"""Milvus hybrid retriever — BGE-Large-zh-v1.5 (dense) + SPLADE v2 (sparse).

Ported from XIAOMI_SU7_RAG/src/retriever/milvus_retriever.py.
Connects to a local Milvus Lite database (uri from config).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from app.knowledge.models import RetrievedDoc
from app.knowledge.retriever.base import BaseRetriever
from app.shared.config import get_settings

logger = logging.getLogger(__name__)

# ── Optional imports ────────────────────────────────────────────────────────
try:
    import torch
    from pymilvus import (  # type: ignore[import-untyped]
        AnnSearchRequest,
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        WeightedRanker,
        connections,
        utility,
    )
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    _MILVUS_AVAILABLE = True
except ImportError:
    _MILVUS_AVAILABLE = False

# ── Constants (matching original) ───────────────────────────────────────────
EMB_BATCH = 32
MAX_TEXT_LENGTH = 2048
ID_MAX_LENGTH = 100
COL_NAME = "hybrid_bge_large_splade_v2"
SPARSE_TOPK = 200

# Default model paths (override via config or env)
BGE_MODEL = "BAAI/bge-large-zh-v1.5"
SPLADE_MODEL = "naver/splade-v2-doc"


def _mean_pooling(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts


class HybridEmbeddingHandler:
    """BGE-Large dense + SPLADE sparse dual-encoder with GPU auto-detect."""

    def __init__(
        self,
        dense_model_path: str = BGE_MODEL,
        splade_model_path: str = SPLADE_MODEL,
        device: str | None = None,
    ) -> None:
        if not _MILVUS_AVAILABLE:
            raise ImportError("pymilvus, torch, and transformers are required for MilvusRetriever")

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if device is None:
            device = "cuda" if num_gpus >= 1 else "cpu"
        self.device = device

        self.dense_device = f"cuda:0" if num_gpus >= 1 else device
        logger.info("BGE-Large loading on %s", self.dense_device)
        self.dense_tokenizer = AutoTokenizer.from_pretrained(dense_model_path)
        self.dense_model = AutoModel.from_pretrained(
            dense_model_path, torch_dtype=torch.float16, device_map=self.dense_device
        ).eval()

        self.sparse_device = f"cuda:1" if num_gpus >= 2 else (f"cuda:0" if num_gpus >= 1 else device)
        logger.info("SPLADE loading on %s", self.sparse_device)
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(splade_model_path)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(
            splade_model_path, torch_dtype=torch.float16, device_map=self.sparse_device
        ).eval()

        self.dim = {"dense": self.dense_model.config.hidden_size}

    def _encode_dense(self, texts: list[str]) -> list[list[float]]:
        inputs = self.dense_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.dense_device)
        with torch.no_grad():
            outputs = self.dense_model(**inputs)
        pooled = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        return pooled.cpu().detach().numpy().tolist()

    def _encode_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        inputs = self.sparse_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.sparse_device)
        with torch.no_grad():
            outputs = self.sparse_model(**inputs)
        logits = outputs.logits
        attn_mask = inputs["attention_mask"].unsqueeze(-1)
        logits = logits.masked_fill(attn_mask == 0, float("-inf"))
        weights = torch.log1p(torch.relu(logits)).amax(dim=1)

        sparse_vectors: list[dict[int, float]] = []
        for row in weights:
            if SPARSE_TOPK < row.numel():
                values, indices = torch.topk(row, SPARSE_TOPK)
                nonzero = values > 0
                indices = indices[nonzero]
                values = values[nonzero]
            else:
                indices = torch.nonzero(row > 0, as_tuple=False).squeeze(1)
                values = row[indices]
            sparse_vectors.append({int(i): float(v) for i, v in zip(indices, values)})
        return sparse_vectors

    def encode(self, texts: list[str], batch_size: int = EMB_BATCH) -> dict[str, list]:
        dense_all: list[list[float]] = []
        sparse_all: list[dict[int, float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            dense_all.extend(self._encode_dense(batch))
            sparse_all.extend(self._encode_sparse(batch))
            torch.cuda.empty_cache()
        return {"dense": dense_all, "sparse": sparse_all}

    def encode_queries(self, queries: list[str]) -> dict[str, list]:
        return self.encode(queries)


class MilvusRetriever(BaseRetriever):
    """Milvus hybrid retriever using BGE-Large (dense) + SPLADE v2 (sparse).

    Indexes documents with dual-vector encoding and retrieves via
    Milvus WeightedRanker hybrid search.
    """

    def __init__(
        self,
        documents: list[str],
        source: str = "local_docs",
        dense_model_path: str = BGE_MODEL,
        splade_model_path: str = SPLADE_MODEL,
        milvus_uri: str | None = None,
    ) -> None:
        if not _MILVUS_AVAILABLE:
            raise ImportError(
                "MilvusRetriever requires: pip install pymilvus torch transformers"
            )

        self._source = source
        settings = get_settings()
        uri = milvus_uri or f"data/knowledge/saved_index/milvus.db"

        # ── Embedding handler ──
        self._embedder = HybridEmbeddingHandler(
            dense_model_path=dense_model_path,
            splade_model_path=splade_model_path,
        )

        # ── Connect Milvus ──
        connections.connect(uri=uri)

        # ── Schema ──
        fields = [
            FieldSchema(name="unique_id", dtype=DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LENGTH),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self._embedder.dim["dense"]),
        ]
        schema = CollectionSchema(fields)

        # Drop existing to rebuild
        if utility.has_collection(COL_NAME):
            Collection(COL_NAME).drop()
        self._col = Collection(COL_NAME, schema, consistency_level="Strong")

        # Create indexes
        self._col.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
        self._col.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
        self._col.load()

        # ── Index documents ──
        self._index_documents(documents)

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        query_emb = self._embedder.encode_queries([query])
        results = self._col.hybrid_search(
            [
                AnnSearchRequest([query_emb["sparse"][0]], "sparse_vector", {"metric_type": "IP", "params": {}}, limit=top_k),
                AnnSearchRequest([query_emb["dense"][0]], "dense_vector", {"metric_type": "IP", "params": {}}, limit=top_k),
            ],
            rerank=WeightedRanker(1.0, 1.0),
            limit=top_k,
            output_fields=["unique_id", "text"],
        )[0]

        docs: list[RetrievedDoc] = []
        for hit in results:
            docs.append(
                RetrievedDoc(
                    content=hit.entity.get("text", ""),
                    source=self._source,
                    score=hit.distance,
                )
            )
        return docs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_documents(self, documents: list[str]) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        truncated = [t[:MAX_TEXT_LENGTH] for t in documents]
        unique_ids = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in truncated]

        logger.info("Encoding %d documents for Milvus...", len(documents))
        embeddings = self._embedder.encode(truncated, batch_size=EMB_BATCH)

        logger.info("Inserting into Milvus...")
        for i in range(0, len(documents), EMB_BATCH):
            batch_entities = [
                unique_ids[i : i + EMB_BATCH],
                truncated[i : i + EMB_BATCH],
                embeddings["sparse"][i : i + EMB_BATCH],
                embeddings["dense"][i : i + EMB_BATCH],
            ]
            self._col.insert(batch_entities)
            torch.cuda.empty_cache()

        logger.info("Milvus index ready: %d entities", self._col.num_entities)
