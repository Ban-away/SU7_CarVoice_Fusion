"""FAISS retriever with sentence-transformer embeddings.

Uses ``sentence-transformers`` for dense embeddings and FAISS
``IndexFlatIP`` (inner-product) for similarity search.  Falls back to
random embeddings when the required libraries are unavailable.
"""

import logging
from typing import Optional

from app.knowledge.models import RetrievedDoc
from app.knowledge.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

try:
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_TRANSFORMERS = False


# ---------------------------------------------------------------------------
# FAISSRetriever
# ---------------------------------------------------------------------------

class FAISSRetriever(BaseRetriever):
    """Dense retriever using FAISS inner-product search.

    Parameters:
        documents: List of text strings to index.
        source: Source identifier applied to all returned RetrievedDoc
            entries (default ``"local_docs"``).
        model_name: Name of the sentence-transformer model to use
            (only meaningful when ``sentence-transformers`` is installed).
        embedding_dim: Embedding vector dimension (used for random
            fallback embeddings).
    """

    def __init__(
        self,
        documents: list[str],
        source: str = "local_docs",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        embedding_dim: int = 512,
    ) -> None:
        self._source = source
        self._raw_docs = documents
        self._embedding_dim = embedding_dim

        # ── Embedding model ──
        self._encoder: Optional[SentenceTransformer] = None
        if _HAS_TRANSFORMERS:
            try:
                self._encoder = SentenceTransformer(model_name)
                logger.info("Loaded embedding model: %s", model_name)
            except Exception:
                logger.warning(
                    "Failed to load sentence-transformer model '%s'; "
                    "falling back to random embeddings",
                    model_name,
                )
        else:
            logger.info(
                "sentence-transformers not installed; using random embeddings"
            )

        # ── Build FAISS index ──
        self._index: Optional[faiss.IndexFlatIP] = None
        self._doc_embeddings: Optional[np.ndarray] = None

        if _HAS_NUMPY and _HAS_FAISS and documents:
            self._build_index()
        elif not _HAS_NUMPY:
            logger.warning("numpy not installed; FAISS retriever disabled")
        elif not _HAS_FAISS:
            logger.warning("faiss not installed; FAISS retriever disabled")

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDoc]:
        """Return up to *top_k* documents ranked by inner-product similarity."""
        if not self._raw_docs:
            return []

        if self._index is None or self._doc_embeddings is None:
            return self._random_fallback(query, top_k)

        try:
            query_vec = self._encode_text(query)
            query_vec = query_vec.reshape(1, -1).astype(np.float32)

            # Normalise for inner-product
            faiss.normalize_L2(query_vec)

            scores, indices = self._index.search(query_vec, min(top_k, len(self._raw_docs)))  # type: ignore[union-attr]

            results: list[RetrievedDoc] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._raw_docs):
                    continue
                results.append(
                    RetrievedDoc(
                        content=self._raw_docs[int(idx)],
                        source=self._source,
                        score=round(float(score), 4),
                    )
                )
            return results

        except Exception:
            logger.exception("FAISS retrieval failed; using random fallback")
            return self._random_fallback(query, top_k)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Encode all documents and build a FAISS IndexFlatIP."""
        import numpy as np  # local import for type-checking clarity

        embeddings = self._encode_documents(self._raw_docs)
        self._doc_embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(self._doc_embeddings)

        dim = self._doc_embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # type: ignore[union-attr]
        self._index.add(self._doc_embeddings)  # type: ignore[union-attr]
        logger.info(
            "Built FAISS index with %d documents (dim=%d)",
            len(self._raw_docs),
            dim,
        )

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector."""
        import numpy as np

        if self._encoder is not None:
            emb = self._encoder.encode([text], normalize_embeddings=True)
            return np.asarray(emb[0], dtype=np.float32)
        return self._random_embedding(len(text))

    def _encode_documents(self, docs: list[str]) -> np.ndarray:
        """Encode a list of document texts into a (N, D) matrix."""
        import numpy as np

        if self._encoder is not None:
            embs = self._encoder.encode(docs, normalize_embeddings=True)
            return np.asarray(embs, dtype=np.float32)

        rng = np.random.RandomState(42)
        return rng.randn(len(docs), self._embedding_dim).astype(np.float32)

    def _random_embedding(self, seed: int) -> np.ndarray:
        """Generate a deterministic random embedding from *seed*."""
        import numpy as np

        rng = np.random.RandomState(abs(hash(str(seed))) % (2**31))
        return rng.randn(self._embedding_dim).astype(np.float32)

    def _random_fallback(self, query: str, top_k: int) -> list[RetrievedDoc]:
        """Return pseudo-random documents when the index is unavailable."""
        if not self._raw_docs:
            return []
        # Use a deterministic hash of the query to pick documents
        seed = abs(hash(query)) % (2**31)
        import random
        rng = random.Random(seed)
        indices = list(range(len(self._raw_docs)))
        rng.shuffle(indices)
        results = []
        for idx in indices[:top_k]:
            results.append(
                RetrievedDoc(
                    content=self._raw_docs[idx],
                    source=self._source,
                    score=0.5,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self._raw_docs)
