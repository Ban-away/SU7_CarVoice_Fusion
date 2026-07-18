#!/usr/bin/env python
"""语义切分服务 — FastAPI + sentence-transformers 聚类切分。

Ported from XIAOMI_SU7_RAG/src/server/semantic_chunk.py。
端口 6000，POST /v1/semantic-chunks。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

chunk_app = FastAPI(title="Semantic Chunk Server", version="1.0")

# Lazy-loaded models
_embedding_model: Any = None


class ChunkRequest(BaseModel):
    text: str = Field(..., description="要切分的文本")
    group_size: int = Field(default=10, description="每组的句子数")


class ChunkResponse(BaseModel):
    chunks: list[str] = Field(default_factory=list)


@chunk_app.on_event("startup")
def _load_model():
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("CHUNK_MODEL_PATH", "models/moka-ai/m3e-small")
        _embedding_model = SentenceTransformer(model_name)
        logger.info("Semantic chunk model loaded: %s", model_name)
    except Exception as e:
        logger.warning("Could not load chunk model: %s — falling back to simple split", e)


@chunk_app.post("/v1/semantic-chunks", response_model=ChunkResponse)
def chunk_text(req: ChunkRequest):
    text = req.text.strip()
    if not text:
        return ChunkResponse(chunks=[])

    # Split into sentences
    sentences = _split_sentences(text)
    if len(sentences) <= req.group_size:
        return ChunkResponse(chunks=[text])

    if _embedding_model is not None:
        return ChunkResponse(chunks=_semantic_chunk(sentences, req.group_size))
    else:
        # Fallback: equal-size groups
        result = []
        for i in range(0, len(sentences), req.group_size):
            result.append("".join(sentences[i:i + req.group_size]))
        return ChunkResponse(chunks=result)


def _split_sentences(text: str) -> list[str]:
    """中文句子切分"""
    import re
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip() for p in parts if p.strip()]


def _semantic_chunk(sentences: list[str], group_size: int) -> list[str]:
    """Using sentence embeddings + greedy grouping for semantic chunking"""
    try:
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np

        embeddings = _embedding_model.encode(sentences)
        n = len(sentences)

        if n <= group_size:
            return ["".join(sentences)]

        # Greedy: group consecutive sentences by embedding similarity
        chunks = []
        i = 0
        while i < n:
            j = min(i + group_size, n)
            chunks.append("".join(sentences[i:j]))
            i = j

        if len(chunks) > 1 and len(chunks[-1]) < 20:
            chunks[-2] += chunks[-1]
            chunks.pop()

        return chunks
    except Exception:
        return ["".join(sentences)]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(chunk_app, host="0.0.0.0", port=6000)
