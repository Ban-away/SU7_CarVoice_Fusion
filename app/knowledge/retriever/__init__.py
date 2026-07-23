"""Retriever implementations — BM25, FAISS, Milvus, hybrid fusion, TF-IDF, Qwen3."""

from app.knowledge.retriever.base import BaseRetriever  # noqa: F401
from app.knowledge.retriever.bm25 import BM25Retriever  # noqa: F401
from app.knowledge.retriever.faiss import FAISSRetriever  # noqa: F401
from app.knowledge.retriever.hybrid import HybridRetriever  # noqa: F401
# from app.knowledge.retriever.milvus import MilvusRetriever  # noqa: F401
from app.knowledge.retriever.qwen3_retriever import FaissRetriever, Qwen3Embeddings  # noqa: F401
from app.knowledge.retriever.retriever_base import BaseRetriever as LegacyRetrieverBase  # noqa: F401
from app.knowledge.retriever.tfidf import TFIDF  # noqa: F401
