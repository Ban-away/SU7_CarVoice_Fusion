"""Knowledge / RAG pipeline for the SU7 CarVoice Fusion assistant.

Provides retrieval-augmented generation:
- Document retrieval (BM25, FAISS, hybrid with WRRF fusion)
- Reranking (MiniCPM with mock fallback)
- Answer generation via LLM
- Citation synthesis
- Web search (mock / config-driven)
- Document parsing (PDF) and semantic chunking
"""

from app.knowledge.service import KnowledgeService  # noqa: F401
