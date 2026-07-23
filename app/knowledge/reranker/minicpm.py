"""MiniCPM reranker with mock fallback.

When the MiniCPM model is unavailable, falls back to a simple
keyword-overlap score — preserving the reranker interface so that
callers work unchanged.
"""

import logging
from typing import Optional

from app.knowledge.models import RetrievedDoc
from app.knowledge.reranker.base import BaseReranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


class MiniCPMReranker(BaseReranker):
    """Reranker powered by MiniCPM embedding / cross-encoder.

    When the model cannot be loaded a simple keyword-overlap scorer is
    used instead, ensuring the pipeline always produces results.

    Parameters:
        model_path: HuggingFace model name or local path.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        use_fp16: Whether to load the model in half precision.
    """

    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/SU7_CarVoice_Fusion/models/BAAI/bge-reranker-v2-minicpm-layerwise",
        device: str = "cpu",
        use_fp16: bool = False,
    ) -> None:
        self._model_path = model_path
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None

        if _HAS_TORCH:
            try:
                self._load_model(model_path, device, use_fp16)
            except Exception:
                logger.warning(
                    "Failed to load MiniCPM reranker model '%s'; "
                    "using keyword-overlap fallback",
                    model_path,
                )
        else:
            logger.info("torch not installed; using keyword-overlap fallback")

        if self._model is None:
            logger.info("MiniCPMReranker running in fallback (keyword-overlap) mode")

    # ------------------------------------------------------------------
    # BaseReranker interface
    # ------------------------------------------------------------------

    def rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """Rerank *docs* by relevance to *query*."""
        if not docs:
            return []

        if self._model is not None:
            return self._model_rerank(query, docs)
        return self._keyword_rerank(query, docs)

    # ------------------------------------------------------------------
    # MiniCPM model path
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, device: str, use_fp16: bool) -> None:
        """Attempt to load the MiniCPM reranker model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.warning("transformers not installed; using keyword-overlap fallback")
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            # Load as CausalLM to get lm_head weights, use inner model for embeddings
            self._model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
            )
            self._model = self._model.to(device)  # type: ignore[union-attr]
            self._model.eval()  # type: ignore[union-attr]
            logger.info("Loaded MiniCPM reranker: %s on %s", model_path, device)
        except Exception as exc:
            logger.warning("MiniCPM model load failed: %s", exc)
            self._model = None
            self._tokenizer = None

    def _model_rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """Rerank using the loaded MiniCPM model."""
        import torch

        pairs = []
        for doc in docs:
            pairs.append([query, doc.content])

        if self._tokenizer is None or self._model is None:
            return self._keyword_rerank(query, docs)

        try:
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}  # type: ignore[union-attr]
                if self._model is None:
                    new_scores = [0.0] * len(docs)
                else:
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                        # Mean pool → take L2 norm as relevance signal
                        hidden = outputs.last_hidden_state.float()  # type: ignore[union-attr]
                        pooled = hidden.mean(dim=1)  # [batch, dim]
                        new_scores = pooled.norm(dim=-1).tolist()

            for doc, ns in zip(docs, new_scores):
                doc.score = round(float(ns), 4)

            docs.sort(key=lambda d: d.score, reverse=True)
            return docs
        except Exception:
            logger.exception("Model rerank failed; falling back to keyword overlap")
            return self._keyword_rerank(query, docs)

    # ------------------------------------------------------------------
    # Keyword-overlap fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_rerank(query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """Simple keyword-overlap scorer as a fallback reranker."""
        query_terms = set(query.lower())
        for doc in docs:
            content_lower = doc.content.lower()
            overlap = sum(1 for t in query_terms if t in content_lower)
            # Blend original score with keyword overlap
            doc.score = round(doc.score * 0.4 + overlap * 0.6, 4)

        docs.sort(key=lambda d: d.score, reverse=True)
        return docs
