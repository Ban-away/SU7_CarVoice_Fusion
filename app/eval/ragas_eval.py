"""RAGas evaluation wrapper — context recall and precision metrics.

Ported from XIAOMI_SU7_RAG/final_score.py.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str = ""


@dataclass
class EvalResult:
    context_recall: float = 0.0
    context_precision: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0


def evaluate_ragas(samples: list[EvalSample]) -> dict:
    """Run RAGas-style evaluation on a list of samples.

    When the ragas library is available, uses the actual LLM-based metrics.
    Otherwise falls back to heuristic approximations suitable for development.

    Returns a dict with aggregated scores.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )

        dataset = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }

        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        return result

    except ImportError:
        logger.info("ragas library not installed — using heuristic evaluation")
        return _heuristic_eval(samples)


def _heuristic_eval(samples: list[EvalSample]) -> dict:
    """Fast heuristic approximations of RAGas metrics."""
    from app.eval.scorer import semantic_similarity

    total = len(samples)
    if total == 0:
        return {}

    recall_sum = 0.0
    precision_sum = 0.0
    faithfulness_sum = 0.0
    relevancy_sum = 0.0

    for s in samples:
        # Context recall: how much of the answer is covered by context
        ctx_text = " ".join(s.contexts)
        recall_sum += semantic_similarity(s.answer, ctx_text) if ctx_text else 0.0

        # Context precision: how much of context is used in answer
        precision_sum += semantic_similarity(ctx_text, s.answer) if ctx_text else 0.0

        # Faithfulness: how well answer matches ground truth
        faithfulness_sum += semantic_similarity(s.answer, s.ground_truth) if s.ground_truth else 0.5

        # Answer relevancy: question-answer similarity
        relevancy_sum += semantic_similarity(s.question, s.answer)

    return {
        "context_recall": round(recall_sum / total, 4),
        "context_precision": round(precision_sum / total, 4),
        "faithfulness": round(faithfulness_sum / total, 4),
        "answer_relevancy": round(relevancy_sum / total, 4),
    }
