"""Custom scoring for RAG evaluation — semantic similarity + keyword hit rate.

Ported from XIAOMI_SU7_RAG/final_score.py (report_score function).
"""

import re
from difflib import SequenceMatcher


def fuzzy_keyword_match(pred: str, ref_keywords: list[str]) -> float:
    """Calculate keyword hit rate via fuzzy substring matching.

    Args:
        pred: Predicted answer text.
        ref_keywords: List of expected keywords.

    Returns:
        Fraction of keywords found in the prediction (0.0–1.0).
    """
    if not ref_keywords:
        return 1.0

    hits = 0
    for kw in ref_keywords:
        # Simple fuzzy: check if keyword or ≥70% similar substring exists
        if kw in pred:
            hits += 1
        else:
            # Sliding window fuzzy match
            kw_len = len(kw)
            if kw_len <= 1:
                continue
            for i in range(len(pred) - kw_len + 1):
                window = pred[i:i + kw_len]
                if SequenceMatcher(None, kw, window).ratio() >= 0.7:
                    hits += 1
                    break
    return hits / len(ref_keywords)


def semantic_similarity(pred: str, ref: str) -> float:
    """Estimate semantic similarity between prediction and reference.

    Uses character-level Jaccard as a fast proxy when no embedding model
    is available.
    """
    if not pred or not ref:
        return 0.0
    set_p = set(pred)
    set_r = set(ref)
    if not set_p or not set_r:
        return 0.0
    intersection = set_p & set_r
    union = set_p | set_r
    return len(intersection) / len(union) if union else 0.0


def report_score(
    pred: str,
    ref: str,
    ref_keywords: list[str] | None = None,
    semantic_weight: float = 0.7,
) -> float:
    """Composite score: weighted combination of semantic similarity and keyword hits.

    Args:
        pred: Model-generated answer.
        ref: Reference answer.
        ref_keywords: Expected keywords from the reference.
        semantic_weight: Weight for semantic similarity (0–1).

    Returns:
        Score in [0, 1].
    """
    sem_score = semantic_similarity(pred, ref)
    kw_score = fuzzy_keyword_match(pred, ref_keywords or [])
    return semantic_weight * sem_score + (1 - semantic_weight) * kw_score


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """Extract likely keywords from reference text using simple heuristics."""
    # Remove punctuation, split, filter short tokens
    cleaned = re.sub(r"[，。！？、；：""''（）【】《》 ]", " ", text)
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) >= 2]
    # Simple TF heuristic: longer tokens and those with technical characters
    scored = sorted(tokens, key=lambda t: (
        -len(t),
        -sum(1 for c in t if '一' <= c <= '鿿'),
    ))
    # Dedup
    seen = set()
    result = []
    for t in scored:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) >= top_n:
            break
    return result
