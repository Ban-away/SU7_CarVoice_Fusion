"""Tests for the evaluation framework."""

from app.eval.scorer import (
    fuzzy_keyword_match,
    semantic_similarity,
    report_score,
    extract_keywords,
)
from app.eval.ragas_eval import EvalSample, _heuristic_eval


class TestScorer:
    def test_fuzzy_keyword_match_perfect(self):
        assert fuzzy_keyword_match("SU7 续航700km", ["续航", "700km"]) == 1.0

    def test_fuzzy_keyword_match_partial(self):
        score = fuzzy_keyword_match("SU7 续航", ["续航", "充电"])
        assert 0 < score < 1.0

    def test_fuzzy_keyword_match_empty(self):
        assert fuzzy_keyword_match("anything", []) == 1.0

    def test_semantic_similarity_identical(self):
        assert semantic_similarity("SU7 续航", "SU7 续航") > 0.5

    def test_semantic_similarity_different(self):
        assert semantic_similarity("SU7 续航", "天气不错") < 0.5

    def test_report_score(self):
        score = report_score("SU7 续航700km", "CLTC 续航约700km", ref_keywords=["续航", "700"])
        assert 0 <= score <= 1.0

    def test_extract_keywords(self):
        keywords = extract_keywords("小米 SU7 标准版 CLTC 续航约 700km")
        assert len(keywords) > 0
        assert "续航" in keywords or "700km" in keywords


class TestRagasEval:
    def test_heuristic_eval(self):
        samples = [
            EvalSample(
                question="SU7 续航",
                answer="700km",
                contexts=["小米 SU7 标准版 CLTC 续航 700km"],
                ground_truth="700km",
            )
        ]
        result = _heuristic_eval(samples)
        assert "context_recall" in result
        assert "faithfulness" in result

    def test_heuristic_eval_empty(self):
        result = _heuristic_eval([])
        assert result == {}
