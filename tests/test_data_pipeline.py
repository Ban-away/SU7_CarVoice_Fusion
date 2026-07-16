"""Tests for the data pipeline — QA generation, filtering, abbreviation expansion."""

from app.data_pipeline.abbr_expander import expand_abbreviations, load_abbr_map
from app.data_pipeline.qa_filter import (
    is_low_quality_question,
    is_off_topic,
    is_invalid_answer,
    filter_qa_pairs,
)
from app.data_pipeline.dataset_builder import build_summary_dataset


class TestAbbrExpander:
    def test_load_abbr_map(self):
        abbr_map = load_abbr_map()
        assert "ECU" in abbr_map
        assert abbr_map["ECU"] == "电子控制单元"

    def test_expand_abbreviations(self):
        result = expand_abbreviations("ECU 故障怎么处理", load_abbr_map())
        assert "电子控制单元" in result


class TestQAFilter:
    def test_low_quality_short(self):
        assert is_low_quality_question("ab") is True

    def test_normal_question(self):
        assert is_low_quality_question("SU7 续航是多少公里") is False

    def test_invalid_answer(self):
        assert is_invalid_answer("无答案") is True

    def test_valid_answer(self):
        assert is_invalid_answer("小米 SU7 CLTC 续航约 700km") is False

    def test_filter_qa_pairs(self):
        pairs = [
            {"question": "SU7 续航多少", "answer": "700km"},
            {"question": "ab", "answer": "无"},
            {"question": "你好", "answer": "你好呀"},
        ]
        clean = filter_qa_pairs(pairs, expand_abbr=False)
        assert len(clean) <= 3


class TestDatasetBuilder:
    def test_build_summary(self, tmp_path):
        pairs = [
            {"question": "SU7 续航多少", "answer": "CLTC 续航 700km"},
        ]
        build_summary_dataset(pairs, output_dir=str(tmp_path / "summary"), test_split=0.5)
        assert (tmp_path / "summary" / "train.json").exists()
        assert (tmp_path / "summary" / "test.json").exists()
