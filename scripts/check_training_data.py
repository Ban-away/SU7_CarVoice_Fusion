#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练数据质量检查脚本。

对应 XIAOMI_SU7_RAG/check_training_data.py 的功能：
  检查训练数据文件的完整性、格式正确性和基本统计信息。

用法:
  python scripts/check_training_data.py
  python scripts/check_training_data.py --dir data/training
  python scripts/check_training_data.py --all  # 检查所有数据类型
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

TRAINING_DIR = Path("data/training")


def _check_file(path: str, desc: str) -> dict:
    """Check a single file and return stats."""
    p = Path(path)
    result = {"path": str(p), "exists": p.exists(), "size": 0, "count": 0, "desc": desc, "issues": []}

    if not p.exists():
        result["issues"].append("文件不存在")
        return result

    result["size"] = p.stat().st_size
    try:
        with open(p, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        result["count"] = len(lines)

        # Check for common issues
        empty_ratio = sum(1 for l in lines if not l) / max(len(lines), 1)
        if empty_ratio > 0.3:
            result["issues"].append(f"空行比例过高: {empty_ratio:.1%}")

        # Check format: expect "label\ttext" for training data
        has_tab = sum(1 for l in lines if "\t" in l)
        if has_tab > 0 and has_tab < len(lines) * 0.5:
            result["issues"].append(f"仅 {has_tab}/{len(lines)} 行含制表符，格式可能不一致")

    except Exception as exc:
        result["issues"].append(f"读取失败: {exc}")

    return result


def _check_json(path: str, desc: str, expected_keys: list[str] | None = None) -> dict:
    """Check a JSON file."""
    p = Path(path)
    result = {"path": str(p), "exists": p.exists(), "size": 0, "count": 0, "desc": desc, "issues": []}

    if not p.exists():
        result["issues"].append("文件不存在")
        return result

    result["size"] = p.stat().st_size
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            result["count"] = len(data)
            if data and expected_keys:
                sample = data[0]
                missing_keys = [k for k in expected_keys if k not in sample]
                if missing_keys:
                    result["issues"].append(f"缺少期望字段: {missing_keys}")
                # Count entries with all expected keys
                complete = sum(1 for d in data if all(k in d for k in expected_keys))
                if complete < len(data):
                    result["issues"].append(f"仅 {complete}/{len(data)} 条包含完整字段")
        elif isinstance(data, dict):
            result["count"] = len(data)
        else:
            result["issues"].append(f"非预期的 JSON 类型: {type(data).__name__}")

    except json.JSONDecodeError as exc:
        result["issues"].append(f"JSON 解析失败: {exc}")
    except Exception as exc:
        result["issues"].append(f"读取失败: {exc}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="训练数据质量检查")
    parser.add_argument("--dir", default=str(TRAINING_DIR), help="训练数据目录")
    parser.add_argument("--all", action="store_true", help="检查所有数据类型")
    args = parser.parse_args()

    base = Path(args.dir)
    results = []

    # Check intent data
    results.append(_check_file(str(base / "intent/train.txt"), "意图训练集"))
    results.append(_check_file(str(base / "intent/dev.txt"), "意图验证集"))
    results.append(_check_file(str(base / "intent/test.txt"), "意图测试集"))
    results.append(_check_file(str(base / "intent/class.txt"), "意图类别列表"))

    # Check reject data
    results.append(_check_file(str(base / "reject/train.txt"), "拒识训练集"))
    results.append(_check_file(str(base / "reject/dev.txt"), "拒识验证集"))
    results.append(_check_file(str(base / "reject/test.txt"), "拒识测试集"))

    if args.all:
        # Check QA pairs (JSON)
        results.append(_check_json(
            str(base / "qa_pairs/qa_pair.json"),
            "QA 对 (原始)",
            expected_keys=["question", "answer"],
        ))
        results.append(_check_json(
            str(base / "qa_pairs/qa_pair_filtered.json"),
            "QA 对 (过滤后)",
            expected_keys=["question", "answer"],
        ))

        # Check summary data
        results.append(_check_json(
            str(base / "sft_data/summary_train.json"),
            "Summary 训练集",
            expected_keys=["instruction", "output"],
        ))
        results.append(_check_json(
            str(base / "sft_data/summary_test.json"),
            "Summary 测试集",
            expected_keys=["instruction", "output"],
        ))

    # Print report
    print()
    print("=" * 70)
    print("  📋 训练数据质量检查报告")
    print(f"  目录: {base}")
    print("=" * 70)

    total_issues = 0
    for r in results:
        status = "✅" if r["exists"] and not r["issues"] else "⚠️" if r["exists"] else "❌"
        size_mb = r["size"] / (1024 * 1024) if r["size"] else 0
        count_str = f", {r['count']} 条" if r["count"] else ""

        print(f"  {status} {r['desc']}")
        print(f"     路径: {r['path']}")
        print(f"     大小: {size_mb:.1f} MB{count_str}")

        if r["issues"]:
            total_issues += len(r["issues"])
            for issue in r["issues"]:
                print(f"     ⚠️  {issue}")

    print("=" * 70)

    # Summary
    exists_count = sum(1 for r in results if r["exists"])
    good_count = sum(1 for r in results if r["exists"] and not r["issues"])

    print(f"  文件状态: {exists_count}/{len(results)} 存在, {good_count} 无问题")
    if total_issues:
        print(f"  发现问题: {total_issues} 项")
    else:
        print(f"  所有检查通过 ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
