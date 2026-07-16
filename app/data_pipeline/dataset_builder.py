"""Training dataset builder — converts QA pairs to SFT/rerank training data.

Ported from XIAOMI_SU7_RAG/generate_sft_data.py.
"""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """你是一个专业的汽车知识问答助手。请根据提供的参考文档，准确、简洁地回答用户的问题。

参考文档：
{context}

请回答用户的问题：{question}"""


def build_summary_dataset(
    qa_pairs: list[dict[str, str]],
    docs_by_question: dict[str, list[str]] | None = None,
    output_dir: str = "data/training/summary",
    test_split: float = 0.08,
) -> None:
    """Build instruction-format summary training data.

    Args:
        qa_pairs: list of {"question": ..., "answer": ...}
        docs_by_question: optional mapping of question → context documents
        output_dir: where to write train.json / test.json
        test_split: fraction reserved for test set
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    for pair in qa_pairs:
        q = pair["question"]
        a = pair["answer"]
        ctx_docs = (docs_by_question or {}).get(q, [])
        context = "\n".join(ctx_docs) if ctx_docs else q
        samples.append({
            "instruction": SUMMARY_SYSTEM_PROMPT.format(context=context, question=q),
            "input": "",
            "output": a,
        })

    random.shuffle(samples)
    split_idx = max(1, int(len(samples) * (1 - test_split)))
    train = samples[:split_idx]
    test = samples[split_idx:]

    with open(out / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(out / "test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    logger.info("Summary dataset: %d train, %d test → %s", len(train), len(test), output_dir)


def build_rerank_dataset(
    queries: list[str],
    pos_docs: list[list[str]],
    neg_docs: list[list[str]] | None = None,
    output_dir: str = "data/training/rerank",
) -> None:
    """Build reranker training triples (query, doc, label).

    label: 2 = top hit, 1 = other positive, 0 = negative/hard negative.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    triples: list[dict] = []
    for i, query in enumerate(queries):
        pos = pos_docs[i] if i < len(pos_docs) else []
        neg = neg_docs[i] if neg_docs and i < len(neg_docs) else []
        for j, doc in enumerate(pos):
            label = 2 if j == 0 else 1
            triples.append({"query": query, "document": doc, "label": label})
        for doc in neg:
            triples.append({"query": query, "document": doc, "label": 0})

    random.shuffle(triples)
    dev_count = min(1000, len(triples) // 10)

    with open(out / "train.json", "w", encoding="utf-8") as f:
        json.dump(triples[:-dev_count], f, ensure_ascii=False, indent=2)
    with open(out / "dev.json", "w", encoding="utf-8") as f:
        json.dump(triples[-dev_count:], f, ensure_ascii=False, indent=2)

    logger.info("Rerank dataset: %d train, %d dev → %s", len(triples) - dev_count, dev_count, output_dir)
