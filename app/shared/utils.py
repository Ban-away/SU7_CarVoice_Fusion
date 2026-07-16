"""Shared utility functions — WRRF fusion, doc merging, post-processing.

Ported from XIAOMI_SU7_RAG/src/utils.py.
"""

import re


def wrrf_fusion(
    results_list: list[list],
    weights: list[float] | None = None,
    k: int = 60,
) -> list:
    """Weighted Reciprocal Rank Fusion.

    Formula: score(d) = Σ (w_i / (k + rank_i(d)))

    Args:
        results_list: One list of documents per retriever.
        weights: Per-retriever weight; defaults to equal weights.
        k: Rank-decay constant (typically 60–100).

    Returns:
        Documents sorted by descending WRRF score.
    """
    if not results_list or all(not results for results in results_list):
        return []

    if weights is None:
        weights = [1.0] * len(results_list)

    doc_scores: dict[str, float] = {}
    doc_map: dict[str, object] = {}

    for idx, results in enumerate(results_list):
        weight = weights[idx]
        for rank, doc in enumerate(results, 1):
            unique_id = _doc_uid(doc)
            if unique_id not in doc_scores:
                doc_scores[unique_id] = 0.0
            doc_scores[unique_id] += weight / (k + rank)
            if unique_id not in doc_map:
                doc_map[unique_id] = doc

    sorted_ids = sorted(doc_scores.keys(), key=lambda x: -doc_scores[x])
    return [doc_map[uid] for uid in sorted_ids]


def _doc_uid(doc) -> str:
    """Extract a unique identifier from a document-like object."""
    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
        return doc.metadata.get("unique_id", str(id(doc)))
    if isinstance(doc, dict):
        return doc.get("unique_id", doc.get("source", str(id(doc))))
    return str(id(doc))


def merge_docs(
    docs1: list,
    docs2: list,
    use_wrrf: bool = True,
) -> list:
    """Merge two retrieval result lists with per-list dedup and optional WRRF.

    Each list is deduped internally first, then the two are fused.
    """
    def _dedup(docs: list) -> list:
        seen: set[str] = set()
        result: list = []
        for doc in docs:
            uid = _doc_uid(doc)
            if uid not in seen:
                seen.add(uid)
                result.append(doc)
        return result

    d1 = _dedup(docs1)
    d2 = _dedup(docs2)

    if use_wrrf:
        return wrrf_fusion([d1, d2], weights=[0.7, 0.7], k=60)

    final_ids: set[str] = set()
    final: list = []
    for doc in d1 + d2:
        uid = _doc_uid(doc)
        if uid not in final_ids:
            final_ids.add(uid)
            final.append(doc)
    return final


def post_processing(response: str, docs: list) -> dict:
    """Post-process an LLM answer: extract citations, strip markers, collect pages.

    Returns:
        {"answer": str, "cite_pages": list[int], "related_images": list}
    """
    all_cites = re.findall(r"[【](.*?)[】]", response)
    cites: list[int] = []
    for cite in all_cites:
        cite = re.sub(r"[{} 【】]", "", cite)
        cite = cite.replace(",", "，")
        cites.extend(int(k) for k in cite.split("，") if k.isdigit())
    cites = list(set(cites))

    answer = re.sub(r"[【](.*?)[】]", "", response)
    answer = re.sub(r"[{}【】]", "", answer)

    pages: list[int] = []
    related_images: list = []
    for index in cites:
        if index > len(docs) or index < 1:
            continue
        doc_ref = docs[index - 1]
        meta = getattr(doc_ref, "metadata", {}) if hasattr(doc_ref, "metadata") else {}
        if isinstance(doc_ref, dict):
            meta = doc_ref.get("metadata", {})
        images = meta.get("images_info", [])
        page = meta.get("page")
        if page is not None:
            pages.append(page)
        for img in images:
            if img.get("title"):
                related_images.append(img)

    pages = sorted(set(pages))
    if answer.strip() in ("无答案", "没有答案", "无", ""):
        pages = []
        related_images = []

    return {"answer": answer, "cite_pages": pages, "related_images": related_images}
