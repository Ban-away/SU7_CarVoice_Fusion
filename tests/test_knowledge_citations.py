from app.services.knowledge.service import KnowledgeService


def test_knowledge_citation_structure_contains_source_and_page() -> None:
    service = KnowledgeService(web_search_enabled=False)
    docs = service.search_local_docs("SU7 续航")
    answer, citations = service.synthesize_with_citations("SU7 续航", docs)

    assert answer
    assert citations
    assert "source" in citations[0]
    assert "page" in citations[0]
