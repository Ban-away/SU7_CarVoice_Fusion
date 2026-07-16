"""Shared test fixtures for SU7_CarVoice_Fusion."""

import pytest

from app.orchestrator.router import ChatOrchestrator
from app.services.knowledge.service import KnowledgeService
from app.services.skills.registry import SkillsRegistry


@pytest.fixture
def skills_registry() -> SkillsRegistry:
    """Return a fresh skills registry with all default skills registered."""
    return SkillsRegistry()


@pytest.fixture
def knowledge_service() -> KnowledgeService:
    """Return a knowledge service with web search disabled for deterministic tests."""
    return KnowledgeService(web_search_enabled=False)


@pytest.fixture
def orchestrator() -> ChatOrchestrator:
    """Return a fully wired orchestrator for integration-style tests."""
    return ChatOrchestrator()
