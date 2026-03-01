"""Shared fixtures for langchain-scavio tests."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_scavio import ScavioSearch

MOCK_API_KEY = "sk_live_test_key_12345"


def make_light_response(**overrides: Any) -> dict[str, Any]:
    """Build a minimal light-mode API response."""
    base: dict[str, Any] = {
        "query": "test query",
        "page": 1,
        "country_code": "",
        "language": "",
        "response_time": 0.45,
        "credits_used": 1,
        "credits_remaining": 999,
        "results": [
            {
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "description": f"Description for result {i}",
                "position": i,
            }
            for i in range(1, 11)
        ],
    }
    base.update(overrides)
    return base


def make_full_response(**overrides: Any) -> dict[str, Any]:
    """Build a full-mode API response with all optional fields."""
    base = make_light_response(credits_used=2)
    base.update(
        {
            "total_results": 1250000,
            "search_url": "https://www.google.com/search?q=test+query",
            "knowledge_graph": {
                "title": "Test Subject",
                "subtitle": "A test entity",
                "factoids": [
                    {"title": "Founded", "content": "2020"},
                    {"title": "Headquarters", "content": "San Francisco"},
                ],
            },
            "questions": [
                {
                    "question": "What is test subject?",
                    "answer": "Test subject is an entity used for testing.",
                },
            ],
            "related_queries": [
                {"title": "test subject reviews", "position": 0, "link": "test subject reviews"},
                {"title": "test subject alternatives", "position": 1, "link": "test subject alternatives"},
            ],
        }
    )
    base.update(overrides)
    return base


def make_error_response(
    status: int, code: str, message: str
) -> dict[str, Any]:
    """Build an API error response body."""
    return {"error": {"code": code, "message": message}}


@pytest.fixture()
def tool() -> ScavioSearch:
    """ScavioSearch with default settings and a test API key."""
    return ScavioSearch(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def full_tool() -> ScavioSearch:
    """ScavioSearch configured for full-mode responses."""
    return ScavioSearch(
        scavio_api_key=MOCK_API_KEY,
        light_request=False,
        include_knowledge_graph=True,
        include_questions=True,
        include_related=True,
    )
