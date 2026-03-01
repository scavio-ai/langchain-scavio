"""Tests for ScavioSearch tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import responses
from langchain_core.tools import ToolException

from langchain_scavio import ScavioSearch
from langchain_scavio._utilities import SCAVIO_API_URL

from .conftest import (
    MOCK_API_KEY,
    make_error_response,
    make_full_response,
    make_light_response,
)

API_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/google"


class TestInstantiation:
    def test_default_params(self, tool: ScavioSearch) -> None:
        assert tool.name == "scavio_search"
        assert tool.max_results == 5
        assert tool.light_request is None
        assert tool.include_knowledge_graph is True
        assert tool.include_questions is True
        assert tool.include_related is False
        assert tool.handle_tool_error is True

    def test_custom_params(self) -> None:
        tool = ScavioSearch(
            scavio_api_key=MOCK_API_KEY,
            max_results=10,
            light_request=False,
            include_knowledge_graph=False,
            include_questions=False,
            include_related=True,
        )
        assert tool.max_results == 10
        assert tool.light_request is False
        assert tool.include_knowledge_graph is False
        assert tool.include_questions is False
        assert tool.include_related is True

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioSearch(scavio_api_key=MOCK_API_KEY)
        assert (
            tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY
        )

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioSearch(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioSearch()
        assert (
            tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY
        )


class TestRun:
    @responses.activate
    def test_successful_search(self, tool: ScavioSearch) -> None:
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_light_response(),
            status=200,
        )
        result = tool.invoke({"query": "python frameworks"})
        assert "results" in result
        assert len(result["results"]) == 5  # truncated by max_results

    @responses.activate
    def test_max_results_truncation(self) -> None:
        tool = ScavioSearch(scavio_api_key=MOCK_API_KEY, max_results=3)
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_light_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert len(result["results"]) == 3

    @responses.activate
    def test_full_mode_includes_all_fields(self, full_tool: ScavioSearch) -> None:
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_full_response(),
            status=200,
        )
        result = full_tool.invoke({"query": "test"})
        assert "knowledge_graph" in result
        assert "questions" in result
        assert "related_queries" in result

    @responses.activate
    def test_knowledge_graph_stripped_when_disabled(self) -> None:
        tool = ScavioSearch(
            scavio_api_key=MOCK_API_KEY, include_knowledge_graph=False
        )
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_full_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert "knowledge_graph" not in result

    @responses.activate
    def test_questions_stripped_when_disabled(self) -> None:
        tool = ScavioSearch(
            scavio_api_key=MOCK_API_KEY, include_questions=False
        )
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_full_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert "questions" not in result

    @responses.activate
    def test_related_stripped_by_default(self, tool: ScavioSearch) -> None:
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_full_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert "related_queries" not in result

    @responses.activate
    def test_empty_results_raises_tool_exception(
        self, tool: ScavioSearch
    ) -> None:
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_light_response(results=[]),
            status=200,
        )
        # handle_tool_error=True means ToolException is caught by LangChain
        # and returned as a string. When invoked directly, it surfaces as str.
        result = tool.invoke({"query": "xyzzy obscure"})
        assert "No search results found" in result

    @responses.activate
    def test_api_error_returns_error_dict(self, tool: ScavioSearch) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, API_ENDPOINT, json=error, status=401)
        result = tool.invoke({"query": "test"})
        # API errors are caught and returned as {"error": "..."} or as
        # a string via handle_tool_error depending on the exception path.
        assert "error" in str(result).lower()

    @responses.activate
    def test_search_type_forwarded(self, tool: ScavioSearch) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_light_response(),
            status=200,
        )
        tool.invoke({"query": "latest news", "search_type": "news"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["search_type"] == "news"

    @responses.activate
    def test_country_code_forwarded(self, tool: ScavioSearch) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_light_response(),
            status=200,
        )
        tool.invoke({"query": "restaurants", "country_code": "fr"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["country_code"] == "fr"


class TestForbiddenParams:
    def test_init_only_params_rejected_at_invocation(
        self, tool: ScavioSearch
    ) -> None:
        # max_results is an init-only param. Passing it to _run raises
        # ValueError before any API call is made.
        with pytest.raises(ValueError, match="instantiation"):
            tool._run(query="test", max_results=10)


class TestAsync:
    @pytest.mark.asyncio
    async def test_async_search(self, tool: ScavioSearch) -> None:
        mock_resp = make_light_response()
        with patch(
            "langchain_scavio._utilities.ScavioSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await tool.ainvoke({"query": "async test"})
            assert "results" in result
            assert len(result["results"]) == 5

    @pytest.mark.asyncio
    async def test_async_empty_results(self, tool: ScavioSearch) -> None:
        mock_resp = make_light_response(results=[])
        with patch(
            "langchain_scavio._utilities.ScavioSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await tool.ainvoke({"query": "xyzzy"})
            assert "No search results found" in result


class TestInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        schema = ScavioSearch.model_json_schema()
        # The tool's args_schema should expose the right properties
        tool = ScavioSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        props = input_schema["properties"]
        assert "query" in props
        assert "search_type" in props
        assert "country_code" in props
        assert "language" in props
        assert "device" in props
        assert "page" in props

    def test_query_is_required(self) -> None:
        tool = ScavioSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "query" in input_schema.get("required", [])
