"""Tests for ScavioSearch tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio import ScavioSearch
from langchain_scavio._utilities import SCAVIO_API_URL

from .conftest import (
    MOCK_API_KEY,
    make_error_response,
    make_google_v2_full_response,
    make_google_v2_response,
    make_normalized_google_response,
)

API_ENDPOINT = f"{SCAVIO_API_URL}/api/v2/google"
NEWS_ENDPOINT = f"{SCAVIO_API_URL}/api/v2/google/news"

# Every request parameter POST /api/v2/google accepts, under its wire name.
# The tool must expose all of them per call; anything missing here is an
# endpoint feature an agent cannot reach.
V2_WIRE_PARAMS = (
    "query",
    "device",
    "start",
    "include_html",
    "hl",
    "gl",
    "google_domain",
    "location",
    "uule",
    "lr",
    "cr",
    "safe",
    "nfpr",
    "filter",
    "time_period",
    "resolve_ai_overview",
)


def _enum_values(schema: dict, field: str) -> list:
    """Pull the enum members out of an Optional[Literal[...]] JSON schema.

    A single-member Literal serializes as ``const``, not ``enum``.
    """
    for option in [schema["properties"][field]] + schema["properties"][field].get(
        "anyOf", []
    ):
        if "enum" in option:
            return option["enum"]
        if "const" in option:
            return [option["const"]]
    return []


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
            json=make_google_v2_response(),
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
            json=make_google_v2_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert len(result["results"]) == 3

    @responses.activate
    def test_full_mode_includes_all_fields(self, full_tool: ScavioSearch) -> None:
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_google_v2_full_response(),
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
            json=make_google_v2_full_response(),
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
            json=make_google_v2_full_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert "questions" not in result

    @responses.activate
    def test_related_stripped_by_default(self, tool: ScavioSearch) -> None:
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_google_v2_full_response(),
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
            json=make_google_v2_response(organic_results=[]),
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
    def test_search_type_news_routes_to_news_endpoint(
        self, tool: ScavioSearch
    ) -> None:
        responses.add(
            responses.POST,
            NEWS_ENDPOINT,
            json={"news_results": [{"title": "n1"}], "credits_used": 1},
            status=200,
        )
        result = tool.invoke({"query": "latest news", "search_type": "news"})
        assert responses.calls[0].request.url == NEWS_ENDPOINT
        assert result["results"] == [{"title": "n1"}]

    @responses.activate
    def test_country_code_forwarded_as_gl(self, tool: ScavioSearch) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_google_v2_response(),
            status=200,
        )
        tool.invoke({"query": "restaurants", "country_code": "fr"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["gl"] == "fr"
        assert "country_code" not in body

    @responses.activate
    def test_serp_filters_forwarded_verbatim(self, tool: ScavioSearch) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_google_v2_response(),
            status=200,
        )
        tool.invoke(
            {
                "query": "rust web framework",
                "location": "Austin, Texas, United States",
                "uule": "w+CAIQICI",
                "lr": "lang_en",
                "cr": "countryUS",
                "safe": "active",
                "nfpr": True,
                "filter": "0",
                "time_period": "last_week",
                "resolve_ai_overview": False,
                "include_html": True,
            }
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["location"] == "Austin, Texas, United States"
        assert body["uule"] == "w+CAIQICI"
        assert body["lr"] == "lang_en"
        assert body["cr"] == "countryUS"
        assert body["safe"] == "active"
        assert body["nfpr"] is True
        assert body["filter"] == "0"
        assert body["time_period"] == "last_week"
        assert body["resolve_ai_overview"] is False
        assert body["include_html"] is True

    @responses.activate
    def test_unset_serp_filters_are_not_sent(self, tool: ScavioSearch) -> None:
        """Absent params must stay off the wire, nfpr included."""
        import json as json_mod

        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_google_v2_response(),
            status=200,
        )
        tool.invoke({"query": "rust web framework"})
        body = json_mod.loads(responses.calls[0].request.body)
        for key in ("nfpr", "safe", "filter", "time_period", "include_html"):
            assert key not in body

    @responses.activate
    def test_init_defaults_apply_when_the_call_omits_them(self) -> None:
        import json as json_mod

        tool = ScavioSearch(
            scavio_api_key=MOCK_API_KEY, safe="active", time_period="last_day"
        )
        responses.add(
            responses.POST,
            API_ENDPOINT,
            json=make_google_v2_response(),
            status=200,
        )
        tool.invoke({"query": "rust", "time_period": "last_year"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["safe"] == "active"
        # The per-call value wins over the constructor default.
        assert body["time_period"] == "last_year"

    @responses.activate
    def test_classic_only_filters_dropped_on_news(
        self, tool: ScavioSearch
    ) -> None:
        """News has a narrower schema; SERP filters must not be forwarded."""
        import json as json_mod

        responses.add(
            responses.POST,
            NEWS_ENDPOINT,
            json={"news_results": [{"title": "n1"}], "credits_used": 1},
            status=200,
        )
        tool.invoke(
            {
                "query": "elections",
                "search_type": "news",
                "gl": "us",
                "safe": "active",
                "time_period": "last_week",
                "lr": "lang_en",
                "cr": "countryUS",
                "filter": "0",
                "nfpr": True,
                "location": "Austin, Texas, United States",
                "uule": "w+CAIQICI",
                "resolve_ai_overview": True,
                "include_html": True,
            }
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["gl"] == "us"
        for key in (
            "safe",
            "time_period",
            "lr",
            "cr",
            "filter",
            "nfpr",
            "location",
            "uule",
            "resolve_ai_overview",
            "include_html",
            "device",
        ):
            assert key not in body


class TestForbiddenParams:
    def test_init_only_params_rejected_at_invocation(
        self, tool: ScavioSearch
    ) -> None:
        # max_results is an init-only param. Passing it to _run raises
        # ValueError before any API call is made.
        with pytest.raises(ValueError, match="instantiation"):
            tool._run(query="test", max_results=10)


class TestAsync:
    # raw_results_async is patched directly, so mocks return the normalized
    # shape (what the wrapper emits), not the v2 wire shape.
    @pytest.mark.asyncio
    async def test_async_search(self, tool: ScavioSearch) -> None:
        mock_resp = make_normalized_google_response()
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
        mock_resp = make_normalized_google_response(results=[])
        with patch(
            "langchain_scavio._utilities.ScavioSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await tool.ainvoke({"query": "xyzzy"})
            assert "No search results found" in result


class TestInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        props = input_schema["properties"]
        assert "query" in props
        assert "search_type" in props
        assert "country_code" in props
        assert "language" in props
        assert "device" in props
        assert "page" in props

    def test_every_v2_wire_param_is_agent_visible(self) -> None:
        """A param the endpoint takes but the schema hides is unusable."""
        tool = ScavioSearch(scavio_api_key=MOCK_API_KEY)
        props = tool.get_input_schema().model_json_schema()["properties"]
        assert set(V2_WIRE_PARAMS) <= set(props)

    def test_enum_params_use_the_wire_values(self) -> None:
        tool = ScavioSearch(scavio_api_key=MOCK_API_KEY)
        schema = tool.get_input_schema().model_json_schema()
        assert _enum_values(schema, "safe") == ["active"]
        # filter is a string enum on the wire, not a number.
        assert _enum_values(schema, "filter") == ["0", "1"]
        assert _enum_values(schema, "time_period") == [
            "last_hour",
            "last_day",
            "last_week",
            "last_month",
            "last_year",
        ]

    def test_query_is_required(self) -> None:
        tool = ScavioSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "query" in input_schema.get("required", [])
