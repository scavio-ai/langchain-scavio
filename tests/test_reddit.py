"""Tests for ScavioRedditSearch and ScavioRedditPost."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio._utilities import SCAVIO_API_URL
from langchain_scavio.scavio_reddit import ScavioRedditPost, ScavioRedditSearch

from .conftest import (
    MOCK_API_KEY,
    make_error_response,
    make_reddit_post_response,
    make_reddit_search_response,
)

SEARCH_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/reddit/search"
POST_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/reddit/post"


class TestRedditSearchInstantiation:
    def test_default_params(self, reddit_search_tool: ScavioRedditSearch) -> None:
        assert reddit_search_tool.name == "scavio_reddit_search"
        assert reddit_search_tool.max_results == 5
        assert reddit_search_tool.handle_tool_error is True

    def test_custom_max_results(self) -> None:
        tool = ScavioRedditSearch(scavio_api_key=MOCK_API_KEY, max_results=10)
        assert tool.max_results == 10

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioRedditSearch(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioRedditSearch(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioRedditSearch()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestRedditSearchRun:
    @responses.activate
    def test_successful_search(self, reddit_search_tool: ScavioRedditSearch) -> None:
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_reddit_search_response(),
            status=200,
        )
        result = reddit_search_tool.invoke({"query": "langchain"})
        assert "data" in result
        assert len(result["data"]["posts"]) == 5  # truncated by max_results

    @responses.activate
    def test_max_results_truncation(self) -> None:
        tool = ScavioRedditSearch(scavio_api_key=MOCK_API_KEY, max_results=3)
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_reddit_search_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert len(result["data"]["posts"]) == 3

    @responses.activate
    def test_empty_results_raises_tool_exception(
        self, reddit_search_tool: ScavioRedditSearch
    ) -> None:
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_reddit_search_response(
                data={
                    "searchQuery": "xyzzy",
                    "totalResults": 0,
                    "nextCursor": None,
                    "posts": [],
                }
            ),
            status=200,
        )
        result = reddit_search_tool.invoke({"query": "xyzzy obscure"})
        assert "No Reddit results found" in result

    @responses.activate
    def test_api_error_returns_error_dict(
        self, reddit_search_tool: ScavioRedditSearch
    ) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, SEARCH_ENDPOINT, json=error, status=401)
        result = reddit_search_tool.invoke({"query": "test"})
        assert "error" in str(result).lower()

    @responses.activate
    def test_sort_and_type_forwarded(
        self, reddit_search_tool: ScavioRedditSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_reddit_search_response(),
            status=200,
        )
        reddit_search_tool.invoke(
            {"query": "python", "sort": "top", "type": "posts"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["query"] == "python"
        assert body["sort"] == "top"
        assert body["type"] == "posts"

    @responses.activate
    def test_cursor_forwarded(self, reddit_search_tool: ScavioRedditSearch) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_reddit_search_response(),
            status=200,
        )
        reddit_search_tool.invoke({"query": "python", "cursor": "abc123"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["cursor"] == "abc123"

    @responses.activate
    def test_none_fields_not_sent(
        self, reddit_search_tool: ScavioRedditSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_reddit_search_response(),
            status=200,
        )
        reddit_search_tool.invoke({"query": "python"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert "sort" not in body
        assert "type" not in body
        assert "cursor" not in body


class TestRedditSearchForbiddenParams:
    def test_init_only_params_rejected_at_invocation(
        self, reddit_search_tool: ScavioRedditSearch
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            reddit_search_tool._run(query="test", max_results=10)


class TestRedditSearchAsync:
    @pytest.mark.asyncio
    async def test_async_search(
        self, reddit_search_tool: ScavioRedditSearch
    ) -> None:
        mock_resp = make_reddit_search_response()
        with patch(
            "langchain_scavio._utilities.ScavioRedditSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await reddit_search_tool.ainvoke({"query": "async test"})
            assert "data" in result
            assert len(result["data"]["posts"]) == 5

    @pytest.mark.asyncio
    async def test_async_empty_results(
        self, reddit_search_tool: ScavioRedditSearch
    ) -> None:
        mock_resp = make_reddit_search_response(
            data={
                "searchQuery": "xyzzy",
                "totalResults": 0,
                "nextCursor": None,
                "posts": [],
            }
        )
        with patch(
            "langchain_scavio._utilities.ScavioRedditSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await reddit_search_tool.ainvoke({"query": "xyzzy"})
            assert "No Reddit results found" in result


class TestRedditSearchInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioRedditSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        props = input_schema["properties"]
        assert "query" in props
        assert "type" in props
        assert "sort" in props
        assert "cursor" in props

    def test_query_is_required(self) -> None:
        tool = ScavioRedditSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "query" in input_schema.get("required", [])


class TestRedditPostInstantiation:
    def test_default_params(self, reddit_post_tool: ScavioRedditPost) -> None:
        assert reddit_post_tool.name == "scavio_reddit_post"
        assert reddit_post_tool.handle_tool_error is True

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioRedditPost(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioRedditPost(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioRedditPost()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestRedditPostRun:
    @responses.activate
    def test_successful_post_lookup(
        self, reddit_post_tool: ScavioRedditPost
    ) -> None:
        responses.add(
            responses.POST,
            POST_ENDPOINT,
            json=make_reddit_post_response(),
            status=200,
        )
        result = reddit_post_tool.invoke(
            {"url": "https://www.reddit.com/r/programming/comments/abc123/ex/"}
        )
        assert "data" in result
        assert result["data"]["post"]["id"] == "t3_abc123"
        assert len(result["data"]["comments"]) == 2

    @responses.activate
    def test_empty_post_raises_tool_exception(
        self, reddit_post_tool: ScavioRedditPost
    ) -> None:
        responses.add(
            responses.POST,
            POST_ENDPOINT,
            json=make_reddit_post_response(data={"post": None, "comments": []}),
            status=200,
        )
        result = reddit_post_tool.invoke(
            {"url": "https://www.reddit.com/r/test/comments/zzz/"}
        )
        assert "No Reddit post found" in result

    @responses.activate
    def test_api_error_returns_error_dict(
        self, reddit_post_tool: ScavioRedditPost
    ) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, POST_ENDPOINT, json=error, status=401)
        result = reddit_post_tool.invoke(
            {"url": "https://www.reddit.com/r/test/comments/abc/"}
        )
        assert "error" in str(result).lower()

    @responses.activate
    def test_url_forwarded(self, reddit_post_tool: ScavioRedditPost) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            POST_ENDPOINT,
            json=make_reddit_post_response(),
            status=200,
        )
        url = "https://www.reddit.com/r/programming/comments/abc123/example_post/"
        reddit_post_tool.invoke({"url": url})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["url"] == url


class TestRedditPostAsync:
    @pytest.mark.asyncio
    async def test_async_post_lookup(
        self, reddit_post_tool: ScavioRedditPost
    ) -> None:
        mock_resp = make_reddit_post_response()
        with patch(
            "langchain_scavio._utilities.ScavioRedditPostAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await reddit_post_tool.ainvoke(
                {"url": "https://www.reddit.com/r/programming/comments/abc123/ex/"}
            )
            assert "data" in result

    @pytest.mark.asyncio
    async def test_async_empty_post(
        self, reddit_post_tool: ScavioRedditPost
    ) -> None:
        mock_resp = make_reddit_post_response(data={"post": None, "comments": []})
        with patch(
            "langchain_scavio._utilities.ScavioRedditPostAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await reddit_post_tool.ainvoke(
                {"url": "https://www.reddit.com/r/test/comments/zzz/"}
            )
            assert "No Reddit post found" in result


class TestRedditPostInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioRedditPost(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "url" in input_schema["properties"]

    def test_url_is_required(self) -> None:
        tool = ScavioRedditPost(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "url" in input_schema.get("required", [])
