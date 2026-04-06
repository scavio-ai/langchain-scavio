"""Tests for ScavioYouTubeSearch and ScavioYouTubeMetadata."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio._utilities import SCAVIO_API_URL
from langchain_scavio.scavio_youtube import (
    ScavioYouTubeMetadata,
    ScavioYouTubeSearch,
)

from .conftest import (
    MOCK_API_KEY,
    make_error_response,
    make_youtube_metadata_response,
    make_youtube_search_response,
)

SEARCH_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/search"
METADATA_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/metadata"


class TestYouTubeSearchInstantiation:
    def test_default_params(self, youtube_search_tool: ScavioYouTubeSearch) -> None:
        assert youtube_search_tool.name == "scavio_youtube_search"
        assert youtube_search_tool.max_results == 5
        assert youtube_search_tool.fourk is None
        assert youtube_search_tool.hdr is None
        assert youtube_search_tool.handle_tool_error is True

    def test_custom_params(self) -> None:
        tool = ScavioYouTubeSearch(
            scavio_api_key=MOCK_API_KEY,
            max_results=10,
            fourk=True,
            hdr=False,
        )
        assert tool.max_results == 10
        assert tool.fourk is True
        assert tool.hdr is False

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioYouTubeSearch(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioYouTubeSearch(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioYouTubeSearch()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestYouTubeSearchRun:
    @responses.activate
    def test_successful_search(self, youtube_search_tool: ScavioYouTubeSearch) -> None:
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(),
            status=200,
        )
        result = youtube_search_tool.invoke({"query": "python tutorial"})
        assert "data" in result
        assert len(result["data"]["results"]) == 5  # truncated by max_results

    @responses.activate
    def test_max_results_truncation(self) -> None:
        tool = ScavioYouTubeSearch(scavio_api_key=MOCK_API_KEY, max_results=3)
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert len(result["data"]["results"]) == 3

    @responses.activate
    def test_empty_results_raises_tool_exception(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(data={"results": []}),
            status=200,
        )
        result = youtube_search_tool.invoke({"query": "xyzzy obscure"})
        assert "No YouTube results found" in result

    @responses.activate
    def test_api_error_returns_error_dict(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, SEARCH_ENDPOINT, json=error, status=401)
        result = youtube_search_tool.invoke({"query": "test"})
        assert "error" in str(result).lower()

    @responses.activate
    def test_query_mapped_to_search_field(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(),
            status=200,
        )
        youtube_search_tool.invoke({"query": "python async"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["search"] == "python async"
        assert "query" not in body

    @responses.activate
    def test_duration_filter_forwarded(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(),
            status=200,
        )
        youtube_search_tool.invoke({"query": "lecture", "duration": "long"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["duration"] == "long"

    @responses.activate
    def test_video_type_mapped_to_type_field(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(),
            status=200,
        )
        youtube_search_tool.invoke({"query": "playlist", "video_type": "playlist"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["type"] == "playlist"
        assert "video_type" not in body


class TestYouTubeSearchForbiddenParams:
    def test_init_only_params_rejected_at_invocation(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            youtube_search_tool._run(query="test", max_results=10)


class TestYouTubeSearchAsync:
    @pytest.mark.asyncio
    async def test_async_search(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        mock_resp = make_youtube_search_response()
        with patch(
            "langchain_scavio._utilities.ScavioYouTubeSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await youtube_search_tool.ainvoke({"query": "async test"})
            assert "data" in result
            assert len(result["data"]["results"]) == 5

    @pytest.mark.asyncio
    async def test_async_empty_results(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        mock_resp = make_youtube_search_response(data={"results": []})
        with patch(
            "langchain_scavio._utilities.ScavioYouTubeSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await youtube_search_tool.ainvoke({"query": "xyzzy"})
            assert "No YouTube results found" in result


class TestYouTubeSearchInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioYouTubeSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        props = input_schema["properties"]
        assert "query" in props
        assert "upload_date" in props
        assert "duration" in props
        assert "sort_by" in props
        assert "video_type" in props

    def test_query_is_required(self) -> None:
        tool = ScavioYouTubeSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "query" in input_schema.get("required", [])


class TestYouTubeMetadataInstantiation:
    def test_default_params(
        self, youtube_metadata_tool: ScavioYouTubeMetadata
    ) -> None:
        assert youtube_metadata_tool.name == "scavio_youtube_metadata"
        assert youtube_metadata_tool.handle_tool_error is True

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioYouTubeMetadata(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioYouTubeMetadata(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioYouTubeMetadata()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestYouTubeMetadataRun:
    @responses.activate
    def test_successful_metadata_lookup(
        self, youtube_metadata_tool: ScavioYouTubeMetadata
    ) -> None:
        responses.add(
            responses.POST,
            METADATA_ENDPOINT,
            json=make_youtube_metadata_response(),
            status=200,
        )
        result = youtube_metadata_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert "data" in result
        assert result["data"]["title"] == "Test Video Title"

    @responses.activate
    def test_empty_metadata_raises_tool_exception(
        self, youtube_metadata_tool: ScavioYouTubeMetadata
    ) -> None:
        responses.add(
            responses.POST,
            METADATA_ENDPOINT,
            json=make_youtube_metadata_response(data=None),
            status=200,
        )
        result = youtube_metadata_tool.invoke({"video_id": "invalidid"})
        assert "No metadata found" in result

    @responses.activate
    def test_api_error_returns_error_dict(
        self, youtube_metadata_tool: ScavioYouTubeMetadata
    ) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, METADATA_ENDPOINT, json=error, status=401)
        result = youtube_metadata_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert "error" in str(result).lower()

    @responses.activate
    def test_video_id_forwarded(
        self, youtube_metadata_tool: ScavioYouTubeMetadata
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            METADATA_ENDPOINT,
            json=make_youtube_metadata_response(),
            status=200,
        )
        youtube_metadata_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["video_id"] == "dQw4w9WgXcQ"


class TestYouTubeMetadataAsync:
    @pytest.mark.asyncio
    async def test_async_metadata_lookup(
        self, youtube_metadata_tool: ScavioYouTubeMetadata
    ) -> None:
        mock_resp = make_youtube_metadata_response()
        with patch(
            "langchain_scavio._utilities.ScavioYouTubeMetadataAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await youtube_metadata_tool.ainvoke({"video_id": "dQw4w9WgXcQ"})
            assert "data" in result

    @pytest.mark.asyncio
    async def test_async_empty_metadata(
        self, youtube_metadata_tool: ScavioYouTubeMetadata
    ) -> None:
        mock_resp = make_youtube_metadata_response(data=None)
        with patch(
            "langchain_scavio._utilities.ScavioYouTubeMetadataAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await youtube_metadata_tool.ainvoke({"video_id": "invalidid"})
            assert "No metadata found" in result


class TestYouTubeMetadataInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioYouTubeMetadata(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "video_id" in input_schema["properties"]

    def test_video_id_is_required(self) -> None:
        tool = ScavioYouTubeMetadata(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "video_id" in input_schema.get("required", [])
