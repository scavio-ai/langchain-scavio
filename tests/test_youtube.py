"""Tests for the Scavio YouTube tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio._utilities import SCAVIO_API_URL
from langchain_scavio.scavio_youtube import (
    ScavioYouTubeChannel,
    ScavioYouTubeChannelVideos,
    ScavioYouTubeComments,
    ScavioYouTubeMetadata,
    ScavioYouTubeSearch,
    ScavioYouTubeStreams,
    ScavioYouTubeTranscript,
    ScavioYouTubeVideo,
)

from .conftest import (
    MOCK_API_KEY,
    make_error_response,
    make_youtube_channel_response,
    make_youtube_channel_videos_response,
    make_youtube_comments_response,
    make_youtube_metadata_response,
    make_youtube_search_response,
    make_youtube_streams_response,
    make_youtube_transcript_response,
    make_youtube_video_response,
)

SEARCH_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/search"
VIDEO_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/video"
# Metadata is a deprecated alias of the video endpoint.
METADATA_ENDPOINT = VIDEO_ENDPOINT
COMMENTS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/comments"
TRANSCRIPT_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/transcript"
CHANNEL_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/channel"
CHANNEL_VIDEOS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/channel/videos"
STREAMS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/youtube/streams"


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

    @responses.activate
    def test_features_forwarded(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(),
            status=200,
        )
        youtube_search_tool.invoke(
            {"query": "lecture", "features": ["hd", "subtitles"]}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["features"] == ["hd", "subtitles"]

    @responses.activate
    def test_cursor_forwarded(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(),
            status=200,
        )
        youtube_search_tool.invoke({"query": "lecture", "cursor": "PAGE2TOKEN"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["cursor"] == "PAGE2TOKEN"

    @responses.activate
    def test_legacy_boolean_flags_still_sent(
        self, youtube_search_tool: ScavioYouTubeSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_youtube_search_response(),
            status=200,
        )
        youtube_search_tool.invoke(
            {"query": "lecture", "hd": True, "creative_commons": True}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["hd"] is True
        assert body["creative_commons"] is True


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
        assert "features" in props
        assert "cursor" in props

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


class TestYouTubeMetadataAliasesVideoEndpoint:
    @responses.activate
    def test_metadata_hits_video_endpoint(
        self, youtube_metadata_tool: ScavioYouTubeMetadata
    ) -> None:
        responses.add(
            responses.POST,
            VIDEO_ENDPOINT,
            json=make_youtube_video_response(),
            status=200,
        )
        result = youtube_metadata_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert result["data"]["video_id"] == "dQw4w9WgXcQ"
        assert responses.calls[0].request.url == VIDEO_ENDPOINT


class TestYouTubeVideo:
    def test_default_params(self, youtube_video_tool: ScavioYouTubeVideo) -> None:
        assert youtube_video_tool.name == "scavio_youtube_video"
        assert youtube_video_tool.handle_tool_error is True

    def test_forwards_rate_limit(self) -> None:
        tool = ScavioYouTubeVideo(
            scavio_api_key=MOCK_API_KEY, max_requests_per_second=10
        )
        assert tool.api_wrapper.max_requests_per_second == 10

    @responses.activate
    def test_successful_lookup(
        self, youtube_video_tool: ScavioYouTubeVideo
    ) -> None:
        responses.add(
            responses.POST,
            VIDEO_ENDPOINT,
            json=make_youtube_video_response(),
            status=200,
        )
        result = youtube_video_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert result["data"]["title"] == "Test Video Title"
        assert "captions" in result["data"]

    @responses.activate
    def test_video_id_forwarded(
        self, youtube_video_tool: ScavioYouTubeVideo
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            VIDEO_ENDPOINT,
            json=make_youtube_video_response(),
            status=200,
        )
        youtube_video_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["video_id"] == "dQw4w9WgXcQ"

    @responses.activate
    def test_empty_raises_tool_exception(
        self, youtube_video_tool: ScavioYouTubeVideo
    ) -> None:
        responses.add(
            responses.POST,
            VIDEO_ENDPOINT,
            json=make_youtube_video_response(data=None),
            status=200,
        )
        result = youtube_video_tool.invoke({"video_id": "bad"})
        assert "No details found" in result

    @pytest.mark.asyncio
    async def test_async_lookup(
        self, youtube_video_tool: ScavioYouTubeVideo
    ) -> None:
        mock_resp = make_youtube_video_response()
        with patch(
            "langchain_scavio._utilities.ScavioYouTubeVideoAPIWrapper."
            "raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await youtube_video_tool.ainvoke({"video_id": "dQw4w9WgXcQ"})
            assert "data" in result


class TestYouTubeComments:
    def test_default_params(
        self, youtube_comments_tool: ScavioYouTubeComments
    ) -> None:
        assert youtube_comments_tool.name == "scavio_youtube_comments"
        assert youtube_comments_tool.max_results == 10

    @responses.activate
    def test_successful_lookup_truncates(
        self, youtube_comments_tool: ScavioYouTubeComments
    ) -> None:
        responses.add(
            responses.POST,
            COMMENTS_ENDPOINT,
            json=make_youtube_comments_response(),
            status=200,
        )
        result = youtube_comments_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert len(result["data"]["comments"]) == 10  # truncated by max_results

    @responses.activate
    def test_cursor_forwarded(
        self, youtube_comments_tool: ScavioYouTubeComments
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            COMMENTS_ENDPOINT,
            json=make_youtube_comments_response(),
            status=200,
        )
        youtube_comments_tool.invoke(
            {"video_id": "dQw4w9WgXcQ", "cursor": "NEXT"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["video_id"] == "dQw4w9WgXcQ"
        assert body["cursor"] == "NEXT"

    @responses.activate
    def test_empty_raises_tool_exception(
        self, youtube_comments_tool: ScavioYouTubeComments
    ) -> None:
        responses.add(
            responses.POST,
            COMMENTS_ENDPOINT,
            json=make_youtube_comments_response(data={"comments": []}),
            status=200,
        )
        result = youtube_comments_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert "No comments found" in result

    def test_init_only_param_rejected(
        self, youtube_comments_tool: ScavioYouTubeComments
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            youtube_comments_tool._run(video_id="x", max_results=3)


class TestYouTubeTranscript:
    def test_default_params(
        self, youtube_transcript_tool: ScavioYouTubeTranscript
    ) -> None:
        assert youtube_transcript_tool.name == "scavio_youtube_transcript"

    @responses.activate
    def test_successful_lookup(
        self, youtube_transcript_tool: ScavioYouTubeTranscript
    ) -> None:
        responses.add(
            responses.POST,
            TRANSCRIPT_ENDPOINT,
            json=make_youtube_transcript_response(),
            status=200,
        )
        result = youtube_transcript_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert result["data"]["content"]

    @responses.activate
    def test_language_and_format_forwarded(
        self, youtube_transcript_tool: ScavioYouTubeTranscript
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            TRANSCRIPT_ENDPOINT,
            json=make_youtube_transcript_response(),
            status=200,
        )
        youtube_transcript_tool.invoke(
            {"video_id": "dQw4w9WgXcQ", "language": "es", "format": "srt"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["language"] == "es"
        assert body["format"] == "srt"

    @responses.activate
    def test_empty_raises_tool_exception(
        self, youtube_transcript_tool: ScavioYouTubeTranscript
    ) -> None:
        responses.add(
            responses.POST,
            TRANSCRIPT_ENDPOINT,
            json=make_youtube_transcript_response(data={"content": ""}),
            status=200,
        )
        result = youtube_transcript_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert "No transcript found" in result


class TestYouTubeChannel:
    def test_default_params(
        self, youtube_channel_tool: ScavioYouTubeChannel
    ) -> None:
        assert youtube_channel_tool.name == "scavio_youtube_channel"

    @responses.activate
    def test_successful_lookup(
        self, youtube_channel_tool: ScavioYouTubeChannel
    ) -> None:
        responses.add(
            responses.POST,
            CHANNEL_ENDPOINT,
            json=make_youtube_channel_response(),
            status=200,
        )
        result = youtube_channel_tool.invoke({"channel_id": "@TestChannel"})
        assert result["data"]["title"] == "Test Channel"

    @responses.activate
    def test_channel_id_forwarded(
        self, youtube_channel_tool: ScavioYouTubeChannel
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            CHANNEL_ENDPOINT,
            json=make_youtube_channel_response(),
            status=200,
        )
        youtube_channel_tool.invoke({"channel_id": "@TestChannel"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["channel_id"] == "@TestChannel"

    @responses.activate
    def test_empty_raises_tool_exception(
        self, youtube_channel_tool: ScavioYouTubeChannel
    ) -> None:
        responses.add(
            responses.POST,
            CHANNEL_ENDPOINT,
            json=make_youtube_channel_response(data=None),
            status=200,
        )
        result = youtube_channel_tool.invoke({"channel_id": "bad"})
        assert "No channel found" in result


class TestYouTubeChannelVideos:
    def test_default_params(
        self, youtube_channel_videos_tool: ScavioYouTubeChannelVideos
    ) -> None:
        assert (
            youtube_channel_videos_tool.name == "scavio_youtube_channel_videos"
        )
        assert youtube_channel_videos_tool.max_results == 5

    @responses.activate
    def test_successful_lookup_truncates(
        self, youtube_channel_videos_tool: ScavioYouTubeChannelVideos
    ) -> None:
        responses.add(
            responses.POST,
            CHANNEL_VIDEOS_ENDPOINT,
            json=make_youtube_channel_videos_response(),
            status=200,
        )
        result = youtube_channel_videos_tool.invoke(
            {"channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw"}
        )
        assert len(result["data"]["results"]) == 5  # truncated by max_results

    @responses.activate
    def test_cursor_forwarded(
        self, youtube_channel_videos_tool: ScavioYouTubeChannelVideos
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            CHANNEL_VIDEOS_ENDPOINT,
            json=make_youtube_channel_videos_response(),
            status=200,
        )
        youtube_channel_videos_tool.invoke(
            {"channel_id": "UCx", "cursor": "NEXT"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["channel_id"] == "UCx"
        assert body["cursor"] == "NEXT"

    @responses.activate
    def test_empty_raises_tool_exception(
        self, youtube_channel_videos_tool: ScavioYouTubeChannelVideos
    ) -> None:
        responses.add(
            responses.POST,
            CHANNEL_VIDEOS_ENDPOINT,
            json=make_youtube_channel_videos_response(data={"results": []}),
            status=200,
        )
        result = youtube_channel_videos_tool.invoke({"channel_id": "UCx"})
        assert "No videos found" in result


class TestYouTubeStreams:
    def test_default_params(
        self, youtube_streams_tool: ScavioYouTubeStreams
    ) -> None:
        assert youtube_streams_tool.name == "scavio_youtube_streams"

    @responses.activate
    def test_successful_lookup(
        self, youtube_streams_tool: ScavioYouTubeStreams
    ) -> None:
        responses.add(
            responses.POST,
            STREAMS_ENDPOINT,
            json=make_youtube_streams_response(),
            status=200,
        )
        result = youtube_streams_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert result["data"]["formats"]

    @responses.activate
    def test_empty_raises_tool_exception(
        self, youtube_streams_tool: ScavioYouTubeStreams
    ) -> None:
        responses.add(
            responses.POST,
            STREAMS_ENDPOINT,
            json=make_youtube_streams_response(
                data={"formats": [], "adaptive_formats": []}
            ),
            status=200,
        )
        result = youtube_streams_tool.invoke({"video_id": "dQw4w9WgXcQ"})
        assert "No streams found" in result

    @pytest.mark.asyncio
    async def test_async_lookup(
        self, youtube_streams_tool: ScavioYouTubeStreams
    ) -> None:
        mock_resp = make_youtube_streams_response()
        with patch(
            "langchain_scavio._utilities.ScavioYouTubeStreamsAPIWrapper."
            "raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await youtube_streams_tool.ainvoke(
                {"video_id": "dQw4w9WgXcQ"}
            )
            assert "data" in result
