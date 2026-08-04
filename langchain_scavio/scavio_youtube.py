"""Scavio YouTube tools for LangChain agents."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import (
    ScavioYouTubeChannelAPIWrapper,
    ScavioYouTubeChannelCommunityAPIWrapper,
    ScavioYouTubeChannelResolveAPIWrapper,
    ScavioYouTubeChannelSearchAPIWrapper,
    ScavioYouTubeChannelShortsAPIWrapper,
    ScavioYouTubeChannelVideosAPIWrapper,
    ScavioYouTubeCommentRepliesAPIWrapper,
    ScavioYouTubeCommentsAPIWrapper,
    ScavioYouTubeMetadataAPIWrapper,
    ScavioYouTubeRelatedAPIWrapper,
    ScavioYouTubeSearchAPIWrapper,
    ScavioYouTubeShortsAPIWrapper,
    ScavioYouTubeStreamsAPIWrapper,
    ScavioYouTubeSuggestionsAPIWrapper,
    ScavioYouTubeTranscriptAPIWrapper,
    ScavioYouTubeVideoAPIWrapper,
)

logger = logging.getLogger(__name__)

# The feature flags moved into the args_schema in 3.4 under their API names
# (four_k, video_360, video_3d, hdr, vr180). The pre-3.4 constructor spellings
# below still work as per-tool defaults but remain rejected at invocation, so a
# model that guesses the old name gets told where the flag lives instead of
# having it silently dropped.
_SEARCH_INIT_ONLY_PARAMS = frozenset(
    {"max_results", "fourk", "three_sixty", "threed"}
)

_LIST_INIT_ONLY_PARAMS = frozenset({"max_results"})


def _forward_api_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract API wrapper kwargs from tool kwargs."""
    api_kwargs: dict[str, Any] = {}
    if "scavio_api_key" in kwargs:
        api_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
    if "api_base_url" in kwargs:
        api_kwargs["api_base_url"] = kwargs.pop("api_base_url")
    if "max_requests_per_second" in kwargs:
        api_kwargs["max_requests_per_second"] = kwargs.pop("max_requests_per_second")
    return api_kwargs

class ScavioYouTubeSearchInput(BaseModel):
    """Input schema for ScavioYouTubeSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(description="YouTube search query (e.g., 'python tutorial')")

    upload_date: Optional[
        Literal["last_hour", "today", "this_week", "this_month", "this_year"]
    ] = Field(
        default=None,
        description=(
            "Filter by upload date. Options: last_hour, today, this_week, "
            "this_month, this_year."
        ),
    )

    type: Optional[Literal["video", "channel", "playlist", "movie"]] = Field(
        default=None,
        description=(
            "Filter by result type. Options: video, channel, playlist, movie."
        ),
    )

    video_type: Optional[Literal["video", "channel", "playlist"]] = Field(
        default=None,
        description=(
            "Deprecated alias of type, kept for backwards compatibility. "
            "Prefer type; when both are given type wins."
        ),
    )

    duration: Optional[Literal["short", "medium", "long"]] = Field(
        default=None,
        description=(
            "Filter by video duration. Options: short (<4 min), "
            "medium (4-20 min), long (>20 min)."
        ),
    )

    sort_by: Optional[
        Literal["relevance", "date", "view_count", "rating"]
    ] = Field(
        default=None,
        description=(
            "Sort order for results. Options: relevance (default), date, "
            "view_count, rating."
        ),
    )

    hd: Optional[bool] = Field(
        default=None,
        description="Filter for HD videos only.",
    )

    subtitles: Optional[bool] = Field(
        default=None,
        description="Filter for videos with subtitles/closed captions.",
    )

    creative_commons: Optional[bool] = Field(
        default=None,
        description="Filter for Creative Commons licensed videos only.",
    )

    live: Optional[bool] = Field(
        default=None,
        description="Filter for live streams only.",
    )

    four_k: Optional[bool] = Field(
        default=None,
        description="Filter for 4K videos only. Sent on the wire as '4k'.",
    )

    hdr: Optional[bool] = Field(
        default=None,
        description="Filter for HDR videos only.",
    )

    video_360: Optional[bool] = Field(
        default=None,
        description=(
            "Filter for 360-degree videos only. Sent on the wire as '360'."
        ),
    )

    video_3d: Optional[bool] = Field(
        default=None,
        description="Filter for 3D videos only. Sent on the wire as '3d'.",
    )

    vr180: Optional[bool] = Field(
        default=None,
        description="Filter for VR180 videos only.",
    )

    location: Optional[bool] = Field(
        default=None,
        description="Filter for videos tagged with a location.",
    )

    features: Optional[
        list[
            Literal[
                "hd",
                "4k",
                "subtitles",
                "creative_commons",
                "live",
                "360",
                "3d",
                "hdr",
                "vr180",
            ]
        ]
    ] = Field(
        default=None,
        description=(
            "Granular feature filters to require, as a list. Options: hd, 4k, "
            "subtitles, creative_commons, live, 360, 3d, hdr, vr180. "
            "An alternative to the individual boolean flags."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. "
            "Keep the query and filters the same across paginated calls."
        ),
    )


class ScavioYouTubeSearch(BaseTool):  # type: ignore[override]
    """Search YouTube videos using the Scavio API.

    Returns video titles, channels, view counts, durations, and thumbnails.
    Supports filtering by upload date, duration, result type, sort order and
    the per-feature flags (hd, four_k, hdr, subtitles, creative_commons, live,
    video_360, video_3d, vr180), all settable per call.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeSearch

            tool = ScavioYouTubeSearch(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "python tutorial", "duration": "medium"})
    """

    name: str = "scavio_youtube_search"
    description: str = (
        "Search YouTube videos using the Scavio API. "
        "Returns video titles, channels, view counts, durations, and video IDs. "
        "Supports filtering by upload date, duration, result type (video, channel, "
        "playlist, movie), sort order and feature flags such as hd, four_k, hdr, "
        "subtitles, creative_commons, live, video_360, video_3d and vr180. "
        "Page with cursor, taken from the previous response's next_cursor. "
        "Input should be a search query. "
        "Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeSearchInput
    handle_tool_error: bool = True

    # max_results is instantiation-only. The feature flags below are the
    # pre-3.4 constructor spellings of four_k/hdr/video_360/video_3d/vr180:
    # they now act as defaults for the matching args_schema fields.
    max_results: Optional[int] = 5
    fourk: Optional[bool] = None
    hdr: Optional[bool] = None
    three_sixty: Optional[bool] = None
    threed: Optional[bool] = None
    vr180: Optional[bool] = None

    api_wrapper: ScavioYouTubeSearchAPIWrapper = Field(
        default_factory=ScavioYouTubeSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_wrapper_kwargs: dict[str, Any] = {}
        if "scavio_api_key" in kwargs:
            api_wrapper_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
        if "api_base_url" in kwargs:
            api_wrapper_kwargs["api_base_url"] = kwargs.pop("api_base_url")
        if "max_requests_per_second" in kwargs:
            api_wrapper_kwargs["max_requests_per_second"] = kwargs.pop(
                "max_requests_per_second"
            )
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeSearchAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _build_params(
        self,
        query: str,
        upload_date: Optional[str],
        type: Optional[str],
        video_type: Optional[str],
        duration: Optional[str],
        sort_by: Optional[str],
        hd: Optional[bool],
        subtitles: Optional[bool],
        creative_commons: Optional[bool],
        live: Optional[bool],
        four_k: Optional[bool],
        hdr: Optional[bool],
        video_360: Optional[bool],
        video_3d: Optional[bool],
        vr180: Optional[bool],
        location: Optional[bool],
        features: Optional[list[str]],
        cursor: Optional[str],
    ) -> dict[str, Any]:
        """Map tool arguments onto the wire body.

        The wire spells the feature flags ``4k``/``360``/``3d``, which are not
        valid Python identifiers, and the result filter ``type``. Per-call
        values win; the constructor attributes are the fallback so a tool
        pinned to, say, 4K at instantiation keeps that default.
        """
        return {
            "search": query,
            "upload_date": upload_date,
            "type": type or video_type,
            "duration": duration,
            "sort_by": sort_by,
            "hd": hd,
            "subtitles": subtitles,
            "creative_commons": creative_commons,
            "live": live,
            "location": location,
            "features": features,
            "cursor": cursor,
            "4k": self.fourk if four_k is None else four_k,
            "hdr": self.hdr if hdr is None else hdr,
            "360": self.three_sixty if video_360 is None else video_360,
            "3d": self.threed if video_3d is None else video_3d,
            "vr180": self.vr180 if vr180 is None else vr180,
        }

    def _run(
        self,
        query: str,
        upload_date: Optional[str] = None,
        type: Optional[str] = None,
        video_type: Optional[str] = None,
        duration: Optional[str] = None,
        sort_by: Optional[str] = None,
        hd: Optional[bool] = None,
        subtitles: Optional[bool] = None,
        creative_commons: Optional[bool] = None,
        live: Optional[bool] = None,
        four_k: Optional[bool] = None,
        hdr: Optional[bool] = None,
        video_360: Optional[bool] = None,
        video_3d: Optional[bool] = None,
        vr180: Optional[bool] = None,
        location: Optional[bool] = None,
        features: Optional[list[str]] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a synchronous YouTube video search."""
        forbidden = _SEARCH_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            params = self._build_params(
                query, upload_date, type, video_type, duration, sort_by, hd,
                subtitles, creative_commons, live, four_k, hdr, video_360,
                video_3d, vr180, location, features, cursor,
            )
            raw = self.api_wrapper.raw_results(**params)
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        upload_date: Optional[str] = None,
        type: Optional[str] = None,
        video_type: Optional[str] = None,
        duration: Optional[str] = None,
        sort_by: Optional[str] = None,
        hd: Optional[bool] = None,
        subtitles: Optional[bool] = None,
        creative_commons: Optional[bool] = None,
        live: Optional[bool] = None,
        four_k: Optional[bool] = None,
        hdr: Optional[bool] = None,
        video_360: Optional[bool] = None,
        video_3d: Optional[bool] = None,
        vr180: Optional[bool] = None,
        location: Optional[bool] = None,
        features: Optional[list[str]] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an asynchronous YouTube video search."""
        forbidden = _SEARCH_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            params = self._build_params(
                query, upload_date, type, video_type, duration, sort_by, hd,
                subtitles, creative_commons, live, four_k, hdr, video_360,
                video_3d, vr180, location, features, cursor,
            )
            raw = await self.api_wrapper.raw_results_async(**params)
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], query: str) -> dict[str, Any]:
        """Truncate results and raise ToolException if empty."""
        data = raw.get("data") or {}
        results = data.get("results") if isinstance(data, dict) else None
        if self.max_results and results:
            raw["data"]["results"] = results[: self.max_results]
        if not (isinstance(data, dict) and data.get("results")):
            raise ToolException(
                f"No YouTube results found for '{query}'. "
                "Try broadening the query or removing filters."
            )
        return raw


class ScavioYouTubeMetadataInput(BaseModel):
    """Input schema for ScavioYouTubeMetadata tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(
        description=(
            "YouTube video ID (the part after 'v=' in the URL, e.g., 'dQw4w9WgXcQ'). "
            "Use ScavioYouTubeSearch first to find video IDs if needed."
        )
    )


class ScavioYouTubeMetadata(BaseTool):  # type: ignore[override]
    """Fetch full metadata for a YouTube video by video ID.

    Deprecated alias of :class:`ScavioYouTubeVideo`; both hit the same
    ``/api/v1/youtube/video`` endpoint. Prefer ScavioYouTubeVideo in new code.

    Returns title, description, view count, channel info, keywords,
    chapters, captions, and the thumbnail URL.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeMetadata

            tool = ScavioYouTubeMetadata()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "dQw4w9WgXcQ"})
    """

    name: str = "scavio_youtube_metadata"
    description: str = (
        "Fetch full metadata for a YouTube video by video ID. "
        "Deprecated alias of scavio_youtube_video; prefer that tool. "
        "Returns title, description, view count, channel info, and keywords. "
        "Use ScavioYouTubeSearch to find video IDs. "
        "Input should be a YouTube video ID or watch URL (e.g., 'dQw4w9WgXcQ'). "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeMetadataInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYouTubeMetadataAPIWrapper = Field(
        default_factory=ScavioYouTubeMetadataAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_wrapper_kwargs: dict[str, Any] = {}
        if "scavio_api_key" in kwargs:
            api_wrapper_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
        if "api_base_url" in kwargs:
            api_wrapper_kwargs["api_base_url"] = kwargs.pop("api_base_url")
        if "max_requests_per_second" in kwargs:
            api_wrapper_kwargs["max_requests_per_second"] = kwargs.pop(
                "max_requests_per_second"
            )
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeMetadataAPIWrapper(
                **api_wrapper_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch YouTube video metadata synchronously."""
        try:
            raw = self.api_wrapper.raw_results(video_id=video_id)
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        video_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch YouTube video metadata asynchronously."""
        try:
            raw = await self.api_wrapper.raw_results_async(video_id=video_id)
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], video_id: str
    ) -> dict[str, Any]:
        """Raise ToolException if no metadata returned."""
        if not raw.get("data"):
            raise ToolException(
                f"No metadata found for YouTube video '{video_id}'. "
                "Verify the video ID is correct and the video is publicly accessible."
            )
        return raw


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class ScavioYouTubeVideoInput(BaseModel):
    """Input schema for ScavioYouTubeVideo tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(
        description=(
            "YouTube video ID (the part after 'v=' in the URL, e.g., "
            "'dQw4w9WgXcQ') or a full watch URL. "
            "Use ScavioYouTubeSearch to find video IDs if needed."
        )
    )


class ScavioYouTubeVideo(BaseTool):  # type: ignore[override]
    """Fetch full details for a YouTube video by video ID or watch URL.

    Returns title, author, channel, publish date, description, length,
    view count, keywords, chapters, captions, and the thumbnail URL.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeVideo

            tool = ScavioYouTubeVideo()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "dQw4w9WgXcQ"})
    """

    name: str = "scavio_youtube_video"
    description: str = (
        "Fetch full details for a YouTube video by video ID or watch URL. "
        "Returns title, author, channel, publish date, description, length, "
        "view count, keywords, chapters, and captions. "
        "Use ScavioYouTubeSearch to find video IDs. "
        "Input should be a YouTube video ID or watch URL. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeVideoInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYouTubeVideoAPIWrapper = Field(
        default_factory=ScavioYouTubeVideoAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeVideoAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch YouTube video details synchronously."""
        try:
            raw = self.api_wrapper.raw_results(video_id=video_id)
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        video_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch YouTube video details asynchronously."""
        try:
            raw = await self.api_wrapper.raw_results_async(video_id=video_id)
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], video_id: str
    ) -> dict[str, Any]:
        """Raise ToolException if no video data returned."""
        if not raw.get("data"):
            raise ToolException(
                f"No details found for YouTube video '{video_id}'. "
                "Verify the video ID is correct and the video is publicly accessible."
            )
        return raw


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


class ScavioYouTubeCommentsInput(BaseModel):
    """Input schema for ScavioYouTubeComments tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(
        description=(
            "YouTube video ID or watch URL. "
            "Use ScavioYouTubeSearch to find video IDs if needed."
        )
    )
    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. "
            "Keep video_id the same across paginated calls."
        ),
    )


class ScavioYouTubeComments(BaseTool):  # type: ignore[override]
    """Fetch comments on a YouTube video.

    Returns a paginated list of comments with text, like/reply counts,
    publish time, and author info. Use ``data.next_cursor`` for pagination;
    stop when ``data.has_more`` is false. A comment's ``reply_cursor`` can be
    used to fetch its replies.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeComments

            tool = ScavioYouTubeComments(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "dQw4w9WgXcQ"})
    """

    name: str = "scavio_youtube_comments"
    description: str = (
        "Fetch comments on a YouTube video. "
        "Returns comment text, like/reply counts, publish time, and author info. "
        "Supports pagination. Input should be a YouTube video ID or watch URL. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeCommentsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioYouTubeCommentsAPIWrapper = Field(
        default_factory=ScavioYouTubeCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(video_id=video_id, cursor=cursor)
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        video_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                video_id=video_id, cursor=cursor
            )
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], video_id: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        comments = data.get("comments") if isinstance(data, dict) else None
        if self.max_results and comments:
            raw["data"]["comments"] = comments[: self.max_results]
        if not (isinstance(data, dict) and data.get("comments")):
            raise ToolException(
                f"No comments found for YouTube video '{video_id}'. "
                "The video may have comments disabled or none yet."
            )
        return raw


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


class ScavioYouTubeTranscriptInput(BaseModel):
    """Input schema for ScavioYouTubeTranscript tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(
        description=(
            "YouTube video ID or watch URL. "
            "Use ScavioYouTubeSearch to find video IDs if needed."
        )
    )
    language: Optional[str] = Field(
        default=None,
        description="Caption language code (ISO 639-1, e.g. 'en'). Defaults to en.",
    )
    format: Optional[Literal["text", "srt"]] = Field(
        default=None,
        description=(
            'Output format. "text" (default) returns a plain transcript; '
            '"srt" returns timed subtitles.'
        ),
    )


class ScavioYouTubeTranscript(BaseTool):  # type: ignore[override]
    """Fetch the transcript (captions) of a YouTube video.

    Returns the transcript as plain text or timed SRT subtitles. Note this
    endpoint costs 8 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeTranscript

            tool = ScavioYouTubeTranscript()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "dQw4w9WgXcQ", "format": "text"})
    """

    name: str = "scavio_youtube_transcript"
    description: str = (
        "Fetch the transcript (captions) of a YouTube video. "
        "Returns plain text (format=text) or timed SRT subtitles (format=srt). "
        "Costs 8 credits per call. "
        "Input should be a YouTube video ID or watch URL."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeTranscriptInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYouTubeTranscriptAPIWrapper = Field(
        default_factory=ScavioYouTubeTranscriptAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeTranscriptAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        language: Optional[str] = None,
        format: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                video_id=video_id, language=language, format=format
            )
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        video_id: str,
        language: Optional[str] = None,
        format: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                video_id=video_id, language=language, format=format
            )
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], video_id: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        content = data.get("content") if isinstance(data, dict) else None
        if not content:
            raise ToolException(
                f"No transcript found for YouTube video '{video_id}'. "
                "The video may not have captions in the requested language."
            )
        return raw


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------


class ScavioYouTubeChannelInput(BaseModel):
    """Input schema for ScavioYouTubeChannel tool."""

    model_config = ConfigDict(extra="allow")

    channel_id: str = Field(
        description=(
            "YouTube channel ID (e.g. 'UC...'), an @handle, or a channel URL."
        )
    )


class ScavioYouTubeChannel(BaseTool):  # type: ignore[override]
    """Fetch details for a YouTube channel by ID, @handle, or URL.

    Returns title, description, subscriber/video/view counts, country,
    creation date, verification status, avatar, banner, and social links.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeChannel

            tool = ScavioYouTubeChannel()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"channel_id": "@YouTube"})
    """

    name: str = "scavio_youtube_channel"
    description: str = (
        "Fetch details for a YouTube channel by ID, @handle, or URL. "
        "Returns title, description, subscriber/video/view counts, country, "
        "creation date, verification status, avatar, banner, and links. "
        "Input should be a channel ID, @handle, or channel URL. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeChannelInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYouTubeChannelAPIWrapper = Field(
        default_factory=ScavioYouTubeChannelAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeChannelAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        channel_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(channel_id=channel_id)
            return self._process_response(raw, channel_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        channel_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(channel_id=channel_id)
            return self._process_response(raw, channel_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], channel_id: str
    ) -> dict[str, Any]:
        if not raw.get("data"):
            raise ToolException(
                f"No channel found for '{channel_id}'. "
                "Verify the channel ID, @handle, or URL is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# Channel Videos
# ---------------------------------------------------------------------------


class ScavioYouTubeChannelVideosInput(BaseModel):
    """Input schema for ScavioYouTubeChannelVideos tool."""

    model_config = ConfigDict(extra="allow")

    channel_id: str = Field(
        description=(
            "YouTube channel ID (e.g. 'UC...'). "
            "Use ScavioYouTubeChannel to resolve an @handle or URL first."
        )
    )
    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. "
            "Keep channel_id the same across paginated calls."
        ),
    )


class ScavioYouTubeChannelVideos(BaseTool):  # type: ignore[override]
    """Fetch a YouTube channel's uploaded videos.

    Returns a paginated list of videos with title, URL, thumbnail, duration,
    view count, publish time, and live status. Use ``data.next_cursor`` for
    pagination; stop when ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeChannelVideos

            tool = ScavioYouTubeChannelVideos(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw"})
    """

    name: str = "scavio_youtube_channel_videos"
    description: str = (
        "Fetch a YouTube channel's uploaded videos. "
        "Returns videos with title, URL, thumbnail, duration, view count, "
        "and publish time. Requires a channel ID. Supports pagination. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeChannelVideosInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioYouTubeChannelVideosAPIWrapper = Field(
        default_factory=ScavioYouTubeChannelVideosAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeChannelVideosAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        channel_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(channel_id=channel_id, cursor=cursor)
            return self._process_response(raw, channel_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        channel_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                channel_id=channel_id, cursor=cursor
            )
            return self._process_response(raw, channel_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], channel_id: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        videos = data.get("results") if isinstance(data, dict) else None
        if self.max_results and videos:
            raw["data"]["results"] = videos[: self.max_results]
        if not (isinstance(data, dict) and data.get("results")):
            raise ToolException(
                f"No videos found for YouTube channel '{channel_id}'. "
                "Verify the channel ID is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------


class ScavioYouTubeStreamsInput(BaseModel):
    """Input schema for ScavioYouTubeStreams tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(
        description=(
            "YouTube video ID or watch URL. "
            "Use ScavioYouTubeSearch to find video IDs if needed."
        )
    )


class ScavioYouTubeStreams(BaseTool):  # type: ignore[override]
    """Fetch playable/downloadable stream URLs for a YouTube video.

    Returns muxed and adaptive formats with direct URLs, mime types,
    bitrates, resolutions, quality labels, and available qualities. Note
    this endpoint costs 3 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeStreams

            tool = ScavioYouTubeStreams()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "dQw4w9WgXcQ"})
    """

    name: str = "scavio_youtube_streams"
    description: str = (
        "Fetch playable/downloadable stream URLs for a YouTube video. "
        "Returns muxed and adaptive formats with direct URLs, mime types, "
        "bitrates, resolutions, and quality labels. Costs 3 credits per call. "
        "Input should be a YouTube video ID or watch URL."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeStreamsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYouTubeStreamsAPIWrapper = Field(
        default_factory=ScavioYouTubeStreamsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeStreamsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(video_id=video_id)
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        video_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(video_id=video_id)
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], video_id: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        has_formats = isinstance(data, dict) and (
            data.get("formats") or data.get("adaptive_formats")
        )
        if not has_formats:
            raise ToolException(
                f"No streams found for YouTube video '{video_id}'. "
                "The video may be unavailable, private, or region-locked."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioYouTubeShorts
# ---------------------------------------------------------------------------


class ScavioYouTubeShortsInput(BaseModel):
    """Input schema for ScavioYouTubeShorts tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Shorts search query (e.g. 'funny cats').",
    )

    sort_by: Optional[Literal["relevance", "date", "view_count", "rating"]] = Field(
        default=None,
        description=(
            "Sort order. Options: relevance (default), date, view_count, rating."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioYouTubeShorts(BaseTool):  # type: ignore[override]
    """Search YouTube Shorts using the Scavio API.

    Returns short-form videos under ``data.results`` with title, url,
    thumbnail, duration, view count and channel. Paginate with
    ``data.next_cursor`` and stop when ``data.has_more`` is false.

    This endpoint has no upload_date, type, duration or feature filters --
    use ScavioYouTubeSearch for those.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeShorts

            tool = ScavioYouTubeShorts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "funny cats", "sort_by": "view_count"})
    """

    name: str = "scavio_youtube_shorts"
    description: str = (
        "Search YouTube Shorts. Returns short-form videos under data.results with "
        "title, url, thumbnail, duration, view count and channel. Supports sort_by and "
        "cursor pagination. Costs 2 credits per call. Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeShortsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioYouTubeShortsAPIWrapper = Field(
        default_factory=ScavioYouTubeShortsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeShortsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        sort_by: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search YouTube Shorts (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                search=query,
                sort_by=sort_by,
                cursor=cursor,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        sort_by: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search YouTube Shorts (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                search=query,
                sort_by=sort_by,
                cursor=cursor,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        results = data.get("results")
        if self.max_results and results:
            raw["data"]["results"] = results[: self.max_results]
        if not results:
            raise ToolException(
                f"No YouTube Shorts found for '{query}'. Try broadening the query or "
                "changing sort_by."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioYouTubeSuggestions
# ---------------------------------------------------------------------------


class ScavioYouTubeSuggestionsInput(BaseModel):
    """Input schema for ScavioYouTubeSuggestions tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Partial search query to autocomplete.",
    )

    language: Optional[str] = Field(
        default=None,
        description="Suggestion language as an ISO 639-1 code (default 'en').",
    )

    region: Optional[str] = Field(
        default=None,
        description=(
            "Region code as ISO 3166-1 alpha-2 (default 'US'). The only geo parameter "
            "in the YouTube family."
        ),
    )


class ScavioYouTubeSuggestions(BaseTool):  # type: ignore[override]
    """Fetch YouTube search autocomplete suggestions.

    Returns the autocomplete list under ``data.suggestions`` (plain strings)
    with ``data.total_count``.

    Useful for keyword expansion before running a real search, and much
    cheaper than searching speculatively.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeSuggestions

            tool = ScavioYouTubeSuggestions()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "python tut", "region": "US"})
    """

    name: str = "scavio_youtube_suggestions"
    description: str = (
        "Fetch YouTube search autocomplete suggestions for a partial query. Returns a "
        "list of suggestion strings under data.suggestions. Use it to expand keywords "
        "before searching. Costs 1 credit per call. Input should be a partial search "
        "query."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeSuggestionsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioYouTubeSuggestionsAPIWrapper = Field(
        default_factory=ScavioYouTubeSuggestionsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeSuggestionsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        language: Optional[str] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch YouTube search autocomplete suggestions (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                search=query,
                language=language,
                region=region,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        language: Optional[str] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch YouTube search autocomplete suggestions (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                search=query,
                language=language,
                region=region,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        suggestions = data.get("suggestions")
        if self.max_results and suggestions:
            raw["data"]["suggestions"] = suggestions[: self.max_results]
        if not suggestions:
            raise ToolException(
                f"No YouTube suggestions found for '{query}'. Try a shorter or more "
                "common prefix."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioYouTubeCommentReplies
# ---------------------------------------------------------------------------


class ScavioYouTubeCommentRepliesInput(BaseModel):
    """Input schema for ScavioYouTubeCommentReplies tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(
        description="YouTube video id or watch URL the comment belongs to.",
    )

    reply_cursor: str = Field(
        description=(
            "The reply_cursor field of a comment returned by ScavioYouTubeComments. "
            "Required -- this endpoint cannot be called from a video id alone."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Cursor for page 2 and beyond. When set it overrides reply_cursor for that "
            "call."
        ),
    )


class ScavioYouTubeCommentReplies(BaseTool):  # type: ignore[override]
    """Fetch the replies to a YouTube comment using the Scavio API.

    Returns replies under ``data.replies`` using the same comment shape as
    ScavioYouTubeComments. Paginate with ``data.next_cursor`` and stop when
    ``data.has_more`` is false.

    Both ``video_id`` and ``reply_cursor`` are required; take
    ``reply_cursor`` from a comment in a ScavioYouTubeComments response.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeCommentReplies

            tool = ScavioYouTubeCommentReplies()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {"video_id": "dQw4w9WgXcQ", "reply_cursor": "Eg0SC2..."}
            )
    """

    name: str = "scavio_youtube_comment_replies"
    description: str = (
        "Fetch replies to a specific YouTube comment. Requires both the video_id and "
        "the reply_cursor taken from a comment returned by the YouTube comments tool. "
        "Returns replies under data.replies. Supports cursor pagination. Costs 1 "
        "credit per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeCommentRepliesInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioYouTubeCommentRepliesAPIWrapper = Field(
        default_factory=ScavioYouTubeCommentRepliesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeCommentRepliesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        reply_cursor: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the replies to a YouTube comment (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                video_id=video_id,
                reply_cursor=reply_cursor,
                cursor=cursor,
            )
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        video_id: str,
        reply_cursor: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the replies to a YouTube comment (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                video_id=video_id,
                reply_cursor=reply_cursor,
                cursor=cursor,
            )
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], video_id: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        replies = data.get("replies")
        if self.max_results and replies:
            raw["data"]["replies"] = replies[: self.max_results]
        if not replies:
            raise ToolException(
                f"No replies found for that comment on YouTube video '{video_id}'. "
                "Verify the reply_cursor came from a comment on this video."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioYouTubeRelated
# ---------------------------------------------------------------------------


class ScavioYouTubeRelatedInput(BaseModel):
    """Input schema for ScavioYouTubeRelated tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(
        description="YouTube video id or watch URL.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor. Note this endpoint returns no next_cursor of its own."
        ),
    )


class ScavioYouTubeRelated(BaseTool):  # type: ignore[override]
    """Fetch videos related to a YouTube video.

    Returns related videos under ``data.results`` with title, url,
    thumbnail, duration, view count, publish time and channel, plus
    ``data.total_count``.

    Unlike the other cursor-taking YouTube endpoints this one returns no
    ``next_cursor`` and no ``has_more``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeRelated

            tool = ScavioYouTubeRelated()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "dQw4w9WgXcQ"})
    """

    name: str = "scavio_youtube_related"
    description: str = (
        "Fetch videos related to a given YouTube video. Returns related videos under "
        "data.results with title, url, thumbnail, duration, view count and channel. "
        "Use it to widen a topic from one seed video. Costs 1 credit per call. Input "
        "should be a video id or watch URL."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeRelatedInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioYouTubeRelatedAPIWrapper = Field(
        default_factory=ScavioYouTubeRelatedAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeRelatedAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch videos related to a YouTube video (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                video_id=video_id,
                cursor=cursor,
            )
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        video_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch videos related to a YouTube video (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                video_id=video_id,
                cursor=cursor,
            )
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], video_id: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        results = data.get("results")
        if self.max_results and results:
            raw["data"]["results"] = results[: self.max_results]
        if not results:
            raise ToolException(
                f"No related videos found for YouTube video '{video_id}'. Verify the "
                "video id and that the video is public."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioYouTubeChannelSearch
# ---------------------------------------------------------------------------


class ScavioYouTubeChannelSearchInput(BaseModel):
    """Input schema for ScavioYouTubeChannelSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Channel search query (e.g. 'mrbeast').",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioYouTubeChannelSearch(BaseTool):  # type: ignore[override]
    """Search YouTube channels using the Scavio API.

    Returns channel briefs under ``data.results``: channel_id, name, handle,
    url, thumbnail, subscriber_count, description and verified. Paginate
    with ``data.next_cursor``.

    Use this when you need to find a channel by name; use
    ScavioYouTubeChannelResolve when you already have an @handle or URL.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeChannelSearch

            tool = ScavioYouTubeChannelSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "mrbeast"})
    """

    name: str = "scavio_youtube_channel_search"
    description: str = (
        "Search for YouTube channels by name. Returns channel briefs under "
        "data.results with channel_id, name, handle, url, subscriber count and "
        "verification. Supports cursor pagination. Costs 1 credit per call. Input "
        "should be a channel name or keyword."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeChannelSearchInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioYouTubeChannelSearchAPIWrapper = Field(
        default_factory=ScavioYouTubeChannelSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeChannelSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search YouTube channels (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                search=query,
                cursor=cursor,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search YouTube channels (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                search=query,
                cursor=cursor,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        results = data.get("results")
        if self.max_results and results:
            raw["data"]["results"] = results[: self.max_results]
        if not results:
            raise ToolException(
                f"No YouTube channels found for '{query}'. Try a different name or "
                "fewer words."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioYouTubeChannelShorts
# ---------------------------------------------------------------------------


class ScavioYouTubeChannelShortsInput(BaseModel):
    """Input schema for ScavioYouTubeChannelShorts tool."""

    model_config = ConfigDict(extra="allow")

    channel_id: str = Field(
        description=(
            "YouTube channel id (e.g. 'UCX6OQ3DkcsbYNE6H8uQQuVA'), an @handle, or a "
            "channel URL."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioYouTubeChannelShorts(BaseTool):  # type: ignore[override]
    """Fetch the Shorts posted by a YouTube channel.

    Returns Shorts under ``data.results`` with video_id, title, url and
    thumbnail, plus ``data.channel_id`` and ``data.total_count``. Paginate
    with ``data.next_cursor``.

    View counts are deliberately omitted here: the upstream field for Shorts
    carries promotional text rather than a reliable number.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeChannelShorts

            tool = ScavioYouTubeChannelShorts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"channel_id": "@MrBeast"})
    """

    name: str = "scavio_youtube_channel_shorts"
    description: str = (
        "Fetch the Shorts posted by a YouTube channel, by channel id, @handle or URL. "
        "Returns Shorts under data.results with video_id, title, url and thumbnail (no "
        "view counts -- upstream data is unreliable for Shorts). Supports cursor "
        "pagination. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeChannelShortsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioYouTubeChannelShortsAPIWrapper = Field(
        default_factory=ScavioYouTubeChannelShortsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeChannelShortsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        channel_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the Shorts posted by a YouTube channel (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                channel_id=channel_id,
                cursor=cursor,
            )
            return self._process_response(raw, channel_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        channel_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the Shorts posted by a YouTube channel (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                channel_id=channel_id,
                cursor=cursor,
            )
            return self._process_response(raw, channel_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], channel_id: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        results = data.get("results")
        if self.max_results and results:
            raw["data"]["results"] = results[: self.max_results]
        if not results:
            raise ToolException(
                f"No Shorts found for YouTube channel '{channel_id}'. The channel "
                "may not post Shorts, or the id/handle may be wrong."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioYouTubeChannelCommunity
# ---------------------------------------------------------------------------


class ScavioYouTubeChannelCommunityInput(BaseModel):
    """Input schema for ScavioYouTubeChannelCommunity tool."""

    model_config = ConfigDict(extra="allow")

    channel_id: str = Field(
        description=(
            "YouTube channel id (e.g. 'UCX6OQ3DkcsbYNE6H8uQQuVA'), an @handle, or a "
            "channel URL."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioYouTubeChannelCommunity(BaseTool):  # type: ignore[override]
    """Fetch the community posts of a YouTube channel.

    Returns posts under ``data.posts`` -- the only YouTube endpoint whose
    list key is ``posts`` rather than ``results`` -- with post_id, url,
    text, author, published_time, vote_count, comment_count, attachment_type
    and images. Paginate with ``data.next_cursor``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeChannelCommunity

            tool = ScavioYouTubeChannelCommunity()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"channel_id": "@MrBeast"})
    """

    name: str = "scavio_youtube_channel_community"
    description: str = (
        "Fetch a YouTube channel's community posts, by channel id, @handle or URL. "
        "Returns posts under data.posts (not data.results) with text, url, published "
        "time, vote and comment counts and attached images. Supports cursor "
        "pagination. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeChannelCommunityInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioYouTubeChannelCommunityAPIWrapper = Field(
        default_factory=ScavioYouTubeChannelCommunityAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeChannelCommunityAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        channel_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the community posts of a YouTube channel (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                channel_id=channel_id,
                cursor=cursor,
            )
            return self._process_response(raw, channel_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        channel_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the community posts of a YouTube channel (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                channel_id=channel_id,
                cursor=cursor,
            )
            return self._process_response(raw, channel_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], channel_id: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        posts = data.get("posts")
        if self.max_results and posts:
            raw["data"]["posts"] = posts[: self.max_results]
        if not posts:
            raise ToolException(
                f"No community posts found for YouTube channel '{channel_id}'. The "
                "channel may have no community tab."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioYouTubeChannelResolve
# ---------------------------------------------------------------------------


class ScavioYouTubeChannelResolveInput(BaseModel):
    """Input schema for ScavioYouTubeChannelResolve tool."""

    model_config = ConfigDict(extra="allow")

    channel: str = Field(
        description=(
            "A channel @handle, bare name or channel URL to resolve (e.g. '@MrBeast'). "
            "The field is channel, not channel_id."
        ),
    )


class ScavioYouTubeChannelResolve(BaseTool):  # type: ignore[override]
    """Resolve a YouTube @handle or URL to a channel id.

    Returns ``data.channel_id`` and ``data.channel_url``.

    The argument is named ``channel``, not ``channel_id`` -- this is the odd
    one out among the channel endpoints. Resolving once and reusing the UC
    id avoids an extra upstream lookup on every later channel call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeChannelResolve

            tool = ScavioYouTubeChannelResolve()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"channel": "@MrBeast"})
    """

    name: str = "scavio_youtube_channel_resolve"
    description: str = (
        "Resolve a YouTube @handle, bare channel name or channel URL to its canonical "
        "UC channel id. Returns channel_id and channel_url under data. Resolve once "
        "and reuse the id for the other channel tools. Costs 1 credit per call. Input "
        "should be an @handle or channel URL."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeChannelResolveInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYouTubeChannelResolveAPIWrapper = Field(
        default_factory=ScavioYouTubeChannelResolveAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeChannelResolveAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        channel: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolve a YouTube @handle or URL to a channel id (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                channel=channel,
            )
            return self._process_response(raw, channel)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        channel: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resolve a YouTube @handle or URL to a channel id (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                channel=channel,
            )
            return self._process_response(raw, channel)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], channel: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not data.get("channel_id"):
            raise ToolException(
                f"Could not resolve YouTube channel '{channel}'. Try the full "
                "channel URL or the exact @handle."
            )
        return raw
