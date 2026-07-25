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
    ScavioYouTubeChannelVideosAPIWrapper,
    ScavioYouTubeCommentsAPIWrapper,
    ScavioYouTubeMetadataAPIWrapper,
    ScavioYouTubeSearchAPIWrapper,
    ScavioYouTubeStreamsAPIWrapper,
    ScavioYouTubeTranscriptAPIWrapper,
    ScavioYouTubeVideoAPIWrapper,
)

logger = logging.getLogger(__name__)

_SEARCH_INIT_ONLY_PARAMS = frozenset(
    {"max_results", "fourk", "hdr", "three_sixty", "threed", "vr180"}
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

    video_type: Optional[Literal["video", "channel", "playlist"]] = Field(
        default=None,
        description="Filter by content type. Options: video, channel, playlist.",
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
    Supports filtering by upload date, duration, content type, and more.

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
        "Supports filtering by upload date, duration, content type, and sort order. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeSearchInput
    handle_tool_error: bool = True

    # Instantiation-only parameters (not controllable by the LLM).
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

    def _run(
        self,
        query: str,
        upload_date: Optional[str] = None,
        video_type: Optional[str] = None,
        duration: Optional[str] = None,
        sort_by: Optional[str] = None,
        hd: Optional[bool] = None,
        subtitles: Optional[bool] = None,
        creative_commons: Optional[bool] = None,
        live: Optional[bool] = None,
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
            params: dict[str, Any] = {
                "search": query,
                "upload_date": upload_date,
                "type": video_type,
                "duration": duration,
                "sort_by": sort_by,
                "hd": hd,
                "subtitles": subtitles,
                "creative_commons": creative_commons,
                "live": live,
                "location": location,
                "features": features,
                "cursor": cursor,
                "4k": self.fourk,
                "hdr": self.hdr,
                "360": self.three_sixty,
                "3d": self.threed,
                "vr180": self.vr180,
            }
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
        video_type: Optional[str] = None,
        duration: Optional[str] = None,
        sort_by: Optional[str] = None,
        hd: Optional[bool] = None,
        subtitles: Optional[bool] = None,
        creative_commons: Optional[bool] = None,
        live: Optional[bool] = None,
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
            params: dict[str, Any] = {
                "search": query,
                "upload_date": upload_date,
                "type": video_type,
                "duration": duration,
                "sort_by": sort_by,
                "hd": hd,
                "subtitles": subtitles,
                "creative_commons": creative_commons,
                "live": live,
                "location": location,
                "features": features,
                "cursor": cursor,
                "4k": self.fourk,
                "hdr": self.hdr,
                "360": self.three_sixty,
                "3d": self.threed,
                "vr180": self.vr180,
            }
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
        "Input should be a YouTube video ID or watch URL (e.g., 'dQw4w9WgXcQ')."
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
        "Input should be a YouTube video ID or watch URL."
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
        "Supports pagination. Input should be a YouTube video ID or watch URL."
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
        "Input should be a channel ID, @handle, or channel URL."
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
        "and publish time. Requires a channel ID. Supports pagination."
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
