"""ScavioYouTubeSearch, ScavioYouTubeMetadata, and ScavioYouTubeTranscript tools."""

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
    ScavioYouTubeMetadataAPIWrapper,
    ScavioYouTubeSearchAPIWrapper,
    ScavioYouTubeTranscriptAPIWrapper,
)

logger = logging.getLogger(__name__)

_SEARCH_INIT_ONLY_PARAMS = frozenset(
    {"max_results", "fourk", "hdr", "three_sixty", "threed", "vr180"}
)
_TRANSCRIPT_INIT_ONLY_PARAMS = frozenset({"max_segments"})


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

    Returns title, description, view count, like count, channel info,
    tags, categories, and thumbnail URLs.

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
        "Returns title, description, view count, like count, channel info, and tags. "
        "Use ScavioYouTubeSearch to find video IDs. "
        "Input should be a YouTube video ID (e.g., 'dQw4w9WgXcQ')."
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


class ScavioYouTubeTranscriptInput(BaseModel):
    """Input schema for ScavioYouTubeTranscript tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(
        description=(
            "YouTube video ID (e.g., 'dQw4w9WgXcQ'). "
            "Use ScavioYouTubeSearch first to find video IDs if needed."
        )
    )

    language: Optional[str] = Field(
        default=None,
        description=(
            'ISO 639-1 language code for the transcript (e.g., "en", "es", "fr"). '
            "Default is 'en'."
        ),
    )

    transcript_origin: Optional[Literal["auto_generated", "uploader_provided"]] = Field(
        default=None,
        description=(
            "Transcript source preference. "
            "Options: auto_generated (auto-captions), "
            "uploader_provided (manual captions)."
        ),
    )


class ScavioYouTubeTranscript(BaseTool):  # type: ignore[override]
    """Fetch the full transcript of a YouTube video by video ID.

    Returns timestamped transcript segments with text, start time, and duration.
    Useful for summarizing videos, extracting quotes, or analyzing content.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYouTubeTranscript

            tool = ScavioYouTubeTranscript(max_segments=200)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "dQw4w9WgXcQ"})
    """

    name: str = "scavio_youtube_transcript"
    description: str = (
        "Fetch the full transcript of a YouTube video by video ID. "
        "Returns timestamped text segments. "
        "Useful for summarizing videos, extracting quotes, or analyzing content. "
        "Input should be a YouTube video ID (e.g., 'dQw4w9WgXcQ')."
    )
    args_schema: Type[BaseModel] = ScavioYouTubeTranscriptInput
    handle_tool_error: bool = True

    # Instantiation-only parameters (not controllable by the LLM).
    max_segments: Optional[int] = None

    api_wrapper: ScavioYouTubeTranscriptAPIWrapper = Field(
        default_factory=ScavioYouTubeTranscriptAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_wrapper_kwargs: dict[str, Any] = {}
        if "scavio_api_key" in kwargs:
            api_wrapper_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
        if "api_base_url" in kwargs:
            api_wrapper_kwargs["api_base_url"] = kwargs.pop("api_base_url")
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYouTubeTranscriptAPIWrapper(
                **api_wrapper_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        language: Optional[str] = None,
        transcript_origin: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch YouTube video transcript synchronously."""
        forbidden = _TRANSCRIPT_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                video_id=video_id,
                language=language,
                transcript_origin=transcript_origin,
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
        transcript_origin: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch YouTube video transcript asynchronously."""
        forbidden = _TRANSCRIPT_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                video_id=video_id,
                language=language,
                transcript_origin=transcript_origin,
            )
            return self._process_response(raw, video_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], video_id: str
    ) -> dict[str, Any]:
        """Truncate segments and raise ToolException if empty."""
        data = raw.get("data") or {}
        transcripts = data.get("transcripts") if isinstance(data, dict) else None
        if self.max_segments and transcripts:
            raw["data"]["transcripts"] = transcripts[: self.max_segments]
        if not (isinstance(data, dict) and data.get("transcripts")):
            raise ToolException(
                f"No transcript available for YouTube video '{video_id}'. "
                "The video may not have captions. "
                "Try language='en' or transcript_origin='auto_generated'."
            )
        return raw
