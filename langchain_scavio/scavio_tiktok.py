"""Scavio TikTok tools for LangChain agents."""

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
    ScavioTikTokCommentRepliesAPIWrapper,
    ScavioTikTokHashtagAPIWrapper,
    ScavioTikTokHashtagVideosAPIWrapper,
    ScavioTikTokProfileAPIWrapper,
    ScavioTikTokSearchUsersAPIWrapper,
    ScavioTikTokSearchVideosAPIWrapper,
    ScavioTikTokUserFollowersAPIWrapper,
    ScavioTikTokUserFollowingsAPIWrapper,
    ScavioTikTokUserPostsAPIWrapper,
    ScavioTikTokVideoAPIWrapper,
    ScavioTikTokVideoCommentsAPIWrapper,
)

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# 1. Profile
# ---------------------------------------------------------------------------


class ScavioTikTokProfileInput(BaseModel):
    """Input schema for ScavioTikTokProfile tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description="TikTok handle without the @ symbol.",
    )
    sec_user_id: Optional[str] = Field(
        default=None,
        description=(
            "Secure user identifier (from a previous profile"
            " or search response)."
        ),
    )


class ScavioTikTokProfile(BaseTool):  # type: ignore[override]
    """Look up a TikTok user profile.

    Returns follower/following counts, bio, avatar, and the ``sec_uid``
    needed by other TikTok endpoints.  Provide either ``username`` or
    ``sec_user_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokProfile

            tool = ScavioTikTokProfile()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "tiktok"})
    """

    name: str = "scavio_tiktok_profile"
    description: str = (
        "Look up a TikTok user profile by username or sec_user_id. "
        "Returns follower/following counts, bio, avatar, and sec_uid. "
        "Provide either username (without @) or sec_user_id."
    )
    args_schema: Type[BaseModel] = ScavioTikTokProfileInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTikTokProfileAPIWrapper = Field(
        default_factory=ScavioTikTokProfileAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokProfileAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        sec_user_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                username=username, sec_user_id=sec_user_id
            )
            return self._process_response(raw, username or sec_user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        sec_user_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username, sec_user_id=sec_user_id
            )
            return self._process_response(raw, username or sec_user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        user = data.get("user") if isinstance(data, dict) else None
        if not user:
            raise ToolException(
                f"No TikTok user found for '{identifier}'. "
                "Check the username or sec_user_id and try again."
            )
        return raw


# ---------------------------------------------------------------------------
# 2. User Posts
# ---------------------------------------------------------------------------


class ScavioTikTokUserPostsInput(BaseModel):
    """Input schema for ScavioTikTokUserPosts tool."""

    model_config = ConfigDict(extra="allow")

    sec_user_id: str = Field(
        description=(
            "Secure user ID from a profile lookup. "
            "Use ScavioTikTokProfile first to obtain this value."
        )
    )
    cursor: Optional[str] = Field(
        default=None,
        description='Pagination cursor from a previous response. Defaults to "0".',
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Results per page (1-30, default 20).",
    )
    sort_type: Optional[Literal["0", "1"]] = Field(
        default=None,
        description='Sort order. "0" = latest (default), "1" = popular.',
    )


class ScavioTikTokUserPosts(BaseTool):  # type: ignore[override]
    """Fetch a TikTok user's posted videos.

    Returns a paginated list of videos with statistics (plays, likes,
    comments, shares).  Use ``data.max_cursor`` for the next page;
    stop when ``data.has_more`` is 0.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokUserPosts

            tool = ScavioTikTokUserPosts(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"sec_user_id": "MS4wLjAB..."})
    """

    name: str = "scavio_tiktok_user_posts"
    description: str = (
        "Fetch a TikTok user's posted videos. "
        "Returns videos with play/like/comment/share counts. "
        "Requires sec_user_id from a profile lookup. "
        "Supports pagination and sort by latest or popular."
    )
    args_schema: Type[BaseModel] = ScavioTikTokUserPostsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioTikTokUserPostsAPIWrapper = Field(
        default_factory=ScavioTikTokUserPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokUserPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        sec_user_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        sort_type: Optional[str] = None,
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
            raw = self.api_wrapper.raw_results(
                sec_user_id=sec_user_id,
                cursor=cursor,
                count=count,
                sort_type=sort_type,
            )
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        sec_user_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        sort_type: Optional[str] = None,
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
                sec_user_id=sec_user_id,
                cursor=cursor,
                count=count,
                sort_type=sort_type,
            )
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        data = raw.get("data") or {}
        videos = data.get("aweme_list") if isinstance(data, dict) else None
        if self.max_results and videos:
            raw["data"]["aweme_list"] = videos[: self.max_results]
        if not (isinstance(data, dict) and data.get("aweme_list")):
            raise ToolException(
                "No posts found for the given TikTok user. "
                "Verify the sec_user_id is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# 3. Video Detail
# ---------------------------------------------------------------------------


class ScavioTikTokVideoInput(BaseModel):
    """Input schema for ScavioTikTokVideo tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(description="TikTok video identifier.")


class ScavioTikTokVideo(BaseTool):  # type: ignore[override]
    """Fetch details for a single TikTok video.

    Returns video metadata including description, statistics, hashtags,
    music, cover image, and playback URLs.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokVideo

            tool = ScavioTikTokVideo()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "7123456789012345678"})
    """

    name: str = "scavio_tiktok_video"
    description: str = (
        "Fetch details for a single TikTok video. "
        "Returns description, statistics (plays, likes, comments, shares), "
        "hashtags, music, cover image, and playback URLs. "
        "Input should be a TikTok video ID."
    )
    args_schema: Type[BaseModel] = ScavioTikTokVideoInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTikTokVideoAPIWrapper = Field(
        default_factory=ScavioTikTokVideoAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokVideoAPIWrapper(**api_kwargs)
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
        detail = data.get("aweme_detail") if isinstance(data, dict) else None
        if not detail:
            raise ToolException(
                f"No TikTok video found for ID '{video_id}'. "
                "Check the video ID and try again."
            )
        return raw


# ---------------------------------------------------------------------------
# 4. Video Comments
# ---------------------------------------------------------------------------


class ScavioTikTokVideoCommentsInput(BaseModel):
    """Input schema for ScavioTikTokVideoComments tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(description="TikTok video identifier.")
    cursor: Optional[str] = Field(
        default=None,
        description='Pagination cursor from a previous response. Defaults to "0".',
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Results per page (1-50, default 20).",
    )


class ScavioTikTokVideoComments(BaseTool):  # type: ignore[override]
    """Fetch comments on a TikTok video.

    Returns a paginated list of comments with text, likes, reply counts,
    and commenter info.  Use ``data.cursor`` for pagination; stop when
    ``data.has_more`` is 0.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokVideoComments

            tool = ScavioTikTokVideoComments(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"video_id": "7123456789012345678"})
    """

    name: str = "scavio_tiktok_video_comments"
    description: str = (
        "Fetch comments on a TikTok video. "
        "Returns comment text, likes, reply counts, and commenter info. "
        "Supports pagination. Input should be a TikTok video ID."
    )
    args_schema: Type[BaseModel] = ScavioTikTokVideoCommentsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioTikTokVideoCommentsAPIWrapper = Field(
        default_factory=ScavioTikTokVideoCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokVideoCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                video_id=video_id, cursor=cursor, count=count
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
        count: Optional[int] = None,
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
                video_id=video_id, cursor=cursor, count=count
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
                f"No comments found for TikTok video '{video_id}'. "
                "The video may have comments disabled or none yet."
            )
        return raw


# ---------------------------------------------------------------------------
# 5. Comment Replies
# ---------------------------------------------------------------------------


class ScavioTikTokCommentRepliesInput(BaseModel):
    """Input schema for ScavioTikTokCommentReplies tool."""

    model_config = ConfigDict(extra="allow")

    video_id: str = Field(description="TikTok video identifier.")
    comment_id: str = Field(
        description=(
            "Comment ID (cid) from the video comments endpoint. "
            "Use ScavioTikTokVideoComments first to find comment IDs."
        )
    )
    cursor: Optional[str] = Field(
        default=None,
        description='Pagination cursor from a previous response. Defaults to "0".',
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Results per page (1-50, default 20).",
    )


class ScavioTikTokCommentReplies(BaseTool):  # type: ignore[override]
    """Fetch replies to a specific comment on a TikTok video.

    Returns a paginated list of reply comments.  Use ``data.cursor``
    for pagination; stop when ``data.has_more`` is 0.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokCommentReplies

            tool = ScavioTikTokCommentReplies(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({
                "video_id": "7123456789012345678",
                "comment_id": "7123456789012345679",
            })
    """

    name: str = "scavio_tiktok_comment_replies"
    description: str = (
        "Fetch replies to a specific comment on a TikTok video. "
        "Requires both video_id and comment_id (from the video comments endpoint). "
        "Supports pagination."
    )
    args_schema: Type[BaseModel] = ScavioTikTokCommentRepliesInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioTikTokCommentRepliesAPIWrapper = Field(
        default_factory=ScavioTikTokCommentRepliesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokCommentRepliesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        video_id: str,
        comment_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                video_id=video_id,
                comment_id=comment_id,
                cursor=cursor,
                count=count,
            )
            return self._process_response(raw, comment_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        video_id: str,
        comment_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
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
                video_id=video_id,
                comment_id=comment_id,
                cursor=cursor,
                count=count,
            )
            return self._process_response(raw, comment_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], comment_id: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        comments = data.get("comments") if isinstance(data, dict) else None
        if self.max_results and comments:
            raw["data"]["comments"] = comments[: self.max_results]
        if not (isinstance(data, dict) and data.get("comments")):
            raise ToolException(
                f"No replies found for comment '{comment_id}'. "
                "The comment may have no replies yet."
            )
        return raw


# ---------------------------------------------------------------------------
# 6. Search Videos
# ---------------------------------------------------------------------------


class ScavioTikTokSearchVideosInput(BaseModel):
    """Input schema for ScavioTikTokSearchVideos tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description="Search query, 1-500 characters.",
        min_length=1,
        max_length=500,
    )
    cursor: Optional[str] = Field(
        default=None,
        description='Pagination offset from a previous response. Defaults to "0".',
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Results per page (1-30, default 20).",
    )
    sort_type: Optional[Literal["0", "1"]] = Field(
        default=None,
        description='Sort method. "0" = relevance (default), "1" = most likes.',
    )
    publish_time: Optional[Literal["0", "1", "7", "30", "90", "180"]] = Field(
        default=None,
        description=(
            'Time filter. "0" = all time (default), "1" = past day, '
            '"7" = past week, "30" = past month, "90" = past 3 months, '
            '"180" = past 6 months.'
        ),
    )


class ScavioTikTokSearchVideos(BaseTool):  # type: ignore[override]
    """Search TikTok videos by keyword.

    Returns a paginated list of videos matching the query with
    statistics, author info, and music metadata.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokSearchVideos

            tool = ScavioTikTokSearchVideos(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "python tutorial"})
    """

    name: str = "scavio_tiktok_search_videos"
    description: str = (
        "Search TikTok videos by keyword. "
        "Returns videos with statistics, author info, and music metadata. "
        "Supports sort by relevance or most likes, and time range filters. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioTikTokSearchVideosInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioTikTokSearchVideosAPIWrapper = Field(
        default_factory=ScavioTikTokSearchVideosAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokSearchVideosAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        sort_type: Optional[str] = None,
        publish_time: Optional[str] = None,
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
            raw = self.api_wrapper.raw_results(
                keyword=keyword,
                cursor=cursor,
                count=count,
                sort_type=sort_type,
                publish_time=publish_time,
            )
            return self._process_response(raw, keyword)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        keyword: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        sort_type: Optional[str] = None,
        publish_time: Optional[str] = None,
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
                keyword=keyword,
                cursor=cursor,
                count=count,
                sort_type=sort_type,
                publish_time=publish_time,
            )
            return self._process_response(raw, keyword)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], keyword: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        items = (
            data.get("search_item_list") if isinstance(data, dict) else None
        )
        if self.max_results and items:
            raw["data"]["search_item_list"] = items[: self.max_results]
        if not (isinstance(data, dict) and data.get("search_item_list")):
            raise ToolException(
                f"No TikTok videos found for '{keyword}'. "
                "Try broadening the query or changing filters."
            )
        return raw


# ---------------------------------------------------------------------------
# 7. Search Users
# ---------------------------------------------------------------------------


class ScavioTikTokSearchUsersInput(BaseModel):
    """Input schema for ScavioTikTokSearchUsers tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description="Search query, 1-500 characters.",
        min_length=1,
        max_length=500,
    )
    cursor: Optional[str] = Field(
        default=None,
        description='Pagination offset from a previous response. Defaults to "0".',
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Results per page (1-30, default 20).",
    )


class ScavioTikTokSearchUsers(BaseTool):  # type: ignore[override]
    """Search TikTok users by keyword.

    Returns a paginated list of user profiles matching the query with
    follower counts, bios, and sec_uid for follow-up requests.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokSearchUsers

            tool = ScavioTikTokSearchUsers(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "cooking"})
    """

    name: str = "scavio_tiktok_search_users"
    description: str = (
        "Search TikTok users by keyword. "
        "Returns user profiles with follower counts, bios, and sec_uid. "
        "Supports pagination. Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioTikTokSearchUsersInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioTikTokSearchUsersAPIWrapper = Field(
        default_factory=ScavioTikTokSearchUsersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokSearchUsersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                keyword=keyword, cursor=cursor, count=count
            )
            return self._process_response(raw, keyword)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        keyword: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
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
                keyword=keyword, cursor=cursor, count=count
            )
            return self._process_response(raw, keyword)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], keyword: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        users = data.get("user_list") if isinstance(data, dict) else None
        if self.max_results and users:
            raw["data"]["user_list"] = users[: self.max_results]
        if not (isinstance(data, dict) and data.get("user_list")):
            raise ToolException(
                f"No TikTok users found for '{keyword}'. "
                "Try broadening the query."
            )
        return raw


# ---------------------------------------------------------------------------
# 8. Hashtag Info
# ---------------------------------------------------------------------------


class ScavioTikTokHashtagInput(BaseModel):
    """Input schema for ScavioTikTokHashtag tool."""

    model_config = ConfigDict(extra="allow")

    hashtag_name: Optional[str] = Field(
        default=None,
        description="Hashtag text without the # symbol.",
    )
    hashtag_id: Optional[str] = Field(
        default=None,
        description="Numeric hashtag identifier.",
    )


class ScavioTikTokHashtag(BaseTool):  # type: ignore[override]
    """Look up TikTok hashtag information.

    Returns the hashtag title, description, video count, and view count.
    Provide either ``hashtag_name`` or ``hashtag_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokHashtag

            tool = ScavioTikTokHashtag()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"hashtag_name": "python"})
    """

    name: str = "scavio_tiktok_hashtag"
    description: str = (
        "Look up TikTok hashtag information. "
        "Returns hashtag title, description, video count, and view count. "
        "Provide either hashtag_name (without #) or hashtag_id."
    )
    args_schema: Type[BaseModel] = ScavioTikTokHashtagInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTikTokHashtagAPIWrapper = Field(
        default_factory=ScavioTikTokHashtagAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokHashtagAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        hashtag_name: Optional[str] = None,
        hashtag_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                hashtag_name=hashtag_name, hashtag_id=hashtag_id
            )
            return self._process_response(raw, hashtag_name or hashtag_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        hashtag_name: Optional[str] = None,
        hashtag_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                hashtag_name=hashtag_name, hashtag_id=hashtag_id
            )
            return self._process_response(raw, hashtag_name or hashtag_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        info = data.get("challengeInfo") if isinstance(data, dict) else None
        if not info:
            raise ToolException(
                f"No TikTok hashtag found for '{identifier}'. "
                "Check the hashtag name or ID and try again."
            )
        return raw


# ---------------------------------------------------------------------------
# 9. Hashtag Videos
# ---------------------------------------------------------------------------


class ScavioTikTokHashtagVideosInput(BaseModel):
    """Input schema for ScavioTikTokHashtagVideos tool."""

    model_config = ConfigDict(extra="allow")

    hashtag_id: str = Field(
        description=(
            "Numeric hashtag ID from the hashtag info endpoint. "
            "Use ScavioTikTokHashtag first to obtain this value."
        )
    )
    cursor: Optional[str] = Field(
        default=None,
        description='Pagination cursor from a previous response. Defaults to "0".',
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Results per page (1-30, default 20).",
    )


class ScavioTikTokHashtagVideos(BaseTool):  # type: ignore[override]
    """Fetch TikTok videos for a specific hashtag.

    Returns a paginated list of videos tagged with the given hashtag.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokHashtagVideos

            tool = ScavioTikTokHashtagVideos(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"hashtag_id": "123456"})
    """

    name: str = "scavio_tiktok_hashtag_videos"
    description: str = (
        "Fetch TikTok videos for a specific hashtag. "
        "Returns videos with statistics and author info. "
        "Requires hashtag_id from the hashtag info endpoint. "
        "Supports pagination."
    )
    args_schema: Type[BaseModel] = ScavioTikTokHashtagVideosInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioTikTokHashtagVideosAPIWrapper = Field(
        default_factory=ScavioTikTokHashtagVideosAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokHashtagVideosAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        hashtag_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                hashtag_id=hashtag_id, cursor=cursor, count=count
            )
            return self._process_response(raw, hashtag_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        hashtag_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
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
                hashtag_id=hashtag_id, cursor=cursor, count=count
            )
            return self._process_response(raw, hashtag_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], hashtag_id: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        videos = data.get("aweme_list") if isinstance(data, dict) else None
        if self.max_results and videos:
            raw["data"]["aweme_list"] = videos[: self.max_results]
        if not (isinstance(data, dict) and data.get("aweme_list")):
            raise ToolException(
                f"No videos found for TikTok hashtag '{hashtag_id}'. "
                "Verify the hashtag_id is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# 10. User Followers
# ---------------------------------------------------------------------------


class ScavioTikTokUserFollowersInput(BaseModel):
    """Input schema for ScavioTikTokUserFollowers tool."""

    model_config = ConfigDict(extra="allow")

    sec_user_id: str = Field(
        description=(
            "Secure user ID from a profile lookup. "
            "Use ScavioTikTokProfile first to obtain this value."
        )
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Results per page (1-20, default 20).",
    )
    page_token: Optional[str] = Field(
        default=None,
        description="Pagination token from a previous response's data.next_page_token.",
    )
    min_time: Optional[int] = Field(
        default=None,
        description="Pagination field from a previous response's data.min_time.",
    )


class ScavioTikTokUserFollowers(BaseTool):  # type: ignore[override]
    """Fetch a TikTok user's followers.

    Returns a paginated list of follower profiles.  Pass both
    ``page_token`` and ``min_time`` from the previous response for
    pagination; stop when ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokUserFollowers

            tool = ScavioTikTokUserFollowers(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"sec_user_id": "MS4wLjAB..."})
    """

    name: str = "scavio_tiktok_user_followers"
    description: str = (
        "Fetch a TikTok user's followers. "
        "Returns follower profiles with usernames, follower counts, and bios. "
        "Requires sec_user_id from a profile lookup. "
        "Supports pagination via page_token and min_time."
    )
    args_schema: Type[BaseModel] = ScavioTikTokUserFollowersInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioTikTokUserFollowersAPIWrapper = Field(
        default_factory=ScavioTikTokUserFollowersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokUserFollowersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        sec_user_id: str,
        count: Optional[int] = None,
        page_token: Optional[str] = None,
        min_time: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                sec_user_id=sec_user_id,
                count=count,
                page_token=page_token,
                min_time=min_time,
            )
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        sec_user_id: str,
        count: Optional[int] = None,
        page_token: Optional[str] = None,
        min_time: Optional[int] = None,
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
                sec_user_id=sec_user_id,
                count=count,
                page_token=page_token,
                min_time=min_time,
            )
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        data = raw.get("data") or {}
        followers = data.get("followers") if isinstance(data, dict) else None
        if self.max_results and followers:
            raw["data"]["followers"] = followers[: self.max_results]
        if not (isinstance(data, dict) and data.get("followers")):
            raise ToolException(
                "No followers found for the given TikTok user. "
                "Verify the sec_user_id is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# 11. User Followings
# ---------------------------------------------------------------------------


class ScavioTikTokUserFollowingsInput(BaseModel):
    """Input schema for ScavioTikTokUserFollowings tool."""

    model_config = ConfigDict(extra="allow")

    sec_user_id: str = Field(
        description=(
            "Secure user ID from a profile lookup. "
            "Use ScavioTikTokProfile first to obtain this value."
        )
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Results per page (1-20, default 20).",
    )
    page_token: Optional[str] = Field(
        default=None,
        description="Pagination token from a previous response's data.next_page_token.",
    )
    min_time: Optional[int] = Field(
        default=None,
        description="Pagination field from a previous response's data.min_time.",
    )


class ScavioTikTokUserFollowings(BaseTool):  # type: ignore[override]
    """Fetch accounts a TikTok user is following.

    Returns a paginated list of followed profiles.  Pass both
    ``page_token`` and ``min_time`` from the previous response for
    pagination; stop when ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokUserFollowings

            tool = ScavioTikTokUserFollowings(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"sec_user_id": "MS4wLjAB..."})
    """

    name: str = "scavio_tiktok_user_followings"
    description: str = (
        "Fetch accounts a TikTok user is following. "
        "Returns followed profiles with usernames, follower counts, and bios. "
        "Requires sec_user_id from a profile lookup. "
        "Supports pagination via page_token and min_time."
    )
    args_schema: Type[BaseModel] = ScavioTikTokUserFollowingsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioTikTokUserFollowingsAPIWrapper = Field(
        default_factory=ScavioTikTokUserFollowingsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokUserFollowingsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        sec_user_id: str,
        count: Optional[int] = None,
        page_token: Optional[str] = None,
        min_time: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                sec_user_id=sec_user_id,
                count=count,
                page_token=page_token,
                min_time=min_time,
            )
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        sec_user_id: str,
        count: Optional[int] = None,
        page_token: Optional[str] = None,
        min_time: Optional[int] = None,
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
                sec_user_id=sec_user_id,
                count=count,
                page_token=page_token,
                min_time=min_time,
            )
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        data = raw.get("data") or {}
        followings = data.get("followings") if isinstance(data, dict) else None
        if self.max_results and followings:
            raw["data"]["followings"] = followings[: self.max_results]
        if not (isinstance(data, dict) and data.get("followings")):
            raise ToolException(
                "No followings found for the given TikTok user. "
                "Verify the sec_user_id is correct."
            )
        return raw
