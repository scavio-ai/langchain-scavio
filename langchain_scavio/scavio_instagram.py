"""Scavio Instagram tools for LangChain agents."""

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
    ScavioInstagramCommentRepliesAPIWrapper,
    ScavioInstagramPostAPIWrapper,
    ScavioInstagramPostCommentsAPIWrapper,
    ScavioInstagramProfileAPIWrapper,
    ScavioInstagramSearchHashtagsAPIWrapper,
    ScavioInstagramSearchUsersAPIWrapper,
    ScavioInstagramStoriesAPIWrapper,
    ScavioInstagramTaggedPostsAPIWrapper,
    ScavioInstagramUserFollowersAPIWrapper,
    ScavioInstagramUserFollowingsAPIWrapper,
    ScavioInstagramUserPostsAPIWrapper,
    ScavioInstagramUserReelsAPIWrapper,
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


class ScavioInstagramProfileInput(BaseModel):
    """Input schema for ScavioInstagramProfile tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description="Instagram handle without the @ symbol.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric user identifier (from a previous profile"
            " or search response)."
        ),
    )


class ScavioInstagramProfile(BaseTool):  # type: ignore[override]
    """Look up an Instagram user profile.

    Returns follower/following counts, bio, avatar, and the ``user_id``
    needed by other Instagram endpoints.  Provide either ``username`` or
    ``user_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramProfile

            tool = ScavioInstagramProfile()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "instagram"})
    """

    name: str = "scavio_instagram_profile"
    description: str = (
        "Look up an Instagram user profile by username or user_id. "
        "Returns follower/following counts, bio, avatar, and user_id. "
        "Provide either username (without @) or user_id."
    )
    args_schema: Type[BaseModel] = ScavioInstagramProfileInput
    handle_tool_error: bool = True

    api_wrapper: ScavioInstagramProfileAPIWrapper = Field(
        default_factory=ScavioInstagramProfileAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramProfileAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                username=username, user_id=user_id
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username, user_id=user_id
            )
            return self._process_response(raw, username or user_id or "")
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
                f"No Instagram user found for '{identifier}'. "
                "Check the username or user_id and try again."
            )
        return raw


# ---------------------------------------------------------------------------
# 2. User Posts
# ---------------------------------------------------------------------------


class ScavioInstagramUserPostsInput(BaseModel):
    """Input schema for ScavioInstagramUserPosts tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description="Instagram handle without the @ symbol.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric user ID from a profile lookup. "
            "Use ScavioInstagramProfile first to obtain this value."
        ),
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Results per page (1-50, default 12).",
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )


class ScavioInstagramUserPosts(BaseTool):  # type: ignore[override]
    """Fetch an Instagram user's posts.

    Returns a paginated list of posts with statistics (likes, comments,
    media type, captions).  Use ``data.next_cursor`` for the next page;
    stop when ``data.has_more`` is false.  Provide either ``username``
    or ``user_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramUserPosts

            tool = ScavioInstagramUserPosts(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "instagram"})
    """

    name: str = "scavio_instagram_user_posts"
    description: str = (
        "Fetch an Instagram user's posts. "
        "Returns posts with like/comment counts, media type, and captions. "
        "Provide either username (without @) or user_id. "
        "Supports pagination via cursor."
    )
    args_schema: Type[BaseModel] = ScavioInstagramUserPostsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioInstagramUserPostsAPIWrapper = Field(
        default_factory=ScavioInstagramUserPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramUserPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        posts = data.get("posts") if isinstance(data, dict) else None
        if self.max_results and posts:
            raw["data"]["posts"] = posts[: self.max_results]
        if not (isinstance(data, dict) and data.get("posts")):
            raise ToolException(
                f"No posts found for Instagram user '{identifier}'. "
                "Verify the username or user_id is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# 3. User Reels
# ---------------------------------------------------------------------------


class ScavioInstagramUserReelsInput(BaseModel):
    """Input schema for ScavioInstagramUserReels tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description="Instagram handle without the @ symbol.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric user ID from a profile lookup. "
            "Use ScavioInstagramProfile first to obtain this value."
        ),
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Results per page (1-50, default 12).",
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )


class ScavioInstagramUserReels(BaseTool):  # type: ignore[override]
    """Fetch an Instagram user's reels.

    Returns a paginated list of reels with statistics (plays, likes,
    comments, captions).  Use ``data.next_cursor`` for the next page;
    stop when ``data.has_more`` is false.  Provide either ``username``
    or ``user_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramUserReels

            tool = ScavioInstagramUserReels(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "instagram"})
    """

    name: str = "scavio_instagram_user_reels"
    description: str = (
        "Fetch an Instagram user's reels. "
        "Returns reels with play/like/comment counts and captions. "
        "Provide either username (without @) or user_id. "
        "Supports pagination via cursor."
    )
    args_schema: Type[BaseModel] = ScavioInstagramUserReelsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioInstagramUserReelsAPIWrapper = Field(
        default_factory=ScavioInstagramUserReelsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramUserReelsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        reels = data.get("reels") if isinstance(data, dict) else None
        if self.max_results and reels:
            raw["data"]["reels"] = reels[: self.max_results]
        if not (isinstance(data, dict) and data.get("reels")):
            raise ToolException(
                f"No reels found for Instagram user '{identifier}'. "
                "Verify the username or user_id is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# 4. Tagged Posts
# ---------------------------------------------------------------------------


class ScavioInstagramTaggedPostsInput(BaseModel):
    """Input schema for ScavioInstagramTaggedPosts tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description="Instagram handle without the @ symbol.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric user ID from a profile lookup. "
            "Use ScavioInstagramProfile first to obtain this value."
        ),
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Results per page (1-50, default 12).",
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )


class ScavioInstagramTaggedPosts(BaseTool):  # type: ignore[override]
    """Fetch posts an Instagram user is tagged in.

    Returns a paginated list of posts where the user is tagged, with
    statistics and captions.  Use ``data.next_cursor`` for the next
    page; stop when ``data.has_more`` is false.  Provide either
    ``username`` or ``user_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramTaggedPosts

            tool = ScavioInstagramTaggedPosts(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "instagram"})
    """

    name: str = "scavio_instagram_user_tagged"
    description: str = (
        "Fetch posts an Instagram user is tagged in. "
        "Returns tagged posts with like/comment counts and captions. "
        "Provide either username (without @) or user_id. "
        "Supports pagination via cursor."
    )
    args_schema: Type[BaseModel] = ScavioInstagramTaggedPostsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioInstagramTaggedPostsAPIWrapper = Field(
        default_factory=ScavioInstagramTaggedPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramTaggedPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        posts = data.get("posts") if isinstance(data, dict) else None
        if self.max_results and posts:
            raw["data"]["posts"] = posts[: self.max_results]
        if not (isinstance(data, dict) and data.get("posts")):
            raise ToolException(
                f"No tagged posts found for Instagram user '{identifier}'. "
                "Verify the username or user_id is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# 5. Stories
# ---------------------------------------------------------------------------


class ScavioInstagramStoriesInput(BaseModel):
    """Input schema for ScavioInstagramStories tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description="Instagram handle without the @ symbol.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric user ID from a profile lookup. "
            "Use ScavioInstagramProfile first to obtain this value."
        ),
    )


class ScavioInstagramStories(BaseTool):  # type: ignore[override]
    """Fetch an Instagram user's active stories.

    Returns the user's currently active stories with media URLs and
    timestamps.  Provide either ``username`` or ``user_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramStories

            tool = ScavioInstagramStories()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "instagram"})
    """

    name: str = "scavio_instagram_user_stories"
    description: str = (
        "Fetch an Instagram user's active stories. "
        "Returns active stories with media URLs and timestamps. "
        "Provide either username (without @) or user_id."
    )
    args_schema: Type[BaseModel] = ScavioInstagramStoriesInput
    handle_tool_error: bool = True

    api_wrapper: ScavioInstagramStoriesAPIWrapper = Field(
        default_factory=ScavioInstagramStoriesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramStoriesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                username=username, user_id=user_id
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username, user_id=user_id
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        if not (isinstance(data, dict) and data.get("stories")):
            raise ToolException(
                f"No active stories found for Instagram user '{identifier}'. "
                "The user may have no active stories right now."
            )
        return raw


# ---------------------------------------------------------------------------
# 6. Post
# ---------------------------------------------------------------------------


class ScavioInstagramPostInput(BaseModel):
    """Input schema for ScavioInstagramPost tool."""

    model_config = ConfigDict(extra="allow")

    url: Optional[str] = Field(
        default=None,
        description="Full URL of the Instagram post or reel.",
    )
    media_id: Optional[str] = Field(
        default=None,
        description="Numeric media identifier of the post.",
    )
    shortcode: Optional[str] = Field(
        default=None,
        description="Shortcode from the post URL (the part after /p/ or /reel/).",
    )


class ScavioInstagramPost(BaseTool):  # type: ignore[override]
    """Fetch details for a single Instagram post or reel.

    Returns post metadata including caption, statistics, media type,
    media URLs, and author info.  Provide one of ``url``, ``media_id``,
    or ``shortcode``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramPost

            tool = ScavioInstagramPost()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"shortcode": "C1a2b3c4d5e"})
    """

    name: str = "scavio_instagram_post"
    description: str = (
        "Fetch details for a single Instagram post or reel. "
        "Returns caption, statistics (likes, comments), media type, "
        "media URLs, and author info. "
        "Provide one of url, media_id, or shortcode."
    )
    args_schema: Type[BaseModel] = ScavioInstagramPostInput
    handle_tool_error: bool = True

    api_wrapper: ScavioInstagramPostAPIWrapper = Field(
        default_factory=ScavioInstagramPostAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramPostAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        url: Optional[str] = None,
        media_id: Optional[str] = None,
        shortcode: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                url=url, media_id=media_id, shortcode=shortcode
            )
            return self._process_response(
                raw, url or media_id or shortcode or ""
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        url: Optional[str] = None,
        media_id: Optional[str] = None,
        shortcode: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                url=url, media_id=media_id, shortcode=shortcode
            )
            return self._process_response(
                raw, url or media_id or shortcode or ""
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        post = data.get("post") if isinstance(data, dict) else None
        if not post:
            raise ToolException(
                f"No Instagram post found for '{identifier}'. "
                "Check the url, media_id, or shortcode and try again."
            )
        return raw


# ---------------------------------------------------------------------------
# 7. Post Comments
# ---------------------------------------------------------------------------


class ScavioInstagramPostCommentsInput(BaseModel):
    """Input schema for ScavioInstagramPostComments tool."""

    model_config = ConfigDict(extra="allow")

    shortcode: Optional[str] = Field(
        default=None,
        description="Shortcode from the post URL (the part after /p/ or /reel/).",
    )
    url: Optional[str] = Field(
        default=None,
        description="Full URL of the Instagram post or reel.",
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )
    sort_order: Optional[Literal["popular", "newest"]] = Field(
        default=None,
        description='Sort order. "popular" (default) or "newest".',
    )


class ScavioInstagramPostComments(BaseTool):  # type: ignore[override]
    """Fetch comments on an Instagram post.

    Returns a paginated list of comments with text, likes, reply counts,
    and commenter info.  Use ``data.next_cursor`` for pagination; stop
    when ``data.has_more`` is false.  Provide either ``shortcode`` or
    ``url``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramPostComments

            tool = ScavioInstagramPostComments(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"shortcode": "C1a2b3c4d5e"})
    """

    name: str = "scavio_instagram_post_comments"
    description: str = (
        "Fetch comments on an Instagram post. "
        "Returns comment text, likes, reply counts, and commenter info. "
        "Provide either shortcode or url. "
        "Supports pagination and sort by popular or newest."
    )
    args_schema: Type[BaseModel] = ScavioInstagramPostCommentsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioInstagramPostCommentsAPIWrapper = Field(
        default_factory=ScavioInstagramPostCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramPostCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        shortcode: Optional[str] = None,
        url: Optional[str] = None,
        cursor: Optional[str] = None,
        sort_order: Optional[str] = None,
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
                shortcode=shortcode,
                url=url,
                cursor=cursor,
                sort_order=sort_order,
            )
            return self._process_response(raw, shortcode or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        shortcode: Optional[str] = None,
        url: Optional[str] = None,
        cursor: Optional[str] = None,
        sort_order: Optional[str] = None,
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
                shortcode=shortcode,
                url=url,
                cursor=cursor,
                sort_order=sort_order,
            )
            return self._process_response(raw, shortcode or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        comments = data.get("comments") if isinstance(data, dict) else None
        if self.max_results and comments:
            raw["data"]["comments"] = comments[: self.max_results]
        if not (isinstance(data, dict) and data.get("comments")):
            raise ToolException(
                f"No comments found for Instagram post '{identifier}'. "
                "The post may have comments disabled or none yet."
            )
        return raw


# ---------------------------------------------------------------------------
# 8. Comment Replies
# ---------------------------------------------------------------------------


class ScavioInstagramCommentRepliesInput(BaseModel):
    """Input schema for ScavioInstagramCommentReplies tool."""

    model_config = ConfigDict(extra="allow")

    media_id: str = Field(
        description="Numeric media identifier of the post the comment belongs to."
    )
    comment_id: str = Field(
        description=(
            "Comment ID from the post comments endpoint. "
            "Use ScavioInstagramPostComments first to find comment IDs."
        )
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )


class ScavioInstagramCommentReplies(BaseTool):  # type: ignore[override]
    """Fetch replies to a specific comment on an Instagram post.

    Returns a paginated list of reply comments.  Use ``data.next_cursor``
    for pagination; stop when ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramCommentReplies

            tool = ScavioInstagramCommentReplies(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({
                "media_id": "1234567890123456789",
                "comment_id": "9876543210987654321",
            })
    """

    name: str = "scavio_instagram_comment_replies"
    description: str = (
        "Fetch replies to a specific comment on an Instagram post. "
        "Requires both media_id and comment_id "
        "(from the post comments endpoint). "
        "Supports pagination."
    )
    args_schema: Type[BaseModel] = ScavioInstagramCommentRepliesInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioInstagramCommentRepliesAPIWrapper = Field(
        default_factory=ScavioInstagramCommentRepliesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramCommentRepliesAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        media_id: str,
        comment_id: str,
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
            raw = self.api_wrapper.raw_results(
                media_id=media_id,
                comment_id=comment_id,
                cursor=cursor,
            )
            return self._process_response(raw, comment_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        media_id: str,
        comment_id: str,
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
                media_id=media_id,
                comment_id=comment_id,
                cursor=cursor,
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
# 9. Search Users
# ---------------------------------------------------------------------------


class ScavioInstagramSearchUsersInput(BaseModel):
    """Input schema for ScavioInstagramSearchUsers tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description="Search query, 1-500 characters.",
        min_length=1,
        max_length=500,
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )


class ScavioInstagramSearchUsers(BaseTool):  # type: ignore[override]
    """Search Instagram users by keyword.

    Returns a paginated list of user profiles matching the query with
    follower counts, bios, and user_id for follow-up requests.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramSearchUsers

            tool = ScavioInstagramSearchUsers(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "cooking"})
    """

    name: str = "scavio_instagram_search_users"
    description: str = (
        "Search Instagram users by keyword. "
        "Returns user profiles with follower counts, bios, and user_id. "
        "Supports pagination. Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioInstagramSearchUsersInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioInstagramSearchUsersAPIWrapper = Field(
        default_factory=ScavioInstagramSearchUsersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramSearchUsersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
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
            raw = self.api_wrapper.raw_results(keyword=keyword, cursor=cursor)
            return self._process_response(raw, keyword)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        keyword: str,
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
                keyword=keyword, cursor=cursor
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
        users = data.get("users") if isinstance(data, dict) else None
        if self.max_results and users:
            raw["data"]["users"] = users[: self.max_results]
        if not (isinstance(data, dict) and data.get("users")):
            raise ToolException(
                f"No Instagram users found for '{keyword}'. "
                "Try broadening the query."
            )
        return raw


# ---------------------------------------------------------------------------
# 10. Search Hashtags
# ---------------------------------------------------------------------------


class ScavioInstagramSearchHashtagsInput(BaseModel):
    """Input schema for ScavioInstagramSearchHashtags tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description="Search query, 1-500 characters.",
        min_length=1,
        max_length=500,
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )


class ScavioInstagramSearchHashtags(BaseTool):  # type: ignore[override]
    """Search Instagram hashtags by keyword.

    Returns a paginated list of hashtags matching the query with their
    names and media counts.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramSearchHashtags

            tool = ScavioInstagramSearchHashtags(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "travel"})
    """

    name: str = "scavio_instagram_search_hashtags"
    description: str = (
        "Search Instagram hashtags by keyword. "
        "Returns hashtags with their names and media counts. "
        "Supports pagination. Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioInstagramSearchHashtagsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioInstagramSearchHashtagsAPIWrapper = Field(
        default_factory=ScavioInstagramSearchHashtagsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramSearchHashtagsAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
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
            raw = self.api_wrapper.raw_results(keyword=keyword, cursor=cursor)
            return self._process_response(raw, keyword)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        keyword: str,
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
                keyword=keyword, cursor=cursor
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
        hashtags = data.get("hashtags") if isinstance(data, dict) else None
        if self.max_results and hashtags:
            raw["data"]["hashtags"] = hashtags[: self.max_results]
        if not (isinstance(data, dict) and data.get("hashtags")):
            raise ToolException(
                f"No Instagram hashtags found for '{keyword}'. "
                "Try broadening the query."
            )
        return raw


# ---------------------------------------------------------------------------
# 11. User Followers
# ---------------------------------------------------------------------------


class ScavioInstagramUserFollowersInput(BaseModel):
    """Input schema for ScavioInstagramUserFollowers tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description="Instagram handle without the @ symbol.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric user ID from a profile lookup. "
            "Use ScavioInstagramProfile first to obtain this value."
        ),
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Results per page (1-50, default 12).",
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )


class ScavioInstagramUserFollowers(BaseTool):  # type: ignore[override]
    """Fetch an Instagram user's followers.

    Returns a paginated list of follower profiles.  Use
    ``data.next_cursor`` for pagination; stop when ``data.has_more`` is
    false.  Provide either ``username`` or ``user_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramUserFollowers

            tool = ScavioInstagramUserFollowers(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "instagram"})
    """

    name: str = "scavio_instagram_user_followers"
    description: str = (
        "Fetch an Instagram user's followers. "
        "Returns follower profiles with usernames, follower counts, and bios. "
        "Provide either username (without @) or user_id. "
        "Supports pagination via cursor."
    )
    args_schema: Type[BaseModel] = ScavioInstagramUserFollowersInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioInstagramUserFollowersAPIWrapper = Field(
        default_factory=ScavioInstagramUserFollowersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramUserFollowersAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        followers = data.get("followers") if isinstance(data, dict) else None
        if self.max_results and followers:
            raw["data"]["followers"] = followers[: self.max_results]
        if not (isinstance(data, dict) and data.get("followers")):
            raise ToolException(
                f"No followers found for Instagram user '{identifier}'. "
                "Verify the username or user_id is correct."
            )
        return raw


# ---------------------------------------------------------------------------
# 12. User Followings
# ---------------------------------------------------------------------------


class ScavioInstagramUserFollowingsInput(BaseModel):
    """Input schema for ScavioInstagramUserFollowings tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description="Instagram handle without the @ symbol.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric user ID from a profile lookup. "
            "Use ScavioInstagramProfile first to obtain this value."
        ),
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Results per page (1-50, default 12).",
    )
    cursor: Optional[str] = Field(
        default=None,
        description="Pagination cursor from a previous response.",
    )


class ScavioInstagramUserFollowings(BaseTool):  # type: ignore[override]
    """Fetch accounts an Instagram user is following.

    Returns a paginated list of followed profiles.  Use
    ``data.next_cursor`` for pagination; stop when ``data.has_more`` is
    false.  Provide either ``username`` or ``user_id``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioInstagramUserFollowings

            tool = ScavioInstagramUserFollowings(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "instagram"})
    """

    name: str = "scavio_instagram_user_followings"
    description: str = (
        "Fetch accounts an Instagram user is following. "
        "Returns followed profiles with usernames, follower counts, and bios. "
        "Provide either username (without @) or user_id. "
        "Supports pagination via cursor."
    )
    args_schema: Type[BaseModel] = ScavioInstagramUserFollowingsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioInstagramUserFollowingsAPIWrapper = Field(
        default_factory=ScavioInstagramUserFollowingsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioInstagramUserFollowingsAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
            raw = self.api_wrapper.raw_results(
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        count: Optional[int] = None,
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
                username=username,
                user_id=user_id,
                count=count,
                cursor=cursor,
            )
            return self._process_response(raw, username or user_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        followings = data.get("followings") if isinstance(data, dict) else None
        if self.max_results and followings:
            raw["data"]["followings"] = followings[: self.max_results]
        if not (isinstance(data, dict) and data.get("followings")):
            raise ToolException(
                f"No followings found for Instagram user '{identifier}'. "
                "Verify the username or user_id is correct."
            )
        return raw
