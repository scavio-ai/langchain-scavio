"""Scavio Kuaishou (China) tools for LangChain agents.

Every URL, parameter name, enum and credit cost below is copied from the
Scavio route definition rather than derived from the tool name.
"""

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
    ScavioKuaishouCommentRepliesAPIWrapper,
    ScavioKuaishouProfileAPIWrapper,
    ScavioKuaishouSearchAPIWrapper,
    ScavioKuaishouSearchLiveAPIWrapper,
    ScavioKuaishouSearchUsersAPIWrapper,
    ScavioKuaishouSearchVideosAPIWrapper,
    ScavioKuaishouTagFeedAPIWrapper,
    ScavioKuaishouTrendingAPIWrapper,
    ScavioKuaishouUserLiveAPIWrapper,
    ScavioKuaishouUserPostsAPIWrapper,
    ScavioKuaishouUserResolveAPIWrapper,
    ScavioKuaishouVideoAPIWrapper,
    ScavioKuaishouVideoCommentsAPIWrapper,
    ScavioKuaishouVideosBatchAPIWrapper,
)

logger = logging.getLogger(__name__)


def _forward_api_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Split the API wrapper kwargs out of the tool kwargs."""
    api_kwargs: dict[str, Any] = {}
    for key in ("scavio_api_key", "api_base_url", "max_requests_per_second"):
        if key in kwargs:
            api_kwargs[key] = kwargs.pop(key)
    return api_kwargs


def _first_identifier(*values: Any) -> str:
    """Return the first non-empty value, used only in error messages."""
    for value in values:
        if value not in (None, "", [], {}):
            return str(value)
    return "this request"


# --------------------------------------------------------------------------
# ScavioKuaishouProfile
# --------------------------------------------------------------------------


class ScavioKuaishouProfileInput(BaseModel):
    """Input schema for the ScavioKuaishouProfile tool."""

    model_config = ConfigDict(extra="allow")

    user_id: str = Field(
        description=(
            "Kuaishou user id."
        ),
    )


class ScavioKuaishouProfile(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Profile details for a Kuaishou user.

    Costs 10 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouProfile

            tool = ScavioKuaishouProfile()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"user_id": "3xnmvnpnyzqxqzm"})
    """

    name: str = "scavio_kuaishou_profile"
    description: str = (
        "Kuaishou (China): Profile details for a Kuaishou user. Costs 10 credits per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouProfileInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouProfileAPIWrapper = Field(
        default_factory=ScavioKuaishouProfileAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouProfileAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        user_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/profile (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                user_id=user_id,
            )
            return self._process_response(raw, _first_identifier(user_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        user_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/profile (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                user_id=user_id,
            )
            return self._process_response(raw, _first_identifier(user_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) profile found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouUserPosts
# --------------------------------------------------------------------------


class ScavioKuaishouUserPostsInput(BaseModel):
    """Input schema for the ScavioKuaishouUserPosts tool."""

    model_config = ConfigDict(extra="allow")

    user_id: str = Field(
        description=(
            "Kuaishou user id."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioKuaishouUserPosts(BaseTool):  # type: ignore[override]
    """Kuaishou (China): A user's top posts, cursor-paginated.

    Costs 1 credit per call.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouUserPosts

            tool = ScavioKuaishouUserPosts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"user_id": "3xnmvnpnyzqxqzm"})
    """

    name: str = "scavio_kuaishou_user_posts"
    description: str = (
        "Kuaishou (China): A user's top posts, cursor-paginated. Pagination: cursor -> "
        "next_cursor. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouUserPostsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouUserPostsAPIWrapper = Field(
        default_factory=ScavioKuaishouUserPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouUserPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        user_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/user/posts (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                user_id=user_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(user_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        user_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/user/posts (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                user_id=user_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(user_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) posts found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouUserLive
# --------------------------------------------------------------------------


class ScavioKuaishouUserLiveInput(BaseModel):
    """Input schema for the ScavioKuaishouUserLive tool."""

    model_config = ConfigDict(extra="allow")

    user_id: str = Field(
        description=(
            "Kuaishou user id."
        ),
    )


class ScavioKuaishouUserLive(BaseTool):  # type: ignore[override]
    """Kuaishou (China): A user's current live-stream status.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouUserLive

            tool = ScavioKuaishouUserLive()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"user_id": "3xnmvnpnyzqxqzm"})
    """

    name: str = "scavio_kuaishou_user_live"
    description: str = (
        "Kuaishou (China): A user's current live-stream status. Costs 1 credit per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouUserLiveInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouUserLiveAPIWrapper = Field(
        default_factory=ScavioKuaishouUserLiveAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouUserLiveAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        user_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/user/live (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                user_id=user_id,
            )
            return self._process_response(raw, _first_identifier(user_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        user_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/user/live (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                user_id=user_id,
            )
            return self._process_response(raw, _first_identifier(user_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) live status found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouUserResolve
# --------------------------------------------------------------------------


class ScavioKuaishouUserResolveInput(BaseModel):
    """Input schema for the ScavioKuaishouUserResolve tool."""

    model_config = ConfigDict(extra="allow")

    share_link: str = Field(
        description=(
            "A kuaishou.com or v.kuaishou.com share link. kwai.com links are NOT "
            "supported -- TikHub does not serve Kwai international."
        ),
    )


class ScavioKuaishouUserResolve(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Turns a Kuaishou share link into a user id.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouUserResolve

            tool = ScavioKuaishouUserResolve()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"share_link": "https://v.kuaishou.com/abc123"})
    """

    name: str = "scavio_kuaishou_user_resolve"
    description: str = (
        "Kuaishou (China): Turns a Kuaishou share link into a user id. Costs 1 credit "
        "per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouUserResolveInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouUserResolveAPIWrapper = Field(
        default_factory=ScavioKuaishouUserResolveAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouUserResolveAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        share_link: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/user/resolve (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                share_link=share_link,
            )
            return self._process_response(raw, _first_identifier(share_link))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        share_link: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/user/resolve (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                share_link=share_link,
            )
            return self._process_response(raw, _first_identifier(share_link))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) user found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouVideo
# --------------------------------------------------------------------------


class ScavioKuaishouVideoInput(BaseModel):
    """Input schema for the ScavioKuaishouVideo tool."""

    model_config = ConfigDict(extra="allow")

    photo_id: Optional[str] = Field(
        default=None,
        description=(
            "Kuaishou photo (video) id."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "A kuaishou.com video URL, usable instead of photo_id."
        ),
    )


class ScavioKuaishouVideo(BaseTool):  # type: ignore[override]
    """Kuaishou (China): A single Kuaishou video by photo id or URL.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouVideo

            tool = ScavioKuaishouVideo()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"photo_id": "3xf8v9pmcvexbhi"})
    """

    name: str = "scavio_kuaishou_video"
    description: str = (
        "Kuaishou (China): A single Kuaishou video by photo id or URL. Costs 2 credits "
        "per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouVideoInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouVideoAPIWrapper = Field(
        default_factory=ScavioKuaishouVideoAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouVideoAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        photo_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/video (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                photo_id=photo_id,
                url=url,
            )
            return self._process_response(raw, _first_identifier(photo_id, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        photo_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/video (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                photo_id=photo_id,
                url=url,
            )
            return self._process_response(raw, _first_identifier(photo_id, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) video found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouVideoComments
# --------------------------------------------------------------------------


class ScavioKuaishouVideoCommentsInput(BaseModel):
    """Input schema for the ScavioKuaishouVideoComments tool."""

    model_config = ConfigDict(extra="allow")

    photo_id: str = Field(
        description=(
            "Kuaishou photo (video) id."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioKuaishouVideoComments(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Comments on a video, cursor-paginated.

    Costs 1 credit per call.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouVideoComments

            tool = ScavioKuaishouVideoComments()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"photo_id": "3xf8v9pmcvexbhi"})
    """

    name: str = "scavio_kuaishou_video_comments"
    description: str = (
        "Kuaishou (China): Comments on a video, cursor-paginated. Pagination: cursor "
        "-> next_cursor. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouVideoCommentsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouVideoCommentsAPIWrapper = Field(
        default_factory=ScavioKuaishouVideoCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouVideoCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        photo_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/video/comments (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                photo_id=photo_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(photo_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        photo_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/video/comments (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                photo_id=photo_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(photo_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) comments found for '{identifier}'. Try "
                "broadening the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouCommentReplies
# --------------------------------------------------------------------------


class ScavioKuaishouCommentRepliesInput(BaseModel):
    """Input schema for the ScavioKuaishouCommentReplies tool."""

    model_config = ConfigDict(extra="allow")

    photo_id: str = Field(
        description=(
            "Kuaishou photo (video) id."
        ),
    )

    root_comment_id: str = Field(
        description=(
            "Id of the root comment whose replies you want."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )

    count: Optional[int] = Field(
        default=None,
        description=(
            "Replies to return in this page, 1-50."
        ),
    )


class ScavioKuaishouCommentReplies(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Replies under a root comment on a Kuaishou video.

    Costs 1 credit per call.

    Pagination: cursor -> next_cursor; `count` sizes the page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouCommentReplies

            tool = ScavioKuaishouCommentReplies()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "photo_id": "3xf8v9pmcvexbhi",
                    "root_comment_id": "1234567890",
                }
            )
    """

    name: str = "scavio_kuaishou_comment_replies"
    description: str = (
        "Kuaishou (China): Replies under a root comment on a Kuaishou video. "
        "Pagination: cursor -> next_cursor; `count` sizes the page. Costs 1 credit per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouCommentRepliesInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouCommentRepliesAPIWrapper = Field(
        default_factory=ScavioKuaishouCommentRepliesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouCommentRepliesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        photo_id: str,
        root_comment_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/video/sub-comments (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                photo_id=photo_id,
                root_comment_id=root_comment_id,
                cursor=cursor,
                count=count,
            )
            return self._process_response(
                raw, _first_identifier(photo_id, root_comment_id)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        photo_id: str,
        root_comment_id: str,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/video/sub-comments (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                photo_id=photo_id,
                root_comment_id=root_comment_id,
                cursor=cursor,
                count=count,
            )
            return self._process_response(
                raw, _first_identifier(photo_id, root_comment_id)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) replies found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouVideosBatch
# --------------------------------------------------------------------------


class ScavioKuaishouVideosBatchInput(BaseModel):
    """Input schema for the ScavioKuaishouVideosBatch tool."""

    model_config = ConfigDict(extra="allow")

    photo_ids: list[str] = Field(
        description=(
            "Kuaishou photo ids to fetch in one call. Hard cap of 20 ids."
        ),
    )


class ScavioKuaishouVideosBatch(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Several Kuaishou videos in one call, max 20 photo ids.

    Costs 40 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouVideosBatch

            tool = ScavioKuaishouVideosBatch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"photo_ids": ["3xf8v9pmcvexbhi", "3x8kzxpmn7t6xyq"]})
    """

    name: str = "scavio_kuaishou_videos_batch"
    description: str = (
        "Kuaishou (China): Several Kuaishou videos in one call, max 20 photo ids. "
        "Costs 40 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouVideosBatchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouVideosBatchAPIWrapper = Field(
        default_factory=ScavioKuaishouVideosBatchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouVideosBatchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        photo_ids: list[str],
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/videos/batch (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                photo_ids=photo_ids,
            )
            return self._process_response(raw, _first_identifier(photo_ids))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        photo_ids: list[str],
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/videos/batch (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                photo_ids=photo_ids,
            )
            return self._process_response(raw, _first_identifier(photo_ids))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) videos found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouSearch
# --------------------------------------------------------------------------


class ScavioKuaishouSearchInput(BaseModel):
    """Input schema for the ScavioKuaishouSearch tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description=(
            "Search keyword."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioKuaishouSearch(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Mixed-result search across Kuaishou.

    Costs 10 credits per call.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouSearch

            tool = ScavioKuaishouSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "coffee"})
    """

    name: str = "scavio_kuaishou_search"
    description: str = (
        "Kuaishou (China): Mixed-result search across Kuaishou. Pagination: cursor -> "
        "next_cursor. Costs 10 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouSearchAPIWrapper = Field(
        default_factory=ScavioKuaishouSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                keyword=keyword,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(keyword))
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
        """Call POST /api/v1/kuaishou/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                keyword=keyword,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(keyword))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) results found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouSearchVideos
# --------------------------------------------------------------------------


class ScavioKuaishouSearchVideosInput(BaseModel):
    """Input schema for the ScavioKuaishouSearchVideos tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description=(
            "Search keyword."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioKuaishouSearchVideos(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Kuaishou video search results.

    Costs 10 credits per call.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouSearchVideos

            tool = ScavioKuaishouSearchVideos()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "coffee"})
    """

    name: str = "scavio_kuaishou_search_videos"
    description: str = (
        "Kuaishou (China): Kuaishou video search results. Pagination: cursor -> "
        "next_cursor. Costs 10 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouSearchVideosInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouSearchVideosAPIWrapper = Field(
        default_factory=ScavioKuaishouSearchVideosAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouSearchVideosAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/search/videos (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                keyword=keyword,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(keyword))
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
        """Call POST /api/v1/kuaishou/search/videos (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                keyword=keyword,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(keyword))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) video results found for '{identifier}'. Try "
                "broadening the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouSearchUsers
# --------------------------------------------------------------------------


class ScavioKuaishouSearchUsersInput(BaseModel):
    """Input schema for the ScavioKuaishouSearchUsers tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description=(
            "Search keyword."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioKuaishouSearchUsers(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Kuaishou user search results.

    Costs 10 credits per call.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouSearchUsers

            tool = ScavioKuaishouSearchUsers()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "coffee"})
    """

    name: str = "scavio_kuaishou_search_users"
    description: str = (
        "Kuaishou (China): Kuaishou user search results. Pagination: cursor -> "
        "next_cursor. Costs 10 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouSearchUsersInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouSearchUsersAPIWrapper = Field(
        default_factory=ScavioKuaishouSearchUsersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouSearchUsersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/search/users (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                keyword=keyword,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(keyword))
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
        """Call POST /api/v1/kuaishou/search/users (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                keyword=keyword,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(keyword))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) user results found for '{identifier}'. Try "
                "broadening the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouSearchLive
# --------------------------------------------------------------------------


class ScavioKuaishouSearchLiveInput(BaseModel):
    """Input schema for the ScavioKuaishouSearchLive tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description=(
            "Search keyword."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioKuaishouSearchLive(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Kuaishou live-stream search results.

    Costs 10 credits per call.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouSearchLive

            tool = ScavioKuaishouSearchLive()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "coffee"})
    """

    name: str = "scavio_kuaishou_search_live"
    description: str = (
        "Kuaishou (China): Kuaishou live-stream search results. Pagination: cursor -> "
        "next_cursor. Costs 10 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouSearchLiveInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouSearchLiveAPIWrapper = Field(
        default_factory=ScavioKuaishouSearchLiveAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouSearchLiveAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/search/live (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                keyword=keyword,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(keyword))
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
        """Call POST /api/v1/kuaishou/search/live (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                keyword=keyword,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(keyword))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) live results found for '{identifier}'. Try "
                "broadening the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouTagFeed
# --------------------------------------------------------------------------


class ScavioKuaishouTagFeedInput(BaseModel):
    """Input schema for the ScavioKuaishouTagFeed tool."""

    model_config = ConfigDict(extra="allow")

    tag: str = Field(
        description=(
            "Hashtag to read the feed for, without the leading #."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioKuaishouTagFeed(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Posts under a Kuaishou hashtag, cursor-paginated.

    Costs 1 credit per call.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouTagFeed

            tool = ScavioKuaishouTagFeed()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"tag": "coffee"})
    """

    name: str = "scavio_kuaishou_tag_feed"
    description: str = (
        "Kuaishou (China): Posts under a Kuaishou hashtag, cursor-paginated. "
        "Pagination: cursor -> next_cursor. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouTagFeedInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouTagFeedAPIWrapper = Field(
        default_factory=ScavioKuaishouTagFeedAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouTagFeedAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        tag: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/tag/feed (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                tag=tag,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(tag))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        tag: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/tag/feed (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                tag=tag,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(tag))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) posts found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKuaishouTrending
# --------------------------------------------------------------------------


class ScavioKuaishouTrendingInput(BaseModel):
    """Input schema for the ScavioKuaishouTrending tool."""

    model_config = ConfigDict(extra="allow")

    board: Optional[Literal["hot", "live", "shopping", "brand", "music"]] = Field(
        default=None,
        description=(
            "Which leaderboard to return. Options: hot, live, shopping, brand, music. "
            "Default: hot."
        ),
    )


class ScavioKuaishouTrending(BaseTool):  # type: ignore[override]
    """Kuaishou (China): Kuaishou hot / live / shopping / brand / music leaderboards.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKuaishouTrending

            tool = ScavioKuaishouTrending()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"board": "hot"})
    """

    name: str = "scavio_kuaishou_trending"
    description: str = (
        "Kuaishou (China): Kuaishou hot / live / shopping / brand / music "
        "leaderboards. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioKuaishouTrendingInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKuaishouTrendingAPIWrapper = Field(
        default_factory=ScavioKuaishouTrendingAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKuaishouTrendingAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        board: Optional[Literal["hot", "live", "shopping", "brand", "music"]] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/trending (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                board=board,
            )
            return self._process_response(raw, _first_identifier(board))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        board: Optional[Literal["hot", "live", "shopping", "brand", "music"]] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/kuaishou/trending (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                board=board,
            )
            return self._process_response(raw, _first_identifier(board))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No Kuaishou (China) trending data found for '{identifier}'. Try "
                "broadening the query or removing filters."
            )
        return raw
