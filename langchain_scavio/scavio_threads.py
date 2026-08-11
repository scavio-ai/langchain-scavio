"""Scavio Threads tools for LangChain agents.

Every URL, parameter name, enum and credit cost below is copied from the
Scavio route definition rather than derived from the tool name.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import (
    ScavioThreadsPostAPIWrapper,
    ScavioThreadsPostCommentsAPIWrapper,
    ScavioThreadsProfileAPIWrapper,
    ScavioThreadsSearchUsersAPIWrapper,
    ScavioThreadsUserPostsAPIWrapper,
    ScavioThreadsUserRepliesAPIWrapper,
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
# ScavioThreadsProfile
# --------------------------------------------------------------------------


class ScavioThreadsProfileInput(BaseModel):
    """Input schema for the ScavioThreadsProfile tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description=(
            "Threads handle without the @. Costs 2 extra credits because the handle "
            "has to be resolved with a second upstream call -- prefer user_id."
        ),
    )

    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric Threads user id, e.g. 63625256886. This is the cheap path."
        ),
    )


class ScavioThreadsProfile(BaseTool):  # type: ignore[override]
    """Threads: Profile details for a Threads user, by user_id (2cr) or username (4cr).

    Costs 2 credits when addressed by user_id and 4 credits when addressed by
    username -- the handle needs a second upstream lookup, so prefer user_id.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioThreadsProfile

            tool = ScavioThreadsProfile()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"user_id": "63625256886"})
    """

    name: str = "scavio_threads_profile"
    description: str = (
        "Threads: Profile details for a Threads user, by user_id (2cr) or username "
        "(4cr). Costs 2 credits when addressed by user_id and 4 credits when addressed "
        "by username -- the handle needs a second upstream lookup, so prefer user_id."
    )
    args_schema: Type[BaseModel] = ScavioThreadsProfileInput
    handle_tool_error: bool = True

    api_wrapper: ScavioThreadsProfileAPIWrapper = Field(
        default_factory=ScavioThreadsProfileAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioThreadsProfileAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/profile (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                username=username,
                user_id=user_id,
            )
            return self._process_response(raw, _first_identifier(username, user_id))
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
        """Call POST /api/v1/threads/profile (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username,
                user_id=user_id,
            )
            return self._process_response(raw, _first_identifier(username, user_id))
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
                f"No Threads profile found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioThreadsUserPosts
# --------------------------------------------------------------------------


class ScavioThreadsUserPostsInput(BaseModel):
    """Input schema for the ScavioThreadsUserPosts tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description=(
            "Threads handle without the @. Costs 2 extra credits because the handle "
            "has to be resolved with a second upstream call -- prefer user_id."
        ),
    )

    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric Threads user id, e.g. 63625256886. This is the cheap path."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioThreadsUserPosts(BaseTool):  # type: ignore[override]
    """Threads: A user's Threads posts, cursor-paginated.

    Costs 2 credits when addressed by user_id and 4 credits when addressed by
    username.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioThreadsUserPosts

            tool = ScavioThreadsUserPosts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"user_id": "63625256886"})
    """

    name: str = "scavio_threads_user_posts"
    description: str = (
        "Threads: A user's Threads posts, cursor-paginated. Pagination: cursor -> "
        "next_cursor. Costs 2 credits when addressed by user_id and 4 credits when "
        "addressed by username."
    )
    args_schema: Type[BaseModel] = ScavioThreadsUserPostsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioThreadsUserPostsAPIWrapper = Field(
        default_factory=ScavioThreadsUserPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioThreadsUserPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/user/posts (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                username=username,
                user_id=user_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(username, user_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/user/posts (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username,
                user_id=user_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(username, user_id))
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
                f"No Threads posts found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioThreadsUserReplies
# --------------------------------------------------------------------------


class ScavioThreadsUserRepliesInput(BaseModel):
    """Input schema for the ScavioThreadsUserReplies tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description=(
            "Threads handle without the @. Costs 2 extra credits because the handle "
            "has to be resolved with a second upstream call -- prefer user_id."
        ),
    )

    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric Threads user id, e.g. 63625256886. This is the cheap path."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioThreadsUserReplies(BaseTool):  # type: ignore[override]
    """Threads: A user's replies, cursor-paginated.

    Costs 2 credits when addressed by user_id and 4 credits when addressed by
    username.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioThreadsUserReplies

            tool = ScavioThreadsUserReplies()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"user_id": "63625256886"})
    """

    name: str = "scavio_threads_user_replies"
    description: str = (
        "Threads: A user's replies, cursor-paginated. Pagination: cursor -> "
        "next_cursor. Costs 2 credits when addressed by user_id and 4 credits when "
        "addressed by username."
    )
    args_schema: Type[BaseModel] = ScavioThreadsUserRepliesInput
    handle_tool_error: bool = True

    api_wrapper: ScavioThreadsUserRepliesAPIWrapper = Field(
        default_factory=ScavioThreadsUserRepliesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioThreadsUserRepliesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/user/replies (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                username=username,
                user_id=user_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(username, user_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        user_id: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/user/replies (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username,
                user_id=user_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(username, user_id))
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
                f"No Threads replies found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioThreadsPost
# --------------------------------------------------------------------------


class ScavioThreadsPostInput(BaseModel):
    """Input schema for the ScavioThreadsPost tool."""

    model_config = ConfigDict(extra="allow")

    post_id: Optional[str] = Field(
        default=None,
        description=(
            "Threads post id."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "A threads.net post URL, usable instead of post_id."
        ),
    )


class ScavioThreadsPost(BaseTool):  # type: ignore[override]
    """Threads: A single Threads post by id or threads.net URL.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioThreadsPost

            tool = ScavioThreadsPost()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"post_id": "3141592653589793"})
    """

    name: str = "scavio_threads_post"
    description: str = (
        "Threads: A single Threads post by id or threads.net URL. Costs 2 credits per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioThreadsPostInput
    handle_tool_error: bool = True

    api_wrapper: ScavioThreadsPostAPIWrapper = Field(
        default_factory=ScavioThreadsPostAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioThreadsPostAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        post_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/post (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                post_id=post_id,
                url=url,
            )
            return self._process_response(raw, _first_identifier(post_id, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        post_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/post (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                post_id=post_id,
                url=url,
            )
            return self._process_response(raw, _first_identifier(post_id, url))
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
                f"No Threads post found for '{identifier}'. Verify the identifiers you "
                "passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioThreadsPostComments
# --------------------------------------------------------------------------


class ScavioThreadsPostCommentsInput(BaseModel):
    """Input schema for the ScavioThreadsPostComments tool."""

    model_config = ConfigDict(extra="allow")

    post_id: str = Field(
        description=(
            "Threads post id."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor taken from a previous response's next_cursor. Keep the "
            "other arguments identical across paginated calls."
        ),
    )


class ScavioThreadsPostComments(BaseTool):  # type: ignore[override]
    """Threads: Replies to a Threads post, cursor-paginated.

    Costs 2 credits per call.

    Pagination: cursor -> next_cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioThreadsPostComments

            tool = ScavioThreadsPostComments()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"post_id": "3141592653589793"})
    """

    name: str = "scavio_threads_post_comments"
    description: str = (
        "Threads: Replies to a Threads post, cursor-paginated. Pagination: cursor -> "
        "next_cursor. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioThreadsPostCommentsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioThreadsPostCommentsAPIWrapper = Field(
        default_factory=ScavioThreadsPostCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioThreadsPostCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        post_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/post/comments (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                post_id=post_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(post_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        post_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/post/comments (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                post_id=post_id,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(post_id))
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
                f"No Threads comments found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioThreadsSearchUsers
# --------------------------------------------------------------------------


class ScavioThreadsSearchUsersInput(BaseModel):
    """Input schema for the ScavioThreadsSearchUsers tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Name or handle to look for."
        ),
    )


class ScavioThreadsSearchUsers(BaseTool):  # type: ignore[override]
    """Threads: Threads profiles matching a name or handle. This is the ONLY search
    Threads exposes.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioThreadsSearchUsers

            tool = ScavioThreadsSearchUsers()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "langchain"})
    """

    name: str = "scavio_threads_search_users"
    description: str = (
        "Threads: Threads profiles matching a name or handle. This is the ONLY search "
        "Threads exposes. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioThreadsSearchUsersInput
    handle_tool_error: bool = True

    api_wrapper: ScavioThreadsSearchUsersAPIWrapper = Field(
        default_factory=ScavioThreadsSearchUsersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioThreadsSearchUsersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/search/users (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/threads/search/users (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
            )
            return self._process_response(raw, _first_identifier(query))
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
                f"No Threads user results found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw
