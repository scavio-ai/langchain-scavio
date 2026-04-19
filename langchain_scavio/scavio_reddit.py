"""ScavioRedditSearch and ScavioRedditPost tools for LangChain agents."""

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
    ScavioRedditPostAPIWrapper,
    ScavioRedditSearchAPIWrapper,
)

logger = logging.getLogger(__name__)

_SEARCH_INIT_ONLY_PARAMS = frozenset({"max_results"})


class ScavioRedditSearchInput(BaseModel):
    """Input schema for ScavioRedditSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Reddit search query, 1-500 characters.",
        min_length=1,
        max_length=500,
    )

    type: Optional[Literal["posts", "comments"]] = Field(
        default=None,
        description='Search scope. Options: "posts" (default), "comments".',
    )

    sort: Optional[
        Literal["new", "relevance", "hot", "top", "comments"]
    ] = Field(
        default=None,
        description=(
            'Sort order. Options: "new" (default), "relevance", "hot", "top", '
            '"comments".'
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's nextCursor. "
            "Keep query/type/sort the same across paginated calls."
        ),
    )


class ScavioRedditSearch(BaseTool):  # type: ignore[override]
    """Search Reddit posts or comments using the Scavio API.

    Returns Reddit posts with titles, URLs, subreddits, authors, and timestamps.
    Supports pagination via an opaque cursor.

    Note: Reddit requires JS rendering; responses typically take 5-15 seconds.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditSearch

            tool = ScavioRedditSearch(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "langchain", "sort": "new"})
    """

    name: str = "scavio_reddit_search"
    description: str = (
        "Search Reddit posts or comments using the Scavio API. "
        "Returns post titles, URLs, subreddits, authors, and timestamps. "
        "Supports sort orders (new, relevance, hot, top, comments) and pagination. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioRedditSearchInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 5

    api_wrapper: ScavioRedditSearchAPIWrapper = Field(
        default_factory=ScavioRedditSearchAPIWrapper  # type: ignore[arg-type]
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
            kwargs["api_wrapper"] = ScavioRedditSearchAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        type: Optional[str] = None,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a synchronous Reddit search."""
        forbidden = _SEARCH_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                type=type,
                sort=sort,
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
        type: Optional[str] = None,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an asynchronous Reddit search."""
        forbidden = _SEARCH_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                type=type,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], query: str) -> dict[str, Any]:
        """Truncate posts and raise ToolException if empty."""
        data = raw.get("data") or {}
        posts = data.get("posts") if isinstance(data, dict) else None
        if self.max_results and posts:
            raw["data"]["posts"] = posts[: self.max_results]
        if not (isinstance(data, dict) and data.get("posts")):
            raise ToolException(
                f"No Reddit results found for '{query}'. "
                "Try broadening the query or changing the sort order."
            )
        return raw


class ScavioRedditPostInput(BaseModel):
    """Input schema for ScavioRedditPost tool."""

    model_config = ConfigDict(extra="allow")

    url: str = Field(
        description=(
            "Full Reddit post URL (www., old., or new. subdomains accepted). "
            "Use ScavioRedditSearch first to find post URLs if needed."
        )
    )


class ScavioRedditPost(BaseTool):  # type: ignore[override]
    """Fetch a Reddit post's metadata and comment thread.

    Returns post fields (title, body, score, awards, media) plus a flat list
    of comments with a ``depth`` field for reconstructing the reply tree.
    Pairs with ScavioRedditSearch -- feed any post URL from search results
    directly into this tool.

    Note: Reddit requires JS rendering; responses typically take 5-15 seconds.
    Only the initial comment view is returned; lazy-loaded "more replies" are
    not fetched, so ``commentCount`` may exceed ``len(comments)``.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditPost

            tool = ScavioRedditPost()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({
                "url": "https://www.reddit.com/r/programming/comments/abc123/example/"
            })
    """

    name: str = "scavio_reddit_post"
    description: str = (
        "Fetch a single Reddit post's metadata and comment thread. "
        "Returns the post (title, body, score, media, flair) and a flat list "
        "of comments with depth for reconstructing the tree. "
        "Use ScavioRedditSearch to find post URLs. "
        "Input should be a full Reddit post URL."
    )
    args_schema: Type[BaseModel] = ScavioRedditPostInput
    handle_tool_error: bool = True

    api_wrapper: ScavioRedditPostAPIWrapper = Field(
        default_factory=ScavioRedditPostAPIWrapper  # type: ignore[arg-type]
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
            kwargs["api_wrapper"] = ScavioRedditPostAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        url: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Reddit post details synchronously."""
        try:
            raw = self.api_wrapper.raw_results(url=url)
            return self._process_response(raw, url)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        url: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Reddit post details asynchronously."""
        try:
            raw = await self.api_wrapper.raw_results_async(url=url)
            return self._process_response(raw, url)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], url: str) -> dict[str, Any]:
        """Raise ToolException if no post data returned."""
        data = raw.get("data") or {}
        post = data.get("post") if isinstance(data, dict) else None
        if not post:
            raise ToolException(
                f"No Reddit post found at '{url}'. "
                "Verify the URL points to a valid Reddit post."
            )
        return raw
