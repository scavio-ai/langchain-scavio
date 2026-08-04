"""Scavio Reddit tools for LangChain agents (12 endpoints, 1 credit each)."""

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
    ScavioRedditCommentRepliesAPIWrapper,
    ScavioRedditPopularAPIWrapper,
    ScavioRedditPostAPIWrapper,
    ScavioRedditPostCommentsAPIWrapper,
    ScavioRedditSearchAPIWrapper,
    ScavioRedditSearchSuggestionsAPIWrapper,
    ScavioRedditSubredditAPIWrapper,
    ScavioRedditSubredditPostsAPIWrapper,
    ScavioRedditTrendingAPIWrapper,
    ScavioRedditUserAPIWrapper,
    ScavioRedditUserCommentsAPIWrapper,
    ScavioRedditUserPostsAPIWrapper,
)

logger = logging.getLogger(__name__)

_SEARCH_INIT_ONLY_PARAMS = frozenset({"max_results"})
_LIST_INIT_ONLY_PARAMS = frozenset({"max_results"})

# Comments and user feeds. Values are UPPERCASE and passed through verbatim.
Sort = Literal["HOT", "NEW", "TOP", "BEST", "CONTROVERSIAL"]
# The subreddit feed is the only endpoint that also accepts RISING.
FeedSort = Literal["BEST", "HOT", "NEW", "TOP", "CONTROVERSIAL", "RISING"]


def _forward_api_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract API wrapper kwargs from tool kwargs."""
    api_kwargs: dict[str, Any] = {}
    if "scavio_api_key" in kwargs:
        api_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
    if "api_base_url" in kwargs:
        api_kwargs["api_base_url"] = kwargs.pop("api_base_url")
    if "max_requests_per_second" in kwargs:
        api_kwargs["max_requests_per_second"] = kwargs.pop(
            "max_requests_per_second"
        )
    return api_kwargs


class ScavioRedditSearchInput(BaseModel):
    """Input schema for ScavioRedditSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Reddit search query, 1-500 characters.",
        min_length=1,
        max_length=500,
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. "
            "Keep query the same across paginated calls."
        ),
    )


class ScavioRedditSearch(BaseTool):  # type: ignore[override]
    """Search Reddit posts using the Scavio API.

    Returns Reddit posts with titles, URLs, subreddits, authors, and timestamps
    under ``data.results``. Supports pagination via an opaque cursor.

    The API returns relevance order only; it has no sort or result-type filter.

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

            result = tool.invoke({"query": "langchain"})
    """

    name: str = "scavio_reddit_search"
    description: str = (
        "Search Reddit posts using the Scavio API. "
        "Returns post titles, URLs, subreddits, authors, and timestamps "
        "under data.results, in relevance order. Supports cursor pagination. "
        "Costs 1 credit per call. Input should be a search query."
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
                cursor=cursor,
            )
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
        if not results:
            raise ToolException(
                f"No Reddit results found for '{query}'. Try broadening the query."
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
    """Fetch a single Reddit post's metadata.

    Returns the post as a flat object under ``data``: post_id, title, text,
    url, subreddit, author, score, upvote_ratio, num_comments, created_at,
    is_nsfw, is_video, thumbnail, media. Pairs with ScavioRedditSearch --
    feed any post URL from search results directly into this tool.

    This endpoint does NOT return comments. Use the post_id it returns with
    the Scavio /reddit/post/comments endpoint to fetch the comment thread.

    Note: Reddit requires JS rendering; responses typically take 5-15 seconds.

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
        "Fetch a single Reddit post's metadata by URL. "
        "Returns a flat post object (title, text, score, upvote_ratio, "
        "num_comments, media) under data. Does not return comments. "
        "Use ScavioRedditSearch to find post URLs. Costs 1 credit per call. "
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
        """Raise ToolException if no post data returned.

        /api/v1/reddit/post returns the post as a flat object under `data`
        (post_id, title, text, ...), not nested under a `post` key.
        """
        data = raw.get("data") or {}
        if not (isinstance(data, dict) and data.get("post_id")):
            raise ToolException(
                f"No Reddit post found at '{url}'. "
                "Verify the URL points to a valid Reddit post."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditSearchSuggestions
# ---------------------------------------------------------------------------


class ScavioRedditSearchSuggestionsInput(BaseModel):
    """Input schema for ScavioRedditSearchSuggestions tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Partial search query to autocomplete, 1-500 characters.",
        min_length=1,
        max_length=500,
    )


class ScavioRedditSearchSuggestions(BaseTool):  # type: ignore[override]
    """Fetch Reddit search autocomplete suggestions using the Scavio API.

    Returns a plain list of strings under ``data.suggestions`` plus
    ``data.total_count``. Useful for query expansion before calling
    ScavioRedditSearch. There is no cursor on this endpoint.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditSearchSuggestions

            tool = ScavioRedditSearchSuggestions()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "python"})
    """

    name: str = "scavio_reddit_search_suggestions"
    description: str = (
        "Fetch Reddit search autocomplete suggestions for a partial query. Returns a "
        "list of strings under data.suggestions with data.total_count. Use it to "
        "expand a query before calling scavio_reddit_search. Costs 1 credit per call. "
        "Input should be a partial search query."
    )
    args_schema: Type[BaseModel] = ScavioRedditSearchSuggestionsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioRedditSearchSuggestionsAPIWrapper = Field(
        default_factory=ScavioRedditSearchSuggestionsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditSearchSuggestionsAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Reddit search suggestions (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(query=query)
            return self._process_response(raw, query)
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
        """Fetch Reddit search suggestions (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(query=query)
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], query: str) -> dict[str, Any]:
        """Truncate the suggestion list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        suggestions = data.get("suggestions")
        if self.max_results and suggestions:
            raw["data"]["suggestions"] = suggestions[: self.max_results]
        if not suggestions:
            raise ToolException(
                f"No Reddit search suggestions found for '{query}'. Try a shorter "
                "or more common prefix."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditPostComments
# ---------------------------------------------------------------------------


class ScavioRedditPostCommentsInput(BaseModel):
    """Input schema for ScavioRedditPostComments tool."""

    model_config = ConfigDict(extra="allow")

    post_id: str = Field(
        description=(
            "Post fullname ('t3_1v6ngaf'), a bare post id, or a full Reddit post "
            "URL. ScavioRedditPost returns the post_id in data.post_id."
        ),
        min_length=1,
    )

    sort: Optional[Sort] = Field(
        default=None,
        description=(
            "Comment sort order: HOT, NEW, TOP, BEST or CONTROVERSIAL. "
            "UPPERCASE; the server defaults to TOP."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioRedditPostComments(BaseTool):  # type: ignore[override]
    """Fetch the top-level comments on a Reddit post using the Scavio API.

    Returns comments under ``data.comments`` with comment_id, text, author,
    score, created_at, depth and reply_cursor. Paginate with
    ``data.next_cursor`` and stop when ``data.has_more`` is false.

    Feed a comment's ``reply_cursor`` to ScavioRedditCommentReplies to expand
    that comment's thread -- ``next_cursor`` will not work there.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditPostComments

            tool = ScavioRedditPostComments()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"post_id": "t3_1v6ngaf", "sort": "NEW"})
    """

    name: str = "scavio_reddit_post_comments"
    description: str = (
        "Fetch top-level comments on a Reddit post. Returns comments under "
        "data.comments with comment_id, text, author, score, created_at, depth and "
        "reply_cursor. sort takes UPPERCASE HOT, NEW, TOP, BEST or CONTROVERSIAL "
        "(default TOP). Supports cursor pagination. Costs 1 credit per call. Input "
        "should be a post fullname, bare post id or post URL."
    )
    args_schema: Type[BaseModel] = ScavioRedditPostCommentsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioRedditPostCommentsAPIWrapper = Field(
        default_factory=ScavioRedditPostCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditPostCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        post_id: str,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a Reddit post's top-level comments (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                post_id=post_id,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, post_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        post_id: str,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a Reddit post's top-level comments (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                post_id=post_id,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, post_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], post_id: str) -> dict[str, Any]:
        """Truncate the comment list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        comments = data.get("comments")
        if self.max_results and comments:
            raw["data"]["comments"] = comments[: self.max_results]
        if not comments:
            raise ToolException(
                f"No comments found for Reddit post '{post_id}'. The post may have "
                "no comments, or be locked or removed."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditCommentReplies
# ---------------------------------------------------------------------------


class ScavioRedditCommentRepliesInput(BaseModel):
    """Input schema for ScavioRedditCommentReplies tool."""

    model_config = ConfigDict(extra="allow")

    post_id: str = Field(
        description=(
            "Post fullname ('t3_1v6ngaf'), a bare post id, or a full Reddit post URL."
        ),
        min_length=1,
    )

    cursor: str = Field(
        description=(
            "Required. The reply_cursor of the comment to expand, taken from a "
            "comment returned by ScavioRedditPostComments. A next_cursor will not "
            "work here."
        ),
        min_length=1,
    )

    sort: Optional[Sort] = Field(
        default=None,
        description=(
            "Reply sort order: HOT, NEW, TOP, BEST or CONTROVERSIAL. "
            "UPPERCASE; the server defaults to TOP."
        ),
    )


class ScavioRedditCommentReplies(BaseTool):  # type: ignore[override]
    """Fetch the replies to one Reddit comment using the Scavio API.

    Returns replies under ``data.replies`` in the same comment shape as
    ScavioRedditPostComments. Paginate with ``data.next_cursor``.

    ``cursor`` is required here and must be the ``reply_cursor`` of the comment
    being expanded, not a ``next_cursor``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditCommentReplies

            tool = ScavioRedditCommentReplies()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({
                "post_id": "t3_1v6ngaf",
                "cursor": comment["reply_cursor"],
            })
    """

    name: str = "scavio_reddit_comment_replies"
    description: str = (
        "Fetch replies to a specific Reddit comment. Returns replies under "
        "data.replies in the same comment shape as scavio_reddit_post_comments. "
        "cursor is REQUIRED and must be the reply_cursor of the comment to expand, "
        "not a next_cursor. sort takes UPPERCASE HOT, NEW, TOP, BEST or "
        "CONTROVERSIAL (default TOP). Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioRedditCommentRepliesInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioRedditCommentRepliesAPIWrapper = Field(
        default_factory=ScavioRedditCommentRepliesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditCommentRepliesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        post_id: str,
        cursor: str,
        sort: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch replies to a Reddit comment (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                post_id=post_id,
                cursor=cursor,
                sort=sort,
            )
            return self._process_response(raw, post_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        post_id: str,
        cursor: str,
        sort: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch replies to a Reddit comment (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                post_id=post_id,
                cursor=cursor,
                sort=sort,
            )
            return self._process_response(raw, post_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], post_id: str) -> dict[str, Any]:
        """Truncate the reply list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        replies = data.get("replies")
        if self.max_results and replies:
            raw["data"]["replies"] = replies[: self.max_results]
        if not replies:
            raise ToolException(
                f"No replies found on Reddit post '{post_id}'. Verify cursor is the "
                "reply_cursor of a comment from scavio_reddit_post_comments."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditSubreddit
# ---------------------------------------------------------------------------


class ScavioRedditSubredditInput(BaseModel):
    """Input schema for ScavioRedditSubreddit tool."""

    model_config = ConfigDict(extra="allow")

    subreddit: str = Field(
        description=(
            "Subreddit name without the r/ prefix (e.g. 'AskReddit'), 1-100 "
            "characters."
        ),
        min_length=1,
        max_length=100,
    )


class ScavioRedditSubreddit(BaseTool):  # type: ignore[override]
    """Fetch a subreddit's metadata using the Scavio API.

    Returns the subreddit as a flat object under ``data``: id, name,
    prefixed_name, title, description, public_description, subscribers,
    active_count, type, is_nsfw, icon, banner, primary_color, created_at and
    url. It does NOT return posts -- use ScavioRedditSubredditPosts for those.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditSubreddit

            tool = ScavioRedditSubreddit()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"subreddit": "AskReddit"})
    """

    name: str = "scavio_reddit_subreddit"
    description: str = (
        "Fetch a subreddit's metadata by name. Returns a flat object under data with "
        "title, description, subscribers, active_count, type, is_nsfw, icon, banner "
        "and created_at. Does not return posts -- use scavio_reddit_subreddit_posts "
        "for those. Costs 1 credit per call. Input should be a subreddit name "
        "without the r/ prefix."
    )
    args_schema: Type[BaseModel] = ScavioRedditSubredditInput
    handle_tool_error: bool = True

    api_wrapper: ScavioRedditSubredditAPIWrapper = Field(
        default_factory=ScavioRedditSubredditAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditSubredditAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        subreddit: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch subreddit metadata (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(subreddit=subreddit)
            return self._process_response(raw, subreddit)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        subreddit: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch subreddit metadata (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(subreddit=subreddit)
            return self._process_response(raw, subreddit)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], subreddit: str) -> dict[str, Any]:
        """Raise ToolException when the response carries no subreddit."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not data.get("name"):
            raise ToolException(
                f"No Reddit subreddit found for '{subreddit}'. Pass the bare name "
                "without the r/ prefix and check it is public."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditSubredditPosts
# ---------------------------------------------------------------------------


class ScavioRedditSubredditPostsInput(BaseModel):
    """Input schema for ScavioRedditSubredditPosts tool."""

    model_config = ConfigDict(extra="allow")

    subreddit: str = Field(
        description=(
            "Subreddit name without the r/ prefix (e.g. 'AskReddit'), 1-100 "
            "characters."
        ),
        min_length=1,
        max_length=100,
    )

    sort: Optional[FeedSort] = Field(
        default=None,
        description=(
            "Feed sort order: BEST, HOT, NEW, TOP, CONTROVERSIAL or RISING. "
            "UPPERCASE; the server defaults to HOT. This is the only Reddit "
            "endpoint that accepts RISING."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioRedditSubredditPosts(BaseTool):  # type: ignore[override]
    """Fetch a subreddit's post feed using the Scavio API.

    Returns posts under ``data.posts`` with post_id, title, author,
    author_icon, subreddit, created_at, score, num_comments, share_count and
    url. This feed shape carries no ``text``, ``thumbnail`` or ``is_nsfw`` --
    fetch a post_id through ScavioRedditPost for the full body. Paginate with
    ``data.next_cursor`` and stop when ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditSubredditPosts

            tool = ScavioRedditSubredditPosts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"subreddit": "programming", "sort": "RISING"})
    """

    name: str = "scavio_reddit_subreddit_posts"
    description: str = (
        "Fetch a subreddit's post feed. Returns posts under data.posts with post_id, "
        "title, author, subreddit, created_at, score, num_comments, share_count and "
        "url (no body text). sort takes UPPERCASE BEST, HOT, NEW, TOP, CONTROVERSIAL "
        "or RISING (default HOT) -- the only endpoint accepting RISING. Supports "
        "cursor pagination. Costs 1 credit per call. Input should be a subreddit name "
        "without the r/ prefix."
    )
    args_schema: Type[BaseModel] = ScavioRedditSubredditPostsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioRedditSubredditPostsAPIWrapper = Field(
        default_factory=ScavioRedditSubredditPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditSubredditPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        subreddit: str,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a subreddit's post feed (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                subreddit=subreddit,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, subreddit)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        subreddit: str,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a subreddit's post feed (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                subreddit=subreddit,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, subreddit)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], subreddit: str) -> dict[str, Any]:
        """Truncate the post list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        posts = data.get("posts")
        if self.max_results and posts:
            raw["data"]["posts"] = posts[: self.max_results]
        if not posts:
            raise ToolException(
                f"No posts found in Reddit subreddit '{subreddit}'. Verify the name "
                "and that the subreddit is public."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditUser
# ---------------------------------------------------------------------------


class ScavioRedditUserInput(BaseModel):
    """Input schema for ScavioRedditUser tool."""

    model_config = ConfigDict(extra="allow")

    username: str = Field(
        description=(
            "Reddit username without the u/ prefix (e.g. 'spez'), 1-100 characters."
        ),
        min_length=1,
        max_length=100,
    )


class ScavioRedditUser(BaseTool):  # type: ignore[override]
    """Fetch a redditor's profile using the Scavio API.

    Returns the profile as a flat object under ``data``: id, name,
    is_employee, is_verified, account_type, is_accepting_pms, is_nsfw, avatar,
    karma, post_karma, comment_karma, description and created_at. It does NOT
    return submissions -- use ScavioRedditUserPosts or ScavioRedditUserComments.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditUser

            tool = ScavioRedditUser()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "spez"})
    """

    name: str = "scavio_reddit_user"
    description: str = (
        "Fetch a redditor's profile by username. Returns a flat object under data "
        "with karma, post_karma, comment_karma, avatar, description, is_verified and "
        "created_at. Does not return submissions -- use scavio_reddit_user_posts or "
        "scavio_reddit_user_comments. Costs 1 credit per call. Input should be a "
        "username without the u/ prefix."
    )
    args_schema: Type[BaseModel] = ScavioRedditUserInput
    handle_tool_error: bool = True

    api_wrapper: ScavioRedditUserAPIWrapper = Field(
        default_factory=ScavioRedditUserAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditUserAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a redditor's profile (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(username=username)
            return self._process_response(raw, username)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a redditor's profile (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(username=username)
            return self._process_response(raw, username)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], username: str) -> dict[str, Any]:
        """Raise ToolException when the response carries no profile."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not data.get("name"):
            raise ToolException(
                f"No Reddit user found for '{username}'. Pass the bare handle "
                "without the u/ prefix; the account may be suspended or deleted."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditUserPosts
# ---------------------------------------------------------------------------


class ScavioRedditUserPostsInput(BaseModel):
    """Input schema for ScavioRedditUserPosts tool."""

    model_config = ConfigDict(extra="allow")

    username: str = Field(
        description=(
            "Reddit username without the u/ prefix (e.g. 'spez'), 1-100 characters."
        ),
        min_length=1,
        max_length=100,
    )

    sort: Optional[Sort] = Field(
        default=None,
        description=(
            "Sort order: HOT, NEW, TOP, BEST or CONTROVERSIAL. UPPERCASE; the "
            "server defaults to NEW. RISING is not accepted here."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioRedditUserPosts(BaseTool):  # type: ignore[override]
    """Fetch a redditor's submitted posts using the Scavio API.

    Returns posts under ``data.posts`` with post_id, title, subreddit, author,
    score, num_comments, url, created_at, is_nsfw and thumbnail. Paginate with
    ``data.next_cursor`` and stop when ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditUserPosts

            tool = ScavioRedditUserPosts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "spez", "sort": "TOP"})
    """

    name: str = "scavio_reddit_user_posts"
    description: str = (
        "Fetch a redditor's submitted posts. Returns posts under data.posts with "
        "post_id, title, subreddit, score, num_comments, url, created_at, is_nsfw "
        "and thumbnail. sort takes UPPERCASE HOT, NEW, TOP, BEST or CONTROVERSIAL "
        "(default NEW). Supports cursor pagination. Costs 1 credit per call. Input "
        "should be a username without the u/ prefix."
    )
    args_schema: Type[BaseModel] = ScavioRedditUserPostsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioRedditUserPostsAPIWrapper = Field(
        default_factory=ScavioRedditUserPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditUserPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: str,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a redditor's submitted posts (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                username=username,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, username)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: str,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a redditor's submitted posts (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, username)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], username: str) -> dict[str, Any]:
        """Truncate the post list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        posts = data.get("posts")
        if self.max_results and posts:
            raw["data"]["posts"] = posts[: self.max_results]
        if not posts:
            raise ToolException(
                f"No posts found for Reddit user '{username}'. The account may have "
                "only comments, or be suspended or deleted."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditUserComments
# ---------------------------------------------------------------------------


class ScavioRedditUserCommentsInput(BaseModel):
    """Input schema for ScavioRedditUserComments tool."""

    model_config = ConfigDict(extra="allow")

    username: str = Field(
        description=(
            "Reddit username without the u/ prefix (e.g. 'spez'), 1-100 characters."
        ),
        min_length=1,
        max_length=100,
    )

    sort: Optional[Sort] = Field(
        default=None,
        description=(
            "Sort order: HOT, NEW, TOP, BEST or CONTROVERSIAL. UPPERCASE; the "
            "server defaults to NEW."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioRedditUserComments(BaseTool):  # type: ignore[override]
    """Fetch a redditor's comments using the Scavio API.

    Returns comments under ``data.comments`` with comment_id, text, author,
    score, created_at and a nested ``post`` of {id, title}. This shape has no
    ``depth`` and no ``reply_cursor``, unlike ScavioRedditPostComments.
    Paginate with ``data.next_cursor``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditUserComments

            tool = ScavioRedditUserComments()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "spez", "sort": "TOP"})
    """

    name: str = "scavio_reddit_user_comments"
    description: str = (
        "Fetch a redditor's comments. Returns comments under data.comments with "
        "comment_id, text, score, created_at and a nested post of id and title -- no "
        "depth and no reply_cursor here. sort takes UPPERCASE HOT, NEW, TOP, BEST or "
        "CONTROVERSIAL (default NEW). Supports cursor pagination. Costs 1 credit per "
        "call. Input should be a username without the u/ prefix."
    )
    args_schema: Type[BaseModel] = ScavioRedditUserCommentsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioRedditUserCommentsAPIWrapper = Field(
        default_factory=ScavioRedditUserCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditUserCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: str,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a redditor's comments (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                username=username,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, username)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: str,
        sort: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a redditor's comments (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username,
                sort=sort,
                cursor=cursor,
            )
            return self._process_response(raw, username)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], username: str) -> dict[str, Any]:
        """Truncate the comment list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        comments = data.get("comments")
        if self.max_results and comments:
            raw["data"]["comments"] = comments[: self.max_results]
        if not comments:
            raise ToolException(
                f"No comments found for Reddit user '{username}'. The account may "
                "have only posts, or be suspended or deleted."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditPopular
# ---------------------------------------------------------------------------


class ScavioRedditPopularInput(BaseModel):
    """Input schema for ScavioRedditPopular tool."""

    model_config = ConfigDict(extra="allow")

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Omit it for "
            "the first page. This is the endpoint's only parameter."
        ),
    )


class ScavioRedditPopular(BaseTool):  # type: ignore[override]
    """Fetch the site-wide Reddit popular feed using the Scavio API.

    Returns posts under ``data.posts`` with post_id, title, subreddit, author,
    score, num_comments, url, created_at, is_nsfw and thumbnail. There is no
    sort and no subreddit filter: ``cursor`` is the only parameter. Paginate
    with ``data.next_cursor`` and stop when ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditPopular

            tool = ScavioRedditPopular()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({})
    """

    name: str = "scavio_reddit_popular"
    description: str = (
        "Fetch the site-wide Reddit popular feed (r/popular). Returns posts under "
        "data.posts with post_id, title, subreddit, author, score, num_comments, url, "
        "created_at, is_nsfw and thumbnail. cursor is the only parameter -- there is "
        "no sort and no subreddit filter. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioRedditPopularInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioRedditPopularAPIWrapper = Field(
        default_factory=ScavioRedditPopularAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditPopularAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the Reddit popular feed (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(cursor=cursor)
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the Reddit popular feed (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(cursor=cursor)
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Truncate the post list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        posts = data.get("posts")
        if self.max_results and posts:
            raw["data"]["posts"] = posts[: self.max_results]
        if not posts:
            raise ToolException(
                "No posts found in the Reddit popular feed. If a cursor was passed, "
                "the feed may be exhausted -- retry without it."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioRedditTrending
# ---------------------------------------------------------------------------


class ScavioRedditTrendingInput(BaseModel):
    """Input schema for ScavioRedditTrending tool (no parameters)."""

    model_config = ConfigDict(extra="allow")


class ScavioRedditTrending(BaseTool):  # type: ignore[override]
    """Fetch the current trending Reddit search queries using the Scavio API.

    Returns trends under ``data.trending``, each ``{query, raw_query}``, plus
    ``data.total_count``. The endpoint takes no parameters and has no cursor.
    Feed a ``query`` straight into ScavioRedditSearch.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedditTrending

            tool = ScavioRedditTrending()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({})
    """

    name: str = "scavio_reddit_trending"
    description: str = (
        "Fetch the Reddit search queries trending right now. Returns them under "
        "data.trending as query and raw_query pairs, with data.total_count. Takes no "
        "parameters and has no cursor. Feed a query into scavio_reddit_search. Costs "
        "1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioRedditTrendingInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioRedditTrendingAPIWrapper = Field(
        default_factory=ScavioRedditTrendingAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedditTrendingAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch trending Reddit search queries (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results()
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch trending Reddit search queries (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async()
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Truncate the trend list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        trending = data.get("trending")
        if self.max_results and trending:
            raw["data"]["trending"] = trending[: self.max_results]
        if not trending:
            raise ToolException(
                "No trending Reddit queries returned. This is transient -- retry, or "
                "use scavio_reddit_popular for the popular feed instead."
            )
        return raw
