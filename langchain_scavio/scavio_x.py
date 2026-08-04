"""Scavio X (Twitter) tools for LangChain agents."""

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
    ScavioXSearchAPIWrapper,
    ScavioXTrendingAPIWrapper,
    ScavioXTweetAPIWrapper,
    ScavioXTweetCommentsAPIWrapper,
    ScavioXTweetRetweetersAPIWrapper,
    ScavioXUserAPIWrapper,
    ScavioXUserFollowersAPIWrapper,
    ScavioXUserFollowingsAPIWrapper,
    ScavioXUserMediaAPIWrapper,
    ScavioXUserRepliesAPIWrapper,
    ScavioXUserTweetsAPIWrapper,
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
        api_kwargs["max_requests_per_second"] = kwargs.pop(
            "max_requests_per_second"
        )
    return api_kwargs


# ---------------------------------------------------------------------------
# ScavioXSearch
# ---------------------------------------------------------------------------


class ScavioXSearchInput(BaseModel):
    """Input schema for ScavioXSearch tool."""

    model_config = ConfigDict(extra="allow")

    search: str = Field(
        description=(
            "Search query, 1-500 characters. Supports X search operators such as "
            "from:handle, #hashtag and -exclude."
        ),
    )

    search_type: Optional[
        Literal["Top", "Latest", "People", "Photos", "Videos"]
    ] = Field(
        default=None,
        description=(
            "Result category. Options: Top (default), Latest, People, Photos, Videos. "
            "Values are case-sensitive and capitalised."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioXSearch(BaseTool):  # type: ignore[override]
    """Search tweets and people on X (Twitter) using the Scavio API.

    Returns matching tweets under ``data.timeline`` with text, engagement
    counts, author and media. Paginate with ``data.next_cursor`` and stop
    when ``data.has_more`` is false.

    The query argument is named ``search``, mirroring the API wire field.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXSearch

            tool = ScavioXSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"search": "langchain", "search_type": "Latest"})
    """

    name: str = "scavio_x_search"
    description: str = (
        "Search tweets and people on X (Twitter). Returns tweets under data.timeline "
        "with text, favorites/retweets/replies/views, author and media. search_type "
        "selects Top, Latest, People, Photos or Videos. Supports cursor pagination. "
        "Costs 1 credit per call. Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioXSearchInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXSearchAPIWrapper = Field(
        default_factory=ScavioXSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        search: str,
        search_type: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search tweets and people on X (Twitter) (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                search=search,
                search_type=search_type,
                cursor=cursor,
            )
            return self._process_response(raw, search)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        search: str,
        search_type: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search tweets and people on X (Twitter) (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                search=search,
                search_type=search_type,
                cursor=cursor,
            )
            return self._process_response(raw, search)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], search: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        timeline = data.get("timeline")
        if self.max_results and timeline:
            raw["data"]["timeline"] = timeline[: self.max_results]
        if not timeline:
            raise ToolException(
                f"No X results found for '{search}'. Try broadening the query or a "
                "different search_type."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXTweet
# ---------------------------------------------------------------------------


class ScavioXTweetInput(BaseModel):
    """Input schema for ScavioXTweet tool."""

    model_config = ConfigDict(extra="allow")

    tweet_id: str = Field(
        description=(
            "Numeric tweet id as a string (e.g. '1808168603721650364'). Use "
            "ScavioXSearch to find tweet ids if needed."
        ),
    )


class ScavioXTweet(BaseTool):  # type: ignore[override]
    """Fetch a single tweet by id using the Scavio API.

    Returns the tweet as a flat object under ``data``: tweet_id, text, lang,
    created_at, favorites, retweets, replies, quotes, bookmarks, views,
    source, conversation_id, author and media, plus reply_to,
    in_reply_to_screen_name, in_reply_to_status_id, in_reply_to_user_id and
    sensitive.

    Retweets and quotes are resolved one level deep into ``retweeted_tweet``
    and ``quoted``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXTweet

            tool = ScavioXTweet()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"tweet_id": "1808168603721650364"})
    """

    name: str = "scavio_x_tweet"
    description: str = (
        "Fetch a single X (Twitter) tweet by its numeric id. Returns a flat tweet "
        "object under data with text, engagement counts, author, media and reply "
        "context. Costs 1 credit per call. Input should be a numeric tweet id."
    )
    args_schema: Type[BaseModel] = ScavioXTweetInput
    handle_tool_error: bool = True

    api_wrapper: ScavioXTweetAPIWrapper = Field(
        default_factory=ScavioXTweetAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXTweetAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        tweet_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a single tweet by id (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                tweet_id=tweet_id,
            )
            return self._process_response(raw, tweet_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        tweet_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a single tweet by id (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                tweet_id=tweet_id,
            )
            return self._process_response(raw, tweet_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], tweet_id: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not data.get("tweet_id"):
            raise ToolException(
                f"No X tweet found for id '{tweet_id}'. Verify the id is correct and "
                "the tweet is public."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXTweetComments
# ---------------------------------------------------------------------------


class ScavioXTweetCommentsInput(BaseModel):
    """Input schema for ScavioXTweetComments tool."""

    model_config = ConfigDict(extra="allow")

    tweet_id: str = Field(
        description="Numeric tweet id as a string.",
    )

    rank: Optional[Literal["top", "latest"]] = Field(
        default=None,
        description=(
            "Reply ordering. 'top' (default) returns ranked replies, 'latest' returns "
            "them chronologically. Lowercase, unlike search_type."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioXTweetComments(BaseTool):  # type: ignore[override]
    """Fetch replies to a tweet using the Scavio API.

    Returns replies under ``data.timeline`` using the same tweet shape as
    search. Paginate with ``data.next_cursor`` and stop when
    ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXTweetComments

            tool = ScavioXTweetComments()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"tweet_id": "1808168603721650364", "rank": "latest"})
    """

    name: str = "scavio_x_tweet_comments"
    description: str = (
        "Fetch replies to an X (Twitter) tweet. Returns reply tweets under "
        "data.timeline with text, engagement counts and author. rank=top (default) is "
        "ranked, rank=latest is chronological. Supports cursor pagination. Costs 1 "
        "credit per call. Input should be a tweet id."
    )
    args_schema: Type[BaseModel] = ScavioXTweetCommentsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXTweetCommentsAPIWrapper = Field(
        default_factory=ScavioXTweetCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXTweetCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        tweet_id: str,
        rank: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch replies to a tweet (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                tweet_id=tweet_id,
                rank=rank,
                cursor=cursor,
            )
            return self._process_response(raw, tweet_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        tweet_id: str,
        rank: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch replies to a tweet (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                tweet_id=tweet_id,
                rank=rank,
                cursor=cursor,
            )
            return self._process_response(raw, tweet_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], tweet_id: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        timeline = data.get("timeline")
        if self.max_results and timeline:
            raw["data"]["timeline"] = timeline[: self.max_results]
        if not timeline:
            raise ToolException(
                f"No replies found for X tweet '{tweet_id}'. The tweet may have no "
                "replies or restrict who can reply."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXTweetRetweeters
# ---------------------------------------------------------------------------


class ScavioXTweetRetweetersInput(BaseModel):
    """Input schema for ScavioXTweetRetweeters tool."""

    model_config = ConfigDict(extra="allow")

    tweet_id: str = Field(
        description="Numeric tweet id as a string.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioXTweetRetweeters(BaseTool):  # type: ignore[override]
    """Fetch the users who retweeted a tweet using the Scavio API.

    Returns user briefs under ``data.retweeters`` (note the response key is
    ``retweeters``): user_id, screen_name, name, description,
    followers_count, friends_count, statuses_count, media_count,
    profile_image, blue_verified, verified, location, website and
    created_at. Paginate with ``data.next_cursor``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXTweetRetweeters

            tool = ScavioXTweetRetweeters()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"tweet_id": "1808168603721650364"})
    """

    name: str = "scavio_x_tweet_retweeters"
    description: str = (
        "Fetch the X (Twitter) users who retweeted a tweet. Returns user profiles "
        "under data.retweeters with handle, name, bio and follower counts. Supports "
        "cursor pagination. Costs 1 credit per call. Input should be a tweet id."
    )
    args_schema: Type[BaseModel] = ScavioXTweetRetweetersInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXTweetRetweetersAPIWrapper = Field(
        default_factory=ScavioXTweetRetweetersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXTweetRetweetersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        tweet_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the users who retweeted a tweet (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                tweet_id=tweet_id,
                cursor=cursor,
            )
            return self._process_response(raw, tweet_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        tweet_id: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the users who retweeted a tweet (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                tweet_id=tweet_id,
                cursor=cursor,
            )
            return self._process_response(raw, tweet_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], tweet_id: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        retweeters = data.get("retweeters")
        if self.max_results and retweeters:
            raw["data"]["retweeters"] = retweeters[: self.max_results]
        if not retweeters:
            raise ToolException(
                f"No retweeters found for X tweet '{tweet_id}'. The tweet may have "
                "no retweets yet."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXUser
# ---------------------------------------------------------------------------


class ScavioXUserInput(BaseModel):
    """Input schema for ScavioXUser tool."""

    model_config = ConfigDict(extra="allow")

    screen_name: str = Field(
        description="An X handle without the leading @ (e.g. 'elonmusk').",
    )


class ScavioXUser(BaseTool):  # type: ignore[override]
    """Fetch an X (Twitter) profile by handle using the Scavio API.

    Returns a flat profile under ``data``: user_id, screen_name, name,
    description, followers_count, friends_count, statuses_count,
    media_count, blue_verified, verified_type, protected, location, website,
    avatar, header_image, created_at and pinned_tweet_ids.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXUser

            tool = ScavioXUser()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"screen_name": "elonmusk"})
    """

    name: str = "scavio_x_user"
    description: str = (
        "Fetch an X (Twitter) user profile by handle. Returns a flat profile under "
        "data with user_id, display name, bio, follower/following/tweet counts, "
        "verification, location, website and avatar. Costs 1 credit per call. Input "
        "should be a handle without the @."
    )
    args_schema: Type[BaseModel] = ScavioXUserInput
    handle_tool_error: bool = True

    api_wrapper: ScavioXUserAPIWrapper = Field(
        default_factory=ScavioXUserAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXUserAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        screen_name: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch an X (Twitter) profile by handle (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                screen_name=screen_name,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        screen_name: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch an X (Twitter) profile by handle (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                screen_name=screen_name,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], screen_name: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not data.get("user_id"):
            raise ToolException(
                f"No X profile found for '{screen_name}'. Verify the handle (without "
                "the @) and that the account is public."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXUserTweets
# ---------------------------------------------------------------------------


class ScavioXUserTweetsInput(BaseModel):
    """Input schema for ScavioXUserTweets tool."""

    model_config = ConfigDict(extra="allow")

    screen_name: str = Field(
        description="An X handle without the leading @.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioXUserTweets(BaseTool):  # type: ignore[override]
    """Fetch a user's tweets from X using the Scavio API.

    Returns the timeline under ``data.timeline`` plus ``data.pinned`` (a
    tweet object or null) and ``data.user`` (the full profile). Paginate
    with ``data.next_cursor``; this endpoint has no ``has_more`` key.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXUserTweets

            tool = ScavioXUserTweets()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"screen_name": "elonmusk"})
    """

    name: str = "scavio_x_user_tweets"
    description: str = (
        "Fetch an X (Twitter) user's tweets by handle. Returns tweets under "
        "data.timeline plus data.pinned and data.user. Supports cursor pagination. "
        "Costs 1 credit per call. Input should be a handle without the @."
    )
    args_schema: Type[BaseModel] = ScavioXUserTweetsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXUserTweetsAPIWrapper = Field(
        default_factory=ScavioXUserTweetsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXUserTweetsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a user's tweets from X (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a user's tweets from X (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], screen_name: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        timeline = data.get("timeline")
        if self.max_results and timeline:
            raw["data"]["timeline"] = timeline[: self.max_results]
        if not timeline:
            raise ToolException(
                f"No tweets found for X user '{screen_name}'. The account may be "
                "protected, suspended or have no posts."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXUserReplies
# ---------------------------------------------------------------------------


class ScavioXUserRepliesInput(BaseModel):
    """Input schema for ScavioXUserReplies tool."""

    model_config = ConfigDict(extra="allow")

    screen_name: str = Field(
        description="An X handle without the leading @.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioXUserReplies(BaseTool):  # type: ignore[override]
    """Fetch a user's tweets and replies from X using the Scavio API.

    Returns the replies timeline under ``data.timeline`` plus ``data.user``.
    There is no ``pinned`` and no ``has_more`` key on this endpoint;
    paginate with ``data.next_cursor``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXUserReplies

            tool = ScavioXUserReplies()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"screen_name": "elonmusk"})
    """

    name: str = "scavio_x_user_replies"
    description: str = (
        "Fetch an X (Twitter) user's tweets and replies by handle. Returns them under "
        "data.timeline plus data.user. Use this instead of user_tweets when "
        "conversational replies matter. Supports cursor pagination. Costs 1 credit per "
        "call. Input should be a handle without the @."
    )
    args_schema: Type[BaseModel] = ScavioXUserRepliesInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXUserRepliesAPIWrapper = Field(
        default_factory=ScavioXUserRepliesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXUserRepliesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a user's tweets and replies from X (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a user's tweets and replies from X (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], screen_name: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        timeline = data.get("timeline")
        if self.max_results and timeline:
            raw["data"]["timeline"] = timeline[: self.max_results]
        if not timeline:
            raise ToolException(
                f"No replies found for X user '{screen_name}'. The account may be "
                "protected, suspended or have no replies."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXUserMedia
# ---------------------------------------------------------------------------


class ScavioXUserMediaInput(BaseModel):
    """Input schema for ScavioXUserMedia tool."""

    model_config = ConfigDict(extra="allow")

    screen_name: str = Field(
        description="An X handle without the leading @.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioXUserMedia(BaseTool):  # type: ignore[override]
    """Fetch a user's media tweets from X using the Scavio API.

    Returns only tweets carrying photos or videos under ``data.timeline``,
    plus ``data.user``. Each tweet's ``media`` holds ``photos`` and
    ``videos`` with direct URLs. Paginate with ``data.next_cursor``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXUserMedia

            tool = ScavioXUserMedia()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"screen_name": "nasa"})
    """

    name: str = "scavio_x_user_media"
    description: str = (
        "Fetch an X (Twitter) user's media tweets by handle. Returns only tweets with "
        "photos or videos under data.timeline, each carrying direct media URLs. "
        "Supports cursor pagination. Costs 1 credit per call. Input should be a handle "
        "without the @."
    )
    args_schema: Type[BaseModel] = ScavioXUserMediaInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXUserMediaAPIWrapper = Field(
        default_factory=ScavioXUserMediaAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXUserMediaAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a user's media tweets from X (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a user's media tweets from X (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], screen_name: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        timeline = data.get("timeline")
        if self.max_results and timeline:
            raw["data"]["timeline"] = timeline[: self.max_results]
        if not timeline:
            raise ToolException(
                f"No media tweets found for X user '{screen_name}'. The account may "
                "have no photo or video posts."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXUserFollowers
# ---------------------------------------------------------------------------


class ScavioXUserFollowersInput(BaseModel):
    """Input schema for ScavioXUserFollowers tool."""

    model_config = ConfigDict(extra="allow")

    screen_name: str = Field(
        description="An X handle without the leading @.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioXUserFollowers(BaseTool):  # type: ignore[override]
    """Fetch a user's followers on X using the Scavio API.

    Returns user briefs under ``data.followers`` along with
    ``data.followers_count``. Paginate with ``data.next_cursor`` and stop
    when ``data.has_more`` is false.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXUserFollowers

            tool = ScavioXUserFollowers()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"screen_name": "elonmusk"})
    """

    name: str = "scavio_x_user_followers"
    description: str = (
        "Fetch an X (Twitter) user's followers by handle. Returns follower profiles "
        "under data.followers with handle, name, bio and counts, plus "
        "data.followers_count. Supports cursor pagination. Costs 1 credit per call. "
        "Input should be a handle without the @."
    )
    args_schema: Type[BaseModel] = ScavioXUserFollowersInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXUserFollowersAPIWrapper = Field(
        default_factory=ScavioXUserFollowersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXUserFollowersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a user's followers on X (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a user's followers on X (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], screen_name: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        followers = data.get("followers")
        if self.max_results and followers:
            raw["data"]["followers"] = followers[: self.max_results]
        if not followers:
            raise ToolException(
                f"No followers found for X user '{screen_name}'. The account may be "
                "protected or have no followers."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXUserFollowings
# ---------------------------------------------------------------------------


class ScavioXUserFollowingsInput(BaseModel):
    """Input schema for ScavioXUserFollowings tool."""

    model_config = ConfigDict(extra="allow")

    screen_name: str = Field(
        description="An X handle without the leading @.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Pagination cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioXUserFollowings(BaseTool):  # type: ignore[override]
    """Fetch the accounts a user follows on X using the Scavio API.

    Returns user briefs under ``data.following`` -- the array key is
    ``following``, not ``followings``, and there is no count field to match
    ``followers_count``. Paginate with ``data.next_cursor``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXUserFollowings

            tool = ScavioXUserFollowings()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"screen_name": "elonmusk"})
    """

    name: str = "scavio_x_user_followings"
    description: str = (
        "Fetch the accounts an X (Twitter) user follows, by handle. Returns profiles "
        "under data.following (singular key) with handle, name, bio and counts. "
        "Supports cursor pagination. Costs 1 credit per call. Input should be a handle "
        "without the @."
    )
    args_schema: Type[BaseModel] = ScavioXUserFollowingsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXUserFollowingsAPIWrapper = Field(
        default_factory=ScavioXUserFollowingsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXUserFollowingsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the accounts a user follows on X (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        screen_name: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the accounts a user follows on X (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                screen_name=screen_name,
                cursor=cursor,
            )
            return self._process_response(raw, screen_name)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], screen_name: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        following = data.get("following")
        if self.max_results and following:
            raw["data"]["following"] = following[: self.max_results]
        if not following:
            raise ToolException(
                f"No followings found for X user '{screen_name}'. The account may be "
                "protected or follow nobody."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioXTrending
# ---------------------------------------------------------------------------


class ScavioXTrendingInput(BaseModel):
    """Input schema for ScavioXTrending tool."""

    model_config = ConfigDict(extra="allow")

    country: Optional[str] = Field(
        default=None,
        description=(
            "Country NAME, not an ISO code (e.g. 'UnitedStates', 'UnitedKingdom', "
            "'Japan'). Defaults to UnitedStates."
        ),
    )


class ScavioXTrending(BaseTool):  # type: ignore[override]
    """Fetch trending topics on X for a country using the Scavio API.

    Returns trends under ``data.trends``, each with name, description and
    context. There is no cursor and no ``has_more`` on this endpoint.

    The country argument takes a country NAME such as 'UnitedStates', not an
    ISO country code.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioXTrending

            tool = ScavioXTrending()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"country": "UnitedStates"})
    """

    name: str = "scavio_x_trending"
    description: str = (
        "Fetch trending topics on X (Twitter) for a country. Returns trends under "
        "data.trends with name, description and context. The country argument is a "
        "country NAME such as UnitedStates, not an ISO code. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioXTrendingInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioXTrendingAPIWrapper = Field(
        default_factory=ScavioXTrendingAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioXTrendingAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        country: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch trending topics on X for a country (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                country=country,
            )
            return self._process_response(raw, country or "UnitedStates")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        country: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch trending topics on X for a country (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                country=country,
            )
            return self._process_response(raw, country or "UnitedStates")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], country: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        trends = data.get("trends")
        if self.max_results and trends:
            raw["data"]["trends"] = trends[: self.max_results]
        if not trends:
            raise ToolException(
                f"No X trends found for '{country}'. Verify the country name (e.g. "
                "'UnitedStates') -- it is a name, not an ISO code."
            )
        return raw
