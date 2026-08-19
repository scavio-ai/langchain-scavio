"""Scavio API wrappers for raw HTTP calls."""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    model_validator,
)

logger = logging.getLogger(__name__)

SCAVIO_API_URL = "https://api.scavio.dev"


class _RateLimiter:
    """Sliding-window rate limiter for API requests."""

    def __init__(self, max_per_second: int) -> None:
        self._max = max_per_second
        self._timestamps: collections.deque[float] = collections.deque()
        self._sync_lock = threading.Lock()

    def _cleanup(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 1.0:
            self._timestamps.popleft()

    def wait(self) -> None:
        """Block until a request slot is available (sync)."""
        with self._sync_lock:
            self._cleanup()
            if len(self._timestamps) >= self._max:
                sleep_time = 1.0 - (time.monotonic() - self._timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._cleanup()
            self._timestamps.append(time.monotonic())

    async def wait_async(self) -> None:
        """Wait until a request slot is available (async)."""
        self._cleanup()
        if len(self._timestamps) >= self._max:
            sleep_time = 1.0 - (time.monotonic() - self._timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self._cleanup()
        self._timestamps.append(time.monotonic())


class ScavioBaseAPIWrapper(BaseModel):
    """Base wrapper for Scavio API endpoints.

    Provides shared auth, headers, and HTTP plumbing.
    Subclasses override ``_build_url()`` to target a specific endpoint.
    """

    scavio_api_key: SecretStr
    api_base_url: Optional[str] = None
    max_requests_per_second: int = Field(
        default=1,
        ge=1,
        le=50,
        description=(
            "Maximum number of API requests per second. "
            "Default is 1, which matches the free and pay-as-you-go plans. "
            "Paid plans allow more (up to 50 on Growth); "
            "Enterprise plans are unlimited."
        ),
    )

    _rate_limiter: _RateLimiter = PrivateAttr()

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        try:
            scavio_api_key = get_from_dict_or_env(
                values, "scavio_api_key", "SCAVIO_API_KEY"
            )
        except ValueError:
            raise ValueError(
                "No SCAVIO_API_KEY found. "
                "Get your free API key at https://dashboard.scavio.dev"
            )
        values["scavio_api_key"] = scavio_api_key
        return values

    def model_post_init(self, __context: Any) -> None:
        self._rate_limiter = _RateLimiter(self.max_requests_per_second)

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.scavio_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-scavio",
        }

    def _build_url(self) -> str:
        raise NotImplementedError("Subclasses must override _build_url()")

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute a synchronous POST request against ``url``.

        Raises:
            ValueError: If the API returns a non-200 status code.
        """
        response = requests.post(
            url,
            json=body,
            headers=self._build_headers(),
            timeout=30,
        )
        if response.status_code != 200:
            error = response.json().get("error", "Unknown error")
            if isinstance(error, dict):
                error = error.get("message", "Unknown error")
            if response.status_code == 429:
                raise ValueError(
                    f"Rate limit exceeded: {error}. "
                    "Upgrade your plan at https://dashboard.scavio.dev/billing"
                )
            raise ValueError(f"Error {response.status_code}: {error}")
        return response.json()

    async def _post_async(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute an asynchronous POST request against ``url``.

        Raises:
            ValueError: If the API returns a non-200 status code.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=self._build_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                text = await response.text()
                if response.status != 200:
                    error = json.loads(text).get("error", "Unknown error")
                    if isinstance(error, dict):
                        error = error.get("message", "Unknown error")
                    if response.status == 429:
                        raise ValueError(
                            f"Rate limit exceeded: {error}. "
                            "Upgrade your plan at "
                            "https://dashboard.scavio.dev/billing"
                        )
                    raise ValueError(
                        f"Error {response.status}: {error}"
                    )
                return json.loads(text)

    def raw_results(self, **params: Any) -> dict[str, Any]:
        """Execute a synchronous POST request.

        Args:
            **params: Request body parameters.
                None values are automatically filtered out.

        Returns:
            Parsed JSON response from the API.

        Raises:
            ValueError: If the API returns a non-200 status code.
        """
        params = {k: v for k, v in params.items() if v is not None}
        self._rate_limiter.wait()
        return self._post(self._build_url(), params)

    async def raw_results_async(self, **params: Any) -> dict[str, Any]:
        """Execute an asynchronous POST request.

        Args:
            **params: Request body parameters.
                None values are automatically filtered out.

        Returns:
            Parsed JSON response from the API.

        Raises:
            Exception: If the API returns a non-200 status code.
        """
        params = {k: v for k, v in params.items() if v is not None}
        await self._rate_limiter.wait_async()
        return await self._post_async(self._build_url(), params)


# Google v2 endpoints, routed by the v1-style search_type parameter.
# v1 (/api/v1/google) retired on 2026-07-20; v2 splits surfaces into
# dedicated endpoints and returns Google's own response structure.
_V2_GOOGLE_PATHS = {
    "classic": "/api/v2/google",
    "news": "/api/v2/google/news",
    "maps": "/api/v2/google/maps/search",
}


# Params only /api/v2/google (classic) accepts. The news and maps endpoints
# have narrower schemas -- maps takes start/ll/hl/gl/google_domain, news takes
# its drivers plus hl/gl/google_domain/so -- so these are dropped rather than
# forwarded into a request that cannot use them.
_CLASSIC_ONLY_PARAMS = (
    "device",
    "nfpr",
    "include_html",
    "location",
    "uule",
    "lr",
    "cr",
    "safe",
    "filter",
    "time_period",
    "resolve_ai_overview",
)


def _domain_from_url(url: Optional[str]) -> Optional[str]:
    """Extract the hostname from a URL, mirroring the v1 ``domain`` field."""
    if not url:
        return None
    try:
        return urlparse(url).netloc or None
    except ValueError:
        return None


def _translate_google_params(
    params: dict[str, Any],
) -> tuple[str, dict[str, Any], int]:
    """Translate v1-style Google params to a v2 search_type + request body.

    Native v2 params (gl, hl, start, google_domain, ...) are passed straight
    through. The pre-3.2 aliases are mapped first and then overwritten by any
    native value: country_code -> gl, language -> hl, page (1-indexed) -> start
    (a 0-based offset), light_request -> dropped (v2 always returns full
    results for 1 credit).

    Returns:
        Tuple of (search_type, v2 request body, requested page).
    """
    p = {k: v for k, v in params.items() if v is not None}
    search_type = p.pop("search_type", "classic") or "classic"
    if search_type in ("images", "lens"):
        logger.warning(
            "search_type %r is not available on Google v2; "
            "falling back to classic web search",
            search_type,
        )
        search_type = "classic"
    body: dict[str, Any] = {"query": p.pop("query")}
    if "country_code" in p:
        body["gl"] = p.pop("country_code")
    if "language" in p:
        body["hl"] = p.pop("language")
    page = p.pop("page", None) or 1
    p.pop("light_request", None)
    if search_type == "classic":
        if page > 1:
            body["start"] = (page - 1) * 10
    else:
        # News and Maps take none of the classic SERP filters, and news has
        # no result offset at all (maps does, in multiples of 20).
        for key in _CLASSIC_ONLY_PARAMS:
            p.pop(key, None)
        if search_type == "news":
            p.pop("start", None)
    body.update(p)
    return search_type, body, page


# v2 response keys carried over unchanged into the normalized shape.
_V2_PASSTHROUGH_KEYS = (
    "knowledge_graph",
    "top_stories",
    "local_results",
    "news_results",
    "top_ads",
    "bottom_ads",
    "response_time",
    "credits_used",
    "credits_remaining",
)


def _normalize_google_v2_response(
    raw: dict[str, Any],
    search_type: str,
    query: str,
    page: int,
    country_code: Optional[str],
    language: Optional[str],
) -> dict[str, Any]:
    """Map a Google v2 response back to the v1-normalized shape.

    Keeps downstream consumers of the pre-2.11 output working:
    organic_results -> results (position/title/url/domain/content),
    related_questions -> questions, related_searches -> related_queries,
    ai_overview -> ai_overviews.
    """
    out: dict[str, Any] = {"query": query}
    results: list[Any] = [
        {
            "position": item.get("position"),
            "title": item.get("title"),
            "url": item.get("link"),
            "domain": _domain_from_url(item.get("link")),
            "content": item.get("snippet"),
        }
        for item in raw.get("organic_results") or []
    ]
    if not results:
        # News/Maps return their primary list under a different key.
        for key in ("news_results", "places", "local_results", "results"):
            value = raw.get(key)
            if isinstance(value, list) and value:
                results = value
                break
    out["results"] = results
    if search_type == "maps":
        out["maps_results"] = results
    for key in _V2_PASSTHROUGH_KEYS:
        if key in raw:
            out[key] = raw[key]
    if "related_questions" in raw:
        out["questions"] = raw["related_questions"]
    if "related_searches" in raw:
        out["related_queries"] = raw["related_searches"]
    if "ai_overview" in raw:
        out["ai_overviews"] = raw["ai_overview"]
    out["page"] = page
    if country_code:
        out["country_code"] = country_code
    if language:
        out["language"] = language
    return out


class ScavioSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Google Search endpoint (POST /api/v2/google).

    Accepts v1-style parameters (search_type, country_code, language, page)
    and translates them to the v2 wire format, normalizing the response back
    to the v1 shape so downstream consumers are unaffected.
    """

    def _build_url(self, search_type: str = "classic") -> str:
        base = self.api_base_url or SCAVIO_API_URL
        path = _V2_GOOGLE_PATHS.get(search_type, _V2_GOOGLE_PATHS["classic"])
        return f"{base}{path}"

    def raw_results(self, **params: Any) -> dict[str, Any]:
        search_type, body, page = _translate_google_params(params)
        self._rate_limiter.wait()
        raw = self._post(self._build_url(search_type), body)
        return _normalize_google_v2_response(
            raw, search_type, body.get("query", ""), page,
            body.get("gl"), body.get("hl"),
        )

    async def raw_results_async(self, **params: Any) -> dict[str, Any]:
        search_type, body, page = _translate_google_params(params)
        await self._rate_limiter.wait_async()
        raw = await self._post_async(self._build_url(search_type), body)
        return _normalize_google_v2_response(
            raw, search_type, body.get("query", ""), page,
            body.get("gl"), body.get("hl"),
        )


class ScavioAmazonSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Amazon Search endpoint (POST /api/v1/amazon/search)."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/amazon/search"


class ScavioAmazonProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Amazon Product endpoint (POST /api/v1/amazon/product)."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/amazon/product"


class ScavioAmazonOffersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Amazon Offers endpoint (POST /api/v1/amazon/offers)."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/amazon/offers"


class ScavioWalmartSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Walmart Search endpoint (POST /api/v1/walmart/search)."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/walmart/search"


class ScavioWalmartProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Walmart Product endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/walmart/product"


class ScavioYouTubeSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio YouTube Search endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/search"


class ScavioYouTubeVideoAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio YouTube Video endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/video"


class ScavioYouTubeMetadataAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the deprecated Scavio YouTube Metadata alias.

    Metadata is a deprecated alias of the YouTube Video endpoint; both
    resolve to ``/api/v1/youtube/video``.
    """

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/video"


class ScavioYouTubeCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio YouTube Comments endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/comments"


class ScavioYouTubeTranscriptAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio YouTube Transcript endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/transcript"


class ScavioYouTubeChannelAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio YouTube Channel endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/channel"


class ScavioYouTubeChannelVideosAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio YouTube Channel Videos endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/channel/videos"


class ScavioYouTubeStreamsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio YouTube Streams endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/streams"


class ScavioRedditSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Reddit Search endpoint (POST /api/v1/reddit/search)."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/search"


class ScavioRedditPostAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Reddit Post endpoint (POST /api/v1/reddit/post)."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/post"


# The remaining 10 Reddit endpoints, added in 3.3. All 12 cost 1 credit.


class ScavioRedditSearchSuggestionsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/search/suggestions."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/search/suggestions"


class ScavioRedditPostCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/post/comments."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/post/comments"


class ScavioRedditCommentRepliesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/post/comments/replies."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/post/comments/replies"


class ScavioRedditSubredditAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/subreddit."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/subreddit"


class ScavioRedditSubredditPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/subreddit/posts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/subreddit/posts"


class ScavioRedditUserAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/user."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/user"


class ScavioRedditUserPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/user/posts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/user/posts"


class ScavioRedditUserCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/user/comments."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/user/comments"


class ScavioRedditPopularAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/popular."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/popular"


class ScavioRedditTrendingAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/reddit/trending."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/reddit/trending"


class ScavioTikTokProfileAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Profile endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/profile"


class ScavioTikTokUserPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok User Posts endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/user/posts"


class ScavioTikTokVideoAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Video endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/video"


class ScavioTikTokVideoCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Video Comments endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/video/comments"


class ScavioTikTokCommentRepliesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Comment Replies endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/video/comments/replies"


class ScavioTikTokSearchVideosAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Search Videos endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/search/videos"


class ScavioTikTokSearchUsersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Search Users endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/search/users"


class ScavioTikTokHashtagAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Hashtag endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/hashtag"


class ScavioTikTokHashtagVideosAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Hashtag Videos endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/hashtag/videos"


class ScavioTikTokUserFollowersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok User Followers endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/user/followers"


class ScavioTikTokUserFollowingsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok User Followings endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok/user/followings"


class ScavioInstagramProfileAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram Profile endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/profile"


class ScavioInstagramUserPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram User Posts endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/user/posts"


class ScavioInstagramUserReelsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram User Reels endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/user/reels"


class ScavioInstagramTaggedPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram User Tagged endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/user/tagged"


class ScavioInstagramStoriesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram User Stories endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/user/stories"


class ScavioInstagramPostAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram Post endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/post"


class ScavioInstagramPostCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram Post Comments endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/post/comments"


class ScavioInstagramCommentRepliesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram Comment Replies endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/post/comments/replies"


class ScavioInstagramSearchUsersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram Search Users endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/search/users"


class ScavioInstagramSearchHashtagsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram Search Hashtags endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/search/hashtags"


class ScavioInstagramUserFollowersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram User Followers endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/user/followers"


class ScavioInstagramUserFollowingsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Instagram User Followings endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/instagram/user/followings"



class ScavioTikTokShopSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Shop Search endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok-shop/search"


class ScavioTikTokShopSearchSuggestionsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Shop Search Suggestions endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok-shop/search/suggestions"


class ScavioTikTokShopProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Shop Product Details endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok-shop/product"


class ScavioTikTokShopProductReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Shop Product Reviews endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok-shop/product/reviews"


class ScavioTikTokShopCategoriesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Shop Categories endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok-shop/categories"


class ScavioTikTokShopCategoryProductsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Shop Category Products endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok-shop/category/products"


class ScavioTikTokShopShopProductsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Shop Shop Products endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok-shop/shop/products"


class ScavioTikTokShopResolveAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio TikTok Shop URL Resolver endpoint."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tiktok-shop/resolve"

# YouTube endpoints added in 3.2.


class ScavioYouTubeShortsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/youtube/shorts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/shorts"


class ScavioYouTubeSuggestionsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/youtube/suggestions."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/suggestions"


class ScavioYouTubeCommentRepliesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/youtube/comments/replies."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/comments/replies"


class ScavioYouTubeRelatedAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/youtube/related."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/related"


class ScavioYouTubeChannelSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/youtube/channel/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/channel/search"


class ScavioYouTubeChannelShortsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/youtube/channel/shorts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/channel/shorts"


class ScavioYouTubeChannelCommunityAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/youtube/channel/community."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/channel/community"


class ScavioYouTubeChannelResolveAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/youtube/channel/resolve."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube/channel/resolve"

# Google v2 verticals added in 3.2 (native v2 params, flat responses).


class ScavioGoogleAIModeAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/ai-mode."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/ai-mode"


class ScavioGoogleMapsPlaceAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/maps/place."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/maps/place"


class ScavioGoogleMapsReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/maps/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/maps/reviews"


class ScavioGoogleShoppingAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/shopping."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/shopping"


class ScavioGoogleShoppingProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/shopping/product."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/shopping/product"


class ScavioGoogleShoppingStoresAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/shopping/product/stores."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/shopping/product/stores"


class ScavioGoogleFlightsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/flights."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/flights"


class ScavioGoogleHotelsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/hotels."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/hotels"


class ScavioGoogleHotelsDetailAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/hotels/detail."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/hotels/detail"


class ScavioGoogleTrendsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/trends."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/trends"


class ScavioGoogleTrendingAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v2/google/trending."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v2/google/trending"

# X (Twitter) endpoints, 1 credit each.


class ScavioXSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/search"


class ScavioXTweetAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/tweet."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/tweet"


class ScavioXTweetCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/tweet/comments."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/tweet/comments"


class ScavioXTweetRetweetersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/tweet/retweeters."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/tweet/retweeters"


class ScavioXUserAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/user."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/user"


class ScavioXUserTweetsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/user/tweets."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/user/tweets"


class ScavioXUserRepliesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/user/replies."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/user/replies"


class ScavioXUserMediaAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/user/media."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/user/media"


class ScavioXUserFollowersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/user/followers."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/user/followers"


class ScavioXUserFollowingsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/user/followings."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/user/followings"


class ScavioXTrendingAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/x/trending."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/x/trending"

# LinkedIn endpoints (the 5 retired 410 routes are not wrapped).


class ScavioLinkedInPersonAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/person."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/person"


class ScavioLinkedInPersonAboutAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/person/about."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/person/about"


class ScavioLinkedInPersonPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/person/posts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/person/posts"


class ScavioLinkedInCompanyAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/company."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/company"


class ScavioLinkedInCompanyPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/company/posts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/company/posts"


class ScavioLinkedInSearchJobsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/search/jobs."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/search/jobs"


class ScavioLinkedInJobAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/job."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/job"


class ScavioLinkedInPostAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/post."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/post"


class ScavioLinkedInPostCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/linkedin/post/comments."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/linkedin/post/comments"


# --------------------------------------------------------------------------
# Endpoint wrappers for the 23-platform fanout shipped in 4.0.
# Every path below is copied verbatim from the route definition. Meta Ad
# Library serves /api/v1/meta-ads/* while its route key is metaads, and
# walmart/seller-products, companieshouse/filing-history and
# kuaishou/video/sub-comments all diverge from their method names too.
# --------------------------------------------------------------------------


# Walmart


class ScavioWalmartReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/walmart/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/walmart/reviews"


class ScavioWalmartCategoryAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/walmart/category."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/walmart/category"


class ScavioWalmartOffersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/walmart/offers."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/walmart/offers"


class ScavioWalmartSellerAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/walmart/seller."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/walmart/seller"


class ScavioWalmartSellerProductsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/walmart/seller-products."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/walmart/seller-products"


# Threads


class ScavioThreadsProfileAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/threads/profile."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/threads/profile"


class ScavioThreadsUserPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/threads/user/posts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/threads/user/posts"


class ScavioThreadsUserRepliesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/threads/user/replies."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/threads/user/replies"


class ScavioThreadsPostAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/threads/post."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/threads/post"


class ScavioThreadsPostCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/threads/post/comments."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/threads/post/comments"


class ScavioThreadsSearchUsersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/threads/search/users."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/threads/search/users"


# Kuaishou (China)


class ScavioKuaishouProfileAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/profile."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/profile"


class ScavioKuaishouUserPostsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/user/posts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/user/posts"


class ScavioKuaishouUserLiveAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/user/live."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/user/live"


class ScavioKuaishouUserResolveAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/user/resolve."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/user/resolve"


class ScavioKuaishouVideoAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/video."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/video"


class ScavioKuaishouVideoCommentsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/video/comments."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/video/comments"


class ScavioKuaishouCommentRepliesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/video/sub-comments."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/video/sub-comments"


class ScavioKuaishouVideosBatchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/videos/batch."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/videos/batch"


class ScavioKuaishouSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/search"


class ScavioKuaishouSearchVideosAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/search/videos."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/search/videos"


class ScavioKuaishouSearchUsersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/search/users."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/search/users"


class ScavioKuaishouSearchLiveAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/search/live."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/search/live"


class ScavioKuaishouTagFeedAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/tag/feed."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/tag/feed"


class ScavioKuaishouTrendingAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/kuaishou/trending."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/kuaishou/trending"


# eBay


class ScavioEbaySearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/ebay/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/ebay/search"


class ScavioEbayProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/ebay/product."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/ebay/product"


class ScavioEbaySellerAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/ebay/seller."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/ebay/seller"


# Target


class ScavioTargetSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/target/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/target/search"


class ScavioTargetCategoryAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/target/category."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/target/category"


class ScavioTargetProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/target/product."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/target/product"


class ScavioTargetReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/target/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/target/reviews"


# Home Depot


class ScavioHomeDepotSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/homedepot/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/homedepot/search"


class ScavioHomeDepotProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/homedepot/product."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/homedepot/product"


class ScavioHomeDepotReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/homedepot/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/homedepot/reviews"


# Zillow


class ScavioZillowSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/zillow/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/zillow/search"


class ScavioZillowPropertyAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/zillow/property."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/zillow/property"


class ScavioZillowAgentReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/zillow/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/zillow/reviews"


# Booking.com


class ScavioBookingSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/booking/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/booking/search"


class ScavioBookingHotelAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/booking/hotel."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/booking/hotel"


class ScavioBookingReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/booking/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/booking/reviews"


# Tripadvisor


class ScavioTripadvisorLocationsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/tripadvisor/locations."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tripadvisor/locations"


class ScavioTripadvisorSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/tripadvisor/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tripadvisor/search"


class ScavioTripadvisorLocationAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/tripadvisor/location."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tripadvisor/location"


class ScavioTripadvisorReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/tripadvisor/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/tripadvisor/reviews"


# Indeed


class ScavioIndeedSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/indeed/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/indeed/search"


class ScavioIndeedJobAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/indeed/job."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/indeed/job"


class ScavioIndeedCompanyAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/indeed/company."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/indeed/company"


class ScavioIndeedCompanyReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/indeed/company/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/indeed/company/reviews"


# Airbnb


class ScavioAirbnbSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/airbnb/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/airbnb/search"


class ScavioAirbnbListingAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/airbnb/listing."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/airbnb/listing"


class ScavioAirbnbReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/airbnb/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/airbnb/reviews"


# Glassdoor


class ScavioGlassdoorCompaniesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/glassdoor/companies."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/glassdoor/companies"


class ScavioGlassdoorCompanyAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/glassdoor/company."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/glassdoor/company"


class ScavioGlassdoorReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/glassdoor/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/glassdoor/reviews"


class ScavioGlassdoorSalariesAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/glassdoor/salaries."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/glassdoor/salaries"


# Yelp


class ScavioYelpSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/yelp/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/yelp/search"


class ScavioYelpBusinessAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/yelp/business."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/yelp/business"


class ScavioYelpReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/yelp/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/yelp/reviews"


# Apple App Store


class ScavioAppStoreSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/appstore/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/appstore/search"


class ScavioAppStoreAppAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/appstore/app."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/appstore/app"


class ScavioAppStoreReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/appstore/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/appstore/reviews"


# Google Play


class ScavioGooglePlaySearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/googleplay/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/googleplay/search"


class ScavioGooglePlayAppAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/googleplay/app."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/googleplay/app"


class ScavioGooglePlayReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/googleplay/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/googleplay/reviews"


# SEC EDGAR


class ScavioSECLookupAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/sec/lookup."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/sec/lookup"


class ScavioSECCompanyAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/sec/company."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/sec/company"


class ScavioSECFilingsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/sec/filings."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/sec/filings"


class ScavioSECConceptAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/sec/concept."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/sec/concept"


class ScavioSECFactsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/sec/facts."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/sec/facts"


class ScavioSECSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/sec/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/sec/search"


# Redfin


class ScavioRedfinSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/redfin/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/redfin/search"


class ScavioRedfinPropertyAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/redfin/property."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/redfin/property"


class ScavioRedfinMarketAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/redfin/market."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/redfin/market"


# Companies House


class ScavioCompaniesHouseSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/companieshouse/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/companieshouse/search"


class ScavioCompaniesHouseCompanyAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/companieshouse/company."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/companieshouse/company"


class ScavioCompaniesHouseOfficersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/companieshouse/officers."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/companieshouse/officers"


class ScavioCompaniesHouseFilingHistoryAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/companieshouse/filing-history."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/companieshouse/filing-history"


# G2


class ScavioG2SearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/g2/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/g2/search"


class ScavioG2ProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/g2/product."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/g2/product"


class ScavioG2ReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/g2/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/g2/reviews"


# Capterra


class ScavioCapterraSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/capterra/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/capterra/search"


class ScavioCapterraProductAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/capterra/product."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/capterra/product"


class ScavioCapterraReviewsAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/capterra/reviews."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/capterra/reviews"


# Google Ads Transparency


class ScavioGoogleAdsAdvertisersAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/googleads/advertisers."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/googleads/advertisers"


class ScavioGoogleAdsSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/googleads/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/googleads/search"


class ScavioGoogleAdsCreativeAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/googleads/creative."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/googleads/creative"


# Meta Ad Library


class ScavioMetaAdsSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/meta-ads/search."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/meta-ads/search"


class ScavioMetaAdsAdvertiserAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/meta-ads/advertiser."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/meta-ads/advertiser"


class ScavioMetaAdsAdAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/meta-ads/ad."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/meta-ads/ad"


# Extract


class ScavioExtractAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio endpoint POST /api/v1/extract."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/extract"
