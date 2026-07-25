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
        le=10,
        description=(
            "Maximum number of API requests per second. "
            "Default is 1 (free plan). Enterprise plans support up to 10."
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

    v1 -> v2 mapping: country_code -> gl, language -> hl,
    page (1-indexed) -> start (offset), light_request -> dropped
    (v2 always returns full results for 1 credit).

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
        for key in ("device", "nfpr"):
            if key in p:
                body[key] = p.pop(key)
    else:
        # News and Maps endpoints do not take device/nfpr/start.
        p.pop("device", None)
        p.pop("nfpr", None)
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


