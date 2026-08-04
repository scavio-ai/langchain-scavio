"""Endpoint-coverage guard.

Every Scavio API endpoint this package claims to support must be reachable
through an exported tool class, and no tool may point at a retired endpoint.
These assertions are deliberately exact: adding or removing a tool without
updating this file is a test failure, not a silent drift.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from langchain_core.tools import BaseTool

import langchain_scavio
from langchain_scavio._utilities import _V2_GOOGLE_PATHS

from .conftest import MOCK_API_KEY

# Endpoints the API still serves but that this package deliberately does not
# expose as agent tools: they always answer HTTP 410 and are never billed.
RETIRED_LINKEDIN_PATHS = {
    "/api/v1/linkedin/person/contact",
    "/api/v1/linkedin/company/people",
    "/api/v1/linkedin/company/jobs",
    "/api/v1/linkedin/search/people",
    "/api/v1/linkedin/search/posts",
}

GOOGLE_V2_PATHS = {
    "/api/v2/google",
    "/api/v2/google/ai-mode",
    "/api/v2/google/maps/search",
    "/api/v2/google/maps/place",
    "/api/v2/google/maps/reviews",
    "/api/v2/google/shopping",
    "/api/v2/google/shopping/product",
    "/api/v2/google/shopping/product/stores",
    "/api/v2/google/flights",
    "/api/v2/google/hotels",
    "/api/v2/google/hotels/detail",
    "/api/v2/google/news",
    "/api/v2/google/trends",
    "/api/v2/google/trending",
}

YOUTUBE_PATHS = {
    "/api/v1/youtube/search",
    "/api/v1/youtube/shorts",
    "/api/v1/youtube/suggestions",
    "/api/v1/youtube/video",
    "/api/v1/youtube/comments",
    "/api/v1/youtube/comments/replies",
    "/api/v1/youtube/transcript",
    "/api/v1/youtube/related",
    "/api/v1/youtube/channel/search",
    "/api/v1/youtube/channel",
    "/api/v1/youtube/channel/videos",
    "/api/v1/youtube/channel/shorts",
    "/api/v1/youtube/channel/community",
    "/api/v1/youtube/channel/resolve",
    "/api/v1/youtube/streams",
}

X_PATHS = {
    "/api/v1/x/search",
    "/api/v1/x/tweet",
    "/api/v1/x/tweet/comments",
    "/api/v1/x/tweet/retweeters",
    "/api/v1/x/user",
    "/api/v1/x/user/tweets",
    "/api/v1/x/user/replies",
    "/api/v1/x/user/media",
    "/api/v1/x/user/followers",
    "/api/v1/x/user/followings",
    "/api/v1/x/trending",
}

REDDIT_PATHS = {
    "/api/v1/reddit/search",
    "/api/v1/reddit/search/suggestions",
    "/api/v1/reddit/post",
    "/api/v1/reddit/post/comments",
    "/api/v1/reddit/post/comments/replies",
    "/api/v1/reddit/subreddit",
    "/api/v1/reddit/subreddit/posts",
    "/api/v1/reddit/user",
    "/api/v1/reddit/user/posts",
    "/api/v1/reddit/user/comments",
    "/api/v1/reddit/popular",
    "/api/v1/reddit/trending",
}

LINKEDIN_PATHS = {
    "/api/v1/linkedin/person",
    "/api/v1/linkedin/person/about",
    "/api/v1/linkedin/person/posts",
    "/api/v1/linkedin/company",
    "/api/v1/linkedin/company/posts",
    "/api/v1/linkedin/search/jobs",
    "/api/v1/linkedin/job",
    "/api/v1/linkedin/post",
    "/api/v1/linkedin/post/comments",
}

# Per-family tool-class counts. YouTube is 16 classes over 15 endpoints because
# ScavioYouTubeMetadata is a kept-for-compatibility alias of ScavioYouTubeVideo.
EXPECTED_TOOL_COUNTS = {
    "ScavioGoogle": 11,
    "ScavioYouTube": 16,
    "ScavioX": 11,
    "ScavioLinkedIn": 9,
    "ScavioInstagram": 12,
    "ScavioTikTokShop": 8,
    "ScavioAmazon": 3,
    "ScavioWalmart": 2,
    "ScavioReddit": 12,
}

TOTAL_TOOL_CLASSES = 96


def _tool_classes() -> list[type[BaseTool]]:
    out = []
    for name in langchain_scavio.__all__:
        obj = getattr(langchain_scavio, name, None)
        if inspect.isclass(obj) and issubclass(obj, BaseTool):
            out.append(obj)
    return out


def _paths_for(cls: type[BaseTool]) -> set[str]:
    tool: Any = cls(scavio_api_key=MOCK_API_KEY)
    if cls is langchain_scavio.ScavioSearch:
        # One tool, three Google v2 surfaces selected by search_type.
        return {
            tool.api_wrapper._build_url(kind).split(".dev", 1)[1]
            for kind in _V2_GOOGLE_PATHS
        }
    return {tool.api_wrapper._build_url().split(".dev", 1)[1]}


ALL_TOOLS = _tool_classes()


def test_all_exports_are_tools_and_counted() -> None:
    assert len(ALL_TOOLS) == TOTAL_TOOL_CLASSES
    assert len(langchain_scavio.__all__) == TOTAL_TOOL_CLASSES + 1  # + __version__


@pytest.mark.parametrize("prefix,expected", sorted(EXPECTED_TOOL_COUNTS.items()))
def test_family_tool_counts(prefix: str, expected: int) -> None:
    if prefix == "ScavioTikTokShop":
        found = [c for c in ALL_TOOLS if c.__name__.startswith("ScavioTikTokShop")]
    elif prefix == "ScavioX":
        # Guard against ScavioXyz-style false positives.
        found = [
            c
            for c in ALL_TOOLS
            if c.__name__.startswith("ScavioX") and c.__name__[7].isupper()
        ]
    else:
        found = [c for c in ALL_TOOLS if c.__name__.startswith(prefix)]
    assert len(found) == expected, sorted(c.__name__ for c in found)


def test_tiktok_family_count() -> None:
    tiktok = [
        c
        for c in ALL_TOOLS
        if c.__name__.startswith("ScavioTikTok")
        and not c.__name__.startswith("ScavioTikTokShop")
    ]
    assert len(tiktok) == 11


@pytest.mark.parametrize(
    "prefix,expected_paths",
    [
        ("google", GOOGLE_V2_PATHS),
        ("youtube", YOUTUBE_PATHS),
        ("x", X_PATHS),
        ("linkedin", LINKEDIN_PATHS),
        ("reddit", REDDIT_PATHS),
    ],
)
def test_endpoint_paths_fully_covered(prefix: str, expected_paths: set[str]) -> None:
    covered: set[str] = set()
    for cls in ALL_TOOLS:
        covered |= _paths_for(cls)
    family = {p for p in covered if f"/{prefix}" in p}
    assert expected_paths <= family, expected_paths - family


def test_no_google_v1_anywhere() -> None:
    """Google v1 was retired on 2026-08-04 and now returns 410."""
    for cls in ALL_TOOLS:
        for path in _paths_for(cls):
            assert not path.startswith("/api/v1/google"), cls.__name__


def test_retired_linkedin_endpoints_are_not_exposed() -> None:
    for cls in ALL_TOOLS:
        for path in _paths_for(cls):
            assert path not in RETIRED_LINKEDIN_PATHS, cls.__name__


def test_deprecated_youtube_metadata_alias_targets_video() -> None:
    """/youtube/metadata is a deprecated alias; the class must hit /video."""
    assert _paths_for(langchain_scavio.ScavioYouTubeMetadata) == {
        "/api/v1/youtube/video"
    }


# Every tool now states its credit cost in its description. The Reddit pair was
# the last exemption and was closed in 3.3, so this set must stay empty.
CREDIT_STATEMENT_EXEMPT: set[str] = set()


def test_every_tool_declares_its_credit_cost() -> None:
    missing = [
        cls.__name__
        for cls in ALL_TOOLS
        if "credit" not in cls(scavio_api_key=MOCK_API_KEY).description.lower()
    ]
    assert set(missing) == CREDIT_STATEMENT_EXEMPT, missing
