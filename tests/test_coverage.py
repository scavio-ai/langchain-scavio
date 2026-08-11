"""Endpoint-coverage guard.

Every Scavio API endpoint this package claims to support must be reachable
through an exported tool class, and no tool may point at a retired endpoint.
These assertions are deliberately exact: adding or removing a tool without
updating this file is a test failure, not a silent drift. The path sets below
are the whole public API surface as of 4.0 -- 32 platforms, 187 tool classes
over 188 distinct endpoints (ScavioSearch alone covers three Google surfaces
via search_type, and ScavioYouTubeMetadata is an alias of ScavioYouTubeVideo,
so those two classes share /api/v1/youtube/video).
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

AMAZON_PATHS = {
    "/api/v1/amazon/search",
    "/api/v1/amazon/product",
    "/api/v1/amazon/offers",
}

TIKTOK_PATHS = {
    "/api/v1/tiktok/profile",
    "/api/v1/tiktok/user/posts",
    "/api/v1/tiktok/video",
    "/api/v1/tiktok/video/comments",
    "/api/v1/tiktok/video/comments/replies",
    "/api/v1/tiktok/search/videos",
    "/api/v1/tiktok/search/users",
    "/api/v1/tiktok/hashtag",
    "/api/v1/tiktok/hashtag/videos",
    "/api/v1/tiktok/user/followers",
    "/api/v1/tiktok/user/followings",
}

TIKTOK_SHOP_PATHS = {
    "/api/v1/tiktok-shop/search",
    "/api/v1/tiktok-shop/search/suggestions",
    "/api/v1/tiktok-shop/product",
    "/api/v1/tiktok-shop/product/reviews",
    "/api/v1/tiktok-shop/categories",
    "/api/v1/tiktok-shop/category/products",
    "/api/v1/tiktok-shop/shop/products",
    "/api/v1/tiktok-shop/resolve",
}

INSTAGRAM_PATHS = {
    "/api/v1/instagram/profile",
    "/api/v1/instagram/user/posts",
    "/api/v1/instagram/user/reels",
    "/api/v1/instagram/user/tagged",
    "/api/v1/instagram/user/stories",
    "/api/v1/instagram/post",
    "/api/v1/instagram/post/comments",
    "/api/v1/instagram/post/comments/replies",
    "/api/v1/instagram/search/users",
    "/api/v1/instagram/search/hashtags",
    "/api/v1/instagram/user/followers",
    "/api/v1/instagram/user/followings",
}

# --- 4.0: the 22-platform + core-extract fanout ---------------------------
# Paths are copied verbatim from the route definitions. Three of them do not
# match their method names and must never be derived from one: Meta Ad Library
# serves /api/v1/meta-ads/* under the route key metaads, Walmart's seller
# catalogue is /seller-products, Companies House is /filing-history, and
# Kuaishou comment replies are /video/sub-comments.

WALMART_PATHS = {
    "/api/v1/walmart/search",
    "/api/v1/walmart/product",
    "/api/v1/walmart/reviews",
    "/api/v1/walmart/category",
    "/api/v1/walmart/offers",
    "/api/v1/walmart/seller",
    "/api/v1/walmart/seller-products",
}

THREADS_PATHS = {
    "/api/v1/threads/profile",
    "/api/v1/threads/user/posts",
    "/api/v1/threads/user/replies",
    "/api/v1/threads/post",
    "/api/v1/threads/post/comments",
    "/api/v1/threads/search/users",
}

KUAISHOU_PATHS = {
    "/api/v1/kuaishou/profile",
    "/api/v1/kuaishou/user/posts",
    "/api/v1/kuaishou/user/live",
    "/api/v1/kuaishou/user/resolve",
    "/api/v1/kuaishou/video",
    "/api/v1/kuaishou/video/comments",
    "/api/v1/kuaishou/video/sub-comments",
    "/api/v1/kuaishou/videos/batch",
    "/api/v1/kuaishou/search",
    "/api/v1/kuaishou/search/videos",
    "/api/v1/kuaishou/search/users",
    "/api/v1/kuaishou/search/live",
    "/api/v1/kuaishou/tag/feed",
    "/api/v1/kuaishou/trending",
}

EBAY_PATHS = {
    "/api/v1/ebay/search",
    "/api/v1/ebay/product",
    "/api/v1/ebay/seller",
}

TARGET_PATHS = {
    "/api/v1/target/search",
    "/api/v1/target/category",
    "/api/v1/target/product",
    "/api/v1/target/reviews",
}

HOMEDEPOT_PATHS = {
    "/api/v1/homedepot/search",
    "/api/v1/homedepot/product",
    "/api/v1/homedepot/reviews",
}

ZILLOW_PATHS = {
    "/api/v1/zillow/search",
    "/api/v1/zillow/property",
    "/api/v1/zillow/reviews",
}

BOOKING_PATHS = {
    "/api/v1/booking/search",
    "/api/v1/booking/hotel",
    "/api/v1/booking/reviews",
}

TRIPADVISOR_PATHS = {
    "/api/v1/tripadvisor/locations",
    "/api/v1/tripadvisor/search",
    "/api/v1/tripadvisor/location",
    "/api/v1/tripadvisor/reviews",
}

INDEED_PATHS = {
    "/api/v1/indeed/search",
    "/api/v1/indeed/job",
    "/api/v1/indeed/company",
    "/api/v1/indeed/company/reviews",
}

AIRBNB_PATHS = {
    "/api/v1/airbnb/search",
    "/api/v1/airbnb/listing",
    "/api/v1/airbnb/reviews",
}

GLASSDOOR_PATHS = {
    "/api/v1/glassdoor/companies",
    "/api/v1/glassdoor/company",
    "/api/v1/glassdoor/reviews",
    "/api/v1/glassdoor/salaries",
}

YELP_PATHS = {
    "/api/v1/yelp/search",
    "/api/v1/yelp/business",
    "/api/v1/yelp/reviews",
}

APPSTORE_PATHS = {
    "/api/v1/appstore/search",
    "/api/v1/appstore/app",
    "/api/v1/appstore/reviews",
}

GOOGLEPLAY_PATHS = {
    "/api/v1/googleplay/search",
    "/api/v1/googleplay/app",
    "/api/v1/googleplay/reviews",
}

SEC_PATHS = {
    "/api/v1/sec/lookup",
    "/api/v1/sec/company",
    "/api/v1/sec/filings",
    "/api/v1/sec/concept",
    "/api/v1/sec/facts",
    "/api/v1/sec/search",
}

REDFIN_PATHS = {
    "/api/v1/redfin/search",
    "/api/v1/redfin/property",
    "/api/v1/redfin/market",
}

COMPANIESHOUSE_PATHS = {
    "/api/v1/companieshouse/search",
    "/api/v1/companieshouse/company",
    "/api/v1/companieshouse/officers",
    "/api/v1/companieshouse/filing-history",
}

G2_PATHS = {
    "/api/v1/g2/search",
    "/api/v1/g2/product",
    "/api/v1/g2/reviews",
}

CAPTERRA_PATHS = {
    "/api/v1/capterra/search",
    "/api/v1/capterra/product",
    "/api/v1/capterra/reviews",
}

GOOGLEADS_PATHS = {
    "/api/v1/googleads/search",
    "/api/v1/googleads/advertisers",
    "/api/v1/googleads/creative",
}

METAADS_PATHS = {
    "/api/v1/meta-ads/search",
    "/api/v1/meta-ads/advertiser",
    "/api/v1/meta-ads/ad",
}

# Extract is a CORE endpoint, not a platform: it reads any URL.
CORE_PATHS = {"/api/v1/extract"}

ALL_ENDPOINT_PATHS = (
    GOOGLE_V2_PATHS
    | YOUTUBE_PATHS
    | X_PATHS
    | REDDIT_PATHS
    | LINKEDIN_PATHS
    | AMAZON_PATHS
    | TIKTOK_PATHS
    | TIKTOK_SHOP_PATHS
    | INSTAGRAM_PATHS
    | WALMART_PATHS
    | THREADS_PATHS
    | KUAISHOU_PATHS
    | EBAY_PATHS
    | TARGET_PATHS
    | HOMEDEPOT_PATHS
    | ZILLOW_PATHS
    | BOOKING_PATHS
    | TRIPADVISOR_PATHS
    | INDEED_PATHS
    | AIRBNB_PATHS
    | GLASSDOOR_PATHS
    | YELP_PATHS
    | APPSTORE_PATHS
    | GOOGLEPLAY_PATHS
    | SEC_PATHS
    | REDFIN_PATHS
    | COMPANIESHOUSE_PATHS
    | G2_PATHS
    | CAPTERRA_PATHS
    | GOOGLEADS_PATHS
    | METAADS_PATHS
    | CORE_PATHS
)

# Class-name prefixes resolved LONGEST FIRST, so ScavioGooglePlaySearch is Google
# Play and not Google, and ScavioTikTokShopSearch is TikTok Shop and not TikTok.
# Every exported tool must resolve to exactly one family.
FAMILY_PREFIXES = [
    ("ScavioGooglePlay", "Google Play"),
    ("ScavioGoogleAds", "Google Ads"),
    ("ScavioGoogle", "Google"),
    ("ScavioSearch", "Google"),
    ("ScavioAmazon", "Amazon"),
    ("ScavioWalmart", "Walmart"),
    ("ScavioTarget", "Target"),
    ("ScavioEbay", "eBay"),
    ("ScavioHomeDepot", "Home Depot"),
    ("ScavioYouTube", "YouTube"),
    ("ScavioReddit", "Reddit"),
    ("ScavioTikTokShop", "TikTok Shop"),
    ("ScavioTikTok", "TikTok"),
    ("ScavioInstagram", "Instagram"),
    ("ScavioX", "X"),
    ("ScavioThreads", "Threads"),
    ("ScavioKuaishou", "Kuaishou"),
    ("ScavioLinkedIn", "LinkedIn"),
    ("ScavioIndeed", "Indeed"),
    ("ScavioGlassdoor", "Glassdoor"),
    ("ScavioZillow", "Zillow"),
    ("ScavioRedfin", "Redfin"),
    ("ScavioBooking", "Booking.com"),
    ("ScavioAirbnb", "Airbnb"),
    ("ScavioTripadvisor", "Tripadvisor"),
    ("ScavioYelp", "Yelp"),
    ("ScavioAppStore", "App Store"),
    ("ScavioSEC", "SEC EDGAR"),
    ("ScavioCompaniesHouse", "Companies House"),
    ("ScavioG2", "G2"),
    ("ScavioCapterra", "Capterra"),
    ("ScavioMetaAds", "Meta Ads"),
    ("ScavioExtract", "Extract"),
]

# YouTube is 16 classes over 15 endpoints because ScavioYouTubeMetadata is a
# kept-for-compatibility alias of ScavioYouTubeVideo.
EXPECTED_TOOL_COUNTS = {
    "Google": 12,
    "Google Play": 3,
    "Google Ads": 3,
    "Amazon": 3,
    "Walmart": 7,
    "Target": 4,
    "eBay": 3,
    "Home Depot": 3,
    "YouTube": 16,
    "Reddit": 12,
    "TikTok Shop": 8,
    "TikTok": 11,
    "Instagram": 12,
    "X": 11,
    "Threads": 6,
    "Kuaishou": 14,
    "LinkedIn": 9,
    "Indeed": 4,
    "Glassdoor": 4,
    "Zillow": 3,
    "Redfin": 3,
    "Booking.com": 3,
    "Airbnb": 3,
    "Tripadvisor": 4,
    "Yelp": 3,
    "App Store": 3,
    "SEC EDGAR": 6,
    "Companies House": 4,
    "G2": 3,
    "Capterra": 3,
    "Meta Ads": 3,
    "Extract": 1,
}

TOTAL_TOOL_CLASSES = 187
TOTAL_ENDPOINTS = 188


def _tool_classes() -> list[type[BaseTool]]:
    out = []
    for name in langchain_scavio.__all__:
        obj = getattr(langchain_scavio, name, None)
        if inspect.isclass(obj) and issubclass(obj, BaseTool):
            out.append(obj)
    return out


def _family(cls: type[BaseTool]) -> str | None:
    for prefix, family in sorted(FAMILY_PREFIXES, key=lambda p: -len(p[0])):
        if cls.__name__.startswith(prefix):
            return family
    return None


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
COVERED_PATHS = {path for cls in ALL_TOOLS for path in _paths_for(cls)}


def test_all_exports_are_tools_and_counted() -> None:
    assert len(ALL_TOOLS) == TOTAL_TOOL_CLASSES
    assert len(langchain_scavio.__all__) == TOTAL_TOOL_CLASSES + 1  # + __version__


def test_every_tool_resolves_to_exactly_one_family() -> None:
    unassigned = [cls.__name__ for cls in ALL_TOOLS if _family(cls) is None]
    assert unassigned == []


@pytest.mark.parametrize("family,expected", sorted(EXPECTED_TOOL_COUNTS.items()))
def test_family_tool_counts(family: str, expected: int) -> None:
    found = [cls for cls in ALL_TOOLS if _family(cls) == family]
    assert len(found) == expected, sorted(cls.__name__ for cls in found)


def test_family_counts_sum_to_the_package_total() -> None:
    assert sum(EXPECTED_TOOL_COUNTS.values()) == TOTAL_TOOL_CLASSES


def test_declared_endpoint_count() -> None:
    assert len(ALL_ENDPOINT_PATHS) == TOTAL_ENDPOINTS


@pytest.mark.parametrize(
    "family,expected_paths",
    [
        ("google-v2", GOOGLE_V2_PATHS),
        ("youtube", YOUTUBE_PATHS),
        ("x", X_PATHS),
        ("linkedin", LINKEDIN_PATHS),
        ("reddit", REDDIT_PATHS),
        ("amazon", AMAZON_PATHS),
        ("tiktok", TIKTOK_PATHS),
        ("tiktok-shop", TIKTOK_SHOP_PATHS),
        ("instagram", INSTAGRAM_PATHS),
        ("walmart", WALMART_PATHS),
        ("threads", THREADS_PATHS),
        ("kuaishou", KUAISHOU_PATHS),
        ("ebay", EBAY_PATHS),
        ("target", TARGET_PATHS),
        ("homedepot", HOMEDEPOT_PATHS),
        ("zillow", ZILLOW_PATHS),
        ("booking", BOOKING_PATHS),
        ("tripadvisor", TRIPADVISOR_PATHS),
        ("indeed", INDEED_PATHS),
        ("airbnb", AIRBNB_PATHS),
        ("glassdoor", GLASSDOOR_PATHS),
        ("yelp", YELP_PATHS),
        ("appstore", APPSTORE_PATHS),
        ("googleplay", GOOGLEPLAY_PATHS),
        ("sec", SEC_PATHS),
        ("redfin", REDFIN_PATHS),
        ("companieshouse", COMPANIESHOUSE_PATHS),
        ("g2", G2_PATHS),
        ("capterra", CAPTERRA_PATHS),
        ("googleads", GOOGLEADS_PATHS),
        ("meta-ads", METAADS_PATHS),
        ("core", CORE_PATHS),
    ],
)
def test_endpoint_paths_fully_covered(family: str, expected_paths: set[str]) -> None:
    assert expected_paths <= COVERED_PATHS, expected_paths - COVERED_PATHS


def test_no_endpoint_is_reachable_that_is_not_declared_here() -> None:
    """The reverse of the coverage check: nothing undeclared may slip in."""
    assert COVERED_PATHS == ALL_ENDPOINT_PATHS, {
        "undeclared": sorted(COVERED_PATHS - ALL_ENDPOINT_PATHS),
        "unreachable": sorted(ALL_ENDPOINT_PATHS - COVERED_PATHS),
    }


def test_no_google_v1_anywhere() -> None:
    """Google v1 was retired on 2026-08-04 and now returns 410.

    /api/v1/googleplay and /api/v1/googleads are different products that merely
    share the prefix, so the check is on the exact segment, not startswith.
    """
    for cls in ALL_TOOLS:
        for path in _paths_for(cls):
            assert path != "/api/v1/google", cls.__name__
            assert not path.startswith("/api/v1/google/"), cls.__name__


def test_retired_linkedin_endpoints_are_not_exposed() -> None:
    for cls in ALL_TOOLS:
        for path in _paths_for(cls):
            assert path not in RETIRED_LINKEDIN_PATHS, cls.__name__


def test_deprecated_youtube_metadata_alias_targets_video() -> None:
    """/youtube/metadata is a deprecated alias; the class must hit /video."""
    assert _paths_for(langchain_scavio.ScavioYouTubeMetadata) == {
        "/api/v1/youtube/video"
    }


def test_meta_ads_path_is_hyphenated() -> None:
    """The route key is metaads but the served path is /meta-ads/ -- never derive it."""
    for cls in (
        langchain_scavio.ScavioMetaAdsSearch,
        langchain_scavio.ScavioMetaAdsAdvertiser,
        langchain_scavio.ScavioMetaAdsAd,
    ):
        for path in _paths_for(cls):
            assert path.startswith("/api/v1/meta-ads/"), cls.__name__


def test_walmart_retired_params_are_gone() -> None:
    """device / delivery_zip / store_id are retired and answered with warnings[]."""
    retired = {"device", "delivery_zip", "store_id"}
    for cls in ALL_TOOLS:
        if not cls.__name__.startswith("ScavioWalmart"):
            continue
        tool: Any = cls(scavio_api_key=MOCK_API_KEY)
        assert retired & set(tool.args_schema.model_fields) == set(), cls.__name__


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


# Surfaces whose credit cost is a function of the request BODY. A flat "Costs N
# credits" on any of these would misprice the call for an agent.
BODY_PRICED_TOOLS = {
    "ScavioWalmartSearch": ("com.mx", "2 credits"),
    "ScavioWalmartCategory": ("com.mx", "2 credits"),
    "ScavioThreadsProfile": ("username", "4 credits"),
    "ScavioThreadsUserPosts": ("username", "4 credits"),
    "ScavioThreadsUserReplies": ("username", "4 credits"),
    "ScavioExtract": ("ultra", "2 credits"),
}


@pytest.mark.parametrize("name,markers", sorted(BODY_PRICED_TOOLS.items()))
def test_body_priced_tools_never_state_a_flat_cost(
    name: str, markers: tuple[str, str]
) -> None:
    tool: Any = getattr(langchain_scavio, name)(scavio_api_key=MOCK_API_KEY)
    for marker in markers:
        assert marker in tool.description, (name, marker, tool.description)


def test_kuaishou_is_priced_per_endpoint_not_flat() -> None:
    """Kuaishou costs 1, 2, 10 or 40 credits depending on the endpoint."""
    expected = {
        "ScavioKuaishouProfile": "10 credits",
        "ScavioKuaishouVideo": "2 credits",
        "ScavioKuaishouVideosBatch": "40 credits",
        "ScavioKuaishouSearch": "10 credits",
        "ScavioKuaishouSearchVideos": "10 credits",
        "ScavioKuaishouSearchUsers": "10 credits",
        "ScavioKuaishouSearchLive": "10 credits",
        "ScavioKuaishouUserPosts": "1 credit",
        "ScavioKuaishouUserLive": "1 credit",
        "ScavioKuaishouUserResolve": "1 credit",
        "ScavioKuaishouVideoComments": "1 credit",
        "ScavioKuaishouCommentReplies": "1 credit",
        "ScavioKuaishouTagFeed": "1 credit",
        "ScavioKuaishouTrending": "1 credit",
    }
    for name, cost in expected.items():
        tool: Any = getattr(langchain_scavio, name)(scavio_api_key=MOCK_API_KEY)
        assert cost in tool.description, (name, cost)


def test_kuaishou_is_never_called_kwai() -> None:
    """TikHub does not serve kwai.com, so no surface may advertise Kwai."""
    for cls in ALL_TOOLS:
        if not cls.__name__.startswith("ScavioKuaishou"):
            continue
        tool: Any = cls(scavio_api_key=MOCK_API_KEY)
        assert "kwai" not in tool.description.lower(), cls.__name__
