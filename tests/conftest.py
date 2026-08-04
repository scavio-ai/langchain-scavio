"""Shared fixtures for langchain-scavio tests."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_scavio import ScavioSearch
from langchain_scavio.scavio_amazon import ScavioAmazonProduct, ScavioAmazonSearch
from langchain_scavio.scavio_reddit import ScavioRedditPost, ScavioRedditSearch
from langchain_scavio.scavio_tiktok import (
    ScavioTikTokCommentReplies,
    ScavioTikTokHashtag,
    ScavioTikTokHashtagVideos,
    ScavioTikTokProfile,
    ScavioTikTokSearchUsers,
    ScavioTikTokSearchVideos,
    ScavioTikTokUserFollowers,
    ScavioTikTokUserFollowings,
    ScavioTikTokUserPosts,
    ScavioTikTokVideo,
    ScavioTikTokVideoComments,
)
from langchain_scavio.scavio_tiktok_shop import (
    ScavioTikTokShopCategories,
    ScavioTikTokShopCategoryProducts,
    ScavioTikTokShopProduct,
    ScavioTikTokShopProductReviews,
    ScavioTikTokShopResolve,
    ScavioTikTokShopSearch,
    ScavioTikTokShopSearchSuggestions,
    ScavioTikTokShopShopProducts,
)
from langchain_scavio.scavio_walmart import ScavioWalmartProduct, ScavioWalmartSearch
from langchain_scavio.scavio_youtube import (
    ScavioYouTubeChannel,
    ScavioYouTubeChannelVideos,
    ScavioYouTubeComments,
    ScavioYouTubeMetadata,
    ScavioYouTubeSearch,
    ScavioYouTubeStreams,
    ScavioYouTubeTranscript,
    ScavioYouTubeVideo,
)

MOCK_API_KEY = "sk_live_test_key_12345"


def make_google_v2_response(**overrides: Any) -> dict[str, Any]:
    """Build a minimal Google v2 API response (wire shape)."""
    base: dict[str, Any] = {
        "search_information": {"total_results": 1250000},
        "organic_results": [
            {
                "position": i,
                "title": f"Result {i}",
                "link": f"https://example.com/{i}",
                "snippet": f"Description for result {i}",
            }
            for i in range(1, 11)
        ],
        "response_time": 0.45,
        "credits_used": 1,
        "credits_remaining": 999,
        "cached": False,
    }
    base.update(overrides)
    return base


def make_google_v2_full_response(**overrides: Any) -> dict[str, Any]:
    """Build a Google v2 API response with all optional blocks."""
    base = make_google_v2_response()
    base.update(
        {
            "knowledge_graph": {
                "title": "Test Subject",
                "subtitle": "A test entity",
                "factoids": [
                    {"title": "Founded", "content": "2020"},
                    {"title": "Headquarters", "content": "San Francisco"},
                ],
            },
            "related_questions": [
                {
                    "question": "What is test subject?",
                    "answer": "Test subject is an entity used for testing.",
                },
            ],
            "related_searches": [
                {"query": "test subject reviews", "position": 0},
                {"query": "test subject alternatives", "position": 1},
            ],
        }
    )
    base.update(overrides)
    return base


def make_normalized_google_response(**overrides: Any) -> dict[str, Any]:
    """Build a normalized Google response (what raw_results returns)."""
    base: dict[str, Any] = {
        "query": "test query",
        "page": 1,
        "response_time": 0.45,
        "credits_used": 1,
        "credits_remaining": 999,
        "results": [
            {
                "position": i,
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "domain": "example.com",
                "content": f"Description for result {i}",
            }
            for i in range(1, 11)
        ],
    }
    base.update(overrides)
    return base


def make_error_response(
    status: int, code: str, message: str
) -> dict[str, Any]:
    """Build an API error response body."""
    return {"error": {"code": code, "message": message}}


@pytest.fixture()
def tool() -> ScavioSearch:
    """ScavioSearch with default settings and a test API key."""
    return ScavioSearch(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def full_tool() -> ScavioSearch:
    """ScavioSearch configured for full-mode responses."""
    return ScavioSearch(
        scavio_api_key=MOCK_API_KEY,
        light_request=False,
        include_knowledge_graph=True,
        include_questions=True,
        include_related=True,
    )


def make_amazon_search_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock Amazon search API response (actual structure: data.products)."""
    products = [
        {
            "asin": f"B00000{i:04d}",
            "title": f"Product {i}",
            "url": f"https://www.amazon.com/dp/B00000{i:04d}",
            "image": f"https://images.amazon.com/images/P/B00000{i:04d}.jpg",
            "price": 9.99 + i,
            "currency": "USD",
            "rating": 4.5,
            "reviews_count": 100 + i,
            "is_sponsored": False,
            "position": i,
            "badge": "Best Seller" if i == 1 else None,
            "sales_volume": None,
            "delivery": {"is_free": True, "date": None, "fastest_date": None},
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {
            "query": "wireless headphones",
            "page": 1,
            "total_results": 50000,
            "count": len(products),
            "products": products,
        },
        "response_time": 0.5,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_amazon_product_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock Amazon product detail API response."""
    base: dict[str, Any] = {
        "data": {
            "asin": "B001234567",
            "title": "Test Product",
            "brand": "TestBrand",
            "url": "https://www.amazon.com/dp/B001234567",
            "price": 29.99,
            "list_price": 39.99,
            "currency": "USD",
            "rating": 4.3,
            "reviews_count": 512,
            "image": "https://images.amazon.com/images/P/B001234567.jpg",
            "is_prime": True,
            "has_buy_box": True,
            "availability": "In Stock",
        },
        "response_time": 0.4,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_walmart_search_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock Walmart search API response (actual structure: data.products)."""
    products = [
        {
            "id": f"100000{i}",
            "title": f"Walmart Product {i}",
            "url": f"/ip/product-{i}/100000{i}",
            "price": 7.99 + i,
            "currency": "USD",
            "rating": 4.2,
            "rating_count": 50 + i,
            "image": f"https://i5.walmartimages.com/product-{i}.jpg",
            "fulfillment": {"delivery": True, "free_shipping": True},
            "out_of_stock": False,
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {"page": 1, "html": "", "products": products},
        "response_time": 0.45,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_walmart_product_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock Walmart product detail API response."""
    base: dict[str, Any] = {
        "data": {
            "id": "987654321",
            "title": "Test Walmart Product",
            "price": 19.99,
            "currency": "USD",
            "rating": 4.1,
            "rating_count": 203,
            "images": ["https://i5.walmartimages.com/test-product.jpg"],
            "fulfillment": {"delivery": True, "free_shipping": False},
            "out_of_stock": False,
        },
        "response_time": 0.38,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_youtube_search_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube search API response (actual structure: data.results)."""
    results = [
        {
            "videoId": f"vid{i:08d}",
            "title": {"runs": [{"text": f"Test Video {i}"}]},
            "longBylineText": {"runs": [{"text": "Test Channel"}]},
            "thumbnail": {
                "thumbnails": [
                    {"url": f"https://i.ytimg.com/vi/vid{i:08d}/hqdefault.jpg"}
                ]
            },
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {"results": results},
        "response_time": 0.42,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_youtube_video_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube video API response (data is the video object)."""
    base: dict[str, Any] = {
        "data": {
            "video_id": "dQw4w9WgXcQ",
            "title": "Test Video Title",
            "author": "Test Channel",
            "channel_id": "UCtest123",
            "channel_url": "https://www.youtube.com/channel/UCtest123",
            "published_at": "2026-01-15",
            "description": "A test video description.",
            "length_seconds": 330,
            "view_count": 50000,
            "keywords": ["python", "tutorial"],
            "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
            "playability_status": "OK",
            "chapters": [],
            "captions": [
                {
                    "language_code": "en",
                    "language_name": "English",
                    "url": "https://www.youtube.com/api/timedtext?v=dQw4w9WgXcQ",
                }
            ],
        },
        "response_time": 0.35,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_youtube_metadata_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube metadata API response (alias of the video shape)."""
    return make_youtube_video_response(**overrides)


def make_youtube_comments_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube comments API response (data.comments)."""
    comments = [
        {
            "comment_id": f"UgxComment{i:04d}",
            "text": f"Great video number {i}",
            "like_count": 100 + i,
            "reply_count": i % 3,
            "published_time": "1 year ago",
            "reply_cursor": f"reply_cursor_{i}",
            "author": {
                "channel_id": f"UCcommenter{i}",
                "name": f"Commenter {i}",
                "url": f"https://www.youtube.com/channel/UCcommenter{i}",
                "avatar": f"https://yt3.ggpht.com/commenter{i}.jpg",
                "is_verified": False,
                "is_creator": i == 1,
            },
        }
        for i in range(1, 21)
    ]
    base: dict[str, Any] = {
        "data": {
            "comments": comments,
            "next_cursor": "eyJjb250aW51YXRpb24iOiJ0ZXN0In0=",
            "has_more": True,
        },
        "response_time": 0.4,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_youtube_transcript_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube transcript API response."""
    base: dict[str, Any] = {
        "data": {
            "video_id": "dQw4w9WgXcQ",
            "language_code": "en",
            "language_name": "English",
            "format": "txt",
            "content": "Never gonna give you up, never gonna let you down.",
        },
        "response_time": 0.6,
        "credits_used": 8,
        "credits_remaining": 992,
    }
    base.update(overrides)
    return base


def make_youtube_channel_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube channel API response (data is the channel object)."""
    base: dict[str, Any] = {
        "data": {
            "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",
            "title": "Test Channel",
            "description": "A test channel description.",
            "handle": "@TestChannel",
            "url": "https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw",
            "subscriber_count": 509000000,
            "video_count": 993,
            "view_count": 134561410625,
            "country": "US",
            "creation_date": "2015-05-01",
            "verified": True,
            "has_business_email": True,
            "avatar": "https://yt3.ggpht.com/avatar.jpg",
            "banner": "https://yt3.ggpht.com/banner.jpg",
            "links": [{"name": "Website", "url": "https://example.com"}],
        },
        "response_time": 0.4,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_youtube_channel_videos_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube channel videos API response (data.results)."""
    results = [
        {
            "video_id": f"chvid{i:06d}",
            "title": f"Channel Video {i}",
            "url": f"https://www.youtube.com/watch?v=chvid{i:06d}",
            "thumbnail": f"https://i.ytimg.com/vi/chvid{i:06d}/hqdefault.jpg",
            "duration_text": "10:30",
            "view_count": 79260000 + i,
            "published_time": "3 weeks ago",
            "is_live": False,
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {
            "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",
            "results": results,
            "next_cursor": "eyJjb250aW51YXRpb24iOiJjaGFubmVsIn0=",
            "has_more": True,
        },
        "response_time": 0.45,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_youtube_streams_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube streams API response (data with formats)."""
    base: dict[str, Any] = {
        "data": {
            "video_id": "dQw4w9WgXcQ",
            "title": "Test Video Title",
            "author": "Test Channel",
            "length_seconds": 330,
            "view_count": 50000,
            "is_live": False,
            "formats": [
                {
                    "itag": 22,
                    "url": "https://rr1---sn-test.googlevideo.com/videoplayback",
                    "mime_type": "video/mp4",
                    "bitrate": 1500000,
                    "width": 1280,
                    "height": 720,
                    "quality_label": "720p",
                    "fps": 30,
                    "audio_quality": "AUDIO_QUALITY_MEDIUM",
                    "audio_sample_rate": 44100,
                    "content_length": 12345678,
                    "has_signature": False,
                }
            ],
            "adaptive_formats": [],
            "available_qualities": ["720p", "360p"],
            "expires_in_seconds": 21540,
        },
        "response_time": 0.5,
        "credits_used": 3,
        "credits_remaining": 997,
    }
    base.update(overrides)
    return base



@pytest.fixture()
def amazon_search_tool() -> ScavioAmazonSearch:
    """ScavioAmazonSearch with default settings and a test API key."""
    return ScavioAmazonSearch(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def amazon_product_tool() -> ScavioAmazonProduct:
    """ScavioAmazonProduct with default settings and a test API key."""
    return ScavioAmazonProduct(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def walmart_search_tool() -> ScavioWalmartSearch:
    """ScavioWalmartSearch with default settings and a test API key."""
    return ScavioWalmartSearch(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def walmart_product_tool() -> ScavioWalmartProduct:
    """ScavioWalmartProduct with default settings and a test API key."""
    return ScavioWalmartProduct(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def youtube_search_tool() -> ScavioYouTubeSearch:
    """ScavioYouTubeSearch with default settings and a test API key."""
    return ScavioYouTubeSearch(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def youtube_metadata_tool() -> ScavioYouTubeMetadata:
    """ScavioYouTubeMetadata with default settings and a test API key."""
    return ScavioYouTubeMetadata(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def youtube_video_tool() -> ScavioYouTubeVideo:
    """ScavioYouTubeVideo with default settings and a test API key."""
    return ScavioYouTubeVideo(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def youtube_comments_tool() -> ScavioYouTubeComments:
    """ScavioYouTubeComments with default settings and a test API key."""
    return ScavioYouTubeComments(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def youtube_transcript_tool() -> ScavioYouTubeTranscript:
    """ScavioYouTubeTranscript with default settings and a test API key."""
    return ScavioYouTubeTranscript(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def youtube_channel_tool() -> ScavioYouTubeChannel:
    """ScavioYouTubeChannel with default settings and a test API key."""
    return ScavioYouTubeChannel(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def youtube_channel_videos_tool() -> ScavioYouTubeChannelVideos:
    """ScavioYouTubeChannelVideos with default settings and a test API key."""
    return ScavioYouTubeChannelVideos(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def youtube_streams_tool() -> ScavioYouTubeStreams:
    """ScavioYouTubeStreams with default settings and a test API key."""
    return ScavioYouTubeStreams(scavio_api_key=MOCK_API_KEY)


def make_reddit_search_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock Reddit search API response (actual structure: data.results)."""
    results = [
        {
            "post_id": f"t3_{i:08d}",
            "title": f"Reddit Post {i}",
            "text": f"body text {i}",
            "url": (
                f"https://www.reddit.com/r/test/comments/{i:08d}/reddit_post_{i}/"
            ),
            "subreddit": "test",
            "author": f"user{i}",
            "score": 10 + i,
            "upvote_ratio": 0.95,
            "num_comments": i,
            "created_at": "2026-04-15T16:34:40.389000+0000",
            "is_nsfw": False,
            "is_video": False,
            "thumbnail": None,
            "media": [],
        }
        for i in range(0, 10)
    ]
    base: dict[str, Any] = {
        "data": {
            "results": results,
            "next_cursor": "eyJjYW5kaWRhdGVzX3JldH...",
            "has_more": True,
        },
        "response_time": 5200,
        "credits_used": 1,
        "credits_remaining": 498,
    }
    base.update(overrides)
    return base


def make_reddit_post_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock Reddit post detail response (flat post object under data)."""
    base: dict[str, Any] = {
        "data": {
            "post_id": "t3_abc123",
            "title": "Example post title",
            "text": "post body",
            "url": (
                "https://www.reddit.com/r/programming/comments/abc123/"
                "example_post/"
            ),
            "subreddit": "programming",
            "author": "op_user",
            "score": 42,
            "upvote_ratio": 0.97,
            "num_comments": 87,
            "created_at": "2026-04-15T16:34:40.389000+0000",
            "is_nsfw": False,
            "is_video": False,
            "thumbnail": None,
            "media": [],
        },
        "response_time": 5200,
        "credits_used": 1,
        "credits_remaining": 498,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def reddit_search_tool() -> ScavioRedditSearch:
    """ScavioRedditSearch with default settings and a test API key."""
    return ScavioRedditSearch(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def reddit_post_tool() -> ScavioRedditPost:
    """ScavioRedditPost with default settings and a test API key."""
    return ScavioRedditPost(scavio_api_key=MOCK_API_KEY)


# -- TikTok mock response builders ------------------------------------------


def make_tiktok_profile_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok profile API response."""
    base: dict[str, Any] = {
        "data": {
            "user": {
                "unique_id": "testuser",
                "nickname": "Test User",
                "sec_uid": "MS4wLjABAAAAtest123",
                "uid": "1234567890",
                "signature": "Just a test bio",
                "bio_url": "https://example.com",
                "follower_count": 50000,
                "following_count": 200,
                "aweme_count": 150,
                "total_favorited": 1000000,
                "avatar_larger": {
                    "url_list": ["https://p16-sign.tiktokcdn.com/avatar.jpg"],
                },
            }
        },
        "response_time": 320,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_user_posts_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok user posts API response."""
    aweme_list = [
        {
            "aweme_id": f"712345678901234{i:04d}",
            "desc": f"Test video {i}",
            "create_time": 1715000000 + i * 3600,
            "statistics": {
                "digg_count": 100 + i,
                "comment_count": 10 + i,
                "play_count": 5000 + i * 100,
                "share_count": 5 + i,
                "collect_count": 20 + i,
            },
            "author": {"unique_id": "testuser", "nickname": "Test User"},
            "music": {"title": f"Sound {i}", "author": f"Artist {i}"},
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {
            "aweme_list": aweme_list,
            "max_cursor": "10",
            "has_more": 1,
        },
        "response_time": 450,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_video_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok video detail API response."""
    base: dict[str, Any] = {
        "data": {
            "aweme_detail": {
                "aweme_id": "7123456789012345678",
                "desc": "Test video description #test",
                "create_time": 1715000000,
                "statistics": {
                    "digg_count": 5000,
                    "comment_count": 300,
                    "play_count": 100000,
                    "share_count": 50,
                    "collect_count": 800,
                },
                "video": {
                    "play_addr": {
                        "url_list": ["https://v16.tiktokcdn.com/video.mp4"],
                    },
                    "cover": {
                        "url_list": ["https://p16-sign.tiktokcdn.com/cover.jpg"],
                    },
                    "duration": 15000,
                },
                "author": {"unique_id": "testuser", "nickname": "Test User"},
                "music": {"title": "Original Sound", "author": "testuser"},
                "cha_list": [{"cid": "123", "cha_name": "test"}],
                "text_extra": [],
            }
        },
        "response_time": 280,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_video_comments_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok video comments API response."""
    comments = [
        {
            "cid": f"710000000000000{i:04d}",
            "text": f"Comment number {i}",
            "create_time": 1715000000 + i * 60,
            "digg_count": 10 + i,
            "reply_comment_total": i % 3,
            "user": {"unique_id": f"commenter{i}", "nickname": f"Commenter {i}"},
            "is_author_digged": i % 2,
        }
        for i in range(1, 21)
    ]
    base: dict[str, Any] = {
        "data": {
            "comments": comments,
            "cursor": "20",
            "has_more": 1,
        },
        "response_time": 350,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_comment_replies_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok comment replies API response."""
    comments = [
        {
            "cid": f"720000000000000{i:04d}",
            "text": f"Reply number {i}",
            "create_time": 1715000000 + i * 60,
            "digg_count": 2 + i,
            "user": {"unique_id": f"replier{i}", "nickname": f"Replier {i}"},
        }
        for i in range(1, 6)
    ]
    base: dict[str, Any] = {
        "data": {
            "comments": comments,
            "cursor": "5",
            "has_more": 0,
        },
        "response_time": 300,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_search_videos_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok search videos API response."""
    search_item_list = [
        {
            "aweme_info": {
                "aweme_id": f"713000000000000{i:04d}",
                "desc": f"Search result video {i}",
                "create_time": 1715000000 + i * 3600,
                "statistics": {
                    "digg_count": 500 + i * 10,
                    "comment_count": 20 + i,
                    "play_count": 10000 + i * 500,
                    "share_count": 10 + i,
                    "collect_count": 50 + i,
                },
                "author": {
                    "unique_id": f"creator{i}",
                    "nickname": f"Creator {i}",
                },
            },
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {
            "search_item_list": search_item_list,
            "cursor": 10,
            "has_more": 1,
        },
        "response_time": 400,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_search_users_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok search users API response."""
    user_list = [
        {
            "user_info": {
                "uid": f"900000000{i}",
                "unique_id": f"searchuser{i}",
                "nickname": f"Search User {i}",
                "sec_uid": f"MS4wLjABAAAAsearch{i}",
                "follower_count": 1000 * i,
                "signature": f"Bio for user {i}",
            }
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {
            "user_list": user_list,
            "cursor": "10",
            "has_more": 1,
        },
        "response_time": 380,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_hashtag_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok hashtag info API response."""
    base: dict[str, Any] = {
        "data": {
            "challengeInfo": {
                "challenge": {
                    "id": "123456",
                    "title": "python",
                    "desc": "Videos about Python programming",
                },
                "stats": {
                    "videoCount": 500000,
                    "viewCount": 2000000000,
                },
            }
        },
        "response_time": 250,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_hashtag_videos_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok hashtag videos API response."""
    aweme_list = [
        {
            "aweme_id": f"714000000000000{i:04d}",
            "desc": f"Hashtag video {i} #python",
            "create_time": 1715000000 + i * 3600,
            "statistics": {
                "digg_count": 200 + i * 5,
                "comment_count": 15 + i,
                "play_count": 8000 + i * 300,
                "share_count": 8 + i,
                "collect_count": 30 + i,
            },
            "author": {"unique_id": f"htcreator{i}", "nickname": f"HT Creator {i}"},
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {
            "aweme_list": aweme_list,
            "cursor": "10",
            "has_more": 1,
        },
        "response_time": 420,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_user_followers_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok user followers API response."""
    followers = [
        {
            "unique_id": f"follower{i}",
            "nickname": f"Follower {i}",
            "sec_uid": f"MS4wLjABAAAAfollower{i}",
            "uid": f"800000000{i}",
            "follower_count": 500 * i,
            "aweme_count": 10 * i,
            "signature": f"Follower {i} bio",
            "avatar_thumb": {
                "url_list": [f"https://p16-sign.tiktokcdn.com/follower{i}.jpg"],
            },
        }
        for i in range(1, 21)
    ]
    base: dict[str, Any] = {
        "data": {
            "followers": followers,
            "has_more": True,
            "next_page_token": "page2token",
            "min_time": 1715000000,
        },
        "response_time": 500,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_user_followings_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock TikTok user followings API response."""
    followings = [
        {
            "unique_id": f"following{i}",
            "nickname": f"Following {i}",
            "sec_uid": f"MS4wLjABAAAAfollowing{i}",
            "uid": f"700000000{i}",
            "follower_count": 2000 * i,
            "aweme_count": 30 * i,
            "signature": f"Following {i} bio",
            "avatar_thumb": {
                "url_list": [f"https://p16-sign.tiktokcdn.com/following{i}.jpg"],
            },
        }
        for i in range(1, 21)
    ]
    base: dict[str, Any] = {
        "data": {
            "followings": followings,
            "has_more": True,
            "next_page_token": "page2token",
            "min_time": 1715000000,
        },
        "response_time": 480,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


# -- TikTok fixtures --------------------------------------------------------


@pytest.fixture()
def tiktok_profile_tool() -> ScavioTikTokProfile:
    return ScavioTikTokProfile(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_user_posts_tool() -> ScavioTikTokUserPosts:
    return ScavioTikTokUserPosts(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_video_tool() -> ScavioTikTokVideo:
    return ScavioTikTokVideo(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_video_comments_tool() -> ScavioTikTokVideoComments:
    return ScavioTikTokVideoComments(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_comment_replies_tool() -> ScavioTikTokCommentReplies:
    return ScavioTikTokCommentReplies(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_search_videos_tool() -> ScavioTikTokSearchVideos:
    return ScavioTikTokSearchVideos(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_search_users_tool() -> ScavioTikTokSearchUsers:
    return ScavioTikTokSearchUsers(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_hashtag_tool() -> ScavioTikTokHashtag:
    return ScavioTikTokHashtag(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_hashtag_videos_tool() -> ScavioTikTokHashtagVideos:
    return ScavioTikTokHashtagVideos(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_user_followers_tool() -> ScavioTikTokUserFollowers:
    return ScavioTikTokUserFollowers(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_user_followings_tool() -> ScavioTikTokUserFollowings:
    return ScavioTikTokUserFollowings(scavio_api_key=MOCK_API_KEY)




# -- TikTok Shop response builders ------------------------------------------
#
# Every shape below is the NORMALIZED response the backend emits, taken from
# running backend/src/lib/tikhub/tiktok-shop-normalize.ts over the recorded
# fixtures in backend/tests/fixtures/tiktok-shop/ -- not from the contract
# prose. Key names here are the key names the tools must read.


def make_tiktok_shop_envelope(data: Any, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "data": data,
        "response_time": 1.42,
        "credits_used": 1,
        "credits_remaining": 999,
    }
    base.update(overrides)
    return base


def make_tiktok_shop_card(i: int = 1) -> dict[str, Any]:
    return {
        "product_id": f"17324831328036832{i:02d}",
        "title": f"Pink Cherry Blossom Phone Case {i}",
        "url": f"https://shop.tiktok.com/us/pdp/17324831328036832{i:02d}",
        "image": "https://p19-oec-general.ttcdn-us.com/image.webp",
        "price": {
            "current": 4.88,
            "original": None,
            "currency": "USD",
            "discount_percent": None,
            "savings": None,
            "min": 4.88,
            "max": 7.88,
        },
        "rating": {"score": 4.7, "review_count": 15},
        "sold_count": 103,
        "variant_count": 25,
        "brand": None,
        "shop": {
            "shop_id": "7494676034572093351",
            "shop_name": "AmiShell",
            "shop_logo": "https://p16-oec-general.ttcdn-us.com/logo.png",
        },
        "labels": ["Free shipping"],
    }


def make_tiktok_shop_review(i: int = 1) -> dict[str, Any]:
    return {
        "review_id": f"763255059759346663{i}",
        "rating": 5,
        "text": f"Review body {i}",
        "created_at": "2026-04-25T04:34:57.611Z",
        "reviewer_name": "C**",
        "reviewer_avatar": "https://p16-common-sign.tiktokcdn-us.com/avatar.jpg",
        "images": ["https://p16-oec-general-useast5.ttcdn-us.com/r.webp"],
        "is_verified_purchase": True,
        "is_incentivized": False,
        "variant": "Default",
        "country": "US",
    }


def make_tiktok_shop_search_response(
    num_products: int = 30, **overrides: Any
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "query": "phone case",
        "products": [make_tiktok_shop_card(i) for i in range(1, num_products + 1)],
        "shops": [
            {
                "shop_id": "7494676034572093351",
                "shop_name": "MAGIC JOHN",
                "shop_logo": "https://p16-oec-general.ttcdn-us.com/logo.png",
            }
        ],
        "next_cursor": "eyJrIjoic2VhcmNoIiwibyI6MzB9",
        "has_more": True,
        "degraded": False,
    }
    data.update(overrides)
    return make_tiktok_shop_envelope(data)


def make_tiktok_shop_suggestions_response(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "query": "wireless",
        "region": "US",
        "suggestions": [
            "wireless charger",
            "wireless apple carplay",
            "wireless headphones",
        ],
    }
    data.update(overrides)
    return make_tiktok_shop_envelope(data)


def make_tiktok_shop_product_response(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "product_id": "1732293553906094315",
        "title": "[medicube] NAD+ EGF Firming Serum",
        "description": "Plain text description.",
        "url": "https://shop.tiktok.com/us/pdp/1732293553906094315",
        "images": ["https://p19-oec-general.ttcdn-us.com/img.webp"],
        # Upstream masks every price on the product page: current and original
        # are always null here. Exact prices come from the listing endpoints.
        "price": {
            "currency": "USD",
            "current": None,
            "original": None,
            "discount_percent": 31,
            "savings": None,
        },
        "rating": {
            "score": 4.7,
            "review_count": 12561,
            "distribution": {"1": 431, "2": 221, "3": 571, "4": 1175, "5": 10163},
        },
        "sold_count": 231615,
        "variants": [
            {
                "sku_id": "1732293560756310251",
                "name": "PMEUS43022R00",
                "in_stock": True,
                "available_quantity": 64657,
                "properties": [{"name": "Specifications", "value": "Default"}],
                "weight_kg": 0.1,
                "dimensions_cm": {"length": 7.0, "width": 3.0, "height": 3.0},
            }
        ],
        "shipping": {
            "fee": 4.22,
            "currency": "USD",
            "delivery_min_days": 6,
            "delivery_max_days": 9,
            "delivery_min_business_days": 4,
            "delivery_max_business_days": 7,
            "cod_available": False,
            "fulfillable": True,
        },
        "shop": {
            "shop_id": "7495514739648989419",
            "shop_name": "medicube US Store",
            "shop_logo": "https://p16-oec-general.ttcdn-us.com/logo.png",
            "shop_url": "https://shop.tiktok.com/us/store/7495514739648989419",
            "rating": 4.6,
            "review_count": 455798,
            "sold_count": 7938115,
            "followers_count": 588860,
            "product_count": 153,
            "video_count": 1006,
            "region": "US",
            "is_official": True,
        },
        "categories": [
            {
                "category_id": "601450",
                "name": "Beauty & Personal Care",
                "slug": "beauty-personal-care",
            }
        ],
        "breadcrumbs": [
            {"name": "Skincare", "url": "https://shop.tiktok.com/us/c/skincare/848776"}
        ],
        "top_reviews": [make_tiktok_shop_review(1)],
        "seller": {
            "business_name": "APR US INC",
            "business_address": "41 GREENFIELD, Irvine, California, United States",
        },
    }
    data.update(overrides)
    return make_tiktok_shop_envelope(data)


def make_tiktok_shop_reviews_response(
    num_reviews: int = 20, **overrides: Any
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "product_id": "1732293553906094315",
        "page": 1,
        "page_size": 20,
        "filters_applied": {
            "sort": "relevant",
            "rating": None,
            "has_media": False,
            "verified_only": False,
        },
        "total_reviews": 12561,
        "rating": {
            "score": 4.6,
            "review_count": 12561,
            "distribution": {"1": 431, "2": 221, "3": 571, "4": 1175, "5": 10163},
        },
        "reviews": [make_tiktok_shop_review(i) for i in range(1, num_reviews + 1)],
        "has_more": True,
    }
    data.update(overrides)
    return make_tiktok_shop_envelope(data)


def make_tiktok_shop_categories_response(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "categories": [
            {
                "category_id": "601450",
                "name": "Beauty & Personal Care",
                "slug": "beauty-personal-care",
                "level": 1,
                "parent_id": None,
                "image": "https://lf16-tiktok-common.tiktokcdn-us.com/beauty.png",
                "children": [
                    {
                        "category_id": "849032",
                        "name": "Hand & Foot Care",
                        "slug": "hand-foot-care",
                        "level": 2,
                        "parent_id": "601450",
                        "image": "https://lf16-tiktok-common.tiktokcdn-us.com/h.png",
                        "children": [],
                    }
                ],
            }
        ],
        "total_categories": 240,
    }
    data.update(overrides)
    return make_tiktok_shop_envelope(data)


def make_tiktok_shop_category_products_response(
    num_products: int = 15, **overrides: Any
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "category_id": "601450",
        "products": [make_tiktok_shop_card(i) for i in range(1, num_products + 1)],
        "next_cursor": "eyJrIjoiY2F0ZWdvcnkiLCJvIjoyMH0",
        "has_more": True,
    }
    data.update(overrides)
    return make_tiktok_shop_envelope(data)


def make_tiktok_shop_shop_products_response(
    num_products: int = 30, **overrides: Any
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "shop_id": "7495514739648989419",
        "shop": {
            "shop_id": "7495514739648989419",
            "shop_name": "medicube US Store",
            "shop_logo": "https://p16-oec-general.ttcdn-us.com/logo.png",
        },
        "products": [make_tiktok_shop_card(i) for i in range(1, num_products + 1)],
        "next_cursor": "eyJrIjoic2hvcCIsInMiOiIzMF9XemN5In0",
        "has_more": True,
    }
    data.update(overrides)
    return make_tiktok_shop_envelope(data)


def make_tiktok_shop_resolve_response(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "product",
        "product_id": "8651224669119091502",
        "shop_id": None,
        "url": "https://shop.tiktok.com/us/pdp/8651224669119091502",
        "resolved_by": "share_link",
    }
    data.update(overrides)
    return make_tiktok_shop_envelope(data)


def make_tiktok_shop_not_found_response(error: str) -> dict[str, Any]:
    """The billed 404 body: an error plus the credit fields, no data key."""
    return {"error": error, "credits_used": 1, "credits_remaining": 999}


# -- TikTok Shop fixtures ---------------------------------------------------


@pytest.fixture()
def tiktok_shop_search_tool() -> ScavioTikTokShopSearch:
    return ScavioTikTokShopSearch(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_shop_suggestions_tool() -> ScavioTikTokShopSearchSuggestions:
    return ScavioTikTokShopSearchSuggestions(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_shop_product_tool() -> ScavioTikTokShopProduct:
    return ScavioTikTokShopProduct(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_shop_reviews_tool() -> ScavioTikTokShopProductReviews:
    return ScavioTikTokShopProductReviews(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_shop_categories_tool() -> ScavioTikTokShopCategories:
    return ScavioTikTokShopCategories(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_shop_category_products_tool() -> ScavioTikTokShopCategoryProducts:
    return ScavioTikTokShopCategoryProducts(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_shop_shop_products_tool() -> ScavioTikTokShopShopProducts:
    return ScavioTikTokShopShopProducts(scavio_api_key=MOCK_API_KEY)


@pytest.fixture()
def tiktok_shop_resolve_tool() -> ScavioTikTokShopResolve:
    return ScavioTikTokShopResolve(scavio_api_key=MOCK_API_KEY)
