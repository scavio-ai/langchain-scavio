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
from langchain_scavio.scavio_walmart import ScavioWalmartProduct, ScavioWalmartSearch
from langchain_scavio.scavio_youtube import (
    ScavioYouTubeMetadata,
    ScavioYouTubeSearch,
)

MOCK_API_KEY = "sk_live_test_key_12345"


def make_light_response(**overrides: Any) -> dict[str, Any]:
    """Build a minimal light-mode API response."""
    base: dict[str, Any] = {
        "query": "test query",
        "page": 1,
        "country_code": "",
        "language": "",
        "response_time": 0.45,
        "credits_used": 1,
        "credits_remaining": 999,
        "results": [
            {
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "description": f"Description for result {i}",
                "position": i,
            }
            for i in range(1, 11)
        ],
    }
    base.update(overrides)
    return base


def make_full_response(**overrides: Any) -> dict[str, Any]:
    """Build a full-mode API response with all optional fields."""
    base = make_light_response(credits_used=2)
    base.update(
        {
            "total_results": 1250000,
            "search_url": "https://www.google.com/search?q=test+query",
            "knowledge_graph": {
                "title": "Test Subject",
                "subtitle": "A test entity",
                "factoids": [
                    {"title": "Founded", "content": "2020"},
                    {"title": "Headquarters", "content": "San Francisco"},
                ],
            },
            "questions": [
                {
                    "question": "What is test subject?",
                    "answer": "Test subject is an entity used for testing.",
                },
            ],
            "related_queries": [
                {"title": "test subject reviews", "position": 0,
                 "link": "test subject reviews"},
                {"title": "test subject alternatives", "position": 1,
                 "link": "test subject alternatives"},
            ],
        }
    )
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
            "url": f"/dp/B00000{i:04d}",
            "price": 9.99 + i,
            "currency": "USD",
            "rating": 4.5,
            "reviews_count": 100 + i,
            "url_image": f"https://images.amazon.com/images/P/B00000{i:04d}.jpg",
            "is_prime": True,
            "best_seller": i == 1,
            "is_sponsored": False,
        }
        for i in range(1, 11)
    ]
    base: dict[str, Any] = {
        "data": {"page": 1, "html": "", "products": products},
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
            "price": 29.99,
            "currency": "USD",
            "rating": 4.3,
            "reviews_count": 512,
            "url_image": "https://images.amazon.com/images/P/B001234567.jpg",
            "is_prime": True,
            "best_seller": False,
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


def make_youtube_metadata_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock YouTube metadata API response."""
    base: dict[str, Any] = {
        "data": {
            "title": "Test Video Title",
            "description": "A test video description.",
            "upload_date": "2026-01-15",
            "duration": 330,
            "view_count": 50000,
            "like_count": 1200,
            "comment_count": 340,
            "categories": ["Education"],
            "tags": ["python", "tutorial"],
            "channel_id": "UCtest123",
            "channel_url": "https://www.youtube.com/channel/UCtest123",
            "uploader": "Test Channel",
            "age_limit": 0,
        },
        "response_time": 0.35,
        "credits_used": 1,
        "credits_remaining": 999,
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


def make_reddit_search_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock Reddit search API response (actual structure: data.posts)."""
    posts = [
        {
            "position": i,
            "id": f"t3_{i:08d}",
            "title": f"Reddit Post {i}",
            "url": (
                f"https://www.reddit.com/r/test/comments/{i:08d}/reddit_post_{i}/"
            ),
            "subreddit": "test",
            "author": f"user{i}",
            "timestamp": "2026-04-15T16:34:40.389000+0000",
            "nsfw": False,
        }
        for i in range(0, 10)
    ]
    base: dict[str, Any] = {
        "data": {
            "searchQuery": "test query",
            "totalResults": len(posts),
            "nextCursor": "eyJjYW5kaWRhdGVzX3JldH...",
            "posts": posts,
        },
        "response_time": 5200,
        "credits_used": 2,
        "credits_remaining": 498,
    }
    base.update(overrides)
    return base


def make_reddit_post_response(**overrides: Any) -> dict[str, Any]:
    """Build a mock Reddit post detail API response."""
    base: dict[str, Any] = {
        "data": {
            "post": {
                "id": "t3_abc123",
                "title": "Example post title",
                "author": "op_user",
                "subreddit": "programming",
                "url": (
                    "https://www.reddit.com/r/programming/comments/abc123/"
                    "example_post/"
                ),
                "contentUrl": "https://external-site.com/article",
                "permalink": "/r/programming/comments/abc123/example_post/",
                "body": "post body",
                "score": 42,
                "commentCount": 87,
                "awardCount": 13,
                "timestamp": "2026-04-15T16:34:40.389000+0000",
                "nsfw": False,
                "postType": "text",
                "domain": "self.programming",
                "flair": None,
                "featuredAward": None,
                "media": [],
            },
            "comments": [
                {
                    "id": "t1_c1",
                    "author": "user1",
                    "body": "top-level reply",
                    "score": 5,
                    "depth": 0,
                    "timestamp": "2026-04-16T07:00:00.000000+0000",
                    "permalink": "/r/programming/comments/abc123/comment/c1/",
                },
                {
                    "id": "t1_c2",
                    "author": "user2",
                    "body": "nested reply",
                    "score": 2,
                    "depth": 1,
                    "timestamp": "2026-04-16T08:00:00.000000+0000",
                    "permalink": "/r/programming/comments/abc123/comment/c2/",
                },
            ],
        },
        "response_time": 5200,
        "credits_used": 2,
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


