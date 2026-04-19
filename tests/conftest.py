"""Shared fixtures for langchain-scavio tests."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_scavio import ScavioSearch
from langchain_scavio.scavio_amazon import ScavioAmazonProduct, ScavioAmazonSearch
from langchain_scavio.scavio_reddit import ScavioRedditPost, ScavioRedditSearch
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


