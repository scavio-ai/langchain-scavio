"""Tests for all Scavio Instagram tools.

These lock the response-field names the tools read. The Instagram API returns
profile/post fields directly under ``data``; lists of posts, reels, tagged
posts, and stories under ``data.items``; followers/followings under
``data.users``; and comment replies under ``data.child_comments`` -- not the
singular wrapper keys an earlier version assumed.
"""

from __future__ import annotations

from typing import Any

import responses

from langchain_scavio.scavio_instagram import (
    ScavioInstagramCommentReplies,
    ScavioInstagramPost,
    ScavioInstagramPostComments,
    ScavioInstagramProfile,
    ScavioInstagramSearchHashtags,
    ScavioInstagramSearchUsers,
    ScavioInstagramStories,
    ScavioInstagramTaggedPosts,
    ScavioInstagramUserFollowers,
    ScavioInstagramUserFollowings,
    ScavioInstagramUserPosts,
    ScavioInstagramUserReels,
)

from .conftest import MOCK_API_KEY

BASE = "https://api.scavio.dev/api/v1/instagram"


def _envelope(data: Any) -> dict[str, Any]:
    # Instagram endpoints cost 8 credits, except user posts which costs 2.
    return {
        "data": data,
        "response_time": 300,
        "credits_used": 8,
        "credits_remaining": 992,
    }


_PROFILE = _envelope(
    {"pk": "787132", "id": "787132", "username": "natgeo", "follower_count": 269000000}
)
_ITEMS = _envelope(
    {"items": [{"pk": f"{i}", "like_count": 100 + i} for i in range(6)],
     "count": 6, "next_max_id": "abc"}
)
_STORIES = _envelope({"items": [{"pk": "1", "taken_at": 1715000000}], "count": 1})
_POST = _envelope(
    {
        "pk": "3920",
        "shortcode": "DZpQwxqimz2",
        "code": "DZpQwxqimz2",
        "like_count": 5000,
    }
)
_COMMENTS = _envelope(
    {
        "comments": [{"pk": f"{i}", "text": f"c{i}"} for i in range(6)],
        "comment_count": 6,
    }
)
_REPLIES = _envelope(
    {
        "child_comments": [{"pk": f"{i}", "text": f"r{i}"} for i in range(6)],
        "child_comment_count": 6,
    }
)
_USERS = _envelope(
    {"users": [{"pk": f"{i}", "username": f"u{i}"} for i in range(6)], "has_more": True}
)
_HASHTAGS = _envelope(
    {
        "hashtags": [
            {"id": f"{i}", "name": f"tag{i}", "media_count": 1000} for i in range(6)
        ]
    }
)


class TestInstagramReturnsData:
    """Each tool must return the payload, not a 'not found' error."""

    @responses.activate
    def test_profile(self) -> None:
        responses.add(responses.POST, f"{BASE}/profile", json=_PROFILE, status=200)
        result = ScavioInstagramProfile(scavio_api_key=MOCK_API_KEY).invoke(
            {"username": "natgeo"}
        )
        assert result["data"]["username"] == "natgeo"

    @responses.activate
    def test_user_posts(self) -> None:
        responses.add(responses.POST, f"{BASE}/user/posts", json=_ITEMS, status=200)
        result = ScavioInstagramUserPosts(
            scavio_api_key=MOCK_API_KEY, max_results=3
        ).invoke({"username": "natgeo"})
        assert len(result["data"]["items"]) == 3  # trimmed to max_results

    @responses.activate
    def test_user_reels(self) -> None:
        responses.add(responses.POST, f"{BASE}/user/reels", json=_ITEMS, status=200)
        result = ScavioInstagramUserReels(scavio_api_key=MOCK_API_KEY).invoke(
            {"username": "natgeo"}
        )
        assert result["data"]["items"]

    @responses.activate
    def test_user_tagged(self) -> None:
        responses.add(responses.POST, f"{BASE}/user/tagged", json=_ITEMS, status=200)
        result = ScavioInstagramTaggedPosts(scavio_api_key=MOCK_API_KEY).invoke(
            {"username": "natgeo"}
        )
        assert result["data"]["items"]

    @responses.activate
    def test_user_stories(self) -> None:
        responses.add(responses.POST, f"{BASE}/user/stories", json=_STORIES, status=200)
        result = ScavioInstagramStories(scavio_api_key=MOCK_API_KEY).invoke(
            {"username": "natgeo"}
        )
        assert result["data"]["items"]

    @responses.activate
    def test_post(self) -> None:
        responses.add(responses.POST, f"{BASE}/post", json=_POST, status=200)
        result = ScavioInstagramPost(scavio_api_key=MOCK_API_KEY).invoke(
            {"shortcode": "DZpQwxqimz2"}
        )
        assert result["data"]["shortcode"] == "DZpQwxqimz2"

    @responses.activate
    def test_post_comments(self) -> None:
        responses.add(
            responses.POST, f"{BASE}/post/comments", json=_COMMENTS, status=200
        )
        result = ScavioInstagramPostComments(
            scavio_api_key=MOCK_API_KEY, max_results=3
        ).invoke({"shortcode": "DZpQwxqimz2"})
        assert len(result["data"]["comments"]) == 3

    @responses.activate
    def test_comment_replies(self) -> None:
        responses.add(
            responses.POST,
            f"{BASE}/post/comments/replies",
            json=_REPLIES,
            status=200,
        )
        result = ScavioInstagramCommentReplies(
            scavio_api_key=MOCK_API_KEY, max_results=3
        ).invoke({"media_id": "3920", "comment_id": "18093"})
        assert len(result["data"]["child_comments"]) == 3

    @responses.activate
    def test_user_followers(self) -> None:
        responses.add(
            responses.POST, f"{BASE}/user/followers", json=_USERS, status=200
        )
        result = ScavioInstagramUserFollowers(scavio_api_key=MOCK_API_KEY).invoke(
            {"username": "natgeo"}
        )
        assert result["data"]["users"]

    @responses.activate
    def test_user_followings(self) -> None:
        responses.add(
            responses.POST, f"{BASE}/user/followings", json=_USERS, status=200
        )
        result = ScavioInstagramUserFollowings(scavio_api_key=MOCK_API_KEY).invoke(
            {"username": "natgeo"}
        )
        assert result["data"]["users"]

    @responses.activate
    def test_search_users(self) -> None:
        responses.add(
            responses.POST, f"{BASE}/search/users", json=_USERS, status=200
        )
        result = ScavioInstagramSearchUsers(scavio_api_key=MOCK_API_KEY).invoke(
            {"keyword": "coffee"}
        )
        assert result["data"]["users"]

    @responses.activate
    def test_search_hashtags(self) -> None:
        responses.add(
            responses.POST, f"{BASE}/search/hashtags", json=_HASHTAGS, status=200
        )
        result = ScavioInstagramSearchHashtags(scavio_api_key=MOCK_API_KEY).invoke(
            {"keyword": "coffee"}
        )
        assert result["data"]["hashtags"]


class TestInstagramEmptyRaises:
    """Genuinely empty payloads still surface a clear not-found message."""

    @responses.activate
    def test_profile_empty(self) -> None:
        responses.add(responses.POST, f"{BASE}/profile", json=_envelope({}), status=200)
        result = ScavioInstagramProfile(scavio_api_key=MOCK_API_KEY).invoke(
            {"username": "nobody"}
        )
        assert "No Instagram user found" in result

    @responses.activate
    def test_posts_empty(self) -> None:
        responses.add(
            responses.POST,
            f"{BASE}/user/posts",
            json=_envelope({"items": []}),
            status=200,
        )
        result = ScavioInstagramUserPosts(scavio_api_key=MOCK_API_KEY).invoke(
            {"username": "nobody"}
        )
        assert "No posts found" in result
