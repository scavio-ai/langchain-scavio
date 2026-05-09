"""Tests for all Scavio TikTok tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio._utilities import SCAVIO_API_URL
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

from .conftest import (
    MOCK_API_KEY,
    make_error_response,
    make_tiktok_comment_replies_response,
    make_tiktok_hashtag_response,
    make_tiktok_hashtag_videos_response,
    make_tiktok_profile_response,
    make_tiktok_search_users_response,
    make_tiktok_search_videos_response,
    make_tiktok_user_followers_response,
    make_tiktok_user_followings_response,
    make_tiktok_user_posts_response,
    make_tiktok_video_comments_response,
    make_tiktok_video_response,
)

PROFILE_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/profile"
USER_POSTS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/user/posts"
VIDEO_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/video"
VIDEO_COMMENTS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/video/comments"
COMMENT_REPLIES_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/video/comments/replies"
SEARCH_VIDEOS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/search/videos"
SEARCH_USERS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/search/users"
HASHTAG_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/hashtag"
HASHTAG_VIDEOS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/hashtag/videos"
USER_FOLLOWERS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/user/followers"
USER_FOLLOWINGS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok/user/followings"


# ===========================================================================
# Profile
# ===========================================================================


class TestTikTokProfileInstantiation:
    def test_default_params(
        self, tiktok_profile_tool: ScavioTikTokProfile
    ) -> None:
        assert tiktok_profile_tool.name == "scavio_tiktok_profile"
        assert tiktok_profile_tool.handle_tool_error is True

    def test_api_key_forwarded(self) -> None:
        tool = ScavioTikTokProfile(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioTikTokProfile(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioTikTokProfile()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestTikTokProfileRun:
    @responses.activate
    def test_successful_lookup(
        self, tiktok_profile_tool: ScavioTikTokProfile
    ) -> None:
        responses.add(
            responses.POST, PROFILE_ENDPOINT,
            json=make_tiktok_profile_response(), status=200,
        )
        result = tiktok_profile_tool.invoke({"username": "testuser"})
        assert "data" in result
        assert result["data"]["user"]["unique_id"] == "testuser"

    @responses.activate
    def test_empty_user_raises_tool_exception(
        self, tiktok_profile_tool: ScavioTikTokProfile
    ) -> None:
        responses.add(
            responses.POST, PROFILE_ENDPOINT,
            json=make_tiktok_profile_response(data={"user": None}), status=200,
        )
        result = tiktok_profile_tool.invoke({"username": "nobody"})
        assert "No TikTok user found" in result

    @responses.activate
    def test_api_error(
        self, tiktok_profile_tool: ScavioTikTokProfile
    ) -> None:
        responses.add(
            responses.POST, PROFILE_ENDPOINT,
            json=make_error_response(401, "unauthorized", "Invalid API key"),
            status=401,
        )
        result = tiktok_profile_tool.invoke({"username": "test"})
        assert "error" in str(result).lower()

    @responses.activate
    def test_username_forwarded(
        self, tiktok_profile_tool: ScavioTikTokProfile
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST, PROFILE_ENDPOINT,
            json=make_tiktok_profile_response(), status=200,
        )
        tiktok_profile_tool.invoke({"username": "testuser"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["username"] == "testuser"


class TestTikTokProfileAsync:
    @pytest.mark.asyncio
    async def test_async_lookup(
        self, tiktok_profile_tool: ScavioTikTokProfile
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokProfileAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_profile_response(),
        ):
            result = await tiktok_profile_tool.ainvoke({"username": "testuser"})
            assert "data" in result

    @pytest.mark.asyncio
    async def test_async_empty_user(
        self, tiktok_profile_tool: ScavioTikTokProfile
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokProfileAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_profile_response(data={"user": None}),
        ):
            result = await tiktok_profile_tool.ainvoke({"username": "nobody"})
            assert "No TikTok user found" in result


class TestTikTokProfileInputSchema:
    def test_schema_fields(self) -> None:
        tool = ScavioTikTokProfile(scavio_api_key=MOCK_API_KEY)
        props = tool.get_input_schema().model_json_schema()["properties"]
        assert "username" in props
        assert "sec_user_id" in props


# ===========================================================================
# User Posts
# ===========================================================================


class TestTikTokUserPostsInstantiation:
    def test_default_params(
        self, tiktok_user_posts_tool: ScavioTikTokUserPosts
    ) -> None:
        assert tiktok_user_posts_tool.name == "scavio_tiktok_user_posts"
        assert tiktok_user_posts_tool.max_results == 5

    def test_custom_max_results(self) -> None:
        tool = ScavioTikTokUserPosts(scavio_api_key=MOCK_API_KEY, max_results=3)
        assert tool.max_results == 3


class TestTikTokUserPostsRun:
    @responses.activate
    def test_successful_fetch(
        self, tiktok_user_posts_tool: ScavioTikTokUserPosts
    ) -> None:
        responses.add(
            responses.POST, USER_POSTS_ENDPOINT,
            json=make_tiktok_user_posts_response(), status=200,
        )
        result = tiktok_user_posts_tool.invoke(
            {"sec_user_id": "MS4wLjABAAAAtest123"}
        )
        assert "data" in result
        assert len(result["data"]["aweme_list"]) == 5

    @responses.activate
    def test_max_results_truncation(self) -> None:
        tool = ScavioTikTokUserPosts(scavio_api_key=MOCK_API_KEY, max_results=3)
        responses.add(
            responses.POST, USER_POSTS_ENDPOINT,
            json=make_tiktok_user_posts_response(), status=200,
        )
        result = tool.invoke({"sec_user_id": "MS4wLjABAAAAtest123"})
        assert len(result["data"]["aweme_list"]) == 3

    @responses.activate
    def test_empty_posts_raises_tool_exception(
        self, tiktok_user_posts_tool: ScavioTikTokUserPosts
    ) -> None:
        responses.add(
            responses.POST, USER_POSTS_ENDPOINT,
            json=make_tiktok_user_posts_response(
                data={"aweme_list": [], "max_cursor": "0", "has_more": 0}
            ),
            status=200,
        )
        result = tiktok_user_posts_tool.invoke(
            {"sec_user_id": "MS4wLjABAAAAtest123"}
        )
        assert "No posts found" in result

    @responses.activate
    def test_sort_type_forwarded(
        self, tiktok_user_posts_tool: ScavioTikTokUserPosts
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST, USER_POSTS_ENDPOINT,
            json=make_tiktok_user_posts_response(), status=200,
        )
        tiktok_user_posts_tool.invoke(
            {"sec_user_id": "MS4wLjABAAAAtest123", "sort_type": "1"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["sort_type"] == "1"


class TestTikTokUserPostsForbiddenParams:
    def test_max_results_rejected_at_invocation(
        self, tiktok_user_posts_tool: ScavioTikTokUserPosts
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            tiktok_user_posts_tool._run(
                sec_user_id="MS4wLjABAAAAtest123", max_results=10
            )


class TestTikTokUserPostsAsync:
    @pytest.mark.asyncio
    async def test_async_fetch(
        self, tiktok_user_posts_tool: ScavioTikTokUserPosts
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokUserPostsAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_user_posts_response(),
        ):
            result = await tiktok_user_posts_tool.ainvoke(
                {"sec_user_id": "MS4wLjABAAAAtest123"}
            )
            assert "data" in result
            assert len(result["data"]["aweme_list"]) == 5


# ===========================================================================
# Video Detail
# ===========================================================================


class TestTikTokVideoInstantiation:
    def test_default_params(
        self, tiktok_video_tool: ScavioTikTokVideo
    ) -> None:
        assert tiktok_video_tool.name == "scavio_tiktok_video"


class TestTikTokVideoRun:
    @responses.activate
    def test_successful_lookup(
        self, tiktok_video_tool: ScavioTikTokVideo
    ) -> None:
        responses.add(
            responses.POST, VIDEO_ENDPOINT,
            json=make_tiktok_video_response(), status=200,
        )
        result = tiktok_video_tool.invoke(
            {"video_id": "7123456789012345678"}
        )
        assert "data" in result
        assert result["data"]["aweme_detail"]["aweme_id"] == "7123456789012345678"

    @responses.activate
    def test_empty_video_raises_tool_exception(
        self, tiktok_video_tool: ScavioTikTokVideo
    ) -> None:
        responses.add(
            responses.POST, VIDEO_ENDPOINT,
            json=make_tiktok_video_response(data={"aweme_detail": None}),
            status=200,
        )
        result = tiktok_video_tool.invoke({"video_id": "0000"})
        assert "No TikTok video found" in result

    @responses.activate
    def test_video_id_forwarded(
        self, tiktok_video_tool: ScavioTikTokVideo
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST, VIDEO_ENDPOINT,
            json=make_tiktok_video_response(), status=200,
        )
        tiktok_video_tool.invoke({"video_id": "7123456789012345678"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["video_id"] == "7123456789012345678"


class TestTikTokVideoAsync:
    @pytest.mark.asyncio
    async def test_async_lookup(
        self, tiktok_video_tool: ScavioTikTokVideo
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokVideoAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_video_response(),
        ):
            result = await tiktok_video_tool.ainvoke(
                {"video_id": "7123456789012345678"}
            )
            assert "data" in result


# ===========================================================================
# Video Comments
# ===========================================================================


class TestTikTokVideoCommentsInstantiation:
    def test_default_params(
        self, tiktok_video_comments_tool: ScavioTikTokVideoComments
    ) -> None:
        assert tiktok_video_comments_tool.name == "scavio_tiktok_video_comments"
        assert tiktok_video_comments_tool.max_results == 10


class TestTikTokVideoCommentsRun:
    @responses.activate
    def test_successful_fetch(
        self, tiktok_video_comments_tool: ScavioTikTokVideoComments
    ) -> None:
        responses.add(
            responses.POST, VIDEO_COMMENTS_ENDPOINT,
            json=make_tiktok_video_comments_response(), status=200,
        )
        result = tiktok_video_comments_tool.invoke(
            {"video_id": "7123456789012345678"}
        )
        assert "data" in result
        assert len(result["data"]["comments"]) == 10

    @responses.activate
    def test_empty_comments_raises_tool_exception(
        self, tiktok_video_comments_tool: ScavioTikTokVideoComments
    ) -> None:
        responses.add(
            responses.POST, VIDEO_COMMENTS_ENDPOINT,
            json=make_tiktok_video_comments_response(
                data={"comments": [], "cursor": "0", "has_more": 0}
            ),
            status=200,
        )
        result = tiktok_video_comments_tool.invoke(
            {"video_id": "7123456789012345678"}
        )
        assert "No comments found" in result


class TestTikTokVideoCommentsForbiddenParams:
    def test_max_results_rejected_at_invocation(
        self, tiktok_video_comments_tool: ScavioTikTokVideoComments
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            tiktok_video_comments_tool._run(
                video_id="7123456789012345678", max_results=5
            )


class TestTikTokVideoCommentsAsync:
    @pytest.mark.asyncio
    async def test_async_fetch(
        self, tiktok_video_comments_tool: ScavioTikTokVideoComments
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokVideoCommentsAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_video_comments_response(),
        ):
            result = await tiktok_video_comments_tool.ainvoke(
                {"video_id": "7123456789012345678"}
            )
            assert "data" in result
            assert len(result["data"]["comments"]) == 10


# ===========================================================================
# Comment Replies
# ===========================================================================


class TestTikTokCommentRepliesInstantiation:
    def test_default_params(
        self, tiktok_comment_replies_tool: ScavioTikTokCommentReplies
    ) -> None:
        assert tiktok_comment_replies_tool.name == "scavio_tiktok_comment_replies"
        assert tiktok_comment_replies_tool.max_results == 10


class TestTikTokCommentRepliesRun:
    @responses.activate
    def test_successful_fetch(
        self, tiktok_comment_replies_tool: ScavioTikTokCommentReplies
    ) -> None:
        responses.add(
            responses.POST, COMMENT_REPLIES_ENDPOINT,
            json=make_tiktok_comment_replies_response(), status=200,
        )
        result = tiktok_comment_replies_tool.invoke(
            {"video_id": "7123456789012345678", "comment_id": "710000000000000001"}
        )
        assert "data" in result
        assert len(result["data"]["comments"]) == 5

    @responses.activate
    def test_empty_replies_raises_tool_exception(
        self, tiktok_comment_replies_tool: ScavioTikTokCommentReplies
    ) -> None:
        responses.add(
            responses.POST, COMMENT_REPLIES_ENDPOINT,
            json=make_tiktok_comment_replies_response(
                data={"comments": [], "cursor": "0", "has_more": 0}
            ),
            status=200,
        )
        result = tiktok_comment_replies_tool.invoke(
            {"video_id": "7123456789012345678", "comment_id": "710000000000000001"}
        )
        assert "No replies found" in result

    @responses.activate
    def test_params_forwarded(
        self, tiktok_comment_replies_tool: ScavioTikTokCommentReplies
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST, COMMENT_REPLIES_ENDPOINT,
            json=make_tiktok_comment_replies_response(), status=200,
        )
        tiktok_comment_replies_tool.invoke({
            "video_id": "vid1",
            "comment_id": "cid1",
            "cursor": "5",
        })
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["video_id"] == "vid1"
        assert body["comment_id"] == "cid1"
        assert body["cursor"] == "5"


class TestTikTokCommentRepliesAsync:
    @pytest.mark.asyncio
    async def test_async_fetch(
        self, tiktok_comment_replies_tool: ScavioTikTokCommentReplies
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokCommentRepliesAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_comment_replies_response(),
        ):
            result = await tiktok_comment_replies_tool.ainvoke(
                {"video_id": "vid1", "comment_id": "cid1"}
            )
            assert "data" in result


# ===========================================================================
# Search Videos
# ===========================================================================


class TestTikTokSearchVideosInstantiation:
    def test_default_params(
        self, tiktok_search_videos_tool: ScavioTikTokSearchVideos
    ) -> None:
        assert tiktok_search_videos_tool.name == "scavio_tiktok_search_videos"
        assert tiktok_search_videos_tool.max_results == 5


class TestTikTokSearchVideosRun:
    @responses.activate
    def test_successful_search(
        self, tiktok_search_videos_tool: ScavioTikTokSearchVideos
    ) -> None:
        responses.add(
            responses.POST, SEARCH_VIDEOS_ENDPOINT,
            json=make_tiktok_search_videos_response(), status=200,
        )
        result = tiktok_search_videos_tool.invoke({"keyword": "python"})
        assert "data" in result
        assert len(result["data"]["search_item_list"]) == 5

    @responses.activate
    def test_max_results_truncation(self) -> None:
        tool = ScavioTikTokSearchVideos(scavio_api_key=MOCK_API_KEY, max_results=3)
        responses.add(
            responses.POST, SEARCH_VIDEOS_ENDPOINT,
            json=make_tiktok_search_videos_response(), status=200,
        )
        result = tool.invoke({"keyword": "test"})
        assert len(result["data"]["search_item_list"]) == 3

    @responses.activate
    def test_empty_results_raises_tool_exception(
        self, tiktok_search_videos_tool: ScavioTikTokSearchVideos
    ) -> None:
        responses.add(
            responses.POST, SEARCH_VIDEOS_ENDPOINT,
            json=make_tiktok_search_videos_response(
                data={
                    "search_item_list": [],
                    "cursor": 0,
                    "has_more": 0,
                }
            ),
            status=200,
        )
        result = tiktok_search_videos_tool.invoke({"keyword": "xyzzy"})
        assert "No TikTok videos found" in result

    @responses.activate
    def test_filters_forwarded(
        self, tiktok_search_videos_tool: ScavioTikTokSearchVideos
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST, SEARCH_VIDEOS_ENDPOINT,
            json=make_tiktok_search_videos_response(), status=200,
        )
        tiktok_search_videos_tool.invoke({
            "keyword": "python",
            "sort_type": "1",
            "publish_time": "7",
        })
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["keyword"] == "python"
        assert body["sort_type"] == "1"
        assert body["publish_time"] == "7"

    @responses.activate
    def test_none_fields_not_sent(
        self, tiktok_search_videos_tool: ScavioTikTokSearchVideos
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST, SEARCH_VIDEOS_ENDPOINT,
            json=make_tiktok_search_videos_response(), status=200,
        )
        tiktok_search_videos_tool.invoke({"keyword": "python"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert "sort_type" not in body
        assert "publish_time" not in body
        assert "cursor" not in body


class TestTikTokSearchVideosForbiddenParams:
    def test_max_results_rejected_at_invocation(
        self, tiktok_search_videos_tool: ScavioTikTokSearchVideos
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            tiktok_search_videos_tool._run(keyword="test", max_results=10)


class TestTikTokSearchVideosAsync:
    @pytest.mark.asyncio
    async def test_async_search(
        self, tiktok_search_videos_tool: ScavioTikTokSearchVideos
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokSearchVideosAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_search_videos_response(),
        ):
            result = await tiktok_search_videos_tool.ainvoke(
                {"keyword": "python"}
            )
            assert "data" in result
            assert len(result["data"]["search_item_list"]) == 5


class TestTikTokSearchVideosInputSchema:
    def test_schema_fields(self) -> None:
        tool = ScavioTikTokSearchVideos(scavio_api_key=MOCK_API_KEY)
        props = tool.get_input_schema().model_json_schema()["properties"]
        assert "keyword" in props
        assert "sort_type" in props
        assert "publish_time" in props
        assert "cursor" in props
        assert "count" in props

    def test_keyword_is_required(self) -> None:
        tool = ScavioTikTokSearchVideos(scavio_api_key=MOCK_API_KEY)
        schema = tool.get_input_schema().model_json_schema()
        assert "keyword" in schema.get("required", [])


# ===========================================================================
# Search Users
# ===========================================================================


class TestTikTokSearchUsersInstantiation:
    def test_default_params(
        self, tiktok_search_users_tool: ScavioTikTokSearchUsers
    ) -> None:
        assert tiktok_search_users_tool.name == "scavio_tiktok_search_users"
        assert tiktok_search_users_tool.max_results == 5


class TestTikTokSearchUsersRun:
    @responses.activate
    def test_successful_search(
        self, tiktok_search_users_tool: ScavioTikTokSearchUsers
    ) -> None:
        responses.add(
            responses.POST, SEARCH_USERS_ENDPOINT,
            json=make_tiktok_search_users_response(), status=200,
        )
        result = tiktok_search_users_tool.invoke({"keyword": "cooking"})
        assert "data" in result
        assert len(result["data"]["user_list"]) == 5

    @responses.activate
    def test_empty_results_raises_tool_exception(
        self, tiktok_search_users_tool: ScavioTikTokSearchUsers
    ) -> None:
        responses.add(
            responses.POST, SEARCH_USERS_ENDPOINT,
            json=make_tiktok_search_users_response(
                data={"user_list": [], "cursor": "0", "has_more": 0}
            ),
            status=200,
        )
        result = tiktok_search_users_tool.invoke({"keyword": "xyzzy"})
        assert "No TikTok users found" in result


class TestTikTokSearchUsersAsync:
    @pytest.mark.asyncio
    async def test_async_search(
        self, tiktok_search_users_tool: ScavioTikTokSearchUsers
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokSearchUsersAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_search_users_response(),
        ):
            result = await tiktok_search_users_tool.ainvoke(
                {"keyword": "cooking"}
            )
            assert "data" in result
            assert len(result["data"]["user_list"]) == 5


# ===========================================================================
# Hashtag Info
# ===========================================================================


class TestTikTokHashtagInstantiation:
    def test_default_params(
        self, tiktok_hashtag_tool: ScavioTikTokHashtag
    ) -> None:
        assert tiktok_hashtag_tool.name == "scavio_tiktok_hashtag"


class TestTikTokHashtagRun:
    @responses.activate
    def test_successful_lookup(
        self, tiktok_hashtag_tool: ScavioTikTokHashtag
    ) -> None:
        responses.add(
            responses.POST, HASHTAG_ENDPOINT,
            json=make_tiktok_hashtag_response(), status=200,
        )
        result = tiktok_hashtag_tool.invoke({"hashtag_name": "python"})
        assert "data" in result
        assert result["data"]["challengeInfo"]["challenge"]["title"] == "python"

    @responses.activate
    def test_empty_hashtag_raises_tool_exception(
        self, tiktok_hashtag_tool: ScavioTikTokHashtag
    ) -> None:
        responses.add(
            responses.POST, HASHTAG_ENDPOINT,
            json=make_tiktok_hashtag_response(data={"challengeInfo": None}),
            status=200,
        )
        result = tiktok_hashtag_tool.invoke({"hashtag_name": "xyzzy"})
        assert "No TikTok hashtag found" in result


class TestTikTokHashtagAsync:
    @pytest.mark.asyncio
    async def test_async_lookup(
        self, tiktok_hashtag_tool: ScavioTikTokHashtag
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokHashtagAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_hashtag_response(),
        ):
            result = await tiktok_hashtag_tool.ainvoke(
                {"hashtag_name": "python"}
            )
            assert "data" in result


class TestTikTokHashtagInputSchema:
    def test_schema_fields(self) -> None:
        tool = ScavioTikTokHashtag(scavio_api_key=MOCK_API_KEY)
        props = tool.get_input_schema().model_json_schema()["properties"]
        assert "hashtag_name" in props
        assert "hashtag_id" in props


# ===========================================================================
# Hashtag Videos
# ===========================================================================


class TestTikTokHashtagVideosInstantiation:
    def test_default_params(
        self, tiktok_hashtag_videos_tool: ScavioTikTokHashtagVideos
    ) -> None:
        assert tiktok_hashtag_videos_tool.name == "scavio_tiktok_hashtag_videos"
        assert tiktok_hashtag_videos_tool.max_results == 5


class TestTikTokHashtagVideosRun:
    @responses.activate
    def test_successful_fetch(
        self, tiktok_hashtag_videos_tool: ScavioTikTokHashtagVideos
    ) -> None:
        responses.add(
            responses.POST, HASHTAG_VIDEOS_ENDPOINT,
            json=make_tiktok_hashtag_videos_response(), status=200,
        )
        result = tiktok_hashtag_videos_tool.invoke({"hashtag_id": "123456"})
        assert "data" in result
        assert len(result["data"]["aweme_list"]) == 5

    @responses.activate
    def test_empty_results_raises_tool_exception(
        self, tiktok_hashtag_videos_tool: ScavioTikTokHashtagVideos
    ) -> None:
        responses.add(
            responses.POST, HASHTAG_VIDEOS_ENDPOINT,
            json=make_tiktok_hashtag_videos_response(
                data={"aweme_list": [], "cursor": "0", "has_more": 0}
            ),
            status=200,
        )
        result = tiktok_hashtag_videos_tool.invoke({"hashtag_id": "999999"})
        assert "No videos found" in result


class TestTikTokHashtagVideosForbiddenParams:
    def test_max_results_rejected_at_invocation(
        self, tiktok_hashtag_videos_tool: ScavioTikTokHashtagVideos
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            tiktok_hashtag_videos_tool._run(hashtag_id="123456", max_results=10)


class TestTikTokHashtagVideosAsync:
    @pytest.mark.asyncio
    async def test_async_fetch(
        self, tiktok_hashtag_videos_tool: ScavioTikTokHashtagVideos
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokHashtagVideosAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_hashtag_videos_response(),
        ):
            result = await tiktok_hashtag_videos_tool.ainvoke(
                {"hashtag_id": "123456"}
            )
            assert "data" in result
            assert len(result["data"]["aweme_list"]) == 5


# ===========================================================================
# User Followers
# ===========================================================================


class TestTikTokUserFollowersInstantiation:
    def test_default_params(
        self, tiktok_user_followers_tool: ScavioTikTokUserFollowers
    ) -> None:
        assert tiktok_user_followers_tool.name == "scavio_tiktok_user_followers"
        assert tiktok_user_followers_tool.max_results == 10


class TestTikTokUserFollowersRun:
    @responses.activate
    def test_successful_fetch(
        self, tiktok_user_followers_tool: ScavioTikTokUserFollowers
    ) -> None:
        responses.add(
            responses.POST, USER_FOLLOWERS_ENDPOINT,
            json=make_tiktok_user_followers_response(), status=200,
        )
        result = tiktok_user_followers_tool.invoke(
            {"sec_user_id": "MS4wLjABAAAAtest123"}
        )
        assert "data" in result
        assert len(result["data"]["followers"]) == 10

    @responses.activate
    def test_empty_followers_raises_tool_exception(
        self, tiktok_user_followers_tool: ScavioTikTokUserFollowers
    ) -> None:
        responses.add(
            responses.POST, USER_FOLLOWERS_ENDPOINT,
            json=make_tiktok_user_followers_response(
                data={"followers": [], "has_more": False}
            ),
            status=200,
        )
        result = tiktok_user_followers_tool.invoke(
            {"sec_user_id": "MS4wLjABAAAAtest123"}
        )
        assert "No followers found" in result

    @responses.activate
    def test_pagination_params_forwarded(
        self, tiktok_user_followers_tool: ScavioTikTokUserFollowers
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST, USER_FOLLOWERS_ENDPOINT,
            json=make_tiktok_user_followers_response(), status=200,
        )
        tiktok_user_followers_tool.invoke({
            "sec_user_id": "MS4wLjABAAAAtest123",
            "page_token": "token123",
            "min_time": 1715000000,
        })
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["page_token"] == "token123"
        assert body["min_time"] == 1715000000


class TestTikTokUserFollowersForbiddenParams:
    def test_max_results_rejected_at_invocation(
        self, tiktok_user_followers_tool: ScavioTikTokUserFollowers
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            tiktok_user_followers_tool._run(
                sec_user_id="MS4wLjABAAAAtest123", max_results=5
            )


class TestTikTokUserFollowersAsync:
    @pytest.mark.asyncio
    async def test_async_fetch(
        self, tiktok_user_followers_tool: ScavioTikTokUserFollowers
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokUserFollowersAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_user_followers_response(),
        ):
            result = await tiktok_user_followers_tool.ainvoke(
                {"sec_user_id": "MS4wLjABAAAAtest123"}
            )
            assert "data" in result
            assert len(result["data"]["followers"]) == 10


# ===========================================================================
# User Followings
# ===========================================================================


class TestTikTokUserFollowingsInstantiation:
    def test_default_params(
        self, tiktok_user_followings_tool: ScavioTikTokUserFollowings
    ) -> None:
        assert tiktok_user_followings_tool.name == "scavio_tiktok_user_followings"
        assert tiktok_user_followings_tool.max_results == 10


class TestTikTokUserFollowingsRun:
    @responses.activate
    def test_successful_fetch(
        self, tiktok_user_followings_tool: ScavioTikTokUserFollowings
    ) -> None:
        responses.add(
            responses.POST, USER_FOLLOWINGS_ENDPOINT,
            json=make_tiktok_user_followings_response(), status=200,
        )
        result = tiktok_user_followings_tool.invoke(
            {"sec_user_id": "MS4wLjABAAAAtest123"}
        )
        assert "data" in result
        assert len(result["data"]["followings"]) == 10

    @responses.activate
    def test_empty_followings_raises_tool_exception(
        self, tiktok_user_followings_tool: ScavioTikTokUserFollowings
    ) -> None:
        responses.add(
            responses.POST, USER_FOLLOWINGS_ENDPOINT,
            json=make_tiktok_user_followings_response(
                data={"followings": [], "has_more": False}
            ),
            status=200,
        )
        result = tiktok_user_followings_tool.invoke(
            {"sec_user_id": "MS4wLjABAAAAtest123"}
        )
        assert "No followings found" in result


class TestTikTokUserFollowingsForbiddenParams:
    def test_max_results_rejected_at_invocation(
        self, tiktok_user_followings_tool: ScavioTikTokUserFollowings
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            tiktok_user_followings_tool._run(
                sec_user_id="MS4wLjABAAAAtest123", max_results=5
            )


class TestTikTokUserFollowingsAsync:
    @pytest.mark.asyncio
    async def test_async_fetch(
        self, tiktok_user_followings_tool: ScavioTikTokUserFollowings
    ) -> None:
        with patch(
            "langchain_scavio._utilities.ScavioTikTokUserFollowingsAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_user_followings_response(),
        ):
            result = await tiktok_user_followings_tool.ainvoke(
                {"sec_user_id": "MS4wLjABAAAAtest123"}
            )
            assert "data" in result
            assert len(result["data"]["followings"]) == 10
