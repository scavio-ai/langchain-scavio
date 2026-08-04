"""Tests for the 10 Reddit tools added in 3.3.

``test_reddit.py`` covers ScavioRedditSearch and ScavioRedditPost, which
predate this file. Together the two files cover all 12 Reddit endpoints.
"""

from __future__ import annotations

import json as json_mod
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio import (
    ScavioRedditCommentReplies,
    ScavioRedditPopular,
    ScavioRedditPostComments,
    ScavioRedditSearchSuggestions,
    ScavioRedditSubreddit,
    ScavioRedditSubredditPosts,
    ScavioRedditTrending,
    ScavioRedditUser,
    ScavioRedditUserComments,
    ScavioRedditUserPosts,
)
from langchain_scavio._utilities import SCAVIO_API_URL

from .conftest import MOCK_API_KEY, make_error_response

# (tool class, endpoint path, invoke payload, mocked payload, list key or None,
#  expected wire body subset, fragment of the empty-result error)
CASES: list[tuple[Any, ...]] = [
    (
        ScavioRedditSearchSuggestions,
        "/api/v1/reddit/search/suggestions",
        {"query": "python"},
        {"suggestions": [f"python {n}" for n in range(12)], "total_count": 12},
        "suggestions",
        {"query": "python"},
        "No Reddit search suggestions found for",
    ),
    (
        ScavioRedditPostComments,
        "/api/v1/reddit/post/comments",
        {"post_id": "t3_1v6ngaf", "sort": "NEW"},
        {
            "comments": [{"comment_id": f"t1_{n}"} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "comments",
        {"post_id": "t3_1v6ngaf", "sort": "NEW"},
        "No comments found for Reddit post",
    ),
    (
        ScavioRedditCommentReplies,
        "/api/v1/reddit/post/comments/replies",
        {"post_id": "t3_1v6ngaf", "cursor": "reply_cursor_abc"},
        {
            "replies": [{"comment_id": f"t1_{n}"} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "replies",
        {"post_id": "t3_1v6ngaf", "cursor": "reply_cursor_abc"},
        "No replies found on Reddit post",
    ),
    (
        ScavioRedditSubreddit,
        "/api/v1/reddit/subreddit",
        {"subreddit": "AskReddit"},
        {"id": "t5_2qh1i", "name": "AskReddit", "subscribers": 47000000},
        None,
        {"subreddit": "AskReddit"},
        "No Reddit subreddit found for",
    ),
    (
        ScavioRedditSubredditPosts,
        "/api/v1/reddit/subreddit/posts",
        {"subreddit": "programming", "sort": "RISING"},
        {
            "posts": [{"post_id": f"t3_{n}"} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "posts",
        {"subreddit": "programming", "sort": "RISING"},
        "No posts found in Reddit subreddit",
    ),
    (
        ScavioRedditUser,
        "/api/v1/reddit/user",
        {"username": "spez"},
        {"id": "t2_1w72", "name": "spez", "karma": 800000},
        None,
        {"username": "spez"},
        "No Reddit user found for",
    ),
    (
        ScavioRedditUserPosts,
        "/api/v1/reddit/user/posts",
        {"username": "spez", "sort": "TOP"},
        {
            "posts": [{"post_id": f"t3_{n}"} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "posts",
        {"username": "spez", "sort": "TOP"},
        "No posts found for Reddit user",
    ),
    (
        ScavioRedditUserComments,
        "/api/v1/reddit/user/comments",
        {"username": "spez", "sort": "TOP"},
        {
            "comments": [{"comment_id": f"t1_{n}"} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "comments",
        {"username": "spez", "sort": "TOP"},
        "No comments found for Reddit user",
    ),
    (
        ScavioRedditPopular,
        "/api/v1/reddit/popular",
        {},
        {
            "posts": [{"post_id": f"t3_{n}"} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "posts",
        {},
        "No posts found in the Reddit popular feed",
    ),
    (
        ScavioRedditTrending,
        "/api/v1/reddit/trending",
        {},
        {
            "trending": [{"query": f"q{n}", "raw_query": f"q{n}"} for n in range(12)],
            "total_count": 12,
        },
        "trending",
        {},
        "No trending Reddit queries returned",
    ),
]

IDS = [case[0].__name__ for case in CASES]


def _mk(cls: Any) -> Any:
    return cls(scavio_api_key=MOCK_API_KEY)


def _wrap(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a payload in the response envelope the Reddit family uses."""
    return {"data": payload, "response_time": 0.4, "credits_used": 1}


def test_every_tool_is_exported() -> None:
    """Every tool in this family must be importable from the package root."""
    import langchain_scavio

    for case in CASES:
        assert case[0].__name__ in langchain_scavio.__all__
        assert getattr(langchain_scavio, case[0].__name__) is case[0]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_endpoint_path(case: tuple[Any, ...]) -> None:
    cls, path = case[0], case[1]
    assert _mk(cls).api_wrapper._build_url() == f"{SCAVIO_API_URL}{path}"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_tool_metadata(case: tuple[Any, ...]) -> None:
    tool = _mk(case[0])
    assert tool.name.startswith("scavio_reddit_")
    assert tool.handle_tool_error is True
    assert "1 credit" in tool.description.lower()
    assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_required_args_in_schema(case: tuple[Any, ...]) -> None:
    cls, payload = case[0], case[2]
    schema = _mk(cls).get_input_schema().model_json_schema()
    for key in payload:
        assert key in schema["properties"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
@responses.activate
def test_successful_call(case: tuple[Any, ...]) -> None:
    cls, path, payload, data, list_key = case[:5]
    responses.add(
        responses.POST, f"{SCAVIO_API_URL}{path}", json=_wrap(data), status=200
    )
    result = _mk(cls).invoke(payload)
    assert "error" not in result
    if list_key:
        assert len(result["data"][list_key]) == 10  # truncated by max_results


@pytest.mark.parametrize("case", CASES, ids=IDS)
@responses.activate
def test_wire_body(case: tuple[Any, ...]) -> None:
    cls, path, payload, data, _, wire = case[:6]
    responses.add(
        responses.POST, f"{SCAVIO_API_URL}{path}", json=_wrap(data), status=200
    )
    _mk(cls).invoke(payload)
    sent = json_mod.loads(responses.calls[0].request.body)
    assert sent == wire


@pytest.mark.parametrize("case", CASES, ids=IDS)
@responses.activate
def test_empty_payload_raises_tool_exception(case: tuple[Any, ...]) -> None:
    cls, path, payload, _, list_key, _, err = case
    empty = {list_key: []} if list_key else {}
    responses.add(
        responses.POST, f"{SCAVIO_API_URL}{path}", json=_wrap(empty), status=200
    )
    result = _mk(cls).invoke(payload)
    assert err in str(result)


@pytest.mark.parametrize("case", CASES, ids=IDS)
@responses.activate
def test_api_error_returns_error_dict(case: tuple[Any, ...]) -> None:
    cls, path, payload = case[0], case[1], case[2]
    responses.add(
        responses.POST,
        f"{SCAVIO_API_URL}{path}",
        json=make_error_response(401, "unauthorized", "Invalid API key"),
        status=401,
    )
    result = _mk(cls).invoke(payload)
    assert "error" in str(result).lower()


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.asyncio
async def test_async_call(case: tuple[Any, ...]) -> None:
    cls, _, payload, data, list_key = case[:5]
    target = f"langchain_scavio._utilities.{cls.__name__}APIWrapper.raw_results_async"
    with patch(target, new_callable=AsyncMock, return_value=_wrap(data)):
        result = await _mk(cls).ainvoke(payload)
    assert "error" not in result
    if list_key:
        assert len(result["data"][list_key]) == 10


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_max_results_is_instantiation_only(case: tuple[Any, ...]) -> None:
    cls, _, payload, _, list_key = case[:5]
    if not list_key:
        pytest.skip("detail tools do not truncate results")
    with pytest.raises(ValueError, match="instantiation"):
        _mk(cls)._run(**payload, max_results=3)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_api_base_url_and_key_forwarded(case: tuple[Any, ...]) -> None:
    tool = case[0](
        scavio_api_key=MOCK_API_KEY, api_base_url="https://custom.api.dev"
    )
    assert tool.api_wrapper.api_base_url == "https://custom.api.dev"
    assert tool.api_wrapper._build_url() == f"https://custom.api.dev{case[1]}"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_env_var_api_key(
    case: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
    tool = case[0]()
    assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


def test_sort_enums_match_the_api() -> None:
    """Only the subreddit feed accepts RISING; the rest use the 5-value set."""
    feed = _mk(ScavioRedditSubredditPosts).get_input_schema().model_json_schema()
    feed_sorts = set(feed["properties"]["sort"]["anyOf"][0]["enum"])
    assert feed_sorts == {"BEST", "HOT", "NEW", "TOP", "CONTROVERSIAL", "RISING"}

    for cls in (
        ScavioRedditPostComments,
        ScavioRedditCommentReplies,
        ScavioRedditUserPosts,
        ScavioRedditUserComments,
    ):
        schema = _mk(cls).get_input_schema().model_json_schema()
        sorts = set(schema["properties"]["sort"]["anyOf"][0]["enum"])
        assert sorts == {"HOT", "NEW", "TOP", "BEST", "CONTROVERSIAL"}, cls.__name__


def test_comment_replies_cursor_is_required() -> None:
    """cursor is optional everywhere else but mandatory on the replies endpoint."""
    schema = _mk(ScavioRedditCommentReplies).get_input_schema().model_json_schema()
    assert set(schema["required"]) == {"post_id", "cursor"}

    comments = _mk(ScavioRedditPostComments).get_input_schema().model_json_schema()
    assert comments["required"] == ["post_id"]


def test_trending_takes_no_parameters() -> None:
    schema = _mk(ScavioRedditTrending).get_input_schema().model_json_schema()
    assert schema.get("required", []) == []
    assert schema.get("properties", {}) == {}


@responses.activate
def test_popular_paginates_with_cursor() -> None:
    responses.add(
        responses.POST,
        f"{SCAVIO_API_URL}/api/v1/reddit/popular",
        json=_wrap({"posts": [{"post_id": "t3_1"}], "next_cursor": "c2"}),
        status=200,
    )
    _mk(ScavioRedditPopular).invoke({"cursor": "c1"})
    assert json_mod.loads(responses.calls[0].request.body) == {"cursor": "c1"}
