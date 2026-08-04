"""Tests for the Scavio X (Twitter) tools (11 endpoints)."""

from __future__ import annotations

import json as json_mod
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio import (
    ScavioXSearch,
    ScavioXTrending,
    ScavioXTweet,
    ScavioXTweetComments,
    ScavioXTweetRetweeters,
    ScavioXUser,
    ScavioXUserFollowers,
    ScavioXUserFollowings,
    ScavioXUserMedia,
    ScavioXUserReplies,
    ScavioXUserTweets,
)
from langchain_scavio._utilities import SCAVIO_API_URL

from .conftest import MOCK_API_KEY, make_error_response

# (tool class, endpoint path, invoke payload, mocked payload, list key or None,
#  expected wire body subset, fragment of the empty-result error)
CASES: list[tuple[Any, ...]] = [
    (
        ScavioXSearch,
        "/api/v1/x/search",
        {"search": "langchain", "search_type": "Latest"},
        {
            "timeline": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "timeline",
        {"search": "langchain", "search_type": "Latest"},
        "No X results found for",
    ),
    (
        ScavioXTweet,
        "/api/v1/x/tweet",
        {"tweet_id": "1808168603721650364"},
        {"tweet_id": "1808168603721650364", "text": "hello"},
        None,
        {"tweet_id": "1808168603721650364"},
        "No X tweet found for id",
    ),
    (
        ScavioXTweetComments,
        "/api/v1/x/tweet/comments",
        {"tweet_id": "1808168603721650364", "rank": "latest"},
        {
            "timeline": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "timeline",
        {"tweet_id": "1808168603721650364", "rank": "latest"},
        "No replies found for X tweet",
    ),
    (
        ScavioXTweetRetweeters,
        "/api/v1/x/tweet/retweeters",
        {"tweet_id": "1808168603721650364"},
        {
            "retweeters": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "retweeters",
        {"tweet_id": "1808168603721650364"},
        "No retweeters found for X tweet",
    ),
    (
        ScavioXUser,
        "/api/v1/x/user",
        {"screen_name": "elonmusk"},
        {"user_id": "44196397", "screen_name": "elonmusk"},
        None,
        {"screen_name": "elonmusk"},
        "No X profile found for",
    ),
    (
        ScavioXUserTweets,
        "/api/v1/x/user/tweets",
        {"screen_name": "elonmusk"},
        {
            "timeline": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "timeline",
        {"screen_name": "elonmusk"},
        "No tweets found for X user",
    ),
    (
        ScavioXUserReplies,
        "/api/v1/x/user/replies",
        {"screen_name": "elonmusk"},
        {
            "timeline": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "timeline",
        {"screen_name": "elonmusk"},
        "No replies found for X user",
    ),
    (
        ScavioXUserMedia,
        "/api/v1/x/user/media",
        {"screen_name": "nasa"},
        {
            "timeline": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "timeline",
        {"screen_name": "nasa"},
        "No media tweets found for X user",
    ),
    (
        ScavioXUserFollowers,
        "/api/v1/x/user/followers",
        {"screen_name": "elonmusk"},
        {
            "followers": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "followers",
        {"screen_name": "elonmusk"},
        "No followers found for X user",
    ),
    (
        ScavioXUserFollowings,
        "/api/v1/x/user/followings",
        {"screen_name": "elonmusk"},
        {
            "following": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "following",
        {"screen_name": "elonmusk"},
        "No followings found for X user",
    ),
    (
        ScavioXTrending,
        "/api/v1/x/trending",
        {"country": "UnitedStates"},
        {
            "trends": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "trends",
        {"country": "UnitedStates"},
        "No X trends found for",
    ),
]

IDS = [case[0].__name__ for case in CASES]

ENVELOPE = "data"


def _mk(cls: Any) -> Any:
    return cls(scavio_api_key=MOCK_API_KEY)


def _wrap(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a payload in the response envelope the endpoint family uses."""
    if ENVELOPE == "data":
        return {"data": payload, "response_time": 0.4}
    return {**payload, "response_time": 0.4, "credits_used": 1}


def _body(raw: dict[str, Any]) -> dict[str, Any]:
    return raw["data"] if ENVELOPE == "data" else raw


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
    assert tool.name.startswith("scavio_")
    assert tool.handle_tool_error is True
    assert "credit" in tool.description.lower()
    assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_required_args_in_schema(case: tuple[Any, ...]) -> None:
    cls, _, payload = case[0], case[1], case[2]
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
        assert len(_body(result)[list_key]) == 10  # truncated by max_results


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
        assert len(_body(result)[list_key]) == 10


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_max_results_is_instantiation_only(case: tuple[Any, ...]) -> None:
    cls, _, payload, _, list_key = case[:5]
    if not list_key:
        pytest.skip("detail tools do not truncate results")
    with pytest.raises(ValueError, match="instantiation"):
        _mk(cls)._run(**payload, max_results=3)
