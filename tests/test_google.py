"""Tests for the Scavio Google v2 vertical tools (11 endpoints)."""

from __future__ import annotations

import json as json_mod
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio import (
    ScavioGoogleAIMode,
    ScavioGoogleFlights,
    ScavioGoogleHotels,
    ScavioGoogleHotelsDetail,
    ScavioGoogleMapsPlace,
    ScavioGoogleMapsReviews,
    ScavioGoogleShopping,
    ScavioGoogleShoppingProduct,
    ScavioGoogleShoppingStores,
    ScavioGoogleTrending,
    ScavioGoogleTrends,
)
from langchain_scavio._utilities import SCAVIO_API_URL

from .conftest import MOCK_API_KEY, make_error_response

# (tool class, endpoint path, invoke payload, mocked payload, list key or None,
#  expected wire body subset, fragment of the empty-result error)
CASES: list[tuple[Any, ...]] = [
    (
        ScavioGoogleAIMode,
        "/api/v2/google/ai-mode",
        {"query": "how to cache llm responses", "gl": "us"},
        {
            "text_blocks": [{"type": "paragraph", "snippet": "answer"}],
            "references": [{"link": "https://example.com"}],
        },
        None,
        {"query": "how to cache llm responses", "gl": "us"},
        "No Google AI Mode answer returned for",
    ),
    (
        ScavioGoogleMapsPlace,
        "/api/v2/google/maps/place",
        {"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"},
        {"place_results": {"title": "Place", "rating": 4.5}},
        None,
        {"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"},
        "No Google Maps place found for",
    ),
    (
        ScavioGoogleMapsReviews,
        "/api/v2/google/maps/reviews",
        {"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4", "sort_by": "newest"},
        {
            "reviews": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "reviews",
        {"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4", "sort_by": "newest"},
        "No Google Maps reviews found for",
    ),
    (
        ScavioGoogleShopping,
        "/api/v2/google/shopping",
        {"query": "mechanical keyboard", "start": 60},
        {
            "shopping_results": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "shopping_results",
        {"query": "mechanical keyboard", "start": 60},
        "No Google Shopping results found for",
    ),
    (
        ScavioGoogleShoppingProduct,
        "/api/v2/google/shopping/product",
        {"catalog_id": "1234567890", "query": "mechanical keyboard"},
        {"product_results": {"title": "Keyboard", "stores": [{"name": "Shop"}]}},
        None,
        {"catalog_id": "1234567890", "query": "mechanical keyboard"},
        "No Google Shopping product found for",
    ),
    (
        ScavioGoogleShoppingStores,
        "/api/v2/google/shopping/product/stores",
        {"catalog_id": "1234567890", "next_page_token": "CAoQAA"},
        {"product_results": {"stores": [{"name": "Shop"}]}},
        None,
        {"catalog_id": "1234567890", "next_page_token": "CAoQAA"},
        "No further Google Shopping sellers found for catalog_id",
    ),
    (
        ScavioGoogleFlights,
        "/api/v2/google/flights",
        {
            "departure_id": "JFK",
            "arrival_id": "LHR",
            "outbound_date": "2026-09-01",
            "type": 2,
        },
        {
            "best_flights": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "best_flights",
        {
            "departure_id": "JFK",
            "arrival_id": "LHR",
            "outbound_date": "2026-09-01",
            "type": 2,
        },
        "No Google Flights results found for",
    ),
    (
        ScavioGoogleHotels,
        "/api/v2/google/hotels",
        {
            "query": "Lisbon hotels",
            "check_in_date": "2026-09-01",
            "check_out_date": "2026-09-04",
        },
        {
            "properties": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "properties",
        {
            "query": "Lisbon hotels",
            "check_in_date": "2026-09-01",
            "check_out_date": "2026-09-04",
        },
        "No Google Hotels properties found for",
    ),
    (
        ScavioGoogleHotelsDetail,
        "/api/v2/google/hotels/detail",
        {
            "detail_token": "CggIu",
            "check_in_date": "2026-09-01",
            "check_out_date": "2026-09-04",
        },
        {"property": {"name": "Hotel", "booking_sources": []}},
        None,
        {
            "detail_token": "CggIu",
            "check_in_date": "2026-09-01",
            "check_out_date": "2026-09-04",
        },
        "No Google Hotels detail found for that detail_token",
    ),
    (
        ScavioGoogleTrends,
        "/api/v2/google/trends",
        {"query": "langchain,llamaindex", "geo": "US"},
        {
            "interest_over_time": {"timeline_data": [{"date": "2026-01"}]},
            "interest_by_region": [{"location": "US"}],
        },
        None,
        {"query": "langchain,llamaindex", "geo": "US"},
        "No Google Trends data found for",
    ),
    (
        ScavioGoogleTrending,
        "/api/v2/google/trending",
        {"geo": "US", "hours": 24},
        {
            "trends": [{"i": n} for n in range(12)],
            "next_cursor": "cur",
            "has_more": True,
        },
        "trends",
        {"geo": "US", "hours": 24},
        "No Google trending searches found for",
    ),
]

IDS = [case[0].__name__ for case in CASES]

ENVELOPE = "flat"


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


class TestGoogleAIModeIncludeHtml:
    """include_html is exposed but must stay opt-in, never a silent default."""

    def test_include_html_is_on_the_schema(self) -> None:
        schema = _mk(ScavioGoogleAIMode).get_input_schema().model_json_schema()
        assert "include_html" in schema["properties"]
        assert schema["properties"]["include_html"]["default"] is None

    @responses.activate
    def test_include_html_not_sent_unless_asked(self) -> None:
        responses.add(
            responses.POST,
            f"{SCAVIO_API_URL}/api/v2/google/ai-mode",
            json=_wrap({"text_blocks": [{"snippet": "a"}]}),
            status=200,
        )
        _mk(ScavioGoogleAIMode).invoke({"query": "cache llm responses"})
        assert "include_html" not in json_mod.loads(responses.calls[0].request.body)

    @responses.activate
    def test_include_html_forwarded_when_requested(self) -> None:
        responses.add(
            responses.POST,
            f"{SCAVIO_API_URL}/api/v2/google/ai-mode",
            json=_wrap({"text_blocks": [{"snippet": "a"}], "html": "<html>"}),
            status=200,
        )
        result = _mk(ScavioGoogleAIMode).invoke(
            {"query": "cache llm responses", "include_html": True}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["include_html"] is True
        assert result["html"] == "<html>"
