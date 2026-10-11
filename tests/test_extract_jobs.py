"""Map, crawl and batch: the multi-page members of the Extract family."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio import (
    ScavioExtractBatch,
    ScavioExtractBatchStatus,
    ScavioExtractCrawl,
    ScavioExtractCrawlStatus,
    ScavioExtractMap,
    _utilities,
)

from .conftest import MOCK_API_KEY

BASE = "https://api.scavio.dev/api/v1/extract"
KEY: dict[str, Any] = {"scavio_api_key": MOCK_API_KEY, "max_requests_per_second": 50}
JOB_ID = "crawl_01a12d21-fb42-7000-88cf-6887f653ee5d"


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_utilities, "WEB_JOB_POLL_INTERVAL", 0.0)


def _created(job_id: str = JOB_ID, kind: str = "crawl") -> dict[str, Any]:
    return {
        "data": {
            "id": job_id,
            "type": kind,
            "status": "queued",
            "credits_per_page": 1,
            "status_url": f"{BASE}/{kind}/{job_id}",
        },
        "credits_used": 0,
    }


def _status(status: str, job_id: str = JOB_ID, **extra: Any) -> dict[str, Any]:
    return {
        "data": {
            "id": job_id,
            "status": status,
            "pages": {"discovered": 3, "queued": 0, "succeeded": 3},
            "credits_used": 3,
            "results": [{"url": "https://hono.dev/docs", "status": "succeeded"}],
            "next_cursor": None,
            **extra,
        }
    }


@responses.activate
def test_map_posts_body_and_returns_links() -> None:
    payload = {
        "data": {
            "url": "https://scavio.dev/",
            "links": [{"url": "https://scavio.dev/", "source": "page"}],
            "total_found": 1,
            "returned": 1,
            "truncated": False,
            "sitemaps_read": 1,
        },
        "credits_used": 1,
    }
    responses.add(responses.POST, f"{BASE}/map", json=payload, status=200)
    tool = ScavioExtractMap(**KEY)
    result = tool.invoke(
        {"url": "scavio.dev", "limit": 5, "include_paths": ["/blog/**"]}
    )
    assert result["data"]["links"][0]["source"] == "page"
    body = json.loads(responses.calls[0].request.body)
    assert body == {"url": "scavio.dev", "limit": 5, "include_paths": ["/blog/**"]}


@responses.activate
def test_crawl_wait_zero_returns_the_202_without_polling() -> None:
    responses.add(responses.POST, f"{BASE}/crawl", json=_created(), status=202)
    tool = ScavioExtractCrawl(**KEY)
    result = tool.invoke({"url": "https://hono.dev/docs", "wait_seconds": 0})
    assert result["data"]["id"] == JOB_ID
    assert len(responses.calls) == 1
    body = json.loads(responses.calls[0].request.body)
    assert body == {"url": "https://hono.dev/docs"}
    assert "wait_seconds" not in body


@responses.activate
def test_crawl_waits_until_completed_then_reads_results() -> None:
    responses.add(responses.POST, f"{BASE}/crawl", json=_created(), status=202)
    status_url = f"{BASE}/crawl/{JOB_ID}"
    responses.add(responses.GET, status_url, json=_status("running"))
    responses.add(responses.GET, status_url, json=_status("completed"))
    responses.add(responses.GET, status_url, json=_status("completed"))
    tool = ScavioExtractCrawl(**KEY)
    result = tool.invoke(
        {
            "url": "https://hono.dev/docs",
            "max_pages": 3,
            "wait_seconds": 30,
            "limit": 10,
        }
    )
    assert result["data"]["status"] == "completed"
    assert json.loads(responses.calls[0].request.body) == {
        "url": "https://hono.dev/docs",
        "max_pages": 3,
    }
    # Progress probes skip content; the final read carries the caller's paging.
    assert "include_content=false" in responses.calls[1].request.url
    assert responses.calls[-1].request.url.endswith("?limit=10")


@responses.activate
def test_crawl_still_running_when_the_wait_ends_returns_its_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every clock reading is 10 s after the last, so a 5 s wait runs out
    # after a single progress probe.
    ticks = iter(range(0, 10_000, 10))
    monkeypatch.setattr(_utilities.time, "monotonic", lambda: float(next(ticks)))
    responses.add(responses.POST, f"{BASE}/crawl", json=_created(), status=202)
    status_url = f"{BASE}/crawl/{JOB_ID}"
    responses.add(responses.GET, status_url, json=_status("running"))
    tool = ScavioExtractCrawl(**KEY)
    result = tool.invoke({"url": "https://hono.dev/docs", "wait_seconds": 5})
    assert result["data"]["status"] == "running"
    assert result["data"]["id"] == JOB_ID


@responses.activate
def test_crawl_status_passes_cursor_and_booleans_as_query() -> None:
    status_url = f"{BASE}/crawl/{JOB_ID}"
    responses.add(responses.GET, status_url, json=_status("completed"))
    tool = ScavioExtractCrawlStatus(**KEY)
    tool.invoke({"id": JOB_ID, "cursor": "25", "include_content": False})
    url = responses.calls[0].request.url
    assert "cursor=25" in url and "include_content=false" in url


@responses.activate
def test_batch_posts_urls_and_reads_the_batch_job() -> None:
    batch_id = "batch_01a12d22-1d54-7000-9210-4263f1735ce9"
    responses.add(
        responses.POST,
        f"{BASE}/batch",
        json=_created(batch_id, "batch"),
        status=202,
    )
    status_url = f"{BASE}/batch/{batch_id}"
    responses.add(responses.GET, status_url, json=_status("completed", batch_id))
    responses.add(responses.GET, status_url, json=_status("completed", batch_id))
    tool = ScavioExtractBatch(**KEY)
    result = tool.invoke(
        {"urls": ["https://example.com/", "https://hono.dev/docs"], "mode": "normal"}
    )
    assert result["data"]["id"] == batch_id
    assert json.loads(responses.calls[0].request.body) == {
        "urls": ["https://example.com/", "https://hono.dev/docs"],
        "mode": "normal",
    }


@responses.activate
def test_batch_status_reads_the_batch_path() -> None:
    batch_id = "batch_1"
    responses.add(responses.GET, f"{BASE}/batch/{batch_id}", json=_status("running"))
    ScavioExtractBatchStatus(**KEY).invoke({"id": batch_id})
    assert responses.calls[0].request.url == f"{BASE}/batch/{batch_id}?limit=10"


@responses.activate
def test_job_errors_surface_the_api_message() -> None:
    responses.add(
        responses.POST,
        f"{BASE}/crawl",
        json={"error": "Active job limit reached"},
        status=429,
    )
    result = ScavioExtractCrawl(**KEY).invoke(
        {"url": "https://hono.dev/docs"}
    )
    assert "Rate limit exceeded" in result["error"]
    assert "Active job limit reached" in result["error"]


@pytest.mark.asyncio
async def test_async_crawl_waits_for_completion() -> None:
    tool = ScavioExtractCrawl(**KEY)
    send = AsyncMock(
        side_effect=[_created(), _status("completed"), _status("completed")]
    )
    with patch.object(_utilities._ScavioExtractJobAPIWrapper, "_send_async", send):
        result = await tool.ainvoke({"url": "https://hono.dev/docs"})
    assert result["data"]["status"] == "completed"
    assert send.await_args_list[0].args[0] == "POST"
    assert send.await_args_list[1].args[0] == "GET"


def test_descriptions_state_the_price_and_never_the_fetch_method() -> None:
    banned = ("proxy", "residential", "datacenter", "headless", "browser")
    for cls in (
        ScavioExtractMap,
        ScavioExtractCrawl,
        ScavioExtractCrawlStatus,
        ScavioExtractBatch,
        ScavioExtractBatchStatus,
    ):
        tool: Any = cls(**KEY)
        text = (
            tool.description + json.dumps(tool.args_schema.model_json_schema())
        ).lower()
        assert "credit" in text, cls.__name__
        for word in banned:
            assert word not in text, (cls.__name__, word)
