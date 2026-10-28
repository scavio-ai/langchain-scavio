"""HEB tools: store_id targets one store and is forwarded in the request body."""

from __future__ import annotations

import json

import responses

from langchain_scavio import ScavioHebProduct, ScavioHebSearch
from langchain_scavio._utilities import SCAVIO_API_URL

from .conftest import MOCK_API_KEY

_OK = {"data": {"store": {"store_id": "441", "name": "Uvalde H-E-B"}}, "credits_used": 2}


@responses.activate
def test_search_forwards_store_id() -> None:
    responses.add(responses.POST, f"{SCAVIO_API_URL}/api/v1/heb/search", json=_OK, status=200)
    ScavioHebSearch(scavio_api_key=MOCK_API_KEY).invoke(
        {"query": "whole milk", "store_id": "441", "page_size": 10}
    )
    body = json.loads(responses.calls[0].request.body)
    assert body == {"query": "whole milk", "store_id": "441", "page_size": 10}


@responses.activate
def test_product_accepts_product_id_alone_with_store_id() -> None:
    responses.add(responses.POST, f"{SCAVIO_API_URL}/api/v1/heb/product", json=_OK, status=200)
    ScavioHebProduct(scavio_api_key=MOCK_API_KEY).invoke({"product_id": "314130", "store_id": "441"})
    body = json.loads(responses.calls[0].request.body)
    assert body == {"product_id": "314130", "store_id": "441"}


@responses.activate
def test_search_without_store_id_omits_it() -> None:
    responses.add(responses.POST, f"{SCAVIO_API_URL}/api/v1/heb/search", json=_OK, status=200)
    ScavioHebSearch(scavio_api_key=MOCK_API_KEY).invoke({"query": "eggs"})
    assert "store_id" not in json.loads(responses.calls[0].request.body)
