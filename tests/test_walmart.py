"""Tests for ScavioWalmartSearch and ScavioWalmartProduct tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio._utilities import SCAVIO_API_URL
from langchain_scavio.scavio_walmart import ScavioWalmartProduct, ScavioWalmartSearch

from .conftest import (
    MOCK_API_KEY,
    make_error_response,
    make_walmart_product_response,
    make_walmart_search_response,
)

SEARCH_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/walmart/search"
PRODUCT_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/walmart/product"


class TestWalmartSearchInstantiation:
    def test_default_params(self, walmart_search_tool: ScavioWalmartSearch) -> None:
        assert walmart_search_tool.name == "scavio_walmart_search"
        assert walmart_search_tool.max_results == 5
        assert walmart_search_tool.handle_tool_error is True

    def test_custom_params(self) -> None:
        tool = ScavioWalmartSearch(scavio_api_key=MOCK_API_KEY, max_results=8)
        assert tool.max_results == 8

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioWalmartSearch(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioWalmartSearch(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioWalmartSearch()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestWalmartSearchRun:
    @responses.activate
    def test_successful_search(self, walmart_search_tool: ScavioWalmartSearch) -> None:
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_walmart_search_response(),
            status=200,
        )
        result = walmart_search_tool.invoke({"query": "air fryer"})
        assert "data" in result
        assert len(result["data"]["products"]) == 5  # truncated by max_results

    @responses.activate
    def test_max_results_truncation(self) -> None:
        tool = ScavioWalmartSearch(scavio_api_key=MOCK_API_KEY, max_results=2)
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_walmart_search_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert len(result["data"]["products"]) == 2

    @responses.activate
    def test_empty_results_raises_tool_exception(
        self, walmart_search_tool: ScavioWalmartSearch
    ) -> None:
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_walmart_search_response(data={"products": []}),
            status=200,
        )
        result = walmart_search_tool.invoke({"query": "xyzzy obscure product"})
        assert "No Walmart results found" in result

    @responses.activate
    def test_api_error_returns_error_dict(
        self, walmart_search_tool: ScavioWalmartSearch
    ) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, SEARCH_ENDPOINT, json=error, status=401)
        result = walmart_search_tool.invoke({"query": "test"})
        assert "error" in str(result).lower()

    @responses.activate
    def test_sort_by_forwarded(self, walmart_search_tool: ScavioWalmartSearch) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_walmart_search_response(),
            status=200,
        )
        walmart_search_tool.invoke({"query": "blender", "sort_by": "price_low"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["sort_by"] == "price_low"

    @responses.activate
    def test_price_filters_forwarded(
        self, walmart_search_tool: ScavioWalmartSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_walmart_search_response(),
            status=200,
        )
        walmart_search_tool.invoke(
            {"query": "toy", "min_price": 500, "max_price": 2000}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["min_price"] == 500
        assert body["max_price"] == 2000


class TestWalmartSearchForbiddenParams:
    def test_init_only_params_rejected_at_invocation(
        self, walmart_search_tool: ScavioWalmartSearch
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            walmart_search_tool._run(query="test", max_results=10)


class TestWalmartSearchAsync:
    @pytest.mark.asyncio
    async def test_async_search(
        self, walmart_search_tool: ScavioWalmartSearch
    ) -> None:
        mock_resp = make_walmart_search_response()
        with patch(
            "langchain_scavio._utilities.ScavioWalmartSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await walmart_search_tool.ainvoke({"query": "async test"})
            assert "data" in result
            assert len(result["data"]["products"]) == 5

    @pytest.mark.asyncio
    async def test_async_empty_results(
        self, walmart_search_tool: ScavioWalmartSearch
    ) -> None:
        mock_resp = make_walmart_search_response(data={"products": []})
        with patch(
            "langchain_scavio._utilities.ScavioWalmartSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await walmart_search_tool.ainvoke({"query": "xyzzy"})
            assert "No Walmart results found" in result


class TestWalmartSearchInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioWalmartSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        props = input_schema["properties"]
        assert "query" in props
        assert "sort_by" in props
        assert "min_price" in props
        assert "max_price" in props
        assert "fulfillment_speed" in props

    def test_query_is_required(self) -> None:
        tool = ScavioWalmartSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "query" in input_schema.get("required", [])


class TestWalmartProductInstantiation:
    def test_default_params(self, walmart_product_tool: ScavioWalmartProduct) -> None:
        assert walmart_product_tool.name == "scavio_walmart_product"
        assert walmart_product_tool.handle_tool_error is True

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioWalmartProduct(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioWalmartProduct(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioWalmartProduct()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestWalmartProductRun:
    @responses.activate
    def test_successful_product_lookup(
        self, walmart_product_tool: ScavioWalmartProduct
    ) -> None:
        responses.add(
            responses.POST,
            PRODUCT_ENDPOINT,
            json=make_walmart_product_response(),
            status=200,
        )
        result = walmart_product_tool.invoke({"product_id": "987654321"})
        assert "data" in result
        assert result["data"]["id"] == "987654321"

    @responses.activate
    def test_empty_product_raises_tool_exception(
        self, walmart_product_tool: ScavioWalmartProduct
    ) -> None:
        responses.add(
            responses.POST,
            PRODUCT_ENDPOINT,
            json=make_walmart_product_response(data=None),
            status=200,
        )
        result = walmart_product_tool.invoke({"product_id": "000000000"})
        assert "No Walmart product found" in result

    @responses.activate
    def test_api_error_returns_error_dict(
        self, walmart_product_tool: ScavioWalmartProduct
    ) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, PRODUCT_ENDPOINT, json=error, status=401)
        result = walmart_product_tool.invoke({"product_id": "987654321"})
        assert "error" in str(result).lower()

    @responses.activate
    def test_product_id_forwarded(
        self, walmart_product_tool: ScavioWalmartProduct
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            PRODUCT_ENDPOINT,
            json=make_walmart_product_response(),
            status=200,
        )
        walmart_product_tool.invoke({"product_id": "987654321"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["product_id"] == "987654321"


class TestWalmartProductAsync:
    @pytest.mark.asyncio
    async def test_async_product_lookup(
        self, walmart_product_tool: ScavioWalmartProduct
    ) -> None:
        mock_resp = make_walmart_product_response()
        with patch(
            "langchain_scavio._utilities.ScavioWalmartProductAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await walmart_product_tool.ainvoke({"product_id": "987654321"})
            assert "data" in result

    @pytest.mark.asyncio
    async def test_async_empty_product(
        self, walmart_product_tool: ScavioWalmartProduct
    ) -> None:
        mock_resp = make_walmart_product_response(data=None)
        with patch(
            "langchain_scavio._utilities.ScavioWalmartProductAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await walmart_product_tool.ainvoke({"product_id": "000000000"})
            assert "No Walmart product found" in result


class TestWalmartProductInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioWalmartProduct(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        props = input_schema["properties"]
        assert "product_id" in props
        assert "delivery_zip" in props

    def test_product_id_is_required(self) -> None:
        tool = ScavioWalmartProduct(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "product_id" in input_schema.get("required", [])
