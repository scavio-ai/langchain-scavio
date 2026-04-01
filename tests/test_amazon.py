"""Tests for ScavioAmazonSearch and ScavioAmazonProduct tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio._utilities import SCAVIO_API_URL
from langchain_scavio.scavio_amazon import ScavioAmazonProduct, ScavioAmazonSearch

from .conftest import (
    MOCK_API_KEY,
    make_amazon_product_response,
    make_amazon_search_response,
    make_error_response,
)

SEARCH_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/amazon/search"
PRODUCT_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/amazon/product"


class TestAmazonSearchInstantiation:
    def test_default_params(self, amazon_search_tool: ScavioAmazonSearch) -> None:
        assert amazon_search_tool.name == "scavio_amazon_search"
        assert amazon_search_tool.max_results == 5
        assert amazon_search_tool.pages is None
        assert amazon_search_tool.autoselect_variant is None
        assert amazon_search_tool.handle_tool_error is True

    def test_custom_params(self) -> None:
        tool = ScavioAmazonSearch(
            scavio_api_key=MOCK_API_KEY,
            max_results=10,
            pages=2,
            autoselect_variant=True,
        )
        assert tool.max_results == 10
        assert tool.pages == 2
        assert tool.autoselect_variant is True

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioAmazonSearch(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioAmazonSearch(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioAmazonSearch()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestAmazonSearchRun:
    @responses.activate
    def test_successful_search(self, amazon_search_tool: ScavioAmazonSearch) -> None:
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_amazon_search_response(),
            status=200,
        )
        result = amazon_search_tool.invoke({"query": "wireless headphones"})
        assert "data" in result
        assert len(result["data"]["products"]) == 5  # truncated by max_results

    @responses.activate
    def test_max_results_truncation(self) -> None:
        tool = ScavioAmazonSearch(scavio_api_key=MOCK_API_KEY, max_results=3)
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_amazon_search_response(),
            status=200,
        )
        result = tool.invoke({"query": "test"})
        assert len(result["data"]["products"]) == 3

    @responses.activate
    def test_empty_results_raises_tool_exception(
        self, amazon_search_tool: ScavioAmazonSearch
    ) -> None:
        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_amazon_search_response(data={"products": []}),
            status=200,
        )
        result = amazon_search_tool.invoke({"query": "xyzzy obscure product"})
        assert "No Amazon results found" in result

    @responses.activate
    def test_api_error_returns_error_dict(
        self, amazon_search_tool: ScavioAmazonSearch
    ) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, SEARCH_ENDPOINT, json=error, status=401)
        result = amazon_search_tool.invoke({"query": "test"})
        assert "error" in str(result).lower()

    @responses.activate
    def test_sort_by_forwarded(self, amazon_search_tool: ScavioAmazonSearch) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_amazon_search_response(),
            status=200,
        )
        amazon_search_tool.invoke(
            {"query": "laptop", "sort_by": "price_low_to_high"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["sort_by"] == "price_low_to_high"

    @responses.activate
    def test_domain_forwarded(self, amazon_search_tool: ScavioAmazonSearch) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_amazon_search_response(),
            status=200,
        )
        amazon_search_tool.invoke({"query": "book", "domain": "co.uk"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["domain"] == "co.uk"


class TestAmazonSearchForbiddenParams:
    def test_init_only_params_rejected_at_invocation(
        self, amazon_search_tool: ScavioAmazonSearch
    ) -> None:
        with pytest.raises(ValueError, match="instantiation"):
            amazon_search_tool._run(query="test", max_results=10)


class TestAmazonSearchAsync:
    @pytest.mark.asyncio
    async def test_async_search(
        self, amazon_search_tool: ScavioAmazonSearch
    ) -> None:
        mock_resp = make_amazon_search_response()
        with patch(
            "langchain_scavio._utilities.ScavioAmazonSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await amazon_search_tool.ainvoke({"query": "async test"})
            assert "data" in result
            assert len(result["data"]["products"]) == 5

    @pytest.mark.asyncio
    async def test_async_empty_results(
        self, amazon_search_tool: ScavioAmazonSearch
    ) -> None:
        mock_resp = make_amazon_search_response(data={"products": []})
        with patch(
            "langchain_scavio._utilities.ScavioAmazonSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await amazon_search_tool.ainvoke({"query": "xyzzy"})
            assert "No Amazon results found" in result


class TestAmazonSearchInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioAmazonSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        props = input_schema["properties"]
        assert "query" in props
        assert "domain" in props
        assert "sort_by" in props
        assert "start_page" in props

    def test_query_is_required(self) -> None:
        tool = ScavioAmazonSearch(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "query" in input_schema.get("required", [])


class TestAmazonProductInstantiation:
    def test_default_params(self, amazon_product_tool: ScavioAmazonProduct) -> None:
        assert amazon_product_tool.name == "scavio_amazon_product"
        assert amazon_product_tool.handle_tool_error is True

    def test_api_key_forwarded_to_wrapper(self) -> None:
        tool = ScavioAmazonProduct(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioAmazonProduct(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioAmazonProduct()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY


class TestAmazonProductRun:
    @responses.activate
    def test_successful_product_lookup(
        self, amazon_product_tool: ScavioAmazonProduct
    ) -> None:
        responses.add(
            responses.POST,
            PRODUCT_ENDPOINT,
            json=make_amazon_product_response(),
            status=200,
        )
        result = amazon_product_tool.invoke({"query": "B001234567"})
        assert "data" in result
        assert result["data"]["asin"] == "B001234567"

    @responses.activate
    def test_empty_product_raises_tool_exception(
        self, amazon_product_tool: ScavioAmazonProduct
    ) -> None:
        responses.add(
            responses.POST,
            PRODUCT_ENDPOINT,
            json=make_amazon_product_response(data=None),
            status=200,
        )
        result = amazon_product_tool.invoke({"query": "BADINVALID"})
        assert "No Amazon product found" in result

    @responses.activate
    def test_api_error_returns_error_dict(
        self, amazon_product_tool: ScavioAmazonProduct
    ) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, PRODUCT_ENDPOINT, json=error, status=401)
        result = amazon_product_tool.invoke({"query": "B001234567"})
        assert "error" in str(result).lower()

    @responses.activate
    def test_asin_forwarded_as_query(
        self, amazon_product_tool: ScavioAmazonProduct
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            PRODUCT_ENDPOINT,
            json=make_amazon_product_response(),
            status=200,
        )
        amazon_product_tool.invoke({"query": "B001234567"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["query"] == "B001234567"


class TestAmazonProductAsync:
    @pytest.mark.asyncio
    async def test_async_product_lookup(
        self, amazon_product_tool: ScavioAmazonProduct
    ) -> None:
        mock_resp = make_amazon_product_response()
        with patch(
            "langchain_scavio._utilities.ScavioAmazonProductAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await amazon_product_tool.ainvoke({"query": "B001234567"})
            assert "data" in result

    @pytest.mark.asyncio
    async def test_async_empty_product(
        self, amazon_product_tool: ScavioAmazonProduct
    ) -> None:
        mock_resp = make_amazon_product_response(data=None)
        with patch(
            "langchain_scavio._utilities.ScavioAmazonProductAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await amazon_product_tool.ainvoke({"query": "BADINVALID"})
            assert "No Amazon product found" in result


class TestAmazonProductInputSchema:
    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioAmazonProduct(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        props = input_schema["properties"]
        assert "query" in props
        assert "domain" in props
        assert "autoselect_variant" in props

    def test_query_is_required(self) -> None:
        tool = ScavioAmazonProduct(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "query" in input_schema.get("required", [])
