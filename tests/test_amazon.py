"""Tests for the ScavioAmazonSearch, ScavioAmazonProduct and ScavioAmazonOffers tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import responses
from pydantic import ValidationError

from langchain_scavio._utilities import SCAVIO_API_URL
from langchain_scavio.scavio_amazon import (
    ScavioAmazonOffers,
    ScavioAmazonOffersInput,
    ScavioAmazonProduct,
    ScavioAmazonProductInput,
    ScavioAmazonSearch,
)

from .conftest import (
    MOCK_API_KEY,
    make_amazon_product_response,
    make_amazon_search_response,
    make_error_response,
)

SEARCH_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/amazon/search"
PRODUCT_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/amazon/product"
OFFERS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/amazon/offers"

# Params the old provider accepted and the current one has no equivalent for.
# They must not reappear as typed fields: a tool schema advertising a filter the
# API silently drops is worse than no filter at all.
RETIRED_PARAMS = (
    "sort_by",
    "pages",
    "category_id",
    "merchant_id",
    "language",
    "currency",
    "device",
    "zip_code",
    "autoselect_variant",
)


class TestAmazonSearchInstantiation:
    def test_default_params(self, amazon_search_tool: ScavioAmazonSearch) -> None:
        assert amazon_search_tool.name == "scavio_amazon_search"
        assert amazon_search_tool.max_results == 5
        assert amazon_search_tool.handle_tool_error is True

    def test_custom_params(self) -> None:
        tool = ScavioAmazonSearch(scavio_api_key=MOCK_API_KEY, max_results=10)
        assert tool.max_results == 10

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
    def test_country_and_page_forwarded(
        self, amazon_search_tool: ScavioAmazonSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_amazon_search_response(),
            status=200,
        )
        amazon_search_tool.invoke({"query": "laptop", "country": "gb", "page": 2})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["country"] == "gb"
        assert body["page"] == 2

    @responses.activate
    def test_deprecated_aliases_still_forwarded(
        self, amazon_search_tool: ScavioAmazonSearch
    ) -> None:
        """domain/start_page are deprecated but must still reach the wire."""
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_amazon_search_response(),
            status=200,
        )
        amazon_search_tool.invoke(
            {"query": "book", "domain": "co.uk", "start_page": 3}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["domain"] == "co.uk"
        assert body["start_page"] == 3

    @responses.activate
    def test_retired_params_never_reach_the_wire(
        self, amazon_search_tool: ScavioAmazonSearch
    ) -> None:
        import json as json_mod

        responses.add(
            responses.POST,
            SEARCH_ENDPOINT,
            json=make_amazon_search_response(),
            status=200,
        )
        amazon_search_tool.invoke(
            {"query": "laptop", **{p: "x" for p in RETIRED_PARAMS}}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert not set(body) & set(RETIRED_PARAMS)


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
        assert "country" in props
        assert "page" in props
        assert "domain" in props
        assert "start_page" in props
        assert not set(props) & set(RETIRED_PARAMS)

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
        assert "asin" in props
        assert "query" in props
        assert "country" in props
        assert "domain" in props
        assert not set(props) & set(RETIRED_PARAMS)

    def test_asin_is_required(self) -> None:
        tool = ScavioAmazonProduct(scavio_api_key=MOCK_API_KEY)
        input_schema = tool.get_input_schema().model_json_schema()
        assert "asin" in input_schema.get("required", [])
        assert "query" not in input_schema.get("required", [])

    def test_query_alias_satisfies_the_asin_requirement(self) -> None:
        """Pre-3.4 callers pass query; validation must still accept that."""
        parsed = ScavioAmazonProductInput.model_validate({"query": "B001234567"})
        assert parsed.asin == "B001234567"

    def test_neither_asin_nor_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScavioAmazonProductInput.model_validate({"country": "us"})

    @responses.activate
    def test_asin_field_forwarded_as_query(self) -> None:
        import json as json_mod

        tool = ScavioAmazonProduct(scavio_api_key=MOCK_API_KEY)
        responses.add(
            responses.POST,
            PRODUCT_ENDPOINT,
            json=make_amazon_product_response(),
            status=200,
        )
        tool.invoke({"asin": "B001234567", "domain": "co.uk"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["query"] == "B001234567"
        assert body["domain"] == "co.uk"
        assert "asin" not in body


class TestAmazonOffers:
    def test_instantiation(self) -> None:
        tool = ScavioAmazonOffers(scavio_api_key=MOCK_API_KEY)
        assert tool.name == "scavio_amazon_offers"
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_schema_has_expected_fields(self) -> None:
        tool = ScavioAmazonOffers(scavio_api_key=MOCK_API_KEY)
        props = tool.get_input_schema().model_json_schema()["properties"]
        assert "asin" in props
        assert "query" in props
        assert "country" in props
        assert "domain" in props
        assert not set(props) & set(RETIRED_PARAMS)

    def test_offers_query_alias_satisfies_the_asin_requirement(self) -> None:
        parsed = ScavioAmazonOffersInput.model_validate({"query": "B001234567"})
        assert parsed.asin == "B001234567"

    @responses.activate
    def test_asin_forwarded_as_query(self) -> None:
        import json as json_mod

        tool = ScavioAmazonOffers(scavio_api_key=MOCK_API_KEY)
        responses.add(
            responses.POST,
            OFFERS_ENDPOINT,
            json={"data": {"asin": "B001234567", "offers": [], "count": 0}},
            status=200,
        )
        tool.invoke({"query": "B001234567", "country": "de"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body["query"] == "B001234567"
        assert body["country"] == "de"

    @responses.activate
    def test_empty_offers_list_is_not_an_error(self) -> None:
        """Amazon-only ASINs legitimately have zero third-party offers."""
        tool = ScavioAmazonOffers(scavio_api_key=MOCK_API_KEY)
        responses.add(
            responses.POST,
            OFFERS_ENDPOINT,
            json={
                "data": {
                    "asin": "B001234567",
                    "offers": [],
                    "count": 0,
                    "note": "Ships from and sold by Amazon.com",
                }
            },
            status=200,
        )
        result = tool.invoke({"query": "B001234567"})
        assert result["data"]["count"] == 0

    @responses.activate
    def test_missing_data_raises_tool_exception(self) -> None:
        tool = ScavioAmazonOffers(scavio_api_key=MOCK_API_KEY)
        responses.add(responses.POST, OFFERS_ENDPOINT, json={"data": None}, status=200)
        result = tool.invoke({"query": "BADINVALID"})
        assert "No Amazon offer listing found" in result

    @pytest.mark.asyncio
    async def test_async_offers(self) -> None:
        tool = ScavioAmazonOffers(scavio_api_key=MOCK_API_KEY)
        with patch(
            "langchain_scavio._utilities.ScavioAmazonOffersAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value={"data": {"asin": "B001234567", "offers": [], "count": 0}},
        ):
            result = await tool.ainvoke({"query": "B001234567"})
            assert "data" in result
