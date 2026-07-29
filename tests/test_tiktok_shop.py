"""Tests for all Scavio TikTok Shop tools.

The response bodies come from tests/conftest.py, whose shapes were produced by
running the backend normalizers over the recorded fixtures, so a tool reading
the wrong key fails here rather than at runtime.
"""

from __future__ import annotations

import json as json_mod
from unittest.mock import AsyncMock, patch

import pytest
import responses

from langchain_scavio._utilities import SCAVIO_API_URL
from langchain_scavio.scavio_tiktok_shop import (
    ScavioTikTokShopCategories,
    ScavioTikTokShopCategoryProducts,
    ScavioTikTokShopProduct,
    ScavioTikTokShopProductReviews,
    ScavioTikTokShopResolve,
    ScavioTikTokShopSearch,
    ScavioTikTokShopSearchSuggestions,
    ScavioTikTokShopShopProducts,
)

from .conftest import (
    MOCK_API_KEY,
    make_error_response,
    make_tiktok_shop_categories_response,
    make_tiktok_shop_category_products_response,
    make_tiktok_shop_not_found_response,
    make_tiktok_shop_product_response,
    make_tiktok_shop_resolve_response,
    make_tiktok_shop_reviews_response,
    make_tiktok_shop_search_response,
    make_tiktok_shop_shop_products_response,
    make_tiktok_shop_suggestions_response,
)

SEARCH_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok-shop/search"
SUGGESTIONS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok-shop/search/suggestions"
PRODUCT_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok-shop/product"
REVIEWS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok-shop/product/reviews"
CATEGORIES_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok-shop/categories"
CATEGORY_PRODUCTS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok-shop/category/products"
SHOP_PRODUCTS_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok-shop/shop/products"
RESOLVE_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/tiktok-shop/resolve"


# ===========================================================================
# Search
# ===========================================================================


class TestTikTokShopSearchInstantiation:
    def test_default_params(
        self, tiktok_shop_search_tool: ScavioTikTokShopSearch
    ) -> None:
        assert tiktok_shop_search_tool.name == "scavio_tiktok_shop_search"
        assert tiktok_shop_search_tool.max_results == 10
        assert tiktok_shop_search_tool.handle_tool_error is True

    def test_api_key_forwarded(self) -> None:
        tool = ScavioTikTokShopSearch(scavio_api_key=MOCK_API_KEY)
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_base_url_forwarded(self) -> None:
        tool = ScavioTikTokShopSearch(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert tool.api_wrapper.api_base_url == "https://custom.api.dev"

    def test_env_var_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        tool = ScavioTikTokShopSearch()
        assert tool.api_wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_description_states_search_is_not_a_detail_pipeline(self) -> None:
        tool = ScavioTikTokShopSearch(scavio_api_key=MOCK_API_KEY)
        assert "44%" in tool.description
        assert "exact prices" in tool.description


class TestTikTokShopSearchRun:
    @responses.activate
    def test_successful_search(
        self, tiktok_shop_search_tool: ScavioTikTokShopSearch
    ) -> None:
        responses.add(
            responses.POST, SEARCH_ENDPOINT,
            json=make_tiktok_shop_search_response(), status=200,
        )
        result = tiktok_shop_search_tool.invoke({"search": "phone case"})
        assert result["data"]["products"][0]["product_id"] == "1732483132803683201"
        assert result["data"]["products"][0]["price"]["current"] == 4.88
        assert result["data"]["has_more"] is True
        assert result["data"]["degraded"] is False

    @responses.activate
    def test_truncates_to_max_results(
        self, tiktok_shop_search_tool: ScavioTikTokShopSearch
    ) -> None:
        responses.add(
            responses.POST, SEARCH_ENDPOINT,
            json=make_tiktok_shop_search_response(num_products=30), status=200,
        )
        result = tiktok_shop_search_tool.invoke({"search": "phone case"})
        assert len(result["data"]["products"]) == 10

    @responses.activate
    def test_wire_param_is_search_not_query(
        self, tiktok_shop_search_tool: ScavioTikTokShopSearch
    ) -> None:
        responses.add(
            responses.POST, SEARCH_ENDPOINT,
            json=make_tiktok_shop_search_response(), status=200,
        )
        tiktok_shop_search_tool.invoke({"search": "phone case", "cursor": "abc"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body == {"search": "phone case", "cursor": "abc"}

    @responses.activate
    def test_empty_products_raises_tool_exception(
        self, tiktok_shop_search_tool: ScavioTikTokShopSearch
    ) -> None:
        responses.add(
            responses.POST, SEARCH_ENDPOINT,
            json=make_tiktok_shop_search_response(num_products=0), status=200,
        )
        result = tiktok_shop_search_tool.invoke({"search": "zzzz"})
        assert "No TikTok Shop products found" in result

    @responses.activate
    def test_api_error(
        self, tiktok_shop_search_tool: ScavioTikTokShopSearch
    ) -> None:
        responses.add(
            responses.POST, SEARCH_ENDPOINT,
            json=make_error_response(401, "unauthorized", "Invalid API key"),
            status=401,
        )
        result = tiktok_shop_search_tool.invoke({"search": "phone case"})
        assert "error" in str(result).lower()


class TestTikTokShopSearchAsync:
    @pytest.mark.asyncio
    async def test_async_search(
        self, tiktok_shop_search_tool: ScavioTikTokShopSearch
    ) -> None:
        with patch(
            "langchain_scavio._utilities."
            "ScavioTikTokShopSearchAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_shop_search_response(),
        ):
            result = await tiktok_shop_search_tool.ainvoke({"search": "phone case"})
            assert len(result["data"]["products"]) == 10


class TestTikTokShopSearchInputSchema:
    def test_schema_fields(self) -> None:
        tool = ScavioTikTokShopSearch(scavio_api_key=MOCK_API_KEY)
        props = tool.get_input_schema().model_json_schema()["properties"]
        assert "search" in props
        assert "cursor" in props
        assert "region" not in props


# ===========================================================================
# Search Suggestions
# ===========================================================================


class TestTikTokShopSuggestions:
    def test_default_params(
        self, tiktok_shop_suggestions_tool: ScavioTikTokShopSearchSuggestions
    ) -> None:
        assert (
            tiktok_shop_suggestions_tool.name
            == "scavio_tiktok_shop_search_suggestions"
        )

    @responses.activate
    def test_returns_plain_strings(
        self, tiktok_shop_suggestions_tool: ScavioTikTokShopSearchSuggestions
    ) -> None:
        responses.add(
            responses.POST, SUGGESTIONS_ENDPOINT,
            json=make_tiktok_shop_suggestions_response(), status=200,
        )
        result = tiktok_shop_suggestions_tool.invoke({"search": "wireless"})
        suggestions = result["data"]["suggestions"]
        assert suggestions[0] == "wireless charger"
        assert all(isinstance(s, str) for s in suggestions)

    @responses.activate
    def test_region_forwarded(
        self, tiktok_shop_suggestions_tool: ScavioTikTokShopSearchSuggestions
    ) -> None:
        responses.add(
            responses.POST, SUGGESTIONS_ENDPOINT,
            json=make_tiktok_shop_suggestions_response(region="TH"), status=200,
        )
        tiktok_shop_suggestions_tool.invoke({"search": "wireless", "region": "TH"})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body == {"search": "wireless", "region": "TH"}

    @responses.activate
    def test_empty_suggestions_raises_tool_exception(
        self, tiktok_shop_suggestions_tool: ScavioTikTokShopSearchSuggestions
    ) -> None:
        responses.add(
            responses.POST, SUGGESTIONS_ENDPOINT,
            json=make_tiktok_shop_suggestions_response(suggestions=[]), status=200,
        )
        result = tiktok_shop_suggestions_tool.invoke({"search": "wireless"})
        assert "No TikTok Shop keyword suggestions" in result

    @pytest.mark.asyncio
    async def test_async(
        self, tiktok_shop_suggestions_tool: ScavioTikTokShopSearchSuggestions
    ) -> None:
        with patch(
            "langchain_scavio._utilities."
            "ScavioTikTokShopSearchSuggestionsAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_shop_suggestions_response(),
        ):
            result = await tiktok_shop_suggestions_tool.ainvoke(
                {"search": "wireless"}
            )
            assert len(result["data"]["suggestions"]) == 3


# ===========================================================================
# Product Details
# ===========================================================================


class TestTikTokShopProduct:
    def test_default_params(
        self, tiktok_shop_product_tool: ScavioTikTokShopProduct
    ) -> None:
        assert tiktok_shop_product_tool.name == "scavio_tiktok_shop_product"

    def test_description_carries_both_honesty_notes(self) -> None:
        tool = ScavioTikTokShopProduct(scavio_api_key=MOCK_API_KEY)
        assert "does NOT return a price" in tool.description
        assert "44%" in tool.description
        assert "normal outcome" in tool.description

    @responses.activate
    def test_successful_lookup(
        self, tiktok_shop_product_tool: ScavioTikTokShopProduct
    ) -> None:
        responses.add(
            responses.POST, PRODUCT_ENDPOINT,
            json=make_tiktok_shop_product_response(), status=200,
        )
        result = tiktok_shop_product_tool.invoke(
            {"product_id": "1732293553906094315"}
        )
        data = result["data"]
        assert data["product_id"] == "1732293553906094315"
        assert data["shop"]["followers_count"] == 588860
        assert data["rating"]["distribution"]["5"] == 10163
        assert data["variants"][0]["in_stock"] is True
        assert data["top_reviews"][0]["is_verified_purchase"] is True

    @responses.activate
    def test_price_is_null_on_detail(
        self, tiktok_shop_product_tool: ScavioTikTokShopProduct
    ) -> None:
        """Upstream masks the digits; the detail endpoint never carries a price."""
        responses.add(
            responses.POST, PRODUCT_ENDPOINT,
            json=make_tiktok_shop_product_response(), status=200,
        )
        result = tiktok_shop_product_tool.invoke(
            {"product_id": "1732293553906094315"}
        )
        assert result["data"]["price"]["current"] is None
        assert result["data"]["price"]["original"] is None

    @responses.activate
    def test_404_is_a_normal_not_found_not_an_error(
        self, tiktok_shop_product_tool: ScavioTikTokShopProduct
    ) -> None:
        responses.add(
            responses.POST, PRODUCT_ENDPOINT,
            json=make_tiktok_shop_not_found_response(
                "Product not found in this region."
            ),
            status=404,
        )
        result = tiktok_shop_product_tool.invoke(
            {"product_id": "1732293553906094315"}
        )
        assert result["not_found"] is True
        assert result["data"] is None
        assert "error" not in result
        assert "44%" in result["guidance"]

    @responses.activate
    def test_region_forwarded(
        self, tiktok_shop_product_tool: ScavioTikTokShopProduct
    ) -> None:
        responses.add(
            responses.POST, PRODUCT_ENDPOINT,
            json=make_tiktok_shop_product_response(), status=200,
        )
        tiktok_shop_product_tool.invoke(
            {"product_id": "1732293553906094315", "region": "GB"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body == {"product_id": "1732293553906094315", "region": "GB"}

    @pytest.mark.asyncio
    async def test_async(
        self, tiktok_shop_product_tool: ScavioTikTokShopProduct
    ) -> None:
        with patch(
            "langchain_scavio._utilities."
            "ScavioTikTokShopProductAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_shop_product_response(),
        ):
            result = await tiktok_shop_product_tool.ainvoke(
                {"product_id": "1732293553906094315"}
            )
            assert result["data"]["title"].startswith("[medicube]")


# ===========================================================================
# Product Reviews
# ===========================================================================


class TestTikTokShopReviews:
    def test_default_params(
        self, tiktok_shop_reviews_tool: ScavioTikTokShopProductReviews
    ) -> None:
        assert (
            tiktok_shop_reviews_tool.name == "scavio_tiktok_shop_product_reviews"
        )
        assert tiktok_shop_reviews_tool.max_results == 20

    def test_description_warns_about_total_reviews_drift(self) -> None:
        tool = ScavioTikTokShopProductReviews(scavio_api_key=MOCK_API_KEY)
        assert "drifts" in tool.description

    @responses.activate
    def test_successful_reviews(
        self, tiktok_shop_reviews_tool: ScavioTikTokShopProductReviews
    ) -> None:
        responses.add(
            responses.POST, REVIEWS_ENDPOINT,
            json=make_tiktok_shop_reviews_response(), status=200,
        )
        result = tiktok_shop_reviews_tool.invoke(
            {"product_id": "1732293553906094315"}
        )
        data = result["data"]
        assert data["reviews"][0]["reviewer_name"] == "C**"
        assert data["rating"]["distribution"]["1"] == 431
        assert data["filters_applied"]["sort"] == "relevant"
        assert data["has_more"] is True

    @responses.activate
    def test_truncates_to_max_results(
        self, tiktok_shop_reviews_tool: ScavioTikTokShopProductReviews
    ) -> None:
        responses.add(
            responses.POST, REVIEWS_ENDPOINT,
            json=make_tiktok_shop_reviews_response(num_reviews=50), status=200,
        )
        result = tiktok_shop_reviews_tool.invoke(
            {"product_id": "1732293553906094315"}
        )
        assert len(result["data"]["reviews"]) == 20

    @responses.activate
    def test_all_filters_forwarded(
        self, tiktok_shop_reviews_tool: ScavioTikTokShopProductReviews
    ) -> None:
        responses.add(
            responses.POST, REVIEWS_ENDPOINT,
            json=make_tiktok_shop_reviews_response(), status=200,
        )
        tiktok_shop_reviews_tool.invoke(
            {
                "product_id": "1732293553906094315",
                "page": 2,
                "page_size": 100,
                "sort": "recent",
                "rating": 5,
                "has_media": True,
                "verified_only": False,
                "region": "US",
            }
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body == {
            "product_id": "1732293553906094315",
            "page": 2,
            "page_size": 100,
            "sort": "recent",
            "rating": 5,
            "has_media": True,
            "verified_only": False,
            "region": "US",
        }

    @responses.activate
    def test_empty_page_raises_tool_exception(
        self, tiktok_shop_reviews_tool: ScavioTikTokShopProductReviews
    ) -> None:
        responses.add(
            responses.POST, REVIEWS_ENDPOINT,
            json=make_tiktok_shop_reviews_response(reviews=[], has_more=False),
            status=200,
        )
        result = tiktok_shop_reviews_tool.invoke(
            {"product_id": "1732293553906094315"}
        )
        assert "No TikTok Shop reviews found" in result

    @pytest.mark.asyncio
    async def test_async(
        self, tiktok_shop_reviews_tool: ScavioTikTokShopProductReviews
    ) -> None:
        with patch(
            "langchain_scavio._utilities."
            "ScavioTikTokShopProductReviewsAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_shop_reviews_response(),
        ):
            result = await tiktok_shop_reviews_tool.ainvoke(
                {"product_id": "1732293553906094315"}
            )
            assert result["data"]["total_reviews"] == 12561


# ===========================================================================
# Categories
# ===========================================================================


class TestTikTokShopCategories:
    def test_default_params(
        self, tiktok_shop_categories_tool: ScavioTikTokShopCategories
    ) -> None:
        assert tiktok_shop_categories_tool.name == "scavio_tiktok_shop_categories"

    @responses.activate
    def test_returns_tree(
        self, tiktok_shop_categories_tool: ScavioTikTokShopCategories
    ) -> None:
        responses.add(
            responses.POST, CATEGORIES_ENDPOINT,
            json=make_tiktok_shop_categories_response(), status=200,
        )
        result = tiktok_shop_categories_tool.invoke({})
        data = result["data"]
        assert data["total_categories"] == 240
        top = data["categories"][0]
        assert top["parent_id"] is None
        assert top["children"][0]["parent_id"] == "601450"

    @responses.activate
    def test_sends_empty_body(
        self, tiktok_shop_categories_tool: ScavioTikTokShopCategories
    ) -> None:
        responses.add(
            responses.POST, CATEGORIES_ENDPOINT,
            json=make_tiktok_shop_categories_response(), status=200,
        )
        tiktok_shop_categories_tool.invoke({})
        body = json_mod.loads(responses.calls[0].request.body)
        assert body == {}

    def test_schema_has_no_params(self) -> None:
        tool = ScavioTikTokShopCategories(scavio_api_key=MOCK_API_KEY)
        props = tool.get_input_schema().model_json_schema().get("properties", {})
        assert "region" not in props

    @pytest.mark.asyncio
    async def test_async(
        self, tiktok_shop_categories_tool: ScavioTikTokShopCategories
    ) -> None:
        with patch(
            "langchain_scavio._utilities."
            "ScavioTikTokShopCategoriesAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_shop_categories_response(),
        ):
            result = await tiktok_shop_categories_tool.ainvoke({})
            assert result["data"]["total_categories"] == 240


# ===========================================================================
# Category Products
# ===========================================================================


class TestTikTokShopCategoryProducts:
    def test_default_params(
        self,
        tiktok_shop_category_products_tool: ScavioTikTokShopCategoryProducts,
    ) -> None:
        assert (
            tiktok_shop_category_products_tool.name
            == "scavio_tiktok_shop_category_products"
        )

    def test_description_warns_about_page_size(self) -> None:
        tool = ScavioTikTokShopCategoryProducts(scavio_api_key=MOCK_API_KEY)
        assert "inconsistent" in tool.description

    @responses.activate
    def test_successful_listing(
        self,
        tiktok_shop_category_products_tool: ScavioTikTokShopCategoryProducts,
    ) -> None:
        responses.add(
            responses.POST, CATEGORY_PRODUCTS_ENDPOINT,
            json=make_tiktok_shop_category_products_response(), status=200,
        )
        result = tiktok_shop_category_products_tool.invoke(
            {"category_id": "601450"}
        )
        assert result["data"]["category_id"] == "601450"
        assert result["data"]["products"][0]["price"]["currency"] == "USD"
        assert result["data"]["next_cursor"]

    @responses.activate
    def test_truncates_to_max_results(
        self,
        tiktok_shop_category_products_tool: ScavioTikTokShopCategoryProducts,
    ) -> None:
        responses.add(
            responses.POST, CATEGORY_PRODUCTS_ENDPOINT,
            json=make_tiktok_shop_category_products_response(num_products=15),
            status=200,
        )
        result = tiktok_shop_category_products_tool.invoke(
            {"category_id": "601450"}
        )
        assert len(result["data"]["products"]) == 10

    @responses.activate
    def test_unknown_category_is_not_found(
        self,
        tiktok_shop_category_products_tool: ScavioTikTokShopCategoryProducts,
    ) -> None:
        responses.add(
            responses.POST, CATEGORY_PRODUCTS_ENDPOINT,
            json=make_tiktok_shop_not_found_response(
                "No products found for this category id."
            ),
            status=404,
        )
        result = tiktok_shop_category_products_tool.invoke(
            {"category_id": "999999"}
        )
        assert result["not_found"] is True
        assert result["data"] is None

    @responses.activate
    def test_cursor_and_region_forwarded(
        self,
        tiktok_shop_category_products_tool: ScavioTikTokShopCategoryProducts,
    ) -> None:
        responses.add(
            responses.POST, CATEGORY_PRODUCTS_ENDPOINT,
            json=make_tiktok_shop_category_products_response(), status=200,
        )
        tiktok_shop_category_products_tool.invoke(
            {"category_id": "601450", "cursor": "eyJ", "region": "GB"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body == {
            "category_id": "601450",
            "cursor": "eyJ",
            "region": "GB",
        }

    @pytest.mark.asyncio
    async def test_async(
        self,
        tiktok_shop_category_products_tool: ScavioTikTokShopCategoryProducts,
    ) -> None:
        with patch(
            "langchain_scavio._utilities."
            "ScavioTikTokShopCategoryProductsAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_shop_category_products_response(),
        ):
            result = await tiktok_shop_category_products_tool.ainvoke(
                {"category_id": "601450"}
            )
            assert len(result["data"]["products"]) == 10


# ===========================================================================
# Shop Products
# ===========================================================================


class TestTikTokShopShopProducts:
    def test_default_params(
        self, tiktok_shop_shop_products_tool: ScavioTikTokShopShopProducts
    ) -> None:
        assert (
            tiktok_shop_shop_products_tool.name
            == "scavio_tiktok_shop_shop_products"
        )

    def test_description_disclaims_follower_count(self) -> None:
        tool = ScavioTikTokShopShopProducts(scavio_api_key=MOCK_API_KEY)
        assert "follower count" in tool.description

    @responses.activate
    def test_successful_catalog(
        self, tiktok_shop_shop_products_tool: ScavioTikTokShopShopProducts
    ) -> None:
        responses.add(
            responses.POST, SHOP_PRODUCTS_ENDPOINT,
            json=make_tiktok_shop_shop_products_response(), status=200,
        )
        result = tiktok_shop_shop_products_tool.invoke(
            {"shop_id": "7495514739648989419"}
        )
        data = result["data"]
        assert data["shop"]["shop_name"] == "medicube US Store"
        assert "followers_count" not in data["shop"]
        assert data["products"][0]["price"]["current"] == 4.88

    @responses.activate
    def test_unknown_shop_is_not_found(
        self, tiktok_shop_shop_products_tool: ScavioTikTokShopShopProducts
    ) -> None:
        responses.add(
            responses.POST, SHOP_PRODUCTS_ENDPOINT,
            json=make_tiktok_shop_not_found_response(
                "Shop not found or has no products."
            ),
            status=404,
        )
        result = tiktok_shop_shop_products_tool.invoke({"shop_id": "123456"})
        assert result["not_found"] is True

    @responses.activate
    def test_params_forwarded(
        self, tiktok_shop_shop_products_tool: ScavioTikTokShopShopProducts
    ) -> None:
        responses.add(
            responses.POST, SHOP_PRODUCTS_ENDPOINT,
            json=make_tiktok_shop_shop_products_response(), status=200,
        )
        tiktok_shop_shop_products_tool.invoke(
            {"shop_id": "7495514739648989419", "cursor": "eyJ", "region": "SG"}
        )
        body = json_mod.loads(responses.calls[0].request.body)
        assert body == {
            "shop_id": "7495514739648989419",
            "cursor": "eyJ",
            "region": "SG",
        }

    @pytest.mark.asyncio
    async def test_async(
        self, tiktok_shop_shop_products_tool: ScavioTikTokShopShopProducts
    ) -> None:
        with patch(
            "langchain_scavio._utilities."
            "ScavioTikTokShopShopProductsAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_shop_shop_products_response(),
        ):
            result = await tiktok_shop_shop_products_tool.ainvoke(
                {"shop_id": "7495514739648989419"}
            )
            assert len(result["data"]["products"]) == 10


# ===========================================================================
# Resolve
# ===========================================================================


class TestTikTokShopResolve:
    def test_default_params(
        self, tiktok_shop_resolve_tool: ScavioTikTokShopResolve
    ) -> None:
        assert tiktok_shop_resolve_tool.name == "scavio_tiktok_shop_resolve"

    @responses.activate
    def test_successful_resolve(
        self, tiktok_shop_resolve_tool: ScavioTikTokShopResolve
    ) -> None:
        responses.add(
            responses.POST, RESOLVE_ENDPOINT,
            json=make_tiktok_shop_resolve_response(), status=200,
        )
        result = tiktok_shop_resolve_tool.invoke(
            {"url": "https://vt.tiktok.com/ZT2AHoGsE/"}
        )
        data = result["data"]
        assert data["type"] == "product"
        assert data["product_id"] == "8651224669119091502"
        assert data["url"].startswith("https://shop.tiktok.com/")
        assert data["resolved_by"] == "share_link"

    @responses.activate
    def test_shop_resolve(
        self, tiktok_shop_resolve_tool: ScavioTikTokShopResolve
    ) -> None:
        responses.add(
            responses.POST, RESOLVE_ENDPOINT,
            json=make_tiktok_shop_resolve_response(
                type="shop",
                product_id=None,
                shop_id="7495514739648989419",
                url="https://shop.tiktok.com/us/store/7495514739648989419",
                resolved_by="url_pattern",
            ),
            status=200,
        )
        result = tiktok_shop_resolve_tool.invoke(
            {"url": "https://shop.tiktok.com/us/store/x/7495514739648989419"}
        )
        assert result["data"]["shop_id"] == "7495514739648989419"
        assert result["data"]["resolved_by"] == "url_pattern"

    @responses.activate
    def test_dead_link_is_not_found(
        self, tiktok_shop_resolve_tool: ScavioTikTokShopResolve
    ) -> None:
        responses.add(
            responses.POST, RESOLVE_ENDPOINT,
            json=make_tiktok_shop_not_found_response(
                "Could not resolve this link. It may have expired."
            ),
            status=404,
        )
        result = tiktok_shop_resolve_tool.invoke(
            {"url": "https://vt.tiktok.com/DEAD/"}
        )
        assert result["not_found"] is True

    @responses.activate
    def test_unsupported_url_is_an_error(
        self, tiktok_shop_resolve_tool: ScavioTikTokShopResolve
    ) -> None:
        responses.add(
            responses.POST, RESOLVE_ENDPOINT,
            json={"error": "Unsupported TikTok Shop URL."}, status=400,
        )
        result = tiktok_shop_resolve_tool.invoke({"url": "https://example.com/"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_async(
        self, tiktok_shop_resolve_tool: ScavioTikTokShopResolve
    ) -> None:
        with patch(
            "langchain_scavio._utilities."
            "ScavioTikTokShopResolveAPIWrapper.raw_results_async",
            new_callable=AsyncMock,
            return_value=make_tiktok_shop_resolve_response(),
        ):
            result = await tiktok_shop_resolve_tool.ainvoke(
                {"url": "https://vt.tiktok.com/ZT2AHoGsE/"}
            )
            assert result["data"]["type"] == "product"


# ===========================================================================
# Package exports
# ===========================================================================


def test_all_tools_exported_from_package() -> None:
    import langchain_scavio

    for name in (
        "ScavioTikTokShopSearch",
        "ScavioTikTokShopSearchSuggestions",
        "ScavioTikTokShopProduct",
        "ScavioTikTokShopProductReviews",
        "ScavioTikTokShopCategories",
        "ScavioTikTokShopCategoryProducts",
        "ScavioTikTokShopShopProducts",
        "ScavioTikTokShopResolve",
    ):
        assert name in langchain_scavio.__all__
        assert hasattr(langchain_scavio, name)
