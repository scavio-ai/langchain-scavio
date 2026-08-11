"""Scavio eBay tools for LangChain agents.

Every URL, parameter name, enum and credit cost below is copied from the
Scavio route definition rather than derived from the tool name.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import (
    ScavioEbayProductAPIWrapper,
    ScavioEbaySearchAPIWrapper,
    ScavioEbaySellerAPIWrapper,
)

logger = logging.getLogger(__name__)


def _forward_api_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Split the API wrapper kwargs out of the tool kwargs."""
    api_kwargs: dict[str, Any] = {}
    for key in ("scavio_api_key", "api_base_url", "max_requests_per_second"):
        if key in kwargs:
            api_kwargs[key] = kwargs.pop(key)
    return api_kwargs


def _first_identifier(*values: Any) -> str:
    """Return the first non-empty value, used only in error messages."""
    for value in values:
        if value not in (None, "", [], {}):
            return str(value)
    return "this request"


# --------------------------------------------------------------------------
# ScavioEbaySearch
# --------------------------------------------------------------------------


class ScavioEbaySearchInput(BaseModel):
    """Input schema for the ScavioEbaySearch tool."""

    model_config = ConfigDict(extra="allow")

    query: Optional[str] = Field(
        default=None,
        description=(
            "Keyword query. Optional: a seller-scoped search works with no query at "
            "all."
        ),
    )

    seller: Optional[str] = Field(
        default=None,
        description=(
            "Scope the search to one seller. Works with no query, which is the only "
            "paginated way to list a seller's whole catalogue."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number, 1-based."
        ),
    )

    sort_by: Optional[
        Literal[
            "best_match",
            "ending_soonest",
            "newly_listed",
            "price_low",
            "price_high",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: best_match, ending_soonest, "
            "newly_listed, price_low, price_high. Default: best_match."
        ),
    )

    min_price: Optional[float] = Field(
        default=None,
        description=(
            "Minimum price filter."
        ),
    )

    max_price: Optional[float] = Field(
        default=None,
        description=(
            "Maximum price filter."
        ),
    )

    condition: Optional[
        Literal["new", "open_box", "refurbished", "used", "for_parts"]
    ] = Field(
        default=None,
        description=(
            "Item condition. refurbished is eBay's parent condition, not one of its "
            "three graded tiers. Options: new, open_box, refurbished, used, for_parts."
        ),
    )

    buying_format: Optional[Literal["auction", "buy_it_now", "best_offer"]] = Field(
        default=None,
        description=(
            "Listing format filter. Options: auction, buy_it_now, best_offer."
        ),
    )

    free_shipping: Optional[bool] = Field(
        default=None,
        description=(
            "Only return listings with free shipping."
        ),
    )

    sold: Optional[bool] = Field(
        default=None,
        description=(
            "Search completed listings that actually SOLD -- the price-research view. "
            "eBay publishes no headline count there, so total_results comes back null."
        ),
    )

    category_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric eBay category id. A non-numeric value returns the UNFILTERED set "
            "under a 200."
        ),
    )

    per_page: Optional[Literal[60, 120, 240]] = Field(
        default=None,
        description=(
            "Listings per page. eBay accepts only 60, 120 or 240 and silently falls "
            "back to 60 for anything else. Options: 60, 120, 240. Default: 60."
        ),
    )


class ScavioEbaySearch(BaseTool):  # type: ignore[override]
    """eBay: Search live or SOLD eBay listings: price, condition, bids, shipping,
    seller, feedback.

    Costs 1 credit per call.

    Pagination: page; per_page accepts ONLY 60, 120 or 240 (silent fallback to 60).

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioEbaySearch

            tool = ScavioEbaySearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "airpods pro", "sold": True})
    """

    name: str = "scavio_ebay_search"
    description: str = (
        "eBay: Search live or SOLD eBay listings: price, condition, bids, shipping, "
        "seller, feedback. Pagination: page; per_page accepts ONLY 60, 120 or 240 "
        "(silent fallback to 60). Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioEbaySearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioEbaySearchAPIWrapper = Field(
        default_factory=ScavioEbaySearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioEbaySearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: Optional[str] = None,
        seller: Optional[str] = None,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal[
                "best_match",
                "ending_soonest",
                "newly_listed",
                "price_low",
                "price_high",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        condition: Optional[
            Literal["new", "open_box", "refurbished", "used", "for_parts"]
        ] = None,
        buying_format: Optional[Literal["auction", "buy_it_now", "best_offer"]] = None,
        free_shipping: Optional[bool] = None,
        sold: Optional[bool] = None,
        category_id: Optional[str] = None,
        per_page: Optional[Literal[60, 120, 240]] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/ebay/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                seller=seller,
                page=page,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
                condition=condition,
                buying_format=buying_format,
                free_shipping=free_shipping,
                sold=sold,
                category_id=category_id,
                per_page=per_page,
            )
            return self._process_response(
                raw, _first_identifier(query, seller, category_id)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: Optional[str] = None,
        seller: Optional[str] = None,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal[
                "best_match",
                "ending_soonest",
                "newly_listed",
                "price_low",
                "price_high",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        condition: Optional[
            Literal["new", "open_box", "refurbished", "used", "for_parts"]
        ] = None,
        buying_format: Optional[Literal["auction", "buy_it_now", "best_offer"]] = None,
        free_shipping: Optional[bool] = None,
        sold: Optional[bool] = None,
        category_id: Optional[str] = None,
        per_page: Optional[Literal[60, 120, 240]] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/ebay/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                seller=seller,
                page=page,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
                condition=condition,
                buying_format=buying_format,
                free_shipping=free_shipping,
                sold=sold,
                category_id=category_id,
                per_page=per_page,
            )
            return self._process_response(
                raw, _first_identifier(query, seller, category_id)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No eBay results found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioEbayProduct
# --------------------------------------------------------------------------


class ScavioEbayProductInput(BaseModel):
    """Input schema for the ScavioEbayProduct tool."""

    model_config = ConfigDict(extra="allow")

    item_id: str = Field(
        description=(
            "eBay item number or a full ebay.com/itm/... URL. Tracking parameters are "
            "discarded."
        ),
    )


class ScavioEbayProduct(BaseTool):  # type: ignore[override]
    """eBay: One eBay listing in full: price, condition, images, item specifics,
    shipping, returns, auction state, seller.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioEbayProduct

            tool = ScavioEbayProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"item_id": "126544332211"})
    """

    name: str = "scavio_ebay_product"
    description: str = (
        "eBay: One eBay listing in full: price, condition, images, item specifics, "
        "shipping, returns, auction state, seller. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioEbayProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioEbayProductAPIWrapper = Field(
        default_factory=ScavioEbayProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioEbayProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        item_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/ebay/product (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                item_id=item_id,
            )
            return self._process_response(raw, _first_identifier(item_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        item_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/ebay/product (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                item_id=item_id,
            )
            return self._process_response(raw, _first_identifier(item_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No eBay product found for '{identifier}'. Verify the identifiers you "
                "passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioEbaySeller
# --------------------------------------------------------------------------


class ScavioEbaySellerInput(BaseModel):
    """Input schema for the ScavioEbaySeller tool."""

    model_config = ConfigDict(extra="allow")

    seller: str = Field(
        description=(
            "eBay username as it appears in ebay.com/usr/<name>."
        ),
    )


class ScavioEbaySeller(BaseTool):  # type: ignore[override]
    """eBay: eBay seller profile card: store name, feedback score and %, items sold,
    followers, location, categories.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioEbaySeller

            tool = ScavioEbaySeller()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"seller": "musicmagpie"})
    """

    name: str = "scavio_ebay_seller"
    description: str = (
        "eBay: eBay seller profile card: store name, feedback score and %, items sold, "
        "followers, location, categories. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioEbaySellerInput
    handle_tool_error: bool = True

    api_wrapper: ScavioEbaySellerAPIWrapper = Field(
        default_factory=ScavioEbaySellerAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioEbaySellerAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        seller: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/ebay/seller (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                seller=seller,
            )
            return self._process_response(raw, _first_identifier(seller))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        seller: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/ebay/seller (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                seller=seller,
            )
            return self._process_response(raw, _first_identifier(seller))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        if not raw.get("data"):
            raise ToolException(
                f"No eBay seller found for '{identifier}'. Verify the identifiers you "
                "passed."
            )
        return raw
