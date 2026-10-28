"""Scavio HEB tools for LangChain agents.

2 credits per call. HEB is a Texas-based grocery chain.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import (
    ScavioHebProductAPIWrapper,
    ScavioHebSearchAPIWrapper,
    ScavioHebStoresAPIWrapper,
)

logger = logging.getLogger(__name__)


def _forward_api_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    api_kwargs: dict[str, Any] = {}
    for key in ("scavio_api_key", "api_base_url", "max_requests_per_second"):
        if key in kwargs:
            api_kwargs[key] = kwargs.pop(key)
    return api_kwargs


def _first_identifier(*values: Any) -> str:
    for value in values:
        if value not in (None, "", [], {}):
            return str(value)
    return "this request"


# --------------------------------------------------------------------------
# ScavioHebSearch
# --------------------------------------------------------------------------


class ScavioHebSearchInput(BaseModel):
    """Input schema for the ScavioHebSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Search keywords, e.g. 'organic milk'.",
    )
    page: Optional[int] = Field(
        default=None,
        description="Page, 1-based.",
    )
    page_size: Optional[int] = Field(
        default=None,
        description="Results per page, 1-60. Defaults to 60.",
    )
    store_id: Optional[str] = Field(
        default=None,
        description=(
            "HEB store number from ScavioHebStores (store_number), e.g. '441'. "
            "Prices, stock, aisle location and assortment come from that store; "
            "the store used is echoed in data.store {store_id, name}. US only. "
            "Omit for HEB's default store."
        ),
    )


class ScavioHebSearch(BaseTool):  # type: ignore[override]
    """HEB: Search grocery products: prices (cents), UPC, stock, EBT/SNAP, aisle.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioHebSearch

            tool = ScavioHebSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "organic milk"})
    """

    name: str = "scavio_heb_search"
    description: str = (
        "HEB: Search grocery products by keyword. Returns price (cents), list price, "
        "sale status, unit price, size, UPC, stock, EBT/SNAP eligibility, category, "
        "aisle location and product URL. Pass store_id (store_number from "
        "scavio_heb_stores) for one store's prices, stock and assortment. "
        "Costs 2 credits."
    )
    args_schema: Type[BaseModel] = ScavioHebSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioHebSearchAPIWrapper = Field(
        default_factory=ScavioHebSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioHebSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                query=query, page=page, page_size=page_size, store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query, page=page, page_size=page_size, store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        if not raw.get("data"):
            raise ToolException(
                f"No HEB results found for '{identifier}'. Try broadening the query."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioHebProduct
# --------------------------------------------------------------------------


class ScavioHebProductInput(BaseModel):
    """Input schema for the ScavioHebProduct tool."""

    model_config = ConfigDict(extra="allow")

    product_url: Optional[str] = Field(
        default=None,
        description=(
            "Full or relative product URL from heb.com, e.g. "
            "'/product-detail/central-market-organics-whole-milk-1-gal/931316'. Preferred."
        ),
    )
    product_id: Optional[str] = Field(
        default=None,
        description=(
            "HEB product id (numeric id at the end of a product URL). Needs slug, "
            "unless store_id is set (then product_id alone works)."
        ),
    )
    slug: Optional[str] = Field(
        default=None,
        description=(
            "Product slug (readable part of the URL). Required with product_id "
            "unless store_id is set."
        ),
    )
    store_id: Optional[str] = Field(
        default=None,
        description=(
            "HEB store number from ScavioHebStores (store_number), e.g. '441'. "
            "Prices, stock, aisle location and assortment come from that store; "
            "the store used is echoed in data.store {store_id, name}. US only. "
            "Omit for HEB's default store."
        ),
    )


class ScavioHebProduct(BaseTool):  # type: ignore[override]
    """HEB: Full product detail: nutrition, ingredients, images, coupons, EBT/SNAP.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioHebProduct

            tool = ScavioHebProduct()
    """

    name: str = "scavio_heb_product"
    description: str = (
        "HEB: Full product detail: nutrition facts, ingredients, description, images, "
        "similar products, coupons, EBT/SNAP eligibility, breadcrumbs, aisle. "
        "Pass product_url or slug + product_id; with store_id (store_number from "
        "scavio_heb_stores) product_id alone works and price, stock and aisle are for "
        "that store. Costs 2 credits."
    )
    args_schema: Type[BaseModel] = ScavioHebProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioHebProductAPIWrapper = Field(
        default_factory=ScavioHebProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioHebProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        slug: Optional[str] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                product_url=product_url, product_id=product_id, slug=slug,
                store_id=store_id,
            )
            return self._process_response(
                raw, _first_identifier(product_url, product_id)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        slug: Optional[str] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_url=product_url, product_id=product_id, slug=slug,
                store_id=store_id,
            )
            return self._process_response(
                raw, _first_identifier(product_url, product_id)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        if not raw.get("data"):
            raise ToolException(
                f"No HEB product found for '{identifier}'."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioHebStores
# --------------------------------------------------------------------------


class ScavioHebStoresInput(BaseModel):
    """Input schema for the ScavioHebStores tool."""

    model_config = ConfigDict(extra="allow")

    zip: Optional[str] = Field(
        default=None,
        description="5-digit US ZIP code, e.g. '78701'.",
    )
    city: Optional[str] = Field(
        default=None,
        description="City name, e.g. 'Austin'. Pass zip or city.",
    )


class ScavioHebStores(BaseTool):  # type: ignore[override]
    """HEB: Find stores near a US ZIP or city: address, hours, pharmacy, fulfillment.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioHebStores

            tool = ScavioHebStores()
    """

    name: str = "scavio_heb_stores"
    description: str = (
        "HEB: Find stores near a US ZIP code or city. Returns store number, address, "
        "phone, hours, pharmacy availability, fulfillment channels (curbside, delivery, "
        "in-store), distance and store URL. Costs 2 credits."
    )
    args_schema: Type[BaseModel] = ScavioHebStoresInput
    handle_tool_error: bool = True

    api_wrapper: ScavioHebStoresAPIWrapper = Field(
        default_factory=ScavioHebStoresAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioHebStoresAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        zip: Optional[str] = None,
        city: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(zip=zip, city=city)
            return self._process_response(raw, _first_identifier(zip, city))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        zip: Optional[str] = None,
        city: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(zip=zip, city=city)
            return self._process_response(raw, _first_identifier(zip, city))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        if not raw.get("data"):
            raise ToolException(
                f"No HEB stores found for '{identifier}'."
            )
        return raw
