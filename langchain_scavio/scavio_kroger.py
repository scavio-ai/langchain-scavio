"""Scavio Kroger tools for LangChain agents.

2 credits per call. Kroger operates Kroger, Ralphs, Fred Meyer, Harris Teeter,
King Soopers, Fry's and other banners.
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
    ScavioKrogerProductAPIWrapper,
    ScavioKrogerSearchAPIWrapper,
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
# ScavioKrogerSearch
# --------------------------------------------------------------------------


class ScavioKrogerSearchInput(BaseModel):
    """Input schema for the ScavioKrogerSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Search keywords, e.g. 'organic milk'.",
    )
    page: Optional[int] = Field(
        default=None,
        description="Page, 1-based.",
    )


class ScavioKrogerSearch(BaseTool):  # type: ignore[override]
    """Kroger: Search grocery products: prices (cents), promo, UPC, stock, SNAP, dietary.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKrogerSearch

            tool = ScavioKrogerSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "organic milk"})
    """

    name: str = "scavio_kroger_search"
    description: str = (
        "Kroger: Search grocery products by keyword. Kroger operates Kroger, Ralphs, "
        "Fred Meyer, Harris Teeter, King Soopers, Fry's and other banners. Returns "
        "price (cents), promo price, n-for deals, unit price, size, UPC, stock, SNAP "
        "eligibility, dietary info, rating and category. Costs 2 credits."
    )
    args_schema: Type[BaseModel] = ScavioKrogerSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKrogerSearchAPIWrapper = Field(
        default_factory=ScavioKrogerSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKrogerSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(query=query, page=page)
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(query=query, page=page)
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
                f"No Kroger results found for '{identifier}'. Try broadening the query."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioKrogerProduct
# --------------------------------------------------------------------------


class ScavioKrogerProductInput(BaseModel):
    """Input schema for the ScavioKrogerProduct tool."""

    model_config = ConfigDict(extra="allow")

    product_url: Optional[str] = Field(
        default=None,
        description=(
            "Full or relative product URL from kroger.com, e.g. "
            "'/p/kroger-2-reduced-fat-milk-gallon/0001111041700'. Preferred."
        ),
    )
    upc: Optional[str] = Field(
        default=None,
        description="Kroger product UPC (from search or product URLs: /p/{slug}/{upc}).",
    )
    slug: Optional[str] = Field(
        default=None,
        description="Product slug for readable URL. Optional with upc.",
    )


class ScavioKrogerProduct(BaseTool):  # type: ignore[override]
    """Kroger: Full product detail: nutrition, allergens, dietary flags, rating, stock.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioKrogerProduct

            tool = ScavioKrogerProduct()
    """

    name: str = "scavio_kroger_product"
    description: str = (
        "Kroger: Full product detail by URL or UPC: nutrition facts (allergens, "
        "ingredients, nutrients with daily values), description, images, rating "
        "distribution, store aisle location, stock level, dietary flags (organic, "
        "non-GMO, gluten-free), SNAP eligibility, dimensions and country of origin. "
        "Pass product_url or upc. Costs 2 credits."
    )
    args_schema: Type[BaseModel] = ScavioKrogerProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioKrogerProductAPIWrapper = Field(
        default_factory=ScavioKrogerProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioKrogerProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_url: Optional[str] = None,
        upc: Optional[str] = None,
        slug: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                product_url=product_url, upc=upc, slug=slug,
            )
            return self._process_response(
                raw, _first_identifier(product_url, upc)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_url: Optional[str] = None,
        upc: Optional[str] = None,
        slug: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_url=product_url, upc=upc, slug=slug,
            )
            return self._process_response(
                raw, _first_identifier(product_url, upc)
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
                f"No Kroger product found for '{identifier}'."
            )
        return raw
