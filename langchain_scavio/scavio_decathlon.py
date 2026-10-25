"""Scavio Decathlon tools for LangChain agents.

2 credits per call. Supports 7 country domains: FR, ES, IT, IN, UK, PT and
NET (international).
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
    ScavioDecathlonProductAPIWrapper,
    ScavioDecathlonSearchAPIWrapper,
)

logger = logging.getLogger(__name__)

DecathlonCountry = Literal["fr", "es", "it", "in", "uk", "pt", "net"]


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
# ScavioDecathlonSearch
# --------------------------------------------------------------------------


class ScavioDecathlonSearchInput(BaseModel):
    """Input schema for the ScavioDecathlonSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Search keywords, e.g. 'running shoes'.",
    )
    country: DecathlonCountry = Field(
        description="Country domain: fr, es, it, in, uk, pt or net.",
    )
    page: Optional[int] = Field(
        default=None,
        description="Page, 1-based.",
    )


class ScavioDecathlonSearch(BaseTool):  # type: ignore[override]
    """Decathlon: Search products across 7 country domains.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioDecathlonSearch

            tool = ScavioDecathlonSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "running shoes", "country": "fr"})
    """

    name: str = "scavio_decathlon_search"
    description: str = (
        "Decathlon: Search products by keyword across 7 country domains "
        "(FR, ES, IT, IN, UK, PT, NET). Returns product name, brand, "
        "price (cents), currency, availability, rating, review count and "
        "image. Costs 2 credits."
    )
    args_schema: Type[BaseModel] = ScavioDecathlonSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioDecathlonSearchAPIWrapper = Field(
        default_factory=ScavioDecathlonSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioDecathlonSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        country: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(query=query, country=country, page=page)
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        country: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(query=query, country=country, page=page)
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
                f"No Decathlon results found for '{identifier}'. Try broadening the query."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioDecathlonProduct
# --------------------------------------------------------------------------


class ScavioDecathlonProductInput(BaseModel):
    """Input schema for the ScavioDecathlonProduct tool."""

    model_config = ConfigDict(extra="allow")

    country: DecathlonCountry = Field(
        description="Country domain: fr, es, it, in, uk, pt or net.",
    )
    product_url: Optional[str] = Field(
        default=None,
        description=(
            "Full or relative product URL from any Decathlon domain, e.g. "
            "'/p/mens-running-shoes/123456/...'. Preferred."
        ),
    )
    product_id: Optional[str] = Field(
        default=None,
        description="Decathlon numeric product ID (from the /p/{slug}/{id}/... URL pattern).",
    )


class ScavioDecathlonProduct(BaseTool):  # type: ignore[override]
    """Decathlon: Full product detail by URL or product ID.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioDecathlonProduct

            tool = ScavioDecathlonProduct()
    """

    name: str = "scavio_decathlon_product"
    description: str = (
        "Decathlon: Full product detail by URL or product ID: description, "
        "images, SKU, GTIN, color, price (cents), list price, availability, "
        "offers/variants, rating, review count, category and breadcrumbs. "
        "Costs 2 credits."
    )
    args_schema: Type[BaseModel] = ScavioDecathlonProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioDecathlonProductAPIWrapper = Field(
        default_factory=ScavioDecathlonProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioDecathlonProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        country: str,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(
                country=country, product_url=product_url, product_id=product_id,
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
        country: str,
        product_url: Optional[str] = None,
        product_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                country=country, product_url=product_url, product_id=product_id,
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
                f"No Decathlon product found for '{identifier}'."
            )
        return raw
