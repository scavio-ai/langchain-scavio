"""ScavioWalmartSearch and ScavioWalmartProduct tools for LangChain agents."""

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
    ScavioWalmartProductAPIWrapper,
    ScavioWalmartSearchAPIWrapper,
)

logger = logging.getLogger(__name__)

_SEARCH_INIT_ONLY_PARAMS = frozenset({"max_results"})


class ScavioWalmartSearchInput(BaseModel):
    """Input schema for ScavioWalmartSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Product search query (e.g., 'air fryer')")

    domain: Optional[str] = Field(
        default=None,
        description="Walmart domain to search. Default is the primary US domain.",
    )

    device: Optional[Literal["desktop", "mobile", "tablet"]] = Field(
        default=None,
        description="Device type. Options: desktop (default), mobile, tablet.",
    )

    sort_by: Optional[
        Literal["best_match", "price_low", "price_high", "best_seller"]
    ] = Field(
        default=None,
        description=(
            "Sort order for results. Options: best_match (default), price_low, "
            "price_high, best_seller."
        ),
    )

    start_page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number (1-indexed). "
            "Only increase if the user asks for more results."
        ),
    )

    min_price: Optional[int] = Field(
        default=None,
        description="Minimum price filter in cents (e.g., 1000 = $10.00).",
    )

    max_price: Optional[int] = Field(
        default=None,
        description="Maximum price filter in cents (e.g., 5000 = $50.00).",
    )

    fulfillment_speed: Optional[
        Literal["today", "tomorrow", "2_days", "anytime"]
    ] = Field(
        default=None,
        description=(
            "Delivery speed filter. Options: today, tomorrow, 2_days, anytime."
        ),
    )

    fulfillment_type: Optional[str] = Field(
        default=None,
        description='Fulfillment method filter (e.g., "in_store" for pickup).',
    )

    delivery_zip: Optional[str] = Field(
        default=None,
        description="ZIP code for delivery availability and local pricing.",
    )

    store_id: Optional[str] = Field(
        default=None,
        description="Walmart store ID to filter in-store availability.",
    )


class ScavioWalmartSearch(BaseTool):  # type: ignore[override]
    """Search Walmart product listings using the Scavio API.

    Returns product listings with names, prices, ratings, product IDs,
    and fulfillment options. Use for product research or shopping queries.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartSearch

            tool = ScavioWalmartSearch(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "air fryer", "max_price": 5000})
    """

    name: str = "scavio_walmart_search"
    description: str = (
        "Search Walmart product listings using the Scavio API. "
        "Returns product names, prices, ratings, and fulfillment options. "
        "Supports price range filters and delivery speed filters. "
        "Input should be a product search query."
    )
    args_schema: Type[BaseModel] = ScavioWalmartSearchInput
    handle_tool_error: bool = True

    # Instantiation-only parameters (not controllable by the LLM).
    max_results: Optional[int] = 5

    api_wrapper: ScavioWalmartSearchAPIWrapper = Field(
        default_factory=ScavioWalmartSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_wrapper_kwargs: dict[str, Any] = {}
        if "scavio_api_key" in kwargs:
            api_wrapper_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
        if "api_base_url" in kwargs:
            api_wrapper_kwargs["api_base_url"] = kwargs.pop("api_base_url")
        if "max_requests_per_second" in kwargs:
            api_wrapper_kwargs["max_requests_per_second"] = kwargs.pop(
                "max_requests_per_second"
            )
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartSearchAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        domain: Optional[str] = None,
        device: Optional[str] = None,
        sort_by: Optional[str] = None,
        start_page: Optional[int] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        fulfillment_speed: Optional[str] = None,
        fulfillment_type: Optional[str] = None,
        delivery_zip: Optional[str] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a synchronous Walmart product search."""
        forbidden = _SEARCH_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                domain=domain,
                device=device,
                sort_by=sort_by,
                start_page=start_page,
                min_price=min_price,
                max_price=max_price,
                fulfillment_speed=fulfillment_speed,
                fulfillment_type=fulfillment_type,
                delivery_zip=delivery_zip,
                store_id=store_id,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        domain: Optional[str] = None,
        device: Optional[str] = None,
        sort_by: Optional[str] = None,
        start_page: Optional[int] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        fulfillment_speed: Optional[str] = None,
        fulfillment_type: Optional[str] = None,
        delivery_zip: Optional[str] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an asynchronous Walmart product search."""
        forbidden = _SEARCH_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                domain=domain,
                device=device,
                sort_by=sort_by,
                start_page=start_page,
                min_price=min_price,
                max_price=max_price,
                fulfillment_speed=fulfillment_speed,
                fulfillment_type=fulfillment_type,
                delivery_zip=delivery_zip,
                store_id=store_id,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], query: str) -> dict[str, Any]:
        """Truncate results and raise ToolException if empty."""
        data = raw.get("data") or {}
        products = data.get("products") if isinstance(data, dict) else None
        if self.max_results and products:
            raw["data"]["products"] = products[: self.max_results]
        if not (isinstance(data, dict) and data.get("products")):
            raise ToolException(
                f"No Walmart results found for '{query}'. "
                "Try broadening the query or removing price/fulfillment filters."
            )
        return raw


class ScavioWalmartProductInput(BaseModel):
    """Input schema for ScavioWalmartProduct tool."""

    model_config = ConfigDict(extra="allow")

    product_id: str = Field(
        description=(
            "Walmart product ID. "
            "Use ScavioWalmartSearch first to find product IDs if needed."
        )
    )

    domain: Optional[str] = Field(
        default=None,
        description="Walmart domain. Default is the primary US domain.",
    )

    device: Optional[Literal["desktop", "mobile", "tablet"]] = Field(
        default=None,
        description="Device type. Options: desktop (default), mobile, tablet.",
    )

    delivery_zip: Optional[str] = Field(
        default=None,
        description="ZIP code for delivery availability and local pricing.",
    )

    store_id: Optional[str] = Field(
        default=None,
        description="Walmart store ID for in-store availability.",
    )


class ScavioWalmartProduct(BaseTool):  # type: ignore[override]
    """Fetch full details for a specific Walmart product by product ID.

    Returns product name, description, price, rating, reviews, and availability.
    Use after ScavioWalmartSearch to get detailed product information.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartProduct

            tool = ScavioWalmartProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "123456789"})
    """

    name: str = "scavio_walmart_product"
    description: str = (
        "Fetch full details for a specific Walmart product by product ID. "
        "Returns name, description, price, rating, reviews, and availability. "
        "Use ScavioWalmartSearch first to find product IDs. "
        "Input should be a Walmart product ID."
    )
    args_schema: Type[BaseModel] = ScavioWalmartProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioWalmartProductAPIWrapper = Field(
        default_factory=ScavioWalmartProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_wrapper_kwargs: dict[str, Any] = {}
        if "scavio_api_key" in kwargs:
            api_wrapper_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
        if "api_base_url" in kwargs:
            api_wrapper_kwargs["api_base_url"] = kwargs.pop("api_base_url")
        if "max_requests_per_second" in kwargs:
            api_wrapper_kwargs["max_requests_per_second"] = kwargs.pop(
                "max_requests_per_second"
            )
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartProductAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: str,
        domain: Optional[str] = None,
        device: Optional[str] = None,
        delivery_zip: Optional[str] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Walmart product details synchronously."""
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
                domain=domain,
                device=device,
                delivery_zip=delivery_zip,
                store_id=store_id,
            )
            return self._process_response(raw, product_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_id: str,
        domain: Optional[str] = None,
        device: Optional[str] = None,
        delivery_zip: Optional[str] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Walmart product details asynchronously."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
                domain=domain,
                device=device,
                delivery_zip=delivery_zip,
                store_id=store_id,
            )
            return self._process_response(raw, product_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], product_id: str
    ) -> dict[str, Any]:
        """Raise ToolException if no product data returned."""
        if not raw.get("data"):
            raise ToolException(
                f"No Walmart product found for ID '{product_id}'. "
                "Verify the product ID is correct."
            )
        return raw
