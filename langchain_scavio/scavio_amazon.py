"""ScavioAmazonSearch and ScavioAmazonProduct tools for LangChain agents."""

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
    ScavioAmazonProductAPIWrapper,
    ScavioAmazonSearchAPIWrapper,
)

logger = logging.getLogger(__name__)

_SEARCH_INIT_ONLY_PARAMS = frozenset({"max_results", "pages", "autoselect_variant"})


class ScavioAmazonSearchInput(BaseModel):
    """Input schema for ScavioAmazonSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Product search query (e.g., 'wireless headphones')")

    domain: Optional[str] = Field(
        default=None,
        description=(
            'Amazon domain to search (e.g., "com", "co.uk", "de", "fr", "co.jp", '
            '"ca", "it", "es", "in", "com.au"). Default is "com".'
        ),
    )

    sort_by: Optional[
        Literal[
            "most_recent",
            "price_low_to_high",
            "price_high_to_low",
            "featured",
            "average_review",
            "bestsellers",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for results. Options: most_recent, price_low_to_high, "
            "price_high_to_low, featured, average_review, bestsellers."
        ),
    )

    start_page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number (1-indexed). "
            "Only increase if the user asks for more results or the previous page "
            "did not contain the needed information."
        ),
    )

    category_id: Optional[str] = Field(
        default=None,
        description="Amazon category ID to filter results within a department.",
    )

    merchant_id: Optional[str] = Field(
        default=None,
        description="Amazon merchant/seller ID to filter results to a specific seller.",
    )

    country: Optional[str] = Field(
        default=None,
        description=(
            "Country code for localized pricing and availability "
            '(e.g., "US", "GB", "DE").'
        ),
    )

    language: Optional[str] = Field(
        default=None,
        description='ISO 639-1 language code for results (e.g., "en", "de", "fr").',
    )

    currency: Optional[str] = Field(
        default=None,
        description='ISO 4217 currency code for prices (e.g., "USD", "EUR", "GBP").',
    )

    device: Optional[Literal["desktop", "mobile", "tablet"]] = Field(
        default=None,
        description="Device type. Options: desktop (default), mobile, tablet.",
    )

    zip_code: Optional[str] = Field(
        default=None,
        description="Postal/ZIP code for local pricing and delivery availability.",
    )


class ScavioAmazonSearch(BaseTool):  # type: ignore[override]
    """Search Amazon product listings using the Scavio API.

    Returns product listings with names, prices, ratings, ASINs, and availability.
    Use for product research, price comparisons, or shopping queries.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAmazonSearch

            tool = ScavioAmazonSearch(
                max_results=5,
                domain="com",
            )

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "wireless headphones"})
    """

    name: str = "scavio_amazon_search"
    description: str = (
        "Search Amazon product listings using the Scavio API. "
        "Returns product names, prices, ratings, ASINs, and availability. "
        "Use for product research, price comparisons, or shopping queries. "
        "Input should be a product search query."
    )
    args_schema: Type[BaseModel] = ScavioAmazonSearchInput
    handle_tool_error: bool = True

    # Instantiation-only parameters (not controllable by the LLM).
    max_results: Optional[int] = 5
    pages: Optional[int] = None
    autoselect_variant: Optional[bool] = None

    api_wrapper: ScavioAmazonSearchAPIWrapper = Field(
        default_factory=ScavioAmazonSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_wrapper_kwargs: dict[str, Any] = {}
        if "scavio_api_key" in kwargs:
            api_wrapper_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
        if "api_base_url" in kwargs:
            api_wrapper_kwargs["api_base_url"] = kwargs.pop("api_base_url")
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAmazonSearchAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        domain: Optional[str] = None,
        sort_by: Optional[str] = None,
        start_page: Optional[int] = None,
        category_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        currency: Optional[str] = None,
        device: Optional[str] = None,
        zip_code: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a synchronous Amazon product search."""
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
                sort_by=sort_by,
                start_page=start_page,
                pages=self.pages,
                category_id=category_id,
                merchant_id=merchant_id,
                country=country,
                language=language,
                currency=currency,
                device=device,
                zip_code=zip_code,
                autoselect_variant=self.autoselect_variant,
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
        sort_by: Optional[str] = None,
        start_page: Optional[int] = None,
        category_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        currency: Optional[str] = None,
        device: Optional[str] = None,
        zip_code: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an asynchronous Amazon product search."""
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
                sort_by=sort_by,
                start_page=start_page,
                pages=self.pages,
                category_id=category_id,
                merchant_id=merchant_id,
                country=country,
                language=language,
                currency=currency,
                device=device,
                zip_code=zip_code,
                autoselect_variant=self.autoselect_variant,
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
                f"No Amazon results found for '{query}'. "
                "Try broadening the query, changing the domain, or removing filters."
            )
        return raw


class ScavioAmazonProductInput(BaseModel):
    """Input schema for ScavioAmazonProduct tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Amazon product ASIN code (e.g., 'B08N5WRWNW'). "
            "Use ScavioAmazonSearch first to find ASINs if needed."
        )
    )

    domain: Optional[str] = Field(
        default=None,
        description='Amazon domain (e.g., "com", "co.uk", "de"). Default is "com".',
    )

    country: Optional[str] = Field(
        default=None,
        description='Country code for localized data (e.g., "US", "GB").',
    )

    language: Optional[str] = Field(
        default=None,
        description='ISO 639-1 language code (e.g., "en", "de").',
    )

    currency: Optional[str] = Field(
        default=None,
        description='ISO 4217 currency code (e.g., "USD", "EUR").',
    )

    device: Optional[Literal["desktop", "mobile", "tablet"]] = Field(
        default=None,
        description="Device type. Options: desktop (default), mobile, tablet.",
    )

    zip_code: Optional[str] = Field(
        default=None,
        description="Postal/ZIP code for local pricing.",
    )

    autoselect_variant: Optional[bool] = Field(
        default=None,
        description="Automatically select the default product variant.",
    )


class ScavioAmazonProduct(BaseTool):  # type: ignore[override]
    """Fetch full details for a specific Amazon product by ASIN.

    Returns product name, description, price, rating, reviews, images,
    and availability. Use after ScavioAmazonSearch to get detailed product info.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAmazonProduct

            tool = ScavioAmazonProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "B08N5WRWNW"})
    """

    name: str = "scavio_amazon_product"
    description: str = (
        "Fetch full details for a specific Amazon product by ASIN. "
        "Returns name, description, price, rating, reviews, and availability. "
        "Use ScavioAmazonSearch first to find the ASIN. "
        "Input should be an Amazon ASIN code."
    )
    args_schema: Type[BaseModel] = ScavioAmazonProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioAmazonProductAPIWrapper = Field(
        default_factory=ScavioAmazonProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_wrapper_kwargs: dict[str, Any] = {}
        if "scavio_api_key" in kwargs:
            api_wrapper_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
        if "api_base_url" in kwargs:
            api_wrapper_kwargs["api_base_url"] = kwargs.pop("api_base_url")
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAmazonProductAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        domain: Optional[str] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        currency: Optional[str] = None,
        device: Optional[str] = None,
        zip_code: Optional[str] = None,
        autoselect_variant: Optional[bool] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Amazon product details synchronously."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                domain=domain,
                country=country,
                language=language,
                currency=currency,
                device=device,
                zip_code=zip_code,
                autoselect_variant=autoselect_variant,
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
        country: Optional[str] = None,
        language: Optional[str] = None,
        currency: Optional[str] = None,
        device: Optional[str] = None,
        zip_code: Optional[str] = None,
        autoselect_variant: Optional[bool] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Amazon product details asynchronously."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                domain=domain,
                country=country,
                language=language,
                currency=currency,
                device=device,
                zip_code=zip_code,
                autoselect_variant=autoselect_variant,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], asin: str) -> dict[str, Any]:
        """Raise ToolException if no product data returned."""
        if not raw.get("data"):
            raise ToolException(
                f"No Amazon product found for ASIN '{asin}'. "
                "Verify the ASIN is correct and the domain matches the marketplace."
            )
        return raw
