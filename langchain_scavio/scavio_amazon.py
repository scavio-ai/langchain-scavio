"""Scavio Amazon tools for LangChain agents.

The Amazon API moved to a new upstream provider in 2026-07 and the request
surface shrank hard. These params are gone and are NOT accepted here any more:
``sort_by``, ``pages``, ``category_id``, ``merchant_id``, ``language``,
``currency``, ``device``, ``zip_code`` and ``autoselect_variant``.

They are removed rather than kept as silent no-ops. ``sort_by`` in particular
was verified against the marketplace: every sort value returns the identical
unordered result set, so a tool schema advertising it is a promise the model
plans against and the API cannot keep. The API answers with a top-level
``warnings`` array if one is sent anyway.

``country`` (ISO 3166-1 alpha-2, e.g. ``us``, ``gb``, ``de``) is the canonical
marketplace selector. ``domain`` and ``start_page`` are deprecated wire aliases
that the API still accepts; they are declared on the schema so nothing the
endpoint takes is unreachable, but every description points back at ``country``
and ``page``. A native value always wins over its alias.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_scavio._utilities import (
    ScavioAmazonOffersAPIWrapper,
    ScavioAmazonProductAPIWrapper,
    ScavioAmazonSearchAPIWrapper,
)

logger = logging.getLogger(__name__)

_COUNTRY_DESCRIPTION = (
    "Marketplace country code (ISO 3166-1 alpha-2), not a domain. "
    'Supported: "us" (default), "gb" (United Kingdom - note gb, not uk), '
    '"ca", "de", "fr", "es", "it", "jp", "in", "au", "br", "mx", "nl", '
    '"pl", "se", "sg", "ae", "sa", "eg", "cn", "be", "tr". '
    "An unrecognised code silently falls back to us."
)

_ASIN_DESCRIPTION = (
    "Amazon ASIN, the 10-character product code (e.g., 'B08N5WRWNW'). "
    "Use ScavioAmazonSearch first to find ASINs if needed."
)

_DOMAIN_DESCRIPTION = (
    "Deprecated: Amazon domain suffix ('com', 'co.uk'). Prefer country, which "
    "is what the marketplace actually selects on; when both are given country "
    "wins."
)

_START_PAGE_DESCRIPTION = (
    "Deprecated alias of page. Prefer page; when both are given page wins."
)


class ScavioAmazonSearchInput(BaseModel):
    """Input schema for ScavioAmazonSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Product search query (e.g., 'wireless headphones')")

    country: Optional[str] = Field(default=None, description=_COUNTRY_DESCRIPTION)

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number (1-indexed). One page per call, 1 credit each. "
            "Only increase if the user asks for more results or the previous "
            "page did not contain the needed information."
        ),
    )

    domain: Optional[str] = Field(default=None, description=_DOMAIN_DESCRIPTION)

    start_page: Optional[int] = Field(
        default=None, description=_START_PAGE_DESCRIPTION
    )


class ScavioAmazonSearch(BaseTool):  # type: ignore[override]
    """Search Amazon product listings using the Scavio API.

    Returns product listings with titles, prices, ratings, ASINs, badges, and
    delivery estimates. Use for product research, price comparisons, or
    shopping queries.

    There is no sort, category, merchant or price filter: the upstream
    marketplace ignores them. Rank the returned products yourself.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAmazonSearch

            tool = ScavioAmazonSearch(max_results=5)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "wireless headphones", "country": "us"})
    """

    name: str = "scavio_amazon_search"
    description: str = (
        "Search Amazon product listings using the Scavio API. "
        "Returns products with asin, title, url, image, price, currency, "
        "rating, reviews_count, is_sponsored, position, badge, sales_volume "
        "and delivery. Results are NOT sorted and cannot be sorted or filtered "
        "by category, merchant or price - rank them yourself. "
        "reviews_count is Amazon's rounded display value and is approximate "
        "above 1000. Input should be a product search query. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioAmazonSearchInput
    handle_tool_error: bool = True

    # Instantiation-only parameter (not controllable by the LLM).
    max_results: Optional[int] = 5

    api_wrapper: ScavioAmazonSearchAPIWrapper = Field(
        default_factory=ScavioAmazonSearchAPIWrapper  # type: ignore[arg-type]
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
            kwargs["api_wrapper"] = ScavioAmazonSearchAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        country: Optional[str] = None,
        page: Optional[int] = None,
        domain: Optional[str] = None,
        start_page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a synchronous Amazon product search."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                country=country,
                page=page,
                domain=domain,
                start_page=start_page,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        country: Optional[str] = None,
        page: Optional[int] = None,
        domain: Optional[str] = None,
        start_page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an asynchronous Amazon product search."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                country=country,
                page=page,
                domain=domain,
                start_page=start_page,
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
                "Try broadening the query or changing the country."
            )
        return raw


class _AsinAliasMixin(BaseModel):
    """Fill ``asin`` from the deprecated ``query`` spelling.

    The wire field on /amazon/product and /amazon/offers is called ``query``
    even though it only ever carries an ASIN. ``asin`` is the name the rest of
    Scavio uses, so it is the required, model-facing one here; ``query`` stays
    accepted so pre-3.4 callers keep working.
    """

    @model_validator(mode="before")
    @classmethod
    def _asin_from_query(cls, values: Any) -> Any:
        if isinstance(values, dict) and not values.get("asin"):
            if values.get("query"):
                values = {**values, "asin": values["query"]}
        return values


class ScavioAmazonProductInput(_AsinAliasMixin):
    """Input schema for ScavioAmazonProduct tool."""

    model_config = ConfigDict(extra="allow")

    asin: str = Field(description=_ASIN_DESCRIPTION)

    country: Optional[str] = Field(default=None, description=_COUNTRY_DESCRIPTION)

    domain: Optional[str] = Field(default=None, description=_DOMAIN_DESCRIPTION)

    query: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated spelling of asin, kept for backwards compatibility. "
            "Prefer asin; when both are given asin wins."
        ),
    )


class ScavioAmazonProduct(BaseTool):  # type: ignore[override]
    """Fetch full details for a specific Amazon product by ASIN.

    Returns title, brand, description, features, price, list price, rating,
    availability, images, videos, best-seller ranks and specifications. Use
    after ScavioAmazonSearch to get detailed product info.

    Takes the ASIN as ``asin``. The wire field is called ``query``, and the
    pre-3.4 ``query`` argument still works and fills ``asin`` when it is the
    only one given.

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

            result = tool.invoke({"asin": "B08N5WRWNW"})
    """

    name: str = "scavio_amazon_product"
    description: str = (
        "Fetch full details for a specific Amazon product by ASIN. "
        "Returns title, brand, description, features, price, list_price, "
        "rating, reviews_count, availability, images, best_sellers_rank and "
        "specifications. `price` is the buy-box price only - use "
        "scavio_amazon_offers for competing sellers. `reviews` is review "
        "metadata with no review text. "
        "Use ScavioAmazonSearch first to find the ASIN. "
        "Pass it as asin; country selects the marketplace. "
        "Costs 1 credit per call."
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
        if "max_requests_per_second" in kwargs:
            api_wrapper_kwargs["max_requests_per_second"] = kwargs.pop(
                "max_requests_per_second"
            )
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAmazonProductAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        asin: Optional[str] = None,
        country: Optional[str] = None,
        domain: Optional[str] = None,
        query: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Amazon product details synchronously."""
        identifier = asin or query
        try:
            raw = self.api_wrapper.raw_results(
                query=identifier,
                country=country,
                domain=domain,
            )
            return self._process_response(raw, identifier or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        asin: Optional[str] = None,
        country: Optional[str] = None,
        domain: Optional[str] = None,
        query: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Amazon product details asynchronously."""
        identifier = asin or query
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=identifier,
                country=country,
                domain=domain,
            )
            return self._process_response(raw, identifier or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], asin: str) -> dict[str, Any]:
        """Raise ToolException if no product data returned."""
        if not raw.get("data"):
            raise ToolException(
                f"No Amazon product found for ASIN '{asin}'. "
                "Verify the ASIN is correct and the country matches the "
                "marketplace it is listed on."
            )
        return raw


class ScavioAmazonOffersInput(_AsinAliasMixin):
    """Input schema for ScavioAmazonOffers tool."""

    model_config = ConfigDict(extra="allow")

    asin: str = Field(description=_ASIN_DESCRIPTION)

    country: Optional[str] = Field(default=None, description=_COUNTRY_DESCRIPTION)

    domain: Optional[str] = Field(default=None, description=_DOMAIN_DESCRIPTION)

    query: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated spelling of asin, kept for backwards compatibility. "
            "Prefer asin; when both are given asin wins."
        ),
    )


class ScavioAmazonOffers(BaseTool):  # type: ignore[override]
    """List every seller offer for an Amazon ASIN.

    Returns each seller's price, condition, shipping, and which offer holds the
    buy box. Use for price comparison, reseller research, and buy-box
    monitoring.

    Takes the ASIN as ``asin``. The wire field is called ``query``, and the
    pre-3.4 ``query`` argument still works and fills ``asin`` when it is the
    only one given.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAmazonOffers

            tool = ScavioAmazonOffers()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"asin": "B08N5WRWNW"})
    """

    name: str = "scavio_amazon_offers"
    description: str = (
        "List every seller offer for an Amazon ASIN. Returns condition, "
        "seller_name, ships_from, is_buy_box_winner, is_prime, price, "
        "list_price, shipping_price and discount_percentage per offer. "
        "`price` excludes shipping_price, and the buy-box winner is not always "
        "the cheapest offer. Page 1 only. An ASIN sold only by Amazon returns "
        "an empty offers list plus a `note` - that is a normal answer, not an "
        "error. Pass the ASIN as asin; country selects the marketplace. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioAmazonOffersInput
    handle_tool_error: bool = True

    api_wrapper: ScavioAmazonOffersAPIWrapper = Field(
        default_factory=ScavioAmazonOffersAPIWrapper  # type: ignore[arg-type]
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
            kwargs["api_wrapper"] = ScavioAmazonOffersAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        asin: Optional[str] = None,
        country: Optional[str] = None,
        domain: Optional[str] = None,
        query: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the offer listing synchronously."""
        identifier = asin or query
        try:
            raw = self.api_wrapper.raw_results(
                query=identifier,
                country=country,
                domain=domain,
            )
            return self._process_response(raw, identifier or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        asin: Optional[str] = None,
        country: Optional[str] = None,
        domain: Optional[str] = None,
        query: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the offer listing asynchronously."""
        identifier = asin or query
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=identifier,
                country=country,
                domain=domain,
            )
            return self._process_response(raw, identifier or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], asin: str) -> dict[str, Any]:
        """Raise ToolException only when the listing itself is missing.

        An empty ``offers`` list is a legitimate answer (Amazon is the sole
        seller), so it is returned as-is rather than raised.
        """
        if not raw.get("data"):
            raise ToolException(
                f"No Amazon offer listing found for ASIN '{asin}'. "
                "Verify the ASIN is correct and the country matches the "
                "marketplace it is listed on."
            )
        return raw
