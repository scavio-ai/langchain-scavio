"""Scavio Walmart tools for LangChain agents.

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
    ScavioWalmartCategoryAPIWrapper,
    ScavioWalmartOffersAPIWrapper,
    ScavioWalmartProductAPIWrapper,
    ScavioWalmartReviewsAPIWrapper,
    ScavioWalmartSearchAPIWrapper,
    ScavioWalmartSellerAPIWrapper,
    ScavioWalmartSellerProductsAPIWrapper,
)

logger = logging.getLogger(__name__)

_INIT_ONLY_PARAMS = frozenset({"max_results"})


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
# ScavioWalmartSearch
# --------------------------------------------------------------------------


class ScavioWalmartSearchInput(BaseModel):
    """Input schema for the ScavioWalmartSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Product search query."
        ),
    )

    start_page: Optional[int] = Field(
        default=None,
        description=(
            "Deprecated alias for page."
        ),
    )

    fulfillment_speed: Optional[Literal["today", "tomorrow"]] = Field(
        default=None,
        description=(
            "Delivery-speed filter. 2_days and anytime are deliberately not offered: "
            "2_days leaks 3-4 day items and anytime is a no-op, so omit the parameter "
            "instead. Options: today, tomorrow."
        ),
    )

    fulfillment_type: Optional[Literal["in_store"]] = Field(
        default=None,
        description=(
            "Set to in_store to only return pickup stock. Options: in_store."
        ),
    )

    domain: Optional[Literal["com", "ca", "com.mx"]] = Field(
        default=None,
        description=(
            "Walmart storefront. com and ca cost 1 credit, com.mx costs 2. Options: "
            "com, ca, com.mx. Default: com."
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
            "price_low",
            "price_high",
            "best_seller",
            "rating_high",
            "new",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: best_match, price_low, price_high, "
            "best_seller, rating_high, new. Default: best_match."
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


class ScavioWalmartSearch(BaseTool):  # type: ignore[override]
    """Walmart: Search Walmart and get structured product rows (products[] +
    products_count + location).

    Costs 1 credit on domain com or ca and 2 credits on com.mx.

    Pagination: page (integer >= 1); start_page is a deprecated alias.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartSearch

            tool = ScavioWalmartSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "air fryer"})
    """

    name: str = "scavio_walmart_search"
    description: str = (
        "Walmart: Search Walmart and get structured product rows (products[] + "
        "products_count + location). Pagination: page (integer >= 1); start_page is a "
        "deprecated alias. Costs 1 credit on domain com or ca and 2 credits on com.mx."
    )
    args_schema: Type[BaseModel] = ScavioWalmartSearchInput
    handle_tool_error: bool = True

    # Instantiation-only parameter, not controllable by the LLM.
    max_results: Optional[int] = 5

    api_wrapper: ScavioWalmartSearchAPIWrapper = Field(
        default_factory=ScavioWalmartSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        start_page: Optional[int] = None,
        fulfillment_speed: Optional[Literal["today", "tomorrow"]] = None,
        fulfillment_type: Optional[Literal["in_store"]] = None,
        domain: Optional[Literal["com", "ca", "com.mx"]] = None,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal[
                "best_match",
                "price_low",
                "price_high",
                "best_seller",
                "rating_high",
                "new",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/search (synchronously)."""
        forbidden = _INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                start_page=start_page,
                fulfillment_speed=fulfillment_speed,
                fulfillment_type=fulfillment_type,
                domain=domain,
                page=page,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        start_page: Optional[int] = None,
        fulfillment_speed: Optional[Literal["today", "tomorrow"]] = None,
        fulfillment_type: Optional[Literal["in_store"]] = None,
        domain: Optional[Literal["com", "ca", "com.mx"]] = None,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal[
                "best_match",
                "price_low",
                "price_high",
                "best_seller",
                "rating_high",
                "new",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/search (asynchronously)."""
        forbidden = _INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                start_page=start_page,
                fulfillment_speed=fulfillment_speed,
                fulfillment_type=fulfillment_type,
                domain=domain,
                page=page,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], identifier: str
    ) -> dict[str, Any]:
        """Raise ToolException when the API returned no usable data."""
        data = raw.get("data") or {}
        rows = data.get("products") if isinstance(data, dict) else None
        if self.max_results and rows:
            raw["data"]["products"] = rows[: self.max_results]
        if not rows:
            raise ToolException(
                f"No Walmart results found for '{identifier}'. Try broadening the "
                "query or removing the price and fulfillment filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioWalmartProduct
# --------------------------------------------------------------------------


class ScavioWalmartProductInput(BaseModel):
    """Input schema for the ScavioWalmartProduct tool."""

    model_config = ConfigDict(extra="allow")

    product_id: str = Field(
        description=(
            "Walmart item id (usItemId), e.g. 13544111159."
        ),
    )


class ScavioWalmartProduct(BaseTool):  # type: ignore[override]
    """Walmart: Full Walmart product detail: price, rating, images, specifications,
    availability, seller.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartProduct

            tool = ScavioWalmartProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "13544111159"})
    """

    name: str = "scavio_walmart_product"
    description: str = (
        "Walmart: Full Walmart product detail: price, rating, images, specifications, "
        "availability, seller. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioWalmartProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioWalmartProductAPIWrapper = Field(
        default_factory=ScavioWalmartProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/product (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
            )
            return self._process_response(raw, _first_identifier(product_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/product (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
            )
            return self._process_response(raw, _first_identifier(product_id))
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
                f"No Walmart product found for ID '{identifier}'. Verify the item id "
                "is correct."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioWalmartReviews
# --------------------------------------------------------------------------


class ScavioWalmartReviewsInput(BaseModel):
    """Input schema for the ScavioWalmartReviews tool."""

    model_config = ConfigDict(extra="allow")

    product_id: str = Field(
        description=(
            "Walmart item id (usItemId), e.g. 13544111159."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 10 reviews per page."
        ),
    )

    sort: Optional[
        Literal[
            "relevancy",
            "submission-desc",
            "submission-asc",
            "rating-desc",
            "rating-asc",
            "helpful-desc",
        ]
    ] = Field(
        default=None,
        description=(
            "Review sort order. Options: relevancy, submission-desc, submission-asc, "
            "rating-desc, rating-asc, helpful-desc."
        ),
    )


class ScavioWalmartReviews(BaseTool):  # type: ignore[override]
    """Walmart: Customer reviews with ratings, text, author, date and the rating
    breakdown.

    Costs 1 credit per call.

    Pagination: page (10 reviews per page).

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartReviews

            tool = ScavioWalmartReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "13544111159", "page": 2})
    """

    name: str = "scavio_walmart_reviews"
    description: str = (
        "Walmart: Customer reviews with ratings, text, author, date and the rating "
        "breakdown. Pagination: page (10 reviews per page). Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioWalmartReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioWalmartReviewsAPIWrapper = Field(
        default_factory=ScavioWalmartReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: str,
        page: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevancy",
                "submission-desc",
                "submission-asc",
                "rating-desc",
                "rating-asc",
                "helpful-desc",
            ]
        ] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
                page=page,
                sort=sort,
            )
            return self._process_response(raw, _first_identifier(product_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_id: str,
        page: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevancy",
                "submission-desc",
                "submission-asc",
                "rating-desc",
                "rating-asc",
                "helpful-desc",
            ]
        ] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
                page=page,
                sort=sort,
            )
            return self._process_response(raw, _first_identifier(product_id))
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
                f"No Walmart reviews found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioWalmartCategory
# --------------------------------------------------------------------------


class ScavioWalmartCategoryInput(BaseModel):
    """Input schema for the ScavioWalmartCategory tool."""

    model_config = ConfigDict(extra="allow")

    category_id: str = Field(
        description=(
            "Leaf category id (1095191) or the full underscore path "
            "(3944_133251_1095191)."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Trims the products list after fetching. It does NOT reduce the credit "
            "cost."
        ),
    )

    fulfillment_speed: Optional[Literal["today", "tomorrow"]] = Field(
        default=None,
        description=(
            "Delivery-speed filter. 2_days and anytime are deliberately not offered: "
            "2_days leaks 3-4 day items and anytime is a no-op, so omit the parameter "
            "instead. Options: today, tomorrow."
        ),
    )

    domain: Optional[Literal["com", "ca", "com.mx"]] = Field(
        default=None,
        description=(
            "Walmart storefront. com and ca cost 1 credit, com.mx costs 2. Options: "
            "com, ca, com.mx. Default: com."
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
            "price_low",
            "price_high",
            "best_seller",
            "rating_high",
            "new",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: best_match, price_low, price_high, "
            "best_seller, rating_high, new. Default: best_match."
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


class ScavioWalmartCategory(BaseTool):  # type: ignore[override]
    """Walmart: Products within a Walmart category, same product shape as search.

    Costs 1 credit on domain com or ca and 2 credits on com.mx.

    Pagination: page; `limit` trims after fetching and does NOT reduce cost.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartCategory

            tool = ScavioWalmartCategory()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"category_id": "1095191"})
    """

    name: str = "scavio_walmart_category"
    description: str = (
        "Walmart: Products within a Walmart category, same product shape as search. "
        "Pagination: page; `limit` trims after fetching and does NOT reduce cost. "
        "Costs 1 credit on domain com or ca and 2 credits on com.mx."
    )
    args_schema: Type[BaseModel] = ScavioWalmartCategoryInput
    handle_tool_error: bool = True

    api_wrapper: ScavioWalmartCategoryAPIWrapper = Field(
        default_factory=ScavioWalmartCategoryAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartCategoryAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        category_id: str,
        limit: Optional[int] = None,
        fulfillment_speed: Optional[Literal["today", "tomorrow"]] = None,
        domain: Optional[Literal["com", "ca", "com.mx"]] = None,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal[
                "best_match",
                "price_low",
                "price_high",
                "best_seller",
                "rating_high",
                "new",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/category (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                category_id=category_id,
                limit=limit,
                fulfillment_speed=fulfillment_speed,
                domain=domain,
                page=page,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
            )
            return self._process_response(raw, _first_identifier(category_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        category_id: str,
        limit: Optional[int] = None,
        fulfillment_speed: Optional[Literal["today", "tomorrow"]] = None,
        domain: Optional[Literal["com", "ca", "com.mx"]] = None,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal[
                "best_match",
                "price_low",
                "price_high",
                "best_seller",
                "rating_high",
                "new",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/category (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                category_id=category_id,
                limit=limit,
                fulfillment_speed=fulfillment_speed,
                domain=domain,
                page=page,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
            )
            return self._process_response(raw, _first_identifier(category_id))
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
                f"No Walmart category products found for '{identifier}'. Try "
                "broadening the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioWalmartOffers
# --------------------------------------------------------------------------


class ScavioWalmartOffersInput(BaseModel):
    """Input schema for the ScavioWalmartOffers tool."""

    model_config = ConfigDict(extra="allow")

    product_id: str = Field(
        description=(
            "Walmart item id (usItemId), e.g. 13544111159."
        ),
    )


class ScavioWalmartOffers(BaseTool):  # type: ignore[override]
    """Walmart: Seller offers for a product: price, seller, condition, buy-box flag.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartOffers

            tool = ScavioWalmartOffers()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "13544111159"})
    """

    name: str = "scavio_walmart_offers"
    description: str = (
        "Walmart: Seller offers for a product: price, seller, condition, buy-box flag. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioWalmartOffersInput
    handle_tool_error: bool = True

    api_wrapper: ScavioWalmartOffersAPIWrapper = Field(
        default_factory=ScavioWalmartOffersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartOffersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/offers (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
            )
            return self._process_response(raw, _first_identifier(product_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/offers (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
            )
            return self._process_response(raw, _first_identifier(product_id))
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
                f"No Walmart offers found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioWalmartSeller
# --------------------------------------------------------------------------


class ScavioWalmartSellerInput(BaseModel):
    """Input schema for the ScavioWalmartSeller tool."""

    model_config = ConfigDict(extra="allow")

    seller_id: str = Field(
        description=(
            "NUMERIC catalog seller id (the seller_catalog_id field). The GUID form of "
            "seller_id returns 404."
        ),
    )


class ScavioWalmartSeller(BaseTool):  # type: ignore[override]
    """Walmart: Marketplace seller storefront: name, rating, review count, Pro Seller
    badge, business details.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartSeller

            tool = ScavioWalmartSeller()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"seller_id": "101040442"})
    """

    name: str = "scavio_walmart_seller"
    description: str = (
        "Walmart: Marketplace seller storefront: name, rating, review count, Pro "
        "Seller badge, business details. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioWalmartSellerInput
    handle_tool_error: bool = True

    api_wrapper: ScavioWalmartSellerAPIWrapper = Field(
        default_factory=ScavioWalmartSellerAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartSellerAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        seller_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/seller (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                seller_id=seller_id,
            )
            return self._process_response(raw, _first_identifier(seller_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        seller_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/seller (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                seller_id=seller_id,
            )
            return self._process_response(raw, _first_identifier(seller_id))
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
                f"No Walmart seller found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioWalmartSellerProducts
# --------------------------------------------------------------------------


class ScavioWalmartSellerProductsInput(BaseModel):
    """Input schema for the ScavioWalmartSellerProducts tool."""

    model_config = ConfigDict(extra="allow")

    seller_id: str = Field(
        description=(
            "NUMERIC catalog seller id (the seller_catalog_id field). The GUID form of "
            "seller_id returns 404."
        ),
    )


class ScavioWalmartSellerProducts(BaseTool):  # type: ignore[override]
    """Walmart: A seller's catalog; ~40 items server-rendered, total_count is the real
    catalog size.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioWalmartSellerProducts

            tool = ScavioWalmartSellerProducts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"seller_id": "101040442"})
    """

    name: str = "scavio_walmart_seller_products"
    description: str = (
        "Walmart: A seller's catalog; ~40 items server-rendered, total_count is the "
        "real catalog size. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioWalmartSellerProductsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioWalmartSellerProductsAPIWrapper = Field(
        default_factory=ScavioWalmartSellerProductsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioWalmartSellerProductsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        seller_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/seller-products (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                seller_id=seller_id,
            )
            return self._process_response(raw, _first_identifier(seller_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        seller_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/walmart/seller-products (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                seller_id=seller_id,
            )
            return self._process_response(raw, _first_identifier(seller_id))
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
                f"No Walmart seller products found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw
