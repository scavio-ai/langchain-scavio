"""Scavio Home Depot tools for LangChain agents.

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
    ScavioHomeDepotProductAPIWrapper,
    ScavioHomeDepotReviewsAPIWrapper,
    ScavioHomeDepotSearchAPIWrapper,
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
# ScavioHomeDepotSearch
# --------------------------------------------------------------------------


class ScavioHomeDepotSearchInput(BaseModel):
    """Input schema for the ScavioHomeDepotSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Product search query."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 12 products per page, fixed."
        ),
    )

    sort_by: Optional[
        Literal["best_match", "top_sellers", "top_rated", "price_low", "price_high"]
    ] = Field(
        default=None,
        description=(
            "Sort order. The set is closed because Home Depot answers an unknown sort "
            "with an empty page rather than falling back. Options: best_match, "
            "top_sellers, top_rated, price_low, price_high. Default: best_match."
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


class ScavioHomeDepotSearch(BaseTool):  # type: ignore[override]
    """Home Depot: Search Home Depot: price and promotions, brand and model, ratings,
    badges, per-store pickup/delivery.

    Costs 2 credits per call.

    Pagination: page -- page size is FIXED at 12 and cannot be changed.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioHomeDepotSearch

            tool = ScavioHomeDepotSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "cordless drill"})
    """

    name: str = "scavio_home_depot_search"
    description: str = (
        "Home Depot: Search Home Depot: price and promotions, brand and model, "
        "ratings, badges, per-store pickup/delivery. Pagination: page -- page size is "
        "FIXED at 12 and cannot be changed. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioHomeDepotSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioHomeDepotSearchAPIWrapper = Field(
        default_factory=ScavioHomeDepotSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioHomeDepotSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal["best_match", "top_sellers", "top_rated", "price_low", "price_high"]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/homedepot/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
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
        page: Optional[int] = None,
        sort_by: Optional[
            Literal["best_match", "top_sellers", "top_rated", "price_low", "price_high"]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/homedepot/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
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
        if not raw.get("data"):
            raise ToolException(
                f"No Home Depot results found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioHomeDepotProduct
# --------------------------------------------------------------------------


class ScavioHomeDepotProductInput(BaseModel):
    """Input schema for the ScavioHomeDepotProduct tool."""

    model_config = ConfigDict(extra="allow")

    item_id: str = Field(
        description=(
            "Home Depot item id or a full homedepot.com/p/... URL. Tracking parameters "
            "are discarded."
        ),
    )


class ScavioHomeDepotProduct(BaseTool):  # type: ignore[override]
    """Home Depot: Full Home Depot item detail: pricing, images and videos, spec table,
    dimensions, bullets, documents, return policy.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioHomeDepotProduct

            tool = ScavioHomeDepotProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"item_id": "313021355"})
    """

    name: str = "scavio_home_depot_product"
    description: str = (
        "Home Depot: Full Home Depot item detail: pricing, images and videos, spec "
        "table, dimensions, bullets, documents, return policy. Costs 2 credits per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioHomeDepotProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioHomeDepotProductAPIWrapper = Field(
        default_factory=ScavioHomeDepotProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioHomeDepotProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        item_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/homedepot/product (synchronously)."""
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
        """Call POST /api/v1/homedepot/product (asynchronously)."""
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
                f"No Home Depot product found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioHomeDepotReviews
# --------------------------------------------------------------------------


class ScavioHomeDepotReviewsInput(BaseModel):
    """Input schema for the ScavioHomeDepotReviews tool."""

    model_config = ConfigDict(extra="allow")

    item_id: str = Field(
        description=(
            "Home Depot item id or a full homedepot.com/p/... URL. Tracking parameters "
            "are discarded."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 30 reviews per page; asking past total_pages is a "
            "404."
        ),
    )


class ScavioHomeDepotReviews(BaseTool):  # type: ignore[override]
    """Home Depot: One page of full Home Depot review bodies, rating distribution, per-
    attribute ratings, photos, seller responses.

    Costs 2 credits per call.

    Pagination: page -- 30 per page; total_pages is the last that exists and asking
    past it is a 404.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioHomeDepotReviews

            tool = ScavioHomeDepotReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"item_id": "313021355", "page": 2})
    """

    name: str = "scavio_home_depot_reviews"
    description: str = (
        "Home Depot: One page of full Home Depot review bodies, rating distribution, "
        "per-attribute ratings, photos, seller responses. Pagination: page -- 30 per "
        "page; total_pages is the last that exists and asking past it is a 404. Costs "
        "2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioHomeDepotReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioHomeDepotReviewsAPIWrapper = Field(
        default_factory=ScavioHomeDepotReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioHomeDepotReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        item_id: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/homedepot/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                item_id=item_id,
                page=page,
            )
            return self._process_response(raw, _first_identifier(item_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        item_id: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/homedepot/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                item_id=item_id,
                page=page,
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
                f"No Home Depot reviews found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw
