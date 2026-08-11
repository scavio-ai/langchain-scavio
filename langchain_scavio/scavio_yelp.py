"""Scavio Yelp tools for LangChain agents.

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
    ScavioYelpBusinessAPIWrapper,
    ScavioYelpReviewsAPIWrapper,
    ScavioYelpSearchAPIWrapper,
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
# ScavioYelpSearch
# --------------------------------------------------------------------------


class ScavioYelpSearchInput(BaseModel):
    """Input schema for the ScavioYelpSearch tool."""

    model_config = ConfigDict(extra="allow")

    term: Optional[str] = Field(
        default=None,
        description=(
            "What to look for, e.g. 'coffee' or a business name."
        ),
    )

    location: Optional[str] = Field(
        default=None,
        description=(
            "City, neighbourhood or address. Effectively required: without it Yelp "
            "geolocates off the proxy exit and the same request answers about a "
            "different metro run to run."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. Yelp fixes the page size at 10."
        ),
    )

    sort: Optional[Literal["recommended", "rating", "review_count"]] = Field(
        default=None,
        description=(
            "Sort order. Closed set: Yelp IGNORES an unrecognised value and serves "
            "default ranking under a billed 200. Options: recommended, rating, "
            "review_count. Default: recommended."
        ),
    )

    price: Optional[list[int]] = Field(
        default=None,
        description=(
            "Price bands to include, 1 (cheapest) to 4 (priciest). Options: 1, 2, 3, "
            "4."
        ),
    )

    open_now: Optional[bool] = Field(
        default=None,
        description=(
            "Only return businesses open right now."
        ),
    )

    attributes: Optional[list[str]] = Field(
        default=None,
        description=(
            "Raw Yelp filter aliases sent through as attrs, e.g. RestaurantsDelivery, "
            "GoodForKids, WheelchairAccessible. This is a passthrough, not a closed "
            "enum: an alias Yelp does not know is ignored and results come back "
            "unfiltered."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full yelp.com/search URL, usable instead of term plus location."
        ),
    )


class ScavioYelpSearch(BaseTool):  # type: ignore[override]
    """Yelp: Businesses in Yelp's ranked order: rating, review count, price band,
    categories, address, contact rails, hours, photos, review snippet.

    Costs 2 credits per call.

    Pagination: page -- Yelp fixes the page size at 10.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYelpSearch

            tool = ScavioYelpSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"term": "coffee", "location": "Austin, TX"})
    """

    name: str = "scavio_yelp_search"
    description: str = (
        "Yelp: Businesses in Yelp's ranked order: rating, review count, price band, "
        "categories, address, contact rails, hours, photos, review snippet. "
        "Pagination: page -- Yelp fixes the page size at 10. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioYelpSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYelpSearchAPIWrapper = Field(
        default_factory=ScavioYelpSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYelpSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        term: Optional[str] = None,
        location: Optional[str] = None,
        page: Optional[int] = None,
        sort: Optional[Literal["recommended", "rating", "review_count"]] = None,
        price: Optional[list[int]] = None,
        open_now: Optional[bool] = None,
        attributes: Optional[list[str]] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/yelp/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                term=term,
                location=location,
                page=page,
                sort=sort,
                price=price,
                open_now=open_now,
                attributes=attributes,
                url=url,
            )
            return self._process_response(raw, _first_identifier(term, location, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        term: Optional[str] = None,
        location: Optional[str] = None,
        page: Optional[int] = None,
        sort: Optional[Literal["recommended", "rating", "review_count"]] = None,
        price: Optional[list[int]] = None,
        open_now: Optional[bool] = None,
        attributes: Optional[list[str]] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/yelp/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                term=term,
                location=location,
                page=page,
                sort=sort,
                price=price,
                open_now=open_now,
                attributes=attributes,
                url=url,
            )
            return self._process_response(raw, _first_identifier(term, location, url))
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
                f"No Yelp results found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioYelpBusiness
# --------------------------------------------------------------------------


class ScavioYelpBusinessInput(BaseModel):
    """Input schema for the ScavioYelpBusiness tool."""

    model_config = ConfigDict(extra="allow")

    business_id: Optional[str] = Field(
        default=None,
        description=(
            "Yelp alias (desnudo-coffee-austin-2), opaque encid, or a yelp.com/biz "
            "URL."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full listing URL, usable instead of the id fields."
        ),
    )


class ScavioYelpBusiness(BaseTool):  # type: ignore[override]
    """Yelp: One Yelp business in full: per-star histogram, price band, address and
    coordinates, hours, amenities, photos, health inspections, Q&A -- PLUS the first
    page of reviews at no extra cost.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYelpBusiness

            tool = ScavioYelpBusiness()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"business_id": "desnudo-coffee-austin-2"})
    """

    name: str = "scavio_yelp_business"
    description: str = (
        "Yelp: One Yelp business in full: per-star histogram, price band, address and "
        "coordinates, hours, amenities, photos, health inspections, Q&A -- PLUS the "
        "first page of reviews at no extra cost. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioYelpBusinessInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYelpBusinessAPIWrapper = Field(
        default_factory=ScavioYelpBusinessAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYelpBusinessAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        business_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/yelp/business (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                business_id=business_id,
                url=url,
            )
            return self._process_response(raw, _first_identifier(business_id, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        business_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/yelp/business (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                business_id=business_id,
                url=url,
            )
            return self._process_response(raw, _first_identifier(business_id, url))
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
                f"No Yelp business found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioYelpReviews
# --------------------------------------------------------------------------


class ScavioYelpReviewsInput(BaseModel):
    """Input schema for the ScavioYelpReviews tool."""

    model_config = ConfigDict(extra="allow")

    business_id: Optional[str] = Field(
        default=None,
        description=(
            "Yelp alias (desnudo-coffee-austin-2), opaque encid, or a yelp.com/biz "
            "URL."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full listing URL, usable instead of the id fields."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. PAGE 1 IS REDUNDANT with the business endpoint and "
            "costs another 2 credits -- start at page 2. A page past the last is a "
            "404."
        ),
    )

    sort: Optional[
        Literal["relevance", "newest", "oldest", "rating_high", "rating_low", "elites"]
    ] = Field(
        default=None,
        description=(
            "Sort order. Closed set: Yelp IGNORES an unrecognised value and serves "
            "default ranking under a billed 200. Options: relevance, newest, oldest, "
            "rating_high, rating_low, elites. Default: relevance."
        ),
    )

    rating: Optional[Literal[1, 2, 3, 4, 5]] = Field(
        default=None,
        description=(
            "Only return reviews with this star rating. Changes filtered_review_count, "
            "not review_count. Options: 1, 2, 3, 4, 5."
        ),
    )


class ScavioYelpReviews(BaseTool):  # type: ignore[override]
    """Yelp: A page of Yelp reviews: rating, full text, language, author profile and
    expertise counts, attached photos, reaction counts, owner response.

    Costs 2 credits per call.

    Pagination: page -- 10 per page; a page past the last is a 404, not an empty
    result.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioYelpReviews

            tool = ScavioYelpReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"business_id": "desnudo-coffee-austin-2", "page": 2})
    """

    name: str = "scavio_yelp_reviews"
    description: str = (
        "Yelp: A page of Yelp reviews: rating, full text, language, author profile and "
        "expertise counts, attached photos, reaction counts, owner response. "
        "Pagination: page -- 10 per page; a page past the last is a 404, not an empty "
        "result. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioYelpReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioYelpReviewsAPIWrapper = Field(
        default_factory=ScavioYelpReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioYelpReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        business_id: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevance",
                "newest",
                "oldest",
                "rating_high",
                "rating_low",
                "elites",
            ]
        ] = None,
        rating: Optional[Literal[1, 2, 3, 4, 5]] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/yelp/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                business_id=business_id,
                url=url,
                page=page,
                sort=sort,
                rating=rating,
            )
            return self._process_response(raw, _first_identifier(business_id, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        business_id: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevance",
                "newest",
                "oldest",
                "rating_high",
                "rating_low",
                "elites",
            ]
        ] = None,
        rating: Optional[Literal[1, 2, 3, 4, 5]] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/yelp/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                business_id=business_id,
                url=url,
                page=page,
                sort=sort,
                rating=rating,
            )
            return self._process_response(raw, _first_identifier(business_id, url))
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
                f"No Yelp reviews found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw
