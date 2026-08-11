"""Scavio Tripadvisor tools for LangChain agents.

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
    ScavioTripadvisorLocationAPIWrapper,
    ScavioTripadvisorLocationsAPIWrapper,
    ScavioTripadvisorReviewsAPIWrapper,
    ScavioTripadvisorSearchAPIWrapper,
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
# ScavioTripadvisorLocations
# --------------------------------------------------------------------------


class ScavioTripadvisorLocationsInput(BaseModel):
    """Input schema for the ScavioTripadvisorLocations tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Place or business NAME to resolve into TripAdvisor ids."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Maximum rows to return, 1-20. Default: 12."
        ),
    )


class ScavioTripadvisorLocations(BaseTool):  # type: ignore[override]
    """Tripadvisor: START HERE. Resolve a place or business NAME to the TripAdvisor
    geo_id / location_id pair every other endpoint needs.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTripadvisorLocations

            tool = ScavioTripadvisorLocations()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "Franklin Barbecue"})
    """

    name: str = "scavio_tripadvisor_locations"
    description: str = (
        "Tripadvisor: START HERE. Resolve a place or business NAME to the TripAdvisor "
        "geo_id / location_id pair every other endpoint needs. Costs 2 credits per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioTripadvisorLocationsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTripadvisorLocationsAPIWrapper = Field(
        default_factory=ScavioTripadvisorLocationsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTripadvisorLocationsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/tripadvisor/locations (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                limit=limit,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/tripadvisor/locations (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                limit=limit,
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
                f"No Tripadvisor locations found for '{identifier}'. Resolve the place "
                "with ScavioTripadvisorLocations first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioTripadvisorSearch
# --------------------------------------------------------------------------


class ScavioTripadvisorSearchInput(BaseModel):
    """Input schema for the ScavioTripadvisorSearch tool."""

    model_config = ConfigDict(extra="allow")

    geo_id: Optional[str] = Field(
        default=None,
        description=(
            "TripAdvisor geo id. Accepts 30196, g30196, or a URL carrying one."
        ),
    )

    category: Optional[Literal["restaurants", "hotels", "attractions"]] = Field(
        default=None,
        description=(
            "Which family the location belongs to. On reviews it also sets the page "
            "size (15 for restaurants, 10 for hotels and attractions), so it must "
            "match the location's own type on any page past the first. Options: "
            "restaurants, hotels, attractions. Default: restaurants."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 30 locations per page; a page beyond the last is a "
            "404, not an empty result."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full tripadvisor.com listing URL, usable instead of the ids. Country "
            "sites are accepted."
        ),
    )


class ScavioTripadvisorSearch(BaseTool):  # type: ignore[override]
    """Tripadvisor: Restaurants / hotels / attractions in a TripAdvisor geo,
    TripAdvisor-ranked; each row carries the location_id + geo_id pair.

    Costs 2 credits per call.

    Pagination: page -- 30 locations per page; a page beyond the last is a 404, not
    an empty result.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTripadvisorSearch

            tool = ScavioTripadvisorSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"geo_id": "30196", "category": "restaurants"})
    """

    name: str = "scavio_tripadvisor_search"
    description: str = (
        "Tripadvisor: Restaurants / hotels / attractions in a TripAdvisor geo, "
        "TripAdvisor-ranked; each row carries the location_id + geo_id pair. "
        "Pagination: page -- 30 locations per page; a page beyond the last is a 404, "
        "not an empty result. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioTripadvisorSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTripadvisorSearchAPIWrapper = Field(
        default_factory=ScavioTripadvisorSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTripadvisorSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        geo_id: Optional[str] = None,
        category: Optional[Literal["restaurants", "hotels", "attractions"]] = None,
        page: Optional[int] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/tripadvisor/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                geo_id=geo_id,
                category=category,
                page=page,
                url=url,
            )
            return self._process_response(raw, _first_identifier(geo_id, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        geo_id: Optional[str] = None,
        category: Optional[Literal["restaurants", "hotels", "attractions"]] = None,
        page: Optional[int] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/tripadvisor/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                geo_id=geo_id,
                category=category,
                page=page,
                url=url,
            )
            return self._process_response(raw, _first_identifier(geo_id, url))
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
                f"No Tripadvisor results found for '{identifier}'. Resolve the place "
                "with ScavioTripadvisorLocations first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioTripadvisorLocation
# --------------------------------------------------------------------------


class ScavioTripadvisorLocationInput(BaseModel):
    """Input schema for the ScavioTripadvisorLocation tool."""

    model_config = ConfigDict(extra="allow")

    location_id: Optional[str] = Field(
        default=None,
        description=(
            "TripAdvisor location id. Accepts 1899234, d1899234, or a full _Review "
            "URL."
        ),
    )

    geo_id: Optional[str] = Field(
        default=None,
        description=(
            "TripAdvisor geo id. Accepts 30196, g30196, or a URL carrying one."
        ),
    )

    category: Optional[Literal["restaurants", "hotels", "attractions"]] = Field(
        default=None,
        description=(
            "Which family the location belongs to. On reviews it also sets the page "
            "size (15 for restaurants, 10 for hotels and attractions), so it must "
            "match the location's own type on any page past the first. Options: "
            "restaurants, hotels, attractions. Default: restaurants."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full tripadvisor.com listing URL, usable instead of the ids. Country "
            "sites are accepted."
        ),
    )


class ScavioTripadvisorLocation(BaseTool):  # type: ignore[override]
    """Tripadvisor: One TripAdvisor location in full: rating histogram, sub-ratings,
    city ranking, amenities, contact, photos, and the FIRST PAGE OF REVIEWS.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTripadvisorLocation

            tool = ScavioTripadvisorLocation()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"location_id": "1899234", "geo_id": "30196"})
    """

    name: str = "scavio_tripadvisor_location"
    description: str = (
        "Tripadvisor: One TripAdvisor location in full: rating histogram, sub-ratings, "
        "city ranking, amenities, contact, photos, and the FIRST PAGE OF REVIEWS. "
        "Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioTripadvisorLocationInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTripadvisorLocationAPIWrapper = Field(
        default_factory=ScavioTripadvisorLocationAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTripadvisorLocationAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        location_id: Optional[str] = None,
        geo_id: Optional[str] = None,
        category: Optional[Literal["restaurants", "hotels", "attractions"]] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/tripadvisor/location (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                location_id=location_id,
                geo_id=geo_id,
                category=category,
                url=url,
            )
            return self._process_response(
                raw, _first_identifier(location_id, geo_id, url)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        location_id: Optional[str] = None,
        geo_id: Optional[str] = None,
        category: Optional[Literal["restaurants", "hotels", "attractions"]] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/tripadvisor/location (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                location_id=location_id,
                geo_id=geo_id,
                category=category,
                url=url,
            )
            return self._process_response(
                raw, _first_identifier(location_id, geo_id, url)
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
                f"No Tripadvisor location found for '{identifier}'. Resolve the place "
                "with ScavioTripadvisorLocations first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioTripadvisorReviews
# --------------------------------------------------------------------------


class ScavioTripadvisorReviewsInput(BaseModel):
    """Input schema for the ScavioTripadvisorReviews tool."""

    model_config = ConfigDict(extra="allow")

    location_id: Optional[str] = Field(
        default=None,
        description=(
            "TripAdvisor location id. Accepts 1899234, d1899234, or a full _Review "
            "URL."
        ),
    )

    geo_id: Optional[str] = Field(
        default=None,
        description=(
            "TripAdvisor geo id. Accepts 30196, g30196, or a URL carrying one."
        ),
    )

    category: Optional[Literal["restaurants", "hotels", "attractions"]] = Field(
        default=None,
        description=(
            "Which family the location belongs to. On reviews it also sets the page "
            "size (15 for restaurants, 10 for hotels and attractions), so it must "
            "match the location's own type on any page past the first. Options: "
            "restaurants, hotels, attractions. Default: restaurants."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full tripadvisor.com listing URL, usable instead of the ids. Country "
            "sites are accepted."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. Page 1 is already inside the location endpoint -- "
            "use this to page PAST it. Past the last page is a 404."
        ),
    )


class ScavioTripadvisorReviews(BaseTool):  # type: ignore[override]
    """Tripadvisor: A page of TripAdvisor reviews: rating, trip date and type, reviewer
    home town and contribution count, management response.

    Costs 2 credits per call.

    Pagination: page -- 15 per page for restaurants, 10 for hotels and attractions,
    so `category` must match the location's own type on any page past the first.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTripadvisorReviews

            tool = ScavioTripadvisorReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "location_id": "1899234",
                    "geo_id": "30196",
                    "page": 2,
                }
            )
    """

    name: str = "scavio_tripadvisor_reviews"
    description: str = (
        "Tripadvisor: A page of TripAdvisor reviews: rating, trip date and type, "
        "reviewer home town and contribution count, management response. Pagination: "
        "page -- 15 per page for restaurants, 10 for hotels and attractions, so "
        "`category` must match the location's own type on any page past the first. "
        "Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioTripadvisorReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTripadvisorReviewsAPIWrapper = Field(
        default_factory=ScavioTripadvisorReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTripadvisorReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        location_id: Optional[str] = None,
        geo_id: Optional[str] = None,
        category: Optional[Literal["restaurants", "hotels", "attractions"]] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/tripadvisor/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                location_id=location_id,
                geo_id=geo_id,
                category=category,
                url=url,
                page=page,
            )
            return self._process_response(
                raw, _first_identifier(location_id, geo_id, url)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        location_id: Optional[str] = None,
        geo_id: Optional[str] = None,
        category: Optional[Literal["restaurants", "hotels", "attractions"]] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/tripadvisor/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                location_id=location_id,
                geo_id=geo_id,
                category=category,
                url=url,
                page=page,
            )
            return self._process_response(
                raw, _first_identifier(location_id, geo_id, url)
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
                f"No Tripadvisor reviews found for '{identifier}'. Resolve the place "
                "with ScavioTripadvisorLocations first."
            )
        return raw
