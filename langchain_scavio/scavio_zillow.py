"""Scavio Zillow tools for LangChain agents.

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
    ScavioZillowAgentReviewsAPIWrapper,
    ScavioZillowPropertyAPIWrapper,
    ScavioZillowSearchAPIWrapper,
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
# ScavioZillowSearch
# --------------------------------------------------------------------------


class ScavioZillowSearchInput(BaseModel):
    """Input schema for the ScavioZillowSearch tool."""

    model_config = ConfigDict(extra="allow")

    location: str = Field(
        description=(
            "Zillow region slug, human city name, ZIP, or a pasted search URL. A bare "
            "ZIP works alone but CANNOT be combined with a filter or a sort -- use the "
            "city name there."
        ),
    )

    listing_status: Optional[Literal["for_sale", "for_rent", "sold"]] = Field(
        default=None,
        description=(
            "Which listing state to return. Options: for_sale, for_rent, sold. "
            "Default: for_sale."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number, 1-based."
        ),
    )

    sort: Optional[
        Literal[
            "relevance",
            "recommended",
            "newest",
            "price_low",
            "price_high",
            "payment_low",
            "payment_high",
            "beds",
            "baths",
            "sqft",
            "lot_size",
            "zestimate_low",
            "zestimate_high",
            "recent_change",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: relevance, recommended, newest, "
            "price_low, price_high, payment_low, payment_high, beds, baths, sqft, "
            "lot_size, zestimate_low, zestimate_high, recent_change."
        ),
    )

    min_price: Optional[float] = Field(
        default=None,
        description=(
            "Minimum price. On listing_status=for_rent this means MONTHLY RENT."
        ),
    )

    max_price: Optional[float] = Field(
        default=None,
        description=(
            "Maximum price. On listing_status=for_rent this means MONTHLY RENT."
        ),
    )

    beds_min: Optional[int] = Field(
        default=None,
        description=(
            "Minimum number of bedrooms."
        ),
    )

    beds_max: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of bedrooms."
        ),
    )

    baths_min: Optional[float] = Field(
        default=None,
        description=(
            "Minimum number of bathrooms. Half-baths allowed (1.5)."
        ),
    )

    baths_max: Optional[float] = Field(
        default=None,
        description=(
            "Maximum number of bathrooms."
        ),
    )

    sqft_min: Optional[int] = Field(
        default=None,
        description=(
            "Minimum living area in square feet."
        ),
    )

    sqft_max: Optional[int] = Field(
        default=None,
        description=(
            "Maximum living area in square feet."
        ),
    )

    lot_size_min: Optional[int] = Field(
        default=None,
        description=(
            "Minimum lot size in square feet."
        ),
    )

    lot_size_max: Optional[int] = Field(
        default=None,
        description=(
            "Maximum lot size in square feet."
        ),
    )

    year_built_min: Optional[int] = Field(
        default=None,
        description=(
            "Earliest year built."
        ),
    )

    year_built_max: Optional[int] = Field(
        default=None,
        description=(
            "Latest year built."
        ),
    )

    max_hoa: Optional[float] = Field(
        default=None,
        description=(
            "Maximum monthly HOA fee."
        ),
    )

    home_type: Optional[
        Literal[
            "houses",
            "townhomes",
            "multi_family",
            "condos",
            "apartments",
            "manufactured",
            "lots_land",
        ]
    ] = Field(
        default=None,
        description=(
            "Property type filter. Options: houses, townhomes, multi_family, condos, "
            "apartments, manufactured, lots_land."
        ),
    )

    days_on_zillow: Optional[
        Literal["1", "7", "14", "30", "90", "6m", "12m", "24m", "36m"]
    ] = Field(
        default=None,
        description=(
            "How recently the listing appeared. Closed set: an unrecognised value "
            "returns the UNFILTERED result set under a 200. Options: 1, 7, 14, 30, 90, "
            "6m, 12m, 24m, 36m."
        ),
    )

    keywords: Optional[str] = Field(
        default=None,
        description=(
            "Extra keywords to match inside the listing text."
        ),
    )

    has_pool: Optional[bool] = Field(
        default=None,
        description=(
            "Only return properties with a pool."
        ),
    )

    has_garage: Optional[bool] = Field(
        default=None,
        description=(
            "Only return properties with a garage."
        ),
    )

    has_air_conditioning: Optional[bool] = Field(
        default=None,
        description=(
            "Only return properties with air conditioning."
        ),
    )

    is_waterfront: Optional[bool] = Field(
        default=None,
        description=(
            "Only return waterfront properties."
        ),
    )

    has_basement: Optional[bool] = Field(
        default=None,
        description=(
            "Only return properties with a basement."
        ),
    )

    is_new_construction: Optional[bool] = Field(
        default=None,
        description=(
            "Only return new construction."
        ),
    )

    has_open_house: Optional[bool] = Field(
        default=None,
        description=(
            "Only return listings with an open house scheduled."
        ),
    )

    price_reduced: Optional[bool] = Field(
        default=None,
        description=(
            "Only return listings whose price was reduced."
        ),
    )

    is_3d_tour: Optional[bool] = Field(
        default=None,
        description=(
            "Only return listings with a 3D tour."
        ),
    )


class ScavioZillowSearch(BaseTool):  # type: ignore[override]
    """Zillow: Zillow listings in a region: price, beds, baths, living area, Zestimate,
    coordinates, images, days on market.

    Costs 1 credit per call.

    Pagination: page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioZillowSearch

            tool = ScavioZillowSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "location": "Austin, TX",
                    "listing_status": "for_sale",
                }
            )
    """

    name: str = "scavio_zillow_search"
    description: str = (
        "Zillow: Zillow listings in a region: price, beds, baths, living area, "
        "Zestimate, coordinates, images, days on market. Pagination: page. Costs 1 "
        "credit per call."
    )
    args_schema: Type[BaseModel] = ScavioZillowSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioZillowSearchAPIWrapper = Field(
        default_factory=ScavioZillowSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioZillowSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        location: str,
        listing_status: Optional[Literal["for_sale", "for_rent", "sold"]] = None,
        page: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevance",
                "recommended",
                "newest",
                "price_low",
                "price_high",
                "payment_low",
                "payment_high",
                "beds",
                "baths",
                "sqft",
                "lot_size",
                "zestimate_low",
                "zestimate_high",
                "recent_change",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        beds_min: Optional[int] = None,
        beds_max: Optional[int] = None,
        baths_min: Optional[float] = None,
        baths_max: Optional[float] = None,
        sqft_min: Optional[int] = None,
        sqft_max: Optional[int] = None,
        lot_size_min: Optional[int] = None,
        lot_size_max: Optional[int] = None,
        year_built_min: Optional[int] = None,
        year_built_max: Optional[int] = None,
        max_hoa: Optional[float] = None,
        home_type: Optional[
            Literal[
                "houses",
                "townhomes",
                "multi_family",
                "condos",
                "apartments",
                "manufactured",
                "lots_land",
            ]
        ] = None,
        days_on_zillow: Optional[
            Literal["1", "7", "14", "30", "90", "6m", "12m", "24m", "36m"]
        ] = None,
        keywords: Optional[str] = None,
        has_pool: Optional[bool] = None,
        has_garage: Optional[bool] = None,
        has_air_conditioning: Optional[bool] = None,
        is_waterfront: Optional[bool] = None,
        has_basement: Optional[bool] = None,
        is_new_construction: Optional[bool] = None,
        has_open_house: Optional[bool] = None,
        price_reduced: Optional[bool] = None,
        is_3d_tour: Optional[bool] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/zillow/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                location=location,
                listing_status=listing_status,
                page=page,
                sort=sort,
                min_price=min_price,
                max_price=max_price,
                beds_min=beds_min,
                beds_max=beds_max,
                baths_min=baths_min,
                baths_max=baths_max,
                sqft_min=sqft_min,
                sqft_max=sqft_max,
                lot_size_min=lot_size_min,
                lot_size_max=lot_size_max,
                year_built_min=year_built_min,
                year_built_max=year_built_max,
                max_hoa=max_hoa,
                home_type=home_type,
                days_on_zillow=days_on_zillow,
                keywords=keywords,
                has_pool=has_pool,
                has_garage=has_garage,
                has_air_conditioning=has_air_conditioning,
                is_waterfront=is_waterfront,
                has_basement=has_basement,
                is_new_construction=is_new_construction,
                has_open_house=has_open_house,
                price_reduced=price_reduced,
                is_3d_tour=is_3d_tour,
            )
            return self._process_response(raw, _first_identifier(location))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        location: str,
        listing_status: Optional[Literal["for_sale", "for_rent", "sold"]] = None,
        page: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevance",
                "recommended",
                "newest",
                "price_low",
                "price_high",
                "payment_low",
                "payment_high",
                "beds",
                "baths",
                "sqft",
                "lot_size",
                "zestimate_low",
                "zestimate_high",
                "recent_change",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        beds_min: Optional[int] = None,
        beds_max: Optional[int] = None,
        baths_min: Optional[float] = None,
        baths_max: Optional[float] = None,
        sqft_min: Optional[int] = None,
        sqft_max: Optional[int] = None,
        lot_size_min: Optional[int] = None,
        lot_size_max: Optional[int] = None,
        year_built_min: Optional[int] = None,
        year_built_max: Optional[int] = None,
        max_hoa: Optional[float] = None,
        home_type: Optional[
            Literal[
                "houses",
                "townhomes",
                "multi_family",
                "condos",
                "apartments",
                "manufactured",
                "lots_land",
            ]
        ] = None,
        days_on_zillow: Optional[
            Literal["1", "7", "14", "30", "90", "6m", "12m", "24m", "36m"]
        ] = None,
        keywords: Optional[str] = None,
        has_pool: Optional[bool] = None,
        has_garage: Optional[bool] = None,
        has_air_conditioning: Optional[bool] = None,
        is_waterfront: Optional[bool] = None,
        has_basement: Optional[bool] = None,
        is_new_construction: Optional[bool] = None,
        has_open_house: Optional[bool] = None,
        price_reduced: Optional[bool] = None,
        is_3d_tour: Optional[bool] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/zillow/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                location=location,
                listing_status=listing_status,
                page=page,
                sort=sort,
                min_price=min_price,
                max_price=max_price,
                beds_min=beds_min,
                beds_max=beds_max,
                baths_min=baths_min,
                baths_max=baths_max,
                sqft_min=sqft_min,
                sqft_max=sqft_max,
                lot_size_min=lot_size_min,
                lot_size_max=lot_size_max,
                year_built_min=year_built_min,
                year_built_max=year_built_max,
                max_hoa=max_hoa,
                home_type=home_type,
                days_on_zillow=days_on_zillow,
                keywords=keywords,
                has_pool=has_pool,
                has_garage=has_garage,
                has_air_conditioning=has_air_conditioning,
                is_waterfront=is_waterfront,
                has_basement=has_basement,
                is_new_construction=is_new_construction,
                has_open_house=has_open_house,
                price_reduced=price_reduced,
                is_3d_tour=is_3d_tour,
            )
            return self._process_response(raw, _first_identifier(location))
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
                f"No Zillow results found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioZillowProperty
# --------------------------------------------------------------------------


class ScavioZillowPropertyInput(BaseModel):
    """Input schema for the ScavioZillowProperty tool."""

    model_config = ConfigDict(extra="allow")

    zpid: str = Field(
        description=(
            "Zillow property id, a /homedetails/ URL, or a zillow.com/apartments/ "
            "building URL. Rental buildings have no visible zpid -- pass the URL."
        ),
    )


class ScavioZillowProperty(BaseTool):  # type: ignore[override]
    """Zillow: Full Zillow listing: price and price history, Zestimate, tax history,
    RESO facts, rooms, schools, open houses, photos.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioZillowProperty

            tool = ScavioZillowProperty()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"zpid": "29444874"})
    """

    name: str = "scavio_zillow_property"
    description: str = (
        "Zillow: Full Zillow listing: price and price history, Zestimate, tax history, "
        "RESO facts, rooms, schools, open houses, photos. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioZillowPropertyInput
    handle_tool_error: bool = True

    api_wrapper: ScavioZillowPropertyAPIWrapper = Field(
        default_factory=ScavioZillowPropertyAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioZillowPropertyAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        zpid: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/zillow/property (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                zpid=zpid,
            )
            return self._process_response(raw, _first_identifier(zpid))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        zpid: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/zillow/property (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                zpid=zpid,
            )
            return self._process_response(raw, _first_identifier(zpid))
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
                f"No Zillow property found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioZillowAgentReviews
# --------------------------------------------------------------------------


class ScavioZillowAgentReviewsInput(BaseModel):
    """Input schema for the ScavioZillowAgentReviews tool."""

    model_config = ConfigDict(extra="allow")

    screen_name: str = Field(
        description=(
            "The AGENT's zillow.com/profile/<name>/ screen name, or the full profile "
            "URL. This endpoint addresses an agent, not a property."
        ),
    )


class ScavioZillowAgentReviews(BaseTool):  # type: ignore[override]
    """Zillow: A Zillow AGENT's profile and reviews: rating, bodies with sub-ratings,
    specialties, licenses, service areas, sales counts.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioZillowAgentReviews

            tool = ScavioZillowAgentReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"screen_name": "jane-smith"})
    """

    name: str = "scavio_zillow_agent_reviews"
    description: str = (
        "Zillow: A Zillow AGENT's profile and reviews: rating, bodies with "
        "sub-ratings, specialties, licenses, service areas, sales counts. Costs 1 "
        "credit per call."
    )
    args_schema: Type[BaseModel] = ScavioZillowAgentReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioZillowAgentReviewsAPIWrapper = Field(
        default_factory=ScavioZillowAgentReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioZillowAgentReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        screen_name: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/zillow/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                screen_name=screen_name,
            )
            return self._process_response(raw, _first_identifier(screen_name))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        screen_name: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/zillow/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                screen_name=screen_name,
            )
            return self._process_response(raw, _first_identifier(screen_name))
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
                f"No Zillow agent found for '{identifier}'. Verify the identifiers you "
                "passed."
            )
        return raw
