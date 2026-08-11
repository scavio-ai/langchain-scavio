"""Scavio Airbnb tools for LangChain agents.

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
    ScavioAirbnbListingAPIWrapper,
    ScavioAirbnbReviewsAPIWrapper,
    ScavioAirbnbSearchAPIWrapper,
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
# ScavioAirbnbSearch
# --------------------------------------------------------------------------


class ScavioAirbnbSearchInput(BaseModel):
    """Input schema for the ScavioAirbnbSearch tool."""

    model_config = ConfigDict(extra="allow")

    location: str = Field(
        description=(
            "City, region, ZIP, or a pasted airbnb.com/s/ URL. An unresolvable place "
            "is a 404."
        ),
    )

    check_in: Optional[str] = Field(
        default=None,
        description=(
            "Check-in date, YYYY-MM-DD. Must be sent with check_out. Omitting both "
            "makes Airbnb A/B both the window AND the prices -- the response flags "
            "that as dates_are_defaulted. Default: +30d when omitted (transport)."
        ),
    )

    check_out: Optional[str] = Field(
        default=None,
        description=(
            "Check-out date, YYYY-MM-DD. Must be sent with check_in. Default: check_in "
            "+ 5 nights when omitted."
        ),
    )

    adults: Optional[int] = Field(
        default=None,
        description=(
            "Number of adult guests."
        ),
    )

    children: Optional[int] = Field(
        default=None,
        description=(
            "Number of children in the party (ages 2-12)."
        ),
    )

    infants: Optional[int] = Field(
        default=None,
        description=(
            "Number of infants in the party."
        ),
    )

    pets: Optional[int] = Field(
        default=None,
        description=(
            "Number of pets travelling."
        ),
    )

    min_price: Optional[float] = Field(
        default=None,
        description=(
            "Minimum WHOLE-STAY total, not a per-night rate."
        ),
    )

    max_price: Optional[float] = Field(
        default=None,
        description=(
            "Maximum WHOLE-STAY total, not a per-night rate."
        ),
    )

    room_type: Optional[
        Literal["entire_home", "private_room", "shared_room", "hotel_room"]
    ] = Field(
        default=None,
        description=(
            "Room type filter. Options: entire_home, private_room, shared_room, "
            "hotel_room."
        ),
    )

    min_bedrooms: Optional[int] = Field(
        default=None,
        description=(
            "Minimum number of bedrooms."
        ),
    )

    min_beds: Optional[int] = Field(
        default=None,
        description=(
            "Minimum number of beds."
        ),
    )

    min_bathrooms: Optional[int] = Field(
        default=None,
        description=(
            "Minimum number of bathrooms."
        ),
    )

    superhost: Optional[bool] = Field(
        default=None,
        description=(
            "Only return Superhost listings."
        ),
    )

    instant_book: Optional[bool] = Field(
        default=None,
        description=(
            "Only return instant-book listings."
        ),
    )

    guest_favorite: Optional[bool] = Field(
        default=None,
        description=(
            "Only return Guest Favourite listings."
        ),
    )

    free_cancellation: Optional[bool] = Field(
        default=None,
        description=(
            "Only return listings with free cancellation."
        ),
    )

    amenities: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated amenity filter. Named vocabulary: wifi, air_conditioning, "
            "pool, kitchen, free_parking, washer, self_check_in, tv -- or raw numeric "
            "Airbnb amenity ids. An unrecognised NAME is rejected before the scrape. "
            "Options: wifi, air_conditioning, pool, kitchen, free_parking, washer, "
            "self_check_in, tv."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        description=(
            "ISO 4217 currency code the prices come back in. Default: USD."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 18 listings per page. Cannot be combined with "
            "cursor."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "next_cursor from a previous response. Wins over page, so sending both is "
            "rejected."
        ),
    )


class ScavioAirbnbSearch(BaseTool):  # type: ignore[override]
    """Airbnb: Airbnb stays: stay-total and per-night price with the full discount
    ledger, rating, bedrooms/beds/baths, coordinates, badges, images.

    Costs 1 credit per call.

    Pagination: page XOR cursor -- `cursor` WINS over `page`, so sending both is
    REJECTED. 18 listings per page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAirbnbSearch

            tool = ScavioAirbnbSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "location": "Lisbon",
                    "check_in": "2026-09-10",
                    "check_out": "2026-09-15",
                }
            )
    """

    name: str = "scavio_airbnb_search"
    description: str = (
        "Airbnb: Airbnb stays: stay-total and per-night price with the full discount "
        "ledger, rating, bedrooms/beds/baths, coordinates, badges, images. Pagination: "
        "page XOR cursor -- `cursor` WINS over `page`, so sending both is REJECTED. 18 "
        "listings per page. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioAirbnbSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioAirbnbSearchAPIWrapper = Field(
        default_factory=ScavioAirbnbSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAirbnbSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        location: str,
        check_in: Optional[str] = None,
        check_out: Optional[str] = None,
        adults: Optional[int] = None,
        children: Optional[int] = None,
        infants: Optional[int] = None,
        pets: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        room_type: Optional[
            Literal["entire_home", "private_room", "shared_room", "hotel_room"]
        ] = None,
        min_bedrooms: Optional[int] = None,
        min_beds: Optional[int] = None,
        min_bathrooms: Optional[int] = None,
        superhost: Optional[bool] = None,
        instant_book: Optional[bool] = None,
        guest_favorite: Optional[bool] = None,
        free_cancellation: Optional[bool] = None,
        amenities: Optional[str] = None,
        currency: Optional[str] = None,
        page: Optional[int] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/airbnb/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                location=location,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=children,
                infants=infants,
                pets=pets,
                min_price=min_price,
                max_price=max_price,
                room_type=room_type,
                min_bedrooms=min_bedrooms,
                min_beds=min_beds,
                min_bathrooms=min_bathrooms,
                superhost=superhost,
                instant_book=instant_book,
                guest_favorite=guest_favorite,
                free_cancellation=free_cancellation,
                amenities=amenities,
                currency=currency,
                page=page,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(location))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        location: str,
        check_in: Optional[str] = None,
        check_out: Optional[str] = None,
        adults: Optional[int] = None,
        children: Optional[int] = None,
        infants: Optional[int] = None,
        pets: Optional[int] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        room_type: Optional[
            Literal["entire_home", "private_room", "shared_room", "hotel_room"]
        ] = None,
        min_bedrooms: Optional[int] = None,
        min_beds: Optional[int] = None,
        min_bathrooms: Optional[int] = None,
        superhost: Optional[bool] = None,
        instant_book: Optional[bool] = None,
        guest_favorite: Optional[bool] = None,
        free_cancellation: Optional[bool] = None,
        amenities: Optional[str] = None,
        currency: Optional[str] = None,
        page: Optional[int] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/airbnb/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                location=location,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=children,
                infants=infants,
                pets=pets,
                min_price=min_price,
                max_price=max_price,
                room_type=room_type,
                min_bedrooms=min_bedrooms,
                min_beds=min_beds,
                min_bathrooms=min_bathrooms,
                superhost=superhost,
                instant_book=instant_book,
                guest_favorite=guest_favorite,
                free_cancellation=free_cancellation,
                amenities=amenities,
                currency=currency,
                page=page,
                cursor=cursor,
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
                f"No Airbnb results found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioAirbnbListing
# --------------------------------------------------------------------------


class ScavioAirbnbListingInput(BaseModel):
    """Input schema for the ScavioAirbnbListing tool."""

    model_config = ConfigDict(extra="allow")

    listing_id: str = Field(
        description=(
            "Airbnb listing id or a full /rooms/ URL. Query parameters are discarded "
            "because they carry someone else's dates."
        ),
    )

    check_in: Optional[str] = Field(
        default=None,
        description=(
            "Check-in date, YYYY-MM-DD. Must be sent with check_out. Omitting both "
            "makes Airbnb A/B both the window AND the prices -- the response flags "
            "that as dates_are_defaulted."
        ),
    )

    check_out: Optional[str] = Field(
        default=None,
        description=(
            "Check-out date, YYYY-MM-DD. Must be sent with check_in."
        ),
    )

    adults: Optional[int] = Field(
        default=None,
        description=(
            "Number of adult guests."
        ),
    )

    children: Optional[int] = Field(
        default=None,
        description=(
            "Number of children in the party (ages 2-12)."
        ),
    )

    infants: Optional[int] = Field(
        default=None,
        description=(
            "Number of infants in the party."
        ),
    )

    pets: Optional[int] = Field(
        default=None,
        description=(
            "Number of pets travelling."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        description=(
            "ISO 4217 currency code the prices come back in. Default: USD."
        ),
    )


class ScavioAirbnbListing(BaseTool):  # type: ignore[override]
    """Airbnb: One Airbnb listing in full: description, capacity, the complete grouped
    amenity list, host profile, house rules, photo tour, and the RATING BREAKDOWN.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAirbnbListing

            tool = ScavioAirbnbListing()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"listing_id": "12345678"})
    """

    name: str = "scavio_airbnb_listing"
    description: str = (
        "Airbnb: One Airbnb listing in full: description, capacity, the complete "
        "grouped amenity list, host profile, house rules, photo tour, and the RATING "
        "BREAKDOWN. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioAirbnbListingInput
    handle_tool_error: bool = True

    api_wrapper: ScavioAirbnbListingAPIWrapper = Field(
        default_factory=ScavioAirbnbListingAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAirbnbListingAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        listing_id: str,
        check_in: Optional[str] = None,
        check_out: Optional[str] = None,
        adults: Optional[int] = None,
        children: Optional[int] = None,
        infants: Optional[int] = None,
        pets: Optional[int] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/airbnb/listing (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                listing_id=listing_id,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=children,
                infants=infants,
                pets=pets,
                currency=currency,
            )
            return self._process_response(raw, _first_identifier(listing_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        listing_id: str,
        check_in: Optional[str] = None,
        check_out: Optional[str] = None,
        adults: Optional[int] = None,
        children: Optional[int] = None,
        infants: Optional[int] = None,
        pets: Optional[int] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/airbnb/listing (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                listing_id=listing_id,
                check_in=check_in,
                check_out=check_out,
                adults=adults,
                children=children,
                infants=infants,
                pets=pets,
                currency=currency,
            )
            return self._process_response(raw, _first_identifier(listing_id))
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
                f"No Airbnb listing found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioAirbnbReviews
# --------------------------------------------------------------------------


class ScavioAirbnbReviewsInput(BaseModel):
    """Input schema for the ScavioAirbnbReviews tool."""

    model_config = ConfigDict(extra="allow")

    listing_id: str = Field(
        description=(
            "Airbnb listing id or a full /rooms/ URL. Query parameters are discarded "
            "because they carry someone else's dates."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        description=(
            "ISO 4217 currency code the prices come back in. Default: USD."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Reviews per page, 1-50. Airbnb returns a fixed 7 when no explicit limit "
            "is sent, so always set it. Default: 30."
        ),
    )

    offset: Optional[int] = Field(
        default=None,
        description=(
            "Zero-based review offset for paging. Default: 0."
        ),
    )


class ScavioAirbnbReviews(BaseTool):  # type: ignore[override]
    """Airbnb: Airbnb review BODIES with per-review rating, date, and reviewer
    name/photo/location.

    Costs 1 credit per call.

    Pagination: limit + offset.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAirbnbReviews

            tool = ScavioAirbnbReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"listing_id": "12345678", "limit": 30})
    """

    name: str = "scavio_airbnb_reviews"
    description: str = (
        "Airbnb: Airbnb review BODIES with per-review rating, date, and reviewer "
        "name/photo/location. Pagination: limit + offset. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioAirbnbReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioAirbnbReviewsAPIWrapper = Field(
        default_factory=ScavioAirbnbReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAirbnbReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        listing_id: str,
        currency: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/airbnb/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                listing_id=listing_id,
                currency=currency,
                limit=limit,
                offset=offset,
            )
            return self._process_response(raw, _first_identifier(listing_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        listing_id: str,
        currency: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/airbnb/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                listing_id=listing_id,
                currency=currency,
                limit=limit,
                offset=offset,
            )
            return self._process_response(raw, _first_identifier(listing_id))
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
                f"No Airbnb reviews found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw
