"""Scavio Booking.com tools for LangChain agents.

Every URL, parameter name, enum and credit cost below is copied from the
Scavio route definition rather than derived from the tool name.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import (
    ScavioBookingHotelAPIWrapper,
    ScavioBookingReviewsAPIWrapper,
    ScavioBookingSearchAPIWrapper,
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
# ScavioBookingSearch
# --------------------------------------------------------------------------


class ScavioBookingSearchInput(BaseModel):
    """Input schema for the ScavioBookingSearch tool."""

    model_config = ConfigDict(extra="allow")

    destination: Optional[str] = Field(
        default=None,
        description=(
            "Destination name. Either destination or dest_id is required -- a search "
            "with neither returns Booking's homepage and still costs a credit."
        ),
    )

    dest_id: Optional[str] = Field(
        default=None,
        description=(
            "Booking's numeric destination id."
        ),
    )

    dest_type: Optional[
        Literal["city", "region", "country", "district", "landmark", "airport", "hotel"]
    ] = Field(
        default=None,
        description=(
            "What dest_id refers to. Requires dest_id. Options: city, region, country, "
            "district, landmark, airport, hotel."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 25 properties per page."
        ),
    )

    sort_by: Optional[
        Literal[
            "popularity",
            "price_low",
            "price_high",
            "stars_high",
            "stars_low",
            "stars_and_price",
            "distance",
            "review_score",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: popularity, price_low, price_high, "
            "stars_high, stars_low, stars_and_price, distance, review_score. Default: "
            "popularity."
        ),
    )

    min_price: Optional[float] = Field(
        default=None,
        description=(
            "Minimum price PER NIGHT, in `currency`."
        ),
    )

    max_price: Optional[float] = Field(
        default=None,
        description=(
            "Maximum price PER NIGHT, in `currency`."
        ),
    )

    stars: Optional[list[int]] = Field(
        default=None,
        description=(
            "Star ratings to include. Values are OR'd together."
        ),
    )

    min_review_score: Optional[Literal["6", "7", "8", "9"]] = Field(
        default=None,
        description=(
            "Minimum guest review score. Only 6, 7, 8 and 9 are accepted -- any other "
            "threshold is silently dropped upstream. Options: 6, 7, 8, 9."
        ),
    )

    property_type: Optional[Union[str, int]] = Field(
        default=None,
        description=(
            "Accommodation type: one of the named values, or a raw numeric Booking "
            "accommodation-type id. Options: apartments, hostels, hotels, motels, "
            "resorts, bed_and_breakfasts, villas, campgrounds, vacation_homes, lodges, "
            "homestays."
        ),
    )

    free_cancellation: Optional[bool] = Field(
        default=None,
        description=(
            "Only return rates with free cancellation."
        ),
    )

    no_prepayment: Optional[bool] = Field(
        default=None,
        description=(
            "Only return rates with no prepayment."
        ),
    )

    breakfast_included: Optional[bool] = Field(
        default=None,
        description=(
            "Only return rates that include breakfast."
        ),
    )

    checkin: Optional[str] = Field(
        default=None,
        description=(
            "Check-in date, YYYY-MM-DD. Must be sent together with checkout."
        ),
    )

    checkout: Optional[str] = Field(
        default=None,
        description=(
            "Check-out date, YYYY-MM-DD. Must be sent together with checkin."
        ),
    )

    adults: Optional[int] = Field(
        default=None,
        description=(
            "Number of adult guests. Default: 2."
        ),
    )

    children_ages: Optional[list[int]] = Field(
        default=None,
        description=(
            "Ages of the children in the party, one entry per child. Ages, not a "
            "count."
        ),
    )

    rooms: Optional[int] = Field(
        default=None,
        description=(
            "Number of rooms required. Default: 1."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        description=(
            "ISO 4217 currency code the prices come back in. Default: USD."
        ),
    )


class ScavioBookingSearch(BaseTool):  # type: ignore[override]
    """Booking.com: Booking.com properties for a destination and stay: live nightly
    price, review score, star rating, room type, deal badges.

    Costs 1 credit per call.

    Pagination: page -- 25 properties per page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioBookingSearch

            tool = ScavioBookingSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "destination": "Lisbon",
                    "checkin": "2026-09-10",
                    "checkout": "2026-09-13",
                }
            )
    """

    name: str = "scavio_booking_search"
    description: str = (
        "Booking.com: Booking.com properties for a destination and stay: live nightly "
        "price, review score, star rating, room type, deal badges. Pagination: page -- "
        "25 properties per page. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioBookingSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioBookingSearchAPIWrapper = Field(
        default_factory=ScavioBookingSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioBookingSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        destination: Optional[str] = None,
        dest_id: Optional[str] = None,
        dest_type: Optional[
            Literal[
                "city",
                "region",
                "country",
                "district",
                "landmark",
                "airport",
                "hotel",
            ]
        ] = None,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal[
                "popularity",
                "price_low",
                "price_high",
                "stars_high",
                "stars_low",
                "stars_and_price",
                "distance",
                "review_score",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        stars: Optional[list[int]] = None,
        min_review_score: Optional[Literal["6", "7", "8", "9"]] = None,
        property_type: Optional[Union[str, int]] = None,
        free_cancellation: Optional[bool] = None,
        no_prepayment: Optional[bool] = None,
        breakfast_included: Optional[bool] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
        adults: Optional[int] = None,
        children_ages: Optional[list[int]] = None,
        rooms: Optional[int] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/booking/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                destination=destination,
                dest_id=dest_id,
                dest_type=dest_type,
                page=page,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
                stars=stars,
                min_review_score=min_review_score,
                property_type=property_type,
                free_cancellation=free_cancellation,
                no_prepayment=no_prepayment,
                breakfast_included=breakfast_included,
                checkin=checkin,
                checkout=checkout,
                adults=adults,
                children_ages=children_ages,
                rooms=rooms,
                currency=currency,
            )
            return self._process_response(
                raw, _first_identifier(destination, dest_id, checkin)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        destination: Optional[str] = None,
        dest_id: Optional[str] = None,
        dest_type: Optional[
            Literal[
                "city",
                "region",
                "country",
                "district",
                "landmark",
                "airport",
                "hotel",
            ]
        ] = None,
        page: Optional[int] = None,
        sort_by: Optional[
            Literal[
                "popularity",
                "price_low",
                "price_high",
                "stars_high",
                "stars_low",
                "stars_and_price",
                "distance",
                "review_score",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        stars: Optional[list[int]] = None,
        min_review_score: Optional[Literal["6", "7", "8", "9"]] = None,
        property_type: Optional[Union[str, int]] = None,
        free_cancellation: Optional[bool] = None,
        no_prepayment: Optional[bool] = None,
        breakfast_included: Optional[bool] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
        adults: Optional[int] = None,
        children_ages: Optional[list[int]] = None,
        rooms: Optional[int] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/booking/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                destination=destination,
                dest_id=dest_id,
                dest_type=dest_type,
                page=page,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
                stars=stars,
                min_review_score=min_review_score,
                property_type=property_type,
                free_cancellation=free_cancellation,
                no_prepayment=no_prepayment,
                breakfast_included=breakfast_included,
                checkin=checkin,
                checkout=checkout,
                adults=adults,
                children_ages=children_ages,
                rooms=rooms,
                currency=currency,
            )
            return self._process_response(
                raw, _first_identifier(destination, dest_id, checkin)
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
                f"No Booking.com results found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioBookingHotel
# --------------------------------------------------------------------------


class ScavioBookingHotelInput(BaseModel):
    """Input schema for the ScavioBookingHotel tool."""

    model_config = ConfigDict(extra="allow")

    hotel: str = Field(
        description=(
            "booking.com property URL or the bare page slug. Query parameters are "
            "discarded."
        ),
    )

    country_code: Optional[str] = Field(
        default=None,
        description=(
            "Two-letter country code. Only consulted when `hotel` is a bare slug; a "
            "wrong one is a real, BILLED 404. Default: us."
        ),
    )

    checkin: Optional[str] = Field(
        default=None,
        description=(
            "Check-in date, YYYY-MM-DD. Must be sent together with checkout."
        ),
    )

    checkout: Optional[str] = Field(
        default=None,
        description=(
            "Check-out date, YYYY-MM-DD. Must be sent together with checkin."
        ),
    )

    adults: Optional[int] = Field(
        default=None,
        description=(
            "Number of adult guests. Default: 2."
        ),
    )

    children_ages: Optional[list[int]] = Field(
        default=None,
        description=(
            "Ages of the children in the party, one entry per child."
        ),
    )

    rooms: Optional[int] = Field(
        default=None,
        description=(
            "Number of rooms required. Default: 1."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        description=(
            "ISO 4217 currency code the prices come back in. Default: USD."
        ),
    )


class ScavioBookingHotel(BaseTool):  # type: ignore[override]
    """Booking.com: One Booking.com property in full: rooms and rate plans, facilities,
    house rules, policies, images, review scores.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioBookingHotel

            tool = ScavioBookingHotel()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "hotel": "memmo-alfama",
                    "checkin": "2026-09-10",
                    "checkout": "2026-09-13",
                }
            )
    """

    name: str = "scavio_booking_hotel"
    description: str = (
        "Booking.com: One Booking.com property in full: rooms and rate plans, "
        "facilities, house rules, policies, images, review scores. Costs 1 credit per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioBookingHotelInput
    handle_tool_error: bool = True

    api_wrapper: ScavioBookingHotelAPIWrapper = Field(
        default_factory=ScavioBookingHotelAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioBookingHotelAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        hotel: str,
        country_code: Optional[str] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
        adults: Optional[int] = None,
        children_ages: Optional[list[int]] = None,
        rooms: Optional[int] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/booking/hotel (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                hotel=hotel,
                country_code=country_code,
                checkin=checkin,
                checkout=checkout,
                adults=adults,
                children_ages=children_ages,
                rooms=rooms,
                currency=currency,
            )
            return self._process_response(raw, _first_identifier(hotel))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        hotel: str,
        country_code: Optional[str] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
        adults: Optional[int] = None,
        children_ages: Optional[list[int]] = None,
        rooms: Optional[int] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/booking/hotel (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                hotel=hotel,
                country_code=country_code,
                checkin=checkin,
                checkout=checkout,
                adults=adults,
                children_ages=children_ages,
                rooms=rooms,
                currency=currency,
            )
            return self._process_response(raw, _first_identifier(hotel))
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
                f"No Booking.com property found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioBookingReviews
# --------------------------------------------------------------------------


class ScavioBookingReviewsInput(BaseModel):
    """Input schema for the ScavioBookingReviews tool."""

    model_config = ConfigDict(extra="allow")

    hotel: str = Field(
        description=(
            "booking.com property URL or the bare page slug. Query parameters are "
            "discarded."
        ),
    )

    country_code: Optional[str] = Field(
        default=None,
        description=(
            "Two-letter country code. Only consulted when `hotel` is a bare slug; a "
            "wrong one is a real, BILLED 404. Default: us."
        ),
    )

    checkin: Optional[str] = Field(
        default=None,
        description=(
            "Check-in date, YYYY-MM-DD. Must be sent together with checkout."
        ),
    )

    checkout: Optional[str] = Field(
        default=None,
        description=(
            "Check-out date, YYYY-MM-DD. Must be sent together with checkin."
        ),
    )

    adults: Optional[int] = Field(
        default=None,
        description=(
            "Number of adult guests. Default: 2."
        ),
    )

    children_ages: Optional[list[int]] = Field(
        default=None,
        description=(
            "Ages of the children in the party, one entry per child."
        ),
    )

    rooms: Optional[int] = Field(
        default=None,
        description=(
            "Number of rooms required. Default: 1."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        description=(
            "ISO 4217 currency code the prices come back in. Default: USD."
        ),
    )


class ScavioBookingReviews(BaseTool):  # type: ignore[override]
    """Booking.com: Booking.com guest reviews with the score breakdown by category and
    Booking's own praise/complaint summary.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioBookingReviews

            tool = ScavioBookingReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"hotel": "memmo-alfama"})
    """

    name: str = "scavio_booking_reviews"
    description: str = (
        "Booking.com: Booking.com guest reviews with the score breakdown by category "
        "and Booking's own praise/complaint summary. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioBookingReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioBookingReviewsAPIWrapper = Field(
        default_factory=ScavioBookingReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioBookingReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        hotel: str,
        country_code: Optional[str] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
        adults: Optional[int] = None,
        children_ages: Optional[list[int]] = None,
        rooms: Optional[int] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/booking/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                hotel=hotel,
                country_code=country_code,
                checkin=checkin,
                checkout=checkout,
                adults=adults,
                children_ages=children_ages,
                rooms=rooms,
                currency=currency,
            )
            return self._process_response(raw, _first_identifier(hotel))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        hotel: str,
        country_code: Optional[str] = None,
        checkin: Optional[str] = None,
        checkout: Optional[str] = None,
        adults: Optional[int] = None,
        children_ages: Optional[list[int]] = None,
        rooms: Optional[int] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/booking/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                hotel=hotel,
                country_code=country_code,
                checkin=checkin,
                checkout=checkout,
                adults=adults,
                children_ages=children_ages,
                rooms=rooms,
                currency=currency,
            )
            return self._process_response(raw, _first_identifier(hotel))
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
                f"No Booking.com reviews found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw
