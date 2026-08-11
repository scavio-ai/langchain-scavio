"""Scavio Redfin tools for LangChain agents.

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
    ScavioRedfinMarketAPIWrapper,
    ScavioRedfinPropertyAPIWrapper,
    ScavioRedfinSearchAPIWrapper,
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
# ScavioRedfinSearch
# --------------------------------------------------------------------------


class ScavioRedfinSearchInput(BaseModel):
    """Input schema for the ScavioRedfinSearch tool."""

    model_config = ConfigDict(extra="allow")

    location: Optional[str] = Field(
        default=None,
        description=(
            "A redfin.com region URL (/city/, /neighborhood/, /county/, /zipcode/) or "
            "a bare 5-digit ZIP. CITY NAMES ARE NOT ACCEPTED."
        ),
    )

    region_id: Optional[int] = Field(
        default=None,
        description=(
            "Redfin's own numeric region id. NOT a ZIP code -- different number "
            "spaces, and a ZIP here resolves to another city rather than failing. Must "
            "be sent together with region_type."
        ),
    )

    region_type: Optional[Literal[1, 2, 5, 6]] = Field(
        default=None,
        description=(
            "What region_id refers to: 1 neighborhood, 2 ZIP, 5 county, 6 city. Must "
            "be sent together with region_id. Options: 1, 2, 5, 6."
        ),
    )

    listing_status: Optional[Literal["for_sale", "sold", "for_rent"]] = Field(
        default=None,
        description=(
            "Which listing state to return. Options: for_sale, sold, for_rent. "
            "Default: for_sale."
        ),
    )

    sold_within_days: Optional[int] = Field(
        default=None,
        description=(
            "How far back to look for sold homes. Only valid with listing_status=sold, "
            "where it defaults to 90. Default: 90."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number, 1-based."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Listings per page, 1-350. Default: 100."
        ),
    )

    sort: Optional[
        Literal[
            "recommended",
            "price_low",
            "price_high",
            "newest",
            "oldest",
            "sqft_low",
            "sqft_high",
            "price_per_sqft_low",
            "price_per_sqft_high",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: recommended, price_low, price_high, "
            "newest, oldest, sqft_low, sqft_high, price_per_sqft_low, "
            "price_per_sqft_high. Default: recommended."
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

    baths_min: Optional[int] = Field(
        default=None,
        description=(
            "Minimum number of bathrooms. WHOLE baths only -- fractional bounds are "
            "rejected because Redfin truncates them."
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

    property_type: Optional[
        Literal["house", "condo", "townhouse", "multi_family", "land", "other", "co_op"]
    ] = Field(
        default=None,
        description=(
            "Property type filter. Options: house, condo, townhouse, multi_family, "
            "land, other, co_op."
        ),
    )

    has_pool: Optional[bool] = Field(
        default=None,
        description=(
            "Only return properties with a pool."
        ),
    )

    max_days_on_market: Optional[int] = Field(
        default=None,
        description=(
            "Maximum days on market. Cannot be combined with min_days_on_market: "
            "Redfin expresses both through one parameter."
        ),
    )

    min_days_on_market: Optional[int] = Field(
        default=None,
        description=(
            "Minimum days on market. Cannot be combined with max_days_on_market."
        ),
    )


class ScavioRedfinSearch(BaseTool):  # type: ignore[override]
    """Redfin: Redfin listings: price, price per sqft, beds, baths, living area, lot
    size, year built, coordinates, listing remarks, full photo galleries.

    Costs 1 credit per call.

    Pagination: page + limit -- up to 350 per page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedfinSearch

            tool = ScavioRedfinSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "location": "https://www.redfin.com/city/30749/TX/Austin",
                }
            )
    """

    name: str = "scavio_redfin_search"
    description: str = (
        "Redfin: Redfin listings: price, price per sqft, beds, baths, living area, lot "
        "size, year built, coordinates, listing remarks, full photo galleries. "
        "Pagination: page + limit -- up to 350 per page. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioRedfinSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioRedfinSearchAPIWrapper = Field(
        default_factory=ScavioRedfinSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedfinSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        location: Optional[str] = None,
        region_id: Optional[int] = None,
        region_type: Optional[Literal[1, 2, 5, 6]] = None,
        listing_status: Optional[Literal["for_sale", "sold", "for_rent"]] = None,
        sold_within_days: Optional[int] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        sort: Optional[
            Literal[
                "recommended",
                "price_low",
                "price_high",
                "newest",
                "oldest",
                "sqft_low",
                "sqft_high",
                "price_per_sqft_low",
                "price_per_sqft_high",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        beds_min: Optional[int] = None,
        beds_max: Optional[int] = None,
        baths_min: Optional[int] = None,
        sqft_min: Optional[int] = None,
        sqft_max: Optional[int] = None,
        lot_size_min: Optional[int] = None,
        year_built_min: Optional[int] = None,
        year_built_max: Optional[int] = None,
        max_hoa: Optional[float] = None,
        property_type: Optional[
            Literal[
                "house",
                "condo",
                "townhouse",
                "multi_family",
                "land",
                "other",
                "co_op",
            ]
        ] = None,
        has_pool: Optional[bool] = None,
        max_days_on_market: Optional[int] = None,
        min_days_on_market: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/redfin/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                location=location,
                region_id=region_id,
                region_type=region_type,
                listing_status=listing_status,
                sold_within_days=sold_within_days,
                page=page,
                limit=limit,
                sort=sort,
                min_price=min_price,
                max_price=max_price,
                beds_min=beds_min,
                beds_max=beds_max,
                baths_min=baths_min,
                sqft_min=sqft_min,
                sqft_max=sqft_max,
                lot_size_min=lot_size_min,
                year_built_min=year_built_min,
                year_built_max=year_built_max,
                max_hoa=max_hoa,
                property_type=property_type,
                has_pool=has_pool,
                max_days_on_market=max_days_on_market,
                min_days_on_market=min_days_on_market,
            )
            return self._process_response(raw, _first_identifier(location))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        location: Optional[str] = None,
        region_id: Optional[int] = None,
        region_type: Optional[Literal[1, 2, 5, 6]] = None,
        listing_status: Optional[Literal["for_sale", "sold", "for_rent"]] = None,
        sold_within_days: Optional[int] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        sort: Optional[
            Literal[
                "recommended",
                "price_low",
                "price_high",
                "newest",
                "oldest",
                "sqft_low",
                "sqft_high",
                "price_per_sqft_low",
                "price_per_sqft_high",
            ]
        ] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        beds_min: Optional[int] = None,
        beds_max: Optional[int] = None,
        baths_min: Optional[int] = None,
        sqft_min: Optional[int] = None,
        sqft_max: Optional[int] = None,
        lot_size_min: Optional[int] = None,
        year_built_min: Optional[int] = None,
        year_built_max: Optional[int] = None,
        max_hoa: Optional[float] = None,
        property_type: Optional[
            Literal[
                "house",
                "condo",
                "townhouse",
                "multi_family",
                "land",
                "other",
                "co_op",
            ]
        ] = None,
        has_pool: Optional[bool] = None,
        max_days_on_market: Optional[int] = None,
        min_days_on_market: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/redfin/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                location=location,
                region_id=region_id,
                region_type=region_type,
                listing_status=listing_status,
                sold_within_days=sold_within_days,
                page=page,
                limit=limit,
                sort=sort,
                min_price=min_price,
                max_price=max_price,
                beds_min=beds_min,
                beds_max=beds_max,
                baths_min=baths_min,
                sqft_min=sqft_min,
                sqft_max=sqft_max,
                lot_size_min=lot_size_min,
                year_built_min=year_built_min,
                year_built_max=year_built_max,
                max_hoa=max_hoa,
                property_type=property_type,
                has_pool=has_pool,
                max_days_on_market=max_days_on_market,
                min_days_on_market=min_days_on_market,
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
                f"No Redfin results found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioRedfinProperty
# --------------------------------------------------------------------------


class ScavioRedfinPropertyInput(BaseModel):
    """Input schema for the ScavioRedfinProperty tool."""

    model_config = ConfigDict(extra="allow")

    property_id: str = Field(
        description=(
            "Redfin property id or any redfin.com listing URL carrying one."
        ),
    )


class ScavioRedfinProperty(BaseTool):  # type: ignore[override]
    """Redfin: One Redfin listing in full: price, Redfin Estimate and rental estimate,
    complete MLS fact sheet, price and tax history, agents, schools, climate risk,
    comparable sales, photos.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedfinProperty

            tool = ScavioRedfinProperty()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"property_id": "170072526"})
    """

    name: str = "scavio_redfin_property"
    description: str = (
        "Redfin: One Redfin listing in full: price, Redfin Estimate and rental "
        "estimate, complete MLS fact sheet, price and tax history, agents, schools, "
        "climate risk, comparable sales, photos. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioRedfinPropertyInput
    handle_tool_error: bool = True

    api_wrapper: ScavioRedfinPropertyAPIWrapper = Field(
        default_factory=ScavioRedfinPropertyAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedfinPropertyAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        property_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/redfin/property (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                property_id=property_id,
            )
            return self._process_response(raw, _first_identifier(property_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        property_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/redfin/property (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                property_id=property_id,
            )
            return self._process_response(raw, _first_identifier(property_id))
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
                f"No Redfin property found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioRedfinMarket
# --------------------------------------------------------------------------


class ScavioRedfinMarketInput(BaseModel):
    """Input schema for the ScavioRedfinMarket tool."""

    model_config = ConfigDict(extra="allow")

    location: Optional[str] = Field(
        default=None,
        description=(
            "A redfin.com region URL (/city/, /neighborhood/, /county/, /zipcode/) or "
            "a bare 5-digit ZIP. CITY NAMES ARE NOT ACCEPTED."
        ),
    )

    region_id: Optional[int] = Field(
        default=None,
        description=(
            "Redfin's own numeric region id. NOT a ZIP code -- different number "
            "spaces, and a ZIP here resolves to another city rather than failing. Must "
            "be sent together with region_type."
        ),
    )

    region_type: Optional[Literal[1, 2, 5, 6]] = Field(
        default=None,
        description=(
            "What region_id refers to: 1 neighborhood, 2 ZIP, 5 county, 6 city. Must "
            "be sent together with region_id. Options: 1, 2, 5, 6."
        ),
    )


class ScavioRedfinMarket(BaseTool):  # type: ignore[override]
    """Redfin: Redfin housing-market stats for a region: median list and sale price,
    price per sqft, sale-to-list ratio, average offers and days on market, YoY
    movement, 0-100 compete score, live inventory.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioRedfinMarket

            tool = ScavioRedfinMarket()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "location": "https://www.redfin.com/city/30749/TX/Austin",
                }
            )
    """

    name: str = "scavio_redfin_market"
    description: str = (
        "Redfin: Redfin housing-market stats for a region: median list and sale price, "
        "price per sqft, sale-to-list ratio, average offers and days on market, YoY "
        "movement, 0-100 compete score, live inventory. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioRedfinMarketInput
    handle_tool_error: bool = True

    api_wrapper: ScavioRedfinMarketAPIWrapper = Field(
        default_factory=ScavioRedfinMarketAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioRedfinMarketAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        location: Optional[str] = None,
        region_id: Optional[int] = None,
        region_type: Optional[Literal[1, 2, 5, 6]] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/redfin/market (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                location=location,
                region_id=region_id,
                region_type=region_type,
            )
            return self._process_response(raw, _first_identifier(location))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        location: Optional[str] = None,
        region_id: Optional[int] = None,
        region_type: Optional[Literal[1, 2, 5, 6]] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/redfin/market (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                location=location,
                region_id=region_id,
                region_type=region_type,
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
                f"No Redfin market data found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw
