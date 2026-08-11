"""Scavio Google Ads Transparency tools for LangChain agents.

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
    ScavioGoogleAdsAdvertisersAPIWrapper,
    ScavioGoogleAdsCreativeAPIWrapper,
    ScavioGoogleAdsSearchAPIWrapper,
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
# ScavioGoogleAdsAdvertisers
# --------------------------------------------------------------------------


class ScavioGoogleAdsAdvertisersInput(BaseModel):
    """Input schema for the ScavioGoogleAdsAdvertisers tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Advertiser name or domain to resolve."
        ),
    )

    region: Optional[str] = Field(
        default=None,
        description=(
            "ISO alpha-2 country (US, GB, DE) or a Google geo criteria id as a string. "
            "It also scopes the deep links on every row. Default: worldwide."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Rows per arm, 1-20. Advertisers and domains are capped separately, so a "
            "name query can return up to twice this many rows. Default: 10."
        ),
    )


class ScavioGoogleAdsAdvertisers(BaseTool):  # type: ignore[override]
    """Google Ads Transparency: START HERE. Resolve a brand name or domain to the
    advertiser_id /search and /creative are keyed by. Returns `advertiser` rows (id,
    verified name, verification country, total ad count as a RANGE) and `domain`
    rows.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleAdsAdvertisers

            tool = ScavioGoogleAdsAdvertisers()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "Stripe"})
    """

    name: str = "scavio_google_ads_advertisers"
    description: str = (
        "Google Ads Transparency: START HERE. Resolve a brand name or domain to the "
        "advertiser_id /search and /creative are keyed by. Returns `advertiser` rows "
        "(id, verified name, verification country, total ad count as a RANGE) and "
        "`domain` rows. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleAdsAdvertisersInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleAdsAdvertisersAPIWrapper = Field(
        default_factory=ScavioGoogleAdsAdvertisersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleAdsAdvertisersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        region: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleads/advertisers (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                region=region,
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
        region: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleads/advertisers (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                region=region,
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
                f"No Google Ads Transparency advertisers found for '{identifier}'. "
                "Resolve the advertiser with ScavioGoogleAdsAdvertisers first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioGoogleAdsSearch
# --------------------------------------------------------------------------


class ScavioGoogleAdsSearchInput(BaseModel):
    """Input schema for the ScavioGoogleAdsSearch tool."""

    model_config = ConfigDict(extra="allow")

    domain: Optional[str] = Field(
        default=None,
        description=(
            "Advertiser website: bare host, www host or full URL, reduced to the "
            "registrable host. This is the ONLY way to get the `domain` field back on "
            "each row."
        ),
    )

    advertiser_id: Optional[str] = Field(
        default=None,
        description=(
            "Google advertiser id, e.g. AR16735076323512287233. The shape is checked "
            "before any request, so a typo costs nothing."
        ),
    )

    region: Optional[str] = Field(
        default=None,
        description=(
            "ISO alpha-2 country (US, GB, DE) or a Google geo criteria id as a string. "
            "It also scopes the deep links on every row. Default: worldwide."
        ),
    )

    format: Optional[Literal["text", "image", "video"]] = Field(
        default=None,
        description=(
            "Creative format. The three sets are DISJOINT -- an advertiser's text, "
            "image and video ads share no creatives. Default: all formats. Options: "
            "text, image, video."
        ),
    )

    platform: Optional[
        Literal["play", "maps", "search", "shopping", "youtube"]
    ] = Field(
        default=None,
        description=(
            "Surface the ad ran on. Default: all surfaces. Options: play, maps, "
            "search, shopping, youtube."
        ),
    )

    topic: Optional[Literal["all", "political"]] = Field(
        default=None,
        description=(
            "Ad topic filter. Options: all, political. Default: all."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Creatives per page, 1-100. 100 is a HARD UPSTREAM CEILING, not our "
            "policy: Google answers a larger request with ZERO rows rather than an "
            "error. Default: 40."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "next_cursor from the previous response. Re-send the SAME filters "
            "alongside it. Null once the result set is exhausted."
        ),
    )


class ScavioGoogleAdsSearch(BaseTool):  # type: ignore[override]
    """Google Ads Transparency: Every ad Google is running for one advertiser: the
    creative, advertiser id and name, format, first/last seen dates, days actually
    run, plus total_ads_min/total_ads_max.

    Costs 1 credit per call.

    Pagination: cursor -> next_cursor, 100 creatives per page. Re-send the SAME
    filters alongside the cursor.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleAdsSearch

            tool = ScavioGoogleAdsSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"domain": "stripe.com", "region": "GB"})
    """

    name: str = "scavio_google_ads_search"
    description: str = (
        "Google Ads Transparency: Every ad Google is running for one advertiser: the "
        "creative, advertiser id and name, format, first/last seen dates, days "
        "actually run, plus total_ads_min/total_ads_max. Pagination: cursor -> "
        "next_cursor, 100 creatives per page. Re-send the SAME filters alongside the "
        "cursor. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleAdsSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleAdsSearchAPIWrapper = Field(
        default_factory=ScavioGoogleAdsSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleAdsSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        domain: Optional[str] = None,
        advertiser_id: Optional[str] = None,
        region: Optional[str] = None,
        format: Optional[Literal["text", "image", "video"]] = None,
        platform: Optional[
            Literal["play", "maps", "search", "shopping", "youtube"]
        ] = None,
        topic: Optional[Literal["all", "political"]] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleads/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                domain=domain,
                advertiser_id=advertiser_id,
                region=region,
                format=format,
                platform=platform,
                topic=topic,
                limit=limit,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(advertiser_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        domain: Optional[str] = None,
        advertiser_id: Optional[str] = None,
        region: Optional[str] = None,
        format: Optional[Literal["text", "image", "video"]] = None,
        platform: Optional[
            Literal["play", "maps", "search", "shopping", "youtube"]
        ] = None,
        topic: Optional[Literal["all", "political"]] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleads/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                domain=domain,
                advertiser_id=advertiser_id,
                region=region,
                format=format,
                platform=platform,
                topic=topic,
                limit=limit,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(advertiser_id))
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
                f"No Google Ads Transparency results found for '{identifier}'. Resolve "
                "the advertiser with ScavioGoogleAdsAdvertisers first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioGoogleAdsCreative
# --------------------------------------------------------------------------


class ScavioGoogleAdsCreativeInput(BaseModel):
    """Input schema for the ScavioGoogleAdsCreative tool."""

    model_config = ConfigDict(extra="allow")

    advertiser_id: str = Field(
        description=(
            "Google advertiser id owning the creative."
        ),
    )

    creative_id: str = Field(
        description=(
            "Creative id. It must belong to the advertiser_id sent with it -- the "
            "lookup is keyed by the pair and a mismatch is a 404."
        ),
    )


class ScavioGoogleAdsCreative(BaseTool):  # type: ignore[override]
    """Google Ads Transparency: One Google ad creative in full and the ONLY endpoint
    carrying its history: every size variation, the impression bucket, the per-
    region breakdown with first/last shown dates and a per-surface impression split,
    and the funder disclosure on political ads.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleAdsCreative

            tool = ScavioGoogleAdsCreative()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "advertiser_id": "AR16735076323512287233",
                    "creative_id": "CR12345678901234567890",
                }
            )
    """

    name: str = "scavio_google_ads_creative"
    description: str = (
        "Google Ads Transparency: One Google ad creative in full and the ONLY endpoint "
        "carrying its history: every size variation, the impression bucket, the "
        "per-region breakdown with first/last shown dates and a per-surface impression "
        "split, and the funder disclosure on political ads. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleAdsCreativeInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleAdsCreativeAPIWrapper = Field(
        default_factory=ScavioGoogleAdsCreativeAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleAdsCreativeAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        advertiser_id: str,
        creative_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleads/creative (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                advertiser_id=advertiser_id,
                creative_id=creative_id,
            )
            return self._process_response(
                raw, _first_identifier(advertiser_id, creative_id)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        advertiser_id: str,
        creative_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleads/creative (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                advertiser_id=advertiser_id,
                creative_id=creative_id,
            )
            return self._process_response(
                raw, _first_identifier(advertiser_id, creative_id)
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
                f"No Google Ads Transparency creative found for '{identifier}'. "
                "Resolve the advertiser with ScavioGoogleAdsAdvertisers first."
            )
        return raw
