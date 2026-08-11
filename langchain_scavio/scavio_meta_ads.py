"""Scavio Meta Ad Library tools for LangChain agents.

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
    ScavioMetaAdsAdAPIWrapper,
    ScavioMetaAdsAdvertiserAPIWrapper,
    ScavioMetaAdsSearchAPIWrapper,
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
# ScavioMetaAdsSearch
# --------------------------------------------------------------------------


class ScavioMetaAdsSearchInput(BaseModel):
    """Input schema for the ScavioMetaAdsSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Keyword, brand or advertiser name to search the library for."
        ),
    )

    country: Optional[str] = Field(
        default=None,
        description=(
            "Two-letter country code for the ad library storefront. Default: US."
        ),
    )

    active_status: Optional[Literal["all", "active", "inactive"]] = Field(
        default=None,
        description=(
            "Whether to return running, stopped or all ads. Options: all, active, "
            "inactive. Default: all."
        ),
    )

    ad_type: Optional[Literal["all", "political_and_issue_ads"]] = Field(
        default=None,
        description=(
            "Set to political_and_issue_ads to expose spend, reach, impressions and "
            "the paid-for-by disclosure. Commercial ads leave those null. Options: "
            "all, political_and_issue_ads. Default: all."
        ),
    )

    media_type: Optional[
        Literal["all", "image", "video", "meme", "image_and_meme", "none"]
    ] = Field(
        default=None,
        description=(
            "Creative media type filter. Options: all, image, video, meme, "
            "image_and_meme, none."
        ),
    )

    search_type: Optional[Literal["keyword_unordered", "keyword_exact_phrase"]] = Field(
        default=None,
        description=(
            "Whether the query is matched as an exact phrase. Options: "
            "keyword_unordered, keyword_exact_phrase. Default: keyword_unordered."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "next_cursor from the previous response. Page 1 is 30 ads, then 10 per "
            "page. ALL OTHER FILTERS ARE IGNORED when a cursor is present -- the "
            "cursor already carries them."
        ),
    )


class ScavioMetaAdsSearch(BaseTool):  # type: ignore[override]
    """Meta Ad Library: Search the Meta Ad Library: 30 ads on page 1 with full creative
    (page name, ad copy, headline, CTA, images and videos, platforms, run dates),
    then cursor-paginated.

    Costs 1 credit per call.

    Pagination: cursor -> next_cursor. Page 1 is 30 ads, then 10 per page; walk
    has_next_page to read a whole query.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioMetaAdsSearch

            tool = ScavioMetaAdsSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "running shoes", "country": "GB"})
    """

    name: str = "scavio_meta_ads_search"
    description: str = (
        "Meta Ad Library: Search the Meta Ad Library: 30 ads on page 1 with full "
        "creative (page name, ad copy, headline, CTA, images and videos, platforms, "
        "run dates), then cursor-paginated. Pagination: cursor -> next_cursor. Page 1 "
        "is 30 ads, then 10 per page; walk has_next_page to read a whole query. Costs "
        "1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioMetaAdsSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioMetaAdsSearchAPIWrapper = Field(
        default_factory=ScavioMetaAdsSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioMetaAdsSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        country: Optional[str] = None,
        active_status: Optional[Literal["all", "active", "inactive"]] = None,
        ad_type: Optional[Literal["all", "political_and_issue_ads"]] = None,
        media_type: Optional[
            Literal["all", "image", "video", "meme", "image_and_meme", "none"]
        ] = None,
        search_type: Optional[
            Literal["keyword_unordered", "keyword_exact_phrase"]
        ] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/meta-ads/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                country=country,
                active_status=active_status,
                ad_type=ad_type,
                media_type=media_type,
                search_type=search_type,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        country: Optional[str] = None,
        active_status: Optional[Literal["all", "active", "inactive"]] = None,
        ad_type: Optional[Literal["all", "political_and_issue_ads"]] = None,
        media_type: Optional[
            Literal["all", "image", "video", "meme", "image_and_meme", "none"]
        ] = None,
        search_type: Optional[
            Literal["keyword_unordered", "keyword_exact_phrase"]
        ] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/meta-ads/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                country=country,
                active_status=active_status,
                ad_type=ad_type,
                media_type=media_type,
                search_type=search_type,
                cursor=cursor,
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
                f"No Meta Ad Library results found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioMetaAdsAdvertiser
# --------------------------------------------------------------------------


class ScavioMetaAdsAdvertiserInput(BaseModel):
    """Input schema for the ScavioMetaAdsAdvertiser tool."""

    model_config = ConfigDict(extra="allow")

    page_id: str = Field(
        description=(
            "The advertiser's numeric Facebook Page id."
        ),
    )

    country: Optional[str] = Field(
        default=None,
        description=(
            "Two-letter country code for the ad library storefront. Default: US."
        ),
    )

    active_status: Optional[Literal["all", "active", "inactive"]] = Field(
        default=None,
        description=(
            "Whether to return running, stopped or all ads. Options: all, active, "
            "inactive. Default: all."
        ),
    )

    ad_type: Optional[Literal["all", "political_and_issue_ads"]] = Field(
        default=None,
        description=(
            "Set to political_and_issue_ads to expose spend, reach, impressions and "
            "the paid-for-by disclosure. Commercial ads leave those null. Options: "
            "all, political_and_issue_ads. Default: all."
        ),
    )

    media_type: Optional[
        Literal["all", "image", "video", "meme", "image_and_meme", "none"]
    ] = Field(
        default=None,
        description=(
            "Creative media type filter. Options: all, image, video, meme, "
            "image_and_meme, none."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "next_cursor from the previous response. Page 1 is 30 ads, then 10 per "
            "page."
        ),
    )


class ScavioMetaAdsAdvertiser(BaseTool):  # type: ignore[override]
    """Meta Ad Library: Every ad a Facebook Page is running, by numeric page id -- 30
    ads on page 1 with the same creative detail as search, cursor-paginated at 10
    per page thereafter.

    Costs 1 credit per call.

    Pagination: cursor -> next_cursor; page 1 = 30 ads, then 10 per page. Walk
    has_next_page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioMetaAdsAdvertiser

            tool = ScavioMetaAdsAdvertiser()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"page_id": "20531316728"})
    """

    name: str = "scavio_meta_ads_advertiser"
    description: str = (
        "Meta Ad Library: Every ad a Facebook Page is running, by numeric page id -- "
        "30 ads on page 1 with the same creative detail as search, cursor-paginated at "
        "10 per page thereafter. Pagination: cursor -> next_cursor; page 1 = 30 ads, "
        "then 10 per page. Walk has_next_page. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioMetaAdsAdvertiserInput
    handle_tool_error: bool = True

    api_wrapper: ScavioMetaAdsAdvertiserAPIWrapper = Field(
        default_factory=ScavioMetaAdsAdvertiserAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioMetaAdsAdvertiserAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        page_id: str,
        country: Optional[str] = None,
        active_status: Optional[Literal["all", "active", "inactive"]] = None,
        ad_type: Optional[Literal["all", "political_and_issue_ads"]] = None,
        media_type: Optional[
            Literal["all", "image", "video", "meme", "image_and_meme", "none"]
        ] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/meta-ads/advertiser (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                page_id=page_id,
                country=country,
                active_status=active_status,
                ad_type=ad_type,
                media_type=media_type,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(page_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        page_id: str,
        country: Optional[str] = None,
        active_status: Optional[Literal["all", "active", "inactive"]] = None,
        ad_type: Optional[Literal["all", "political_and_issue_ads"]] = None,
        media_type: Optional[
            Literal["all", "image", "video", "meme", "image_and_meme", "none"]
        ] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/meta-ads/advertiser (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                page_id=page_id,
                country=country,
                active_status=active_status,
                ad_type=ad_type,
                media_type=media_type,
                cursor=cursor,
            )
            return self._process_response(raw, _first_identifier(page_id))
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
                f"No Meta Ad Library ads found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioMetaAdsAd
# --------------------------------------------------------------------------


class ScavioMetaAdsAdInput(BaseModel):
    """Input schema for the ScavioMetaAdsAd tool."""

    model_config = ConfigDict(extra="allow")

    ad_archive_id: str = Field(
        description=(
            "The ad's numeric archive id."
        ),
    )


class ScavioMetaAdsAd(BaseTool):  # type: ignore[override]
    """Meta Ad Library: One Meta ad in full by archive id: creative, advertiser, run
    dates, platforms, any political disclosure.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioMetaAdsAd

            tool = ScavioMetaAdsAd()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"ad_archive_id": "1234567890123456"})
    """

    name: str = "scavio_meta_ads_ad"
    description: str = (
        "Meta Ad Library: One Meta ad in full by archive id: creative, advertiser, run "
        "dates, platforms, any political disclosure. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioMetaAdsAdInput
    handle_tool_error: bool = True

    api_wrapper: ScavioMetaAdsAdAPIWrapper = Field(
        default_factory=ScavioMetaAdsAdAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioMetaAdsAdAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        ad_archive_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/meta-ads/ad (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                ad_archive_id=ad_archive_id,
            )
            return self._process_response(raw, _first_identifier(ad_archive_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        ad_archive_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/meta-ads/ad (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                ad_archive_id=ad_archive_id,
            )
            return self._process_response(raw, _first_identifier(ad_archive_id))
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
                f"No Meta Ad Library ad found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw
