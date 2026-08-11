"""Scavio Apple App Store tools for LangChain agents.

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
    ScavioAppStoreAppAPIWrapper,
    ScavioAppStoreReviewsAPIWrapper,
    ScavioAppStoreSearchAPIWrapper,
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
# ScavioAppStoreSearch
# --------------------------------------------------------------------------


class ScavioAppStoreSearchInput(BaseModel):
    """Input schema for the ScavioAppStoreSearch tool."""

    model_config = ConfigDict(extra="allow")

    term: str = Field(
        description=(
            "Search term. Matches the app name, a keyword, OR a publisher name -- "
            "searching a developer returns their catalogue."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Apps to return, 1-200. This is the ONLY lever on result volume: App Store "
            "search has no pagination and every offset spelling is silently ignored. "
            "Default: 25."
        ),
    )

    country: Optional[str] = Field(
        default=None,
        description=(
            "Two-letter storefront code. It decides price, currency, localised title "
            "and whether the app is sold there at all. Anything that is not exactly "
            "two letters silently falls back to us. Default: us."
        ),
    )

    entity: Optional[Literal["software", "ipad_software", "mac_software"]] = Field(
        default=None,
        description=(
            "Which App Store catalogue to search. Options: software, ipad_software, "
            "mac_software. Default: software."
        ),
    )

    lang: Optional[str] = Field(
        default=None,
        description=(
            "Five-letter locale, e.g. en_us. Independent of country: the storefront "
            "sets the price, this sets the words."
        ),
    )


class ScavioAppStoreSearch(BaseTool):  # type: ignore[override]
    """Apple App Store: Up to 200 fully-shaped App Store apps (the same 43-field row as
    /app). Doubles as a bulk metadata fetch and a publisher lookup.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAppStoreSearch

            tool = ScavioAppStoreSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"term": "notion", "limit": 25})
    """

    name: str = "scavio_app_store_search"
    description: str = (
        "Apple App Store: Up to 200 fully-shaped App Store apps (the same 43-field row "
        "as /app). Doubles as a bulk metadata fetch and a publisher lookup. Costs 1 "
        "credit per call."
    )
    args_schema: Type[BaseModel] = ScavioAppStoreSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioAppStoreSearchAPIWrapper = Field(
        default_factory=ScavioAppStoreSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAppStoreSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        term: str,
        limit: Optional[int] = None,
        country: Optional[str] = None,
        entity: Optional[Literal["software", "ipad_software", "mac_software"]] = None,
        lang: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/appstore/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                term=term,
                limit=limit,
                country=country,
                entity=entity,
                lang=lang,
            )
            return self._process_response(raw, _first_identifier(term))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        term: str,
        limit: Optional[int] = None,
        country: Optional[str] = None,
        entity: Optional[Literal["software", "ipad_software", "mac_software"]] = None,
        lang: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/appstore/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                term=term,
                limit=limit,
                country=country,
                entity=entity,
                lang=lang,
            )
            return self._process_response(raw, _first_identifier(term))
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
                f"No Apple App Store results found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioAppStoreApp
# --------------------------------------------------------------------------


class ScavioAppStoreAppInput(BaseModel):
    """Input schema for the ScavioAppStoreApp tool."""

    model_config = ConfigDict(extra="allow")

    app_id: str = Field(
        description=(
            "Numeric App Store id OR a bundle id (notion.id, com.burbn.instagram). A "
            "pasted apps.apple.com URL is rejected with a free 400."
        ),
    )

    country: Optional[str] = Field(
        default=None,
        description=(
            "Two-letter storefront code. It decides price, currency, localised title "
            "and whether the app is sold there at all. Anything that is not exactly "
            "two letters silently falls back to us. Default: us."
        ),
    )


class ScavioAppStoreApp(BaseTool):  # type: ignore[override]
    """Apple App Store: Full App Store listing: title, description, developer identity,
    price, all-time and current-version ratings, release notes, genres, advisories,
    screenshots, size, minimum OS, supported devices.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAppStoreApp

            tool = ScavioAppStoreApp()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"app_id": "1232780281"})
    """

    name: str = "scavio_app_store_app"
    description: str = (
        "Apple App Store: Full App Store listing: title, description, developer "
        "identity, price, all-time and current-version ratings, release notes, genres, "
        "advisories, screenshots, size, minimum OS, supported devices. Costs 1 credit "
        "per call."
    )
    args_schema: Type[BaseModel] = ScavioAppStoreAppInput
    handle_tool_error: bool = True

    api_wrapper: ScavioAppStoreAppAPIWrapper = Field(
        default_factory=ScavioAppStoreAppAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAppStoreAppAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        app_id: str,
        country: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/appstore/app (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                app_id=app_id,
                country=country,
            )
            return self._process_response(raw, _first_identifier(app_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        app_id: str,
        country: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/appstore/app (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                app_id=app_id,
                country=country,
            )
            return self._process_response(raw, _first_identifier(app_id))
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
                f"No Apple App Store app found for '{identifier}'. Verify the "
                "identifiers you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioAppStoreReviews
# --------------------------------------------------------------------------


class ScavioAppStoreReviewsInput(BaseModel):
    """Input schema for the ScavioAppStoreReviews tool."""

    model_config = ConfigDict(extra="allow")

    app_id: str = Field(
        description=(
            "Numeric App Store id. NUMERIC ONLY here -- the reviews feed has no "
            "bundle-id form."
        ),
    )

    country: Optional[str] = Field(
        default=None,
        description=(
            "Two-letter storefront code. It decides price, currency, localised title "
            "and whether the app is sold there at all. Anything that is not exactly "
            "two letters silently falls back to us. Default: us."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-10, at 50 reviews each. Apple hard-stops at page 10; reach "
            "further by asking a different country. Default: 1."
        ),
    )

    sort: Optional[Literal["most_recent", "most_helpful"]] = Field(
        default=None,
        description=(
            "Review sort order. Under most_recent almost every review is too new to "
            "have been voted on, so the vote fields come back as zeroes. Options: "
            "most_recent, most_helpful. Default: most_recent."
        ),
    )


class ScavioAppStoreReviews(BaseTool):  # type: ignore[override]
    """Apple App Store: A page of App Store reviews: star rating, title, full text,
    author, and the APP VERSION it was written against.

    Costs 1 credit per call.

    Pagination: page 1..10 at 50 per page -- a HARD STOP at page 10 (500 reviews per
    storefront is Apple's anonymous ceiling). Reach further by asking a different
    `country`.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioAppStoreReviews

            tool = ScavioAppStoreReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"app_id": "1232780281", "page": 1})
    """

    name: str = "scavio_app_store_reviews"
    description: str = (
        "Apple App Store: A page of App Store reviews: star rating, title, full text, "
        "author, and the APP VERSION it was written against. Pagination: page 1..10 at "
        "50 per page -- a HARD STOP at page 10 (500 reviews per storefront is Apple's "
        "anonymous ceiling). Reach further by asking a different `country`. Costs 1 "
        "credit per call."
    )
    args_schema: Type[BaseModel] = ScavioAppStoreReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioAppStoreReviewsAPIWrapper = Field(
        default_factory=ScavioAppStoreReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioAppStoreReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        app_id: str,
        country: Optional[str] = None,
        page: Optional[int] = None,
        sort: Optional[Literal["most_recent", "most_helpful"]] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/appstore/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                app_id=app_id,
                country=country,
                page=page,
                sort=sort,
            )
            return self._process_response(raw, _first_identifier(app_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        app_id: str,
        country: Optional[str] = None,
        page: Optional[int] = None,
        sort: Optional[Literal["most_recent", "most_helpful"]] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/appstore/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                app_id=app_id,
                country=country,
                page=page,
                sort=sort,
            )
            return self._process_response(raw, _first_identifier(app_id))
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
                f"No Apple App Store reviews found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw
