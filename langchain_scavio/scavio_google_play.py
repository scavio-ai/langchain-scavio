"""Scavio Google Play tools for LangChain agents.

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
    ScavioGooglePlayAppAPIWrapper,
    ScavioGooglePlayReviewsAPIWrapper,
    ScavioGooglePlaySearchAPIWrapper,
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
# ScavioGooglePlaySearch
# --------------------------------------------------------------------------


class ScavioGooglePlaySearchInput(BaseModel):
    """Input schema for the ScavioGooglePlaySearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Search query. There is no pagination -- one shelf of ~30 apps."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "Interface language. It moves the whole storefront, not only the strings: "
            "title, description, install formatting and content rating all follow it. "
            "Default: en."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Storefront country code. Default: us."
        ),
    )


class ScavioGooglePlaySearch(BaseTool):  # type: ignore[override]
    """Google Play: Ranked Google Play apps: package name, title, developer, rating,
    install count, price and IAP range, content rating, icon, screenshots.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGooglePlaySearch

            tool = ScavioGooglePlaySearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "notion"})
    """

    name: str = "scavio_google_play_search"
    description: str = (
        "Google Play: Ranked Google Play apps: package name, title, developer, rating, "
        "install count, price and IAP range, content rating, icon, screenshots. Costs "
        "2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioGooglePlaySearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGooglePlaySearchAPIWrapper = Field(
        default_factory=ScavioGooglePlaySearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGooglePlaySearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleplay/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                hl=hl,
                gl=gl,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleplay/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                hl=hl,
                gl=gl,
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
                f"No Google Play results found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioGooglePlayApp
# --------------------------------------------------------------------------


class ScavioGooglePlayAppInput(BaseModel):
    """Input schema for the ScavioGooglePlayApp tool."""

    model_config = ConfigDict(extra="allow")

    app_id: str = Field(
        description=(
            "Android package name, or any play.google.com URL carrying one in its id "
            "parameter."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "Interface language. It moves the whole storefront, not only the strings: "
            "title, description, install formatting and content rating all follow it. "
            "Default: en."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Storefront country code. Default: us."
        ),
    )


class ScavioGooglePlayApp(BaseTool):  # type: ignore[override]
    """Google Play: Full Google Play store listing: installs (incl. the REAL unrendered
    count), star histogram, developer identity, IAPs, permission tree, Data safety
    table, the 20 server-rendered reviews.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGooglePlayApp

            tool = ScavioGooglePlayApp()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"app_id": "notion.id"})
    """

    name: str = "scavio_google_play_app"
    description: str = (
        "Google Play: Full Google Play store listing: installs (incl. the REAL "
        "unrendered count), star histogram, developer identity, IAPs, permission tree, "
        "Data safety table, the 20 server-rendered reviews. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioGooglePlayAppInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGooglePlayAppAPIWrapper = Field(
        default_factory=ScavioGooglePlayAppAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGooglePlayAppAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        app_id: str,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleplay/app (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                app_id=app_id,
                hl=hl,
                gl=gl,
            )
            return self._process_response(raw, _first_identifier(app_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        app_id: str,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleplay/app (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                app_id=app_id,
                hl=hl,
                gl=gl,
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
                f"No Google Play app found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioGooglePlayReviews
# --------------------------------------------------------------------------


class ScavioGooglePlayReviewsInput(BaseModel):
    """Input schema for the ScavioGooglePlayReviews tool."""

    model_config = ConfigDict(extra="allow")

    app_id: str = Field(
        description=(
            "Android package name, or any play.google.com URL carrying one in its id "
            "parameter."
        ),
    )

    sort: Optional[Literal["relevance", "newest", "rating"]] = Field(
        default=None,
        description=(
            "Review sort order. Options: relevance, newest, rating. Default: newest."
        ),
    )

    count: Optional[int] = Field(
        default=None,
        description=(
            "Reviews to return, 1-200. Default: 50."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "next_cursor from a previous response. OPAQUE and SINGLE-USE, and it "
            "encodes the sort as well as the position -- send it back with the SAME "
            "sort it came from. A cursor past the last review is a 404."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "Interface language. It moves the whole storefront, not only the strings: "
            "title, description, install formatting and content rating all follow it. "
            "Default: en."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Storefront country code. Default: us."
        ),
    )


class ScavioGooglePlayReviews(BaseTool):  # type: ignore[override]
    """Google Play: A page of Google Play reviews: star score, full text, author,
    thumbs-up count, developer reply, and the APP VERSION the reviewer was running.

    Costs 2 credits per call.

    Pagination: cursor -> next_cursor. The cursor is OPAQUE, SINGLE-USE, and encodes
    the SORT as well as the position -- send it back with the SAME `sort` it came
    from. A cursor past the last review is a 404, not an empty page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGooglePlayReviews

            tool = ScavioGooglePlayReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"app_id": "notion.id"})
    """

    name: str = "scavio_google_play_reviews"
    description: str = (
        "Google Play: A page of Google Play reviews: star score, full text, author, "
        "thumbs-up count, developer reply, and the APP VERSION the reviewer was "
        "running. Pagination: cursor -> next_cursor. The cursor is OPAQUE, SINGLE-USE, "
        "and encodes the SORT as well as the position -- send it back with the SAME "
        "`sort` it came from. A cursor past the last review is a 404, not an empty "
        "page. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioGooglePlayReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGooglePlayReviewsAPIWrapper = Field(
        default_factory=ScavioGooglePlayReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGooglePlayReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        app_id: str,
        sort: Optional[Literal["relevance", "newest", "rating"]] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleplay/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                app_id=app_id,
                sort=sort,
                count=count,
                cursor=cursor,
                hl=hl,
                gl=gl,
            )
            return self._process_response(raw, _first_identifier(app_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        app_id: str,
        sort: Optional[Literal["relevance", "newest", "rating"]] = None,
        count: Optional[int] = None,
        cursor: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/googleplay/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                app_id=app_id,
                sort=sort,
                count=count,
                cursor=cursor,
                hl=hl,
                gl=gl,
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
                f"No Google Play reviews found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw
