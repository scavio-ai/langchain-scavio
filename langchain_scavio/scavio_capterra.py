"""Scavio Capterra tools for LangChain agents.

Every URL, parameter name, enum and credit cost below is copied from the
Scavio route definition rather than derived from the tool name.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import (
    ScavioCapterraProductAPIWrapper,
    ScavioCapterraReviewsAPIWrapper,
    ScavioCapterraSearchAPIWrapper,
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
# ScavioCapterraSearch
# --------------------------------------------------------------------------


class ScavioCapterraSearchInput(BaseModel):
    """Input schema for the ScavioCapterraSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: Optional[str] = Field(
        default=None,
        description=(
            "Software product or category. Required unless you pass a url: a term-less "
            "search serves a fixed popular-products list that has nothing to do with "
            "the caller."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full capterra.com/search URL. capterra.co.uk and capterra.com.br are "
            "accepted."
        ),
    )


class ScavioCapterraSearch(BaseTool):  # type: ignore[override]
    """Capterra: Search Capterra for B2B software: 20 ranked products with name, vendor
    description, rating, review count, logo, paid-placement flag; each row carries
    product_id and slug.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioCapterraSearch

            tool = ScavioCapterraSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "project management"})
    """

    name: str = "scavio_capterra_search"
    description: str = (
        "Capterra: Search Capterra for B2B software: 20 ranked products with name, "
        "vendor description, rating, review count, logo, paid-placement flag; each row "
        "carries product_id and slug. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioCapterraSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioCapterraSearchAPIWrapper = Field(
        default_factory=ScavioCapterraSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioCapterraSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/capterra/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                url=url,
            )
            return self._process_response(raw, _first_identifier(query, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/capterra/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                url=url,
            )
            return self._process_response(raw, _first_identifier(query, url))
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
                f"No Capterra results found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioCapterraProduct
# --------------------------------------------------------------------------


class ScavioCapterraProductInput(BaseModel):
    """Input schema for the ScavioCapterraProduct tool."""

    model_config = ConfigDict(extra="allow")

    product_id: Optional[str] = Field(
        default=None,
        description=(
            "The number in /p/186596/Notion/, as a STRING -- a JSON number is "
            "rejected."
        ),
    )

    slug: Optional[str] = Field(
        default=None,
        description=(
            "Product slug. Cosmetic here: /p/186596/Zzzjunk/ returns Notion's profile "
            "byte for byte."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full listing URL, usable instead of the id fields."
        ),
    )


class ScavioCapterraProduct(BaseTool):  # type: ignore[override]
    """Capterra: Full Capterra profile: per-star histogram and the four scored
    criteria, complete pricing table, every rated feature and integration, AI
    pros/cons with the quoted review, buyer profile, PLUS the 25 most recent
    reviews.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioCapterraProduct

            tool = ScavioCapterraProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "186596"})
    """

    name: str = "scavio_capterra_product"
    description: str = (
        "Capterra: Full Capterra profile: per-star histogram and the four scored "
        "criteria, complete pricing table, every rated feature and integration, AI "
        "pros/cons with the quoted review, buyer profile, PLUS the 25 most recent "
        "reviews. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioCapterraProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioCapterraProductAPIWrapper = Field(
        default_factory=ScavioCapterraProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioCapterraProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: Optional[str] = None,
        slug: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/capterra/product (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
                slug=slug,
                url=url,
            )
            return self._process_response(raw, _first_identifier(product_id, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_id: Optional[str] = None,
        slug: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/capterra/product (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
                slug=slug,
                url=url,
            )
            return self._process_response(raw, _first_identifier(product_id, url))
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
                f"No Capterra product found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioCapterraReviews
# --------------------------------------------------------------------------


class ScavioCapterraReviewsInput(BaseModel):
    """Input schema for the ScavioCapterraReviews tool."""

    model_config = ConfigDict(extra="allow")

    product_id: Optional[str] = Field(
        default=None,
        description=(
            "The number in /p/186596/Notion/, as a STRING -- a JSON number is "
            "rejected."
        ),
    )

    slug: Optional[str] = Field(
        default=None,
        description=(
            "Product slug. LOAD-BEARING here: it is case-sensitive upstream and a "
            "wrong one silently serves PAGE ONE under a billed 200. Pass back the slug "
            "from search or product."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Passing back reviews_url from the product endpoint is the reliable way to "
            "page."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-100, at 25 reviews each. Past page 100 Capterra answers "
            "200 with page ONE."
        ),
    )


class ScavioCapterraReviews(BaseTool):  # type: ignore[override]
    """Capterra: A page of Capterra reviews: overall score plus five per-criterion
    scores, pros, cons, advice, usage duration, alternatives considered, vendor
    response -- plus a rich competitor list with rating histograms and starting
    prices.

    Costs 2 credits per call.

    Pagination: page -- 25 per page, capped at page 100 (CAPTERRA_MAX_REVIEW_PAGE);
    past it Capterra answers 200 with PAGE ONE and the page quietly dropped from the
    canonical.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioCapterraReviews

            tool = ScavioCapterraReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "186596", "slug": "Notion", "page": 2})
    """

    name: str = "scavio_capterra_reviews"
    description: str = (
        "Capterra: A page of Capterra reviews: overall score plus five per-criterion "
        "scores, pros, cons, advice, usage duration, alternatives considered, vendor "
        "response -- plus a rich competitor list with rating histograms and starting "
        "prices. Pagination: page -- 25 per page, capped at page 100 "
        "(CAPTERRA_MAX_REVIEW_PAGE); past it Capterra answers 200 with PAGE ONE and "
        "the page quietly dropped from the canonical. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioCapterraReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioCapterraReviewsAPIWrapper = Field(
        default_factory=ScavioCapterraReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioCapterraReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: Optional[str] = None,
        slug: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/capterra/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
                slug=slug,
                url=url,
                page=page,
            )
            return self._process_response(raw, _first_identifier(product_id, url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_id: Optional[str] = None,
        slug: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/capterra/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
                slug=slug,
                url=url,
                page=page,
            )
            return self._process_response(raw, _first_identifier(product_id, url))
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
                f"No Capterra reviews found for '{identifier}'. Try broadening the "
                "query or removing filters."
            )
        return raw
