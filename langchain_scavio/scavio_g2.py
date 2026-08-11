"""Scavio G2 tools for LangChain agents.

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
    ScavioG2ProductAPIWrapper,
    ScavioG2ReviewsAPIWrapper,
    ScavioG2SearchAPIWrapper,
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
# ScavioG2Search
# --------------------------------------------------------------------------


class ScavioG2SearchInput(BaseModel):
    """Input schema for the ScavioG2Search tool."""

    model_config = ConfigDict(extra="allow")

    query: Optional[str] = Field(
        default=None,
        description=(
            "Software product or category to search for."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 20 per page unless limit says otherwise."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Products per page, 1-100. Capped at 100 so one request cannot ask for a "
            "multi-megabyte page. Default: 20."
        ),
    )

    sort: Optional[Literal["relevance", "popular", "alphabetical", "rating"]] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: relevance, popular, alphabetical, "
            "rating. Default: relevance."
        ),
    )

    rating: Optional[Literal[1, 2, 3, 4, 5]] = Field(
        default=None,
        description=(
            "Only return products at or above this star rating. Options: 1, 2, 3, 4, "
            "5."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full g2.com/search URL, usable instead of query."
        ),
    )


class ScavioG2Search(BaseTool):  # type: ignore[override]
    """G2: Search G2 for B2B software products: star rating, review count, vendor,
    categories, seller description, logo; each row carries product_id and slug.

    Costs 5 credits per call.

    Pagination: page + limit -- `limit` capped at 100 on our side; G2 itself keeps
    paginating at any size.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioG2Search

            tool = ScavioG2Search()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "project management"})
    """

    name: str = "scavio_g2_search"
    description: str = (
        "G2: Search G2 for B2B software products: star rating, review count, vendor, "
        "categories, seller description, logo; each row carries product_id and slug. "
        "Pagination: page + limit -- `limit` capped at 100 on our side; G2 itself "
        "keeps paginating at any size. Costs 5 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioG2SearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioG2SearchAPIWrapper = Field(
        default_factory=ScavioG2SearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioG2SearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: Optional[str] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        sort: Optional[
            Literal["relevance", "popular", "alphabetical", "rating"]
        ] = None,
        rating: Optional[Literal[1, 2, 3, 4, 5]] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/g2/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                page=page,
                limit=limit,
                sort=sort,
                rating=rating,
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
        page: Optional[int] = None,
        limit: Optional[int] = None,
        sort: Optional[
            Literal["relevance", "popular", "alphabetical", "rating"]
        ] = None,
        rating: Optional[Literal[1, 2, 3, 4, 5]] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/g2/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                page=page,
                limit=limit,
                sort=sort,
                rating=rating,
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
                f"No G2 results found for '{identifier}'. Try broadening the query or "
                "removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioG2Product
# --------------------------------------------------------------------------


class ScavioG2ProductInput(BaseModel):
    """Input schema for the ScavioG2Product tool."""

    model_config = ConfigDict(extra="allow")

    product_id: Optional[str] = Field(
        default=None,
        description=(
            "G2 slug (notion) or the numeric G2 id (82623) as a string. Both resolve "
            "on the same upstream path."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full listing URL, usable instead of the id fields."
        ),
    )


class ScavioG2Product(BaseTool):  # type: ignore[override]
    """G2: Full G2 software profile: rating and per-star histogram, vendor, pricing
    editions with parsed amounts, feature groups, integrations, alternatives,
    comparisons, and G2's AI-derived pros and cons.

    Costs 5 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioG2Product

            tool = ScavioG2Product()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "notion"})
    """

    name: str = "scavio_g2_product"
    description: str = (
        "G2: Full G2 software profile: rating and per-star histogram, vendor, pricing "
        "editions with parsed amounts, feature groups, integrations, alternatives, "
        "comparisons, and G2's AI-derived pros and cons. Costs 5 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioG2ProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioG2ProductAPIWrapper = Field(
        default_factory=ScavioG2ProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioG2ProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/g2/product (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
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
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/g2/product (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
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
                f"No G2 product found for '{identifier}'. Verify the identifiers you "
                "passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioG2Reviews
# --------------------------------------------------------------------------


class ScavioG2ReviewsInput(BaseModel):
    """Input schema for the ScavioG2Reviews tool."""

    model_config = ConfigDict(extra="allow")

    product_id: Optional[str] = Field(
        default=None,
        description=(
            "G2 slug (notion) or the numeric G2 id (82623) as a string. Both resolve "
            "on the same upstream path."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full listing URL, usable instead of the id fields."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. Fixed at 10 reviews per page, and it paginates well "
            "past the 10 pages G2's own widget links to."
        ),
    )

    sort: Optional[
        Literal["relevance", "newest", "most_helpful", "rating_high", "rating_low"]
    ] = Field(
        default=None,
        description=(
            "Sort order. Closed set: an unknown value is silently accepted upstream "
            "and the sort never runs. Options: relevance, newest, most_helpful, "
            "rating_high, rating_low. Default: relevance."
        ),
    )

    rating: Optional[Literal[1, 2, 3, 4, 5]] = Field(
        default=None,
        description=(
            "Star bucket. HALF-STAR-INCLUSIVE: 1 returns 0, 0.5 and 1-star reviews. "
            "Options: 1, 2, 3, 4, 5."
        ),
    )

    company_size: Optional[
        Literal["small_business", "mid_market", "enterprise"]
    ] = Field(
        default=None,
        description=(
            "Reviewer company size: small_business (<=50), mid_market (51-1000), "
            "enterprise (>1000). Options: small_business, mid_market, enterprise."
        ),
    )

    role: Optional[
        Literal[
            "user",
            "administrator",
            "executive_sponsor",
            "internal_consultant",
            "consultant",
            "agency",
            "industry_analyst",
        ]
    ] = Field(
        default=None,
        description=(
            "Reviewer role filter. Options: user, administrator, executive_sponsor, "
            "internal_consultant, consultant, agency, industry_analyst."
        ),
    )

    region: Optional[
        Literal[
            "north_america",
            "europe",
            "asia",
            "latin_america",
            "anz",
            "middle_east",
            "africa",
        ]
    ] = Field(
        default=None,
        description=(
            "Reviewer region filter. Options: north_america, europe, asia, "
            "latin_america, anz, middle_east, africa."
        ),
    )

    query: Optional[str] = Field(
        default=None,
        description=(
            "Full-text search inside the reviews. Narrows the list AND every facet "
            "count."
        ),
    )


class ScavioG2Reviews(BaseTool):  # type: ignore[override]
    """G2: A page of G2 software reviews: rating, title, likes/dislikes, problems
    solved, reviewer job title, industry and company size, validated/incentivized
    flags -- PLUS exact per-star counts and faceted counts.

    Costs 5 credits per call.

    Pagination: page -- fixed at 10 per page and paginates well past the 10 pages
    G2's own widget links to.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioG2Reviews

            tool = ScavioG2Reviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "notion", "page": 2})
    """

    name: str = "scavio_g2_reviews"
    description: str = (
        "G2: A page of G2 software reviews: rating, title, likes/dislikes, problems "
        "solved, reviewer job title, industry and company size, validated/incentivized "
        "flags -- PLUS exact per-star counts and faceted counts. Pagination: page -- "
        "fixed at 10 per page and paginates well past the 10 pages G2's own widget "
        "links to. Costs 5 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioG2ReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioG2ReviewsAPIWrapper = Field(
        default_factory=ScavioG2ReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioG2ReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        sort: Optional[
            Literal["relevance", "newest", "most_helpful", "rating_high", "rating_low"]
        ] = None,
        rating: Optional[Literal[1, 2, 3, 4, 5]] = None,
        company_size: Optional[
            Literal["small_business", "mid_market", "enterprise"]
        ] = None,
        role: Optional[
            Literal[
                "user",
                "administrator",
                "executive_sponsor",
                "internal_consultant",
                "consultant",
                "agency",
                "industry_analyst",
            ]
        ] = None,
        region: Optional[
            Literal[
                "north_america",
                "europe",
                "asia",
                "latin_america",
                "anz",
                "middle_east",
                "africa",
            ]
        ] = None,
        query: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/g2/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
                url=url,
                page=page,
                sort=sort,
                rating=rating,
                company_size=company_size,
                role=role,
                region=region,
                query=query,
            )
            return self._process_response(
                raw, _first_identifier(product_id, url, query)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_id: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        sort: Optional[
            Literal["relevance", "newest", "most_helpful", "rating_high", "rating_low"]
        ] = None,
        rating: Optional[Literal[1, 2, 3, 4, 5]] = None,
        company_size: Optional[
            Literal["small_business", "mid_market", "enterprise"]
        ] = None,
        role: Optional[
            Literal[
                "user",
                "administrator",
                "executive_sponsor",
                "internal_consultant",
                "consultant",
                "agency",
                "industry_analyst",
            ]
        ] = None,
        region: Optional[
            Literal[
                "north_america",
                "europe",
                "asia",
                "latin_america",
                "anz",
                "middle_east",
                "africa",
            ]
        ] = None,
        query: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/g2/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
                url=url,
                page=page,
                sort=sort,
                rating=rating,
                company_size=company_size,
                role=role,
                region=region,
                query=query,
            )
            return self._process_response(
                raw, _first_identifier(product_id, url, query)
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
                f"No G2 reviews found for '{identifier}'. Try broadening the query or "
                "removing filters."
            )
        return raw
