"""Scavio Target tools for LangChain agents.

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
    ScavioTargetCategoryAPIWrapper,
    ScavioTargetProductAPIWrapper,
    ScavioTargetReviewsAPIWrapper,
    ScavioTargetSearchAPIWrapper,
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
# ScavioTargetSearch
# --------------------------------------------------------------------------


class ScavioTargetSearchInput(BaseModel):
    """Input schema for the ScavioTargetSearch tool."""

    model_config = ConfigDict(extra="allow")

    keyword: str = Field(
        description=(
            "Product search query."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number, 1-based."
        ),
    )

    count: Optional[int] = Field(
        default=None,
        description=(
            "Products per page. Target rejects anything above 28 outright. Default: "
            "24."
        ),
    )

    sort: Optional[
        Literal[
            "relevance",
            "featured",
            "price_low",
            "price_high",
            "rating_high",
            "best_seller",
            "newest",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: relevance, featured, price_low, "
            "price_high, rating_high, best_seller, newest. Default: relevance."
        ),
    )

    store_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric Target store id. Unlike Walmart this is a real request parameter: "
            "it decides prices and availability. Default: 3991."
        ),
    )


class ScavioTargetSearch(BaseTool):  # type: ignore[override]
    """Target: Search Target.com: prices, ratings, badges and promotions.

    Costs 1 credit per call.

    Pagination: page + count.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTargetSearch

            tool = ScavioTargetSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"keyword": "office chair"})
    """

    name: str = "scavio_target_search"
    description: str = (
        "Target: Search Target.com: prices, ratings, badges and promotions. "
        "Pagination: page + count. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioTargetSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTargetSearchAPIWrapper = Field(
        default_factory=ScavioTargetSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTargetSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        keyword: str,
        page: Optional[int] = None,
        count: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevance",
                "featured",
                "price_low",
                "price_high",
                "rating_high",
                "best_seller",
                "newest",
            ]
        ] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/target/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                keyword=keyword,
                page=page,
                count=count,
                sort=sort,
                store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(keyword))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        keyword: str,
        page: Optional[int] = None,
        count: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevance",
                "featured",
                "price_low",
                "price_high",
                "rating_high",
                "best_seller",
                "newest",
            ]
        ] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/target/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                keyword=keyword,
                page=page,
                count=count,
                sort=sort,
                store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(keyword))
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
                f"No Target results found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioTargetCategory
# --------------------------------------------------------------------------


class ScavioTargetCategoryInput(BaseModel):
    """Input schema for the ScavioTargetCategory tool."""

    model_config = ConfigDict(extra="allow")

    category_id: str = Field(
        description=(
            "The segment after `N-` in a target.com /c/ URL."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number, 1-based."
        ),
    )

    count: Optional[int] = Field(
        default=None,
        description=(
            "Products per page. Target rejects anything above 28 outright. Default: "
            "24."
        ),
    )

    sort: Optional[
        Literal[
            "relevance",
            "featured",
            "price_low",
            "price_high",
            "rating_high",
            "best_seller",
            "newest",
        ]
    ] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: relevance, featured, price_low, "
            "price_high, rating_high, best_seller, newest. Default: relevance."
        ),
    )

    store_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric Target store id. Unlike Walmart this is a real request parameter: "
            "it decides prices and availability. Default: 3991."
        ),
    )


class ScavioTargetCategory(BaseTool):  # type: ignore[override]
    """Target: Products in a Target category, same shape as search plus the category
    breadcrumb.

    Costs 1 credit per call.

    Pagination: page + count.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTargetCategory

            tool = ScavioTargetCategory()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"category_id": "5xtg6"})
    """

    name: str = "scavio_target_category"
    description: str = (
        "Target: Products in a Target category, same shape as search plus the category "
        "breadcrumb. Pagination: page + count. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioTargetCategoryInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTargetCategoryAPIWrapper = Field(
        default_factory=ScavioTargetCategoryAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTargetCategoryAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        category_id: str,
        page: Optional[int] = None,
        count: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevance",
                "featured",
                "price_low",
                "price_high",
                "rating_high",
                "best_seller",
                "newest",
            ]
        ] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/target/category (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                category_id=category_id,
                page=page,
                count=count,
                sort=sort,
                store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(category_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        category_id: str,
        page: Optional[int] = None,
        count: Optional[int] = None,
        sort: Optional[
            Literal[
                "relevance",
                "featured",
                "price_low",
                "price_high",
                "rating_high",
                "best_seller",
                "newest",
            ]
        ] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/target/category (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                category_id=category_id,
                page=page,
                count=count,
                sort=sort,
                store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(category_id))
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
                f"No Target category products found for '{identifier}'. Try broadening "
                "the query or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioTargetProduct
# --------------------------------------------------------------------------


class ScavioTargetProductInput(BaseModel):
    """Input schema for the ScavioTargetProduct tool."""

    model_config = ConfigDict(extra="allow")

    tcin: str = Field(
        description=(
            "Target catalog item number. A child TCIN is answered by its variation "
            "parent, with the child present under variants."
        ),
    )

    store_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric Target store id. Unlike Walmart this is a real request parameter: "
            "it decides prices and availability. Default: 3991."
        ),
    )


class ScavioTargetProduct(BaseTool):  # type: ignore[override]
    """Target: Target product details by TCIN: price, rating, images, specifications,
    variants, return policy, fulfillment.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTargetProduct

            tool = ScavioTargetProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"tcin": "87095665"})
    """

    name: str = "scavio_target_product"
    description: str = (
        "Target: Target product details by TCIN: price, rating, images, "
        "specifications, variants, return policy, fulfillment. Costs 1 credit per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioTargetProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTargetProductAPIWrapper = Field(
        default_factory=ScavioTargetProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTargetProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        tcin: str,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/target/product (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                tcin=tcin,
                store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(tcin))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        tcin: str,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/target/product (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                tcin=tcin,
                store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(tcin))
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
                f"No Target product found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioTargetReviews
# --------------------------------------------------------------------------


class ScavioTargetReviewsInput(BaseModel):
    """Input schema for the ScavioTargetReviews tool."""

    model_config = ConfigDict(extra="allow")

    tcin: str = Field(
        description=(
            "Target catalog item number. A child TCIN is answered by its variation "
            "parent, with the child present under variants."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "TRIMS the returned bodies only. Target publishes 8 reviews anonymously "
            "and offers no paging, so this cannot fetch more."
        ),
    )

    store_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric Target store id. Unlike Walmart this is a real request parameter: "
            "it decides prices and availability. Default: 3991."
        ),
    )


class ScavioTargetReviews(BaseTool):  # type: ignore[override]
    """Target: Target reviews with the rating breakdown, per-attribute averages and
    guest photos.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTargetReviews

            tool = ScavioTargetReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"tcin": "87095665"})
    """

    name: str = "scavio_target_reviews"
    description: str = (
        "Target: Target reviews with the rating breakdown, per-attribute averages and "
        "guest photos. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioTargetReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTargetReviewsAPIWrapper = Field(
        default_factory=ScavioTargetReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTargetReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        tcin: str,
        limit: Optional[int] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/target/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                tcin=tcin,
                limit=limit,
                store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(tcin))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        tcin: str,
        limit: Optional[int] = None,
        store_id: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/target/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                tcin=tcin,
                limit=limit,
                store_id=store_id,
            )
            return self._process_response(raw, _first_identifier(tcin))
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
                f"No Target reviews found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw
