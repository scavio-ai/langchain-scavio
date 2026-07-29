"""Scavio TikTok Shop tools for LangChain agents."""

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
    ScavioTikTokShopCategoriesAPIWrapper,
    ScavioTikTokShopCategoryProductsAPIWrapper,
    ScavioTikTokShopProductAPIWrapper,
    ScavioTikTokShopProductReviewsAPIWrapper,
    ScavioTikTokShopResolveAPIWrapper,
    ScavioTikTokShopSearchAPIWrapper,
    ScavioTikTokShopSearchSuggestionsAPIWrapper,
    ScavioTikTokShopShopProductsAPIWrapper,
)

logger = logging.getLogger(__name__)

_LIST_INIT_ONLY_PARAMS = frozenset({"max_results"})

# Regions the provider actually serves. Search and categories take no region at
# all (US-only / globally identical), and category listings are US/GB only.
Region = Literal["US", "GB", "SG", "MY", "PH", "TH", "VN", "ID"]
ListingRegion = Literal["US", "GB"]

# The two facts every caller has to know about this product area. They are
# repeated verbatim in the tool descriptions so an agent reading only the tool
# manifest still sees them.
PRODUCT_COVERAGE_NOTE = (
    "Only about 44% of the product ids returned by scavio_tiktok_shop_search "
    "resolve on this endpoint. Upstream has no detail data for the rest, so a "
    "404 with data null is a normal outcome rather than an error: skip that "
    "product instead of retrying. Search is a listing source, not the first leg "
    "of a reliable search-then-detail pipeline."
)
PRODUCT_PRICE_NOTE = (
    "This endpoint does NOT return a price -- upstream masks it on the product "
    "page, so price.current and price.original come back null. Exact prices are "
    "returned by scavio_tiktok_shop_search, scavio_tiktok_shop_shop_products and "
    "scavio_tiktok_shop_category_products; read prices from those."
)

_ERROR_404_PREFIX = "Error 404: "


def _forward_api_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract API wrapper kwargs from tool kwargs."""
    api_kwargs: dict[str, Any] = {}
    if "scavio_api_key" in kwargs:
        api_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
    if "api_base_url" in kwargs:
        api_kwargs["api_base_url"] = kwargs.pop("api_base_url")
    if "max_requests_per_second" in kwargs:
        api_kwargs["max_requests_per_second"] = kwargs.pop("max_requests_per_second")
    return api_kwargs


def _not_found_detail(err: Exception) -> Optional[str]:
    """Return the API message when ``err`` is a 404, else ``None``.

    A TikTok Shop 404 means the provider answered and there is genuinely no
    record: the id does not resolve, the shop has no products, or the link is
    dead.  It is a determinate answer, not a failure, so it is surfaced as a
    structured ``not_found`` result rather than an error string.
    """
    message = str(err)
    if message.startswith(_ERROR_404_PREFIX):
        return message[len(_ERROR_404_PREFIX) :]
    return None


def _not_found_result(detail: str, guidance: str) -> dict[str, Any]:
    """Build the structured not-found payload returned instead of an error."""
    return {"data": None, "not_found": True, "reason": detail, "guidance": guidance}


# ---------------------------------------------------------------------------
# 1. Search
# ---------------------------------------------------------------------------


class ScavioTikTokShopSearchInput(BaseModel):
    """Input schema for ScavioTikTokShopSearch tool."""

    model_config = ConfigDict(extra="allow")

    search: str = Field(
        description="Search keyword, 1-200 characters.",
        min_length=1,
        max_length=200,
    )
    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque cursor from a previous response's data.next_cursor. "
            "Omit for the first page."
        ),
    )


class ScavioTikTokShopSearch(BaseTool):  # type: ignore[override]
    """Search TikTok Shop products by keyword (US catalog).

    Returns up to 30 product cards per page with exact prices, ratings,
    sold counts and shop details.  Paginate with ``data.next_cursor`` and
    stop when ``data.has_more`` is false; dedupe by ``product_id`` across
    pages because pages can overlap.

    ``data.degraded`` is true when the retry budget was exhausted and the
    page came back short -- that is a thin page, not the end of results.

    Product ids returned here are not guaranteed to resolve on
    ``ScavioTikTokShopProduct``: only about 44% do.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokShopSearch

            tool = ScavioTikTokShopSearch(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"search": "phone case"})
    """

    name: str = "scavio_tiktok_shop_search"
    description: str = (
        "Search TikTok Shop products by keyword (US catalog only). "
        "Returns up to 30 products per page with exact prices, ratings, "
        "sold counts and shop details. Paginate with data.next_cursor and "
        "dedupe by product_id across pages. "
        "Product ids returned here are not guaranteed to resolve on "
        "scavio_tiktok_shop_product: only about 44% do, so treat this as a "
        "listing source, not the first leg of a search-then-detail pipeline. "
        "This endpoint returns exact prices; the product endpoint does not."
    )
    args_schema: Type[BaseModel] = ScavioTikTokShopSearchInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioTikTokShopSearchAPIWrapper = Field(
        default_factory=ScavioTikTokShopSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokShopSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        search: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(search=search, cursor=cursor)
            return self._process_response(raw, search)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        search: str,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                search=search, cursor=cursor
            )
            return self._process_response(raw, search)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], search: str) -> dict[str, Any]:
        data = raw.get("data") or {}
        products = data.get("products") if isinstance(data, dict) else None
        if self.max_results and products:
            raw["data"]["products"] = products[: self.max_results]
        if not products:
            raise ToolException(
                f"No TikTok Shop products found for '{search}'. "
                "TikTok Shop search covers the US catalog only. "
                "Try a broader keyword."
            )
        return raw


# ---------------------------------------------------------------------------
# 2. Search Suggestions
# ---------------------------------------------------------------------------


class ScavioTikTokShopSearchSuggestionsInput(BaseModel):
    """Input schema for ScavioTikTokShopSearchSuggestions tool."""

    model_config = ConfigDict(extra="allow")

    search: str = Field(
        description="Partial search keyword, 1-100 characters.",
        min_length=1,
        max_length=100,
    )
    region: Optional[Region] = Field(
        default=None,
        description="Marketplace region (default US). All 8 regions work here.",
    )


class ScavioTikTokShopSearchSuggestions(BaseTool):  # type: ignore[override]
    """Keyword autocomplete and expansion for a partial TikTok Shop query.

    Returns ``data.suggestions``, a list of plain strings (no volume, no
    score -- upstream provides neither).  Suggestions are not guaranteed
    prefix matches: a misspelling returns typo corrections, and results can
    include brand and shop names.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokShopSearchSuggestions

            tool = ScavioTikTokShopSearchSuggestions()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"search": "wireless"})
    """

    name: str = "scavio_tiktok_shop_search_suggestions"
    description: str = (
        "Keyword autocomplete and expansion for a partial TikTok Shop query, "
        "across 8 marketplace regions. Returns a plain list of suggestion "
        "strings with no search volume or score. Suggestions are not "
        "guaranteed prefix matches: a misspelling returns typo corrections, "
        "and results can include brand and shop names."
    )
    args_schema: Type[BaseModel] = ScavioTikTokShopSearchSuggestionsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTikTokShopSearchSuggestionsAPIWrapper = Field(
        default_factory=ScavioTikTokShopSearchSuggestionsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokShopSearchSuggestionsAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        search: str,
        region: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(search=search, region=region)
            return self._process_response(raw, search)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        search: str,
        region: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                search=search, region=region
            )
            return self._process_response(raw, search)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any], search: str) -> dict[str, Any]:
        data = raw.get("data") or {}
        suggestions = data.get("suggestions") if isinstance(data, dict) else None
        if not suggestions:
            raise ToolException(
                f"No TikTok Shop keyword suggestions found for '{search}'. "
                "This endpoint normally returns typo corrections even for "
                "nonsense input, so an empty list is unusual; try again."
            )
        return raw


# ---------------------------------------------------------------------------
# 3. Product Details
# ---------------------------------------------------------------------------


class ScavioTikTokShopProductInput(BaseModel):
    """Input schema for ScavioTikTokShopProduct tool."""

    model_config = ConfigDict(extra="allow")

    product_id: str = Field(
        description="TikTok Shop product id, 6-25 digits.",
        pattern=r"^\d{6,25}$",
    )
    region: Optional[Region] = Field(
        default=None,
        description="Marketplace region (default US).",
    )


class ScavioTikTokShopProduct(BaseTool):  # type: ignore[override]
    """Fetch full TikTok Shop product detail.

    Returns description, images, variants with stock, shipping, the full
    shop profile, category path, breadcrumbs and up to 3 top reviews.

    Two limits worth knowing before you call it:

    * It resolves only about 44% of the product ids returned by
      ``ScavioTikTokShopSearch``.  Upstream has no detail data for the
      rest, so a 404 is a normal outcome, not an error -- the tool returns
      ``{"data": None, "not_found": True, ...}`` and the caller should skip
      that product rather than retry.
    * It does **not** return a price.  Upstream masks the digits on the
      product page, so ``price.current`` and ``price.original`` are null.
      Exact prices come from ``ScavioTikTokShopSearch``,
      ``ScavioTikTokShopShopProducts`` or
      ``ScavioTikTokShopCategoryProducts``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokShopProduct

            tool = ScavioTikTokShopProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "1732293553906094315"})
    """

    name: str = "scavio_tiktok_shop_product"
    description: str = (
        "Fetch full TikTok Shop product detail: description, images, variants "
        "with stock, shipping, shop profile, category path and top reviews. "
        + PRODUCT_PRICE_NOTE
        + " "
        + PRODUCT_COVERAGE_NOTE
        + " A not_found result is returned as data null with not_found true; "
        "treat it as a normal answer and move on."
    )
    args_schema: Type[BaseModel] = ScavioTikTokShopProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTikTokShopProductAPIWrapper = Field(
        default_factory=ScavioTikTokShopProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokShopProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: str,
        region: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(product_id=product_id, region=region)
            return self._process_response(raw, product_id)
        except ToolException:
            raise
        except Exception as e:
            return self._handle_exception(e)

    async def _arun(
        self,
        product_id: str,
        region: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id, region=region
            )
            return self._process_response(raw, product_id)
        except ToolException:
            raise
        except Exception as e:
            return self._handle_exception(e)

    def _handle_exception(self, err: Exception) -> dict[str, Any]:
        detail = _not_found_detail(err)
        if detail is not None:
            return _not_found_result(
                detail, PRODUCT_COVERAGE_NOTE + " " + PRODUCT_PRICE_NOTE
            )
        return {"error": str(err)}

    def _process_response(
        self, raw: dict[str, Any], product_id: str
    ) -> dict[str, Any]:
        data = raw.get("data")
        if not isinstance(data, dict) or not data.get("product_id"):
            return _not_found_result(
                f"No TikTok Shop detail data upstream for product '{product_id}'.",
                PRODUCT_COVERAGE_NOTE + " " + PRODUCT_PRICE_NOTE,
            )
        return raw


# ---------------------------------------------------------------------------
# 4. Product Reviews
# ---------------------------------------------------------------------------


class ScavioTikTokShopProductReviewsInput(BaseModel):
    """Input schema for ScavioTikTokShopProductReviews tool."""

    model_config = ConfigDict(extra="allow")

    product_id: str = Field(
        description="TikTok Shop product id, 6-25 digits.",
        pattern=r"^\d{6,25}$",
    )
    page: Optional[int] = Field(
        default=None,
        ge=1,
        le=500,
        description="1-based page number (default 1).",
    )
    page_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Reviews per page (1-200, default 20).",
    )
    sort: Optional[Literal["relevant", "recent"]] = Field(
        default=None,
        description=(
            '"relevant" (default) returns text-complete, image-heavy reviews. '
            '"recent" is fresher but far more text-sparse -- many rows carry '
            "stars only, with no text and no images."
        ),
    )
    rating: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Only reviews with this star rating.",
    )
    has_media: Optional[bool] = Field(
        default=None,
        description="Only reviews with a photo or video.",
    )
    verified_only: Optional[bool] = Field(
        default=None,
        description=(
            "Only verified purchases. Upstream allows one filter at a time, so "
            "this is ignored when has_media is true; the response echoes what "
            "was really applied in data.filters_applied."
        ),
    )
    region: Optional[Region] = Field(
        default=None,
        description="Marketplace region (default US).",
    )


class ScavioTikTokShopProductReviews(BaseTool):  # type: ignore[override]
    """Fetch paginated TikTok Shop product reviews.

    Returns review text, images, star histogram, verified-purchase and
    incentivized flags, up to 200 rows per call.

    ``data.total_reviews`` drifts between calls minutes apart and must not
    be used to compute a page count -- page with ``data.has_more`` instead.
    Reviewer names arrive pre-masked by the platform ("C\\*\\*") and reviewer
    avatars and review images are signed URLs that expire, so do not treat
    them as stable identifiers.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokShopProductReviews

            tool = ScavioTikTokShopProductReviews(max_results=20)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"product_id": "1732293553906094315"})
    """

    name: str = "scavio_tiktok_shop_product_reviews"
    description: str = (
        "Fetch paginated TikTok Shop product reviews with text, images, star "
        "histogram and verified-purchase flags, up to 200 per call. "
        "data.total_reviews drifts between calls and must not be used to "
        "compute a page count; page with data.has_more instead. "
        'sort="recent" is fresher but far more text-sparse than the default '
        '"relevant". Reviewer names are pre-masked by the platform and image '
        "URLs are signed and expire."
    )
    args_schema: Type[BaseModel] = ScavioTikTokShopProductReviewsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 20

    api_wrapper: ScavioTikTokShopProductReviewsAPIWrapper = Field(
        default_factory=ScavioTikTokShopProductReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokShopProductReviewsAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        product_id: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort: Optional[str] = None,
        rating: Optional[int] = None,
        has_media: Optional[bool] = None,
        verified_only: Optional[bool] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                product_id=product_id,
                page=page,
                page_size=page_size,
                sort=sort,
                rating=rating,
                has_media=has_media,
                verified_only=verified_only,
                region=region,
            )
            return self._process_response(raw, product_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        product_id: str,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        sort: Optional[str] = None,
        rating: Optional[int] = None,
        has_media: Optional[bool] = None,
        verified_only: Optional[bool] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                product_id=product_id,
                page=page,
                page_size=page_size,
                sort=sort,
                rating=rating,
                has_media=has_media,
                verified_only=verified_only,
                region=region,
            )
            return self._process_response(raw, product_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], product_id: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        reviews = data.get("reviews") if isinstance(data, dict) else None
        if self.max_results and reviews:
            raw["data"]["reviews"] = reviews[: self.max_results]
        if not reviews:
            raise ToolException(
                f"No TikTok Shop reviews found for product '{product_id}'. "
                "The product may have no reviews, the filters may exclude all "
                "of them, or the page may be past the end of the list -- an "
                "empty page with has_more false is the end, not a failure."
            )
        return raw


# ---------------------------------------------------------------------------
# 5. Categories
# ---------------------------------------------------------------------------


class ScavioTikTokShopCategoriesInput(BaseModel):
    """Input schema for ScavioTikTokShopCategories tool."""

    model_config = ConfigDict(extra="allow")


class ScavioTikTokShopCategories(BaseTool):  # type: ignore[override]
    """Fetch the global TikTok Shop category tree.

    Returns 28 top-level categories, 240 nodes, exactly two levels deep --
    that is all upstream provides, the depth is not faked.  Category ids
    are identical in every region and names are always English, so this
    endpoint takes no parameters.  The tree is stable: cache it rather than
    calling it per request.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokShopCategories

            tool = ScavioTikTokShopCategories()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({})
    """

    name: str = "scavio_tiktok_shop_categories"
    description: str = (
        "Fetch the global TikTok Shop category tree: 28 top-level categories, "
        "240 nodes, two levels deep. Category ids are identical in every "
        "region and names are always English, so this tool takes no "
        "parameters. Use a category_id from here with "
        "scavio_tiktok_shop_category_products. The tree is stable; cache it."
    )
    args_schema: Type[BaseModel] = ScavioTikTokShopCategoriesInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTikTokShopCategoriesAPIWrapper = Field(
        default_factory=ScavioTikTokShopCategoriesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokShopCategoriesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results()
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async()
            return self._process_response(raw)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        data = raw.get("data") or {}
        categories = data.get("categories") if isinstance(data, dict) else None
        if not categories:
            raise ToolException(
                "The TikTok Shop category tree came back empty. "
                "This tree is static, so an empty result means the request "
                "failed rather than that there are no categories."
            )
        return raw


# ---------------------------------------------------------------------------
# 6. Category Products
# ---------------------------------------------------------------------------


class ScavioTikTokShopCategoryProductsInput(BaseModel):
    """Input schema for ScavioTikTokShopCategoryProducts tool."""

    model_config = ConfigDict(extra="allow")

    category_id: str = Field(
        description=(
            "Category id from scavio_tiktok_shop_categories. "
            "Level 1 and level 2 ids both work."
        ),
        pattern=r"^\d{4,20}$",
    )
    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque cursor from a previous response's data.next_cursor. "
            "Omit for the first page."
        ),
    )
    region: Optional[ListingRegion] = Field(
        default=None,
        description=(
            "Marketplace region (default US). Category listings are served for "
            "US and GB only."
        ),
    )


class ScavioTikTokShopCategoryProducts(BaseTool):  # type: ignore[override]
    """List TikTok Shop products under a category id, with exact prices.

    Page size is inconsistent upstream (15 to 20 per page), so always
    paginate with ``data.next_cursor`` rather than assuming a fixed size.
    Category listings are shallow: after a few pages the source stops
    returning new products and ``has_more`` turns false, which is the end
    of the listing rather than an error.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokShopCategoryProducts

            tool = ScavioTikTokShopCategoryProducts(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"category_id": "601450"})
    """

    name: str = "scavio_tiktok_shop_category_products"
    description: str = (
        "List TikTok Shop products under a category id from "
        "scavio_tiktok_shop_categories, with exact prices. US and GB only. "
        "Page size is inconsistent upstream (15 to 20 per page), so paginate "
        "with data.next_cursor and never assume a fixed size. Listings are "
        "shallow: has_more turning false after a few pages is the end of the "
        "listing, not an error. This endpoint returns exact prices; the "
        "product endpoint does not."
    )
    args_schema: Type[BaseModel] = ScavioTikTokShopCategoryProductsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioTikTokShopCategoryProductsAPIWrapper = Field(
        default_factory=ScavioTikTokShopCategoryProductsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokShopCategoryProductsAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        category_id: str,
        cursor: Optional[str] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                category_id=category_id, cursor=cursor, region=region
            )
            return self._process_response(raw, category_id)
        except ToolException:
            raise
        except Exception as e:
            return self._handle_exception(e)

    async def _arun(
        self,
        category_id: str,
        cursor: Optional[str] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                category_id=category_id, cursor=cursor, region=region
            )
            return self._process_response(raw, category_id)
        except ToolException:
            raise
        except Exception as e:
            return self._handle_exception(e)

    def _handle_exception(self, err: Exception) -> dict[str, Any]:
        detail = _not_found_detail(err)
        if detail is not None:
            return _not_found_result(
                detail,
                "Check the category_id against scavio_tiktok_shop_categories. "
                "Category listings are served for US and GB only.",
            )
        return {"error": str(err)}

    def _process_response(
        self, raw: dict[str, Any], category_id: str
    ) -> dict[str, Any]:
        data = raw.get("data") or {}
        products = data.get("products") if isinstance(data, dict) else None
        if self.max_results and products:
            raw["data"]["products"] = products[: self.max_results]
        if not products:
            raise ToolException(
                f"No TikTok Shop products returned for category "
                f"'{category_id}'. An empty later page with has_more false is "
                "the end of a shallow listing, not a failure."
            )
        return raw


# ---------------------------------------------------------------------------
# 7. Shop Products
# ---------------------------------------------------------------------------


class ScavioTikTokShopShopProductsInput(BaseModel):
    """Input schema for ScavioTikTokShopShopProducts tool."""

    model_config = ConfigDict(extra="allow")

    shop_id: str = Field(
        description=(
            "TikTok Shop seller id, 6-25 digits "
            "(also called seller_id elsewhere on TikTok)."
        ),
        pattern=r"^\d{6,25}$",
    )
    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque cursor from a previous response's data.next_cursor. "
            "Omit for the first page."
        ),
    )
    region: Optional[Region] = Field(
        default=None,
        description="Marketplace region (default US).",
    )


class ScavioTikTokShopShopProducts(BaseTool):  # type: ignore[override]
    """Fetch a TikTok Shop seller's product catalog, with exact prices.

    Returns 30 product cards per page.  ``data.shop`` carries only the shop
    id, name and logo: follower count, shop location and shop-level rating
    are **not** available from this endpoint -- call
    ``ScavioTikTokShopProduct`` for the full shop profile.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokShopShopProducts

            tool = ScavioTikTokShopShopProducts(max_results=10)

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"shop_id": "7495514739648989419"})
    """

    name: str = "scavio_tiktok_shop_shop_products"
    description: str = (
        "Fetch a TikTok Shop seller's product catalog, 30 per page, with "
        "exact prices. Paginate with data.next_cursor. Shop follower count, "
        "location and shop-level rating are not available here -- call "
        "scavio_tiktok_shop_product for the full shop profile. This endpoint "
        "returns exact prices; the product endpoint does not."
    )
    args_schema: Type[BaseModel] = ScavioTikTokShopShopProductsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioTikTokShopShopProductsAPIWrapper = Field(
        default_factory=ScavioTikTokShopShopProductsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokShopShopProductsAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        shop_id: str,
        cursor: Optional[str] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                shop_id=shop_id, cursor=cursor, region=region
            )
            return self._process_response(raw, shop_id)
        except ToolException:
            raise
        except Exception as e:
            return self._handle_exception(e)

    async def _arun(
        self,
        shop_id: str,
        cursor: Optional[str] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                shop_id=shop_id, cursor=cursor, region=region
            )
            return self._process_response(raw, shop_id)
        except ToolException:
            raise
        except Exception as e:
            return self._handle_exception(e)

    def _handle_exception(self, err: Exception) -> dict[str, Any]:
        detail = _not_found_detail(err)
        if detail is not None:
            return _not_found_result(
                detail,
                "Check the shop_id, or resolve a storefront URL with "
                "scavio_tiktok_shop_resolve first.",
            )
        return {"error": str(err)}

    def _process_response(self, raw: dict[str, Any], shop_id: str) -> dict[str, Any]:
        data = raw.get("data") or {}
        products = data.get("products") if isinstance(data, dict) else None
        if self.max_results and products:
            raw["data"]["products"] = products[: self.max_results]
        if not products:
            raise ToolException(
                f"No TikTok Shop products returned for shop '{shop_id}'. "
                "An empty later page with has_more false is the end of the "
                "catalog, not a failure."
            )
        return raw


# ---------------------------------------------------------------------------
# 8. URL Resolver
# ---------------------------------------------------------------------------


class ScavioTikTokShopResolveInput(BaseModel):
    """Input schema for ScavioTikTokShopResolve tool."""

    model_config = ConfigDict(extra="allow")

    url: str = Field(
        description=(
            "A TikTok Shop URL or share link. Accepted: shop.tiktok.com "
            "product or store pages, tiktok.com/view/product or /view/shop "
            "links, affiliate-*.tiktok.com share links, and vt.tiktok.com or "
            "tiktok.com/t short links."
        ),
        max_length=2000,
    )


class ScavioTikTokShopResolve(BaseTool):  # type: ignore[override]
    """Resolve a TikTok Shop URL or share link to a product_id or shop_id.

    Returns ``data.type`` ("product" or "shop"), the id, a canonical
    ``https://shop.tiktok.com/...`` URL, and ``resolved_by``
    ("url_pattern" when the id was read straight out of the link,
    "share_link" when the link had to be followed).

    A dead or expired share link returns
    ``{"data": None, "not_found": True, ...}``; an unsupported URL format
    is an input error and is not billed.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioTikTokShopResolve

            tool = ScavioTikTokShopResolve()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"url": "https://vt.tiktok.com/ZT2AHoGsE/"})
    """

    name: str = "scavio_tiktok_shop_resolve"
    description: str = (
        "Resolve any TikTok Shop URL or share link to a product_id or "
        "shop_id, ready to pass to the other TikTok Shop tools. Accepts "
        "shop.tiktok.com product and store pages, tiktok.com/view links, "
        "affiliate share links and vt.tiktok.com short links. Returns the id, "
        "a canonical https URL and whether it was read from the URL pattern "
        "or by following the share link. A dead link returns not_found."
    )
    args_schema: Type[BaseModel] = ScavioTikTokShopResolveInput
    handle_tool_error: bool = True

    api_wrapper: ScavioTikTokShopResolveAPIWrapper = Field(
        default_factory=ScavioTikTokShopResolveAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioTikTokShopResolveAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        url: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = self.api_wrapper.raw_results(url=url)
            return self._process_response(raw, url)
        except ToolException:
            raise
        except Exception as e:
            return self._handle_exception(e)

    async def _arun(
        self,
        url: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            raw = await self.api_wrapper.raw_results_async(url=url)
            return self._process_response(raw, url)
        except ToolException:
            raise
        except Exception as e:
            return self._handle_exception(e)

    def _handle_exception(self, err: Exception) -> dict[str, Any]:
        detail = _not_found_detail(err)
        if detail is not None:
            return _not_found_result(
                detail,
                "The link may have expired or may not point to a product or "
                "shop. Try the canonical shop.tiktok.com URL instead.",
            )
        return {"error": str(err)}

    def _process_response(self, raw: dict[str, Any], url: str) -> dict[str, Any]:
        data = raw.get("data")
        if not isinstance(data, dict) or not (
            data.get("product_id") or data.get("shop_id")
        ):
            raise ToolException(
                f"Could not resolve '{url}' to a TikTok Shop product or shop."
            )
        return raw
