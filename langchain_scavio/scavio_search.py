"""Scavio Google tools for LangChain agents.

``ScavioSearch`` is the general web-search tool; the ``ScavioGoogle*`` tools
below cover the remaining Google v2 verticals. Google v1 (``/api/v1/google``)
was retired on 2026-08-04 and now returns HTTP 410 -- every tool here targets
``/api/v2/google*`` and exposes the v2 wire parameters (``gl``, ``hl``,
``start``, ``google_domain``, ``device``) natively. ``start`` is a 0-based
result offset, never a page number.
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
    ScavioGoogleAIModeAPIWrapper,
    ScavioGoogleFlightsAPIWrapper,
    ScavioGoogleHotelsAPIWrapper,
    ScavioGoogleHotelsDetailAPIWrapper,
    ScavioGoogleMapsPlaceAPIWrapper,
    ScavioGoogleMapsReviewsAPIWrapper,
    ScavioGoogleShoppingAPIWrapper,
    ScavioGoogleShoppingProductAPIWrapper,
    ScavioGoogleShoppingStoresAPIWrapper,
    ScavioGoogleTrendingAPIWrapper,
    ScavioGoogleTrendsAPIWrapper,
    ScavioSearchAPIWrapper,
)

logger = logging.getLogger(__name__)

_LIST_INIT_ONLY_PARAMS = frozenset({"max_results"})


def _forward_api_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract API wrapper kwargs from tool kwargs."""
    api_kwargs: dict[str, Any] = {}
    if "scavio_api_key" in kwargs:
        api_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
    if "api_base_url" in kwargs:
        api_kwargs["api_base_url"] = kwargs.pop("api_base_url")
    if "max_requests_per_second" in kwargs:
        api_kwargs["max_requests_per_second"] = kwargs.pop(
            "max_requests_per_second"
        )
    return api_kwargs


# Parameters that can only be set at instantiation, not by the LLM.
_INIT_ONLY_PARAMS = frozenset(
    {
        "max_results",
        "light_request",
        "include_knowledge_graph",
        "include_questions",
        "include_related",
        "include_maps_results",
        "include_ai_overviews",
        "include_local_results",
        "include_top_stories",
        "include_hotel_results",
        "include_news_results",
        "include_shopping_ads",
        "include_top_ads",
        "include_bottom_ads",
        "nfpr",
    }
)


class ScavioSearchInput(BaseModel):
    """Input schema for ScavioSearch tool.

    Defines the parameters that an LLM agent can set dynamically
    at invocation time.
    """

    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Search query to look up")

    search_type: Optional[Literal["classic", "news", "maps"]] = (
        Field(
            default=None,
            description=(
                'Type of search to perform. '
                'Use "classic" (default) for most queries, INCLUDING those asking for '
                '"latest" or "recent" general information. '
                'Use "news" ONLY for politics, sports, or major current events '
                'covered by mainstream media. '
                'Use "maps" for local businesses, restaurants, or place lookups.'
            ),
        )
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            'Country of the search as an ISO 3166-1 alpha-2 code (e.g., "us", '
            '"gb", "de", "fr", "jp"). Native Google v2 parameter. Set this when '
            "the user mentions a country or region, or when results should be "
            "geographically relevant."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            'UI language as an ISO 639-1 code (e.g., "en", "fr", "de"). '
            "Native Google v2 parameter. Set this when the user requests "
            "results in a specific language."
        ),
    )

    start: Optional[int] = Field(
        default=None,
        description=(
            "Result OFFSET, not a page number: 0 is the first page, 10 the "
            "second, 20 the third. Native Google v2 parameter. Only set this if "
            "the user explicitly asks for more results, or the previous batch "
            "did not contain the needed information. On maps searches the "
            "offset must be a multiple of 20; news does not accept it."
        ),
    )

    google_domain: Optional[str] = Field(
        default=None,
        description=(
            'Regional Google domain to query (e.g., "google.co.uk"). '
            "Native Google v2 parameter."
        ),
    )

    country_code: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated alias of gl, kept for backwards compatibility. "
            "Prefer gl; when both are given gl wins."
        ),
    )

    language: Optional[str] = Field(
        default=None,
        description=(
            "Deprecated alias of hl, kept for backwards compatibility. "
            "Prefer hl; when both are given hl wins."
        ),
    )

    device: Optional[Literal["desktop", "mobile"]] = Field(
        default=None,
        description=(
            "Device type for search results. "
            'Use "desktop" (default) for most queries. '
            'Use "mobile" only when the user specifically asks for mobile results. '
            'Note: "news" search_type only supports "desktop".'
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Deprecated 1-indexed page number, translated to the v2 start "
            "offset as (page - 1) * 10. Prefer start, which is what the API "
            "actually takes; when both are given start wins."
        ),
    )


def _generate_suggestions(
    search_type: Optional[str] = None,
    country_code: Optional[str] = None,
    language: Optional[str] = None,
) -> list[str]:
    """Generate actionable suggestions for the LLM when results are empty."""
    suggestions = ["Try broadening the query"]
    if country_code:
        suggestions.append("Remove country_code filter")
    if language:
        suggestions.append("Remove language filter")
    if search_type and search_type != "classic":
        suggestions.append('Try search_type "classic"')
    else:
        suggestions.append("Try a different search_type")
    return suggestions


class ScavioSearch(BaseTool):  # type: ignore[override]
    """Search the web using the Scavio Search API.

    Returns search results with titles, URLs, descriptions, knowledge graphs,
    news results, and related questions. Use for any query requiring real-time
    or recent web information.

    Setup:
        Install ``langchain-scavio`` and set the ``SCAVIO_API_KEY`` environment
        variable.

        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioSearch

            tool = ScavioSearch(
                max_results=5,
                include_knowledge_graph=True,
                include_questions=True,
            )

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "best python web frameworks 2026"})

    Use with an agent:
        .. code-block:: python

            from langchain.agents import create_agent
            from langchain_openai import ChatOpenAI
            from langchain_scavio import ScavioSearch

            agent = create_agent(
                model=ChatOpenAI(model="gpt-5.5"),
                tools=[ScavioSearch()],
                system_prompt="You are a research assistant.",
            )
    """

    name: str = "scavio_search"
    description: str = (
        "Search the web using the Scavio Search API. Returns search results "
        "with titles, URLs, descriptions, knowledge graphs, news results, "
        "and related questions. Use for any query requiring real-time or "
        "recent web information. Input should be a search query. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioSearchInput
    handle_tool_error: bool = True

    # Instantiation-only parameters (not controllable by the LLM).
    max_results: Optional[int] = 5
    # Deprecated: ignored since 2.11 (Google v2 always returns full results
    # for 1 credit). Kept so existing configs keep working.
    light_request: Optional[bool] = None
    include_knowledge_graph: bool = True
    include_questions: bool = True
    include_related: bool = False
    include_maps_results: bool = False
    include_ai_overviews: bool = False
    include_local_results: bool = False
    include_top_stories: bool = False
    include_hotel_results: bool = False
    include_news_results: bool = False
    include_shopping_ads: bool = False
    include_top_ads: bool = False
    include_bottom_ads: bool = False
    nfpr: bool = False

    # Default search parameters (used when the LLM doesn't specify).
    # gl/hl/start/google_domain are the native Google v2 names; country_code,
    # language and page are the pre-3.2 aliases and lose to them when both set.
    gl: Optional[str] = None
    hl: Optional[str] = None
    start: Optional[int] = None
    google_domain: Optional[str] = None
    country_code: Optional[str] = None
    language: Optional[str] = None
    search_type: Optional[str] = None
    device: Optional[str] = None
    page: Optional[int] = None

    api_wrapper: ScavioSearchAPIWrapper = Field(
        default_factory=ScavioSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ScavioSearch.

        Accepts ``scavio_api_key`` and ``api_base_url`` directly,
        forwarding them to the underlying API wrapper.
        """
        api_wrapper_kwargs: dict[str, Any] = {}
        if "scavio_api_key" in kwargs:
            api_wrapper_kwargs["scavio_api_key"] = kwargs.pop("scavio_api_key")
        if "api_base_url" in kwargs:
            api_wrapper_kwargs["api_base_url"] = kwargs.pop("api_base_url")
        if "max_requests_per_second" in kwargs:
            api_wrapper_kwargs["max_requests_per_second"] = kwargs.pop(
                "max_requests_per_second"
            )
        if api_wrapper_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioSearchAPIWrapper(**api_wrapper_kwargs)
        super().__init__(**kwargs)

    def _resolve_params(
        self,
        search_type: Optional[str],
        country_code: Optional[str],
        language: Optional[str],
        device: Optional[str],
        page: Optional[int],
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        start: Optional[int] = None,
        google_domain: Optional[str] = None,
    ) -> dict[str, Any]:
        """Merge LLM-provided params with init-time defaults.

        Native v2 arguments (gl, hl, start, google_domain) are passed through
        untouched; the legacy country_code/language/page aliases are only used
        when their v2 counterpart is absent.
        """
        return {
            "search_type": search_type or self.search_type or "classic",
            "country_code": country_code or self.country_code,
            "language": language or self.language,
            "device": device or self.device or "desktop",
            "page": page or self.page or 1,
            "gl": gl or self.gl,
            "hl": hl or self.hl,
            "start": start if start is not None else self.start,
            "google_domain": google_domain or self.google_domain,
        }

    def _run(
        self,
        query: str,
        search_type: Optional[str] = None,
        country_code: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        page: Optional[int] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        start: Optional[int] = None,
        google_domain: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a synchronous search."""
        forbidden = _INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        params = self._resolve_params(
            search_type, country_code, language, device, page,
            gl, hl, start, google_domain,
        )
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                **params,
                light_request=self.light_request,
                nfpr=self.nfpr,
            )
            return self._process_response(
                raw,
                query,
                params["search_type"],
                params["country_code"],
                params["language"],
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        search_type: Optional[str] = None,
        country_code: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        page: Optional[int] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        start: Optional[int] = None,
        google_domain: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute an asynchronous search."""
        forbidden = _INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        params = self._resolve_params(
            search_type, country_code, language, device, page,
            gl, hl, start, google_domain,
        )
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                **params,
                light_request=self.light_request,
                nfpr=self.nfpr,
            )
            return self._process_response(
                raw,
                query,
                params["search_type"],
                params["country_code"],
                params["language"],
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self,
        raw: dict[str, Any],
        query: str,
        search_type: Optional[str],
        country_code: Optional[str],
        language: Optional[str],
    ) -> dict[str, Any]:
        """Truncate results, strip disabled fields, handle empty results."""
        if self.max_results:
            for key in ("results", "maps_results", "local_results", "news_results"):
                if key in raw and raw[key]:
                    raw[key] = raw[key][: self.max_results]

        if not self.include_knowledge_graph:
            raw.pop("knowledge_graph", None)
        if not self.include_questions:
            raw.pop("questions", None)
        if not self.include_related:
            raw.pop("related_queries", None)
            raw.pop("related_searches", None)
        if not self.include_maps_results:
            raw.pop("maps_results", None)
        if not self.include_ai_overviews:
            raw.pop("ai_overviews", None)
        if not self.include_local_results:
            raw.pop("local_results", None)
        if not self.include_top_stories:
            raw.pop("top_stories", None)
        if not self.include_hotel_results:
            raw.pop("hotel_results", None)
        if not self.include_news_results:
            raw.pop("news_results", None)
        if not self.include_shopping_ads:
            raw.pop("shopping_ads", None)
        if not self.include_top_ads:
            raw.pop("top_ads", None)
        if not self.include_bottom_ads:
            raw.pop("bottom_ads", None)

        has_results = (
            raw.get("results")
            or raw.get("maps_results")
            or raw.get("local_results")
            or raw.get("news_results")
            or raw.get("hotel_results")
        )
        if not has_results:
            suggestions = _generate_suggestions(search_type, country_code, language)
            raise ToolException(
                f"No search results found for '{query}'. "
                f"Suggestions: {', '.join(suggestions)}. "
                "Try modifying your search with one of these approaches."
            )

        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleAIMode
# ---------------------------------------------------------------------------


class ScavioGoogleAIModeInput(BaseModel):
    """Input schema for ScavioGoogleAIMode tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Question or prompt, 1-500 characters.",
    )

    device: Optional[Literal["desktop", "mobile"]] = Field(
        default=None,
        description="Device to emulate. Options: desktop (default), mobile.",
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Country of the search as a 2-letter ISO 3166-1 alpha-2 code (e.g. 'us', "
            "'gb'). Native v2 parameter."
        ),
    )

    google_domain: Optional[str] = Field(
        default=None,
        description="Regional Google domain to query (e.g. 'google.co.uk').",
    )

    location: Optional[str] = Field(
        default=None,
        description=(
            "Canonical location name (e.g. 'Austin, Texas, United States'), encoded to "
            "a UULE string server-side."
        ),
    )

    uule: Optional[str] = Field(
        default=None,
        description="Pre-encoded UULE location string. Takes priority over location.",
    )

    safe: Optional[Literal["active"]] = Field(
        default=None,
        description="SafeSearch filter. The only accepted value is 'active'.",
    )


class ScavioGoogleAIMode(BaseTool):  # type: ignore[override]
    """Ask Google AI Mode a question using the Scavio API.

    Returns Google's conversational answer as ``text_blocks`` with the cited
    sources in ``references``, plus ``shopping_results`` when the prompt is
    commercial. The response is flat -- there is no ``data`` wrapper.

    This endpoint is a strict subset of the SERP parameters: there is no
    ``start``, no time filter and no spelling-correction toggle.

    The API also accepts ``include_html``, which inlines Google's raw HTML.
    It is deliberately not exposed here (nor on ``ScavioSearch``): the payload
    is large enough to swamp a model's context and carries nothing the parsed
    fields do not already have.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleAIMode

            tool = ScavioGoogleAIMode()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {"query": "best way to cache LLM responses", "gl": "us"}
            )
    """

    name: str = "scavio_google_ai_mode"
    description: str = (
        "Ask Google AI Mode a question and get its generated answer with citations. "
        "Returns text_blocks and references at the top level (no data wrapper). Use "
        "for synthesised answers rather than a raw result list. Costs 1 credit per "
        "call. Input should be a question or prompt."
    )
    args_schema: Type[BaseModel] = ScavioGoogleAIModeInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleAIModeAPIWrapper = Field(
        default_factory=ScavioGoogleAIModeAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleAIModeAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        device: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        google_domain: Optional[str] = None,
        location: Optional[str] = None,
        uule: Optional[str] = None,
        safe: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ask Google AI Mode a question (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                device=device,
                hl=hl,
                gl=gl,
                google_domain=google_domain,
                location=location,
                uule=uule,
                safe=safe,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        device: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        google_domain: Optional[str] = None,
        location: Optional[str] = None,
        uule: Optional[str] = None,
        safe: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ask Google AI Mode a question (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                device=device,
                hl=hl,
                gl=gl,
                google_domain=google_domain,
                location=location,
                uule=uule,
                safe=safe,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        if not (raw.get("text_blocks") or raw.get("references")):
            raise ToolException(
                f"No Google AI Mode answer returned for '{query}'. Try rephrasing "
                "the prompt or dropping the gl/hl filters."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleMapsPlace
# ---------------------------------------------------------------------------


class ScavioGoogleMapsPlaceInput(BaseModel):
    """Input schema for ScavioGoogleMapsPlace tool."""

    model_config = ConfigDict(extra="allow")

    place_id: Optional[str] = Field(
        default=None,
        description="Google place id in ChIJ... form. Provide place_id or data_cid.",
    )

    data_cid: Optional[str] = Field(
        default=None,
        description=(
            "Numeric Google CID (e.g. '6324466594394968992'), as an alternative to "
            "place_id."
        ),
    )


class ScavioGoogleMapsPlace(BaseTool):  # type: ignore[override]
    """Fetch Google Maps place details using the Scavio API.

    Returns the place under a flat ``place_results`` key: name, address,
    phone, website, rating, review count, hours, coordinates and category.
    There is no ``data`` wrapper.

    Provide either ``place_id`` or ``data_cid``. This endpoint has no locale
    parameters -- hl, gl and google_domain do not exist here.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleMapsPlace

            tool = ScavioGoogleMapsPlace()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"})
    """

    name: str = "scavio_google_maps_place"
    description: str = (
        "Fetch full Google Maps details for one place by place_id or data_cid. Returns "
        "place_results at the top level with address, phone, website, rating, review "
        "count, hours and coordinates. Use the maps search tool first to obtain a "
        "place_id. Costs 1 credit per call. Provide place_id or data_cid."
    )
    args_schema: Type[BaseModel] = ScavioGoogleMapsPlaceInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleMapsPlaceAPIWrapper = Field(
        default_factory=ScavioGoogleMapsPlaceAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleMapsPlaceAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        place_id: Optional[str] = None,
        data_cid: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Maps place details (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                place_id=place_id,
                data_cid=data_cid,
            )
            return self._process_response(raw, place_id or data_cid or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        place_id: Optional[str] = None,
        data_cid: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Maps place details (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                place_id=place_id,
                data_cid=data_cid,
            )
            return self._process_response(raw, place_id or data_cid or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        if not raw.get("place_results"):
            raise ToolException(
                f"No Google Maps place found for '{subject}'. Verify the place_id or "
                "data_cid taken from a maps search result."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleMapsReviews
# ---------------------------------------------------------------------------


class ScavioGoogleMapsReviewsInput(BaseModel):
    """Input schema for ScavioGoogleMapsReviews tool."""

    model_config = ConfigDict(extra="allow")

    data_id: Optional[str] = Field(
        default=None,
        description="Maps data id in 0xHEX:0xHEX form. Provide data_id or place_id.",
    )

    place_id: Optional[str] = Field(
        default=None,
        description="Google place id in ChIJ... form, as an alternative to data_id.",
    )

    num: Optional[int] = Field(
        default=None,
        description="Reviews per page, 1-20.",
    )

    next_page_token: Optional[str] = Field(
        default=None,
        description="Pagination token from a previous response.",
    )

    sort_by: Optional[
        Literal["relevance", "newest", "highest_rating", "lowest_rating"]
    ] = Field(
        default=None,
        description=(
            "Review sort order. Options: relevance, newest, highest_rating, "
            "lowest_rating."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Country of the search as a 2-letter ISO 3166-1 alpha-2 code (e.g. 'us', "
            "'gb'). Native v2 parameter."
        ),
    )

    google_domain: Optional[str] = Field(
        default=None,
        description="Regional Google domain to query (e.g. 'google.co.uk').",
    )


class ScavioGoogleMapsReviews(BaseTool):  # type: ignore[override]
    """Fetch Google Maps reviews for a place using the Scavio API.

    Returns reviews at the top level under ``reviews``, with ``place_info``,
    ``topics`` and ``pagination`` alongside. There is no ``data`` wrapper.

    Provide either ``data_id`` or ``place_id``. There is no keyword filter
    for reviews -- the upstream does not support one.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleMapsReviews

            tool = ScavioGoogleMapsReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4", "sort_by": "newest"}
            )
    """

    name: str = "scavio_google_maps_reviews"
    description: str = (
        "Fetch Google Maps reviews for a place by data_id or place_id. Returns reviews "
        "at the top level with rating, text, author and date, plus place_info and "
        "pagination. sort_by accepts relevance, newest, highest_rating or "
        "lowest_rating. Costs 1 credit per call. Provide data_id or place_id."
    )
    args_schema: Type[BaseModel] = ScavioGoogleMapsReviewsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioGoogleMapsReviewsAPIWrapper = Field(
        default_factory=ScavioGoogleMapsReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleMapsReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        data_id: Optional[str] = None,
        place_id: Optional[str] = None,
        num: Optional[int] = None,
        next_page_token: Optional[str] = None,
        sort_by: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        google_domain: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Maps reviews for a place (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                data_id=data_id,
                place_id=place_id,
                num=num,
                next_page_token=next_page_token,
                sort_by=sort_by,
                hl=hl,
                gl=gl,
                google_domain=google_domain,
            )
            return self._process_response(raw, data_id or place_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        data_id: Optional[str] = None,
        place_id: Optional[str] = None,
        num: Optional[int] = None,
        next_page_token: Optional[str] = None,
        sort_by: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        google_domain: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Maps reviews for a place (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                data_id=data_id,
                place_id=place_id,
                num=num,
                next_page_token=next_page_token,
                sort_by=sort_by,
                hl=hl,
                gl=gl,
                google_domain=google_domain,
            )
            return self._process_response(raw, data_id or place_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        reviews = raw.get("reviews")
        if self.max_results and reviews:
            raw["reviews"] = reviews[: self.max_results]
        if not reviews:
            raise ToolException(
                f"No Google Maps reviews found for '{subject}'. The place may have "
                "no reviews, or the data_id/place_id may be wrong."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleShopping
# ---------------------------------------------------------------------------


class ScavioGoogleShoppingInput(BaseModel):
    """Input schema for ScavioGoogleShopping tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Product search query, 1-500 characters.",
    )

    device: Optional[Literal["desktop", "mobile"]] = Field(
        default=None,
        description="Device to emulate. Options: desktop (default), mobile.",
    )

    start: Optional[int] = Field(
        default=None,
        description=(
            "Result OFFSET, not a page number: 0 is the first page, 60 skips the first "
            "60 listings. Follow pagination.next in the response."
        ),
    )

    min_price: Optional[int] = Field(
        default=None,
        description="Minimum price filter.",
    )

    max_price: Optional[int] = Field(
        default=None,
        description="Maximum price filter.",
    )

    sort_by: Optional[int] = Field(
        default=None,
        description=(
            "Sort order as a NUMBER: 0 relevance, 1 price ascending, 2 price "
            "descending."
        ),
    )

    free_shipping: Optional[bool] = Field(
        default=None,
        description="Only listings with free shipping.",
    )

    on_sale: Optional[bool] = Field(
        default=None,
        description="Only listings currently on sale.",
    )

    shoprs: Optional[str] = Field(
        default=None,
        description=(
            "Opaque Google Shopping filter token lifted from filters or "
            "carousel_filters in a previous response."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Country of the search as a 2-letter ISO 3166-1 alpha-2 code (e.g. 'us', "
            "'gb'). Native v2 parameter."
        ),
    )

    google_domain: Optional[str] = Field(
        default=None,
        description="Regional Google domain to query (e.g. 'google.co.uk').",
    )

    location: Optional[str] = Field(
        default=None,
        description=(
            "Canonical location name (e.g. 'Austin, Texas, United States'), encoded to "
            "a UULE string server-side."
        ),
    )

    uule: Optional[str] = Field(
        default=None,
        description="Pre-encoded UULE location string. Takes priority over location.",
    )


class ScavioGoogleShopping(BaseTool):  # type: ignore[override]
    """Search Google Shopping listings using the Scavio API.

    Returns listings at the top level under ``shopping_results`` with title,
    price, merchant, rating and product links, plus ``filters``,
    ``carousel_filters`` and ``pagination``. There is no ``data`` wrapper.

    ``start`` is a result offset, not a page index, and ``sort_by`` is a
    number here (it is a string enum on the shopping product endpoint).

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleShopping

            tool = ScavioGoogleShopping()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {"query": "mechanical keyboard", "max_price": 150, "gl": "us"}
            )
    """

    name: str = "scavio_google_shopping"
    description: str = (
        "Search Google Shopping product listings. Returns shopping_results at the top "
        "level with title, price, merchant and rating. Supports price bounds, "
        "free_shipping/on_sale filters and numeric sort_by (0 relevance, 1 price asc, "
        "2 price desc). start is a 0-based result offset, not a page number. Costs 1 "
        "credit per call. Input should be a product query."
    )
    args_schema: Type[BaseModel] = ScavioGoogleShoppingInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioGoogleShoppingAPIWrapper = Field(
        default_factory=ScavioGoogleShoppingAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleShoppingAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        device: Optional[str] = None,
        start: Optional[int] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        sort_by: Optional[int] = None,
        free_shipping: Optional[bool] = None,
        on_sale: Optional[bool] = None,
        shoprs: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        google_domain: Optional[str] = None,
        location: Optional[str] = None,
        uule: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search Google Shopping listings (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                device=device,
                start=start,
                min_price=min_price,
                max_price=max_price,
                sort_by=sort_by,
                free_shipping=free_shipping,
                on_sale=on_sale,
                shoprs=shoprs,
                hl=hl,
                gl=gl,
                google_domain=google_domain,
                location=location,
                uule=uule,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        device: Optional[str] = None,
        start: Optional[int] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        sort_by: Optional[int] = None,
        free_shipping: Optional[bool] = None,
        on_sale: Optional[bool] = None,
        shoprs: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        google_domain: Optional[str] = None,
        location: Optional[str] = None,
        uule: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search Google Shopping listings (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                device=device,
                start=start,
                min_price=min_price,
                max_price=max_price,
                sort_by=sort_by,
                free_shipping=free_shipping,
                on_sale=on_sale,
                shoprs=shoprs,
                hl=hl,
                gl=gl,
                google_domain=google_domain,
                location=location,
                uule=uule,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        shopping_results = raw.get("shopping_results")
        if self.max_results and shopping_results:
            raw["shopping_results"] = shopping_results[: self.max_results]
        if not shopping_results:
            raise ToolException(
                f"No Google Shopping results found for '{query}'. Try broadening the "
                "query or removing the price filters."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleShoppingProduct
# ---------------------------------------------------------------------------


class ScavioGoogleShoppingProductInput(BaseModel):
    """Input schema for ScavioGoogleShoppingProduct tool."""

    model_config = ConfigDict(extra="allow")

    catalog_id: Optional[str] = Field(
        default=None,
        description="Durable product catalog id. When set, query is also required.",
    )

    query: Optional[str] = Field(
        default=None,
        description="Product query. Required whenever catalog_id is supplied.",
    )

    immersive_product_page_token: Optional[str] = Field(
        default=None,
        description="Immersive product page token from a shopping result.",
    )

    page_token: Optional[str] = Field(
        default=None,
        description="Alias of immersive_product_page_token.",
    )

    product_id: Optional[str] = Field(
        default=None,
        description="Google product id.",
    )

    device: Optional[Literal["desktop", "mobile", "tablet"]] = Field(
        default=None,
        description=(
            "Device to emulate. This is the only Google endpoint that also accepts "
            "tablet."
        ),
    )

    google_domain: Optional[str] = Field(
        default=None,
        description="Regional Google domain to query (e.g. 'google.co.uk').",
    )

    sort_by: Optional[
        Literal["base_price", "total_price", "promotion", "seller_rating"]
    ] = Field(
        default=None,
        description=(
            "Seller sort order as a STRING here: base_price, total_price, promotion or "
            "seller_rating."
        ),
    )

    load_all_stores: Optional[bool] = Field(
        default=None,
        description="Load every available store.",
    )

    more_stores: Optional[bool] = Field(
        default=None,
        description="Fetch additional stores.",
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Country of the search as a 2-letter ISO 3166-1 alpha-2 code (e.g. 'us', "
            "'gb'). Native v2 parameter."
        ),
    )

    location: Optional[str] = Field(
        default=None,
        description=(
            "Canonical location name (e.g. 'Austin, Texas, United States'), encoded to "
            "a UULE string server-side."
        ),
    )

    uule: Optional[str] = Field(
        default=None,
        description="Pre-encoded UULE location string. Takes priority over location.",
    )


class ScavioGoogleShoppingProduct(BaseTool):  # type: ignore[override]
    """Fetch a Google Shopping product and its sellers.

    Returns the product at the top level under ``product_results``,
    including specifications, reviews and a ``stores`` list of sellers with
    prices. There is no ``data`` wrapper.

    Provide one of ``catalog_id`` (with ``query``),
    ``immersive_product_page_token``, ``page_token`` or ``product_id``. Use
    ScavioGoogleShoppingStores to page past the first batch of sellers.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleShoppingProduct

            tool = ScavioGoogleShoppingProduct()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {"catalog_id": "1234567890", "query": "mechanical keyboard"}
            )
    """

    name: str = "scavio_google_shopping_product"
    description: str = (
        "Fetch Google Shopping product detail and its sellers. Returns product_results "
        "at the top level with specs, reviews and a stores list of merchants and "
        "prices. Provide catalog_id (plus query), immersive_product_page_token, "
        "page_token or product_id. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleShoppingProductInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleShoppingProductAPIWrapper = Field(
        default_factory=ScavioGoogleShoppingProductAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleShoppingProductAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        catalog_id: Optional[str] = None,
        query: Optional[str] = None,
        immersive_product_page_token: Optional[str] = None,
        page_token: Optional[str] = None,
        product_id: Optional[str] = None,
        device: Optional[str] = None,
        google_domain: Optional[str] = None,
        sort_by: Optional[str] = None,
        load_all_stores: Optional[bool] = None,
        more_stores: Optional[bool] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        location: Optional[str] = None,
        uule: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a Google Shopping product and its sellers (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                catalog_id=catalog_id,
                query=query,
                immersive_product_page_token=immersive_product_page_token,
                page_token=page_token,
                product_id=product_id,
                device=device,
                google_domain=google_domain,
                sort_by=sort_by,
                load_all_stores=load_all_stores,
                more_stores=more_stores,
                hl=hl,
                gl=gl,
                location=location,
                uule=uule,
            )
            return self._process_response(raw, catalog_id or product_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        catalog_id: Optional[str] = None,
        query: Optional[str] = None,
        immersive_product_page_token: Optional[str] = None,
        page_token: Optional[str] = None,
        product_id: Optional[str] = None,
        device: Optional[str] = None,
        google_domain: Optional[str] = None,
        sort_by: Optional[str] = None,
        load_all_stores: Optional[bool] = None,
        more_stores: Optional[bool] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        location: Optional[str] = None,
        uule: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a Google Shopping product and its sellers (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                catalog_id=catalog_id,
                query=query,
                immersive_product_page_token=immersive_product_page_token,
                page_token=page_token,
                product_id=product_id,
                device=device,
                google_domain=google_domain,
                sort_by=sort_by,
                load_all_stores=load_all_stores,
                more_stores=more_stores,
                hl=hl,
                gl=gl,
                location=location,
                uule=uule,
            )
            return self._process_response(raw, catalog_id or product_id or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        if not raw.get("product_results"):
            raise ToolException(
                f"No Google Shopping product found for '{subject}'. Note that query "
                "is required alongside catalog_id."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleShoppingStores
# ---------------------------------------------------------------------------


class ScavioGoogleShoppingStoresInput(BaseModel):
    """Input schema for ScavioGoogleShoppingStores tool."""

    model_config = ConfigDict(extra="allow")

    catalog_id: str = Field(
        description="The same catalog_id used on the shopping product call.",
    )

    next_page_token: str = Field(
        description=(
            "Continuation token from a previous shopping product or stores response."
        ),
    )


class ScavioGoogleShoppingStores(BaseTool):  # type: ignore[override]
    """Page through more sellers for a Google Shopping product.

    Returns the next batch of merchants under ``product_results.stores``.
    There is no ``data`` wrapper.

    Both arguments are required and both come from a prior
    ScavioGoogleShoppingProduct response. This endpoint takes no locale
    parameters at all.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleShoppingStores

            tool = ScavioGoogleShoppingStores()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {"catalog_id": "1234567890", "next_page_token": "CAoQAA"}
            )
    """

    name: str = "scavio_google_shopping_stores"
    description: str = (
        "Fetch more sellers for a Google Shopping product, continuing a shopping "
        "product lookup. Returns product_results.stores at the top level. Requires the "
        "same catalog_id plus the next_page_token from the previous response. Costs 1 "
        "credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleShoppingStoresInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleShoppingStoresAPIWrapper = Field(
        default_factory=ScavioGoogleShoppingStoresAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleShoppingStoresAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        catalog_id: str,
        next_page_token: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Page through more sellers for a Google Shopping product (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                catalog_id=catalog_id,
                next_page_token=next_page_token,
            )
            return self._process_response(raw, catalog_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        catalog_id: str,
        next_page_token: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Page through more sellers for a Google Shopping product (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                catalog_id=catalog_id,
                next_page_token=next_page_token,
            )
            return self._process_response(raw, catalog_id)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], catalog_id: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        if not raw.get("product_results"):
            raise ToolException(
                "No further Google Shopping sellers found for catalog_id "
                f"'{catalog_id}'. The next_page_token may be stale or exhausted."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleFlights
# ---------------------------------------------------------------------------


class ScavioGoogleFlightsInput(BaseModel):
    """Input schema for ScavioGoogleFlights tool."""

    model_config = ConfigDict(extra="allow")

    departure_id: str = Field(
        description=(
            "Departure airport IATA code (e.g. 'JFK'). Comma-separated multiples are "
            "allowed."
        ),
    )

    arrival_id: str = Field(
        description=(
            "Arrival airport IATA code (e.g. 'LHR'). Comma-separated multiples are "
            "allowed."
        ),
    )

    outbound_date: str = Field(
        description="Outbound date as YYYY-MM-DD.",
    )

    type: Optional[int] = Field(
        default=None,
        description="Trip type: 1 round trip, 2 one way, 3 multi-city.",
    )

    return_date: Optional[str] = Field(
        default=None,
        description="Return date as YYYY-MM-DD. Required when type is 1 (round trip).",
    )

    adults: Optional[int] = Field(
        default=None,
        description="Number of adults, 1-9.",
    )

    children: Optional[int] = Field(
        default=None,
        description="Number of children, 0-9.",
    )

    infants_in_seat: Optional[int] = Field(
        default=None,
        description="Infants in their own seat, 0-4.",
    )

    infants_on_lap: Optional[int] = Field(
        default=None,
        description="Infants on a lap, 0-4.",
    )

    travel_class: Optional[int] = Field(
        default=None,
        description="Cabin: 1 economy, 2 premium economy, 3 business, 4 first.",
    )

    stops: Optional[int] = Field(
        default=None,
        description=(
            "Stops: 0 any, 1 nonstop only, 2 at most 1 stop, 3 at most 2 stops."
        ),
    )

    sort_by: Optional[int] = Field(
        default=None,
        description=(
            "Sort: 1 top, 2 price, 3 departure time, 4 arrival time, 5 duration, 6 "
            "emissions."
        ),
    )

    include_airlines: Optional[str] = Field(
        default=None,
        description="Comma-separated airline or alliance codes to include.",
    )

    exclude_airlines: Optional[str] = Field(
        default=None,
        description="Comma-separated airline or alliance codes to exclude.",
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Country of the search as a 2-letter ISO 3166-1 alpha-2 code (e.g. 'us', "
            "'gb'). Native v2 parameter."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        description="Currency as a 3-letter ISO 4217 code (e.g. 'USD').",
    )


class ScavioGoogleFlights(BaseTool):  # type: ignore[override]
    """Search Google Flights using the Scavio API.

    Returns itineraries at the top level under ``best_flights`` and
    ``other_flights``, each with legs, airlines, duration, stops, emissions
    and price. There is no ``data`` wrapper.

    ``return_date`` is required when ``type`` is 1 (round trip). Airport
    codes are IATA and may be comma-separated for multi-airport searches.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleFlights

            tool = ScavioGoogleFlights()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "departure_id": "JFK",
                    "arrival_id": "LHR",
                    "outbound_date": "2026-09-01",
                    "type": 2,
                }
            )
    """

    name: str = "scavio_google_flights"
    description: str = (
        "Search Google Flights between two airports. Returns best_flights and "
        "other_flights at the top level with legs, airlines, duration, stops, "
        "emissions and price. Airports are IATA codes; dates are YYYY-MM-DD; "
        "return_date is required for round trips (type=1). Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleFlightsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioGoogleFlightsAPIWrapper = Field(
        default_factory=ScavioGoogleFlightsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleFlightsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        departure_id: str,
        arrival_id: str,
        outbound_date: str,
        type: Optional[int] = None,
        return_date: Optional[str] = None,
        adults: Optional[int] = None,
        children: Optional[int] = None,
        infants_in_seat: Optional[int] = None,
        infants_on_lap: Optional[int] = None,
        travel_class: Optional[int] = None,
        stops: Optional[int] = None,
        sort_by: Optional[int] = None,
        include_airlines: Optional[str] = None,
        exclude_airlines: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search Google Flights (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                departure_id=departure_id,
                arrival_id=arrival_id,
                outbound_date=outbound_date,
                type=type,
                return_date=return_date,
                adults=adults,
                children=children,
                infants_in_seat=infants_in_seat,
                infants_on_lap=infants_on_lap,
                travel_class=travel_class,
                stops=stops,
                sort_by=sort_by,
                include_airlines=include_airlines,
                exclude_airlines=exclude_airlines,
                hl=hl,
                gl=gl,
                currency=currency,
            )
            return self._process_response(raw, f"{departure_id}-{arrival_id}")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        departure_id: str,
        arrival_id: str,
        outbound_date: str,
        type: Optional[int] = None,
        return_date: Optional[str] = None,
        adults: Optional[int] = None,
        children: Optional[int] = None,
        infants_in_seat: Optional[int] = None,
        infants_on_lap: Optional[int] = None,
        travel_class: Optional[int] = None,
        stops: Optional[int] = None,
        sort_by: Optional[int] = None,
        include_airlines: Optional[str] = None,
        exclude_airlines: Optional[str] = None,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        currency: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search Google Flights (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                departure_id=departure_id,
                arrival_id=arrival_id,
                outbound_date=outbound_date,
                type=type,
                return_date=return_date,
                adults=adults,
                children=children,
                infants_in_seat=infants_in_seat,
                infants_on_lap=infants_on_lap,
                travel_class=travel_class,
                stops=stops,
                sort_by=sort_by,
                include_airlines=include_airlines,
                exclude_airlines=exclude_airlines,
                hl=hl,
                gl=gl,
                currency=currency,
            )
            return self._process_response(raw, f"{departure_id}-{arrival_id}")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], route: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        best_flights = raw.get("best_flights")
        if self.max_results and best_flights:
            raw["best_flights"] = best_flights[: self.max_results]
        other_flights = raw.get("other_flights")
        if self.max_results and other_flights:
            raw["other_flights"] = other_flights[: self.max_results]
        if not (best_flights or other_flights):
            raise ToolException(
                f"No Google Flights results found for '{route}'. Check the IATA "
                "codes and the date format (YYYY-MM-DD); round trips also need "
                "return_date."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleHotels
# ---------------------------------------------------------------------------


class ScavioGoogleHotelsInput(BaseModel):
    """Input schema for ScavioGoogleHotels tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Search query, up to 200 characters. Use a '<City> hotels' form.",
    )

    check_in_date: str = Field(
        description="Check-in date as YYYY-MM-DD.",
    )

    check_out_date: str = Field(
        description="Check-out date as YYYY-MM-DD.",
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Country of the search as a 2-letter ISO 3166-1 alpha-2 code (e.g. 'us', "
            "'gb'). Native v2 parameter."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        description="Currency as a 3-letter ISO 4217 code (e.g. 'USD').",
    )

    sort_by: Optional[int] = Field(
        default=None,
        description="Sort: 3 lowest price, 8 highest rating, 13 most reviewed.",
    )

    min_price: Optional[int] = Field(
        default=None,
        description="Minimum nightly price.",
    )

    max_price: Optional[int] = Field(
        default=None,
        description="Maximum nightly price.",
    )

    rating: Optional[int] = Field(
        default=None,
        description="Minimum guest rating: 7 for 3.5+, 8 for 4.0+, 9 for 4.5+.",
    )

    hotel_class: Optional[str] = Field(
        default=None,
        description="Comma-separated star ratings 2-5 as a string (e.g. '4,5').",
    )

    amenities: Optional[str] = Field(
        default=None,
        description="Comma-separated amenity ids.",
    )

    property_types: Optional[str] = Field(
        default=None,
        description="Comma-separated property-type ids ('12' is vacation rentals).",
    )

    free_cancellation: Optional[bool] = Field(
        default=None,
        description="Only properties offering free cancellation.",
    )

    eco_certified: Optional[bool] = Field(
        default=None,
        description="Only eco-certified properties.",
    )

    special_offers: Optional[bool] = Field(
        default=None,
        description="Only properties with special offers.",
    )

    next_page_token: Optional[str] = Field(
        default=None,
        description="Pagination token from a previous response.",
    )

    limit: Optional[int] = Field(
        default=None,
        description="Number of properties to return, 1-20.",
    )


class ScavioGoogleHotels(BaseTool):  # type: ignore[override]
    """Search Google Hotels using the Scavio API.

    Returns properties at the top level under ``properties`` with name,
    rate, rating, review count, amenities and a ``detail_token``. There is
    no ``data`` wrapper.

    Feed a property's ``detail_token`` into ScavioGoogleHotelsDetail for
    booking sources. Both dates are required and must be re-sent on the
    detail call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleHotels

            tool = ScavioGoogleHotels()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "query": "Lisbon hotels",
                    "check_in_date": "2026-09-01",
                    "check_out_date": "2026-09-04",
                }
            )
    """

    name: str = "scavio_google_hotels"
    description: str = (
        "Search Google Hotels for a destination and date range. Returns properties at "
        "the top level with name, nightly rate, rating, review count, amenities and a "
        "detail_token for the hotel detail tool. Supports price, rating, star-class "
        "and amenity filters. Costs 1 credit per call. Query should use a '<City> "
        "hotels' form."
    )
    args_schema: Type[BaseModel] = ScavioGoogleHotelsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioGoogleHotelsAPIWrapper = Field(
        default_factory=ScavioGoogleHotelsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleHotelsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        check_in_date: str,
        check_out_date: str,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        currency: Optional[str] = None,
        sort_by: Optional[int] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        rating: Optional[int] = None,
        hotel_class: Optional[str] = None,
        amenities: Optional[str] = None,
        property_types: Optional[str] = None,
        free_cancellation: Optional[bool] = None,
        eco_certified: Optional[bool] = None,
        special_offers: Optional[bool] = None,
        next_page_token: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search Google Hotels (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                check_in_date=check_in_date,
                check_out_date=check_out_date,
                hl=hl,
                gl=gl,
                currency=currency,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
                rating=rating,
                hotel_class=hotel_class,
                amenities=amenities,
                property_types=property_types,
                free_cancellation=free_cancellation,
                eco_certified=eco_certified,
                special_offers=special_offers,
                next_page_token=next_page_token,
                limit=limit,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        check_in_date: str,
        check_out_date: str,
        hl: Optional[str] = None,
        gl: Optional[str] = None,
        currency: Optional[str] = None,
        sort_by: Optional[int] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        rating: Optional[int] = None,
        hotel_class: Optional[str] = None,
        amenities: Optional[str] = None,
        property_types: Optional[str] = None,
        free_cancellation: Optional[bool] = None,
        eco_certified: Optional[bool] = None,
        special_offers: Optional[bool] = None,
        next_page_token: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search Google Hotels (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                check_in_date=check_in_date,
                check_out_date=check_out_date,
                hl=hl,
                gl=gl,
                currency=currency,
                sort_by=sort_by,
                min_price=min_price,
                max_price=max_price,
                rating=rating,
                hotel_class=hotel_class,
                amenities=amenities,
                property_types=property_types,
                free_cancellation=free_cancellation,
                eco_certified=eco_certified,
                special_offers=special_offers,
                next_page_token=next_page_token,
                limit=limit,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        properties = raw.get("properties")
        if self.max_results and properties:
            raw["properties"] = properties[: self.max_results]
        if not properties:
            raise ToolException(
                f"No Google Hotels properties found for '{query}'. Try a '<City> "
                "hotels' query, different dates, or looser price filters."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleHotelsDetail
# ---------------------------------------------------------------------------


class ScavioGoogleHotelsDetailInput(BaseModel):
    """Input schema for ScavioGoogleHotelsDetail tool."""

    model_config = ConfigDict(extra="allow")

    detail_token: str = Field(
        description="detail_token taken from a property in a Google Hotels listing.",
    )

    check_in_date: str = Field(
        description=(
            "Check-in date as YYYY-MM-DD. Must be re-sent -- the token alone is not "
            "enough."
        ),
    )

    check_out_date: str = Field(
        description="Check-out date as YYYY-MM-DD.",
    )

    currency: Optional[str] = Field(
        default=None,
        description="Currency as a 3-letter ISO 4217 code (e.g. 'USD').",
    )

    gl: Optional[str] = Field(
        default=None,
        description=(
            "Country of the search as a 2-letter ISO 3166-1 alpha-2 code (e.g. 'us', "
            "'gb'). Native v2 parameter."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )


class ScavioGoogleHotelsDetail(BaseTool):  # type: ignore[override]
    """Fetch Google Hotels property details using the Scavio API.

    Returns the property at the top level under ``property``, including
    ``property.booking_sources`` -- the per-site rates. There is no ``data``
    wrapper.

    The ``detail_token`` comes from a ScavioGoogleHotels result and the same
    check-in and check-out dates must be sent again.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleHotelsDetail

            tool = ScavioGoogleHotelsDetail()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "detail_token": "CggIu...",
                    "check_in_date": "2026-09-01",
                    "check_out_date": "2026-09-04",
                }
            )
    """

    name: str = "scavio_google_hotels_detail"
    description: str = (
        "Fetch full Google Hotels details for one property from its detail_token, "
        "including booking_sources with per-site rates. Returns property at the top "
        "level. The same check-in and check-out dates used in the hotel search must be "
        "re-sent. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleHotelsDetailInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleHotelsDetailAPIWrapper = Field(
        default_factory=ScavioGoogleHotelsDetailAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleHotelsDetailAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        detail_token: str,
        check_in_date: str,
        check_out_date: str,
        currency: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Hotels property details (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                detail_token=detail_token,
                check_in_date=check_in_date,
                check_out_date=check_out_date,
                currency=currency,
                gl=gl,
                hl=hl,
            )
            return self._process_response(raw, detail_token)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        detail_token: str,
        check_in_date: str,
        check_out_date: str,
        currency: Optional[str] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Hotels property details (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                detail_token=detail_token,
                check_in_date=check_in_date,
                check_out_date=check_out_date,
                currency=currency,
                gl=gl,
                hl=hl,
            )
            return self._process_response(raw, detail_token)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], detail_token: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        if not raw.get("property"):
            raise ToolException(
                "No Google Hotels detail found for that detail_token. Tokens "
                "expire; re-run the hotel search to get a fresh one."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleTrends
# ---------------------------------------------------------------------------


class ScavioGoogleTrendsInput(BaseModel):
    """Input schema for ScavioGoogleTrends tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description="Search term, or several comma-separated terms to compare.",
    )

    geo: Optional[str] = Field(
        default=None,
        description=(
            "Location code in UPPERCASE (e.g. 'US', 'GB', 'US-CA'). Worldwide when "
            "omitted. This endpoint has no gl parameter."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )

    date: Optional[str] = Field(
        default=None,
        description=(
            "Time range string (e.g. 'today 12-m', 'now 7-d', '2024-01-01 "
            "2024-12-31')."
        ),
    )

    tz: Optional[str] = Field(
        default=None,
        description="Timezone offset in minutes, sent as a string.",
    )

    data_type: Optional[
        Literal[
            "TIMESERIES",
            "GEO_MAP",
            "GEO_MAP_0",
            "RELATED_QUERIES",
            "RELATED_TOPICS",
        ]
    ] = Field(
        default=None,
        description=(
            "Which dataset to return, UPPERCASE: TIMESERIES, GEO_MAP, GEO_MAP_0, "
            "RELATED_QUERIES or RELATED_TOPICS."
        ),
    )

    cat: Optional[str] = Field(
        default=None,
        description="Category id as a string (e.g. '71').",
    )

    gprop: Optional[Literal["images", "news", "youtube", "froogle"]] = Field(
        default=None,
        description=(
            "Google property filter: images, news, youtube or froogle. Omit for web "
            "search."
        ),
    )

    region: Optional[Literal["COUNTRY", "REGION", "DMA", "CITY"]] = Field(
        default=None,
        description=(
            "Resolution for GEO_MAP data, UPPERCASE: COUNTRY, REGION, DMA or CITY."
        ),
    )


class ScavioGoogleTrends(BaseTool):  # type: ignore[override]
    """Fetch Google Trends interest data using the Scavio API.

    Returns ``interest_over_time`` (with ``timeline_data``) and
    ``interest_by_region`` at the top level. There is no ``data`` wrapper.

    Comma-separated terms in ``query`` produce a comparison. Note this
    endpoint uses an uppercase ``geo`` instead of the ``gl`` used elsewhere
    in the Google family.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleTrends

            tool = ScavioGoogleTrends()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {"query": "langchain,llamaindex", "geo": "US", "date": "today 12-m"}
            )
    """

    name: str = "scavio_google_trends"
    description: str = (
        "Fetch Google Trends interest data for one or more search terms (comma- "
        "separate to compare). Returns interest_over_time and interest_by_region at "
        "the top level. Location uses an uppercase geo code (US, GB, US-CA), not gl. "
        "data_type selects TIMESERIES, GEO_MAP, RELATED_QUERIES or RELATED_TOPICS. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleTrendsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGoogleTrendsAPIWrapper = Field(
        default_factory=ScavioGoogleTrendsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleTrendsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        geo: Optional[str] = None,
        hl: Optional[str] = None,
        date: Optional[str] = None,
        tz: Optional[str] = None,
        data_type: Optional[str] = None,
        cat: Optional[str] = None,
        gprop: Optional[str] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Trends interest data (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                geo=geo,
                hl=hl,
                date=date,
                tz=tz,
                data_type=data_type,
                cat=cat,
                gprop=gprop,
                region=region,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        geo: Optional[str] = None,
        hl: Optional[str] = None,
        date: Optional[str] = None,
        tz: Optional[str] = None,
        data_type: Optional[str] = None,
        cat: Optional[str] = None,
        gprop: Optional[str] = None,
        region: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Trends interest data (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                geo=geo,
                hl=hl,
                date=date,
                tz=tz,
                data_type=data_type,
                cat=cat,
                gprop=gprop,
                region=region,
            )
            return self._process_response(raw, query)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        if not (raw.get("interest_over_time") or raw.get("interest_by_region")):
            raise ToolException(
                f"No Google Trends data found for '{query}'. Low-volume terms return "
                "nothing; try a broader term, a wider date range or no geo."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioGoogleTrending
# ---------------------------------------------------------------------------


class ScavioGoogleTrendingInput(BaseModel):
    """Input schema for ScavioGoogleTrending tool."""

    model_config = ConfigDict(extra="allow")

    geo: str = Field(
        description=(
            "Country code (e.g. 'US'). This is the required driver -- the endpoint has "
            "no query field."
        ),
    )

    hl: Optional[str] = Field(
        default=None,
        description=(
            "UI language as an ISO 639-1 code (e.g. 'en', 'fr'). Native v2 parameter."
        ),
    )

    hours: Optional[int] = Field(
        default=None,
        description="Trending window in hours: 4, 24, 48 or 168.",
    )

    cat: Optional[int] = Field(
        default=None,
        description="Category id as a NUMBER, 0-20 (0 is all).",
    )

    sort: Optional[Literal["relevance", "search_volume", "recency", "title"]] = Field(
        default=None,
        description=(
            "Sort order. The field is named sort, not sort_by. Options: relevance, "
            "search_volume, recency, title."
        ),
    )

    status: Optional[Literal["all", "active"]] = Field(
        default=None,
        description="Filter by trend status: all or active.",
    )


class ScavioGoogleTrending(BaseTool):  # type: ignore[override]
    """Fetch Google Trending Now for a country using the Scavio API.

    Returns currently trending searches at the top level under ``trends``,
    with search volume and status. There is no ``data`` wrapper.

    This is the only Google endpoint with no query field at all -- ``geo``
    is the driver. The sort argument is named ``sort``, not ``sort_by``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGoogleTrending

            tool = ScavioGoogleTrending()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"geo": "US", "hours": 24})
    """

    name: str = "scavio_google_trending"
    description: str = (
        "Fetch Google Trending Now for a country. Returns trends at the top level with "
        "search volume and status. geo is required and replaces the query; hours "
        "selects the 4/24/48/168-hour window; sort (not sort_by) accepts relevance, "
        "search_volume, recency or title. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGoogleTrendingInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioGoogleTrendingAPIWrapper = Field(
        default_factory=ScavioGoogleTrendingAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGoogleTrendingAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        geo: str,
        hl: Optional[str] = None,
        hours: Optional[int] = None,
        cat: Optional[int] = None,
        sort: Optional[str] = None,
        status: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Trending Now for a country (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                geo=geo,
                hl=hl,
                hours=hours,
                cat=cat,
                sort=sort,
                status=status,
            )
            return self._process_response(raw, geo)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        geo: str,
        hl: Optional[str] = None,
        hours: Optional[int] = None,
        cat: Optional[int] = None,
        sort: Optional[str] = None,
        status: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch Google Trending Now for a country (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                geo=geo,
                hl=hl,
                hours=hours,
                cat=cat,
                sort=sort,
                status=status,
            )
            return self._process_response(raw, geo)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], geo: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        trends = raw.get("trends")
        if self.max_results and trends:
            raw["trends"] = trends[: self.max_results]
        if not trends:
            raise ToolException(
                f"No Google trending searches found for '{geo}'. Verify the country "
                "code and try a wider hours window."
            )
        return raw
