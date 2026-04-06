"""ScavioSearch tool for LangChain agents."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import ScavioSearchAPIWrapper

logger = logging.getLogger(__name__)

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

    search_type: Optional[Literal["classic", "news", "maps", "images", "lens"]] = (
        Field(
            default=None,
            description=(
                'Type of search to perform. '
                'Use "classic" (default) for most queries, INCLUDING those asking for '
                '"latest" or "recent" general information. '
                'Use "news" ONLY for politics, sports, or major current events '
                'covered by mainstream media. '
                'Use "maps" for local businesses, restaurants, or place lookups. '
                'Use "images" when the user explicitly asks for visual results.'
            ),
        )
    )

    country_code: Optional[str] = Field(
        default=None,
        description=(
            'ISO 3166-1 alpha-2 country code for localized results '
            '(e.g., "us", "gb", "de", "fr", "jp"). '
            "Set this when the user explicitly mentions a country or region, "
            "or when results should be geographically relevant. "
            "Default is None (no country filter)."
        ),
    )

    language: Optional[str] = Field(
        default=None,
        description=(
            'ISO 639-1 language code (e.g., "en", "fr", "de"). '
            "Set this when the user requests results in a specific language. "
            "Default is None (no language filter)."
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
            "Result page number (1-indexed). "
            "Only increase this if the user explicitly asks for more results "
            "or the previous page did not contain the needed information."
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
                model=ChatOpenAI(model="gpt-4o"),
                tools=[ScavioSearch()],
                system_prompt="You are a research assistant.",
            )
    """

    name: str = "scavio_search"
    description: str = (
        "Search the web using the Scavio Search API. Returns search results "
        "with titles, URLs, descriptions, knowledge graphs, news results, "
        "and related questions. Use for any query requiring real-time or "
        "recent web information. Input should be a search query."
    )
    args_schema: Type[BaseModel] = ScavioSearchInput
    handle_tool_error: bool = True

    # Instantiation-only parameters (not controllable by the LLM).
    max_results: Optional[int] = 5
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
    ) -> dict[str, Any]:
        """Merge LLM-provided params with init-time defaults."""
        return {
            "search_type": search_type or self.search_type or "classic",
            "country_code": country_code or self.country_code,
            "language": language or self.language,
            "device": device or self.device or "desktop",
            "page": page or self.page or 1,
        }

    def _run(
        self,
        query: str,
        search_type: Optional[str] = None,
        country_code: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        page: Optional[int] = None,
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
        params = self._resolve_params(search_type, country_code, language, device, page)
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
        params = self._resolve_params(search_type, country_code, language, device, page)
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
