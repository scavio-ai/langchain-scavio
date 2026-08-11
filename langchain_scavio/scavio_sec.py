"""Scavio SEC EDGAR tools for LangChain agents.

Every URL, parameter name, enum and credit cost below is copied from the
Scavio route definition rather than derived from the tool name.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import (
    ScavioSECCompanyAPIWrapper,
    ScavioSECConceptAPIWrapper,
    ScavioSECFactsAPIWrapper,
    ScavioSECFilingsAPIWrapper,
    ScavioSECLookupAPIWrapper,
    ScavioSECSearchAPIWrapper,
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
# ScavioSECLookup
# --------------------------------------------------------------------------


class ScavioSECLookupInput(BaseModel):
    """Input schema for the ScavioSECLookup tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Ticker, company name, or a fragment of either."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Maximum filers to return, 1-100. Default: 10."
        ),
    )

    exchange: Optional[str] = Field(
        default=None,
        description=(
            "Listing exchange filter, matched case-insensitively. Filers listed with "
            "no exchange are excluded by any value. Options: NASDAQ, NYSE, OTC, CBOE."
        ),
    )


class ScavioSECLookup(BaseTool):  # type: ignore[override]
    """SEC EDGAR: START HERE. Resolve a company name or ticker (AAPL) to the CIK
    (0000320193) every other SEC EDGAR endpoint is keyed by.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioSECLookup

            tool = ScavioSECLookup()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "AAPL"})
    """

    name: str = "scavio_sec_lookup"
    description: str = (
        "SEC EDGAR: START HERE. Resolve a company name or ticker (AAPL) to the CIK "
        "(0000320193) every other SEC EDGAR endpoint is keyed by. Costs 1 credit per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioSECLookupInput
    handle_tool_error: bool = True

    api_wrapper: ScavioSECLookupAPIWrapper = Field(
        default_factory=ScavioSECLookupAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioSECLookupAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        limit: Optional[int] = None,
        exchange: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/lookup (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                limit=limit,
                exchange=exchange,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        limit: Optional[int] = None,
        exchange: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/lookup (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                limit=limit,
                exchange=exchange,
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
                f"No SEC EDGAR filers found for '{identifier}'. Resolve the CIK with "
                "ScavioSECLookup first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioSECCompany
# --------------------------------------------------------------------------


class ScavioSECCompanyInput(BaseModel):
    """Input schema for the ScavioSECCompany tool."""

    model_config = ConfigDict(extra="allow")

    cik: Optional[str] = Field(
        default=None,
        description=(
            "Central Index Key: 320193, 0000320193 or CIK0000320193. A ticker is "
            "accepted here too."
        ),
    )

    ticker: Optional[str] = Field(
        default=None,
        description=(
            "Stock ticker, dotted or dashed (BRK.B / BRK-B). WINS over cik when both "
            "are given."
        ),
    )


class ScavioSECCompany(BaseTool):  # type: ignore[override]
    """SEC EDGAR: SEC filer profile: legal and former names, SIC industry, EIN, LEI,
    state of incorporation, fiscal year end, addresses, every ticker, and a preview
    of its 10 most recent filings.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioSECCompany

            tool = ScavioSECCompany()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"ticker": "AAPL"})
    """

    name: str = "scavio_sec_company"
    description: str = (
        "SEC EDGAR: SEC filer profile: legal and former names, SIC industry, EIN, LEI, "
        "state of incorporation, fiscal year end, addresses, every ticker, and a "
        "preview of its 10 most recent filings. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioSECCompanyInput
    handle_tool_error: bool = True

    api_wrapper: ScavioSECCompanyAPIWrapper = Field(
        default_factory=ScavioSECCompanyAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioSECCompanyAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        cik: Optional[str] = None,
        ticker: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/company (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                cik=cik,
                ticker=ticker,
            )
            return self._process_response(raw, _first_identifier(cik, ticker))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        cik: Optional[str] = None,
        ticker: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/company (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                cik=cik,
                ticker=ticker,
            )
            return self._process_response(raw, _first_identifier(cik, ticker))
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
                f"No SEC EDGAR company found for '{identifier}'. Resolve the CIK with "
                "ScavioSECLookup first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioSECFilings
# --------------------------------------------------------------------------


class ScavioSECFilingsInput(BaseModel):
    """Input schema for the ScavioSECFilings tool."""

    model_config = ConfigDict(extra="allow")

    cik: Optional[str] = Field(
        default=None,
        description=(
            "Central Index Key: 320193, 0000320193 or CIK0000320193. A ticker is "
            "accepted here too."
        ),
    )

    ticker: Optional[str] = Field(
        default=None,
        description=(
            "Stock ticker, dotted or dashed (BRK.B / BRK-B). WINS over cik when both "
            "are given."
        ),
    )

    form: Optional[Union[str, list[str]]] = Field(
        default=None,
        description=(
            "Form filter: \"10-K\", [\"10-K\", \"10-Q\"] or \"10-K,8-K\". Matched "
            "against the form AND its root form, so 10-K also returns 10-K/A "
            "amendments."
        ),
    )

    date_from: Optional[str] = Field(
        default=None,
        description=(
            "Earliest date to include, YYYY-MM-DD."
        ),
    )

    date_to: Optional[str] = Field(
        default=None,
        description=(
            "Latest date to include, YYYY-MM-DD."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page number, 1-based."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Filings to return, 1-500. Default: 50."
        ),
    )

    include_history: Optional[bool] = Field(
        default=None,
        description=(
            "Also read the archived filing shards (up to 10). Still one credit; "
            "history_truncated flags a filer that had more. Default: False."
        ),
    )


class ScavioSECFilings(BaseTool):  # type: ignore[override]
    """SEC EDGAR: A page of one filer's filings: accession number, form and root form,
    filing and period dates, 8-K item codes, direct document links.

    Costs 1 credit per call.

    Pagination: page + limit.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioSECFilings

            tool = ScavioSECFilings()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"ticker": "AAPL", "form": "10-K"})
    """

    name: str = "scavio_sec_filings"
    description: str = (
        "SEC EDGAR: A page of one filer's filings: accession number, form and root "
        "form, filing and period dates, 8-K item codes, direct document links. "
        "Pagination: page + limit. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioSECFilingsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioSECFilingsAPIWrapper = Field(
        default_factory=ScavioSECFilingsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioSECFilingsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        cik: Optional[str] = None,
        ticker: Optional[str] = None,
        form: Optional[Union[str, list[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        include_history: Optional[bool] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/filings (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                cik=cik,
                ticker=ticker,
                form=form,
                date_from=date_from,
                date_to=date_to,
                page=page,
                limit=limit,
                include_history=include_history,
            )
            return self._process_response(raw, _first_identifier(cik, ticker, form))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        cik: Optional[str] = None,
        ticker: Optional[str] = None,
        form: Optional[Union[str, list[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: Optional[int] = None,
        limit: Optional[int] = None,
        include_history: Optional[bool] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/filings (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                cik=cik,
                ticker=ticker,
                form=form,
                date_from=date_from,
                date_to=date_to,
                page=page,
                limit=limit,
                include_history=include_history,
            )
            return self._process_response(raw, _first_identifier(cik, ticker, form))
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
                f"No SEC EDGAR filings found for '{identifier}'. Resolve the CIK with "
                "ScavioSECLookup first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioSECConcept
# --------------------------------------------------------------------------


class ScavioSECConceptInput(BaseModel):
    """Input schema for the ScavioSECConcept tool."""

    model_config = ConfigDict(extra="allow")

    cik: Optional[str] = Field(
        default=None,
        description=(
            "Central Index Key: 320193, 0000320193 or CIK0000320193. A ticker is "
            "accepted here too."
        ),
    )

    ticker: Optional[str] = Field(
        default=None,
        description=(
            "Stock ticker, dotted or dashed (BRK.B / BRK-B). WINS over cik when both "
            "are given."
        ),
    )

    concept: str = Field(
        description=(
            "XBRL tag, CASE-SENSITIVE: 'netincomeloss' is a 404 upstream, not a match. "
            "Use the facts endpoint to list what a filer actually reports."
        ),
    )

    taxonomy: Optional[str] = Field(
        default=None,
        description=(
            "XBRL taxonomy: us-gaap, dei, ifrs-full or srt. Default: us-gaap."
        ),
    )

    unit: Optional[str] = Field(
        default=None,
        description=(
            "Unit of measure to filter on, e.g. USD or USD/shares."
        ),
    )

    form: Optional[str] = Field(
        default=None,
        description=(
            "Form filter. EXACT match here, so '10-K' excludes 10-K/A."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Values to return, 1-2000. Default: 250."
        ),
    )


class ScavioSECConcept(BaseTool):  # type: ignore[override]
    """SEC EDGAR: Every value a filer reported for one XBRL concept, newest period
    first, with the form and filing each number came from.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioSECConcept

            tool = ScavioSECConcept()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"ticker": "AAPL", "concept": "NetIncomeLoss"})
    """

    name: str = "scavio_sec_concept"
    description: str = (
        "SEC EDGAR: Every value a filer reported for one XBRL concept, newest period "
        "first, with the form and filing each number came from. Costs 1 credit per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioSECConceptInput
    handle_tool_error: bool = True

    api_wrapper: ScavioSECConceptAPIWrapper = Field(
        default_factory=ScavioSECConceptAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioSECConceptAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        concept: str,
        cik: Optional[str] = None,
        ticker: Optional[str] = None,
        taxonomy: Optional[str] = None,
        unit: Optional[str] = None,
        form: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/concept (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                concept=concept,
                cik=cik,
                ticker=ticker,
                taxonomy=taxonomy,
                unit=unit,
                form=form,
                limit=limit,
            )
            return self._process_response(raw, _first_identifier(concept))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        concept: str,
        cik: Optional[str] = None,
        ticker: Optional[str] = None,
        taxonomy: Optional[str] = None,
        unit: Optional[str] = None,
        form: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/concept (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                concept=concept,
                cik=cik,
                ticker=ticker,
                taxonomy=taxonomy,
                unit=unit,
                form=form,
                limit=limit,
            )
            return self._process_response(raw, _first_identifier(concept))
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
                f"No SEC EDGAR values found for '{identifier}'. Resolve the CIK with "
                "ScavioSECLookup first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioSECFacts
# --------------------------------------------------------------------------


class ScavioSECFactsInput(BaseModel):
    """Input schema for the ScavioSECFacts tool."""

    model_config = ConfigDict(extra="allow")

    cik: Optional[str] = Field(
        default=None,
        description=(
            "Central Index Key: 320193, 0000320193 or CIK0000320193. A ticker is "
            "accepted here too."
        ),
    )

    ticker: Optional[str] = Field(
        default=None,
        description=(
            "Stock ticker, dotted or dashed (BRK.B / BRK-B). WINS over cik when both "
            "are given."
        ),
    )

    taxonomy: Optional[str] = Field(
        default=None,
        description=(
            "Restrict the index to one XBRL taxonomy."
        ),
    )

    query: Optional[str] = Field(
        default=None,
        description=(
            "Case-insensitive substring matched against the tag name and its label."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        description=(
            "Concepts to return, 1-2000. Default: 250."
        ),
    )


class ScavioSECFacts(BaseTool):  # type: ignore[override]
    """SEC EDGAR: The index of every XBRL concept a filer reports -- tag, label,
    description, units, most recent value. This is how you find what to ask
    /sec/concept for.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioSECFacts

            tool = ScavioSECFacts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"ticker": "AAPL", "query": "revenue"})
    """

    name: str = "scavio_sec_facts"
    description: str = (
        "SEC EDGAR: The index of every XBRL concept a filer reports -- tag, label, "
        "description, units, most recent value. This is how you find what to ask "
        "/sec/concept for. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioSECFactsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioSECFactsAPIWrapper = Field(
        default_factory=ScavioSECFactsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioSECFactsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        cik: Optional[str] = None,
        ticker: Optional[str] = None,
        taxonomy: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/facts (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                cik=cik,
                ticker=ticker,
                taxonomy=taxonomy,
                query=query,
                limit=limit,
            )
            return self._process_response(raw, _first_identifier(cik, ticker, query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        cik: Optional[str] = None,
        ticker: Optional[str] = None,
        taxonomy: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/facts (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                cik=cik,
                ticker=ticker,
                taxonomy=taxonomy,
                query=query,
                limit=limit,
            )
            return self._process_response(raw, _first_identifier(cik, ticker, query))
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
                f"No SEC EDGAR concepts found for '{identifier}'. Resolve the CIK with "
                "ScavioSECLookup first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioSECSearch
# --------------------------------------------------------------------------


class ScavioSECSearchInput(BaseModel):
    """Input schema for the ScavioSECSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: Optional[str] = Field(
        default=None,
        description=(
            "Full-text query. A quoted phrase is exact, bare words are a bag of terms. "
            "Optional -- a cik, form or date filter on its own is a valid search."
        ),
    )

    cik: Optional[Union[str, list[str]]] = Field(
        default=None,
        description=(
            "One CIK or a list of them. Tickers are accepted here too."
        ),
    )

    ticker: Optional[Union[str, list[str]]] = Field(
        default=None,
        description=(
            "One ticker or a list of them."
        ),
    )

    form: Optional[Union[str, list[str]]] = Field(
        default=None,
        description=(
            "One form type or a list of them."
        ),
    )

    date_from: Optional[str] = Field(
        default=None,
        description=(
            "Earliest filing date, YYYY-MM-DD. Coverage starts 2001."
        ),
    )

    date_to: Optional[str] = Field(
        default=None,
        description=(
            "Latest date to include, YYYY-MM-DD."
        ),
    )

    location: Optional[Union[str, list[str]]] = Field(
        default=None,
        description=(
            "EDGAR's own jurisdiction codes: CA, NY, and alphanumeric codes for "
            "foreign jurisdictions. One code or a list."
        ),
    )

    sort: Optional[Literal["relevance", "newest", "oldest"]] = Field(
        default=None,
        description=(
            "Sort order for the results. Options: relevance, newest, oldest. Default: "
            "relevance."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-100, at 100 documents each. The index refuses a result "
            "window past 10,000."
        ),
    )


class ScavioSECSearch(BaseTool):  # type: ignore[override]
    """SEC EDGAR: EDGAR full-text search, 2001-today: each hit is the matching DOCUMENT
    with its URL, form, filing date and filer identity, plus facets by company,
    form, industry and state.

    Costs 1 credit per call.

    Pagination: page -- capped at 100 (100 docs/page) because the index refuses a
    result window past 10,000.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioSECSearch

            tool = ScavioSECSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "climate risk", "form": "10-K"})
    """

    name: str = "scavio_sec_search"
    description: str = (
        "SEC EDGAR: EDGAR full-text search, 2001-today: each hit is the matching "
        "DOCUMENT with its URL, form, filing date and filer identity, plus facets by "
        "company, form, industry and state. Pagination: page -- capped at 100 (100 "
        "docs/page) because the index refuses a result window past 10,000. Costs 1 "
        "credit per call."
    )
    args_schema: Type[BaseModel] = ScavioSECSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioSECSearchAPIWrapper = Field(
        default_factory=ScavioSECSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioSECSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: Optional[str] = None,
        cik: Optional[Union[str, list[str]]] = None,
        ticker: Optional[Union[str, list[str]]] = None,
        form: Optional[Union[str, list[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        location: Optional[Union[str, list[str]]] = None,
        sort: Optional[Literal["relevance", "newest", "oldest"]] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                cik=cik,
                ticker=ticker,
                form=form,
                date_from=date_from,
                date_to=date_to,
                location=location,
                sort=sort,
                page=page,
            )
            return self._process_response(raw, _first_identifier(query, cik, ticker))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: Optional[str] = None,
        cik: Optional[Union[str, list[str]]] = None,
        ticker: Optional[Union[str, list[str]]] = None,
        form: Optional[Union[str, list[str]]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        location: Optional[Union[str, list[str]]] = None,
        sort: Optional[Literal["relevance", "newest", "oldest"]] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/sec/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                cik=cik,
                ticker=ticker,
                form=form,
                date_from=date_from,
                date_to=date_to,
                location=location,
                sort=sort,
                page=page,
            )
            return self._process_response(raw, _first_identifier(query, cik, ticker))
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
                f"No SEC EDGAR results found for '{identifier}'. Resolve the CIK with "
                "ScavioSECLookup first."
            )
        return raw
