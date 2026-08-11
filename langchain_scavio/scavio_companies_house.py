"""Scavio Companies House tools for LangChain agents.

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
    ScavioCompaniesHouseCompanyAPIWrapper,
    ScavioCompaniesHouseFilingHistoryAPIWrapper,
    ScavioCompaniesHouseOfficersAPIWrapper,
    ScavioCompaniesHouseSearchAPIWrapper,
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
# ScavioCompaniesHouseSearch
# --------------------------------------------------------------------------


class ScavioCompaniesHouseSearchInput(BaseModel):
    """Input schema for the ScavioCompaniesHouseSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Company name or number. Matches CURRENT AND FORMER names."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-50, at 20 rows each. Capped at 50 because the register "
            "only serves the first 1000 matches per term whatever hit count it prints. "
            "Default: 1."
        ),
    )


class ScavioCompaniesHouseSearch(BaseTool):  # type: ignore[override]
    """Companies House: START HERE. Search the UK register by name and get the
    company_number every other endpoint is keyed by, plus status, incorporation date
    and registered office.

    Costs 1 credit per call.

    Pagination: page -- 20 per page, CAPPED AT PAGE 50. The register serves a
    1000-result WINDOW per term whatever hit count it prints (it claims 10,000 for a
    broad term then answers page 51 with HTTP 416).

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioCompaniesHouseSearch

            tool = ScavioCompaniesHouseSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "Monzo"})
    """

    name: str = "scavio_companies_house_search"
    description: str = (
        "Companies House: START HERE. Search the UK register by name and get the "
        "company_number every other endpoint is keyed by, plus status, incorporation "
        "date and registered office. Pagination: page -- 20 per page, CAPPED AT PAGE "
        "50. The register serves a 1000-result WINDOW per term whatever hit count it "
        "prints (it claims 10,000 for a broad term then answers page 51 with HTTP "
        "416). Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioCompaniesHouseSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioCompaniesHouseSearchAPIWrapper = Field(
        default_factory=ScavioCompaniesHouseSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioCompaniesHouseSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/companieshouse/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                page=page,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/companieshouse/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                page=page,
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
                f"No Companies House results found for '{identifier}'. Resolve the "
                "company number with ScavioCompaniesHouseSearch first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioCompaniesHouseCompany
# --------------------------------------------------------------------------


class ScavioCompaniesHouseCompanyInput(BaseModel):
    """Input schema for the ScavioCompaniesHouseCompany tool."""

    model_config = ConfigDict(extra="allow")

    company_number: str = Field(
        description=(
            "Company number. Zero-padded and upper-cased for you, so numbers off a "
            "letterhead or a spreadsheet that ate leading zeros still resolve. SC, NI, "
            "OC, SO, NC, FC, BR and CE prefixes are supported."
        ),
    )


class ScavioCompaniesHouseCompany(BaseTool):  # type: ignore[override]
    """Companies House: Full UK register entry: status, type, incorporation and
    dissolution dates, registered office, SIC codes, previous names, accounts and
    confirmation-statement due dates with overdue flags.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioCompaniesHouseCompany

            tool = ScavioCompaniesHouseCompany()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"company_number": "09446231"})
    """

    name: str = "scavio_companies_house_company"
    description: str = (
        "Companies House: Full UK register entry: status, type, incorporation and "
        "dissolution dates, registered office, SIC codes, previous names, accounts and "
        "confirmation-statement due dates with overdue flags. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioCompaniesHouseCompanyInput
    handle_tool_error: bool = True

    api_wrapper: ScavioCompaniesHouseCompanyAPIWrapper = Field(
        default_factory=ScavioCompaniesHouseCompanyAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioCompaniesHouseCompanyAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        company_number: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/companieshouse/company (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                company_number=company_number,
            )
            return self._process_response(raw, _first_identifier(company_number))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        company_number: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/companieshouse/company (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                company_number=company_number,
            )
            return self._process_response(raw, _first_identifier(company_number))
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
                f"No Companies House company found for '{identifier}'. Resolve the "
                "company number with ScavioCompaniesHouseSearch first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioCompaniesHouseOfficers
# --------------------------------------------------------------------------


class ScavioCompaniesHouseOfficersInput(BaseModel):
    """Input schema for the ScavioCompaniesHouseOfficers tool."""

    model_config = ConfigDict(extra="allow")

    company_number: str = Field(
        description=(
            "Company number. Zero-padded and upper-cased for you, so numbers off a "
            "letterhead or a spreadsheet that ate leading zeros still resolve. SC, NI, "
            "OC, SO, NC, FC, BR and CE prefixes are supported."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based, 35 officers per page. No upper bound: past the last "
            "page the register answers a plain 200 with an empty list. Default: 1."
        ),
    )


class ScavioCompaniesHouseOfficers(BaseTool):  # type: ignore[override]
    """Companies House: UK company officers current and resigned: name, role,
    appointment and resignation dates, correspondence address, nationality, month-
    and-year DOB, identity-verification status.

    Costs 1 credit per call.

    Pagination: page -- 35 per page, NO upper page bound. Past the last page the
    register answers an ordinary 200 with an empty list, identical to a company with
    no officers.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioCompaniesHouseOfficers

            tool = ScavioCompaniesHouseOfficers()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"company_number": "09446231"})
    """

    name: str = "scavio_companies_house_officers"
    description: str = (
        "Companies House: UK company officers current and resigned: name, role, "
        "appointment and resignation dates, correspondence address, nationality, "
        "month-and-year DOB, identity-verification status. Pagination: page -- 35 per "
        "page, NO upper page bound. Past the last page the register answers an "
        "ordinary 200 with an empty list, identical to a company with no officers. "
        "Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioCompaniesHouseOfficersInput
    handle_tool_error: bool = True

    api_wrapper: ScavioCompaniesHouseOfficersAPIWrapper = Field(
        default_factory=ScavioCompaniesHouseOfficersAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioCompaniesHouseOfficersAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        company_number: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/companieshouse/officers (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                company_number=company_number,
                page=page,
            )
            return self._process_response(raw, _first_identifier(company_number))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        company_number: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/companieshouse/officers (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                company_number=company_number,
                page=page,
            )
            return self._process_response(raw, _first_identifier(company_number))
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
                f"No Companies House officers found for '{identifier}'. Resolve the "
                "company number with ScavioCompaniesHouseSearch first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioCompaniesHouseFilingHistory
# --------------------------------------------------------------------------


class ScavioCompaniesHouseFilingHistoryInput(BaseModel):
    """Input schema for the ScavioCompaniesHouseFilingHistory tool."""

    model_config = ConfigDict(extra="allow")

    company_number: str = Field(
        description=(
            "Company number. Zero-padded and upper-cased for you, so numbers off a "
            "letterhead or a spreadsheet that ate leading zeros still resolve. SC, NI, "
            "OC, SO, NC, FC, BR and CE prefixes are supported."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. No upper bound: past the last page the register "
            "answers a plain 200 with an empty list. Default: 1."
        ),
    )


class ScavioCompaniesHouseFilingHistory(BaseTool):  # type: ignore[override]
    """Companies House: UK filings, most recent first: date, filing type code (AA,
    CS01, SH03), description, register annotations and child documents, link to the
    filed PDF with page count.

    Costs 1 credit per call.

    Pagination: page -- NO upper page bound; past the last page it is an ordinary
    200 with an empty list.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioCompaniesHouseFilingHistory

            tool = ScavioCompaniesHouseFilingHistory()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"company_number": "09446231"})
    """

    name: str = "scavio_companies_house_filing_history"
    description: str = (
        "Companies House: UK filings, most recent first: date, filing type code (AA, "
        "CS01, SH03), description, register annotations and child documents, link to "
        "the filed PDF with page count. Pagination: page -- NO upper page bound; past "
        "the last page it is an ordinary 200 with an empty list. Costs 1 credit per "
        "call."
    )
    args_schema: Type[BaseModel] = ScavioCompaniesHouseFilingHistoryInput
    handle_tool_error: bool = True

    api_wrapper: ScavioCompaniesHouseFilingHistoryAPIWrapper = Field(
        default_factory=(
            ScavioCompaniesHouseFilingHistoryAPIWrapper  # type: ignore[arg-type]
        )
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioCompaniesHouseFilingHistoryAPIWrapper(
                **api_kwargs
            )
        super().__init__(**kwargs)

    def _run(
        self,
        company_number: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/companieshouse/filing-history (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                company_number=company_number,
                page=page,
            )
            return self._process_response(raw, _first_identifier(company_number))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        company_number: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/companieshouse/filing-history (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                company_number=company_number,
                page=page,
            )
            return self._process_response(raw, _first_identifier(company_number))
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
                f"No Companies House filings found for '{identifier}'. Resolve the "
                "company number with ScavioCompaniesHouseSearch first."
            )
        return raw
