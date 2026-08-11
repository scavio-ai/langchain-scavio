"""Scavio Glassdoor tools for LangChain agents.

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
    ScavioGlassdoorCompaniesAPIWrapper,
    ScavioGlassdoorCompanyAPIWrapper,
    ScavioGlassdoorReviewsAPIWrapper,
    ScavioGlassdoorSalariesAPIWrapper,
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
# ScavioGlassdoorCompanies
# --------------------------------------------------------------------------


class ScavioGlassdoorCompaniesInput(BaseModel):
    """Input schema for the ScavioGlassdoorCompanies tool."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(
        description=(
            "Company NAME to resolve into an employer_id."
        ),
    )


class ScavioGlassdoorCompanies(BaseTool):  # type: ignore[override]
    """Glassdoor: START HERE. Search Glassdoor by company NAME and resolve it to the
    employer_id every other endpoint needs.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGlassdoorCompanies

            tool = ScavioGlassdoorCompanies()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "Stripe"})
    """

    name: str = "scavio_glassdoor_companies"
    description: str = (
        "Glassdoor: START HERE. Search Glassdoor by company NAME and resolve it to the "
        "employer_id every other endpoint needs. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGlassdoorCompaniesInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGlassdoorCompaniesAPIWrapper = Field(
        default_factory=ScavioGlassdoorCompaniesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGlassdoorCompaniesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/glassdoor/companies (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
            )
            return self._process_response(raw, _first_identifier(query))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/glassdoor/companies (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
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
                f"No Glassdoor companies found for '{identifier}'. Resolve the "
                "employer_id with ScavioGlassdoorCompanies first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioGlassdoorCompany
# --------------------------------------------------------------------------


class ScavioGlassdoorCompanyInput(BaseModel):
    """Input schema for the ScavioGlassdoorCompany tool."""

    model_config = ConfigDict(extra="allow")

    employer_id: Optional[str] = Field(
        default=None,
        description=(
            "Glassdoor employer id as a STRING -- a JSON number is rejected. Accepts "
            "1699, E1699 or IE1699."
        ),
    )

    company: Optional[str] = Field(
        default=None,
        description=(
            "Company name. COSMETIC only: the profile resolves on employer_id alone, "
            "it is ignored entirely when url is set, and it does not satisfy the "
            "required-identifier rule."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Any glassdoor.com employer URL (/Overview/, /Reviews/, /Salary/). "
            "Non-glassdoor.com hosts are rejected."
        ),
    )


class ScavioGlassdoorCompany(BaseTool):  # type: ignore[override]
    """Glassdoor: Glassdoor employer profile: ratings, star distribution, CEO approval,
    size/revenue bands, awards, five server-rendered reviews, plus reviews_url /
    salaries_url.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGlassdoorCompany

            tool = ScavioGlassdoorCompany()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"employer_id": "1699"})
    """

    name: str = "scavio_glassdoor_company"
    description: str = (
        "Glassdoor: Glassdoor employer profile: ratings, star distribution, CEO "
        "approval, size/revenue bands, awards, five server-rendered reviews, plus "
        "reviews_url / salaries_url. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGlassdoorCompanyInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGlassdoorCompanyAPIWrapper = Field(
        default_factory=ScavioGlassdoorCompanyAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGlassdoorCompanyAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        employer_id: Optional[str] = None,
        company: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/glassdoor/company (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                employer_id=employer_id,
                company=company,
                url=url,
            )
            return self._process_response(
                raw, _first_identifier(employer_id, company, url)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        employer_id: Optional[str] = None,
        company: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/glassdoor/company (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                employer_id=employer_id,
                company=company,
                url=url,
            )
            return self._process_response(
                raw, _first_identifier(employer_id, company, url)
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
                f"No Glassdoor company found for '{identifier}'. Resolve the "
                "employer_id with ScavioGlassdoorCompanies first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioGlassdoorReviews
# --------------------------------------------------------------------------


class ScavioGlassdoorReviewsInput(BaseModel):
    """Input schema for the ScavioGlassdoorReviews tool."""

    model_config = ConfigDict(extra="allow")

    employer_id: Optional[str] = Field(
        default=None,
        description=(
            "Glassdoor employer id as a STRING -- a JSON number is rejected. Accepts "
            "1699, E1699 or IE1699."
        ),
    )

    company: Optional[str] = Field(
        default=None,
        description=(
            "Company name. COSMETIC only: the profile resolves on employer_id alone, "
            "it is ignored entirely when url is set, and it does not satisfy the "
            "required-identifier rule."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Pass back reviews_url from the company endpoint to skip the resolve fetch "
            "-- addressing this endpoint by employer_id costs two upstream fetches."
        ),
    )

    category: Optional[
        Literal[
            "career_development",
            "compensation",
            "culture",
            "diversity_and_inclusion",
            "management",
            "work_life_balance",
        ]
    ] = Field(
        default=None,
        description=(
            "Review category filter. Closed set: Glassdoor IGNORES an unknown value "
            "and returns the unfiltered set under a 200. Options: career_development, "
            "compensation, culture, diversity_and_inclusion, management, "
            "work_life_balance."
        ),
    )

    employment_status: Optional[
        Literal["full_time", "part_time", "contract", "intern"]
    ] = Field(
        default=None,
        description=(
            "Reviewer employment status filter. Closed set for the same reason as "
            "category. Options: full_time, part_time, contract, intern."
        ),
    )


class ScavioGlassdoorReviews(BaseTool):  # type: ignore[override]
    """Glassdoor: Up to THREE full Glassdoor reviews with per-axis scores, pros, cons,
    advice, employer response -- plus complete rating statistics and per-job-title
    review counts.

    Costs 1 credit per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGlassdoorReviews

            tool = ScavioGlassdoorReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"employer_id": "1699"})
    """

    name: str = "scavio_glassdoor_reviews"
    description: str = (
        "Glassdoor: Up to THREE full Glassdoor reviews with per-axis scores, pros, "
        "cons, advice, employer response -- plus complete rating statistics and "
        "per-job-title review counts. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGlassdoorReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGlassdoorReviewsAPIWrapper = Field(
        default_factory=ScavioGlassdoorReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGlassdoorReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        employer_id: Optional[str] = None,
        company: Optional[str] = None,
        url: Optional[str] = None,
        category: Optional[
            Literal[
                "career_development",
                "compensation",
                "culture",
                "diversity_and_inclusion",
                "management",
                "work_life_balance",
            ]
        ] = None,
        employment_status: Optional[
            Literal["full_time", "part_time", "contract", "intern"]
        ] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/glassdoor/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                employer_id=employer_id,
                company=company,
                url=url,
                category=category,
                employment_status=employment_status,
            )
            return self._process_response(
                raw, _first_identifier(employer_id, company, url)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        employer_id: Optional[str] = None,
        company: Optional[str] = None,
        url: Optional[str] = None,
        category: Optional[
            Literal[
                "career_development",
                "compensation",
                "culture",
                "diversity_and_inclusion",
                "management",
                "work_life_balance",
            ]
        ] = None,
        employment_status: Optional[
            Literal["full_time", "part_time", "contract", "intern"]
        ] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/glassdoor/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                employer_id=employer_id,
                company=company,
                url=url,
                category=category,
                employment_status=employment_status,
            )
            return self._process_response(
                raw, _first_identifier(employer_id, company, url)
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
                f"No Glassdoor reviews found for '{identifier}'. Resolve the "
                "employer_id with ScavioGlassdoorCompanies first."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioGlassdoorSalaries
# --------------------------------------------------------------------------


class ScavioGlassdoorSalariesInput(BaseModel):
    """Input schema for the ScavioGlassdoorSalaries tool."""

    model_config = ConfigDict(extra="allow")

    employer_id: Optional[str] = Field(
        default=None,
        description=(
            "Glassdoor employer id as a STRING -- a JSON number is rejected. Accepts "
            "1699, E1699 or IE1699."
        ),
    )

    company: Optional[str] = Field(
        default=None,
        description=(
            "Company name. COSMETIC only: the profile resolves on employer_id alone, "
            "it is ignored entirely when url is set, and it does not satisfy the "
            "required-identifier rule."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Pass back salaries_url from the company endpoint to skip the resolve "
            "fetch."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 10 job titles per page; page_count on the response "
            "says how many exist."
        ),
    )


class ScavioGlassdoorSalaries(BaseTool):  # type: ignore[override]
    """Glassdoor: Glassdoor salaries by job title: base-pay and total-pay percentiles
    P10-P90 with medians, sample counts, currency, pay period, last-reported date.

    Costs 1 credit per call.

    Pagination: page -- 10 job titles per page; page_count on the response is how
    many exist.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioGlassdoorSalaries

            tool = ScavioGlassdoorSalaries()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"employer_id": "1699"})
    """

    name: str = "scavio_glassdoor_salaries"
    description: str = (
        "Glassdoor: Glassdoor salaries by job title: base-pay and total-pay "
        "percentiles P10-P90 with medians, sample counts, currency, pay period, "
        "last-reported date. Pagination: page -- 10 job titles per page; page_count on "
        "the response is how many exist. Costs 1 credit per call."
    )
    args_schema: Type[BaseModel] = ScavioGlassdoorSalariesInput
    handle_tool_error: bool = True

    api_wrapper: ScavioGlassdoorSalariesAPIWrapper = Field(
        default_factory=ScavioGlassdoorSalariesAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioGlassdoorSalariesAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        employer_id: Optional[str] = None,
        company: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/glassdoor/salaries (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                employer_id=employer_id,
                company=company,
                url=url,
                page=page,
            )
            return self._process_response(
                raw, _first_identifier(employer_id, company, url)
            )
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        employer_id: Optional[str] = None,
        company: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/glassdoor/salaries (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                employer_id=employer_id,
                company=company,
                url=url,
                page=page,
            )
            return self._process_response(
                raw, _first_identifier(employer_id, company, url)
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
                f"No Glassdoor salaries found for '{identifier}'. Resolve the "
                "employer_id with ScavioGlassdoorCompanies first."
            )
        return raw
