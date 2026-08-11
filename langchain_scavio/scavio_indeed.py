"""Scavio Indeed tools for LangChain agents.

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
    ScavioIndeedCompanyAPIWrapper,
    ScavioIndeedCompanyReviewsAPIWrapper,
    ScavioIndeedJobAPIWrapper,
    ScavioIndeedSearchAPIWrapper,
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
# ScavioIndeedSearch
# --------------------------------------------------------------------------


class ScavioIndeedSearchInput(BaseModel):
    """Input schema for the ScavioIndeedSearch tool."""

    model_config = ConfigDict(extra="allow")

    query: Optional[str] = Field(
        default=None,
        description=(
            "Job title, keyword or company. Optional if location is set."
        ),
    )

    location: Optional[str] = Field(
        default=None,
        description=(
            "City and state, postal code, state, country or 'Remote'. Usable with no "
            "query at all -- that returns every posting in a metro."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 10 postings per page."
        ),
    )

    radius: Optional[Literal[0, 5, 10, 15, 25, 35, 50, 100]] = Field(
        default=None,
        description=(
            "Search radius in miles. Closed set: Indeed IGNORES any other value and "
            "returns the unfiltered set, so asking for 7 would silently buy 50. "
            "Options: 0, 5, 10, 15, 25, 35, 50, 100. Default: 50."
        ),
    )

    max_age_days: Optional[Literal[1, 3, 7, 14]] = Field(
        default=None,
        description=(
            "Only postings published within this many days. Closed set for the same "
            "reason as radius. Options: 1, 3, 7, 14."
        ),
    )

    job_type: Optional[
        Literal["full_time", "part_time", "contract", "temporary", "internship"]
    ] = Field(
        default=None,
        description=(
            "Employment type filter. Options: full_time, part_time, contract, "
            "temporary, internship."
        ),
    )

    min_salary: Optional[float] = Field(
        default=None,
        description=(
            "Minimum salary. This filters on INDEED'S OWN ESTIMATE for the role, not a "
            "posted figure, so postings that publish no salary still match."
        ),
    )

    remote: Optional[bool] = Field(
        default=None,
        description=(
            "Only return remote roles."
        ),
    )


class ScavioIndeedSearch(BaseTool):  # type: ignore[override]
    """Indeed: Indeed job postings: title, employer, rating, location, salary range,
    job type, benefits, posting age, apply route.

    Costs 2 credits per call.

    Pagination: page -- 10 postings per page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioIndeedSearch

            tool = ScavioIndeedSearch()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"query": "data engineer", "location": "Austin, TX"})
    """

    name: str = "scavio_indeed_search"
    description: str = (
        "Indeed: Indeed job postings: title, employer, rating, location, salary range, "
        "job type, benefits, posting age, apply route. Pagination: page -- 10 postings "
        "per page. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioIndeedSearchInput
    handle_tool_error: bool = True

    api_wrapper: ScavioIndeedSearchAPIWrapper = Field(
        default_factory=ScavioIndeedSearchAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioIndeedSearchAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        query: Optional[str] = None,
        location: Optional[str] = None,
        page: Optional[int] = None,
        radius: Optional[Literal[0, 5, 10, 15, 25, 35, 50, 100]] = None,
        max_age_days: Optional[Literal[1, 3, 7, 14]] = None,
        job_type: Optional[
            Literal["full_time", "part_time", "contract", "temporary", "internship"]
        ] = None,
        min_salary: Optional[float] = None,
        remote: Optional[bool] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/indeed/search (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                query=query,
                location=location,
                page=page,
                radius=radius,
                max_age_days=max_age_days,
                job_type=job_type,
                min_salary=min_salary,
                remote=remote,
            )
            return self._process_response(raw, _first_identifier(query, location))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: Optional[str] = None,
        location: Optional[str] = None,
        page: Optional[int] = None,
        radius: Optional[Literal[0, 5, 10, 15, 25, 35, 50, 100]] = None,
        max_age_days: Optional[Literal[1, 3, 7, 14]] = None,
        job_type: Optional[
            Literal["full_time", "part_time", "contract", "temporary", "internship"]
        ] = None,
        min_salary: Optional[float] = None,
        remote: Optional[bool] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/indeed/search (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                query=query,
                location=location,
                page=page,
                radius=radius,
                max_age_days=max_age_days,
                job_type=job_type,
                min_salary=min_salary,
                remote=remote,
            )
            return self._process_response(raw, _first_identifier(query, location))
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
                f"No Indeed results found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioIndeedJob
# --------------------------------------------------------------------------


class ScavioIndeedJobInput(BaseModel):
    """Input schema for the ScavioIndeedJob tool."""

    model_config = ConfigDict(extra="allow")

    job_id: str = Field(
        description=(
            "16-hex Indeed job key, or any indeed.com URL carrying jk= (/viewjob, "
            "/rc/clk, /pagead/clk)."
        ),
    )


class ScavioIndeedJob(BaseTool):  # type: ignore[override]
    """Indeed: One Indeed posting in full: description text and HTML, structured
    salary, employment types, benefits, geocoded address, original ATS link.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioIndeedJob

            tool = ScavioIndeedJob()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"job_id": "a1b2c3d4e5f60718"})
    """

    name: str = "scavio_indeed_job"
    description: str = (
        "Indeed: One Indeed posting in full: description text and HTML, structured "
        "salary, employment types, benefits, geocoded address, original ATS link. "
        "Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioIndeedJobInput
    handle_tool_error: bool = True

    api_wrapper: ScavioIndeedJobAPIWrapper = Field(
        default_factory=ScavioIndeedJobAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioIndeedJobAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        job_id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/indeed/job (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                job_id=job_id,
            )
            return self._process_response(raw, _first_identifier(job_id))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        job_id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/indeed/job (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                job_id=job_id,
            )
            return self._process_response(raw, _first_identifier(job_id))
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
                f"No Indeed job found for '{identifier}'. Verify the identifiers you "
                "passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioIndeedCompany
# --------------------------------------------------------------------------


class ScavioIndeedCompanyInput(BaseModel):
    """Input schema for the ScavioIndeedCompany tool."""

    model_config = ConfigDict(extra="allow")

    company: str = Field(
        description=(
            "indeed.com/cmp/<slug> slug or a full profile URL. Slugs are untidy, e.g. "
            "'Tata-Consultancy-Services-(tcs)'."
        ),
    )


class ScavioIndeedCompany(BaseTool):  # type: ignore[override]
    """Indeed: Indeed employer profile: description, industry, HQ, size, revenue, CEO
    approval, per-category ratings, reported salaries, open roles.

    Costs 2 credits per call.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioIndeedCompany

            tool = ScavioIndeedCompany()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"company": "Stripe"})
    """

    name: str = "scavio_indeed_company"
    description: str = (
        "Indeed: Indeed employer profile: description, industry, HQ, size, revenue, "
        "CEO approval, per-category ratings, reported salaries, open roles. Costs 2 "
        "credits per call."
    )
    args_schema: Type[BaseModel] = ScavioIndeedCompanyInput
    handle_tool_error: bool = True

    api_wrapper: ScavioIndeedCompanyAPIWrapper = Field(
        default_factory=ScavioIndeedCompanyAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioIndeedCompanyAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        company: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/indeed/company (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                company=company,
            )
            return self._process_response(raw, _first_identifier(company))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        company: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/indeed/company (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                company=company,
            )
            return self._process_response(raw, _first_identifier(company))
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
                f"No Indeed company found for '{identifier}'. Verify the identifiers "
                "you passed."
            )
        return raw


# --------------------------------------------------------------------------
# ScavioIndeedCompanyReviews
# --------------------------------------------------------------------------


class ScavioIndeedCompanyReviewsInput(BaseModel):
    """Input schema for the ScavioIndeedCompanyReviews tool."""

    model_config = ConfigDict(extra="allow")

    company: str = Field(
        description=(
            "indeed.com/cmp/<slug> slug or a full profile URL. Slugs are untidy, e.g. "
            "'Tata-Consultancy-Services-(tcs)'."
        ),
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "Result page, 1-based. 20 reviews per page."
        ),
    )


class ScavioIndeedCompanyReviews(BaseTool):  # type: ignore[override]
    """Indeed: Indeed employee reviews with per-category ratings, pros/cons, reviewer
    job title and location, plus aggregated sentiment and breakdowns.

    Costs 2 credits per call.

    Pagination: page -- 20 reviews per page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioIndeedCompanyReviews

            tool = ScavioIndeedCompanyReviews()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"company": "Stripe", "page": 2})
    """

    name: str = "scavio_indeed_company_reviews"
    description: str = (
        "Indeed: Indeed employee reviews with per-category ratings, pros/cons, "
        "reviewer job title and location, plus aggregated sentiment and breakdowns. "
        "Pagination: page -- 20 reviews per page. Costs 2 credits per call."
    )
    args_schema: Type[BaseModel] = ScavioIndeedCompanyReviewsInput
    handle_tool_error: bool = True

    api_wrapper: ScavioIndeedCompanyReviewsAPIWrapper = Field(
        default_factory=ScavioIndeedCompanyReviewsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioIndeedCompanyReviewsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        company: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/indeed/company/reviews (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                company=company,
                page=page,
            )
            return self._process_response(raw, _first_identifier(company))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        company: str,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/indeed/company/reviews (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                company=company,
                page=page,
            )
            return self._process_response(raw, _first_identifier(company))
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
                f"No Indeed reviews found for '{identifier}'. Try broadening the query "
                "or removing filters."
            )
        return raw
