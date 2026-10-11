"""Scavio  tools for LangChain agents.

Every URL, parameter name, enum and credit cost below is copied from the
Scavio route definition rather than derived from the tool name.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_scavio._utilities import (
    WEB_JOB_MAX_WAIT,
    ScavioExtractAPIWrapper,
    ScavioExtractBatchAPIWrapper,
    ScavioExtractBatchStatusAPIWrapper,
    ScavioExtractCrawlAPIWrapper,
    ScavioExtractCrawlStatusAPIWrapper,
    ScavioExtractMapAPIWrapper,
    _ScavioExtractJobAPIWrapper,
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
# ScavioExtract
# --------------------------------------------------------------------------


class ScavioExtractInput(BaseModel):
    """Input schema for the ScavioExtract tool."""

    model_config = ConfigDict(extra="allow")

    url: str = Field(
        description=(
            "Page to read. http(s) only; a bare host is upgraded to https. Loopback, "
            "private, link-local and metadata hosts are rejected with a 400."
        ),
    )

    format: Optional[Literal["html", "markdown", "text"]] = Field(
        default=None,
        description=(
            "Output format: html is the raw page, markdown is a readability "
            "extraction, text is that markdown flattened to plain text. Options: html, "
            "markdown, text. Default: markdown."
        ),
    )

    mode: Optional[Literal["normal", "advanced", "ultra"]] = Field(
        default=None,
        description=(
            "Fetch tier and THE PRICE-BEARING PARAMETER: normal and advanced cost 1 "
            "credit, ultra costs 2. Escalate only when a plain fetch comes back empty. "
            "Options: normal, advanced, ultra. Default: normal."
        ),
    )


class ScavioExtract(BaseTool):  # type: ignore[override]
    """Read ANY URL and get it back as raw HTML, readability Markdown, or plain text.
    The read-a-page primitive: { url, format, mode, content, content_length }.

    Costs 1 credit in normal or advanced mode and 2 credits in ultra mode, and is
    billed only on a successful extraction.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioExtract

            tool = ScavioExtract()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke(
                {
                    "url": "https://example.com/article",
                    "format": "markdown",
                }
            )
    """

    name: str = "scavio_extract"
    description: str = (
        "Read ANY URL and get it back as raw HTML, readability Markdown, or plain "
        "text. The read-a-page primitive: { url, format, mode, content, content_length "
        "}. Costs 1 credit in normal or advanced mode and 2 credits in ultra mode, and "
        "is billed only on a successful extraction."
    )
    args_schema: Type[BaseModel] = ScavioExtractInput
    handle_tool_error: bool = True

    api_wrapper: ScavioExtractAPIWrapper = Field(
        default_factory=ScavioExtractAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioExtractAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        url: str,
        format: Optional[Literal["html", "markdown", "text"]] = None,
        mode: Optional[Literal["normal", "advanced", "ultra"]] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/extract (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                url=url,
                format=format,
                mode=mode,
            )
            return self._process_response(raw, _first_identifier(url))
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        url: str,
        format: Optional[Literal["html", "markdown", "text"]] = None,
        mode: Optional[Literal["normal", "advanced", "ultra"]] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/extract (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                url=url,
                format=format,
                mode=mode,
            )
            return self._process_response(raw, _first_identifier(url))
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
                f"No content could be extracted from '{identifier}'. The page may be "
                "empty, blocked or unreachable -- retry with mode='advanced' to "
                "render it, or mode='ultra' for a hard-blocked site. A failed "
                "extraction is not billed."
            )
        return raw


# --------------------------------------------------------------------------
# Map, crawl and batch: the multi-page members of the Extract family.
#
# Map answers in one call. Crawl and batch run as background jobs: the start
# tools post the job and then wait up to ``wait_seconds`` (default 45) for it
# to finish, returning the job's status and its first page of results either
# way. A job still running when the wait ends keeps going server-side; read
# the rest with the matching *Status tool and the returned ``data.id``.
# --------------------------------------------------------------------------

_MODE_DESCRIPTION = (
    "Fetch tier and THE PRICE-BEARING PARAMETER: normal is the standard fetch, "
    "advanced renders JavaScript-built pages, ultra reaches the hardest pages. "
    "Options: normal, advanced, ultra. Default: normal."
)

_PATHS_DESCRIPTION = (
    "Path globs (max 20). * matches within one path segment, ** across "
    "segments; a pattern without wildcards matches that path and everything "
    'under it. Example: ["/blog/**"].'
)

_JOB_PRICE = (
    "Billed per SUCCESSFUL page at Extract's price for the mode: normal and "
    "advanced 1 credit, ultra 2 credits. Failed and skipped pages are free, and "
    "if the balance runs out the job stops cleanly."
)

_STATUS_PRICE = "Free: reading a job's status and results costs 0 credits."


def _limit_field() -> Any:
    return Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Results per response, 1-100. Default: 10, to keep page content "
            "inside the context window; page on with next_cursor."
        ),
    )


def _include_content_field() -> Any:
    return Field(
        default=None,
        description=(
            "Set false to list results without their page content (smaller "
            "responses). Default: true."
        ),
    )


def _format_field() -> Any:
    return Field(
        default=None,
        description=(
            "Format for each page, as on Extract: html, markdown or text. "
            "Default: markdown."
        ),
    )


def _mode_field() -> Any:
    return Field(default=None, description=_MODE_DESCRIPTION)


def _wait_field() -> Any:
    return Field(
        default=45,
        ge=0,
        le=WEB_JOB_MAX_WAIT,
        description=(
            "How long to wait for the job to finish before returning, 0-300 "
            "seconds. 0 returns the job id at once. A job still running when "
            "the wait ends keeps going; read it later with its status tool."
        ),
    )


class _JobStatusInput(BaseModel):
    id: str = Field(description="Job id returned when the job was started (data.id).")

    cursor: Optional[str] = Field(
        default=None,
        description="next_cursor from the previous response, to read the next page.",
    )

    limit: Optional[int] = _limit_field()

    status: Optional[Literal["succeeded", "failed", "skipped"]] = Field(
        default=None, description="Only results with this status."
    )

    include_content: Optional[bool] = _include_content_field()


class _ScavioExtractToolBase(BaseTool):  # type: ignore[override]
    handle_tool_error: bool = True

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            wrapper_cls = type(self).model_fields["api_wrapper"].annotation
            kwargs["api_wrapper"] = wrapper_cls(**api_kwargs)  # type: ignore[misc]
        super().__init__(**kwargs)


# --------------------------------------------------------------------------
# ScavioExtractMap
# --------------------------------------------------------------------------


class ScavioExtractMapInput(BaseModel):
    """Input schema for the ScavioExtractMap tool."""

    model_config = ConfigDict(extra="allow")

    url: str = Field(description="The site to map. A bare host is upgraded to https.")

    search: Optional[str] = Field(
        default=None,
        description=(
            "Only return URLs containing every word of this text (case-insensitive)."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000,
        description=(
            "Maximum links to return, 1-5000. Default 1000 (or the plan's limit "
            "if lower: Free 500, PAYG 1,000, paid plans 5,000)."
        ),
    )

    sitemap: Optional[Literal["include", "only", "skip"]] = Field(
        default=None,
        description=(
            "include combines the site's sitemaps with the links on the start "
            "page; only uses sitemaps alone; skip uses page links alone. "
            "Default: include."
        ),
    )

    include_subdomains: Optional[bool] = Field(
        default=None,
        description="Also return URLs on subdomains of the site. Default: false.",
    )

    include_paths: Optional[List[str]] = Field(
        default=None, description="Keep only URLs matching these. " + _PATHS_DESCRIPTION
    )

    exclude_paths: Optional[List[str]] = Field(
        default=None, description="Drop URLs matching these. " + _PATHS_DESCRIPTION
    )

    ignore_query_parameters: Optional[bool] = Field(
        default=None,
        description=(
            "Treat URLs that differ only by query string as one URL. Default: true."
        ),
    )

    mode: Optional[Literal["normal", "advanced", "ultra"]] = _mode_field()


class ScavioExtractMap(_ScavioExtractToolBase):  # type: ignore[override]
    """List a website's page URLs from its sitemaps and start page, deduplicated
    and filtered by path. Use it to find the right pages before extracting or
    crawling them.

    Costs 1 credit in normal mode; advanced costs up to 2 and ultra up to 10,
    priced by what the call needed.

    Invoke directly:
        .. code-block:: python

            from langchain_scavio import ScavioExtractMap

            result = ScavioExtractMap().invoke(
                {"url": "https://docs.example.com", "search": "pricing"}
            )
            [link["url"] for link in result["data"]["links"]]
    """

    name: str = "scavio_map_website"
    description: str = (
        "List a website's page URLs, gathered from its sitemaps and its start "
        "page, deduplicated, filtered by path or search words and limited: "
        "{ url, links: [{ url, source, lastmod }], total_found, returned, "
        "truncated }. Pages the site's robots.txt disallows are left out. Use it "
        "to find the right pages before extracting or crawling. Costs 1 credit in "
        "normal mode; advanced costs up to 2 credits and ultra up to 10, priced "
        "by what the call needed."
    )
    args_schema: Type[BaseModel] = ScavioExtractMapInput

    api_wrapper: ScavioExtractMapAPIWrapper = Field(
        default_factory=ScavioExtractMapAPIWrapper  # type: ignore[arg-type]
    )

    def _run(
        self,
        url: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/extract/map (synchronously)."""
        try:
            return self.api_wrapper.raw_results(url=url, **kwargs)
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        url: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call POST /api/v1/extract/map (asynchronously)."""
        try:
            return await self.api_wrapper.raw_results_async(url=url, **kwargs)
        except Exception as e:
            return {"error": str(e)}


# --------------------------------------------------------------------------
# Job tools: start-and-wait plus status
# --------------------------------------------------------------------------

_RESULT_KEYS = ("limit", "include_content")


def _split_result_query(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: kwargs.pop(k) for k in _RESULT_KEYS if k in kwargs}


class _ScavioJobStartTool(_ScavioExtractToolBase):  # type: ignore[override]
    """Start a job, then wait for it (bounded) and return its status page."""

    api_wrapper: _ScavioExtractJobAPIWrapper

    def _start_and_wait(
        self, body: dict[str, Any], wait_seconds: int
    ) -> dict[str, Any]:
        query = _split_result_query(body)
        created = self.api_wrapper.start(**body)
        job_id = (created.get("data") or {}).get("id")
        if not job_id or wait_seconds <= 0:
            return created
        return self.api_wrapper.wait(job_id, wait_seconds, **query)

    async def _start_and_wait_async(
        self, body: dict[str, Any], wait_seconds: int
    ) -> dict[str, Any]:
        query = _split_result_query(body)
        created = await self.api_wrapper.start_async(**body)
        job_id = (created.get("data") or {}).get("id")
        if not job_id or wait_seconds <= 0:
            return created
        return await self.api_wrapper.wait_async(job_id, wait_seconds, **query)


class _ScavioJobStatusTool(_ScavioExtractToolBase):  # type: ignore[override]
    args_schema: Type[BaseModel] = _JobStatusInput
    api_wrapper: _ScavioExtractJobAPIWrapper

    def _run(
        self,
        id: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            return self.api_wrapper.status(id, **kwargs)
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        id: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            return await self.api_wrapper.status_async(id, **kwargs)
        except Exception as e:
            return {"error": str(e)}


# --------------------------------------------------------------------------
# ScavioExtractCrawl / ScavioExtractCrawlStatus
# --------------------------------------------------------------------------


class ScavioExtractCrawlInput(BaseModel):
    """Input schema for the ScavioExtractCrawl tool."""

    url: str = Field(
        description="Where the crawl starts. Only pages on the same site are crawled."
    )

    max_pages: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description=(
            "Maximum pages to fetch. Default 100. Plan limits: Free 10, PAYG 100, "
            "Project 500, Bootstrap 2,000, Startup 5,000, Growth 10,000."
        ),
    )

    max_depth: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description=(
            "How many links away from the start page to follow, 0-10. 0 fetches "
            "only the start page. Default: 3."
        ),
    )

    include_paths: Optional[List[str]] = Field(
        default=None,
        description="Crawl only paths matching these. " + _PATHS_DESCRIPTION,
    )

    exclude_paths: Optional[List[str]] = Field(
        default=None, description="Skip paths matching these. " + _PATHS_DESCRIPTION
    )

    allow_subdomains: Optional[bool] = Field(
        default=None, description="Also crawl subdomains of the site. Default: false."
    )

    ignore_query_parameters: Optional[bool] = Field(
        default=None,
        description=(
            "Treat URLs that differ only by query string as one page. Default: true."
        ),
    )

    sitemap: Optional[Literal["include", "skip", "only"]] = Field(
        default=None,
        description=(
            "include also starts from the site's sitemap; skip follows links only; "
            "only fetches sitemap pages without following links. Default: include."
        ),
    )

    format: Optional[Literal["html", "markdown", "text"]] = _format_field()

    mode: Optional[Literal["normal", "advanced", "ultra"]] = _mode_field()

    wait_seconds: int = _wait_field()

    limit: Optional[int] = _limit_field()

    include_content: Optional[bool] = _include_content_field()


class ScavioExtractCrawl(_ScavioJobStartTool):  # type: ignore[override]
    """Crawl a website from one URL and get every page back as Markdown, text or
    HTML. Starts the crawl, waits up to ``wait_seconds`` for it to finish and
    returns its status with the first page of results.

    Billed per successful page at Extract's price for the mode: normal and
    advanced 1 credit, ultra 2 credits. Failed and skipped pages are free.

    Invoke directly:
        .. code-block:: python

            from langchain_scavio import ScavioExtractCrawl, ScavioExtractCrawlStatus

            job = ScavioExtractCrawl().invoke(
                {"url": "https://docs.example.com", "max_pages": 10}
            )
            job["data"]["status"], job["data"]["results"]
            # Still running, or more pages than one response holds:
            ScavioExtractCrawlStatus().invoke(
                {"id": job["data"]["id"], "cursor": job["data"]["next_cursor"]}
            )
    """

    name: str = "scavio_crawl_website"
    description: str = (
        "Crawl a website from one start URL: same-site pages are discovered and "
        "extracted breadth-first within the page, depth and path limits, each "
        "returned as Markdown, text or HTML. Pages the site's robots.txt "
        "disallows are skipped. Waits up to wait_seconds (default 45) and returns "
        "{ id, status, pages, credits_used, results, next_cursor }; if status is "
        "still queued or running, call scavio_crawl_status with the id later. "
        + _JOB_PRICE
    )
    args_schema: Type[BaseModel] = ScavioExtractCrawlInput

    api_wrapper: ScavioExtractCrawlAPIWrapper = Field(
        default_factory=ScavioExtractCrawlAPIWrapper  # type: ignore[arg-type]
    )

    def _run(
        self,
        url: str,
        wait_seconds: int = 45,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start a crawl (POST /api/v1/extract/crawl) and wait for it."""
        try:
            return self._start_and_wait({"url": url, **kwargs}, wait_seconds)
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        url: str,
        wait_seconds: int = 45,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start a crawl and wait for it (asynchronously)."""
        try:
            return await self._start_and_wait_async(
                {"url": url, **kwargs}, wait_seconds
            )
        except Exception as e:
            return {"error": str(e)}


class ScavioExtractCrawlStatus(_ScavioJobStatusTool):  # type: ignore[override]
    """Read a crawl's progress and its page results, paged by cursor.

    Free. Results are kept for 3 days after the crawl finishes.
    """

    name: str = "scavio_crawl_status"
    description: str = (
        "Read a crawl's progress and its page results by job id, paged by "
        "cursor: { status, stopped_reason, pages, credits_used, results: [{ url, "
        "status, content }], next_cursor }. Results are kept for 3 days after the "
        "crawl finishes. " + _STATUS_PRICE
    )

    api_wrapper: ScavioExtractCrawlStatusAPIWrapper = Field(
        default_factory=ScavioExtractCrawlStatusAPIWrapper  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# ScavioExtractBatch / ScavioExtractBatchStatus
# --------------------------------------------------------------------------


class ScavioExtractBatchInput(BaseModel):
    """Input schema for the ScavioExtractBatch tool."""

    urls: List[str] = Field(
        min_length=1,
        max_length=1000,
        description=(
            "The pages to extract (duplicates removed). Plan limits: Free 10, "
            "PAYG 50, Project 100, Bootstrap 500, Startup and Growth 1,000."
        ),
    )

    format: Optional[Literal["html", "markdown", "text"]] = _format_field()

    mode: Optional[Literal["normal", "advanced", "ultra"]] = _mode_field()

    wait_seconds: int = _wait_field()

    limit: Optional[int] = _limit_field()

    include_content: Optional[bool] = _include_content_field()


class ScavioExtractBatch(_ScavioJobStartTool):  # type: ignore[override]
    """Extract a list of URLs in one job, each returned as Markdown, text or
    HTML exactly like Extract. Waits up to ``wait_seconds`` for it to finish.

    Billed per successful page at Extract's price for the mode: normal and
    advanced 1 credit, ultra 2 credits. Failed pages are free.
    """

    name: str = "scavio_batch_extract"
    description: str = (
        "Extract a list of URLs (up to 1,000, plan-limited) in one job, each "
        "returned as Markdown, text or HTML exactly like scavio_extract. Waits "
        "up to wait_seconds (default 45) and returns { id, status, pages, "
        "credits_used, results, next_cursor }; if status is still queued or "
        "running, call scavio_batch_status with the id later. " + _JOB_PRICE
    )
    args_schema: Type[BaseModel] = ScavioExtractBatchInput

    api_wrapper: ScavioExtractBatchAPIWrapper = Field(
        default_factory=ScavioExtractBatchAPIWrapper  # type: ignore[arg-type]
    )

    def _run(
        self,
        urls: List[str],
        wait_seconds: int = 45,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start a batch (POST /api/v1/extract/batch) and wait for it."""
        try:
            return self._start_and_wait({"urls": urls, **kwargs}, wait_seconds)
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        urls: List[str],
        wait_seconds: int = 45,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Start a batch and wait for it (asynchronously)."""
        try:
            return await self._start_and_wait_async(
                {"urls": urls, **kwargs}, wait_seconds
            )
        except Exception as e:
            return {"error": str(e)}


class ScavioExtractBatchStatus(_ScavioJobStatusTool):  # type: ignore[override]
    """Read a batch's progress and its page results, paged by cursor. Free."""

    name: str = "scavio_batch_status"
    description: str = (
        "Read a batch extract's progress and its page results by job id, paged "
        "by cursor: { status, pages, credits_used, results: [{ url, status, "
        "content }], next_cursor }. Results are kept for 3 days after the batch "
        "finishes. " + _STATUS_PRICE
    )

    api_wrapper: ScavioExtractBatchStatusAPIWrapper = Field(
        default_factory=ScavioExtractBatchStatusAPIWrapper  # type: ignore[arg-type]
    )
