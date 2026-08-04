"""Scavio LinkedIn tools for LangChain agents."""

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
    ScavioLinkedInCompanyAPIWrapper,
    ScavioLinkedInCompanyPostsAPIWrapper,
    ScavioLinkedInJobAPIWrapper,
    ScavioLinkedInPersonAboutAPIWrapper,
    ScavioLinkedInPersonAPIWrapper,
    ScavioLinkedInPersonPostsAPIWrapper,
    ScavioLinkedInPostAPIWrapper,
    ScavioLinkedInPostCommentsAPIWrapper,
    ScavioLinkedInSearchJobsAPIWrapper,
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


# ---------------------------------------------------------------------------
# ScavioLinkedInPerson
# ---------------------------------------------------------------------------


class ScavioLinkedInPersonInput(BaseModel):
    """Input schema for ScavioLinkedInPerson tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description=(
            "Public identifier / vanity handle from the profile URL (e.g. "
            "'williamhgates'). Provide username or url."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description="Full LinkedIn profile URL, as an alternative to username.",
    )


class ScavioLinkedInPerson(BaseTool):  # type: ignore[override]
    """Fetch a full LinkedIn member profile using the Scavio API.

    Returns a flat profile under ``data``: public_identifier, full_name,
    headline, about, location, avatar, follower_count, connection_count,
    current_company, experiences, educations, honors_and_awards, bio_links,
    people_also_viewed and similar_profiles.

    Provide either ``username`` (the vanity handle) or ``url``. Contact
    details are not available -- that endpoint was retired upstream.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInPerson

            tool = ScavioLinkedInPerson()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "williamhgates"})
    """

    name: str = "scavio_linkedin_person"
    description: str = (
        "Fetch a full LinkedIn member profile by vanity handle or profile URL. Returns "
        "name, headline, about, location, current company, experience, education, "
        "honours and links under data. Costs 1 credit per call. Provide username or "
        "url."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInPersonInput
    handle_tool_error: bool = True

    api_wrapper: ScavioLinkedInPersonAPIWrapper = Field(
        default_factory=ScavioLinkedInPersonAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInPersonAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a full LinkedIn member profile (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                username=username,
                url=url,
            )
            return self._process_response(raw, username or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a full LinkedIn member profile (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username,
                url=url,
            )
            return self._process_response(raw, username or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not (data.get("public_identifier") or data.get("full_name")):
            raise ToolException(
                f"No LinkedIn profile found for '{subject}'. Verify the vanity "
                "handle or URL, and note that private profiles are not retrievable."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioLinkedInPersonAbout
# ---------------------------------------------------------------------------


class ScavioLinkedInPersonAboutInput(BaseModel):
    """Input schema for ScavioLinkedInPersonAbout tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description=(
            "Public identifier / vanity handle from the profile URL (e.g. "
            "'williamhgates'). Provide username or url."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description="Full LinkedIn profile URL, as an alternative to username.",
    )


class ScavioLinkedInPersonAbout(BaseTool):  # type: ignore[override]
    """Fetch the about/overview section of a LinkedIn member.

    Returns ``data`` with about, headline, education_summary, experiences,
    educations, honors_and_awards and bio_links -- the narrative parts of a
    profile without the social graph blocks.

    Provide either ``username`` or ``url``. Use ScavioLinkedInPerson when
    follower counts and similar-profile suggestions are also needed.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInPersonAbout

            tool = ScavioLinkedInPersonAbout()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "williamhgates"})
    """

    name: str = "scavio_linkedin_person_about"
    description: str = (
        "Fetch the about/overview metadata of a LinkedIn member: summary, headline, "
        "experience, education, honours and links. A trimmed-down version of the full "
        "person profile. Costs 1 credit per call. Provide username or url."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInPersonAboutInput
    handle_tool_error: bool = True

    api_wrapper: ScavioLinkedInPersonAboutAPIWrapper = Field(
        default_factory=ScavioLinkedInPersonAboutAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInPersonAboutAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the about/overview section of a LinkedIn member (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                username=username,
                url=url,
            )
            return self._process_response(raw, username or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the about/overview section of a LinkedIn member (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username,
                url=url,
            )
            return self._process_response(raw, username or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not (data.get("about") or data.get("headline") or data.get("experiences")):
            raise ToolException(
                f"No LinkedIn about section found for '{subject}'. Verify the vanity "
                "handle or URL."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioLinkedInPersonPosts
# ---------------------------------------------------------------------------


class ScavioLinkedInPersonPostsInput(BaseModel):
    """Input schema for ScavioLinkedInPersonPosts tool."""

    model_config = ConfigDict(extra="allow")

    username: Optional[str] = Field(
        default=None,
        description=(
            "Public identifier / vanity handle from the profile URL (e.g. "
            "'williamhgates'). Provide username or url."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description="Full LinkedIn profile URL, as an alternative to username.",
    )

    type: Optional[Literal["posts", "comments", "reactions"]] = Field(
        default=None,
        description=(
            "Which feed to return: 'posts' (default) for the member's own posts, "
            "'comments' for posts they commented on, 'reactions' for posts they "
            "reacted to."
        ),
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioLinkedInPersonPosts(BaseTool):  # type: ignore[override]
    """Fetch a LinkedIn member's post feed using the Scavio API.

    Returns 50 posts per page under ``data.data`` with text, url,
    created_at, reaction breakdown, comment/repost counts, images, article
    and author. Paginate with ``data.next_cursor`` and stop when
    ``data.has_more`` is false.

    Costs 10 credits per page -- more than most Scavio endpoints, so avoid
    paginating further than needed.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInPersonPosts

            tool = ScavioLinkedInPersonPosts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"username": "williamhgates", "type": "posts"})
    """

    name: str = "scavio_linkedin_person_posts"
    description: str = (
        "Fetch a LinkedIn member's posts, or the posts they commented on or reacted to "
        "(type=posts|comments|reactions). Returns 50 per page under data.data with "
        "text, reactions, comment counts and author. Supports cursor pagination. Costs "
        "10 credits per page. Provide username or url."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInPersonPostsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioLinkedInPersonPostsAPIWrapper = Field(
        default_factory=ScavioLinkedInPersonPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInPersonPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        username: Optional[str] = None,
        url: Optional[str] = None,
        type: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a LinkedIn member's post feed (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                username=username,
                url=url,
                type=type,
                cursor=cursor,
            )
            return self._process_response(raw, username or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        username: Optional[str] = None,
        url: Optional[str] = None,
        type: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a LinkedIn member's post feed (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                username=username,
                url=url,
                type=type,
                cursor=cursor,
            )
            return self._process_response(raw, username or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        data = data.get("data")
        if self.max_results and data:
            raw["data"]["data"] = data[: self.max_results]
        if not data:
            raise ToolException(
                f"No LinkedIn posts found for '{subject}'. The member may not post "
                "publicly, or the requested feed type may be empty."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioLinkedInCompany
# ---------------------------------------------------------------------------


class ScavioLinkedInCompanyInput(BaseModel):
    """Input schema for ScavioLinkedInCompany tool."""

    model_config = ConfigDict(extra="allow")

    company: Optional[str] = Field(
        default=None,
        description=(
            "Company universal name / slug from the company URL (e.g. 'microsoft'). "
            "Provide company or url."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description="Full LinkedIn company URL, as an alternative to company.",
    )


class ScavioLinkedInCompany(BaseTool):  # type: ignore[override]
    """Fetch a LinkedIn company profile using the Scavio API.

    Returns a flat profile under ``data``: name, description, about,
    website, industries, specialties, company_size, employee_count,
    follower_count, headquarters, locations, logo, similar_companies,
    affiliated_companies and recent_updates.

    ``featured_employees`` carries a 4-6 person sample of staff -- it is the
    documented substitute for the retired employee-directory endpoint.
    Provide either ``company`` (the slug) or ``url``.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInCompany

            tool = ScavioLinkedInCompany()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"company": "microsoft"})
    """

    name: str = "scavio_linkedin_company"
    description: str = (
        "Fetch a LinkedIn company profile by slug or company URL. Returns description, "
        "website, industries, specialties, size, employee and follower counts, "
        "locations, similar/affiliated companies and a small featured_employees "
        "sample. Costs 1 credit per call. Provide company or url."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInCompanyInput
    handle_tool_error: bool = True

    api_wrapper: ScavioLinkedInCompanyAPIWrapper = Field(
        default_factory=ScavioLinkedInCompanyAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInCompanyAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        company: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a LinkedIn company profile (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                company=company,
                url=url,
            )
            return self._process_response(raw, company or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        company: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a LinkedIn company profile (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                company=company,
                url=url,
            )
            return self._process_response(raw, company or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not (data.get("universal_name") or data.get("name")):
            raise ToolException(
                f"No LinkedIn company found for '{subject}'. Verify the company slug "
                "(the segment after /company/ in the URL)."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioLinkedInCompanyPosts
# ---------------------------------------------------------------------------


class ScavioLinkedInCompanyPostsInput(BaseModel):
    """Input schema for ScavioLinkedInCompanyPosts tool."""

    model_config = ConfigDict(extra="allow")

    company: Optional[str] = Field(
        default=None,
        description=(
            "Company universal name / slug from the company URL (e.g. 'microsoft'). "
            "Provide company or url."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description="Full LinkedIn company URL, as an alternative to company.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioLinkedInCompanyPosts(BaseTool):  # type: ignore[override]
    """Fetch a LinkedIn company's recent posts using the Scavio API.

    Returns 50 posts per page under ``data.data``, using the same feed item
    shape as the member post feed. Paginate with ``data.next_cursor`` and
    stop when ``data.has_more`` is false.

    Costs 10 credits per page. There is no ``type`` argument here -- that is
    member-only.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInCompanyPosts

            tool = ScavioLinkedInCompanyPosts()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"company": "microsoft"})
    """

    name: str = "scavio_linkedin_company_posts"
    description: str = (
        "Fetch a LinkedIn company's recent posts by slug or company URL. Returns 50 "
        "per page under data.data with text, reactions, comment counts, images and "
        "author. Supports cursor pagination. Costs 10 credits per page. Provide "
        "company or url."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInCompanyPostsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioLinkedInCompanyPostsAPIWrapper = Field(
        default_factory=ScavioLinkedInCompanyPostsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInCompanyPostsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        company: Optional[str] = None,
        url: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a LinkedIn company's recent posts (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                company=company,
                url=url,
                cursor=cursor,
            )
            return self._process_response(raw, company or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        company: Optional[str] = None,
        url: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a LinkedIn company's recent posts (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                company=company,
                url=url,
                cursor=cursor,
            )
            return self._process_response(raw, company or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        data = data.get("data")
        if self.max_results and data:
            raw["data"]["data"] = data[: self.max_results]
        if not data:
            raise ToolException(
                f"No LinkedIn company posts found for '{subject}'. The page may not "
                "post publicly."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioLinkedInSearchJobs
# ---------------------------------------------------------------------------


class ScavioLinkedInSearchJobsInput(BaseModel):
    """Input schema for ScavioLinkedInSearchJobs tool."""

    model_config = ConfigDict(extra="allow")

    search: str = Field(
        description=(
            "Job search keyword (e.g. 'software engineer'). The field is named search, "
            "mirroring the API wire field."
        ),
    )

    location: Optional[str] = Field(
        default=None,
        description="Geographic filter (e.g. 'London'). Omit to search everywhere.",
    )

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque cursor from a previous response's next_cursor. Keep the other "
            "arguments the same across paginated calls."
        ),
    )


class ScavioLinkedInSearchJobs(BaseTool):  # type: ignore[override]
    """Search LinkedIn job listings using the Scavio API.

    Returns 25 job briefs per page under ``data.data``: id, title, url,
    company, company_url, company_logo, location, posted_at, workplace_type
    and salary. Paginate with ``data.next_cursor``.

    The upstream rotates its result set, so repeat calls return different
    listings and pages overlap slightly -- dedupe by job id. Costs 10
    credits per page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInSearchJobs

            tool = ScavioLinkedInSearchJobs()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"search": "software engineer", "location": "London"})
    """

    name: str = "scavio_linkedin_search_jobs"
    description: str = (
        "Search LinkedIn job listings by keyword and optional location. Returns 25 job "
        "briefs per page under data.data with title, company, location, posted_at, "
        "workplace_type and salary. Results rotate between calls, so dedupe by job id. "
        "Supports cursor pagination. Costs 10 credits per page."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInSearchJobsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioLinkedInSearchJobsAPIWrapper = Field(
        default_factory=ScavioLinkedInSearchJobsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInSearchJobsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        search: str,
        location: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search LinkedIn job listings (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                search=search,
                location=location,
                cursor=cursor,
            )
            return self._process_response(raw, search)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        search: str,
        location: Optional[str] = None,
        cursor: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search LinkedIn job listings (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                search=search,
                location=location,
                cursor=cursor,
            )
            return self._process_response(raw, search)
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], search: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        data = data.get("data")
        if self.max_results and data:
            raw["data"]["data"] = data[: self.max_results]
        if not data:
            raise ToolException(
                f"No LinkedIn jobs found for '{search}'. Try broadening the keyword "
                "or removing the location filter."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioLinkedInJob
# ---------------------------------------------------------------------------


class ScavioLinkedInJobInput(BaseModel):
    """Input schema for ScavioLinkedInJob tool."""

    model_config = ConfigDict(extra="allow")

    job_id: Optional[str] = Field(
        default=None,
        description=(
            "Job id from a search result (e.g. '4415427228'). Provide job_id or url."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description="Full LinkedIn job URL, as an alternative to job_id.",
    )


class ScavioLinkedInJob(BaseTool):  # type: ignore[override]
    """Fetch full details for a single LinkedIn job listing.

    Returns a flat listing under ``data``: title, description, location,
    employment_type, experience_level, job_functions, industries, benefits,
    skills, is_remote, is_closed, posted_at, applicant_count, salary and a
    nested hiring ``company`` object.

    This is the most expensive endpoint in the API at 30 credits per call,
    so fetch detail only for shortlisted ids. Roughly one job id in five
    returned by job search has no detail record and answers with an unbilled
    404.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInJob

            tool = ScavioLinkedInJob()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"job_id": "4415427228"})
    """

    name: str = "scavio_linkedin_job"
    description: str = (
        "Fetch full details for a single LinkedIn job listing by job id or job URL. "
        "Returns description, location, employment type, seniority, skills, benefits, "
        "applicant count, salary and the hiring company. Costs 30 credits per call -- "
        "the most expensive Scavio endpoint, so only call it for shortlisted jobs. "
        "Provide job_id or url."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInJobInput
    handle_tool_error: bool = True

    api_wrapper: ScavioLinkedInJobAPIWrapper = Field(
        default_factory=ScavioLinkedInJobAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInJobAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        job_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch full details for a single LinkedIn job listing (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                job_id=job_id,
                url=url,
            )
            return self._process_response(raw, job_id or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        job_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch full details for a single LinkedIn job listing (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                job_id=job_id,
                url=url,
            )
            return self._process_response(raw, job_id or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not (data.get("id") or data.get("title")):
            raise ToolException(
                f"No LinkedIn job detail found for '{subject}'. Expired or delisted "
                "postings often stay in search results after their detail page is "
                "gone; that lookup is not billed."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioLinkedInPost
# ---------------------------------------------------------------------------


class ScavioLinkedInPostInput(BaseModel):
    """Input schema for ScavioLinkedInPost tool."""

    model_config = ConfigDict(extra="allow")

    post_id: Optional[str] = Field(
        default=None,
        description=(
            "Post id or activity urn (a bare id, urn:li:activity:<id>, "
            "urn:li:ugcPost:<id> or urn:li:share:<id>). Provide post_id or url."
        ),
    )

    url: Optional[str] = Field(
        default=None,
        description="Full LinkedIn post URL, as an alternative to post_id.",
    )


class ScavioLinkedInPost(BaseTool):  # type: ignore[override]
    """Fetch a single LinkedIn post using the Scavio API.

    Returns a flat post under ``data``: post_type, title, headline, text,
    url, created_at, hashtags, embedded_links, images, videos, num_likes,
    num_comments, tagged_companies, tagged_people, author and
    ``top_comments`` (the visible comments).

    For the full comment thread with replies use ScavioLinkedInPostComments.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInPost

            tool = ScavioLinkedInPost()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"post_id": "7488618410256523265"})
    """

    name: str = "scavio_linkedin_post"
    description: str = (
        "Fetch a single LinkedIn post by post id, activity urn or post URL. Returns "
        "text, media, hashtags, like/comment counts, tagged entities, author and the "
        "top visible comments under data. Costs 1 credit per call. Provide post_id or "
        "url."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInPostInput
    handle_tool_error: bool = True

    api_wrapper: ScavioLinkedInPostAPIWrapper = Field(
        default_factory=ScavioLinkedInPostAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInPostAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        post_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a single LinkedIn post (synchronously)."""
        try:
            raw = self.api_wrapper.raw_results(
                post_id=post_id,
                url=url,
            )
            return self._process_response(raw, post_id or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        post_id: Optional[str] = None,
        url: Optional[str] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch a single LinkedIn post (asynchronously)."""
        try:
            raw = await self.api_wrapper.raw_results_async(
                post_id=post_id,
                url=url,
            )
            return self._process_response(raw, post_id or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Raise ToolException when the response carries no payload."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if not (data.get("id") or data.get("text")):
            raise ToolException(
                f"No LinkedIn post found for '{subject}'. Verify the post id, "
                "activity urn or URL, and that the post is public."
            )
        return raw


# ---------------------------------------------------------------------------
# ScavioLinkedInPostComments
# ---------------------------------------------------------------------------


class ScavioLinkedInPostCommentsInput(BaseModel):
    """Input schema for ScavioLinkedInPostComments tool."""

    model_config = ConfigDict(extra="allow")

    post_id: Optional[str] = Field(
        default=None,
        description="Post id or activity urn. Provide post_id or url.",
    )

    url: Optional[str] = Field(
        default=None,
        description="Full LinkedIn post URL, as an alternative to post_id.",
    )

    page: Optional[int] = Field(
        default=None,
        description=(
            "1-based page number (default 1). This endpoint pages by number, not by "
            "cursor. Page size varies, so keep paging until a page comes back empty."
        ),
    )


class ScavioLinkedInPostComments(BaseTool):  # type: ignore[override]
    """Fetch the comments on a LinkedIn post using the Scavio API.

    Returns comments under ``data.data``, each with text, url, created_at,
    is_pinned, author and nested ``replies``. The response also carries
    ``data.page``, ``data.total`` (page 1 only), ``data.has_more`` and
    ``data.next_page``.

    This is the only LinkedIn endpoint that paginates by a 1-based ``page``
    instead of a cursor. Costs 10 credits per page.

    Setup:
        .. code-block:: bash

            pip install langchain-scavio
            export SCAVIO_API_KEY="sk_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_scavio import ScavioLinkedInPostComments

            tool = ScavioLinkedInPostComments()

    Invoke directly:
        .. code-block:: python

            result = tool.invoke({"post_id": "7488618410256523265", "page": 1})
    """

    name: str = "scavio_linkedin_post_comments"
    description: str = (
        "Fetch the comments on a LinkedIn post, with their replies, by post id or post "
        "URL. Returns comments under data.data. Paginate with a 1-based page number "
        "(not a cursor) and keep going until a page comes back empty. Costs 10 credits "
        "per page. Provide post_id or url."
    )
    args_schema: Type[BaseModel] = ScavioLinkedInPostCommentsInput
    handle_tool_error: bool = True

    max_results: Optional[int] = 10

    api_wrapper: ScavioLinkedInPostCommentsAPIWrapper = Field(
        default_factory=ScavioLinkedInPostCommentsAPIWrapper  # type: ignore[arg-type]
    )

    def __init__(self, **kwargs: Any) -> None:
        api_kwargs = _forward_api_kwargs(kwargs)
        if api_kwargs and "api_wrapper" not in kwargs:
            kwargs["api_wrapper"] = ScavioLinkedInPostCommentsAPIWrapper(**api_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        post_id: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the comments on a LinkedIn post (synchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = self.api_wrapper.raw_results(
                post_id=post_id,
                url=url,
                page=page,
            )
            return self._process_response(raw, post_id or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        post_id: Optional[str] = None,
        url: Optional[str] = None,
        page: Optional[int] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fetch the comments on a LinkedIn post (asynchronously)."""
        forbidden = _LIST_INIT_ONLY_PARAMS & set(kwargs)
        if forbidden:
            raise ValueError(
                f"Parameters {forbidden} can only be set at instantiation, "
                "not during invocation."
            )
        try:
            raw = await self.api_wrapper.raw_results_async(
                post_id=post_id,
                url=url,
                page=page,
            )
            return self._process_response(raw, post_id or url or "")
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _process_response(
        self, raw: dict[str, Any], subject: str
    ) -> dict[str, Any]:
        """Truncate the result list and raise ToolException if empty."""
        data = raw.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        data = data.get("data")
        if self.max_results and data:
            raw["data"]["data"] = data[: self.max_results]
        if not data:
            raise ToolException(
                f"No LinkedIn comments found for '{subject}'. The post may have no "
                "comments, or the page number may be past the end of the thread."
            )
        return raw
