"""Scavio  tools for LangChain agents.

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
    ScavioExtractAPIWrapper,
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
