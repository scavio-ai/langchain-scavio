"""Scavio API wrappers for raw HTTP calls."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

logger = logging.getLogger(__name__)

SCAVIO_API_URL = "https://api.scavio.dev"


class ScavioBaseAPIWrapper(BaseModel):
    """Base wrapper for Scavio API endpoints.

    Provides shared auth, headers, and HTTP plumbing.
    Subclasses override ``_build_url()`` to target a specific endpoint.
    """

    scavio_api_key: SecretStr
    api_base_url: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        scavio_api_key = get_from_dict_or_env(
            values, "scavio_api_key", "SCAVIO_API_KEY"
        )
        values["scavio_api_key"] = scavio_api_key
        return values

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.scavio_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-scavio",
        }

    def _build_url(self) -> str:
        raise NotImplementedError("Subclasses must override _build_url()")

    def raw_results(self, **params: Any) -> dict[str, Any]:
        """Execute a synchronous POST request.

        Args:
            **params: Request body parameters.
                None values are automatically filtered out.

        Returns:
            Parsed JSON response from the API.

        Raises:
            ValueError: If the API returns a non-200 status code.
        """
        params = {k: v for k, v in params.items() if v is not None}
        response = requests.post(
            self._build_url(),
            json=params,
            headers=self._build_headers(),
            timeout=30,
        )
        if response.status_code != 200:
            error = response.json().get("error", "Unknown error")
            if isinstance(error, dict):
                error = error.get("message", "Unknown error")
            raise ValueError(f"Error {response.status_code}: {error}")
        return response.json()

    async def raw_results_async(self, **params: Any) -> dict[str, Any]:
        """Execute an asynchronous POST request.

        Args:
            **params: Request body parameters.
                None values are automatically filtered out.

        Returns:
            Parsed JSON response from the API.

        Raises:
            Exception: If the API returns a non-200 status code.
        """
        params = {k: v for k, v in params.items() if v is not None}

        async def fetch() -> dict[str, Any]:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._build_url(),
                    json=params,
                    headers=self._build_headers(),
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    text = await response.text()
                    if response.status != 200:
                        error = json.loads(text).get("error", "Unknown error")
                        if isinstance(error, dict):
                            error = error.get("message", "Unknown error")
                        raise ValueError(f"Error {response.status}: {error}")
                    return json.loads(text)

        return await fetch()


class ScavioSearchAPIWrapper(ScavioBaseAPIWrapper):
    """Wrapper for the Scavio Google Search endpoint (POST /api/v1/google)."""

    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/google"
