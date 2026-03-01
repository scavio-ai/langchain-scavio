"""Tests for ScavioSearchAPIWrapper."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import responses

from langchain_scavio._utilities import SCAVIO_API_URL, ScavioSearchAPIWrapper

from .conftest import MOCK_API_KEY, make_error_response, make_light_response

API_ENDPOINT = f"{SCAVIO_API_URL}/api/v1/google"


class TestValidation:
    def test_api_key_from_param(self) -> None:
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        assert wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SCAVIO_API_KEY", MOCK_API_KEY)
        wrapper = ScavioSearchAPIWrapper()
        assert wrapper.scavio_api_key.get_secret_value() == MOCK_API_KEY

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SCAVIO_API_KEY", raising=False)
        with pytest.raises(ValueError):
            ScavioSearchAPIWrapper()

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            ScavioSearchAPIWrapper(
                scavio_api_key=MOCK_API_KEY, unknown_field="bad"
            )

    def test_custom_base_url(self) -> None:
        wrapper = ScavioSearchAPIWrapper(
            scavio_api_key=MOCK_API_KEY,
            api_base_url="https://custom.api.dev",
        )
        assert wrapper._build_url() == "https://custom.api.dev/api/v1/google"


class TestSyncRequests:
    @responses.activate
    def test_successful_search(self) -> None:
        mock_resp = make_light_response()
        responses.add(
            responses.POST, API_ENDPOINT, json=mock_resp, status=200
        )
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        result = wrapper.raw_results(query="test")
        assert result["query"] == "test query"
        assert len(result["results"]) == 10

    @responses.activate
    def test_sends_correct_headers(self) -> None:
        responses.add(
            responses.POST, API_ENDPOINT, json=make_light_response(), status=200
        )
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        wrapper.raw_results(query="test")
        sent_headers = responses.calls[0].request.headers
        assert sent_headers["Authorization"] == f"Bearer {MOCK_API_KEY}"
        assert sent_headers["X-Client-Source"] == "langchain-scavio"
        assert sent_headers["Content-Type"] == "application/json"

    @responses.activate
    def test_filters_none_params(self) -> None:
        responses.add(
            responses.POST, API_ENDPOINT, json=make_light_response(), status=200
        )
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        wrapper.raw_results(query="test", country_code=None, language=None)
        body = json.loads(responses.calls[0].request.body)
        assert "country_code" not in body
        assert "language" not in body
        assert body["query"] == "test"

    @responses.activate
    def test_light_request_none_excluded(self) -> None:
        responses.add(
            responses.POST, API_ENDPOINT, json=make_light_response(), status=200
        )
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        wrapper.raw_results(query="test", light_request=None)
        body = json.loads(responses.calls[0].request.body)
        assert "light_request" not in body

    @responses.activate
    def test_light_request_false_included(self) -> None:
        responses.add(
            responses.POST, API_ENDPOINT, json=make_light_response(), status=200
        )
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        wrapper.raw_results(query="test", light_request=False)
        body = json.loads(responses.calls[0].request.body)
        assert body["light_request"] is False

    @responses.activate
    def test_401_raises_value_error(self) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        responses.add(responses.POST, API_ENDPOINT, json=error, status=401)
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        with pytest.raises(ValueError, match="Invalid API key"):
            wrapper.raw_results(query="test")

    @responses.activate
    def test_429_raises_value_error(self) -> None:
        error = make_error_response(429, "rate_limit_exceeded", "Rate limited")
        responses.add(responses.POST, API_ENDPOINT, json=error, status=429)
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        with pytest.raises(ValueError, match="Rate limited"):
            wrapper.raw_results(query="test")

    @responses.activate
    def test_402_raises_value_error(self) -> None:
        error = make_error_response(
            402, "insufficient_credits", "Insufficient credits"
        )
        responses.add(responses.POST, API_ENDPOINT, json=error, status=402)
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)
        with pytest.raises(ValueError, match="Insufficient credits"):
            wrapper.raw_results(query="test")


class TestAsyncRequests:
    @pytest.mark.asyncio
    async def test_successful_async_search(self) -> None:
        mock_resp = make_light_response()
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps(mock_resp))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)

        with patch(
            "langchain_scavio._utilities.aiohttp.ClientSession",
            return_value=mock_session_ctx,
        ):
            result = await wrapper.raw_results_async(query="test")
            assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_async_error_raises(self) -> None:
        error = make_error_response(401, "unauthorized", "Invalid API key")
        wrapper = ScavioSearchAPIWrapper(scavio_api_key=MOCK_API_KEY)

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value=json.dumps(error))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)

        with patch(
            "langchain_scavio._utilities.aiohttp.ClientSession",
            return_value=mock_session_ctx,
        ):
            with pytest.raises(ValueError, match="Invalid API key"):
                await wrapper.raw_results_async(query="test")
