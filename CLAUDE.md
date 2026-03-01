# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development

```bash
# Create venv and install (from langchain-scavio/ directory)
python3 -m venv .venv
source .venv/bin/activate
pip install -e "."

# Install test deps
pip install pytest pytest-asyncio responses

# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run a single test file
.venv/bin/python -m pytest tests/test_search.py -v

# Run a single test
.venv/bin/python -m pytest tests/test_search.py::TestRun::test_successful_search -v

# Integration tests (requires SCAVIO_API_KEY env var)
.venv/bin/python -m pytest tests/ -m integration -v
```

Build system is Poetry (`pyproject.toml`), but no `poetry` CLI is required -- `pip install -e .` works directly.

## Architecture

Two-layer design modeled after `langchain-tavily`:

**`_utilities.py` (ScavioSearchAPIWrapper)** -- Raw HTTP layer. Pydantic `BaseModel` with `ConfigDict(extra="forbid")`. Handles auth (`SecretStr` + `SCAVIO_API_KEY` env var via `get_from_dict_or_env`), request building, sync (`requests`) and async (`aiohttp`) calls to `POST /api/v1/google`. New `aiohttp.ClientSession` per request (no persistent session -- avoids event loop issues).

**`scavio_search.py` (ScavioSearch)** -- LangChain `BaseTool` subclass. Custom `__init__` intercepts `scavio_api_key`/`api_base_url` and forwards to wrapper. Two-tier parameter model:
- **Init-only** (developer-controlled): `max_results`, `light_request`, `include_knowledge_graph`, `include_questions`, `include_related`. Enforced by `_INIT_ONLY_PARAMS` check in `_run`/`_arun` -- raises `ValueError` before the try block.
- **LLM-controllable** (via `ScavioSearchInput` args_schema): `query`, `search_type`, `country_code`, `language`, `device`, `page`.

**Error handling**: `handle_tool_error=True` on the tool. Empty results raise `ToolException` with suggestions (caught by LangChain, passed to LLM). API errors caught and returned as `{"error": "..."}` dict.

## Key Pydantic Patterns

- `ScavioSearchInput`: `ConfigDict(extra="allow")` -- required so LangChain can pass internal kwargs like `run_manager`
- `ScavioSearchAPIWrapper`: `ConfigDict(extra="forbid")` -- strict validation. This means `patch.object()` on wrapper instances won't work in tests; mock at class level or HTTP level instead
- `@model_validator(mode="before")` for env var fallback

## API Gotchas

- `light_request`: omit entirely for light mode (1 credit), send `false` for full mode (2 credits). `None` values are filtered out of the request body.
- Optional response fields (`knowledge_graph`, `questions`, `related_searches`, `news_results`, `top_stories`) are absent when empty, not `null`.
- `related_searches` items are `{"query": "string"}` objects, not bare strings.
- `knowledge_graph.factoids` uses `title`/`content` fields (not `label`/`value`).
- `search_type: "news"` only works with `device: "desktop"`.

## Testing Patterns

- Sync HTTP mocked with `responses` library (`@responses.activate`)
- Async HTTP mocked by patching `langchain_scavio._utilities.aiohttp.ClientSession` at class level (not instance, due to `extra="forbid"`)
- Mock response builders in `conftest.py`: `make_light_response()`, `make_full_response()`, `make_error_response()`
- Fixtures: `tool` (default config), `full_tool` (all features enabled)

## Reference

- PRD: `../prd-langchain.md`
- Handoff doc (API contract, gotchas): `../handoff-langchain.md`
- Reference implementation: [langchain-tavily](https://github.com/tavily-ai/langchain-tavily)
