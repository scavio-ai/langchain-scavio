# Contributing to langchain-scavio

## Setup

```bash
git clone https://github.com/scavio-ai/langchain-scavio.git
cd langchain-scavio
python3 -m venv .venv
source .venv/bin/activate
pip install -e "."
pip install pytest pytest-asyncio responses ruff mypy types-requests
```

## Development workflow

1. Create a branch from `main`
2. Make changes
3. Run tests: `python -m pytest tests/ -v`
4. Run linter: `ruff check langchain_scavio/`
5. Run type checker: `mypy langchain_scavio/`
6. Open a PR against `main`

## Running tests

```bash
# All tests
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_search.py -v

# Single test
python -m pytest tests/test_search.py::TestRun::test_successful_search -v

# Integration tests (requires SCAVIO_API_KEY)
python -m pytest tests/ -m integration -v
```

## Code style

- Formatter/linter: `ruff` (rules: E, F, I)
- Type checking: `mypy` with `disallow_untyped_defs`
- Pydantic v2 patterns throughout

## Testing patterns

- Sync HTTP: mock with `responses` library (`@responses.activate`)
- Async HTTP: patch `langchain_scavio._utilities.aiohttp.ClientSession` at class level
- Mock response builders live in `tests/conftest.py`

## Architecture

Two-layer design:

- `_utilities.py` -- Raw HTTP layer (`ScavioBaseAPIWrapper` / `ScavioSearchAPIWrapper`)
- `scavio_search.py` -- LangChain `BaseTool` wrapper (`ScavioSearch`)

Init-only params (developer-controlled) vs LLM-controllable params (via `args_schema`).

## Pull requests

- PRs require CI to pass and 1 approval before merging
- Keep changes focused -- one feature/fix per PR
- Include tests for new functionality
