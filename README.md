# langchain-scavio

LangChain integration for the [Scavio Search API](https://scavio.dev). Provides real-time web search with structured SERP data including knowledge graphs, "People Also Ask", and related searches.

## Installation

```bash
pip install langchain-scavio
```

## Quick Start

```python
from langchain_scavio import ScavioSearch

# Uses SCAVIO_API_KEY environment variable
tool = ScavioSearch()

result = tool.invoke({"query": "best python web frameworks 2026"})
```

## Configuration

```python
tool = ScavioSearch(
    scavio_api_key="sk_live_...",       # or set SCAVIO_API_KEY env var
    max_results=5,                       # truncate results (default: 5)
    light_request=None,                  # None=light/1 credit, False=full/2 credits
    nfpr=False,                          # disable autocorrection (default: False)

    # Response field filters (include/exclude sections from results)
    include_knowledge_graph=True,        # default: True
    include_questions=True,              # "people also ask" (default: True)
    include_related=False,               # related queries + searches (default: False)
    include_maps_results=False,          # maps results (default: False)
    include_ai_overviews=False,          # AI overviews (default: False)
    include_news_results=False,          # news results (default: False)
    include_local_results=False,         # local results (default: False)
    include_top_stories=False,           # top stories (default: False)
    include_hotel_results=False,         # hotel results (default: False)
    include_shopping_ads=False,          # shopping ads (default: False)
    include_top_ads=False,               # top ads (default: False)
    include_bottom_ads=False,            # bottom ads (default: False)

    # Default search parameters (LLM can override per-query)
    country_code="us",                   # ISO 3166-1 alpha-2
    language="en",                       # ISO 639-1
    search_type="classic",               # classic|news|maps|images|lens
    device="desktop",                    # desktop|mobile
    page=1,                              # result page number
)
```

## Agent Usage

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_scavio import ScavioSearch

tool = ScavioSearch(max_results=5)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[tool],
    system_prompt="You are a research assistant with web search access.",
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What are the latest AI regulations in the EU?"}]
})
```

## LangGraph

```python
from langgraph.prebuilt import ToolNode
from langchain_scavio import ScavioSearch

tool_node = ToolNode([ScavioSearch()])
```

## Async

```python
result = await tool.ainvoke({"query": "async python frameworks"})
```

## Agent-Controllable Parameters

The LLM can dynamically set these at invocation time (overrides init defaults):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Search query (1-500 chars) |
| `search_type` | `classic\|news\|maps\|images\|lens` | `classic` | Type of search |
| `country_code` | `str` | `None` | ISO 3166-1 alpha-2 country code |
| `language` | `str` | `None` | ISO 639-1 language code |
| `device` | `desktop\|mobile` | `desktop` | Device type (`news` only supports `desktop`) |
| `page` | `int` | `1` | Result page number |

## Error Handling

- Empty results raise `ToolException` with actionable suggestions for the LLM
- API errors return `{"error": "message"}` without crashing the agent
- `handle_tool_error=True` ensures LangChain passes errors to the LLM as context

## Architecture

Two-layer design:

```
ScavioBaseAPIWrapper          # Auth, headers, sync/async HTTP POST
  └── ScavioSearchAPIWrapper  # _build_url() -> /api/v1/google

ScavioSearch(BaseTool)        # LangChain tool wrapping ScavioSearchAPIWrapper
```

- **`ScavioBaseAPIWrapper`** -- shared plumbing (API key via `SCAVIO_API_KEY` env var, `_build_headers()`, `raw_results()`, `raw_results_async()`). Subclasses override `_build_url()`.
- **`ScavioSearchAPIWrapper`** -- thin subclass targeting the Google Search endpoint.
- **`ScavioSearch`** -- LangChain `BaseTool` with init-only params (developer-controlled) and an `args_schema` for LLM-controllable params.

## Extending

To add a new Scavio endpoint (e.g. YouTube):

```python
# langchain_scavio/_utilities.py
class ScavioYouTubeAPIWrapper(ScavioBaseAPIWrapper):
    def _build_url(self) -> str:
        base = self.api_base_url or SCAVIO_API_URL
        return f"{base}/api/v1/youtube"

# langchain_scavio/scavio_youtube.py
class ScavioYouTube(BaseTool):
    api_wrapper: ScavioYouTubeAPIWrapper = Field(
        default_factory=ScavioYouTubeAPIWrapper
    )
    # ... define args_schema, _run, _arun
```

## License

MIT
