# langchain-scavio

[![PyPI version](https://img.shields.io/pypi/v/langchain-scavio.svg)](https://pypi.org/project/langchain-scavio/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-scavio.svg)](https://pypi.org/project/langchain-scavio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-integration-blueviolet)](https://python.langchain.com/)

LangChain integration for the [Scavio Search API](https://scavio.dev). Real-time structured data from Google, Amazon, Walmart, YouTube, and Reddit — all through a single package.

**Why Scavio?** Multi-platform coverage, structured knowledge graph data, and competitive pricing at $0.005/credit.

## Installation

```bash
pip install langchain-scavio
```

## Tools

| Tool | Description |
|------|-------------|
| `ScavioSearch` | Google web search with knowledge graphs, PAA questions, news |
| `ScavioAmazonSearch` | Search Amazon product listings |
| `ScavioAmazonProduct` | Fetch full details for an Amazon product by ASIN |
| `ScavioWalmartSearch` | Search Walmart product listings |
| `ScavioWalmartProduct` | Fetch full details for a Walmart product by ID |
| `ScavioYouTubeSearch` | Search YouTube videos with duration/date/type filters |
| `ScavioYouTubeMetadata` | Fetch metadata for a YouTube video by video ID |
| `ScavioRedditSearch` | Search Reddit posts or comments with sort/pagination |
| `ScavioRedditPost` | Fetch a Reddit post's metadata and comment thread by URL |

## Quick Start

Get your API key at [dashboard.scavio.dev](https://dashboard.scavio.dev/).

```python
import os
from langchain_scavio import ScavioSearch

os.environ["SCAVIO_API_KEY"] = "sk_live_..."

tool = ScavioSearch()
result = tool.invoke({"query": "best python web frameworks 2026"})
```

## Use with a LangChain Agent

Scavio tools plug into the current [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) API from `langchain.agents`:

```python
from langchain.agents import create_agent
from langchain_scavio import (
    ScavioSearch,
    ScavioAmazonSearch, ScavioAmazonProduct,
    ScavioWalmartSearch,
    ScavioYouTubeSearch, ScavioYouTubeMetadata,
    ScavioRedditSearch, ScavioRedditPost,
)

agent = create_agent(
    "openai:gpt-4o",
    tools=[
        ScavioSearch(max_results=5),
        ScavioAmazonSearch(max_results=5),
        ScavioAmazonProduct(),
        ScavioWalmartSearch(max_results=5),
        ScavioYouTubeSearch(max_results=5),
        ScavioYouTubeMetadata(),
        ScavioRedditSearch(max_results=5),
        ScavioRedditPost(),
    ],
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Find me a Python book on Amazon under $30"}]
})
```

## Async Support

All tools support async invocation:

```python
result = await tool.ainvoke({"query": "async python frameworks"})
```

## Configuration

### Google Search

```python
from langchain_scavio import ScavioSearch

tool = ScavioSearch(
    scavio_api_key="sk_live_...",       # or SCAVIO_API_KEY env var
    max_results=5,
    light_request=None,                  # None=light/1 credit, False=full/2 credits
    include_knowledge_graph=True,
    include_questions=True,
    include_related=False,
    country_code="us",
    language="en",
    search_type="classic",               # classic|news|maps|images|lens
    device="desktop",
)
```

### Amazon

```python
from langchain_scavio import ScavioAmazonSearch, ScavioAmazonProduct

search = ScavioAmazonSearch(
    max_results=5,
    pages=1,                             # number of result pages to fetch
    domain="com",                        # see supported marketplaces below
)

product = ScavioAmazonProduct()
result = product.invoke({"query": "B08N5WRWNW"})  # query = ASIN
```

> **Targeting a marketplace:** use `domain` to pick which Amazon store to search — **do not** use a country code. Supported domains: `com` (US), `co.uk` (UK), `ca`, `de`, `fr`, `es`, `it`, `co.jp`, `in`, `com.au`, `com.br`, `com.mx`, `nl`, `pl`, `se`, `sg`, `ae`, `sa`, `eg`, `cn`, `com.be`, `com.tr`.

### Walmart

```python
from langchain_scavio import ScavioWalmartSearch, ScavioWalmartProduct

search = ScavioWalmartSearch(max_results=5)
result = search.invoke({
    "query": "air fryer",
    "sort_by": "price_low",              # best_match|price_low|price_high|best_seller
    "max_price": 5000,                   # in cents
    "fulfillment_speed": "2_days",       # today|tomorrow|2_days|anytime
})

product = ScavioWalmartProduct()
result = product.invoke({"product_id": "123456789"})
```

### YouTube

```python
from langchain_scavio import ScavioYouTubeSearch, ScavioYouTubeMetadata

search = ScavioYouTubeSearch(max_results=5)
result = search.invoke({
    "query": "python tutorial",
    "duration": "medium",                # short|medium|long
    "upload_date": "this_month",         # last_hour|today|this_week|this_month|this_year
    "sort_by": "view_count",             # relevance|date|view_count|rating
    "video_type": "video",               # video|channel|playlist
})

metadata = ScavioYouTubeMetadata()
result = metadata.invoke({"video_id": "dQw4w9WgXcQ"})
```

### Reddit

Reddit endpoints cost 2 credits each and typically take 5-15 seconds (JS rendering required).

```python
from langchain_scavio import ScavioRedditSearch, ScavioRedditPost

search = ScavioRedditSearch(max_results=5)
result = search.invoke({
    "query": "langchain",
    "sort": "top",                       # new|relevance|hot|top|comments
    "type": "posts",                     # posts|comments
})

# Paginate by passing back the previous response's nextCursor
next_page = search.invoke({
    "query": "langchain",
    "sort": "top",
    "cursor": result["data"]["nextCursor"],
})

post = ScavioRedditPost()
result = post.invoke({
    "url": "https://www.reddit.com/r/programming/comments/abc123/example_post/"
})
# result["data"]["post"] + result["data"]["comments"] (flat list with `depth`)
```

## Agent-Controllable Parameters

### ScavioSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query |
| `search_type` | `classic\|news\|maps\|images\|lens` | Type of search |
| `country_code` | `str` | ISO 3166-1 alpha-2 |
| `language` | `str` | ISO 639-1 |
| `device` | `desktop\|mobile` | Device type |
| `page` | `int` | Result page number |

### ScavioAmazonSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Product search query |
| `domain` | `str` | Amazon marketplace — the only way to select a store (com, co.uk, de, co.jp, ...) |
| `sort_by` | `str` | featured\|most_recent\|price_low_to_high\|price_high_to_low\|average_review\|bestsellers |
| `start_page` | `int` | Page number |
| `category_id` | `str` | Category filter |
| `merchant_id` | `str` | Seller filter |
| `language` / `currency` | `str` | Localization |
| `zip_code` | `str` | Local pricing |

### ScavioWalmartSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Product search query |
| `sort_by` | `str` | best_match\|price_low\|price_high\|best_seller |
| `min_price` / `max_price` | `int` | Price range in cents |
| `fulfillment_speed` | `str` | today\|tomorrow\|2_days\|anytime |
| `delivery_zip` | `str` | Delivery ZIP code |

### ScavioYouTubeSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query |
| `upload_date` | `str` | last_hour\|today\|this_week\|this_month\|this_year |
| `video_type` | `str` | video\|channel\|playlist |
| `duration` | `str` | short\|medium\|long |
| `sort_by` | `str` | relevance\|date\|view_count\|rating |
| `hd` / `subtitles` / `live` | `bool` | Content filters |

### ScavioRedditSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Reddit search query (1-500 chars) |
| `type` | `str` | posts\|comments |
| `sort` | `str` | new\|relevance\|hot\|top\|comments |
| `cursor` | `str` | Opaque pagination cursor from prior response's `nextCursor` |

### ScavioRedditPost

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | Full Reddit post URL (www., old., or new. subdomains accepted) |

## Error Handling

- Empty results raise `ToolException` with actionable suggestions for the LLM
- API errors return `{"error": "message"}` without crashing the agent
- `handle_tool_error=True` ensures LangChain passes errors to the LLM as context

## Architecture

```
ScavioBaseAPIWrapper                      # Auth, headers, sync/async HTTP POST
  +-- ScavioSearchAPIWrapper              # -> /api/v1/google
  +-- ScavioAmazonSearchAPIWrapper        # -> /api/v1/amazon/search
  +-- ScavioAmazonProductAPIWrapper       # -> /api/v1/amazon/product
  +-- ScavioWalmartSearchAPIWrapper       # -> /api/v1/walmart/search
  +-- ScavioWalmartProductAPIWrapper      # -> /api/v1/walmart/product
  +-- ScavioYouTubeSearchAPIWrapper       # -> /api/v1/youtube/search
  +-- ScavioYouTubeMetadataAPIWrapper     # -> /api/v1/youtube/metadata
  +-- ScavioRedditSearchAPIWrapper        # -> /api/v1/reddit/search
  +-- ScavioRedditPostAPIWrapper          # -> /api/v1/reddit/post
```

Each tool splits parameters into **init-only** (developer-controlled, e.g. `max_results`, `domain`) and **LLM-controllable** (passed via `args_schema` at invocation time, e.g. `query`, `sort_by`).

## Migrating from Tavily

```diff
- from langchain_tavily import TavilySearch
+ from langchain_scavio import ScavioSearch

- tool = TavilySearch(max_results=5)
+ tool = ScavioSearch(max_results=5)
```

See the full [migration guide](.docs/tavily-migration.md) for parameter mapping and feature comparison.

## License

MIT
