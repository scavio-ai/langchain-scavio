# Migrating from Tavily to Scavio

## Installation

```diff
- pip install langchain-tavily
+ pip install langchain-scavio
```

## Import Changes

```diff
- from langchain_tavily import TavilySearch
+ from langchain_scavio import ScavioSearch
```

## Parameter Mapping

| Tavily | Scavio | Notes |
|--------|--------|-------|
| `max_results` | `max_results` | Same behavior |
| `search_depth="basic"` | `light_request=None` | Light mode (1 credit) |
| `search_depth="advanced"` | `light_request=False` | Full mode (2 credits) |
| `include_answer` | -- | Not applicable; use an LLM to synthesize |
| `include_raw_content` | -- | Scavio returns structured data by default |
| `include_images` | `search_type="images"` | Dedicated image search mode |
| `topic="news"` | `search_type="news"` | Dedicated news search mode |
| `days` (news recency) | -- | Use date-range operators in query |
| -- | `include_knowledge_graph` | Scavio-only: structured entity data |
| -- | `include_questions` | Scavio-only: People Also Ask results |
| -- | `include_related` | Scavio-only: related search queries |
| -- | `include_ai_overviews` | Scavio-only: AI overview snippets |
| -- | `country_code` | Scavio-only: geo-targeted results |
| -- | `language` | Scavio-only: language targeting |
| -- | `device` | Scavio-only: desktop vs mobile results |

## Feature Comparison

| Feature | Tavily | Scavio |
|---------|--------|--------|
| Google web search | Yes | Yes |
| Read any URL | Yes | Yes (`ScavioExtract`, Markdown/text/HTML) |
| Amazon product search | No | Yes (22 marketplaces) |
| Walmart / Target / eBay / Home Depot | No | Yes |
| YouTube search | No | Yes |
| Reddit search | No | Yes |
| Real estate (Zillow, Redfin) | No | Yes |
| Travel (Booking.com, Airbnb, Tripadvisor) | No | Yes |
| Jobs and employers (Indeed, Glassdoor) | No | Yes |
| App stores (Apple, Google Play) | No | Yes |
| Filings (SEC EDGAR, Companies House) | No | Yes |
| Software reviews (G2, Capterra) | No | Yes |
| Ad transparency (Google Ads, Meta Ad Library) | No | Yes |
| Knowledge graphs | No | Yes |
| People Also Ask | No | Yes |
| AI Overviews | No | Yes |
| Structured product data | No | Yes (price, rating, reviews, availability) |
| Total tools | 1 | 187 |
| Pricing | $0.01/search | $0.005/credit |
| Async support | Yes | Yes |

## Agent Migration

```diff
  from langchain.agents import create_agent
- from langchain_tavily import TavilySearch
+ from langchain_scavio import (
+     ScavioSearch,
+     ScavioAmazonSearch,
+     ScavioWalmartSearch,
+     ScavioYouTubeSearch,
+     ScavioRedditSearch,
+ )

  agent = create_agent(
      "openai:gpt-4o",
-     tools=[TavilySearch(max_results=5)],
+     tools=[
+         ScavioSearch(max_results=5),
+         ScavioAmazonSearch(max_results=5),
+         ScavioWalmartSearch(max_results=5),
+         ScavioYouTubeSearch(max_results=5),
+         ScavioRedditSearch(max_results=5),
+     ],
  )
```

## Get Your API Key

Sign up at [dashboard.scavio.dev](https://dashboard.scavio.dev/) to get your free API key.
