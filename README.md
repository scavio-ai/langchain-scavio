# langchain-scavio

[![PyPI version](https://img.shields.io/pypi/v/langchain-scavio.svg)](https://pypi.org/project/langchain-scavio/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-scavio.svg)](https://pypi.org/project/langchain-scavio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-integration-blueviolet)](https://python.langchain.com/)

**47 LangChain tools for real-time search across Google, Amazon, Walmart, YouTube, Reddit, TikTok, TikTok Shop, and Instagram** -- structured data with knowledge graphs, all through a single package.

```bash
pip install langchain-scavio
```

Get your free API key at [dashboard.scavio.dev](https://dashboard.scavio.dev/).

## Why Scavio over Tavily?

Scavio is a full [Tavily alternative](https://scavio.dev/alternatives/tavily) built for multi-platform agents — here is [Tavily vs Scavio](https://scavio.dev/compare/tavily/vs-scavio) at a glance:

| | Scavio | Tavily | SerpAPI |
|---|---|---|---|
| **Platforms** | Google, YouTube, Amazon, Walmart, Reddit, TikTok, TikTok Shop, Instagram, X, LinkedIn | Google only | Google + others |
| **Tools** | 47 | 1 | 1 per wrapper |
| **Knowledge graphs** | Yes | No | Partial |
| **Product data** (price, rating, reviews) | Yes | No | No |
| **Pricing** | $0.005/credit | $0.01/search | $0.05/search |
| **Amazon marketplace coverage** | 22 countries | -- | -- |
| **LangChain async** | Yes | Yes | Yes |

## What Can You Build?

- **Shopping agents** -- search Amazon and Walmart, compare prices, find deals across 22 marketplaces
- **Product research agents** -- Google reviews + Amazon listings + YouTube reviews + Reddit opinions in one query
- **Content research agents** -- YouTube trends + Reddit sentiment + Google news in a single workflow
- **Brand monitoring** -- track what Reddit and Google say about any topic in real time
- **Social media agents** -- TikTok and Instagram profile analytics, hashtag tracking, post/video comments, and trend discovery

## Quick Start

```python
import os
from langchain_scavio import ScavioSearch

os.environ["SCAVIO_API_KEY"] = "sk_live_..."

tool = ScavioSearch()
result = tool.invoke({"query": "best python web frameworks 2026"})
```

## All 47 Tools

| Tool | Description |
|------|-------------|
| `ScavioSearch` | Google web search with knowledge graphs, PAA questions, news |
| `ScavioAmazonSearch` | Search Amazon product listings across 22 marketplaces |
| `ScavioAmazonProduct` | Fetch full details for an Amazon product by ASIN |
| `ScavioAmazonOffers` | Every seller offer for an ASIN: price, seller, condition, buy box |
| `ScavioWalmartSearch` | Search Walmart product listings with price/fulfillment filters |
| `ScavioWalmartProduct` | Fetch full details for a Walmart product by ID |
| `ScavioYouTubeSearch` | Search YouTube videos with duration/date/type/feature filters |
| `ScavioYouTubeVideo` | Fetch full details for a YouTube video (chapters, captions) |
| `ScavioYouTubeMetadata` | Deprecated alias of `ScavioYouTubeVideo` |
| `ScavioYouTubeComments` | Fetch comments on a YouTube video with pagination |
| `ScavioYouTubeTranscript` | Fetch a YouTube video transcript as text or SRT |
| `ScavioYouTubeChannel` | Fetch channel details by ID, @handle, or URL |
| `ScavioYouTubeChannelVideos` | Fetch a YouTube channel's uploaded videos |
| `ScavioYouTubeStreams` | Fetch playable/downloadable stream URLs for a video |
| `ScavioRedditSearch` | Search Reddit posts with cursor pagination |
| `ScavioRedditPost` | Fetch a Reddit post's metadata by URL (no comments) |
| `ScavioTikTokProfile` | Look up a TikTok user profile by username or sec_user_id |
| `ScavioTikTokUserPosts` | Fetch a TikTok user's posted videos with statistics |
| `ScavioTikTokVideo` | Fetch details for a single TikTok video |
| `ScavioTikTokVideoComments` | Fetch comments on a TikTok video |
| `ScavioTikTokCommentReplies` | Fetch replies to a specific comment on a TikTok video |
| `ScavioTikTokSearchVideos` | Search TikTok videos by keyword with sort/time filters |
| `ScavioTikTokSearchUsers` | Search TikTok users by keyword |
| `ScavioTikTokHashtag` | Look up TikTok hashtag info (video count, views) |
| `ScavioTikTokHashtagVideos` | Fetch TikTok videos for a specific hashtag |
| `ScavioTikTokUserFollowers` | Fetch a TikTok user's followers |
| `ScavioTikTokUserFollowings` | Fetch accounts a TikTok user is following |
| `ScavioTikTokShopSearch` | Search TikTok Shop products by keyword (US catalog) with exact prices |
| `ScavioTikTokShopSearchSuggestions` | Keyword autocomplete for TikTok Shop across 8 regions |
| `ScavioTikTokShopProduct` | Full TikTok Shop product detail (no price -- upstream masks it) |
| `ScavioTikTokShopProductReviews` | Paginated TikTok Shop reviews, up to 200 per call |
| `ScavioTikTokShopCategories` | The global TikTok Shop category tree (240 nodes, 2 levels) |
| `ScavioTikTokShopCategoryProducts` | Products under a TikTok Shop category, with exact prices |
| `ScavioTikTokShopShopProducts` | A TikTok Shop seller's catalog, with exact prices |
| `ScavioTikTokShopResolve` | Resolve a TikTok Shop URL or share link to a product_id / shop_id |
| `ScavioInstagramProfile` | Look up an Instagram user profile by username or user_id |
| `ScavioInstagramUserPosts` | Fetch an Instagram user's posts with statistics |
| `ScavioInstagramUserReels` | Fetch an Instagram user's reels with statistics |
| `ScavioInstagramTaggedPosts` | Fetch posts an Instagram user is tagged in |
| `ScavioInstagramStories` | Fetch an Instagram user's active stories |
| `ScavioInstagramPost` | Fetch details for a single Instagram post or reel |
| `ScavioInstagramPostComments` | Fetch comments on an Instagram post |
| `ScavioInstagramCommentReplies` | Fetch replies to a specific comment on an Instagram post |
| `ScavioInstagramSearchUsers` | Search Instagram users by keyword |
| `ScavioInstagramSearchHashtags` | Search Instagram hashtags by keyword |
| `ScavioInstagramUserFollowers` | Fetch an Instagram user's followers |
| `ScavioInstagramUserFollowings` | Fetch accounts an Instagram user is following |

## Use with a LangChain Agent

Scavio tools plug into the current [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) API from `langchain.agents`:

```python
from langchain.agents import create_agent
from langchain_scavio import (
    ScavioSearch,
    ScavioAmazonSearch, ScavioAmazonProduct,
    ScavioWalmartSearch,
    ScavioYouTubeSearch, ScavioYouTubeVideo, ScavioYouTubeTranscript,
    ScavioRedditSearch, ScavioRedditPost,
    ScavioTikTokSearchVideos, ScavioTikTokProfile, ScavioTikTokVideo,
)

agent = create_agent(
    "openai:gpt-5.5",
    tools=[
        ScavioSearch(max_results=5),
        ScavioAmazonSearch(max_results=5),
        ScavioAmazonProduct(),
        ScavioWalmartSearch(max_results=5),
        ScavioYouTubeSearch(max_results=5),
        ScavioYouTubeVideo(),
        ScavioYouTubeTranscript(),
        ScavioRedditSearch(max_results=5),
        ScavioRedditPost(),
        ScavioTikTokSearchVideos(max_results=5),
        ScavioTikTokProfile(),
        ScavioTikTokVideo(),
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
    light_request=None,                  # deprecated, ignored (v2 always full, 1 credit)
    include_knowledge_graph=True,
    include_questions=True,
    include_related=False,
    country_code="us",
    language="en",
    search_type="classic",               # classic|news|maps
    device="desktop",
)
```

### Amazon

```python
from langchain_scavio import ScavioAmazonSearch, ScavioAmazonProduct, ScavioAmazonOffers

search = ScavioAmazonSearch(max_results=5)
search.invoke({"query": "wireless headphones", "country": "us", "page": 1})

product = ScavioAmazonProduct()
product.invoke({"query": "B08N5WRWNW"})           # query = ASIN

offers = ScavioAmazonOffers()
offers.invoke({"query": "B08N5WRWNW"})            # every seller for that ASIN
```

> **Targeting a marketplace:** `country` takes a two-letter code, not a domain. Supported: `us` (default), `gb` (the UK is `gb`, not `uk`), `ca`, `de`, `fr`, `es`, `it`, `jp`, `in`, `au`, `br`, `mx`, `nl`, `pl`, `se`, `sg`, `ae`, `sa`, `eg`, `cn`, `be`, `tr`. An unrecognised code falls back to `us`.

> **Amazon changed in 3.0 (breaking).** The upstream provider moved and the request surface shrank. `sort_by`, `pages`, `category_id`, `merchant_id`, `language`, `currency`, `device`, `zip_code` and `autoselect_variant` are gone from all Amazon tools -- the marketplace ignores every one of them, so they are removed rather than kept as silent no-ops (`sort_by` was verified: all six sort values return the identical unordered set). `domain` and `start_page` still work on the wire and are still forwarded, but they are no longer in the tool schemas: use `country` and `page`. Response fields were renamed too -- `url_image` is now `image`, `best_seller`/`is_amazons_choice` collapsed into `badge`, and `buybox` is gone (use `ScavioAmazonOffers`).

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
from langchain_scavio import (
    ScavioYouTubeSearch, ScavioYouTubeVideo, ScavioYouTubeComments,
    ScavioYouTubeTranscript, ScavioYouTubeChannel,
    ScavioYouTubeChannelVideos, ScavioYouTubeStreams,
)

# Video search (2 credits per call)
search = ScavioYouTubeSearch(max_results=5)
result = search.invoke({
    "query": "python tutorial",
    "duration": "medium",                # short|medium|long
    "upload_date": "this_month",         # last_hour|today|this_week|this_month|this_year
    "sort_by": "view_count",             # relevance|date|view_count|rating
    "video_type": "video",               # video|channel|playlist
    "features": ["hd", "subtitles"],     # hd|4k|subtitles|creative_commons|live|360|3d|hdr|vr180
})

# Paginate with the previous response's next_cursor
next_page = search.invoke({
    "query": "python tutorial",
    "cursor": result["data"]["next_cursor"],
})

# Full video details (chapters, captions, keywords)
video = ScavioYouTubeVideo()
result = video.invoke({"video_id": "dQw4w9WgXcQ"})  # video ID or watch URL

# Comments (paginate via data.next_cursor)
comments = ScavioYouTubeComments(max_results=10)
result = comments.invoke({"video_id": "dQw4w9WgXcQ"})

# Transcript as plain text or timed SRT (8 credits per call)
transcript = ScavioYouTubeTranscript()
result = transcript.invoke({"video_id": "dQw4w9WgXcQ", "format": "text"})

# Channel details and uploads
channel = ScavioYouTubeChannel()
result = channel.invoke({"channel_id": "@YouTube"})  # ID, @handle, or URL

channel_videos = ScavioYouTubeChannelVideos(max_results=5)
result = channel_videos.invoke({"channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw"})

# Playable/downloadable stream URLs (3 credits per call)
streams = ScavioYouTubeStreams()
result = streams.invoke({"video_id": "dQw4w9WgXcQ"})

# ScavioYouTubeMetadata is a deprecated alias of ScavioYouTubeVideo
```

### Reddit

Reddit endpoints cost 1 credit each.

```python
from langchain_scavio import ScavioRedditSearch, ScavioRedditPost

search = ScavioRedditSearch(max_results=5)
result = search.invoke({"query": "langchain"})
# result["data"]["results"] + next_cursor + has_more
# Relevance order only: the endpoint has no sort or result-type filter

# Paginate by passing back the previous response's next_cursor
next_page = search.invoke({
    "query": "langchain",
    "cursor": result["data"]["next_cursor"],
})

post = ScavioRedditPost()
result = post.invoke({
    "url": "https://www.reddit.com/r/programming/comments/abc123/example_post/"
})
# result["data"] is a flat post object (post_id, title, text, score,
# upvote_ratio, num_comments, media). It does NOT return comments.
```

### TikTok

```python
from langchain_scavio import (
    ScavioTikTokProfile, ScavioTikTokUserPosts, ScavioTikTokVideo,
    ScavioTikTokVideoComments, ScavioTikTokCommentReplies,
    ScavioTikTokSearchVideos, ScavioTikTokSearchUsers,
    ScavioTikTokHashtag, ScavioTikTokHashtagVideos,
    ScavioTikTokUserFollowers, ScavioTikTokUserFollowings,
)

# Look up a user profile (returns sec_uid needed by other tools)
profile = ScavioTikTokProfile()
result = profile.invoke({"username": "tiktok"})
sec_uid = result["data"]["user"]["sec_uid"]

# Fetch their recent posts
posts = ScavioTikTokUserPosts(max_results=5)
result = posts.invoke({"sec_user_id": sec_uid, "sort_type": "1"})  # popular

# Search videos by keyword
search = ScavioTikTokSearchVideos(max_results=5)
result = search.invoke({
    "keyword": "python tutorial",
    "sort_type": "1",                        # 0=relevance, 1=most likes
    "publish_time": "30",                    # 0=all, 1=day, 7=week, 30=month
})

# Get video details and comments
video = ScavioTikTokVideo()
result = video.invoke({"video_id": "7123456789012345678"})

comments = ScavioTikTokVideoComments(max_results=10)
result = comments.invoke({"video_id": "7123456789012345678"})

# Hashtag research
hashtag = ScavioTikTokHashtag()
result = hashtag.invoke({"hashtag_name": "python"})
hashtag_id = result["data"]["challengeInfo"]["challenge"]["id"]

hashtag_videos = ScavioTikTokHashtagVideos(max_results=5)
result = hashtag_videos.invoke({"hashtag_id": hashtag_id})
```

### TikTok Shop

Eight tools over the TikTok Shop catalog. Two things to know before you wire
them together:

1. **`ScavioTikTokShopProduct` resolves only about 44% of the product ids that
   `ScavioTikTokShopSearch` returns.** Upstream has no detail data for the rest,
   so a not-found result is a normal outcome rather than an error -- skip the
   product instead of retrying. Search is a listing source, not the first leg of
   a reliable search-then-detail pipeline.
2. **`ScavioTikTokShopProduct` does not return a price.** Upstream masks the
   digits on the product page, so `price.current` and `price.original` come back
   null. Exact prices are on `ScavioTikTokShopSearch`,
   `ScavioTikTokShopShopProducts` and `ScavioTikTokShopCategoryProducts`.

```python
from langchain_scavio import (
    ScavioTikTokShopSearch, ScavioTikTokShopSearchSuggestions,
    ScavioTikTokShopProduct, ScavioTikTokShopProductReviews,
    ScavioTikTokShopCategories, ScavioTikTokShopCategoryProducts,
    ScavioTikTokShopShopProducts, ScavioTikTokShopResolve,
)

# Search the US catalog -- this is where exact prices live
search = ScavioTikTokShopSearch(max_results=10)
result = search.invoke({"search": "phone case"})
for product in result["data"]["products"]:
    print(product["title"], product["price"]["current"], product["rating"]["score"])

# Paginate with the opaque cursor; dedupe by product_id across pages
if result["data"]["has_more"]:
    page2 = search.invoke({
        "search": "phone case",
        "cursor": result["data"]["next_cursor"],
    })

# Product detail: rich, but priceless (literally) and only ~44% resolvable
detail = ScavioTikTokShopProduct()
result = detail.invoke({"product_id": "1732293553906094315"})
if result.get("not_found"):
    pass                                     # normal: skip it, do not retry
else:
    result["data"]["variants"]               # stock per SKU
    result["data"]["shop"]["followers_count"]

# Reviews: page with has_more, never with total_reviews (it drifts)
reviews = ScavioTikTokShopProductReviews(max_results=20)
result = reviews.invoke({
    "product_id": "1732293553906094315",
    "page_size": 100,
    "sort": "relevant",                      # "recent" is fresher but text-sparse
    "has_media": True,
})

# Category browse (US and GB only)
categories = ScavioTikTokShopCategories()
tree = categories.invoke({})
category_id = tree["data"]["categories"][0]["category_id"]

listing = ScavioTikTokShopCategoryProducts(max_results=10)
result = listing.invoke({"category_id": category_id})

# A seller's whole catalog, with exact prices
shop = ScavioTikTokShopShopProducts(max_results=10)
result = shop.invoke({"shop_id": "7495514739648989419"})

# Turn any share link into an id
resolve = ScavioTikTokShopResolve()
result = resolve.invoke({"url": "https://vt.tiktok.com/ZT2AHoGsE/"})
result["data"]["product_id"], result["data"]["type"]

# Keyword expansion, the only endpoint with genuine 8-region coverage
suggestions = ScavioTikTokShopSearchSuggestions()
result = suggestions.invoke({"search": "wireless", "region": "GB"})
result["data"]["suggestions"]                # plain strings, no volume or score
```

### Instagram

Instagram is priced per endpoint, not flat: 10 credits by default, 8 for
`ScavioInstagramPost` and `ScavioInstagramCommentReplies`, and 2 for
`ScavioInstagramUserPosts`.

```python
from langchain_scavio import (
    ScavioInstagramProfile, ScavioInstagramUserPosts, ScavioInstagramUserReels,
    ScavioInstagramTaggedPosts, ScavioInstagramStories,
    ScavioInstagramPost, ScavioInstagramPostComments,
    ScavioInstagramCommentReplies, ScavioInstagramSearchUsers,
    ScavioInstagramSearchHashtags,
    ScavioInstagramUserFollowers, ScavioInstagramUserFollowings,
)

# Look up a user profile (returns user_id usable by other tools)
profile = ScavioInstagramProfile()
result = profile.invoke({"username": "instagram"})
user_id = result["data"]["user"]["id"]

# Fetch their recent posts and reels
posts = ScavioInstagramUserPosts(max_results=5)
result = posts.invoke({"username": "instagram"})

reels = ScavioInstagramUserReels(max_results=5)
result = reels.invoke({"username": "instagram"})

# Get a single post's details and comments
post = ScavioInstagramPost()
result = post.invoke({"shortcode": "C1a2b3c4d5e"})

comments = ScavioInstagramPostComments(max_results=10)
result = comments.invoke({
    "shortcode": "C1a2b3c4d5e",
    "sort_order": "newest",                  # popular (default) or newest
})

# Search users and hashtags
search_users = ScavioInstagramSearchUsers(max_results=5)
result = search_users.invoke({"keyword": "cooking"})

search_hashtags = ScavioInstagramSearchHashtags(max_results=5)
result = search_hashtags.invoke({"keyword": "travel"})
```

## Agent-Controllable Parameters

### ScavioSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query |
| `search_type` | `classic\|news\|maps` | Type of search |
| `country_code` | `str` | ISO 3166-1 alpha-2 |
| `language` | `str` | ISO 639-1 |
| `device` | `desktop\|mobile` | Device type |
| `page` | `int` | Result page number |

### ScavioAmazonSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Product search query |
| `country` | `str` | Two-letter marketplace code (us, gb, de, jp, ...). Defaults to us |
| `page` | `int` | Result page, 1-based. One page per call, 1 credit each |

There is no sort, category, merchant or price filter: the marketplace ignores
them. Rank results yourself.

### ScavioAmazonProduct / ScavioAmazonOffers

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The ASIN |
| `country` | `str` | Two-letter marketplace code. Defaults to us |

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
| `features` | `list[str]` | hd\|4k\|subtitles\|creative_commons\|live\|360\|3d\|hdr\|vr180 |
| `cursor` | `str` | Pagination cursor from a prior response's `next_cursor` |

### ScavioYouTubeVideo / ScavioYouTubeMetadata

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_id` | `str` | YouTube video ID or watch URL |

### ScavioYouTubeComments

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_id` | `str` | YouTube video ID or watch URL |
| `cursor` | `str` | Pagination cursor from a prior response's `next_cursor` |

### ScavioYouTubeTranscript

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_id` | `str` | YouTube video ID or watch URL |
| `language` | `str` | Caption language code (ISO 639-1, default en) |
| `format` | `str` | text (default) or srt |

### ScavioYouTubeChannel

| Parameter | Type | Description |
|-----------|------|-------------|
| `channel_id` | `str` | Channel ID, @handle, or channel URL |

### ScavioYouTubeChannelVideos

| Parameter | Type | Description |
|-----------|------|-------------|
| `channel_id` | `str` | Channel ID |
| `cursor` | `str` | Pagination cursor from a prior response's `next_cursor` |

### ScavioYouTubeStreams

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_id` | `str` | YouTube video ID or watch URL |

### ScavioRedditSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Reddit search query (1-500 chars) |
| `cursor` | `str` | Opaque pagination cursor from prior response's `next_cursor` |

Results come back in relevance order. There is no sort or result-type
parameter: the endpoint accepts only `query` and `cursor`.

### ScavioRedditPost

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | Full Reddit post URL (www., old., or new. subdomains accepted) |

### ScavioTikTokProfile

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | TikTok handle without @ (provide this or `sec_user_id`) |
| `sec_user_id` | `str` | Secure user ID from a previous lookup |

### ScavioTikTokUserPosts

| Parameter | Type | Description |
|-----------|------|-------------|
| `sec_user_id` | `str` | Secure user ID from a profile lookup |
| `cursor` | `str` | Pagination cursor (from `data.max_cursor`) |
| `count` | `int` | Results per page (1-30, default 20) |
| `sort_type` | `str` | 0=latest (default), 1=popular |

### ScavioTikTokVideo

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_id` | `str` | TikTok video identifier |

### ScavioTikTokVideoComments

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_id` | `str` | TikTok video identifier |
| `cursor` | `str` | Pagination cursor |
| `count` | `int` | Results per page (1-50, default 20) |

### ScavioTikTokCommentReplies

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_id` | `str` | TikTok video identifier |
| `comment_id` | `str` | Comment ID from the comments endpoint |
| `cursor` | `str` | Pagination cursor |
| `count` | `int` | Results per page (1-50, default 20) |

### ScavioTikTokSearchVideos

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | `str` | Search query (1-500 chars) |
| `cursor` | `str` | Pagination offset |
| `count` | `int` | Results per page (1-30, default 20) |
| `sort_type` | `str` | 0=relevance (default), 1=most likes |
| `publish_time` | `str` | 0=all, 1=day, 7=week, 30=month, 90=3mo, 180=6mo |

### ScavioTikTokSearchUsers

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | `str` | Search query (1-500 chars) |
| `cursor` | `str` | Pagination offset |
| `count` | `int` | Results per page (1-30, default 20) |

### ScavioTikTokHashtag

| Parameter | Type | Description |
|-----------|------|-------------|
| `hashtag_name` | `str` | Hashtag text without # (provide this or `hashtag_id`) |
| `hashtag_id` | `str` | Numeric hashtag identifier |

### ScavioTikTokHashtagVideos

| Parameter | Type | Description |
|-----------|------|-------------|
| `hashtag_id` | `str` | Hashtag ID from the hashtag info endpoint |
| `cursor` | `str` | Pagination cursor |
| `count` | `int` | Results per page (1-30, default 20) |

### ScavioTikTokUserFollowers / ScavioTikTokUserFollowings

| Parameter | Type | Description |
|-----------|------|-------------|
| `sec_user_id` | `str` | Secure user ID from a profile lookup |
| `count` | `int` | Results per page (1-20, default 20) |
| `page_token` | `str` | Pagination token from `data.next_page_token` |
| `min_time` | `int` | Pagination field from `data.min_time` |

### ScavioInstagramProfile / ScavioInstagramStories

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | Instagram handle without @ (provide this or `user_id`) |
| `user_id` | `str` | Numeric user ID from a previous lookup |

### ScavioInstagramUserPosts / ScavioInstagramUserReels / ScavioInstagramTaggedPosts / ScavioInstagramUserFollowers / ScavioInstagramUserFollowings

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | Instagram handle without @ (provide this or `user_id`) |
| `user_id` | `str` | Numeric user ID from a profile lookup |
| `count` | `int` | Results per page (1-50, default 12) |
| `cursor` | `str` | Pagination cursor from a prior response |

### ScavioInstagramPost

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | Full Instagram post or reel URL |
| `media_id` | `str` | Numeric media identifier (provide one of url, media_id, shortcode) |
| `shortcode` | `str` | Shortcode from the post URL (after /p/ or /reel/) |

### ScavioInstagramPostComments

| Parameter | Type | Description |
|-----------|------|-------------|
| `shortcode` | `str` | Post shortcode (provide this or `url`) |
| `url` | `str` | Full Instagram post or reel URL |
| `cursor` | `str` | Pagination cursor |
| `sort_order` | `str` | popular (default) or newest |

### ScavioInstagramCommentReplies

| Parameter | Type | Description |
|-----------|------|-------------|
| `media_id` | `str` | Numeric media ID of the post |
| `comment_id` | `str` | Comment ID from the post comments endpoint |
| `cursor` | `str` | Pagination cursor |

### ScavioInstagramSearchUsers / ScavioInstagramSearchHashtags

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | `str` | Search query (1-500 chars) |
| `cursor` | `str` | Pagination cursor |

## Error Handling

- Empty results raise `ToolException` with actionable suggestions for the LLM
- API errors return `{"error": "message"}` without crashing the agent
- `handle_tool_error=True` ensures LangChain passes errors to the LLM as context

## Architecture

```
ScavioBaseAPIWrapper                      # Auth, headers, sync/async HTTP POST
  +-- ScavioSearchAPIWrapper              # -> /api/v2/google (+ news, maps)
  +-- ScavioAmazonSearchAPIWrapper        # -> /api/v1/amazon/search
  +-- ScavioAmazonProductAPIWrapper       # -> /api/v1/amazon/product
  +-- ScavioAmazonOffersAPIWrapper        # -> /api/v1/amazon/offers
  +-- ScavioWalmartSearchAPIWrapper       # -> /api/v1/walmart/search
  +-- ScavioWalmartProductAPIWrapper      # -> /api/v1/walmart/product
  +-- ScavioYouTubeSearchAPIWrapper       # -> /api/v1/youtube/search
  +-- ScavioYouTubeVideoAPIWrapper        # -> /api/v1/youtube/video
  +-- ScavioYouTubeMetadataAPIWrapper     # -> /api/v1/youtube/video (deprecated alias)
  +-- ScavioYouTubeCommentsAPIWrapper     # -> /api/v1/youtube/comments
  +-- ScavioYouTubeTranscriptAPIWrapper   # -> /api/v1/youtube/transcript
  +-- ScavioYouTubeChannelAPIWrapper      # -> /api/v1/youtube/channel
  +-- ScavioYouTubeChannelVideosAPIWrapper# -> /api/v1/youtube/channel/videos
  +-- ScavioYouTubeStreamsAPIWrapper      # -> /api/v1/youtube/streams
  +-- ScavioRedditSearchAPIWrapper        # -> /api/v1/reddit/search
  +-- ScavioRedditPostAPIWrapper          # -> /api/v1/reddit/post
  +-- ScavioTikTokProfileAPIWrapper       # -> /api/v1/tiktok/profile
  +-- ScavioTikTokUserPostsAPIWrapper     # -> /api/v1/tiktok/user/posts
  +-- ScavioTikTokVideoAPIWrapper         # -> /api/v1/tiktok/video
  +-- ScavioTikTokVideoCommentsAPIWrapper # -> /api/v1/tiktok/video/comments
  +-- ScavioTikTokCommentRepliesAPIWrapper# -> /api/v1/tiktok/video/comments/replies
  +-- ScavioTikTokSearchVideosAPIWrapper  # -> /api/v1/tiktok/search/videos
  +-- ScavioTikTokSearchUsersAPIWrapper   # -> /api/v1/tiktok/search/users
  +-- ScavioTikTokHashtagAPIWrapper       # -> /api/v1/tiktok/hashtag
  +-- ScavioTikTokHashtagVideosAPIWrapper # -> /api/v1/tiktok/hashtag/videos
  +-- ScavioTikTokUserFollowersAPIWrapper # -> /api/v1/tiktok/user/followers
  +-- ScavioTikTokUserFollowingsAPIWrapper# -> /api/v1/tiktok/user/followings
  +-- ScavioInstagramProfileAPIWrapper        # -> /api/v1/instagram/profile
  +-- ScavioInstagramUserPostsAPIWrapper      # -> /api/v1/instagram/user/posts
  +-- ScavioInstagramUserReelsAPIWrapper      # -> /api/v1/instagram/user/reels
  +-- ScavioInstagramTaggedPostsAPIWrapper    # -> /api/v1/instagram/user/tagged
  +-- ScavioInstagramStoriesAPIWrapper        # -> /api/v1/instagram/user/stories
  +-- ScavioInstagramPostAPIWrapper           # -> /api/v1/instagram/post
  +-- ScavioInstagramPostCommentsAPIWrapper   # -> /api/v1/instagram/post/comments
  +-- ScavioInstagramCommentRepliesAPIWrapper # -> /api/v1/instagram/post/comments/replies
  +-- ScavioInstagramSearchUsersAPIWrapper    # -> /api/v1/instagram/search/users
  +-- ScavioInstagramSearchHashtagsAPIWrapper # -> /api/v1/instagram/search/hashtags
  +-- ScavioInstagramUserFollowersAPIWrapper  # -> /api/v1/instagram/user/followers
  +-- ScavioInstagramUserFollowingsAPIWrapper # -> /api/v1/instagram/user/followings
```

Each tool splits parameters into **init-only** (developer-controlled, e.g. `max_results`, `domain`) and **LLM-controllable** (passed via `args_schema` at invocation time, e.g. `query`, `sort_by`).

## Migrating from Tavily

```diff
- from langchain_tavily import TavilySearch
+ from langchain_scavio import ScavioSearch

- tool = TavilySearch(max_results=5)
+ tool = ScavioSearch(max_results=5)
```

See the full [migration guide](.docs/tavily-migration.md) for parameter mapping and feature comparison, or read more on [migrating from Tavily](https://scavio.dev/alternatives/tavily).

## License

MIT


## About Scavio

[Scavio](https://scavio.dev) is a unified [search API for AI agents](https://scavio.dev/search-api-for-ai-agents) — one API key, structured JSON, no scraping or proxies. A real-time [Tavily alternative](https://scavio.dev/alternatives/tavily) and [SerpAPI alternative](https://scavio.dev/alternatives/serpapi) with data from:

- [Google Search API](https://scavio.dev/google-search-api) — SERP results, news, images, maps, and knowledge graph
- [Amazon Product API](https://scavio.dev/amazon-product-api) and [Walmart Product API](https://scavio.dev/walmart-product-api) — product search and details
- [YouTube API](https://scavio.dev/youtube-transcript-api), [TikTok API](https://scavio.dev/tiktok-api), and [Instagram API](https://scavio.dev/instagram-api) — video and social media data
- [Reddit API](https://scavio.dev/reddit-api) — posts and threaded comments
- TikTok Shop, X (formerly Twitter), and LinkedIn — product listings, tweets, profiles, company pages, and job listings

X and LinkedIn are covered by the Scavio API but do not yet have tools in this package.

Get a free [API key](https://dashboard.scavio.dev) and explore the [documentation](https://scavio.dev/docs/introduction).
