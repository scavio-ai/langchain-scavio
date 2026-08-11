# langchain-scavio

[![PyPI version](https://img.shields.io/pypi/v/langchain-scavio.svg)](https://pypi.org/project/langchain-scavio/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-scavio.svg)](https://pypi.org/project/langchain-scavio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-integration-blueviolet)](https://python.langchain.com/)

**187 LangChain tools across 32 platforms** -- Google, YouTube, Amazon, Walmart, Target, eBay, Home Depot, Reddit, TikTok, TikTok Shop, Instagram, X (Twitter), Threads, Kuaishou, LinkedIn, Indeed, Glassdoor, Zillow, Redfin, Booking.com, Airbnb, Tripadvisor, Yelp, the Apple App Store, Google Play, SEC EDGAR, Companies House, G2, Capterra, Google Ads Transparency and the Meta Ad Library -- plus `ScavioExtract`, which reads **any** URL as Markdown, plain text or raw HTML. Structured JSON, one API key, no scraping or proxies.

```bash
pip install langchain-scavio
```

Get your free API key at [dashboard.scavio.dev](https://dashboard.scavio.dev/).

## Why Scavio over Tavily?

Scavio is a full [Tavily alternative](https://scavio.dev/alternatives/tavily) built for multi-platform agents — here is [Tavily vs Scavio](https://scavio.dev/compare/tavily/vs-scavio) at a glance:

| | Scavio | Tavily | SerpAPI |
|---|---|---|---|
| **Platforms** | 32 (Google, retail, real estate, travel, jobs, app stores, filings, software reviews, ad transparency) | Google only | Google + others |
| **Tools** | 187 | 1 | 1 per wrapper |
| **Read any URL** | Yes (`ScavioExtract`) | Yes | No |
| **Knowledge graphs** | Yes | No | Partial |
| **Product data** (price, rating, reviews) | Yes | No | No |
| **Pricing** | $0.005/credit | $0.01/search | $0.05/search |
| **Amazon marketplace coverage** | 22 countries | -- | -- |
| **LangChain async** | Yes | Yes | Yes |

## What Can You Build?

- **Shopping agents** -- Amazon, Walmart, Target, eBay and Home Depot in one basket-comparison loop, including eBay *sold* prices
- **Product research agents** -- Google reviews + Amazon listings + YouTube reviews + Reddit opinions in one query
- **Real-estate agents** -- Zillow and Redfin listings, price history, Zestimates, market stats and agent reviews
- **Travel agents** -- Booking.com stays priced for real dates, Airbnb listings and review bodies, Tripadvisor rankings, Google Flights and Hotels
- **Local intelligence agents** -- Yelp businesses, reviews and attribute filters behind the same API key
- **Recruiting and employer-brand agents** -- Indeed postings and company reviews, Glassdoor ratings and salary bands, LinkedIn jobs
- **App-store agents** -- App Store and Google Play listings, install counts, versioned review streams
- **Financial and company-research agents** -- SEC EDGAR full-text search, filings and XBRL facts; the UK Companies House register
- **Competitive-intelligence agents** -- G2 and Capterra reviews with facets, plus Google Ads Transparency and the Meta Ad Library for what rivals are actually running
- **Content research agents** -- YouTube trends + Reddit sentiment + Google news in a single workflow
- **Social media agents** -- TikTok, Instagram, X, Threads and Kuaishou profile analytics, hashtag tracking and comment mining
- **Read-anything agents** -- point `ScavioExtract` at any URL the other 186 tools do not cover and get clean Markdown back

## Quick Start

```python
import os
from langchain_scavio import ScavioSearch

os.environ["SCAVIO_API_KEY"] = "sk_live_..."

tool = ScavioSearch()
result = tool.invoke({"query": "best python web frameworks 2026"})
```

## All 187 Tools

Per platform: YouTube 16, Kuaishou 14, Google 12, Instagram 12, Reddit 12,
TikTok 11, X 11, LinkedIn 9, TikTok Shop 8, Walmart 7, SEC EDGAR 6, Threads 6,
Target 4, Indeed 4, Glassdoor 4, Tripadvisor 4, Companies House 4, Amazon 3,
eBay 3, Home Depot 3, Zillow 3, Redfin 3, Booking.com 3, Airbnb 3, Yelp 3,
App Store 3, Google Play 3, G2 3, Capterra 3, Google Ads 3, Meta Ads 3, and
`ScavioExtract` on its own.

> **New in 4.0.** 96 tools -> 187, 10 platforms -> 32. Twenty-one new platforms
> (Threads, Kuaishou, eBay, Target, Home Depot, Zillow, Redfin, Booking.com,
> Airbnb, Tripadvisor, Indeed, Glassdoor, Yelp, Apple App Store, Google Play,
> SEC EDGAR, Companies House, G2, Capterra, Google Ads Transparency, Meta Ad
> Library), plus `ScavioExtract` -- a read-any-URL tool that returns Markdown,
> plain text or raw HTML. **Walmart is the one breaking change:** it goes from
> 2 tools to all 7 endpoints, and `device`, `delivery_zip` and `store_id` are
> **removed** from `ScavioWalmartSearch` and `ScavioWalmartProduct` because the
> endpoint retired them -- sending them now only earns a `warnings[]` entry. In
> exchange `domain` (`com`, `ca`, `com.mx`), `page`, and the `rating_high` and
> `new` sort orders are exposed, and `ScavioWalmartProduct` takes `product_id`
> and nothing else. Five platforms are **lookup-first** and their resolver tool
> is the one to call before anything else: `ScavioTripadvisorLocations`,
> `ScavioGlassdoorCompanies`, `ScavioSECLookup`, `ScavioGoogleAdsAdvertisers`
> and `ScavioCompaniesHouseSearch`.

> **New in 3.4.** Parameter parity: every tool now exposes the full parameter
> set of the endpoint behind it, in the agent-visible schema rather than as a
> constructor attribute an agent cannot vary per call. `ScavioSearch` gained the
> ten remaining Google v2 params (`location`, `uule`, `lr`, `cr`, `safe`,
> `nfpr`, `filter`, `time_period`, `resolve_ai_overview`, `include_html`);
> `ScavioYouTubeSearch` gained `type` plus the `four_k`/`hdr`/`video_360`/
> `video_3d`/`vr180` feature flags; `ScavioGoogleAIMode` gained `include_html`;
> `ScavioRedditPost` accepts `post_id` as well as `url`; the Amazon tools take
> `asin` under its own name plus the deprecated `domain`/`start_page` aliases.
> Nothing was removed -- every pre-3.4 spelling still works and still wins or
> loses exactly as documented.

> **New in 3.3.** Reddit goes from 2 tools to all 12 endpoints: search
> suggestions, post comments, comment replies, subreddit metadata and feed,
> user profile/posts/comments, the popular feed and trending queries. Every
> Reddit endpoint costs 1 credit. `ScavioRedditSearch` and `ScavioRedditPost`
> are unchanged apart from their descriptions, which now state the cost -- so
> every tool in the package states its cost.

> **New in 3.2.** X (11 tools) and LinkedIn (9) are new platforms; Google gained
> its 11 remaining v2 verticals (AI Mode, Maps place/reviews, Shopping, Flights,
> Hotels, Trends, Trending) and YouTube its 8 remaining endpoints (Shorts,
> suggestions, comment replies, related, channel search/shorts/community/resolve).
> Google tools now take the v2 parameters natively -- `gl`, `hl`, `start`,
> `google_domain` -- with `country_code`, `language` and `page` kept as aliases.
> Every tool description now states its credit cost.

| Tool | Description |
|------|-------------|
| `ScavioSearch` | Google web search with knowledge graphs, PAA questions, news |
| `ScavioGoogleAIMode` | Google AI Mode answer with cited references |
| `ScavioGoogleMapsPlace` | Google Maps place details by place_id or data_cid |
| `ScavioGoogleMapsReviews` | Google Maps reviews for a place, with sorting |
| `ScavioGoogleShopping` | Google Shopping listings with price and shipping filters |
| `ScavioGoogleShoppingProduct` | Google Shopping product detail and its sellers |
| `ScavioGoogleShoppingStores` | More sellers for a Google Shopping product |
| `ScavioGoogleFlights` | Google Flights search between two airports |
| `ScavioGoogleHotels` | Google Hotels search for a destination and date range |
| `ScavioGoogleHotelsDetail` | Google Hotels property detail with booking sources |
| `ScavioGoogleTrends` | Google Trends interest over time and by region |
| `ScavioGoogleTrending` | Google Trending Now for a country |
| `ScavioAmazonSearch` | Search Amazon product listings across 22 marketplaces |
| `ScavioAmazonProduct` | Fetch full details for an Amazon product by ASIN |
| `ScavioAmazonOffers` | Every seller offer for an ASIN: price, seller, condition, buy box |
| `ScavioWalmartSearch` | Search Walmart, com/ca/com.mx, with price and fulfillment filters |
| `ScavioWalmartProduct` | Full Walmart product detail by item id |
| `ScavioWalmartReviews` | Walmart customer reviews with the rating breakdown, 10 per page |
| `ScavioWalmartCategory` | Products inside a Walmart category, same shape as search |
| `ScavioWalmartOffers` | Walmart buy-box seller for a product (not the full offer list) |
| `ScavioWalmartSeller` | Walmart marketplace seller storefront and Pro Seller badge |
| `ScavioWalmartSellerProducts` | A Walmart seller's catalog, ~40 server-rendered items |
| `ScavioTargetSearch` | Search Target with per-store prices and availability |
| `ScavioTargetCategory` | Products in a Target category, same shape as search |
| `ScavioTargetProduct` | Target product detail by TCIN, with variants and fulfillment |
| `ScavioTargetReviews` | Target reviews, 8 bodies max, with per-attribute averages |
| `ScavioEbaySearch` | Search eBay live or SOLD listings, filters and 60/120/240 per page |
| `ScavioEbayProduct` | One eBay listing in full, including auction state |
| `ScavioEbaySeller` | eBay seller profile card (a profile, not a catalogue) |
| `ScavioHomeDepotSearch` | Search Home Depot, 12 products per page (fixed) |
| `ScavioHomeDepotProduct` | Home Depot item detail with the full spec table |
| `ScavioHomeDepotReviews` | Home Depot review bodies, 30 per page |
| `ScavioYouTubeSearch` | Search YouTube videos with duration/date/type/feature filters |
| `ScavioYouTubeShorts` | Search YouTube Shorts with sorting and pagination |
| `ScavioYouTubeSuggestions` | YouTube search autocomplete for keyword expansion |
| `ScavioYouTubeMetadata` | Deprecated alias of `ScavioYouTubeVideo` |
| `ScavioYouTubeVideo` | Fetch full details for a YouTube video (chapters, captions) |
| `ScavioYouTubeComments` | Fetch comments on a YouTube video with pagination |
| `ScavioYouTubeCommentReplies` | Fetch replies to a specific YouTube comment |
| `ScavioYouTubeTranscript` | 8 |
| `ScavioYouTubeRelated` | Fetch videos related to a YouTube video |
| `ScavioYouTubeChannelSearch` | Search YouTube channels by name |
| `ScavioYouTubeChannel` | Fetch channel details by ID, @handle, or URL |
| `ScavioYouTubeChannelVideos` | Fetch a YouTube channel's uploaded videos |
| `ScavioYouTubeChannelShorts` | Fetch a YouTube channel's Shorts |
| `ScavioYouTubeChannelCommunity` | Fetch a YouTube channel's community posts |
| `ScavioYouTubeChannelResolve` | Resolve an @handle or URL to a channel ID |
| `ScavioYouTubeStreams` | 3 |
| `ScavioRedditSearch` | Search Reddit posts with cursor pagination |
| `ScavioRedditSearchSuggestions` | Reddit search autocomplete for query expansion |
| `ScavioRedditPost` | Fetch a Reddit post's metadata by URL or post_id (no comments) |
| `ScavioRedditPostComments` | Top-level comments on a Reddit post, with sorting |
| `ScavioRedditCommentReplies` | Replies to one comment (needs its `reply_cursor`) |
| `ScavioRedditSubreddit` | Subreddit metadata: subscribers, description, icon |
| `ScavioRedditSubredditPosts` | A subreddit's post feed (the only `RISING` sort) |
| `ScavioRedditUser` | A redditor's profile: karma breakdown, avatar, bio |
| `ScavioRedditUserPosts` | A redditor's submitted posts with sorting |
| `ScavioRedditUserComments` | A redditor's comments, each with its parent post |
| `ScavioRedditPopular` | The site-wide r/popular feed (cursor only) |
| `ScavioRedditTrending` | Reddit search queries trending right now |
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
| `ScavioInstagramUserPosts` | 2 |
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
| `ScavioXSearch` | Search X (Twitter) tweets and people, Top/Latest/People/Photos/Videos |
| `ScavioXTweet` | Fetch a single tweet with engagement counts and reply context |
| `ScavioXTweetComments` | Fetch replies to a tweet, ranked or chronological |
| `ScavioXTweetRetweeters` | Fetch the users who retweeted a tweet |
| `ScavioXUser` | Fetch an X profile by handle |
| `ScavioXUserTweets` | Fetch a user's tweets, plus their pinned tweet |
| `ScavioXUserReplies` | Fetch a user's tweets and replies |
| `ScavioXUserMedia` | Fetch a user's media tweets with direct photo/video URLs |
| `ScavioXUserFollowers` | Fetch an X user's followers |
| `ScavioXUserFollowings` | Fetch accounts an X user follows |
| `ScavioXTrending` | Fetch trending topics on X for a country |
| `ScavioThreadsProfile` | Threads profile by user_id (2cr) or username (4cr) |
| `ScavioThreadsUserPosts` | A Threads user's posts, cursor-paginated |
| `ScavioThreadsUserReplies` | A Threads user's replies, cursor-paginated |
| `ScavioThreadsPost` | One Threads post by id or threads.net URL |
| `ScavioThreadsPostComments` | Replies to a Threads post, cursor-paginated |
| `ScavioThreadsSearchUsers` | Threads people search (the only search Threads exposes) |
| `ScavioKuaishouProfile` | Kuaishou profile by user id |
| `ScavioKuaishouUserPosts` | A Kuaishou user's top posts, cursor-paginated |
| `ScavioKuaishouUserLive` | A Kuaishou user's current live-stream status |
| `ScavioKuaishouUserResolve` | Turn a Kuaishou share link into a user id |
| `ScavioKuaishouVideo` | One Kuaishou video by photo id or URL |
| `ScavioKuaishouVideoComments` | Comments on a Kuaishou video, cursor-paginated |
| `ScavioKuaishouCommentReplies` | Replies under a Kuaishou root comment |
| `ScavioKuaishouVideosBatch` | Up to 20 Kuaishou videos in one call (40 credits) |
| `ScavioKuaishouSearch` | Mixed-result Kuaishou search |
| `ScavioKuaishouSearchVideos` | Kuaishou video search |
| `ScavioKuaishouSearchUsers` | Kuaishou user search |
| `ScavioKuaishouSearchLive` | Kuaishou live-stream search |
| `ScavioKuaishouTagFeed` | Kuaishou hashtag feed, cursor-paginated |
| `ScavioKuaishouTrending` | Kuaishou hot / live / shopping / brand / music leaderboards |
| `ScavioLinkedInPerson` | Full LinkedIn member profile with experience and education |
| `ScavioLinkedInPersonAbout` | About/overview section of a LinkedIn member |
| `ScavioLinkedInPersonPosts` | A member's posts, comments, or reactions feed |
| `ScavioLinkedInCompany` | LinkedIn company profile with locations and specialties |
| `ScavioLinkedInCompanyPosts` | A company's recent LinkedIn posts |
| `ScavioLinkedInSearchJobs` | Search LinkedIn job listings by keyword and location |
| `ScavioLinkedInJob` | 30 |
| `ScavioLinkedInPost` | A single LinkedIn post with its top visible comments |
| `ScavioLinkedInPostComments` | Comments on a LinkedIn post, with replies |
| `ScavioIndeedSearch` | Indeed job postings, 10 per page, location-only search allowed |
| `ScavioIndeedJob` | One Indeed posting in full, with the original ATS link |
| `ScavioIndeedCompany` | Indeed employer profile with ratings and reported salaries |
| `ScavioIndeedCompanyReviews` | Indeed employee reviews, 20 per page, with sentiment |
| `ScavioGlassdoorCompanies` | START HERE: resolve a company NAME to a Glassdoor employer_id |
| `ScavioGlassdoorCompany` | Glassdoor employer profile, plus reviews_url and salaries_url |
| `ScavioGlassdoorReviews` | Up to THREE Glassdoor reviews plus full rating statistics |
| `ScavioGlassdoorSalaries` | Glassdoor salary estimates by job title, 10 per page |
| `ScavioZillowSearch` | Zillow for-sale / for-rent / sold listings with 25+ filters |
| `ScavioZillowProperty` | Zillow listing detail, price and tax history, Zestimate, schools |
| `ScavioZillowAgentReviews` | A Zillow AGENT's profile and reviews (not a property) |
| `ScavioRedfinSearch` | Redfin listings, up to 350 per page, region URL or region id |
| `ScavioRedfinProperty` | Redfin listing detail with MLS facts, history and comps |
| `ScavioRedfinMarket` | Redfin housing-market stats for a region |
| `ScavioBookingSearch` | Booking.com properties priced for a real stay |
| `ScavioBookingHotel` | One Booking.com property with rooms, rate plans and policies |
| `ScavioBookingReviews` | Booking.com guest reviews with the category score breakdown |
| `ScavioAirbnbSearch` | Airbnb stays with the full price and discount ledger |
| `ScavioAirbnbListing` | One Airbnb listing with amenities, host, rules and photos |
| `ScavioAirbnbReviews` | Airbnb review BODIES with limit/offset paging |
| `ScavioTripadvisorLocations` | START HERE: resolve a place NAME to TripAdvisor ids |
| `ScavioTripadvisorSearch` | Restaurants / hotels / attractions ranked inside a geo |
| `ScavioTripadvisorLocation` | One TripAdvisor location plus its first page of reviews |
| `ScavioTripadvisorReviews` | TripAdvisor reviews, used to page PAST the location page |
| `ScavioYelpSearch` | Yelp businesses in ranked order, 10 per page |
| `ScavioYelpBusiness` | One Yelp business in full, plus its first page of reviews |
| `ScavioYelpReviews` | Yelp reviews -- start at page 2, page 1 duplicates the business |
| `ScavioAppStoreSearch` | Up to 200 App Store apps in one call, no pagination |
| `ScavioAppStoreApp` | App Store listing by numeric id or bundle id |
| `ScavioAppStoreReviews` | App Store reviews, 50 per page, hard stop at page 10 |
| `ScavioGooglePlaySearch` | Google Play app search, one shelf of ~30 apps |
| `ScavioGooglePlayApp` | Google Play listing including the real unrendered install count |
| `ScavioGooglePlayReviews` | Google Play reviews via an opaque single-use cursor |
| `ScavioSECLookup` | START HERE: resolve a ticker or name to an SEC CIK |
| `ScavioSECCompany` | SEC filer profile: SIC, EIN, addresses, tickers, filing habits |
| `ScavioSECFilings` | SEC filings with direct document links, optional archive history |
| `ScavioSECConcept` | Every value a filer reported for one XBRL concept |
| `ScavioSECFacts` | The index of every XBRL concept a filer reports |
| `ScavioSECSearch` | EDGAR full-text search from 2001, with facets |
| `ScavioCompaniesHouseSearch` | START HERE: search the UK register, current and former names |
| `ScavioCompaniesHouseCompany` | Full Companies House register entry |
| `ScavioCompaniesHouseOfficers` | Companies House officers, current and resigned, 35 per page |
| `ScavioCompaniesHouseFilingHistory` | Companies House filing history with PDF links |
| `ScavioG2Search` | Search G2 software products (5 credits, the dearest platform) |
| `ScavioG2Product` | G2 product profile, pricing editions and comparisons (no reviews) |
| `ScavioG2Reviews` | G2 reviews with exact per-star counts and full facets |
| `ScavioCapterraSearch` | Capterra product search, a fixed 20 rows, no pagination |
| `ScavioCapterraProduct` | Capterra profile with the complete pricing table |
| `ScavioCapterraReviews` | Capterra reviews, 25 per page, with a rich competitor list |
| `ScavioGoogleAdsAdvertisers` | START HERE: resolve a name or domain to a Google advertiser id |
| `ScavioGoogleAdsSearch` | Every ad Google runs for one advertiser or domain, cursor-paged |
| `ScavioGoogleAdsCreative` | One Google creative with its per-region impression history |
| `ScavioMetaAdsSearch` | Search the Meta Ad Library, fully cursor-paginated |
| `ScavioMetaAdsAdvertiser` | Every ad a Facebook Page is running, cursor-paginated |
| `ScavioMetaAdsAd` | One Meta ad in full by archive id |
| `ScavioExtract` | Read ANY URL as raw HTML, readability Markdown or plain text |

## Use with a LangChain Agent

Scavio tools plug into the current [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) API from `langchain.agents`:

```python
from langchain.agents import create_agent
from langchain_scavio import (
    ScavioSearch, ScavioExtract,
    ScavioAmazonSearch, ScavioAmazonProduct,
    ScavioWalmartSearch, ScavioTargetSearch, ScavioEbaySearch,
    ScavioYouTubeSearch, ScavioYouTubeVideo, ScavioYouTubeTranscript,
    ScavioRedditSearch, ScavioRedditPost,
)

agent = create_agent(
    "openai:gpt-5.5",
    tools=[
        ScavioSearch(max_results=5),
        ScavioExtract(),
        ScavioAmazonSearch(max_results=5),
        ScavioAmazonProduct(),
        ScavioWalmartSearch(max_results=5),
        ScavioTargetSearch(),
        ScavioEbaySearch(),
        ScavioYouTubeSearch(max_results=5),
        ScavioYouTubeVideo(),
        ScavioYouTubeTranscript(),
        ScavioRedditSearch(max_results=5),
        ScavioRedditPost(),
    ],
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "Find me a Python book on Amazon under $30"}]
})
```

With 187 tools available, hand a given agent only the ones its job needs -- a
price-comparison agent wants the retail tools, a recruiting agent wants Indeed,
Glassdoor and LinkedIn. `ScavioExtract` is the useful catch-all: it reads any
URL the other tools do not cover.

```python
# A lookup-first platform: resolve the id, then use it
from langchain.agents import create_agent
from langchain_scavio import (
    ScavioGlassdoorCompanies, ScavioGlassdoorCompany,
    ScavioGlassdoorReviews, ScavioGlassdoorSalaries,
    ScavioIndeedSearch, ScavioIndeedCompanyReviews,
)

recruiting_agent = create_agent(
    "openai:gpt-5.5",
    tools=[
        ScavioGlassdoorCompanies(),   # call this FIRST -- it returns employer_id
        ScavioGlassdoorCompany(),
        ScavioGlassdoorReviews(),
        ScavioGlassdoorSalaries(),
        ScavioIndeedSearch(),
        ScavioIndeedCompanyReviews(),
    ],
)
```

## Async Support

All tools support async invocation:

```python
result = await tool.ainvoke({"query": "async python frameworks"})
```

## Configuration

### Google Search

Every Google tool targets the v2 API (`/api/v2/google*`) and takes the v2 wire
parameters directly: `gl`, `hl`, `start`, `google_domain` and `device`. Google
v1 was retired on 2026-08-04 and now returns HTTP 410.

`start` is a **0-based result offset, not a page number**: 0 is the first page,
10 the second, 20 the third.

```python
from langchain_scavio import ScavioSearch

tool = ScavioSearch(
    scavio_api_key="sk_live_...",       # or SCAVIO_API_KEY env var
    max_results=5,
    light_request=None,                  # deprecated, ignored (v2 always full, 1 credit)
    include_knowledge_graph=True,
    include_questions=True,
    include_related=False,
    gl="us",                             # native v2 country
    hl="en",                             # native v2 UI language
    search_type="classic",               # classic|news|maps
    device="desktop",
)

result = tool.invoke({"query": "vector databases", "start": 10})  # page 2
```

`country_code`, `language` and `page` still work as pre-3.2 aliases of `gl`,
`hl` and `start`, but the native names win when both are supplied.

### Google verticals

Eleven more Google surfaces, 1 credit each. All of them return a **flat**
response -- there is no `data` wrapper.

```python
from langchain_scavio import (
    ScavioGoogleAIMode,
    ScavioGoogleFlights,
    ScavioGoogleHotels,
    ScavioGoogleHotelsDetail,
    ScavioGoogleMapsPlace,
    ScavioGoogleMapsReviews,
    ScavioGoogleShopping,
    ScavioGoogleShoppingProduct,
    ScavioGoogleShoppingStores,
    ScavioGoogleTrending,
    ScavioGoogleTrends,
)

# AI Mode: a synthesised answer with citations
result = ScavioGoogleAIMode().invoke({"query": "how to cache LLM responses"})
# result["text_blocks"] + result["references"]

# Maps: place details and reviews (search lives on ScavioSearch search_type="maps")
result = ScavioGoogleMapsPlace().invoke({"place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4"})
result = ScavioGoogleMapsReviews(max_results=10).invoke({
    "place_id": "ChIJN1t_tDeuEmsRUsoyG83frY4",
    "sort_by": "newest",                 # relevance|newest|highest_rating|lowest_rating
})

# Shopping: sort_by is a NUMBER here, and start is an offset
result = ScavioGoogleShopping(max_results=10).invoke({
    "query": "mechanical keyboard",
    "max_price": 150,
    "sort_by": 1,                        # 0 relevance, 1 price asc, 2 price desc
    "start": 60,
})
# ... but a STRING enum on the product endpoint
result = ScavioGoogleShoppingProduct().invoke({
    "catalog_id": "1234567890",
    "query": "mechanical keyboard",      # required whenever catalog_id is set
    "sort_by": "total_price",
})
result = ScavioGoogleShoppingStores().invoke({
    "catalog_id": "1234567890",
    "next_page_token": "...",            # from the product response
})

# Travel
result = ScavioGoogleFlights().invoke({
    "departure_id": "JFK",
    "arrival_id": "LHR",
    "outbound_date": "2026-09-01",
    "type": 2,                           # 1 round trip (needs return_date), 2 one way
})
hotels = ScavioGoogleHotels(max_results=10).invoke({
    "query": "Lisbon hotels",
    "check_in_date": "2026-09-01",
    "check_out_date": "2026-09-04",
})
# feed a property's detail_token back in -- and re-send both dates
result = ScavioGoogleHotelsDetail().invoke({
    "detail_token": hotels["properties"][0]["detail_token"],
    "check_in_date": "2026-09-01",
    "check_out_date": "2026-09-04",
})

# Trends uses an UPPERCASE geo, not gl; Trending has no query at all
result = ScavioGoogleTrends().invoke({
    "query": "langchain,llamaindex",     # comma-separate to compare terms
    "geo": "US",
    "date": "today 12-m",
})
result = ScavioGoogleTrending(max_results=10).invoke({"geo": "US", "hours": 24})
```

### Amazon

```python
from langchain_scavio import ScavioAmazonSearch, ScavioAmazonProduct, ScavioAmazonOffers

search = ScavioAmazonSearch(max_results=5)
search.invoke({"query": "wireless headphones", "country": "us", "page": 1})

product = ScavioAmazonProduct()
product.invoke({"asin": "B08N5WRWNW"})            # `query` still accepted

offers = ScavioAmazonOffers()
offers.invoke({"asin": "B08N5WRWNW"})             # every seller for that ASIN
```

> **Targeting a marketplace:** `country` takes a two-letter code, not a domain. Supported: `us` (default), `gb` (the UK is `gb`, not `uk`), `ca`, `de`, `fr`, `es`, `it`, `jp`, `in`, `au`, `br`, `mx`, `nl`, `pl`, `se`, `sg`, `ae`, `sa`, `eg`, `cn`, `be`, `tr`. An unrecognised code falls back to `us`.

> **Amazon changed in 3.0 (breaking).** The upstream provider moved and the request surface shrank. `sort_by`, `pages`, `category_id`, `merchant_id`, `language`, `currency`, `device`, `zip_code` and `autoselect_variant` are gone from all Amazon tools -- the marketplace ignores every one of them, so they are removed rather than kept as silent no-ops (`sort_by` was verified: all six sort values return the identical unordered set). `domain` and `start_page` are deprecated wire aliases: they still work and, since 3.4, are declared on the schemas so nothing the endpoint accepts is unreachable -- but prefer `country` and `page`, which win when both are given. Response fields were renamed too -- `url_image` is now `image`, `best_seller`/`is_amazons_choice` collapsed into `badge`, and `buybox` is gone (use `ScavioAmazonOffers`).

### Walmart

```python
from langchain_scavio import (
    ScavioWalmartSearch, ScavioWalmartProduct, ScavioWalmartReviews,
    ScavioWalmartCategory, ScavioWalmartOffers, ScavioWalmartSeller,
    ScavioWalmartSellerProducts,
)

search = ScavioWalmartSearch(max_results=5)
result = search.invoke({
    "query": "air fryer",
    "sort_by": "price_low",       # best_match|price_low|price_high|best_seller|rating_high|new
    "max_price": 50,
    "fulfillment_speed": "tomorrow",   # today|tomorrow only
    "domain": "com",                   # com|ca = 1 credit, com.mx = 2 credits
    "page": 2,
})

# Product detail takes product_id and nothing else
result = ScavioWalmartProduct().invoke({"product_id": "13544111159"})

# Reviews page 10 at a time
result = ScavioWalmartReviews().invoke({"product_id": "13544111159", "page": 2})

# Category browse -- same product shape as search; `limit` TRIMS and does not
# reduce the credit cost
result = ScavioWalmartCategory().invoke({"category_id": "3944_133251_1095191"})

# Offers is the BUY-BOX seller only, not the full offer list
result = ScavioWalmartOffers().invoke({"product_id": "13544111159"})

# seller_id must be the NUMERIC catalog seller id (seller_catalog_id).
# The GUID form of seller_id returns 404.
result = ScavioWalmartSeller().invoke({"seller_id": "101040442"})
result = ScavioWalmartSellerProducts().invoke({"seller_id": "101040442"})
```

> **Breaking in 4.0.** `device`, `delivery_zip` and `store_id` were retired
> upstream and are gone from the schemas. `domain` was *not* retired -- it is
> live on `search` and `category` only (walmart.ca product pages cannot be
> fetched at all) and it is the price-bearing parameter: `com` and `ca` cost 1
> credit, `com.mx` costs 2. `fulfillment_speed` no longer offers `2_days` (it
> leaks 3-4 day items) or `anytime` (a no-op -- omit the parameter instead).

### Target, eBay and Home Depot

```python
from langchain_scavio import (
    ScavioTargetSearch, ScavioTargetProduct, ScavioTargetReviews,
    ScavioEbaySearch, ScavioEbayProduct, ScavioEbaySeller,
    ScavioHomeDepotSearch, ScavioHomeDepotReviews,
)

# Target: store_id IS a real parameter here (unlike Walmart) -- it decides
# price and availability. Rendered pages are slow: search ~9s, category ~37s.
ScavioTargetSearch().invoke({"keyword": "office chair", "store_id": "3991"})
ScavioTargetProduct().invoke({"tcin": "87095665"})
# 8 review bodies maximum, no paging -- `limit` only trims
ScavioTargetReviews().invoke({"tcin": "87095665"})

# eBay: `sold=True` searches completed listings that actually sold -- the
# price-research view. total_results is null there because eBay publishes no
# headline count. per_page accepts ONLY 60, 120 or 240.
ScavioEbaySearch().invoke({"query": "airpods pro", "sold": True, "per_page": 120})
# A seller-scoped search needs no query and is the only way to page a catalogue
ScavioEbaySearch().invoke({"seller": "musicmagpie", "page": 2})
ScavioEbaySeller().invoke({"seller": "musicmagpie"})   # a profile, not a catalogue
ScavioEbayProduct().invoke({"item_id": "126544332211"})

# Home Depot: 12 products per page, fixed. Reviews are 30 per page and asking
# past total_pages is a 404. sort_by is closed -- an unknown sort returns an
# empty page that still bills.
ScavioHomeDepotSearch().invoke({"query": "cordless drill", "sort_by": "top_rated"})
ScavioHomeDepotReviews().invoke({"item_id": "313021355", "page": 2})
```

### Zillow and Redfin

```python
from langchain_scavio import (
    ScavioZillowSearch, ScavioZillowProperty, ScavioZillowAgentReviews,
    ScavioRedfinSearch, ScavioRedfinProperty, ScavioRedfinMarket,
)

# Zillow: a bare ZIP works alone but CANNOT be combined with a filter or a sort
# -- Zillow resolves it by geolocation on that request shape and answers about
# another city. Use the city name there.
ScavioZillowSearch().invoke({
    "location": "Austin, TX",
    "listing_status": "for_rent",
    "max_price": 3000,          # MONTHLY RENT on for_rent, not a sale price
    "beds_min": 2,
})
ScavioZillowProperty().invoke({"zpid": "29444874"})
# /reviews addresses an AGENT profile, not a property
ScavioZillowAgentReviews().invoke({"screen_name": "jane-smith"})

# Redfin: city NAMES are not accepted. Pass a redfin.com region URL or the
# region_id + region_type pair (region_id is NOT a ZIP code).
ScavioRedfinSearch().invoke({
    "location": "https://www.redfin.com/city/30749/TX/Austin",
    "listing_status": "sold",
    "sold_within_days": 90,     # only valid with listing_status=sold
    "limit": 350,
})
ScavioRedfinMarket().invoke({"region_id": 30749, "region_type": 6})
ScavioRedfinProperty().invoke({"property_id": "170072526"})
```

### Booking.com, Airbnb and Tripadvisor

```python
from langchain_scavio import (
    ScavioBookingSearch, ScavioBookingHotel, ScavioBookingReviews,
    ScavioAirbnbSearch, ScavioAirbnbListing, ScavioAirbnbReviews,
    ScavioTripadvisorLocations, ScavioTripadvisorSearch,
    ScavioTripadvisorLocation, ScavioTripadvisorReviews,
)

# Booking prices a STAY: checkin and checkout must be sent TOGETHER, or
# Booking prices its own default range and returns real prices for dates you
# never asked for. Chaining the `url` a search row returns beats a bare slug.
ScavioBookingSearch().invoke({
    "destination": "Lisbon",
    "checkin": "2026-09-10", "checkout": "2026-09-13",
    "adults": 2, "currency": "EUR", "min_review_score": "8",
})
ScavioBookingHotel().invoke({"hotel": "memmo-alfama", "country_code": "pt"})

# Airbnb: dates matter for the same reason. A dateless search defaults to
# +30d / 5 nights and A/Bs both the window and the prices; the response flags
# that as dates_are_defaulted. The listing endpoint has NO price field --
# prices are search-only.
ScavioAirbnbSearch().invoke({
    "location": "Lisbon", "check_in": "2026-09-10", "check_out": "2026-09-15",
    "room_type": "entire_home", "amenities": "wifi,kitchen",
})
# Always set limit -- upstream returns a fixed 7 reviews without it
ScavioAirbnbReviews().invoke({"listing_id": "12345678", "limit": 30, "offset": 30})

# Tripadvisor is LOOKUP-FIRST: everything else is keyed by ids that only exist
# inside Tripadvisor URLs.
found = ScavioTripadvisorLocations().invoke({"query": "Franklin Barbecue"})
ScavioTripadvisorSearch().invoke({"geo_id": "30196", "category": "restaurants"})
# Page 1 of reviews already rides along with /location -- use /reviews to page
# past it, and de-duplicate on review_id across page boundaries.
ScavioTripadvisorReviews().invoke({
    "location_id": "1899234", "geo_id": "30196",
    "category": "restaurants", "page": 2,
})
```

### Yelp

```python
from langchain_scavio import ScavioYelpSearch, ScavioYelpBusiness, ScavioYelpReviews

# `location` is effectively required: without it Yelp geolocates off our proxy
# exit and the same request answers about a different metro run to run.
ScavioYelpSearch().invoke({
    "term": "coffee", "location": "Austin, TX",
    "price": [1, 2], "open_now": True,
    "attributes": ["RestaurantsDelivery"],   # raw Yelp aliases, passthrough
})
ScavioYelpBusiness().invoke({"business_id": "desnudo-coffee-austin-2"})
# Page 1 re-fetches the document /business already returned and costs another
# 2 credits -- start at page 2.
ScavioYelpReviews().invoke({"business_id": "desnudo-coffee-austin-2", "page": 2})
```

### Indeed and Glassdoor

```python
from langchain_scavio import (
    ScavioIndeedSearch, ScavioIndeedJob, ScavioIndeedCompany,
    ScavioIndeedCompanyReviews,
    ScavioGlassdoorCompanies, ScavioGlassdoorCompany,
    ScavioGlassdoorReviews, ScavioGlassdoorSalaries,
)

# radius and max_age_days are CLOSED sets: Indeed ignores anything else and
# returns the unfiltered set, so "7 miles" would silently buy a 50-mile search.
# min_salary filters on Indeed's own ESTIMATE, so postings with no published
# salary still match. A location-only search (no query) is valid.
ScavioIndeedSearch().invoke({
    "query": "data engineer", "location": "Austin, TX",
    "radius": 25, "max_age_days": 7, "job_type": "full_time",
})
ScavioIndeedCompanyReviews().invoke({"company": "Stripe", "page": 2})

# Glassdoor is LOOKUP-FIRST, and addressing reviews/salaries by employer_id
# costs TWO upstream fetches. Pass back reviews_url / salaries_url from the
# company response as `url` to make it one.
match = ScavioGlassdoorCompanies().invoke({"query": "Stripe"})
company = ScavioGlassdoorCompany().invoke({"employer_id": "1699"})
ScavioGlassdoorReviews().invoke({
    "url": company["data"]["reviews_url"],
    "category": "work_life_balance",
})
ScavioGlassdoorSalaries().invoke({"url": company["data"]["salaries_url"], "page": 2})
```

> Glassdoor reviews are capped at **three per response** by Glassdoor's login
> wall, which is why there is deliberately no `page` parameter there. Move the
> window with `category` and `employment_status`, and read
> `filtered_review_count` to see how many match.

### Apple App Store and Google Play

```python
from langchain_scavio import (
    ScavioAppStoreSearch, ScavioAppStoreApp, ScavioAppStoreReviews,
    ScavioGooglePlaySearch, ScavioGooglePlayApp, ScavioGooglePlayReviews,
)

# App Store search has NO pagination: `limit` (1-200) is the only lever, and a
# search doubles as a bulk metadata fetch -- the same 43-field row as /app.
ScavioAppStoreSearch().invoke({"term": "notion", "limit": 200, "country": "gb"})
# /app takes a numeric id OR a bundle id; /reviews is NUMERIC ONLY.
ScavioAppStoreApp().invoke({"app_id": "com.burbn.instagram"})
# 50 reviews per page, hard stop at page 10 -- reach further via `country`.
ScavioAppStoreReviews().invoke({"app_id": "1232780281", "page": 2, "sort": "most_helpful"})

# Google Play search returns one shelf of ~30 apps with no pagination either.
ScavioGooglePlaySearch().invoke({"query": "notion", "hl": "pt-BR", "gl": "br"})
# /app carries the REAL install count Play publishes but never renders.
ScavioGooglePlayApp().invoke({"app_id": "notion.id"})
# The reviews cursor is OPAQUE and SINGLE-USE and encodes the sort -- send it
# back with the SAME sort it came from.
page = ScavioGooglePlayReviews().invoke({"app_id": "notion.id", "sort": "newest"})
ScavioGooglePlayReviews().invoke({
    "app_id": "notion.id", "sort": "newest", "cursor": page["data"]["next_cursor"],
})
```

### SEC EDGAR and Companies House

```python
from langchain_scavio import (
    ScavioSECLookup, ScavioSECCompany, ScavioSECFilings,
    ScavioSECConcept, ScavioSECFacts, ScavioSECSearch,
    ScavioCompaniesHouseSearch, ScavioCompaniesHouseCompany,
    ScavioCompaniesHouseOfficers, ScavioCompaniesHouseFilingHistory,
)

# EDGAR is keyed by CIK; callers hold a ticker. Start with the lookup.
ScavioSECLookup().invoke({"query": "AAPL", "exchange": "NASDAQ"})
# `form` matches the form AND its root form, so 10-K also returns 10-K/A
ScavioSECFilings().invoke({"ticker": "AAPL", "form": ["10-K", "10-Q"], "limit": 100})
# XBRL concept tags are CASE-SENSITIVE -- list what a filer reports first
ScavioSECFacts().invoke({"ticker": "AAPL", "query": "revenue"})
ScavioSECConcept().invoke({"ticker": "AAPL", "concept": "NetIncomeLoss", "unit": "USD"})
# Full-text search covers 2001-today and accepts NO query at all
ScavioSECSearch().invoke({"form": "8-K", "date_from": "2026-01-01", "sort": "newest"})

# Companies House is lookup-first too, and the company number is normalised
# for you (padded and upper-cased), so numbers off a letterhead still resolve.
ScavioCompaniesHouseSearch().invoke({"query": "Monzo", "page": 1})
ScavioCompaniesHouseOfficers().invoke({"company_number": "09446231"})
ScavioCompaniesHouseFilingHistory().invoke({"company_number": "09446231", "page": 2})
```

### G2 and Capterra

```python
from langchain_scavio import (
    ScavioG2Search, ScavioG2Product, ScavioG2Reviews,
    ScavioCapterraSearch, ScavioCapterraProduct, ScavioCapterraReviews,
)

# G2 is the only 5-credit platform, and a bot wall arrives as a billed 502 --
# budget accordingly and do not loop blindly.
ScavioG2Search().invoke({"query": "project management", "limit": 50, "rating": 4})
# The G2 profile carries NO review text -- call /reviews for it, and only
# /reviews has exact per-star counts and the company-size / role / industry
# facets. rating buckets are half-star-inclusive.
ScavioG2Reviews().invoke({
    "product_id": "notion", "company_size": "enterprise",
    "region": "north_america", "page": 2,
})

# Capterra search does NOT paginate: a fixed 20 rows. `slug` is cosmetic on
# /product but LOAD-BEARING on /reviews -- a wrong one silently serves page one
# under a billed 200, so pass back the slug or reviews_url you were given.
ScavioCapterraSearch().invoke({"query": "project management"})
ScavioCapterraReviews().invoke({"product_id": "186596", "slug": "Notion", "page": 2})
```

### Google Ads Transparency and the Meta Ad Library

```python
from langchain_scavio import (
    ScavioGoogleAdsSearch, ScavioGoogleAdsAdvertisers, ScavioGoogleAdsCreative,
    ScavioMetaAdsSearch, ScavioMetaAdsAdvertiser, ScavioMetaAdsAd,
)

# Lookup first: /advertisers resolves a name or domain to an advertiser id.
ScavioGoogleAdsAdvertisers().invoke({"query": "Stripe", "region": "GB"})
# Querying by `domain` is the ONLY way to get the `domain` field back per row.
# The text / image / video format sets are DISJOINT.
page = ScavioGoogleAdsSearch().invoke({
    "domain": "stripe.com", "region": "GB", "format": "image", "limit": 100,
})
# Re-send the SAME filters alongside the cursor
ScavioGoogleAdsSearch().invoke({
    "domain": "stripe.com", "region": "GB", "format": "image",
    "cursor": page["data"]["next_cursor"],
})
# Impressions and reach are EEA-ONLY (DSA-compelled): US creatives return null.
ScavioGoogleAdsCreative().invoke({
    "advertiser_id": "AR16735076323512287233", "creative_id": "CR12345678901234567890",
})

# Meta: page 1 is 30 ads, then 10 per page off next_cursor. The cursor is a
# self-contained blob, so THE OTHER FILTERS ARE IGNORED when one is present.
first = ScavioMetaAdsSearch().invoke({
    "query": "running shoes", "country": "GB",
    "ad_type": "political_and_issue_ads",   # exposes spend / reach / impressions
})
while first["data"]["has_next_page"]:
    first = ScavioMetaAdsSearch().invoke({"query": "running shoes",
                                          "cursor": first["data"]["next_cursor"]})
ScavioMetaAdsAdvertiser().invoke({"page_id": "20531316728"})
```

> `total_results` on the Meta Ad Library caps at 50000 with
> `total_is_capped: true` -- Meta only ever reports ">50,000". Each page costs
> 1 credit, so "scrape the whole library" is real but its cost scales with
> depth (roughly 10 ads per credit past the first 30).

### Threads and Kuaishou

```python
from langchain_scavio import (
    ScavioThreadsProfile, ScavioThreadsUserPosts, ScavioThreadsSearchUsers,
    ScavioKuaishouProfile, ScavioKuaishouUserResolve, ScavioKuaishouVideo,
    ScavioKuaishouVideosBatch, ScavioKuaishouTrending,
)

# Threads: user_id is the CHEAP path (2 credits). A username costs 4, because
# the handle lookup needs a second upstream call.
ScavioThreadsProfile().invoke({"user_id": "63625256886"})
ScavioThreadsUserPosts().invoke({"user_id": "63625256886", "cursor": "..."})
# There is NO Threads content search -- people search is the only search.
ScavioThreadsSearchUsers().invoke({"query": "langchain"})

# Kuaishou (China) is priced PER ENDPOINT: 1, 2, 10 or 40 credits.
ScavioKuaishouUserResolve().invoke({"share_link": "https://v.kuaishou.com/abc123"})
ScavioKuaishouProfile().invoke({"user_id": "3xnmvnpnyzqxqzm"})      # 10 credits
ScavioKuaishouVideo().invoke({"photo_id": "3xf8v9pmcvexbhi"})       # 2 credits
ScavioKuaishouVideosBatch().invoke({"photo_ids": ["a", "b"]})       # 40 credits, max 20 ids
ScavioKuaishouTrending().invoke({"board": "shopping"})              # 1 credit
```

> This is **Kuaishou (China)**, never "Kwai". Kwai international (kwai.com) is
> not served: a real kwai.com photo id returns an empty envelope, and
> `ScavioKuaishouUserResolve` rejects kwai.com links.

### Extract (any URL)

`ScavioExtract` is the read-a-page primitive: point it at any URL the other 186
tools do not cover.

```python
from langchain_scavio import ScavioExtract

extract = ScavioExtract()

# Default: readability Markdown, 1 credit
result = extract.invoke({"url": "https://example.com/article"})
result["data"]["content"], result["data"]["content_length"]

# format: html (raw page) | markdown (readability) | text (markdown flattened)
extract.invoke({"url": "example.com", "format": "text"})   # bare host -> https

# mode is the PRICE-BEARING parameter. Escalate only when a plain fetch comes
# back empty: normal 1 credit, advanced 1 credit (renders JS), ultra 2 credits.
extract.invoke({"url": "https://example.com/spa", "mode": "advanced"})
```

> Billing is on a **successful** extraction only -- a dead link, bot wall or
> timeout costs nothing. `http(s)` only; loopback, private, link-local and
> metadata hosts are rejected with a 400.

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
    "type": "video",                     # video|channel|playlist|movie
    "features": ["hd", "subtitles"],     # hd|4k|subtitles|creative_commons|live|360|3d|hdr|vr180
    "four_k": True,                      # or hdr / video_360 / video_3d / vr180
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

The other eight YouTube endpoints:

```python
from langchain_scavio import (
    ScavioYouTubeChannelCommunity, ScavioYouTubeChannelResolve,
    ScavioYouTubeChannelSearch, ScavioYouTubeChannelShorts,
    ScavioYouTubeCommentReplies, ScavioYouTubeRelated,
    ScavioYouTubeShorts, ScavioYouTubeSuggestions,
)

# Shorts search (2 credits per call)
result = ScavioYouTubeShorts(max_results=10).invoke({"query": "funny cats"})

# Keyword expansion before you spend a search
result = ScavioYouTubeSuggestions().invoke({"query": "python tut", "region": "US"})

# Replies need BOTH the video id and a comment's reply_cursor
comments = ScavioYouTubeComments().invoke({"video_id": "dQw4w9WgXcQ"})
result = ScavioYouTubeCommentReplies().invoke({
    "video_id": "dQw4w9WgXcQ",
    "reply_cursor": comments["data"]["comments"][0]["reply_cursor"],
})

# Related videos -- note: no next_cursor on this endpoint
result = ScavioYouTubeRelated(max_results=10).invoke({"video_id": "dQw4w9WgXcQ"})

# Find a channel, then reuse its UC id everywhere else
result = ScavioYouTubeChannelSearch(max_results=5).invoke({"query": "mrbeast"})
resolved = ScavioYouTubeChannelResolve().invoke({"channel": "@MrBeast"})
channel_id = resolved["data"]["channel_id"]

result = ScavioYouTubeChannelShorts(max_results=10).invoke({"channel_id": channel_id})
# community posts land under data.posts, not data.results
result = ScavioYouTubeChannelCommunity(max_results=10).invoke({
    "channel_id": channel_id,
})
```

### Reddit

All 12 Reddit endpoints cost 1 credit each.

```python
from langchain_scavio import (
    ScavioRedditSearch, ScavioRedditSearchSuggestions,
    ScavioRedditPost, ScavioRedditPostComments, ScavioRedditCommentReplies,
    ScavioRedditSubreddit, ScavioRedditSubredditPosts,
    ScavioRedditUser, ScavioRedditUserPosts, ScavioRedditUserComments,
    ScavioRedditPopular, ScavioRedditTrending,
)

search = ScavioRedditSearch(max_results=5)
result = search.invoke({"query": "langchain"})
# result["data"]["results"] + next_cursor + has_more
# Relevance order only: the endpoint has no sort or result-type filter

# Paginate by passing back the previous response's next_cursor
next_page = search.invoke({
    "query": "langchain",
    "cursor": result["data"]["next_cursor"],
})

# Expand a query before searching
ScavioRedditSearchSuggestions().invoke({"query": "langchain"})
# result["data"]["suggestions"] is a list of strings + total_count

post = ScavioRedditPost()
result = post.invoke({
    "url": "https://www.reddit.com/r/programming/comments/abc123/example_post/"
})
# or, equivalently: post.invoke({"post_id": "t3_abc123"})
# result["data"] is a flat post object (post_id, title, text, score,
# upvote_ratio, num_comments, media). It does NOT return comments.
```

Comments are a separate endpoint, and replies are a separate endpoint again:

```python
comments = ScavioRedditPostComments(max_results=10).invoke({
    "post_id": result["data"]["post_id"],   # 't3_...', a bare id, or a post URL
    "sort": "TOP",                          # UPPERCASE; default TOP
})
# comments["data"]["comments"] -> comment_id, text, author, score, created_at,
# depth, reply_cursor

# To expand one comment's thread, pass THAT comment's reply_cursor -- a
# next_cursor will not work here, and cursor is required.
ScavioRedditCommentReplies().invoke({
    "post_id": result["data"]["post_id"],
    "cursor": comments["data"]["comments"][0]["reply_cursor"],
})
# -> data.replies, same comment shape
```

Subreddits, redditors and the site-wide feeds:

```python
ScavioRedditSubreddit().invoke({"subreddit": "programming"})
# flat data: subscribers, active_count, description, icon, banner, is_nsfw

ScavioRedditSubredditPosts(max_results=10).invoke({
    "subreddit": "programming",
    "sort": "RISING",   # BEST|HOT|NEW|TOP|CONTROVERSIAL|RISING, default HOT
})
# data.posts -- this feed shape has no body text, thumbnail or is_nsfw;
# fetch a post_id through ScavioRedditPost for the full body

ScavioRedditUser().invoke({"username": "spez"})          # flat profile + karma
ScavioRedditUserPosts().invoke({"username": "spez", "sort": "TOP"})     # data.posts
ScavioRedditUserComments().invoke({"username": "spez"})  # data.comments

ScavioRedditPopular().invoke({})     # r/popular; cursor is its only parameter
ScavioRedditTrending().invoke({})    # data.trending -> {query, raw_query}
```

Sort values are UPPERCASE and differ by endpoint: `RISING` is accepted only by
`ScavioRedditSubredditPosts`, and the server default is `TOP` for comments,
`HOT` for the subreddit feed and `NEW` for the user feeds.

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

### X (Twitter)

Eleven endpoints, 1 credit each. The search field is literally `search`, and
handles are passed without the leading `@`.

```python
from langchain_scavio import (
    ScavioXSearch,
    ScavioXTrending,
    ScavioXTweetComments,
    ScavioXUser,
    ScavioXUserFollowings,
    ScavioXUserTweets,
)

# Search tweets -- the field is `search`, not `query`
search = ScavioXSearch(max_results=10)
result = search.invoke({
    "search": "langchain",
    "search_type": "Latest",              # Top (default), Latest, People, Photos, Videos
})
# result["data"]["timeline"] + next_cursor + has_more

# Profile and timeline
result = ScavioXUser().invoke({"screen_name": "elonmusk"})
result = ScavioXUserTweets().invoke({"screen_name": "elonmusk"})
# user timelines return data.timeline + data.pinned + data.user (no has_more)

# Replies to a tweet, ranked or chronological
result = ScavioXTweetComments().invoke({
    "tweet_id": "1808168603721650364",
    "rank": "latest",                     # lowercase, unlike search_type
})

# Followings come back under data.following -- singular, not "followings"
result = ScavioXUserFollowings().invoke({"screen_name": "elonmusk"})

# Trending takes a country NAME, not an ISO code
result = ScavioXTrending().invoke({"country": "UnitedStates"})
```

### LinkedIn

Nine live endpoints across three credit tiers: profile and single-post reads
cost 1, paginated list endpoints cost 10 per page, and job detail costs 30.

```python
from langchain_scavio import (
    ScavioLinkedInCompany,
    ScavioLinkedInJob,
    ScavioLinkedInPersonPosts,
    ScavioLinkedInPostComments,
    ScavioLinkedInSearchJobs,
)

# Profiles are addressed by vanity handle or full URL
result = ScavioLinkedInCompany().invoke({"company": "microsoft"})
# data.featured_employees is a 4-6 person sample -- the employee directory
# endpoint was retired upstream and is not exposed by this package

# Post feeds: 50 per page, 10 credits per page
posts = ScavioLinkedInPersonPosts(max_results=10)
result = posts.invoke({
    "username": "williamhgates",
    "type": "posts",                      # posts (default), comments, reactions
})

# Job search rotates its result set between calls -- dedupe by job id
jobs = ScavioLinkedInSearchJobs(max_results=10)
result = jobs.invoke({"search": "software engineer", "location": "London"})

# Job detail is the most expensive endpoint in the API (30 credits)
result = ScavioLinkedInJob().invoke({"job_id": "4415427228"})

# Post comments page by a 1-based number, not a cursor
result = ScavioLinkedInPostComments().invoke({
    "post_id": "7488618410256523265",
    "page": 1,
})
```

Five LinkedIn endpoints (`person/contact`, `company/people`, `company/jobs`,
`search/people`, `search/posts`) were retired upstream and always return HTTP
410. They are deliberately **not** exposed as tools: an agent calling them
would only burn a turn. Use `ScavioLinkedInCompany` (`featured_employees`) and
`ScavioLinkedInSearchJobs` with the company name instead.

## Credit Costs

Most endpoints cost 1 credit. The exceptions:

| Tool | Credits |
|------|---------|
| `ScavioLinkedInJob` | 30 |
| `ScavioKuaishouVideosBatch` | 40 |
| `ScavioInstagram*` (all but `UserPosts`, `Post`, `CommentReplies`), `ScavioLinkedInPersonPosts`, `ScavioLinkedInCompanyPosts`, `ScavioLinkedInSearchJobs`, `ScavioLinkedInPostComments`, `ScavioKuaishouProfile`, `ScavioKuaishouSearch`, `ScavioKuaishouSearchVideos`, `ScavioKuaishouSearchUsers`, `ScavioKuaishouSearchLive` | 10 |
| `ScavioYouTubeTranscript`, `ScavioInstagramPost`, `ScavioInstagramCommentReplies` | 8 |
| `ScavioG2Search`, `ScavioG2Product`, `ScavioG2Reviews` | 5 |
| `ScavioYouTubeStreams` | 3 |
| `ScavioYouTubeSearch`, `ScavioYouTubeShorts`, `ScavioInstagramUserPosts`, `ScavioKuaishouVideo`, all `ScavioHomeDepot*`, all `ScavioIndeed*`, all `ScavioTripadvisor*`, all `ScavioYelp*`, all `ScavioGooglePlay*`, all `ScavioCapterra*`, all `ScavioThreads*` (by `user_id`) | 2 |
| everything else | 1 |

**Body-priced tools.** Four surfaces cost a different number of credits
depending on what you send, so no flat figure is correct for them:

| Tool | Rule |
|------|------|
| `ScavioWalmartSearch`, `ScavioWalmartCategory` | `domain=com` or `ca` -> 1 credit; `domain=com.mx` -> 2 |
| `ScavioThreadsProfile`, `ScavioThreadsUserPosts`, `ScavioThreadsUserReplies` | `user_id` -> 2 credits; `username` -> 4 (the handle needs a second upstream lookup) |
| `ScavioExtract` | `mode=normal` or `advanced` -> 1 credit; `mode=ultra` -> 2. Billed only on a successful extraction |
| `ScavioKuaishou*` | priced per endpoint: 1, 2, 10 or 40 -- see each tool's description |

Every tool states its own cost in its `description`, so an agent can see the
price before it calls.

## Agent-Controllable Parameters

### ScavioSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query |
| `search_type` | `classic\|news\|maps` | Type of search |
| `gl` | `str` | Country the search runs from, ISO 3166-1 alpha-2 |
| `hl` | `str` | UI language, ISO 639-1 |
| `start` | `int` | Result OFFSET, not a page: 0, 10, 20, ... up to 990 |
| `google_domain` | `str` | Regional Google domain, e.g. `google.co.uk` |
| `device` | `desktop\|mobile` | Device type |
| `location` | `str` | Canonical location name, UULE-encoded server-side |
| `uule` | `str` | Pre-encoded UULE string; wins over `location` |
| `lr` | `str` | Language restrict on the pages, e.g. `lang_en` |
| `cr` | `str` | Country restrict on the pages, e.g. `countryUS` |
| `safe` | `active` | SafeSearch; `active` is the only accepted value |
| `nfpr` | `bool` | Disable spelling correction |
| `filter` | `"0"\|"1"` | Omitted-results filter, a STRING not a number |
| `time_period` | `str` | last_hour\|last_day\|last_week\|last_month\|last_year |
| `resolve_ai_overview` | `bool` | Resolve a deferred AI Overview (default true) |
| `include_html` | `bool` | Inline raw Google HTML. Off by default and very large |
| `country_code` / `language` / `page` | | Deprecated aliases of `gl` / `hl` / `start` |

Everything from `location` down is classic-search only: on `news` and `maps`
those filters are dropped rather than sent to an endpoint that cannot use them.

### ScavioAmazonSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Product search query |
| `country` | `str` | Two-letter marketplace code (us, gb, de, jp, ...). Defaults to us |
| `page` | `int` | Result page, 1-based. One page per call, 1 credit each |
| `domain` / `start_page` | | Deprecated aliases of `country` / `page` |

There is no sort, category, merchant or price filter: the marketplace ignores
them. Rank results yourself.

### ScavioAmazonProduct / ScavioAmazonOffers

| Parameter | Type | Description |
|-----------|------|-------------|
| `asin` | `str` | The ASIN. Sent on the wire as `query` |
| `country` | `str` | Two-letter marketplace code. Defaults to us |
| `domain` | `str` | Deprecated alias of `country` |
| `query` | `str` | Deprecated spelling of `asin`; `asin` wins when both are given |

### ScavioYouTubeSearch

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query |
| `upload_date` | `str` | last_hour\|today\|this_week\|this_month\|this_year |
| `type` | `str` | video\|channel\|playlist\|movie (`video_type` is the old alias) |
| `duration` | `str` | short\|medium\|long |
| `sort_by` | `str` | relevance\|date\|view_count\|rating |
| `hd` / `subtitles` / `creative_commons` / `live` | `bool` | Content filters |
| `four_k` / `hdr` / `video_360` / `video_3d` / `vr180` | `bool` | Sent as `4k`, `hdr`, `360`, `3d`, `vr180` |
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

### ScavioRedditSearchSuggestions

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Partial query to autocomplete (1-500 chars) |

### ScavioRedditPost

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | Full Reddit post URL (www., old., or new. subdomains accepted) |
| `post_id` | `str` | Post fullname `t3_...` or the bare id, instead of `url` |

Supply one of the two; supplying neither is a validation error.

### ScavioRedditPostComments

| Parameter | Type | Description |
|-----------|------|-------------|
| `post_id` | `str` | Post fullname `t3_...`, a bare post id, or a post URL |
| `sort` | `HOT\|NEW\|TOP\|BEST\|CONTROVERSIAL` | UPPERCASE, default `TOP` |
| `cursor` | `str` | Pagination cursor from a prior response's `next_cursor` |

### ScavioRedditCommentReplies

| Parameter | Type | Description |
|-----------|------|-------------|
| `post_id` | `str` | Post fullname `t3_...`, a bare post id, or a post URL |
| `cursor` | `str` | **Required.** The `reply_cursor` of the comment to expand |
| `sort` | `HOT\|NEW\|TOP\|BEST\|CONTROVERSIAL` | UPPERCASE, default `TOP` |

`cursor` is the one place a `next_cursor` is not accepted: it must be the
`reply_cursor` carried by a comment from `ScavioRedditPostComments`.

### ScavioRedditSubreddit

| Parameter | Type | Description |
|-----------|------|-------------|
| `subreddit` | `str` | Subreddit name without the `r/` prefix (1-100 chars) |

### ScavioRedditSubredditPosts

| Parameter | Type | Description |
|-----------|------|-------------|
| `subreddit` | `str` | Subreddit name without the `r/` prefix (1-100 chars) |
| `sort` | `BEST\|HOT\|NEW\|TOP\|CONTROVERSIAL\|RISING` | UPPERCASE, default `HOT` |
| `cursor` | `str` | Pagination cursor from a prior response's `next_cursor` |

This is the only Reddit endpoint that accepts `RISING`.

### ScavioRedditUser

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | Reddit username without the `u/` prefix (1-100 chars) |

### ScavioRedditUserPosts

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | Reddit username without the `u/` prefix (1-100 chars) |
| `sort` | `HOT\|NEW\|TOP\|BEST\|CONTROVERSIAL` | UPPERCASE, default `NEW` |
| `cursor` | `str` | Pagination cursor from a prior response's `next_cursor` |

### ScavioRedditUserComments

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | Reddit username without the `u/` prefix (1-100 chars) |
| `sort` | `HOT\|NEW\|TOP\|BEST\|CONTROVERSIAL` | UPPERCASE, default `NEW` |
| `cursor` | `str` | Pagination cursor from a prior response's `next_cursor` |

### ScavioRedditPopular

| Parameter | Type | Description |
|-----------|------|-------------|
| `cursor` | `str` | Pagination cursor from a prior response's `next_cursor` |

`cursor` is the endpoint's only parameter: no sort, no subreddit filter.

### ScavioRedditTrending

Takes no parameters. Invoke it with an empty dict.

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

The 93 tools added or rewritten in 4.0 follow. Every row is the endpoint's
own parameter, spelled exactly as the API takes it.

### ScavioWalmartSearch

`scavio_walmart_search` -> `POST /api/v1/walmart/search`. Costs 1 credit on domain com or ca and 2 credits on com.mx.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Product search query **(required)** |
| `start_page` | `int` | Deprecated alias for page |
| `fulfillment_speed` | `today\|tomorrow` | Delivery-speed filter. 2_days and anytime are deliberately not offered: 2_days leaks 3-4 day items and anytime is a no-op, so omit the parameter instead |
| `fulfillment_type` | `in_store` | Set to in_store to only return pickup stock |
| `domain` | `com\|ca\|com.mx` | Walmart storefront. com and ca cost 1 credit, com.mx costs 2. Default: com |
| `page` | `int` | Result page number, 1-based |
| `sort_by` | `best_match\|price_low\|price_high\|best_seller\|rating_high\|new` | Sort order for the results. Default: best_match |
| `min_price` | `float` | Minimum price filter |
| `max_price` | `float` | Maximum price filter |

### ScavioWalmartProduct

`scavio_walmart_product` -> `POST /api/v1/walmart/product`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | `str` | Walmart item id (usItemId), e.g. 13544111159 **(required)** |

### ScavioWalmartReviews

`scavio_walmart_reviews` -> `POST /api/v1/walmart/reviews`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | `str` | Walmart item id (usItemId), e.g. 13544111159 **(required)** |
| `page` | `int` | Result page, 1-based. 10 reviews per page |
| `sort` | `relevancy\|submission-desc\|submission-asc\|rating-desc\|rating-asc\|helpful-desc` | Review sort order |

### ScavioWalmartCategory

`scavio_walmart_category` -> `POST /api/v1/walmart/category`. Costs 1 credit on domain com or ca and 2 credits on com.mx.

| Parameter | Type | Description |
|-----------|------|-------------|
| `category_id` | `str` | Leaf category id (1095191) or the full underscore path (3944_133251_1095191) **(required)** |
| `limit` | `int` | Trims the products list after fetching. It does NOT reduce the credit cost |
| `fulfillment_speed` | `today\|tomorrow` | Delivery-speed filter. 2_days and anytime are deliberately not offered: 2_days leaks 3-4 day items and anytime is a no-op, so omit the parameter instead |
| `domain` | `com\|ca\|com.mx` | Walmart storefront. com and ca cost 1 credit, com.mx costs 2. Default: com |
| `page` | `int` | Result page number, 1-based |
| `sort_by` | `best_match\|price_low\|price_high\|best_seller\|rating_high\|new` | Sort order for the results. Default: best_match |
| `min_price` | `float` | Minimum price filter |
| `max_price` | `float` | Maximum price filter |

### ScavioWalmartOffers

`scavio_walmart_offers` -> `POST /api/v1/walmart/offers`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | `str` | Walmart item id (usItemId), e.g. 13544111159 **(required)** |

### ScavioWalmartSeller

`scavio_walmart_seller` -> `POST /api/v1/walmart/seller`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `seller_id` | `str` | NUMERIC catalog seller id (the seller_catalog_id field). The GUID form of seller_id returns 404 **(required)** |

### ScavioWalmartSellerProducts

`scavio_walmart_seller_products` -> `POST /api/v1/walmart/seller-products`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `seller_id` | `str` | NUMERIC catalog seller id (the seller_catalog_id field). The GUID form of seller_id returns 404 **(required)** |

### ScavioThreadsProfile

`scavio_threads_profile` -> `POST /api/v1/threads/profile`. Costs 2 credits when addressed by user_id and 4 credits when addressed by username -- the handle needs a second upstream lookup, so prefer user_id.

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | Threads handle without the @. Costs 2 extra credits because the handle has to be resolved with a second upstream call -- prefer user_id |
| `user_id` | `str` | Numeric Threads user id, e.g. 63625256886. This is the cheap path |

### ScavioThreadsUserPosts

`scavio_threads_user_posts` -> `POST /api/v1/threads/user/posts`. Costs 2 credits when addressed by user_id and 4 credits when addressed by username.

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | Threads handle without the @. Costs 2 extra credits because the handle has to be resolved with a second upstream call -- prefer user_id |
| `user_id` | `str` | Numeric Threads user id, e.g. 63625256886. This is the cheap path |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioThreadsUserReplies

`scavio_threads_user_replies` -> `POST /api/v1/threads/user/replies`. Costs 2 credits when addressed by user_id and 4 credits when addressed by username.

| Parameter | Type | Description |
|-----------|------|-------------|
| `username` | `str` | Threads handle without the @. Costs 2 extra credits because the handle has to be resolved with a second upstream call -- prefer user_id |
| `user_id` | `str` | Numeric Threads user id, e.g. 63625256886. This is the cheap path |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioThreadsPost

`scavio_threads_post` -> `POST /api/v1/threads/post`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `post_id` | `str` | Threads post id |
| `url` | `str` | A threads.net post URL, usable instead of post_id |

### ScavioThreadsPostComments

`scavio_threads_post_comments` -> `POST /api/v1/threads/post/comments`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `post_id` | `str` | Threads post id **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioThreadsSearchUsers

`scavio_threads_search_users` -> `POST /api/v1/threads/search/users`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Name or handle to look for **(required)** |

### ScavioKuaishouProfile

`scavio_kuaishou_profile` -> `POST /api/v1/kuaishou/profile`. Costs 10 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | Kuaishou user id **(required)** |

### ScavioKuaishouUserPosts

`scavio_kuaishou_user_posts` -> `POST /api/v1/kuaishou/user/posts`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | Kuaishou user id **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioKuaishouUserLive

`scavio_kuaishou_user_live` -> `POST /api/v1/kuaishou/user/live`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | `str` | Kuaishou user id **(required)** |

### ScavioKuaishouUserResolve

`scavio_kuaishou_user_resolve` -> `POST /api/v1/kuaishou/user/resolve`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `share_link` | `str` | A kuaishou.com or v.kuaishou.com share link. kwai.com links are NOT supported -- our upstream source does not serve Kwai international **(required)** |

### ScavioKuaishouVideo

`scavio_kuaishou_video` -> `POST /api/v1/kuaishou/video`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `photo_id` | `str` | Kuaishou photo (video) id |
| `url` | `str` | A kuaishou.com video URL, usable instead of photo_id |

### ScavioKuaishouVideoComments

`scavio_kuaishou_video_comments` -> `POST /api/v1/kuaishou/video/comments`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `photo_id` | `str` | Kuaishou photo (video) id **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioKuaishouCommentReplies

`scavio_kuaishou_comment_replies` -> `POST /api/v1/kuaishou/video/sub-comments`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `photo_id` | `str` | Kuaishou photo (video) id **(required)** |
| `root_comment_id` | `str` | Id of the root comment whose replies you want **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |
| `count` | `int` | Replies to return in this page, 1-50 |

### ScavioKuaishouVideosBatch

`scavio_kuaishou_videos_batch` -> `POST /api/v1/kuaishou/videos/batch`. Costs 40 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `photo_ids` | `list[str]` | Kuaishou photo ids to fetch in one call. Hard cap of 20 ids **(required)** |

### ScavioKuaishouSearch

`scavio_kuaishou_search` -> `POST /api/v1/kuaishou/search`. Costs 10 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | `str` | Search keyword **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioKuaishouSearchVideos

`scavio_kuaishou_search_videos` -> `POST /api/v1/kuaishou/search/videos`. Costs 10 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | `str` | Search keyword **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioKuaishouSearchUsers

`scavio_kuaishou_search_users` -> `POST /api/v1/kuaishou/search/users`. Costs 10 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | `str` | Search keyword **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioKuaishouSearchLive

`scavio_kuaishou_search_live` -> `POST /api/v1/kuaishou/search/live`. Costs 10 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | `str` | Search keyword **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioKuaishouTagFeed

`scavio_kuaishou_tag_feed` -> `POST /api/v1/kuaishou/tag/feed`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tag` | `str` | Hashtag to read the feed for, without the leading # **(required)** |
| `cursor` | `str` | Pagination cursor taken from a previous response's next_cursor. Keep the other arguments identical across paginated calls |

### ScavioKuaishouTrending

`scavio_kuaishou_trending` -> `POST /api/v1/kuaishou/trending`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `board` | `hot\|live\|shopping\|brand\|music` | Which leaderboard to return. Default: hot |

### ScavioEbaySearch

`scavio_ebay_search` -> `POST /api/v1/ebay/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Keyword query. Optional: a seller-scoped search works with no query at all |
| `seller` | `str` | Scope the search to one seller. Works with no query, which is the only paginated way to list a seller's whole catalogue |
| `page` | `int` | Result page number, 1-based |
| `sort_by` | `best_match\|ending_soonest\|newly_listed\|price_low\|price_high` | Sort order for the results. Default: best_match |
| `min_price` | `float` | Minimum price filter |
| `max_price` | `float` | Maximum price filter |
| `condition` | `new\|open_box\|refurbished\|used\|for_parts` | Item condition. refurbished is eBay's parent condition, not one of its three graded tiers |
| `buying_format` | `auction\|buy_it_now\|best_offer` | Listing format filter |
| `free_shipping` | `bool` | Only return listings with free shipping |
| `sold` | `bool` | Search completed listings that actually SOLD -- the price-research view. eBay publishes no headline count there, so total_results comes back null |
| `category_id` | `str` | Numeric eBay category id. A non-numeric value returns the UNFILTERED set under a 200 |
| `per_page` | `60\|120\|240` | Listings per page. eBay accepts only 60, 120 or 240 and silently falls back to 60 for anything else. Default: 60 |

### ScavioEbayProduct

`scavio_ebay_product` -> `POST /api/v1/ebay/product`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `item_id` | `str` | eBay item number or a full ebay.com/itm/... URL. Tracking parameters are discarded **(required)** |

### ScavioEbaySeller

`scavio_ebay_seller` -> `POST /api/v1/ebay/seller`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `seller` | `str` | eBay username as it appears in ebay.com/usr/<name> **(required)** |

### ScavioTargetSearch

`scavio_target_search` -> `POST /api/v1/target/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `keyword` | `str` | Product search query **(required)** |
| `page` | `int` | Result page number, 1-based |
| `count` | `int` | Products per page. Target rejects anything above 28 outright. Default: 24 |
| `sort` | `relevance\|featured\|price_low\|price_high\|rating_high\|best_seller\|newest` | Sort order for the results. Default: relevance |
| `store_id` | `str` | Numeric Target store id. Unlike Walmart this is a real request parameter: it decides prices and availability. Default: 3991 |

### ScavioTargetCategory

`scavio_target_category` -> `POST /api/v1/target/category`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `category_id` | `str` | The segment after `N-` in a target.com /c/ URL **(required)** |
| `page` | `int` | Result page number, 1-based |
| `count` | `int` | Products per page. Target rejects anything above 28 outright. Default: 24 |
| `sort` | `relevance\|featured\|price_low\|price_high\|rating_high\|best_seller\|newest` | Sort order for the results. Default: relevance |
| `store_id` | `str` | Numeric Target store id. Unlike Walmart this is a real request parameter: it decides prices and availability. Default: 3991 |

### ScavioTargetProduct

`scavio_target_product` -> `POST /api/v1/target/product`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tcin` | `str` | Target catalog item number. A child TCIN is answered by its variation parent, with the child present under variants **(required)** |
| `store_id` | `str` | Numeric Target store id. Unlike Walmart this is a real request parameter: it decides prices and availability. Default: 3991 |

### ScavioTargetReviews

`scavio_target_reviews` -> `POST /api/v1/target/reviews`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tcin` | `str` | Target catalog item number. A child TCIN is answered by its variation parent, with the child present under variants **(required)** |
| `limit` | `int` | TRIMS the returned bodies only. Target publishes 8 reviews anonymously and offers no paging, so this cannot fetch more |
| `store_id` | `str` | Numeric Target store id. Unlike Walmart this is a real request parameter: it decides prices and availability. Default: 3991 |

### ScavioHomeDepotSearch

`scavio_home_depot_search` -> `POST /api/v1/homedepot/search`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Product search query **(required)** |
| `page` | `int` | Result page, 1-based. 12 products per page, fixed |
| `sort_by` | `best_match\|top_sellers\|top_rated\|price_low\|price_high` | Sort order. The set is closed because Home Depot answers an unknown sort with an empty page rather than falling back. Default: best_match |
| `min_price` | `float` | Minimum price filter |
| `max_price` | `float` | Maximum price filter |

### ScavioHomeDepotProduct

`scavio_home_depot_product` -> `POST /api/v1/homedepot/product`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `item_id` | `str` | Home Depot item id or a full homedepot.com/p/... URL. Tracking parameters are discarded **(required)** |

### ScavioHomeDepotReviews

`scavio_home_depot_reviews` -> `POST /api/v1/homedepot/reviews`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `item_id` | `str` | Home Depot item id or a full homedepot.com/p/... URL. Tracking parameters are discarded **(required)** |
| `page` | `int` | Result page, 1-based. 30 reviews per page; asking past total_pages is a 404 |

### ScavioZillowSearch

`scavio_zillow_search` -> `POST /api/v1/zillow/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `location` | `str` | Zillow region slug, human city name, ZIP, or a pasted search URL. A bare ZIP works alone but CANNOT be combined with a filter or a sort -- use the city name there **(required)** |
| `listing_status` | `for_sale\|for_rent\|sold` | Which listing state to return. Default: for_sale |
| `page` | `int` | Result page number, 1-based |
| `sort` | `relevance\|recommended\|newest\|price_low\|price_high\|payment_low\|payment_high\|beds\|baths\|sqft\|lot_size\|zestimate_low\|zestimate_high\|recent_change` | Sort order for the results |
| `min_price` | `float` | Minimum price. On listing_status=for_rent this means MONTHLY RENT |
| `max_price` | `float` | Maximum price. On listing_status=for_rent this means MONTHLY RENT |
| `beds_min` | `int` | Minimum number of bedrooms |
| `beds_max` | `int` | Maximum number of bedrooms |
| `baths_min` | `float` | Minimum number of bathrooms. Half-baths allowed (1.5) |
| `baths_max` | `float` | Maximum number of bathrooms |
| `sqft_min` | `int` | Minimum living area in square feet |
| `sqft_max` | `int` | Maximum living area in square feet |
| `lot_size_min` | `int` | Minimum lot size in square feet |
| `lot_size_max` | `int` | Maximum lot size in square feet |
| `year_built_min` | `int` | Earliest year built |
| `year_built_max` | `int` | Latest year built |
| `max_hoa` | `float` | Maximum monthly HOA fee |
| `home_type` | `houses\|townhomes\|multi_family\|condos\|apartments\|manufactured\|lots_land` | Property type filter |
| `days_on_zillow` | `1\|7\|14\|30\|90\|6m\|12m\|24m\|36m` | How recently the listing appeared. Closed set: an unrecognised value returns the UNFILTERED result set under a 200 |
| `keywords` | `str` | Extra keywords to match inside the listing text |
| `has_pool` | `bool` | Only return properties with a pool |
| `has_garage` | `bool` | Only return properties with a garage |
| `has_air_conditioning` | `bool` | Only return properties with air conditioning |
| `is_waterfront` | `bool` | Only return waterfront properties |
| `has_basement` | `bool` | Only return properties with a basement |
| `is_new_construction` | `bool` | Only return new construction |
| `has_open_house` | `bool` | Only return listings with an open house scheduled |
| `price_reduced` | `bool` | Only return listings whose price was reduced |
| `is_3d_tour` | `bool` | Only return listings with a 3D tour |

### ScavioZillowProperty

`scavio_zillow_property` -> `POST /api/v1/zillow/property`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `zpid` | `str` | Zillow property id, a /homedetails/ URL, or a zillow.com/apartments/ building URL. Rental buildings have no visible zpid -- pass the URL **(required)** |

### ScavioZillowAgentReviews

`scavio_zillow_agent_reviews` -> `POST /api/v1/zillow/reviews`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `screen_name` | `str` | The AGENT's zillow.com/profile/<name>/ screen name, or the full profile URL. This endpoint addresses an agent, not a property **(required)** |

### ScavioBookingSearch

`scavio_booking_search` -> `POST /api/v1/booking/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `destination` | `str` | Destination name. Either destination or dest_id is required -- a search with neither returns Booking's homepage and still costs a credit |
| `dest_id` | `str` | Booking's numeric destination id |
| `dest_type` | `city\|region\|country\|district\|landmark\|airport\|hotel` | What dest_id refers to. Requires dest_id |
| `page` | `int` | Result page, 1-based. 25 properties per page |
| `sort_by` | `popularity\|price_low\|price_high\|stars_high\|stars_low\|stars_and_price\|distance\|review_score` | Sort order for the results. Default: popularity |
| `min_price` | `float` | Minimum price PER NIGHT, in `currency` |
| `max_price` | `float` | Maximum price PER NIGHT, in `currency` |
| `stars` | `list[int]` | Star ratings to include. Values are OR'd together |
| `min_review_score` | `6\|7\|8\|9` | Minimum guest review score. Only 6, 7, 8 and 9 are accepted -- any other threshold is silently dropped upstream |
| `property_type` | `str\|int` | Accommodation type: one of the named values, or a raw numeric Booking accommodation-type id |
| `free_cancellation` | `bool` | Only return rates with free cancellation |
| `no_prepayment` | `bool` | Only return rates with no prepayment |
| `breakfast_included` | `bool` | Only return rates that include breakfast |
| `checkin` | `str` | Check-in date, YYYY-MM-DD. Must be sent together with checkout |
| `checkout` | `str` | Check-out date, YYYY-MM-DD. Must be sent together with checkin |
| `adults` | `int` | Number of adult guests. Default: 2 |
| `children_ages` | `list[int]` | Ages of the children in the party, one entry per child. Ages, not a count |
| `rooms` | `int` | Number of rooms required. Default: 1 |
| `currency` | `str` | ISO 4217 currency code the prices come back in. Default: USD |

### ScavioBookingHotel

`scavio_booking_hotel` -> `POST /api/v1/booking/hotel`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `hotel` | `str` | booking.com property URL or the bare page slug. Query parameters are discarded **(required)** |
| `country_code` | `str` | Two-letter country code. Only consulted when `hotel` is a bare slug; a wrong one is a real, BILLED 404. Default: us |
| `checkin` | `str` | Check-in date, YYYY-MM-DD. Must be sent together with checkout |
| `checkout` | `str` | Check-out date, YYYY-MM-DD. Must be sent together with checkin |
| `adults` | `int` | Number of adult guests. Default: 2 |
| `children_ages` | `list[int]` | Ages of the children in the party, one entry per child |
| `rooms` | `int` | Number of rooms required. Default: 1 |
| `currency` | `str` | ISO 4217 currency code the prices come back in. Default: USD |

### ScavioBookingReviews

`scavio_booking_reviews` -> `POST /api/v1/booking/reviews`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `hotel` | `str` | booking.com property URL or the bare page slug. Query parameters are discarded **(required)** |
| `country_code` | `str` | Two-letter country code. Only consulted when `hotel` is a bare slug; a wrong one is a real, BILLED 404. Default: us |
| `checkin` | `str` | Check-in date, YYYY-MM-DD. Must be sent together with checkout |
| `checkout` | `str` | Check-out date, YYYY-MM-DD. Must be sent together with checkin |
| `adults` | `int` | Number of adult guests. Default: 2 |
| `children_ages` | `list[int]` | Ages of the children in the party, one entry per child |
| `rooms` | `int` | Number of rooms required. Default: 1 |
| `currency` | `str` | ISO 4217 currency code the prices come back in. Default: USD |

### ScavioTripadvisorLocations

`scavio_tripadvisor_locations` -> `POST /api/v1/tripadvisor/locations`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Place or business NAME to resolve into TripAdvisor ids **(required)** |
| `limit` | `int` | Maximum rows to return, 1-20. Default: 12 |

### ScavioTripadvisorSearch

`scavio_tripadvisor_search` -> `POST /api/v1/tripadvisor/search`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `geo_id` | `str` | TripAdvisor geo id. Accepts 30196, g30196, or a URL carrying one |
| `category` | `restaurants\|hotels\|attractions` | Which family the location belongs to. On reviews it also sets the page size (15 for restaurants, 10 for hotels and attractions), so it must match the location's own type on any page past the first. Default: restaurants |
| `page` | `int` | Result page, 1-based. 30 locations per page; a page beyond the last is a 404, not an empty result |
| `url` | `str` | Full tripadvisor.com listing URL, usable instead of the ids. Country sites are accepted |

### ScavioTripadvisorLocation

`scavio_tripadvisor_location` -> `POST /api/v1/tripadvisor/location`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `location_id` | `str` | TripAdvisor location id. Accepts 1899234, d1899234, or a full _Review URL |
| `geo_id` | `str` | TripAdvisor geo id. Accepts 30196, g30196, or a URL carrying one |
| `category` | `restaurants\|hotels\|attractions` | Which family the location belongs to. On reviews it also sets the page size (15 for restaurants, 10 for hotels and attractions), so it must match the location's own type on any page past the first. Default: restaurants |
| `url` | `str` | Full tripadvisor.com listing URL, usable instead of the ids. Country sites are accepted |

### ScavioTripadvisorReviews

`scavio_tripadvisor_reviews` -> `POST /api/v1/tripadvisor/reviews`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `location_id` | `str` | TripAdvisor location id. Accepts 1899234, d1899234, or a full _Review URL |
| `geo_id` | `str` | TripAdvisor geo id. Accepts 30196, g30196, or a URL carrying one |
| `category` | `restaurants\|hotels\|attractions` | Which family the location belongs to. On reviews it also sets the page size (15 for restaurants, 10 for hotels and attractions), so it must match the location's own type on any page past the first. Default: restaurants |
| `url` | `str` | Full tripadvisor.com listing URL, usable instead of the ids. Country sites are accepted |
| `page` | `int` | Result page, 1-based. Page 1 is already inside the location endpoint -- use this to page PAST it. Past the last page is a 404 |

### ScavioIndeedSearch

`scavio_indeed_search` -> `POST /api/v1/indeed/search`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Job title, keyword or company. Optional if location is set |
| `location` | `str` | City and state, postal code, state, country or 'Remote'. Usable with no query at all -- that returns every posting in a metro |
| `page` | `int` | Result page, 1-based. 10 postings per page |
| `radius` | `0\|5\|10\|15\|25\|35\|50\|100` | Search radius in miles. Closed set: Indeed IGNORES any other value and returns the unfiltered set, so asking for 7 would silently buy 50. Default: 50 |
| `max_age_days` | `1\|3\|7\|14` | Only postings published within this many days. Closed set for the same reason as radius |
| `job_type` | `full_time\|part_time\|contract\|temporary\|internship` | Employment type filter |
| `min_salary` | `float` | Minimum salary. This filters on INDEED'S OWN ESTIMATE for the role, not a posted figure, so postings that publish no salary still match |
| `remote` | `bool` | Only return remote roles |

### ScavioIndeedJob

`scavio_indeed_job` -> `POST /api/v1/indeed/job`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | `str` | 16-hex Indeed job key, or any indeed.com URL carrying jk= (/viewjob, /rc/clk, /pagead/clk) **(required)** |

### ScavioIndeedCompany

`scavio_indeed_company` -> `POST /api/v1/indeed/company`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `company` | `str` | indeed.com/cmp/<slug> slug or a full profile URL. Slugs are untidy, e.g. 'Tata-Consultancy-Services-(tcs)' **(required)** |

### ScavioIndeedCompanyReviews

`scavio_indeed_company_reviews` -> `POST /api/v1/indeed/company/reviews`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `company` | `str` | indeed.com/cmp/<slug> slug or a full profile URL. Slugs are untidy, e.g. 'Tata-Consultancy-Services-(tcs)' **(required)** |
| `page` | `int` | Result page, 1-based. 20 reviews per page |

### ScavioAirbnbSearch

`scavio_airbnb_search` -> `POST /api/v1/airbnb/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `location` | `str` | City, region, ZIP, or a pasted airbnb.com/s/ URL. An unresolvable place is a 404 **(required)** |
| `check_in` | `str` | Check-in date, YYYY-MM-DD. Must be sent with check_out. Omitting both makes Airbnb A/B both the window AND the prices -- the response flags that as dates_are_defaulted. Default: +30d when omitted (transport) |
| `check_out` | `str` | Check-out date, YYYY-MM-DD. Must be sent with check_in. Default: check_in + 5 nights when omitted |
| `adults` | `int` | Number of adult guests |
| `children` | `int` | Number of children in the party (ages 2-12) |
| `infants` | `int` | Number of infants in the party |
| `pets` | `int` | Number of pets travelling |
| `min_price` | `float` | Minimum WHOLE-STAY total, not a per-night rate |
| `max_price` | `float` | Maximum WHOLE-STAY total, not a per-night rate |
| `room_type` | `entire_home\|private_room\|shared_room\|hotel_room` | Room type filter |
| `min_bedrooms` | `int` | Minimum number of bedrooms |
| `min_beds` | `int` | Minimum number of beds |
| `min_bathrooms` | `int` | Minimum number of bathrooms |
| `superhost` | `bool` | Only return Superhost listings |
| `instant_book` | `bool` | Only return instant-book listings |
| `guest_favorite` | `bool` | Only return Guest Favourite listings |
| `free_cancellation` | `bool` | Only return listings with free cancellation |
| `amenities` | `str` | Comma-separated amenity filter. Named vocabulary: wifi, air_conditioning, pool, kitchen, free_parking, washer, self_check_in, tv -- or raw numeric Airbnb amenity ids. An unrecognised NAME is rejected before the scrape |
| `currency` | `str` | ISO 4217 currency code the prices come back in. Default: USD |
| `page` | `int` | Result page, 1-based. 18 listings per page. Cannot be combined with cursor |
| `cursor` | `str` | next_cursor from a previous response. Wins over page, so sending both is rejected |

### ScavioAirbnbListing

`scavio_airbnb_listing` -> `POST /api/v1/airbnb/listing`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `listing_id` | `str` | Airbnb listing id or a full /rooms/ URL. Query parameters are discarded because they carry someone else's dates **(required)** |
| `check_in` | `str` | Check-in date, YYYY-MM-DD. Must be sent with check_out. Omitting both makes Airbnb A/B both the window AND the prices -- the response flags that as dates_are_defaulted |
| `check_out` | `str` | Check-out date, YYYY-MM-DD. Must be sent with check_in |
| `adults` | `int` | Number of adult guests |
| `children` | `int` | Number of children in the party (ages 2-12) |
| `infants` | `int` | Number of infants in the party |
| `pets` | `int` | Number of pets travelling |
| `currency` | `str` | ISO 4217 currency code the prices come back in. Default: USD |

### ScavioAirbnbReviews

`scavio_airbnb_reviews` -> `POST /api/v1/airbnb/reviews`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `listing_id` | `str` | Airbnb listing id or a full /rooms/ URL. Query parameters are discarded because they carry someone else's dates **(required)** |
| `currency` | `str` | ISO 4217 currency code the prices come back in. Default: USD |
| `limit` | `int` | Reviews per page, 1-50. Airbnb returns a fixed 7 when no explicit limit is sent, so always set it. Default: 30 |
| `offset` | `int` | Zero-based review offset for paging. Default: 0 |

### ScavioGlassdoorCompanies

`scavio_glassdoor_companies` -> `POST /api/v1/glassdoor/companies`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Company NAME to resolve into an employer_id **(required)** |

### ScavioGlassdoorCompany

`scavio_glassdoor_company` -> `POST /api/v1/glassdoor/company`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `employer_id` | `str` | Glassdoor employer id as a STRING -- a JSON number is rejected. Accepts 1699, E1699 or IE1699 |
| `company` | `str` | Company name. COSMETIC only: the profile resolves on employer_id alone, it is ignored entirely when url is set, and it does not satisfy the required-identifier rule |
| `url` | `str` | Any glassdoor.com employer URL (/Overview/, /Reviews/, /Salary/). Non-glassdoor.com hosts are rejected |

### ScavioGlassdoorReviews

`scavio_glassdoor_reviews` -> `POST /api/v1/glassdoor/reviews`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `employer_id` | `str` | Glassdoor employer id as a STRING -- a JSON number is rejected. Accepts 1699, E1699 or IE1699 |
| `company` | `str` | Company name. COSMETIC only: the profile resolves on employer_id alone, it is ignored entirely when url is set, and it does not satisfy the required-identifier rule |
| `url` | `str` | Pass back reviews_url from the company endpoint to skip the resolve fetch -- addressing this endpoint by employer_id costs two upstream fetches |
| `category` | `career_development\|compensation\|culture\|diversity_and_inclusion\|management\|work_life_balance` | Review category filter. Closed set: Glassdoor IGNORES an unknown value and returns the unfiltered set under a 200 |
| `employment_status` | `full_time\|part_time\|contract\|intern` | Reviewer employment status filter. Closed set for the same reason as category |

### ScavioGlassdoorSalaries

`scavio_glassdoor_salaries` -> `POST /api/v1/glassdoor/salaries`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `employer_id` | `str` | Glassdoor employer id as a STRING -- a JSON number is rejected. Accepts 1699, E1699 or IE1699 |
| `company` | `str` | Company name. COSMETIC only: the profile resolves on employer_id alone, it is ignored entirely when url is set, and it does not satisfy the required-identifier rule |
| `url` | `str` | Pass back salaries_url from the company endpoint to skip the resolve fetch |
| `page` | `int` | Result page, 1-based. 10 job titles per page; page_count on the response says how many exist |

### ScavioYelpSearch

`scavio_yelp_search` -> `POST /api/v1/yelp/search`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `term` | `str` | What to look for, e.g. 'coffee' or a business name |
| `location` | `str` | City, neighbourhood or address. Effectively required: without it Yelp geolocates off the proxy exit and the same request answers about a different metro run to run |
| `page` | `int` | Result page, 1-based. Yelp fixes the page size at 10 |
| `sort` | `recommended\|rating\|review_count` | Sort order. Closed set: Yelp IGNORES an unrecognised value and serves default ranking under a billed 200. Default: recommended |
| `price` | `list[int]` | Price bands to include, 1 (cheapest) to 4 (priciest) |
| `open_now` | `bool` | Only return businesses open right now |
| `attributes` | `list[str]` | Raw Yelp filter aliases sent through as attrs, e.g. RestaurantsDelivery, GoodForKids, WheelchairAccessible. This is a passthrough, not a closed enum: an alias Yelp does not know is ignored and results come back unfiltered |
| `url` | `str` | Full yelp.com/search URL, usable instead of term plus location |

### ScavioYelpBusiness

`scavio_yelp_business` -> `POST /api/v1/yelp/business`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `business_id` | `str` | Yelp alias (desnudo-coffee-austin-2), opaque encid, or a yelp.com/biz URL |
| `url` | `str` | Full listing URL, usable instead of the id fields |

### ScavioYelpReviews

`scavio_yelp_reviews` -> `POST /api/v1/yelp/reviews`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `business_id` | `str` | Yelp alias (desnudo-coffee-austin-2), opaque encid, or a yelp.com/biz URL |
| `url` | `str` | Full listing URL, usable instead of the id fields |
| `page` | `int` | Result page, 1-based. PAGE 1 IS REDUNDANT with the business endpoint and costs another 2 credits -- start at page 2. A page past the last is a 404 |
| `sort` | `relevance\|newest\|oldest\|rating_high\|rating_low\|elites` | Sort order. Closed set: Yelp IGNORES an unrecognised value and serves default ranking under a billed 200. Default: relevance |
| `rating` | `1\|2\|3\|4\|5` | Only return reviews with this star rating. Changes filtered_review_count, not review_count |

### ScavioAppStoreSearch

`scavio_app_store_search` -> `POST /api/v1/appstore/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `term` | `str` | Search term. Matches the app name, a keyword, OR a publisher name -- searching a developer returns their catalogue **(required)** |
| `limit` | `int` | Apps to return, 1-200. This is the ONLY lever on result volume: App Store search has no pagination and every offset spelling is silently ignored. Default: 25 |
| `country` | `str` | Two-letter storefront code. It decides price, currency, localised title and whether the app is sold there at all. Anything that is not exactly two letters silently falls back to us. Default: us |
| `entity` | `software\|ipad_software\|mac_software` | Which App Store catalogue to search. Default: software |
| `lang` | `str` | Five-letter locale, e.g. en_us. Independent of country: the storefront sets the price, this sets the words |

### ScavioAppStoreApp

`scavio_app_store_app` -> `POST /api/v1/appstore/app`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `app_id` | `str` | Numeric App Store id OR a bundle id (notion.id, com.burbn.instagram). A pasted apps.apple.com URL is rejected with a free 400 **(required)** |
| `country` | `str` | Two-letter storefront code. It decides price, currency, localised title and whether the app is sold there at all. Anything that is not exactly two letters silently falls back to us. Default: us |

### ScavioAppStoreReviews

`scavio_app_store_reviews` -> `POST /api/v1/appstore/reviews`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `app_id` | `str` | Numeric App Store id. NUMERIC ONLY here -- the reviews feed has no bundle-id form **(required)** |
| `country` | `str` | Two-letter storefront code. It decides price, currency, localised title and whether the app is sold there at all. Anything that is not exactly two letters silently falls back to us. Default: us |
| `page` | `int` | Result page, 1-10, at 50 reviews each. Apple hard-stops at page 10; reach further by asking a different country. Default: 1 |
| `sort` | `most_recent\|most_helpful` | Review sort order. Under most_recent almost every review is too new to have been voted on, so the vote fields come back as zeroes. Default: most_recent |

### ScavioGooglePlaySearch

`scavio_google_play_search` -> `POST /api/v1/googleplay/search`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Search query. There is no pagination -- one shelf of ~30 apps **(required)** |
| `hl` | `str` | Interface language. It moves the whole storefront, not only the strings: title, description, install formatting and content rating all follow it. Default: en |
| `gl` | `str` | Storefront country code. Default: us |

### ScavioGooglePlayApp

`scavio_google_play_app` -> `POST /api/v1/googleplay/app`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `app_id` | `str` | Android package name, or any play.google.com URL carrying one in its id parameter **(required)** |
| `hl` | `str` | Interface language. It moves the whole storefront, not only the strings: title, description, install formatting and content rating all follow it. Default: en |
| `gl` | `str` | Storefront country code. Default: us |

### ScavioGooglePlayReviews

`scavio_google_play_reviews` -> `POST /api/v1/googleplay/reviews`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `app_id` | `str` | Android package name, or any play.google.com URL carrying one in its id parameter **(required)** |
| `sort` | `relevance\|newest\|rating` | Review sort order. Default: newest |
| `count` | `int` | Reviews to return, 1-200. Default: 50 |
| `cursor` | `str` | next_cursor from a previous response. OPAQUE and SINGLE-USE, and it encodes the sort as well as the position -- send it back with the SAME sort it came from. A cursor past the last review is a 404 |
| `hl` | `str` | Interface language. It moves the whole storefront, not only the strings: title, description, install formatting and content rating all follow it. Default: en |
| `gl` | `str` | Storefront country code. Default: us |

### ScavioSECLookup

`scavio_sec_lookup` -> `POST /api/v1/sec/lookup`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Ticker, company name, or a fragment of either **(required)** |
| `limit` | `int` | Maximum filers to return, 1-100. Default: 10 |
| `exchange` | `str` | Listing exchange filter, matched case-insensitively. Filers listed with no exchange are excluded by any value |

### ScavioSECCompany

`scavio_sec_company` -> `POST /api/v1/sec/company`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cik` | `str` | Central Index Key: 320193, 0000320193 or CIK0000320193. A ticker is accepted here too |
| `ticker` | `str` | Stock ticker, dotted or dashed (BRK.B / BRK-B). WINS over cik when both are given |

### ScavioSECFilings

`scavio_sec_filings` -> `POST /api/v1/sec/filings`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cik` | `str` | Central Index Key: 320193, 0000320193 or CIK0000320193. A ticker is accepted here too |
| `ticker` | `str` | Stock ticker, dotted or dashed (BRK.B / BRK-B). WINS over cik when both are given |
| `form` | `str\|list[str]` | Form filter: "10-K", ["10-K", "10-Q"] or "10-K,8-K". Matched against the form AND its root form, so 10-K also returns 10-K/A amendments |
| `date_from` | `str` | Earliest date to include, YYYY-MM-DD |
| `date_to` | `str` | Latest date to include, YYYY-MM-DD |
| `page` | `int` | Result page number, 1-based |
| `limit` | `int` | Filings to return, 1-500. Default: 50 |
| `include_history` | `bool` | Also read the archived filing shards (up to 10). Still one credit; history_truncated flags a filer that had more. Default: False |

### ScavioSECConcept

`scavio_sec_concept` -> `POST /api/v1/sec/concept`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cik` | `str` | Central Index Key: 320193, 0000320193 or CIK0000320193. A ticker is accepted here too |
| `ticker` | `str` | Stock ticker, dotted or dashed (BRK.B / BRK-B). WINS over cik when both are given |
| `concept` | `str` | XBRL tag, CASE-SENSITIVE: 'netincomeloss' is a 404 upstream, not a match. Use the facts endpoint to list what a filer actually reports **(required)** |
| `taxonomy` | `str` | XBRL taxonomy: us-gaap, dei, ifrs-full or srt. Default: us-gaap |
| `unit` | `str` | Unit of measure to filter on, e.g. USD or USD/shares |
| `form` | `str` | Form filter. EXACT match here, so '10-K' excludes 10-K/A |
| `limit` | `int` | Values to return, 1-2000. Default: 250 |

### ScavioSECFacts

`scavio_sec_facts` -> `POST /api/v1/sec/facts`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cik` | `str` | Central Index Key: 320193, 0000320193 or CIK0000320193. A ticker is accepted here too |
| `ticker` | `str` | Stock ticker, dotted or dashed (BRK.B / BRK-B). WINS over cik when both are given |
| `taxonomy` | `str` | Restrict the index to one XBRL taxonomy |
| `query` | `str` | Case-insensitive substring matched against the tag name and its label |
| `limit` | `int` | Concepts to return, 1-2000. Default: 250 |

### ScavioSECSearch

`scavio_sec_search` -> `POST /api/v1/sec/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Full-text query. A quoted phrase is exact, bare words are a bag of terms. Optional -- a cik, form or date filter on its own is a valid search |
| `cik` | `str\|list[str]` | One CIK or a list of them. Tickers are accepted here too |
| `ticker` | `str\|list[str]` | One ticker or a list of them |
| `form` | `str\|list[str]` | One form type or a list of them |
| `date_from` | `str` | Earliest filing date, YYYY-MM-DD. Coverage starts 2001 |
| `date_to` | `str` | Latest date to include, YYYY-MM-DD |
| `location` | `str\|list[str]` | EDGAR's own jurisdiction codes: CA, NY, and alphanumeric codes for foreign jurisdictions. One code or a list |
| `sort` | `relevance\|newest\|oldest` | Sort order for the results. Default: relevance |
| `page` | `int` | Result page, 1-100, at 100 documents each. The index refuses a result window past 10,000 |

### ScavioRedfinSearch

`scavio_redfin_search` -> `POST /api/v1/redfin/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `location` | `str` | A redfin.com region URL (/city/, /neighborhood/, /county/, /zipcode/) or a bare 5-digit ZIP. CITY NAMES ARE NOT ACCEPTED |
| `region_id` | `int` | Redfin's own numeric region id. NOT a ZIP code -- different number spaces, and a ZIP here resolves to another city rather than failing. Must be sent together with region_type |
| `region_type` | `1\|2\|5\|6` | What region_id refers to: 1 neighborhood, 2 ZIP, 5 county, 6 city. Must be sent together with region_id |
| `listing_status` | `for_sale\|sold\|for_rent` | Which listing state to return. Default: for_sale |
| `sold_within_days` | `int` | How far back to look for sold homes. Only valid with listing_status=sold, where it defaults to 90. Default: 90 |
| `page` | `int` | Result page number, 1-based |
| `limit` | `int` | Listings per page, 1-350. Default: 100 |
| `sort` | `recommended\|price_low\|price_high\|newest\|oldest\|sqft_low\|sqft_high\|price_per_sqft_low\|price_per_sqft_high` | Sort order for the results. Default: recommended |
| `min_price` | `float` | Minimum price. On listing_status=for_rent this means MONTHLY RENT |
| `max_price` | `float` | Maximum price. On listing_status=for_rent this means MONTHLY RENT |
| `beds_min` | `int` | Minimum number of bedrooms |
| `beds_max` | `int` | Maximum number of bedrooms |
| `baths_min` | `int` | Minimum number of bathrooms. WHOLE baths only -- fractional bounds are rejected because Redfin truncates them |
| `sqft_min` | `int` | Minimum living area in square feet |
| `sqft_max` | `int` | Maximum living area in square feet |
| `lot_size_min` | `int` | Minimum lot size in square feet |
| `year_built_min` | `int` | Earliest year built |
| `year_built_max` | `int` | Latest year built |
| `max_hoa` | `float` | Maximum monthly HOA fee |
| `property_type` | `house\|condo\|townhouse\|multi_family\|land\|other\|co_op` | Property type filter |
| `has_pool` | `bool` | Only return properties with a pool |
| `max_days_on_market` | `int` | Maximum days on market. Cannot be combined with min_days_on_market: Redfin expresses both through one parameter |
| `min_days_on_market` | `int` | Minimum days on market. Cannot be combined with max_days_on_market |

### ScavioRedfinProperty

`scavio_redfin_property` -> `POST /api/v1/redfin/property`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `property_id` | `str` | Redfin property id or any redfin.com listing URL carrying one **(required)** |

### ScavioRedfinMarket

`scavio_redfin_market` -> `POST /api/v1/redfin/market`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `location` | `str` | A redfin.com region URL (/city/, /neighborhood/, /county/, /zipcode/) or a bare 5-digit ZIP. CITY NAMES ARE NOT ACCEPTED |
| `region_id` | `int` | Redfin's own numeric region id. NOT a ZIP code -- different number spaces, and a ZIP here resolves to another city rather than failing. Must be sent together with region_type |
| `region_type` | `1\|2\|5\|6` | What region_id refers to: 1 neighborhood, 2 ZIP, 5 county, 6 city. Must be sent together with region_id |

### ScavioCompaniesHouseSearch

`scavio_companies_house_search` -> `POST /api/v1/companieshouse/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Company name or number. Matches CURRENT AND FORMER names **(required)** |
| `page` | `int` | Result page, 1-50, at 20 rows each. Capped at 50 because the register only serves the first 1000 matches per term whatever hit count it prints. Default: 1 |

### ScavioCompaniesHouseCompany

`scavio_companies_house_company` -> `POST /api/v1/companieshouse/company`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `company_number` | `str` | Company number. Zero-padded and upper-cased for you, so numbers off a letterhead or a spreadsheet that ate leading zeros still resolve. SC, NI, OC, SO, NC, FC, BR and CE prefixes are supported **(required)** |

### ScavioCompaniesHouseOfficers

`scavio_companies_house_officers` -> `POST /api/v1/companieshouse/officers`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `company_number` | `str` | Company number. Zero-padded and upper-cased for you, so numbers off a letterhead or a spreadsheet that ate leading zeros still resolve. SC, NI, OC, SO, NC, FC, BR and CE prefixes are supported **(required)** |
| `page` | `int` | Result page, 1-based, 35 officers per page. No upper bound: past the last page the register answers a plain 200 with an empty list. Default: 1 |

### ScavioCompaniesHouseFilingHistory

`scavio_companies_house_filing_history` -> `POST /api/v1/companieshouse/filing-history`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `company_number` | `str` | Company number. Zero-padded and upper-cased for you, so numbers off a letterhead or a spreadsheet that ate leading zeros still resolve. SC, NI, OC, SO, NC, FC, BR and CE prefixes are supported **(required)** |
| `page` | `int` | Result page, 1-based. No upper bound: past the last page the register answers a plain 200 with an empty list. Default: 1 |

### ScavioG2Search

`scavio_g2_search` -> `POST /api/v1/g2/search`. Costs 5 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Software product or category to search for |
| `page` | `int` | Result page, 1-based. 20 per page unless limit says otherwise |
| `limit` | `int` | Products per page, 1-100. Capped at 100 so one request cannot ask for a multi-megabyte page. Default: 20 |
| `sort` | `relevance\|popular\|alphabetical\|rating` | Sort order for the results. Default: relevance |
| `rating` | `1\|2\|3\|4\|5` | Only return products at or above this star rating |
| `url` | `str` | Full g2.com/search URL, usable instead of query |

### ScavioG2Product

`scavio_g2_product` -> `POST /api/v1/g2/product`. Costs 5 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | `str` | G2 slug (notion) or the numeric G2 id (82623) as a string. Both resolve on the same upstream path |
| `url` | `str` | Full listing URL, usable instead of the id fields |

### ScavioG2Reviews

`scavio_g2_reviews` -> `POST /api/v1/g2/reviews`. Costs 5 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | `str` | G2 slug (notion) or the numeric G2 id (82623) as a string. Both resolve on the same upstream path |
| `url` | `str` | Full listing URL, usable instead of the id fields |
| `page` | `int` | Result page, 1-based. Fixed at 10 reviews per page, and it paginates well past the 10 pages G2's own widget links to |
| `sort` | `relevance\|newest\|most_helpful\|rating_high\|rating_low` | Sort order. Closed set: an unknown value is silently accepted upstream and the sort never runs. Default: relevance |
| `rating` | `1\|2\|3\|4\|5` | Star bucket. HALF-STAR-INCLUSIVE: 1 returns 0, 0.5 and 1-star reviews |
| `company_size` | `small_business\|mid_market\|enterprise` | Reviewer company size: small_business (<=50), mid_market (51-1000), enterprise (>1000) |
| `role` | `user\|administrator\|executive_sponsor\|internal_consultant\|consultant\|agency\|industry_analyst` | Reviewer role filter |
| `region` | `north_america\|europe\|asia\|latin_america\|anz\|middle_east\|africa` | Reviewer region filter |
| `query` | `str` | Full-text search inside the reviews. Narrows the list AND every facet count |

### ScavioCapterraSearch

`scavio_capterra_search` -> `POST /api/v1/capterra/search`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Software product or category. Required unless you pass a url: a term-less search serves a fixed popular-products list that has nothing to do with the caller |
| `url` | `str` | Full capterra.com/search URL. capterra.co.uk and capterra.com.br are accepted |

### ScavioCapterraProduct

`scavio_capterra_product` -> `POST /api/v1/capterra/product`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | `str` | The number in /p/186596/Notion/, as a STRING -- a JSON number is rejected |
| `slug` | `str` | Product slug. Cosmetic here: /p/186596/Zzzjunk/ returns Notion's profile byte for byte |
| `url` | `str` | Full listing URL, usable instead of the id fields |

### ScavioCapterraReviews

`scavio_capterra_reviews` -> `POST /api/v1/capterra/reviews`. Costs 2 credits per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `product_id` | `str` | The number in /p/186596/Notion/, as a STRING -- a JSON number is rejected |
| `slug` | `str` | Product slug. LOAD-BEARING here: it is case-sensitive upstream and a wrong one silently serves PAGE ONE under a billed 200. Pass back the slug from search or product |
| `url` | `str` | Passing back reviews_url from the product endpoint is the reliable way to page |
| `page` | `int` | Result page, 1-100, at 25 reviews each. Past page 100 Capterra answers 200 with page ONE |

### ScavioGoogleAdsAdvertisers

`scavio_google_ads_advertisers` -> `POST /api/v1/googleads/advertisers`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Advertiser name or domain to resolve **(required)** |
| `region` | `str` | ISO alpha-2 country (US, GB, DE) or a Google geo criteria id as a string. It also scopes the deep links on every row. Default: worldwide |
| `limit` | `int` | Rows per arm, 1-20. Advertisers and domains are capped separately, so a name query can return up to twice this many rows. Default: 10 |

### ScavioGoogleAdsSearch

`scavio_google_ads_search` -> `POST /api/v1/googleads/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `domain` | `str` | Advertiser website: bare host, www host or full URL, reduced to the registrable host. This is the ONLY way to get the `domain` field back on each row |
| `advertiser_id` | `str` | Google advertiser id, e.g. AR16735076323512287233. The shape is checked before any request, so a typo costs nothing |
| `region` | `str` | ISO alpha-2 country (US, GB, DE) or a Google geo criteria id as a string. It also scopes the deep links on every row. Default: worldwide |
| `format` | `text\|image\|video` | Creative format. The three sets are DISJOINT -- an advertiser's text, image and video ads share no creatives. Default: all formats |
| `platform` | `play\|maps\|search\|shopping\|youtube` | Surface the ad ran on. Default: all surfaces |
| `topic` | `all\|political` | Ad topic filter. Default: all |
| `limit` | `int` | Creatives per page, 1-100. 100 is a HARD UPSTREAM CEILING, not our policy: Google answers a larger request with ZERO rows rather than an error. Default: 40 |
| `cursor` | `str` | next_cursor from the previous response. Re-send the SAME filters alongside it. Null once the result set is exhausted |

### ScavioGoogleAdsCreative

`scavio_google_ads_creative` -> `POST /api/v1/googleads/creative`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `advertiser_id` | `str` | Google advertiser id owning the creative **(required)** |
| `creative_id` | `str` | Creative id. It must belong to the advertiser_id sent with it -- the lookup is keyed by the pair and a mismatch is a 404 **(required)** |

### ScavioMetaAdsSearch

`scavio_meta_ads_search` -> `POST /api/v1/meta-ads/search`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Keyword, brand or advertiser name to search the library for **(required)** |
| `country` | `str` | Two-letter country code for the ad library storefront. Default: US |
| `active_status` | `all\|active\|inactive` | Whether to return running, stopped or all ads. Default: all |
| `ad_type` | `all\|political_and_issue_ads` | Set to political_and_issue_ads to expose spend, reach, impressions and the paid-for-by disclosure. Commercial ads leave those null. Default: all |
| `media_type` | `all\|image\|video\|meme\|image_and_meme\|none` | Creative media type filter |
| `search_type` | `keyword_unordered\|keyword_exact_phrase` | Whether the query is matched as an exact phrase. Default: keyword_unordered |
| `cursor` | `str` | next_cursor from the previous response. Page 1 is 30 ads, then 10 per page. ALL OTHER FILTERS ARE IGNORED when a cursor is present -- the cursor already carries them |

### ScavioMetaAdsAdvertiser

`scavio_meta_ads_advertiser` -> `POST /api/v1/meta-ads/advertiser`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `page_id` | `str` | The advertiser's numeric Facebook Page id **(required)** |
| `country` | `str` | Two-letter country code for the ad library storefront. Default: US |
| `active_status` | `all\|active\|inactive` | Whether to return running, stopped or all ads. Default: all |
| `ad_type` | `all\|political_and_issue_ads` | Set to political_and_issue_ads to expose spend, reach, impressions and the paid-for-by disclosure. Commercial ads leave those null. Default: all |
| `media_type` | `all\|image\|video\|meme\|image_and_meme\|none` | Creative media type filter |
| `cursor` | `str` | next_cursor from the previous response. Page 1 is 30 ads, then 10 per page |

### ScavioMetaAdsAd

`scavio_meta_ads_ad` -> `POST /api/v1/meta-ads/ad`. Costs 1 credit per call.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ad_archive_id` | `str` | The ad's numeric archive id **(required)** |

### ScavioExtract

`scavio_extract` -> `POST /api/v1/extract`. Costs 1 credit in normal or advanced mode and 2 credits in ultra mode, and is billed only on a successful extraction.

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | Page to read. http(s) only; a bare host is upgraded to https. Loopback, private, link-local and metadata hosts are rejected with a 400 **(required)** |
| `format` | `html\|markdown\|text` | Output format: html is the raw page, markdown is a readability extraction, text is that markdown flattened to plain text. Default: markdown |
| `mode` | `normal\|advanced\|ultra` | Fetch tier and THE PRICE-BEARING PARAMETER: normal and advanced cost 1 credit, ultra costs 2. Escalate only when a plain fetch comes back empty. Default: normal |

## Error Handling

- Empty results raise `ToolException` with actionable suggestions for the LLM
- API errors return `{"error": "message"}` without crashing the agent
- `handle_tool_error=True` ensures LangChain passes errors to the LLM as context

## Architecture

One `BaseTool` subclass per endpoint, each backed by an `APIWrapper` that owns
a single URL. 187 tools over 188 distinct endpoints: `ScavioSearch` alone
covers three Google surfaces via `search_type`, and `ScavioYouTubeMetadata` is a
deprecated alias of `ScavioYouTubeVideo`, so those two share one endpoint.

```
ScavioBaseAPIWrapper                    # Auth, headers, rate limit, sync/async POST
  |
  +-- YouTube     16 tools -> /api/v1/youtube/*      (15 endpoints)
  +-- Kuaishou    14 tools -> /api/v1/kuaishou/*
  +-- Google      12 tools -> /api/v2/google*        (14 endpoints)
  +-- Instagram   12 tools -> /api/v1/instagram/*
  +-- Reddit      12 tools -> /api/v1/reddit/*
  +-- TikTok      11 tools -> /api/v1/tiktok/*
  +-- X           11 tools -> /api/v1/x/*
  +-- LinkedIn     9 tools -> /api/v1/linkedin/*     (9 live; 5 retired, not exposed)
  +-- TikTok Shop  8 tools -> /api/v1/tiktok-shop/*
  +-- Walmart      7 tools -> /api/v1/walmart/*      (seller-products is hyphenated)
  +-- SEC EDGAR    6 tools -> /api/v1/sec/*
  +-- Threads      6 tools -> /api/v1/threads/*
  +-- Target       4 tools -> /api/v1/target/*
  +-- Indeed       4 tools -> /api/v1/indeed/*
  +-- Glassdoor    4 tools -> /api/v1/glassdoor/*
  +-- Tripadvisor  4 tools -> /api/v1/tripadvisor/*
  +-- Cos. House   4 tools -> /api/v1/companieshouse/*  (filing-history is hyphenated)
  +-- Amazon       3 tools -> /api/v1/amazon/*
  +-- eBay         3 tools -> /api/v1/ebay/*
  +-- Home Depot   3 tools -> /api/v1/homedepot/*
  +-- Zillow       3 tools -> /api/v1/zillow/*
  +-- Redfin       3 tools -> /api/v1/redfin/*
  +-- Booking.com  3 tools -> /api/v1/booking/*
  +-- Airbnb       3 tools -> /api/v1/airbnb/*
  +-- Yelp         3 tools -> /api/v1/yelp/*
  +-- App Store    3 tools -> /api/v1/appstore/*
  +-- Google Play  3 tools -> /api/v1/googleplay/*
  +-- G2           3 tools -> /api/v1/g2/*
  +-- Capterra     3 tools -> /api/v1/capterra/*
  +-- Google Ads   3 tools -> /api/v1/googleads/*
  +-- Meta Ads     3 tools -> /api/v1/meta-ads/*     (route key metaads, path meta-ads)
  +-- Extract      1 tool  -> /api/v1/extract        (core, not a platform)
```

Source layout: one module per platform -- `scavio_search.py` (Google),
`scavio_youtube.py`, `scavio_instagram.py`, `scavio_tiktok.py`,
`scavio_tiktok_shop.py`, `scavio_x.py`, `scavio_threads.py`,
`scavio_kuaishou.py`, `scavio_linkedin.py`, `scavio_amazon.py`,
`scavio_walmart.py`, `scavio_target.py`, `scavio_ebay.py`,
`scavio_home_depot.py`, `scavio_zillow.py`, `scavio_redfin.py`,
`scavio_booking.py`, `scavio_airbnb.py`, `scavio_tripadvisor.py`,
`scavio_yelp.py`, `scavio_indeed.py`, `scavio_glassdoor.py`,
`scavio_app_store.py`, `scavio_google_play.py`, `scavio_sec.py`,
`scavio_companies_house.py`, `scavio_g2.py`, `scavio_capterra.py`,
`scavio_google_ads.py`, `scavio_meta_ads.py`, `scavio_reddit.py` and
`scavio_extract.py` -- with every wrapper in `_utilities.py`.

This package owns its **own HTTP stack** (`requests` + `aiohttp`, sliding-window
rate limiting and error translation in `_utilities.py`). It does not depend on
the `scavio` Python SDK, so every URL and parameter here is declared against the
endpoint directly. `tests/test_fanout_platforms.py` pins each of the 93 tools
added or rewritten in 4.0 to its endpoint path and exact parameter set, and
`tests/test_coverage.py` asserts the reachable endpoint set matches the declared
one in both directions.

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
- [Reddit API](https://scavio.dev/reddit-api) — posts, comments, subreddits, redditors, popular and trending feeds
- TikTok Shop, X (formerly Twitter), Threads, Kuaishou, and LinkedIn — product listings, tweets, profiles, company pages, and job listings
- Target, eBay and Home Depot — search, product detail, reviews, sold-price research
- Zillow and Redfin — listings, price history, Zestimates and market statistics
- Booking.com, Airbnb and Tripadvisor — stays priced for real dates, listings, review bodies and rankings
- Yelp, the Apple App Store and Google Play — local businesses, app listings and versioned review streams
- Indeed and Glassdoor — job postings, employer profiles, review sentiment and salary bands
- SEC EDGAR and Companies House — filings, XBRL facts, officers and the UK register
- G2, Capterra, Google Ads Transparency and the Meta Ad Library — software reviews with facets, and the ads competitors are actually running
- Extract — read any URL as Markdown, plain text or raw HTML

Every billable Scavio endpoint has a tool in this package.

Get a free [API key](https://dashboard.scavio.dev) and explore the [documentation](https://scavio.dev/docs/introduction).
