"""Per-endpoint guard for the 23-platform fanout shipped in 4.0.

Each row pins one tool to the endpoint path, request-body parameter set and
sample call it was specified against. The table is deliberately explicit: a
tool that changes its URL, gains a parameter or loses one fails here rather
than shipping green against a backend that moved.
"""

from __future__ import annotations

import json as json_mod
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import responses
from langchain_core.tools import BaseTool

import langchain_scavio
from langchain_scavio._utilities import SCAVIO_API_URL

from .conftest import MOCK_API_KEY

# (tool class name, API wrapper class name, endpoint path, sample body, params)
FANOUT_ENDPOINTS: list[tuple[str, str, str, dict[str, Any], tuple[str, ...]]] = [
    (
        "ScavioWalmartSearch",
        "ScavioWalmartSearchAPIWrapper",
        "/api/v1/walmart/search",
        {"query": "air fryer"},
        (
            "query",
            "start_page",
            "fulfillment_speed",
            "fulfillment_type",
            "domain",
            "page",
            "sort_by",
            "min_price",
            "max_price",
        ),
    ),
    (
        "ScavioWalmartProduct",
        "ScavioWalmartProductAPIWrapper",
        "/api/v1/walmart/product",
        {"product_id": "13544111159"},
        (
            "product_id",
        ),
    ),
    (
        "ScavioWalmartReviews",
        "ScavioWalmartReviewsAPIWrapper",
        "/api/v1/walmart/reviews",
        {"product_id": "13544111159", "page": 2},
        (
            "product_id",
            "page",
            "sort",
        ),
    ),
    (
        "ScavioWalmartCategory",
        "ScavioWalmartCategoryAPIWrapper",
        "/api/v1/walmart/category",
        {"category_id": "1095191"},
        (
            "category_id",
            "limit",
            "fulfillment_speed",
            "domain",
            "page",
            "sort_by",
            "min_price",
            "max_price",
        ),
    ),
    (
        "ScavioWalmartOffers",
        "ScavioWalmartOffersAPIWrapper",
        "/api/v1/walmart/offers",
        {"product_id": "13544111159"},
        (
            "product_id",
        ),
    ),
    (
        "ScavioWalmartSeller",
        "ScavioWalmartSellerAPIWrapper",
        "/api/v1/walmart/seller",
        {"seller_id": "101040442"},
        (
            "seller_id",
        ),
    ),
    (
        "ScavioWalmartSellerProducts",
        "ScavioWalmartSellerProductsAPIWrapper",
        "/api/v1/walmart/seller-products",
        {"seller_id": "101040442"},
        (
            "seller_id",
        ),
    ),
    (
        "ScavioThreadsProfile",
        "ScavioThreadsProfileAPIWrapper",
        "/api/v1/threads/profile",
        {"user_id": "63625256886"},
        (
            "username",
            "user_id",
        ),
    ),
    (
        "ScavioThreadsUserPosts",
        "ScavioThreadsUserPostsAPIWrapper",
        "/api/v1/threads/user/posts",
        {"user_id": "63625256886"},
        (
            "username",
            "user_id",
            "cursor",
        ),
    ),
    (
        "ScavioThreadsUserReplies",
        "ScavioThreadsUserRepliesAPIWrapper",
        "/api/v1/threads/user/replies",
        {"user_id": "63625256886"},
        (
            "username",
            "user_id",
            "cursor",
        ),
    ),
    (
        "ScavioThreadsPost",
        "ScavioThreadsPostAPIWrapper",
        "/api/v1/threads/post",
        {"post_id": "3141592653589793"},
        (
            "post_id",
            "url",
        ),
    ),
    (
        "ScavioThreadsPostComments",
        "ScavioThreadsPostCommentsAPIWrapper",
        "/api/v1/threads/post/comments",
        {"post_id": "3141592653589793"},
        (
            "post_id",
            "cursor",
        ),
    ),
    (
        "ScavioThreadsSearchUsers",
        "ScavioThreadsSearchUsersAPIWrapper",
        "/api/v1/threads/search/users",
        {"query": "langchain"},
        (
            "query",
        ),
    ),
    (
        "ScavioKuaishouProfile",
        "ScavioKuaishouProfileAPIWrapper",
        "/api/v1/kuaishou/profile",
        {"user_id": "3xnmvnpnyzqxqzm"},
        (
            "user_id",
        ),
    ),
    (
        "ScavioKuaishouUserPosts",
        "ScavioKuaishouUserPostsAPIWrapper",
        "/api/v1/kuaishou/user/posts",
        {"user_id": "3xnmvnpnyzqxqzm"},
        (
            "user_id",
            "cursor",
        ),
    ),
    (
        "ScavioKuaishouUserLive",
        "ScavioKuaishouUserLiveAPIWrapper",
        "/api/v1/kuaishou/user/live",
        {"user_id": "3xnmvnpnyzqxqzm"},
        (
            "user_id",
        ),
    ),
    (
        "ScavioKuaishouUserResolve",
        "ScavioKuaishouUserResolveAPIWrapper",
        "/api/v1/kuaishou/user/resolve",
        {"share_link": "https://v.kuaishou.com/abc123"},
        (
            "share_link",
        ),
    ),
    (
        "ScavioKuaishouVideo",
        "ScavioKuaishouVideoAPIWrapper",
        "/api/v1/kuaishou/video",
        {"photo_id": "3xf8v9pmcvexbhi"},
        (
            "photo_id",
            "url",
        ),
    ),
    (
        "ScavioKuaishouVideoComments",
        "ScavioKuaishouVideoCommentsAPIWrapper",
        "/api/v1/kuaishou/video/comments",
        {"photo_id": "3xf8v9pmcvexbhi"},
        (
            "photo_id",
            "cursor",
        ),
    ),
    (
        "ScavioKuaishouCommentReplies",
        "ScavioKuaishouCommentRepliesAPIWrapper",
        "/api/v1/kuaishou/video/sub-comments",
        {"photo_id": "3xf8v9pmcvexbhi", "root_comment_id": "1234567890"},
        (
            "photo_id",
            "root_comment_id",
            "cursor",
            "count",
        ),
    ),
    (
        "ScavioKuaishouVideosBatch",
        "ScavioKuaishouVideosBatchAPIWrapper",
        "/api/v1/kuaishou/videos/batch",
        {"photo_ids": ["3xf8v9pmcvexbhi", "3x8kzxpmn7t6xyq"]},
        (
            "photo_ids",
        ),
    ),
    (
        "ScavioKuaishouSearch",
        "ScavioKuaishouSearchAPIWrapper",
        "/api/v1/kuaishou/search",
        {"keyword": "coffee"},
        (
            "keyword",
            "cursor",
        ),
    ),
    (
        "ScavioKuaishouSearchVideos",
        "ScavioKuaishouSearchVideosAPIWrapper",
        "/api/v1/kuaishou/search/videos",
        {"keyword": "coffee"},
        (
            "keyword",
            "cursor",
        ),
    ),
    (
        "ScavioKuaishouSearchUsers",
        "ScavioKuaishouSearchUsersAPIWrapper",
        "/api/v1/kuaishou/search/users",
        {"keyword": "coffee"},
        (
            "keyword",
            "cursor",
        ),
    ),
    (
        "ScavioKuaishouSearchLive",
        "ScavioKuaishouSearchLiveAPIWrapper",
        "/api/v1/kuaishou/search/live",
        {"keyword": "coffee"},
        (
            "keyword",
            "cursor",
        ),
    ),
    (
        "ScavioKuaishouTagFeed",
        "ScavioKuaishouTagFeedAPIWrapper",
        "/api/v1/kuaishou/tag/feed",
        {"tag": "coffee"},
        (
            "tag",
            "cursor",
        ),
    ),
    (
        "ScavioKuaishouTrending",
        "ScavioKuaishouTrendingAPIWrapper",
        "/api/v1/kuaishou/trending",
        {"board": "hot"},
        (
            "board",
        ),
    ),
    (
        "ScavioEbaySearch",
        "ScavioEbaySearchAPIWrapper",
        "/api/v1/ebay/search",
        {"query": "airpods pro", "sold": True},
        (
            "query",
            "seller",
            "page",
            "sort_by",
            "min_price",
            "max_price",
            "condition",
            "buying_format",
            "free_shipping",
            "sold",
            "category_id",
            "per_page",
        ),
    ),
    (
        "ScavioEbayProduct",
        "ScavioEbayProductAPIWrapper",
        "/api/v1/ebay/product",
        {"item_id": "126544332211"},
        (
            "item_id",
        ),
    ),
    (
        "ScavioEbaySeller",
        "ScavioEbaySellerAPIWrapper",
        "/api/v1/ebay/seller",
        {"seller": "musicmagpie"},
        (
            "seller",
        ),
    ),
    (
        "ScavioTargetSearch",
        "ScavioTargetSearchAPIWrapper",
        "/api/v1/target/search",
        {"keyword": "office chair"},
        (
            "keyword",
            "page",
            "count",
            "sort",
            "store_id",
        ),
    ),
    (
        "ScavioTargetCategory",
        "ScavioTargetCategoryAPIWrapper",
        "/api/v1/target/category",
        {"category_id": "5xtg6"},
        (
            "category_id",
            "page",
            "count",
            "sort",
            "store_id",
        ),
    ),
    (
        "ScavioTargetProduct",
        "ScavioTargetProductAPIWrapper",
        "/api/v1/target/product",
        {"tcin": "87095665"},
        (
            "tcin",
            "store_id",
        ),
    ),
    (
        "ScavioTargetReviews",
        "ScavioTargetReviewsAPIWrapper",
        "/api/v1/target/reviews",
        {"tcin": "87095665"},
        (
            "tcin",
            "limit",
            "store_id",
        ),
    ),
    (
        "ScavioHomeDepotSearch",
        "ScavioHomeDepotSearchAPIWrapper",
        "/api/v1/homedepot/search",
        {"query": "cordless drill"},
        (
            "query",
            "page",
            "sort_by",
            "min_price",
            "max_price",
        ),
    ),
    (
        "ScavioHomeDepotProduct",
        "ScavioHomeDepotProductAPIWrapper",
        "/api/v1/homedepot/product",
        {"item_id": "313021355"},
        (
            "item_id",
        ),
    ),
    (
        "ScavioHomeDepotReviews",
        "ScavioHomeDepotReviewsAPIWrapper",
        "/api/v1/homedepot/reviews",
        {"item_id": "313021355", "page": 2},
        (
            "item_id",
            "page",
        ),
    ),
    (
        "ScavioZillowSearch",
        "ScavioZillowSearchAPIWrapper",
        "/api/v1/zillow/search",
        {"location": "Austin, TX", "listing_status": "for_sale"},
        (
            "location",
            "listing_status",
            "page",
            "sort",
            "min_price",
            "max_price",
            "beds_min",
            "beds_max",
            "baths_min",
            "baths_max",
            "sqft_min",
            "sqft_max",
            "lot_size_min",
            "lot_size_max",
            "year_built_min",
            "year_built_max",
            "max_hoa",
            "home_type",
            "days_on_zillow",
            "keywords",
            "has_pool",
            "has_garage",
            "has_air_conditioning",
            "is_waterfront",
            "has_basement",
            "is_new_construction",
            "has_open_house",
            "price_reduced",
            "is_3d_tour",
        ),
    ),
    (
        "ScavioZillowProperty",
        "ScavioZillowPropertyAPIWrapper",
        "/api/v1/zillow/property",
        {"zpid": "29444874"},
        (
            "zpid",
        ),
    ),
    (
        "ScavioZillowAgentReviews",
        "ScavioZillowAgentReviewsAPIWrapper",
        "/api/v1/zillow/reviews",
        {"screen_name": "jane-smith"},
        (
            "screen_name",
        ),
    ),
    (
        "ScavioBookingSearch",
        "ScavioBookingSearchAPIWrapper",
        "/api/v1/booking/search",
        {"destination": "Lisbon", "checkin": "2026-09-10", "checkout": "2026-09-13"},
        (
            "destination",
            "dest_id",
            "dest_type",
            "page",
            "sort_by",
            "min_price",
            "max_price",
            "stars",
            "min_review_score",
            "property_type",
            "free_cancellation",
            "no_prepayment",
            "breakfast_included",
            "checkin",
            "checkout",
            "adults",
            "children_ages",
            "rooms",
            "currency",
        ),
    ),
    (
        "ScavioBookingHotel",
        "ScavioBookingHotelAPIWrapper",
        "/api/v1/booking/hotel",
        {"hotel": "memmo-alfama", "checkin": "2026-09-10", "checkout": "2026-09-13"},
        (
            "hotel",
            "country_code",
            "checkin",
            "checkout",
            "adults",
            "children_ages",
            "rooms",
            "currency",
        ),
    ),
    (
        "ScavioBookingReviews",
        "ScavioBookingReviewsAPIWrapper",
        "/api/v1/booking/reviews",
        {"hotel": "memmo-alfama"},
        (
            "hotel",
            "country_code",
            "checkin",
            "checkout",
            "adults",
            "children_ages",
            "rooms",
            "currency",
        ),
    ),
    (
        "ScavioTripadvisorLocations",
        "ScavioTripadvisorLocationsAPIWrapper",
        "/api/v1/tripadvisor/locations",
        {"query": "Franklin Barbecue"},
        (
            "query",
            "limit",
        ),
    ),
    (
        "ScavioTripadvisorSearch",
        "ScavioTripadvisorSearchAPIWrapper",
        "/api/v1/tripadvisor/search",
        {"geo_id": "30196", "category": "restaurants"},
        (
            "geo_id",
            "category",
            "page",
            "url",
        ),
    ),
    (
        "ScavioTripadvisorLocation",
        "ScavioTripadvisorLocationAPIWrapper",
        "/api/v1/tripadvisor/location",
        {"location_id": "1899234", "geo_id": "30196"},
        (
            "location_id",
            "geo_id",
            "category",
            "url",
        ),
    ),
    (
        "ScavioTripadvisorReviews",
        "ScavioTripadvisorReviewsAPIWrapper",
        "/api/v1/tripadvisor/reviews",
        {"location_id": "1899234", "geo_id": "30196", "page": 2},
        (
            "location_id",
            "geo_id",
            "category",
            "url",
            "page",
        ),
    ),
    (
        "ScavioIndeedSearch",
        "ScavioIndeedSearchAPIWrapper",
        "/api/v1/indeed/search",
        {"query": "data engineer", "location": "Austin, TX"},
        (
            "query",
            "location",
            "page",
            "radius",
            "max_age_days",
            "job_type",
            "min_salary",
            "remote",
        ),
    ),
    (
        "ScavioIndeedJob",
        "ScavioIndeedJobAPIWrapper",
        "/api/v1/indeed/job",
        {"job_id": "a1b2c3d4e5f60718"},
        (
            "job_id",
        ),
    ),
    (
        "ScavioIndeedCompany",
        "ScavioIndeedCompanyAPIWrapper",
        "/api/v1/indeed/company",
        {"company": "Stripe"},
        (
            "company",
        ),
    ),
    (
        "ScavioIndeedCompanyReviews",
        "ScavioIndeedCompanyReviewsAPIWrapper",
        "/api/v1/indeed/company/reviews",
        {"company": "Stripe", "page": 2},
        (
            "company",
            "page",
        ),
    ),
    (
        "ScavioAirbnbSearch",
        "ScavioAirbnbSearchAPIWrapper",
        "/api/v1/airbnb/search",
        {"location": "Lisbon", "check_in": "2026-09-10", "check_out": "2026-09-15"},
        (
            "location",
            "check_in",
            "check_out",
            "adults",
            "children",
            "infants",
            "pets",
            "min_price",
            "max_price",
            "room_type",
            "min_bedrooms",
            "min_beds",
            "min_bathrooms",
            "superhost",
            "instant_book",
            "guest_favorite",
            "free_cancellation",
            "amenities",
            "currency",
            "page",
            "cursor",
        ),
    ),
    (
        "ScavioAirbnbListing",
        "ScavioAirbnbListingAPIWrapper",
        "/api/v1/airbnb/listing",
        {"listing_id": "12345678"},
        (
            "listing_id",
            "check_in",
            "check_out",
            "adults",
            "children",
            "infants",
            "pets",
            "currency",
        ),
    ),
    (
        "ScavioAirbnbReviews",
        "ScavioAirbnbReviewsAPIWrapper",
        "/api/v1/airbnb/reviews",
        {"listing_id": "12345678", "limit": 30},
        (
            "listing_id",
            "currency",
            "limit",
            "offset",
        ),
    ),
    (
        "ScavioGlassdoorCompanies",
        "ScavioGlassdoorCompaniesAPIWrapper",
        "/api/v1/glassdoor/companies",
        {"query": "Stripe"},
        (
            "query",
        ),
    ),
    (
        "ScavioGlassdoorCompany",
        "ScavioGlassdoorCompanyAPIWrapper",
        "/api/v1/glassdoor/company",
        {"employer_id": "1699"},
        (
            "employer_id",
            "company",
            "url",
        ),
    ),
    (
        "ScavioGlassdoorReviews",
        "ScavioGlassdoorReviewsAPIWrapper",
        "/api/v1/glassdoor/reviews",
        {"employer_id": "1699"},
        (
            "employer_id",
            "company",
            "url",
            "category",
            "employment_status",
        ),
    ),
    (
        "ScavioGlassdoorSalaries",
        "ScavioGlassdoorSalariesAPIWrapper",
        "/api/v1/glassdoor/salaries",
        {"employer_id": "1699"},
        (
            "employer_id",
            "company",
            "url",
            "page",
        ),
    ),
    (
        "ScavioYelpSearch",
        "ScavioYelpSearchAPIWrapper",
        "/api/v1/yelp/search",
        {"term": "coffee", "location": "Austin, TX"},
        (
            "term",
            "location",
            "page",
            "sort",
            "price",
            "open_now",
            "attributes",
            "url",
        ),
    ),
    (
        "ScavioYelpBusiness",
        "ScavioYelpBusinessAPIWrapper",
        "/api/v1/yelp/business",
        {"business_id": "desnudo-coffee-austin-2"},
        (
            "business_id",
            "url",
        ),
    ),
    (
        "ScavioYelpReviews",
        "ScavioYelpReviewsAPIWrapper",
        "/api/v1/yelp/reviews",
        {"business_id": "desnudo-coffee-austin-2", "page": 2},
        (
            "business_id",
            "url",
            "page",
            "sort",
            "rating",
        ),
    ),
    (
        "ScavioAppStoreSearch",
        "ScavioAppStoreSearchAPIWrapper",
        "/api/v1/appstore/search",
        {"term": "notion", "limit": 25},
        (
            "term",
            "limit",
            "country",
            "entity",
            "lang",
        ),
    ),
    (
        "ScavioAppStoreApp",
        "ScavioAppStoreAppAPIWrapper",
        "/api/v1/appstore/app",
        {"app_id": "1232780281"},
        (
            "app_id",
            "country",
        ),
    ),
    (
        "ScavioAppStoreReviews",
        "ScavioAppStoreReviewsAPIWrapper",
        "/api/v1/appstore/reviews",
        {"app_id": "1232780281", "page": 1},
        (
            "app_id",
            "country",
            "page",
            "sort",
        ),
    ),
    (
        "ScavioGooglePlaySearch",
        "ScavioGooglePlaySearchAPIWrapper",
        "/api/v1/googleplay/search",
        {"query": "notion"},
        (
            "query",
            "hl",
            "gl",
        ),
    ),
    (
        "ScavioGooglePlayApp",
        "ScavioGooglePlayAppAPIWrapper",
        "/api/v1/googleplay/app",
        {"app_id": "notion.id"},
        (
            "app_id",
            "hl",
            "gl",
        ),
    ),
    (
        "ScavioGooglePlayReviews",
        "ScavioGooglePlayReviewsAPIWrapper",
        "/api/v1/googleplay/reviews",
        {"app_id": "notion.id"},
        (
            "app_id",
            "sort",
            "count",
            "cursor",
            "hl",
            "gl",
        ),
    ),
    (
        "ScavioSECLookup",
        "ScavioSECLookupAPIWrapper",
        "/api/v1/sec/lookup",
        {"query": "AAPL"},
        (
            "query",
            "limit",
            "exchange",
        ),
    ),
    (
        "ScavioSECCompany",
        "ScavioSECCompanyAPIWrapper",
        "/api/v1/sec/company",
        {"ticker": "AAPL"},
        (
            "cik",
            "ticker",
        ),
    ),
    (
        "ScavioSECFilings",
        "ScavioSECFilingsAPIWrapper",
        "/api/v1/sec/filings",
        {"ticker": "AAPL", "form": "10-K"},
        (
            "cik",
            "ticker",
            "form",
            "date_from",
            "date_to",
            "page",
            "limit",
            "include_history",
        ),
    ),
    (
        "ScavioSECConcept",
        "ScavioSECConceptAPIWrapper",
        "/api/v1/sec/concept",
        {"ticker": "AAPL", "concept": "NetIncomeLoss"},
        (
            "cik",
            "ticker",
            "concept",
            "taxonomy",
            "unit",
            "form",
            "limit",
        ),
    ),
    (
        "ScavioSECFacts",
        "ScavioSECFactsAPIWrapper",
        "/api/v1/sec/facts",
        {"ticker": "AAPL", "query": "revenue"},
        (
            "cik",
            "ticker",
            "taxonomy",
            "query",
            "limit",
        ),
    ),
    (
        "ScavioSECSearch",
        "ScavioSECSearchAPIWrapper",
        "/api/v1/sec/search",
        {"query": "climate risk", "form": "10-K"},
        (
            "query",
            "cik",
            "ticker",
            "form",
            "date_from",
            "date_to",
            "location",
            "sort",
            "page",
        ),
    ),
    (
        "ScavioRedfinSearch",
        "ScavioRedfinSearchAPIWrapper",
        "/api/v1/redfin/search",
        {"location": "https://www.redfin.com/city/30749/TX/Austin"},
        (
            "location",
            "region_id",
            "region_type",
            "listing_status",
            "sold_within_days",
            "page",
            "limit",
            "sort",
            "min_price",
            "max_price",
            "beds_min",
            "beds_max",
            "baths_min",
            "sqft_min",
            "sqft_max",
            "lot_size_min",
            "year_built_min",
            "year_built_max",
            "max_hoa",
            "property_type",
            "has_pool",
            "max_days_on_market",
            "min_days_on_market",
        ),
    ),
    (
        "ScavioRedfinProperty",
        "ScavioRedfinPropertyAPIWrapper",
        "/api/v1/redfin/property",
        {"property_id": "170072526"},
        (
            "property_id",
        ),
    ),
    (
        "ScavioRedfinMarket",
        "ScavioRedfinMarketAPIWrapper",
        "/api/v1/redfin/market",
        {"location": "https://www.redfin.com/city/30749/TX/Austin"},
        (
            "location",
            "region_id",
            "region_type",
        ),
    ),
    (
        "ScavioCompaniesHouseSearch",
        "ScavioCompaniesHouseSearchAPIWrapper",
        "/api/v1/companieshouse/search",
        {"query": "Monzo"},
        (
            "query",
            "page",
        ),
    ),
    (
        "ScavioCompaniesHouseCompany",
        "ScavioCompaniesHouseCompanyAPIWrapper",
        "/api/v1/companieshouse/company",
        {"company_number": "09446231"},
        (
            "company_number",
        ),
    ),
    (
        "ScavioCompaniesHouseOfficers",
        "ScavioCompaniesHouseOfficersAPIWrapper",
        "/api/v1/companieshouse/officers",
        {"company_number": "09446231"},
        (
            "company_number",
            "page",
        ),
    ),
    (
        "ScavioCompaniesHouseFilingHistory",
        "ScavioCompaniesHouseFilingHistoryAPIWrapper",
        "/api/v1/companieshouse/filing-history",
        {"company_number": "09446231"},
        (
            "company_number",
            "page",
        ),
    ),
    (
        "ScavioG2Search",
        "ScavioG2SearchAPIWrapper",
        "/api/v1/g2/search",
        {"query": "project management"},
        (
            "query",
            "page",
            "limit",
            "sort",
            "rating",
            "url",
        ),
    ),
    (
        "ScavioG2Product",
        "ScavioG2ProductAPIWrapper",
        "/api/v1/g2/product",
        {"product_id": "notion"},
        (
            "product_id",
            "url",
        ),
    ),
    (
        "ScavioG2Reviews",
        "ScavioG2ReviewsAPIWrapper",
        "/api/v1/g2/reviews",
        {"product_id": "notion", "page": 2},
        (
            "product_id",
            "url",
            "page",
            "sort",
            "rating",
            "company_size",
            "role",
            "region",
            "query",
        ),
    ),
    (
        "ScavioCapterraSearch",
        "ScavioCapterraSearchAPIWrapper",
        "/api/v1/capterra/search",
        {"query": "project management"},
        (
            "query",
            "url",
        ),
    ),
    (
        "ScavioCapterraProduct",
        "ScavioCapterraProductAPIWrapper",
        "/api/v1/capterra/product",
        {"product_id": "186596"},
        (
            "product_id",
            "slug",
            "url",
        ),
    ),
    (
        "ScavioCapterraReviews",
        "ScavioCapterraReviewsAPIWrapper",
        "/api/v1/capterra/reviews",
        {"product_id": "186596", "slug": "Notion", "page": 2},
        (
            "product_id",
            "slug",
            "url",
            "page",
        ),
    ),
    (
        "ScavioGoogleAdsAdvertisers",
        "ScavioGoogleAdsAdvertisersAPIWrapper",
        "/api/v1/googleads/advertisers",
        {"query": "Stripe"},
        (
            "query",
            "region",
            "limit",
        ),
    ),
    (
        "ScavioGoogleAdsSearch",
        "ScavioGoogleAdsSearchAPIWrapper",
        "/api/v1/googleads/search",
        {"domain": "stripe.com", "region": "GB"},
        (
            "domain",
            "advertiser_id",
            "region",
            "format",
            "platform",
            "topic",
            "limit",
            "cursor",
        ),
    ),
    (
        "ScavioGoogleAdsCreative",
        "ScavioGoogleAdsCreativeAPIWrapper",
        "/api/v1/googleads/creative",
        {"advertiser_id": "AR16735076323512287233", "creative_id": "CR12345678901234567890"},
        (
            "advertiser_id",
            "creative_id",
        ),
    ),
    (
        "ScavioMetaAdsSearch",
        "ScavioMetaAdsSearchAPIWrapper",
        "/api/v1/meta-ads/search",
        {"query": "running shoes", "country": "GB"},
        (
            "query",
            "country",
            "active_status",
            "ad_type",
            "media_type",
            "search_type",
            "cursor",
        ),
    ),
    (
        "ScavioMetaAdsAdvertiser",
        "ScavioMetaAdsAdvertiserAPIWrapper",
        "/api/v1/meta-ads/advertiser",
        {"page_id": "20531316728"},
        (
            "page_id",
            "country",
            "active_status",
            "ad_type",
            "media_type",
            "cursor",
        ),
    ),
    (
        "ScavioMetaAdsAd",
        "ScavioMetaAdsAdAPIWrapper",
        "/api/v1/meta-ads/ad",
        {"ad_archive_id": "1234567890123456"},
        (
            "ad_archive_id",
        ),
    ),
    (
        "ScavioExtract",
        "ScavioExtractAPIWrapper",
        "/api/v1/extract",
        {"url": "https://example.com/article", "format": "markdown"},
        (
            "url",
            "format",
            "mode",
        ),
    ),
]

IDS = [row[0] for row in FANOUT_ENDPOINTS]


def _ok_payload() -> dict[str, Any]:
    """A minimal 200 envelope that satisfies every tool's empty-data check."""
    return {
        "data": {"products": [{"id": "1"}], "results": [{"id": "1"}]},
        "response_time": 0.42,
        "credits_used": 1,
        "credits_remaining": 999,
    }


@pytest.mark.parametrize(
    "cls_name,wrapper_name,path,sample,params", FANOUT_ENDPOINTS, ids=IDS
)
class TestFanoutEndpoint:
    def test_targets_the_frozen_path(
        self,
        cls_name: str,
        wrapper_name: str,
        path: str,
        sample: dict[str, Any],
        params: tuple[str, ...],
    ) -> None:
        tool = getattr(langchain_scavio, cls_name)(scavio_api_key=MOCK_API_KEY)
        assert isinstance(tool, BaseTool)
        assert tool.api_wrapper._build_url() == f"{SCAVIO_API_URL}{path}"
        assert type(tool.api_wrapper).__name__ == wrapper_name

    def test_exposes_exactly_the_endpoint_parameters(
        self,
        cls_name: str,
        wrapper_name: str,
        path: str,
        sample: dict[str, Any],
        params: tuple[str, ...],
    ) -> None:
        tool = getattr(langchain_scavio, cls_name)(scavio_api_key=MOCK_API_KEY)
        assert set(tool.args_schema.model_fields) == set(params)

    def test_states_its_credit_cost(
        self,
        cls_name: str,
        wrapper_name: str,
        path: str,
        sample: dict[str, Any],
        params: tuple[str, ...],
    ) -> None:
        tool = getattr(langchain_scavio, cls_name)(scavio_api_key=MOCK_API_KEY)
        assert "credit" in tool.description.lower()

    @responses.activate
    def test_posts_the_sample_body_to_the_endpoint(
        self,
        cls_name: str,
        wrapper_name: str,
        path: str,
        sample: dict[str, Any],
        params: tuple[str, ...],
    ) -> None:
        tool = getattr(langchain_scavio, cls_name)(scavio_api_key=MOCK_API_KEY)
        responses.add(
            responses.POST,
            f"{SCAVIO_API_URL}{path}",
            json=_ok_payload(),
            status=200,
        )
        result = tool.invoke(dict(sample))
        assert "data" in result
        body = json_mod.loads(responses.calls[0].request.body)
        assert body == sample

    @responses.activate
    def test_empty_data_raises_a_tool_exception(
        self,
        cls_name: str,
        wrapper_name: str,
        path: str,
        sample: dict[str, Any],
        params: tuple[str, ...],
    ) -> None:
        tool = getattr(langchain_scavio, cls_name)(scavio_api_key=MOCK_API_KEY)
        responses.add(
            responses.POST,
            f"{SCAVIO_API_URL}{path}",
            json={"data": None, "response_time": 0.1},
            status=200,
        )
        result = tool.invoke(dict(sample))
        assert isinstance(result, str)
        assert result.startswith("No ")

    @pytest.mark.asyncio
    async def test_async_path_uses_the_same_wrapper(
        self,
        cls_name: str,
        wrapper_name: str,
        path: str,
        sample: dict[str, Any],
        params: tuple[str, ...],
    ) -> None:
        tool = getattr(langchain_scavio, cls_name)(scavio_api_key=MOCK_API_KEY)
        target = f"langchain_scavio._utilities.{wrapper_name}.raw_results_async"
        with patch(target, new_callable=AsyncMock, return_value=_ok_payload()):
            result = await tool.ainvoke(dict(sample))
        assert "data" in result
