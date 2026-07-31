"""LangChain integration for Scavio Search API."""

from importlib import metadata

from langchain_scavio.scavio_amazon import (
    ScavioAmazonOffers,
    ScavioAmazonProduct,
    ScavioAmazonSearch,
)
from langchain_scavio.scavio_instagram import (
    ScavioInstagramCommentReplies,
    ScavioInstagramPost,
    ScavioInstagramPostComments,
    ScavioInstagramProfile,
    ScavioInstagramSearchHashtags,
    ScavioInstagramSearchUsers,
    ScavioInstagramStories,
    ScavioInstagramTaggedPosts,
    ScavioInstagramUserFollowers,
    ScavioInstagramUserFollowings,
    ScavioInstagramUserPosts,
    ScavioInstagramUserReels,
)
from langchain_scavio.scavio_reddit import ScavioRedditPost, ScavioRedditSearch
from langchain_scavio.scavio_search import ScavioSearch
from langchain_scavio.scavio_tiktok import (
    ScavioTikTokCommentReplies,
    ScavioTikTokHashtag,
    ScavioTikTokHashtagVideos,
    ScavioTikTokProfile,
    ScavioTikTokSearchUsers,
    ScavioTikTokSearchVideos,
    ScavioTikTokUserFollowers,
    ScavioTikTokUserFollowings,
    ScavioTikTokUserPosts,
    ScavioTikTokVideo,
    ScavioTikTokVideoComments,
)
from langchain_scavio.scavio_tiktok_shop import (
    ScavioTikTokShopCategories,
    ScavioTikTokShopCategoryProducts,
    ScavioTikTokShopProduct,
    ScavioTikTokShopProductReviews,
    ScavioTikTokShopResolve,
    ScavioTikTokShopSearch,
    ScavioTikTokShopSearchSuggestions,
    ScavioTikTokShopShopProducts,
)
from langchain_scavio.scavio_walmart import ScavioWalmartProduct, ScavioWalmartSearch
from langchain_scavio.scavio_youtube import (
    ScavioYouTubeChannel,
    ScavioYouTubeChannelVideos,
    ScavioYouTubeComments,
    ScavioYouTubeMetadata,
    ScavioYouTubeSearch,
    ScavioYouTubeStreams,
    ScavioYouTubeTranscript,
    ScavioYouTubeVideo,
)

try:
    __version__: str = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "ScavioSearch",
    "ScavioAmazonSearch",
    "ScavioAmazonProduct",
    "ScavioAmazonOffers",
    "ScavioWalmartSearch",
    "ScavioWalmartProduct",
    "ScavioYouTubeSearch",
    "ScavioYouTubeMetadata",
    "ScavioYouTubeVideo",
    "ScavioYouTubeComments",
    "ScavioYouTubeTranscript",
    "ScavioYouTubeChannel",
    "ScavioYouTubeChannelVideos",
    "ScavioYouTubeStreams",
    "ScavioRedditSearch",
    "ScavioRedditPost",
    "ScavioTikTokProfile",
    "ScavioTikTokUserPosts",
    "ScavioTikTokVideo",
    "ScavioTikTokVideoComments",
    "ScavioTikTokCommentReplies",
    "ScavioTikTokSearchVideos",
    "ScavioTikTokSearchUsers",
    "ScavioTikTokHashtag",
    "ScavioTikTokHashtagVideos",
    "ScavioTikTokUserFollowers",
    "ScavioTikTokUserFollowings",
    "ScavioTikTokShopSearch",
    "ScavioTikTokShopSearchSuggestions",
    "ScavioTikTokShopProduct",
    "ScavioTikTokShopProductReviews",
    "ScavioTikTokShopCategories",
    "ScavioTikTokShopCategoryProducts",
    "ScavioTikTokShopShopProducts",
    "ScavioTikTokShopResolve",
    "ScavioInstagramProfile",
    "ScavioInstagramUserPosts",
    "ScavioInstagramUserReels",
    "ScavioInstagramTaggedPosts",
    "ScavioInstagramStories",
    "ScavioInstagramPost",
    "ScavioInstagramPostComments",
    "ScavioInstagramCommentReplies",
    "ScavioInstagramSearchUsers",
    "ScavioInstagramSearchHashtags",
    "ScavioInstagramUserFollowers",
    "ScavioInstagramUserFollowings",
    "__version__",
]
