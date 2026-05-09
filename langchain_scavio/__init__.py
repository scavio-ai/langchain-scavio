"""LangChain integration for Scavio Search API."""

from importlib import metadata

from langchain_scavio.scavio_amazon import ScavioAmazonProduct, ScavioAmazonSearch
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
from langchain_scavio.scavio_walmart import ScavioWalmartProduct, ScavioWalmartSearch
from langchain_scavio.scavio_youtube import (
    ScavioYouTubeMetadata,
    ScavioYouTubeSearch,
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
    "ScavioWalmartSearch",
    "ScavioWalmartProduct",
    "ScavioYouTubeSearch",
    "ScavioYouTubeMetadata",
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
    "__version__",
]
