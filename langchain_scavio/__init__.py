"""LangChain integration for Scavio Search API."""

from importlib import metadata

from langchain_scavio.scavio_search import ScavioSearch

try:
    __version__: str = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "ScavioSearch",
    "__version__",
]
