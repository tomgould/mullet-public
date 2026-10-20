"""Source connectors for automated document acquisition."""

from .base import BaseSourceConnector, SyncResult
from .github_connector import GitHubSourceConnector
from .github_models import GitHubSourceConfig, RepoInfo
from .web_crawler import WebSourceConfig, WebSourceConnector

__all__ = [
    "BaseSourceConnector",
    "GitHubSourceConfig",
    "GitHubSourceConnector",
    "RepoInfo",
    "SyncResult",
    "WebSourceConfig",
    "WebSourceConnector",
]
