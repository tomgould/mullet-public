"""
Client configuration and loading.

Each client has:
- config.yaml: Configuration
- documents/: Source documents
- index/: Vector DB and BM25 index
- prompts/: Optional custom system prompts
"""

import difflib
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from mullet.knowledge.engine.logging_config import AuditConfig, load_audit_config
from mullet.knowledge.code import CodeChunker
from mullet.knowledge.engine.reference_config import ReferenceConfig, load_reference_config

logger = logging.getLogger(__name__)

# Default document extensions for ingestion
DEFAULT_DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".html", ".txt", ".md", ".mbox")

# Well-known extension-less files that should be ingestible as documents
# (Jenkinsfile, Dockerfile, Makefile, etc.). Pools opt in via
# documents.filenames in config.yaml; nothing is added implicitly.
DEFAULT_DOCUMENT_FILENAMES: tuple = ()

# Default client configuration
DEFAULT_CONFIG = {
    "name": "Default RAG",
    "machine_name": "default",
    "paths": {
        "documents": "documents",
        "index": "index",
        "prompts": "prompts",
    },
    "documents": {
        "extensions": list(DEFAULT_DOCUMENT_EXTENSIONS),
        "exclude_extensions": [],  # e.g., [".csv", ".xlsx"] to skip spreadsheets
    },
    "embedding": {
        "model": "all-MiniLM-L6-v2",  # sentence-transformers model name
        "dimension": 384,  # Embedding dimension (must match model)
        "query_prefix": "",  # Prefix for queries (e.g., "query: " for E5 models)
        "passage_prefix": "",  # Prefix for passages (e.g., "passage: " for E5 models)
        "contextualize": False,  # Prepend document context to chunks before embedding
        "hnsw_ef_construction": 200,  # HNSW build-time: neighbor list size during construction
        "hnsw_max_neighbors": 16,     # HNSW build-time: max connections per node (M)
        "hnsw_ef_search": 100,        # HNSW query-time: neighbor list size during search
    },
    "search": {
        "bm25_weight": 0.5,
        "vector_weight": 0.5,
        "top_k": 50,
        "rerank_top_n": 50,
        "min_score": None,  # No filtering by default; use -2.0 for strict compliance filtering
        "tokenization": {
            "stemmer": None,  # "porter", "snowball", or None
            "synonyms": {},  # e.g., {"cdd": ["customer due diligence", "KYC"]}
        },
        # Adaptive reranking settings
        "adaptive_rerank": True,  # Reduce rerank depth on high-confidence matches
        "adaptive_rerank_threshold": 0.9,  # Combined score threshold to trigger
        "adaptive_rerank_depth": 15,  # Reduced rerank depth when triggered
        # Merge strategy
        "merge_strategy": "weighted",  # "weighted" or "rrf"
        "rrf_k": 60,  # RRF constant (higher = less impact of rank position)
        # MMR diversity
        "mmr_lambda": None,  # None=disabled, 0.0-1.0 enabled (1.0=pure relevance, 0.0=pure diversity)
        # Code priority settings
        "code_priority": False,  # Boost code chunks above docs (code is source of truth)
        "code_priority_boost": 0.5,  # Score boost for code chunks when code_priority=True
        "cross_reference_code": False,  # Auto-fetch related code when docs are found
        "cross_reference_top_k": 3,  # Number of related code chunks to fetch per doc
        # Reranker backend
        "reranker": "local",  # "local" or "api"
        "reranker_model": None,  # None = use default cross-encoder
        "reranker_endpoint": None,  # API endpoint URL (required for "api" backend)
        # BM25 parameter tuning
        "bm25_k1": 1.5,  # BM25 term frequency saturation (>0, higher = more weight to term frequency)
        "bm25_b": 0.75,  # BM25 document length normalization (0-1, higher = more penalty for long docs)
        # Score normalization
        "score_normalization": "minmax",  # "minmax", "zscore", or "raw"
        # HyDE (Hypothetical Document Embeddings)
        "hyde": False,  # Generate hypothetical answer via LLM for better vector search
        # Multi-query / RAG-Fusion
        "multi_query": False,  # Generate query variants and merge via RRF
        "multi_query_count": 3,  # Number of query variants to generate (2-7)
        # Search result caching
        "cache_ttl": 300,        # Seconds before cache entries expire (0 = disabled)
        "cache_max_size": 128,   # Max cached queries per client
    },
    "llm": {
        "provider": "lmstudio",
        "max_context_chars": 24000,
        "max_chunks": 50,
        "context_filter": False,  # Drop chunks below rerank score threshold before LLM
        "context_filter_threshold": 0.1,  # Minimum rerank score for context filtering
        # Token budgeting ratios (proportional allocation by chunk category)
        "doc_budget_ratio": 0.6,   # Budget share for document chunks
        "code_budget_ratio": 0.3,  # Budget share for code chunks
        "reference_budget_ratio": 0.1,  # Budget share for reference chunks
        "citations": False,  # Enable inline [1] citation format
    },
    "code": {
        "enabled": False,
        "paths": [],
        "languages": {
            "python": {"extensions": [".py"], "parser": "tree-sitter"},
            "javascript": {"extensions": [".js", ".jsx", ".mjs"], "parser": "tree-sitter"},
            "typescript": {"extensions": [".ts", ".tsx"], "parser": "tree-sitter"},
            "rust": {"extensions": [".rs"], "parser": "tree-sitter"},
            "go": {"extensions": [".go"], "parser": "tree-sitter"},
            "java": {"extensions": [".java"], "parser": "tree-sitter"},
            "cpp": {"extensions": [".cpp", ".cc", ".cxx", ".h", ".hpp"], "parser": "tree-sitter"},
            "c": {"extensions": [".c", ".h"], "parser": "tree-sitter"},
        },
        "parallel_workers": 4,
        "max_file_size_kb": 500,
        "git": {
            "detect_repos": True,
            "follow_symlinks": True,
            "exclude_branches": [],
        },
    },
}


# Known top-level config keys
_KNOWN_TOP_LEVEL_KEYS = {
    "name", "machine_name", "description", "paths", "documents",
    "embedding", "search", "llm", "code", "source_sync", "audit",
    "references", "chunking", "graph", "sources",
}

# Known code config keys
_KNOWN_CODE_KEYS = {
    "enabled", "paths", "languages", "parallel_workers",
    "max_file_size_kb", "git", "exclude_patterns",
}

# Known code.git config keys
_KNOWN_CODE_GIT_KEYS = {
    "detect_repos", "follow_symlinks", "exclude_branches",
}

# Known graph config keys
_KNOWN_GRAPH_KEYS = {
    "enabled", "entity_types", "extract_batch_size",
    "max_description_chars", "search_mode",
    "extract_workers", "extract_save_every", "extract_text_budget",
    "exclude_extensions",
    "embed_batch_size", "inter_batch_sleep",
}

# Known search config keys
_KNOWN_SEARCH_KEYS = {
    "bm25_weight", "vector_weight", "top_k", "rerank_top_n", "min_score",
    "tokenization", "adaptive_rerank", "adaptive_rerank_threshold",
    "adaptive_rerank_depth", "merge_strategy", "rrf_k", "mmr_lambda",
    "code_priority", "code_priority_boost", "cross_reference_code",
    "cross_reference_top_k",
    "reranker", "reranker_model", "reranker_endpoint",
    "bm25_k1", "bm25_b",
    "score_normalization",
    "hyde", "multi_query", "multi_query_count",
    "cache_ttl", "cache_max_size",
}

# Known embedding model dimensions
_KNOWN_EMBEDDING_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}


def validate_config(config: dict) -> list[str]:
    """
    Validate client config against expected schema.

    Returns list of warnings (empty if config is valid).
    Does not raise — invalid configs still load with warnings.
    """
    warnings = []

    # Required fields
    if not config.get("machine_name"):
        warnings.append("Missing required field: 'machine_name'")
    elif len(config["machine_name"]) < 3:
        warnings.append(
            f"machine_name '{config['machine_name']}' is too short "
            f"(min 3 chars, required by ChromaDB collection naming)"
        )

    # Check for unknown top-level keys
    for key in config:
        if key not in _KNOWN_TOP_LEVEL_KEYS:
            warnings.append(f"Unknown top-level key: '{key}'")

    # Validate search config
    search_cfg = config.get("search", {})
    if isinstance(search_cfg, dict):
        for key in search_cfg:
            if key not in _KNOWN_SEARCH_KEYS:
                warnings.append(f"Unknown search key: 'search.{key}'")

        # Type and range checks
        for field, expected_type in [
            ("bm25_weight", (int, float)),
            ("vector_weight", (int, float)),
            ("top_k", int),
            ("rerank_top_n", int),
            ("rrf_k", int),
            ("adaptive_rerank_depth", int),
        ]:
            val = search_cfg.get(field)
            if val is not None and not isinstance(val, expected_type):
                warnings.append(
                    f"search.{field} should be {expected_type.__name__ if isinstance(expected_type, type) else 'numeric'}, "
                    f"got {type(val).__name__}"
                )

        # Range checks for weights (0-1)
        for field in ("bm25_weight", "vector_weight"):
            val = search_cfg.get(field)
            if val is not None and isinstance(val, (int, float)):
                if not (0 <= val <= 1):
                    warnings.append(f"search.{field} should be between 0 and 1, got {val}")

        # Positive integer checks
        for field in ("top_k", "rerank_top_n"):
            val = search_cfg.get(field)
            if val is not None and isinstance(val, int) and val <= 0:
                warnings.append(f"search.{field} should be > 0, got {val}")

        # MMR lambda: None or 0-1
        mmr = search_cfg.get("mmr_lambda")
        if mmr is not None and isinstance(mmr, (int, float)):
            if not (0 <= mmr <= 1):
                warnings.append(f"search.mmr_lambda should be between 0 and 1, got {mmr}")

        # Merge strategy
        strategy = search_cfg.get("merge_strategy")
        if strategy is not None and strategy not in ("weighted", "rrf"):
            warnings.append(f"search.merge_strategy should be 'weighted' or 'rrf', got '{strategy}'")

        # Reranker backend
        reranker = search_cfg.get("reranker")
        if reranker is not None and reranker not in ("local", "api"):
            warnings.append(f"search.reranker should be 'local' or 'api', got '{reranker}'")

        # BM25 parameter tuning
        bm25_k1 = search_cfg.get("bm25_k1")
        if bm25_k1 is not None:
            if not isinstance(bm25_k1, (int, float)):
                warnings.append(f"search.bm25_k1 should be numeric, got {type(bm25_k1).__name__}")
            elif bm25_k1 <= 0:
                warnings.append(f"search.bm25_k1 should be > 0, got {bm25_k1}")

        bm25_b = search_cfg.get("bm25_b")
        if bm25_b is not None:
            if not isinstance(bm25_b, (int, float)):
                warnings.append(f"search.bm25_b should be numeric, got {type(bm25_b).__name__}")
            elif not (0 <= bm25_b <= 1):
                warnings.append(f"search.bm25_b should be between 0 and 1, got {bm25_b}")

        # Score normalization
        score_norm = search_cfg.get("score_normalization")
        if score_norm is not None and score_norm not in ("minmax", "zscore", "raw"):
            warnings.append(f"search.score_normalization should be 'minmax', 'zscore', or 'raw', got '{score_norm}'")

        # Multi-query count
        mq_count = search_cfg.get("multi_query_count")
        if mq_count is not None:
            if not isinstance(mq_count, int):
                warnings.append(f"search.multi_query_count should be int, got {type(mq_count).__name__}")
            elif not (2 <= mq_count <= 7):
                warnings.append(f"search.multi_query_count should be between 2 and 7, got {mq_count}")

    # Validate graph config
    graph_cfg = config.get("graph", {})
    if isinstance(graph_cfg, dict):
        for key in graph_cfg:
            if key not in _KNOWN_GRAPH_KEYS:
                warnings.append(f"Unknown graph key: 'graph.{key}'")

        search_mode = graph_cfg.get("search_mode")
        if search_mode is not None and search_mode not in ("naive", "local", "global", "hybrid", "mix"):
            warnings.append(
                f"graph.search_mode should be 'naive', 'local', 'global', 'hybrid', or 'mix', "
                f"got '{search_mode}'"
            )

    # Validate embedding config
    emb_cfg = config.get("embedding", {})
    if isinstance(emb_cfg, dict):
        model = emb_cfg.get("model")
        dim = emb_cfg.get("dimension")
        if model and dim and model in _KNOWN_EMBEDDING_DIMS:
            expected_dim = _KNOWN_EMBEDDING_DIMS[model]
            if dim != expected_dim:
                warnings.append(
                    f"embedding.dimension={dim} doesn't match model '{model}' "
                    f"(expected {expected_dim})"
                )

    # Validate code config
    code_cfg = config.get("code", {})
    if isinstance(code_cfg, dict):
        for key in code_cfg:
            if key not in _KNOWN_CODE_KEYS:
                warnings.append(f"Unknown code key: 'code.{key}'")

        enabled = code_cfg.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            warnings.append(f"code.enabled must be boolean, got {type(enabled).__name__}")

        paths = code_cfg.get("paths")
        if paths is not None:
            if not isinstance(paths, list):
                warnings.append(f"code.paths must be a list, got {type(paths).__name__}")
            else:
                for i, p in enumerate(paths):
                    if not isinstance(p, str):
                        warnings.append(f"code.paths[{i}] must be a string, got {type(p).__name__}")

        workers = code_cfg.get("parallel_workers")
        if workers is not None:
            if not isinstance(workers, int):
                warnings.append(f"code.parallel_workers must be a positive integer, got {type(workers).__name__}")
            elif workers <= 0:
                warnings.append(f"code.parallel_workers must be > 0, got {workers}")

        max_kb = code_cfg.get("max_file_size_kb")
        if max_kb is not None:
            if not isinstance(max_kb, (int, float)) or max_kb <= 0:
                warnings.append(f"code.max_file_size_kb must be a positive number, got {max_kb}")

        git_cfg = code_cfg.get("git", {})
        if isinstance(git_cfg, dict):
            for key in git_cfg:
                if key not in _KNOWN_CODE_GIT_KEYS:
                    warnings.append(f"Unknown code.git key: 'code.git.{key}'")

    return warnings


class Client:
    """Represents a client configuration."""

    def __init__(self, client_path: Path, config: dict):
        """Wrap a client directory at *client_path* with its parsed *config* dict."""
        self.path = client_path
        self.config = config
        self.name = config.get("name", "Unknown")
        self.machine_name = config.get("machine_name", client_path.name)

    @property
    def client_dir(self) -> Path:
        """Path to the client's root directory."""
        return self.path

    @property
    def sources_path(self) -> Path:
        """Path to client's sources directory (GitHub clones, web crawls)."""
        return self.path / "sources"

    @property
    def sources_config(self) -> dict:
        """Source connector configuration."""
        return self.config.get("sources", {})

    @property
    def has_sources(self) -> bool:
        """Whether this client has source files on disk."""
        return self.sources_path.exists() and any(self.sources_path.iterdir())

    @property
    def documents_path(self) -> Path:
        """Path to client's documents directory."""
        return self.path / self.config.get("paths", {}).get("documents", "documents")

    @property
    def index_path(self) -> Path:
        """Path to client's index directory."""
        return self.path / self.config.get("paths", {}).get("index", "index")

    @property
    def prompts_path(self) -> Path:
        """Path to client's prompts directory."""
        return self.path / self.config.get("paths", {}).get("prompts", "prompts")

    @property
    def search_config(self) -> dict:
        """Search configuration."""
        return {**DEFAULT_CONFIG["search"], **self.config.get("search", {})}

    @property
    def tokenization_config(self) -> dict:
        """Tokenization configuration for BM25."""
        default_tok = DEFAULT_CONFIG["search"]["tokenization"]
        client_tok = self.config.get("search", {}).get("tokenization", {})
        return {**default_tok, **client_tok}

    @property
    def embedding_config(self) -> dict:
        """Embedding model configuration."""
        return {**DEFAULT_CONFIG["embedding"], **self.config.get("embedding", {})}

    @property
    def llm_config(self) -> dict:
        """LLM configuration."""
        return {**DEFAULT_CONFIG["llm"], **self.config.get("llm", {})}

    @property
    def documents_config(self) -> dict:
        """Document ingestion configuration."""
        return {**DEFAULT_CONFIG["documents"], **self.config.get("documents", {})}

    @property
    def graph_config(self) -> dict:
        """Knowledge graph configuration."""
        defaults = {
            "enabled": False,
            "entity_types": ["person", "organization", "concept", "location"],
            "extract_batch_size": 512,
            "max_description_chars": 200,
            "search_mode": "naive",
            "extract_workers": 3,
            "extract_save_every": 5,
            "extract_text_budget": 6000,
            "exclude_extensions": [],
        }
        return {**defaults, **self.config.get("graph", {})}

    @property
    def chunking_config(self) -> dict:
        """Chunking strategy configuration."""
        defaults = {
            "strategy": "title",
            "max_chunk_size": 1500,
            "overlap": 100,
            "semantic_threshold": 0.5,
            "min_chunk_size": 100,
            "separators": ["\n\n", "\n", ". ", " "],
        }
        return {**defaults, **self.config.get("chunking", {})}

    def get_document_extensions(self) -> tuple:
        """Get document extensions to process, excluding any specified exclusions."""
        config = self.documents_config
        extensions = set(config.get("extensions", DEFAULT_DOCUMENT_EXTENSIONS))
        exclude = set(config.get("exclude_extensions", []))
        # Normalize extensions (ensure they start with .)
        extensions = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
        exclude = {ext if ext.startswith(".") else f".{ext}" for ext in exclude}
        return tuple(sorted(extensions - exclude))

    def get_document_filenames(self) -> tuple:
        """Bare filenames (no extension) to ingest as documents.

        Pools opt in via documents.filenames in config.yaml to pick up
        extension-less files like Jenkinsfile, Dockerfile, Makefile.
        """
        config = self.documents_config
        names = config.get("filenames", DEFAULT_DOCUMENT_FILENAMES)
        return tuple(sorted(set(names)))

    @property
    def audit_config(self) -> AuditConfig:
        """Audit configuration."""
        return load_audit_config(self.config)

    @property
    def reference_config(self) -> ReferenceConfig:
        """Reference resolution configuration."""
        return load_reference_config(self.config.get("references", {}))

    def get_system_prompt(self) -> Optional[str]:
        """Load custom system prompt if available."""
        prompt_file = self.prompts_path / "system.txt"
        if prompt_file.exists():
            return prompt_file.read_text().strip()
        return None

    @property
    def code_config(self) -> dict:
        """Code intelligence configuration."""
        return self.config.get("code", {})

    def get_code_chunks(self, paths: list[Path] | None = None) -> list[dict]:
        """Extract code chunks from source paths using CodeChunker.

        Args:
            paths: Directories to scan. Defaults to documents_path.

        Returns:
            List of chunk dicts ready for indexing.
        """
        if paths is None:
            paths = [self.documents_path]
        chunker = CodeChunker(self.code_config)
        return chunker.chunk_paths(paths)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.documents_path.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.prompts_path.mkdir(parents=True, exist_ok=True)


def _match_display_name(
    query: str,
    clients_dir: Path,
    known_slugs: list[str],
) -> list[str]:
    """Return slugs whose config.yaml/client.yaml ``name`` matches ``query``.

    Match strategy:
      1. Case-insensitive exact match on display name → wins outright.
      2. Otherwise fuzzy match via difflib.get_close_matches (cutoff 0.8).

    Returns a list of candidate slugs. Empty if no match found.
    """
    display_to_slug: dict[str, str] = {}
    for slug in known_slugs:
        for fname in ("config.yaml", "client.yaml"):
            cfg_path = clients_dir / slug / fname
            if cfg_path.exists():
                try:
                    data = yaml.safe_load(cfg_path.read_text()) or {}
                    display = data.get("name")
                    if isinstance(display, str) and display.strip():
                        display_to_slug[display.strip()] = slug
                except Exception as exc:
                    logger.debug(
                        f"_match_display_name: failed to read {cfg_path}: {exc}"
                    )
                break  # only look at the first config file found per pool

    if not display_to_slug:
        return []

    query_lower = query.strip().lower()
    # 1. Case-insensitive exact match
    exact = [slug for name, slug in display_to_slug.items() if name.lower() == query_lower]
    if exact:
        # Deduplicate while preserving order
        return list(dict.fromkeys(exact))

    # 2. Fuzzy match against display names
    display_names = list(display_to_slug.keys())
    matches = difflib.get_close_matches(query, display_names, n=5, cutoff=0.8)
    return list(dict.fromkeys(display_to_slug[m] for m in matches))


def load_client(
    client_name: str,
    clients_dir: Optional[Path] = None,
    create_if_missing: bool = False,
) -> Client:
    """
    Load a client by name.

    Args:
        client_name: Client directory name
        clients_dir: Base directory for clients (default: ./clients)
        create_if_missing: If True and no config.yaml/client.yaml is found,
            fall back to DEFAULT_CONFIG and create the directory layout
            (used by pool-creation flows). If False (default), raise
            ValueError instead — search/load flows should fail loudly
            rather than silently spawn an empty pool directory.

    Returns:
        Client instance

    Raises:
        ValueError: If no config file is found and ``create_if_missing`` is
            False. The message lists available pools and, when possible,
            suggests the closest matching slug.
    """
    if clients_dir is None:
        from mullet.core.env import get_pools_dir
        clients_dir = get_pools_dir()

    client_path = clients_dir / client_name
    # Support both config.yaml (preferred) and client.yaml (legacy)
    config_file = client_path / "config.yaml"
    if not config_file.exists():
        config_file = client_path / "client.yaml"

    if not config_file.exists():
        if not create_if_missing:
            known = discover_clients(clients_dir)

            # Display-name fuzzy fallback: if the caller passed something that
            # doesn't match any slug, try matching against each pool's
            # config.yaml:name (or client.yaml:name) field. This lets users
            # call load_client("Claude Code Ecosystem") when the on-disk slug
            # is "claude_code_ecosystem".
            matched_slugs = _match_display_name(client_name, clients_dir, known)
            if len(matched_slugs) == 1:
                matched = matched_slugs[0]
                logger.info(
                    f"load_client: interpreted '{client_name}' as display "
                    f"name for slug '{matched}'"
                )
                return load_client(matched, clients_dir, create_if_missing)
            if len(matched_slugs) > 1:
                raise ValueError(
                    f"Pool display name '{client_name}' is ambiguous — matches "
                    f"multiple pools: {', '.join(matched_slugs)}. "
                    f"Pass the exact slug instead."
                )

            lines = [f"Pool '{client_name}' not found in {clients_dir}."]
            if known:
                lines.append(f"Available pools: {', '.join(sorted(known))}.")
            else:
                lines.append("Available pools: (none).")
            best = difflib.get_close_matches(client_name, known, n=1, cutoff=0.6)
            if best:
                lines.append(f"Did you mean '{best[0]}'?  (use exact slug, case-sensitive)")
            raise ValueError(" ".join(lines))

        logger.warning(f"No config.yaml or client.yaml found for {client_name}, using defaults")
        config = {**DEFAULT_CONFIG, "machine_name": client_name}
    else:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        config = {**DEFAULT_CONFIG, **config}

    # Validate config
    config_warnings = validate_config(config)
    for w in config_warnings:
        logger.warning(f"Config warning for {client_name}: {w}")

    client = Client(client_path, config)
    client.ensure_directories()

    logger.info(f"Loaded client: {client.name} ({client.machine_name})")
    return client


def discover_clients(clients_dir: Optional[Path] = None) -> list[str]:
    """
    Discover available clients.

    Args:
        clients_dir: Base directory for clients

    Returns:
        List of client names
    """
    if clients_dir is None:
        from mullet.core.env import get_pools_dir
        clients_dir = get_pools_dir()

    if not clients_dir.exists():
        return []

    clients = []
    for path in clients_dir.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            clients.append(path.name)

    return sorted(clients)


def get_active_client(clients_dir: Optional[Path] = None) -> Optional[Client]:
    """
    Get the active client from environment or first available.

    Checks RAG_CLIENT environment variable first.

    Returns:
        Client instance or None
    """
    client_name = os.getenv("RAG_CLIENT")

    if client_name:
        return load_client(client_name, clients_dir)

    # Fall back to first available client
    available = discover_clients(clients_dir)
    if available:
        return load_client(available[0], clients_dir)

    return None
