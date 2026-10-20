"""Non-interactive pool creation helper extracted from scripts/create_pool.py."""

from pathlib import Path

import yaml

from mullet.core.env import get_pools_dir


def create_pool_silent(
    slug: str,
    machine_name: str,
    embedding_model: str = "nomic-embed-text",
) -> Path:
    """Create a minimal pool directory and config.yaml without prompting.

    Args:
        slug: Directory name under ~/.mullet/pools/.
        machine_name: The machine_name field written into config.yaml.
        embedding_model: Embedding model identifier.

    Returns:
        Path to the created pool directory.
    """
    pools_dir = get_pools_dir()
    pool_path = pools_dir / slug
    pool_path.mkdir(parents=True, exist_ok=True)
    (pool_path / "documents").mkdir(exist_ok=True)
    (pool_path / "index").mkdir(exist_ok=True)
    (pool_path / "prompts").mkdir(exist_ok=True)

    config = {
        "name": slug,
        "machine_name": machine_name,
        "paths": {
            "documents": "documents",
            "index": "index",
            "prompts": "prompts",
        },
        "embedding": {
            "model": embedding_model,
            "dimension": 768,
            "hnsw_ef_construction": 200,
            "hnsw_M": 16,
        },
    }

    config_path = pool_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return pool_path
