"""Prompt discovery — search and manage prompt templates."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_prompts(search_dirs: list[str] | None = None) -> list[dict]:
    """Discover prompt template files from search directories."""
    from mullet.core.paths import data_dir

    dirs = search_dirs or [
        str(data_dir() / "prompts"),
        str(Path.cwd() / ".mullet" / "prompts"),
        str(Path.cwd() / "prompts"),
    ]
    prompts = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*.md")) + sorted(p.glob("*.txt")) + sorted(p.glob("*.json")):
            prompts.append({
                "name": f.stem,
                "path": str(f),
                "format": f.suffix.lstrip("."),
                "size": f.stat().st_size,
            })
    return prompts


def load_prompt(path: str, variables: dict | None = None) -> str:
    """Load a prompt template and optionally substitute variables."""
    content = Path(path).read_text()
    if variables:
        for key, val in variables.items():
            content = content.replace(f"{{{{{key}}}}}", str(val))
    return content


def search_prompts(query: str, prompts: list[dict] | None = None) -> list[dict]:
    """Search prompts by name or content."""
    if prompts is None:
        prompts = discover_prompts()
    query_lower = query.lower()
    results = []
    for p in prompts:
        if query_lower in p["name"].lower():
            results.append(p)
            continue
        try:
            content = Path(p["path"]).read_text()[:1000]
            if query_lower in content.lower():
                results.append(p)
        except Exception as _exc:
            logger.warning("prompts: failed to read prompt content for search %s: %s", p["path"], _exc)
            pass
    return results
