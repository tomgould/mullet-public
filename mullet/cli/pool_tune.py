"""
Hardware-adaptive pool settings tuner.

Probes CPU/RAM/GPU and writes recommended settings into each pool's config.yaml.
Designed for open-source use: works on CPU-only, 4GB VRAM, or high-end GPUs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HEADINGS = ["Pool", "GPU free MB", "workers", "batch_sleep", "embed_batch"]


def _recommend_inter_batch_sleep(free_mb: Optional[int]) -> float:
    if free_mb is None:
        return 2.0  # CPU-only — no VRAM, be conservative
    if free_mb > 8192:
        return 0.0
    if free_mb > 4096:
        return 0.5
    return 1.0


def _load_yaml(path: Path):
    """Load a YAML file. Returns (ruamel_instance_or_None, data)."""
    try:
        from ruamel.yaml import YAML  # pragma: no cover
        y = YAML()  # pragma: no cover
        y.preserve_quotes = True  # pragma: no cover
        with open(path) as f:  # pragma: no cover
            return y, y.load(f)  # pragma: no cover
    except ImportError:
        import yaml
        with open(path) as f:
            return None, yaml.safe_load(f)


def _save_yaml(path: Path, engine, data) -> None:
    if engine is None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    else:
        with open(path, "w") as f:
            engine.dump(data, f)


def _add_tuned_comment(engine, graph_section, timestamp: str) -> None:
    """Add a ruamel comment above the graph section keys (best-effort)."""
    try:
        graph_section.yaml_set_start_comment(
            f" last tuned: {timestamp} by mullet pool tune",
            indent=2,
        )
    except Exception:
        pass  # ruamel comment API varies — not critical


def run_pool_tune(
    pools_dir: Path,
    pool_filter: str | None = None,
    dry_run: bool = False,
) -> None:
    """Probe hardware and write recommended settings into pool config.yaml files.

    Args:
        pools_dir: Directory containing pool subdirectories.
        pool_filter: If set, only process this single pool name.
        dry_run: Print changes without writing any files.
    """
    from mullet.knowledge.engine.resource_manager import ResourceManager

    rm = ResourceManager(headroom=0.15)
    gpu = rm.probe_gpu()
    free_mb: Optional[int] = gpu["free_mb"] if gpu else None

    inter_batch_sleep = _recommend_inter_batch_sleep(free_mb)
    embed_batch_size = min(64, rm.recommend_embed_batch(768))
    extract_workers = 1  # always — graph extraction is GPU-bound, not I/O-bound

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gpu_label = str(free_mb) if free_mb is not None else "none"

    rows: list[tuple] = []

    for pool_dir in sorted(pools_dir.iterdir()):
        if not pool_dir.is_dir():
            continue
        if pool_filter and pool_dir.name != pool_filter:
            continue
        yaml_path = pool_dir / "config.yaml"
        if not yaml_path.exists():
            # Legacy fallback
            yaml_path = pool_dir / "client.yaml"
        if not yaml_path.exists():
            continue

        engine, data = _load_yaml(yaml_path)
        if not isinstance(data, dict) or "graph" not in data:
            continue

        graph = data["graph"]
        old_workers = graph.get("extract_workers")
        old_sleep = graph.get("inter_batch_sleep")
        old_batch = graph.get("embed_batch_size")

        changed = (
            old_workers != extract_workers
            or old_sleep != inter_batch_sleep
            or old_batch != embed_batch_size
        )

        rows.append((
            pool_dir.name,
            gpu_label,
            extract_workers,
            inter_batch_sleep,
            embed_batch_size,
            changed,
        ))

        if not dry_run and changed:
            graph["extract_workers"] = extract_workers
            graph["inter_batch_sleep"] = inter_batch_sleep
            graph["embed_batch_size"] = embed_batch_size
            if engine is not None:  # pragma: no cover — only when ruamel.yaml installed
                _add_tuned_comment(engine, graph, timestamp)  # pragma: no cover
            _save_yaml(yaml_path, engine, data)
            logger.info(f"pool tune: updated {pool_dir.name}")

    _print_table(rows)

    if dry_run:
        print("\n(dry-run — no files written)")
    elif rows:
        changed_count = sum(1 for r in rows if r[5])
        print(f"\nUpdated {changed_count}/{len(rows)} pools.")
    else:
        pool_msg = f" '{pool_filter}'" if pool_filter else ""
        print(f"No pools with graph: section found{pool_msg}.")


def _print_table(rows: list[tuple]) -> None:
    if not rows:
        return

    display_rows = [(r[0], r[1], str(r[2]), str(r[3]), str(r[4])) for r in rows]
    col_w = [
        max(len(_HEADINGS[i]), max(len(dr[i]) for dr in display_rows))
        for i in range(len(_HEADINGS))
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    header = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(_HEADINGS)) + " |"

    print(sep)
    print(header)
    print(sep)
    for dr, r in zip(display_rows, rows):
        flag = " *" if r[5] else "  "
        print("| " + " | ".join(dr[i].ljust(col_w[i]) for i in range(len(_HEADINGS))) + f" |{flag}")
    print(sep)
    if any(r[5] for r in rows):
        print("  * = will be updated")
