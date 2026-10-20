#!/usr/bin/env python3
"""Check OpenRouter live pricing against MODEL_DATABASE; report drift.

Usage:
  scripts/sync_openrouter_pricing.py            # human-readable table
  scripts/sync_openrouter_pricing.py --json     # machine-readable
  scripts/sync_openrouter_pricing.py --threshold 10  # different drift cutoff

Exits 0 if no drift > threshold, 1 if drift detected, 2 on error.

Reads OPENROUTER_API_KEY from env. Compares per-1K-token pricing for every
MODEL_DATABASE entry whose key matches `<provider>/<model>` (the OpenRouter
slug convention).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request


_OR_SLUG_RE = re.compile(r"^[a-z0-9\-]+/[a-z0-9.\-]+$")


def fetch_live_pricing(api_key: str) -> dict[str, tuple[float, float]]:
    """Return {model_id: (in_per_1k, out_per_1k)} from OpenRouter's /models endpoint."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"Error fetching OpenRouter models: {exc}", file=sys.stderr)
        sys.exit(2)

    out: dict[str, tuple[float, float]] = {}
    for m in data.get("data", []):
        slug = m.get("id")
        p = m.get("pricing") or {}
        try:
            in_per_1k = float(p.get("prompt", 0)) * 1000.0
            out_per_1k = float(p.get("completion", 0)) * 1000.0
        except (TypeError, ValueError):
            continue
        if slug:
            out[slug] = (in_per_1k, out_per_1k)
    return out


def drift_pct(live: float, file: float) -> float:
    """Percentage drift between live and file price.

    - 0 when both are zero
    - 100 when file is zero but live is non-zero (cap)
    - abs((live - file) / file) * 100 otherwise
    """
    if file == 0:
        return 0.0 if live == 0 else 100.0
    return abs((live - file) / file) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare OpenRouter live pricing against MODEL_DATABASE.",
    )
    parser.add_argument(
        "--json", dest="json_out", action="store_true",
        help="Emit machine-readable JSON.",
    )
    parser.add_argument(
        "--threshold", type=float, default=5.0,
        help="Drift percentage that counts as 'changed' (default 5.0).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY env var not set.", file=sys.stderr)
        return 2

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mullet.core.providers.model_info import MODEL_DATABASE

    live = fetch_live_pricing(api_key)

    rows = []
    for slug, info in MODEL_DATABASE.items():
        if not _OR_SLUG_RE.match(slug):
            continue
        live_pair = live.get(slug)
        if live_pair is None:
            rows.append({
                "model": slug,
                "status": "not-on-openrouter",
                "live_in": None, "live_out": None,
                "file_in": info.input_cost_per_1k,
                "file_out": info.output_cost_per_1k,
                "drift_in": None, "drift_out": None,
            })
            continue
        live_in, live_out = live_pair
        d_in = drift_pct(live_in, info.input_cost_per_1k)
        d_out = drift_pct(live_out, info.output_cost_per_1k)
        rows.append({
            "model": slug,
            "status": "drift" if max(d_in, d_out) > args.threshold else "ok",
            "live_in": live_in, "live_out": live_out,
            "file_in": info.input_cost_per_1k,
            "file_out": info.output_cost_per_1k,
            "drift_in": d_in, "drift_out": d_out,
        })

    drift_count = sum(1 for r in rows if r["status"] == "drift")
    rows.sort(key=lambda r: -(r["drift_in"] or 0))

    if args.json_out:
        print(json.dumps({
            "threshold_pct": args.threshold,
            "drift_count": drift_count,
            "total_checked": sum(1 for r in rows if r["status"] != "not-on-openrouter"),
            "rows": rows,
        }, indent=2))
    else:
        print(f"{'Model':<40} {'live in':>10} {'live out':>10} {'file in':>10} {'file out':>10} {'drift':>10}")
        print("-" * 94)
        for r in rows:
            if r["status"] == "not-on-openrouter":
                print(f"{r['model']:<40} {'-':>10} {'-':>10} ${r['file_in']:>8.5f} ${r['file_out']:>8.5f}  (not on OR)")
                continue
            mark = "!" if r["status"] == "drift" else " "
            max_drift = max(r["drift_in"], r["drift_out"])
            print(
                f"{r['model']:<40} ${r['live_in']:>8.5f} ${r['live_out']:>8.5f} "
                f"${r['file_in']:>8.5f} ${r['file_out']:>8.5f} {mark}{max_drift:>7.1f}%"
            )
        print()
        print(f"{drift_count} model(s) with drift > {args.threshold}%.")

    return 0 if drift_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
