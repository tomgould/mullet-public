"""Refresh OpenRouter model prices in pricing.yaml from live API data."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

import yaml

PRICING_YAML = Path(__file__).parent.parent / "mullet/core/providers/pricing.yaml"

# Vendor prefixes Mullet's routing code actually references (routing_matrix.py,
# resolver.py, model_registry.py) or that are already curated in pricing.yaml.
# OpenRouter's full catalog has hundreds of irrelevant vendors (image/niche
# models) -- auto-add only considers models from vendors we'd plausibly route to.
KNOWN_VENDOR_PREFIXES: frozenset[str] = frozenset({
    "anthropic", "deepseek", "google", "meta-llama", "minimax",
    "mistralai", "openai", "qwen", "x-ai", "z-ai",
})


def load_yaml(path=PRICING_YAML):
    with open(path) as f:
        return yaml.safe_load(f)


def identify_openrouter_models(pricing: dict) -> list[str]:
    """Return model IDs containing '/' (OpenRouter models)."""
    return [k for k in pricing.get("models", {}) if "/" in k]


def _fetch_openrouter_data(api_key=None) -> dict:
    """Call https://openrouter.ai/api/v1/models and return raw JSON."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY")
    headers = {"User-Agent": "mullet-pricing-refresh/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def fetch_openrouter_prices(api_key=None) -> dict[str, tuple[float, float]]:
    """Call https://openrouter.ai/api/v1/models, return {model_id: (input_per_1k, output_per_1k)}.
    pricing.prompt/completion in API are USD per TOKEN — multiply by 1000 for per-1K.
    Returns {} on error (logs warning)."""
    logger = logging.getLogger(__name__)
    try:
        data = _fetch_openrouter_data(api_key)
        result = {}
        for m in data.get("data", []):
            mid = m.get("id", "")
            p = m.get("pricing", {})
            try:
                inp = float(p.get("prompt", 0)) * 1000
                out = float(p.get("completion", 0)) * 1000
                if inp > 0 or out > 0:
                    result[mid] = (round(inp, 8), round(out, 8))
            except (TypeError, ValueError):
                pass
        return result
    except Exception as e:
        logger.warning("fetch_openrouter_prices failed: %s", e)
        return {}


def fetch_openrouter_catalog(api_key=None) -> tuple[dict[str, dict], int]:
    """Call https://openrouter.ai/api/v1/models, return filtered catalog and skip count.

    Only includes models where:
    - architecture.modality exists and ends with "->text"
    - pricing prompt and completion are both > 0 after ×1000 conversion

    Returns ({model_id: {"input_per_1k": ..., "output_per_1k": ..., "context_window": ...}}, skipped_count).
    Returns ({}, 0) on network/parse error (logs warning).
    """
    logger = logging.getLogger(__name__)
    try:
        data = _fetch_openrouter_data(api_key)
    except Exception as e:
        logger.warning("fetch_openrouter_catalog failed: %s", e)
        return {}, 0

    catalog: dict[str, dict] = {}
    skipped = 0
    for m in data.get("data", []):
        mid = m.get("id", "")
        if "/" not in mid:
            skipped += 1
            continue
        vendor = mid.split("/", 1)[0]
        if vendor not in KNOWN_VENDOR_PREFIXES:
            skipped += 1
            continue

        modality = m.get("architecture", {}).get("modality")
        if not modality or not str(modality).endswith("->text"):
            skipped += 1
            continue

        p = m.get("pricing", {})
        try:
            inp = float(p.get("prompt", 0)) * 1000
            out = float(p.get("completion", 0)) * 1000
            if inp <= 0 or out <= 0:
                skipped += 1
                continue
        except (TypeError, ValueError):
            skipped += 1
            continue

        ctx = m.get("context_length")
        try:
            ctx_int = int(ctx) if ctx is not None else 0
        except (TypeError, ValueError):
            ctx_int = 0

        catalog[mid] = {
            "input_per_1k": round(inp, 8),
            "output_per_1k": round(out, 8),
            "context_window": ctx_int,
        }
    return catalog, skipped


def identify_new_openrouter_models(pricing: dict, catalog: dict) -> list[str]:
    """Return model IDs present in catalog but not already in pricing["models"]."""
    existing = set(pricing.get("models", {}).keys())
    return [mid for mid in catalog if mid not in existing]


def compute_additions(catalog: dict, new_ids: list[str]) -> dict[str, dict]:
    """Return {model_id: fields} for each id in new_ids, sourced from catalog."""
    return {mid: catalog[mid] for mid in new_ids if mid in catalog}


def apply_additions(pricing: dict, additions: dict) -> dict:
    """Return a new pricing dict with added models. Does NOT write to disk."""
    result = copy.deepcopy(pricing)
    for mid, fields in additions.items():
        result["models"][mid] = copy.deepcopy(fields)
    return result


def compute_updates(pricing: dict, live: dict, threshold=0.001) -> dict[str, tuple[float, float]]:
    """Return {model_id: (new_input, new_output)} for models that changed by >threshold (0.1%)."""
    updates = {}
    for mid in identify_openrouter_models(pricing):
        if mid not in live:
            continue
        entry = pricing["models"][mid]
        cur_inp = entry.get("input_per_1k", 0)
        cur_out = entry.get("output_per_1k", 0)
        new_inp, new_out = live[mid]
        inp_changed = cur_inp == 0 or abs(new_inp - cur_inp) / max(cur_inp, 1e-12) > threshold
        out_changed = cur_out == 0 or abs(new_out - cur_out) / max(cur_out, 1e-12) > threshold
        if inp_changed or out_changed:
            updates[mid] = (new_inp, new_out)
    return updates


def apply_updates(pricing: dict, updates: dict) -> dict:
    """Return a new pricing dict with updated prices. Does NOT write to disk."""
    result = copy.deepcopy(pricing)
    for mid, (inp, out) in updates.items():
        result["models"][mid]["input_per_1k"] = inp
        result["models"][mid]["output_per_1k"] = out
    return result


def _format_change(mid: str, old_inp: float, old_out: float, new_inp: float, new_out: float) -> str:
    return (
        f"  {mid}:\n"
        f"    input_per_1k:  {old_inp:.8f} -> {new_inp:.8f}\n"
        f"    output_per_1k: {old_out:.8f} -> {new_out:.8f}"
    )


def _format_addition(mid: str, fields: dict) -> str:
    return (
        f"  {mid}:\n"
        f"    input_per_1k:  {fields['input_per_1k']:.8f}\n"
        f"    output_per_1k: {fields['output_per_1k']:.8f}\n"
        f"    context_window: {fields['context_window']}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Refresh OpenRouter prices in pricing.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key (or env OPENROUTER_API_KEY)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    pricing = load_yaml()
    live = fetch_openrouter_prices(api_key=args.api_key)

    if not live:
        logger.error("No live prices retrieved — API may be down or returned empty data.")
        return 2

    updates = compute_updates(pricing, live)

    catalog, skipped = fetch_openrouter_catalog(api_key=args.api_key)
    new_ids = identify_new_openrouter_models(pricing, catalog)
    additions = compute_additions(catalog, new_ids)

    if skipped:
        print(f"Skipped {skipped} non-text/unpriced catalog entries.")

    if not updates and not additions:
        logger.info("No price changes or new models detected.")
        return 1

    if updates:
        print(f"Detected {len(updates)} price update(s):\n")
        for mid, (new_inp, new_out) in updates.items():
            entry = pricing["models"][mid]
            old_inp = entry.get("input_per_1k", 0)
            old_out = entry.get("output_per_1k", 0)
            print(_format_change(mid, old_inp, old_out, new_inp, new_out))

    if additions:
        print(f"\nDetected {len(additions)} new model(s):\n")
        for mid, fields in additions.items():
            print(_format_addition(mid, fields))

    if args.dry_run:
        print("\n(Dry run — no files modified.)")
        return 0

    updated = apply_updates(pricing, updates)
    updated = apply_additions(updated, additions)
    with open(PRICING_YAML, "w") as f:
        yaml.dump(updated, f, sort_keys=False, allow_unicode=True)
    print(f"\nWrote updated prices to {PRICING_YAML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
