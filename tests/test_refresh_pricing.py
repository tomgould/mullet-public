"""Tests for scripts/refresh_pricing.py helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from refresh_pricing import (
    apply_additions,
    apply_updates,
    compute_additions,
    compute_updates,
    fetch_openrouter_catalog,
    identify_new_openrouter_models,
    identify_openrouter_models,
)


def test_identify_openrouter_models():
    pricing = {"models": {
        "deepseek/deepseek-chat": {"input_per_1k": 0.00027},
        "kimi-k2.7-code": {"input_per_1k": 0.00095},
        "meta-llama/llama-4-scout": {"input_per_1k": 0.00017},
    }}
    result = identify_openrouter_models(pricing)
    assert "deepseek/deepseek-chat" in result
    assert "meta-llama/llama-4-scout" in result
    assert "kimi-k2.7-code" not in result


def test_compute_updates_no_change():
    pricing = {"models": {"deepseek/deepseek-chat": {"input_per_1k": 0.00027, "output_per_1k": 0.0011}}}
    live = {"deepseek/deepseek-chat": (0.00027, 0.0011)}
    assert compute_updates(pricing, live) == {}


def test_compute_updates_detects_change():
    pricing = {"models": {"deepseek/deepseek-chat": {"input_per_1k": 0.00030, "output_per_1k": 0.0011}}}
    live = {"deepseek/deepseek-chat": (0.00027, 0.0011)}
    updates = compute_updates(pricing, live)
    assert "deepseek/deepseek-chat" in updates
    assert updates["deepseek/deepseek-chat"][0] == 0.00027


def test_apply_updates_does_not_mutate_original():
    import copy
    pricing = {"models": {"deepseek/deepseek-chat": {"input_per_1k": 0.00030, "output_per_1k": 0.0011}}}
    original = copy.deepcopy(pricing)
    result = apply_updates(pricing, {"deepseek/deepseek-chat": (0.00027, 0.0011)})
    assert pricing == original  # original not mutated
    assert result["models"]["deepseek/deepseek-chat"]["input_per_1k"] == 0.00027


def test_fetch_openrouter_catalog_filters_modality_and_pricing(monkeypatch):
    raw = {
        "data": [
            # Allowed vendors, valid modality/pricing -> included
            {"id": "deepseek/model-b", "architecture": {"modality": "text->text"}, "pricing": {"prompt": 0.001, "completion": 0.002}, "context_length": 128000},
            {"id": "deepseek/model-c", "architecture": {"modality": "text+image->text"}, "pricing": {"prompt": 0.001, "completion": 0.002}, "context_length": 64000},
            # Allowed vendor but invalid modality/pricing -> skipped
            {"id": "deepseek/model-d", "architecture": {"modality": "text->image"}, "pricing": {"prompt": 0.001, "completion": 0.002}, "context_length": 128000},
            {"id": "deepseek/model-e", "pricing": {"prompt": 0.001, "completion": 0.002}, "context_length": 128000},
            {"id": "deepseek/model-f", "architecture": {"modality": "text->text"}, "pricing": {"prompt": 0, "completion": 0.002}, "context_length": 128000},
            {"id": "deepseek/model-g", "architecture": {"modality": "text->text"}, "pricing": {"prompt": 0.001, "completion": 0}, "context_length": 128000},
            # Disallowed vendor, valid modality/pricing -> skipped
            {"id": "someweirdvendor/model-x", "architecture": {"modality": "text->text"}, "pricing": {"prompt": 0.001, "completion": 0.002}, "context_length": 128000},
            # Malformed id (no slash), valid modality/pricing -> skipped
            {"id": "no-slash-model", "architecture": {"modality": "text->text"}, "pricing": {"prompt": 0.001, "completion": 0.002}, "context_length": 128000},
        ]
    }
    monkeypatch.setattr("refresh_pricing._fetch_openrouter_data", lambda api_key=None: raw)
    catalog, skipped = fetch_openrouter_catalog()
    assert set(catalog.keys()) == {"deepseek/model-b", "deepseek/model-c"}
    assert catalog["deepseek/model-b"]["context_window"] == 128000
    assert catalog["deepseek/model-c"]["context_window"] == 64000
    assert skipped == 6  # d (text->image), e (missing modality), f (zero prompt), g (zero completion), someweirdvendor (unknown vendor), no-slash-model (malformed id)


def test_identify_new_openrouter_models():
    pricing = {"models": {"a/b": {}, "a/c": {}}}
    catalog = {"a/c": {}, "a/d": {}, "a/e": {}}
    assert identify_new_openrouter_models(pricing, catalog) == ["a/d", "a/e"]


def test_compute_additions():
    catalog = {
        "a/b": {"input_per_1k": 0.001, "output_per_1k": 0.002, "context_window": 128000},
        "a/c": {"input_per_1k": 0.003, "output_per_1k": 0.004, "context_window": 64000},
    }
    additions = compute_additions(catalog, ["a/c"])
    assert additions == {"a/c": catalog["a/c"]}


def test_apply_additions_does_not_mutate_original():
    import copy
    pricing = {"models": {"a/b": {"input_per_1k": 0.001}}}
    original = copy.deepcopy(pricing)
    result = apply_additions(pricing, {"a/c": {"input_per_1k": 0.002, "output_per_1k": 0.003, "context_window": 128000}})
    assert pricing == original
    assert result["models"]["a/c"]["input_per_1k"] == 0.002
    assert result["models"]["a/c"]["context_window"] == 128000


def test_main_dry_run_additions_only(tmp_path, monkeypatch, capsys):
    pricing_yaml = tmp_path / "pricing.yaml"
    pricing_yaml.write_text("""models:
  existing/model:
    input_per_1k: 0.001
    output_per_1k: 0.002
    context_window: 128000
""")
    monkeypatch.setattr("refresh_pricing.PRICING_YAML", pricing_yaml)
    monkeypatch.setattr("refresh_pricing.fetch_openrouter_prices", lambda api_key=None: {"existing/model": (0.001, 0.002)})
    monkeypatch.setattr("refresh_pricing.fetch_openrouter_catalog", lambda api_key=None: (
        {"new/vendor-model": {"input_per_1k": 0.003, "output_per_1k": 0.004, "context_window": 64000}},
        2,
    ))
    import refresh_pricing
    rc = refresh_pricing.main(["--dry-run"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Skipped 2 non-text/unpriced catalog entries." in captured.out
    assert "Detected 1 new model(s):" in captured.out
    assert "new/vendor-model" in captured.out
    # File must be untouched
    content = pricing_yaml.read_text()
    assert "new/vendor-model" not in content


def test_script_file_exists():
    assert Path("scripts/refresh_pricing.py").exists()
