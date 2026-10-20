"""Lock-in test: MODEL_DATABASE costs must match pricing.yaml.

Prevents the drift that bit Kimi K2.5 (two different prices in two files).
After the SSOT consolidation, MODEL_DATABASE costs are sourced from pricing.yaml,
so these tests serve as regression tests, not just parity checks.
"""

from mullet.core.providers.model_info import MODEL_DATABASE
from mullet.core.providers.pricing import load_pricing


def test_every_priced_model_has_yaml_entry() -> None:
    """Every model in MODEL_DATABASE with non-zero input_cost_per_1k
    must have a corresponding entry in pricing.yaml.
    """
    yaml_data = load_pricing()
    missing: list[str] = []

    for model_id, info in MODEL_DATABASE.items():
        if info.input_cost_per_1k != 0.0 and model_id not in yaml_data:
            missing.append(model_id)

    if missing:
        missing.sort()
        raise AssertionError(
            f"{len(missing)} priced model(s) missing from pricing.yaml:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


def test_input_costs_match_yaml() -> None:
    """input_cost_per_1k in MODEL_DATABASE must equal input_per_1k
    in pricing.yaml (within floating-point tolerance).
    """
    yaml_data = load_pricing()
    mismatches: list[tuple[str, float, float]] = []

    for model_id, info in MODEL_DATABASE.items():
        if info.input_cost_per_1k == 0.0:
            continue
        entry = yaml_data.get(model_id)
        if entry is None:
            continue
        yaml_value = float(entry["input_per_1k"])
        if abs(info.input_cost_per_1k - yaml_value) >= 1e-9:
            mismatches.append((model_id, info.input_cost_per_1k, yaml_value))

    if mismatches:
        mismatches.sort(key=lambda t: t[0])
        lines = "\n".join(
            f"  - {m}: MODEL_DATABASE={db:.9f}  pricing.yaml={yaml:.9f}"
            for m, db, yaml in mismatches
        )
        raise AssertionError(
            f"{len(mismatches)} model(s) have input cost mismatches:\n{lines}"
        )


def test_output_costs_match_yaml() -> None:
    """output_cost_per_1k in MODEL_DATABASE must equal output_per_1k
    in pricing.yaml (within floating-point tolerance).
    """
    yaml_data = load_pricing()
    mismatches: list[tuple[str, float, float]] = []

    for model_id, info in MODEL_DATABASE.items():
        if info.input_cost_per_1k == 0.0:
            continue
        entry = yaml_data.get(model_id)
        if entry is None:
            continue
        yaml_value = float(entry["output_per_1k"])
        if abs(info.output_cost_per_1k - yaml_value) >= 1e-9:
            mismatches.append((model_id, info.output_cost_per_1k, yaml_value))

    if mismatches:
        mismatches.sort(key=lambda t: t[0])
        lines = "\n".join(
            f"  - {m}: MODEL_DATABASE={db:.9f}  pricing.yaml={yaml:.9f}"
            for m, db, yaml in mismatches
        )
        raise AssertionError(
            f"{len(mismatches)} model(s) have output cost mismatches:\n{lines}"
        )


def test_free_models_exempt() -> None:
    """Models with zero input cost (free/local) are exempt from parity checks.

    We also assert that at least one free model exists so the exemption
    logic is actually exercised.
    """
    free_models = [
        m for m, info in MODEL_DATABASE.items() if info.input_cost_per_1k == 0.0
    ]
    assert len(free_models) >= 1, "Expected at least one free/local model in MODEL_DATABASE"
