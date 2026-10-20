"""Unit tests for mullet/core/providers/spend_caps.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from unittest.mock import patch

from mullet.core.providers.spend_caps import (
    BudgetConfig,
    SpendCap,
    check_spend_cap,
    load_budget_config,
)


# ───────────────────────── SpendCap ─────────────────────────


class TestSpendCap:
    def test_remaining_cap_usd_simple(self):
        cap = SpendCap(provider="openrouter", cap_usd=10.0, current_spend_usd=3.0)
        assert cap.remaining_cap_usd == 7.0

    def test_remaining_cap_usd_never_negative(self):
        cap = SpendCap(provider="openrouter", cap_usd=5.0, current_spend_usd=10.0)
        assert cap.remaining_cap_usd == 0.0

    def test_is_within_cap_true_when_under(self):
        cap = SpendCap(provider="openrouter", cap_usd=10.0, current_spend_usd=9.99)
        assert cap.is_within_cap is True

    def test_is_within_cap_false_at_boundary(self):
        cap = SpendCap(provider="openrouter", cap_usd=10.0, current_spend_usd=10.0)
        assert cap.is_within_cap is False

    def test_is_within_cap_false_when_over(self):
        cap = SpendCap(provider="openrouter", cap_usd=10.0, current_spend_usd=15.0)
        assert cap.is_within_cap is False


# ───────────────────────── BudgetConfig ─────────────────────────


class TestBudgetConfig:
    def test_empty_config_has_no_caps(self):
        cfg = BudgetConfig(caps={})
        assert cfg.caps == {}
        assert cfg.monthly_total_cap_usd is None

    def test_config_with_two_providers(self):
        cfg = BudgetConfig(
            caps={
                "openrouter": SpendCap(provider="openrouter", cap_usd=10.0),
                "kimi": SpendCap(provider="kimi", cap_usd=20.0),
            }
        )
        assert len(cfg.caps) == 2
        assert "openrouter" in cfg.caps
        assert "kimi" in cfg.caps


# ───────────────────────── check_spend_cap ─────────────────────────


class TestCheckSpendCap:
    def test_provider_not_in_config_returns_no_cap_set(self):
        cfg = BudgetConfig(caps={"openrouter": SpendCap(provider="openrouter", cap_usd=10.0)})
        allowed, reason = check_spend_cap("kimi", balance_usd=5.0, config=cfg)
        assert allowed is True
        assert reason == "no cap set"

    def test_balance_none_returns_allowing(self):
        cfg = BudgetConfig(caps={"openrouter": SpendCap(provider="openrouter", cap_usd=10.0)})
        allowed, reason = check_spend_cap("openrouter", balance_usd=None, config=cfg)
        assert allowed is True
        assert reason == "balance unknown — allowing"

    def test_spent_less_than_cap_allowed(self):
        cfg = BudgetConfig(caps={"openrouter": SpendCap(provider="openrouter", cap_usd=10.0)})
        allowed, reason = check_spend_cap("openrouter", balance_usd=5.0, config=cfg)
        assert allowed is True
        assert "within cap" in reason

    def test_spent_equal_to_cap_denied(self):
        cfg = BudgetConfig(caps={"openrouter": SpendCap(provider="openrouter", cap_usd=10.0)})
        allowed, reason = check_spend_cap("openrouter", balance_usd=10.0, config=cfg)
        assert allowed is False
        assert "cap exceeded" in reason

    def test_spent_greater_than_cap_denied(self):
        cfg = BudgetConfig(caps={"openrouter": SpendCap(provider="openrouter", cap_usd=10.0)})
        allowed, reason = check_spend_cap("openrouter", balance_usd=15.0, config=cfg)
        assert allowed is False
        assert "cap exceeded" in reason

    def test_no_config_file_returns_no_cap_set(self, tmp_path: Path, monkeypatch):
        # Patch _find_budget_file so parent-dir walking doesn't find /tmp/.mullet/
        monkeypatch.setattr("mullet.core.providers.spend_caps._find_budget_file", lambda cwd=None: None)
        allowed, reason = check_spend_cap("openrouter", balance_usd=5.0, config=load_budget_config(cwd=tmp_path))
        assert allowed is True
        assert reason == "no cap set"


# ───────────────────────── load_budget_config ─────────────────────────


class TestLoadBudgetConfig:
    def test_no_mullet_dir_returns_empty_config(self, tmp_path: Path, monkeypatch):
        # Patch so parent-dir walking doesn't find /tmp/.mullet/budget.yaml
        monkeypatch.setattr("mullet.core.providers.spend_caps._find_budget_file", lambda cwd=None: None)
        cfg = load_budget_config(cwd=tmp_path)
        assert isinstance(cfg, BudgetConfig)
        assert cfg.caps == {}

    def test_loads_valid_yaml_one_provider(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text("openrouter:\n  cap_usd: 12.5\n", encoding="utf-8")

        cfg = load_budget_config(cwd=tmp_path)
        assert "openrouter" in cfg.caps
        assert cfg.caps["openrouter"].cap_usd == 12.5
        assert cfg.monthly_total_cap_usd is None

    def test_loads_valid_yaml_two_providers_and_monthly_total(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text(
            "openrouter:\n  cap_usd: 10.0\nkimi:\n  cap_usd: 20.0\nmonthly_total_cap_usd: 50.0\n",
            encoding="utf-8",
        )

        cfg = load_budget_config(cwd=tmp_path)
        assert len(cfg.caps) == 2
        assert cfg.caps["openrouter"].cap_usd == 10.0
        assert cfg.caps["kimi"].cap_usd == 20.0
        assert cfg.monthly_total_cap_usd == 50.0

    def test_malformed_yaml_returns_empty_config(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text("not_valid: [ unclosed\n", encoding="utf-8")

        cfg = load_budget_config(cwd=tmp_path)
        assert cfg.caps == {}

    def test_json_fallback_when_no_yaml(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.json"
        budget_file.write_text(
            json.dumps({"openrouter": {"cap_usd": 7.5}, "monthly_total_cap_usd": 30.0}),
            encoding="utf-8",
        )

        cfg = load_budget_config(cwd=tmp_path)
        assert "openrouter" in cfg.caps
        assert cfg.caps["openrouter"].cap_usd == 7.5
        assert cfg.monthly_total_cap_usd == 30.0

    def test_yaml_preferred_over_json(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        yaml_file = mullet_dir / "budget.yaml"
        yaml_file.write_text("openrouter:\n  cap_usd: 1.0\n", encoding="utf-8")
        json_file = mullet_dir / "budget.json"
        json_file.write_text(json.dumps({"openrouter": {"cap_usd": 99.0}}), encoding="utf-8")

        cfg = load_budget_config(cwd=tmp_path)
        assert cfg.caps["openrouter"].cap_usd == 1.0

    def test_loads_yaml_with_integer_cap(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text("openrouter:\n  cap_usd: 10\n", encoding="utf-8")

        cfg = load_budget_config(cwd=tmp_path)
        assert cfg.caps["openrouter"].cap_usd == 10.0

    def test_ignores_invalid_cap_usd_type(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text("openrouter:\n  cap_usd: not_a_number\n", encoding="utf-8")

        cfg = load_budget_config(cwd=tmp_path)
        assert "openrouter" not in cfg.caps

    def test_walks_up_to_parent_dir(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text("openrouter:\n  cap_usd: 5.0\n", encoding="utf-8")

        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        cfg = load_budget_config(cwd=nested)
        assert "openrouter" in cfg.caps
        assert cfg.caps["openrouter"].cap_usd == 5.0

    def test_invalid_monthly_total_logged_but_ignored(self, tmp_path: Path, caplog):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text(
            "openrouter:\n  cap_usd: 5.0\nmonthly_total_cap_usd: bad_value\n",
            encoding="utf-8",
        )

        cfg = load_budget_config(cwd=tmp_path)
        assert cfg.monthly_total_cap_usd is None
        assert "Invalid monthly_total_cap_usd" in caplog.text

    def test_loads_yaml_with_extra_keys_ignored(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.yaml"
        budget_file.write_text(
            "openrouter:\n  cap_usd: 5.0\n  extra_key: ignored\n",
            encoding="utf-8",
        )

        cfg = load_budget_config(cwd=tmp_path)
        assert cfg.caps["openrouter"].cap_usd == 5.0

    def test_non_dict_json_returns_empty_config(self, tmp_path: Path):
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        budget_file = mullet_dir / "budget.json"
        budget_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        cfg = load_budget_config(cwd=tmp_path)
        assert cfg.caps == {}


# --- Coverage gap tests ---

class TestFindBudgetFile:
    def test_cwd_is_file_walks_from_parent(self, tmp_path):
        from mullet.core.providers.spend_caps import _find_budget_file
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        (mullet_dir / "budget.yaml").write_text("openrouter:\n  cap_usd: 5.0\n")
        # Pass a file path (not a directory) — covers line 53 (start = start.parent)
        dummy_file = tmp_path / "dummy.txt"
        dummy_file.write_text("x")
        result = _find_budget_file(dummy_file)
        assert result is not None
        assert result.name == "budget.yaml"

    def test_no_mullet_dir_anywhere_returns_none(self, tmp_path):
        from mullet.core.providers.spend_caps import _find_budget_file
        import mullet.core.providers.spend_caps as sc_mod
        # Walk the real function but cap it at tmp_path so system dirs don't interfere.
        # Replace BUDGET_FILE_NAME and BUDGET_JSON_NAME with names that won't exist.
        orig_yaml = sc_mod.BUDGET_FILE_NAME
        orig_json = sc_mod.BUDGET_JSON_NAME
        sc_mod.BUDGET_FILE_NAME = "__no_such_budget_file__.yaml"
        sc_mod.BUDGET_JSON_NAME = "__no_such_budget_file__.json"
        try:
            nested = tmp_path / "a" / "b" / "c"
            nested.mkdir(parents=True)
            result = _find_budget_file(nested)
            # With non-existent filenames, no .mullet dir will yield a match → None
            assert result is None
        finally:
            sc_mod.BUDGET_FILE_NAME = orig_yaml
            sc_mod.BUDGET_JSON_NAME = orig_json


class TestLoadBudgetConfigGaps:
    def test_invalid_json_content_returns_empty(self, tmp_path):
        """Lines 111-113: JSON file with invalid content → empty config."""
        mullet_dir = tmp_path / ".mullet"
        mullet_dir.mkdir()
        (mullet_dir / "budget.json").write_text("NOT VALID JSON {{{{")
        cfg = load_budget_config(cwd=tmp_path)
        assert cfg.caps == {}

    def test_find_budget_file_raises_returns_empty(self, tmp_path, monkeypatch):
        """Lines 119-121: outer exception in load_budget_config → empty config."""
        monkeypatch.setattr(
            "mullet.core.providers.spend_caps._find_budget_file",
            lambda cwd=None: (_ for _ in ()).throw(OSError("disk error")),
        )
        cfg = load_budget_config(cwd=tmp_path)
        assert cfg.caps == {}

    def test_check_spend_cap_load_raises_returns_allowing(self, monkeypatch):
        """Lines 138-139: load_budget_config raises → check_spend_cap allows."""
        monkeypatch.setattr(
            "mullet.core.providers.spend_caps.load_budget_config",
            lambda cwd=None: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        allowed, reason = check_spend_cap("openrouter", balance_usd=5.0, config=None)
        assert allowed is True
        assert "config load failed" in reason
