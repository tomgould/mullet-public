"""Tests for load_client strict-miss behaviour.

Regression: load_client used to silently create empty pool directories
for any unknown name (display name, typo, wrong case). Now it raises
ValueError unless create_if_missing=True, and leaves disk untouched.
"""
import pytest
import yaml

from mullet.knowledge.engine.client import load_client, discover_clients


def _make_pool(root, slug, config=None):
    pool = root / slug
    pool.mkdir(parents=True)
    (pool / "config.yaml").write_text(yaml.safe_dump(config or {"machine_name": slug}))
    return pool


def test_load_client_raises_on_missing_pool(tmp_path):
    _make_pool(tmp_path, "real_pool")

    with pytest.raises(ValueError) as exc_info:
        load_client("does_not_exist", clients_dir=tmp_path)

    msg = str(exc_info.value)
    assert "does_not_exist" in msg
    assert "not found" in msg
    assert "real_pool" in msg


def test_load_client_create_if_missing_true_creates_dirs(tmp_path):
    client = load_client("brand_new_pool", clients_dir=tmp_path, create_if_missing=True)

    assert (tmp_path / "brand_new_pool").is_dir()
    assert client.documents_path.is_dir()
    assert client.index_path.is_dir()
    assert client.prompts_path.is_dir()


def test_load_client_suggests_close_match(tmp_path):
    _make_pool(tmp_path, "claude_code_ecosystem")

    with pytest.raises(ValueError) as exc_info:
        load_client("claude_code_eco", clients_dir=tmp_path)

    msg = str(exc_info.value)
    assert "Did you mean" in msg
    assert "claude_code_ecosystem" in msg


def test_load_client_no_phantom_dir_on_miss(tmp_path):
    _make_pool(tmp_path, "real_pool")
    candidate = tmp_path / "Claude Code Ecosystem"

    with pytest.raises(ValueError):
        load_client("Claude Code Ecosystem", clients_dir=tmp_path)

    assert not candidate.exists(), "load_client must not create a dir for an unknown pool"
    # discover_clients should still only report the one real pool
    assert discover_clients(tmp_path) == ["real_pool"]


def test_load_client_empty_pools_dir_reports_none(tmp_path):
    # No pools at all — error message should still be readable
    with pytest.raises(ValueError) as exc_info:
        load_client("anything", clients_dir=tmp_path)

    msg = str(exc_info.value)
    assert "(none)" in msg


def test_load_client_resolves_display_name(tmp_path):
    """Passing a display name (from config.yaml:name) resolves to the slug."""
    _make_pool(tmp_path, "claude_code_ecosystem", {
        "name": "Claude Code Ecosystem",
        "machine_name": "claude_code_ecosystem",
    })

    client = load_client("Claude Code Ecosystem", clients_dir=tmp_path)
    assert client.machine_name == "claude_code_ecosystem"
    assert (tmp_path / "claude_code_ecosystem").is_dir()


def test_load_client_resolves_display_name_case_insensitive(tmp_path):
    _make_pool(tmp_path, "claude_code_ecosystem", {
        "name": "Claude Code Ecosystem",
        "machine_name": "claude_code_ecosystem",
    })

    client = load_client("claude code ecosystem", clients_dir=tmp_path)
    assert client.machine_name == "claude_code_ecosystem"


def test_load_client_ambiguous_display_name_errors(tmp_path):
    """If the fuzzy match hits 2+ pools, raise a clear ambiguity error."""
    _make_pool(tmp_path, "pool_one", {
        "name": "Shared Prefix Alpha",
        "machine_name": "pool_one",
    })
    _make_pool(tmp_path, "pool_two", {
        "name": "Shared Prefix Beta",
        "machine_name": "pool_two",
    })

    with pytest.raises(ValueError) as exc_info:
        load_client("Shared Prefix", clients_dir=tmp_path)

    msg = str(exc_info.value)
    assert "ambiguous" in msg.lower()
    assert "pool_one" in msg
    assert "pool_two" in msg


def test_load_client_display_name_no_match_falls_through(tmp_path):
    """If no display name matches, the original 'did you mean' error still fires."""
    _make_pool(tmp_path, "alpha_pool", {
        "name": "Alpha Pool",
        "machine_name": "alpha_pool",
    })

    with pytest.raises(ValueError) as exc_info:
        load_client("totally_unrelated", clients_dir=tmp_path)

    msg = str(exc_info.value)
    assert "not found" in msg
