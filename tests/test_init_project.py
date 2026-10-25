"""Tests for `mullet init` project scaffolding."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure mullet package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mullet.cli.init_project import (
    _target_dir_from_remaining,
    _value_from_remaining,
    run_init,
    run_update,
)


@pytest.fixture
def mock_args(tmp_path):
    """Return a default args namespace pointing at tmp_path."""
    return argparse.Namespace(
        target_dir=str(tmp_path),
        pool="test-pool",
        providers="kimi,gemini",
        model="kimi-k2.6",
        speed="balanced",
        force=False,
        dry_run=False,
        no_claude=False,
        no_memory=False,
        no_pool=False,
        remaining=["init", str(tmp_path)],
    )


# ---------------------------------------------------------------------------
# Helper function edge cases
# ---------------------------------------------------------------------------

def test_value_from_remaining_empty():
    assert _value_from_remaining(None, "--foo", "default") == "default"
    assert _value_from_remaining([], "--foo") is None


def test_value_from_remaining_not_found():
    assert _value_from_remaining(["--bar", "val"], "--foo", "default") == "default"


def test_value_from_remaining_at_end_no_value():
    # flag is last element, so no value follows it
    assert _value_from_remaining(["--foo"], "--foo", "default") == "default"


def test_value_from_remaining_next_is_flag():
    assert _value_from_remaining(["--foo", "--bar"], "--foo", "default") == "default"


def test_value_from_remaining_found():
    assert _value_from_remaining(["--foo", "val"], "--foo") == "val"


def test_target_dir_from_remaining_empty():
    assert _target_dir_from_remaining(None) is None
    assert _target_dir_from_remaining([]) is None


def test_target_dir_from_remaining_only_flags():
    # first item is command, rest are flags — no positional target_dir
    assert _target_dir_from_remaining(["init", "--foo", "--bar"]) is None


def test_target_dir_from_remaining_found():
    assert _target_dir_from_remaining(["init", "/some/path"]) == "/some/path"


# ---------------------------------------------------------------------------
# run_init branches
# ---------------------------------------------------------------------------

@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_writes_mullet_yaml(mock_pool, tmp_path, mock_args):
    run_init(mock_args)
    yaml_path = tmp_path / "mullet.yaml"
    assert yaml_path.exists()
    text = yaml_path.read_text()
    assert "client: test-pool" in text
    assert "[kimi, gemini]" in text


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_creates_claude_md_with_test_policy(mock_pool, tmp_path, mock_args):
    run_init(mock_args)
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    assert claude_md.exists()
    text = claude_md.read_text()
    assert "Testing policy" in text
    assert "test-pool" in text


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_copies_loop_agreement(mock_pool, tmp_path, mock_args):
    run_init(mock_args)
    loop = tmp_path / ".mullet" / "LOOP_AGREEMENT.md"
    assert loop.exists()
    assert "Development Loop Agreement" in loop.read_text()


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_installs_hook(mock_pool, tmp_path, mock_args):
    """The project's settings run the hook through `mullet hook`; no script is copied."""
    run_init(mock_args)
    settings = tmp_path / ".claude" / "settings.json"
    data = json.loads(settings.read_text())
    commands = [h["command"] for g in data["hooks"]["UserPromptSubmit"] for h in g["hooks"]]
    assert commands == ["mullet hook delegation-reminder"]
    assert not (tmp_path / ".claude" / "hooks").exists()


def test_init_hook_command_names_a_real_hook():
    from mullet.cli.init_project import _HOOK_COMMAND
    from mullet.hooks import find

    assert _HOOK_COMMAND.startswith("mullet hook ")
    assert find(_HOOK_COMMAND.split()[-1]) is not None


def test_update_points_old_hook_at_mullet_hook(tmp_path, capsys):
    """A project set up by an older `mullet init` gets its hook command updated."""
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    old = "python3 ${CLAUDE_PROJECT_DIR}/.claude/hooks/delegation_reminder.py"
    settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
        {"matcher": "", "hooks": [{"type": "command", "command": old}]}]}}))
    args = argparse.Namespace(
        target_dir=str(tmp_path), dry_run=False, no_claude=False,
        remaining=["update", str(tmp_path)],
    )
    run_update(args)
    text = settings.read_text()
    assert old not in text
    assert "mullet hook delegation-reminder" in text
    assert "Pointed the delegation hook" in capsys.readouterr().out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_seeds_memory(mock_pool, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    args = argparse.Namespace(
        target_dir=str(tmp_path),
        pool="test-pool",
        providers="kimi,gemini",
        model="kimi-k2.6",
        speed="balanced",
        force=False,
        dry_run=False,
        no_claude=False,
        no_memory=False,
        no_pool=False,
        remaining=["init", str(tmp_path)],
    )
    run_init(args)

    encoded = "-" + str(tmp_path).replace("/", "-")
    mem_dir = home / ".claude" / "projects" / encoded / "memory"
    assert mem_dir.exists()
    assert (mem_dir / "MEMORY.md").exists()
    for fname in (
        "feedback_prefer_concurrent.md",
        "feedback_measure_before_delegate.md",
        "feedback_opencode_for_edits.md",
        "feedback_test_writing_delegation.md",
    ):
        assert (mem_dir / fname).exists(), f"missing {fname}"
    # No personal commit-trailer preference is written into the user's project.
    assert not (mem_dir / "feedback_coauthored_by.md").exists()
    assert "Co-Authored-By" not in (mem_dir / "MEMORY.md").read_text()


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_no_force_skips_existing(mock_pool, tmp_path, mock_args):
    existing = tmp_path / "mullet.yaml"
    existing.write_text("preserve me")
    run_init(mock_args)
    assert existing.read_text() == "preserve me"


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_force_overwrites(mock_pool, tmp_path, mock_args):
    existing = tmp_path / "mullet.yaml"
    existing.write_text("old")
    mock_args.force = True
    run_init(mock_args)
    assert "client: test-pool" in existing.read_text()


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_dry_run_writes_nothing(mock_pool, tmp_path, mock_args):
    mock_args.dry_run = True
    run_init(mock_args)
    assert not (tmp_path / "mullet.yaml").exists()
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".mullet").exists()


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_no_claude_skips_claude_dir(mock_pool, tmp_path, mock_args):
    mock_args.no_claude = True
    run_init(mock_args)
    assert not (tmp_path / ".claude").exists()
    assert (tmp_path / "mullet.yaml").exists()


def test_create_pool_silent(tmp_path, monkeypatch):
    from mullet.cli.create_pool import create_pool_silent

    monkeypatch.setattr(
        "mullet.cli.create_pool.get_pools_dir", lambda: tmp_path / "pools"
    )
    path = create_pool_silent(slug="my-pool", machine_name="my-machine")
    assert path.exists()
    cfg = (path / "config.yaml").read_text()
    assert "machine_name: my-machine" in cfg
    assert "dimension: 768" in cfg
    assert "hnsw_ef_construction: 200" in cfg
    assert "hnsw_M: 16" in cfg
    assert (path / "documents").exists()
    assert (path / "index").exists()
    assert (path / "prompts").exists()


# ---------------------------------------------------------------------------
# Additional run_init branches
# ---------------------------------------------------------------------------

@patch("mullet.cli.init_project._pool_config_path")
@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_pool_already_exists(mock_pool, mock_pool_cfg, tmp_path, mock_args, capsys):
    """Line 118: pool config already exists → skip create_pool_silent."""
    mock_pool_cfg.return_value = tmp_path / "existing_pool" / "config.yaml"
    mock_pool_cfg.return_value.parent.mkdir(parents=True, exist_ok=True)
    mock_pool_cfg.return_value.write_text("exists")
    run_init(mock_args)
    mock_pool.assert_not_called()
    captured = capsys.readouterr()
    assert "already exists" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_loop_agreement_exists_no_force(mock_pool, tmp_path, mock_args, capsys):
    """Line 158: LOOP_AGREEMENT.md exists and not force → skip."""
    loop = tmp_path / ".mullet" / "LOOP_AGREEMENT.md"
    loop.parent.mkdir(parents=True, exist_ok=True)
    loop.write_text("existing")
    run_init(mock_args)
    assert loop.read_text() == "existing"
    captured = capsys.readouterr()
    assert "LOOP_AGREEMENT.md exists, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_loop_agreement_not_found_anywhere(mock_pool, tmp_path, mock_args, capsys, monkeypatch):
    """Line 176: docs/LOOP_AGREEMENT.md not found anywhere → skip with warning."""
    # Monkeypatch Path.exists to False for the docs paths
    original_exists = Path.exists

    def fake_exists(self):
        # Only fake out the docs/LOOP_AGREEMENT.md paths
        if "LOOP_AGREEMENT.md" in str(self):
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    run_init(mock_args)
    captured = capsys.readouterr()
    assert "docs/LOOP_AGREEMENT.md not found, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_claude_md_exists_no_force(mock_pool, tmp_path, mock_args, capsys):
    """Line 188: .claude/CLAUDE.md exists and not force → skip."""
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True, exist_ok=True)
    claude_md.write_text("existing")
    run_init(mock_args)
    assert claude_md.read_text() == "existing"
    captured = capsys.readouterr()
    assert ".claude/CLAUDE.md exists, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_settings_json_exists_no_force(mock_pool, tmp_path, mock_args, capsys):
    """Line 205: .claude/settings.json exists and not force → skip."""
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text("existing")
    run_init(mock_args)
    assert settings.read_text() == "existing"
    captured = capsys.readouterr()
    assert ".claude/settings.json exists, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_gitignore_already_has_patterns(mock_pool, tmp_path, mock_args, capsys):
    """Line 248: .gitignore already contains Mullet patterns → skip."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("# Mullet runtime state\n")
    run_init(mock_args)
    captured = capsys.readouterr()
    assert ".gitignore already contains Mullet patterns, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_gitignore_no_trailing_newline(mock_pool, tmp_path, mock_args):
    """Line 256: .gitignore exists without trailing newline → prepend newline."""
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("existing", encoding="utf-8")
    run_init(mock_args)
    text = gitignore.read_text(encoding="utf-8")
    assert text.startswith("existing\n")
    assert "# Mullet runtime state" in text


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_memory_file_exists(mock_pool, tmp_path, mock_args, capsys, monkeypatch):
    """Lines 274-275: memory file already exists → skip."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    encoded = "-" + str(tmp_path).replace("/", "-")
    mem_dir = home / ".claude" / "projects" / encoded / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "feedback_prefer_concurrent.md").write_text("existing")

    run_init(mock_args)
    captured = capsys.readouterr()
    assert "Memory file feedback_prefer_concurrent.md exists, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_memory_template_not_found(mock_pool, tmp_path, mock_args, capsys, monkeypatch):
    """Line 280: memory template not found → skip with warning."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    # Make all memory template files appear missing
    original_exists = Path.exists

    def fake_exists(self):
        if "feedback_" in str(self) and ".md" in str(self) and "memory" in str(self):
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    run_init(mock_args)
    captured = capsys.readouterr()
    assert "Memory template feedback_prefer_concurrent.md not found, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_memory_index_exists(mock_pool, tmp_path, mock_args, capsys, monkeypatch):
    """Lines 289-291: MEMORY.md exists → skip."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    encoded = "-" + str(tmp_path).replace("/", "-")
    mem_dir = home / ".claude" / "projects" / encoded / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "MEMORY.md").write_text("existing")

    run_init(mock_args)
    captured = capsys.readouterr()
    assert "MEMORY.md exists, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_no_memory_skips_memory(mock_pool, tmp_path, mock_args, capsys):
    """Line 291: --no-memory skips memory seeding."""
    mock_args.no_memory = True
    run_init(mock_args)
    captured = capsys.readouterr()
    assert "Memory seeding skipped" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_no_pool_skips_pool(mock_pool, tmp_path, mock_args, capsys):
    """Line 128: --no-pool skips pool creation."""
    mock_args.no_pool = True
    run_init(mock_args)
    mock_pool.assert_not_called()
    captured = capsys.readouterr()
    assert "Pool creation skipped" in captured.out


# ---------------------------------------------------------------------------
# run_update tests
# ---------------------------------------------------------------------------

@pytest.fixture
def update_args(tmp_path):
    return argparse.Namespace(
        target_dir=str(tmp_path),
        dry_run=False,
        no_claude=False,
        remaining=["update", str(tmp_path)],
    )


def test_update_dry_run_writes_nothing(update_args, tmp_path, capsys):
    """run_update dry_run mode writes nothing."""
    update_args.dry_run = True
    # Create existing files so it would try to update them
    mullet_yaml = tmp_path / "mullet.yaml"
    mullet_yaml.write_text("providers: [kimi, gemini]\n")
    loop = tmp_path / ".mullet" / "LOOP_AGREEMENT.md"
    loop.parent.mkdir(parents=True, exist_ok=True)
    loop.write_text("existing")
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True, exist_ok=True)
    claude_md.write_text("existing")
    hook = tmp_path / ".claude" / "hooks" / "delegation_reminder.py"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("existing")

    run_update(update_args)
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    # Files should be unchanged
    assert mullet_yaml.read_text() == "providers: [kimi, gemini]\n"


def test_update_no_claude_skips_claude_files(update_args, tmp_path, capsys):
    """run_update --no-claude skips hook and claude_md updates."""
    update_args.no_claude = True
    mullet_yaml = tmp_path / "mullet.yaml"
    mullet_yaml.write_text("providers: [kimi, gemini]\n")
    run_update(update_args)
    captured = capsys.readouterr()
    assert "mullet.yaml" in captured.out
    # Should not mention updating hook or claude_md
    assert ".claude/hooks/delegation_reminder.py" not in captured.out
    assert ".claude/CLAUDE.md" not in captured.out


def test_update_loop_agreement_not_found(update_args, tmp_path, capsys, monkeypatch):
    """Line 361: docs/LOOP_AGREEMENT.md not found → skip."""
    original_exists = Path.exists

    def fake_exists(self):
        if "LOOP_AGREEMENT.md" in str(self):
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    run_update(update_args)
    captured = capsys.readouterr()
    assert "docs/LOOP_AGREEMENT.md not found" in captured.out


def test_update_mullet_yaml_old_providers(update_args, tmp_path, capsys):
    """Update mullet.yaml providers from old pattern."""
    mullet_yaml = tmp_path / "mullet.yaml"
    mullet_yaml.write_text("providers: [kimi, gemini, anthropic]\n")
    run_update(update_args)
    captured = capsys.readouterr()
    assert "Updated mullet.yaml providers" in captured.out
    text = mullet_yaml.read_text()
    assert "providers: [openrouter, haiku, kimi, sonnet]" in text


def test_update_mullet_yaml_old_model(update_args, tmp_path, capsys):
    """Update mullet.yaml model from old pattern."""
    mullet_yaml = tmp_path / "mullet.yaml"
    mullet_yaml.write_text("providers: [kimi, gemini]\ndefault_model: kimi-k2.6\n")
    run_update(update_args)
    captured = capsys.readouterr()
    assert "Updated mullet.yaml providers" in captured.out
    text = mullet_yaml.read_text()
    assert "default_model: deepseek/deepseek-v4-flash" in text


def test_update_mullet_yaml_already_up_to_date(update_args, tmp_path, capsys):
    """Line 417: mullet.yaml already up to date → skip."""
    mullet_yaml = tmp_path / "mullet.yaml"
    mullet_yaml.write_text("providers: [openrouter, haiku, kimi, sonnet]\n")
    run_update(update_args)
    captured = capsys.readouterr()
    assert "mullet.yaml providers already up to date, skipping" in captured.out


def test_update_mullet_yaml_not_found(update_args, tmp_path, capsys):
    """Line 419: mullet.yaml not found → warning."""
    run_update(update_args)
    captured = capsys.readouterr()
    assert "mullet.yaml not found" in captured.out


def test_update_claude_md_not_found(update_args, tmp_path, capsys):
    """Line 490: .claude/CLAUDE.md not found → warning."""
    mullet_yaml = tmp_path / "mullet.yaml"
    mullet_yaml.write_text("providers: [openrouter, haiku, kimi, sonnet]\n")
    run_update(update_args)
    captured = capsys.readouterr()
    assert ".claude/CLAUDE.md not found" in captured.out


def test_update_claude_md_already_up_to_date(update_args, tmp_path, capsys):
    """Line 488: .claude/CLAUDE.md already up to date → skip."""
    # Create a CLAUDE.md with the new content already in place, structured
    # exactly as run_update produces it (blank line after table and after
    # hard rules, delegation defaults in their own section).
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True, exist_ok=True)
    claude_md.write_text(
        "| Tier | When to use | How |\n"
        "|---|---|---|\n"
        "| **1. Local LLM** | Free. Economy mode, simple short edits | `mullet_opencode` (auto-selects) or `mullet_run --providers local` |\n"
        "| **2. OpenRouter** | Pay-per-token. Default cloud — cheap, reliable workhorse | `mullet_opencode` (default) or `mullet_run --providers openrouter` |\n"
        "| **3. Anthropic Haiku** | Prepaid (Max plan). Substitute OpenRouter when pressure <60% — saves pay-per-token spend | Direct in-context or `mullet_run --providers haiku` |\n"
        "| **4. Anthropic Sonnet** | Prepaid (Max plan). Complex tasks; substitute when pressure <60% | Direct in-context or `mullet_run --providers sonnet` |\n"
        "| **5. Kimi K2** | Pay-per-token. Large context; OpenRouter fallback | `mullet_opencode` or `mullet_run --providers kimi` |\n"
        "| **6. Gemini** | Last resort — pay-per-token, expensive for large contexts | `mullet_run --providers gemini` |\n"
        "| **7. Opus** | Orchestration and commit decisions only | Direct in-context |\n"
        "\n"
        "**Hard rules:**\n"
        "- File edits >5 lines → `mullet_opencode` always — never in-context Opus writing code\n"
        "- Check `check_budget()` before each task — if Anthropic pressure <60%, substitute Haiku/Sonnet for the OpenRouter task (saves pay-per-token spend)\n"
        "- Default to OpenRouter when Anthropic pressure >60%\n"
        "- Mullet's routing matrix (`routing_matrix.py`) enforces this automatically for API delegation calls\n"
        "\n"
        "## Delegation defaults\n"
        "\n"
        "- Default provider: openrouter, kimi, sonnet, haiku\n"
        "- Default model: deepseek/deepseek-v4-flash\n"
    )
    run_update(update_args)
    captured = capsys.readouterr()
    assert ".claude/CLAUDE.md already up to date, skipping" in captured.out


def test_update_claude_md_updates_ladder(update_args, tmp_path, capsys):
    """run_update replaces old ladder table in CLAUDE.md."""
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True, exist_ok=True)
    claude_md.write_text(
        "| Tier | When to use | How |\n"
        "|---|---|---|\n"
        "| **1. Local LLM** | Free | `mullet_opencode` |\n"
        "| **2. OpenRouter** | Pay | `mullet_run` |\n"
        "\n"
        "**Hard rules:**\n"
        "- Old rule 1\n"
        "- Old rule 2\n"
        "\n"
        "## Delegation defaults\n"
        "\n"
        "- Default provider: old providers\n"
        "- Default model: old model\n"
    )
    run_update(update_args)
    captured = capsys.readouterr()
    assert "Updated .claude/CLAUDE.md delegation ladder and defaults" in captured.out
    text = claude_md.read_text()
    assert "| **5. Kimi K2** | Pay-per-token. Large context; OpenRouter fallback" in text
    assert "- Default provider: openrouter, kimi, sonnet, haiku" in text
    assert "- Default model: deepseek/deepseek-v4-flash" in text


def test_update_mullet_yaml_else_branch(update_args, tmp_path, capsys):
    """Line 406: mullet.yaml update preserves non-matching lines."""
    mullet_yaml = tmp_path / "mullet.yaml"
    mullet_yaml.write_text(
        "providers: [kimi, gemini, anthropic]\n"
        "some_other_key: value\n"
        "default_model: kimi-k2.6\n"
    )
    run_update(update_args)
    captured = capsys.readouterr()
    assert "Updated mullet.yaml providers" in captured.out
    text = mullet_yaml.read_text()
    assert "some_other_key: value" in text
    assert "providers: [openrouter, haiku, kimi, sonnet]" in text
    assert "default_model: deepseek/deepseek-v4-flash" in text


def test_update_claude_md_dry_run(update_args, tmp_path, capsys):
    """Line 482: dry-run shows what would change in CLAUDE.md."""
    update_args.dry_run = True
    claude_md = tmp_path / ".claude" / "CLAUDE.md"
    claude_md.parent.mkdir(parents=True, exist_ok=True)
    claude_md.write_text(
        "| Tier | When to use | How |\n"
        "|---|---|---|\n"
        "| **1. Local LLM** | Free | `mullet_opencode` |\n"
        "\n"
        "**Hard rules:**\n"
        "- Old rule\n"
        "\n"
        "## Delegation defaults\n"
        "\n"
        "- Default provider: old\n"
        "- Default model: old\n"
    )
    run_update(update_args)
    captured = capsys.readouterr()
    assert "[dry-run] Would update .claude/CLAUDE.md delegation ladder and defaults" in captured.out
    # File should remain unchanged in dry-run
    assert "- Old rule" in claude_md.read_text()


def test_update_without_old_hook_skips(update_args, tmp_path, capsys):
    """No settings.json with the old hook command: nothing to update."""
    run_update(update_args)
    captured = capsys.readouterr()
    assert "no old delegation hook to update, skipping" in captured.out


@patch("mullet.cli.create_pool.create_pool_silent")
def test_init_memory_index_template_not_found(mock_pool, tmp_path, mock_args, capsys, monkeypatch):
    """Line 285→296: MEMORY.md template missing → skip seeding and fall through to summary."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    original_exists = Path.exists

    def fake_exists(self):
        s = str(self)
        if s.endswith("memory/MEMORY.md") and "templates" in s:
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    run_init(mock_args)
    captured = capsys.readouterr()
    assert "Memory template MEMORY.md not found, skipping" not in captured.out
    assert "mullet init complete for" in captured.out
