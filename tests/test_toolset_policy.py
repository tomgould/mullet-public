"""Per-provider toolsets for delegated models, set in the Mullet config file."""

from mullet.core.toolset_policy import (
    ALL, SAFE_DEFAULT, ToolsetPolicy, apply_policy, load_policy, parse_policy, risky_grants,
)

EVERY = ["terminal", "file", "web", "code", "browser", "delegation", "session", "git", "testing", "vision", "docker", "lsp"]


def test_claude_providers_keep_every_toolset_by_default():
    for name in ("anthropic", "claude-cli", "claude_max", "Anthropic"):
        assert apply_policy(EVERY, name, ToolsetPolicy()) == EVERY


def test_any_other_provider_gets_only_the_safe_default():
    got = apply_policy(EVERY, "kimi", ToolsetPolicy())
    assert got == [t for t in EVERY if t in SAFE_DEFAULT]
    assert not {"terminal", "file", "code", "docker"} & set(got)


def test_an_unknown_or_missing_provider_name_is_treated_as_untrusted():
    assert "terminal" not in apply_policy(EVERY, None, ToolsetPolicy())
    assert "terminal" not in apply_policy(EVERY, object(), ToolsetPolicy())


def test_the_config_can_give_one_provider_everything_and_another_a_list():
    policy = parse_policy({"kimi": "all", "local": ["terminal", "file", "web"], "default": ["web"]})
    assert apply_policy(EVERY, "kimi", policy) == EVERY
    assert apply_policy(EVERY, "local", policy) == ["terminal", "file", "web"]
    assert apply_policy(EVERY, "gemini", policy) == ["web"]
    assert policy.problems == []


def test_the_config_can_also_narrow_a_claude_provider():
    policy = parse_policy({"anthropic": ["web"]})
    assert apply_policy(EVERY, "anthropic", policy) == ["web"]


def test_bad_entries_are_reported_and_ignored():
    policy = parse_policy({"kimi": 5, "local": ["web", 3], "default": "everything"})
    assert len(policy.problems) == 3
    assert apply_policy(EVERY, "kimi", policy) == [t for t in EVERY if t in SAFE_DEFAULT]


def test_a_non_mapping_block_keeps_the_default_and_says_so():
    policy = parse_policy(["kimi"])
    assert policy.problems and policy.default == list(SAFE_DEFAULT)


def test_load_policy_reads_the_config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("delegation:\n  toolsets:\n    kimi: all\n    default: [web, browser]\n")
    policy = load_policy(path)
    assert policy.allowed("kimi") == ALL and policy.allowed("openrouter") == ["web", "browser"]


def test_a_missing_or_broken_config_file_gives_the_shipped_default(tmp_path):
    assert load_policy(tmp_path / "none.yaml").allowed("kimi") == list(SAFE_DEFAULT)
    broken = tmp_path / "bad.yaml"
    broken.write_text("delegation: [unclosed")
    policy = load_policy(broken)
    assert policy.problems and policy.allowed("kimi") == list(SAFE_DEFAULT)


def test_the_default_config_path_is_used_when_none_is_given(tmp_path, monkeypatch):
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yaml").write_text("delegation:\n  toolsets:\n    kimi: all\n")
    assert apply_policy(EVERY, "kimi") == EVERY


def test_risky_grants_names_every_untrusted_provider_that_can_act_on_the_machine():
    assert risky_grants(ToolsetPolicy(), EVERY) == []
    policy = parse_policy({"kimi": "all", "local": ["web", "terminal"], "default": ["web"]})
    found = dict(risky_grants(policy, EVERY))
    assert found["kimi"] == ["terminal", "file", "code", "git", "testing", "docker"]
    assert found["local"] == ["terminal"] and "default" not in found and "anthropic" not in found


def test_doctor_warns_about_each_provider_that_may_act_on_the_machine():
    from mullet.cli.doctor_checks import toolset_checks

    assert toolset_checks(ToolsetPolicy()) == []
    rows = toolset_checks(parse_policy({"kimi": "all", "default": ["web"]}))
    assert len(rows) == 1 and rows[0][2] == "warn"
    assert "kimi (terminal, file" in rows[0][1] and "delegation.toolsets" in rows[0][1]
