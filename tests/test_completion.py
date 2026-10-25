"""Tests for mullet.cli.completion — shell completion script generation."""
from mullet.cli.completion import bash_completion, zsh_completion, get_completion


class TestBashCompletion:
    def test_returns_string(self):
        assert isinstance(bash_completion(), str)

    def test_contains_function_definition(self):
        script = bash_completion()
        assert "_mullet_complete()" in script
        assert "complete -F _mullet_complete mullet" in script

    def test_contains_core_commands(self):
        script = bash_completion()
        for cmd in ("job", "reindex", "hud", "budget", "mcp"):
            assert cmd in script

    def test_contains_provider_options(self):
        script = bash_completion()
        for p in ("kimi", "gemini", "anthropic", "local"):
            assert p in script

    def test_contains_tier_options(self):
        script = bash_completion()
        for t in ("LOCAL", "STANDARD", "ADVANCED", "CRITICAL"):
            assert t in script

    def test_pool_subcommands_present(self):
        script = bash_completion()
        for cmd in ("list", "delete", "audit"):
            assert cmd in script

    def test_budget_command_present(self):
        assert "budget" in bash_completion()


class TestZshCompletion:
    def test_returns_string(self):
        assert isinstance(zsh_completion(), str)

    def test_starts_with_compdef(self):
        assert zsh_completion().startswith("#compdef mullet")

    def test_contains_core_commands(self):
        script = zsh_completion()
        for cmd in ("job", "reindex", "hud", "budget"):
            assert cmd in script

    def test_contains_provider_descriptions(self):
        script = zsh_completion()
        assert "kimi" in script
        assert "gemini" in script

    def test_contains_tier_flags(self):
        script = zsh_completion()
        assert "LOCAL" in script
        assert "ADVANCED" in script

    def test_budget_command_present(self):
        assert "budget" in zsh_completion()


class TestGetCompletion:
    def test_bash_dispatches_correctly(self):
        result = get_completion("bash")
        assert "_mullet_complete" in result

    def test_zsh_dispatches_correctly(self):
        result = get_completion("zsh")
        assert "#compdef mullet" in result

    def test_unsupported_shell_returns_comment(self):
        result = get_completion("fish")
        assert "Unsupported shell" in result
        assert "fish" in result
