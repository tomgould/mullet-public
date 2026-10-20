"""Tests for pool_tune pure helpers and repl_enhanced pure helpers."""


from mullet.cli.pool_tune import _recommend_inter_batch_sleep
from mullet.cli.repl_enhanced import REPLHistory, TabCompleter, MultiLineInput


# ---------------------------------------------------------------------------
# _recommend_inter_batch_sleep
# ---------------------------------------------------------------------------

class TestRecommendInterBatchSleep:
    def test_none_returns_conservative(self):
        assert _recommend_inter_batch_sleep(None) == 2.0

    def test_above_8192_returns_zero(self):
        assert _recommend_inter_batch_sleep(9000) == 0.0

    def test_exactly_8192_returns_half(self):
        assert _recommend_inter_batch_sleep(8192) == 0.5

    def test_above_4096_returns_half(self):
        assert _recommend_inter_batch_sleep(5000) == 0.5

    def test_exactly_4096_returns_one(self):
        assert _recommend_inter_batch_sleep(4096) == 1.0

    def test_below_4096_returns_one(self):
        assert _recommend_inter_batch_sleep(2000) == 1.0

    def test_zero_returns_one(self):
        assert _recommend_inter_batch_sleep(0) == 1.0

    def test_returns_float(self):
        assert isinstance(_recommend_inter_batch_sleep(5000), float)


# ---------------------------------------------------------------------------
# REPLHistory
# ---------------------------------------------------------------------------

class TestREPLHistory:
    def test_empty_history_on_missing_file(self, tmp_path):
        h = REPLHistory(history_file=str(tmp_path / "history.json"))
        assert h.get_all() == []

    def test_append_stores_command(self, tmp_path):
        h = REPLHistory(history_file=str(tmp_path / "history.json"))
        h.append("ls -l")
        assert "ls -l" in h.get_all()

    def test_append_multiple(self, tmp_path):
        h = REPLHistory(history_file=str(tmp_path / "history.json"))
        h.append("cmd1")
        h.append("cmd2")
        assert h.get_all() == ["cmd1", "cmd2"]

    def test_max_size_enforced(self, tmp_path):
        h = REPLHistory(history_file=str(tmp_path / "history.json"), max_size=3)
        for i in range(5):
            h.append(f"cmd{i}")
        assert len(h.get_all()) == 3

    def test_max_size_keeps_recent(self, tmp_path):
        h = REPLHistory(history_file=str(tmp_path / "history.json"), max_size=2)
        h.append("old1")
        h.append("old2")
        h.append("new1")
        assert "new1" in h.get_all()
        assert "old1" not in h.get_all()

    def test_search_returns_matches(self, tmp_path):
        h = REPLHistory(history_file=str(tmp_path / "history.json"))
        h.append("git status")
        h.append("git push")
        h.append("ls -la")
        results = h.search("git")
        assert len(results) == 2
        assert "ls -la" not in results

    def test_search_no_match_returns_empty(self, tmp_path):
        h = REPLHistory(history_file=str(tmp_path / "history.json"))
        h.append("ls -l")
        assert h.search("nonexistent") == []

    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "history.json")
        h1 = REPLHistory(history_file=path)
        h1.append("persistent cmd")
        h2 = REPLHistory(history_file=path)
        assert "persistent cmd" in h2.get_all()

    def test_invalid_json_file_returns_empty(self, tmp_path):
        path = tmp_path / "history.json"
        path.write_text("not json {")
        h = REPLHistory(history_file=str(path))
        assert h.get_all() == []


# ---------------------------------------------------------------------------
# TabCompleter
# ---------------------------------------------------------------------------

class TestTabCompleter:
    def test_slash_command_first_match(self):
        tc = TabCompleter(slash_commands=["/help", "/exit"])
        assert tc.complete("/h", 0) == "/help"

    def test_slash_command_second_match(self):
        tc = TabCompleter(slash_commands=["/help1", "/help2"])
        assert tc.complete("/help", 1) == "/help2"

    def test_slash_command_no_match_returns_none(self):
        tc = TabCompleter(slash_commands=["/help", "/exit"])
        assert tc.complete("/z", 0) is None

    def test_slash_command_out_of_range_returns_none(self):
        tc = TabCompleter(slash_commands=["/help"])
        assert tc.complete("/h", 1) is None

    def test_empty_slash_commands_returns_none(self):
        tc = TabCompleter(slash_commands=[])
        assert tc.complete("/help", 0) is None

    def test_exact_slash_command_match(self):
        tc = TabCompleter(slash_commands=["/help", "/exit"])
        assert tc.complete("/exit", 0) == "/exit"

    def test_no_slash_commands_defaults_to_empty(self):
        tc = TabCompleter()
        assert tc.complete("/test", 0) is None


# ---------------------------------------------------------------------------
# MultiLineInput._is_incomplete
# ---------------------------------------------------------------------------

class TestMultiLineInputIsIncomplete:
    def setup_method(self):
        self.ml = MultiLineInput()

    def test_simple_string_complete(self):
        assert self.ml._is_incomplete("hello") is False

    def test_unclosed_brace_incomplete(self):
        assert self.ml._is_incomplete("{") is True

    def test_valid_json_object_complete(self):
        assert self.ml._is_incomplete("{}") is False

    def test_valid_json_with_content_complete(self):
        assert self.ml._is_incomplete('{"key": "value"}') is False

    def test_incomplete_json_array_incomplete(self):
        assert self.ml._is_incomplete("[1, 2") is True

    def test_ends_with_colon_incomplete(self):
        assert self.ml._is_incomplete("if x:") is True

    def test_unmatched_paren_incomplete(self):
        assert self.ml._is_incomplete("func(a, b") is True

    def test_matched_parens_complete(self):
        assert self.ml._is_incomplete("func(a, b)") is False

    def test_unclosed_single_quote_incomplete(self):
        assert self.ml._is_incomplete("x = 'hello") is True

    def test_closed_single_quote_complete(self):
        assert self.ml._is_incomplete("x = 'hello'") is False

    def test_unclosed_double_quote_incomplete(self):
        assert self.ml._is_incomplete('x = "hello') is True

    def test_empty_string_complete(self):
        assert self.ml._is_incomplete("") is False

    def test_multiline_complete_statement(self):
        assert self.ml._is_incomplete("a = 1\nb = 2") is False
