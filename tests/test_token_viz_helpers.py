"""Tests for mullet.core.agent.token_viz.TokenUsageViz — pure accounting."""

from mullet.core.agent.token_viz import TokenUsageViz


class TestTokenUsageVizRecordTurn:
    def test_empty_on_init(self):
        viz = TokenUsageViz()
        assert viz.turns == []

    def test_record_adds_turn(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert len(viz.turns) == 1

    def test_record_stores_correct_values(self):
        viz = TokenUsageViz()
        viz.record_turn(200, 75, cached=True)
        assert viz.turns[0]["tokens_in"] == 200
        assert viz.turns[0]["tokens_out"] == 75
        assert viz.turns[0]["cached"] is True

    def test_default_cached_false(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        assert viz.turns[0]["cached"] is False

    def test_multiple_turns(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75, cached=True)
        viz.record_turn(150, 60)
        assert len(viz.turns) == 3


class TestTokenUsageVizFormatTable:
    def test_empty_turns_returns_message(self):
        viz = TokenUsageViz()
        result = viz.format_table()
        assert "No token usage" in result

    def test_table_contains_turn_number(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        table = viz.format_table()
        assert "1" in table

    def test_table_contains_token_values(self):
        viz = TokenUsageViz()
        viz.record_turn(1234, 567)
        table = viz.format_table()
        assert "1234" in table
        assert "567" in table

    def test_cached_shows_yes(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        assert "Yes" in viz.format_table()

    def test_not_cached_shows_no(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=False)
        assert "No" in viz.format_table()

    def test_header_present(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        table = viz.format_table()
        assert "Turn" in table
        assert "Tokens In" in table
        assert "Tokens Out" in table
        assert "Cached" in table

    def test_multiple_turns_numbered(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        table = viz.format_table()
        assert "1" in table
        assert "2" in table


class TestTokenUsageVizTotal:
    def test_empty_turns_returns_zeros(self):
        viz = TokenUsageViz()
        t = viz.total()
        assert t["total_tokens_in"] == 0
        assert t["total_tokens_out"] == 0
        assert t["total_tokens_combined"] == 0

    def test_sums_tokens_correctly(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50)
        viz.record_turn(200, 75)
        t = viz.total()
        assert t["total_tokens_in"] == 300
        assert t["total_tokens_out"] == 125
        assert t["total_tokens_combined"] == 425

    def test_cached_totals_correct(self):
        viz = TokenUsageViz()
        viz.record_turn(100, 50, cached=True)
        viz.record_turn(200, 75, cached=False)
        t = viz.total()
        assert t["total_cached_tokens_in"] == 100
        assert t["total_cached_tokens_out"] == 50
        assert t["total_cached_tokens_combined"] == 150

    def test_uncached_not_counted_in_cached(self):
        viz = TokenUsageViz()
        viz.record_turn(300, 100, cached=False)
        t = viz.total()
        assert t["total_cached_tokens_in"] == 0
        assert t["total_cached_tokens_combined"] == 0

    def test_all_keys_present(self):
        viz = TokenUsageViz()
        t = viz.total()
        for key in ("total_tokens_in", "total_tokens_out", "total_tokens_combined",
                    "total_cached_tokens_in", "total_cached_tokens_out",
                    "total_cached_tokens_combined"):
            assert key in t
