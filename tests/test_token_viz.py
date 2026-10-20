"""Tests for TokenUsageViz — per-turn token tracking and visualization."""

from mullet.core.agent.token_viz import TokenUsageViz


def test_empty_viz_format_table():
    viz = TokenUsageViz()
    assert viz.format_table() == "No token usage recorded yet."


def test_empty_viz_total():
    viz = TokenUsageViz()
    t = viz.total()
    assert t["total_tokens_in"] == 0
    assert t["total_tokens_out"] == 0
    assert t["total_tokens_combined"] == 0
    assert t["total_cached_tokens_in"] == 0
    assert t["total_cached_tokens_out"] == 0
    assert t["total_cached_tokens_combined"] == 0


def test_single_turn():
    viz = TokenUsageViz()
    viz.record_turn(100, 50)
    t = viz.total()
    assert t["total_tokens_in"] == 100
    assert t["total_tokens_out"] == 50
    assert t["total_tokens_combined"] == 150
    assert t["total_cached_tokens_combined"] == 0


def test_cached_turn():
    viz = TokenUsageViz()
    viz.record_turn(200, 75, cached=True)
    t = viz.total()
    assert t["total_cached_tokens_in"] == 200
    assert t["total_cached_tokens_out"] == 75
    assert t["total_cached_tokens_combined"] == 275


def test_multiple_turns():
    viz = TokenUsageViz()
    viz.record_turn(100, 50)
    viz.record_turn(200, 75, cached=True)
    viz.record_turn(150, 60)
    t = viz.total()
    assert t["total_tokens_in"] == 450
    assert t["total_tokens_out"] == 185
    assert t["total_tokens_combined"] == 635
    assert t["total_cached_tokens_in"] == 200
    assert t["total_cached_tokens_out"] == 75


def test_format_table_has_header_and_rows():
    viz = TokenUsageViz()
    viz.record_turn(100, 50)
    viz.record_turn(200, 75, cached=True)
    table = viz.format_table()
    lines = table.strip().splitlines()
    assert len(lines) == 4  # header + separator + 2 data rows
    assert "Turn" in lines[0]
    assert "Tokens In" in lines[0]
    assert "Tokens Out" in lines[0]
    assert "Cached" in lines[0]
    assert "-+-" in lines[1]
    assert "100" in lines[2]
    assert "No" in lines[2]
    assert "200" in lines[3]
    assert "Yes" in lines[3]


def test_turns_list_populated():
    viz = TokenUsageViz()
    viz.record_turn(10, 20)
    viz.record_turn(30, 40, cached=True)
    assert len(viz.turns) == 2
    assert viz.turns[0] == {"tokens_in": 10, "tokens_out": 20, "cached": False}
    assert viz.turns[1] == {"tokens_in": 30, "tokens_out": 40, "cached": True}


def test_token_viz_main_block_runs():
    """The if __name__ == '__main__' block in token_viz.py executes without error."""
    import os
    import runpy
    runpy.run_path(
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../mullet/core/agent/token_viz.py")
        ),
        run_name="__main__",
    )
