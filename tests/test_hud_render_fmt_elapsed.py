"""Tests for mullet.hud.render._fmt_elapsed."""
from __future__ import annotations


from mullet.hud.render import _fmt_elapsed


class TestFmtElapsed:
    def test_zero_seconds(self) -> None:
        assert _fmt_elapsed(0) == "0s"

    def test_single_second(self) -> None:
        assert _fmt_elapsed(1) == "1s"

    def test_59_seconds(self) -> None:
        assert _fmt_elapsed(59) == "59s"

    def test_60_seconds_is_one_minute(self) -> None:
        assert _fmt_elapsed(60) == "1m00s"

    def test_61_seconds(self) -> None:
        assert _fmt_elapsed(61) == "1m01s"

    def test_90_seconds(self) -> None:
        assert _fmt_elapsed(90) == "1m30s"

    def test_3599_seconds(self) -> None:
        assert _fmt_elapsed(3599) == "59m59s"

    def test_3600_seconds_is_one_hour(self) -> None:
        assert _fmt_elapsed(3600) == "1h00m"

    def test_3661_seconds(self) -> None:
        assert _fmt_elapsed(3661) == "1h01m"

    def test_7260_seconds(self) -> None:
        assert _fmt_elapsed(7260) == "2h01m"

    def test_float_truncated(self) -> None:
        # Floats are truncated to int
        assert _fmt_elapsed(59.9) == "59s"
        assert _fmt_elapsed(60.5) == "1m00s"

    def test_returns_string(self) -> None:
        assert isinstance(_fmt_elapsed(42), str)

    def test_sub_minute_no_minutes(self) -> None:
        result = _fmt_elapsed(45)
        assert "m" not in result
        assert result.endswith("s")

    def test_hour_format_no_seconds(self) -> None:
        result = _fmt_elapsed(3600)
        assert result.endswith("m")
        assert "s" not in result
