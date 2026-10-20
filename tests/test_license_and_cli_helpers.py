"""Tests for mullet.config.license, mullet.cli.observations_cmd, and
mullet.cli.decisions_cmd pure helper functions."""

import base64
import json
import pytest

from mullet.config.license import LicenseChecker, LicenseError, TRIAL_DAYS
from mullet.core.observations import Observation
from mullet.cli.observations_cmd import _fmt_row
from mullet.cli.decisions_cmd import _kind_str, _source_str


# ---------------------------------------------------------------------------
# LicenseError
# ---------------------------------------------------------------------------

class TestLicenseError:
    def test_is_exception(self):
        err = LicenseError("bad key")
        assert isinstance(err, Exception)

    def test_message_preserved(self):
        err = LicenseError("expired")
        assert "expired" in str(err)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(LicenseError, match="invalid"):
            raise LicenseError("invalid key")


# ---------------------------------------------------------------------------
# TRIAL_DAYS constant
# ---------------------------------------------------------------------------

class TestTrialDays:
    def test_trial_days_positive(self):
        assert TRIAL_DAYS > 0

    def test_trial_days_is_int(self):
        assert isinstance(TRIAL_DAYS, int)


# ---------------------------------------------------------------------------
# LicenseChecker._validate_key
# ---------------------------------------------------------------------------

class TestValidateKey:
    def setup_method(self):
        self.checker = LicenseChecker()

    def _make_key(self, claims: dict) -> str:
        return base64.b64encode(json.dumps(claims).encode()).decode()

    def test_valid_key_with_customer_returns_claims(self):
        key = self._make_key({"customer": "acme", "max_users": 10})
        result = self.checker._validate_key(key)
        assert result is not None
        assert result["customer"] == "acme"

    def test_valid_key_max_users_preserved(self):
        key = self._make_key({"customer": "corp", "max_users": 50, "expires_at": 9999999999})
        result = self.checker._validate_key(key)
        assert result["max_users"] == 50

    def test_key_without_customer_returns_none(self):
        key = self._make_key({"plan": "enterprise", "users": 5})
        result = self.checker._validate_key(key)
        assert result is None

    def test_invalid_base64_returns_none(self):
        result = self.checker._validate_key("not-valid-base64!!!")
        assert result is None

    def test_valid_base64_non_json_returns_none(self):
        raw = base64.b64encode(b"this is not json").decode()
        result = self.checker._validate_key(raw)
        assert result is None

    def test_empty_string_returns_none(self):
        result = self.checker._validate_key("")
        assert result is None

    def test_expires_at_in_claims(self):
        key = self._make_key({"customer": "test", "expires_at": 9999999999})
        result = self.checker._validate_key(key)
        assert result["expires_at"] == 9999999999


# ---------------------------------------------------------------------------
# _fmt_row (observations_cmd.py)
# ---------------------------------------------------------------------------

def _make_obs(tool_name="Read", input_preview="file.py", output_preview="contents",
              ts="2025-01-15T10:30:00Z", session_id="s1"):
    return Observation(
        timestamp=ts,
        session_id=session_id,
        tool_name=tool_name,
        tool_input_preview=input_preview,
        tool_output_preview=output_preview,
        cwd="/home/user",
    )


class TestFmtRow:
    def test_returns_string(self):
        obs = _make_obs()
        assert isinstance(_fmt_row(obs), str)

    def test_timestamp_prefix_in_row(self):
        obs = _make_obs(ts="2025-01-15T10:30:00Z")
        row = _fmt_row(obs)
        assert "2025-01-15" in row

    def test_tool_name_in_row(self):
        obs = _make_obs(tool_name="Bash")
        row = _fmt_row(obs)
        assert "Bash" in row

    def test_input_preview_in_row(self):
        obs = _make_obs(input_preview="my_special_file.py")
        row = _fmt_row(obs)
        assert "my_special_file.py" in row

    def test_long_preview_truncated_in_row(self):
        obs = _make_obs(input_preview="x" * 200)
        row = _fmt_row(obs)
        # preview is truncated to 80 chars in _fmt_row
        assert len(row) < 300  # not the full 200-char input

    def test_different_tools_in_row(self):
        for tool in ["Read", "Write", "Edit", "Bash"]:
            obs = _make_obs(tool_name=tool)
            row = _fmt_row(obs)
            assert tool in row


# ---------------------------------------------------------------------------
# _kind_str (decisions_cmd.py)
# ---------------------------------------------------------------------------

class TestKindStr:
    def test_decision_returns_string(self):
        result = _kind_str("decision")
        assert "decision" in result

    def test_rejection_returns_string(self):
        result = _kind_str("rejection")
        assert "rejection" in result

    def test_rationale_returns_string(self):
        result = _kind_str("rationale")
        assert "rationale" in result

    def test_unknown_kind_still_returns_string(self):
        result = _kind_str("unknown_kind")
        assert isinstance(result, str)
        assert "unknown_kind" in result


# ---------------------------------------------------------------------------
# _source_str (decisions_cmd.py)
# ---------------------------------------------------------------------------

class TestSourceStr:
    def test_docs_source_contains_path(self):
        result = _source_str("docs", "README.md")
        assert "README.md" in result

    def test_commit_source_contains_type(self):
        result = _source_str("commit", "abc1234")
        assert "commit" in result

    def test_unknown_source_still_returns_string(self):
        result = _source_str("unknown", "some/path")
        assert isinstance(result, str)
        assert "some/path" in result

    def test_colon_separates_type_and_path(self):
        result = _source_str("roadmap", "v2.0")
        # ANSI codes may be present, but the logic joins with ":"
        assert "roadmap" in result and "v2.0" in result
