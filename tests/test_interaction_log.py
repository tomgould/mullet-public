import sqlite3
import pytest
import mullet.core.logging.interaction_log as ilog
from mullet.core.logging.interaction_log import log_interaction

MESSAGES = [{"role": "user", "content": "hello world"}]

@pytest.fixture(autouse=True)
def reset_db(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    monkeypatch.setenv("MULLET_INTERACTIONS_DB", db)
    monkeypatch.delenv("MULLET_LOG_FULL_PROMPTS", raising=False)
    monkeypatch.delenv("MULLET_LOG_FULL_RESPONSES", raising=False)
    ilog._CONNECTION = None
    ilog._DB_INITIALIZED = False
    ilog.MULLET_LOG_FULL_PROMPTS = False
    ilog.MULLET_LOG_FULL_RESPONSES = False
    yield db
    ilog._CONNECTION = None
    ilog._DB_INITIALIZED = False


def _rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM interactions").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def test_creates_table(reset_db):
    log_interaction("test", "test-model", MESSAGES, "test", 10, 5, 100, True)
    assert len(_rows(reset_db)) == 1


def test_stores_provider_and_model(reset_db):
    log_interaction("kimi", "kimi-k2", MESSAGES, "test", 10, 5, 100, True)
    row = _rows(reset_db)[0]
    assert row["provider"] == "kimi"
    assert row["model"] == "kimi-k2"


def test_stores_latency_and_tokens(reset_db):
    log_interaction("test", "test-model", MESSAGES, "test", 10, 5, 150, True)
    row = _rows(reset_db)[0]
    assert row["latency_ms"] == 150
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5


def test_stores_success_true(reset_db):
    log_interaction("test", "test-model", MESSAGES, "test", 10, 5, 100, True)
    assert _rows(reset_db)[0]["success"] == 1


def test_stores_error(reset_db):
    log_interaction("test", "test-model", MESSAGES, None, 0, 0, 100, False, error="timeout")
    row = _rows(reset_db)[0]
    assert row["success"] == 0
    assert row["error"] == "timeout"


def test_prompt_preview_from_last_user_message(reset_db):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello world"}]
    log_interaction("test", "test-model", msgs, "test", 10, 5, 100, True)
    assert _rows(reset_db)[0]["prompt_preview"] == "hello world"


def test_response_preview(reset_db):
    log_interaction("test", "test-model", MESSAGES, "great response", 10, 5, 100, True)
    assert _rows(reset_db)[0]["response_preview"] == "great response"


def test_preview_truncated_at_500(reset_db):
    long = "x" * 600
    log_interaction("test", "test-model", MESSAGES, long, 10, 5, 100, True)
    preview = _rows(reset_db)[0]["response_preview"]
    assert len(preview) <= 503  # 500 chars + possible "..." suffix
    assert preview.startswith("x" * 490)  # at least 490 x's preserved


def test_never_raises_on_bad_input(reset_db):
    # Must not raise even with garbage inputs
    try:
        log_interaction(None, None, [], None, -1, -1, -1, False)
    except Exception as e:
        pytest.fail(f"log_interaction raised unexpectedly: {e}")


def test_full_response_stored_when_env_set(reset_db, monkeypatch):
    monkeypatch.setattr(ilog, "MULLET_LOG_FULL_RESPONSES", True)
    log_interaction("test", "test-model", MESSAGES, "full content", 10, 5, 100, True)
    assert _rows(reset_db)[0]["full_response"] == "full content"
