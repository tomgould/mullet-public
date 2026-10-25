"""Tests for mullet.local.watchdog: probe() and check(), with fakes only."""

from __future__ import annotations

import json

from pathlib import Path

from mullet.local import watchdog
from mullet.local.watchdog import Probe

BASE = "http://localhost:1234/v1"


class FakeFetch:
    """Answers GETs from a dict of url -> (status, json) or exception."""

    def __init__(self, answers):
        self.answers = answers
        self.urls = []

    def __call__(self, url, *, timeout):
        self.urls.append(url)
        answer = self.answers.get(url, OSError("connection refused"))
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakeGeneration:
    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def __call__(self, base_url, model_id, *, timeout):
        self.calls.append((base_url, model_id, timeout))
        return self.answers.get(model_id, True)


def probe_of(*models, reachable=True, latency=0.1):
    def fn(base_url):
        if not reachable:
            return Probe(reachable=False, error="refused")
        return Probe(reachable=True, latency_s=latency, models=list(models))

    return fn


# probe()


def test_probe_uses_native_model_list():
    fetch = FakeFetch({
        "http://localhost:1234/api/v0/models": (200, {"data": [
            {"id": "qwen", "state": "loaded"},
            {"id": "llama", "state": "not-loaded"},
        ]}),
    })
    result = watchdog.probe(BASE, fetch=fetch)
    assert result.reachable is True
    assert result.models == [("qwen", "loaded"), ("llama", "not-loaded")]
    assert result.error is None
    assert result.latency_s is not None
    assert fetch.urls == ["http://localhost:1234/api/v0/models"]


def test_probe_root_strips_v1_and_trailing_slash():
    fetch = FakeFetch({"http://host:9/api/v0/models": (200, {"data": []})})
    result = watchdog.probe("http://host:9/v1/", fetch=fetch)
    assert result.reachable is True
    assert fetch.urls[0] == "http://host:9/api/v0/models"


def test_probe_falls_back_to_openai_models():
    fetch = FakeFetch({
        "http://localhost:1234/v1/models": (200, {"data": [{"id": "qwen"}]}),
    })
    result = watchdog.probe(BASE, fetch=fetch)
    assert result.reachable is True
    assert result.models == [("qwen", "unknown")]
    assert fetch.urls == [
        "http://localhost:1234/api/v0/models",
        "http://localhost:1234/v1/models",
    ]


def test_probe_falls_back_when_native_answer_is_not_a_model_list():
    fetch = FakeFetch({
        "http://localhost:1234/api/v0/models": (200, {"error": "nope"}),
        "http://localhost:1234/v1/models": (200, {"data": [{"id": "m"}]}),
    })
    result = watchdog.probe(BASE, fetch=fetch)
    assert result.models == [("m", "unknown")]


def test_probe_on_fetch_error_is_unreachable_without_raising():
    def broken(url, *, timeout):
        raise TimeoutError("timed out")

    result = watchdog.probe(BASE, fetch=broken)
    assert result.reachable is False
    assert result.latency_s is None
    assert result.models == []
    assert "timed out" in (result.error or "")


def test_probe_measures_latency_with_the_clock():
    ticks = iter([100.0, 102.5])
    fetch = FakeFetch({"http://localhost:1234/api/v0/models": (200, {"data": []})})
    result = watchdog.probe(BASE, fetch=fetch, clock=lambda: next(ticks))
    assert result.latency_s == 2.5


def test_probe_defaults_to_local_llm_url(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", "http://example.invalid:4321/v1")
    fetch = FakeFetch({})
    watchdog.probe(fetch=fetch)
    assert fetch.urls[0] == "http://example.invalid:4321/api/v0/models"


# check(): rule 1, not answering


def test_unreachable_restarts_only_on_second_check(tmp_path):
    state = tmp_path / "s" / "watchdog.json"
    down = probe_of(reachable=False)
    first = watchdog.check(state, base_url=BASE, now=0, probe_fn=down, generation_fn=FakeGeneration())
    assert first.action == "ok"
    assert state.exists()
    second = watchdog.check(state, base_url=BASE, now=120, probe_fn=down, generation_fn=FakeGeneration())
    assert second.action == "restart"
    assert second.reason == "not answering"
    assert second.probe.reachable is False


def test_slow_probe_restarts_only_on_second_check(tmp_path):
    state = tmp_path / "watchdog.json"
    slow = probe_of(latency=12.0)
    assert watchdog.check(state, base_url=BASE, now=0, probe_fn=slow, generation_fn=FakeGeneration()).action == "ok"
    second = watchdog.check(state, base_url=BASE, now=120, probe_fn=slow, generation_fn=FakeGeneration())
    assert (second.action, second.reason) == ("restart", "not answering")


def test_good_check_in_between_resets_not_answering(tmp_path):
    state = tmp_path / "watchdog.json"
    down, up = probe_of(reachable=False), probe_of()
    gen = FakeGeneration()
    assert watchdog.check(state, base_url=BASE, now=0, probe_fn=down, generation_fn=gen).action == "ok"
    assert watchdog.check(state, base_url=BASE, now=120, probe_fn=up, generation_fn=gen).action == "ok"
    assert watchdog.check(state, base_url=BASE, now=240, probe_fn=down, generation_fn=gen).action == "ok"


# check(): rule 2, stuck loading


def test_loading_rule_fires_only_after_five_minutes(tmp_path):
    state = tmp_path / "watchdog.json"
    loading = probe_of(("qwen", "loading"))
    gen = FakeGeneration()
    assert watchdog.check(state, base_url=BASE, now=1000, probe_fn=loading, generation_fn=gen).action == "ok"
    assert watchdog.check(state, base_url=BASE, now=1120, probe_fn=loading, generation_fn=gen).action == "ok"
    assert watchdog.check(state, base_url=BASE, now=1300, probe_fn=loading, generation_fn=gen).action == "ok"
    late = watchdog.check(state, base_url=BASE, now=1301, probe_fn=loading, generation_fn=gen)
    assert (late.action, late.reason) == ("restart", "stuck loading qwen")
    assert gen.calls == []


def test_loading_clock_resets_when_model_finishes_loading(tmp_path):
    state = tmp_path / "watchdog.json"
    gen = FakeGeneration()
    watchdog.check(state, base_url=BASE, now=0, probe_fn=probe_of(("qwen", "loading")), generation_fn=gen)
    watchdog.check(state, base_url=BASE, now=200, probe_fn=probe_of(("qwen", "loaded")), generation_fn=gen)
    again = watchdog.check(state, base_url=BASE, now=400, probe_fn=probe_of(("qwen", "loading")), generation_fn=gen)
    assert again.action == "ok"


def test_loading_first_seen_time_is_kept_in_state(tmp_path):
    state = tmp_path / "watchdog.json"
    watchdog.check(state, base_url=BASE, now=50, probe_fn=probe_of(("qwen", "loading")), generation_fn=FakeGeneration())
    saved = json.loads(state.read_text())
    assert saved["loading_since"] == {"qwen": 50}


# check(): rule 3, generation hangs


def test_generation_hang_restarts_only_on_second_check(tmp_path):
    state = tmp_path / "watchdog.json"
    loaded = probe_of(("qwen", "loaded"))
    gen = FakeGeneration({"qwen": False})
    first = watchdog.check(state, base_url=BASE, now=0, probe_fn=loaded, generation_fn=gen)
    assert first.action == "ok"
    second = watchdog.check(state, base_url=BASE, now=120, probe_fn=loaded, generation_fn=gen)
    assert (second.action, second.reason) == ("restart", "generation hangs on qwen")
    assert gen.calls == [(BASE, "qwen", 30.0), (BASE, "qwen", 30.0)]


def test_good_generation_in_between_resets_hang_rule(tmp_path):
    state = tmp_path / "watchdog.json"
    loaded = probe_of(("qwen", "loaded"))
    hangs, answers = FakeGeneration({"qwen": False}), FakeGeneration({"qwen": True})
    assert watchdog.check(state, base_url=BASE, now=0, probe_fn=loaded, generation_fn=hangs).action == "ok"
    assert watchdog.check(state, base_url=BASE, now=120, probe_fn=loaded, generation_fn=answers).action == "ok"
    assert watchdog.check(state, base_url=BASE, now=240, probe_fn=loaded, generation_fn=hangs).action == "ok"


def test_models_not_loaded_are_never_sent_a_generation(tmp_path):
    state = tmp_path / "watchdog.json"
    mixed = probe_of(("a", "not-loaded"), ("b", "loading"), ("c", "unknown"), ("d", "loaded"))
    gen = FakeGeneration()
    for now in (0, 120, 240):
        watchdog.check(state, base_url=BASE, now=now, probe_fn=mixed, generation_fn=gen)
    assert {model for _, model, _ in gen.calls} == {"d"}


def test_unreachable_server_is_never_sent_a_generation(tmp_path):
    gen = FakeGeneration()
    watchdog.check(tmp_path / "w.json", base_url=BASE, now=0, probe_fn=probe_of(reachable=False), generation_fn=gen)
    assert gen.calls == []


# check(): state file


def test_corrupt_state_file_is_treated_as_no_history(tmp_path):
    state = tmp_path / "watchdog.json"
    state.write_text("{not json")
    result = watchdog.check(state, base_url=BASE, now=0, probe_fn=probe_of(reachable=False), generation_fn=FakeGeneration())
    assert result.action == "ok"
    assert json.loads(state.read_text())["bad_probe"] is True


def test_state_written_atomically_without_leftovers(tmp_path):
    state = tmp_path / "deep" / "dir" / "watchdog.json"
    watchdog.check(state, base_url=BASE, now=0, probe_fn=probe_of(), generation_fn=FakeGeneration())
    assert [p.name for p in state.parent.iterdir()] == ["watchdog.json"]


def test_default_state_path_is_in_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    from mullet.core import paths

    assert watchdog.default_state_path() == paths.state_dir() / "local-watchdog.json"


# restart() helpers


class FakeRun:
    """Record argv and return configured exit codes or raise exceptions."""

    def __init__(self, codes=None, raises=None):
        self.codes = codes or {}
        self.raises = set(raises or ())
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        key = tuple(argv)
        if key in self.raises:
            raise OSError("failed")
        code = self.codes.get(key, 0)
        return type("R", (), {"returncode": code, "stdout": "", "stderr": ""})()


class FakeProbeSequence:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, base_url):
        self.calls.append(base_url)
        answer = self.answers[len(self.calls) - 1]
        if isinstance(answer, BaseException):
            raise answer
        return answer


# restart()


def test_restart_uses_systemctl_when_unit_exists(tmp_path):
    run = FakeRun(codes={
        ("systemctl", "--user", "cat", "lmstudio.service"): 0,
        ("systemctl", "--user", "restart", "lmstudio.service"): 0,
    })
    result = watchdog.restart(
        run=run,
        probe_fn=lambda base: Probe(reachable=True),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=1.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=lambda t, b: None,
    )
    assert result.ok is True
    assert ["systemctl", "--user", "restart", "lmstudio.service"] in result.steps
    assert not any(step[0] == "lms" for step in result.steps)


def test_restart_falls_back_to_lms_server_start_with_port(tmp_path):
    run = FakeRun(codes={
        ("systemctl", "--user", "cat", "lmstudio.service"): 1,
    })
    result = watchdog.restart(
        base_url="http://localhost:1234/v1",
        run=run,
        probe_fn=lambda base: Probe(reachable=True),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=1.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=lambda t, b: None,
    )
    assert result.ok is True
    lms = run.calls[1][0][0]
    assert Path(lms).name == "lms"
    assert run.calls[1][0] == [lms, "server", "stop"]
    assert run.calls[2][0] == [lms, "server", "start", "--port", "1234"]


def test_restart_uses_port_from_custom_base_url(tmp_path):
    run = FakeRun(codes={
        ("systemctl", "--user", "cat", "lmstudio.service"): 1,
    })
    result = watchdog.restart(
        base_url="http://localhost:5678/v1",
        run=run,
        probe_fn=lambda base: Probe(reachable=True),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=1.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=lambda t, b: None,
    )
    lms = run.calls[2][0][0]
    assert run.calls[2][0] == [lms, "server", "start", "--port", "5678"]
    assert result.ok is True


def test_restart_hard_runs_daemon_down_and_up_first(tmp_path):
    run = FakeRun(codes={
        ("systemctl", "--user", "cat", "lmstudio.service"): 1,
    })
    result = watchdog.restart(
        hard=True,
        base_url="http://localhost:1234/v1",
        run=run,
        probe_fn=lambda base: Probe(reachable=True),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=1.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=lambda t, b: None,
    )
    assert result.ok is True
    lms = run.calls[0][0][0]
    assert run.calls[0][0] == [lms, "daemon", "down"]
    assert run.calls[1][0] == [lms, "daemon", "up"]
    assert run.calls[2][0] == ["systemctl", "--user", "cat", "lmstudio.service"]
    assert run.calls[3][0] == [lms, "server", "stop"]
    assert run.calls[4][0] == [lms, "server", "start", "--port", "1234"]


def test_restart_reports_failure_when_probe_never_answers(tmp_path):
    run = FakeRun(codes={
        ("systemctl", "--user", "cat", "lmstudio.service"): 0,
        ("systemctl", "--user", "restart", "lmstudio.service"): 0,
    })
    times = [0.0, 0.5, 1.0, 1.5]
    clock = iter(times).__next__
    probe = FakeProbeSequence([Probe(reachable=False)] * 3)
    result = watchdog.restart(
        run=run,
        probe_fn=probe,
        sleep=lambda s: None,
        clock=clock,
        wait_s=1.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=lambda t, b: None,
    )
    assert result.ok is False
    assert result.error == "probe did not become reachable"


def test_restart_catches_run_exceptions_and_returns_ok_false(tmp_path):
    def broken(argv, **kwargs):
        raise OSError("boom")

    result = watchdog.restart(
        run=broken,
        probe_fn=lambda base: Probe(reachable=False),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=0.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=lambda t, b: None,
    )
    assert result.ok is False
    assert "OSError: boom" in (result.error or "")


def test_restart_logs_one_json_line_per_restart(tmp_path):
    log = tmp_path / "log.jsonl"
    run = FakeRun(codes={
        ("systemctl", "--user", "cat", "lmstudio.service"): 0,
        ("systemctl", "--user", "restart", "lmstudio.service"): 0,
    })
    watchdog.restart(
        reason="not answering",
        run=run,
        probe_fn=lambda base: Probe(reachable=True),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=1.0,
        log_path=log,
        notify_fn=lambda t, b: None,
    )
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["reason"] == "not answering"
    assert entry["ok"] is True
    assert entry["hard"] is False
    assert entry["error"] is None
    assert ["systemctl", "--user", "restart", "lmstudio.service"] in entry["steps"]


def test_restart_notifies_with_fixed_wording(tmp_path):
    notified = []

    def capture(title, body):
        notified.append((title, body))

    watchdog.restart(
        reason="not answering",
        run=FakeRun(codes={("systemctl", "--user", "cat", "lmstudio.service"): 1}),
        probe_fn=lambda base: Probe(reachable=True),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=1.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=capture,
    )
    assert notified == [("Local LLM restarted: not answering", "The local LLM is answering again.")]


def test_restart_notifies_failure_body_when_probe_does_not_come_back(tmp_path):
    notified = []

    def capture(title, body):
        notified.append((title, body))

    watchdog.restart(
        reason="generation hangs on qwen",
        run=FakeRun(codes={("systemctl", "--user", "cat", "lmstudio.service"): 1}),
        probe_fn=lambda base: Probe(reachable=False),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=0.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=capture,
    )
    assert notified == [
        ("Local LLM restarted: generation hangs on qwen", "The local LLM did not come back.")
    ]


def test_restart_result_is_unchanged_when_notify_raises(tmp_path):
    def boom(title, body):
        raise RuntimeError("no notify")

    result = watchdog.restart(
        run=FakeRun(codes={("systemctl", "--user", "cat", "lmstudio.service"): 1}),
        probe_fn=lambda base: Probe(reachable=True),
        sleep=lambda s: None,
        clock=lambda: 0.0,
        wait_s=1.0,
        log_path=tmp_path / "log.jsonl",
        notify_fn=boom,
    )
    assert result.ok is True


def test_last_restart_returns_last_log_line(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"time": 1.0, "reason": "old"}) + "\n"
        + json.dumps({"time": 2.0, "reason": "new"}) + "\n"
    )
    entry = watchdog.last_restart(log)
    assert entry == {"time": 2.0, "reason": "new"}


def test_last_restart_tolerates_missing_file_and_bad_lines(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text("not json\n")
    assert watchdog.last_restart(log) is None
    assert watchdog.last_restart(tmp_path / "missing.jsonl") is None


# check(): cooldown


def test_check_is_busy_within_five_minutes_of_logged_restart(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"time": 100.0, "reason": "not answering", "hard": False, "ok": True, "steps": [], "error": None})
        + "\n"
    )
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"bad_probe": True}))
    result = watchdog.check(
        state,
        base_url=BASE,
        now=300,
        probe_fn=probe_of(reachable=False),
        generation_fn=FakeGeneration(),
        log_path=log,
    )
    assert result.action == "busy"
    assert result.reason == "not answering"


def test_check_restarts_after_five_minute_cooldown(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"time": 100.0, "reason": "not answering", "hard": False, "ok": True, "steps": [], "error": None})
        + "\n"
    )
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"bad_probe": True}))
    result = watchdog.check(
        state,
        base_url=BASE,
        now=401,
        probe_fn=probe_of(reachable=False),
        generation_fn=FakeGeneration(),
        log_path=log,
    )
    assert result.action == "restart"
    assert result.reason == "not answering"


def test_check_without_log_path_uses_default_state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"bad_probe": True}))
    result = watchdog.check(
        state,
        base_url=BASE,
        now=0,
        probe_fn=probe_of(reachable=False),
        generation_fn=FakeGeneration(),
    )
    assert result.action == "restart"


def test_no_generation_while_the_gpu_lease_is_held(tmp_path):
    # A local job attempt holds the lease; a probe would queue behind its work,
    # time out and restart the server under it.
    state = tmp_path / "state.json"
    gen = FakeGeneration({"m": False})
    for now in (0, 120, 240):
        verdict = watchdog.check(state, base_url=BASE, now=now, probe_fn=probe_of(("m", "loaded")),
                                 generation_fn=gen, gpu_busy_fn=lambda: "job 1 step 2")
        assert verdict.action == "ok" and "GPU is in use (job 1 step 2)" in verdict.reason
    assert gen.calls == []
    # Once the GPU is free the checks resume, and a hang still needs two checks.
    free = lambda: None  # noqa: E731
    assert watchdog.check(state, base_url=BASE, now=360, probe_fn=probe_of(("m", "loaded")),
                          generation_fn=gen, gpu_busy_fn=free).action == "ok"
    assert watchdog.check(state, base_url=BASE, now=480, probe_fn=probe_of(("m", "loaded")),
                          generation_fn=gen, gpu_busy_fn=free).action == "restart"
