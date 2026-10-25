"""The key proxy forwards an allowlist of headers and clamps `n` and output-token fields."""

import json
from pathlib import Path

from mullet.jobs.executors.keyproxy import FORWARD_HEADERS, KeyProxy
from tests.test_jobs_keyproxy import KEY, _Upstream, call, proxy, upstream  # noqa: F401  (fixtures)


def _bare(**kwargs) -> KeyProxy:
    return KeyProxy(Path("/nonexistent/sock"), "https://example.com/v1", KEY, **kwargs)


def test_unlisted_client_headers_are_not_forwarded(proxy):
    p = proxy()
    call(p, headers={
        "OpenAI-Organization": "org-victim", "X-Forwarded-For": "1.2.3.4", "X-Title": "kept",
        "Accept": "application/json", "Content-Type": "application/json",
    })
    seen = {k.lower(): v for k, v in _Upstream.seen[-1]["headers"].items()}
    assert "openai-organization" not in seen and "x-forwarded-for" not in seen
    assert seen["x-title"] == "kept" and seen["accept"] == "application/json"
    assert seen["authorization"] == f"Bearer {KEY}"


def test_the_allowlist_is_a_parameter(proxy):
    p = proxy(forward_headers={"x-custom"})
    call(p, headers={"X-Custom": "yes", "X-Title": "no"})
    seen = {k.lower() for k in _Upstream.seen[-1]["headers"]}
    assert "x-custom" in seen and "x-title" not in seen


def test_the_default_allowlist_has_no_organization_header():
    assert "openai-organization" not in FORWARD_HEADERS
    assert {"content-type", "accept", "user-agent", "accept-language", "x-title",
            "http-referer"} <= FORWARD_HEADERS


def test_a_keyless_proxy_still_passes_the_clients_own_credentials(proxy):
    p = proxy(key=None)
    call(p)
    assert _Upstream.seen[-1]["headers"]["Authorization"] == "Bearer mullet-dummy"


def test_n_is_clamped_to_one():
    p = _bare()
    out = json.loads(p.with_fields(json.dumps({"model": "m", "n": 8}).encode()))
    assert out == {"model": "m", "n": 1}


def test_n_of_one_leaves_the_body_byte_identical():
    p = _bare()
    body = b'{"model": "m",  "n": 1}'
    assert p.with_fields(body) == body


def test_a_non_integer_n_is_replaced():
    p = _bare()
    assert json.loads(p.with_fields(b'{"n": "100"}')) == {"n": 1}
    assert json.loads(p.with_fields(b'{"n": true}')) == {"n": 1}


def test_max_n_is_configurable():
    p = _bare(max_n=3)
    assert json.loads(p.with_fields(b'{"n": 9}')) == {"n": 3}
    assert json.loads(p.with_fields(b'{"n": 2}')) == {"n": 2}


def test_token_fields_are_untouched_by_default():
    p = _bare()
    body = b'{"max_tokens": 999999, "max_completion_tokens": 999999}'
    assert p.with_fields(body) == body


def test_token_fields_are_clamped_when_a_limit_is_set():
    p = _bare(max_output_tokens=4096)
    body = json.dumps({"max_tokens": 10**6, "max_completion_tokens": 100,
                       "max_output_tokens": "lots", "model": "m"}).encode()
    out = json.loads(p.with_fields(body))
    assert out == {"max_tokens": 4096, "max_completion_tokens": 100,
                   "max_output_tokens": 4096, "model": "m"}


def test_a_body_with_a_repeated_key_is_refused():
    assert _bare().with_fields(b'{"n": 1, "n": 50}') is None


def test_a_non_json_body_passes_when_no_fields_are_set():
    assert _bare().with_fields(b"--multipart--") == b"--multipart--"


def test_body_fields_still_apply_with_the_clamp():
    p = _bare(body_fields={"provider": {"order": ["x"]}})
    out = json.loads(p.with_fields(b'{"n": 5}'))
    assert out == {"n": 1, "provider": {"order": ["x"]}}


def test_the_clamp_reaches_the_upstream(proxy):
    p = proxy(max_output_tokens=256)
    call(p, body=json.dumps({"n": 10, "max_tokens": 5000}).encode())
    assert json.loads(_Upstream.seen[-1]["body"]) == {"n": 1, "max_tokens": 256}
