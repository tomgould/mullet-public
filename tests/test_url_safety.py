import pytest

from mullet.core.tools.url_safety import is_safe_url

def test_safe_http_url():
    assert is_safe_url("http://example.com") is True

def test_safe_https_url():
    assert is_safe_url("https://secure.com/path") is True

def test_unsafe_file_scheme():
    assert is_safe_url("file:///etc/passwd") is False

def test_unsafe_ftp_scheme():
    assert is_safe_url("ftp://internal.server/data") is False

def test_localhost_ip():
    assert is_safe_url("http://127.0.0.1/admin") is False

def test_private_ip_range():
    assert is_safe_url("http://192.168.1.100/resource") is False

def test_private_ip_range_another():
    assert is_safe_url("http://10.0.0.5/api") is False



def test_localhost_with_port_is_refused():
    assert is_safe_url("http://localhost:1234/") is False


def test_name_that_resolves_to_loopback_is_refused(monkeypatch):
    monkeypatch.setattr("mullet.core.tools.url_safety._resolve", lambda host: {"127.0.0.1"})
    assert is_safe_url("http://localtest.me/") is False


def test_one_private_answer_among_public_ones_is_refused(monkeypatch):
    monkeypatch.setattr(
        "mullet.core.tools.url_safety._resolve", lambda host: {"93.184.216.34", "10.0.0.7"}
    )
    assert is_safe_url("https://mixed.example/") is False


def test_lookup_failure_is_refused(monkeypatch):
    def fail(host):
        raise OSError("no such host")

    monkeypatch.setattr("mullet.core.tools.url_safety._resolve", fail)
    assert is_safe_url("https://nowhere.example/") is False


def test_ipv4_mapped_ipv6_loopback_is_refused():
    assert is_safe_url("http://[::ffff:127.0.0.1]/") is False


def test_carrier_grade_nat_and_metadata_addresses_are_refused():
    assert is_safe_url("http://100.64.0.1/") is False
    assert is_safe_url("http://169.254.169.254/latest/meta-data/") is False


def test_redirect_to_an_internal_address_is_refused():
    import urllib.error
    import urllib.request

    from mullet.core.tools.url_safety import _CheckedRedirects

    req = urllib.request.Request("https://public.example/")
    with pytest.raises(urllib.error.URLError):
        _CheckedRedirects().redirect_request(
            req, None, 302, "Found", {}, "http://127.0.0.1:8080/admin"
        )


def test_safe_urlopen_refuses_a_private_first_url():
    import urllib.error

    from mullet.core.tools.url_safety import safe_urlopen

    with pytest.raises(urllib.error.URLError):
        safe_urlopen("http://192.168.1.1/", timeout=1)
