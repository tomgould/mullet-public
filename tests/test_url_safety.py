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

