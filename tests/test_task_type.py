"""Tests for task type classification."""
from mullet.core.router.complexity import classify_task_type, TASK_TYPE_PROVIDERS


def test_code_tasks():
    assert classify_task_type("write a function to parse JSON") == "code"
    assert classify_task_type("implement the login page") == "code"
    assert classify_task_type("fix the bug in router") == "code"
    assert classify_task_type("refactor the database module") == "code"
    assert classify_task_type("create a new class for users") == "code"


def test_search_tasks():
    assert classify_task_type("find where the router is defined") == "search"
    assert classify_task_type("search for API endpoints") == "search"
    assert classify_task_type("read the configuration file") == "search"
    assert classify_task_type("where is the test directory") == "search"


def test_generate_tasks():
    assert classify_task_type("explain how compression works") == "generate"
    assert classify_task_type("summarize the changes made") == "generate"
    assert classify_task_type("describe the architecture") == "generate"


def test_lookup_tasks():
    assert classify_task_type("what is the version of Python") == "lookup"
    assert classify_task_type("how many tests are there") == "lookup"
    assert classify_task_type("what is the current status") == "lookup"


def test_default_is_code():
    assert classify_task_type("do something") == "code"


def test_task_type_providers():
    assert TASK_TYPE_PROVIDERS["code"] == "ADVANCED"
    assert TASK_TYPE_PROVIDERS["search"] == "LOW"
    assert TASK_TYPE_PROVIDERS["generate"] == "STANDARD"
    assert TASK_TYPE_PROVIDERS["lookup"] == "LOW"
