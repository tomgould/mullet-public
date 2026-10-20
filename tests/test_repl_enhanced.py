import pytest
from unittest.mock import patch
from mullet.cli.repl_enhanced import REPLHistory, TabCompleter, MultiLineInput

@pytest.fixture
def history_file(tmp_path):
    return tmp_path / "history.json"

@pytest.fixture
def repl_history(history_file):
    history = REPLHistory(history_file=str(history_file), max_size=3)
    history.history = []
    return history

def test_repl_history_append_and_get(repl_history):
    repl_history.append("cmd1")
    repl_history.append("cmd2")
    assert repl_history.get_all() == ["cmd1", "cmd2"]

def test_repl_history_max_size(repl_history):
    repl_history.append("cmd1")
    repl_history.append("cmd2")
    repl_history.append("cmd3")
    repl_history.append("cmd4")
    assert repl_history.get_all() == ["cmd2", "cmd3", "cmd4"]

def test_repl_history_search(repl_history):
    repl_history.append("list files")
    repl_history.append("cd /tmp")
    repl_history.append("ls -l")
    # "ls" is a substring of "ls -l" but NOT of "list files"
    assert repl_history.search("ls") == ["ls -l"]
    assert repl_history.search("tmp") == ["cd /tmp"]
    assert repl_history.search("list") == ["list files"]

def test_repl_history_load_save(history_file):
    history1 = REPLHistory(history_file=str(history_file), max_size=2)
    history1.append("first")
    history1.append("second")
    del history1

    history2 = REPLHistory(history_file=str(history_file), max_size=2)
    assert history2.get_all() == ["first", "second"]
    history2.append("third")
    assert history2.get_all() == ["second", "third"]

def test_tab_completer_slash_commands():
    completer = TabCompleter(slash_commands=["/help", "/exit", "/search"])
    assert completer.complete("/h", 0) == "/help"
    assert completer.complete("/e", 0) == "/exit"
    assert completer.complete("/s", 0) == "/search"
    assert completer.complete("/exi", 0) == "/exit"
    assert completer.complete("/x", 0) is None

def test_tab_completer_file_paths(tmp_path):
    """Test file path completion with a real temp directory."""
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.py").touch()
    (tmp_path / "folder").mkdir()

    completer = TabCompleter(base_path=str(tmp_path))
    # Complete files starting with "fi"
    matches = []
    i = 0
    while True:
        m = completer.complete("fi", i)
        if m is None:
            break
        matches.append(m)
        i += 1
    assert "file1.txt" in matches
    assert "file2.py" in matches

    # Complete "fo" -> folder/
    assert completer.complete("fo", 0) == "folder/"

@patch('builtins.input', side_effect=['{', '  "key": "value"', '}'])
def test_multi_line_input_json(mock_input):
    ml_input = MultiLineInput()
    result = ml_input.get_multiline_input()
    assert result == '{\n  "key": "value"\n}'

@patch('builtins.input', side_effect=['[', '1, 2,', '3', ']'])
def test_multi_line_input_incomplete_list(mock_input):
    ml_input = MultiLineInput()
    result = ml_input.get_multiline_input()
    assert result == '[\n1, 2,\n3\n]'

@patch('builtins.input', side_effect=['def func():', '    pass', ''])
def test_multi_line_input_python(mock_input):
    ml_input = MultiLineInput()
    result = ml_input.get_multiline_input()
    # "def func():" ends with colon so is incomplete, "    pass" + empty line completes it
    assert "def func():" in result
    assert "pass" in result

@patch('builtins.input', side_effect=['hello', ''])
def test_multi_line_input_simple(mock_input):
    ml_input = MultiLineInput()
    result = ml_input.get_multiline_input()
    # "hello" is complete (no unmatched brackets/quotes, no trailing colon)
    # so it returns immediately without reading more lines
    assert result == "hello"

@patch('builtins.input', side_effect=['', EOFError])
def test_multi_line_input_eof(mock_input):
    ml_input = MultiLineInput()
    result = ml_input.get_multiline_input()
    assert result == ""

@patch('builtins.input', side_effect=[KeyboardInterrupt])
def test_multi_line_input_keyboard_interrupt(mock_input):
    ml_input = MultiLineInput()
    result = ml_input.get_multiline_input()
    assert result is None


def test_repl_history_load_json_decode_error(tmp_path, capsys):
    bad_file = tmp_path / "bad_history.json"
    bad_file.write_text("not json")
    history = REPLHistory(history_file=str(bad_file), max_size=3)
    assert history.get_all() == []


def test_repl_history_save_os_error(tmp_path, capsys):
    history = REPLHistory(history_file=str(tmp_path / "history.json"), max_size=3)
    history.history = ["cmd"]
    # Make the parent non-writable by pointing at a directory path
    with patch("builtins.open", side_effect=OSError("read-only")):
        history._save_history()
    # Should not raise; history unchanged in memory
    assert history.get_all() == ["cmd"]


def test_main_block_runs():
    import runpy
    with patch("builtins.input", return_value="hello"):
        with patch("builtins.print"):
            runpy.run_module("mullet.cli.repl_enhanced", run_name="__main__", alter_sys=True)


def test_is_incomplete_unmatched_quotes():
    ml_input = MultiLineInput()
    assert ml_input._is_incomplete('print("hello)') is True
    assert ml_input._is_incomplete("print('hello)") is True


def test_is_incomplete_unmatched_brackets():
    ml_input = MultiLineInput()
    assert ml_input._is_incomplete("print([1, 2") is True
    assert ml_input._is_incomplete("print({1: 2") is True
    assert ml_input._is_incomplete("print((1, 2") is True
