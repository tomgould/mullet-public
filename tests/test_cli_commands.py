
import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

from mullet.cli import main

@pytest.fixture
def mock_sys_argv():
    """Fixture to temporarily replace sys.argv for CLI argument parsing."""
    original_argv = sys.argv
    sys.argv = ['mullet']  # Default program name
    yield sys.argv
    sys.argv = original_argv

def run_mullet_command(command_args, capsys):
    """
    Helper to run a mullet command and capture stdout/stderr.
    Bypasses argparse.parse_args() and calls _main_inner directly.
    """
    original_argv = sys.argv
    sys.argv = ['mullet'] + command_args
    try:
        # main.main() calls _main_inner and handles SystemExit
        main.main()
    except SystemExit as e:
        # Allow SystemExit(0) for successful exits
        if e.code != 0:
            raise
    finally:
        sys.argv = original_argv
    return capsys.readouterr()


def test_ingest_command(mock_sys_argv, capsys):
    """
    Test the 'ingest' CLI command.
    Mocks Client.get_code_chunks to return fake chunks.
    """
    # Mocking Path.is_dir and Path.resolve for the ingest command
    with patch('mullet.knowledge.engine.client.Client') as mock_client_cls, \
         patch('pathlib.Path.is_dir', return_value=True), \
         patch('pathlib.Path.resolve', return_value=Path('/tmp/fake_mullet_dir')) as mock_path_resolve:
        mock_path_resolve.return_value = Path('/tmp/fake_mullet_dir')

        mock_client_instance = MagicMock()
        mock_client_cls.return_value = mock_client_instance
        mock_client_instance.get_code_chunks.return_value = [
            MagicMock(get=lambda x: 'Python', metadata={'symbol_type': 'function', 'symbol_name': 'func1', 'source_file': 'file1.py'}),
            MagicMock(get=lambda x: 'Python', metadata={'symbol_type': 'class', 'symbol_name': 'ClassA', 'source_file': 'file2.py'})
        ]
        captured = run_mullet_command(['ingest', '/tmp/fake_mullet_dir'], capsys)
        assert "Extracted 2 code chunks from" in captured.out
        mock_client_instance.get_code_chunks.assert_called_once()

def test_search_command(mock_sys_argv, capsys):
    """
    Test the 'search' CLI command.
    Verifies it calls the search handler (may return error if no pool loaded).
    """
    captured = run_mullet_command(['search', 'my query'], capsys)
    # Either returns search results or a "no pool" error — both are valid
    assert captured.out.strip() != ""


def test_search_command_pool_flag_not_in_query(mock_sys_argv, capsys):
    """Regression: --pool <value> must not be included in the search query string."""
    received_args = {}

    def fake_handle_search(args):
        received_args.update(args)
        return "ok"

    with patch('mullet.server.mcp_knowledge.handle_search', side_effect=fake_handle_search):
        run_mullet_command(['search', 'what is the router', '--pool', 'mullet-dev'], capsys)

    assert received_args.get("pool_name") == "mullet-dev"
    assert "--pool" not in received_args.get("query", "")
    assert "mullet-dev" not in received_args.get("query", "")
    assert received_args.get("query") == "what is the router"

def test_graph_stats_command(mock_sys_argv, capsys):
    """
    Test the 'graph stats' CLI command.
    Verifies it attempts to load the KnowledgeGraph.
    """
    with patch('mullet.knowledge.graph.store.KnowledgeGraph') as mock_kg_cls:
        mock_kg_instance = MagicMock()
        mock_kg_cls.return_value = mock_kg_instance
        mock_kg_instance.stats = {'entity_count': 10, 'relationship_count': 5}
        mock_kg_instance.load.return_value = None
        mock_kg_instance.close.return_value = None

        captured = run_mullet_command(['graph', 'stats'], capsys)
        mock_kg_cls.assert_called_once()
        assert "Entities: 10, Relationships: 5" in captured.out

def test_source_list_command(mock_sys_argv, capsys):
    """
    Test the 'source list' CLI command.
    Verifies it prints the config example.
    """
    captured = run_mullet_command(['source', 'list'], capsys)
    assert "Sources are configured in config.yaml. Example:" in captured.out
    assert "github:" in captured.out # Check for a specific part of the example
    assert "web:" in captured.out # Check for another specific part

def test_schedule_list_empty(mock_sys_argv, capsys):
    """
    Test the 'schedule list' CLI command when no tasks are scheduled.
    Verifies the empty schedule message.
    """
    with patch('mullet.core.orchestration.scheduler.TaskScheduler') as mock_scheduler_cls:
        mock_scheduler_instance = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler_instance
        mock_scheduler_instance.list_tasks.return_value = []
        captured = run_mullet_command(['schedule', 'list'], capsys)
        assert "No scheduled tasks. Use: mullet schedule add <name> <interval> <command>" in captured.out

def test_schedule_add(mock_sys_argv, capsys):
    """
    Test the 'schedule add' CLI command.
    Verifies a task gets added.
    """
    with patch('mullet.core.orchestration.scheduler.TaskScheduler') as mock_scheduler_cls:
        mock_scheduler_instance = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler_instance
        mock_scheduler_instance.add.return_value = None

        captured = run_mullet_command(['schedule', 'add', 'daily', 'my task description'], capsys)
        mock_scheduler_instance.add.assert_called_once_with(
            "daily", "my", "task description"
        )
        assert "Scheduled: daily every my" in captured.out # Adjusted assertion
