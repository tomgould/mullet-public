from pathlib import Path
from unittest.mock import MagicMock, patch


from mullet.knowledge.engine.ingest_progress import IngestionProgress, IngestionCheckpoint


def test_percent_zero_total():
    p = IngestionProgress(total_files=0)
    assert p.percent == 100.0


def test_percent_half():
    p = IngestionProgress(total_files=10, processed_files=5)
    assert p.percent == 50.0


def test_eta_none():
    p = IngestionProgress(total_files=10)
    assert p.eta_seconds is None


@patch("time.time")
def test_eta_calculation(mock_time):
    mock_time.return_value = 100.0
    p = IngestionProgress(total_files=10, processed_files=2, start_time=90.0)
    assert p.eta_seconds == 40.0


def test_status_completed_with_errors():
    p = IngestionProgress(total_files=5, processed_files=5, failed_files=["a.py"])
    assert p.status == "completed_with_errors"


@patch("time.time")
def test_format_eta_minutes_seconds(mock_time):
    mock_time.return_value = 10.0
    p = IngestionProgress(total_files=10, processed_files=1, start_time=0.0)
    assert p.format_eta() == "1m 30s"


def test_checkpoint_path():
    assert IngestionCheckpoint.checkpoint_path(Path("/tmp/client")) == Path("/tmp/client/.ingest_checkpoint.json")


def test_remove_calls_unlink():
    mock_path = MagicMock(spec=Path)
    IngestionCheckpoint.remove(None, mock_path)
    mock_path.unlink.assert_called_once_with(missing_ok=True)
