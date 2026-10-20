from pathlib import Path
from unittest.mock import mock_open, patch

from mullet.cli.create_pool import create_pool_silent


def test_create_pool_silent_returns_path():
    with patch("mullet.cli.create_pool.get_pools_dir") as mock_get_dir, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("mullet.cli.create_pool.yaml.dump") as mock_yaml:
        mock_get_dir.return_value = Path("/fake/pools")
        result = create_pool_silent("test-slug", "Test Machine")
        assert result == Path("/fake/pools/test-slug")


def test_create_pool_silent_makes_directories():
    with patch("mullet.cli.create_pool.get_pools_dir") as mock_get_dir, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("mullet.cli.create_pool.yaml.dump") as mock_yaml:
        mock_get_dir.return_value = Path("/fake/pools")
        create_pool_silent("slug", "Machine")
        assert mock_mkdir.call_count >= 4


def test_create_pool_silent_default_embedding():
    with patch("mullet.cli.create_pool.get_pools_dir") as mock_get_dir, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("mullet.cli.create_pool.yaml.dump") as mock_yaml:
        mock_get_dir.return_value = Path("/fake/pools")
        create_pool_silent("slug", "Machine")
        config = mock_yaml.call_args[0][0]
        assert config["embedding"]["model"] == "nomic-embed-text"
        assert config["embedding"]["dimension"] == 768


def test_create_pool_silent_custom_embedding():
    with patch("mullet.cli.create_pool.get_pools_dir") as mock_get_dir, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("mullet.cli.create_pool.yaml.dump") as mock_yaml:
        mock_get_dir.return_value = Path("/fake/pools")
        create_pool_silent("slug", "Machine", embedding_model="custom-model")
        config = mock_yaml.call_args[0][0]
        assert config["embedding"]["model"] == "custom-model"


def test_create_pool_silent_writes_correct_config():
    with patch("mullet.cli.create_pool.get_pools_dir") as mock_get_dir, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("mullet.cli.create_pool.yaml.dump") as mock_yaml:
        mock_get_dir.return_value = Path("/fake/pools")
        create_pool_silent("my-pool", "My Machine")
        config = mock_yaml.call_args[0][0]
        assert config["name"] == "my-pool"
        assert config["machine_name"] == "My Machine"
        assert config["paths"]["documents"] == "documents"
        assert config["paths"]["index"] == "index"
        assert config["paths"]["prompts"] == "prompts"
        assert config["embedding"]["hnsw_ef_construction"] == 200
        assert config["embedding"]["hnsw_M"] == 16


def test_create_pool_silent_opens_config_yaml():
    with patch("mullet.cli.create_pool.get_pools_dir") as mock_get_dir, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("mullet.cli.create_pool.yaml.dump") as mock_yaml:
        mock_get_dir.return_value = Path("/fake/pools")
        create_pool_silent("slug", "Machine")
        mock_file.assert_called_once()
        mock_yaml.assert_called_once()


def test_create_pool_silent_yaml_dump_args():
    with patch("mullet.cli.create_pool.get_pools_dir") as mock_get_dir, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("mullet.cli.create_pool.yaml.dump") as mock_yaml:
        mock_get_dir.return_value = Path("/fake/pools")
        create_pool_silent("slug", "Machine")
        assert mock_yaml.call_args[1]["default_flow_style"] is False
        assert mock_yaml.call_args[1]["sort_keys"] is False
