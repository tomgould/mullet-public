"""Tests for mullet.cli.create_pool — pool directory scaffolding."""
import yaml
from pathlib import Path
from unittest.mock import patch

from mullet.cli.create_pool import create_pool_silent


class TestCreatePoolSilent:
    def test_creates_pool_directory(self, tmp_path):
        with patch("mullet.cli.create_pool.get_pools_dir", return_value=tmp_path):
            result = create_pool_silent("mypool", "mymachine")
        assert result == tmp_path / "mypool"
        assert result.is_dir()

    def test_creates_subdirectories(self, tmp_path):
        with patch("mullet.cli.create_pool.get_pools_dir", return_value=tmp_path):
            create_pool_silent("mypool", "mymachine")
        pool = tmp_path / "mypool"
        assert (pool / "documents").is_dir()
        assert (pool / "index").is_dir()
        assert (pool / "prompts").is_dir()

    def test_creates_config_yaml(self, tmp_path):
        with patch("mullet.cli.create_pool.get_pools_dir", return_value=tmp_path):
            create_pool_silent("mypool", "mymachine")
        config_path = tmp_path / "mypool" / "config.yaml"
        assert config_path.exists()

    def test_config_yaml_has_correct_fields(self, tmp_path):
        with patch("mullet.cli.create_pool.get_pools_dir", return_value=tmp_path):
            create_pool_silent("mypool", "mymachine")
        with open(tmp_path / "mypool" / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["name"] == "mypool"
        assert cfg["machine_name"] == "mymachine"
        assert cfg["embedding"]["model"] == "nomic-embed-text"

    def test_custom_embedding_model(self, tmp_path):
        with patch("mullet.cli.create_pool.get_pools_dir", return_value=tmp_path):
            create_pool_silent("mypool", "mymachine", embedding_model="text-embedding-3")
        with open(tmp_path / "mypool" / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["embedding"]["model"] == "text-embedding-3"

    def test_returns_pool_path(self, tmp_path):
        with patch("mullet.cli.create_pool.get_pools_dir", return_value=tmp_path):
            result = create_pool_silent("testpool", "machine")
        assert isinstance(result, Path)
        assert result.name == "testpool"

    def test_idempotent_on_existing_pool(self, tmp_path):
        with patch("mullet.cli.create_pool.get_pools_dir", return_value=tmp_path):
            create_pool_silent("mypool", "machine1")
            result = create_pool_silent("mypool", "machine2")
        assert result.is_dir()
        with open(tmp_path / "mypool" / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["machine_name"] == "machine2"
