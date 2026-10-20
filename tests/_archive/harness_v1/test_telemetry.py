"""Tests for mullet.harness.telemetry."""

from __future__ import annotations

import json


from mullet.harness.telemetry import TelemetryCollector
from mullet.cli.telemetry_cli import cmd_telemetry


class TestTelemetryCollector:
    def test_disabled_by_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MULLET_TELEMETRY", raising=False)
        cfg_dir = tmp_path / "config" / "mullet"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "config.json").write_text("{}")
        monkeypatch.setattr(TelemetryCollector, "_CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(
            TelemetryCollector, "_BUFFER_PATH", cfg_dir / "telemetry-buffer.jsonl"
        )
        monkeypatch.setattr(
            TelemetryCollector, "_UUID_PATH", tmp_path / "mullet" / "telemetry_id"
        )
        c = TelemetryCollector()
        assert c.enabled is False

    def test_enabled_via_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MULLET_TELEMETRY", "1")
        monkeypatch.setattr(
            TelemetryCollector, "_CONFIG_DIR", tmp_path / "config" / "mullet"
        )
        monkeypatch.setattr(
            TelemetryCollector,
            "_BUFFER_PATH",
            tmp_path / "config" / "mullet" / "telemetry-buffer.jsonl",
        )
        monkeypatch.setattr(
            TelemetryCollector, "_UUID_PATH", tmp_path / "mullet" / "telemetry_id"
        )
        c = TelemetryCollector()
        assert c.enabled is True

    def test_enabled_via_config_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MULLET_TELEMETRY", raising=False)
        cfg_dir = tmp_path / "config" / "mullet"
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "config.json").write_text('{"telemetry": true}')
        monkeypatch.setattr(TelemetryCollector, "_CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(
            TelemetryCollector, "_BUFFER_PATH", cfg_dir / "telemetry-buffer.jsonl"
        )
        monkeypatch.setattr(
            TelemetryCollector, "_UUID_PATH", tmp_path / "mullet" / "telemetry_id"
        )
        c = TelemetryCollector()
        assert c.enabled is True

    def test_event_no_op_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            TelemetryCollector, "_CONFIG_DIR", tmp_path / "config" / "mullet"
        )
        monkeypatch.setattr(
            TelemetryCollector,
            "_BUFFER_PATH",
            tmp_path / "config" / "mullet" / "telemetry-buffer.jsonl",
        )
        monkeypatch.setattr(
            TelemetryCollector, "_UUID_PATH", tmp_path / "mullet" / "telemetry_id"
        )
        c = TelemetryCollector(enabled=False)
        c.event("test", foo="bar")
        assert c._buffer == []

    def test_event_records_when_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            TelemetryCollector, "_CONFIG_DIR", tmp_path / "config" / "mullet"
        )
        monkeypatch.setattr(
            TelemetryCollector,
            "_BUFFER_PATH",
            tmp_path / "config" / "mullet" / "telemetry-buffer.jsonl",
        )
        monkeypatch.setattr(
            TelemetryCollector, "_UUID_PATH", tmp_path / "mullet" / "telemetry_id"
        )
        c = TelemetryCollector(enabled=True)
        c.event("test", foo="bar")
        assert len(c._buffer) == 1
        ev = c._buffer[0]
        assert ev.name == "test"
        assert ev.properties == {"foo": "bar"}
        assert ev.session_id != ""

    def test_uuid_persists_across_runs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            TelemetryCollector, "_CONFIG_DIR", tmp_path / "config" / "mullet"
        )
        monkeypatch.setattr(
            TelemetryCollector,
            "_BUFFER_PATH",
            tmp_path / "config" / "mullet" / "telemetry-buffer.jsonl",
        )
        uuid_path = tmp_path / "mullet" / "telemetry_id"
        monkeypatch.setattr(TelemetryCollector, "_UUID_PATH", uuid_path)
        c1 = TelemetryCollector(enabled=True)
        uuid1 = c1.session_id
        c2 = TelemetryCollector(enabled=True)
        assert c2.session_id == uuid1

    def test_no_path_collected_in_event(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            TelemetryCollector, "_CONFIG_DIR", tmp_path / "config" / "mullet"
        )
        monkeypatch.setattr(
            TelemetryCollector,
            "_BUFFER_PATH",
            tmp_path / "config" / "mullet" / "telemetry-buffer.jsonl",
        )
        monkeypatch.setattr(
            TelemetryCollector, "_UUID_PATH", tmp_path / "mullet" / "telemetry_id"
        )
        c = TelemetryCollector(enabled=True)
        c.event(
            "test",
            path="/home/user/secret.txt",
            nested={"file": "C:\\Users\\x\\doc.pdf"},
        )
        ev = c._buffer[0]
        assert ev.properties["path"] == "[redacted]"
        assert ev.properties["nested"]["file"] == "[redacted]"

    def test_flush_writes_jsonl(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            TelemetryCollector, "_CONFIG_DIR", tmp_path / "config" / "mullet"
        )
        bp = tmp_path / "config" / "mullet" / "telemetry-buffer.jsonl"
        monkeypatch.setattr(TelemetryCollector, "_BUFFER_PATH", bp)
        monkeypatch.setattr(
            TelemetryCollector, "_UUID_PATH", tmp_path / "mullet" / "telemetry_id"
        )
        c = TelemetryCollector(enabled=True)
        c.event("test", foo="bar")
        c.flush()
        assert bp.exists()
        lines = bp.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["name"] == "test"
        assert data["properties"]["foo"] == "bar"


class TestTelemetryCLI:
    def test_cli_status_outputs_disclosure(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "mullet.cli.telemetry_cli._config_path", lambda: tmp_path / "config.json"
        )
        monkeypatch.setattr(
            "mullet.cli.telemetry_cli._buffer_path", lambda: tmp_path / "buffer.jsonl"
        )
        monkeypatch.setattr(
            "mullet.cli.telemetry_cli._uuid_path", lambda: tmp_path / "id"
        )
        cmd_telemetry(["status"])
        captured = capsys.readouterr()
        assert "Telemetry is disabled" in captured.out
        assert "What we collect" in captured.out
        assert "What we do NOT collect" in captured.out

    def test_cli_off_disables_persistently(self, monkeypatch, tmp_path, capsys):
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr(
            "mullet.cli.telemetry_cli._config_path", lambda: cfg_path
        )
        monkeypatch.setattr(
            "mullet.cli.telemetry_cli._buffer_path", lambda: tmp_path / "buffer.jsonl"
        )
        monkeypatch.setattr(
            "mullet.cli.telemetry_cli._uuid_path", lambda: tmp_path / "id"
        )
        cmd_telemetry(["off"])
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert data.get("telemetry") is False
        captured = capsys.readouterr()
        assert "Telemetry disabled" in captured.out
