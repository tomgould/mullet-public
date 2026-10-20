"""Tests for the voice harness (recorder + transcriber + /voice command).

All tests use mocked backends — no real microphone or network calls.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


from mullet.harness.voice.recorder import VoiceRecorder
from mullet.harness.voice.transcriber import VoiceTranscriber
from mullet.harness.commands.voice import handle_voice_command


class TestVoiceRecorder:
    def test_recorder_detects_backend(self, monkeypatch):
        """detect_backend returns the first available tool."""
        monkeypatch.setattr("mullet.harness.voice.recorder.shutil.which", lambda cmd: cmd == "sox")
        recorder = VoiceRecorder()
        assert recorder.detect_backend() == "sox"

    def test_recorder_with_no_backend_returns_none(self, monkeypatch):
        """detect_backend returns None when nothing is installed."""
        monkeypatch.setattr("mullet.harness.voice.recorder.shutil.which", lambda _cmd: None)
        recorder = VoiceRecorder()
        assert recorder.detect_backend() is None

    def test_record_to_returns_dict_with_path(self, monkeypatch, tmp_path):
        """record_to produces a WAV file and returns path/duration info."""
        wav_path = tmp_path / "out.wav"
        fake_proc = MagicMock()
        fake_proc.wait.return_value = 0
        monkeypatch.setattr(
            "mullet.harness.voice.recorder.subprocess.Popen",
            lambda *_a, **_k: fake_proc,
        )
        monkeypatch.setattr(
            "mullet.harness.voice.recorder.shutil.which",
            lambda cmd: cmd == "sox",
        )
        recorder = VoiceRecorder()
        info = recorder.record_to(wav_path, max_seconds=5)
        assert info["path"] == wav_path
        assert isinstance(info["duration"], float)
        assert "silence_stopped" in info


class TestVoiceTranscriber:
    def test_transcriber_with_no_backend_returns_mock_in_test_mode(self, monkeypatch):
        """In test mode (MULLET_TEST_MODE=1) mock is used when no real backend exists."""
        monkeypatch.setenv("MULLET_TEST_MODE", "1")
        monkeypatch.setattr("mullet.harness.voice.transcriber.shutil.which", lambda _cmd: None)
        transcriber = VoiceTranscriber()
        assert transcriber.detect_backend() == "mock"
        text = transcriber.transcribe(Path("/dev/null/dummy.wav"))
        assert text == "[mock transcript]"

    def test_transcriber_openai_called_with_correct_args(self, monkeypatch, tmp_path):
        """OpenAI backend lazy-imports and calls audio.transcriptions.create."""
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"RIFF")

        fake_resp = MagicMock()
        fake_resp.text = "  hello openai  "
        fake_client = MagicMock()
        fake_client.audio.transcriptions.create.return_value = fake_resp

        fake_openai_mod = MagicMock()
        fake_openai_mod.OpenAI.return_value = fake_client
        monkeypatch.setitem(
            __import__("sys").modules,
            "openai",
            fake_openai_mod,
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("mullet.harness.voice.transcriber.shutil.which", lambda _cmd: None)

        transcriber = VoiceTranscriber(prefer="openai")
        text = transcriber.transcribe(wav_path)
        assert text == "hello openai"
        fake_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = fake_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "whisper-1"

    def test_transcriber_whisper_cpp_subprocess(self, monkeypatch, tmp_path):
        """whisper-cpp backend shells out to whisper-cli and reads the txt file."""
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"RIFF")
        txt_path = wav_path.with_suffix(".txt")
        txt_path.write_text("hello whisper")

        model_dir = tmp_path / ".local" / "share" / "whisper" / "models"
        model_dir.mkdir(parents=True)
        (model_dir / "ggml-base.bin").touch()

        fake_result = MagicMock()
        fake_result.stdout = ""

        def fake_run(cmd, **kwargs):
            return fake_result

        monkeypatch.setattr("mullet.harness.voice.transcriber.subprocess.run", fake_run)
        monkeypatch.setattr(
            "mullet.harness.voice.transcriber.Path.home",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "mullet.harness.voice.transcriber.shutil.which",
            lambda cmd: cmd == "whisper-cli",
        )

        transcriber = VoiceTranscriber(prefer="whisper-cpp")
        text = transcriber.transcribe(wav_path)
        assert text == "hello whisper"

    def test_transcriber_prefer_forces_backend(self, monkeypatch):
        """prefer='mock' bypasses environment checks."""
        transcriber = VoiceTranscriber(prefer="mock")
        assert transcriber.detect_backend() == "mock"


class TestVoiceCommand:
    def test_voice_command_record_returns_friendly_when_no_backend(self, monkeypatch):
        """/voice record without sox/ffmpeg/arecord gives a friendly message."""
        monkeypatch.setattr(
            "mullet.harness.voice.recorder.shutil.which",
            lambda _cmd: None,
        )
        engine = SimpleNamespace()
        out = handle_voice_command("record", engine=engine)
        assert "No recording backend found" in out

    def test_voice_command_transcribe_with_explicit_path(self, monkeypatch, tmp_path):
        """/voice transcribe <path> returns the transcript for an existing file."""
        wav_path = tmp_path / "existing.wav"
        wav_path.write_bytes(b"RIFF")

        monkeypatch.setenv("MULLET_TEST_MODE", "1")
        monkeypatch.setattr("mullet.harness.voice.transcriber.shutil.which", lambda _cmd: None)
        engine = SimpleNamespace()
        out = handle_voice_command(f"transcribe {wav_path}", engine=engine)
        assert out == "[mock transcript]"

    def test_voice_command_record_success(self, monkeypatch, tmp_path):
        """/voice record records, transcribes, and queues the prompt on the engine."""
        wav_path = tmp_path / "out.wav"
        wav_path.write_bytes(b"RIFF")

        fake_recorder = MagicMock()
        fake_recorder.detect_backend.return_value = "sox"
        fake_recorder.record_to.return_value = {"path": wav_path, "duration": 1.23, "silence_stopped": False}

        fake_transcriber = MagicMock()
        fake_transcriber.transcribe.return_value = "hello world"

        monkeypatch.setattr(
            "mullet.harness.commands.voice.VoiceRecorder",
            lambda *a, **k: fake_recorder,
        )
        monkeypatch.setattr(
            "mullet.harness.commands.voice.VoiceTranscriber",
            lambda *a, **k: fake_transcriber,
        )

        engine = SimpleNamespace()
        out = handle_voice_command("record 5", engine=engine)
        assert out == "[voice] hello world"
        assert getattr(engine, "_pending_voice_prompt", []) == ["hello world"]
        fake_recorder.record_to.assert_called_once()
        assert fake_recorder.record_to.call_args.kwargs["max_seconds"] == 5

    def test_voice_command_unknown_subcommand(self):
        """Unknown subcommand returns a helpful error."""
        engine = SimpleNamespace()
        out = handle_voice_command("dance", engine=engine)
        assert "Unknown /voice subcommand" in out

    def test_voice_command_requires_engine(self):
        """Missing engine returns an error."""
        out = handle_voice_command("record")
        assert "requires an active engine" in out
