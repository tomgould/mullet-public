"""Tests for the `mullet shell` harness CLI subcommand."""

from __future__ import annotations

import argparse
import json


from mullet.cli.harness_cli import cmd_shell, build_shell_subparser, build_shell_subparser_simple


class TestBuildShellSubparser:
    def test_build_shell_subparser(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        shell_parser = build_shell_subparser(subparsers)
        assert shell_parser is not None
        args = shell_parser.parse_args(["hello", "--provider", "kimi"])
        assert args.prompt == "hello"
        assert args.provider == "kimi"

    def test_build_shell_subparser_simple(self):
        parser = argparse.ArgumentParser()
        build_shell_subparser_simple(parser)
        args = parser.parse_args(["hello", "--output-format", "json"])
        assert args.prompt == "hello"
        assert args.output_format == "json"


class TestListSessions:
    def test_list_sessions_empty(self, isolated_profiles_dir, tmp_session_store, monkeypatch, capsys):
        class MockEngine:
            def list_sessions(self, project_hash=None, limit=20):
                return []

        monkeypatch.setattr("mullet.cli.harness_cli.HarnessEngine", MockEngine)
        args = argparse.Namespace(
            prompt=None,
            list_sessions=True,
            session_id=None,
            provider="kimi",
            model=None,
            output_format="text",
        )
        rc = cmd_shell(args)
        captured = capsys.readouterr()
        assert rc == 0
        assert "No sessions yet" in captured.out


class TestOneShotPrompt:
    def test_one_shot_prompt_text_output(self, monkeypatch, capsys):
        class MockEngine:
            def run(self, prompt, session_id=None, provider="kimi", model=None, images=None):
                return {
                    "session_id": "sess-test",
                    "response": "hello world",
                    "stop_reason": "end_turn",
                    "input_tokens": 10,
                    "output_tokens": 5,
                }

        monkeypatch.setattr("mullet.cli.harness_cli.HarnessEngine", MockEngine)
        args = argparse.Namespace(
            prompt="hi",
            list_sessions=False,
            session_id=None,
            provider="kimi",
            model=None,
            output_format="text",
        )
        rc = cmd_shell(args)
        captured = capsys.readouterr()
        assert rc == 0
        assert "hello world" in captured.out

    def test_one_shot_prompt_json_output(self, monkeypatch, capsys):
        class MockEngine:
            def run(self, prompt, session_id=None, provider="kimi", model=None, images=None):
                return {
                    "session_id": "sess-test",
                    "response": "hello world",
                    "stop_reason": "end_turn",
                    "input_tokens": 10,
                    "output_tokens": 5,
                }

        monkeypatch.setattr("mullet.cli.harness_cli.HarnessEngine", MockEngine)
        args = argparse.Namespace(
            prompt="hi",
            list_sessions=False,
            session_id=None,
            provider="kimi",
            model=None,
            output_format="json",
        )
        rc = cmd_shell(args)
        captured = capsys.readouterr()
        assert rc == 0
        data = json.loads(captured.out)
        assert data["response"] == "hello world"


class TestErrorHandling:
    def test_unknown_provider_returns_1(self, monkeypatch, capsys):
        class MockEngine:
            def run(self, prompt, session_id=None, provider="kimi", model=None, images=None):
                raise NotImplementedError("unknown provider")

        monkeypatch.setattr("mullet.cli.harness_cli.HarnessEngine", MockEngine)
        args = argparse.Namespace(
            prompt="hi",
            list_sessions=False,
            session_id=None,
            provider="bogus",
            model=None,
            output_format="text",
        )
        rc = cmd_shell(args)
        captured = capsys.readouterr()
        assert rc == 1
        assert "unknown provider" in captured.err
