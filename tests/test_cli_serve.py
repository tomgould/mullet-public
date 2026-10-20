"""Tests for mullet/cli/serve.py — add_serve_parser + run_serve argument plumbing."""

import argparse
from unittest.mock import patch



def test_add_serve_parser_registers_subcommand():
    """add_serve_parser attaches a 'serve' subcommand with host + port args."""
    from mullet.cli.serve import add_serve_parser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    serve_p = add_serve_parser(sub)

    args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "8080"])
    assert args.cmd == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 8080


def test_add_serve_parser_defaults():
    """Defaults are localhost:3000."""
    from mullet.cli.serve import add_serve_parser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    add_serve_parser(sub)

    args = parser.parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 3000


def test_run_serve_invokes_server_with_args():
    """run_serve passes host + port through to run_server."""
    from mullet.cli import serve as serve_mod

    with patch("mullet.server.app.run_server") as mock_run_server:
        args = argparse.Namespace(host="192.168.1.5", port=9999)
        serve_mod.run_serve(args)
        mock_run_server.assert_called_once_with(host="192.168.1.5", port=9999)


def test_run_serve_prints_endpoints(capsys):
    """run_serve announces the listening endpoints on stdout."""
    from mullet.cli import serve as serve_mod

    with patch("mullet.server.app.run_server"):
        args = argparse.Namespace(host="127.0.0.1", port=3000)
        serve_mod.run_serve(args)

    captured = capsys.readouterr()
    assert "127.0.0.1:3000" in captured.out
    assert "/v1/chat/completions" in captured.out
    assert "/ws" in captured.out
    assert "/health" in captured.out


def test_run_serve_initialises_logging():
    """run_serve must configure logging before starting the server."""
    from mullet.cli import serve as serve_mod

    with patch("mullet.server.app.run_server"), \
         patch("logging.basicConfig") as mock_basicconfig:
        args = argparse.Namespace(host="127.0.0.1", port=3000)
        serve_mod.run_serve(args)
        mock_basicconfig.assert_called_once()
