"""CLI dispatcher for ``mullet sandbox``: build the image the container backends run commands in.

Wired into ``mullet/cli/main.py`` through the same ``sys.argv[1]`` early
branch as ``mullet job``. The backends live in ``mullet.jobs.sandbox_backends``.
"""

from __future__ import annotations

import argparse
import functools
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from mullet import __version__
from mullet.jobs.sandbox_backends import container
from mullet.jobs.sandbox_backends.settings import load_settings

DOCKERFILE = Path(container.__file__).with_name("Dockerfile")


def build_parser() -> argparse.ArgumentParser:
    """The ``mullet sandbox`` argument parser."""
    parser = argparse.ArgumentParser(prog="mullet sandbox",
                                     description="The image Mullet's docker and podman sandbox backends use.")
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build-image", help="build the sandbox image from Mullet's Dockerfile")
    build.add_argument("--tag", help="image name to build (default: the sandbox.image setting)")
    build.add_argument("--engine", choices=container.ENGINES,
                       help="docker or podman (default: podman when sandbox.backend is podman, else docker)")
    build.add_argument("--print", action="store_true", dest="print_only",
                       help="print the build command instead of running it")
    return parser


def build_command(engine: str, tag: str, context: Path) -> list[str]:
    """The engine's build command for Mullet's Dockerfile, with an empty build context.

    MULLET_VERSION becomes the image's ``mullet.sandbox.version`` label, which
    the backends' check() requires.
    """
    return [shutil.which(engine) or engine, "build", "--tag", tag, "--build-arg", f"MULLET_VERSION={__version__}",
            "--file", str(DOCKERFILE), str(context)]


def cmd_sandbox(argv: list[str], *, out: Callable[[str], None] = print,
                err: Callable[[str], None] = functools.partial(print, file=sys.stderr)) -> int:
    """Entry point used by ``mullet sandbox ...``."""
    args = build_parser().parse_args(argv)
    settings = load_settings()
    for problem in settings.problems:
        err(f"mullet sandbox: config: {problem}")
    engine = args.engine or ("podman" if settings.backend == "podman" else "docker")
    tag = args.tag or settings.image
    if args.print_only:
        out(" ".join(build_command(engine, tag, Path("<empty directory>"))))
        return 0
    if not shutil.which(engine):
        err(f"mullet sandbox: {engine} is not installed")
        return 1
    with tempfile.TemporaryDirectory(prefix="mullet-build-") as context:
        command = build_command(engine, tag, Path(context))
        out(f"Building {tag} with {engine}")
        sys.stdout.flush()  # before the engine's own output
        code = subprocess.run(command, stdin=subprocess.DEVNULL).returncode
    if code != 0:
        err(f"mullet sandbox: the build failed (exit {code})")
        return 1
    if tag != settings.image:
        out(f"Built {tag}. Jobs use {settings.image}; set sandbox.image: {tag} to use this one.")
    else:
        out(f"Built {tag}. Set sandbox.backend: {engine} (or auto) to run jobs in it.")
    return 0
