"""`mullet hook <name>`: run one of Mullet's Claude Code hooks.

Claude Code runs this with the hook's JSON payload on stdin. The hooks live in
`mullet/hooks/`; `mullet setup-claude --hooks <names>` registers them.
"""

from __future__ import annotations

import sys

USAGE = """\
Usage: mullet hook <name>     run a hook (Claude Code sends its JSON on stdin)
       mullet hook list       show the hooks

Register hooks with `mullet setup-claude --hooks <name>,<name>`. None are
registered by default.

Hooks:"""


def cmd_hook(argv: list[str]) -> int:
    """Run `mullet hook` with the arguments after `hook`; returns the exit code."""
    from mullet.hooks import describe, run

    if not argv:
        print(USAGE, file=sys.stderr)
        print(describe(), file=sys.stderr)
        return 2
    if argv[0] in ("-h", "--help", "help", "list"):
        print(USAGE if argv[0] != "list" else "Hooks (register with mullet setup-claude --hooks <names>):")
        print(describe())
        return 0
    if len(argv) > 1:
        print(f"mullet hook: one hook name expected, got {' '.join(argv)!r}", file=sys.stderr)
        return 2
    return run(argv[0])
