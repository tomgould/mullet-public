"""Runs the `pre-agent-guard` hook from this checkout: `mullet hook pre-agent-guard`.

Kept for this repo's own .claude/settings.json. The code is in
mullet/hooks/pre_agent_guard.py; installed copies register `mullet hook pre-agent-guard` instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mullet.hooks import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run("pre-agent-guard"))
