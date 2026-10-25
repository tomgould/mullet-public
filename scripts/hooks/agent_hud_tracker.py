"""Runs the `agent-hud-tracker` hook from this checkout: `mullet hook agent-hud-tracker`.

Kept for this repo's own .claude/settings.json. The code is in
mullet/hooks/agent_hud_tracker.py; installed copies register `mullet hook agent-hud-tracker` instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mullet.hooks import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run("agent-hud-tracker"))
