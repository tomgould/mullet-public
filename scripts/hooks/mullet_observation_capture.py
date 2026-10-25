"""Runs the `observation-capture` hook from this checkout: `mullet hook observation-capture`.

Kept for this repo's own .claude/settings.json. The code is in
mullet/hooks/observation_capture.py; installed copies register `mullet hook observation-capture` instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mullet.hooks import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run("observation-capture"))
