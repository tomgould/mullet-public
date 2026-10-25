"""Runs the `preset-prompt` hook from this checkout: `mullet hook preset-prompt`.

Kept for this repo's own .claude/settings.json. The code is in
mullet/hooks/preset_prompt.py; installed copies register `mullet hook preset-prompt` instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mullet.hooks import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run("preset-prompt"))
