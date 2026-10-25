"""Runs the `opus-override` hook from this checkout: `mullet hook opus-override`.

Kept for this repo's own .claude/settings.json. The code is in
mullet/hooks/user_prompt_detect_override.py; installed copies register `mullet hook opus-override` instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mullet.hooks import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run("opus-override"))
