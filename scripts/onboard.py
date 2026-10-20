"""Mullet onboarding checklist, for running from a source checkout.

`python3 scripts/onboard.py` works before `pip install -e .`; once installed,
the same checklist is the `mullet-onboard` command. The code lives in
mullet/cli/onboard.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mullet.cli.onboard import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
