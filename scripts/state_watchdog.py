#!/usr/bin/env python3
"""Clear stale Mullet state files written before the last system boot."""
import time
from pathlib import Path

STATE_DIR = Path(__file__).parent.parent / ".mullet" / "state"
INTERVAL = 60


def boot_time() -> float:
    return time.time() - float(Path("/proc/uptime").read_text().split()[0])


def clear_stale():
    boot = boot_time()
    for f in STATE_DIR.glob("*.json"):
        try:
            if f.stat().st_mtime < boot:
                f.write_text("{}")
        except OSError:
            pass


if __name__ == "__main__":
    while True:
        clear_stale()
        time.sleep(INTERVAL)
