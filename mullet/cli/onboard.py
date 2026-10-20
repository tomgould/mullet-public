"""Mullet onboarding checklist: detects the system and prints the set-up steps.

Run it as `mullet-onboard` once installed, or as `python3 scripts/onboard.py`
from a source checkout before `pip install -e .` (it needs only the standard
library).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request


def _yes_no(prompt: str, default: bool) -> bool:
    """Ask a yes/no question. default=True means [Y/n], False means [y/N]."""
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix} ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _detect_os() -> str:
    plat = sys.platform
    if plat == "darwin":
        return "macOS"
    if plat == "linux":
        return "Linux"
    if plat == "win32":
        return "Windows"
    return plat


def _detect_arch() -> str:
    return platform.machine()


def _detect_ram_gb() -> float | None:
    """Return total RAM in GB, or None if detection fails."""
    plat = sys.platform
    try:
        if plat == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            bytes_total = int(out.stdout.strip())
            return bytes_total / (1024 ** 3)
        if plat == "linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        kb = int(parts[1])
                        return kb / (1024 ** 2)
        if plat == "win32":
            out = subprocess.run(
                ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            for line in out.stdout.strip().splitlines():
                line = line.strip()
                if line and not line.lower().startswith("total"):
                    bytes_total = int(line)
                    return bytes_total / (1024 ** 3)
    except Exception:
        pass
    return None


def _detect_lm_studio() -> tuple[bool, str | None]:
    """Probe LM Studio at localhost:1234. Returns (is_running, model_name_or_none)."""
    try:
        req = urllib.request.Request(
            "http://localhost:1234/v1/models",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                return True, models[0].get("id")
            return True, None
    except Exception:
        return False, None


def _header(text: str) -> None:
    print()
    print(text)
    print("=" * len(text))


USAGE = """Usage: mullet-onboard

Detects your OS, RAM, Python and LM Studio, asks a few yes/no questions about
the providers you have, and prints a personalised set-up checklist. It changes
nothing on your system.
"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help", "help") for a in args):
        print(USAGE, end="")
        return 0

    os_name = _detect_os()
    arch = _detect_arch()
    py_major, py_minor = sys.version_info[:2]
    ram_gb = _detect_ram_gb()
    lm_running, lm_model = _detect_lm_studio()

    _header("Detected system")
    print(f"OS:           {os_name}")
    print(f"Architecture: {arch}")
    print(f"Python:       {py_major}.{py_minor}")
    if ram_gb is not None:
        print(f"RAM:          {ram_gb:.1f} GB")
    else:
        print("RAM:          (could not detect)")
    if lm_running:
        if lm_model:
            print(f"LM Studio:    running ({lm_model})")
        else:
            print("LM Studio:    running (no model loaded)")
    else:
        print("LM Studio:    not detected")

    if py_major < 3 or (py_major == 3 and py_minor < 10):
        print()
        print("WARNING: Python 3.10+ is required. Please upgrade before continuing.")

    print()
    _header("Quick questions")

    has_claude = _yes_no("Do you have a Claude Max or Pro subscription?", default=True)
    has_kimi = _yes_no("Do you have a Kimi API key? (https://platform.kimi.ai)", default=False)
    has_gemini = _yes_no("Do you have a Google Gemini API key?", default=False)
    has_lm_studio = True if lm_running else _yes_no("Is LM Studio installed?", default=False)

    print()
    _header("Your personalised setup checklist")

    step = 1

    # 1. pip install
    print(f"{step}. pip install -e .")
    step += 1

    # 2. .env
    env_keys: list[str] = []
    if has_kimi:
        env_keys.append("KIMI_API_KEY=sk-...")
    if has_gemini:
        env_keys.append("GOOGLE_API_KEY=...")

    print(f"{step}. Create .env from .env.example in the repo root:")
    if env_keys:
        for key in env_keys:
            print(f"      {key}")
    else:
        print("      (no API keys to add yet)")
    step += 1

    # 3. Claude setup
    print(f"{step}. Run `mullet setup-claude`, then restart Claude Code")
    step += 1

    # 4. doctor
    print(f"{step}. Run `mullet doctor` — green = ready, yellow = works but not ideal, red = broken")
    step += 1

    # 5. first test
    print(f"{step}. First test: `mullet \"what Python version am I running?\"`")
    step += 1

    # 6. verify delegation
    print(f"{step}. Verify delegation: `mullet explain \"refactor the auth module\"`")
    print("      — should show Kimi or local LLM, NOT Anthropic")
    step += 1

    # RAM-specific guidance
    if ram_gb is not None and ram_gb <= 10:
        print()
        print("--- Low-RAM guidance ---")
        print(f"{step}. Do NOT load 7B+ models in LM Studio.")
        step += 1
        print(f"{step}. Recommended small models:")
        print("      - qwen3.5-2b-kimi-and-opus-distillation (1.6GB) — reasoning/analysis tasks")
        print("      - qwen2.5-coder-3b-instruct (1.9GB) — code tasks")
        print("      Mullet's router picks between them automatically based on task type.")
        step += 1
        print(f"{step}. Do NOT run `pip install -e \".[gpu]\"` — pulls pynvml (NVIDIA-only, useless on Mac).")
        step += 1
    elif ram_gb is not None and ram_gb > 10:
        print()
        print("--- High-RAM guidance ---")
        print(f"{step}. With >10 GB RAM you can run larger models for better quality:")
        print("      - qwen3.5-9b")
        print("      - qwen2.5-coder-7b")
        step += 1

    # Apple Silicon notes
    if os_name == "macOS" and arch == "arm64":
        print()
        print("--- Apple Silicon notes ---")
        print(f"{step}. sentence-transformers uses MPS (Metal) automatically — no CUDA needed.")
        step += 1
        print(f"{step}. Download the LM Studio Apple Silicon build (not the Intel one).")
        step += 1
        print(f"{step}. In LM Studio, download MLX format models — NOT GGUF.")
        print("      Search 'mlx-community' in the Discover tab. MLX is native to Apple Silicon.")
        step += 1

    # Provider warning — Anthropic-only setups still work but cap-bound
    if not has_kimi and not has_gemini and not has_lm_studio and not lm_running:
        print()
        print("--- Provider warning ---")
        if has_claude:
            print(f"{step}. You only have Anthropic configured — every task will count")
            print("      against your Claude Max 5-hour quota. Mullet's whole point is to")
            print("      offload cheap work to other providers.")
        else:
            print(f"{step}. You have no provider configured at all.")
            print("      Mullet needs at least one to be useful.")
        print("      Get a Kimi key at https://platform.kimi.ai — £20 gets you started.")
        step += 1

    # Critical rules
    print()
    print("--- Critical rules ---")
    print(f"{step}. ANTHROPIC_API_KEY is optional and billed per token. Without it, Anthropic work runs")
    print("      through Claude Code or the official claude CLI, under your subscription.")
    step += 1
    print(f"{step}. Keys go in .env in the repo root, not in ~/.zshrc or ~/.bashrc.")
    step += 1
    print(f"{step}. The [gpu] extra is for Linux+NVIDIA only — do not install it on Mac.")
    step += 1

    print()
    print("Done. Work through the steps above, then you're ready to go.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
