"""Smoke-import every module under mullet/.

Catches latent ImportErrors that the wider test suite misses because most
tests import only a handful of well-trodden modules. This test would have
caught the three ImportError regressions found during the 2026-05-18
overnight session:

  - mullet.cli.hud_claude       (orphan, referenced nonexistent core.budget)
  - mullet.knowledge.engine.runner  (SearchQuality not re-exported)
  - mullet.cli.status_cli       (load_permissions not defined anywhere)

If your module legitimately can't be imported at module-import time (e.g.
heavy optional dep), add it to EXPECTED_SKIPS with a one-line reason.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import mullet


# Modules that intentionally can't be imported without a runtime resource
# (heavy optional dep, GPU-only, etc.). Empty by default — be sparing.
EXPECTED_SKIPS: dict[str, str] = {
    # "mullet.foo.gpu_only": "requires CUDA",
}


def _discover_modules() -> list[str]:
    out: list[str] = []
    for _finder, modname, _ispkg in pkgutil.walk_packages(
        mullet.__path__, prefix="mullet."
    ):
        if "__pycache__" in modname:
            continue
        out.append(modname)
    return sorted(out)


@pytest.mark.parametrize("modname", _discover_modules())
def test_module_imports(modname: str) -> None:
    if modname in EXPECTED_SKIPS:
        pytest.skip(f"intentional skip: {EXPECTED_SKIPS[modname]}")
    try:
        importlib.import_module(modname)
    except ImportError as e:
        pytest.fail(
            f"{modname} failed to import: {e!s}\n"
            f"Either fix the import or, if intentional, add to EXPECTED_SKIPS."
        )
