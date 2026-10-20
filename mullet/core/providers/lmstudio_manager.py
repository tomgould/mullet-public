"""LM Studio CLI wrapper: load/unload/list models, query server health."""
import shutil
import subprocess
import urllib.error
import urllib.request
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_LMS_SEARCH_PATHS = [
    str(Path.home() / ".cache/lm-studio/bin/lms"),  # common Linux install
    "/usr/local/bin/lms",
    "/usr/bin/lms",
]


def _find_lms_cli() -> str:
    """Locate the lms CLI. Searches PATH then common install locations.

    Returns the resolved path when found, or the bare string 'lms' as a
    last-resort so subprocess calls fail with a clear 'command not found'
    rather than a NoneType-join crash.
    """
    found = shutil.which("lms")
    if found:
        return found
    for p in _LMS_SEARCH_PATHS:
        import os
        if os.path.isfile(p):
            return p
    return "lms"


class LMStudioManager:
    """Inspect and manage the local LM Studio server (model listing, load/unload)."""

    LMS_CLI = _find_lms_cli()

    def __init__(self, port: int = 1234):
        """Bind to the LM Studio HTTP API on *port* (default 1234)."""
        self.port = port
        self.base_url = f"http://localhost:{port}"

    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """Helper to make HTTP GET requests and parse JSON."""
        url = f"{self.base_url}{endpoint}"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return json.loads(response.read().decode('utf-8'))
                else:
                    logger.debug(f"HTTP error {response.status} for {url}")
                    return None
        except urllib.error.URLError as e:
            logger.debug(f"Network error accessing {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error from {url}: {e}")
            return None
        except Exception as e: # Catch any other unexpected errors during request
            logger.debug(f"An unexpected error occurred accessing {url}: {e}")
            return None

    def get_models(self) -> List[Dict]:
        """GET /api/v0/models — returns full list with state info. Returns [] on error."""
        data = self._make_request("/api/v0/models")
        return data.get("data", []) if data else []
    
    def get_loaded_models(self) -> List[Dict]:
        """Filter get_models() to type==llm and state==loaded."""
        all_models = self.get_models()
        return [
            model for model in all_models
            if model.get("type") == "llm" and model.get("state") == "loaded"
        ]
    
    def pick_best_model(self, context_needed: int = 4096, complexity: str = "STANDARD") -> Optional[Dict]:
        """Select the best available local model for the given context and complexity.

        Selection strategy:
        1. Among models whose context window fits context_needed:
           a. Prefer loaded (already hot — no load delay)
           b. Among loaded, prefer smallest context that still fits (avoids VRAM waste)
        2. If nothing fits context_needed, fall back to any loaded LLM (with a warning)
        3. Else any non-embedding LLM, then any LLM
        """
        from mullet.core.providers.model_info import get_model_info

        _SKIP = ("embedding", "clip", "rerank", "vision-encoder")

        all_models = self.get_models()
        candidates = []

        for model in all_models:
            raw_id = model.get("id", "")
            lower_id = raw_id.lower()
            if model.get("type") != "llm":
                continue
            if any(kw in lower_id for kw in _SKIP):
                continue
            info = get_model_info(raw_id)
            is_loaded = model.get("state") == "loaded"
            # Prefer actual loaded_context_length from LM Studio API — the static
            # registry is a best-effort estimate and often over-states the real window.
            actual_ctx = model.get("loaded_context_length", 0)
            effective_ctx = actual_ctx if actual_ctx > 0 else info.context_length
            candidates.append((model, info, is_loaded, effective_ctx))

        if not candidates:
            return None

        # Split into those that fit context_needed vs those that don't
        fitting = [(m, info, loaded, ctx) for m, info, loaded, ctx in candidates if ctx >= context_needed]
        overflow = [(m, info, loaded, ctx) for m, info, loaded, ctx in candidates if ctx < context_needed]

        if fitting:
            # Sort: loaded first, then by smallest fitting context (lean resource use)
            fitting.sort(key=lambda x: (not x[2], x[3]))
            return fitting[0][0]

        # Nothing fits — warn and fall back to largest available
        if overflow:
            overflow.sort(key=lambda x: (not x[2], -x[3]))
            logger.warning(
                "pick_best_model: no local model has context >= %d tokens; "
                "using largest available (%s, %d ctx)",
                context_needed,
                overflow[0][0].get("id"),
                overflow[0][3],
            )
            return overflow[0][0]
    
    def _run_lms_command(self, args: List[str]) -> Tuple[bool, str]:
        """Helper to run lms CLI commands."""
        command = [self.LMS_CLI] + args
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                logger.debug(f"LMS command {' '.join(command)} successful: {result.stdout.strip()}")
                return True, result.stdout.strip()
            else:
                logger.error(f"LMS command {' '.join(command)} failed with code {result.returncode}: {result.stderr.strip()}")
                return False, result.stderr.strip()
        except FileNotFoundError:
            logger.error(f"LMS CLI not found at {self.LMS_CLI}. Please ensure LM Studio is installed and the CLI is accessible.")
            return False, f"LMS CLI not found at {self.LMS_CLI}"
        except Exception as e:
            logger.error(f"Error running LMS command {' '.join(command)}: {e}")
            return False, str(e)

    def ensure_loaded(self, model_id: str, ttl: int = 60) -> bool:
        """Check if model_id is already loaded. If not, run:
           `lms load --ttl {ttl} --yes {model_id}`
        Wait up to 30s for load to complete (poll /api/v0/models every 2s).
        Return True if loaded, False on timeout/error.
        Log progress to logger (DEBUG level).
        """
        start_time = time.time()
        timeout = 30
        poll_interval = 2

        # Check if already loaded
        loaded_models = self.get_loaded_models()
        if any(m.get("id") == model_id for m in loaded_models):
            logger.debug(f"Model {model_id} is already loaded.")
            return True

        # If not loaded, attempt to load
        logger.debug(f"Attempting to load model {model_id} with TTL {ttl}s.")
        success, output = self._run_lms_command(["load", "--ttl", str(ttl), "--yes", model_id])
        if not success:
            logger.error(f"Failed to initiate load for model {model_id}: {output}")
            return False

        # Poll until loaded or timeout
        while time.time() - start_time < timeout:
            loaded_models = self.get_loaded_models()
            if any(m.get("id") == model_id for m in loaded_models):
                logger.debug(f"Model {model_id} successfully loaded after {int(time.time() - start_time)}s.")
                return True
            logger.debug(f"Waiting for model {model_id} to load...")
            time.sleep(poll_interval)
        
        logger.error(f"Timeout waiting for model {model_id} to load after {timeout}s.")
        return False
    
    def unload(self, model_id: str) -> bool:
        """Run `lms unload {model_id}`. Return True on success."""
        logger.debug(f"Attempting to unload model {model_id}.")
        success, output = self._run_lms_command(["unload", model_id])
        if success:
            logger.debug(f"Model {model_id} successfully unloaded: {output}")
        else:
            logger.error(f"Failed to unload model {model_id}: {output}")
        return success
    
    def reload_current_model(self, ttl: int = 60) -> bool:
        """Unload and reload the currently active model to clear KV cache.

        Called after a context-size-exceeded error so the model starts fresh.
        Returns True if reload succeeded.
        """
        loaded = self.get_loaded_models()
        if not loaded:
            logger.warning("reload_current_model: no model loaded, nothing to reload")
            return False
        model_id = loaded[0].get("id") or loaded[0].get("modelKey", "")
        if not model_id:
            return False
        logger.info("Reloading model %s to clear KV cache after context overflow", model_id)
        self.unload(model_id)
        return self.ensure_loaded(model_id, ttl=ttl)

    def get_or_load_provider(self, ttl: int = 60) -> Tuple[Optional[str], bool]:
        """High-level: pick best model, ensure it's loaded (with TTL), return (model_id, was_already_loaded).
        Returns (None, False) if no models available or load fails.
        """
        # First, check if any model is already loaded and is the 'best'
        loaded_models = self.get_loaded_models()
        if loaded_models:
            best_loaded_model = self.pick_best_model() # This will prioritize loaded models
            if best_loaded_model and best_loaded_model.get("state") == "loaded":
                logger.info(f"Using already loaded best model: {best_loaded_model.get('id')}")
                return best_loaded_model.get("id"), True

        # If no suitable model is already loaded, pick the best available and try to load it
        model_to_load = self.pick_best_model()
        if not model_to_load:
            logger.debug("No suitable models found in LM Studio.")
            return None, False
        
        model_id = model_to_load.get("id")
        logger.info(f"Picked model {model_id} for loading.")
        
        if self.ensure_loaded(model_id, ttl):
            return model_id, False # It was not already loaded, so we loaded it
        else:
            logger.error(f"Failed to ensure model {model_id} is loaded.")
            return None, False
