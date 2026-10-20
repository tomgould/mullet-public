"""
Model cache — singleton that keeps embedding models and rerankers in memory.

Avoids redundant GPU loads when multiple clients use the same model.
Thread-safe: a per-key load lock prevents duplicate loads under parallel
reindex workers; clear() acquires the write lock so it cannot race with
an in-progress get.

Loading is thread-safe, but a loaded SentenceTransformer/CrossEncoder is
NOT safe to run concurrent .encode()/.predict() calls on from multiple
threads — concurrent CUDA forward passes on one model instance can hang
indefinitely rather than raise. `inference_lock()` serializes actual
inference calls (not just loads) across every caller of this cache;
callers doing embedding/reranking work must hold it for the duration of
each encode/predict call.
"""

import gc
import logging
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_instance: Optional["ModelCache"] = None
_instance_lock = threading.Lock()


def get_model_cache() -> "ModelCache":
    """Get the singleton ModelCache instance (thread-safe)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ModelCache()
    return _instance


class ModelCache:
    """Cache for embedding models and cross-encoder rerankers.

    Thread-safe for concurrent get_embedder / get_reranker calls.
    Multiple parallel reindex workers sharing the same model will block
    on the first load and then share the single loaded instance.
    """

    def __init__(self):
        self._embedders: dict[str, object] = {}
        self._rerankers: dict[str, object] = {}
        # One lock per cache key prevents concurrent loads of the same model;
        # a global RW-style lock guards clear() vs get_* races.
        self._lock = threading.Lock()
        # Separate lock serializing actual inference calls (see module docstring).
        self._inference_lock = threading.Lock()

    def inference_lock(self) -> threading.Lock:
        """Lock to hold for the duration of any encode()/predict() call.

        Shared across all cached models — this deployment has one GPU, so
        serializing inference is correct, not just a safety fallback.
        """
        return self._inference_lock

    def get_embedder(self, model_name: str, device: str = "cuda"):
        """Get or load a SentenceTransformer model (thread-safe)."""
        key = f"{model_name}:{device}"
        with self._lock:
            if key not in self._embedders:
                from sentence_transformers import SentenceTransformer
                logger.info(f"ModelCache: loading embedder {model_name} on {device}")
                self._embedders[key] = SentenceTransformer(model_name, device=device)
            return self._embedders[key]

    def get_reranker(self, model_name: str, device: str = "cuda"):
        """Get or load a CrossEncoder reranker (thread-safe)."""
        key = f"{model_name}:{device}"
        with self._lock:
            if key not in self._rerankers:
                from sentence_transformers import CrossEncoder
                logger.info(f"ModelCache: loading reranker {model_name} on {device}")
                self._rerankers[key] = CrossEncoder(model_name, device=device)
            return self._rerankers[key]

    def clear(self):
        """Release all cached models and free GPU memory (thread-safe)."""
        with self._lock:
            count = len(self._embedders) + len(self._rerankers)
            self._embedders.clear()
            self._rerankers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if count:
            logger.info(f"ModelCache: cleared {count} cached models")
