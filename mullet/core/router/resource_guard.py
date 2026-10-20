"""Process-level resource guard: clamp concurrent jobs based on RAM/GPU/CPU."""
import gc
import logging
import os
import threading
import time

import psutil

from mullet.knowledge.engine.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class ResourceGuard:
    """
    Manages resource availability for dispatching tasks locally.
    Ensures the system has sufficient CPU, RAM, and GPU resources before allowing new tasks.
    """

    def __init__(self):
        """
        Initializes the ResourceGuard with a ResourceManager and a threading.Semaphore.
        The semaphore is used to limit the number of concurrently dispatched local tasks.
        """
        self.resource_manager = ResourceManager(headroom=0.05)
        self.semaphore = threading.BoundedSemaphore(1)

    def can_dispatch_local(self) -> tuple[bool, str]:
        """
        Checks if the system has enough resources to dispatch a local task.

        Returns:
            tuple[bool, str]: A tuple where the first element is True if resources are
                             available, False otherwise. The second element is a reason
                             string if resources are not available, or 'ok' if they are.
        """
        # Check load average
        cpu_count = psutil.cpu_count()
        load_avg_1, _, _ = os.getloadavg()
        if load_avg_1 > 2 * cpu_count:
            reason = f"High load average: {load_avg_1:.2f} (>{2 * cpu_count})"
            logger.warning(reason)
            return False, reason

        # Check RAM
        virtual_memory = psutil.virtual_memory()
        available_ram_gb = virtual_memory.available / (1024 ** 3)
        if available_ram_gb < 2:
            reason = f"Insufficient RAM: {available_ram_gb:.2f}GB (<2GB available)"
            logger.warning(reason)
            return False, reason

        # Check GPU (if available)
        gpu = self.resource_manager.probe_gpu()
        if gpu is not None:
            if gpu["free_mb"] < 1024:
                reason = f"GPU VRAM low: {gpu['free_mb']}MB free (need 1024MB)"
                logger.warning(reason)
                return False, reason
            if gpu["utilization_pct"] > 95:
                reason = f"GPU saturated: {gpu['utilization_pct']}% utilized"
                logger.warning(reason)
                return False, reason
        
        # Check if TRAG/embedding processes are running (GPU/RAM conflict)
        try:
            for proc in psutil.process_iter(['name', 'cmdline']):
                cmdline = " ".join(proc.info.get('cmdline') or [])
                if 'trove' in cmdline and ('reindex' in cmdline or 'ingest' in cmdline or 'embed' in cmdline):
                    reason = "TRAG reindex running — skip local to avoid GPU conflict"
                    logger.warning(reason)
                    return False, reason
        except (psutil.NoSuchProcess, psutil.AccessDenied) as _exc:
            logger.warning("resource_guard: failed to probe process: %s", _exc)

        # Check semaphore availability
        if not self.semaphore.acquire(blocking=False):
            reason = "Local dispatch slot not available (semaphore acquired)"
            logger.warning(reason)
            return False, reason
        else:
            self.semaphore.release() # Release immediately as this is just a check

        return True, 'ok'

    def acquire_local_slot(self) -> bool:
        """
        Attempts to acquire a local dispatch slot non-blockingly.

        Returns:
            bool: True if the slot was acquired, False otherwise.
        """
        acquired = self.semaphore.acquire(blocking=False)
        if acquired:
            logger.info("Local dispatch slot acquired.")
        else:
            logger.warning("Failed to acquire local dispatch slot.")
        return acquired

    def release_local_slot(self):
        """
        Releases a previously acquired local dispatch slot.
        """
        self.semaphore.release()
        logger.info("Local dispatch slot released.")

    def flush_resources(self) -> dict:
        """Reclaim resources Mullet owns — no sudo needed.

        1. Kill orphaned child processes spawned by Mullet
        2. Force Python garbage collection (releases held memory)
        3. Clear GPU VRAM cache (torch)
        4. Close idle HTTP connections
        5. Invalidate stale caches
        """
        before = self.get_system_snapshot()

        # 1. Kill orphaned Mullet child processes (our own PIDs only)
        our_pid = os.getpid()
        try:
            parent = psutil.Process(our_pid)
            for child in parent.children(recursive=True):
                try:
                    # Only kill if it's been running > 60s (likely stuck)
                    if child.create_time() < (time.time() - 60):
                        child.terminate()
                        logger.info("Terminated orphaned child PID %d: %s", child.pid, child.name())
                except (psutil.NoSuchProcess, psutil.AccessDenied) as _exc:
                    logger.warning("resource_guard: failed to access child process: %s", _exc)
        except psutil.NoSuchProcess as _exc:
            logger.warning("resource_guard: parent process not found: %s", _exc)

        # 2. Aggressive Python GC — collect all generations
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

        # 3. Clear GPU VRAM cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU VRAM cache cleared")
        except (ImportError, Exception) as _exc:
            logger.warning("resource_guard: failed to clear GPU VRAM cache: %s", _exc)

        # 4. GC pass to release idle HTTP connections (OpenAI SDK clients hold pools)
        gc.collect()

        # 5. Invalidate cached GPU probe so next check gets fresh data
        self.resource_manager._gpu_probed = False

        after = self.get_system_snapshot()
        freed_ram = after["ram_available_gb"] - before["ram_available_gb"]
        logger.info("Resource flush: freed %.1fGB RAM", freed_ram)
        return {"before": before, "after": after, "freed_ram_gb": round(freed_ram, 1)}

    def get_system_snapshot(self) -> dict:
        """
        Retrieves a snapshot of current system resource statistics.

        Returns:
            dict: A dictionary containing CPU, RAM, GPU, and load average statistics.
        """
        ram = self.resource_manager.probe_ram()
        cpu = self.resource_manager.probe_cpu()
        snapshot = {
            "load_avg": round(psutil.getloadavg()[0], 2),
            "cpu_cores": cpu["total_cores"],
            "ram_available_gb": round(ram["available_gb"], 1),
            "ram_total_gb": ram["total_gb"],
            "gpu": self.resource_manager.probe_gpu(),
        }
        return snapshot

_guard: ResourceGuard | None = None

def get_resource_guard() -> ResourceGuard:
    """
    Returns the singleton instance of ResourceGuard.

    Returns:
        ResourceGuard: The singleton instance.
    """
    global _guard
    if _guard is None:
        _guard = ResourceGuard()
    return _guard