"""Manages system resources and provides recommendations."""

import psutil
import torch

class ResourceManager:
    """Manages system resources and provides recommendations."""
    def probe(self):
        """Probe system CPU, RAM, and GPU resources."""
        cpu_count = psutil.cpu_count(logical=False)
        ram_gb = psutil.virtual_memory().total / (1024**3)
        gpu_available = False
        gpu_vram_gb = 0

        if torch.cuda.is_available():
            gpu_available = True
            # Assuming a single GPU for simplicity
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "cpu_count": cpu_count,
            "ram_gb": ram_gb,
            "gpu_available": gpu_available,
            "gpu_vram_gb": gpu_vram_gb,
        }

    def recommend_batch_size(self, gpu_available: bool) -> int:
        """Recommend a batch size based on GPU availability."""
        # This is a placeholder for a more sophisticated recommendation logic
        if gpu_available:
            return 32  # Larger batch size for GPU
        else:
            return 8   # Smaller batch size for CPU
