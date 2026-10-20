"""Monitors health status and provides health tracking utilities."""

from enum import Enum
from typing import Dict, Any, Callable

class HealthStatus(Enum):
    """Enumeration of possible health states."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"

class HealthChecker:
    """Runs configured health checks and returns a status summary."""
    def __init__(self,
                 storage_check: Callable = None,
                 index_check: Callable = None,
                 embedding_model_check: Callable = None):
        self._checks = {
            "storage_accessible": storage_check,
            "index_exists": index_check,
            "embedding_model_loadable": embedding_model_check
        }

    def check(self) -> Dict[str, Any]:
        """Run all configured health checks and return the results."""
        results = {"status": HealthStatus.HEALTHY.value, "details": {}}
        all_healthy = True

        for check_name, check_func in self._checks.items():
            if check_func:
                try:
                    is_healthy, message = check_func()
                    results["details"][check_name] = {"healthy": is_healthy, "message": message}
                    if not is_healthy:
                        all_healthy = False
                except Exception as e:
                    results["details"][check_name] = {"healthy": False, "message": f"Error during check: {e}"}
                    all_healthy = False
            else:
                results["details"][check_name] = {"healthy": True, "message": "Check not configured"}

        if not all_healthy:
            results["status"] = HealthStatus.UNHEALTHY.value
            # Could add logic for DEGRADED based on specific checks

        return results

