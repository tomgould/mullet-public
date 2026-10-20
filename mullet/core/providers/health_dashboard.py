"""Provides a health dashboard view of provider performance metrics."""


class ProviderHealthDashboard:
    """Track and report health metrics for LLM providers."""

    def __init__(self):
        self.providers_data = {}

    def record_call(self, provider: str, latency: float, success: bool):
        """Record the result of a provider call."""
        if provider not in self.providers_data:
            self.providers_data[provider] = {
                'total_calls': 0,
                'successful_calls': 0,
                'total_latency': 0.0,
                'min_latency': float('inf'),
                'max_latency': float('-inf'),
            }

        data = self.providers_data[provider]
        data['total_calls'] += 1
        data['total_latency'] += latency
        data['min_latency'] = min(data['min_latency'], latency)
        data['max_latency'] = max(data['max_latency'], latency)

        if success:
            data['successful_calls'] += 1

    def get_report(self) -> dict:
        """Return a summary report of all recorded provider metrics."""
        report = {}
        for provider, data in self.providers_data.items():
            total_calls = data['total_calls']
            successful_calls = data['successful_calls']
            total_latency = data['total_latency']

            error_rate = 0.0
            average_latency = 0.0

            if total_calls > 0:
                error_rate = (total_calls - successful_calls) / total_calls
                average_latency = total_latency / total_calls

            report[provider] = {
                'total_calls': total_calls,
                'error_rate': error_rate,
                'average_latency': average_latency,
                'min_latency': data['min_latency'] if data['min_latency'] != float('inf') else 0.0,
                'max_latency': data['max_latency'] if data['max_latency'] != float('-inf') else 0.0,
            }
        return report
