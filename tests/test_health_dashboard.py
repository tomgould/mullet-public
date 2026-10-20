import unittest
from mullet.core.providers.health_dashboard import ProviderHealthDashboard

class TestProviderHealthDashboard(unittest.TestCase):

    def setUp(self):
        self.dashboard = ProviderHealthDashboard()

    def test_initial_state(self):
        # Test 1: Dashboard is empty initially
        report = self.dashboard.get_report()
        self.assertEqual(report, {})

    def test_record_single_successful_call(self):
        # Test 2: Record a single successful call
        self.dashboard.record_call("provider_A", 0.1, True)
        report = self.dashboard.get_report()
        self.assertIn("provider_A", report)
        self.assertEqual(report["provider_A"]["total_calls"], 1)
        self.assertEqual(report["provider_A"]["error_rate"], 0.0)
        self.assertAlmostEqual(report["provider_A"]["average_latency"], 0.1)
        self.assertAlmostEqual(report["provider_A"]["min_latency"], 0.1)
        self.assertAlmostEqual(report["provider_A"]["max_latency"], 0.1)

    def test_record_single_failed_call(self):
        # Test 3: Record a single failed call
        self.dashboard.record_call("provider_B", 0.2, False)
        report = self.dashboard.get_report()
        self.assertIn("provider_B", report)
        self.assertEqual(report["provider_B"]["total_calls"], 1)
        self.assertAlmostEqual(report["provider_B"]["error_rate"], 1.0)
        self.assertAlmostEqual(report["provider_B"]["average_latency"], 0.2)
        self.assertAlmostEqual(report["provider_B"]["min_latency"], 0.2)
        self.assertAlmostEqual(report["provider_B"]["max_latency"], 0.2)

    def test_record_multiple_calls_same_provider(self):
        # Test 4: Record multiple calls for the same provider
        self.dashboard.record_call("provider_A", 0.1, True)
        self.dashboard.record_call("provider_A", 0.3, False)
        self.dashboard.record_call("provider_A", 0.2, True)
        report = self.dashboard.get_report()
        self.assertEqual(report["provider_A"]["total_calls"], 3)
        self.assertAlmostEqual(report["provider_A"]["error_rate"], 1/3)
        self.assertAlmostEqual(report["provider_A"]["average_latency"], (0.1 + 0.3 + 0.2) / 3)
        self.assertAlmostEqual(report["provider_A"]["min_latency"], 0.1)
        self.assertAlmostEqual(report["provider_A"]["max_latency"], 0.3)

    def test_record_multiple_providers(self):
        # Test 5: Record calls for multiple providers
        self.dashboard.record_call("provider_A", 0.1, True)
        self.dashboard.record_call("provider_B", 0.5, True)
        self.dashboard.record_call("provider_A", 0.2, True)
        report = self.dashboard.get_report()
        self.assertIn("provider_A", report)
        self.assertIn("provider_B", report)
        self.assertEqual(report["provider_A"]["total_calls"], 2)
        self.assertEqual(report["provider_B"]["total_calls"], 1)

    def test_get_report_no_calls(self):
        # Test 6: Report for a provider with no calls should not exist
        self.dashboard.record_call("provider_C", 0.1, True) # Record for a different provider
        report = self.dashboard.get_report()
        self.assertNotIn("provider_D", report)

    def test_latency_updates(self):
        # Test 7: Verify min/max latency updates
        self.dashboard.record_call("provider_X", 0.8, True)
        self.dashboard.record_call("provider_X", 0.1, True)
        self.dashboard.record_call("provider_X", 0.5, True)
        report = self.dashboard.get_report()
        self.assertAlmostEqual(report["provider_X"]["min_latency"], 0.1)
        self.assertAlmostEqual(report["provider_X"]["max_latency"], 0.8)

    def test_error_rate_calculation(self):
        # Test 8: Verify error rate calculation
        self.dashboard.record_call("provider_Y", 0.1, True)
        self.dashboard.record_call("provider_Y", 0.2, False)
        self.dashboard.record_call("provider_Y", 0.3, True)
        self.dashboard.record_call("provider_Y", 0.4, False)
        report = self.dashboard.get_report()
        self.assertEqual(report["provider_Y"]["total_calls"], 4)
        self.assertAlmostEqual(report["provider_Y"]["error_rate"], 2/4) # 2 failures out of 4 calls

if __name__ == '__main__':
    unittest.main()
