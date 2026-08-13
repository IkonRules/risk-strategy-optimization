import unittest

from risk_strategy.distribution_comparison_metrics import total_variation_distance


class DistributionMetricTests(unittest.TestCase):
    def test_identity_has_zero_total_variation(self):
        distribution = {"left": 0.75, "right": 0.25}
        self.assertEqual(total_variation_distance(distribution, distribution), 0.0)

    def test_known_total_variation_distance(self):
        left = {"a": 0.75, "b": 0.25}
        right = {"a": 0.25, "b": 0.75}
        self.assertAlmostEqual(total_variation_distance(left, right), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
