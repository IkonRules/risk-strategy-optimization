import unittest

import numpy as np

from risk_strategy.markov_matrix_probabilities import battle_summary


class CombatKernelTests(unittest.TestCase):
    def test_one_versus_one_probabilities_sum_to_one(self):
        result = battle_summary(1, 1)
        total = result["p_attacker_wins"] + result["p_defender_wins"]
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_one_versus_one_matches_reference_table(self):
        result = battle_summary(1, 1)
        self.assertAlmostEqual(result["p_attacker_wins"], 0.417, places=12)
        self.assertAlmostEqual(result["p_defender_wins"], 0.583, places=12)

    def test_absorbing_outcomes_are_legal_and_nonnegative(self):
        frame = battle_summary(4, 3)["F_df"]
        self.assertTrue(np.isfinite(frame.to_numpy()).all())
        self.assertGreaterEqual(float(frame.to_numpy().min()), -1e-12)
        for label in frame.columns:
            attacker, defender = (int(part) for part in label.strip("()").split(","))
            self.assertNotEqual(attacker == 0, defender == 0)


if __name__ == "__main__":
    unittest.main()
