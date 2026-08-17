import unittest

from examples.run_exact_example import build_exact_example_solver, solve_exact_example


class ExactSolverTests(unittest.TestCase):
    def test_example_distribution_is_normalized(self):
        result = solve_exact_example()
        self.assertAlmostEqual(result.probability_mass, 1.0, places=12)
        self.assertEqual(len(result.terminal_outcomes), 4)

    def test_example_value_and_policy_are_stable(self):
        first = solve_exact_example()
        second = solve_exact_example()
        self.assertEqual(first.canonical_root_action, (0, 1))
        self.assertEqual(first.canonical_root_action, second.canonical_root_action)
        self.assertEqual(first.objective_value, second.objective_value)
        self.assertEqual(first.terminal_outcomes, second.terminal_outcomes)

    def test_example_value_matches_reproduced_reference(self):
        result = solve_exact_example()
        expected = (1.49664762, 2.87878762, 0.58009824)
        for observed, reference in zip(result.objective_value, expected):
            self.assertAlmostEqual(observed, reference, places=8)

    def test_every_reported_successor_is_absorbing(self):
        solver = build_exact_example_solver()
        result = solver.evaluate_start((4,), (1, 1))
        for state in result.absorbing_dist:
            self.assertTrue(solver.is_absorbing(state))


if __name__ == "__main__":
    unittest.main()
