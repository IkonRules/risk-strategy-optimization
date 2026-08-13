import unittest

from risk_strategy.demo import build_exact_example_solver
from risk_strategy.distribution_comparison_metrics import total_variation_distance
from risk_strategy.exact_policy_dag import export_exact_policy_dag, materialize_policy_variant


class PolicyDagTests(unittest.TestCase):
    def setUp(self):
        self.solver = build_exact_example_solver()
        self.root = self.solver.initial_state((4,), (1, 1))
        self.solver.evaluate_start((4,), (1, 1))
        self.dag = export_exact_policy_dag(
            solver=self.solver,
            root_state=self.root,
            retain_mode="exact_ties",
            max_split_depth=1,
        )

    def test_exact_ties_are_preserved(self):
        root = self.dag.nodes[self.root]
        self.assertEqual(len(root.retained_actions), 2)
        self.assertTrue(all(action.is_exact_tied_optimal for action in root.retained_actions))

    def test_tied_policies_keep_separate_normalized_distributions(self):
        root = self.dag.nodes[self.root]
        canonical = materialize_policy_variant(policy_dag=self.dag)
        alternative_signature = next(
            action.action_signature
            for action in root.retained_actions
            if not action.is_canonical_action
        )
        alternative = materialize_policy_variant(
            policy_dag=self.dag,
            action_choices_by_state={self.root: alternative_signature},
        )
        self.assertEqual(canonical.exact_value, alternative.exact_value)
        self.assertAlmostEqual(sum(canonical.terminal_distribution.values()), 1.0, places=12)
        self.assertAlmostEqual(sum(alternative.terminal_distribution.values()), 1.0, places=12)
        self.assertAlmostEqual(
            total_variation_distance(
                canonical.terminal_distribution,
                alternative.terminal_distribution,
            ),
            0.71859114,
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
