import importlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PUBLIC_RESEARCH_MODULES = (
    "project_risk.game_simulation.Board",
    "project_risk.game_simulation.Players",
    "project_risk.game_simulation.SimulationFunctions",
    "project_risk.game_simulation.SimulationEngine",
    "project_risk.mathematical.small_graph_model.markov_matrix_probabilities",
    "project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities",
    "project_risk.mathematical.small_graph_model.exact_finite_solver",
    "project_risk.mathematical.libraries.canonicalize_graphs",
    "project_risk.mathematical.libraries.create_library",
    "project_risk.mathematical.libraries.library_io",
    "project_risk.mathematical.continent_model.approximate_graph_outcome_probabilities",
    "project_risk.mathematical.continent_model.battle_graph_ranking",
    "project_risk.mathematical.transition_prediction_ml.generate_data_ML",
    "project_risk.mathematical.transition_prediction_ml.state_generators",
    "project_risk.mathematical.transition_prediction_ml.train_ML",
    "project_risk.mathematical.transition_prediction_ml.predict_future_states_ML",
    "project_risk.mathematical.transition_prediction_ml.transition_distribution_ML",
    "project_risk.mathematical.transition_prediction_ml.transition_distribution_stage_a_v2",
    "project_risk.mathematical.transition_prediction_ml.transition_distribution_stage_a_v3",
    "project_risk.mathematical.transition_prediction_ml.run_transition_distribution_stage_a",
    "project_risk.mathematical.full_board_model.full_board_state_generators",
    "project_risk.mathematical.full_board_model.full_board_simulation_ML",
    "project_risk.mathematical.full_board_model.full_board_simulation_GT",
    "project_risk.mathematical.full_board_model.strategy_policy_gt",
    "project_risk.mathematical.strategic_evaluation.utility_terminal",
    "project_risk.mathematical.strategic_evaluation.game_theory_commitment",
    "project_risk.infrastructure.log_config",
    "project_risk.validation.distribution_comparison_metrics",
    "project_risk.validation.exact_policy_dag",
    "project_risk.validation.regional_compounding_validation",
    "project_risk.validation.regional_compounding_validation_v2",
    "project_risk.validation.validate_exact_finite_library_builder",
    "project_risk.validation.check_exact_finite_library_contents",
    "project_risk.validation.transition_distribution_validation",
    "project_risk.validation.preflight_checks_training",
    "project_risk.validation.verify_state_set_cap7_full_library",
)


class ResearchSourceImportTests(unittest.TestCase):
    def test_all_extracted_layers_import_without_artifacts_or_execution(self):
        for module_name in PUBLIC_RESEARCH_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


class ResearchSourceExactTests(unittest.TestCase):
    def test_research_combat_kernel_normalizes(self):
        from project_risk.mathematical.small_graph_model.markov_matrix_probabilities import (
            battle_summary,
        )

        result = battle_summary(1, 1)
        self.assertAlmostEqual(
            result["p_attacker_wins"] + result["p_defender_wins"],
            1.0,
            places=12,
        )

    def test_research_exact_solver_matches_public_example_value(self):
        from project_risk.mathematical.small_graph_model.exact_finite_solver import (
            CompactExactTopologySolver,
            combat_df_for_caps,
        )

        solver = CompactExactTopologySolver(
            edges=((0, 1), (0, 2)),
            num_attacker_nodes=1,
            num_defender_nodes=2,
            combat_df=combat_df_for_caps(
                num_attacker_nodes=1,
                num_defender_nodes=2,
                max_attacker_troops=4,
                max_defender_troops=1,
            ),
            utility_mode="local",
            max_total_troops=6,
            cache_distributions=True,
            sort_actions=True,
        )
        result = solver.evaluate_start((4,), (1, 1))
        expected = (1.49664762, 2.87878762, 0.58009824)
        for observed, reference in zip(result.value, expected):
            self.assertAlmostEqual(observed, reference, places=8)
        self.assertAlmostEqual(sum(result.absorbing_dist.values()), 1.0, places=12)

    def test_role_preserving_canonicalization_is_stable(self):
        from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
            canonicalize_edges_with_roles,
        )

        first = ((0, 2), (1, 2), (1, 3))
        relabel = (1, 0, 3, 2)
        second = tuple(
            sorted(
                (min(relabel[u], relabel[v]), max(relabel[u], relabel[v]))
                for u, v in first
            )
        )
        canonical_a, _, _ = canonicalize_edges_with_roles(first, 2, 2)
        canonical_b, _, _ = canonicalize_edges_with_roles(second, 2, 2)
        self.assertEqual(canonical_a, canonical_b)

    def test_tiny_policy_library_build_and_lookup(self):
        from project_risk.mathematical.libraries.create_library import (
            build_libraries_grid_exact_finite,
            graph_path,
            load_library,
        )
        from project_risk.mathematical.libraries.library_io import (
            get_prob_row_payload_from_library,
        )
        from project_risk.mathematical.small_graph_model.exact_finite_solver import (
            combat_df_for_caps,
        )

        edges = [(0, 1)]
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            stats = build_libraries_grid_exact_finite(
                num_attacker_nodes=1,
                num_defender_nodes=1,
                max_attacker_troops=2,
                max_defender_troops=2,
                combat_df=combat_df_for_caps(
                    num_attacker_nodes=1,
                    num_defender_nodes=1,
                    max_attacker_troops=2,
                    max_defender_troops=2,
                ),
                base_dir=base,
                overwrite=True,
                edges_list=[edges],
                chunk_rows=4,
                utility_mode="local",
                multi_policy_options=False,
            )
            self.assertEqual(stats["num_libraries_written"], 1)
            path = graph_path(edges, 1, 1, 2, 2, base_dir=base)
            library = load_library(path)
            payload = get_prob_row_payload_from_library(
                library,
                "(A2,D1)",
                allow_extrapolation=False,
                num_attacker_nodes=1,
                library_pkl_path=str(path),
            )
            self.assertIsNotNone(payload)
            self.assertAlmostEqual(float(payload["p"].sum()), 1.0, places=12)


class StrategicEvaluationTests(unittest.TestCase):
    def test_terminal_utility_scores_compatible_global_states(self):
        from project_risk.game_simulation import Board
        from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
            GlobalState,
            NodeState,
        )
        from project_risk.mathematical.strategic_evaluation.utility_terminal import (
            UtilityWeights,
            continent_outcome_payoff,
        )

        max_node = max(int(index) for index in Board.node_to_territory_dict)
        start_nodes = [NodeState(owner="D", troops=1) for _ in range(max_node + 1)]
        end_nodes = list(start_nodes)
        continent = next(iter(Board.continent_territory_dict))
        captured = int(Board.continent_territory_dict[continent][0]._index)
        end_nodes[captured] = NodeState(owner="A", troops=1)

        breakdown = continent_outcome_payoff(
            continent=continent,
            owner="A",
            start_state=GlobalState(nodes=tuple(start_nodes)),
            end_state=GlobalState(nodes=tuple(end_nodes)),
            weights=UtilityWeights(),
        )
        self.assertEqual(breakdown.new_territories, 1)
        self.assertGreater(breakdown.payoff_total, 0.0)


if __name__ == "__main__":
    unittest.main()
