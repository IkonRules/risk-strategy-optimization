import unittest

from risk_strategy.exact_finite_solver import CompactExactTopologySolver, combat_df_for_caps
from risk_strategy.small_graph_outcome_probabilities import canonicalize_edges_with_roles


EDGES = ((0, 2), (1, 2), (1, 3))
OLD_TO_NEW = (1, 0, 3, 2)
PERMUTED_EDGES = tuple(
    sorted(
        (min(OLD_TO_NEW[u], OLD_TO_NEW[v]), max(OLD_TO_NEW[u], OLD_TO_NEW[v]))
        for u, v in EDGES
    )
)


def make_solver(edges):
    return CompactExactTopologySolver(
        edges=edges,
        num_attacker_nodes=2,
        num_defender_nodes=2,
        combat_df=combat_df_for_caps(
            num_attacker_nodes=2,
            num_defender_nodes=2,
            max_attacker_troops=3,
            max_defender_troops=2,
        ),
        utility_mode="local",
        max_total_troops=10,
        cache_distributions=True,
        sort_actions=True,
    )


def labelled_distribution_in_old_order(solver, distribution, old_to_new=None):
    result = {}
    for state, probability in solver.normalize_distribution(distribution).items():
        nodes = solver.state_to_global_state(state).nodes
        if old_to_new is None:
            signature = tuple((node.owner, node.troops) for node in nodes)
        else:
            signature = tuple(
                (nodes[old_to_new[old]].owner, nodes[old_to_new[old]].troops)
                for old in range(len(nodes))
            )
        result[signature] = result.get(signature, 0.0) + probability
    return result


class CanonicalizationTests(unittest.TestCase):
    def test_role_preserving_isomorphisms_share_a_canonical_graph(self):
        canonical_a, _, _ = canonicalize_edges_with_roles(EDGES, 2, 2)
        canonical_b, _, _ = canonicalize_edges_with_roles(PERMUTED_EDGES, 2, 2)
        self.assertEqual(canonical_a, canonical_b)

    def test_isomorphic_relabelling_preserves_exact_distribution(self):
        original_solver = make_solver(EDGES)
        relabelled_solver = make_solver(PERMUTED_EDGES)
        original = original_solver.evaluate_start((3, 2), (1, 2))
        relabelled = relabelled_solver.evaluate_start((2, 3), (2, 1))

        self.assertEqual(original.value, relabelled.value)
        original_distribution = labelled_distribution_in_old_order(
            original_solver,
            original.absorbing_dist,
        )
        relabelled_distribution = labelled_distribution_in_old_order(
            relabelled_solver,
            relabelled.absorbing_dist,
            OLD_TO_NEW,
        )
        self.assertEqual(set(original_distribution), set(relabelled_distribution))
        for state in original_distribution:
            self.assertAlmostEqual(
                original_distribution[state],
                relabelled_distribution[state],
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
