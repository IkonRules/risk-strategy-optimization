"""Run a small exact example against the authoritative Project Risk source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from project_risk.mathematical.small_graph_model.exact_finite_solver import (  # noqa: E402
    CompactExactTopologySolver,
    combat_df_for_caps,
)
from project_risk.validation.distribution_comparison_metrics import (  # noqa: E402
    total_variation_distance,
)
from project_risk.validation.exact_policy_dag import (  # noqa: E402
    export_exact_policy_dag,
    materialize_policy_variant,
)


EXAMPLE_EDGES = ((0, 1), (0, 2))
EXAMPLE_ATTACKER_TROOPS = (4,)
EXAMPLE_DEFENDER_TROOPS = (1, 1)


@dataclass(frozen=True)
class TerminalOutcome:
    state: str
    probability: float


@dataclass(frozen=True)
class ExactExampleSummary:
    objective_value: Tuple[float, ...]
    canonical_root_action: Tuple[int, int]
    optimal_root_action_count: int
    terminal_outcomes: Tuple[TerminalOutcome, ...]
    probability_mass: float
    tied_policy_total_variation: float
    states_evaluated: int


def build_exact_example_solver() -> CompactExactTopologySolver:
    """Construct the real compact solver for the three-node star example."""
    return CompactExactTopologySolver(
        edges=EXAMPLE_EDGES,
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


def solve_exact_example() -> ExactExampleSummary:
    """Solve the example and summarize value, policy, and terminal states."""
    solver = build_exact_example_solver()
    result = solver.evaluate_start(EXAMPLE_ATTACKER_TROOPS, EXAMPLE_DEFENDER_TROOPS)
    distribution = solver.normalize_distribution(result.absorbing_dist)

    policy_dag = export_exact_policy_dag(
        solver=solver,
        root_state=result.state,
        retain_mode="exact_ties",
        max_split_depth=1,
    )
    root_node = policy_dag.nodes[result.state]
    canonical_variant = materialize_policy_variant(policy_dag=policy_dag)
    alternative_signature = next(
        action.action_signature
        for action in root_node.retained_actions
        if not action.is_canonical_action
    )
    alternative_variant = materialize_policy_variant(
        policy_dag=policy_dag,
        action_choices_by_state={result.state: alternative_signature},
    )

    terminal_outcomes = tuple(
        TerminalOutcome(state=solver.state_label(state), probability=float(probability))
        for state, probability in sorted(
            distribution.items(),
            key=lambda item: (-item[1], solver.state_label(item[0])),
        )
    )
    assert result.root_action is not None
    return ExactExampleSummary(
        objective_value=tuple(float(value) for value in result.value),
        canonical_root_action=result.root_action,
        optimal_root_action_count=len(root_node.retained_actions),
        terminal_outcomes=terminal_outcomes,
        probability_mass=sum(outcome.probability for outcome in terminal_outcomes),
        tied_policy_total_variation=total_variation_distance(
            canonical_variant.terminal_distribution,
            alternative_variant.terminal_distribution,
        ),
        states_evaluated=solver.stats.value_evals,
    )


def main() -> None:
    result = solve_exact_example()

    print("Exact small-graph strategy example")
    print(f"  Edges: {EXAMPLE_EDGES}")
    print(
        "  Initial troops: "
        f"attacker={EXAMPLE_ATTACKER_TROOPS}, defenders={EXAMPLE_DEFENDER_TROOPS}"
    )
    print(
        "  Lexicographic value "
        "(expected new territories, expected attacker troops, conquest probability): "
        + str(tuple(round(value, 9) for value in result.objective_value))
    )
    print(f"  Canonical optimal opening attack: {result.canonical_root_action}")
    print(f"  Exactly tied optimal opening attacks: {result.optimal_root_action_count}")
    print(f"  Evaluated compact states: {result.states_evaluated}")
    print("  Terminal successor-state distribution:")
    for outcome in result.terminal_outcomes:
        print(f"    {outcome.probability:.9f}  {outcome.state}")
    print(f"  Probability mass: {result.probability_mass:.12f}")
    print(
        "  TV distance between the two tied labelled-policy distributions: "
        f"{result.tied_policy_total_variation:.9f}"
    )


if __name__ == "__main__":
    main()
