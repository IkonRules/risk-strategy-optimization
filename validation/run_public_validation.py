"""Run the compact validations reproducible from the public repository alone."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.run_exact_example import solve_exact_example  # noqa: E402
from project_risk.mathematical.small_graph_model.markov_matrix_probabilities import (  # noqa: E402
    battle_summary,
)


def main() -> None:
    combat = battle_summary(1, 1)
    example = solve_exact_example()

    combat_mass = combat["p_attacker_wins"] + combat["p_defender_wins"]
    assert abs(combat_mass - 1.0) < 1e-12
    assert abs(example.probability_mass - 1.0) < 1e-12
    assert example.optimal_root_action_count == 2
    assert example.tied_policy_total_variation > 0.0

    print("Public validation PASS")
    print(f"  One-versus-one combat mass: {combat_mass:.12f}")
    print(f"  Exact terminal distribution mass: {example.probability_mass:.12f}")
    print(f"  Terminal support: {len(example.terminal_outcomes)} states")
    print(f"  Tied optimal opening actions: {example.optimal_root_action_count}")
    print(f"  Tied-policy labelled-state TV: {example.tied_policy_total_variation:.9f}")


if __name__ == "__main__":
    main()
