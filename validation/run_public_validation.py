"""Run the compact validations reproducible from the public repository alone."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from risk_strategy.demo import solve_exact_example  # noqa: E402
from risk_strategy.markov_matrix_probabilities import battle_summary  # noqa: E402


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
