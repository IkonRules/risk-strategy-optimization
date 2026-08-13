"""Run the small exact graph example from a repository checkout."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from risk_strategy.demo import (  # noqa: E402
    EXAMPLE_ATTACKER_TROOPS,
    EXAMPLE_DEFENDER_TROOPS,
    EXAMPLE_EDGES,
    solve_exact_example,
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
