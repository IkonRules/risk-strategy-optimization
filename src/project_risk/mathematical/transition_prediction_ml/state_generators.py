# state_generators.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Any, Optional
import numpy as np

from project_risk.game_simulation import Players
from project_risk.game_simulation import Board
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.transition_prediction_ml.generate_data_ML import (
    ExperimentConstraints, build_full_graph, 
    #compute_basic_macro_metrics, 
    MacroTargets, MacroTolerances)


def _reset_board_state() -> None:
    """Reset singleton Board territory objects without importing simulation/rendering code."""
    for territory in Board.all_territories_list:
        territory._owner = None
        territory._troops = 0




def simple_continent_state_generator(
    target_territory_ratio: float,
    target_troops_ratio: float,
    constraints: ExperimentConstraints,
    rng: np.random.Generator,
) -> Tuple[Sequence["Players.Player"], Any, Any]:
    """
    Simple continent-based state generator.

    Workflow:
      1) Choose continent from constraints.continent_name.
      2) Assign attacker vs defender ownership on *continent nodes* only,
         to roughly match target_territory_ratio.
      3) Assign defender troops, then attacker troops, to roughly match
         target_troops_ratio via the "available attacker troops" definition:
             available_attacker_troops = sum(max(0, troops - 1)) over attacker nodes.
      4) Build:
           - battle_graph  = effective graph for this turn
           - full_graph    = continent + all neighbors (static topology)
      5) Return (players, battle_graph, full_graph).

    Notes:
      - All ownership/troop updates are applied only to the chosen continent's
        territories; other territories retain whatever state SimEng.reset_board_state()
        initializes them to.
      - The battle_graph may exclude some attacker nodes (e.g. 1-troop or non-adjacent
        nodes), by design.
    """

    # --------------------------------------------------------------
    # 0) Figure out which continent we're operating on
    # --------------------------------------------------------------
    continent_name = getattr(constraints, "continent_name", "North America")

    if continent_name not in Board.continent_territory_dict:
        raise ValueError(f"Unknown continent: {continent_name}")

    # --------------------------------------------------------------
    # 1) Players and board reset
    # --------------------------------------------------------------
    p1 = Players.Player("Red")
    p2 = Players.Player("Blue")
    players = [p1, p2]

    # Reset the global board state (ownership/troops)
    _reset_board_state()

    attacker_player = players[0]

    continent_territories = Board.continent_territory_dict[continent_name]
    continent_node_indices = [t._index for t in continent_territories]
    continent_node_count = len(continent_node_indices)

    if continent_node_count == 0:
        raise ValueError(f"Continent {continent_name} has no territories.")

    # --------------------------------------------------------------
    # 2) Decide attacker vs defender territories on the continent
    # --------------------------------------------------------------
    attacker_continent_target_count = int(round(
        target_territory_ratio * continent_node_count
    ))

    # Ensure at least 1 attacker and 1 defender if possible
    attacker_continent_count = max(
        1,
        min(continent_node_count - 1, attacker_continent_target_count),
    )
    defender_continent_count = continent_node_count - attacker_continent_count

    attacker_continent_indices = set(
        rng.choice(continent_node_indices, size=attacker_continent_count, replace=False)
    )
    defender_continent_indices = [
        idx for idx in continent_node_indices if idx not in attacker_continent_indices
    ]

    # --------------------------------------------------------------
    # 3) Assign defender troops on continent nodes
    # --------------------------------------------------------------
    max_defender_per_node = min(constraints.max_defender_troops_per_node, 5)
    defender_troops_per_node = rng.integers(
        1, max_defender_per_node + 1, size=len(defender_continent_indices)
    )
    total_defender_troops = int(defender_troops_per_node.sum())

    # --------------------------------------------------------------
    # 4) Determine feasible attacker "available troops" range
    # --------------------------------------------------------------
    max_attacker_per_node = min(constraints.max_attacker_troops_per_node, 5)
    # Max available attacker troops given per-node caps:
    # each attacker node can contribute at most (maxA - 1) available troops
    max_available_attacker_troops = attacker_continent_count * max(0, max_attacker_per_node - 1)
    min_available_attacker_troops = 0

    if total_defender_troops > 0:
        feasible_min_ratio = 0.0
        feasible_max_ratio = max_available_attacker_troops / total_defender_troops

        clamped_troops_ratio = max(
            feasible_min_ratio,
            min(target_troops_ratio, feasible_max_ratio),
        )

        target_available_attacker_troops = int(
            round(clamped_troops_ratio * total_defender_troops)
        )
        target_available_attacker_troops = max(
            min_available_attacker_troops,
            min(target_available_attacker_troops, max_available_attacker_troops),
        )
    else:
        clamped_troops_ratio = 0.0
        target_available_attacker_troops = 0

    # --------------------------------------------------------------
    # 5) Distribute attacker troops on attacker-owned continent nodes
    # --------------------------------------------------------------
    attacker_troops_per_node = [1] * attacker_continent_count
    extra_troops_to_distribute = target_available_attacker_troops

    # Simple greedy distribution, respecting per-node cap
    while extra_troops_to_distribute > 0:
        any_increment = False
        for idx in range(attacker_continent_count):
            if extra_troops_to_distribute <= 0:
                break
            if attacker_troops_per_node[idx] < max_attacker_per_node:
                attacker_troops_per_node[idx] += 1
                extra_troops_to_distribute -= 1
                any_increment = True
        if not any_increment:
            # All at max; can't place more
            break

    # --------------------------------------------------------------
    # 6) Write ownership & troops back to the Board for continent nodes
    # --------------------------------------------------------------
    # First clear continent territories (neutral with 0 troops)
    for idx in continent_node_indices:
        terr = Board.node_to_territory_dict[idx]
        terr._owner = None
        terr._troops = 0

    # Defender territories (owned by p2 so they are enemies for p1)
    for i, terr_idx in enumerate(defender_continent_indices):
        terr = Board.node_to_territory_dict[terr_idx]
        terr._owner = p2
        terr._troops = int(defender_troops_per_node[i])

    # Attacker territories
    attacker_continent_indices_list = list(attacker_continent_indices)
    for i, terr_idx in enumerate(attacker_continent_indices_list):
        terr = Board.node_to_territory_dict[terr_idx]
        terr._owner = attacker_player
        terr._troops = attacker_troops_per_node[i]

    # --------------------------------------------------------------
    # 7) Build the two graph views:
    #    - battle_graph: effective graph given this turn's rules
    #    - full_graph:   static topology = continent + neighbors
    # --------------------------------------------------------------
    battle_graph = agop.build_continent_battle_graph(continent_name, players)
    full_graph = build_full_graph(continent_name)

    return players, battle_graph, full_graph



def macro_constraints_satisfied(
    realized: dict,
    targets: MacroTargets,
    tol: MacroTolerances,
) -> bool:
    """
    Return True if all non-None targets are satisfied within tolerances.
    """
    # convenience helper
    def check_float(name, target, tol_val):
        if target is None:
            return True
        val = realized[name]
        if np.isnan(val):
            return False
        return abs(val - target) <= tol_val

    def check_int(name, target, tol_val):
        if target is None:
            return True
        val = realized[name]
        return abs(int(val) - int(target)) <= tol_val

    # battle-level
    if not check_float(
        "battle_attacker_territory_ratio",
        targets.battle_attacker_territory_ratio,
        tol.battle_attacker_territory_ratio,
    ):
        return False

    if not check_float(
        "battle_attacker_available_troops_ratio",
        targets.battle_attacker_available_troops_ratio,
        tol.battle_attacker_available_troops_ratio,
    ):
        return False

    if not check_int(
        "battle_total_territory_count",
        targets.battle_total_territory_count,
        tol.battle_total_territory_count,
    ):
        return False

    if not check_int(
        "battle_total_troops_count",
        targets.battle_total_troops_count,
        tol.battle_total_troops_count,
    ):
        return False

    # full graph
    if not check_float(
        "full_attacker_territory_ratio",
        targets.full_attacker_territory_ratio,
        tol.full_attacker_territory_ratio,
    ):
        return False

    if not check_float(
        "full_attacker_troops_ratio",
        targets.full_attacker_troops_ratio,
        tol.full_attacker_troops_ratio,
    ):
        return False

    return True


# def generate_state_with_macro_constraints(
#     constraints: ExperimentConstraints,
#     rng: np.random.Generator,
#     macro_targets: MacroTargets,
#     macro_tolerances: MacroTolerances,
#     max_tries: int = 2000,
# ) -> Tuple[Sequence["Players.Player"], any, any, dict]:
#     """
#     Wrapper generator that keeps sampling continent states until
#     macro-level constraints are (approximately) satisfied.

#     Returns
#     -------
#     players, battle_graph, full_graph, realized_macro_metrics
#     """
#     # If user wants to fix target_territory_ratio / target_troops_ratio,
#     # use those; otherwise sample them freely in a reasonable range.
#     def sample_target_ratios():
#         if macro_targets.target_territory_ratio is not None:
#             tTerr = macro_targets.target_territory_ratio
#         else:
#             tTerr = rng.uniform(0.2, 0.8)

#         if macro_targets.target_troops_ratio is not None:
#             tTroops = macro_targets.target_troops_ratio
#         else:
#             tTroops = rng.uniform(0.2, 2.0)

#         return float(tTerr), float(tTroops)

#     for attempt in range(max_tries):
#         target_territory_ratio, target_troops_ratio = sample_target_ratios()

#         players, battle_graph, full_graph = simple_continent_state_generator(
#             target_territory_ratio=target_territory_ratio,
#             target_troops_ratio=target_troops_ratio,
#             constraints=constraints,
#             rng=rng,
#         )

#         global_state = agop.build_global_state_for_board(players)

#         realized = compute_basic_macro_metrics(
#             global_state=global_state,
#             battle_graph=battle_graph,
#             full_graph=full_graph,
#         )

#         # Optionally store back the requested targets for inspection
#         realized["target_territory_ratio"] = target_territory_ratio
#         realized["target_troops_ratio"] = target_troops_ratio

#         if macro_constraints_satisfied(realized, macro_targets, macro_tolerances):
#             return players, battle_graph, full_graph, realized

#     raise RuntimeError(
#         f"Could not generate a state satisfying macro constraints within {max_tries} attempts."
#     )



def ml_full_graph_state_generator(
    target_territory_ratio: float,
    target_troops_ratio: float,
    constraints: ExperimentConstraints,
    rng: np.random.Generator,
) -> Tuple[Sequence["Players.Player"], Any, Any]:
    """
    ML-focused state generator.

    Goal:
      - Control ATTACKER vs DEFENDER ownership over the *entire* full_graph
        (continent + neighbours), not just the continent.
      - Attacker fraction over full_graph ~ target_territory_ratio ∈ [0,1].
      - Attacker "available troops" vs defender troops ~ target_troops_ratio,
        using the same semantics as the old generator.

    This keeps all nodes (continent + neighbours) in play, so:
      - defender can own all continent + all neighbours,
      - defender can own all continent but not all neighbours,
      - attacker can own some continent nodes and no neighbours,
      - attacker can own everything, etc.

    It returns:
      (players, battle_graph, full_graph)
    compatible with run_node_transition_experiment.
    """

    continent_name = getattr(constraints, "continent_name", "North America")

    # --------------------------------------------------------------
    # 0) Players and full reset
    # --------------------------------------------------------------
    p1 = Players.Player("Red")
    p2 = Players.Player("Blue")
    players = [p1, p2]

    _reset_board_state()

    # --------------------------------------------------------------
    # 1) Build full_graph for this continent
    #    (your existing helper, same as in simple_continent_state_generator)
    # --------------------------------------------------------------
    full_graph = build_full_graph(continent_name)

    try:
        full_nodes_iter = full_graph.nodes()
    except TypeError:
        full_nodes_iter = full_graph.nodes
    full_nodes = list(full_nodes_iter)
    full_node_count = len(full_nodes)

    if full_node_count == 0:
        raise ValueError(f"Full graph for continent {continent_name} has no nodes.")

    # --------------------------------------------------------------
    # 2) Clear ownership & troops for all nodes in full_graph
    # --------------------------------------------------------------
    for idx in full_nodes:
        terr = Board.node_to_territory_dict[idx]
        terr._owner = None
        terr._troops = 0

    # --------------------------------------------------------------
    # 3) Decide attacker vs defender ownership on FULL graph
    #    Attacker fraction over full_graph ≈ target_territory_ratio.
    # --------------------------------------------------------------
    attacker_full_target_count = int(round(target_territory_ratio * full_node_count))

    # Ensure at least 1 attacker and 1 defender (avoid degenerate "one player missing")
    attacker_full_count = max(
        1,
        min(full_node_count - 1, attacker_full_target_count),
    )
    defender_full_count = full_node_count - attacker_full_count

    attacker_full_indices = set(
        rng.choice(full_nodes, size=attacker_full_count, replace=False)
    )
    defender_full_indices = [idx for idx in full_nodes if idx not in attacker_full_indices]

    # --------------------------------------------------------------
    # 4) Assign defender troops on ALL defender nodes in full_graph
    # --------------------------------------------------------------
    max_defender_per_node = min(constraints.max_defender_troops_per_node, 5)
    defender_troops_per_node = rng.integers(
        1, max_defender_per_node + 1, size=len(defender_full_indices)
    )
    total_defender_troops = int(defender_troops_per_node.sum())

    # --------------------------------------------------------------
    # 5) Determine feasible attacker "available troops" range
    #    (same semantics as before: available = sum(max(0, troops-1)) over A nodes)
    # --------------------------------------------------------------
    max_attacker_per_node = min(constraints.max_attacker_troops_per_node, 5)

    max_available_attacker_troops = attacker_full_count * max(0, max_attacker_per_node - 1)
    min_available_attacker_troops = 0

    if total_defender_troops > 0:
        feasible_min_ratio = 0.0
        feasible_max_ratio = (
            max_available_attacker_troops / total_defender_troops
            if total_defender_troops > 0 else 0.0
        )

        clamped_troops_ratio = max(
            feasible_min_ratio,
            min(target_troops_ratio, feasible_max_ratio),
        )

        target_available_attacker_troops = int(
            round(clamped_troops_ratio * total_defender_troops)
        )
        target_available_attacker_troops = max(
            min_available_attacker_troops,
            min(target_available_attacker_troops, max_available_attacker_troops),
        )
    else:
        clamped_troops_ratio = 0.0
        target_available_attacker_troops = 0

    # --------------------------------------------------------------
    # 6) Distribute attacker troops on ALL attacker nodes in full_graph
    #    - Start with 1 troop per attacker node
    #    - Extra troops = target_available_attacker_troops
    #    - Extra troops contribute to "available" (troops - 1)
    # --------------------------------------------------------------
    attacker_troops_per_node = [1] * attacker_full_count
    extra_troops_to_distribute = target_available_attacker_troops

    while extra_troops_to_distribute > 0:
        any_increment = False
        for i in range(attacker_full_count):
            if extra_troops_to_distribute <= 0:
                break
            if attacker_troops_per_node[i] < max_attacker_per_node:
                attacker_troops_per_node[i] += 1
                extra_troops_to_distribute -= 1
                any_increment = True
        if not any_increment:
            break  # all at cap; can't add more

    # --------------------------------------------------------------
    # 7) Write ownership & troops back to the Board for full_graph nodes
    # --------------------------------------------------------------
    # Defender territories (owned by p2)
    for i, terr_idx in enumerate(defender_full_indices):
        terr = Board.node_to_territory_dict[terr_idx]
        terr._owner = p2
        terr._troops = int(defender_troops_per_node[i])

    # Attacker territories (owned by p1)
    attacker_full_indices_list = list(attacker_full_indices)
    for i, terr_idx in enumerate(attacker_full_indices_list):
        terr = Board.node_to_territory_dict[terr_idx]
        terr._owner = p1
        terr._troops = attacker_troops_per_node[i]

    # --------------------------------------------------------------
    # 8) Battle graph for this turn (same rules as before)
    #    - Still continent-based, but now attacker/defender ownership
    #      is controlled over the whole full_graph.
    # --------------------------------------------------------------
    battle_graph = agop.build_continent_battle_graph(continent_name, players)

    return players, battle_graph, full_graph



def generate_ML_initial_state(
    seed: Optional[int] = None,
    continent_name: str = "North America",
    territory_ratio_range: Tuple[float, float] = (0.2, 0.8),
    troops_ratio_range: Tuple[float, float] = (0.5, 2.0),
    constraints: Optional[ExperimentConstraints] = None,
) -> Tuple[Sequence["Players.Player"], Any, Any]:
    """
    Convenience wrapper for the ML simulation:

      - Picks random target_territory_ratio and target_troops_ratio
        within given ranges.
      - Builds default ExperimentConstraints if none are provided.
      - Internally calls ml_full_graph_state_generator(...).

    Returns:
      players, battle_graph, full_graph
    """
    rng = np.random.default_rng(seed)

    if constraints is None:
        constraints = ExperimentConstraints(
            continent_name=continent_name,
            max_attacker_troops_per_node=8,
            max_defender_troops_per_node=8,
        )

    target_territory_ratio = float(
        rng.uniform(territory_ratio_range[0], territory_ratio_range[1])
    )
    target_troops_ratio = float(
        rng.uniform(troops_ratio_range[0], troops_ratio_range[1])
    )

    players, battle_graph, full_graph = ml_full_graph_state_generator(
        target_territory_ratio=target_territory_ratio,
        target_troops_ratio=target_troops_ratio,
        constraints=constraints,
        rng=rng,
    )

    return players, battle_graph, full_graph

