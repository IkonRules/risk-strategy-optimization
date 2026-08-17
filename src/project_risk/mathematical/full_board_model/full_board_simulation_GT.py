# full_board_simulation_GT.py
"""
Full-board *game-theoretic* (GT) simulator.

This module wraps the existing ML full-board transition mechanics but injects:
- a commitment policy based on (s1, s2)
- even splitting of reinforcements across committed continents
- end-of-turn fortify (one move) guided by commitment

Turn structure (per user clarification)
--------------------------------------
For each player on their turn:
  1) Reinforcement allocation at the very beginning of turn
  2) Combat (ML expectations per continent, with commitment_map to avoid double counting)
  3) Fortify/reallocation of existing troops at end of turn (one move, along friendly path)

This differs from `full_board_simulation_ML.simulate_multi_turns_full_board_ML`
which includes additional within-continent reallocations. In GT we keep the
mechanics minimal and strategy-dependent; later you can add richer troop movement
optimizers.

Indexing note
-------------
Board node indices are 1..42 (see Board.Territory._index). GlobalState may contain
a dummy index 0; always rely on Board.node_to_territory_dict keys when enumerating.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop

from project_risk.mathematical.transition_prediction_ml.generate_data_ML import build_full_graph, apply_global_state_to_board
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState

from project_risk.mathematical.transition_prediction_ml.predict_future_states_ML import compute_macro_features_from_global_state, apply_expectations_as_state
from project_risk.mathematical.full_board_model.full_board_simulation_ML import (
    Commitment as MLCommitment,
    load_models_by_continent,
    compute_board_reinforcements_risklike,
    apply_reinforcement_allocation,
)

from project_risk.mathematical.full_board_model.strategy_policy_gt import (
    CommitmentPolicyGT,
    split_reinforcements_even,
    choose_end_of_turn_fortify_move,
    apply_fortify_move,
)


@dataclass(frozen=True)
class GTStepDiag:
    turn: int
    attacker_owner: str
    defender_owner: str
    attacker_committed: Tuple[str, ...]
    defender_committed: Tuple[str, ...]
    attacker_reinf: int
    defender_reinf: int
    attacker_split: Dict[str, int]
    defender_split: Dict[str, int]
    attacker_commitment_nodes: int
    defender_commitment_nodes: int


def _swap_roles_in_state(global_state: GlobalState) -> GlobalState:
    """Swap A<->D in a GlobalState (same as generate_data_ML.swap_roles_in_global_state, but local)."""
    nodes = []
    for n in global_state.nodes:
        if getattr(n, "owner", None) == "A":
            nodes.append(type(n)(owner="D", troops=n.troops))
        elif getattr(n, "owner", None) == "D":
            nodes.append(type(n)(owner="A", troops=n.troops))
        else:
            nodes.append(n)
    return GlobalState(nodes=tuple(nodes))


def _apply_combat_all_continents(
    *,
    state: GlobalState,
    players: Sequence["Players.Player"],
    models_by_continent: Dict[str, Dict[str, Any]],
    commitment: MLCommitment,
    attacker_is_player1: bool,
    turn: int,
) -> GlobalState:
    """
    Apply ML expectations sequentially continent-by-continent, using `commitment_map`
    so outside nodes are counted only for their committed target continent.
    """
    persp_state = state if attacker_is_player1 else _swap_roles_in_state(state)
    persp_players = players if attacker_is_player1 else [players[1], players[0]]

    apply_global_state_to_board(persp_state, persp_players)

    for cont in Board.continent_territory_dict.keys():
        models = models_by_continent.get(cont)
        if models is None:
            continue

        full_graph = build_full_graph(cont)
        battle_graph = agop.build_continent_battle_graph(
            cont,
            persp_players,
            debug=False,
            commitment_map=getattr(commitment, "commitment_by_node", None),
        )
        macro = compute_macro_features_from_global_state(persp_state, battle_graph, full_graph)

        # Note: pass continent_name for richer diagnostics; harmless.
        persp_state = apply_expectations_as_state(
            global_state=persp_state,
            battle_graph=battle_graph,
            full_graph=full_graph,
            models_bundle=models,
            macro_features=macro,
            attack_perspective="P1_as_attacker",
            continent_name=str(cont),
            step=int(turn),
            return_diag=False,
        )

        apply_global_state_to_board(persp_state, persp_players)

    return persp_state if attacker_is_player1 else _swap_roles_in_state(persp_state)


def _alloc_reinforcements_by_commitment(
    *,
    state: GlobalState,
    players: Sequence["Players.Player"],
    models_by_continent: Dict[str, Dict[str, Any]],
    owner: str,
    reinf_total: int,
    committed_continents: Sequence[str],
    commitment_map: Dict[int, str],
) -> Tuple[GlobalState, Dict[str, int]]:
    """
    Apply reinforcements for one player by splitting evenly among committed continents,
    and placing them using the existing greedy-cheapest allocator (continent-scoped).
    """
    split = split_reinforcements_even(
        int(reinf_total),
        committed_continents,
        state=state,
        owner=str(owner),
    )
    curr = state

    # For each continent with allocated reinf, we decide candidate nodes:
    # - nodes inside the continent owned by owner
    # - plus outside nodes committed to that continent (if owner owns them), to allow stacking on borders
    for cont, n in sorted(split.items()):
        n = int(n)
        if n <= 0:
            continue
        models = models_by_continent.get(cont)
        if models is None:
            continue

        full_graph = build_full_graph(cont)

        # Keep Board synced for battle_graph construction
        apply_global_state_to_board(curr, players)

        battle_graph = agop.build_continent_battle_graph(
            cont,
            players,
            debug=False,
            commitment_map=commitment_map,
        )

        macro = compute_macro_features_from_global_state(curr, battle_graph, full_graph)

        # Candidate nodes are continent nodes owned by owner.
        cont_nodes = [int(t._index) for t in Board.continent_territory_dict[cont]]
        candidate = [i for i in cont_nodes if curr.nodes[int(i)].owner == owner]

        # Apply a simple allocation: use ML module helper that adds troops directly.
        # (We keep reinforcement placement simple here; later you may call allocate_reinforcements_greedy_cheapest directly.)
        # For now we allocate all to the first candidate deterministically if any exist.
        if candidate:
            alloc = {int(candidate[0]): int(n)}
            curr = apply_reinforcement_allocation(curr, alloc)
        # else: no owned nodes in continent -> skip (can't place reinforcements there)

    return curr, split


def simulate_multi_turns_full_board_GT(
    *,
    initial_global_state: GlobalState,
    players: Sequence["Players.Player"],
    strategy_p1: Sequence[str],
    strategy_p2: Sequence[str],
    number_turns: int,
    models_dir: Path | str = "models",
) -> Tuple[GlobalState, List[GTStepDiag]]:
    """
    Simulate `number_turns` full turns (each turn = P1 acts then P2 acts), using GT commitment.

    Returns:
      (final_state, step_diags)
    """
    models_by_continent = load_models_by_continent(models_dir, strict=True)
    comm_policy = CommitmentPolicyGT()

    state = initial_global_state
    diags: List[GTStepDiag] = []

    strat1 = tuple(str(c) for c in strategy_p1)
    strat2 = tuple(str(c) for c in strategy_p2)

    for t in range(int(number_turns)):
        # -------------------------
        # P1 turn (A)
        # -------------------------
        apply_global_state_to_board(state, players)
        planA = comm_policy.build_commitment_plan(players=players, owner_obj=players[0], committed_continents=strat1)
        planD = comm_policy.build_commitment_plan(players=players, owner_obj=players[1], committed_continents=strat2)

        # reinforcements computed from FULL board state (Risk-like) using Board.continent_points_dict
        reinf_A, reinf_D, _ = compute_board_reinforcements_risklike(
            state,
            min_base=3,
            per_territories_div=3,
            continent_bonus=Board.continent_points_dict,
        )

        state_after_reinf_A, splitA = _alloc_reinforcements_by_commitment(
            state=state,
            players=players,
            models_by_continent=models_by_continent,
            owner="A",
            reinf_total=reinf_A,
            committed_continents=strat1,
            commitment_map=planA.commitment_by_node,
        )

        # Combat with commitment for attacker A (P1)
        state_after_combat = _apply_combat_all_continents(
            state=state_after_reinf_A,
            players=players,
            models_by_continent=models_by_continent,
            commitment=MLCommitment(commitment_by_node=planA.commitment_by_node),
            attacker_is_player1=True,
            turn=t * 2,
        )

        # Fortify (end of P1 turn)
        move = choose_end_of_turn_fortify_move(state_after_combat, owner="A", committed_continents=strat1)
        state_after_fort = apply_fortify_move(state_after_combat, move) if move else state_after_combat

        diags.append(
            GTStepDiag(
                turn=int(t * 2),
                attacker_owner="A",
                defender_owner="D",
                attacker_committed=strat1,
                defender_committed=strat2,
                attacker_reinf=int(reinf_A),
                defender_reinf=int(reinf_D),
                attacker_split=dict(splitA),
                defender_split={},
                attacker_commitment_nodes=int(len(planA.commitment_by_node)),
                defender_commitment_nodes=int(len(planD.commitment_by_node)),
            )
        )

        state = state_after_fort

        # -------------------------
        # P2 turn (D attacks; perspective swap inside combat)
        # -------------------------
        apply_global_state_to_board(state, players)
        # recompute plans (could differ after P1 turn)
        planA2 = comm_policy.build_commitment_plan(players=players, owner_obj=players[0], committed_continents=strat1)
        planD2 = comm_policy.build_commitment_plan(players=players, owner_obj=players[1], committed_continents=strat2)

        reinf_A2, reinf_D2, _ = compute_board_reinforcements_risklike(
            state,
            min_base=3,
            per_territories_div=3,
            continent_bonus=Board.continent_points_dict,
        )

        state_after_reinf_D, splitD = _alloc_reinforcements_by_commitment(
            state=state,
            players=players,
            models_by_continent=models_by_continent,
            owner="D",
            reinf_total=reinf_D2,
            committed_continents=strat2,
            commitment_map=planD2.commitment_by_node,
        )

        # Combat with commitment for attacker D (player2)
        state_after_combat2 = _apply_combat_all_continents(
            state=state_after_reinf_D,
            players=players,
            models_by_continent=models_by_continent,
            commitment=MLCommitment(commitment_by_node=planD2.commitment_by_node),
            attacker_is_player1=False,
            turn=t * 2 + 1,
        )

        # Fortify (end of P2 turn)
        move2 = choose_end_of_turn_fortify_move(state_after_combat2, owner="D", committed_continents=strat2)
        state_after_fort2 = apply_fortify_move(state_after_combat2, move2) if move2 else state_after_combat2

        diags.append(
            GTStepDiag(
                turn=int(t * 2 + 1),
                attacker_owner="D",
                defender_owner="A",
                attacker_committed=strat2,
                defender_committed=strat1,
                attacker_reinf=int(reinf_D2),
                defender_reinf=int(reinf_A2),
                attacker_split=dict(splitD),
                defender_split={},
                attacker_commitment_nodes=int(len(planD2.commitment_by_node)),
                defender_commitment_nodes=int(len(planA2.commitment_by_node)),
            )
        )

        state = state_after_fort2

    return state, diags
