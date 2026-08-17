# utility_terminal.py
"""
Terminal utility functions for game-theoretic evaluation.

Implements the clarified payoff/utility:

- new_territories(continent) = owned_end - owned_start (per player, independent)
- new_continents(continent)  = 1 if fully owned at end else 0
- continent bonus term       = size_of_continent * weight_2 * new_continents
- troop reinforcement bonus  = weight_3 * (expected_reinf_next_turn - current_reinf_next_turn)
    (weight_3 is 0 for now per user)

Utility u_i(S, T) is computed at the END of the simulation horizon T, and is
the SUM of continent payoffs over the continents in player i's committed set.
Optionally, we can include non-committed continents (currently disabled by default).

Indexing note
-------------
Board territories are indexed 1..42. GlobalState may include dummy slot 0.
Always use Board.node_to_territory_dict.keys() / Board.continent_territory_dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from project_risk.game_simulation import Board
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState

from project_risk.mathematical.full_board_model.full_board_simulation_ML import compute_board_reinforcements_risklike


@dataclass(frozen=True)
class UtilityWeights:
    weight_1: float = 1.0   # territories
    weight_2: float = 0.5   # continent size multiplier
    weight_3: float = 0.0   # reinforcement delta (disabled for now)


@dataclass(frozen=True)
class ContinentPayoffBreakdown:
    continent: str
    owner: str  # "A" or "D"
    owned_start: int
    owned_end: int
    new_territories: int
    fully_owned_end: bool
    continent_size: int
    payoff_territory: float
    payoff_continent: float
    payoff_reinf: float
    payoff_total: float


def _continent_node_indices(continent: str) -> List[int]:
    return [int(t._index) for t in Board.continent_territory_dict[str(continent)]]


def owned_count_in_continent(state: GlobalState, continent: str, owner: str) -> int:
    nodes = _continent_node_indices(continent)
    return int(sum(1 for i in nodes if state.nodes[int(i)].owner == owner))


def fully_owned_continent(state: GlobalState, continent: str, owner: str) -> bool:
    nodes = _continent_node_indices(continent)
    return bool(nodes) and all(state.nodes[int(i)].owner == owner for i in nodes)


def continent_outcome_payoff(
    *,
    continent: str,
    owner: str,
    start_state: GlobalState,
    end_state: GlobalState,
    weights: UtilityWeights,
) -> ContinentPayoffBreakdown:
    nodes = _continent_node_indices(continent)
    size = int(len(nodes))

    owned_start = owned_count_in_continent(start_state, continent, owner)
    owned_end = owned_count_in_continent(end_state, continent, owner)

    new_terr = int(owned_end - owned_start)
    all_owned_end = bool(nodes) and (owned_end == size)

    # territory term (can be negative)
    payoff_terr = float(new_terr) * float(weights.weight_1)

    # continent bonus: size * weight_2 * 1{fully owned at end}
    payoff_cont = float(size) * float(weights.weight_2) * (1.0 if all_owned_end else 0.0)

    # reinf delta term (computed globally; disabled by default)
    payoff_reinf = 0.0

    return ContinentPayoffBreakdown(
        continent=str(continent),
        owner=str(owner),
        owned_start=int(owned_start),
        owned_end=int(owned_end),
        new_territories=int(new_terr),
        fully_owned_end=bool(all_owned_end),
        continent_size=int(size),
        payoff_territory=float(payoff_terr),
        payoff_continent=float(payoff_cont),
        payoff_reinf=float(payoff_reinf),
        payoff_total=float(payoff_terr + payoff_cont + payoff_reinf),
    )


def utility_for_player(
    *,
    owner: str,
    committed_continents: Sequence[str],
    start_state: GlobalState,
    end_state: GlobalState,
    players_for_reinf: Sequence[Any],
    weights: Optional[UtilityWeights] = None,
    include_noncommitted: bool = False,
) -> Tuple[float, List[ContinentPayoffBreakdown], Dict[str, Any]]:
    """
    Returns:
      (utility_value, per_continent_breakdowns, diag)

    diag includes board-level reinforcement expectation at end-of-horizon for both players.
    """
    if weights is None:
        weights = UtilityWeights()

    all_conts = list(Board.continent_territory_dict.keys())
    committed = [str(c) for c in committed_continents]
    if include_noncommitted:
        eval_conts = all_conts
    else:
        eval_conts = committed

    per_cont: List[ContinentPayoffBreakdown] = []
    total = 0.0
    for cont in eval_conts:
        b = continent_outcome_payoff(
            continent=cont,
            owner=owner,
            start_state=start_state,
            end_state=end_state,
            weights=weights,
        )
        per_cont.append(b)
        total += float(b.payoff_total)

    # end-of-horizon reinforcements (next turn), from *end_state*
    # Use true continent bonuses from Board.continent_points_dict (user requirement)
    A_reinf, D_reinf, reinf_diag = compute_board_reinforcements_risklike(
        end_state,
        min_base=3,
        per_territories_div=3,
        continent_bonus=Board.continent_points_dict,  # <-- required
    )

    diag: Dict[str, Any] = {
        "end_reinf_A": int(A_reinf),
        "end_reinf_D": int(D_reinf),
        "end_reinf_diag": reinf_diag,
    }

    # Optional reinforcement delta bonus (currently weight_3=0.0)
    if float(weights.weight_3) != 0.0:
        A0, D0, _ = compute_board_reinforcements_risklike(
            start_state,
            min_base=3,
            per_territories_div=3,
            continent_bonus=Board.continent_points_dict,
        )
        current = A0 if owner == "A" else D0
        expected = A_reinf if owner == "A" else D_reinf
        delta = float(expected - current)
        bonus = delta * float(weights.weight_3)
        total += bonus
        diag["reinf_delta_bonus"] = float(bonus)
        diag["reinf_start"] = int(current)
        diag["reinf_end"] = int(expected)

    return float(total), per_cont, diag
