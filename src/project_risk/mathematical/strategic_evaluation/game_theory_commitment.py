# game_theory_commitment.py
"""
Game-theoretic commitment evaluation layer.

This module produces the required "strategy table" outputs:

C.1) A payoff matrix (P1 strategies on x-axis, P2 strategies on y-axis) where each
     cell is (u1, u2).

C.2) A detailed dataframe with per-continent payoff breakdowns for each strategy pair.

It uses:
- simulate_multi_turns_full_board_GT (full board GT simulation)
- utility_terminal.utility_for_player (terminal utility evaluation)

Strategy space:
- Max 2 continents per player (per user)
- No empty strategy (we generate size 1 and size 2 combos)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState

from project_risk.mathematical.full_board_model.full_board_simulation_GT import simulate_multi_turns_full_board_GT
from project_risk.mathematical.strategic_evaluation.utility_terminal import UtilityWeights, utility_for_player


def enumerate_strategies(*, max_k: int = 2) -> List[Tuple[str, ...]]:
    conts = [str(c) for c in Board.continent_territory_dict.keys()]
    out: List[Tuple[str, ...]] = []
    for k in range(1, int(max_k) + 1):
        out.extend(tuple(cs) for cs in combinations(conts, k))
    return out


@dataclass(frozen=True)
class StrategyEvalConfig:
    number_turns: int = 3
    max_k: int = 2
    models_dir: Path | str = "models"
    weights: UtilityWeights = UtilityWeights()
    include_expected_vars: bool = False  # reserved: currently we only output payoff breakdowns + ownership deltas


def evaluate_strategy_pair(
    *,
    initial_state: GlobalState,
    players: Sequence["Players.Player"],
    s1: Sequence[str],
    s2: Sequence[str],
    config: StrategyEvalConfig,
) -> Dict[str, Any]:
    final_state, diags = simulate_multi_turns_full_board_GT(
        initial_global_state=initial_state,
        players=players,
        strategy_p1=s1,
        strategy_p2=s2,
        number_turns=int(config.number_turns),
        models_dir=config.models_dir,
    )

    u1, per1, diag1 = utility_for_player(
        owner="A",
        committed_continents=tuple(s1),
        start_state=initial_state,
        end_state=final_state,
        players_for_reinf=players,
        weights=config.weights,
        include_noncommitted=False,
    )
    u2, per2, diag2 = utility_for_player(
        owner="D",
        committed_continents=tuple(s2),
        start_state=initial_state,
        end_state=final_state,
        players_for_reinf=players,
        weights=config.weights,
        include_noncommitted=False,
    )

    # detailed rows
    rows = []
    for b in per1:
        rows.append(
            {
                "s1": tuple(s1),
                "s2": tuple(s2),
                "player": "A",
                "continent": b.continent,
                "owned_start": b.owned_start,
                "owned_end": b.owned_end,
                "new_territories": b.new_territories,
                "fully_owned_end": b.fully_owned_end,
                "continent_size": b.continent_size,
                "payoff_territory": b.payoff_territory,
                "payoff_continent": b.payoff_continent,
                "payoff_reinf": b.payoff_reinf,
                "payoff_total": b.payoff_total,
            }
        )
    for b in per2:
        rows.append(
            {
                "s1": tuple(s1),
                "s2": tuple(s2),
                "player": "D",
                "continent": b.continent,
                "owned_start": b.owned_start,
                "owned_end": b.owned_end,
                "new_territories": b.new_territories,
                "fully_owned_end": b.fully_owned_end,
                "continent_size": b.continent_size,
                "payoff_territory": b.payoff_territory,
                "payoff_continent": b.payoff_continent,
                "payoff_reinf": b.payoff_reinf,
                "payoff_total": b.payoff_total,
            }
        )

    return {
        "u1": float(u1),
        "u2": float(u2),
        "final_state": final_state,
        "turn_diags": diags,
        "payoff_rows": rows,
        "u1_diag": diag1,
        "u2_diag": diag2,
    }


def build_payoff_matrix_and_details(
    *,
    initial_state: GlobalState,
    players: Sequence["Players.Player"],
    config: Optional[StrategyEvalConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      payoff_matrix_df: rows=s2, cols=s1, values=(u1,u2)
      details_df: long-form per-continent breakdown rows for each (s1,s2,player,continent)
    """
    if config is None:
        config = StrategyEvalConfig()

    strategies = enumerate_strategies(max_k=int(config.max_k))

    # payoff matrix
    mat = pd.DataFrame(index=[str(s) for s in strategies], columns=[str(s) for s in strategies], dtype=object)

    detail_rows: List[Dict[str, Any]] = []

    for s2 in strategies:
        for s1 in strategies:
            out = evaluate_strategy_pair(
                initial_state=initial_state,
                players=players,
                s1=s1,
                s2=s2,
                config=config,
            )
            mat.loc[str(s2), str(s1)] = (out["u1"], out["u2"])
            detail_rows.extend(out["payoff_rows"])

    details_df = pd.DataFrame(detail_rows)
    return mat, details_df
