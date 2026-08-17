# full_board_state_generators.py
"""
State generators for FULL-board simulations.

Why this module exists
----------------------
The training-time generator `ml_full_graph_state_generator` assigns ownership/troops only on a
continent-scoped "full_graph" (continent + neighbours). That is correct for training *continent*
models, but it does NOT create a coherent full-board state across all territories.

For full-board simulation tests, we need a generator that assigns ownership and troops across
the entire board (all territories), while keeping the same semantics:
  - target_territory_ratio controls attacker share of territories
  - target_troops_ratio controls attacker "available troops" relative to defender troops
    where available troops = sum(max(0, troops-1)) over attacker nodes
"""

from __future__ import annotations

from typing import Sequence, Tuple, Any

import numpy as np
import networkx as nx

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.game_simulation import SimulationEngine as SimEng


def build_world_graph() -> nx.Graph:
    """
    Full board graph: all territories and their adjacency edges.
    """
    G = nx.Graph()
    for terr in Board.all_territories_list:
        G.add_node(int(terr._index))
    for terr in Board.all_territories_list:
        a = int(terr._index)
        for nb in terr._neighbors:
            b = int(nb._index)
            if a != b:
                G.add_edge(a, b)
    return G


def full_board_state_generator(
    target_territory_ratio: float,
    target_troops_ratio: float,
    constraints: Any,
    rng: np.random.Generator,
) -> Tuple[Sequence["Players.Player"], Any, Any]:
    """
    Full-board state generator.

    Returns:
      (players, battle_graph_dummy, full_graph_world)

    Notes:
      - battle_graph_dummy is an empty graph; full-board simulation code should build
        continent battle graphs per turn using `agop.build_continent_battle_graph(...)`.
      - This generator is intended for *simulation initialization / testing*, not training
        continent models.
    """
    # Players
    p1 = Players.HighRiskAttacker(_name="Red")
    p2 = Players.LowRiskAttacker(_name="Blue")
    players = [p1, p2]

    # Reset entire board
    SimEng.reset_board_state()

    # World graph and node list
    full_graph = build_world_graph()
    full_nodes = list(full_graph.nodes())
    N = len(full_nodes)
    if N <= 0:
        raise ValueError("World graph has 0 nodes; cannot generate state.")

    # Clear ownership & troops for all nodes
    for idx in full_nodes:
        terr = Board.node_to_territory_dict[int(idx)]
        terr._owner = None
        terr._troops = 0

    # Choose attacker vs defender nodes (ensure at least 1 each)
    attacker_target = int(round(float(target_territory_ratio) * N))
    attacker_count = max(1, min(N - 1, attacker_target))
    defender_count = N - attacker_count

    attacker_indices = set(rng.choice(full_nodes, size=attacker_count, replace=False))
    defender_indices = [int(i) for i in full_nodes if int(i) not in attacker_indices]

    # Defender troops
    max_def = min(int(getattr(constraints, "max_defender_troops_per_node", 5)), 5)
    def_troops = rng.integers(1, max_def + 1, size=len(defender_indices))
    total_def_troops = int(def_troops.sum())

    # Attacker available troops target (same semantics as training generator)
    max_att = min(int(getattr(constraints, "max_attacker_troops_per_node", 5)), 5)
    max_avail_att = attacker_count * max(0, max_att - 1)

    if total_def_troops > 0:
        feasible_max_ratio = max_avail_att / total_def_troops
        clamped_ratio = max(0.0, min(float(target_troops_ratio), feasible_max_ratio))
        target_avail_att = int(round(clamped_ratio * total_def_troops))
        target_avail_att = max(0, min(target_avail_att, max_avail_att))
    else:
        target_avail_att = 0

    # Distribute attacker troops: start at 1 each, then distribute extras (available troops)
    att_troops = [1] * attacker_count
    extras = int(target_avail_att)

    # round-robin fill up to cap
    while extras > 0:
        any_inc = False
        for i in range(attacker_count):
            if extras <= 0:
                break
            if att_troops[i] < max_att:
                att_troops[i] += 1
                extras -= 1
                any_inc = True
        if not any_inc:
            break

    # Write defender to board
    for i, idx in enumerate(defender_indices):
        terr = Board.node_to_territory_dict[int(idx)]
        terr._owner = p2
        terr._troops = int(def_troops[i])

    # Write attacker to board
    attacker_list = [int(i) for i in attacker_indices]
    for i, idx in enumerate(attacker_list):
        terr = Board.node_to_territory_dict[int(idx)]
        terr._owner = p1
        terr._troops = int(att_troops[i])

    # Dummy battle graph (not used by full-board sim)
    battle_graph_dummy = nx.Graph()
    return players, battle_graph_dummy, full_graph
