# strategy_policy_gt.py
"""
Game-theoretic (GT) policies for full-board Risk simulations.

This module defines **strategy-conditional policies** used by the GT simulation layer:
- CommitmentPolicyGT: assigns cross-continent frontier nodes to committed continents.
- Reinforcement allocation: splits reinforcements across committed continents (even split for now).
- End-of-turn reallocation (fortify): one move per turn across friendly paths, respecting "leave 1 troop".

Important indexing note
-----------------------
This codebase uses classic Risk territory indices **1..42** (see Board.Territory._index).
Many GlobalState objects are built with a dummy slot at index 0, so you should NOT iterate
`enumerate(global_state.nodes)` and assume those are all board nodes.
Instead, use `Board.node_to_territory_dict.keys()` as the authoritative node set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from project_risk.game_simulation import Board

def _board_node_indices():
    return [int(i) for i in Board.node_to_territory_dict.keys()]

def _is_friendly(state, idx, owner_char):
    if idx < 0 or idx >= len(state.nodes):
        return False
    n = state.nodes[idx]
    return n.owner == owner_char and int(n.troops) > 0

from project_risk.game_simulation import Players

from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def all_board_node_indices() -> List[int]:
    """Authoritative node indices present on the Board (usually 1..42)."""
    return sorted(int(i) for i in Board.node_to_territory_dict.keys())


def continent_nodes(continent: str) -> List[int]:
    return [int(t._index) for t in Board.continent_territory_dict[str(continent)]]


def node_continent(node_idx: int) -> str:
    return str(Board.node_to_territory_dict[int(node_idx)]._continent)


def is_cross_continent_frontier_node(
    node_idx: int,
    *,
    owner_obj: Players.Player,
) -> bool:
    """
    A node is 'cross-continent frontier' if it is adjacent to at least one territory in a different continent.
    """
    terr = Board.node_to_territory_dict[int(node_idx)]
    if terr._owner is not owner_obj:
        return False
    for neigh in terr._neighbors:
        if neigh._continent != terr._continent:
            return True
    return False


def cross_continent_neighbor_continents(node_idx: int) -> Set[str]:
    """Set of continents (different from node's own) that this node touches."""
    terr = Board.node_to_territory_dict[int(node_idx)]
    out: Set[str] = set()
    for neigh in terr._neighbors:
        if neigh._continent != terr._continent:
            out.add(str(neigh._continent))
    return out


def eligible_to_be_committed(node_idx: int, *, owner_obj: Players.Player) -> bool:
    """
    User rule (clarification #6):
      - Must be adjacent to enemy nodes on the target continent
      - Must have > 1 troops
    """
    terr = Board.node_to_territory_dict[int(node_idx)]
    if terr._owner is not owner_obj:
        return False
    if int(getattr(terr, "_troops", 0)) <= 1:
        return False
    # enemy adjacency check is continent-specific; done in commitment assignment.
    return True


def enemy_neighbors_in_continent(
    node_idx: int,
    target_continent: str,
    *,
    owner_obj: Players.Player,
) -> int:
    terr = Board.node_to_territory_dict[int(node_idx)]
    cont_set = set(continent_nodes(target_continent))
    score = 0
    for neigh in terr._neighbors:
        if int(neigh._index) not in cont_set:
            continue
        if neigh._owner is None:
            continue
        if neigh._owner is not owner_obj:
            score += 1
    return int(score)


# ---------------------------------------------------------------------
# Commitment policy (GT)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CommitmentPlan:
    """
    commitment_by_node:
        Only includes *cross-continent* nodes that are committed to some continent.
        Nodes that are internal to a continent (not touching others) are implicitly committed to themselves.
    """
    commitment_by_node: Dict[int, str]

    def committed_continents(self) -> Set[str]:
        return set(self.commitment_by_node.values())


class CommitmentPolicyGT:
    """
    Implements user-specified commitment semantics (cap at 2 continents).

    Given a strategy (tuple of committed continents), for each player:
      - Nodes internal to a continent remain there implicitly (no map entry needed).
      - Cross-continent nodes are assigned to *one* committed continent if:
          a) node has >1 troop and is adjacent to enemy nodes on that target continent
          b) the player is committed to that target continent (priority rule)
      - If a node can serve multiple committed continents, distribute "as evenly as possible":
          we assign each such node to the currently least-filled committed continent among its eligible targets.
    """

    def build_commitment_plan(
        self,
        *,
        players: Sequence["Players.Player"],
        owner_obj: Players.Player,
        committed_continents: Sequence[str],
    ) -> CommitmentPlan:
        comm = [str(c) for c in committed_continents]
        if len(comm) == 0:
            # Spec says "no empty"; treat as committed to all, but in GT we won't generate empty anyway.
            comm = list(Board.continent_territory_dict.keys())

        # cap at 2 in the enumerator, but keep robust here too
        if len(comm) > 2:
            comm = comm[:2]

        # candidates: cross-continent nodes owned by player that could be committed somewhere
        candidates: List[int] = []
        for i in all_board_node_indices():
            if not is_cross_continent_frontier_node(i, owner_obj=owner_obj):
                continue
            if not eligible_to_be_committed(i, owner_obj=owner_obj):
                continue
            candidates.append(int(i))

        # For each node, compute eligible target continents among committed set
        eligible_targets: Dict[int, List[str]] = {}
        for i in candidates:
            # Node's own continent is handled implicitly unless player also "uses" it to attack another.
            # Here we only commit across borders to a target continent != node's own continent.
            touch = cross_continent_neighbor_continents(i)
            poss = [c for c in comm if c in touch and c != node_continent(i)]
            # Additionally require: adjacency to enemy in that target continent
            poss2 = [c for c in poss if enemy_neighbors_in_continent(i, c, owner_obj=owner_obj) > 0]
            if poss2:
                eligible_targets[int(i)] = poss2

        # Even distribution heuristic among committed continents, but only for nodes that have >=1 eligible target.
        fill = {c: 0 for c in comm}
        commitment_by_node: Dict[int, str] = {}

        # deterministic order: sort by (max enemy score among targets desc, node idx asc)
        def _node_priority(node_idx: int) -> Tuple[int, int]:
            scores = [enemy_neighbors_in_continent(node_idx, c, owner_obj=owner_obj) for c in eligible_targets[node_idx]]
            return (max(scores) if scores else 0, -node_idx)  # we will sort reverse

        ordered = sorted(eligible_targets.keys(), key=_node_priority, reverse=True)

        for node_idx in ordered:
            targets = eligible_targets[node_idx]
            # choose among targets the currently least-filled; tie-break by higher enemy score then name
            best = None
            best_key = None
            for c in targets:
                key = (
                    -fill.get(c, 0),  # prefer least filled (i.e., larger negative)
                    enemy_neighbors_in_continent(node_idx, c, owner_obj=owner_obj),
                    c,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = c
            if best is not None:
                commitment_by_node[int(node_idx)] = str(best)
                fill[str(best)] = int(fill.get(str(best), 0) + 1)

        return CommitmentPlan(commitment_by_node=commitment_by_node)


# ---------------------------------------------------------------------
# Reinforcement allocation policy (GT)
# ---------------------------------------------------------------------

def split_reinforcements_even(
    total: int,
    committed_continents: Sequence[str],
    *,
    state: GlobalState,
    owner: str,
) -> Dict[str, int]:
    """
    Even split among committed continents where the owner has at least one territory at start of turn.
    Deterministic remainder assignment by sorted continent name.
    """
    total = int(total)
    comm = [str(c) for c in committed_continents]
    if total <= 0:
        return {c: 0 for c in comm}

    eligible = []
    for c in comm:
        nodes = continent_nodes(c)
        if any(state.nodes[int(i)].owner == owner for i in nodes):
            eligible.append(c)

    if not eligible:
        return {c: 0 for c in comm}

    eligible = sorted(set(eligible))
    base = total // len(eligible)
    rem = total - base * len(eligible)

    out = {c: 0 for c in comm}
    for k, c in enumerate(eligible):
        out[c] = base + (1 if k < rem else 0)
    return out


# ---------------------------------------------------------------------
# End-of-turn reallocation (fortify) policy
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FortifyMove:
    """Move troops from src -> dst at end of a turn (must be friendly-connected)."""
    src: int
    dst: int
    troops: int  # troops moved (must leave >=1 behind)


def choose_end_of_turn_fortify_move(
    state: GlobalState,
    *,
    owner: str,
    committed_continents: Sequence[str],
) -> Optional[FortifyMove]:
    """
    Very simple policy (first pass):
      - Identify a destination "front" node inside committed continents that touches an enemy inside its continent.
      - Identify a source node in the same friendly component (board-wide) with surplus troops.
      - Move as many as possible (leave 1).
    If no such move exists, return None.

    NOTE: This is intentionally conservative. You can later replace this with a utility-driven move search.
    """
    comm = [str(c) for c in committed_continents]
    if not comm:
        return None

    # Build world adjacency and BFS components for the owner
    owner_nodes = [i for i in all_board_node_indices() if state.nodes[int(i)].owner == owner]
    if len(owner_nodes) < 2:
        return None

    # adjacency restricted to owner nodes (friendly path)
    adj: Dict[int, List[int]] = {i: [] for i in owner_nodes}
    for i in owner_nodes:
        terr = Board.node_to_territory_dict[int(i)]
        for neigh in terr._neighbors:
            j = int(neigh._index)
            if j in adj:
                adj[i].append(j)

    # compute components via DFS
    seen: Set[int] = set()
    comps: List[Set[int]] = []
    for i in owner_nodes:
        if i in seen:
            continue
        stack = [i]
        comp = set()
        seen.add(i)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)

    # helper: frontier-in-continent
    def _is_front_in_cont(i: int, cont: str) -> bool:
        cont_set = set(continent_nodes(cont))
        terr = Board.node_to_territory_dict[int(i)]
        if int(i) not in cont_set:
            return False
        for neigh in terr._neighbors:
            if int(neigh._index) in cont_set and state.nodes[int(neigh._index)].owner != owner:
                return True
        return False

    # destinations: committed continents only, nodes owned by owner, frontier inside that continent
    dests: List[int] = []
    for cont in comm:
        for i in continent_nodes(cont):
            if state.nodes[int(i)].owner != owner:
                continue
            if _is_front_in_cont(int(i), cont):
                dests.append(int(i))

    if not dests:
        return None

    # deterministic destination preference: smallest troops (needs help), tie by node id
    dests = sorted(dests, key=lambda i: (int(state.nodes[int(i)].troops), int(i)))
    dst = dests[0]

    # Find a source in same component with surplus troops, prefer non-front nodes
    comp = None
    for cset in comps:
        if dst in cset:
            comp = cset
            break
    if not comp or len(comp) < 2:
        return None

    # candidate sources
    def _is_any_front(i: int) -> bool:
        # front relative to ANY continent it's in (its own continent only)
        cont = node_continent(i)
        return _is_front_in_cont(i, cont)

    sources = []
    for i in comp:
        if i == dst:
            continue
        troops = int(state.nodes[int(i)].troops)
        if troops <= 1:
            continue
        sources.append(i)

    if not sources:
        return None

    # prefer non-front (safer), and higher troops
    sources = sorted(sources, key=lambda i: (_is_any_front(i), -int(state.nodes[int(i)].troops), int(i)))
    src = sources[0]
    move = int(state.nodes[int(src)].troops) - 1
    if move <= 0:
        return None
    return FortifyMove(src=int(src), dst=int(dst), troops=int(move))


def apply_fortify_move(state: GlobalState, move: FortifyMove) -> GlobalState:
    """Apply a FortifyMove to a GlobalState (no legality checks here)."""
    src = int(move.src)
    dst = int(move.dst)
    t = int(move.troops)
    if t <= 0:
        return state
    new_nodes = list(state.nodes)
    src_node = new_nodes[src]
    dst_node = new_nodes[dst]
    # leave at least 1
    movable = max(0, int(src_node.troops) - 1)
    t = min(t, movable)
    if t <= 0:
        return state
    new_nodes[src] = type(src_node)(owner=src_node.owner, troops=int(src_node.troops) - t)
    new_nodes[dst] = type(dst_node)(owner=dst_node.owner, troops=int(dst_node.troops) + t)
    return GlobalState(nodes=tuple(new_nodes))
