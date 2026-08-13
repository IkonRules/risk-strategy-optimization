from __future__ import annotations

from itertools import combinations, product, permutations
from dataclasses import dataclass
import pandas as pd
import pickle
from typing import Sequence, Dict, Tuple, List, Any, Optional, Iterable
from functools import lru_cache

from .markov_matrix_probabilities import battle_summary
from pathlib import Path
from collections import Counter
import uuid
import hashlib


import numpy as np

BASE_LIB_DIR = Path("small_graph_libraries")
CANON_DIR = BASE_LIB_DIR / "canonical_topologies"

# ---------------------------------------------------------------------
# OPERATING RULES (conceptual model)
# ---------------------------------------------------------------------
# - An end state in which the attacker has lost all troops is absorbing.
# - If the attacker has more than one troop left on at least one node,
#   he may continue attacking adjacent defender nodes.
# - If there are no defender nodes left, this is an absorbing state
#   (attacker has conquered the region).
# - If no attacker-owned node has more than one troop, this is an absorbing state
#   (attacker cannot attack any further).
# - Each time the player moves on to attack another node he must leave one troop
#   behind on the node he attacked from.
# - After conquering a node:
#     * If there are other enemy nodes adjacent to the origin node, the attacker
#       may either:
#           - move exactly 1 troop to the conquered node (keep stack back), or
#           - move all but 1 troop to the conquered node (push stack).
#     * If there are no other enemy neighbours of the origin node, the attacker
#       must move all but 1 troop to the conquered node.
# - The *actual implementation* assumes a rational attacker who, at every state,
#   chooses actions (including movement mode) to maximize expected utility:
#       U(success) = sum of attacker troops on originally defender-owned nodes
#       U(failure) = 0
#   The resulting probabilities of absorbing states are induced by this policy.
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Graph generation: all connected simple graphs on num_nodes nodes
# ---------------------------------------------------------------------



@dataclass
class PlateauPolicyOption:
    """
    One option-aware plateau policy family.

    action_groups is an ordered list of equally acceptable action groups.
    The chooser scans groups in order and picks the first legal action in the
    first group with at least one legal action.
    """
    option_id: int
    action_groups: List[List[Tuple[int, int]]]
    representative_root_action: Optional[Tuple[int, int]] = None
    support: float = 0.0
    support_rows: int = 0
    total_rows_considered: int = 0
    value_signature: Optional[Tuple[float, ...]] = None
    outcome_signature: Any = None


@dataclass
class PlateauPolicy:
    """
    Approximate high-troop policy for a given small graph (edges_key).

    Backward-compatible single-policy use:
      - edges_order is the old ordered list of preferred edges.

    Option-aware use:
      - action_groups can represent ties among equally acceptable actions.
      - options can represent multiple stable plateau policy families.
    """
    edges_order: List[Tuple[int, int]]
    action_groups: Optional[List[List[Tuple[int, int]]]] = None
    options: Optional[List[PlateauPolicyOption]] = None
    diagnostics: Optional[Dict[str, Any]] = None


def global_state_from_row_label(lbl: str) -> GlobalState:
    """
    Inverse of encode_state_label.

    lbl format: "(A3,D2,D1)"  -> GlobalState(nodes=(NodeState('A',3), NodeState('D',2), ...))

    Robust to whitespace. Allows 0 troops (e.g., "A0") if present in stored rows.
    """
    if not isinstance(lbl, str):
        raise TypeError(f"row label must be a str, got {type(lbl)}")

    s = lbl.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    if not s:
        return GlobalState(nodes=tuple())

    parts = [p.strip() for p in s.split(",") if p.strip()]
    nodes = []
    for p in parts:
        owner = p[0].upper()
        if owner not in ("A", "D"):
            raise ValueError(f"Invalid owner prefix in state label part {p!r} from lbl={lbl!r}")
        try:
            troops = int(p[1:])
        except Exception as e:
            raise ValueError(f"Invalid troop count in state label part {p!r} from lbl={lbl!r}") from e

        nodes.append(NodeState(owner=owner, troops=troops))

    return GlobalState(nodes=tuple(nodes))


def generate_connected_graphs_n_nodes(num_nodes: int):
    """
    Generate all connected simple graphs on nodes 0..num_nodes-1.
    Returns a list of edge-sets; each edge-set is a set of (u, v) with u < v.
    """
    nodes = list(range(num_nodes))
    all_possible_edges = [(i, j) for i in nodes for j in nodes if i < j]
    graphs = []

    def is_connected(edges):
        if not edges:
            return False
        adj = {n: set() for n in nodes}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        visited = set()
        stack = [0]  # start from node 0 (arbitrary)
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for w in adj[u]:
                if w not in visited:
                    stack.append(w)
        return len(visited) == len(nodes)

    for r in range(1, len(all_possible_edges) + 1):
        for combo in combinations(all_possible_edges, r):
            if is_connected(combo):
                graphs.append(set(combo))

    return graphs


# ---------------------------------------------------------------------
# Canonicalization: collapse isomorphic graphs (A/D partition preserved)
# ---------------------------------------------------------------------


def canonicalize_edges_with_roles(
    edges,
    num_attacker_nodes: int,
    num_defender_nodes: int,
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute a canonical representative for a small graph topology with an
    attacker/defender partition.

    We only consider permutations that:
      - permute attacker nodes among themselves
      - permute defender nodes among themselves

    Parameters
    ----------
    edges : iterable of (u, v)
        Edges on node indices 0..(nA+nD-1).
    num_attacker_nodes : int
        Number of attacker-owned nodes (indices 0..nA-1).
    num_defender_nodes : int
        Number of defender-owned nodes (indices nA..nA+nD-1).

    Returns
    -------
    canonical_edges_key : tuple[(u, v), ...]
        Sorted tuple of edges (u < v) representing the canonical form.
    perm_old_to_new : tuple[int, ...]
        Permutation mapping old_index -> canonical_index.
    perm_new_to_old : tuple[int, ...]
        Inverse permutation mapping canonical_index -> old_index.
    """
    total_nodes = num_attacker_nodes + num_defender_nodes
    edges = list(edges)

    attacker_indices = list(range(num_attacker_nodes))
    defender_indices = list(range(num_attacker_nodes, total_nodes))

    best_edges_key: Tuple[Tuple[int, int], ...] | None = None
    best_perm_old_to_new: Tuple[int, ...] | None = None

    for att_perm in permutations(attacker_indices):
        for def_perm in permutations(defender_indices):
            # new_index -> old_index
            new_order = list(att_perm) + list(def_perm)

            # old_index -> new_index
            old_to_new = {old: new for new, old in enumerate(new_order)}

            remapped_edges: List[Tuple[int, int]] = []
            for (u, v) in edges:
                nu = old_to_new[u]
                nv = old_to_new[v]
                if nu < nv:
                    remapped_edges.append((nu, nv))
                else:
                    remapped_edges.append((nv, nu))
            remapped_edges.sort()
            key = tuple(remapped_edges)

            if best_edges_key is None or key < best_edges_key:
                best_edges_key = key
                # store permutation as tuple for hashability
                best_perm_old_to_new = tuple(old_to_new[i] for i in range(total_nodes))

    assert best_edges_key is not None and best_perm_old_to_new is not None

    # Invert perm: canonical_index -> old_index
    perm_new_to_old = [None] * total_nodes
    for old_idx, new_idx in enumerate(best_perm_old_to_new):
        perm_new_to_old[new_idx] = old_idx
    perm_new_to_old = tuple(perm_new_to_old)

    return best_edges_key, best_perm_old_to_new, perm_new_to_old



# ---------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class NodeState:
    owner: str   # 'A' or 'D'
    troops: int  # number of troops on this node


@dataclass(frozen=True)
class GlobalState:
    """
    Immutable representation of a whole configuration of nodes.

    nodes[i] is the NodeState of node i (0, 1, 2, ...).
    """
    nodes: tuple  # tuple[NodeState, ...]


def initial_state_generic(
    attacker_troops: Sequence[int],
    defender_troops: Sequence[int],
) -> GlobalState:
    """
    Create an initial GlobalState for:
      - nodes 0..(nA-1)            : attacker-owned, given troops
      - nodes nA..(nA+nD-1)        : defender-owned, given troops
    """
    nodes = []
    for t in attacker_troops:
        nodes.append(NodeState('A', t))
    for t in defender_troops:
        nodes.append(NodeState('D', t))
    return GlobalState(nodes=tuple(nodes))


# ---------------------------------------------------------------------
# Combat lookup from the 2-node outcome table
# ---------------------------------------------------------------------


def get_combat_outcomes(combat_df: pd.DataFrame, a_init: int, d_init: int):
    """
    Given initial attacker/defender troops (a_init, d_init),
    return list of ((a_end, d_end), probability) from the combat_df.

    combat_df is F_df from battle_summary(A_max, D_max), with:
      - index  : strings "(a,d)" for transient states a>=1,d>=1
      - columns: strings "(0,d)" and "(a,0)" for absorbing states
    """
    row_label = f"({a_init},{d_init})"

    try:
        row = combat_df.loc[row_label]
    except KeyError as e:
        raise KeyError(
            f"No row for initial state {row_label} in combat_df.\n"
            f"Example index entries: {list(combat_df.index)[:10]}"
        ) from e

    outcomes = []
    for col_label, p in row.items():
        if p <= 0:
            continue

        # col_label: "(0,1)", "(3,0)", etc.
        label = col_label.strip("()")
        a_end_str, d_end_str = label.split(",")
        a_end = int(a_end_str)
        d_end = int(d_end_str)

        outcomes.append(((a_end, d_end), float(p)))

    return outcomes


# ---------------------------------------------------------------------
# Utility functions on GlobalState
# ---------------------------------------------------------------------


def adjacency_dict(edges):
    """
    Build adjacency dict from an edge set.

    Parameters
    ----------
    edges : iterable of (u, v) with u < v

    Returns
    -------
    dict[int, set[int]]
        Adjacency list representation.
    """
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    adj = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def possible_actions(state: GlobalState, edges):
    """
    Possible actions (u, v) where:
      - u is an attacker node with troops > 1,
      - v is an adjacent defender node.
    Works for any number of nodes.
    """
    adj = adjacency_dict(edges)
    actions = []
    num_nodes = len(state.nodes)

    for u in range(num_nodes):
        node_u = state.nodes[u]
        if node_u.owner == 'A' and node_u.troops > 1:
            for v in adj.get(u, []):
                node_v = state.nodes[v]
                if node_v.owner == 'D':
                    actions.append((u, v))
    return actions


def is_absorbing(state: GlobalState, edges):
    """
    A state is absorbing if:
    - attacker has no troops on any node, OR
    - there are no defender nodes, OR
    - there are no possible actions.
    """
    any_attacker = any(n.owner == 'A' and n.troops > 0 for n in state.nodes)
    any_defender = any(n.owner == 'D' and n.troops > 0 for n in state.nodes)

    if not any_attacker:
        return True
    if not any_defender:
        return True
    if not possible_actions(state, edges):
        return True
    return False


def is_successful(state: GlobalState) -> bool:
    """
    'Successful' absorbing state:
    - All nodes are owned by the attacker (no defenders remain).
    - And there is at least one attacker troop somewhere.
    """
    all_attacker = all(n.owner == 'A' for n in state.nodes)
    any_troops = any(n.troops > 0 for n in state.nodes)
    return all_attacker and any_troops


# ---------------------------------------------------------------------
# Single combat step and post-combat troop movement (legacy helper)
# ---------------------------------------------------------------------


def state_utility(state: GlobalState, num_attacker_nodes: int) -> Tuple[float, float]:
    """
    Lexicographic utility of an absorbing state for the attacker.

    Returns a pair (p_success, troop_score):

    - If the state is *not* successful:
        p_success = 0.0
        troop_score = 0.0   (irrelevant because failure)
    - If successful:
        p_success = 1.0
        troop_score = sum of attacker troops on originally defender-owned nodes.(HMM DOESENT SEEM RGHT)

    num_attacker_nodes:
        The first num_attacker_nodes indices were originally attacker nodes.
        The rest were originally defender nodes.
    """
    if not is_successful(state):
        return (0.0, 0.0)

    total_nodes = len(state.nodes)
    troop_score = 0
    for i in range(num_attacker_nodes, total_nodes):
        node = state.nodes[i]
        troop_score += node.troops
    return (1.0, float(troop_score))


def better_value(v1: Tuple[float, float], v2: Tuple[float, float], tol: float = 1e-9) -> bool:
    """
    Return True if v1 is strictly better than v2 under lexicographic order:

        1) larger success probability is always better
        2) if success probabilities are (approximately) equal,
           larger troop_score is better
    """
    p1, t1 = v1
    p2, t2 = v2

    if p1 > p2 + tol:
        return True
    if p2 > p1 + tol:
        return False

    # success probs are equal within tolerance -> compare troop scores
    return t1 > t2 + tol




def local_state_value(
    state: GlobalState,
    num_attacker_nodes: int,
    *,
    include_no_gain: bool = False,
) -> Tuple[float, ...]:
    """
    Context-independent terminal value for one absorbing small-graph state.

    Default tuple, larger is better:
      1) new_territories
      2) final_attacker_troops
      3) local_conquest

    If include_no_gain=True, tuple becomes:
      1) new_territories
      2) -no_gain
      3) final_attacker_troops
      4) local_conquest

    The negative no_gain component is used because lower P(no gain) is better,
    but the generic tuple comparison assumes larger is better.
    """
    new_territories = 0
    final_attacker_troops = 0

    for i, node in enumerate(state.nodes):
        if node.owner == "A" and node.troops > 0:
            final_attacker_troops += int(node.troops)
            if i >= num_attacker_nodes:
                new_territories += 1

    local_conquest = 1.0 if is_successful(state) else 0.0

    if include_no_gain:
        no_gain = 1.0 if new_territories == 0 else 0.0
        return (
            float(new_territories),
            -float(no_gain),
            float(final_attacker_troops),
            float(local_conquest),
        )

    return (
        float(new_territories),
        float(final_attacker_troops),
        float(local_conquest),
    )


def default_local_value_tolerances(include_no_gain: bool = False) -> Tuple[float, ...]:
    """Default numerical tolerances matching the active local value tuple."""
    return (1e-9, 1e-9, 1e-9, 1e-9) if include_no_gain else (1e-9, 1e-9, 1e-9)


def value_relation(
    v1: Tuple[float, ...],
    v2: Tuple[float, ...],
    tolerances: Optional[Tuple[float, ...]] = None,
) -> int:
    """
    Lexicographic comparison for arbitrary value tuples.

    Returns:
       1 if v1 is strictly better than v2,
       0 if equivalent within tolerances,
      -1 if v2 is strictly better than v1.
    """
    if len(v1) != len(v2):
        raise ValueError(f"Cannot compare value tuples of different lengths: {len(v1)} != {len(v2)}")

    if tolerances is None:
        tolerances = tuple(1e-9 for _ in v1)
    if len(tolerances) != len(v1):
        raise ValueError(f"tolerances length {len(tolerances)} does not match value length {len(v1)}")

    for x, y, tol in zip(v1, v2, tolerances):
        if float(x) > float(y) + float(tol):
            return 1
        if float(y) > float(x) + float(tol):
            return -1
    return 0


def better_value_tuple(
    v1: Tuple[float, ...],
    v2: Tuple[float, ...],
    tolerances: Optional[Tuple[float, ...]] = None,
) -> bool:
    """Return True iff v1 is strictly better than v2 under value_relation()."""
    return value_relation(v1, v2, tolerances) == 1


def _add_scaled_value(
    acc: Tuple[float, ...],
    scale: float,
    value: Tuple[float, ...],
) -> Tuple[float, ...]:
    return tuple(float(a) + float(scale) * float(b) for a, b in zip(acc, value))


def _normalize_dist(dist: Dict[GlobalState, float]) -> Dict[GlobalState, float]:
    total = float(sum(float(p) for p in dist.values()))
    if total <= 0.0:
        return dict(dist)
    return {s: float(p) / total for s, p in dist.items() if float(p) > 0.0}


def _dist_signature(dist: Dict[GlobalState, float], *, ndigits: int = 12) -> Tuple[Tuple[str, float], ...]:
    """
    Stable signature for a local absorbing distribution.

    This is deliberately based on final local node identities, owners, troops and
    probabilities. If two policies produce the same distribution over the same
    local nodes, the larger graph receives the same final-state information and
    we only need one representative.
    """
    return tuple(
        sorted(
            (encode_state_label(s), round(float(p), ndigits))
            for s, p in _normalize_dist(dist).items()
            if float(p) > 0.0
        )
    )


@dataclass(frozen=True)
class PolicyOption:
    """One saved policy option for a small-graph row."""
    option_id: int
    root_action: Optional[Tuple[int, int]]
    value: Tuple[float, ...]
    absorbing_dist: Dict[GlobalState, float]


@dataclass(frozen=True)
class _StateOption:
    """
    Internal continuation option used by the bottom-up/state-set solver.

    split_depth_used counts how many meaningful decision levels have been allowed
    to preserve multiple alternatives along this continuation.
    """
    value: Tuple[float, ...]
    absorbing_dist: Dict[GlobalState, float]
    root_action: Optional[Tuple[int, int]] = None
    branch_key: Any = None
    split_depth_used: int = 0


def _cap_list(items: List[Any], cap: Optional[int]) -> List[Any]:
    """Apply an optional positive cap. None means no cap."""
    if cap is None:
        return list(items)
    cap_i = max(1, int(cap))
    return list(items)[:cap_i]


def _prune_state_options(
    candidates: List[_StateOption],
    *,
    value_tolerances: Optional[Tuple[float, ...]],
    max_options: Optional[int],
    max_split_depth: Optional[int],
    meaningful_decision: bool,
) -> List[_StateOption]:
    """
    Keep only locally near-best, distinct continuation distributions.

    Pruning rule:
      1. Find best local utility.
      2. Keep candidates equivalent to best within tolerance.
      3. If multiple branch keys are retained at this state, count this as one
         preserved split level.
      4. Drop candidates exceeding max_split_depth, if a depth cap is used.
      5. Deduplicate identical final distributions.
      6. Apply width cap max_options, if used.
    """
    if not candidates:
        return []

    best_value = candidates[0].value
    for opt in candidates[1:]:
        if better_value_tuple(opt.value, best_value, value_tolerances):
            best_value = opt.value

    near_best = [
        opt for opt in candidates
        if value_relation(opt.value, best_value, value_tolerances) == 0
    ]

    if not near_best:
        near_best = [max(candidates, key=lambda o: o.value)]

    # If this state itself preserves alternatives from different branches, that
    # consumes one split level. Forced states, or states where all retained options
    # came from the same branch, do not consume split depth.
    if meaningful_decision:
        branch_keys = {opt.branch_key for opt in near_best}
        if len(branch_keys) > 1:
            near_best = [
                _StateOption(
                    value=opt.value,
                    absorbing_dist=opt.absorbing_dist,
                    root_action=opt.root_action,
                    branch_key=opt.branch_key,
                    split_depth_used=int(opt.split_depth_used) + 1,
                )
                for opt in near_best
            ]

    if max_split_depth is not None:
        max_sd = int(max_split_depth)
        filtered = [opt for opt in near_best if int(opt.split_depth_used) <= max_sd]
        if filtered:
            near_best = filtered
        else:
            # Fallback: do not return an empty option set. Keep the least-depth,
            # best-valued candidate as a conservative single representative.
            near_best = sorted(
                near_best,
                key=lambda opt: (int(opt.split_depth_used), opt.value),
                reverse=False,
            )[:1]

    # Deduplicate by final local distribution. This keeps distinct policies only
    # when they give different final-state distributions.
    seen = set()
    distinct: List[_StateOption] = []
    for opt in near_best:
        sig = _dist_signature(opt.absorbing_dist)
        if sig in seen:
            continue
        seen.add(sig)
        distinct.append(opt)

    # Stable deterministic ordering: better value first, then smaller split depth,
    # then branch key string representation.
    distinct.sort(
        key=lambda opt: (opt.value, -int(opt.split_depth_used), repr(opt.branch_key)),
        reverse=True,
    )

    return _cap_list(distinct, max_options)


def _combine_weighted_child_option_sets(
    weighted_sets: List[Tuple[float, List[_StateOption]]],
    *,
    value_tolerances: Optional[Tuple[float, ...]],
    max_options: Optional[int],
    max_split_depth: Optional[int],
) -> List[_StateOption]:
    """
    Combine independent policy choices available after different stochastic
    combat outcomes of the same action.

    Each combat outcome happens with fixed probability; after the outcome is
    observed, the policy may choose one continuation option in that child state.
    Therefore an action-level option is a Cartesian product of one child option
    from each combat outcome branch.
    """
    if not weighted_sets:
        return []

    # Remove zero-probability branches and empty child sets.
    clean: List[Tuple[float, List[_StateOption]]] = []
    for p, opts in weighted_sets:
        if float(p) <= 0.0:
            continue
        if not opts:
            continue
        clean.append((float(p), list(opts)))

    if not clean:
        return []

    # Incrementally combine and prune to avoid uncontrolled growth.
    partials = [
        _StateOption(
            value=tuple(0.0 for _ in clean[0][1][0].value),
            absorbing_dist={},
            root_action=None,
            branch_key=None,
            split_depth_used=0,
        )
    ]

    for p_branch, opts in clean:
        new_partials: List[_StateOption] = []
        for partial in partials:
            for opt in opts:
                value = _add_scaled_value(partial.value, p_branch, opt.value)

                dist = dict(partial.absorbing_dist)
                for s, ps in opt.absorbing_dist.items():
                    dist[s] = dist.get(s, 0.0) + p_branch * float(ps)

                root_action = partial.root_action if partial.root_action is not None else opt.root_action
                split_used = max(int(partial.split_depth_used), int(opt.split_depth_used))

                new_partials.append(
                    _StateOption(
                        value=value,
                        absorbing_dist=dist,
                        root_action=root_action,
                        branch_key=partial.branch_key,
                        split_depth_used=split_used,
                    )
                )

        partials = _prune_state_options(
            new_partials,
            value_tolerances=value_tolerances,
            max_options=max_options,
            max_split_depth=max_split_depth,
            meaningful_decision=False,
        )

    return partials


# ---------------------------------------------------------------------
# Recursive exploration of optimal policy and absorbing states
# ---------------------------------------------------------------------


def explore_absorbing_states_for_graph(
    edges,
    combat_df: pd.DataFrame,
    start_state: GlobalState,
    num_attacker_nodes: int,
):
    """
    Explore the game tree for a fixed graph under a *rational* attacker
    with lexicographic preferences:

      1) maximize probability of eventual success (conquer all nodes),
      2) among policies with equal success probability, maximize
         expected surviving troops on originally defender-owned nodes.

    Returns
    -------
    absorbing_dist : dict[GlobalState, float]
        Probability distribution over absorbing states under the optimal policy.

    best_value : Tuple[float, float]
        (P_success, E[troop_score]), where troop_score is sum of attacker troops
        on originally defender-owned nodes in successful absorbing states
        (0 in failure states).

    policy : dict[GlobalState, Tuple[int, int] or None]
        For each non-absorbing state, the chosen attack (u, v).
        For absorbing states, the value is None.
    """
    adj = adjacency_dict(edges)
    policy: Dict[GlobalState, Tuple[int, int] | None] = {}

    @lru_cache(maxsize=None)
    def eval_state(state: GlobalState):
        # Base case: absorbing -> point-mass distribution, with its lexicographic value
        if is_absorbing(state, edges):
            value = state_utility(state, num_attacker_nodes)  # (p_success, troop_score)
            dist = {state: 1.0}
            policy[state] = None
            return value, dist

        actions = possible_actions(state, edges)
        if not actions:
            # Safety: should be caught by is_absorbing, but just in case
            value = state_utility(state, num_attacker_nodes)
            dist = {state: 1.0}
            policy[state] = None
            return value, dist

        # best_value is a pair (P_success, E[troop_score])
        best_value: Tuple[float, float] = (-1.0, 0.0)  # impossible baseline
        best_dist: Dict[GlobalState, float] | None = None
        best_action: Tuple[int, int] | None = None

        for (u, v) in actions:
            node_u = state.nodes[u]
            node_v = state.nodes[v]

            T_u = node_u.troops
            T_v = node_v.troops
            assert T_u > 1, f"Cannot attack from node {u} with T_u={T_u}"

            a_init = T_u - 1
            d_init = T_v

            combat_outcomes = get_combat_outcomes(combat_df, a_init, d_init)

            # Expected value (pair) and absorbing distribution for this particular attack
            attack_value = (0.0, 0.0)  # (P_success, E[troop_score])
            attack_dist: Dict[GlobalState, float] = {}

            for (a_avail_end, d_end), p_outcome in combat_outcomes:
                if p_outcome <= 0.0:
                    continue

                origin_after = 1 + a_avail_end
                base_nodes = list(state.nodes)

                if d_end > 0:
                    # Defender holds v, no conquest
                    base_nodes[u] = NodeState('A', origin_after)
                    base_nodes[v] = NodeState('D', d_end)
                    next_state = GlobalState(nodes=tuple(base_nodes))

                    v_next, dist_next = eval_state(next_state)  # (p_succ_next, troop_next)

                    # add p_outcome * v_next to attack_value
                    attack_value = (
                        attack_value[0] + p_outcome * v_next[0],
                        attack_value[1] + p_outcome * v_next[1],
                    )

                    for s, ps in dist_next.items():
                        attack_dist[s] = attack_dist.get(s, 0.0) + p_outcome * ps

                else:
                    # Defender eliminated on v -> attacker chooses troop movement
                    total_at_u_before_move = origin_after

                    other_enemy_neighbors = any(
                        (w != v) and (state.nodes[w].owner == 'D')
                        for w in adj[u]
                    )

                    if other_enemy_neighbors:
                        # Two movement choices:

                        # Option 1: keep stack back (move exactly 1 to v)
                        nodes1 = list(base_nodes)
                        nodes1[u] = NodeState('A', total_at_u_before_move - 1)
                        nodes1[v] = NodeState('A', 1)
                        state1 = GlobalState(nodes=tuple(nodes1))
                        v1, dist1 = eval_state(state1)

                        # Option 2: push stack forward (move all but 1 to v)
                        nodes2 = list(base_nodes)
                        nodes2[u] = NodeState('A', 1)
                        nodes2[v] = NodeState('A', total_at_u_before_move - 1)
                        state2 = GlobalState(nodes=tuple(nodes2))
                        v2, dist2 = eval_state(state2)

                        # Choose movement with higher lexicographic value
                        if better_value(v1, v2):
                            chosen_value = v1
                            chosen_dist = dist1
                        else:
                            chosen_value = v2
                            chosen_dist = dist2
                    else:
                        # No other enemy neighbours: forced move all but one to v
                        nodes3 = list(base_nodes)
                        nodes3[u] = NodeState('A', 1)
                        nodes3[v] = NodeState('A', total_at_u_before_move - 1)
                        state3 = GlobalState(nodes=tuple(nodes3))
                        chosen_value, chosen_dist = eval_state(state3)

                    # Incorporate chosen branch for this outcome
                    attack_value = (
                        attack_value[0] + p_outcome * chosen_value[0],
                        attack_value[1] + p_outcome * chosen_value[1],
                    )
                    for s, ps in chosen_dist.items():
                        attack_dist[s] = attack_dist.get(s, 0.0) + p_outcome * ps

            # Compare attacks by lexicographic expected value
            if best_dist is None or better_value(attack_value, best_value):
                best_value = attack_value
                best_dist = attack_dist
                best_action = (u, v)

        assert best_dist is not None and best_action is not None
        policy[state] = best_action
        return best_value, best_dist

    best_value, absorbing_dist = eval_state(start_state)
    return absorbing_dist, best_value, policy


def success_probability(start_state, edges, combat_df, num_attacker_nodes):
    (_, (p_success, _), _) = explore_absorbing_states_for_graph(
        edges=edges,
        combat_df=combat_df,
        start_state=start_state,
        num_attacker_nodes=num_attacker_nodes,
    )
    return p_success




def explore_absorbing_states_for_graph_local_objective(
    edges,
    combat_df: pd.DataFrame,
    start_state: GlobalState,
    num_attacker_nodes: int,
    *,
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
):
    """
    Exact recursive search using the local context-independent objective.

    Default objective:
        (E[new_territories], E[local_final_attacker_troops], P_local_conquest)

    If include_no_gain_in_value=True:
        (E[new_territories], -P(no_gain), E[local_final_attacker_troops], P_local_conquest)

    This returns the same shape as explore_absorbing_states_for_graph():
        absorbing_dist, best_value, policy
    """
    if value_tolerances is None:
        value_tolerances = default_local_value_tolerances(include_no_gain_in_value)

    adj = adjacency_dict(edges)
    policy: Dict[GlobalState, Tuple[int, int] | None] = {}

    @lru_cache(maxsize=None)
    def eval_state(state: GlobalState):
        if is_absorbing(state, edges):
            value = local_state_value(
                state,
                num_attacker_nodes,
                include_no_gain=include_no_gain_in_value,
            )
            dist = {state: 1.0}
            policy[state] = None
            return value, dist

        actions = possible_actions(state, edges)
        if not actions:
            value = local_state_value(
                state,
                num_attacker_nodes,
                include_no_gain=include_no_gain_in_value,
            )
            dist = {state: 1.0}
            policy[state] = None
            return value, dist

        best_value: Optional[Tuple[float, ...]] = None
        best_dist: Optional[Dict[GlobalState, float]] = None
        best_action: Optional[Tuple[int, int]] = None

        for action in actions:
            attack_value, attack_dist = eval_action(state, action)
            if best_dist is None or best_value is None or better_value_tuple(
                attack_value,
                best_value,
                value_tolerances,
            ):
                best_value = attack_value
                best_dist = attack_dist
                best_action = action

        assert best_value is not None and best_dist is not None and best_action is not None
        policy[state] = best_action
        return best_value, best_dist

    def eval_action(state: GlobalState, action: Tuple[int, int]):
        u, v = action
        node_u = state.nodes[u]
        node_v = state.nodes[v]

        T_u = node_u.troops
        T_v = node_v.troops
        assert T_u > 1, f"Cannot attack from node {u} with T_u={T_u}"

        combat_outcomes = get_combat_outcomes(combat_df, T_u - 1, T_v)

        sample_value = local_state_value(
            state,
            num_attacker_nodes,
            include_no_gain=include_no_gain_in_value,
        )
        attack_value: Tuple[float, ...] = tuple(0.0 for _ in sample_value)
        attack_dist: Dict[GlobalState, float] = {}

        for (a_avail_end, d_end), p_outcome in combat_outcomes:
            if p_outcome <= 0.0:
                continue

            origin_after = 1 + a_avail_end
            base_nodes = list(state.nodes)

            if d_end > 0:
                base_nodes[u] = NodeState("A", origin_after)
                base_nodes[v] = NodeState("D", d_end)
                next_state = GlobalState(nodes=tuple(base_nodes))

                v_next, dist_next = eval_state(next_state)
                attack_value = _add_scaled_value(attack_value, p_outcome, v_next)
                for s, ps in dist_next.items():
                    attack_dist[s] = attack_dist.get(s, 0.0) + p_outcome * ps
                continue

            total_at_u_before_move = origin_after
            other_enemy_neighbors = any(
                (w != v) and (state.nodes[w].owner == "D")
                for w in adj[u]
            )

            if other_enemy_neighbors:
                nodes1 = list(base_nodes)
                nodes1[u] = NodeState("A", total_at_u_before_move - 1)
                nodes1[v] = NodeState("A", 1)
                state1 = GlobalState(nodes=tuple(nodes1))
                v1, dist1 = eval_state(state1)

                nodes2 = list(base_nodes)
                nodes2[u] = NodeState("A", 1)
                nodes2[v] = NodeState("A", total_at_u_before_move - 1)
                state2 = GlobalState(nodes=tuple(nodes2))
                v2, dist2 = eval_state(state2)

                if better_value_tuple(v1, v2, value_tolerances):
                    chosen_value, chosen_dist = v1, dist1
                else:
                    chosen_value, chosen_dist = v2, dist2
            else:
                nodes3 = list(base_nodes)
                nodes3[u] = NodeState("A", 1)
                nodes3[v] = NodeState("A", total_at_u_before_move - 1)
                state3 = GlobalState(nodes=tuple(nodes3))
                chosen_value, chosen_dist = eval_state(state3)

            attack_value = _add_scaled_value(attack_value, p_outcome, chosen_value)
            for s, ps in chosen_dist.items():
                attack_dist[s] = attack_dist.get(s, 0.0) + p_outcome * ps

        return attack_value, attack_dist

    best_value, absorbing_dist = eval_state(start_state)
    return absorbing_dist, best_value, policy


def explore_root_policy_options_for_graph(
    edges,
    combat_df: pd.DataFrame,
    start_state: GlobalState,
    num_attacker_nodes: int,
    *,
    max_policy_options: Optional[int] = 2,
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
) -> List[PolicyOption]:
    """
    Return locally near-equivalent root-action policy options.

    Each option fixes one root action and then uses the local-objective optimal
    continuation thereafter. Obvious local losers are discarded. This preserves
    strategically ambiguous first-attack choices without storing full multi-policy
    trees.
    """
    if value_tolerances is None:
        value_tolerances = default_local_value_tolerances(include_no_gain_in_value)

    if is_absorbing(start_state, edges):
        return [
            PolicyOption(
                option_id=0,
                root_action=None,
                value=local_state_value(
                    start_state,
                    num_attacker_nodes,
                    include_no_gain=include_no_gain_in_value,
                ),
                absorbing_dist={start_state: 1.0},
            )
        ]

    actions = possible_actions(start_state, edges)
    if not actions:
        return [
            PolicyOption(
                option_id=0,
                root_action=None,
                value=local_state_value(
                    start_state,
                    num_attacker_nodes,
                    include_no_gain=include_no_gain_in_value,
                ),
                absorbing_dist={start_state: 1.0},
            )
        ]

    adj = adjacency_dict(edges)

    @lru_cache(maxsize=None)
    def eval_state(state: GlobalState):
        if is_absorbing(state, edges):
            return (
                local_state_value(
                    state,
                    num_attacker_nodes,
                    include_no_gain=include_no_gain_in_value,
                ),
                {state: 1.0},
            )

        st_actions = possible_actions(state, edges)
        if not st_actions:
            return (
                local_state_value(
                    state,
                    num_attacker_nodes,
                    include_no_gain=include_no_gain_in_value,
                ),
                {state: 1.0},
            )

        best_value: Optional[Tuple[float, ...]] = None
        best_dist: Optional[Dict[GlobalState, float]] = None

        for action in st_actions:
            val, dist = eval_action(state, action)
            if best_dist is None or best_value is None or better_value_tuple(val, best_value, value_tolerances):
                best_value, best_dist = val, dist

        assert best_value is not None and best_dist is not None
        return best_value, best_dist

    def eval_action(state: GlobalState, action: Tuple[int, int]):
        u, v = action
        node_u = state.nodes[u]
        node_v = state.nodes[v]
        T_u, T_v = node_u.troops, node_v.troops
        assert T_u > 1, f"Cannot attack from node {u} with T_u={T_u}"

        sample_value = local_state_value(
            state,
            num_attacker_nodes,
            include_no_gain=include_no_gain_in_value,
        )
        attack_value: Tuple[float, ...] = tuple(0.0 for _ in sample_value)
        attack_dist: Dict[GlobalState, float] = {}

        for (a_avail_end, d_end), p_outcome in get_combat_outcomes(combat_df, T_u - 1, T_v):
            if p_outcome <= 0.0:
                continue

            origin_after = 1 + a_avail_end
            base_nodes = list(state.nodes)

            if d_end > 0:
                base_nodes[u] = NodeState("A", origin_after)
                base_nodes[v] = NodeState("D", d_end)
                next_state = GlobalState(nodes=tuple(base_nodes))
                v_next, dist_next = eval_state(next_state)
                attack_value = _add_scaled_value(attack_value, p_outcome, v_next)
                for s, ps in dist_next.items():
                    attack_dist[s] = attack_dist.get(s, 0.0) + p_outcome * ps
                continue

            total_at_u_before_move = origin_after
            other_enemy_neighbors = any(
                (w != v) and (state.nodes[w].owner == "D")
                for w in adj[u]
            )

            if other_enemy_neighbors:
                nodes1 = list(base_nodes)
                nodes1[u] = NodeState("A", total_at_u_before_move - 1)
                nodes1[v] = NodeState("A", 1)
                state1 = GlobalState(nodes=tuple(nodes1))
                v1, dist1 = eval_state(state1)

                nodes2 = list(base_nodes)
                nodes2[u] = NodeState("A", 1)
                nodes2[v] = NodeState("A", total_at_u_before_move - 1)
                state2 = GlobalState(nodes=tuple(nodes2))
                v2, dist2 = eval_state(state2)

                if better_value_tuple(v1, v2, value_tolerances):
                    chosen_value, chosen_dist = v1, dist1
                else:
                    chosen_value, chosen_dist = v2, dist2
            else:
                nodes3 = list(base_nodes)
                nodes3[u] = NodeState("A", 1)
                nodes3[v] = NodeState("A", total_at_u_before_move - 1)
                state3 = GlobalState(nodes=tuple(nodes3))
                chosen_value, chosen_dist = eval_state(state3)

            attack_value = _add_scaled_value(attack_value, p_outcome, chosen_value)
            for s, ps in chosen_dist.items():
                attack_dist[s] = attack_dist.get(s, 0.0) + p_outcome * ps

        return attack_value, attack_dist

    candidates: List[PolicyOption] = []
    for action in actions:
        val, dist = eval_action(start_state, action)
        candidates.append(PolicyOption(-1, action, val, dist))

    best_value = candidates[0].value
    for opt in candidates[1:]:
        if better_value_tuple(opt.value, best_value, value_tolerances):
            best_value = opt.value

    kept = [opt for opt in candidates if value_relation(opt.value, best_value, value_tolerances) == 0]

    # Keep different root actions. Even if their distributions are identical
    # locally, the action identity can matter when the small graph is embedded.
    seen_actions = set()
    deduped: List[PolicyOption] = []
    for opt in kept:
        if opt.root_action in seen_actions:
            continue
        seen_actions.add(opt.root_action)
        deduped.append(opt)

    deduped.sort(key=lambda opt: (opt.value, opt.root_action or (-1, -1)), reverse=True)
    capped = _cap_list(deduped, max_policy_options)

    return [
        PolicyOption(i, opt.root_action, opt.value, opt.absorbing_dist)
        for i, opt in enumerate(capped)
    ]


def explore_state_set_policy_options_for_graph(
    edges,
    combat_df: pd.DataFrame,
    start_state: GlobalState,
    num_attacker_nodes: int,
    *,
    max_policy_options: Optional[int] = 4,
    max_options_per_state: Optional[int] = 2,
    max_split_depth: Optional[int] = 1,
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
) -> List[PolicyOption]:
    """
    Bottom-up bounded option-set solver.

    Instead of saving only different root actions, this lets tied continuation
    choices near the leaves create distinct absorbing distributions. Those
    alternatives are propagated upward and pruned at each parent state.

    Caps:
      - max_options_per_state=None means no width cap inside recursion.
      - max_split_depth=None means no depth cap on preserved split levels.
      - max_policy_options=None means no final row cap.
    """
    if value_tolerances is None:
        value_tolerances = default_local_value_tolerances(include_no_gain_in_value)

    adj = adjacency_dict(edges)

    @lru_cache(maxsize=None)
    def eval_state_options(state: GlobalState) -> Tuple[_StateOption, ...]:
        if is_absorbing(state, edges):
            return (
                _StateOption(
                    value=local_state_value(
                        state,
                        num_attacker_nodes,
                        include_no_gain=include_no_gain_in_value,
                    ),
                    absorbing_dist={state: 1.0},
                    root_action=None,
                    branch_key=None,
                    split_depth_used=0,
                ),
            )

        actions = possible_actions(state, edges)
        if not actions:
            return (
                _StateOption(
                    value=local_state_value(
                        state,
                        num_attacker_nodes,
                        include_no_gain=include_no_gain_in_value,
                    ),
                    absorbing_dist={state: 1.0},
                    root_action=None,
                    branch_key=None,
                    split_depth_used=0,
                ),
            )

        candidates: List[_StateOption] = []
        for action in actions:
            for opt in eval_action_options(state, action):
                candidates.append(
                    _StateOption(
                        value=opt.value,
                        absorbing_dist=opt.absorbing_dist,
                        root_action=action if opt.root_action is None else opt.root_action,
                        branch_key=action,
                        split_depth_used=opt.split_depth_used,
                    )
                )

        kept = _prune_state_options(
            candidates,
            value_tolerances=value_tolerances,
            max_options=max_options_per_state,
            max_split_depth=max_split_depth,
            meaningful_decision=(len(actions) > 1),
        )
        return tuple(kept)

    def eval_action_options(state: GlobalState, action: Tuple[int, int]) -> List[_StateOption]:
        u, v = action
        node_u = state.nodes[u]
        node_v = state.nodes[v]
        T_u, T_v = node_u.troops, node_v.troops
        assert T_u > 1, f"Cannot attack from node {u} with T_u={T_u}"

        weighted_child_sets: List[Tuple[float, List[_StateOption]]] = []

        for (a_avail_end, d_end), p_outcome in get_combat_outcomes(combat_df, T_u - 1, T_v):
            if p_outcome <= 0.0:
                continue

            origin_after = 1 + a_avail_end
            base_nodes = list(state.nodes)

            if d_end > 0:
                base_nodes[u] = NodeState("A", origin_after)
                base_nodes[v] = NodeState("D", d_end)
                next_state = GlobalState(nodes=tuple(base_nodes))
                child_opts = list(eval_state_options(next_state))
                weighted_child_sets.append((p_outcome, child_opts))
                continue

            total_at_u_before_move = origin_after
            other_enemy_neighbors = any(
                (w != v) and (state.nodes[w].owner == "D")
                for w in adj[u]
            )

            if other_enemy_neighbors:
                # Movement is itself a meaningful decision: keep stack back vs push forward.
                nodes1 = list(base_nodes)
                nodes1[u] = NodeState("A", total_at_u_before_move - 1)
                nodes1[v] = NodeState("A", 1)
                state1 = GlobalState(nodes=tuple(nodes1))
                opts1 = [
                    _StateOption(
                        value=o.value,
                        absorbing_dist=o.absorbing_dist,
                        root_action=o.root_action,
                        branch_key=("move_one", u, v),
                        split_depth_used=o.split_depth_used,
                    )
                    for o in eval_state_options(state1)
                ]

                nodes2 = list(base_nodes)
                nodes2[u] = NodeState("A", 1)
                nodes2[v] = NodeState("A", total_at_u_before_move - 1)
                state2 = GlobalState(nodes=tuple(nodes2))
                opts2 = [
                    _StateOption(
                        value=o.value,
                        absorbing_dist=o.absorbing_dist,
                        root_action=o.root_action,
                        branch_key=("push", u, v),
                        split_depth_used=o.split_depth_used,
                    )
                    for o in eval_state_options(state2)
                ]

                movement_opts = _prune_state_options(
                    opts1 + opts2,
                    value_tolerances=value_tolerances,
                    max_options=max_options_per_state,
                    max_split_depth=max_split_depth,
                    meaningful_decision=True,
                )
                weighted_child_sets.append((p_outcome, movement_opts))
            else:
                nodes3 = list(base_nodes)
                nodes3[u] = NodeState("A", 1)
                nodes3[v] = NodeState("A", total_at_u_before_move - 1)
                state3 = GlobalState(nodes=tuple(nodes3))
                child_opts = list(eval_state_options(state3))
                weighted_child_sets.append((p_outcome, child_opts))

        action_options = _combine_weighted_child_option_sets(
            weighted_child_sets,
            value_tolerances=value_tolerances,
            max_options=max_options_per_state,
            max_split_depth=max_split_depth,
        )

        # The first action from the current state should be visible at the root.
        return [
            _StateOption(
                value=opt.value,
                absorbing_dist=opt.absorbing_dist,
                root_action=action if opt.root_action is None else opt.root_action,
                branch_key=action,
                split_depth_used=opt.split_depth_used,
            )
            for opt in action_options
        ]

    root_options = list(eval_state_options(start_state))
    root_options = _prune_state_options(
        root_options,
        value_tolerances=value_tolerances,
        max_options=max_policy_options,
        max_split_depth=max_split_depth,
        meaningful_decision=False,
    )

    return [
        PolicyOption(
            option_id=i,
            root_action=opt.root_action,
            value=opt.value,
            absorbing_dist=opt.absorbing_dist,
        )
        for i, opt in enumerate(root_options)
    ]


# ---------------------------------------------------------------------
# Encoding and absorption table construction
# ---------------------------------------------------------------------


def encode_state_label(state: GlobalState) -> str:
    """
    Encode a GlobalState as a string like "(A3,D2,D1)", meaning:
      node 0: owner A, 3 troops
      node 1: owner D, 2 troops
      node 2: owner D, 1 troop
    """
    parts = []
    for node in state.nodes:
        prefix = 'A' if node.owner == 'A' else 'D'
        parts.append(f"{prefix}{node.troops}")
    return "(" + ",".join(parts) + ")"



def build_absorption_tables_two_player(
    combat_df: pd.DataFrame,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops: int,
    max_defender_troops: int,
    edges_list: Optional[List[Iterable[Tuple[int, int]]]] = None,
    canonical_edges_list: Optional[List[Iterable[Tuple[int, int]]]] = None,
    *,
    return_stats: bool = False,
    utility_mode: str = "legacy",
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
):
    """
    Build absorption tables for small graphs with an A/D partition.

    Graph source priority:
      1) canonical_edges_list (assumed already canonical under A/D-preserving permutations)
      2) edges_list (arbitrary edge sets; will canonicalize+dedupe)
      3) enumerate ALL connected labelled graphs on nA+nD; canonicalize+dedupe

    Canonicalization:
      - A/D-preserving only (permute attacker nodes among themselves and defender nodes among themselves).
      - Result key: canonical_edges_key = tuple(sorted((u,v), ...)) with u < v.

    Returns
    -------
    If return_stats=False:
        tables, policies
    If return_stats=True:
        tables, policies, stats

    tables : Dict[canonical_edges_key, pd.DataFrame]
    policies : Dict[canonical_edges_key, Dict[row_label, Dict[GlobalState, (u,v)|None]]]

    Notes
    -----
    - This function computes *exact* absorption distributions for all initial troop
      configurations up to the given caps.
    - For performance when rebuilding often: precompute canonical topologies once,
      then pass canonical_edges_list to avoid generating/looping over all labelled graphs.
    """
    from itertools import product  # local import avoids accidental shadowing elsewhere

    total_nodes = num_attacker_nodes + num_defender_nodes

    # -----------------------------
    # Choose graph source
    # -----------------------------
    if canonical_edges_list is not None:
        # Treat inputs as canonical reps (still normalize to sets)
        graphs = [set(tuple(sorted(e)) for e in edges) for edges in canonical_edges_list]
        will_canonicalize = False
        graphs_seen = len(graphs)
    elif edges_list is not None:
        graphs = [set(tuple(sorted(e)) for e in edges) for edges in edges_list]
        will_canonicalize = True
        graphs_seen = len(graphs)
    else:
        graphs = generate_connected_graphs_n_nodes(total_nodes)
        # ensure normalized edge tuples
        graphs = [set(tuple(sorted(e)) for e in edges) for edges in graphs]
        will_canonicalize = True
        graphs_seen = len(graphs)

    # -----------------------------
    # Output containers
    # -----------------------------
    tables: Dict[Tuple[Tuple[int, int], ...], pd.DataFrame] = {}
    policies: Dict[
        Tuple[Tuple[int, int], ...],
        Dict[str, Dict[GlobalState, Tuple[int, int] | None]]
    ] = {}

    att_range = range(1, max_attacker_troops + 1)
    def_range = range(1, max_defender_troops + 1)

    # Dedup by canonical key
    seen_canonical: set[Tuple[Tuple[int, int], ...]] = set()

    # Optional: map canonical key -> how many inputs map to it (useful for debugging)
    canon_multiplicity: Dict[Tuple[Tuple[int, int], ...], int] = {}

    # -----------------------------
    # Main loop
    # -----------------------------
    for edges in graphs:
        # Normalize edges as sorted pairs (u < v) inside a set
        edges = set((u, v) if u < v else (v, u) for (u, v) in edges)

        if will_canonicalize:
            canonical_edges_key, _perm_old_to_new, _perm_new_to_old = canonicalize_edges_with_roles(
                edges=sorted(edges),
                num_attacker_nodes=num_attacker_nodes,
                num_defender_nodes=num_defender_nodes,
            )
        else:
            # Assume caller already provided canonical reps
            # Still enforce canonical key format (sorted tuple of sorted edges).
            canonical_edges_key = tuple(sorted((u, v) if u < v else (v, u) for (u, v) in edges))

        canon_multiplicity[canonical_edges_key] = canon_multiplicity.get(canonical_edges_key, 0) + 1

        if canonical_edges_key in seen_canonical:
            continue
        seen_canonical.add(canonical_edges_key)

        # Use canonical representative edges for evaluation
        edges_canonical = set(canonical_edges_key)

        all_abs_states: set[GlobalState] = set()
        row_labels: list[str] = []
        row_dists: list[Dict[GlobalState, float]] = []
        row_policies: Dict[str, Dict[GlobalState, Tuple[int, int] | None]] = {}

        for attacker_troops in product(att_range, repeat=num_attacker_nodes):
            for defender_troops in product(def_range, repeat=num_defender_nodes):
                start = initial_state_generic(attacker_troops, defender_troops)

                if utility_mode == "local":
                    absorbing_dist, _expected_value, policy = explore_absorbing_states_for_graph_local_objective(
                        edges=edges_canonical,
                        combat_df=combat_df,
                        start_state=start,
                        num_attacker_nodes=num_attacker_nodes,
                        value_tolerances=value_tolerances,
                        include_no_gain_in_value=include_no_gain_in_value,
                    )
                elif utility_mode == "legacy":
                    absorbing_dist, _expected_value, policy = explore_absorbing_states_for_graph(
                        edges=edges_canonical,
                        combat_df=combat_df,
                        start_state=start,
                        num_attacker_nodes=num_attacker_nodes,
                    )
                else:
                    raise ValueError(f"utility_mode must be 'legacy' or 'local', got {utility_mode!r}")

                row_label = encode_state_label(start)
                row_labels.append(row_label)
                row_dists.append(absorbing_dist)
                all_abs_states.update(absorbing_dist.keys())
                row_policies[row_label] = policy

        col_states = sorted(all_abs_states, key=lambda s: encode_state_label(s))
        col_labels = [encode_state_label(s) for s in col_states]

        df = pd.DataFrame(0.0, index=row_labels, columns=col_labels)
        for row_label, absorbing_dist in zip(row_labels, row_dists):
            for state, prob in absorbing_dist.items():
                df.loc[row_label, encode_state_label(state)] = prob

        tables[canonical_edges_key] = df
        policies[canonical_edges_key] = row_policies

    if not return_stats:
        return tables, policies

    stats = {
        "nA": num_attacker_nodes,
        "nD": num_defender_nodes,
        "total_nodes": total_nodes,
        "max_attacker_troops": max_attacker_troops,
        "max_defender_troops": max_defender_troops,
        "graphs_seen": graphs_seen,                 # raw inputs iterated
        "graphs_canonical": len(seen_canonical),    # unique canon classes computed
        "canonical_multiplicity": canon_multiplicity,  # how many inputs mapped to each canon key
        "source": (
            "canonical_edges_list" if canonical_edges_list is not None
            else "edges_list" if edges_list is not None
            else "enumerated_labelled"
        ),
    }
    return tables, policies, stats




def parse_row_label(row_label: str) -> Tuple[List[str], List[int]]:
    """
    Parse a row label like "(A3,A2,D1,D1)" into
      owners = ['A','A','D','D']
      troops = [3,2,1,1]
    Adjust parsing logic if your label format differs.
    """
    inner = row_label.strip("()")
    parts = inner.split(",")
    owners: List[str] = []
    troops: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        owners.append(p[0])
        troops.append(int(p[1:]))
    return owners, troops


# ---------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------



def derive_plateau_policy_for_edges(
    row_policies: Dict[str, Dict["GlobalState", Optional[Tuple[int, int]]]],
    row_start_states: Dict[str, "GlobalState"],
    num_attacker_nodes: int,
    high_min_att_troops: int = 3,
    *,
    max_attacker_troops_exact: Optional[int] = None,
    min_support: float = 0.8,
    allow_threshold_relaxation: bool = True,
    allow_global_fallback: bool = True,   # <-- NEW
) -> Optional["PlateauPolicy"]:
    """
    Derive a simple plateau policy from exact-region policies.

    Patch:
      - robust fallback if plateau-like evidence is empty/too weak:
          use modal action among ANY states where attacker can attack.
    """
    # Clamp threshold to what exists in exact region
    eff_thr = high_min_att_troops
    if max_attacker_troops_exact is not None:
        eff_thr = min(eff_thr, max_attacker_troops_exact)

    def _collect_actions_plateau_like(threshold: int) -> List[Tuple[int, int]]:
        actions: List[Tuple[int, int]] = []
        for row_label, policy in row_policies.items():
            owners, troops = parse_row_label(row_label)

            # attacker block first by construction
            if not all(owners[i] == "A" for i in range(num_attacker_nodes)):
                continue

            # plateau-like: all attacker nodes >= threshold
            if not all(troops[i] >= threshold for i in range(num_attacker_nodes)):
                continue

            start_state = row_start_states[row_label]
            action = policy.get(start_state, None)
            if action is not None:
                actions.append(action)
        return actions

    def _collect_actions_any_attack_possible() -> List[Tuple[int, int]]:
        actions: List[Tuple[int, int]] = []
        for row_label, policy in row_policies.items():
            owners, troops = parse_row_label(row_label)

            if not all(owners[i] == "A" for i in range(num_attacker_nodes)):
                continue

            # any attacker node can attack (>=2 troops somewhere)
            if not any(troops[i] >= 2 for i in range(num_attacker_nodes)):
                continue

            start_state = row_start_states[row_label]
            action = policy.get(start_state, None)
            if action is not None:
                actions.append(action)
        return actions

    # Try plateau threshold(s)
    thresholds_to_try = [eff_thr]
    if allow_threshold_relaxation:
        thresholds_to_try += list(range(eff_thr - 1, 1, -1))  # down to 2

    for thr in thresholds_to_try:
        root_actions = _collect_actions_plateau_like(thr)
        if not root_actions:
            continue

        counts = Counter(root_actions)
        most_common_action, freq = counts.most_common(1)[0]
        if (freq / len(root_actions)) >= min_support:
            return PlateauPolicy(edges_order=[most_common_action])

    # Fallback: use any “attack-possible” states
    if allow_global_fallback:
        root_actions = _collect_actions_any_attack_possible()
        if root_actions:
            counts = Counter(root_actions)
            most_common_action, _freq = counts.most_common(1)[0]
            return PlateauPolicy(edges_order=[most_common_action])

    return None



def derive_plateau_policy_options_for_edges(
    row_policy_options: Dict[str, List[PolicyOption]],
    row_start_states: Dict[str, "GlobalState"],
    num_attacker_nodes: int,
    high_min_att_troops: int = 3,
    *,
    max_attacker_troops_exact: Optional[int] = None,
    min_support: float = 0.8,
    allow_threshold_relaxation: bool = True,
    allow_global_fallback: bool = True,
    value_round_digits: int = 10,
) -> Optional["PlateauPolicy"]:
    """
    Derive an option-aware plateau policy from exact multi-option rows.

    Unlike the legacy plateau builder, this does not collapse a row to one
    arbitrary representative action.  For each qualifying row it looks at the
    set of root actions represented by exact optimal/near-optimal options.  An
    action receives support from a row if it appears among that row's options.

    The result may contain several PlateauPolicyOption objects, one for each
    stable action family.  This preserves the fact that multiple high-troop
    plateau policies may exist.
    """
    eff_thr = high_min_att_troops
    if max_attacker_troops_exact is not None:
        eff_thr = min(eff_thr, max_attacker_troops_exact)

    thresholds_to_try = [eff_thr]
    if allow_threshold_relaxation:
        thresholds_to_try += list(range(eff_thr - 1, 1, -1))

    def _row_is_plateau_like(row_label: str, threshold: int) -> bool:
        owners, troops = parse_row_label(row_label)
        if not all(owners[i] == "A" for i in range(num_attacker_nodes)):
            return False
        return all(troops[i] >= threshold for i in range(num_attacker_nodes))

    def _row_attack_possible(row_label: str) -> bool:
        owners, troops = parse_row_label(row_label)
        if not all(owners[i] == "A" for i in range(num_attacker_nodes)):
            return False
        return any(troops[i] >= 2 for i in range(num_attacker_nodes))

    def _collect(predicate) -> Tuple[Counter, Dict[Tuple[int, int], Dict[str, Any]], int, Counter]:
        action_support: Counter = Counter()
        action_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}
        option_count_hist: Counter = Counter()
        rows_used = 0

        for row_label, options in row_policy_options.items():
            if not predicate(row_label):
                continue
            usable = [opt for opt in (options or []) if opt.root_action is not None]
            if not usable:
                continue

            rows_used += 1
            unique_actions = sorted({tuple(opt.root_action) for opt in usable})
            option_count_hist[len(unique_actions)] += 1

            # Row-level support: an action that is present in the row's option set
            # receives one support count from that row, not one count per duplicate.
            for action in unique_actions:
                action_support[action] += 1

            for opt in usable:
                action = tuple(opt.root_action)
                meta = action_meta.setdefault(
                    action,
                    {
                        "value_signatures": Counter(),
                        "outcome_signatures": Counter(),
                        "rows": set(),
                    },
                )
                meta["rows"].add(row_label)
                meta["value_signatures"][tuple(round(float(x), value_round_digits) for x in opt.value)] += 1
                meta["outcome_signatures"][_dist_signature(opt.absorbing_dist, ndigits=value_round_digits)] += 1

        return action_support, action_meta, rows_used, option_count_hist

    selected = None
    selected_threshold = None
    selected_source = None

    for thr in thresholds_to_try:
        collected = _collect(lambda rl, thr=thr: _row_is_plateau_like(rl, thr))
        action_support, _meta, rows_used, _hist = collected
        if rows_used <= 0:
            continue
        stable_actions = [a for a, c in action_support.items() if (c / rows_used) >= float(min_support)]
        if stable_actions:
            selected = collected
            selected_threshold = thr
            selected_source = "plateau_like"
            break

    if selected is None and allow_global_fallback:
        collected = _collect(_row_attack_possible)
        action_support, _meta, rows_used, _hist = collected
        if rows_used > 0:
            stable_actions = [a for a, c in action_support.items() if (c / rows_used) >= float(min_support)]
            if stable_actions:
                selected = collected
                selected_threshold = None
                selected_source = "global_fallback"

    if selected is None:
        return None

    action_support, action_meta, rows_used, option_count_hist = selected
    stable_actions = sorted(
        [a for a, c in action_support.items() if (c / rows_used) >= float(min_support)],
        key=lambda a: (-action_support[a], a),
    )
    if not stable_actions:
        return None

    options: List[PlateauPolicyOption] = []
    for i, action in enumerate(stable_actions):
        meta = action_meta.get(action, {})
        value_sig = None
        if meta.get("value_signatures"):
            value_sig = meta["value_signatures"].most_common(1)[0][0]
        outcome_sig = None
        if meta.get("outcome_signatures"):
            outcome_sig = meta["outcome_signatures"].most_common(1)[0][0]

        options.append(
            PlateauPolicyOption(
                option_id=i,
                action_groups=[[action]],
                representative_root_action=action,
                support=float(action_support[action]) / float(rows_used),
                support_rows=int(action_support[action]),
                total_rows_considered=int(rows_used),
                value_signature=value_sig,
                outcome_signature=outcome_sig,
            )
        )

    edges_order: List[Tuple[int, int]] = []
    for action in stable_actions:
        if action not in edges_order:
            edges_order.append(action)

    diagnostics = {
        "source": selected_source,
        "threshold": selected_threshold,
        "min_support": float(min_support),
        "rows_used": int(rows_used),
        "stable_actions": [list(a) for a in stable_actions],
        "action_support": {str(k): int(v) for k, v in action_support.items()},
        "action_support_fraction": {str(k): float(v) / float(rows_used) for k, v in action_support.items()},
        "option_count_histogram": {int(k): int(v) for k, v in option_count_hist.items()},
        "mode": "option_aware",
    }

    return PlateauPolicy(
        edges_order=edges_order,
        action_groups=[[a] for a in stable_actions],
        options=options,
        diagnostics=diagnostics,
    )


def choose_action_from_plateau(
    state: "GlobalState",
    edges: List[Tuple[int, int]],
    plateau_policy: PlateauPolicy,
) -> Optional[Tuple[int, int]]:
    """
    Choose one legal action according to a plateau policy.

    If action_groups are present, actions inside the same group are treated as
    equally preferred plateau choices.  A deterministic representative is still
    selected for simulation, but the policy object records that alternatives were
    equivalent for inference/diagnostics.
    """
    actions = possible_actions(state, edges)
    if not actions:
        return None

    legal = set(actions)

    groups = plateau_policy.action_groups
    if groups:
        for group in groups:
            for action in group:
                if tuple(action) in legal:
                    return tuple(action)

    for action in plateau_policy.edges_order:
        if tuple(action) in legal:
            return tuple(action)

    return actions[0]


def plateau_policy_from_option(opt: PlateauPolicyOption) -> PlateauPolicy:
    """Convert one option-aware plateau family into a fixed chooser policy."""
    edges_order: List[Tuple[int, int]] = []
    for group in opt.action_groups or []:
        for action in group:
            action = tuple(action)
            if action not in edges_order:
                edges_order.append(action)
    if opt.representative_root_action is not None:
        r = tuple(opt.representative_root_action)
        if r not in edges_order:
            edges_order.insert(0, r)
    return PlateauPolicy(
        edges_order=edges_order,
        action_groups=opt.action_groups,
        options=[opt],
        diagnostics={
            "source": "plateau_policy_option",
            "option_id": int(opt.option_id),
            "support": float(opt.support),
            "support_rows": int(opt.support_rows),
            "total_rows_considered": int(opt.total_rows_considered),
        },
    )


def evaluate_under_fixed_policy(
    edges: List[Tuple[int, int]],
    combat_df: pd.DataFrame,
    start_state: GlobalState,
    plateau_policy: PlateauPolicy,
    num_attacker_nodes: int,
) -> Dict[GlobalState, float]:
    """
    Compute an approximate absorbing distribution for `start_state` under a
    *fixed* action profile (plateau_policy), without searching over alternative actions.

    Movement choices (keep stack vs push stack) still follow the same rational
    lexicographic rule as in explore_absorbing_states_for_graph; the only
    approximation is that we do not search over different attack edges.
    """
    adj = adjacency_dict(edges)

    @lru_cache(maxsize=None)
    def _eval_state(state: GlobalState) -> Dict[GlobalState, float]:
        if is_absorbing(state, edges):
            return {state: 1.0}

        action = choose_action_from_plateau(state, edges, plateau_policy)
        if action is None:
            # No legal actions -> absorbing under this policy
            return {state: 1.0}

        u, v = action

        node_u = state.nodes[u]
        node_v = state.nodes[v]

        T_u = node_u.troops
        T_v = node_v.troops
        assert T_u > 1, f"Cannot attack from node {u} with T_u={T_u}"

        a_init = T_u - 1
        d_init = T_v

        if d_init <= 0 or a_init <= 0:
            return {state: 1.0}

        combat_outcomes = get_combat_outcomes(combat_df, a_init, d_init)

        dist_total: Dict[GlobalState, float] = {}

        for (a_avail_end, d_end), p_outcome in combat_outcomes:
            if p_outcome <= 0.0:
                continue

            origin_after = 1 + a_avail_end
            base_nodes = list(state.nodes)

            if d_end > 0:
                # Defender holds v, no conquest
                base_nodes[u] = NodeState("A", origin_after)
                base_nodes[v] = NodeState("D", d_end)
                next_state = GlobalState(nodes=tuple(base_nodes))

                sub_dist = _eval_state(next_state)

                for s, p_s in sub_dist.items():
                    dist_total[s] = dist_total.get(s, 0.0) + p_outcome * p_s

            else:
                # Defender eliminated on v -> movement choice
                total_at_u_before_move = origin_after

                other_enemy_neighbors = any(
                    (w != v) and (state.nodes[w].owner == "D")
                    for w in adj[u]
                )

                if other_enemy_neighbors:
                    # Two movement choices: keep stack back vs push stack
                    nodes1 = list(base_nodes)
                    nodes1[u] = NodeState("A", total_at_u_before_move - 1)
                    nodes1[v] = NodeState("A", 1)
                    state1 = GlobalState(nodes=tuple(nodes1))
                    dist1 = _eval_state(state1)

                    nodes2 = list(base_nodes)
                    nodes2[u] = NodeState("A", 1)
                    nodes2[v] = NodeState("A", total_at_u_before_move - 1)
                    state2 = GlobalState(nodes=tuple(nodes2))
                    dist2 = _eval_state(state2)

                    # Evaluate lexicographic value for movement (same as state_utility logic)
                    def value_from_dist(dist: Dict[GlobalState, float]) -> Tuple[float, float]:
                        p_succ = 0.0
                        troop_score = 0.0
                        for s, p in dist.items():
                            ps, ts = state_utility(s, num_attacker_nodes)
                            p_succ += p * ps
                            troop_score += p * ts
                        return (p_succ, troop_score)

                    v1 = value_from_dist(dist1)
                    v2 = value_from_dist(dist2)

                    if better_value(v1, v2):
                        chosen_dist = dist1
                    else:
                        chosen_dist = dist2

                    for s, p_s in chosen_dist.items():
                        dist_total[s] = dist_total.get(s, 0.0) + p_outcome * p_s

                else:
                    # Forced: move all but 1 to v
                    nodes3 = list(base_nodes)
                    nodes3[u] = NodeState("A", 1)
                    nodes3[v] = NodeState("A", total_at_u_before_move - 1)
                    state3 = GlobalState(nodes=tuple(nodes3))
                    dist3 = _eval_state(state3)
                    for s, p_s in dist3.items():
                        dist_total[s] = dist_total.get(s, 0.0) + p_outcome * p_s

        return dist_total

    return _eval_state(start_state)



def expected_local_value_from_dist(
    dist: Dict[GlobalState, float],
    *,
    num_attacker_nodes: int,
    include_no_gain_in_value: bool = False,
) -> Tuple[float, ...]:
    """Expected local value tuple for an absorbing-state distribution."""
    acc: Optional[Tuple[float, ...]] = None
    for st, p in dist.items():
        val = local_state_value(
            st,
            num_attacker_nodes,
            include_no_gain=include_no_gain_in_value,
        )
        if acc is None:
            acc = tuple(0.0 for _ in val)
        acc = _add_scaled_value(acc, float(p), val)
    if acc is None:
        return tuple(0.0 for _ in default_local_value_tolerances(include_no_gain_in_value))
    return acc


def _prune_policy_options_by_value_and_distribution(
    candidates: List[PolicyOption],
    *,
    value_tolerances: Optional[Tuple[float, ...]],
    max_policy_options: Optional[int],
) -> List[PolicyOption]:
    """Keep near-best policy options and deduplicate identical distributions."""
    if not candidates:
        return []

    best_value = candidates[0].value
    for opt in candidates[1:]:
        if better_value_tuple(opt.value, best_value, value_tolerances):
            best_value = opt.value

    near_best = [
        opt for opt in candidates
        if value_relation(opt.value, best_value, value_tolerances) == 0
    ]

    seen = set()
    distinct: List[PolicyOption] = []
    for opt in near_best:
        sig = _dist_signature(opt.absorbing_dist)
        if sig in seen:
            continue
        seen.add(sig)
        distinct.append(opt)

    distinct.sort(key=lambda opt: (opt.value, opt.root_action or (-1, -1)), reverse=True)
    capped = _cap_list(distinct, max_policy_options)
    return [PolicyOption(i, opt.root_action, opt.value, opt.absorbing_dist) for i, opt in enumerate(capped)]


def evaluate_under_plateau_policy_options(
    edges: List[Tuple[int, int]],
    combat_df: pd.DataFrame,
    start_state: GlobalState,
    plateau_policy: PlateauPolicy,
    num_attacker_nodes: int,
    *,
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
    max_policy_options: Optional[int] = None,
) -> List[PolicyOption]:
    """
    Evaluate a high-troop state under each option-aware plateau family.

    This returns the same kind of policy-option objects as the exact solver, but
    each option is produced by one inferred plateau policy family rather than by
    full exact search in the high-troop state.
    """
    if value_tolerances is None:
        value_tolerances = default_local_value_tolerances(include_no_gain_in_value)

    raw_options = list(plateau_policy.options or [])
    if not raw_options:
        # Backward-compatible single plateau policy.
        dist = evaluate_under_fixed_policy(
            edges=edges,
            combat_df=combat_df,
            start_state=start_state,
            plateau_policy=plateau_policy,
            num_attacker_nodes=num_attacker_nodes,
        )
        val = expected_local_value_from_dist(
            dist,
            num_attacker_nodes=num_attacker_nodes,
            include_no_gain_in_value=include_no_gain_in_value,
        )
        root_action = choose_action_from_plateau(start_state, edges, plateau_policy)
        return [PolicyOption(0, root_action, val, dist)]

    candidates: List[PolicyOption] = []
    for i, popt in enumerate(raw_options):
        fixed_policy = plateau_policy_from_option(popt)
        dist = evaluate_under_fixed_policy(
            edges=edges,
            combat_df=combat_df,
            start_state=start_state,
            plateau_policy=fixed_policy,
            num_attacker_nodes=num_attacker_nodes,
        )
        val = expected_local_value_from_dist(
            dist,
            num_attacker_nodes=num_attacker_nodes,
            include_no_gain_in_value=include_no_gain_in_value,
        )
        root_action = choose_action_from_plateau(start_state, edges, fixed_policy)
        if root_action is None:
            root_action = popt.representative_root_action
        candidates.append(PolicyOption(i, root_action, val, dist))

    return _prune_policy_options_by_value_and_distribution(
        candidates,
        value_tolerances=value_tolerances,
        max_policy_options=max_policy_options,
    )

def renormalize_rows(df: pd.DataFrame, tol: float = 1e-8) -> pd.DataFrame:
    """
    Row-normalize a (possibly very wide) probability table without densifying.

    Works safely for:
      - float32 / float64
      - SparseDtype(float, fill_value=0)

    Rows with sum <= tol are left unchanged.
    """
    # Compute row sums (this is sparse-safe)
    row_sums = df.sum(axis=1)

    # Factors: divide only where sum > tol, else factor = 1
    factors = row_sums.where(row_sums > tol, 1.0)

    # Fast path: dense DataFrame
    if not isinstance(df.dtypes.iloc[0], pd.SparseDtype):
        return df.div(factors, axis=0)

    # ---- Sparse-safe path ----
    # We scale column-by-column to avoid densification
    out = df.copy()

    for col in out.columns:
        s = out[col]
        if not isinstance(s.dtype, pd.SparseDtype):
            # safety fallback (should not happen)
            out[col] = s / factors
            continue

        # Only operate on stored (non-zero) values
        sp = s.sparse
        if sp.npoints == 0:
            continue  # all zeros

        idx = sp.sp_index.indices
        vals = sp.sp_values / factors.iloc[idx].to_numpy()

        out[col] = pd.arrays.SparseArray(
            vals,
            sparse_index=sp.sp_index,
            fill_value=sp.fill_value,
        )

    return out


def add_precomputed_metrics_to_v2_payload(payload: dict, *, num_attacker_nodes: int) -> dict:
    """
    Given payload with keys: p, owners, troops
    add:
      - is_conquered
      - new_territories
      - final_attacker_troops
    using the exact semantics used in battle_graph_ranking legacy path.
    """
    if payload is None or payload == {}:
        return payload

    if "p" not in payload or "owners" not in payload or "troops" not in payload:
        return payload

    owners = np.asarray(payload["owners"])
    troops = np.asarray(payload["troops"])

    if owners.ndim != 2 or troops.ndim != 2 or owners.shape != troops.shape:
        raise ValueError(f"owners/troops must be 2D and same shape, got owners={owners.shape}, troops={troops.shape}")

    N, M = owners.shape

    # attacker-owned mask
    if owners.dtype.kind in ("U", "S", "O"):
        att_mask = (owners == "A")
    else:
        # adjust if your numeric encoding differs
        att_mask = (owners == 1)

    # new territories = defender-block nodes that end attacker-owned
    if num_attacker_nodes < 0 or num_attacker_nodes > M:
        raise ValueError(f"num_attacker_nodes={num_attacker_nodes} out of bounds for M={M}")

    defender_block = np.zeros((1, M), dtype=bool)
    defender_block[:, num_attacker_nodes:] = True
    new_territories = np.sum(att_mask & defender_block, axis=1).astype(np.int16)

    # is_conquered = all nodes attacker-owned (entire region owned by attacker)
    is_conquered = np.all(att_mask, axis=1).astype(np.uint8)

    # final attacker troops = sum troops on attacker-owned nodes
    final_attacker_troops = np.sum(troops * att_mask, axis=1).astype(np.int32)

    payload = dict(payload)
    payload["new_territories"] = new_territories
    payload["is_conquered"] = is_conquered
    payload["final_attacker_troops"] = final_attacker_troops
    return payload






def build_absorption_tables_two_player_with_plateau(
    combat_df: pd.DataFrame,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops_exact: int,
    max_defender_troops_exact: int,
    max_attacker_troops_extended: int,
    max_defender_troops_extended: int,
    high_min_att_troops: int = 3,
    edges_list: Optional[List[Iterable[Tuple[int, int]]]] = None,
    canonical_edges_list: Optional[List[Iterable[Tuple[int, int]]]] = None,
    *,
    # PATCHED:
    # "chunked_rows" is treated as alias for "chunked_rows_v2" (preferred),
    # but we still support v1 for backward compatibility / migration.
    output_format: str = "auto",  # "auto" | "dataframe" | "chunked_rows" | "chunked_rows_v1" | "chunked_rows_v2"
    chunk_rows: int = 5_000,
    chunk_root_dir: Optional[Path] = None,
    chunk_rel_prefix: str = "_chunked_rows",
    dtype: Any = np.float32,
    fallback_to_exact_search_if_no_plateau: bool = True,
    utility_mode: str = "legacy",
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",  # "root" | "state_set"
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_split_depth: Optional[int] = 1,
    # Heuristic threshold for auto (rows in extended region)
    auto_chunk_threshold_rows: int = 250_000,
) -> Tuple[Dict[Any, Any], Dict[Any, "PlateauPolicy"]]:
    """
    Build exact absorption tables up to (max_*_exact), then extend coverage to
    (max_*_extended).

    Output modes:
      - "dataframe": build a single extended pd.DataFrame per graph (small cases)
      - "chunked_rows_v1": write extended rows to disk as dict-of-probabilities chunks
                           (format="chunked_prob_table_v1")
      - "chunked_rows_v2": write extended rows to disk as array-row payloads
                           (format="chunked_prob_table_v2_rows_v1")
      - "chunked_rows": alias for v2 (preferred)
      - "auto": choose dataframe vs v2 chunked based on extended state-space size

    Chunked layout (per edges_key):
      chunk_root_dir / <edges_hash> / chunk_000000.pkl, chunk_000001.pkl, ...
    and the descriptor includes:
      chunk_dir = f"{chunk_rel_prefix}/{edges_hash}"  (relative to the per-graph .pkl folder)

    IMPORTANT:
      - exact_df remains a DataFrame (small exact region).
      - v2 chunk payload stores arrays; during migration you can still keep exact_df as legacy df.

    V2 METRICS (PATCHED):
      - is_conquered: 1 iff ALL nodes in region end attacker-owned (no defenders remain)
      - new_territories: number of defender-block nodes that end attacker-owned
      - final_attacker_troops: sum of troops on attacker-owned nodes in end state
    """
    import pickle
    import hashlib
    from itertools import product

    # -----------------------------
    # Decide output format (auto)
    # -----------------------------
    total_ext_rows = (max_attacker_troops_extended ** num_attacker_nodes) * (
        max_defender_troops_extended ** num_defender_nodes
    )

    fmt = str(output_format or "auto").strip().lower()
    if fmt == "chunked_rows":
        fmt = "chunked_rows_v2"  # alias -> v2 preferred

    allowed = {"auto", "dataframe", "chunked_rows_v1", "chunked_rows_v2"}
    if fmt not in allowed:
        raise ValueError(
            f"output_format must be one of {sorted(allowed)} (or 'chunked_rows' alias); got {output_format!r}"
        )

    if fmt == "auto":
        use_chunked = total_ext_rows >= auto_chunk_threshold_rows
        fmt = "chunked_rows_v2" if use_chunked else "dataframe"

    if fmt in {"chunked_rows_v1", "chunked_rows_v2"}:
        if chunk_root_dir is None:
            raise ValueError("chunk_root_dir must be provided when output_format is chunked_rows_*")

    # -----------------------------
    # Build EXACT region (always DF)
    # -----------------------------
    tables_exact, policies_exact = build_absorption_tables_two_player(
        combat_df=combat_df,
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
        max_attacker_troops=max_attacker_troops_exact,
        max_defender_troops=max_defender_troops_exact,
        edges_list=edges_list,
        canonical_edges_list=canonical_edges_list,
        utility_mode=utility_mode,
        value_tolerances=value_tolerances,
        include_no_gain_in_value=include_no_gain_in_value,
    )

    tables_extended: Dict[Any, Any] = {}
    plateau_policies: Dict[Any, PlateauPolicy] = {}

    # -----------------------------
    # Helpers
    # -----------------------------
    def _row_label(att_troops: Tuple[int, ...], def_troops: Tuple[int, ...]) -> str:
        return "(" + ",".join([f"A{t}" for t in att_troops] + [f"D{t}" for t in def_troops]) + ")"

    def _row_label_to_state(row_label: str) -> GlobalState:
        owners, troops = parse_row_label(row_label)
        att = [troops[i] for i in range(num_attacker_nodes)]
        deff = [troops[i] for i in range(num_attacker_nodes, len(troops))]
        return initial_state_generic(att, deff)

    def _should_be_in_exact(att_troops: Tuple[int, ...], def_troops: Tuple[int, ...]) -> bool:
        return (max(att_troops) <= max_attacker_troops_exact) and (max(def_troops) <= max_defender_troops_exact)

    def _normalize_row_dict(row: Dict[str, float]) -> Dict[str, float]:
        s = float(sum(row.values()))
        if s <= 0.0:
            return row
        inv = 1.0 / s
        return {k: float(v) * inv for k, v in row.items() if float(v) > 0.0}

    def _eval_row_distribution(
        *,
        edges: List[Tuple[int, int]],
        att_troops: Tuple[int, ...],
        def_troops: Tuple[int, ...],
        plateau_policy: Optional[PlateauPolicy],
    ) -> Dict[str, float]:
        start_state = initial_state_generic(att_troops, def_troops)

        if plateau_policy is not None:
            absorbing_dist = evaluate_under_fixed_policy(
                edges=edges,
                combat_df=combat_df,
                start_state=start_state,
                plateau_policy=plateau_policy,
                num_attacker_nodes=num_attacker_nodes,
            )
        else:
            if not fallback_to_exact_search_if_no_plateau:
                return {encode_state_label(start_state): 1.0}

            if utility_mode == "local":
                absorbing_dist, _val, _pol = explore_absorbing_states_for_graph_local_objective(
                    edges=edges,
                    combat_df=combat_df,
                    start_state=start_state,
                    num_attacker_nodes=num_attacker_nodes,
                    value_tolerances=value_tolerances,
                    include_no_gain_in_value=include_no_gain_in_value,
                )
            elif utility_mode == "legacy":
                absorbing_dist, _val, _pol = explore_absorbing_states_for_graph(
                    edges=edges,
                    combat_df=combat_df,
                    start_state=start_state,
                    num_attacker_nodes=num_attacker_nodes,
                )
            else:
                raise ValueError(f"utility_mode must be 'legacy' or 'local', got {utility_mode!r}")

        row: Dict[str, float] = {}
        for abs_state, p in absorbing_dist.items():
            p = float(p)
            if p <= 0.0:
                continue
            col_label = encode_state_label(abs_state)
            row[col_label] = row.get(col_label, 0.0) + p

        return _normalize_row_dict(row)

    def _edges_key_hash(edges_key: Any) -> str:
        # stable hash of canonical edges key
        s = repr(edges_key).encode("utf-8")
        return hashlib.sha1(s).hexdigest()[:10]

    # ---- v1 chunk writer (dict-of-probabilities) ----
    def _write_chunk_file_v1(
        folder: Path,
        chunk_index: int,
        edges_key: Any,
        rows_obj: Dict[str, Dict[str, float]],
    ) -> str:
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"chunk_{chunk_index:06d}.pkl"
        with p.open("wb") as f:
            pickle.dump(
                {"edges_key": edges_key, "format": "rowdict_chunk_v1", "rows": rows_obj},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return p.name  # filename only

    # ---- v2 chunk writer (array payloads) ----
    def _write_chunk_file_v2(
        folder: Path,
        chunk_index: int,
        edges_key: Any,
        rows_obj: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        V2 chunk payload format:
          {"format": "v2_rowchunk_v1", "edges_key": ..., "rows": {row_label: row_payload}}

        row_payload stores arrays:
          - "p": (N,) float32
          - "owners": (N,M) uint8
          - "troops": (N,M) uint16
          - "is_conquered": (N,) uint8
          - "new_territories": (N,) int16
          - "final_attacker_troops": (N,) int32
        """
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f"chunk_{chunk_index:06d}.pkl"
        with p.open("wb") as f:
            pickle.dump(
                {"edges_key": edges_key, "format": "v2_rowchunk_v1", "rows": rows_obj},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return p.name

    # ---- convert legacy dict-of-probs row -> v2 arrays payload (WITH METRICS) ----
    def _rowdict_to_v2_payload(row: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert {col_label: p} into arrays by decoding each col_label -> GlobalState.

        owners encoding:
          0 = unowned/empty
          1 = attacker ("A")
          2 = defender ("D")

        troops dtype:
          uint16 (safe for your troop caps)

        Also includes metrics required by V2 fast path:
          - is_conquered: 1 iff no defenders remain (all nodes attacker-owned)
          - new_territories: defender-block nodes that end attacker-owned
          - final_attacker_troops: sum troops on attacker-owned nodes
        """
        # Keep stable order for determinism
        items = [(lbl, float(p)) for (lbl, p) in row.items() if float(p) > 0.0]
        M = num_attacker_nodes + num_defender_nodes

        if not items:
            return {
                "p": np.zeros((0,), dtype=np.float32),
                "owners": np.zeros((0, M), dtype=np.uint8),
                "troops": np.zeros((0, M), dtype=np.uint16),
                "is_conquered": np.zeros((0,), dtype=np.uint8),
                "new_territories": np.zeros((0,), dtype=np.int16),
                "final_attacker_troops": np.zeros((0,), dtype=np.int32),
            }

        # Sort by label for stable build artifacts
        items.sort(key=lambda t: t[0])

        N = len(items)

        p_arr = np.empty((N,), dtype=np.float32)
        owners_arr = np.empty((N, M), dtype=np.uint8)
        troops_arr = np.empty((N, M), dtype=np.uint16)

        is_conq_arr = np.empty((N,), dtype=np.uint8)
        new_terr_arr = np.empty((N,), dtype=np.int16)
        final_att_arr = np.empty((N,), dtype=np.int32)

        for i, (lbl, p) in enumerate(items):
            p_arr[i] = np.float32(p)

            st = global_state_from_row_label(lbl)

            any_defender_remaining = False
            new_terr = 0
            final_att_troops = 0

            for j, node in enumerate(st.nodes):
                t = int(node.troops)

                if t <= 0:
                    owners_arr[i, j] = 0
                    troops_arr[i, j] = 0
                    continue

                if node.owner == "A":
                    owners_arr[i, j] = 1
                elif node.owner == "D":
                    owners_arr[i, j] = 2
                else:
                    owners_arr[i, j] = 0

                troops_arr[i, j] = np.uint16(max(t, 0))

                # ---- metrics ----
                if owners_arr[i, j] == 2:
                    any_defender_remaining = True

                if owners_arr[i, j] == 1:
                    final_att_troops += t
                    if j >= num_attacker_nodes:
                        new_terr += 1

            is_conq_arr[i] = np.uint8(0 if any_defender_remaining else 1)
            new_terr_arr[i] = np.int16(new_terr)
            final_att_arr[i] = np.int32(final_att_troops)

        # Defensive renormalize
        s = float(p_arr.sum())
        if s > 0.0 and abs(s - 1.0) > 1e-6:
            p_arr = p_arr / np.float32(s)

        return {
            "p": p_arr,
            "owners": owners_arr,
            "troops": troops_arr,
            "is_conquered": is_conq_arr,
            "new_territories": new_terr_arr,
            "final_attacker_troops": final_att_arr,
        }


    def _dist_to_rowdict(dist: Dict[GlobalState, float]) -> Dict[str, float]:
        row: Dict[str, float] = {}
        for abs_state, p in dist.items():
            p = float(p)
            if p <= 0.0:
                continue
            col_label = encode_state_label(abs_state)
            row[col_label] = row.get(col_label, 0.0) + p
        return _normalize_row_dict(row)

    def _option_payload_from_policy_option(opt: PolicyOption) -> Dict[str, Any]:
        payload = _rowdict_to_v2_payload(_dist_to_rowdict(opt.absorbing_dist))
        p_arr = np.asarray(payload.get("p", []), dtype=np.float64)
        new_terr = np.asarray(payload.get("new_territories", []), dtype=np.float64)
        is_conq = np.asarray(payload.get("is_conquered", []), dtype=np.float64)
        final_att = np.asarray(payload.get("final_attacker_troops", []), dtype=np.float64)

        expected_new = float(np.dot(p_arr, new_terr)) if p_arr.size else 0.0
        expected_final_att = float(np.dot(p_arr, final_att)) if p_arr.size else 0.0
        p_local_conquest = float(np.dot(p_arr, is_conq)) if p_arr.size else 0.0
        p_no_gain = float(p_arr[new_terr == 0].sum()) if p_arr.size else 0.0

        payload = dict(payload)
        payload["option_id"] = int(opt.option_id)
        payload["root_action"] = list(opt.root_action) if opt.root_action is not None else None
        payload["local_value"] = {
            "expected_new_territories": expected_new,
            "expected_final_attacker_troops": expected_final_att,
            "p_local_conquest": p_local_conquest,
            "p_no_gain": p_no_gain,
            "raw_value": tuple(float(x) for x in opt.value),
            "utility_tuple_semantics": (
                ("expected_new_territories", "-p_no_gain", "expected_final_attacker_troops", "p_local_conquest")
                if include_no_gain_in_value
                else ("expected_new_territories", "expected_final_attacker_troops", "p_local_conquest")
            ),
        }
        return payload

    def _policy_options_to_v2_payload(options: List[PolicyOption]) -> Dict[str, Any]:
        return {
            "format": "policy_options_v2",
            "policy_option_mode": str(policy_option_mode),
            "max_policy_options_per_row": None if max_policy_options_per_row is None else int(max_policy_options_per_row),
            "max_options_per_state": None if max_options_per_state is None else int(max_options_per_state),
            "max_split_depth": None if max_split_depth is None else int(max_split_depth),
            "include_no_gain_in_value": bool(include_no_gain_in_value),
            "options": [_option_payload_from_policy_option(opt) for opt in options],
        }

    def _eval_row_policy_options_list(
        *,
        edges: List[Tuple[int, int]],
        att_troops: Tuple[int, ...],
        def_troops: Tuple[int, ...],
        plateau_policy: Optional[PlateauPolicy] = None,
    ) -> List[PolicyOption]:
        start_state = initial_state_generic(att_troops, def_troops)

        if plateau_policy is not None:
            return evaluate_under_plateau_policy_options(
                edges=edges,
                combat_df=combat_df,
                start_state=start_state,
                plateau_policy=plateau_policy,
                num_attacker_nodes=num_attacker_nodes,
                value_tolerances=value_tolerances,
                include_no_gain_in_value=include_no_gain_in_value,
                max_policy_options=max_policy_options_per_row,
            )

        mode = str(policy_option_mode or "root").strip().lower()

        if mode == "root":
            return explore_root_policy_options_for_graph(
                edges=edges,
                combat_df=combat_df,
                start_state=start_state,
                num_attacker_nodes=num_attacker_nodes,
                max_policy_options=max_policy_options_per_row,
                value_tolerances=value_tolerances,
                include_no_gain_in_value=include_no_gain_in_value,
            )
        if mode in {"state_set", "state-set", "bottom_up", "bottom-up"}:
            return explore_state_set_policy_options_for_graph(
                edges=edges,
                combat_df=combat_df,
                start_state=start_state,
                num_attacker_nodes=num_attacker_nodes,
                max_policy_options=max_policy_options_per_row,
                max_options_per_state=max_options_per_state,
                max_split_depth=max_split_depth,
                value_tolerances=value_tolerances,
                include_no_gain_in_value=include_no_gain_in_value,
            )
        raise ValueError(
            "policy_option_mode must be 'root' or 'state_set', "
            f"got {policy_option_mode!r}"
        )

    def _eval_row_policy_options_payload(
        *,
        edges: List[Tuple[int, int]],
        att_troops: Tuple[int, ...],
        def_troops: Tuple[int, ...],
        plateau_policy: Optional[PlateauPolicy] = None,
    ) -> Dict[str, Any]:
        options = _eval_row_policy_options_list(
            edges=edges,
            att_troops=att_troops,
            def_troops=def_troops,
            plateau_policy=plateau_policy,
        )
        return _policy_options_to_v2_payload(options)

    # -----------------------------
    # Extend each canonical topology
    # -----------------------------
    att_range_ext = range(1, max_attacker_troops_extended + 1)
    def_range_ext = range(1, max_defender_troops_extended + 1)

    for edges_key, df_exact in tables_exact.items():
        row_policies = policies_exact[edges_key]
        row_start_states = {rl: _row_label_to_state(rl) for rl in row_policies.keys()}
        edges = list(edges_key) if not isinstance(edges_key, list) else edges_key

        if bool(multi_policy_options):
            # Option-aware plateau inference: exact rows contribute their full set
            # of optimal/near-optimal options instead of one arbitrary chosen action.
            exact_row_policy_options: Dict[str, List[PolicyOption]] = {}
            for rl in row_start_states.keys():
                st = row_start_states[rl]
                att_troops = tuple(int(st.nodes[i].troops) for i in range(num_attacker_nodes))
                def_troops = tuple(int(st.nodes[i].troops) for i in range(num_attacker_nodes, len(st.nodes)))
                exact_row_policy_options[rl] = _eval_row_policy_options_list(
                    edges=edges,
                    att_troops=att_troops,
                    def_troops=def_troops,
                    plateau_policy=None,
                )

            plateau_policy = derive_plateau_policy_options_for_edges(
                row_policy_options=exact_row_policy_options,
                row_start_states=row_start_states,
                num_attacker_nodes=num_attacker_nodes,
                high_min_att_troops=high_min_att_troops,
                max_attacker_troops_exact=max_attacker_troops_exact,
            )
        else:
            plateau_policy = derive_plateau_policy_for_edges(
                row_policies=row_policies,
                row_start_states=row_start_states,
                num_attacker_nodes=num_attacker_nodes,
                high_min_att_troops=high_min_att_troops,
                max_attacker_troops_exact=max_attacker_troops_exact,
            )

        if plateau_policy is not None:
            plateau_policies[edges_key] = plateau_policy

        # Exact DF stays small-ish
        df_exact_small = df_exact.astype(dtype, copy=False)
        exact_index = set(df_exact_small.index)

        # -----------------------------
        # CHUNKED ROW STORE (v1 / v2)
        # -----------------------------
        if fmt in {"chunked_rows_v1", "chunked_rows_v2"}:
            ek = _edges_key_hash(edges_key)

            base_root = Path(chunk_root_dir)
            per_graph_folder = (base_root / ek)
            per_graph_folder.mkdir(parents=True, exist_ok=True)

            row_to_chunk: Dict[str, int] = {}
            chunk_files: List[str] = []
            chunk_index = 0

            chunk_rows_obj_v1: Dict[str, Dict[str, float]] = {}
            chunk_rows_obj_v2: Dict[str, Dict[str, Any]] = {}

            def _flush_v1():
                nonlocal chunk_index, chunk_rows_obj_v1, chunk_files
                if not chunk_rows_obj_v1:
                    return
                fname = _write_chunk_file_v1(per_graph_folder, chunk_index, edges_key, chunk_rows_obj_v1)
                chunk_files.append(fname)
                chunk_index += 1
                chunk_rows_obj_v1 = {}

            def _flush_v2():
                nonlocal chunk_index, chunk_rows_obj_v2, chunk_files
                if not chunk_rows_obj_v2:
                    return
                fname = _write_chunk_file_v2(per_graph_folder, chunk_index, edges_key, chunk_rows_obj_v2)
                chunk_files.append(fname)
                chunk_index += 1
                chunk_rows_obj_v2 = {}

            # If multi-policy options are enabled for V2, store all rows in the
            # chunk store, including exact rows, even when the final option cap is 1.
            # Otherwise exact_df would be checked first by library_io and would hide
            # policy_options_v2 rows from the reader.
            store_policy_options_in_chunks = (
                fmt == "chunked_rows_v2"
                and bool(multi_policy_options)
            )

            for att_troops in product(att_range_ext, repeat=num_attacker_nodes):
                for def_troops in product(def_range_ext, repeat=num_defender_nodes):
                    if (not store_policy_options_in_chunks) and _should_be_in_exact(att_troops, def_troops):
                        continue

                    rl = _row_label(att_troops, def_troops)
                    if (not store_policy_options_in_chunks) and rl in exact_index:
                        continue

                    row_to_chunk[rl] = chunk_index

                    if fmt == "chunked_rows_v1":
                        row = _eval_row_distribution(
                            edges=edges,
                            att_troops=att_troops,
                            def_troops=def_troops,
                            plateau_policy=plateau_policy,
                        )
                        chunk_rows_obj_v1[rl] = row
                        if len(chunk_rows_obj_v1) >= int(chunk_rows):
                            _flush_v1()
                    else:
                        if store_policy_options_in_chunks:
                            # Exact-region rows are solved exactly.  Extended rows
                            # use the option-aware plateau policy when one was
                            # inferred; otherwise they fall back to exact search.
                            pp_for_row = None if _should_be_in_exact(att_troops, def_troops) else plateau_policy
                            chunk_rows_obj_v2[rl] = _eval_row_policy_options_payload(
                                edges=edges,
                                att_troops=att_troops,
                                def_troops=def_troops,
                                plateau_policy=pp_for_row,
                            )
                        else:
                            row = _eval_row_distribution(
                                edges=edges,
                                att_troops=att_troops,
                                def_troops=def_troops,
                                plateau_policy=plateau_policy,
                            )
                            chunk_rows_obj_v2[rl] = _rowdict_to_v2_payload(row)
                        if len(chunk_rows_obj_v2) >= int(chunk_rows):
                            _flush_v2()

            if fmt == "chunked_rows_v1":
                _flush_v1()
                desc_format = "chunked_prob_table_v1"
            else:
                _flush_v2()
                desc_format = "chunked_prob_table_v2_rows_v1"

            chunk_dir_rel = f"{chunk_rel_prefix}/{ek}"

            tables_extended[edges_key] = {
                "format": desc_format,
                "exact_df": None if store_policy_options_in_chunks else df_exact_small,
                "chunks": chunk_files,
                "row_to_chunk": row_to_chunk,
                "chunk_dir": chunk_dir_rel,
                "policy_option_mode": str(policy_option_mode),
                "max_policy_options_per_row": None if max_policy_options_per_row is None else int(max_policy_options_per_row),
                "max_options_per_state": None if max_options_per_state is None else int(max_options_per_state),
                "max_split_depth": None if max_split_depth is None else int(max_split_depth),
                "include_no_gain_in_value": bool(include_no_gain_in_value),
            }
            continue

        # -----------------------------
        # DATAFRAME (small cases)
        # -----------------------------
        rows_dicts: List[Dict[str, float]] = []
        rows_index: List[str] = []

        for att_troops in product(att_range_ext, repeat=num_attacker_nodes):
            for def_troops in product(def_range_ext, repeat=num_defender_nodes):
                if _should_be_in_exact(att_troops, def_troops):
                    continue

                rl = _row_label(att_troops, def_troops)
                if rl in exact_index:
                    continue

                row = _eval_row_distribution(
                    edges=edges,
                    att_troops=att_troops,
                    def_troops=def_troops,
                    plateau_policy=plateau_policy,
                )
                rows_dicts.append(row)
                rows_index.append(rl)

        if rows_index:
            ext_df = (
                pd.DataFrame.from_records(rows_dicts, index=rows_index)
                .astype(dtype, copy=False)
                .fillna(0.0)
            )
            extended_df = pd.concat([df_exact_small, ext_df], axis=0, copy=False, sort=False).fillna(0.0)
        else:
            extended_df = df_exact_small

        extended_df = renormalize_rows(extended_df)
        tables_extended[edges_key] = extended_df

    return tables_extended, plateau_policies




# ---------------------------------------------------------------------
# V2 Array-library helpers (indexed outcomes + arrays)
# ---------------------------------------------------------------------

# Owner encoding for compact storage
OWNER_TO_CODE: Dict[str, np.uint8] = {"D": np.uint8(0), "A": np.uint8(1)}
CODE_TO_OWNER: Dict[int, str] = {0: "D", 1: "A"}


def _stable_outcome_sort_key(
    owners: np.ndarray, troops: np.ndarray
) -> Tuple:
    """
    Deterministic ordering key for outcomes.
    Sort by owners then troops lexicographically (both in local canonical node order).
    """
    # owners, troops are shape (M,)
    return tuple(owners.tolist()) + tuple(troops.tolist())


def global_state_to_owner_troop_arrays(
    state: GlobalState,
    *,
    troops_dtype: Any = np.uint16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GlobalState -> (owners[M] uint8, troops[M] uint16|uint8)
    in local node order.

    owners codes: 0=D, 1=A
    """
    M = len(state.nodes)
    owners = np.empty((M,), dtype=np.uint8)
    troops = np.empty((M,), dtype=troops_dtype)

    for i, n in enumerate(state.nodes):
        owners[i] = OWNER_TO_CODE.get(n.owner, np.uint8(0))
        troops[i] = troops_dtype(n.troops)
    return owners, troops


def label_to_owner_troop_arrays(
    label: str,
    *,
    troops_dtype: Any = np.uint16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a column-label like "(A3,D2,D1)" into (owners[M], troops[M]).
    Uses parse_row_label (same format).
    """
    owners_s, troops_i = parse_row_label(label)
    M = len(owners_s)
    owners = np.empty((M,), dtype=np.uint8)
    troops = np.empty((M,), dtype=troops_dtype)

    for i in range(M):
        owners[i] = OWNER_TO_CODE.get(owners_s[i], np.uint8(0))
        troops[i] = troops_dtype(troops_i[i])
    return owners, troops


def compute_outcome_metrics_from_arrays(
    owners: np.ndarray,
    troops: np.ndarray,
    *,
    num_attacker_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized computation of metric vectors for outcomes.

    owners: (N,M) uint8
    troops: (N,M) uint16|uint8

    Returns:
      is_conquered: (N,) uint8   (1 if no defenders remain AND some troops exist)
      new_territories: (N,) uint8  (# originally-defender nodes that end as A with troops>0)
      final_attacker_troops: (N,) uint32 (sum troops where owner==A and troops>0)
    """
    owners = np.asarray(owners)
    troops = np.asarray(troops)

    # who is attacker after?
    is_A = (owners == OWNER_TO_CODE["A"]) & (troops > 0)
    any_troops = (troops > 0).any(axis=1)

    # "conquered": all nodes owned by A (ignoring troop==0 owner labels) AND any troops exist.
    # If you prefer "no defenders remain" instead, replace with: ~(owners==D & troops>0).any(axis=1)
    no_defenders = ~((owners == OWNER_TO_CODE["D"]) & (troops > 0)).any(axis=1)
    is_conquered = (no_defenders & any_troops).astype(np.uint8)

    # new territories: count nodes that were originally defenders (indices >= num_attacker_nodes)
    # and end as A with troops > 0
    if owners.shape[1] <= num_attacker_nodes:
        new_territories = np.zeros((owners.shape[0],), dtype=np.uint8)
    else:
        new_territories = is_A[:, num_attacker_nodes:].sum(axis=1).astype(np.uint8)

    final_attacker_troops = (troops * is_A).sum(axis=1).astype(np.uint32)

    return is_conquered, new_territories, final_attacker_troops


def build_cdf_from_p(p: np.ndarray) -> np.ndarray:
    """
    Build a float32 CDF suitable for np.searchsorted sampling.
    Ensures last entry is exactly 1.0 to avoid float drift edge cases.
    """
    p = np.asarray(p, dtype=np.float32)
    if p.size == 0:
        return p
    cdf = np.cumsum(p, dtype=np.float32)
    # Defensive: normalize if not quite 1.0 due to rounding
    last = float(cdf[-1])
    if last > 0.0 and abs(last - 1.0) > 1e-6:
        cdf /= np.float32(last)
    cdf[-1] = np.float32(1.0)
    return cdf


def absorbing_dist_to_arrays(
    absorbing_dist: Dict[GlobalState, float],
    *,
    num_attacker_nodes: int,
    troops_dtype: Any = np.uint16,
    store_metrics: bool = True,
    store_cdf: bool = True,
    sort_outcomes: bool = True,
) -> Dict[str, Any]:
    """
    Convert an absorbing distribution dict[GlobalState -> prob] into V2 arrays.

    Returns payload dict with:
      p, owners, troops, [is_conquered, new_territories, final_attacker_troops], [cdf]
    """
    # Filter + normalize
    items = [(s, float(p)) for s, p in absorbing_dist.items() if float(p) > 0.0]
    if not items:
        payload: Dict[str, Any] = {
            "p": np.zeros((0,), dtype=np.float32),
            "owners": np.zeros((0, 0), dtype=np.uint8),
            "troops": np.zeros((0, 0), dtype=troops_dtype),
        }
        if store_metrics:
            payload.update(
                is_conquered=np.zeros((0,), dtype=np.uint8),
                new_territories=np.zeros((0,), dtype=np.uint8),
                final_attacker_troops=np.zeros((0,), dtype=np.uint32),
            )
        if store_cdf:
            payload["cdf"] = np.zeros((0,), dtype=np.float32)
        return payload

    total = float(sum(p for _, p in items))
    if total <= 0.0:
        raise ValueError("absorbing_dist has non-positive total probability mass.")
    inv = 1.0 / total

    # Build outcome arrays
    owners_list: List[np.ndarray] = []
    troops_list: List[np.ndarray] = []
    p_list: List[float] = []

    for s, p in items:
        o, t = global_state_to_owner_troop_arrays(s, troops_dtype=troops_dtype)
        owners_list.append(o)
        troops_list.append(t)
        p_list.append(p * inv)

    owners = np.stack(owners_list, axis=0).astype(np.uint8, copy=False)
    troops = np.stack(troops_list, axis=0).astype(troops_dtype, copy=False)
    p = np.asarray(p_list, dtype=np.float32)

    if sort_outcomes:
        # Deterministic reordering by state signature (owners then troops)
        keys = [(_stable_outcome_sort_key(owners[i], troops[i]), i) for i in range(len(p))]
        keys.sort(key=lambda x: x[0])
        idx = np.array([i for _, i in keys], dtype=np.int64)
        owners = owners[idx]
        troops = troops[idx]
        p = p[idx]

    payload2: Dict[str, Any] = {"p": p, "owners": owners, "troops": troops}

    if store_metrics:
        is_conq, new_terr, fat = compute_outcome_metrics_from_arrays(
            owners, troops, num_attacker_nodes=num_attacker_nodes
        )
        payload2["is_conquered"] = is_conq
        payload2["new_territories"] = new_terr
        payload2["final_attacker_troops"] = fat

    if store_cdf:
        payload2["cdf"] = build_cdf_from_p(p)

    return payload2


def prob_row_dict_labels_to_arrays(
    prob_row: Dict[str, float],
    *,
    num_attacker_nodes: int,
    troops_dtype: Any = np.uint16,
    store_metrics: bool = True,
    store_cdf: bool = True,
    sort_outcomes: bool = True,
) -> Dict[str, Any]:
    """
    Convert legacy row dict { "(A3,D2,...)" : p } into V2 arrays.

    This is useful for:
      - converting an existing DataFrame row to arrays
      - converting chunked rowdict libraries to arrays
    """
    items = [(lbl, float(p)) for lbl, p in prob_row.items() if float(p) > 0.0]
    if not items:
        payload: Dict[str, Any] = {
            "p": np.zeros((0,), dtype=np.float32),
            "owners": np.zeros((0, 0), dtype=np.uint8),
            "troops": np.zeros((0, 0), dtype=troops_dtype),
        }
        if store_metrics:
            payload.update(
                is_conquered=np.zeros((0,), dtype=np.uint8),
                new_territories=np.zeros((0,), dtype=np.uint8),
                final_attacker_troops=np.zeros((0,), dtype=np.uint32),
            )
        if store_cdf:
            payload["cdf"] = np.zeros((0,), dtype=np.float32)
        return payload

    total = float(sum(p for _, p in items))
    if total <= 0.0:
        raise ValueError("prob_row has non-positive total probability mass.")
    inv = 1.0 / total

    owners_list: List[np.ndarray] = []
    troops_list: List[np.ndarray] = []
    p_list: List[float] = []

    for lbl, p in items:
        o, t = label_to_owner_troop_arrays(lbl, troops_dtype=troops_dtype)
        owners_list.append(o)
        troops_list.append(t)
        p_list.append(p * inv)

    owners = np.stack(owners_list, axis=0).astype(np.uint8, copy=False)
    troops = np.stack(troops_list, axis=0).astype(troops_dtype, copy=False)
    p = np.asarray(p_list, dtype=np.float32)

    if sort_outcomes:
        keys = [(_stable_outcome_sort_key(owners[i], troops[i]), i) for i in range(len(p))]
        keys.sort(key=lambda x: x[0])
        idx = np.array([i for _, i in keys], dtype=np.int64)
        owners = owners[idx]
        troops = troops[idx]
        p = p[idx]

    payload2: Dict[str, Any] = {"p": p, "owners": owners, "troops": troops}

    if store_metrics:
        is_conq, new_terr, fat = compute_outcome_metrics_from_arrays(
            owners, troops, num_attacker_nodes=num_attacker_nodes
        )
        payload2["is_conquered"] = is_conq
        payload2["new_territories"] = new_terr
        payload2["final_attacker_troops"] = fat

    if store_cdf:
        payload2["cdf"] = build_cdf_from_p(p)

    return payload2


def save_v2_npz(
    path: Path,
    *,
    format_version: int,
    nA: int,
    nD: int,
    maxA: int,
    maxD: int,
    edges_key_repr: str,
    payload: Dict[str, Any],
) -> None:
    """
    Save a V2 array-library entry to NPZ (compressed).

    Metadata is stored as small numpy scalar/string arrays for convenience.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "format_version": np.array([int(format_version)], dtype=np.int32),
        "nA": np.array([int(nA)], dtype=np.int32),
        "nD": np.array([int(nD)], dtype=np.int32),
        "maxA": np.array([int(maxA)], dtype=np.int32),
        "maxD": np.array([int(maxD)], dtype=np.int32),
        "edges_key_repr": np.array([edges_key_repr], dtype=object),
    }

    # Merge meta + payload and write
    to_save: Dict[str, Any] = {}
    to_save.update(meta)
    to_save.update(payload)

    np.savez_compressed(str(path), **to_save)


def load_v2_npz(path: Path) -> Dict[str, Any]:
    """
    Load a V2 NPZ entry into a dict. Note: np.load returns arrays; metadata
    scalars are stored as 1-element arrays.
    """
    with np.load(str(path), allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}

    # Convenience: unwrap 1-element numeric metadata arrays
    for k in ("format_version", "nA", "nD", "maxA", "maxD"):
        if k in out and isinstance(out[k], np.ndarray) and out[k].size == 1:
            out[k] = int(out[k][0])

    if "edges_key_repr" in out and isinstance(out["edges_key_repr"], np.ndarray) and out["edges_key_repr"].size == 1:
        out["edges_key_repr"] = str(out["edges_key_repr"][0])

    return out


def dataframe_row_to_v2_arrays(
    row: "pd.Series",
    *,
    num_attacker_nodes: int,
    troops_dtype: Any = np.uint16,
    store_metrics: bool = True,
    store_cdf: bool = True,
    sort_outcomes: bool = True,
    drop_zeros: bool = True,
    tol: float = 0.0,
) -> Dict[str, Any]:
    """
    Convert ONE absorption table row (Series) into the V2 array payload.

    Parameters
    ----------
    row:
        A pandas Series representing an absorption distribution over end states.
        - index: column labels like "(A3,D2,...)"
        - values: probabilities
        The Series can be dense or SparseDtype.

    num_attacker_nodes:
        Used for precomputing new_territories + is_conquered metrics.

    troops_dtype:
        dtype for stored troop counts, typically np.uint8 or np.uint16.

    store_metrics / store_cdf:
        Whether to include metric vectors + CDF in the returned payload.

    sort_outcomes:
        If True, reorder outcomes deterministically by (owners,troops) signature.

    drop_zeros:
        If True, ignore zero-prob outcomes.

    tol:
        Drop outcomes with p <= tol (useful if you have tiny numerical noise).

    Returns
    -------
    payload dict containing:
        p: float32 (N,)
        owners: uint8 (N,M)
        troops: troops_dtype (N,M)
        optional metrics arrays
        optional cdf
    """
    # ---- Extract nonzero items without densifying ----
    prob_row: Dict[str, float] = {}

    # Sparse-aware path
    if isinstance(row.dtype, pd.SparseDtype):
        arr = row.array
        if isinstance(arr, pd.arrays.SparseArray):
            idx = arr.sp_index.indices
            vals = arr.sp_values
            cols = row.index
            for j, v in zip(idx, vals):
                fv = float(v)
                if drop_zeros and fv == 0.0:
                    continue
                if fv <= float(tol):
                    continue
                prob_row[str(cols[j])] = fv
        else:
            # Fallback (should be rare)
            for k, v in row.items():
                fv = float(v)
                if drop_zeros and fv == 0.0:
                    continue
                if fv <= float(tol):
                    continue
                prob_row[str(k)] = fv
    else:
        # Dense path
        for k, v in row.items():
            fv = float(v)
            if drop_zeros and fv == 0.0:
                continue
            if fv <= float(tol):
                continue
            prob_row[str(k)] = fv

    # ---- Convert using the existing dict->arrays helper ----
    return prob_row_dict_labels_to_arrays(
        prob_row,
        num_attacker_nodes=num_attacker_nodes,
        troops_dtype=troops_dtype,
        store_metrics=store_metrics,
        store_cdf=store_cdf,
        sort_outcomes=sort_outcomes,
    )

# ---------------------------------------------------------------------
# V2 chunked-rows storage helpers
# ---------------------------------------------------------------------

def make_v2_chunk_descriptor(
    *,
    exact_df: Optional[pd.DataFrame],
    chunk_files: List[str],
    row_to_chunk: Dict[str, int],
    chunk_dir_rel: str,
    metrics_included: bool,
    cdf_included: bool,
) -> Dict[str, Any]:
    """
    Build the runtime descriptor for a V2 chunked row-store.
    `chunk_dir_rel` is relative to the per-graph library folder.
    """
    return {
        "format": "chunked_prob_table_v2_rows_v1",
        "format_version": 2,
        "exact_df": exact_df,
        "chunks": list(chunk_files),
        "row_to_chunk": dict(row_to_chunk),
        "chunk_dir": str(chunk_dir_rel),
        "dtype_meta": {
            "p": "float32",
            "owners": "uint8",
            "troops": "uint16",
            "cdf": "float32",
        },
        "metrics_included": bool(metrics_included),
        "cdf_included": bool(cdf_included),
    }


def write_v2_rowchunk_file(
    folder: Path,
    *,
    chunk_index: int,
    edges_key: Any,
    nA: int,
    nD: int,
    rows_obj: Dict[str, Dict[str, Any]],
) -> str:
    """
    Write one chunk file containing many row_payloads.

    rows_obj: { row_label -> row_payload }
    Returns filename only (not full path), matching your current pattern.
    """
    import pickle

    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"chunk_{chunk_index:06d}.pkl"

    payload = {
        "format": "v2_rowchunk_v1",
        "edges_key": edges_key,
        "nA": int(nA),
        "nD": int(nD),
        "rows": rows_obj,
    }

    with p.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return p.name


def truncate_v2_row_payload(
    row_payload: Dict[str, Any],
    *,
    min_state_prob: float = 0.0,
    max_end_states: Optional[int] = None,
    sort_by_prob_desc: bool = True,
    renormalize: bool = True,
    rebuild_cdf: bool = True,
) -> Dict[str, Any]:
    """
    Apply truncation/capping to a V2 row payload in array form.

    Keeps arrays aligned across keys (p, owners, troops, metrics, cdf).
    Stores total_mass / kept_mass for coverage reporting.
    """
    p = np.asarray(row_payload.get("p", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    if p.size == 0:
        # ensure consistent empties
        out = dict(row_payload)
        out["p"] = p
        out["total_mass"] = 0.0
        out["kept_mass"] = 0.0
        if rebuild_cdf:
            out["cdf"] = np.zeros((0,), dtype=np.float32)
        return out

    total_mass = float(p.sum())
    if total_mass <= 0.0:
        out = dict(row_payload)
        out["p"] = np.zeros((0,), dtype=np.float32)
        out["total_mass"] = float(total_mass)
        out["kept_mass"] = 0.0
        if rebuild_cdf:
            out["cdf"] = np.zeros((0,), dtype=np.float32)
        return out

    idx = np.arange(p.size)

    # threshold
    if min_state_prob and min_state_prob > 0.0:
        mask = p >= float(min_state_prob)
        idx = idx[mask]
        if idx.size == 0:
            out = dict(row_payload)
            # produce empty
            out["p"] = np.zeros((0,), dtype=np.float32)
            out["owners"] = np.zeros((0, row_payload["owners"].shape[1]), dtype=np.uint8)
            out["troops"] = np.zeros((0, row_payload["troops"].shape[1]), dtype=row_payload["troops"].dtype)
            for k in ("is_conquered", "new_territories", "final_attacker_troops", "cdf"):
                if k in out:
                    out[k] = np.zeros((0,), dtype=np.asarray(out[k]).dtype)
            out["total_mass"] = float(total_mass)
            out["kept_mass"] = 0.0
            return out

    # optionally order by prob desc for top-k
    if sort_by_prob_desc:
        idx = idx[np.argsort(p[idx])[::-1]]

    # cap
    if max_end_states is not None and max_end_states > 0 and idx.size > max_end_states:
        idx = idx[: int(max_end_states)]

    kept_p = p[idx]
    kept_mass = float(kept_p.sum())

    # slice all aligned arrays
    out = dict(row_payload)
    out["p"] = kept_p.astype(np.float32, copy=False)

    for k in ("owners", "troops", "is_conquered", "new_territories", "final_attacker_troops"):
        if k in row_payload:
            out[k] = np.asarray(row_payload[k])[idx]

    if renormalize and kept_mass > 0.0:
        out["p"] = (out["p"] / np.float32(kept_mass)).astype(np.float32, copy=False)
        kept_mass = 1.0  # post-renorm mass

    out["total_mass"] = float(total_mass)
    out["kept_mass"] = float(kept_p.sum()) if not renormalize else float(1.0)

    # cdf
    if rebuild_cdf:
        out["cdf"] = build_cdf_from_p(out["p"])

    return out






