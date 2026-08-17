"""Compact shared-cache exact solver for small Risk-like combat graphs.

This module is intentionally experimental and non-invasive: it does not modify
``small_graph_outcome_probabilities.py`` or the current plateau/V2 library
pipeline.  The goal is to provide a faster exact finite solver that can later be
plugged into ``create_library.py`` as a replacement for the per-row recursive
solvers.

Main design differences from the current recursive implementation
-----------------------------------------------------------------
1. A solver object is tied to one topology and owns caches shared by all rows.
2. States are packed into a single Python integer instead of nested dataclasses.
3. Value computation is separated from absorbing-distribution reconstruction.
   This avoids storing a full distribution in the value cache for every internal
   state.
4. Adjacency and combat outcome rows are precomputed/cached outside recursion.

The rules implemented here match the current whole-battle solver semantics:
- legal actions are attacker-owned source nodes with troops > 1 attacking an
  adjacent defender-owned target node;
- a combat action is a whole two-node battle via the F matrix row;
- after conquest, movement is either forced push or a value-maximizing choice
  between "move one" and "push" when the origin has other enemy neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from project_risk.mathematical.small_graph_model.markov_matrix_probabilities import battle_summary
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
    GlobalState,
    NodeState,
    PolicyOption,
    default_local_value_tolerances,
    encode_state_label,
    initial_state_generic,
    value_relation,
)


Action = Tuple[int, int]
ValueTuple = Tuple[float, ...]
StateInt = int
Distribution = Dict[StateInt, float]


class ExactSolverLimitReached(RuntimeError):
    """Controlled interruption used by validation-oriented exact solves."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = str(status)


@dataclass
class ExactSolverStats:
    """Instrumentation counters for one ``CompactExactTopologySolver``."""

    value_evals: int = 0
    value_cache_hits: int = 0
    dist_evals: int = 0
    dist_cache_hits: int = 0
    action_value_evals: int = 0
    action_dist_evals: int = 0
    combat_lookup_misses: int = 0
    combat_outcome_branches: int = 0
    movement_choice_evals: int = 0
    max_depth_seen: int = 0

    rows_evaluated: int = 0
    root_options_evaluated: int = 0
    state_options_evaluated: int = 0
    state_options_cache_hits: int = 0
    state_options_pruned: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "value_evals": int(self.value_evals),
            "value_cache_hits": int(self.value_cache_hits),
            "dist_evals": int(self.dist_evals),
            "dist_cache_hits": int(self.dist_cache_hits),
            "action_value_evals": int(self.action_value_evals),
            "action_dist_evals": int(self.action_dist_evals),
            "combat_lookup_misses": int(self.combat_lookup_misses),
            "combat_outcome_branches": int(self.combat_outcome_branches),
            "movement_choice_evals": int(self.movement_choice_evals),
            "max_depth_seen": int(self.max_depth_seen),
            "rows_evaluated": int(self.rows_evaluated),
            "root_options_evaluated": int(self.root_options_evaluated),
            "state_options_evaluated": int(self.state_options_evaluated),
            "state_options_cache_hits": int(self.state_options_cache_hits),
            "state_options_pruned": int(self.state_options_pruned),
        }


@dataclass(frozen=True)
class ExactStateResult:
    """Exact value and absorbing distribution for one start state."""

    state: StateInt
    value: ValueTuple
    absorbing_dist: Distribution
    root_action: Optional[Action]


@dataclass(frozen=True)
class CompactPolicyOption:
    """Root policy option in compact-state form."""

    option_id: int
    root_action: Optional[Action]
    value: ValueTuple
    absorbing_dist: Distribution
    split_metadata: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class CompactStatePolicyOption:
    """Internal state-set option.

    These options are cached per compact state and may share child-state option
    computations. The public export still flattens them into policy_options_v2.
    """

    option_id: int
    action: Optional[Action]
    value: ValueTuple
    absorbing_dist: Distribution
    child_refs: Tuple[Any, ...] = ()
    split_metadata: Tuple[Any, ...] = ()


@dataclass
class RowBuildResult:
    """Rows produced by evaluating a whole troop grid with one shared cache."""

    rows: Dict[str, Dict[str, float]]
    stats: Dict[str, Any]



def infer_combat_df_limits(combat_df: pd.DataFrame) -> Tuple[int, int]:
    """Infer max available attackers/defenders represented in an F_df table."""
    max_a = 0
    max_d = 0
    for lbl in combat_df.index:
        s = str(lbl).strip().strip("()")
        if not s:
            continue
        a_s, d_s = s.split(",")
        max_a = max(max_a, int(a_s))
        max_d = max(max_d, int(d_s))
    return max_a, max_d



def combat_df_for_total_troops(
    *,
    total_nodes: int,
    troop_cap: int,
    defender_cap: Optional[int] = None,
    as_dataframes: bool = True,
) -> pd.DataFrame:
    """Build a combat table large enough for exact solving at a finite cap.

    The largest available attacking force in one source can never exceed
    ``total_nodes * troop_cap - 1``.  Defender nodes never gain troops while they
    remain defenders, so ``defender_cap`` defaults to ``troop_cap``.
    """
    if total_nodes <= 0:
        raise ValueError("total_nodes must be positive")
    if troop_cap <= 0:
        raise ValueError("troop_cap must be positive")
    if defender_cap is None:
        defender_cap = troop_cap
    a_max = max(1, int(total_nodes) * int(troop_cap) - 1)
    d_max = max(1, int(defender_cap))
    return battle_summary(a_max, d_max, as_dataframes=as_dataframes)["F_df"]


def combat_df_for_caps(
    *,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops: int,
    max_defender_troops: int,
    as_dataframes: bool = True,
) -> pd.DataFrame:
    """Build a combat table large enough for exact solving over a row grid.

    For a grid with attacker nodes capped by ``max_attacker_troops`` and
    defender nodes capped by ``max_defender_troops``, the total initial troops
    can be as large as ``nA*maxA + nD*maxD``.  A source may later contain most
    of those troops, so the combat table needs available-attacker rows up to
    ``total_initial_troops - 1``.
    """
    nA = int(num_attacker_nodes)
    nD = int(num_defender_nodes)
    maxA = int(max_attacker_troops)
    maxD = int(max_defender_troops)
    if nA <= 0 or nD <= 0 or maxA <= 0 or maxD <= 0:
        raise ValueError("nA, nD, maxA and maxD must all be positive")
    total_initial = nA * maxA + nD * maxD
    return battle_summary(max(1, total_initial - 1), maxD, as_dataframes=as_dataframes)["F_df"]


class CompactExactTopologySolver:
    """Exact shared-cache solver for one small graph topology.

    Parameters
    ----------
    edges:
        Iterable of undirected graph edges ``(u, v)`` using local node indices.
    num_attacker_nodes / num_defender_nodes:
        Initial ownership partition.  Nodes ``0..nA-1`` start attacker-owned;
        nodes ``nA..nA+nD-1`` start defender-owned.
    combat_df:
        Whole-battle absorption table returned as ``battle_summary(...)["F_df"]``.
        It must contain rows up to the maximum possible available attacking
        force and defender troop count encountered by the finite cap.
    utility_mode:
        ``"legacy"`` reproduces the older ``(P_success, troop_score)`` value.
        ``"local"`` uses ``(new_territories, final_attacker_troops,
        local_conquest)`` and optional no-gain component.
    max_total_troops:
        Used only to choose the state bit packing width.  If omitted, it is
        inferred from the combat table attacker limit + 1, which is safe for
        ordinary use.
    cache_distributions:
        If True, cache reconstructed absorbing distributions.  This speeds
        repeated row extraction but may use substantial memory.  Value cache is
        always shared.
    """

    def __init__(
        self,
        *,
        edges: Iterable[Tuple[int, int]],
        num_attacker_nodes: int,
        num_defender_nodes: int,
        combat_df: pd.DataFrame,
        utility_mode: str = "local",
        include_no_gain_in_value: bool = False,
        value_tolerances: Optional[Tuple[float, ...]] = None,
        max_total_troops: Optional[int] = None,
        cache_distributions: bool = True,
        sort_actions: bool = True,
        max_states: Optional[int] = None,
        max_runtime_seconds: Optional[float] = None,
        max_cache_entries: Optional[int] = None,
        max_memory_estimate_bytes: Optional[int] = None,
    ) -> None:
        self.nA = int(num_attacker_nodes)
        self.nD = int(num_defender_nodes)
        self.n = self.nA + self.nD
        if self.n <= 0:
            raise ValueError("total node count must be positive")
        if self.nA <= 0 or self.nD <= 0:
            raise ValueError("both attacker and defender node counts must be positive")

        self.edges: Tuple[Tuple[int, int], ...] = tuple(
            sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in edges)
        )
        self.utility_mode = str(utility_mode or "local").lower().strip()
        if self.utility_mode not in {"local", "legacy"}:
            raise ValueError("utility_mode must be 'local' or 'legacy'")
        self.include_no_gain_in_value = bool(include_no_gain_in_value)
        if value_tolerances is None:
            if self.utility_mode == "local":
                value_tolerances = default_local_value_tolerances(self.include_no_gain_in_value)
            else:
                value_tolerances = (1e-9, 1e-9)
        self.value_tolerances = tuple(float(x) for x in value_tolerances)
        self.cache_distributions = bool(cache_distributions)
        self.sort_actions = bool(sort_actions)
        self.max_states = None if max_states is None else max(1, int(max_states))
        self.max_runtime_seconds = (
            None
            if max_runtime_seconds is None
            else max(0.0, float(max_runtime_seconds))
        )
        self.max_cache_entries = (
            None if max_cache_entries is None else max(1, int(max_cache_entries))
        )
        self.max_memory_estimate_bytes = (
            None
            if max_memory_estimate_bytes is None
            else max(1, int(max_memory_estimate_bytes))
        )
        self._evaluation_started = time.perf_counter()

        self._attacker_initial_mask = (1 << self.nA) - 1
        self._all_owned_mask = (1 << self.n) - 1

        self._neighbors: List[Tuple[int, ...]] = [tuple() for _ in range(self.n)]
        nbs = [set() for _ in range(self.n)]
        for u, v in self.edges:
            if not (0 <= u < self.n and 0 <= v < self.n):
                raise ValueError(f"edge {(u, v)} outside node range 0..{self.n - 1}")
            if u == v:
                raise ValueError(f"self-loop edge {(u, v)} is not allowed")
            nbs[u].add(v)
            nbs[v].add(u)
        self._neighbors = [tuple(sorted(x)) for x in nbs]

        self.combat_df = combat_df
        self.combat_A_max, self.combat_D_max = infer_combat_df_limits(combat_df)
        if max_total_troops is None:
            max_total_troops = self.combat_A_max + 1
        self.max_total_troops = int(max_total_troops)
        self._troop_bits = max(1, int(self.max_total_troops).bit_length())
        self._troop_mask = (1 << self._troop_bits) - 1

        self._combat_cache: Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]] = {}
        self._value_cache: Dict[StateInt, Tuple[ValueTuple, Optional[Action]]] = {}
        self._dist_cache: Dict[StateInt, Distribution] = {}
        self._state_options_cache: Dict[Tuple[StateInt, int, int], Tuple[CompactStatePolicyOption, ...]] = {}
        self._leaf_distance_cache: Dict[Tuple[StateInt, int], int] = {}
        self.stats = ExactSolverStats()

    def cache_entry_count(self) -> int:
        """Return the number of top-level entries in all solver caches."""
        return int(
            len(self._combat_cache)
            + len(self._value_cache)
            + len(self._dist_cache)
            + len(self._state_options_cache)
            + len(self._leaf_distance_cache)
        )

    def estimated_cache_bytes(self) -> int:
        """Return a deterministic, explicitly approximate cache-size estimate.

        The estimate intentionally avoids recursive ``getsizeof`` walks during
        solving. Coefficients include Python container/key/value overhead and
        are suitable for validation safeguards and relative reporting, not RSS
        accounting.
        """
        distribution_payload_entries = sum(len(value) for value in self._dist_cache.values())
        state_policy_options = sum(len(value) for value in self._state_options_cache.values())
        combat_outcomes = sum(len(value) for value in self._combat_cache.values())
        return int(
            512
            + 192 * len(self._value_cache)
            + 128 * len(self._dist_cache)
            + 80 * distribution_payload_entries
            + 160 * len(self._state_options_cache)
            + 320 * state_policy_options
            + 128 * len(self._combat_cache)
            + 64 * combat_outcomes
            + 128 * len(self._leaf_distance_cache)
        )

    def _check_validation_limits(self, *, before_new_value_state: bool = False) -> None:
        if (
            self.max_runtime_seconds is not None
            and time.perf_counter() - self._evaluation_started
            >= self.max_runtime_seconds
        ):
            raise ExactSolverLimitReached(
                "runtime_limit",
                f"exact solver exceeded {self.max_runtime_seconds:g} seconds",
            )
        if (
            before_new_value_state
            and self.max_states is not None
            and self.stats.value_evals >= self.max_states
        ):
            raise ExactSolverLimitReached(
                "state_limit",
                f"exact solver reached max_states={self.max_states}",
            )
        if (
            self.max_cache_entries is not None
            and self.cache_entry_count() >= self.max_cache_entries
        ):
            raise ExactSolverLimitReached(
                "memory_limit",
                f"exact solver reached max_cache_entries={self.max_cache_entries}",
            )
        if (
            self.max_memory_estimate_bytes is not None
            and self.estimated_cache_bytes() >= self.max_memory_estimate_bytes
        ):
            raise ExactSolverLimitReached(
                "memory_limit",
                "exact solver reached max_memory_estimate_bytes="
                f"{self.max_memory_estimate_bytes}",
            )

    # ------------------------------------------------------------------
    # State encoding/decoding
    # ------------------------------------------------------------------

    def pack_state(self, owner_mask: int, troops: Sequence[int]) -> StateInt:
        if len(troops) != self.n:
            raise ValueError(f"expected {self.n} troop counts, got {len(troops)}")
        owner_mask = int(owner_mask) & self._all_owned_mask
        s = owner_mask
        shift0 = self.n
        for i, t in enumerate(troops):
            ti = int(t)
            if ti < 0:
                raise ValueError(f"troops cannot be negative: node {i} has {ti}")
            if ti > self._troop_mask:
                raise ValueError(
                    f"node {i} troop count {ti} does not fit in {self._troop_bits} bits; "
                    f"increase max_total_troops"
                )
            s |= ti << (shift0 + self._troop_bits * i)
        return int(s)

    def owner_mask(self, state: StateInt) -> int:
        return int(state) & self._all_owned_mask

    def troop_at(self, state: StateInt, i: int) -> int:
        return (int(state) >> (self.n + self._troop_bits * int(i))) & self._troop_mask

    def troops_tuple(self, state: StateInt) -> Tuple[int, ...]:
        s = int(state)
        return tuple(
            (s >> (self.n + self._troop_bits * i)) & self._troop_mask
            for i in range(self.n)
        )

    def replace_troops_and_owner(
        self,
        state: StateInt,
        *,
        owner_mask: Optional[int] = None,
        updates: Sequence[Tuple[int, int]] = (),
    ) -> StateInt:
        troops = list(self.troops_tuple(state))
        for i, t in updates:
            troops[int(i)] = int(t)
        if owner_mask is None:
            owner_mask = self.owner_mask(state)
        return self.pack_state(owner_mask, troops)

    def initial_state(self, attacker_troops: Sequence[int], defender_troops: Sequence[int]) -> StateInt:
        if len(attacker_troops) != self.nA:
            raise ValueError(f"expected {self.nA} attacker troop counts")
        if len(defender_troops) != self.nD:
            raise ValueError(f"expected {self.nD} defender troop counts")
        return self.pack_state(
            self._attacker_initial_mask,
            tuple(int(x) for x in attacker_troops) + tuple(int(x) for x in defender_troops),
        )

    def state_to_global_state(self, state: StateInt) -> GlobalState:
        mask = self.owner_mask(state)
        troops = self.troops_tuple(state)
        nodes = []
        for i, t in enumerate(troops):
            owner = "A" if ((mask >> i) & 1) else "D"
            nodes.append(NodeState(owner, int(t)))
        return GlobalState(nodes=tuple(nodes))

    def state_label(self, state: StateInt) -> str:
        return encode_state_label(self.state_to_global_state(state))

    def row_label(self, attacker_troops: Sequence[int], defender_troops: Sequence[int]) -> str:
        return "(" + ",".join(
            [f"A{int(t)}" for t in attacker_troops]
            + [f"D{int(t)}" for t in defender_troops]
        ) + ")"

    # ------------------------------------------------------------------
    # Rules/helpers
    # ------------------------------------------------------------------

    def possible_actions(self, state: StateInt) -> Tuple[Action, ...]:
        mask = self.owner_mask(state)
        actions: List[Action] = []
        for u in range(self.n):
            if ((mask >> u) & 1) == 0:
                continue
            if self.troop_at(state, u) <= 1:
                continue
            for v in self._neighbors[u]:
                if ((mask >> v) & 1) == 0:
                    actions.append((u, v))
        if self.sort_actions:
            actions.sort()
        return tuple(actions)

    def is_absorbing(self, state: StateInt) -> bool:
        mask = self.owner_mask(state)
        if mask == self._all_owned_mask:
            return True
        return len(self.possible_actions(state)) == 0

    def terminal_value(self, state: StateInt) -> ValueTuple:
        mask = self.owner_mask(state)
        troops = self.troops_tuple(state)
        conquered = mask == self._all_owned_mask and sum(troops) > 0

        if self.utility_mode == "legacy":
            if not conquered:
                return (0.0, 0.0)
            troop_score = sum(troops[i] for i in range(self.nA, self.n))
            return (1.0, float(troop_score))

        new_territories = 0
        final_attacker_troops = 0
        for i, t in enumerate(troops):
            if ((mask >> i) & 1) and t > 0:
                final_attacker_troops += int(t)
                if i >= self.nA:
                    new_territories += 1
        local_conquest = 1.0 if conquered else 0.0
        if self.include_no_gain_in_value:
            no_gain = 1.0 if new_territories == 0 else 0.0
            return (
                float(new_territories),
                -float(no_gain),
                float(final_attacker_troops),
                float(local_conquest),
            )
        return (float(new_territories), float(final_attacker_troops), float(local_conquest))

    def better_value(self, v1: ValueTuple, v2: ValueTuple) -> bool:
        return value_relation(v1, v2, self.value_tolerances) == 1

    def equivalent_value(self, v1: ValueTuple, v2: ValueTuple) -> bool:
        return value_relation(v1, v2, self.value_tolerances) == 0

    def _add_scaled_value(self, acc: ValueTuple, p: float, val: ValueTuple) -> ValueTuple:
        return tuple(float(a) + float(p) * float(b) for a, b in zip(acc, val))

    def combat_outcomes(self, a_avail: int, d: int) -> Tuple[Tuple[int, int, float], ...]:
        key = (int(a_avail), int(d))
        cached = self._combat_cache.get(key)
        if cached is not None:
            return cached
        self.stats.combat_lookup_misses += 1
        a_avail, d = key
        if a_avail <= 0 or d <= 0:
            raise ValueError(f"invalid combat lookup: a_avail={a_avail}, d={d}")
        if a_avail > self.combat_A_max or d > self.combat_D_max:
            raise KeyError(
                f"combat_df is too small for lookup ({a_avail},{d}). "
                f"Available limits are A≤{self.combat_A_max}, D≤{self.combat_D_max}. "
                f"For exact cap solving, build combat_df with combat_df_for_total_troops(...)."
            )
        row_label = f"({a_avail},{d})"
        row = self.combat_df.loc[row_label]
        out: List[Tuple[int, int, float]] = []
        for col_label, p in row.items():
            fp = float(p)
            if fp <= 0.0:
                continue
            a_s, d_s = str(col_label).strip().strip("()").split(",")
            out.append((int(a_s), int(d_s), fp))
        ans = tuple(out)
        self._combat_cache[key] = ans
        return ans

    def _other_enemy_neighbors_at_origin(self, state: StateInt, u: int, v: int) -> bool:
        mask = self.owner_mask(state)
        for w in self._neighbors[u]:
            if w != v and ((mask >> w) & 1) == 0:
                return True
        return False

    # ------------------------------------------------------------------
    # Exact value recursion
    # ------------------------------------------------------------------

    def evaluate_value(self, state: StateInt, *, _depth: int = 0) -> Tuple[ValueTuple, Optional[Action]]:
        cached = self._value_cache.get(state)
        if cached is not None:
            self.stats.value_cache_hits += 1
            return cached

        self._check_validation_limits(before_new_value_state=True)
        self.stats.value_evals += 1
        if _depth > self.stats.max_depth_seen:
            self.stats.max_depth_seen = int(_depth)

        if self.is_absorbing(state):
            result = (self.terminal_value(state), None)
            self._value_cache[state] = result
            return result

        actions = self.possible_actions(state)
        if not actions:
            result = (self.terminal_value(state), None)
            self._value_cache[state] = result
            return result

        best_value: Optional[ValueTuple] = None
        best_action: Optional[Action] = None
        for action in actions:
            val = self.evaluate_action_value(state, action, _depth=_depth)
            if best_value is None or self.better_value(val, best_value):
                best_value = val
                best_action = action

        assert best_value is not None and best_action is not None
        result = (best_value, best_action)
        self._value_cache[state] = result
        return result

    def evaluate_action_value(self, state: StateInt, action: Action, *, _depth: int = 0) -> ValueTuple:
        self._check_validation_limits()
        self.stats.action_value_evals += 1
        u, v = action
        T_u = self.troop_at(state, u)
        T_v = self.troop_at(state, v)
        if T_u <= 1:
            raise ValueError(f"cannot attack from node {u} with {T_u} troops")
        outcomes = self.combat_outcomes(T_u - 1, T_v)
        self.stats.combat_outcome_branches += len(outcomes)

        # value tuple length follows terminal objective
        acc = tuple(0.0 for _ in self.terminal_value(state))
        old_mask = self.owner_mask(state)

        for a_end, d_end, p in outcomes:
            origin_after = 1 + int(a_end)
            if d_end > 0:
                child = self.replace_troops_and_owner(
                    state,
                    owner_mask=old_mask,
                    updates=((u, origin_after), (v, int(d_end))),
                )
                child_val, _ = self.evaluate_value(child, _depth=_depth + 1)
            else:
                child = self._best_movement_child_after_conquest(
                    state,
                    u=u,
                    v=v,
                    total_at_u_before_move=origin_after,
                    _depth=_depth,
                )
                child_val, _ = self.evaluate_value(child, _depth=_depth + 1)
            acc = self._add_scaled_value(acc, p, child_val)
        return acc

    def _best_movement_child_after_conquest(
        self,
        state: StateInt,
        *,
        u: int,
        v: int,
        total_at_u_before_move: int,
        _depth: int,
    ) -> StateInt:
        old_mask = self.owner_mask(state)
        new_mask = old_mask | (1 << v)

        if not self._other_enemy_neighbors_at_origin(state, u, v):
            return self.replace_troops_and_owner(
                state,
                owner_mask=new_mask,
                updates=((u, 1), (v, int(total_at_u_before_move) - 1)),
            )

        self.stats.movement_choice_evals += 1
        keep_state = self.replace_troops_and_owner(
            state,
            owner_mask=new_mask,
            updates=((u, int(total_at_u_before_move) - 1), (v, 1)),
        )
        push_state = self.replace_troops_and_owner(
            state,
            owner_mask=new_mask,
            updates=((u, 1), (v, int(total_at_u_before_move) - 1)),
        )
        keep_val, _ = self.evaluate_value(keep_state, _depth=_depth + 1)
        push_val, _ = self.evaluate_value(push_state, _depth=_depth + 1)

        # Match current implementation's tie behavior: choose push unless keep is
        # strictly better.
        if self.better_value(keep_val, push_val):
            return keep_state
        return push_state

    # ------------------------------------------------------------------
    # Distribution reconstruction under cached optimal policy
    # ------------------------------------------------------------------

    def absorbing_distribution(self, state: StateInt) -> Distribution:
        self._check_validation_limits()
        if self.cache_distributions:
            cached = self._dist_cache.get(state)
            if cached is not None:
                self.stats.dist_cache_hits += 1
                return cached

        self.stats.dist_evals += 1
        if self.is_absorbing(state):
            dist = {state: 1.0}
            if self.cache_distributions:
                self._dist_cache[state] = dist
            return dist

        _val, action = self.evaluate_value(state)
        if action is None:
            dist = {state: 1.0}
        else:
            dist = self.action_distribution(state, action)

        if self.cache_distributions:
            self._dist_cache[state] = dist
        return dist

    def action_distribution(self, state: StateInt, action: Action) -> Distribution:
        self._check_validation_limits()
        self.stats.action_dist_evals += 1
        u, v = action
        T_u = self.troop_at(state, u)
        T_v = self.troop_at(state, v)
        outcomes = self.combat_outcomes(T_u - 1, T_v)
        old_mask = self.owner_mask(state)
        out: Distribution = {}

        for a_end, d_end, p in outcomes:
            origin_after = 1 + int(a_end)
            if d_end > 0:
                child = self.replace_troops_and_owner(
                    state,
                    owner_mask=old_mask,
                    updates=((u, origin_after), (v, int(d_end))),
                )
            else:
                child = self._best_movement_child_after_conquest(
                    state,
                    u=u,
                    v=v,
                    total_at_u_before_move=origin_after,
                    _depth=0,
                )
            child_dist = self.absorbing_distribution(child)
            for s2, ps in child_dist.items():
                out[s2] = out.get(s2, 0.0) + float(p) * float(ps)
        # Keep the raw product of rounded combat probabilities here so that
        # recursive distributions match the legacy exact solver. Row/payload
        # conversion helpers normalize at the final output boundary.
        return out

    @staticmethod
    def normalize_distribution(dist: Distribution) -> Distribution:
        total = float(sum(dist.values()))
        if total <= 0.0:
            return dict(dist)
        return {int(s): float(p) / total for s, p in dist.items() if float(p) > 0.0}

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def evaluate_start(
        self,
        attacker_troops: Sequence[int],
        defender_troops: Sequence[int],
    ) -> ExactStateResult:
        state = self.initial_state(attacker_troops, defender_troops)
        value, root_action = self.evaluate_value(state)
        dist = self.absorbing_distribution(state)
        self.stats.rows_evaluated += 1
        return ExactStateResult(state=state, value=value, absorbing_dist=dist, root_action=root_action)

    def root_policy_options(
        self,
        attacker_troops: Sequence[int],
        defender_troops: Sequence[int],
        *,
        max_policy_options: Optional[int] = 2,
    ) -> List[CompactPolicyOption]:
        """Return exact near-best root-action options with optimal continuation.

        This mirrors ``explore_root_policy_options_for_graph``.  It does not yet
        implement the deeper state-set option propagation mode.
        """
        state = self.initial_state(attacker_troops, defender_troops)
        if self.is_absorbing(state):
            return [CompactPolicyOption(0, None, self.terminal_value(state), {state: 1.0})]

        actions = self.possible_actions(state)
        if not actions:
            return [CompactPolicyOption(0, None, self.terminal_value(state), {state: 1.0})]

        candidates: List[CompactPolicyOption] = []
        for action in actions:
            val = self.evaluate_action_value(state, action)
            dist = self.action_distribution(state, action)
            candidates.append(CompactPolicyOption(-1, action, val, dist))

        best = candidates[0].value
        for opt in candidates[1:]:
            if self.better_value(opt.value, best):
                best = opt.value

        kept = [opt for opt in candidates if self.equivalent_value(opt.value, best)]
        kept.sort(key=lambda o: (o.value, o.root_action or (-1, -1)), reverse=True)
        if max_policy_options is not None:
            kept = kept[: max(1, int(max_policy_options))]
        self.stats.root_options_evaluated += 1
        return [CompactPolicyOption(i, o.root_action, o.value, o.absorbing_dist) for i, o in enumerate(kept)]

    # ------------------------------------------------------------------
    # State-set policy options
    # ------------------------------------------------------------------

    def _child_state_after_combat_outcome(
        self,
        state: StateInt,
        action: Action,
        a_end: int,
        d_end: int,
    ) -> StateInt:
        u, v = action
        old_mask = self.owner_mask(state)
        origin_after = 1 + int(a_end)
        if int(d_end) > 0:
            return self.replace_troops_and_owner(
                state,
                owner_mask=old_mask,
                updates=((u, origin_after), (v, int(d_end))),
            )
        return self._best_movement_child_after_conquest(
            state,
            u=u,
            v=v,
            total_at_u_before_move=origin_after,
            _depth=0,
        )

    def _state_leaf_distance(self, state: StateInt, cap: int) -> int:
        """Conservative capped distance from a state to terminal play.

        This is used only to annotate whether a split is near the leaf side.
        Values greater than cap are returned as cap + 1.
        """
        cap = int(cap)
        key = (int(state), cap)
        cached = self._leaf_distance_cache.get(key)
        if cached is not None:
            return cached
        if self.is_absorbing(state):
            self._leaf_distance_cache[key] = 0
            return 0
        if cap <= 0:
            self._leaf_distance_cache[key] = 1
            return 1

        best = cap + 1
        for action in self.possible_actions(state):
            try:
                dist = self._action_leaf_distance(state, action, cap)
            except Exception:
                dist = cap + 1
            if dist < best:
                best = dist
        best = min(best, cap + 1)
        self._leaf_distance_cache[key] = best
        return best

    def _action_leaf_distance(self, state: StateInt, action: Action, cap: int) -> int:
        if self.is_absorbing(state):
            return 0
        if cap <= 0:
            return 1
        u, v = action
        T_u = self.troop_at(state, u)
        T_v = self.troop_at(state, v)
        child_max = 0
        for a_end, d_end, _p in self.combat_outcomes(T_u - 1, T_v):
            child = self._child_state_after_combat_outcome(state, action, a_end, d_end)
            child_max = max(child_max, self._state_leaf_distance(child, cap - 1))
            if child_max >= cap:
                return cap + 1
        return min(1 + child_max, cap + 1)

    def _distribution_value(self, dist: Distribution) -> ValueTuple:
        acc = tuple(0.0 for _ in self.terminal_value(next(iter(dist), 0)))
        for state, p in dist.items():
            acc = self._add_scaled_value(acc, float(p), self.terminal_value(int(state)))
        return acc

    def _distribution_signature(
        self,
        dist: Distribution,
        *,
        ndigits: int = 12,
    ) -> Tuple[Tuple[int, float], ...]:
        norm = self.normalize_distribution(dist)
        return tuple(
            (int(s), round(float(p), int(ndigits)))
            for s, p in sorted(norm.items())
            if float(p) > 0.0
        )

    def _policy_trace_signature(self, opt: CompactStatePolicyOption) -> Tuple[Any, ...]:
        return (
            tuple(opt.action) if opt.action is not None else None,
            tuple(opt.split_metadata or ()),
            tuple(opt.child_refs or ()),
        )

    def _dedupe_and_prune_state_options(
        self,
        candidates: Sequence[CompactStatePolicyOption],
        *,
        max_options: Optional[int],
    ) -> Tuple[CompactStatePolicyOption, ...]:
        if not candidates:
            return tuple()

        best = candidates[0].value
        for opt in candidates[1:]:
            if self.better_value(opt.value, best):
                best = opt.value

        near_best = [opt for opt in candidates if self.equivalent_value(opt.value, best)]
        if not near_best:
            near_best = [max(candidates, key=lambda o: o.value)]

        deduped: List[CompactStatePolicyOption] = []
        seen = set()
        for opt in sorted(
            near_best,
            key=lambda o: (o.value, str(self._policy_trace_signature(o))),
            reverse=True,
        ):
            key = (self._distribution_signature(opt.absorbing_dist), self._policy_trace_signature(opt))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(opt)

        if max_options is not None:
            before = len(deduped)
            deduped = deduped[: max(1, int(max_options))]
            self.stats.state_options_pruned += max(0, before - len(deduped))

        return tuple(
            CompactStatePolicyOption(
                option_id=i,
                action=opt.action,
                value=opt.value,
                absorbing_dist=opt.absorbing_dist,
                child_refs=opt.child_refs,
                split_metadata=opt.split_metadata,
            )
            for i, opt in enumerate(deduped)
        )

    def _combine_state_option_branches(
        self,
        branches: Sequence[Tuple[float, StateInt, Sequence[CompactStatePolicyOption]]],
        choices: Sequence[int],
    ) -> Tuple[Distribution, Tuple[Any, ...], Tuple[Any, ...]]:
        out: Distribution = {}
        child_refs: List[Any] = []
        split_meta: List[Any] = []
        for branch_i, (p, child, child_options) in enumerate(branches):
            choice_i = int(choices[branch_i])
            child_opt = child_options[choice_i]
            child_refs.append((int(child), int(child_opt.option_id)))
            if choice_i != 0 or child_opt.split_metadata:
                split_meta.append(
                    (
                        "child_option",
                        int(branch_i),
                        int(child),
                        int(choice_i),
                        tuple(child_opt.split_metadata or ()),
                    )
                )
            for s2, ps in child_opt.absorbing_dist.items():
                out[int(s2)] = out.get(int(s2), 0.0) + float(p) * float(ps)
        return out, tuple(child_refs), tuple(split_meta)

    def _optimal_state_policy_option(self, state: StateInt) -> CompactStatePolicyOption:
        value, action = self.evaluate_value(state)
        dist = self.absorbing_distribution(state)
        return CompactStatePolicyOption(
            option_id=0,
            action=action,
            value=value,
            absorbing_dist=dist,
            split_metadata=(("optimal",),),
        )

    def state_policy_options(
        self,
        state: StateInt,
        *,
        max_leaf_split_depth: int = 1,
        max_options_per_state: Optional[int] = 2,
    ) -> Tuple[CompactStatePolicyOption, ...]:
        """Return bounded shared state-set options for one compact state.

        ``max_leaf_split_depth`` is interpreted as leaf-side split depth. This
        first implementation introduces and propagates alternatives from states
        near terminal play, while memoizing options by compact state so common
        subtrees are shared across root candidates.
        """
        depth = max(0, int(max_leaf_split_depth))
        cap = max(1, int(max_options_per_state or 1))
        key = (int(state), depth, cap)
        cached = self._state_options_cache.get(key)
        if cached is not None:
            self.stats.state_options_cache_hits += 1
            return cached

        self.stats.state_options_evaluated += 1

        if depth <= 0:
            ans = (self._optimal_state_policy_option(state),)
            self._state_options_cache[key] = ans
            return ans

        if self.is_absorbing(state):
            ans = (
                CompactStatePolicyOption(
                    option_id=0,
                    action=None,
                    value=self.terminal_value(state),
                    absorbing_dist={int(state): 1.0},
                    split_metadata=(("terminal", int(state)),),
                ),
            )
            self._state_options_cache[key] = ans
            return ans

        candidates: List[CompactStatePolicyOption] = []
        for action in self.possible_actions(state):
            u, v = action
            T_u = self.troop_at(state, u)
            T_v = self.troop_at(state, v)
            branches: List[Tuple[float, StateInt, Sequence[CompactStatePolicyOption]]] = []
            for a_end, d_end, p in self.combat_outcomes(T_u - 1, T_v):
                child = self._child_state_after_combat_outcome(state, action, a_end, d_end)
                child_opts = self.state_policy_options(
                    child,
                    max_leaf_split_depth=depth,
                    max_options_per_state=cap,
                )
                branches.append((float(p), child, child_opts))

            if not branches:
                continue

            primary_choices = tuple(0 for _ in branches)
            choice_sets = [primary_choices]
            for branch_i, (_p, _child, child_opts) in enumerate(branches):
                for alt_i in range(1, min(len(child_opts), cap)):
                    choices = list(primary_choices)
                    choices[branch_i] = alt_i
                    choice_sets.append(tuple(choices))

            action_leaf_distance = self._action_leaf_distance(state, action, depth)
            for choices in choice_sets:
                dist, child_refs, split_meta = self._combine_state_option_branches(branches, choices)
                if action_leaf_distance <= depth:
                    split_meta = (("leaf_action", tuple(action), int(action_leaf_distance)),) + split_meta
                candidates.append(
                    CompactStatePolicyOption(
                        option_id=-1,
                        action=action,
                        value=self._distribution_value(dist),
                        absorbing_dist=dist,
                        child_refs=child_refs,
                        split_metadata=(("action", tuple(action)),) + tuple(split_meta),
                    )
                )

        ans = self._dedupe_and_prune_state_options(candidates, max_options=cap)
        self._state_options_cache[key] = ans
        return ans

    def state_set_policy_options(
        self,
        attacker_troops: Sequence[int],
        defender_troops: Sequence[int],
        *,
        max_policy_options_per_row: Optional[int] = 2,
        max_options_per_state: Optional[int] = 2,
        max_leaf_split_depth: int = 1,
    ) -> List[CompactPolicyOption]:
        state = self.initial_state(attacker_troops, defender_troops)
        state_opts = self.state_policy_options(
            state,
            max_leaf_split_depth=max_leaf_split_depth,
            max_options_per_state=max_options_per_state,
        )
        if max_policy_options_per_row is not None:
            state_opts = state_opts[: max(1, int(max_policy_options_per_row))]
        return [
            CompactPolicyOption(
                option_id=i,
                root_action=opt.action,
                value=opt.value,
                absorbing_dist=opt.absorbing_dist,
                split_metadata=opt.split_metadata,
            )
            for i, opt in enumerate(state_opts)
        ]

    # ------------------------------------------------------------------
    # Conversion helpers for current infrastructure
    # ------------------------------------------------------------------

    def dist_to_global_dist(self, dist: Distribution) -> Dict[GlobalState, float]:
        return {self.state_to_global_state(s): float(p) for s, p in dist.items() if float(p) > 0.0}

    def dist_to_rowdict(self, dist: Distribution) -> Dict[str, float]:
        row: Dict[str, float] = {}
        for state, p in dist.items():
            if float(p) <= 0.0:
                continue
            lbl = self.state_label(state)
            row[lbl] = row.get(lbl, 0.0) + float(p)
        total = float(sum(row.values()))
        if total > 0.0:
            row = {k: float(v) / total for k, v in row.items() if float(v) > 0.0}
        return row

    def compact_option_to_policy_option(self, opt: CompactPolicyOption) -> PolicyOption:
        return PolicyOption(
            option_id=int(opt.option_id),
            root_action=opt.root_action,
            value=tuple(float(x) for x in opt.value),
            absorbing_dist=self.dist_to_global_dist(opt.absorbing_dist),
        )

    def dist_to_v2_payload(self, dist: Distribution) -> Dict[str, Any]:
        """Convert a compact absorbing distribution into the current V2 row shape."""
        items = [(int(s), float(p)) for s, p in dist.items() if float(p) > 0.0]
        items.sort(key=lambda x: self.state_label(x[0]))
        M = self.n
        N = len(items)
        p_arr = np.empty((N,), dtype=np.float32)
        owners = np.empty((N, M), dtype=np.uint8)
        troops = np.empty((N, M), dtype=np.uint16)
        is_conq = np.empty((N,), dtype=np.uint8)
        new_terr = np.empty((N,), dtype=np.int16)
        final_att = np.empty((N,), dtype=np.int32)

        for r, (state, p) in enumerate(items):
            p_arr[r] = np.float32(p)
            mask = self.owner_mask(state)
            tups = self.troops_tuple(state)
            any_defender = False
            new_count = 0
            final_count = 0
            for i, t in enumerate(tups):
                is_A = bool((mask >> i) & 1)
                owners[r, i] = np.uint8(1 if is_A else 2)
                troops[r, i] = np.uint16(t)
                if is_A:
                    final_count += int(t)
                    if i >= self.nA:
                        new_count += 1
                else:
                    any_defender = True
            is_conq[r] = np.uint8(0 if any_defender else 1)
            new_terr[r] = np.int16(new_count)
            final_att[r] = np.int32(final_count)

        s = float(p_arr.sum())
        if s > 0.0 and abs(s - 1.0) > 1e-6:
            p_arr = (p_arr / np.float32(s)).astype(np.float32)
        cdf = np.cumsum(p_arr, dtype=np.float32)
        if cdf.size:
            cdf[-1] = np.float32(1.0)
        return {
            "p": p_arr,
            "owners": owners,
            "troops": troops,
            "is_conquered": is_conq,
            "new_territories": new_terr,
            "final_attacker_troops": final_att,
            "cdf": cdf,
        }

    def policy_options_to_v2_payload(
        self,
        options: Sequence[CompactPolicyOption],
        *,
        policy_option_mode: str = "root",
        max_policy_options_per_row: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload_options: List[Dict[str, Any]] = []
        for opt in options:
            p = self.dist_to_v2_payload(opt.absorbing_dist)
            probs = np.asarray(p["p"], dtype=np.float64)
            new_terr = np.asarray(p["new_territories"], dtype=np.float64)
            is_conq = np.asarray(p["is_conquered"], dtype=np.float64)
            final_att = np.asarray(p["final_attacker_troops"], dtype=np.float64)
            p_no_gain = float(probs[new_terr == 0].sum()) if probs.size else 0.0
            p = dict(p)
            p["option_id"] = int(opt.option_id)
            p["root_action"] = list(opt.root_action) if opt.root_action is not None else None
            if getattr(opt, "split_metadata", ()):
                p["split_metadata"] = tuple(opt.split_metadata)
            p["local_value"] = {
                "expected_new_territories": float(np.dot(probs, new_terr)) if probs.size else 0.0,
                "expected_final_attacker_troops": float(np.dot(probs, final_att)) if probs.size else 0.0,
                "p_local_conquest": float(np.dot(probs, is_conq)) if probs.size else 0.0,
                "p_no_gain": p_no_gain,
                "raw_value": tuple(float(x) for x in opt.value),
                "utility_tuple_semantics": (
                    ("expected_new_territories", "-p_no_gain", "expected_final_attacker_troops", "p_local_conquest")
                    if self.include_no_gain_in_value
                    else ("expected_new_territories", "expected_final_attacker_troops", "p_local_conquest")
                ),
            }
            payload_options.append(p)
        return {
            "format": "policy_options_v2",
            "policy_option_mode": str(policy_option_mode),
            "max_policy_options_per_row": None if max_policy_options_per_row is None else int(max_policy_options_per_row),
            "include_no_gain_in_value": bool(self.include_no_gain_in_value),
            "options": payload_options,
        }

    # ------------------------------------------------------------------
    # Grid builders / diagnostics
    # ------------------------------------------------------------------

    def build_rowdict_grid(
        self,
        *,
        max_attacker_troops: int,
        max_defender_troops: int,
    ) -> RowBuildResult:
        rows: Dict[str, Dict[str, float]] = {}
        for att in product(range(1, int(max_attacker_troops) + 1), repeat=self.nA):
            for deff in product(range(1, int(max_defender_troops) + 1), repeat=self.nD):
                res = self.evaluate_start(att, deff)
                rows[self.row_label(att, deff)] = self.dist_to_rowdict(res.absorbing_dist)
        stats = self.stats.as_dict()
        stats.update(
            rows=len(rows),
            value_cache_size=len(self._value_cache),
            dist_cache_size=len(self._dist_cache),
            combat_cache_size=len(self._combat_cache),
        )
        return RowBuildResult(rows=rows, stats=stats)

    def build_policy_option_payload_grid(
        self,
        *,
        max_attacker_troops: int,
        max_defender_troops: int,
        max_policy_options_per_row: Optional[int] = 4,
        policy_option_mode: str = "root",
        max_options_per_state: Optional[int] = 2,
        max_leaf_split_depth: int = 1,
        max_split_depth: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        mode = str(policy_option_mode or "root").lower().strip()
        if max_split_depth is not None:
            max_leaf_split_depth = int(max_split_depth)
        if mode in {"state-set", "state_set", "bottom_up", "bottom-up"}:
            mode = "state_set"
        if mode not in {"root", "state_set"}:
            raise ValueError(
                "policy_option_mode must be 'root' or 'state_set', "
                f"got {policy_option_mode!r}"
            )
        rows: Dict[str, Dict[str, Any]] = {}
        for att in product(range(1, int(max_attacker_troops) + 1), repeat=self.nA):
            for deff in product(range(1, int(max_defender_troops) + 1), repeat=self.nD):
                if mode == "root":
                    opts = self.root_policy_options(
                        att,
                        deff,
                        max_policy_options=max_policy_options_per_row,
                    )
                else:
                    opts = self.state_set_policy_options(
                        att,
                        deff,
                        max_policy_options_per_row=max_policy_options_per_row,
                        max_options_per_state=max_options_per_state,
                        max_leaf_split_depth=max_leaf_split_depth,
                    )
                rows[self.row_label(att, deff)] = self.policy_options_to_v2_payload(
                    opts,
                    policy_option_mode=mode,
                    max_policy_options_per_row=max_policy_options_per_row,
                )
        stats = self.stats.as_dict()
        stats.update(
            rows=len(rows),
            value_cache_size=len(self._value_cache),
            dist_cache_size=len(self._dist_cache),
            combat_cache_size=len(self._combat_cache),
            state_options_cache_size=len(self._state_options_cache),
        )
        return rows, stats

    def clear_distribution_cache(self) -> None:
        self._dist_cache.clear()

    def clear_all_caches(self) -> None:
        self._value_cache.clear()
        self._dist_cache.clear()
        self._combat_cache.clear()
        self._state_options_cache.clear()
        self._leaf_distance_cache.clear()
        self.stats = ExactSolverStats()



def compare_rowdicts(
    a: Dict[str, float],
    b: Dict[str, float],
    *,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """Small diagnostic helper for validating against legacy row dictionaries."""
    keys = set(a) | set(b)
    max_abs = 0.0
    l1 = 0.0
    worst_key = None
    for k in keys:
        diff = abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0)))
        l1 += diff
        if diff > max_abs:
            max_abs = diff
            worst_key = k
    return {
        "ok": bool(max_abs <= float(tol)),
        "max_abs_diff": float(max_abs),
        "l1_diff": float(l1),
        "worst_key": worst_key,
        "num_keys_a": len(a),
        "num_keys_b": len(b),
        "num_union_keys": len(keys),
    }


__all__ = [
    "Action",
    "CompactExactTopologySolver",
    "CompactPolicyOption",
    "CompactStatePolicyOption",
    "Distribution",
    "ExactSolverStats",
    "ExactStateResult",
    "RowBuildResult",
    "combat_df_for_total_troops",
    "combat_df_for_caps",
    "compare_rowdicts",
    "infer_combat_df_limits",
]
