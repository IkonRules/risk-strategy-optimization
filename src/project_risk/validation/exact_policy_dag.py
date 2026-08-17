"""Validation-only shared policy-DAG tools for the compact exact solver.

The production solver computes one canonical full-depth continuation.  This
module exposes that continuation and exact action ties without changing solver
selection, library payloads, or ranking behavior.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import pickle
import time
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from project_risk.mathematical.small_graph_model.exact_finite_solver import CompactExactTopologySolver, ExactSolverLimitReached


ActionSignature = Any
StateSignature = Any
ValueTuple = Tuple[float, ...]


@dataclass(frozen=True)
class ExactPolicyDagAction:
    action_signature: ActionSignature
    action_value: ValueTuple
    outcome_probabilities: Tuple[float, ...]
    successor_state_signatures: Tuple[StateSignature, ...]
    is_canonical_action: bool
    is_exact_tied_optimal: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExactPolicyDagNode:
    state_signature: StateSignature
    state_value: ValueTuple
    terminal: bool
    canonical_action_signature: Optional[ActionSignature]
    retained_actions: Tuple[ExactPolicyDagAction, ...]
    all_optimal_action_signatures: Tuple[ActionSignature, ...]
    depth_from_root_min: Optional[int]
    reachable_probability_under_canonical_policy: Optional[float]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExactPolicyDag:
    root_state_signature: StateSignature
    utility_mode: str
    nodes: Mapping[StateSignature, ExactPolicyDagNode]
    canonical_policy_action_by_state: Mapping[StateSignature, ActionSignature]
    branching_state_signatures: Tuple[StateSignature, ...]
    max_observed_depth: int
    terminal_state_count: int
    implied_policy_count: Optional[int]
    construction_status: str
    runtime_seconds: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExactPolicyVariant:
    policy_identity: str
    action_by_state: Mapping[StateSignature, ActionSignature]
    root_action_signature: Optional[ActionSignature]
    branching_choices: Mapping[StateSignature, ActionSignature]
    exact_value: ValueTuple
    terminal_distribution: Mapping[StateSignature, float]
    runtime_seconds: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDagTraceEntry:
    state_signature: StateSignature
    decision_depth: int
    state_rows: Tuple[Any, ...]
    action_signature: Optional[ActionSignature]
    action_label: str
    outcomes: Tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDagTrace:
    root_state_signature: StateSignature
    entries: Tuple[PolicyDagTraceEntry, ...]
    max_decision_depth: int
    truncated: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExactPolicyDagExportCache:
    """Call-suite cache shared by exports at several split depths."""

    action_values: MutableMapping[Any, ValueTuple] = field(default_factory=dict)
    attack_transitions: MutableMapping[Any, Tuple[Any, ...]] = field(default_factory=dict)
    attack_transition_diagnostics: MutableMapping[Any, Tuple[Any, ...]] = field(
        default_factory=dict
    )
    movement_specs: MutableMapping[StateSignature, Mapping[str, Any]] = field(default_factory=dict)
    action_value_hits: int = 0
    action_value_misses: int = 0
    transition_hits: int = 0
    transition_misses: int = 0

    def diagnostics(self) -> Dict[str, int]:
        return {
            "action_value_cache_size": len(self.action_values),
            "action_value_cache_hits": int(self.action_value_hits),
            "action_value_cache_misses": int(self.action_value_misses),
            "transition_cache_size": len(self.attack_transitions),
            "transition_cache_hits": int(self.transition_hits),
            "transition_cache_misses": int(self.transition_misses),
            "transition_diagnostic_count": len(self.attack_transition_diagnostics),
            "movement_spec_count": len(self.movement_specs),
        }


class PolicyDagConstructionLimit(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = str(status)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_ready(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return repr(value)


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_sort_key(value: Any) -> str:
    return repr(value)


def _normalize_distribution(distribution: Mapping[Any, float]) -> Dict[Any, float]:
    output: Dict[Any, float] = {}
    for state, probability in distribution.items():
        value = float(probability)
        if not math.isfinite(value) or value < -1e-12:
            raise ValueError("policy variant produced an invalid probability")
        if value > 0.0:
            output[state] = output.get(state, 0.0) + value
    total = float(sum(output.values()))
    if total <= 0.0:
        raise ValueError("policy variant produced no positive probability mass")
    return {state: value / total for state, value in output.items()}


def _state_rows(
    solver: CompactExactTopologySolver,
    state: int,
    *,
    node_ids: Optional[Sequence[int]],
    territory_names: Optional[Mapping[int, str]],
) -> Tuple[Tuple[Any, ...], Tuple[Tuple[int, str, int], ...]]:
    owner_mask = solver.owner_mask(int(state))
    troops = solver.troops_tuple(int(state))
    local_rows = []
    global_rows = []
    for local_index, troop_count in enumerate(troops):
        owner = "A" if ((owner_mask >> local_index) & 1) else "D"
        global_node = (
            int(node_ids[local_index]) if node_ids is not None else int(local_index)
        )
        name = (
            str(territory_names.get(global_node, global_node))
            if territory_names is not None
            else str(global_node)
        )
        local_rows.append((int(local_index), global_node, name, owner, int(troop_count)))
        global_rows.append((global_node, owner, int(troop_count)))
    return tuple(local_rows), tuple(sorted(global_rows))


def _action_label(
    action_signature: Any,
    *,
    node_ids: Optional[Sequence[int]],
    territory_names: Optional[Mapping[int, str]],
) -> str:
    if action_signature is None:
        return "stop"
    if (
        isinstance(action_signature, tuple)
        and len(action_signature) == 2
        and all(isinstance(value, int) for value in action_signature)
    ):
        source, target = action_signature
        source_global = int(node_ids[source]) if node_ids is not None else int(source)
        target_global = int(node_ids[target]) if node_ids is not None else int(target)
        source_name = (
            str(territory_names.get(source_global, source_global))
            if territory_names is not None
            else str(source_global)
        )
        target_name = (
            str(territory_names.get(target_global, target_global))
            if territory_names is not None
            else str(target_global)
        )
        return f"attack {target_name} from {source_name}"
    if isinstance(action_signature, tuple) and action_signature:
        if action_signature[0] == "move_one":
            return "move one troop into the conquered territory"
        if action_signature[0] == "push":
            return "push the available stack into the conquered territory"
    return repr(action_signature)


def _cached_action_value(
    solver: CompactExactTopologySolver,
    state: int,
    action: Tuple[int, int],
    cache: ExactPolicyDagExportCache,
) -> ValueTuple:
    key = (int(state), tuple(action))
    cached = cache.action_values.get(key)
    if cached is not None:
        cache.action_value_hits += 1
        return tuple(float(value) for value in cached)
    cache.action_value_misses += 1
    value = tuple(float(item) for item in solver.evaluate_action_value(int(state), action))
    cache.action_values[key] = value
    return value


def _movement_state_signature(
    state: int,
    action: Tuple[int, int],
    a_end: int,
    d_end: int,
) -> Tuple[Any, ...]:
    return (
        "movement_decision_v1",
        int(state),
        tuple(action),
        int(a_end),
        int(d_end),
    )


def _attack_transitions(
    solver: CompactExactTopologySolver,
    state: int,
    action: Tuple[int, int],
    cache: ExactPolicyDagExportCache,
) -> Tuple[Tuple[float, StateSignature], ...]:
    key = (int(state), tuple(action))
    cached = cache.attack_transitions.get(key)
    if cached is not None:
        cache.transition_hits += 1
        return tuple((float(p), signature) for p, signature in cached)
    cache.transition_misses += 1
    u, v = action
    source_troops = solver.troop_at(int(state), u)
    target_troops = solver.troop_at(int(state), v)
    old_mask = solver.owner_mask(int(state))
    transitions = []
    combat_rows = []
    for a_end, d_end, probability in solver.combat_outcomes(source_troops - 1, target_troops):
        origin_after = 1 + int(a_end)
        if int(d_end) > 0:
            child = solver.replace_troops_and_owner(
                int(state),
                owner_mask=old_mask,
                updates=((u, origin_after), (v, int(d_end))),
            )
            successor: StateSignature = int(child)
            movement = "not_conquered"
        elif not solver._other_enemy_neighbors_at_origin(int(state), u, v):
            child = solver.replace_troops_and_owner(
                int(state),
                owner_mask=old_mask | (1 << v),
                updates=((u, 1), (v, origin_after - 1)),
            )
            successor = int(child)
            movement = "forced_push"
        else:
            new_mask = old_mask | (1 << v)
            keep_state = solver.replace_troops_and_owner(
                int(state),
                owner_mask=new_mask,
                updates=((u, origin_after - 1), (v, 1)),
            )
            push_state = solver.replace_troops_and_owner(
                int(state),
                owner_mask=new_mask,
                updates=((u, 1), (v, origin_after - 1)),
            )
            keep_value = tuple(float(value) for value in solver.evaluate_value(keep_state)[0])
            push_value = tuple(float(value) for value in solver.evaluate_value(push_state)[0])
            canonical = "move_one" if solver.better_value(keep_value, push_value) else "push"
            movement_signature = _movement_state_signature(
                int(state), action, int(a_end), int(d_end)
            )
            cache.movement_specs[movement_signature] = {
                "parent_state": int(state),
                "attack_action": tuple(action),
                "a_end": int(a_end),
                "d_end": int(d_end),
                "source": int(u),
                "target": int(v),
                "keep_state": int(keep_state),
                "push_state": int(push_state),
                "keep_value": keep_value,
                "push_value": push_value,
                "canonical_choice": canonical,
                "total_troops": int(sum(solver.troops_tuple(keep_state))),
            }
            successor = movement_signature
            movement = "policy_decision"
        transitions.append((float(probability), successor))
        combat_rows.append(
            {
                "a_end": int(a_end),
                "d_end": int(d_end),
                "probability": float(probability),
                "successor_state_signature": successor,
                "movement": movement,
            }
        )
    # Distinct combat outcomes can occasionally produce the same compact child.
    aggregated: Dict[StateSignature, float] = {}
    for probability, successor in transitions:
        aggregated[successor] = aggregated.get(successor, 0.0) + float(probability)
    answer = tuple(
        (float(probability), successor)
        for successor, probability in sorted(aggregated.items(), key=lambda item: _stable_sort_key(item[0]))
    )
    cache.attack_transitions[key] = answer
    cache.attack_transition_diagnostics[key] = tuple(combat_rows)
    return answer


def _split_allowed(depth: int, max_split_depth: Optional[int]) -> bool:
    return max_split_depth is None or int(depth) < int(max_split_depth)


def _node_total_troops(node: ExactPolicyDagNode) -> int:
    return int(node.diagnostics.get("total_troops", 0))


def _canonical_reachability(
    nodes: Mapping[StateSignature, ExactPolicyDagNode],
    root_state: StateSignature,
) -> Tuple[Dict[StateSignature, float], Dict[StateSignature, int]]:
    probabilities: Dict[StateSignature, float] = {root_state: 1.0}
    depths: Dict[StateSignature, int] = {root_state: 0}
    order = sorted(
        nodes,
        key=lambda signature: (
            _node_total_troops(nodes[signature]),
            1 if nodes[signature].diagnostics.get("state_kind") == "movement_decision" else 0,
            _stable_sort_key(signature),
        ),
        reverse=True,
    )
    for signature in order:
        mass = float(probabilities.get(signature, 0.0))
        if mass <= 0.0:
            continue
        node = nodes[signature]
        canonical = next(
            (action for action in node.retained_actions if action.is_canonical_action),
            None,
        )
        if canonical is None:
            continue
        current_depth = int(depths.get(signature, 0))
        is_movement = node.diagnostics.get("state_kind") == "movement_decision"
        next_depth = current_depth if is_movement else current_depth + 1
        for probability, successor in zip(
            canonical.outcome_probabilities,
            canonical.successor_state_signatures,
        ):
            if successor not in nodes:
                continue
            probabilities[successor] = probabilities.get(successor, 0.0) + mass * float(probability)
            depths[successor] = min(depths.get(successor, next_depth), next_depth)
    return probabilities, depths


def _policy_mapping_count(
    nodes: Mapping[StateSignature, ExactPolicyDagNode],
    count_limit: Optional[int],
) -> Tuple[Optional[int], str, int]:
    count = 1
    for node in nodes.values():
        width = len(node.retained_actions)
        if width <= 1:
            continue
        count *= width
        if count_limit is not None and count > int(count_limit):
            return None, "overflow", int(count_limit) + 1
    return int(count), "exact_action_mapping_count", int(count)


def _naive_unrolled_occurrences(
    nodes: Mapping[StateSignature, ExactPolicyDagNode],
    root_state: StateSignature,
    *,
    limit: int = 10**12,
) -> Tuple[int, bool]:
    path_counts: Dict[StateSignature, int] = {root_state: 1}
    order = sorted(
        nodes,
        key=lambda signature: (
            _node_total_troops(nodes[signature]),
            1 if nodes[signature].diagnostics.get("state_kind") == "movement_decision" else 0,
            _stable_sort_key(signature),
        ),
        reverse=True,
    )
    overflow = False
    total = 0
    for signature in order:
        paths = int(path_counts.get(signature, 0))
        total = min(limit, total + paths)
        if total >= limit:
            overflow = True
        for action in nodes[signature].retained_actions:
            for successor in action.successor_state_signatures:
                if successor not in nodes:
                    continue
                value = path_counts.get(successor, 0) + paths
                if value >= limit:
                    value = limit
                    overflow = True
                path_counts[successor] = value
    return int(total), bool(overflow)


def policy_dag_summary(policy_dag: ExactPolicyDag) -> Dict[str, Any]:
    nodes = tuple(policy_dag.nodes.values())
    actions = tuple(action for node in nodes for action in node.retained_actions)
    edge_count = sum(len(action.successor_state_signatures) for action in actions)
    return {
        "construction_status": policy_dag.construction_status,
        "node_count": len(nodes),
        "decision_node_count": sum(not node.terminal for node in nodes),
        "edge_count": edge_count,
        "retained_action_count": len(actions),
        "terminal_node_count": policy_dag.terminal_state_count,
        "branching_state_count": len(policy_dag.branching_state_signatures),
        "maximum_actions_at_state": max((len(node.retained_actions) for node in nodes), default=0),
        "max_observed_depth": policy_dag.max_observed_depth,
        "implied_policy_count": policy_dag.implied_policy_count,
        "implied_policy_count_status": policy_dag.diagnostics.get("implied_policy_count_status"),
        "runtime_seconds": policy_dag.runtime_seconds,
        "pickle_size_bytes": policy_dag.diagnostics.get("pickle_size_bytes"),
        "json_summary_size_bytes": policy_dag.diagnostics.get("json_summary_size_bytes"),
        "estimated_memory_bytes": policy_dag.diagnostics.get("estimated_memory_bytes"),
        "shared_successor_reference_count": policy_dag.diagnostics.get("shared_successor_reference_count"),
        "naive_unrolled_node_occurrences": policy_dag.diagnostics.get("naive_unrolled_node_occurrences"),
        "avoided_duplicate_node_occurrences": policy_dag.diagnostics.get("avoided_duplicate_node_occurrences"),
        "maximum_canonical_decision_depth": policy_dag.diagnostics.get("maximum_canonical_decision_depth"),
        "maximum_retained_alternative_depth": policy_dag.diagnostics.get("maximum_retained_alternative_depth"),
    }


def export_exact_policy_dag(
    *,
    solver: CompactExactTopologySolver,
    root_state: int,
    retain_mode: str = "canonical_only",
    max_split_depth: Optional[int] = None,
    max_actions_per_state: Optional[int] = None,
    reachable_under: str = "retained_actions",
    max_dag_nodes: Optional[int] = None,
    max_dag_edges: Optional[int] = None,
    max_runtime_seconds: Optional[float] = None,
    policy_count_limit: Optional[int] = 1_000_000,
    export_cache: Optional[ExactPolicyDagExportCache] = None,
    node_ids: Optional[Sequence[int]] = None,
    territory_names: Optional[Mapping[int, str]] = None,
) -> ExactPolicyDag:
    """Export a shared full-depth policy DAG from one solved compact state.

    ``max_split_depth`` controls where alternative tied actions are retained.
    It never truncates canonical continuation. Root attack depth is zero, so a
    split depth of one permits alternatives at the root, two also permits them
    after one combat transition, and ``None`` permits them everywhere.
    """
    started = time.perf_counter()
    mode = str(retain_mode).strip().lower()
    if mode not in {"canonical_only", "exact_ties"}:
        raise ValueError(f"Unknown retain_mode: {retain_mode!r}")
    if max_split_depth is not None and int(max_split_depth) < 0:
        raise ValueError("max_split_depth must be non-negative or None")
    if reachable_under not in {"retained_actions", "canonical_policy"}:
        raise ValueError("reachable_under must be 'retained_actions' or 'canonical_policy'")
    if node_ids is not None and len(tuple(node_ids)) != solver.n:
        raise ValueError("node_ids must align with the solver node count")
    cache = export_cache if export_cache is not None else ExactPolicyDagExportCache()
    root_state = int(root_state)
    nodes: Dict[StateSignature, ExactPolicyDagNode] = {}
    canonical_policy: Dict[StateSignature, ActionSignature] = {}
    depth_by_state: Dict[StateSignature, int] = {root_state: 0}
    queue = deque([root_state])
    queued = {root_state}
    edge_count = 0
    status = "completed"
    stop_reason: Optional[str] = None

    def check_limits(*, add_nodes: int = 0, add_edges: int = 0) -> None:
        if (
            max_runtime_seconds is not None
            and time.perf_counter() - started >= float(max_runtime_seconds)
        ):
            raise PolicyDagConstructionLimit("runtime_limit", "policy DAG export runtime limit reached")
        if max_dag_nodes is not None and len(nodes) + int(add_nodes) > int(max_dag_nodes):
            raise PolicyDagConstructionLimit("dag_node_limit", "policy DAG node limit reached")
        if max_dag_edges is not None and edge_count + int(add_edges) > int(max_dag_edges):
            raise PolicyDagConstructionLimit("dag_edge_limit", "policy DAG edge limit reached")

    try:
        solver.evaluate_value(root_state)
        while queue:
            check_limits(add_nodes=1)
            signature = queue.popleft()
            queued.discard(signature)
            if signature in nodes:
                continue
            depth = int(depth_by_state.get(signature, 0))

            if isinstance(signature, tuple) and signature and signature[0] == "movement_decision_v1":
                spec = cache.movement_specs.get(signature)
                if spec is None:
                    raise RuntimeError(f"missing movement specification for {signature!r}")
                canonical_choice = str(spec["canonical_choice"])
                choice_rows = (
                    ("move_one", int(spec["keep_state"]), tuple(spec["keep_value"])),
                    ("push", int(spec["push_state"]), tuple(spec["push_value"])),
                )
                canonical_value = next(value for name, _child, value in choice_rows if name == canonical_choice)
                optimal = tuple(
                    (name, child, value)
                    for name, child, value in choice_rows
                    if solver.equivalent_value(tuple(value), tuple(canonical_value))
                )
                retained = (
                    optimal
                    if mode == "exact_ties"
                    and reachable_under == "retained_actions"
                    and _split_allowed(depth, max_split_depth)
                    else tuple(row for row in choice_rows if row[0] == canonical_choice)
                )
                if max_actions_per_state is not None and len(retained) > int(max_actions_per_state):
                    raise PolicyDagConstructionLimit(
                        "dag_edge_limit",
                        "max_actions_per_state would truncate exactly tied movement actions",
                    )
                action_objects = []
                for name, child, value in sorted(
                    retained,
                    key=lambda row: (row[0] != canonical_choice, _stable_sort_key(row[0])),
                ):
                    action_signature = (str(name), int(spec["source"]), int(spec["target"]))
                    check_limits(add_edges=1)
                    edge_count += 1
                    action_objects.append(
                        ExactPolicyDagAction(
                            action_signature=action_signature,
                            action_value=tuple(float(item) for item in value),
                            outcome_probabilities=(1.0,),
                            successor_state_signatures=(int(child),),
                            is_canonical_action=name == canonical_choice,
                            is_exact_tied_optimal=True,
                            diagnostics={
                                "action_kind": "movement",
                                "movement_choice": str(name),
                                "source_local": int(spec["source"]),
                                "target_local": int(spec["target"]),
                                "label": _action_label(
                                    action_signature,
                                    node_ids=node_ids,
                                    territory_names=territory_names,
                                ),
                            },
                        )
                    )
                    if int(child) not in nodes and int(child) not in queued:
                        depth_by_state[int(child)] = min(depth_by_state.get(int(child), depth), depth)
                        queue.appendleft(int(child))
                        queued.add(int(child))
                parent_rows, parent_global_rows = _state_rows(
                    solver,
                    int(spec["parent_state"]),
                    node_ids=node_ids,
                    territory_names=territory_names,
                )
                canonical_signature = (
                    canonical_choice,
                    int(spec["source"]),
                    int(spec["target"]),
                )
                canonical_policy[signature] = canonical_signature
                nodes[signature] = ExactPolicyDagNode(
                    state_signature=signature,
                    state_value=tuple(float(item) for item in canonical_value),
                    terminal=False,
                    canonical_action_signature=canonical_signature,
                    retained_actions=tuple(action_objects),
                    all_optimal_action_signatures=tuple(
                        (name, int(spec["source"]), int(spec["target"]))
                        for name, _child, _value in optimal
                    ),
                    depth_from_root_min=depth,
                    reachable_probability_under_canonical_policy=None,
                    diagnostics={
                        "state_kind": "movement_decision",
                        "attack_decision_depth": depth,
                        "parent_state_rows": parent_rows,
                        "parent_global_state_signature": parent_global_rows,
                        "total_troops": int(spec["total_troops"]),
                        "attack_action": tuple(spec["attack_action"]),
                    },
                )
                continue

            state = int(signature)
            state_value, canonical_action = solver.evaluate_value(state)
            state_value = tuple(float(value) for value in state_value)
            local_rows, global_rows = _state_rows(
                solver,
                state,
                node_ids=node_ids,
                territory_names=territory_names,
            )
            terminal = solver.is_absorbing(state) or canonical_action is None
            if terminal:
                nodes[state] = ExactPolicyDagNode(
                    state_signature=state,
                    state_value=state_value,
                    terminal=True,
                    canonical_action_signature=None,
                    retained_actions=(),
                    all_optimal_action_signatures=(),
                    depth_from_root_min=depth,
                    reachable_probability_under_canonical_policy=None,
                    diagnostics={
                        "state_kind": "compact_state",
                        "state_rows": local_rows,
                        "global_state_signature": global_rows,
                        "total_troops": int(sum(solver.troops_tuple(state))),
                        "attack_decision_depth": depth,
                    },
                )
                continue

            actions = solver.possible_actions(state)
            values = {
                tuple(action): _cached_action_value(solver, state, tuple(action), cache)
                for action in actions
            }
            optimal_actions = tuple(
                action
                for action in actions
                if solver.equivalent_value(tuple(values[action]), state_value)
            )
            if canonical_action not in optimal_actions:
                optimal_actions = (tuple(canonical_action),) + tuple(
                    action for action in optimal_actions if action != canonical_action
                )
            retained_actions = (
                optimal_actions
                if mode == "exact_ties"
                and reachable_under == "retained_actions"
                and _split_allowed(depth, max_split_depth)
                else (tuple(canonical_action),)
            )
            retained_actions = tuple(
                sorted(
                    set(retained_actions),
                    key=lambda action: (action != tuple(canonical_action), tuple(action)),
                )
            )
            if max_actions_per_state is not None and len(retained_actions) > int(max_actions_per_state):
                raise PolicyDagConstructionLimit(
                    "dag_edge_limit",
                    "max_actions_per_state would truncate exactly tied attack actions",
                )
            action_objects = []
            for action in retained_actions:
                transitions = _attack_transitions(solver, state, action, cache)
                check_limits(add_edges=len(transitions))
                edge_count += len(transitions)
                probabilities = tuple(float(probability) for probability, _successor in transitions)
                successors = tuple(successor for _probability, successor in transitions)
                source_local, target_local = action
                source_global = int(node_ids[source_local]) if node_ids is not None else int(source_local)
                target_global = int(node_ids[target_local]) if node_ids is not None else int(target_local)
                action_objects.append(
                    ExactPolicyDagAction(
                        action_signature=tuple(action),
                        action_value=tuple(values[action]),
                        outcome_probabilities=probabilities,
                        successor_state_signatures=successors,
                        is_canonical_action=tuple(action) == tuple(canonical_action),
                        is_exact_tied_optimal=True,
                        diagnostics={
                            "action_kind": "attack",
                            "source_local": int(source_local),
                            "target_local": int(target_local),
                            "source_global": source_global,
                            "target_global": target_global,
                            "label": _action_label(
                                tuple(action),
                                node_ids=node_ids,
                                territory_names=territory_names,
                            ),
                            "combat_outcomes": cache.attack_transition_diagnostics.get(
                                (int(state), tuple(action)), ()
                            ),
                        },
                    )
                )
                for successor in successors:
                    successor_depth = depth + 1
                    if successor in nodes or successor in queued:
                        depth_by_state[successor] = min(
                            depth_by_state.get(successor, successor_depth), successor_depth
                        )
                        continue
                    depth_by_state[successor] = successor_depth
                    queue.append(successor)
                    queued.add(successor)
            canonical_policy[state] = tuple(canonical_action)
            nodes[state] = ExactPolicyDagNode(
                state_signature=state,
                state_value=state_value,
                terminal=False,
                canonical_action_signature=tuple(canonical_action),
                retained_actions=tuple(action_objects),
                all_optimal_action_signatures=tuple(tuple(action) for action in optimal_actions),
                depth_from_root_min=depth,
                reachable_probability_under_canonical_policy=None,
                diagnostics={
                    "state_kind": "compact_state",
                    "state_rows": local_rows,
                    "global_state_signature": global_rows,
                    "total_troops": int(sum(solver.troops_tuple(state))),
                    "attack_decision_depth": depth,
                    "legal_action_count": len(actions),
                    "exact_optimal_action_count": len(optimal_actions),
                },
            )
    except PolicyDagConstructionLimit as exc:
        status = exc.status
        stop_reason = str(exc)
    except ExactSolverLimitReached as exc:
        status = str(exc.status)
        stop_reason = str(exc)
    except Exception as exc:
        status = "solver_error"
        stop_reason = f"{type(exc).__name__}: {exc}"

    reach, canonical_depths = _canonical_reachability(nodes, root_state)
    for signature, node in tuple(nodes.items()):
        diagnostics = dict(node.diagnostics)
        diagnostics["canonical_decision_depth_min"] = canonical_depths.get(signature)
        nodes[signature] = replace(
            node,
            reachable_probability_under_canonical_policy=float(reach.get(signature, 0.0)),
            diagnostics=diagnostics,
        )

    branching = tuple(
        sorted(
            (signature for signature, node in nodes.items() if len(node.retained_actions) > 1),
            key=_stable_sort_key,
        )
    )
    implied_count, count_status, count_lower_bound = _policy_mapping_count(
        nodes, policy_count_limit
    )
    action_count = sum(len(node.retained_actions) for node in nodes.values())
    terminal_count = sum(node.terminal for node in nodes.values())
    unique_nonroot_successors = {
        successor
        for node in nodes.values()
        for action in node.retained_actions
        for successor in action.successor_state_signatures
        if successor in nodes and successor != root_state
    }
    shared_refs = max(0, edge_count - len(unique_nonroot_successors))
    naive_occurrences, naive_overflow = _naive_unrolled_occurrences(nodes, root_state)
    maximum_canonical_depth = max(
        (
            int(node.diagnostics.get("canonical_decision_depth_min", 0))
            for node in nodes.values()
            if not node.terminal
            and node.diagnostics.get("state_kind") == "compact_state"
            and float(node.reachable_probability_under_canonical_policy or 0.0) > 0.0
        ),
        default=0,
    )
    maximum_alternative_depth = max(
        (int(nodes[signature].depth_from_root_min or 0) for signature in branching),
        default=-1,
    )
    diagnostics = {
        "retain_mode": mode,
        "policy_dag_max_split_depth": max_split_depth,
        "split_depth_semantics": "root_side_attack_transition_depth; canonical continuation is never truncated",
        "reachable_under": reachable_under,
        "exact_tie_comparison": "solver.equivalent_value using the solver's existing numerical comparison semantics",
        "no_new_near_optimal_tolerance": True,
        "partial": status != "completed",
        "stop_reason": stop_reason,
        "node_count": len(nodes),
        "edge_count": int(edge_count),
        "retained_action_count": int(action_count),
        "branching_state_count": len(branching),
        "maximum_canonical_decision_depth": int(maximum_canonical_depth),
        "maximum_retained_alternative_depth": int(maximum_alternative_depth),
        "canonical_decision_state_count": sum(
            not node.terminal
            and node.diagnostics.get("state_kind") == "compact_state"
            and float(node.reachable_probability_under_canonical_policy or 0.0) > 0.0
            for node in nodes.values()
        ),
        "alternative_branching_state_count": len(branching),
        "implied_policy_count_status": count_status,
        "implied_policy_count_lower_bound": int(count_lower_bound),
        "policy_count_limit": policy_count_limit,
        "policy_count_definition": "complete retained-action mappings over shared DAG decision states",
        "shared_successor_reference_count": int(shared_refs),
        "naive_unrolled_node_occurrences": int(naive_occurrences),
        "naive_unrolled_count_overflow": bool(naive_overflow),
        "avoided_duplicate_node_occurrences": max(0, int(naive_occurrences) - len(nodes)),
        "export_cache": cache.diagnostics(),
        "solver_value_cache_size": len(solver._value_cache),
        "solver_distribution_cache_size": len(solver._dist_cache),
        "movement_decisions_are_explicit_pseudo_nodes": True,
    }
    dag = ExactPolicyDag(
        root_state_signature=root_state,
        utility_mode=str(solver.utility_mode),
        nodes=dict(nodes),
        canonical_policy_action_by_state=dict(canonical_policy),
        branching_state_signatures=branching,
        max_observed_depth=max(
            (int(node.depth_from_root_min or 0) for node in nodes.values()), default=0
        ),
        terminal_state_count=int(terminal_count),
        implied_policy_count=implied_count,
        construction_status=status,
        runtime_seconds=float(time.perf_counter() - started),
        diagnostics=diagnostics,
    )
    estimated_memory = int(
        512
        + len(nodes) * 448
        + action_count * 352
        + edge_count * 96
        + sum(len(repr(signature)) for signature in nodes)
    )
    dag = replace(
        dag,
        diagnostics={
            **dict(dag.diagnostics),
            "pickle_size_bytes": 0,
            "json_summary_size_bytes": 0,
            "estimated_memory_bytes": int(estimated_memory),
        },
    )
    # The serialized-size fields affect their own payload sizes. Iterate to a
    # fixed point so the diagnostics describe the final returned object.
    for _iteration in range(8):
        pickle_size = len(pickle.dumps(dag, protocol=pickle.HIGHEST_PROTOCOL))
        json_size = len(
            json.dumps(_json_ready(policy_dag_summary(dag)), sort_keys=True).encode(
                "utf-8"
            )
        )
        if (
            dag.diagnostics.get("pickle_size_bytes") == pickle_size
            and dag.diagnostics.get("json_summary_size_bytes") == json_size
        ):
            break
        dag = replace(
            dag,
            diagnostics={
                **dict(dag.diagnostics),
                "pickle_size_bytes": int(pickle_size),
                "json_summary_size_bytes": int(json_size),
            },
        )
    return dag


def _action_lookup(node: ExactPolicyDagNode) -> Dict[Any, ExactPolicyDagAction]:
    return {action.action_signature: action for action in node.retained_actions}


def materialize_policy_variant(
    *,
    policy_dag: ExactPolicyDag,
    action_choices_by_state: Optional[Mapping[Any, Any]] = None,
    variant_index: Optional[int] = None,
    max_states: Optional[int] = None,
    canonical_distribution_cache: Optional[
        MutableMapping[Any, Mapping[Any, float]]
    ] = None,
) -> ExactPolicyVariant:
    """Materialize one policy lazily from retained DAG actions.

    ``canonical_distribution_cache`` is scoped to one DAG. Canonical subtrees
    that cannot reach an explicitly changed branch are reusable between
    variants, so a late tied choice does not re-evaluate the common prefix or
    unrelated subtrees.
    """
    started = time.perf_counter()
    if policy_dag.construction_status != "completed":
        raise ValueError("cannot materialize a policy from an incomplete DAG")
    if action_choices_by_state is not None and variant_index is not None:
        raise ValueError("provide action_choices_by_state or variant_index, not both")
    choices: Dict[Any, Any] = dict(action_choices_by_state or {})
    branching = tuple(sorted(policy_dag.branching_state_signatures, key=_stable_sort_key))
    if variant_index is not None:
        index = int(variant_index)
        if index < 0:
            raise ValueError("variant_index must be non-negative")
        for signature in branching:
            node = policy_dag.nodes[signature]
            width = len(node.retained_actions)
            digit = index % width
            index //= width
            choices[signature] = node.retained_actions[digit].action_signature
        if index:
            raise IndexError("variant_index exceeds represented policy mappings")

    for signature, action_signature in tuple(choices.items()):
        node = policy_dag.nodes.get(signature)
        if node is None:
            raise KeyError(f"policy DAG has no state {signature!r}")
        if action_signature not in _action_lookup(node):
            raise ValueError(
                f"action {action_signature!r} is not retained at state {signature!r}"
            )
        if action_signature == node.canonical_action_signature:
            choices.pop(signature)

    reverse_predecessors: Dict[Any, set[Any]] = {}
    for parent_signature, node in policy_dag.nodes.items():
        for action in node.retained_actions:
            for successor in action.successor_state_signatures:
                reverse_predecessors.setdefault(successor, set()).add(parent_signature)
    affected_states = set(choices)
    affected_queue = deque(affected_states)
    while affected_queue:
        signature = affected_queue.popleft()
        for predecessor in reverse_predecessors.get(signature, ()):
            if predecessor not in affected_states:
                affected_states.add(predecessor)
                affected_queue.append(predecessor)

    memo: Dict[Any, Dict[Any, float]] = {}
    canonical_cache = canonical_distribution_cache
    canonical_cache_hits = 0
    canonical_cache_misses = 0
    visiting = set()

    def visit(signature: Any) -> Dict[Any, float]:
        nonlocal canonical_cache_hits, canonical_cache_misses
        if signature in memo:
            return memo[signature]
        if (
            signature not in affected_states
            and canonical_cache is not None
            and signature in canonical_cache
        ):
            canonical_cache_hits += 1
            return dict(canonical_cache[signature])
        if signature not in affected_states and canonical_cache is not None:
            canonical_cache_misses += 1
        if max_states is not None and len(memo) + len(visiting) >= int(max_states):
            raise PolicyDagConstructionLimit("dag_node_limit", "policy materialization state limit reached")
        if signature in visiting:
            raise ValueError("policy DAG contains a cycle")
        node = policy_dag.nodes.get(signature)
        if node is None:
            raise KeyError(f"policy DAG is missing successor node {signature!r}")
        if node.terminal:
            answer = {signature: 1.0}
            memo[signature] = answer
            if signature not in affected_states and canonical_cache is not None:
                canonical_cache[signature] = answer
            return answer
        visiting.add(signature)
        chosen_signature = choices.get(signature, node.canonical_action_signature)
        action = _action_lookup(node).get(chosen_signature)
        if action is None:
            raise ValueError(
                f"action {chosen_signature!r} is not retained at state {signature!r}"
            )
        output: Dict[Any, float] = {}
        for probability, successor in zip(
            action.outcome_probabilities, action.successor_state_signatures
        ):
            child_distribution = visit(successor)
            for terminal_state, child_probability in child_distribution.items():
                output[terminal_state] = output.get(terminal_state, 0.0) + float(probability) * float(child_probability)
        visiting.remove(signature)
        # Match CompactExactTopologySolver.absorbing_distribution: preserve
        # products of rounded combat probabilities through recursion and
        # normalize only once at the public root boundary.
        answer = {
            terminal_state: float(probability)
            for terminal_state, probability in output.items()
            if float(probability) > 0.0
        }
        memo[signature] = answer
        if signature not in affected_states and canonical_cache is not None:
            canonical_cache[signature] = answer
        return answer

    distribution = _normalize_distribution(visit(policy_dag.root_state_signature))
    action_by_state: Dict[Any, Any] = {}
    branching_choices: Dict[Any, Any] = {}
    reachable_states = set()
    pending = [policy_dag.root_state_signature]
    while pending:
        signature = pending.pop()
        if signature in reachable_states:
            continue
        if max_states is not None and len(reachable_states) >= int(max_states):
            raise PolicyDagConstructionLimit(
                "dag_node_limit", "policy materialization state limit reached"
            )
        reachable_states.add(signature)
        node = policy_dag.nodes.get(signature)
        if node is None:
            raise KeyError(f"policy DAG is missing successor node {signature!r}")
        if node.terminal:
            continue
        chosen_signature = choices.get(signature, node.canonical_action_signature)
        action = _action_lookup(node).get(chosen_signature)
        if action is None:
            raise ValueError(
                f"action {chosen_signature!r} is not retained at state {signature!r}"
            )
        action_by_state[signature] = chosen_signature
        if chosen_signature != node.canonical_action_signature:
            branching_choices[signature] = chosen_signature
        pending.extend(action.successor_state_signatures)
    distribution_implied_value = tuple(
        sum(
            float(probability)
            * float(policy_dag.nodes[terminal].state_value[index])
            for terminal, probability in distribution.items()
        )
        for index in range(len(policy_dag.nodes[policy_dag.root_state_signature].state_value))
    )
    exact_value = tuple(
        float(value)
        for value in policy_dag.nodes[policy_dag.root_state_signature].state_value
    )
    root_action = action_by_state.get(policy_dag.root_state_signature)
    identity_payload = tuple(
        sorted(
            ((repr(state), repr(action)) for state, action in action_by_state.items()),
            key=lambda row: row[0],
        )
    )
    return ExactPolicyVariant(
        policy_identity="exact_policy_variant_" + _stable_digest(identity_payload)[:24],
        action_by_state=dict(action_by_state),
        root_action_signature=root_action,
        branching_choices=dict(branching_choices),
        exact_value=exact_value,
        terminal_distribution=distribution,
        runtime_seconds=float(time.perf_counter() - started),
        diagnostics={
            "visited_state_count": len(reachable_states),
            "recomputed_state_count": len(memo),
            "canonical_subtree_cache_hits": int(canonical_cache_hits),
            "canonical_subtree_cache_misses": int(canonical_cache_misses),
            "canonical_subtree_cache_size": (
                len(canonical_cache) if canonical_cache is not None else 0
            ),
            "normalization_semantics": "root_boundary_only_like_exact_solver",
            "selected_noncanonical_branch_count": len(branching_choices),
            "distribution_support_size": len(distribution),
            "root_dag_value": policy_dag.nodes[policy_dag.root_state_signature].state_value,
            "distribution_implied_value": distribution_implied_value,
            "distribution_value_component_deltas": tuple(
                float(observed) - float(expected)
                for observed, expected in zip(distribution_implied_value, exact_value)
            ),
            "value_matches_root_numerically": all(
                abs(float(observed) - float(expected)) <= 1e-9
                for observed, expected in zip(distribution_implied_value, exact_value)
            ),
            "variant_index": variant_index,
        },
    )


def policy_variant_distribution_as_global_states(
    policy_dag: ExactPolicyDag,
    variant: ExactPolicyVariant,
) -> Dict[Any, float]:
    output: Dict[Any, float] = {}
    for signature, probability in variant.terminal_distribution.items():
        node = policy_dag.nodes[signature]
        global_signature = node.diagnostics.get("global_state_signature")
        if global_signature is None:
            global_signature = signature
        global_signature = tuple(tuple(row) for row in global_signature) if isinstance(global_signature, (tuple, list)) else global_signature
        output[global_signature] = output.get(global_signature, 0.0) + float(probability)
    return _normalize_distribution(output)


def extract_policy_dag_trace(
    *,
    policy_dag: ExactPolicyDag,
    max_decision_depth: int = 3,
    max_outcomes_per_action: int = 5,
    min_outcome_probability: float = 0.0,
) -> PolicyDagTrace:
    """Extract a compact canonical contingent trace without fabricating prose."""
    if int(max_decision_depth) < 0:
        raise ValueError("max_decision_depth must be non-negative")
    queue = deque([(policy_dag.root_state_signature, 0)])
    visited = set()
    entries = []
    truncated = False
    while queue:
        signature, depth = queue.popleft()
        if signature in visited:
            continue
        visited.add(signature)
        node = policy_dag.nodes.get(signature)
        if node is None or node.terminal:
            continue
        if depth > int(max_decision_depth):
            truncated = True
            continue
        action = next(
            (item for item in node.retained_actions if item.is_canonical_action), None
        )
        if action is None:
            continue
        ranked = sorted(
            zip(action.outcome_probabilities, action.successor_state_signatures),
            key=lambda item: (-float(item[0]), _stable_sort_key(item[1])),
        )
        outcomes = []
        for probability, successor in ranked:
            if float(probability) < float(min_outcome_probability):
                continue
            if len(outcomes) >= int(max_outcomes_per_action):
                truncated = True
                break
            child = policy_dag.nodes.get(successor)
            next_action = child.canonical_action_signature if child is not None else None
            outcomes.append(
                {
                    "probability": float(probability),
                    "successor_state_signature": successor,
                    "successor_state_rows": (
                        child.diagnostics.get("state_rows")
                        or child.diagnostics.get("parent_state_rows")
                        if child is not None
                        else ()
                    ),
                    "next_action_signature": next_action,
                    "next_action_label": (
                        next(
                            (
                                candidate.diagnostics.get("label", repr(next_action))
                                for candidate in (child.retained_actions if child is not None else ())
                                if candidate.is_canonical_action
                            ),
                            repr(next_action),
                        )
                    ),
                }
            )
            if child is not None and not child.terminal:
                next_depth = depth if node.diagnostics.get("state_kind") == "movement_decision" else depth + 1
                queue.append((successor, next_depth))
        entries.append(
            PolicyDagTraceEntry(
                state_signature=signature,
                decision_depth=int(depth),
                state_rows=tuple(
                    node.diagnostics.get("state_rows")
                    or node.diagnostics.get("parent_state_rows")
                    or ()
                ),
                action_signature=action.action_signature,
                action_label=str(action.diagnostics.get("label", repr(action.action_signature))),
                outcomes=tuple(outcomes),
                diagnostics={"state_kind": node.diagnostics.get("state_kind")},
            )
        )
    return PolicyDagTrace(
        root_state_signature=policy_dag.root_state_signature,
        entries=tuple(entries),
        max_decision_depth=int(max_decision_depth),
        truncated=bool(truncated),
        diagnostics={
            "entry_count": len(entries),
            "max_outcomes_per_action": int(max_outcomes_per_action),
            "min_outcome_probability": float(min_outcome_probability),
        },
    )


def _region_index_by_node(partition_signature: Sequence[Sequence[int]]) -> Dict[int, int]:
    output: Dict[int, int] = {}
    for region_index, region in enumerate(partition_signature):
        for node in region:
            output[int(node)] = int(region_index)
    return output


def _next_attack_actions(
    policy_dag: ExactPolicyDag,
    signature: Any,
    *,
    max_hops: int = 2,
) -> Tuple[Any, ...]:
    frontier = [(signature, 0)]
    output = []
    visited = set()
    while frontier:
        current, hops = frontier.pop()
        if current in visited or hops > max_hops:
            continue
        visited.add(current)
        node = policy_dag.nodes.get(current)
        if node is None or node.terminal:
            continue
        action = next((item for item in node.retained_actions if item.is_canonical_action), None)
        if action is None:
            continue
        if action.diagnostics.get("action_kind") == "attack":
            output.append(action)
            continue
        for successor in action.successor_state_signatures:
            frontier.append((successor, hops + 1))
    return tuple(output)


def analyze_policy_dag_sequence_behavior(
    *,
    policy_dag: ExactPolicyDag,
    graph: Any,
    former_partition_signature: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Measure front switching and sequence use visible in exported actions."""
    region_by_node = _region_index_by_node(former_partition_signature)
    cross_partition_actions = set()
    front_switch_states = set()
    outcome_dependent_switch_states = set()
    outcome_dependent_stop_states = set()
    sequence_opening_states = set()
    cross_partition_followup_states = set()
    action_fronts: Dict[Any, Any] = {}

    for signature, node in policy_dag.nodes.items():
        if node.terminal:
            continue
        action = next((item for item in node.retained_actions if item.is_canonical_action), None)
        if action is None or action.diagnostics.get("action_kind") != "attack":
            continue
        source = int(action.diagnostics.get("source_global", action.diagnostics.get("source_local")))
        target = int(action.diagnostics.get("target_global", action.diagnostics.get("target_local")))
        source_region = region_by_node.get(source)
        target_region = region_by_node.get(target)
        front = target_region if target_region is not None else (source_region, target_region)
        action_fronts[signature] = front
        if source_region is not None and target_region is not None and source_region != target_region:
            cross_partition_actions.add((signature, action.action_signature))

        next_fronts = set()
        has_stop = False
        has_continue = False
        for successor in action.successor_state_signatures:
            next_actions = _next_attack_actions(policy_dag, successor)
            if not next_actions:
                has_stop = True
                continue
            has_continue = True
            for next_action in next_actions:
                next_source = int(
                    next_action.diagnostics.get("source_global", next_action.diagnostics.get("source_local"))
                )
                next_target = int(
                    next_action.diagnostics.get("target_global", next_action.diagnostics.get("target_local"))
                )
                next_source_region = region_by_node.get(next_source)
                next_target_region = region_by_node.get(next_target)
                next_front = next_target_region if next_target_region is not None else (
                    next_source_region,
                    next_target_region,
                )
                next_fronts.add(next_front)
                if next_front != front:
                    front_switch_states.add(signature)
                if next_source_region != next_target_region:
                    cross_partition_followup_states.add(signature)
                if next_source == target:
                    sequence_opening_states.add(signature)
        if len(next_fronts) > 1:
            outcome_dependent_switch_states.add(signature)
        if has_stop and has_continue:
            outcome_dependent_stop_states.add(signature)

    canonical_trace_fronts = []
    current = policy_dag.root_state_signature
    seen = set()
    while current not in seen:
        seen.add(current)
        node = policy_dag.nodes.get(current)
        if node is None or node.terminal:
            break
        action = next((item for item in node.retained_actions if item.is_canonical_action), None)
        if action is None:
            break
        if action.diagnostics.get("action_kind") == "attack":
            canonical_trace_fronts.append(action_fronts.get(current))
        if not action.successor_state_signatures:
            break
        index = max(
            range(len(action.successor_state_signatures)),
            key=lambda i: float(action.outcome_probabilities[i]),
        )
        current = action.successor_state_signatures[index]
    front_switch_count = sum(
        left != right
        for left, right in zip(canonical_trace_fronts, canonical_trace_fronts[1:])
    )
    returns = any(
        canonical_trace_fronts[index] in canonical_trace_fronts[: max(0, index - 1)]
        and canonical_trace_fronts[index] != canonical_trace_fronts[index - 1]
        for index in range(1, len(canonical_trace_fronts))
    )
    return {
        "has_outcome_dependent_front_switch": bool(outcome_dependent_switch_states),
        "has_sequence_opening_action": bool(sequence_opening_states),
        "has_cross_partition_followup": bool(cross_partition_followup_states),
        "has_outcome_dependent_stop": bool(outcome_dependent_stop_states),
        "front_switch_state_count": len(front_switch_states),
        "outcome_dependent_front_switch_state_count": len(outcome_dependent_switch_states),
        "cross_partition_action_count": len(cross_partition_actions),
        "cross_partition_followup_state_count": len(cross_partition_followup_states),
        "sequence_opening_state_count": len(sequence_opening_states),
        "outcome_dependent_stop_state_count": len(outcome_dependent_stop_states),
        "canonical_most_likely_path_front_switch_count": int(front_switch_count),
        "canonical_most_likely_path_returns_to_previous_front": bool(returns),
        "measured_from_exported_actions_only": True,
        "survivor_redistribution_inferred": False,
        "former_partition_signature": tuple(tuple(int(node) for node in region) for region in former_partition_signature),
    }


__all__ = [
    "ExactPolicyDagAction",
    "ExactPolicyDagNode",
    "ExactPolicyDag",
    "ExactPolicyVariant",
    "PolicyDagTraceEntry",
    "PolicyDagTrace",
    "ExactPolicyDagExportCache",
    "PolicyDagConstructionLimit",
    "export_exact_policy_dag",
    "materialize_policy_variant",
    "policy_variant_distribution_as_global_states",
    "extract_policy_dag_trace",
    "analyze_policy_dag_sequence_behavior",
    "policy_dag_summary",
]
