"""Deterministic metrics for regional-compounding validation.

Complete successor-state distributions are primary. Marginals, strategic
events, and one-dimensional summaries are derived diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


ProbabilityDistribution = Mapping[Any, float]
CanonicalRiskState = Tuple[Tuple[int, str, int], ...]


def _owner_label(owner: Any) -> str:
    if hasattr(owner, "_name"):
        owner = owner._name
    if isinstance(owner, bool):
        return "A" if owner else "D"
    if isinstance(owner, int):
        if owner == 1:
            return "A"
        if owner == 2:
            return "D"
    text = str(owner)
    upper = text.upper()
    if upper in {"A", "ATTACKER"}:
        return "A"
    if upper in {"D", "DEFENDER"}:
        return "D"
    return text


def canonical_risk_state(state: Any) -> CanonicalRiskState:
    """Convert supported state representations to ``(node, owner, troops)``."""
    nodes = getattr(state, "nodes", None)
    if nodes is not None and not callable(nodes):
        return tuple(
            (int(index), _owner_label(node.owner), int(node.troops))
            for index, node in enumerate(nodes)
        )
    if isinstance(state, Mapping):
        rows = []
        for node, value in state.items():
            if hasattr(value, "owner") and hasattr(value, "troops"):
                owner, troops = value.owner, value.troops
            else:
                owner, troops = value
            rows.append((int(node), _owner_label(owner), int(troops)))
        return tuple(sorted(rows))
    if isinstance(state, (tuple, list)):
        rows = []
        for index, value in enumerate(state):
            if not isinstance(value, (tuple, list)):
                raise TypeError("State rows must be tuples or lists")
            if len(value) == 3:
                node, owner, troops = value
            elif len(value) == 2:
                node, (owner, troops) = index, value
            else:
                raise TypeError(f"Unsupported state row: {value!r}")
            rows.append((int(node), _owner_label(owner), int(troops)))
        return tuple(sorted(rows))
    raise TypeError(f"Unsupported Risk state representation: {type(state).__name__}")


def normalize_probability_distribution(
    distribution: ProbabilityDistribution,
    *,
    tolerance: float = 1e-12,
) -> Dict[Any, float]:
    out: Dict[Any, float] = {}
    for state, probability in distribution.items():
        value = float(probability)
        if not math.isfinite(value):
            raise ValueError("Distribution contains a non-finite probability")
        if value < -float(tolerance):
            raise ValueError("Distribution contains a negative probability")
        if value > 0.0:
            out[state] = out.get(state, 0.0) + value
    total = float(sum(out.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Distribution has no positive finite mass")
    return {state: value / total for state, value in out.items()}


def total_variation_distance(p: ProbabilityDistribution, q: ProbabilityDistribution) -> float:
    p2 = normalize_probability_distribution(p)
    q2 = normalize_probability_distribution(q)
    return 0.5 * sum(abs(p2.get(state, 0.0) - q2.get(state, 0.0)) for state in set(p2) | set(q2))


def jensen_shannon_divergence(p: ProbabilityDistribution, q: ProbabilityDistribution) -> float:
    p2 = normalize_probability_distribution(p)
    q2 = normalize_probability_distribution(q)
    answer = 0.0
    for state in set(p2) | set(q2):
        pv = p2.get(state, 0.0)
        qv = q2.get(state, 0.0)
        midpoint = 0.5 * (pv + qv)
        if pv > 0.0:
            answer += 0.5 * pv * math.log(pv / midpoint)
        if qv > 0.0:
            answer += 0.5 * qv * math.log(qv / midpoint)
    return max(0.0, float(answer))


def probability_mass_overlap(p: ProbabilityDistribution, q: ProbabilityDistribution) -> float:
    p2 = normalize_probability_distribution(p)
    q2 = normalize_probability_distribution(q)
    return float(sum(min(p2.get(state, 0.0), q2.get(state, 0.0)) for state in set(p2) | set(q2)))


def support_comparison(p: ProbabilityDistribution, q: ProbabilityDistribution) -> Dict[str, Any]:
    p2 = normalize_probability_distribution(p)
    q2 = normalize_probability_distribution(q)
    p_support = set(p2)
    q_support = set(q2)
    intersection = p_support & q_support
    union = p_support | q_support
    return {
        "p_support_size": len(p_support),
        "q_support_size": len(q_support),
        "support_intersection_size": len(intersection),
        "support_union_size": len(union),
        "support_jaccard_index": float(len(intersection)) / float(len(union)) if union else 1.0,
        "p_mass_on_intersection": float(sum(p2[state] for state in intersection)),
        "q_mass_on_intersection": float(sum(q2[state] for state in intersection)),
    }


def _top_states(distribution: ProbabilityDistribution, k: int) -> Tuple[Any, ...]:
    normalized = normalize_probability_distribution(distribution)
    return tuple(
        state
        for state, _probability in sorted(
            normalized.items(), key=lambda item: (-item[1], repr(item[0]))
        )[: max(0, int(k))]
    )


def top_k_state_overlap(p: ProbabilityDistribution, q: ProbabilityDistribution, k: int) -> float:
    p_top = _top_states(p, k)
    q_top = _top_states(q, k)
    denominator = min(max(0, int(k)), len(p_top), len(q_top))
    if denominator == 0:
        return 1.0
    return float(len(set(p_top) & set(q_top))) / float(denominator)


def compare_distributions(p: ProbabilityDistribution, q: ProbabilityDistribution) -> Dict[str, Any]:
    p_top = _top_states(p, 1)
    q_top = _top_states(q, 1)
    result = support_comparison(p, q)
    result.update(
        {
            "total_variation": total_variation_distance(p, q),
            "jensen_shannon": jensen_shannon_divergence(p, q),
            "probability_mass_overlap": probability_mass_overlap(p, q),
            "top_1_agreement": bool(p_top == q_top),
            "top_3_overlap": top_k_state_overlap(p, q, 3),
            "top_5_overlap": top_k_state_overlap(p, q, 5),
            "top_10_overlap": top_k_state_overlap(p, q, 10),
        }
    )
    return result


def risk_state_distance(
    state_a: Any,
    state_b: Any,
    *,
    ownership_weight: float = 1.0,
    troop_weight: float = 1.0,
    troop_scale: Optional[float] = None,
) -> float:
    left = {node: (owner, troops) for node, owner, troops in canonical_risk_state(state_a)}
    right = {node: (owner, troops) for node, owner, troops in canonical_risk_state(state_b)}
    if set(left) != set(right):
        raise ValueError("Risk states must contain the same labelled nodes")
    if not left:
        return 0.0
    scale = troop_scale
    if scale is None:
        scale = max(1.0, *(float(troops) for _owner, troops in left.values()), *(float(troops) for _owner, troops in right.values()))
    if float(scale) <= 0.0:
        raise ValueError("troop_scale must be positive")
    total = 0.0
    for node in sorted(left):
        owner_a, troops_a = left[node]
        owner_b, troops_b = right[node]
        total += float(ownership_weight) * float(owner_a != owner_b)
        total += float(troop_weight) * abs(float(troops_a) - float(troops_b)) / float(scale)
    return total / float(len(left))


@dataclass(frozen=True)
class StateAwareDistanceResult:
    status: str
    distance: Optional[float]
    p_support_size: int
    q_support_size: int
    union_support_size: int
    ownership_weight: float
    troop_weight: float
    troop_scale: Optional[float]
    diagnostics: Mapping[str, Any]


def _add_flow_edge(graph: list, source: int, target: int, capacity: float, cost: float) -> None:
    forward = [target, len(graph[target]), float(capacity), float(cost)]
    reverse = [source, len(graph[source]), 0.0, -float(cost)]
    graph[source].append(forward)
    graph[target].append(reverse)


def _transport_min_cost(supplies: Sequence[float], demands: Sequence[float], costs: Sequence[Sequence[float]]) -> float:
    m, n = len(supplies), len(demands)
    source = 0
    p_start = 1
    q_start = p_start + m
    sink = q_start + n
    graph = [[] for _ in range(sink + 1)]
    for i, supply in enumerate(supplies):
        _add_flow_edge(graph, source, p_start + i, float(supply), 0.0)
    for i in range(m):
        for j in range(n):
            _add_flow_edge(graph, p_start + i, q_start + j, 1.0, float(costs[i][j]))
    for j, demand in enumerate(demands):
        _add_flow_edge(graph, q_start + j, sink, float(demand), 0.0)

    flow = 0.0
    cost_total = 0.0
    epsilon = 1e-13
    while flow < 1.0 - 1e-10:
        distances = [math.inf] * len(graph)
        parents: list[Optional[Tuple[int, int]]] = [None] * len(graph)
        distances[source] = 0.0
        for _ in range(len(graph) - 1):
            changed = False
            for u, edges in enumerate(graph):
                if not math.isfinite(distances[u]):
                    continue
                for edge_index, edge in enumerate(edges):
                    v, _reverse, capacity, edge_cost = edge
                    if capacity <= epsilon:
                        continue
                    candidate = distances[u] + edge_cost
                    if candidate < distances[v] - 1e-15:
                        distances[v] = candidate
                        parents[v] = (u, edge_index)
                        changed = True
            if not changed:
                break
        if parents[sink] is None:
            raise RuntimeError("Transportation network could not route all probability mass")
        amount = 1.0 - flow
        node = sink
        while node != source:
            u, edge_index = parents[node]  # type: ignore[misc]
            amount = min(amount, graph[u][edge_index][2])
            node = u
        node = sink
        while node != source:
            u, edge_index = parents[node]  # type: ignore[misc]
            edge = graph[u][edge_index]
            reverse_index = edge[1]
            edge[2] -= amount
            graph[node][reverse_index][2] += amount
            node = u
        flow += amount
        cost_total += amount * distances[sink]
    return float(cost_total)


def risk_state_wasserstein_distance(
    p: ProbabilityDistribution,
    q: ProbabilityDistribution,
    *,
    state_distance_config: Optional[Mapping[str, Any]] = None,
    max_support_size: Optional[int] = None,
) -> StateAwareDistanceResult:
    config = dict(state_distance_config or {})
    ownership_weight = float(config.get("ownership_weight", 1.0))
    troop_weight = float(config.get("troop_weight", 1.0))
    p2 = normalize_probability_distribution(p)
    q2 = normalize_probability_distribution(q)
    union_size = len(set(p2) | set(q2))
    troop_scale = config.get("troop_scale")
    if troop_scale is None:
        all_troops = [
            float(troops)
            for state in set(p2) | set(q2)
            for _node, _owner, troops in canonical_risk_state(state)
        ]
        troop_scale = max([1.0] + all_troops)
    if max_support_size is not None and union_size > int(max_support_size):
        return StateAwareDistanceResult(
            status="support_limit",
            distance=None,
            p_support_size=len(p2),
            q_support_size=len(q2),
            union_support_size=union_size,
            ownership_weight=ownership_weight,
            troop_weight=troop_weight,
            troop_scale=float(troop_scale),
            diagnostics={"max_support_size": int(max_support_size), "solver": "exact_min_cost_flow"},
        )
    p_states = tuple(sorted(p2, key=repr))
    q_states = tuple(sorted(q2, key=repr))
    costs = tuple(
        tuple(
            risk_state_distance(
                left,
                right,
                ownership_weight=ownership_weight,
                troop_weight=troop_weight,
                troop_scale=float(troop_scale),
            )
            for right in q_states
        )
        for left in p_states
    )
    try:
        distance = _transport_min_cost(
            tuple(p2[state] for state in p_states),
            tuple(q2[state] for state in q_states),
            costs,
        )
    except Exception as exc:
        return StateAwareDistanceResult(
            status="solver_error",
            distance=None,
            p_support_size=len(p2),
            q_support_size=len(q2),
            union_support_size=union_size,
            ownership_weight=ownership_weight,
            troop_weight=troop_weight,
            troop_scale=float(troop_scale),
            diagnostics={"solver": "exact_min_cost_flow", "error": f"{type(exc).__name__}: {exc}"},
        )
    return StateAwareDistanceResult(
        status="exact_complete",
        distance=distance,
        p_support_size=len(p2),
        q_support_size=len(q2),
        union_support_size=union_size,
        ownership_weight=ownership_weight,
        troop_weight=troop_weight,
        troop_scale=float(troop_scale),
        diagnostics={"solver": "exact_min_cost_flow", "approximated": False},
    )


def _state_map(state: Any) -> Dict[int, Tuple[str, int]]:
    return {node: (owner, troops) for node, owner, troops in canonical_risk_state(state)}


def derive_distribution_node_marginals(
    distribution: ProbabilityDistribution,
    *,
    initial_state: Optional[Any] = None,
) -> Dict[int, Dict[str, Any]]:
    normalized = normalize_probability_distribution(distribution)
    initial = _state_map(initial_state) if initial_state is not None else {}
    result: Dict[int, Dict[str, Any]] = {}
    for state, probability in normalized.items():
        for node, owner, troops in canonical_risk_state(state):
            row = result.setdefault(
                int(node),
                {
                    "attacker_ownership_probability": 0.0,
                    "defender_ownership_probability": 0.0,
                    "expected_troops": 0.0,
                    "troop_count_distribution": {},
                    "ownership_changed_probability": 0.0,
                },
            )
            if owner == "A":
                row["attacker_ownership_probability"] += probability
            else:
                row["defender_ownership_probability"] += probability
            row["expected_troops"] += probability * int(troops)
            troop_dist = row["troop_count_distribution"]
            troop_dist[int(troops)] = troop_dist.get(int(troops), 0.0) + probability
            if node in initial and owner != initial[node][0]:
                row["ownership_changed_probability"] += probability
    return result


def compare_node_marginals(
    exact_distribution: ProbabilityDistribution,
    approximate_distribution: ProbabilityDistribution,
    *,
    initial_state: Optional[Any] = None,
    partition_boundary_nodes: Iterable[int] = (),
    articulation_nodes: Iterable[int] = (),
) -> Dict[str, Any]:
    exact = derive_distribution_node_marginals(exact_distribution, initial_state=initial_state)
    approximate = derive_distribution_node_marginals(approximate_distribution, initial_state=initial_state)
    initial = _state_map(initial_state) if initial_state is not None else {}
    boundary = {int(node) for node in partition_boundary_nodes}
    articulations = {int(node) for node in articulation_nodes}
    per_node: Dict[int, Dict[str, Any]] = {}
    ownership_errors = []
    troop_errors = []
    for node in sorted(set(exact) | set(approximate)):
        exact_row = exact.get(node, {})
        approximate_row = approximate.get(node, {})
        ownership_error = abs(
            float(exact_row.get("attacker_ownership_probability", 0.0))
            - float(approximate_row.get("attacker_ownership_probability", 0.0))
        )
        troop_error = abs(
            float(exact_row.get("expected_troops", 0.0))
            - float(approximate_row.get("expected_troops", 0.0))
        )
        troop_tv = total_variation_distance(
            exact_row.get("troop_count_distribution", {0: 1.0}),
            approximate_row.get("troop_count_distribution", {0: 1.0}),
        )
        ownership_errors.append(ownership_error)
        troop_errors.append(troop_error)
        per_node[node] = {
            "ownership_probability_error": ownership_error,
            "expected_troop_error": troop_error,
            "troop_distribution_tv": troop_tv,
            "exact": exact_row,
            "approximate": approximate_row,
            "is_partition_boundary": node in boundary,
            "is_articulation_node": node in articulations,
            "initial_owner": initial.get(node, (None, None))[0],
        }
    largest_node = max(per_node, key=lambda node: (per_node[node]["ownership_probability_error"], per_node[node]["expected_troop_error"], -node), default=None)
    return {
        "mean_absolute_ownership_probability_error": sum(ownership_errors) / len(ownership_errors) if ownership_errors else 0.0,
        "maximum_ownership_probability_error": max(ownership_errors, default=0.0),
        "mean_absolute_expected_troop_error": sum(troop_errors) / len(troop_errors) if troop_errors else 0.0,
        "maximum_expected_troop_error": max(troop_errors, default=0.0),
        "largest_error_node": largest_node,
        "per_node": per_node,
    }


def _adjacency(nodes: Iterable[int], edges: Iterable[Tuple[int, int]]) -> Dict[int, set[int]]:
    answer = {int(node): set() for node in nodes}
    for left, right in edges:
        left, right = int(left), int(right)
        answer.setdefault(left, set()).add(right)
        answer.setdefault(right, set()).add(left)
    return answer


def _component_sizes(nodes: Iterable[int], adjacency: Mapping[int, set[int]]) -> Tuple[int, ...]:
    remaining = set(int(node) for node in nodes)
    sizes = []
    while remaining:
        start = min(remaining)
        stack = [start]
        remaining.remove(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency.get(node, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))


def strategic_event_probabilities(
    distribution: ProbabilityDistribution,
    *,
    initial_state: Any,
    edges: Iterable[Tuple[int, int]] = (),
    at_least_k_values: Sequence[int] = (1, 2),
    articulation_nodes: Iterable[int] = (),
    key_territories: Iterable[int] = (),
) -> Dict[str, float]:
    normalized = normalize_probability_distribution(distribution)
    initial = _state_map(initial_state)
    nodes = tuple(sorted(initial))
    initial_attackers = {node for node, (owner, _troops) in initial.items() if owner == "A"}
    initial_defenders = {node for node, (owner, _troops) in initial.items() if owner == "D"}
    articulation_targets = initial_defenders & {int(node) for node in articulation_nodes}
    key_targets = initial_defenders & {int(node) for node in key_territories}
    adjacency = _adjacency(nodes, edges)
    events: Dict[str, float] = {
        "complete_conquest": 0.0,
        "continent_completion": 0.0,
        "no_territory_gained": 0.0,
        "at_least_one_territory_gained": 0.0,
        "attacker_retains_all_initial_territories": 0.0,
        "defender_retains_at_least_one_territory": 0.0,
        "articulation_territory_captured": 0.0,
        "key_territory_captured": 0.0,
        "attacker_remains_connected": 0.0,
        "attacker_ends_in_one_connected_component": 0.0,
    }
    for k in sorted({int(value) for value in at_least_k_values if int(value) >= 0}):
        events[f"at_least_{k}_territories_gained"] = 0.0
        events[f"exactly_{k}_territories_gained"] = 0.0
    for state, probability in normalized.items():
        current = _state_map(state)
        attacker_nodes = {node for node in nodes if current[node][0] == "A"}
        gained = len(initial_defenders & attacker_nodes)
        conquest = not (initial_defenders - attacker_nodes)
        events["complete_conquest"] += probability * conquest
        events["continent_completion"] += probability * conquest
        events["no_territory_gained"] += probability * (gained == 0)
        events["at_least_one_territory_gained"] += probability * (gained >= 1)
        events["attacker_retains_all_initial_territories"] += probability * initial_attackers.issubset(attacker_nodes)
        events["defender_retains_at_least_one_territory"] += probability * bool(initial_defenders - attacker_nodes)
        events["articulation_territory_captured"] += probability * bool(articulation_targets & attacker_nodes)
        events["key_territory_captured"] += probability * bool(key_targets & attacker_nodes)
        component_count = len(_component_sizes(attacker_nodes, adjacency)) if attacker_nodes else 0
        connected = component_count <= 1
        events["attacker_remains_connected"] += probability * connected
        events["attacker_ends_in_one_connected_component"] += probability * connected
        for k in sorted({int(value) for value in at_least_k_values if int(value) >= 0}):
            events[f"at_least_{k}_territories_gained"] += probability * (gained >= k)
            events[f"exactly_{k}_territories_gained"] += probability * (gained == k)
    return {name: float(value) for name, value in events.items()}


def compare_strategic_events(
    exact_distribution: ProbabilityDistribution,
    approximate_distribution: ProbabilityDistribution,
    **kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    exact = strategic_event_probabilities(exact_distribution, **kwargs)
    approximate = strategic_event_probabilities(approximate_distribution, **kwargs)
    return {
        name: {
            "exact_probability": exact.get(name, 0.0),
            "approx_probability": approximate.get(name, 0.0),
            "absolute_error": abs(approximate.get(name, 0.0) - exact.get(name, 0.0)),
            "signed_error": approximate.get(name, 0.0) - exact.get(name, 0.0),
        }
        for name in sorted(set(exact) | set(approximate))
    }


def _one_dimensional_wasserstein(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    p = normalize_probability_distribution(left)
    q = normalize_probability_distribution(right)
    points = sorted(set(int(value) for value in p) | set(int(value) for value in q))
    if len(points) <= 1:
        return 0.0
    cdf_difference = 0.0
    answer = 0.0
    for index, point in enumerate(points[:-1]):
        cdf_difference += p.get(point, 0.0) - q.get(point, 0.0)
        answer += abs(cdf_difference) * float(points[index + 1] - point)
    return float(answer)


def strategic_summary_distributions(
    distribution: ProbabilityDistribution,
    *,
    initial_state: Any,
    edges: Iterable[Tuple[int, int]] = (),
) -> Dict[str, Dict[int, float]]:
    normalized = normalize_probability_distribution(distribution)
    initial = _state_map(initial_state)
    nodes = tuple(sorted(initial))
    initial_defenders = {node for node, (owner, _troops) in initial.items() if owner == "D"}
    adjacency = _adjacency(nodes, edges)
    names = (
        "new_attacker_territories",
        "final_attacker_owned_territories",
        "final_attacker_troop_total",
        "final_defender_troop_total",
        "attacker_connected_components",
        "largest_attacker_connected_component",
        "surviving_defender_components",
    )
    result = {name: {} for name in names}
    for state, probability in normalized.items():
        current = _state_map(state)
        attackers = {node for node in nodes if current[node][0] == "A"}
        defenders = set(nodes) - attackers
        attacker_sizes = _component_sizes(attackers, adjacency) if attackers else ()
        defender_sizes = _component_sizes(defenders, adjacency) if defenders else ()
        values = {
            "new_attacker_territories": len(initial_defenders & attackers),
            "final_attacker_owned_territories": len(attackers),
            "final_attacker_troop_total": sum(current[node][1] for node in attackers),
            "final_defender_troop_total": sum(current[node][1] for node in defenders),
            "attacker_connected_components": len(attacker_sizes),
            "largest_attacker_connected_component": max(attacker_sizes, default=0),
            "surviving_defender_components": len(defender_sizes),
        }
        for name, value in values.items():
            result[name][int(value)] = result[name].get(int(value), 0.0) + probability
    return result


def _pmf_mean_variance(distribution: Mapping[int, float]) -> Tuple[float, float]:
    normalized = normalize_probability_distribution(distribution)
    mean = sum(float(value) * probability for value, probability in normalized.items())
    variance = sum((float(value) - mean) ** 2 * probability for value, probability in normalized.items())
    return float(mean), float(variance)


def compare_strategic_summaries(
    exact_distribution: ProbabilityDistribution,
    approximate_distribution: ProbabilityDistribution,
    *,
    initial_state: Any,
    edges: Iterable[Tuple[int, int]] = (),
) -> Dict[str, Any]:
    exact = strategic_summary_distributions(exact_distribution, initial_state=initial_state, edges=edges)
    approximate = strategic_summary_distributions(approximate_distribution, initial_state=initial_state, edges=edges)
    answer: Dict[str, Any] = {}
    for name in sorted(exact):
        exact_mean, exact_variance = _pmf_mean_variance(exact[name])
        approximate_mean, approximate_variance = _pmf_mean_variance(approximate[name])
        thresholds = sorted(set(exact[name]) | set(approximate[name]))
        tail_errors = [
            abs(
                sum(prob for value, prob in exact[name].items() if value >= threshold)
                - sum(prob for value, prob in approximate[name].items() if value >= threshold)
            )
            for threshold in thresholds
        ]
        answer[name] = {
            "exact_expectation": exact_mean,
            "approximate_expectation": approximate_mean,
            "expectation_difference": approximate_mean - exact_mean,
            "absolute_expectation_error": abs(approximate_mean - exact_mean),
            "exact_variance": exact_variance,
            "approximate_variance": approximate_variance,
            "variance_difference": approximate_variance - exact_variance,
            "one_dimensional_tv": total_variation_distance(exact[name], approximate[name]),
            "one_dimensional_wasserstein": _one_dimensional_wasserstein(exact[name], approximate[name]),
            "maximum_tail_probability_difference": max(tail_errors, default=0.0),
            "exact_distribution": exact[name],
            "approximate_distribution": approximate[name],
        }
    return answer


def _region_event_values(
    state: Mapping[int, Tuple[str, int]],
    initial: Mapping[int, Tuple[str, int]],
    region_nodes: Sequence[int],
    troop_threshold: int,
) -> Dict[str, bool]:
    nodes = {int(node) for node in region_nodes}
    initial_defenders = {node for node in nodes if initial[node][0] == "D"}
    current_attackers = {node for node in nodes if state[node][0] == "A"}
    gained = initial_defenders & current_attackers
    return {
        "full_regional_conquest": initial_defenders.issubset(current_attackers),
        "at_least_one_defender_node_captured": bool(gained),
        "positive_territorial_gain": bool(gained),
        "attacker_survives_in_region": bool(current_attackers),
        "attacker_above_troop_threshold": sum(state[node][1] for node in current_attackers) >= int(troop_threshold),
    }


def _dependence_for_distribution(
    distribution: ProbabilityDistribution,
    *,
    initial_state: Any,
    regions: Sequence[Sequence[int]],
    troop_threshold: int,
) -> Dict[str, Any]:
    normalized = normalize_probability_distribution(distribution)
    initial = _state_map(initial_state)
    event_names = (
        "full_regional_conquest",
        "at_least_one_defender_node_captured",
        "positive_territorial_gain",
        "attacker_survives_in_region",
        "attacker_above_troop_threshold",
    )
    per_state = []
    for state, probability in normalized.items():
        mapped = _state_map(state)
        per_state.append(
            (
                probability,
                tuple(_region_event_values(mapped, initial, region, troop_threshold) for region in regions),
            )
        )
    result: Dict[str, Any] = {"events": {}}
    for event in event_names:
        marginals = [
            sum(probability for probability, values in per_state if values[index][event])
            for index in range(len(regions))
        ]
        pairs: Dict[Tuple[int, int], Dict[str, float]] = {}
        for left in range(len(regions)):
            for right in range(left + 1, len(regions)):
                joint = sum(
                    probability
                    for probability, values in per_state
                    if values[left][event] and values[right][event]
                )
                p_left, p_right = marginals[left], marginals[right]
                covariance = joint - p_left * p_right
                denominator = math.sqrt(max(0.0, p_left * (1.0 - p_left) * p_right * (1.0 - p_right)))
                pairs[(left, right)] = {
                    "p_left": p_left,
                    "p_right": p_right,
                    "p_both": joint,
                    "p_right_given_left": joint / p_left if p_left > 0.0 else 0.0,
                    "p_right_given_left_failure": (p_right - joint) / (1.0 - p_left) if p_left < 1.0 else 0.0,
                    "covariance": covariance,
                    "correlation": covariance / denominator if denominator > 0.0 else 0.0,
                }
        count_distribution: Dict[int, float] = {}
        for probability, values in per_state:
            successes = sum(int(value[event]) for value in values)
            count_distribution[successes] = count_distribution.get(successes, 0.0) + probability
        result["events"][event] = {
            "region_success_probabilities": tuple(marginals),
            "region_pairs": pairs,
            "successful_region_count_distribution": count_distribution,
            "all_regions_succeed": count_distribution.get(len(regions), 0.0),
            "no_regions_succeed": count_distribution.get(0, 0.0),
            "exactly_one_region_succeeds": count_distribution.get(1, 0.0),
        }
    return result


def compare_cross_region_dependence(
    exact_distribution: ProbabilityDistribution,
    approximate_distribution: ProbabilityDistribution,
    *,
    initial_state: Any,
    regions: Sequence[Sequence[int]],
    troop_threshold: int = 2,
) -> Dict[str, Any]:
    exact = _dependence_for_distribution(
        exact_distribution,
        initial_state=initial_state,
        regions=regions,
        troop_threshold=troop_threshold,
    )
    approximate = _dependence_for_distribution(
        approximate_distribution,
        initial_state=initial_state,
        regions=regions,
        troop_threshold=troop_threshold,
    )
    comparisons: Dict[str, Any] = {}
    max_covariance_error = 0.0
    max_joint_error = 0.0
    for event, exact_event in exact["events"].items():
        approximate_event = approximate["events"][event]
        pair_errors = {}
        for pair, exact_pair in exact_event["region_pairs"].items():
            approximate_pair = approximate_event["region_pairs"][pair]
            row = {
                key + "_error": approximate_pair[key] - exact_pair[key]
                for key in (
                    "p_both",
                    "p_right_given_left",
                    "p_right_given_left_failure",
                    "covariance",
                    "correlation",
                )
            }
            pair_errors[pair] = row
            max_covariance_error = max(max_covariance_error, abs(row["covariance_error"]))
            max_joint_error = max(max_joint_error, abs(row["p_both_error"]))
        comparisons[event] = {
            "exact": exact_event,
            "approximate": approximate_event,
            "pair_errors": pair_errors,
            "successful_region_count_tv": total_variation_distance(
                exact_event["successful_region_count_distribution"],
                approximate_event["successful_region_count_distribution"],
            ),
            "all_regions_succeed_error": approximate_event["all_regions_succeed"] - exact_event["all_regions_succeed"],
            "no_regions_succeed_error": approximate_event["no_regions_succeed"] - exact_event["no_regions_succeed"],
            "exactly_one_region_succeeds_error": approximate_event["exactly_one_region_succeeds"] - exact_event["exactly_one_region_succeeds"],
        }
    return {
        "events": comparisons,
        "maximum_absolute_covariance_error": max_covariance_error,
        "maximum_absolute_joint_success_error": max_joint_error,
        "compounded_model_assumption": "selected regional outcome distributions are composed independently",
    }


STATE_DISTANCE_PROFILES: Mapping[str, Mapping[str, float]] = {
    "ownership_dominant": {"ownership_weight": 3.0, "troop_weight": 1.0},
    "balanced": {"ownership_weight": 1.0, "troop_weight": 1.0},
    "troop_dominant": {"ownership_weight": 1.0, "troop_weight": 3.0},
}
