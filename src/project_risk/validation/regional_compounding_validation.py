"""Validation framework for exact full-graph versus regional composition.

The module is additive. It does not alter production partition selection and
keeps exact-policy regret distinct from terminal-distribution value gaps.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import random
import statistics
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.continent_model import battle_graph_ranking as bgr
from project_risk.validation.distribution_comparison_metrics import (
    STATE_DISTANCE_PROFILES,
    canonical_risk_state,
    compare_cross_region_dependence,
    compare_distributions,
    compare_node_marginals,
    compare_strategic_events,
    compare_strategic_summaries,
    normalize_probability_distribution,
    risk_state_wasserstein_distance,
    total_variation_distance,
)
from project_risk.mathematical.small_graph_model.exact_finite_solver import (
    CompactExactTopologySolver,
    ExactSolverLimitReached,
    combat_df_for_caps,
)
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState, NodeState


VALIDATION_FORMAT_VERSION = "regional_compounding_validation_v1"
VALIDATION_SCHEMA_VERSION = "regional_compounding_validation_schema_1"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_ready(item) for item in value]
    return repr(value)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _graph_nodes_edges(graph: Any) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    if isinstance(graph, Mapping):
        nodes = tuple(sorted(int(node) for node in graph.get("nodes", ())))
        edges_raw = graph.get("edges", ())
    elif isinstance(graph, (tuple, list)) and (not graph or len(graph[0]) == 2):
        edges_raw = graph
        nodes = tuple(sorted({int(node) for edge in edges_raw for node in edge}))
    else:
        nodes_attr = graph.nodes()
        edges_raw = graph.edges()
        nodes = tuple(sorted(int(node) for node in nodes_attr))
    edges = tuple(
        sorted(
            {
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in edges_raw
                if int(left) != int(right)
            }
        )
    )
    if not nodes:
        nodes = tuple(sorted({node for edge in edges for node in edge}))
    return nodes, edges


class ValidationGraph:
    """Small networkx-compatible graph used by the validation runner."""

    def __init__(self, nodes: Iterable[int], edges: Iterable[Tuple[int, int]]) -> None:
        self._adj = {int(node): set() for node in nodes}
        for left, right in edges:
            left, right = int(left), int(right)
            self._adj.setdefault(left, set()).add(right)
            self._adj.setdefault(right, set()).add(left)

    def nodes(self) -> List[int]:
        return sorted(self._adj)

    def edges(self) -> List[Tuple[int, int]]:
        return sorted(
            (node, neighbor)
            for node, neighbors in self._adj.items()
            for neighbor in neighbors
            if node < neighbor
        )

    def neighbors(self, node: int) -> List[int]:
        return sorted(self._adj.get(int(node), ()))

    def has_edge(self, left: int, right: int) -> bool:
        return int(right) in self._adj.get(int(left), set())

    def degree(self, node: Optional[int] = None) -> Any:
        if node is None:
            return [(item, len(neighbors)) for item, neighbors in sorted(self._adj.items())]
        return len(self._adj.get(int(node), ()))

    def number_of_nodes(self) -> int:
        return len(self._adj)

    def number_of_edges(self) -> int:
        return len(self.edges())


def _as_validation_graph(graph: Any) -> ValidationGraph:
    if isinstance(graph, ValidationGraph):
        return graph
    nodes, edges = _graph_nodes_edges(graph)
    return ValidationGraph(nodes, edges)


def _global_state_for_nodes(state: Any, nodes: Sequence[int]) -> GlobalState:
    canonical = {node: (owner, troops) for node, owner, troops in canonical_risk_state(state)}
    missing = set(int(node) for node in nodes) - set(canonical)
    if missing:
        raise ValueError(f"Initial state is missing graph nodes: {sorted(missing)}")
    maximum = max(nodes, default=-1)
    return GlobalState(
        nodes=tuple(
            NodeState(*canonical.get(index, ("D", 1)))
            for index in range(maximum + 1)
        )
    )


def canonical_validation_state_signature(state: Any, nodes: Sequence[int]) -> Tuple[Tuple[int, str, int], ...]:
    mapped = {node: (owner, troops) for node, owner, troops in canonical_risk_state(state)}
    return tuple((int(node), str(mapped[int(node)][0]), int(mapped[int(node)][1])) for node in sorted(nodes))


@dataclass(frozen=True)
class ExactComposedDistributionResult:
    status: str
    distribution: Mapping[Any, float]
    raw_cartesian_expansions: int
    unique_states_after_each_region: Tuple[int, ...]
    final_unique_state_count: int
    probability_mass: float
    runtime_seconds: float
    limit_reached: Optional[str]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ExactFullGraphReferenceResult:
    status: str
    graph_signature: Any
    initial_state_signature: Any
    optimal_value: Any
    canonical_optimal_policy: Any
    optimal_policy_set: Tuple[Any, ...]
    canonical_optimal_distribution: Mapping[Any, float]
    optimal_policy_distributions: Tuple[Mapping[Any, float], ...]
    states_evaluated: int
    cache_hits: int
    actions_evaluated: int
    runtime_seconds: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ExactPolicyEvaluationResult:
    status: str
    value: Any
    distribution: Mapping[Any, float]
    runtime_seconds: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RegionalApproximationResult:
    status: str
    supported_partitions: Tuple[Any, ...]
    maximal_partitions: Tuple[Any, ...]
    retained_candidates: Tuple[Any, ...]
    selected_candidate: Any
    selected_partition_signature: Any
    selected_policy_option_indices: Tuple[int, ...]
    selected_candidate_local_utility: Any
    selected_candidate_second_stage_value: Any
    exact_compounded_distribution: Optional[Mapping[Any, float]]
    composition_status: str
    composition_diagnostics: Mapping[str, Any]
    runtime_seconds: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RegionalCompoundingValidationRecord:
    benchmark_id: str
    graph_signature: Any
    topology_family: str
    graph_descriptors: Mapping[str, Any]
    initial_state_signature: Any
    exact_status: str
    approximation_status: str
    exact_composition_status: str
    exact_optimal_value: Any
    approximate_policy_exact_value: Any
    regret: Mapping[str, Any]
    exact_policy_count: int
    approximation_selected_policy: Any
    distribution_metrics: Mapping[str, Any]
    state_aware_metrics: Mapping[str, Any]
    node_marginal_metrics: Mapping[str, Any]
    strategic_event_metrics: Mapping[str, Any]
    strategic_summary_metrics: Mapping[str, Any]
    cross_region_dependence_metrics: Mapping[str, Any]
    exact_runtime_seconds: float
    approximation_runtime_seconds: float
    exact_composition_runtime_seconds: float
    diagnostics: Mapping[str, Any]


def _partial_composition_result(
    *,
    status: str,
    distribution: Mapping[Any, float],
    expansions: int,
    unique_counts: Sequence[int],
    started: float,
    limit: Optional[str],
    diagnostics: Mapping[str, Any],
) -> ExactComposedDistributionResult:
    return ExactComposedDistributionResult(
        status=status,
        distribution=dict(distribution),
        raw_cartesian_expansions=int(expansions),
        unique_states_after_each_region=tuple(int(value) for value in unique_counts),
        final_unique_state_count=len(distribution),
        probability_mass=float(sum(distribution.values())),
        runtime_seconds=float(time.perf_counter() - started),
        limit_reached=limit,
        diagnostics=dict(diagnostics),
    )


def compose_selected_candidate_distribution_exact(
    *,
    selected_candidate: bgr.PartitionPolicyCandidate,
    initial_global_state: GlobalState,
    node_indices: Optional[Sequence[int]] = None,
    max_unique_states: Optional[int] = None,
    max_cartesian_expansions: Optional[int] = None,
    max_runtime_seconds: Optional[float] = None,
    prepared_regional_options: Optional[Mapping[Any, Any]] = None,
    prepared_regional_options_diagnostics: Optional[Mapping[str, Any]] = None,
) -> ExactComposedDistributionResult:
    """Enumerate the exact product distribution for one selected candidate."""
    started = time.perf_counter()
    preparation_seconds = 0.0
    outcome_application_seconds = 0.0
    hashing_merging_seconds = 0.0
    normalization_seconds = 0.0
    region_sets = [set(int(node) for node in ref.region_nodes) for ref in selected_candidate.region_policy_options]
    overlap = set()
    for index, left in enumerate(region_sets):
        for right in region_sets[index + 1 :]:
            overlap.update(left & right)
    if overlap:
        return _partial_composition_result(
            status="invalid_region_overlap",
            distribution={},
            expansions=0,
            unique_counts=(),
            started=started,
            limit=None,
            diagnostics={"overlapping_nodes": tuple(sorted(overlap))},
        )
    try:
        if prepared_regional_options is None:
            preparation_started = time.perf_counter()
            prepared, preparation_diagnostics = bgr.prepare_unique_regional_policy_options(
                (selected_candidate,)
            )
            preparation_seconds = float(time.perf_counter() - preparation_started)
            preparation_reused = False
        else:
            prepared = dict(prepared_regional_options)
            preparation_diagnostics = dict(prepared_regional_options_diagnostics or {})
            preparation_reused = True
    except Exception as exc:
        return _partial_composition_result(
            status="unsupported_payload",
            distribution={},
            expansions=0,
            unique_counts=(),
            started=started,
            limit=None,
            diagnostics={
                "error": f"{type(exc).__name__}: {exc}",
                "regional_distribution_preparation_seconds": preparation_seconds,
            },
        )
    ordered = []
    for ref in selected_candidate.region_policy_options:
        key = bgr.canonical_region_policy_option_key(ref)
        option = prepared.get(key)
        if option is None or not option.normalized_distribution:
            return _partial_composition_result(
                status="unsupported_payload",
                distribution={},
                expansions=0,
                unique_counts=(),
                started=started,
                limit=None,
                diagnostics={"missing_region_option": repr(key)},
            )
        ordered.append(option)
    regional_support_sizes = tuple(len(option.normalized_distribution) for option in ordered)
    raw_cartesian_product_size = int(math.prod(regional_support_sizes)) if ordered else 1
    signature_nodes = tuple(
        sorted(
            {int(node) for node in (node_indices or ())}
            or set(range(len(initial_global_state.nodes)))
        )
    )
    initial_signature = bgr.canonical_two_stage_global_state_signature(
        initial_global_state, node_indices=signature_nodes
    )
    distribution: Dict[Any, float] = {initial_signature: 1.0}
    expansions = 0
    unique_counts: List[int] = []
    expansions_after_each_region: List[int] = []
    duplicates_merged_after_each_region: List[int] = []

    def complexity_diagnostics(**extra: Any) -> Dict[str, Any]:
        final_support = len(distribution)
        return {
            "prepared_regional_options": preparation_diagnostics,
            "regional_option_preparation_reused": preparation_reused,
            "number_of_regions": len(ordered),
            "regional_support_sizes": regional_support_sizes,
            "raw_cartesian_product_size": raw_cartesian_product_size,
            "actual_raw_expansions": expansions,
            "raw_expansions_after_each_region": tuple(expansions_after_each_region),
            "unique_states_after_each_region": tuple(unique_counts),
            "duplicates_merged_after_each_region": tuple(duplicates_merged_after_each_region),
            "final_unique_support_size": final_support,
            "compression_ratio": (
                float(final_support) / float(raw_cartesian_product_size)
                if raw_cartesian_product_size > 0
                else None
            ),
            "regional_distribution_preparation_seconds": preparation_seconds,
            "outcome_application_seconds": outcome_application_seconds,
            "hashing_merging_seconds": hashing_merging_seconds,
            "normalization_seconds": normalization_seconds,
            "duplicate_states_merged_after_each_region": True,
            "low_probability_pruning": False,
            "regional_independence_model": True,
            **extra,
        }

    for region_index, option in enumerate(ordered):
        next_distribution: Dict[Any, float] = {}
        region_expansions = 0
        for partial_signature, partial_probability in sorted(distribution.items(), key=lambda item: repr(item[0])):
            partial_map = {int(node): (str(owner), int(troops)) for node, owner, troops in partial_signature}
            for outcome_index, (_outcome_signature, outcome_probability) in enumerate(option.normalized_distribution):
                if max_runtime_seconds is not None and time.perf_counter() - started >= float(max_runtime_seconds):
                    return _partial_composition_result(
                        status="runtime_limit",
                        distribution=next_distribution,
                        expansions=expansions,
                        unique_counts=unique_counts,
                        started=started,
                        limit="max_runtime_seconds",
                        diagnostics=complexity_diagnostics(
                            stopped_region_index=region_index,
                            partial_distribution_not_normalized=True,
                            partial_current_region_unique_states=len(next_distribution),
                        ),
                    )
                if max_cartesian_expansions is not None and expansions >= int(max_cartesian_expansions):
                    return _partial_composition_result(
                        status="cartesian_expansion_limit",
                        distribution=next_distribution,
                        expansions=expansions,
                        unique_counts=unique_counts,
                        started=started,
                        limit="max_cartesian_expansions",
                        diagnostics=complexity_diagnostics(
                            stopped_region_index=region_index,
                            partial_distribution_not_normalized=True,
                            partial_current_region_unique_states=len(next_distribution),
                        ),
                    )
                expansions += 1
                region_expansions += 1
                application_started = time.perf_counter()
                successor = dict(partial_map)
                owners = option.owners_by_outcome[outcome_index]
                troops = option.troops_by_outcome[outcome_index]
                for local_index, global_index in option.mapping:
                    if int(global_index) not in successor:
                        continue
                    successor[int(global_index)] = (
                        "A" if int(owners[int(local_index)]) == 1 else "D",
                        int(troops[int(local_index)]),
                    )
                successor_signature = tuple(
                    (node, successor[node][0], successor[node][1]) for node in signature_nodes
                )
                outcome_application_seconds += float(time.perf_counter() - application_started)
                merge_started = time.perf_counter()
                if (
                    successor_signature not in next_distribution
                    and max_unique_states is not None
                    and len(next_distribution) >= int(max_unique_states)
                ):
                    return _partial_composition_result(
                        status="unique_state_limit",
                        distribution=next_distribution,
                        expansions=expansions,
                        unique_counts=unique_counts,
                        started=started,
                        limit="max_unique_states",
                        diagnostics=complexity_diagnostics(
                            stopped_region_index=region_index,
                            partial_distribution_not_normalized=True,
                            partial_current_region_unique_states=len(next_distribution),
                        ),
                    )
                next_distribution[successor_signature] = next_distribution.get(successor_signature, 0.0) + float(partial_probability) * float(outcome_probability)
                hashing_merging_seconds += float(time.perf_counter() - merge_started)
        distribution = next_distribution
        unique_counts.append(len(distribution))
        expansions_after_each_region.append(expansions)
        duplicates_merged_after_each_region.append(max(0, region_expansions - len(distribution)))
    mass = float(sum(distribution.values()))
    if not math.isfinite(mass) or mass <= 0.0 or abs(mass - 1.0) > 1e-8:
        return _partial_composition_result(
            status="probability_error",
            distribution=distribution,
            expansions=expansions,
            unique_counts=unique_counts,
            started=started,
            limit=None,
            diagnostics=complexity_diagnostics(probability_mass=mass, renormalized=False),
        )
    normalization_started = time.perf_counter()
    normalized = {state: probability / mass for state, probability in distribution.items()}
    normalization_seconds = float(time.perf_counter() - normalization_started)
    return ExactComposedDistributionResult(
        status="exact_complete",
        distribution=normalized,
        raw_cartesian_expansions=expansions,
        unique_states_after_each_region=tuple(unique_counts),
        final_unique_state_count=len(normalized),
        probability_mass=float(sum(normalized.values())),
        runtime_seconds=float(time.perf_counter() - started),
        limit_reached=None,
        diagnostics=complexity_diagnostics(regions_composed=len(ordered)),
    )


@dataclass(frozen=True)
class _ExactProblem:
    nodes: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    attacker_nodes: Tuple[int, ...]
    defender_nodes: Tuple[int, ...]
    local_to_global: Tuple[int, ...]
    global_to_local: Mapping[int, int]
    attacker_troops: Tuple[int, ...]
    defender_troops: Tuple[int, ...]
    initial_signature: Any


def _build_exact_problem(graph: Any, initial_state: Any) -> _ExactProblem:
    nodes, edges = _graph_nodes_edges(graph)
    state = {node: (owner, troops) for node, owner, troops in canonical_risk_state(initial_state)}
    if set(nodes) - set(state):
        raise ValueError("Initial state and graph node sets do not align")
    attackers = tuple(node for node in nodes if state[node][0] == "A")
    defenders = tuple(node for node in nodes if state[node][0] == "D")
    unknown = tuple(node for node in nodes if state[node][0] not in {"A", "D"})
    if unknown:
        raise ValueError(f"Unknown owner semantics on nodes {unknown}")
    if not attackers or not defenders:
        raise ValueError("Exact battle reference requires both attacker and defender nodes")
    if any(state[node][1] < 1 for node in nodes):
        raise ValueError("Exact battle reference requires at least one troop per node")
    local_to_global = attackers + defenders
    global_to_local = {node: index for index, node in enumerate(local_to_global)}
    return _ExactProblem(
        nodes=nodes,
        edges=edges,
        attacker_nodes=attackers,
        defender_nodes=defenders,
        local_to_global=local_to_global,
        global_to_local=global_to_local,
        attacker_troops=tuple(state[node][1] for node in attackers),
        defender_troops=tuple(state[node][1] for node in defenders),
        initial_signature=canonical_validation_state_signature(initial_state, nodes),
    )


def _solver_for_problem(
    problem: _ExactProblem,
    *,
    utility_mode: str,
    max_runtime_seconds: Optional[float],
    max_states: Optional[int],
    max_cache_entries: Optional[int] = None,
    max_memory_estimate_bytes: Optional[int] = None,
) -> CompactExactTopologySolver:
    local_edges = tuple(
        sorted(
            (
                min(problem.global_to_local[left], problem.global_to_local[right]),
                max(problem.global_to_local[left], problem.global_to_local[right]),
            )
            for left, right in problem.edges
        )
    )
    max_attacker = max(problem.attacker_troops)
    max_defender = max(problem.defender_troops)
    combat_df = combat_df_for_caps(
        num_attacker_nodes=len(problem.attacker_nodes),
        num_defender_nodes=len(problem.defender_nodes),
        max_attacker_troops=max_attacker,
        max_defender_troops=max_defender,
    )
    return CompactExactTopologySolver(
        edges=local_edges,
        num_attacker_nodes=len(problem.attacker_nodes),
        num_defender_nodes=len(problem.defender_nodes),
        combat_df=combat_df,
        utility_mode=utility_mode,
        max_total_troops=sum(problem.attacker_troops) + sum(problem.defender_troops),
        cache_distributions=True,
        max_states=max_states,
        max_runtime_seconds=max_runtime_seconds,
        max_cache_entries=max_cache_entries,
        max_memory_estimate_bytes=max_memory_estimate_bytes,
    )


def _compact_distribution_to_signature(
    solver: CompactExactTopologySolver,
    problem: _ExactProblem,
    distribution: Mapping[int, float],
) -> Dict[Any, float]:
    answer: Dict[Any, float] = {}
    for compact_state, probability in solver.normalize_distribution(dict(distribution)).items():
        owner_mask = solver.owner_mask(compact_state)
        troops = solver.troops_tuple(compact_state)
        rows = []
        for local_index, global_node in enumerate(problem.local_to_global):
            owner = "A" if ((owner_mask >> local_index) & 1) else "D"
            rows.append((int(global_node), owner, int(troops[local_index])))
        signature = tuple(sorted(rows))
        answer[signature] = answer.get(signature, 0.0) + float(probability)
    return normalize_probability_distribution(answer)


def _global_action(problem: _ExactProblem, action: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if action is None:
        return None
    return (problem.local_to_global[int(action[0])], problem.local_to_global[int(action[1])])


def solve_full_graph_exact_reference(
    *,
    graph: Any,
    initial_state: Any,
    utility_mode: str = "local",
    ranking_variable: str = "battle_expected_attacker_territory_count",
    include_all_optimal_policies: bool = True,
    max_runtime_seconds: Optional[float] = None,
    max_states: Optional[int] = None,
    max_cache_entries: Optional[int] = None,
    max_memory_estimate_bytes: Optional[int] = None,
    max_policy_options: Optional[int] = None,
) -> ExactFullGraphReferenceResult:
    """Solve one labelled full graph with the compact exact finite solver."""
    started = time.perf_counter()
    graph_signature: Any = None
    initial_signature: Any = None
    solver: Optional[CompactExactTopologySolver] = None
    try:
        problem = _build_exact_problem(graph, initial_state)
        graph_signature = (problem.nodes, problem.edges)
        initial_signature = problem.initial_signature
        solver = _solver_for_problem(
            problem,
            utility_mode=str(utility_mode),
            max_runtime_seconds=max_runtime_seconds,
            max_states=max_states,
            max_cache_entries=max_cache_entries,
            max_memory_estimate_bytes=max_memory_estimate_bytes,
        )
        result = solver.evaluate_start(problem.attacker_troops, problem.defender_troops)
        if include_all_optimal_policies:
            options = solver.root_policy_options(
                problem.attacker_troops,
                problem.defender_troops,
                max_policy_options=None,
            )
        else:
            options = []
        if max_policy_options is not None and len(options) > int(max_policy_options):
            raise ExactSolverLimitReached(
                "policy_option_limit",
                f"exact reference found {len(options)} root policy options; "
                f"limit={int(max_policy_options)}",
            )
        if not options:
            options = [
                type("CanonicalOption", (), {
                    "root_action": result.root_action,
                    "value": result.value,
                    "absorbing_dist": result.absorbing_dist,
                })()
            ]
        policies = tuple(
            (
                "exact_root_policy_v1",
                _global_action(problem, option.root_action),
                "canonical_optimal_continuation",
            )
            for option in options
        )
        distributions = tuple(
            _compact_distribution_to_signature(solver, problem, option.absorbing_dist)
            for option in options
        )
        canonical_index = next(
            (
                index
                for index, option in enumerate(options)
                if option.root_action == result.root_action
            ),
            0,
        )
        stats = solver.stats.as_dict()
        return ExactFullGraphReferenceResult(
            status="exact_complete",
            graph_signature=graph_signature,
            initial_state_signature=initial_signature,
            optimal_value=tuple(float(value) for value in result.value),
            canonical_optimal_policy=policies[canonical_index],
            optimal_policy_set=policies,
            canonical_optimal_distribution=distributions[canonical_index],
            optimal_policy_distributions=distributions,
            states_evaluated=int(stats["value_evals"]),
            cache_hits=int(stats["value_cache_hits"] + stats["dist_cache_hits"]),
            actions_evaluated=int(stats["action_value_evals"]),
            runtime_seconds=float(time.perf_counter() - started),
            diagnostics={
                "ranking_variable": str(ranking_variable),
                "utility_mode": str(utility_mode),
                "solver_stats": stats,
                "value_cache_size": len(solver._value_cache),
                "distribution_cache_size": len(solver._dist_cache),
                "total_cache_entries": solver.cache_entry_count(),
                "estimated_cache_bytes": solver.estimated_cache_bytes(),
                "memory_estimate_method": "deterministic_python_container_coefficient_estimate_v1",
                "optimal_policy_scope": "all_tied_root_actions_with_canonical_optimal_continuation",
                "complete_downstream_tie_policy_enumeration": False,
                "movement_semantics": "existing compact exact finite solver",
                "policy_trace": {
                    "canonical_root_action": policies[canonical_index][1],
                    "conditional_trace_status": "unavailable",
                    "reason": "the compact reference exports absorbing distributions, not a serialized contingent policy DAG",
                },
            },
        )
    except ExactSolverLimitReached as exc:
        status = exc.status
        error = str(exc)
    except Exception as exc:
        status = "invalid_state" if isinstance(exc, (TypeError, ValueError)) else "solver_error"
        error = f"{type(exc).__name__}: {exc}"
    stats = solver.stats.as_dict() if solver is not None else {}
    return ExactFullGraphReferenceResult(
        status=status,
        graph_signature=graph_signature,
        initial_state_signature=initial_signature,
        optimal_value=None,
        canonical_optimal_policy=None,
        optimal_policy_set=(),
        canonical_optimal_distribution={},
        optimal_policy_distributions=(),
        states_evaluated=int(stats.get("value_evals", 0)),
        cache_hits=int(stats.get("value_cache_hits", 0)) + int(stats.get("dist_cache_hits", 0)),
        actions_evaluated=int(stats.get("action_value_evals", 0)),
        runtime_seconds=float(time.perf_counter() - started),
        diagnostics={
            "error": error,
            "solver_stats": stats,
            "partial_result_not_normalized": True,
            "total_cache_entries": solver.cache_entry_count() if solver is not None else 0,
            "estimated_cache_bytes": solver.estimated_cache_bytes() if solver is not None else 0,
            "memory_estimate_method": "deterministic_python_container_coefficient_estimate_v1",
        },
    )


def _terminal_value_from_signature(
    signature: Any,
    *,
    initial_state: Any,
    utility_mode: str,
) -> Tuple[float, ...]:
    initial = {node: (owner, troops) for node, owner, troops in canonical_risk_state(initial_state)}
    current = {node: (owner, troops) for node, owner, troops in canonical_risk_state(signature)}
    if utility_mode == "legacy":
        conquered = all(owner == "A" for owner, _troops in current.values())
        return (float(conquered), float(sum(troops for node, (owner, troops) in current.items() if conquered and initial[node][0] == "D" and owner == "A")))
    new_territories = sum(1 for node, (owner, _troops) in current.items() if initial[node][0] == "D" and owner == "A")
    attacker_troops = sum(troops for owner, troops in current.values() if owner == "A")
    conquest = float(all(owner == "A" for owner, _troops in current.values()))
    return (float(new_territories), float(attacker_troops), conquest)


def distribution_implied_value(
    distribution: Mapping[Any, float],
    *,
    initial_state: Any,
    utility_mode: str = "local",
) -> Tuple[float, ...]:
    normalized = normalize_probability_distribution(distribution)
    width = len(_terminal_value_from_signature(next(iter(normalized)), initial_state=initial_state, utility_mode=utility_mode))
    answer = [0.0] * width
    for state, probability in normalized.items():
        value = _terminal_value_from_signature(state, initial_state=initial_state, utility_mode=utility_mode)
        for index in range(width):
            answer[index] += probability * value[index]
    return tuple(float(value) for value in answer)


def evaluate_policy_under_exact_full_graph_model(
    *,
    graph: Any,
    initial_state: Any,
    policy: Any,
    utility_mode: str = "local",
    max_runtime_seconds: Optional[float] = None,
    max_states: Optional[int] = None,
    max_cache_entries: Optional[int] = None,
    max_memory_estimate_bytes: Optional[int] = None,
) -> ExactPolicyEvaluationResult:
    """Evaluate explicit root+canonical-continuation policies when representable."""
    started = time.perf_counter()
    if isinstance(policy, Mapping) and "terminal_distribution" in policy:
        distribution = normalize_probability_distribution(policy["terminal_distribution"])
        return ExactPolicyEvaluationResult(
            status="distribution_value_only",
            value=distribution_implied_value(distribution, initial_state=initial_state, utility_mode=utility_mode),
            distribution=distribution,
            runtime_seconds=float(time.perf_counter() - started),
            diagnostics={
                "explicit_full_graph_policy_evaluated": False,
                "reason": "terminal distributions do not encode a full state-to-action policy",
            },
        )
    if isinstance(policy, bgr.PartitionPolicyCandidate):
        return ExactPolicyEvaluationResult(
            status="unsupported_policy_representation",
            value=None,
            distribution={},
            runtime_seconds=float(time.perf_counter() - started),
            diagnostics={
                "explicit_full_graph_policy_evaluated": False,
                "reason": "regional V2 policy payloads omit a complete full-graph state-to-action trace",
            },
        )
    if not (
        isinstance(policy, (tuple, list))
        and len(policy) >= 3
        and policy[0] == "exact_root_policy_v1"
        and policy[2] == "canonical_optimal_continuation"
    ):
        return ExactPolicyEvaluationResult(
            status="unsupported_policy_representation",
            value=None,
            distribution={},
            runtime_seconds=float(time.perf_counter() - started),
            diagnostics={"explicit_full_graph_policy_evaluated": False},
        )
    try:
        problem = _build_exact_problem(graph, initial_state)
        solver = _solver_for_problem(
            problem,
            utility_mode=utility_mode,
            max_runtime_seconds=max_runtime_seconds,
            max_states=max_states,
            max_cache_entries=max_cache_entries,
            max_memory_estimate_bytes=max_memory_estimate_bytes,
        )
        compact_state = solver.initial_state(problem.attacker_troops, problem.defender_troops)
        global_action = policy[1]
        if global_action is None:
            value = solver.terminal_value(compact_state)
            compact_distribution = {compact_state: 1.0}
        else:
            action = (problem.global_to_local[int(global_action[0])], problem.global_to_local[int(global_action[1])])
            if action not in solver.possible_actions(compact_state):
                raise ValueError(f"Policy root action {global_action!r} is not legal")
            value = solver.evaluate_action_value(compact_state, action)
            compact_distribution = solver.action_distribution(compact_state, action)
        distribution = _compact_distribution_to_signature(solver, problem, compact_distribution)
        return ExactPolicyEvaluationResult(
            status="exact_complete",
            value=tuple(float(item) for item in value),
            distribution=distribution,
            runtime_seconds=float(time.perf_counter() - started),
            diagnostics={"explicit_full_graph_policy_evaluated": True, "solver_stats": solver.stats.as_dict()},
        )
    except ExactSolverLimitReached as exc:
        status, error = exc.status, str(exc)
    except Exception as exc:
        status, error = "policy_evaluation_error", f"{type(exc).__name__}: {exc}"
    return ExactPolicyEvaluationResult(
        status=status,
        value=None,
        distribution={},
        runtime_seconds=float(time.perf_counter() - started),
        diagnostics={"error": error, "explicit_full_graph_policy_evaluated": False},
    )


def compare_policy_values(
    optimal_value: Optional[Sequence[float]],
    evaluated_value: Optional[Sequence[float]],
    *,
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    if optimal_value is None or evaluated_value is None:
        return {
            "status": "unavailable",
            "primary_component_regret": None,
            "secondary_component_regret": None,
            "tertiary_component_regret": None,
            "lexicographically_optimal": False,
            "value_tied_with_optimum": False,
        }
    width = min(len(optimal_value), len(evaluated_value))
    regrets = tuple(float(optimal_value[index]) - float(evaluated_value[index]) for index in range(width))
    tied = all(abs(value) <= float(tolerance) for value in regrets)
    lex_optimal = tied
    if not tied:
        for value in regrets:
            if abs(value) <= float(tolerance):
                continue
            lex_optimal = value < -float(tolerance)
            break
    return {
        "status": "complete",
        "component_regrets": regrets,
        "primary_component_regret": regrets[0] if width > 0 else None,
        "secondary_component_regret": regrets[1] if width > 1 else None,
        "tertiary_component_regret": regrets[2] if width > 2 else None,
        "lexicographically_optimal": bool(lex_optimal),
        "value_tied_with_optimum": bool(tied),
        "scalarization_used": False,
    }


def compare_to_exact_optimal_policy_set(
    approximate_distribution: Mapping[Any, float],
    reference: ExactFullGraphReferenceResult,
) -> Dict[str, Any]:
    if not reference.optimal_policy_distributions:
        return {
            "matches_any_exact_optimal_policy": False,
            "minimum_tv_to_exact_optimal_set": None,
            "maximum_tv_to_exact_optimal_set": None,
            "canonical_exact_policy_tv": None,
        }
    tvs = tuple(
        total_variation_distance(approximate_distribution, distribution)
        for distribution in reference.optimal_policy_distributions
    )
    return {
        "matches_any_exact_optimal_policy": bool(min(tvs) <= 1e-10),
        "minimum_tv_to_exact_optimal_set": min(tvs),
        "maximum_tv_to_exact_optimal_set": max(tvs),
        "canonical_exact_policy_tv": total_variation_distance(
            approximate_distribution, reference.canonical_optimal_distribution
        ),
        "exact_optimal_distribution_tv_range": (min(tvs), max(tvs)),
        "identity_interpretation": "distribution equality; explicit regional policy lifting is unavailable",
    }


@contextmanager
def _temporary_board_state(initial_state: Any, graph_nodes: Sequence[int]):
    state = {node: (owner, troops) for node, owner, troops in canonical_risk_state(initial_state)}
    missing_board_nodes = set(int(node) for node in graph_nodes) - set(Board.node_to_territory_dict)
    if missing_board_nodes:
        raise ValueError(f"Graph nodes are not Risk board indices: {sorted(missing_board_nodes)}")
    snapshot = {
        int(index): (territory._owner, territory._troops)
        for index, territory in Board.node_to_territory_dict.items()
    }
    players = (Players.Player("A"), Players.Player("D"))
    attacker, defender = players
    try:
        for index, territory in Board.node_to_territory_dict.items():
            if int(index) in state:
                owner, troops = state[int(index)]
                territory._owner = attacker if owner == "A" else defender
                territory._troops = int(troops)
            else:
                territory._owner = defender
                territory._troops = 1
        yield players, agop.build_global_state_for_board(players)
    finally:
        for index, (owner, troops) in snapshot.items():
            territory = Board.node_to_territory_dict[int(index)]
            territory._owner = owner
            territory._troops = troops


def evaluate_regional_compounding_approximation(
    *,
    graph: Any,
    initial_state: Any,
    library_dir: Path | str,
    partition_config: Optional[Mapping[str, Any]] = None,
    ranking_config: Optional[Mapping[str, Any]] = None,
    exact_compose_selected_candidate: bool = True,
    composition_limits: Optional[Mapping[str, Any]] = None,
) -> RegionalApproximationResult:
    """Run corrected production candidate preparation and selection."""
    started = time.perf_counter()
    partition = dict(partition_config or {})
    ranking = dict(ranking_config or {})
    forbidden_non_null = (
        "utility_abs_tolerance",
        "utility_rel_tolerance",
        "max_candidates_per_partition",
        "max_policy_combos_per_partition",
        "max_total_partition_policy_candidates",
    )
    for name in forbidden_non_null:
        if partition.get(name) is not None:
            raise ValueError(f"Validation requires uncapped corrected semantics; {name} must be None")
    mode = partition.get("partition_candidate_selection_mode", "maximal_per_partition_utility")
    if mode != "maximal_per_partition_utility":
        raise ValueError("Validation requires partition_candidate_selection_mode='maximal_per_partition_utility'")
    nodes, edges = _graph_nodes_edges(graph)
    graph_object = _as_validation_graph(graph)
    try:
        with _temporary_board_state(initial_state, nodes) as (players, board_global_state):
            prepared = bgr.prepare_two_stage_partition_policy_candidates(
                players=players,
                battle_graph=graph_object,
                combat_libraries_base=Path(library_dir),
                max_partitions=int(partition.get("max_partitions", 10000)),
                ranking_variable=str(ranking.get("ranking_variable", "battle_expected_attacker_territory_count")),
                first_stage_value_tolerances=None,
                max_policy_combos_per_partition=None,
                max_total_partition_policy_candidates=None,
                partition_candidate_selection_mode="maximal_per_partition_utility",
                utility_abs_tolerance=None,
                utility_rel_tolerance=None,
                max_candidates_per_partition=None,
                run_expensive_cover_diagnostics=bool(partition.get("run_expensive_cover_diagnostics", False)),
            )
            if not prepared.retained_candidates:
                return RegionalApproximationResult(
                    status="no_valid_candidates",
                    supported_partitions=tuple(prepared.partitions_full),
                    maximal_partitions=tuple(prepared.working_partitions),
                    retained_candidates=(),
                    selected_candidate=None,
                    selected_partition_signature=None,
                    selected_policy_option_indices=(),
                    selected_candidate_local_utility=None,
                    selected_candidate_second_stage_value=None,
                    exact_compounded_distribution=None,
                    composition_status="not_run",
                    composition_diagnostics={},
                    runtime_seconds=float(time.perf_counter() - started),
                    diagnostics=dict(prepared.diagnostics),
                )
            samples = max(1, int(ranking.get("candidate_selection_mc_samples", 5)))
            selection = bgr.evaluate_candidates_at_nested_checkpoints(
                prepared_candidates=prepared,
                players=players,
                battle_graph=graph_object,
                combat_libraries_base=Path(library_dir),
                ranking_variable=str(ranking.get("ranking_variable", "battle_expected_attacker_territory_count")),
                checkpoints=(samples,),
                base_seed=int(ranking.get("candidate_selection_seed", 42)),
                selection_mode="fixed",
                fixed_sample_count=samples,
                global_state_utility_evaluator=ranking.get("global_state_utility_evaluator"),
                profile_second_stage=bool(ranking.get("profile_second_stage", False)),
            )
            selected = selection.selected_candidate
            if selected is None:
                raise RuntimeError(f"Candidate selection failed: {selection.stopping_reason}")
            if exact_compose_selected_candidate:
                limits = dict(composition_limits or {})
                composition = compose_selected_candidate_distribution_exact(
                    selected_candidate=selected,
                    initial_global_state=board_global_state,
                    node_indices=nodes,
                    max_unique_states=limits.get("max_unique_states"),
                    max_cartesian_expansions=limits.get("max_cartesian_expansions"),
                    max_runtime_seconds=limits.get("max_runtime_seconds"),
                )
            else:
                composition = ExactComposedDistributionResult(
                    status="not_run",
                    distribution={},
                    raw_cartesian_expansions=0,
                    unique_states_after_each_region=(),
                    final_unique_state_count=0,
                    probability_mass=0.0,
                    runtime_seconds=0.0,
                    limit_reached=None,
                    diagnostics={},
                )
            checkpoint = selection.checkpoints[-1] if selection.checkpoints else None
            return RegionalApproximationResult(
                status="approximation_complete",
                supported_partitions=tuple(prepared.partitions_full),
                maximal_partitions=tuple(prepared.working_partitions),
                retained_candidates=tuple(prepared.retained_candidates),
                selected_candidate=selected,
                selected_partition_signature=bgr.canonical_partition_signature(selected.partition_regions),
                selected_policy_option_indices=tuple(
                    int(ref.option_index)
                    for ref in sorted(selected.region_policy_options, key=lambda item: tuple(sorted(item.region_nodes)))
                ),
                selected_candidate_local_utility=tuple(selected.first_stage_utility),
                selected_candidate_second_stage_value=(
                    tuple(checkpoint.best_score_mean) if checkpoint is not None else None
                ),
                exact_compounded_distribution=(
                    dict(composition.distribution) if composition.status == "exact_complete" else None
                ),
                composition_status=composition.status,
                composition_diagnostics={
                    **dict(composition.diagnostics),
                    "raw_cartesian_expansions": composition.raw_cartesian_expansions,
                    "unique_states_after_each_region": composition.unique_states_after_each_region,
                    "final_unique_state_count": composition.final_unique_state_count,
                    "probability_mass": composition.probability_mass,
                    "runtime_seconds": composition.runtime_seconds,
                    "limit_reached": composition.limit_reached,
                },
                runtime_seconds=float(time.perf_counter() - started),
                diagnostics={
                    **dict(prepared.diagnostics),
                    "candidate_selection": selection,
                    "graph_edges": edges,
                    "production_semantics": {
                        "partition_candidate_selection_mode": "maximal_per_partition_utility",
                        "utility_tolerance": None,
                        "candidate_cap": None,
                        "policy_combo_cap": None,
                    },
                },
            )
    except Exception as exc:
        return RegionalApproximationResult(
            status="approximation_error",
            supported_partitions=(),
            maximal_partitions=(),
            retained_candidates=(),
            selected_candidate=None,
            selected_partition_signature=None,
            selected_policy_option_indices=(),
            selected_candidate_local_utility=None,
            selected_candidate_second_stage_value=None,
            exact_compounded_distribution=None,
            composition_status="not_run",
            composition_diagnostics={},
            runtime_seconds=float(time.perf_counter() - started),
            diagnostics={"error": f"{type(exc).__name__}: {exc}"},
        )


def _adjacency(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> Dict[int, set[int]]:
    answer = {int(node): set() for node in nodes}
    for left, right in edges:
        answer[int(left)].add(int(right))
        answer[int(right)].add(int(left))
    return answer


def _articulation_nodes(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> Tuple[int, ...]:
    adjacency = _adjacency(nodes, edges)
    discovery: Dict[int, int] = {}
    low: Dict[int, int] = {}
    parent: Dict[int, Optional[int]] = {}
    articulations = set()
    timer = 0

    def visit(node: int) -> None:
        nonlocal timer
        timer += 1
        discovery[node] = low[node] = timer
        children = 0
        for neighbor in sorted(adjacency[node]):
            if neighbor not in discovery:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if parent.get(node) is None and children > 1:
                    articulations.add(node)
                if parent.get(node) is not None and low[neighbor] >= discovery[node]:
                    articulations.add(node)
            elif neighbor != parent.get(node):
                low[node] = min(low[node], discovery[neighbor])

    for node in nodes:
        if node not in discovery:
            parent[node] = None
            visit(node)
    return tuple(sorted(articulations))


def _edge_in_cycle(edge: Tuple[int, int], adjacency: Mapping[int, set[int]]) -> bool:
    left, right = edge
    visited = {left}
    stack = [left]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if {node, neighbor} == {left, right}:
                continue
            if neighbor == right:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return False


def describe_cross_region_interaction_structure(
    *,
    graph: Any,
    initial_state: Any,
    partition_signature: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    nodes, edges = _graph_nodes_edges(graph)
    state = {node: (owner, troops) for node, owner, troops in canonical_risk_state(initial_state)}
    regions = tuple(tuple(sorted(int(node) for node in region)) for region in partition_signature)
    region_for_node = {
        node: index for index, region in enumerate(regions) for node in region
    }
    boundary_edges = tuple(
        edge
        for edge in edges
        if region_for_node.get(edge[0]) is not None
        and region_for_node.get(edge[1]) is not None
        and region_for_node[edge[0]] != region_for_node[edge[1]]
    )
    boundary_nodes = tuple(sorted({node for edge in boundary_edges for node in edge}))
    articulations = _articulation_nodes(nodes, edges)
    adjacency = _adjacency(nodes, edges)
    shared_sources = []
    for node in nodes:
        if state[node][0] != "A":
            continue
        defender_regions = {
            region_for_node.get(neighbor)
            for neighbor in adjacency[node]
            if state[neighbor][0] == "D" and region_for_node.get(neighbor) is not None
        }
        if len(defender_regions) > 1:
            shared_sources.append(node)
    cross_region_defender_edges = tuple(
        edge
        for edge in boundary_edges
        if state[edge[0]][0] == "D" and state[edge[1]][0] == "D"
    )
    active_cross_edges = tuple(
        edge for edge in boundary_edges if state[edge[0]][0] != state[edge[1]][0]
    )
    attacker_boundary_edges = tuple(
        edge for edge in boundary_edges if state[edge[0]][0] == state[edge[1]][0] == "A"
    )
    defender_boundary_edges = tuple(
        edge for edge in boundary_edges if state[edge[0]][0] == state[edge[1]][0] == "D"
    )
    return {
        "number_of_articulation_nodes": len(articulations),
        "articulation_nodes": articulations,
        "attacker_articulation_nodes": tuple(node for node in articulations if state[node][0] == "A"),
        "defender_articulation_nodes": tuple(node for node in articulations if state[node][0] == "D"),
        "number_of_partition_boundary_edges": len(boundary_edges),
        "partition_boundary_edges": boundary_edges,
        "partition_boundary_nodes": boundary_nodes,
        "active_attacker_defender_boundary_edges": active_cross_edges,
        "shared_potential_troop_source_nodes": tuple(shared_sources),
        "shared_troop_source_present": bool(shared_sources),
        "one_region_conquest_opens_adjacency_to_another": bool(cross_region_defender_edges),
        "sequence_opening_present": bool(cross_region_defender_edges),
        "regions_connected_through_attacker_owned_paths": bool(attacker_boundary_edges),
        "regions_connected_through_defender_owned_paths": bool(defender_boundary_edges),
        "cycles_crossing_regional_boundaries": sum(_edge_in_cycle(edge, adjacency) for edge in boundary_edges),
        "exact_optimal_actions_alternate_between_regions": None,
        "policy_trace_observability": "unavailable from compact terminal-distribution payloads",
    }


def validate_state_signature_alignment(
    exact_distribution: Mapping[Any, float],
    approximate_distribution: Mapping[Any, float],
    *,
    expected_nodes: Sequence[int],
) -> Dict[str, Any]:
    expected = tuple(sorted(int(node) for node in expected_nodes))
    errors = []
    owner_values = set()
    for label, distribution in (("exact", exact_distribution), ("approximate", approximate_distribution)):
        for state in distribution:
            canonical = canonical_risk_state(state)
            nodes = tuple(node for node, _owner, _troops in canonical)
            if nodes != expected:
                errors.append(f"{label} state node order {nodes!r} != {expected!r}")
            for _node, owner, troops in canonical:
                owner_values.add(owner)
                if int(troops) < 0:
                    errors.append(f"{label} state contains negative troops")
    if owner_values - {"A", "D"}:
        errors.append(f"unsupported owners: {sorted(owner_values - {'A', 'D'})}")
    return {
        "valid": not errors,
        "errors": tuple(errors),
        "node_order": expected,
        "owner_semantics": tuple(sorted(owner_values)),
        "troop_semantics": "nonnegative integer troops on labelled nodes",
        "terminal_state_semantics_aligned": not errors,
    }


def _betweenness_centrality(nodes: Sequence[int], edges: Sequence[Tuple[int, int]]) -> Dict[int, float]:
    adjacency = _adjacency(nodes, edges)
    centrality = {node: 0.0 for node in nodes}
    for source in nodes:
        stack: List[int] = []
        predecessors = {node: [] for node in nodes}
        paths = {node: 0.0 for node in nodes}
        paths[source] = 1.0
        distance = {node: -1 for node in nodes}
        distance[source] = 0
        queue = [source]
        while queue:
            node = queue.pop(0)
            stack.append(node)
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[node] + 1
                if distance[neighbor] == distance[node] + 1:
                    paths[neighbor] += paths[node]
                    predecessors[neighbor].append(node)
        dependency = {node: 0.0 for node in nodes}
        while stack:
            node = stack.pop()
            if paths[node] > 0.0:
                coefficient = (1.0 + dependency[node]) / paths[node]
                for predecessor in predecessors[node]:
                    dependency[predecessor] += paths[predecessor] * coefficient
            if node != source:
                centrality[node] += dependency[node]
    return {node: value / 2.0 for node, value in centrality.items()}


@dataclass(frozen=True)
class InterpretationThresholds:
    version: str = "regional_interpretation_thresholds_v2"
    near_exact_tv: float = 0.05
    high_joint_tv: float = 0.20
    low_primary_regret: float = 0.05
    high_primary_regret: float = 0.25
    high_ownership_error: float = 0.10
    high_troop_error: float = 0.50
    low_expectation_error: float = 0.05
    high_tail_error: float = 0.15
    high_covariance_error: float = 0.05


def interpretation_profiles(
    *,
    regret: Mapping[str, Any],
    distribution_metrics: Mapping[str, Any],
    node_metrics: Mapping[str, Any],
    strategic_summary_metrics: Mapping[str, Any],
    dependence_metrics: Mapping[str, Any],
    graph_descriptors: Mapping[str, Any],
    thresholds: InterpretationThresholds = InterpretationThresholds(),
) -> Tuple[str, ...]:
    labels = []
    primary_regret = regret.get("primary_component_regret")
    tv = distribution_metrics.get("total_variation")
    ownership_error = node_metrics.get("maximum_ownership_probability_error", 0.0)
    troop_error = node_metrics.get("maximum_expected_troop_error", 0.0)
    max_tail = max(
        (
            float(row.get("maximum_tail_probability_difference", 0.0))
            for row in strategic_summary_metrics.values()
            if isinstance(row, Mapping)
        ),
        default=0.0,
    )
    max_expectation_error = max(
        (
            float(row.get("absolute_expectation_error", 0.0))
            for row in strategic_summary_metrics.values()
            if isinstance(row, Mapping)
        ),
        default=0.0,
    )
    covariance_error = dependence_metrics.get("maximum_absolute_covariance_error", 0.0)
    if primary_regret is not None and float(primary_regret) <= thresholds.low_primary_regret and tv is not None and float(tv) >= thresholds.high_joint_tv:
        labels.append("low_regret_high_distribution_difference")
    if ownership_error <= thresholds.high_ownership_error and tv is not None and float(tv) >= thresholds.high_joint_tv:
        labels.append("accurate_marginals_inaccurate_joint_distribution")
    if ownership_error <= thresholds.high_ownership_error and troop_error >= thresholds.high_troop_error:
        labels.append("ownership_accurate_troops_inaccurate")
    if (
        max_expectation_error <= thresholds.low_expectation_error
        and max_tail >= thresholds.high_tail_error
    ):
        labels.append("accurate_means_inaccurate_tails")
    largest = node_metrics.get("largest_error_node")
    per_node = node_metrics.get("per_node", {})
    if (
        largest in per_node
        and per_node[largest].get("is_partition_boundary")
        and (
            float(per_node[largest].get("ownership_probability_error", 0.0))
            >= thresholds.high_ownership_error
            or float(per_node[largest].get("expected_troop_error", 0.0))
            >= thresholds.high_troop_error
        )
    ):
        labels.append("boundary_node_error")
    if float(covariance_error or 0.0) >= thresholds.high_covariance_error:
        labels.append("cross_region_dependence_error")
    if graph_descriptors.get("sequence_opening_present") and tv is not None and float(tv) >= thresholds.high_joint_tv:
        labels.append("sequence_sensitive_failure")
    if tv is not None and float(tv) <= thresholds.near_exact_tv and (primary_regret is None or float(primary_regret) <= thresholds.low_primary_regret):
        labels.append("near_exact")
    if primary_regret is not None and float(primary_regret) >= thresholds.high_primary_regret:
        labels.append("high_policy_regret")
    return tuple(labels)


def build_regional_compounding_validation_record(
    *,
    benchmark_id: str,
    topology_family: str,
    graph: Any,
    initial_state: Any,
    exact_reference: ExactFullGraphReferenceResult,
    approximation: RegionalApproximationResult,
    state_aware_max_support_size: Optional[int] = 500,
    interpretation_thresholds: InterpretationThresholds = InterpretationThresholds(),
) -> RegionalCompoundingValidationRecord:
    nodes, edges = _graph_nodes_edges(graph)
    partition_signature = approximation.selected_partition_signature or ()
    descriptors = describe_cross_region_interaction_structure(
        graph=graph,
        initial_state=initial_state,
        partition_signature=partition_signature,
    )
    exact_distribution = exact_reference.canonical_optimal_distribution
    approximate_distribution = approximation.exact_compounded_distribution or {}
    if exact_reference.status == "exact_complete" and approximation.composition_status == "exact_complete":
        alignment = validate_state_signature_alignment(
            exact_distribution,
            approximate_distribution,
            expected_nodes=nodes,
        )
    else:
        alignment = {"valid": False, "errors": ("complete distributions unavailable",)}
    if alignment.get("valid"):
        distribution_metrics = compare_distributions(exact_distribution, approximate_distribution)
        distribution_metrics.update(compare_to_exact_optimal_policy_set(approximate_distribution, exact_reference))
        approximate_value = distribution_implied_value(
            approximate_distribution,
            initial_state=initial_state,
            utility_mode=str(exact_reference.diagnostics.get("utility_mode", "local")),
        )
        distribution_value_gap = compare_policy_values(exact_reference.optimal_value, approximate_value)
        regret = compare_policy_values(exact_reference.optimal_value, None)
        regret.update(
            {
                "evaluation_status": "unsupported_policy_representation",
                "explicit_full_graph_policy_lifted": False,
                "reason": "regional V2 payloads do not encode a complete full-graph state-to-action policy",
                "distribution_value_gap": distribution_value_gap,
            }
        )
        state_aware = {
            name: asdict(
                risk_state_wasserstein_distance(
                    exact_distribution,
                    approximate_distribution,
                    state_distance_config=profile,
                    max_support_size=state_aware_max_support_size,
                )
            )
            for name, profile in STATE_DISTANCE_PROFILES.items()
        }
        node_metrics = compare_node_marginals(
            exact_distribution,
            approximate_distribution,
            initial_state=initial_state,
            partition_boundary_nodes=descriptors["partition_boundary_nodes"],
            articulation_nodes=descriptors["articulation_nodes"],
        )
        centrality = _betweenness_centrality(nodes, edges)
        key_nodes = tuple(
            node for node, value in centrality.items() if value == max(centrality.values(), default=0.0)
        )
        event_metrics = compare_strategic_events(
            exact_distribution,
            approximate_distribution,
            initial_state=initial_state,
            edges=edges,
            at_least_k_values=tuple(range(1, min(3, len(nodes)) + 1)),
            articulation_nodes=descriptors["articulation_nodes"],
            key_territories=key_nodes,
        )
        summary_metrics = compare_strategic_summaries(
            exact_distribution,
            approximate_distribution,
            initial_state=initial_state,
            edges=edges,
        )
        if len(partition_signature) >= 2:
            dependence_metrics = compare_cross_region_dependence(
                exact_distribution,
                approximate_distribution,
                initial_state=initial_state,
                regions=partition_signature,
            )
        else:
            dependence_metrics = {
                "status": "insufficient_regions",
                "maximum_absolute_covariance_error": 0.0,
                "maximum_absolute_joint_success_error": 0.0,
            }
        labels = interpretation_profiles(
            regret=regret,
            distribution_metrics=distribution_metrics,
            node_metrics=node_metrics,
            strategic_summary_metrics=summary_metrics,
            dependence_metrics=dependence_metrics,
            graph_descriptors=descriptors,
            thresholds=interpretation_thresholds,
        )
    else:
        approximate_value = None
        regret = compare_policy_values(exact_reference.optimal_value, None)
        distribution_metrics = {}
        state_aware = {}
        node_metrics = {}
        event_metrics = {}
        summary_metrics = {}
        dependence_metrics = {}
        labels = ()
    selected_policy = None
    if approximation.selected_candidate is not None:
        selected_policy = bgr.canonical_partition_policy_candidate_identity(approximation.selected_candidate)
    return RegionalCompoundingValidationRecord(
        benchmark_id=str(benchmark_id),
        graph_signature=(nodes, edges),
        topology_family=str(topology_family),
        graph_descriptors=descriptors,
        initial_state_signature=canonical_validation_state_signature(initial_state, nodes),
        exact_status=exact_reference.status,
        approximation_status=approximation.status,
        exact_composition_status=approximation.composition_status,
        exact_optimal_value=exact_reference.optimal_value,
        approximate_policy_exact_value=approximate_value,
        regret=regret,
        exact_policy_count=len(exact_reference.optimal_policy_set),
        approximation_selected_policy=selected_policy,
        distribution_metrics=distribution_metrics,
        state_aware_metrics=state_aware,
        node_marginal_metrics=node_metrics,
        strategic_event_metrics=event_metrics,
        strategic_summary_metrics=summary_metrics,
        cross_region_dependence_metrics=dependence_metrics,
        exact_runtime_seconds=exact_reference.runtime_seconds,
        approximation_runtime_seconds=approximation.runtime_seconds,
        exact_composition_runtime_seconds=float(approximation.composition_diagnostics.get("runtime_seconds", 0.0)),
        diagnostics={
            "state_alignment": alignment,
            "interpretation_profiles": labels,
            "interpretation_thresholds": asdict(interpretation_thresholds),
            "explicit_policy_regret_available": False,
            "distribution_implied_value_is_policy_regret_proxy": False,
            "distribution_implied_value_is_reported_separately": True,
            "full_graph_reference_policy_scope": exact_reference.diagnostics.get("optimal_policy_scope"),
        },
    )


@dataclass(frozen=True)
class BenchmarkGraphSpec:
    graph_id: str
    topology_family: str
    node_count: int
    attacker_count: int
    edges: Tuple[Tuple[int, int], ...]
    descriptors: Mapping[str, Any] = field(default_factory=dict)

    @property
    def nodes(self) -> Tuple[int, ...]:
        return tuple(range(1, int(self.node_count) + 1))

    def graph(self) -> ValidationGraph:
        return ValidationGraph(self.nodes, self.edges)


@dataclass(frozen=True)
class BenchmarkCase:
    benchmark_id: str
    graph_spec: BenchmarkGraphSpec
    troop_cap: int
    initial_state_signature: Tuple[Tuple[int, str, int], ...]
    state_stratum: str

    @property
    def initial_state(self) -> Tuple[Tuple[int, str, int], ...]:
        return self.initial_state_signature


def _topology_edges(family: str, node_count: int, attacker_count: int) -> Tuple[Tuple[int, int], ...]:
    nodes = tuple(range(1, node_count + 1))
    if family == "chain":
        edges = [(node, node + 1) for node in nodes[:-1]]
    elif family == "star":
        center = 1
        edges = [(center, node) for node in nodes if node != center]
    elif family == "tree":
        edges = [(node // 2, node) for node in nodes[1:]]
    elif family == "cycle":
        edges = [(node, node + 1) for node in nodes[:-1]] + [(nodes[-1], nodes[0])]
    elif family in {"bridge", "two_dense"}:
        split = max(2, node_count // 2)
        left = nodes[:split]
        right = nodes[split:]
        edges = list(itertools.combinations(left, 2)) + list(itertools.combinations(right, 2))
        edges.append((left[-1], right[0]))
    elif family == "double_front":
        attackers = nodes[:attacker_count]
        defenders = nodes[attacker_count:]
        edges = [(attackers[index], attackers[index + 1]) for index in range(len(attackers) - 1)]
        edges += [(defenders[index], defenders[index + 1]) for index in range(len(defenders) - 1)]
        for index, defender in enumerate(defenders):
            edges.append((attackers[index % len(attackers)], defender))
    elif family == "articulation":
        attackers = nodes[:attacker_count]
        defenders = nodes[attacker_count:]
        hub = attackers[-1]
        edges = [(attackers[index], attackers[index + 1]) for index in range(len(attackers) - 1)]
        edges += [(hub, defender) for defender in defenders]
        if len(defenders) > 1:
            edges += [(defenders[index], defenders[index + 1]) for index in range(len(defenders) - 1)]
    elif family == "sequence_opening":
        attackers = nodes[:attacker_count]
        defenders = nodes[attacker_count:]
        edges = [(attackers[-1], defenders[0])]
        edges += [(defenders[index], defenders[index + 1]) for index in range(len(defenders) - 1)]
        edges += [(attackers[index], attackers[index + 1]) for index in range(len(attackers) - 1)]
    else:
        raise ValueError(f"Unknown topology family {family!r}")
    return tuple(sorted({(min(left, right), max(left, right)) for left, right in edges if left != right}))


def generate_benchmark_graph_suite(
    *,
    node_counts: Sequence[int] = (6, 7, 8),
    topology_families: Sequence[str] = (
        "chain",
        "star",
        "tree",
        "cycle",
        "bridge",
        "double_front",
        "articulation",
        "sequence_opening",
    ),
) -> Tuple[BenchmarkGraphSpec, ...]:
    output = []
    for node_count in sorted({int(value) for value in node_counts}):
        if node_count < 4:
            raise ValueError("Benchmark graphs require at least four nodes")
        attacker_count = max(2, node_count // 2)
        for family in topology_families:
            edges = _topology_edges(str(family), node_count, attacker_count)
            graph_id = f"{family}_n{node_count}_{_stable_digest(edges)[:10]}"
            output.append(
                BenchmarkGraphSpec(
                    graph_id=graph_id,
                    topology_family=str(family),
                    node_count=node_count,
                    attacker_count=attacker_count,
                    edges=edges,
                    descriptors={
                        "expected_difficulty": (
                            "easy" if family in {"chain", "star", "tree"} else "difficult"
                        )
                    },
                )
            )
    return tuple(output)


def sample_benchmark_initial_states(
    graph_spec: BenchmarkGraphSpec,
    *,
    troop_cap: int,
    states_per_graph: int,
    random_seed: int,
) -> Tuple[BenchmarkCase, ...]:
    cap = int(troop_cap)
    if cap < 1:
        raise ValueError("troop_cap must be positive")
    output = []
    strata = (
        "balanced",
        "attacker_favored",
        "defender_favored",
        "mixed",
        "all_nodes_at_cap",
    )
    for state_index in range(int(states_per_graph)):
        stratum = strata[state_index % len(strata)]
        seed = int.from_bytes(
            hashlib.sha256(
                f"{random_seed}|{graph_spec.graph_id}|{cap}|{state_index}".encode("ascii")
            ).digest()[:8],
            "big",
        )
        rng = random.Random(seed)
        rows = []
        for node in graph_spec.nodes:
            owner = "A" if node <= graph_spec.attacker_count else "D"
            if stratum == "all_nodes_at_cap":
                low, high = (cap, cap)
            elif stratum == "attacker_favored":
                low, high = ((max(1, (cap + 1) // 2), cap) if owner == "A" else (1, max(1, cap // 2)))
            elif stratum == "defender_favored":
                low, high = ((1, max(1, cap // 2)) if owner == "A" else (max(1, (cap + 1) // 2), cap))
            else:
                low, high = (1, cap)
            rows.append((node, owner, rng.randint(low, high)))
        signature = tuple(rows)
        benchmark_id = "regional_benchmark_" + _stable_digest(
            (graph_spec.graph_id, cap, signature)
        )[:24]
        output.append(
            BenchmarkCase(
                benchmark_id=benchmark_id,
                graph_spec=graph_spec,
                troop_cap=cap,
                initial_state_signature=signature,
                state_stratum=stratum,
            )
        )
    return tuple(output)


class RegionalCompoundingValidationStore:
    LAYOUT = (
        "graphs",
        "states",
        "exact_results",
        "approximation_results",
        "comparison_records",
        "failures",
        "checkpoints",
    )

    def __init__(
        self,
        output_dir: Path | str,
        *,
        config: Optional[Mapping[str, Any]] = None,
        resume: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in self.LAYOUT:
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
        self.config = dict(config or {})
        self.config_fingerprint = _stable_digest(
            {
                "format": VALIDATION_FORMAT_VERSION,
                "schema": VALIDATION_SCHEMA_VERSION,
                "config": self.config,
            }
        )
        manifest_path = self.output_dir / "manifest.json"
        if resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_fingerprint") != self.config_fingerprint:
                raise ValueError("Regional validation resume configuration mismatch")
            self.manifest = manifest
        else:
            self.manifest = {
                "validation_format_version": VALIDATION_FORMAT_VERSION,
                "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                "config_fingerprint": self.config_fingerprint,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "completed_benchmark_ids": [],
                "failed_benchmark_ids": [],
                "layout": list(self.LAYOUT),
            }
            _atomic_write_json(self.output_dir / "config.json", self.config)
            self._write_manifest()

    @property
    def completed_ids(self) -> set[str]:
        return set(str(value) for value in self.manifest.get("completed_benchmark_ids", ()))

    def _write_manifest(self) -> None:
        self.manifest["updated_at"] = _utc_now()
        _atomic_write_json(self.output_dir / "manifest.json", self.manifest)

    def save_graph(self, graph_spec: BenchmarkGraphSpec) -> None:
        _atomic_write_json(
            self.output_dir / "graphs" / f"{graph_spec.graph_id}.json",
            graph_spec,
        )

    def save_state(self, case: BenchmarkCase) -> None:
        _atomic_write_pickle(
            self.output_dir / "states" / f"{case.benchmark_id}.pkl",
            case,
        )

    def save_completed(
        self,
        *,
        case: BenchmarkCase,
        exact: ExactFullGraphReferenceResult,
        approximation: RegionalApproximationResult,
        record: RegionalCompoundingValidationRecord,
    ) -> None:
        self.save_graph(case.graph_spec)
        self.save_state(case)
        _atomic_write_pickle(
            self.output_dir / "exact_results" / f"{case.benchmark_id}.pkl",
            exact,
        )
        _atomic_write_pickle(
            self.output_dir / "approximation_results" / f"{case.benchmark_id}.pkl",
            approximation,
        )
        _atomic_write_pickle(
            self.output_dir / "comparison_records" / f"{case.benchmark_id}.pkl",
            record,
        )
        completed = self.completed_ids
        completed.add(case.benchmark_id)
        self.manifest["completed_benchmark_ids"] = sorted(completed)
        self._write_manifest()
        _atomic_write_json(
            self.output_dir / "checkpoints" / "completed_states.json",
            {
                "completed_benchmark_ids": sorted(completed),
                "updated_at": _utc_now(),
            },
        )

    def save_failure(self, case: BenchmarkCase, exc: BaseException) -> None:
        _atomic_write_json(
            self.output_dir / "failures" / f"{case.benchmark_id}.json",
            {
                "benchmark_id": case.benchmark_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recorded_at": _utc_now(),
            },
        )
        failed = set(str(value) for value in self.manifest.get("failed_benchmark_ids", ()))
        failed.add(case.benchmark_id)
        self.manifest["failed_benchmark_ids"] = sorted(failed)
        self._write_manifest()


def run_regional_compounding_benchmark_case(
    case: BenchmarkCase,
    *,
    library_dir: Path | str,
    exact_max_runtime_seconds: Optional[float],
    exact_max_states: Optional[int],
    composition_limits: Optional[Mapping[str, Any]] = None,
    candidate_selection_mc_samples: int = 5,
    candidate_selection_seed: int = 42,
    state_aware_max_support_size: Optional[int] = 500,
) -> Tuple[ExactFullGraphReferenceResult, RegionalApproximationResult, RegionalCompoundingValidationRecord]:
    graph = case.graph_spec.graph()
    exact = solve_full_graph_exact_reference(
        graph=graph,
        initial_state=case.initial_state,
        utility_mode="local",
        include_all_optimal_policies=True,
        max_runtime_seconds=exact_max_runtime_seconds,
        max_states=exact_max_states,
    )
    approximation = evaluate_regional_compounding_approximation(
        graph=graph,
        initial_state=case.initial_state,
        library_dir=library_dir,
        partition_config={
            "partition_candidate_selection_mode": "maximal_per_partition_utility",
            "utility_abs_tolerance": None,
            "utility_rel_tolerance": None,
            "max_candidates_per_partition": None,
            "max_policy_combos_per_partition": None,
            "max_total_partition_policy_candidates": None,
            "max_partitions": 10000,
        },
        ranking_config={
            "ranking_variable": "battle_expected_attacker_territory_count",
            "candidate_selection_mc_samples": int(candidate_selection_mc_samples),
            "candidate_selection_seed": int(candidate_selection_seed),
        },
        exact_compose_selected_candidate=True,
        composition_limits=composition_limits,
    )
    record = build_regional_compounding_validation_record(
        benchmark_id=case.benchmark_id,
        topology_family=case.graph_spec.topology_family,
        graph=graph,
        initial_state=case.initial_state,
        exact_reference=exact,
        approximation=approximation,
        state_aware_max_support_size=state_aware_max_support_size,
    )
    diagnostics = dict(record.diagnostics)
    diagnostics.update(
        {
            "troop_cap": int(case.troop_cap),
            "state_stratum": case.state_stratum,
            "graph_size": case.graph_spec.node_count,
            "attacker_node_count": case.graph_spec.attacker_count,
            "defender_node_count": case.graph_spec.node_count - case.graph_spec.attacker_count,
            "retained_candidate_count": len(approximation.retained_candidates),
            "region_count": len(approximation.selected_partition_signature or ()),
        }
    )
    record = RegionalCompoundingValidationRecord(
        **{**asdict(record), "diagnostics": diagnostics}
    )
    return exact, approximation, record


def run_regional_compounding_cases(
    cases: Sequence[BenchmarkCase],
    *,
    output_dir: Path | str,
    library_dir: Path | str,
    exact_max_runtime_seconds: Optional[float] = 60.0,
    exact_max_states: Optional[int] = 250000,
    composition_max_unique_states: Optional[int] = 100000,
    composition_max_expansions: Optional[int] = 1000000,
    composition_max_runtime_seconds: Optional[float] = 60.0,
    candidate_selection_mc_samples: int = 5,
    candidate_selection_seed: int = 42,
    state_aware_max_support_size: Optional[int] = 500,
    resume: bool = False,
) -> Dict[str, Any]:
    config = {
        "library_dir": str(Path(library_dir)),
        "exact_max_runtime_seconds": exact_max_runtime_seconds,
        "exact_max_states": exact_max_states,
        "composition_max_unique_states": composition_max_unique_states,
        "composition_max_expansions": composition_max_expansions,
        "composition_max_runtime_seconds": composition_max_runtime_seconds,
        "candidate_selection_mc_samples": int(candidate_selection_mc_samples),
        "candidate_selection_seed": int(candidate_selection_seed),
        "state_aware_max_support_size": state_aware_max_support_size,
        "production_partition_semantics": "maximal_per_partition_utility_uncapped_exact_equality",
    }
    store = RegionalCompoundingValidationStore(output_dir, config=config, resume=resume)
    completed_now = 0
    skipped = 0
    failures = 0
    for case in cases:
        if case.benchmark_id in store.completed_ids:
            skipped += 1
            continue
        try:
            exact, approximation, record = run_regional_compounding_benchmark_case(
                case,
                library_dir=library_dir,
                exact_max_runtime_seconds=exact_max_runtime_seconds,
                exact_max_states=exact_max_states,
                composition_limits={
                    "max_unique_states": composition_max_unique_states,
                    "max_cartesian_expansions": composition_max_expansions,
                    "max_runtime_seconds": composition_max_runtime_seconds,
                },
                candidate_selection_mc_samples=candidate_selection_mc_samples,
                candidate_selection_seed=candidate_selection_seed,
                state_aware_max_support_size=state_aware_max_support_size,
            )
            store.save_completed(
                case=case,
                exact=exact,
                approximation=approximation,
                record=record,
            )
            completed_now += 1
        except Exception as exc:
            store.save_failure(case, exc)
            failures += 1
    return {
        "requested": len(cases),
        "completed_now": completed_now,
        "skipped_existing": skipped,
        "failures": failures,
        "completed_total": len(store.completed_ids),
        "output_dir": str(Path(output_dir)),
    }


def load_validation_records(output_dir: Path | str) -> Tuple[RegionalCompoundingValidationRecord, ...]:
    records = []
    for path in sorted((Path(output_dir) / "comparison_records").glob("*.pkl")):
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if isinstance(value, RegionalCompoundingValidationRecord):
            records.append(value)
        elif isinstance(value, Mapping):
            records.append(RegionalCompoundingValidationRecord(**dict(value)))
    return tuple(records)


def _numeric_summary(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    numbers = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not numbers:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "maximum": None}

    def percentile(fraction: float) -> float:
        if len(numbers) == 1:
            return numbers[0]
        position = fraction * (len(numbers) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return numbers[lower] * (1.0 - weight) + numbers[upper] * weight

    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "maximum": max(numbers),
    }


def _flat_record_metrics(record: RegionalCompoundingValidationRecord) -> Dict[str, Any]:
    balanced = record.state_aware_metrics.get("balanced", {})
    conquest = record.strategic_event_metrics.get("complete_conquest", {})
    no_gain = record.strategic_event_metrics.get("no_territory_gained", {})
    territory = record.strategic_summary_metrics.get("new_attacker_territories", {})
    attacker_troops = record.strategic_summary_metrics.get("final_attacker_troop_total", {})
    distribution_value_gap = record.regret.get("distribution_value_gap", {})
    return {
        "benchmark_id": record.benchmark_id,
        "topology_family": record.topology_family,
        "graph_size": record.diagnostics.get("graph_size"),
        "troop_cap": record.diagnostics.get("troop_cap"),
        "region_count": record.diagnostics.get("region_count"),
        "retained_candidate_count": record.diagnostics.get("retained_candidate_count"),
        "articulation_present": bool(record.graph_descriptors.get("articulation_nodes")),
        "shared_troop_source_present": bool(record.graph_descriptors.get("shared_troop_source_present")),
        "sequence_opening_present": bool(record.graph_descriptors.get("sequence_opening_present")),
        "exact_policy_count": record.exact_policy_count,
        "exact_status": record.exact_status,
        "approximation_status": record.approximation_status,
        "composition_status": record.exact_composition_status,
        "primary_regret": record.regret.get("primary_component_regret"),
        "distribution_value_primary_gap": distribution_value_gap.get("primary_component_regret"),
        "tv": record.distribution_metrics.get("total_variation"),
        "js": record.distribution_metrics.get("jensen_shannon"),
        "wasserstein_balanced": balanced.get("distance"),
        "ownership_error": record.node_marginal_metrics.get("mean_absolute_ownership_probability_error"),
        "troop_error": record.node_marginal_metrics.get("mean_absolute_expected_troop_error"),
        "conquest_probability_error": conquest.get("absolute_error"),
        "no_gain_error": no_gain.get("absolute_error"),
        "territory_expectation_error": territory.get("absolute_expectation_error"),
        "troop_expectation_error": attacker_troops.get("absolute_expectation_error"),
        "cross_region_covariance_error": record.cross_region_dependence_metrics.get("maximum_absolute_covariance_error"),
        "exact_runtime_seconds": record.exact_runtime_seconds,
        "approximation_runtime_seconds": record.approximation_runtime_seconds,
        "composition_runtime_seconds": record.exact_composition_runtime_seconds,
        "interpretation_profiles": record.diagnostics.get("interpretation_profiles", ()),
    }


def _pearson(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> Optional[float]:
    pairs = []
    for row in rows:
        try:
            x, y = float(row.get(left)), float(row.get(right))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator if denominator > 0.0 else None


def summarize_regional_compounding_validation(output_dir: Path | str) -> Dict[str, Any]:
    root = Path(output_dir)
    records = load_validation_records(root)
    rows = [_flat_record_metrics(record) for record in records]
    metric_names = (
        "primary_regret",
        "distribution_value_primary_gap",
        "tv",
        "js",
        "wasserstein_balanced",
        "ownership_error",
        "troop_error",
        "conquest_probability_error",
        "no_gain_error",
        "territory_expectation_error",
        "troop_expectation_error",
        "cross_region_covariance_error",
        "exact_runtime_seconds",
        "approximation_runtime_seconds",
        "composition_runtime_seconds",
    )
    overall = {name: _numeric_summary(row.get(name) for row in rows) for name in metric_names}
    groups: Dict[str, Any] = {}
    for group_name in (
        "graph_size",
        "troop_cap",
        "topology_family",
        "region_count",
        "retained_candidate_count",
        "articulation_present",
        "shared_troop_source_present",
        "sequence_opening_present",
        "exact_policy_count",
    ):
        group_values: Dict[str, Any] = {}
        for value in sorted({repr(row.get(group_name)) for row in rows}):
            subset = [row for row in rows if repr(row.get(group_name)) == value]
            group_values[value] = {
                "records": len(subset),
                **{name: _numeric_summary(row.get(name) for row in subset) for name in metric_names[:12]},
            }
        groups[group_name] = group_values
    correlations = {
        f"tv_vs_{descriptor}": _pearson(rows, "tv", descriptor)
        for descriptor in (
            "graph_size",
            "troop_cap",
            "region_count",
            "retained_candidate_count",
            "exact_policy_count",
        )
    }
    outliers = sorted(
        rows,
        key=lambda row: (
            float(row.get("tv") or -1.0),
            float(row.get("distribution_value_primary_gap") or -1.0),
        ),
        reverse=True,
    )[:20]
    strategic_event_names = sorted(
        {
            name
            for record in records
            for name in record.strategic_event_metrics
        }
    )
    strategic_summary_names = sorted(
        {
            name
            for record in records
            for name in record.strategic_summary_metrics
        }
    )
    state_aware_profile_names = sorted(
        {
            name
            for record in records
            for name in record.state_aware_metrics
        }
    )
    strategic_event_error_summaries = {
        name: _numeric_summary(
            record.strategic_event_metrics.get(name, {}).get("absolute_error")
            for record in records
        )
        for name in strategic_event_names
    }
    strategic_summary_expectation_error_summaries = {
        name: _numeric_summary(
            record.strategic_summary_metrics.get(name, {}).get("absolute_expectation_error")
            for record in records
        )
        for name in strategic_summary_names
    }
    state_aware_profile_summaries = {
        name: {
            "distance": _numeric_summary(
                record.state_aware_metrics.get(name, {}).get("distance")
                for record in records
            ),
            "status_histogram": {
                status: sum(
                    record.state_aware_metrics.get(name, {}).get("status") == status
                    for record in records
                )
                for status in sorted(
                    {
                        record.state_aware_metrics.get(name, {}).get("status")
                        for record in records
                        if record.state_aware_metrics.get(name, {}).get("status") is not None
                    }
                )
            },
        }
        for name in state_aware_profile_names
    }
    summary = {
        "validation_format_version": VALIDATION_FORMAT_VERSION,
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "record_count": len(records),
        "complete_exact_count": sum(record.exact_status == "exact_complete" for record in records),
        "complete_approximation_count": sum(record.approximation_status == "approximation_complete" for record in records),
        "complete_composition_count": sum(record.exact_composition_status == "exact_complete" for record in records),
        "overall": overall,
        "groups": groups,
        "strategic_event_error_summaries": strategic_event_error_summaries,
        "strategic_summary_expectation_error_summaries": strategic_summary_expectation_error_summaries,
        "state_aware_profile_summaries": state_aware_profile_summaries,
        "exact_optimal_distribution_comparison": {
            "matches_any_exact_optimal_distribution": sum(
                bool(record.distribution_metrics.get("matches_any_exact_optimal_policy"))
                for record in records
            ),
            "canonical_exact_distribution_matches": sum(
                float(record.distribution_metrics.get("canonical_exact_policy_tv", math.inf)) <= 1e-10
                for record in records
            ),
            "interpretation": "distribution equality only; regional V2 policies cannot be identity-matched as full policies",
        },
        "correlations": correlations,
        "correlation_interpretation": "descriptive association only; not causal evidence",
        "policy_regret_availability": {
            "exact_lifted_policy_records": sum(
                record.regret.get("evaluation_status") == "exact_complete" for record in records
            ),
            "unavailable_records": sum(
                record.regret.get("evaluation_status") != "exact_complete" for record in records
            ),
            "reason": "regional V2 payloads do not encode a complete full-graph state-to-action policy",
            "distribution_value_primary_gap_is_not_policy_regret": True,
        },
        "outlier_states": outliers,
        "primary_questions_require_manual_review": True,
    }
    _atomic_write_json(root / "benchmark_summary.json", summary)
    if rows:
        csv_path = root / "benchmark_records.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _json_ready(value) for key, value in row.items()})
        temporary.replace(csv_path)
    return summary


def validate_regional_compounding_output(output_dir: Path | str) -> Dict[str, Any]:
    root = Path(output_dir)
    errors = []
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        errors.append("missing manifest.json")
        completed_ids = set()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed_ids = set(manifest.get("completed_benchmark_ids", ()))
        if manifest.get("validation_format_version") != VALIDATION_FORMAT_VERSION:
            errors.append("validation format version mismatch")
    record_paths = sorted((root / "comparison_records").glob("*.pkl"))
    records = load_validation_records(root)
    if len(records) != len(record_paths):
        errors.append("one or more comparison records could not be loaded")
    record_ids = {record.benchmark_id for record in records}
    if record_ids != completed_ids:
        errors.append("manifest completed IDs do not match comparison records")
    for record in records:
        if record.exact_composition_status == "exact_complete":
            path = root / "approximation_results" / f"{record.benchmark_id}.pkl"
            with path.open("rb") as handle:
                approximation = pickle.load(handle)
            distribution = approximation.exact_compounded_distribution or {}
            mass = sum(distribution.values())
            if abs(mass - 1.0) > 1e-8:
                errors.append(f"{record.benchmark_id}: exact composition mass={mass}")
    validation = {
        "valid": not errors,
        "errors": errors,
        "record_count": len(records),
        "completed_manifest_count": len(completed_ids),
        "failure_file_count": len(list((root / "failures").glob("*.json"))),
        "validated_at": _utc_now(),
    }
    _atomic_write_json(root / "validation.json", validation)
    return validation


def run_tractability_pilot(
    *,
    output_dir: Path | str,
    library_dir: Path | str,
    node_counts: Sequence[int],
    troop_caps: Sequence[int],
    topology_families: Sequence[str],
    states_per_cell: int,
    random_seed: int,
    exact_max_runtime_seconds: Optional[float],
    exact_max_states: Optional[int],
    run_regional_approximation: bool = False,
    **case_kwargs: Any,
) -> Dict[str, Any]:
    root = Path(output_dir)
    graph_specs = generate_benchmark_graph_suite(
        node_counts=node_counts,
        topology_families=topology_families,
    )
    cases = tuple(
        case
        for graph_spec in graph_specs
        for cap in troop_caps
        for case in sample_benchmark_initial_states(
            graph_spec,
            troop_cap=int(cap),
            states_per_graph=int(states_per_cell),
            random_seed=int(random_seed),
        )
    )
    if run_regional_approximation:
        run_stats = run_regional_compounding_cases(
            cases,
            output_dir=root,
            library_dir=library_dir,
            exact_max_runtime_seconds=exact_max_runtime_seconds,
            exact_max_states=exact_max_states,
            **case_kwargs,
        )
        records = load_validation_records(root)
        rows = [
            {
                "benchmark_id": record.benchmark_id,
                "node_count": record.diagnostics.get("graph_size"),
                "troop_cap": record.diagnostics.get("troop_cap"),
                "topology_family": record.topology_family,
                "exact_status": record.exact_status,
                "exact_runtime_seconds": record.exact_runtime_seconds,
                "exact_states_evaluated": None,
                "exact_distribution_support_size": record.distribution_metrics.get("p_support_size"),
                "exact_policy_count": record.exact_policy_count,
                "approximation_status": record.approximation_status,
                "approximation_runtime_seconds": record.approximation_runtime_seconds,
                "composition_runtime_seconds": record.exact_composition_runtime_seconds,
            }
            for record in records
        ]
    else:
        run_stats = {"requested": len(cases), "completed_now": 0, "failures": 0}
        rows = []
        for case in cases:
            exact = solve_full_graph_exact_reference(
                graph=case.graph_spec.graph(),
                initial_state=case.initial_state,
                max_runtime_seconds=exact_max_runtime_seconds,
                max_states=exact_max_states,
            )
            rows.append(
                {
                    "benchmark_id": case.benchmark_id,
                    "node_count": case.graph_spec.node_count,
                    "troop_cap": case.troop_cap,
                    "topology_family": case.graph_spec.topology_family,
                    "exact_status": exact.status,
                    "exact_runtime_seconds": exact.runtime_seconds,
                    "exact_states_evaluated": exact.states_evaluated,
                    "exact_distribution_support_size": len(exact.canonical_optimal_distribution),
                    "exact_policy_count": len(exact.optimal_policy_set),
                    "approximation_status": "not_run",
                    "approximation_runtime_seconds": 0.0,
                    "composition_runtime_seconds": 0.0,
                }
            )
    cells = {}
    for key, group in itertools.groupby(
        sorted(rows, key=lambda row: (row["node_count"], row["troop_cap"], row["topology_family"])),
        key=lambda row: (row["node_count"], row["troop_cap"], row["topology_family"]),
    ):
        group_rows = list(group)
        cells[repr(key)] = {
            "states": len(group_rows),
            "exact_complete": sum(row["exact_status"] == "exact_complete" for row in group_rows),
            "status_histogram": {
                status: sum(row["exact_status"] == status for row in group_rows)
                for status in sorted({row["exact_status"] for row in group_rows})
            },
            "runtime_seconds": _numeric_summary(row["exact_runtime_seconds"] for row in group_rows),
            "states_evaluated": _numeric_summary(row["exact_states_evaluated"] for row in group_rows),
            "support_size": _numeric_summary(row["exact_distribution_support_size"] for row in group_rows),
            "optimal_policy_count": _numeric_summary(row["exact_policy_count"] for row in group_rows),
        }
    summary = {
        "generated_at": _utc_now(),
        "cases": len(rows),
        "exact_complete": sum(row["exact_status"] == "exact_complete" for row in rows),
        "cells": cells,
        "limits": {
            "exact_max_runtime_seconds": exact_max_runtime_seconds,
            "exact_max_states": exact_max_states,
        },
        "run_regional_approximation": bool(run_regional_approximation),
        "run_stats": run_stats,
    }
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(root / "tractability_summary.json", summary)
    _atomic_write_pickle(root / "tractability_records.pkl", rows)
    return summary
