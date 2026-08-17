"""Exact regional-candidate selection and v2 validation reporting.

This module is validation-only. It reuses the corrected production candidate
set and second-stage utility semantics without changing production routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import csv
import ctypes
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import random
import statistics
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from project_risk.game_simulation import Board
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.continent_model import battle_graph_ranking as bgr
from project_risk.validation import distribution_comparison_metrics as dcm
from project_risk.validation import regional_compounding_validation as rcv
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState, NodeState


V2_FORMAT_VERSION = "regional_compounding_validation_v2_exact_candidate_selection"
V2_SCHEMA_VERSION = "regional_compounding_validation_v2_schema_1"


@dataclass(frozen=True)
class ExactRegionalCandidateEvaluation:
    candidate_identity: Any
    partition_signature: Any
    policy_option_indices: Tuple[int, ...]
    composition_status: str
    successor_distribution: Optional[Mapping[Any, float]]
    successor_support_size: Optional[int]
    exact_expected_score: Optional[Tuple[float, ...]]
    exact_score_components: Mapping[str, Any]
    raw_cartesian_product_size: Optional[int]
    raw_cartesian_expansions: int
    unique_states_after_each_region: Tuple[int, ...]
    composition_runtime_seconds: float
    global_evaluation_runtime_seconds: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ExactRegionalCandidateSelectionResult:
    status: str
    candidate_evaluations: Tuple[ExactRegionalCandidateEvaluation, ...]
    exact_best_candidate_identities: Tuple[Any, ...]
    canonical_selected_candidate_identity: Any
    selected_partition_signature: Any
    selected_policy_option_indices: Optional[Tuple[int, ...]]
    selected_successor_distribution: Optional[Mapping[Any, float]]
    selected_exact_expected_score: Optional[Tuple[float, ...]]
    total_unique_successor_states: int
    global_evaluation_cache_hits: int
    global_evaluation_cache_misses: int
    runtime_seconds: float
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RegionalCompoundingValidationV2Record:
    benchmark_id: str
    v1_benchmark_id: Optional[str]
    graph_signature: Any
    topology_family: str
    initial_state_signature: Any
    territory_mapping: Mapping[int, str]
    full_exact_status: str
    regional_mc1_status: str
    regional_exact_status: str
    full_exact_distribution: Optional[Mapping[Any, float]]
    regional_mc1_distribution: Optional[Mapping[Any, float]]
    regional_exact_distribution: Optional[Mapping[Any, float]]
    regional_mc1_candidate_identity: Any
    regional_exact_candidate_identity: Any
    mc1_vs_full_metrics: Mapping[str, Any]
    exact_regional_vs_full_metrics: Mapping[str, Any]
    mc1_vs_exact_regional_metrics: Mapping[str, Any]
    candidate_selection_effect: Mapping[str, Any]
    structural_approximation_effect: Mapping[str, Any]
    exact_candidate_selection_diagnostics: Mapping[str, Any]
    composition_complexity_diagnostics: Mapping[str, Any]
    diagnostic_labels: Tuple[str, ...]
    runtimes: Mapping[str, float]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorClassificationThresholds:
    version: str = "regional_error_source_thresholds_v1"
    candidate_selection_tv: float = 0.05
    structural_tv: float = 0.20
    near_exact_tv: float = 0.05
    severe_tv: float = 0.90
    low_wasserstein: float = 0.10
    high_strategic_event_error: float = 0.10
    high_ownership_error: float = 0.10
    low_ownership_error: float = 0.05
    high_troop_error: float = 0.50
    distribution_identity_tolerance: float = 1e-10


@dataclass(frozen=True)
class ExactTractabilityExpansionRecord:
    benchmark_id: str
    graph_signature: Any
    topology_family: str
    graph_size: int
    attacker_node_count: int
    defender_node_count: int
    troop_cap: int
    state_stratum: str
    initial_state_signature: Any
    initial_legal_action_count: int
    status: str
    limit_reached: Optional[str]
    runtime_seconds: float
    states_evaluated: int
    value_cache_entries: int
    distribution_cache_entries: int
    total_cache_entries: int
    cache_hits: int
    actions_evaluated: int
    combat_lookups: int
    terminal_support_size: int
    optimal_policy_count: int
    estimated_cache_bytes: int
    rss_before_bytes: Optional[int]
    rss_after_bytes: Optional[int]
    rss_delta_bytes: Optional[int]
    diagnostics: Mapping[str, Any]


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            pass
    return repr(value)


def _stable_digest(value: Any) -> str:
    payload = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    rcv._atomic_write_json(path, payload)


def _atomic_pickle(path: Path, payload: Any) -> None:
    rcv._atomic_write_pickle(path, payload)


def _candidate_identity(candidate: bgr.PartitionPolicyCandidate) -> Any:
    return bgr.canonical_partition_policy_candidate_identity(candidate)


def _candidate_partition(candidate: bgr.PartitionPolicyCandidate) -> Tuple[Tuple[int, ...], ...]:
    return bgr.canonical_partition_signature(candidate.partition_regions)


def _candidate_option_indices(candidate: bgr.PartitionPolicyCandidate) -> Tuple[int, ...]:
    return tuple(
        int(ref.option_index)
        for ref in sorted(
            candidate.region_policy_options,
            key=lambda item: tuple(sorted(int(node) for node in item.region_nodes)),
        )
    )


def _state_from_partial_signature(base_state: GlobalState, signature: Any) -> GlobalState:
    nodes = list(base_state.nodes)
    for node, owner, troops in dcm.canonical_risk_state(signature):
        index = int(node)
        if index < 0 or index >= len(nodes):
            raise ValueError(f"Successor node {index} is outside the base global state")
        nodes[index] = NodeState(str(owner), int(troops))
    return GlobalState(nodes=tuple(nodes))


def _empty_selection_result(
    status: str,
    *,
    started: float,
    evaluations: Sequence[ExactRegionalCandidateEvaluation] = (),
    cache_hits: int = 0,
    cache_misses: int = 0,
    total_unique_states: Optional[int] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> ExactRegionalCandidateSelectionResult:
    return ExactRegionalCandidateSelectionResult(
        status=str(status),
        candidate_evaluations=tuple(evaluations),
        exact_best_candidate_identities=(),
        canonical_selected_candidate_identity=None,
        selected_partition_signature=None,
        selected_policy_option_indices=None,
        selected_successor_distribution=None,
        selected_exact_expected_score=None,
        total_unique_successor_states=int(
            cache_misses if total_unique_states is None else total_unique_states
        ),
        global_evaluation_cache_hits=int(cache_hits),
        global_evaluation_cache_misses=int(cache_misses),
        runtime_seconds=float(time.perf_counter() - started),
        diagnostics=dict(diagnostics or {}),
    )


def evaluate_regional_candidates_exactly(
    *,
    prepared_candidates: Any,
    initial_global_state: GlobalState,
    ranking_variable: str,
    evaluation_mode: str = "production_second_stage",
    rollout_steps: int = 0,
    global_evaluator: Callable[[GlobalState], Sequence[float]],
    composition_limits: Optional[Mapping[str, Any]] = None,
    evaluation_cache: Optional[MutableMapping[Any, Tuple[float, ...]]] = None,
    node_indices: Optional[Sequence[int]] = None,
) -> ExactRegionalCandidateSelectionResult:
    """Select the exact winner within the fixed regional approximation.

    Every retained candidate is composed. If any composition or global-state
    evaluation is incomplete, no winner is claimed from the tractable subset.
    """
    started = time.perf_counter()
    limits = dict(composition_limits or {})
    candidates_raw = getattr(prepared_candidates, "retained_candidates", prepared_candidates)
    candidates = tuple(sorted(tuple(candidates_raw or ()), key=_candidate_identity))
    if not candidates:
        return _empty_selection_result("no_candidates", started=started, diagnostics={"candidate_count": 0})
    if str(evaluation_mode) not in {"production_second_stage", "custom"} or int(rollout_steps) != 0:
        return _empty_selection_result(
            "invalid_candidate",
            started=started,
            diagnostics={
                "reason": "only current zero-step production second-stage evaluation is supported",
                "evaluation_mode": str(evaluation_mode),
                "rollout_steps": int(rollout_steps),
            },
        )
    identities = tuple(_candidate_identity(candidate) for candidate in candidates)
    if len(set(identities)) != len(identities):
        return _empty_selection_result(
            "invalid_candidate",
            started=started,
            diagnostics={"reason": "duplicate canonical candidate identities"},
        )

    battle_nodes = tuple(
        sorted(
            int(node)
            for node in (
                node_indices
                or getattr(prepared_candidates, "battle_nodes", ())
                or {
                    node
                    for candidate in candidates
                    for ref in candidate.region_policy_options
                    for node in ref.region_nodes
                }
            )
        )
    )
    preparation_started = time.perf_counter()
    try:
        prepared_options, option_diagnostics = bgr.prepare_unique_regional_policy_options(candidates)
        assembly_plans, assembly_diagnostics = bgr.prepare_partition_assembly_plans(
            candidates,
            base_global_state=initial_global_state,
            battle_nodes=battle_nodes,
        )
    except Exception as exc:
        return _empty_selection_result(
            "invalid_candidate",
            started=started,
            diagnostics={"error": f"{type(exc).__name__}: {exc}"},
        )
    preparation_seconds = float(time.perf_counter() - preparation_started)
    cache: MutableMapping[Any, Tuple[float, ...]] = evaluation_cache if evaluation_cache is not None else {}
    cache_hits = 0
    cache_misses = 0
    raw_successor_requests = 0
    requested_successor_signatures = set()
    assembled_state_cache: Dict[Any, Tuple[GlobalState, Any]] = {}
    state_conversion_cache_hits = 0
    state_conversion_cache_misses = 0
    composition_runtime = 0.0
    global_runtime = 0.0
    evaluations: List[ExactRegionalCandidateEvaluation] = []
    incomplete_composition_statuses: List[str] = []
    max_global_evaluations = limits.get("max_global_evaluations")
    max_total_runtime = limits.get("max_total_runtime_seconds")
    max_unique_successors = limits.get("max_unique_successor_states")

    for candidate in candidates:
        if max_total_runtime is not None and time.perf_counter() - started >= float(max_total_runtime):
            return _empty_selection_result(
                "candidate_evaluation_limit",
                started=started,
                evaluations=evaluations,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                diagnostics={"limit_reached": "max_total_runtime_seconds"},
            )
        composition = rcv.compose_selected_candidate_distribution_exact(
            selected_candidate=candidate,
            initial_global_state=initial_global_state,
            node_indices=battle_nodes,
            max_unique_states=limits.get("max_unique_states"),
            max_cartesian_expansions=limits.get("max_cartesian_expansions"),
            max_runtime_seconds=limits.get("max_composition_runtime_seconds"),
            prepared_regional_options=prepared_options,
            prepared_regional_options_diagnostics=option_diagnostics,
        )
        composition_runtime += float(composition.runtime_seconds)
        identity = _candidate_identity(candidate)
        partition = _candidate_partition(candidate)
        options = _candidate_option_indices(candidate)
        complexity = dict(composition.diagnostics)
        if composition.status != "exact_complete":
            evaluations.append(
                ExactRegionalCandidateEvaluation(
                    candidate_identity=identity,
                    partition_signature=partition,
                    policy_option_indices=options,
                    composition_status=composition.status,
                    successor_distribution=None,
                    successor_support_size=None,
                    exact_expected_score=None,
                    exact_score_components={},
                    raw_cartesian_product_size=complexity.get("raw_cartesian_product_size"),
                    raw_cartesian_expansions=composition.raw_cartesian_expansions,
                    unique_states_after_each_region=composition.unique_states_after_each_region,
                    composition_runtime_seconds=composition.runtime_seconds,
                    global_evaluation_runtime_seconds=0.0,
                    diagnostics={"composition": complexity},
                )
            )
            status = (
                "candidate_composition_limit"
                if composition.status in {"unique_state_limit", "cartesian_expansion_limit", "runtime_limit"}
                else "probability_error"
                if composition.status == "probability_error"
                else "invalid_candidate"
            )
            incomplete_composition_statuses.append(status)
            continue

        raw_successor_requests += len(composition.distribution)
        accumulator: Optional[List[float]] = None
        candidate_global_started = time.perf_counter()
        for signature, probability in sorted(composition.distribution.items(), key=lambda item: repr(item[0])):
            if signature in assembled_state_cache:
                state, complete_signature = assembled_state_cache[signature]
                state_conversion_cache_hits += 1
            else:
                state = _state_from_partial_signature(initial_global_state, signature)
                complete_signature = bgr.canonical_two_stage_global_state_signature(state)
                assembled_state_cache[signature] = (state, complete_signature)
                state_conversion_cache_misses += 1
            is_new_requested_state = complete_signature not in requested_successor_signatures
            if (
                is_new_requested_state
                and max_unique_successors is not None
                and len(requested_successor_signatures) >= int(max_unique_successors)
            ):
                return _empty_selection_result(
                    "candidate_evaluation_limit",
                    started=started,
                    evaluations=evaluations,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    total_unique_states=len(requested_successor_signatures),
                    diagnostics={
                        "limit_reached": "max_unique_successor_states",
                        "complete_selection_not_claimed": True,
                    },
                )
            requested_successor_signatures.add(complete_signature)
            if complete_signature in cache:
                score = tuple(float(value) for value in cache[complete_signature])
                cache_hits += 1
            else:
                if max_global_evaluations is not None and cache_misses >= int(max_global_evaluations):
                    return _empty_selection_result(
                        "candidate_evaluation_limit",
                        started=started,
                        evaluations=evaluations,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        diagnostics={
                            "limit_reached": "max_global_evaluations",
                            "complete_selection_not_claimed": True,
                        },
                    )
                try:
                    score = tuple(float(value) for value in global_evaluator(state))
                except Exception as exc:
                    return _empty_selection_result(
                        "global_evaluation_error",
                        started=started,
                        evaluations=evaluations,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        diagnostics={
                            "error": f"{type(exc).__name__}: {exc}",
                            "failed_state_signature": complete_signature,
                            "complete_selection_not_claimed": True,
                        },
                    )
                if not score or any(not math.isfinite(value) for value in score):
                    return _empty_selection_result(
                        "global_evaluation_error",
                        started=started,
                        evaluations=evaluations,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        diagnostics={"reason": "global evaluator returned an empty or non-finite score"},
                    )
                cache[complete_signature] = score
                cache_misses += 1
            if accumulator is None:
                accumulator = [0.0] * len(score)
            if len(score) != len(accumulator):
                return _empty_selection_result(
                    "global_evaluation_error",
                    started=started,
                    evaluations=evaluations,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    diagnostics={"reason": "global evaluator score width changed between states"},
                )
            for index, value in enumerate(score):
                accumulator[index] += float(probability) * value
        candidate_global_runtime = float(time.perf_counter() - candidate_global_started)
        global_runtime += candidate_global_runtime
        expected_score = tuple(float(value) for value in (accumulator or ()))
        component_names = (
            "expected_new_territories",
            "expected_attacker_troops",
            "conquest_probability",
        )
        evaluations.append(
            ExactRegionalCandidateEvaluation(
                candidate_identity=identity,
                partition_signature=partition,
                policy_option_indices=options,
                composition_status="exact_complete",
                successor_distribution=dict(composition.distribution),
                successor_support_size=len(composition.distribution),
                exact_expected_score=expected_score,
                exact_score_components={
                    component_names[index] if index < len(component_names) else f"component_{index}": value
                    for index, value in enumerate(expected_score)
                },
                raw_cartesian_product_size=complexity.get("raw_cartesian_product_size"),
                raw_cartesian_expansions=composition.raw_cartesian_expansions,
                unique_states_after_each_region=composition.unique_states_after_each_region,
                composition_runtime_seconds=composition.runtime_seconds,
                global_evaluation_runtime_seconds=candidate_global_runtime,
                diagnostics={"composition": complexity},
            )
        )

    if incomplete_composition_statuses:
        status_priority = (
            "candidate_composition_limit",
            "probability_error",
            "invalid_candidate",
        )
        status = next(
            value for value in status_priority if value in incomplete_composition_statuses
        )
        return _empty_selection_result(
            status,
            started=started,
            evaluations=evaluations,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            total_unique_states=len(requested_successor_signatures),
            diagnostics={
                "failed_candidate_count": len(incomplete_composition_statuses),
                "failed_selection_statuses": tuple(incomplete_composition_statuses),
                "candidate_count": len(candidates),
                "exactly_composed_candidate_count": sum(
                    item.composition_status == "exact_complete" for item in evaluations
                ),
                "all_retained_candidates_reported": len(evaluations) == len(candidates),
                "complete_selection_not_claimed": True,
            },
        )

    scored = tuple(item for item in evaluations if item.exact_expected_score is not None)
    if len(scored) != len(candidates):
        return _empty_selection_result(
            "candidate_evaluation_limit",
            started=started,
            evaluations=evaluations,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            diagnostics={"complete_selection_not_claimed": True},
        )
    best_score = max(tuple(item.exact_expected_score or ()) for item in scored)
    tied = tuple(item for item in scored if tuple(item.exact_expected_score or ()) == best_score)
    selected = min(tied, key=lambda item: item.candidate_identity)
    return ExactRegionalCandidateSelectionResult(
        status="exact_complete",
        candidate_evaluations=tuple(evaluations),
        exact_best_candidate_identities=tuple(item.candidate_identity for item in tied),
        canonical_selected_candidate_identity=selected.candidate_identity,
        selected_partition_signature=selected.partition_signature,
        selected_policy_option_indices=selected.policy_option_indices,
        selected_successor_distribution=dict(selected.successor_distribution or {}),
        selected_exact_expected_score=selected.exact_expected_score,
        total_unique_successor_states=len(requested_successor_signatures),
        global_evaluation_cache_hits=cache_hits,
        global_evaluation_cache_misses=cache_misses,
        runtime_seconds=float(time.perf_counter() - started),
        diagnostics={
            "candidate_count": len(candidates),
            "exactly_composed_candidate_count": len(evaluations),
            "raw_successor_state_requests": raw_successor_requests,
            "unique_successor_states": len(requested_successor_signatures),
            "global_evaluation_cache_hits": cache_hits,
            "global_evaluation_cache_misses": cache_misses,
            "assembled_state_cache_hits": state_conversion_cache_hits,
            "assembled_state_cache_misses": state_conversion_cache_misses,
            "assembled_state_cache_size": len(assembled_state_cache),
            "regional_option_count": int(option_diagnostics.get("num_candidate_region_references", 0)),
            "unique_regional_option_count": int(option_diagnostics.get("num_unique_region_options", 0)),
            "regional_distribution_preparation_seconds": preparation_seconds,
            "composition_runtime_seconds": composition_runtime,
            "global_evaluation_runtime_seconds": global_runtime,
            "total_runtime_seconds": float(time.perf_counter() - started),
            "prepared_partition_assemblies": assembly_diagnostics,
            "ranking_variable": str(ranking_variable),
            "evaluation_mode": str(evaluation_mode),
            "rollout_steps": int(rollout_steps),
            "strict_lexicographic_comparison": True,
            "score_tolerance": None,
            "all_exact_ties_preserved": True,
            "canonical_tie_break": "lowest canonical candidate identity",
            "all_retained_candidates_evaluated": True,
            "successor_distribution_node_indices": battle_nodes,
        },
    )


def territory_name_for_node(node_id: int) -> str:
    index = int(node_id)
    territory = Board.node_to_territory_dict.get(index)
    if territory is None:
        raise KeyError(f"Unknown authoritative Risk territory node ID {index}")
    name = str(territory._name)
    duplicates = [
        int(other)
        for other, candidate in Board.node_to_territory_dict.items()
        if int(other) != index and str(candidate._name) == name
    ]
    if duplicates:
        raise ValueError(f"Authoritative territory name {name!r} is duplicated at nodes {duplicates}")
    return name


def _board_adjacency() -> Dict[int, set[int]]:
    return {
        int(node): {int(neighbor._index) for neighbor in territory._neighbors}
        for node, territory in Board.node_to_territory_dict.items()
    }


def _find_board_embedding(
    graph: Any,
    *,
    induced: bool,
) -> Optional[Dict[int, int]]:
    nodes, edges = rcv._graph_nodes_edges(graph)
    pattern = {int(node): set() for node in nodes}
    for left, right in edges:
        pattern[int(left)].add(int(right))
        pattern[int(right)].add(int(left))
    board = _board_adjacency()
    board_nodes = tuple(sorted(board))
    if not nodes:
        return {}
    if max(len(pattern[node]) for node in nodes) > max(len(board[node]) for node in board_nodes):
        return None
    order = tuple(sorted(nodes, key=lambda node: (-len(pattern[node]), int(node))))
    mapping: Dict[int, int] = {}
    used: set[int] = set()

    def search(position: int) -> Optional[Dict[int, int]]:
        if position == len(order):
            return dict(mapping)
        pattern_node = int(order[position])
        for board_node in board_nodes:
            if board_node in used or len(board[board_node]) < len(pattern[pattern_node]):
                continue
            compatible = True
            for assigned_pattern, assigned_board in mapping.items():
                pattern_edge = assigned_pattern in pattern[pattern_node]
                board_edge = assigned_board in board[board_node]
                incompatible = (
                    pattern_edge != board_edge
                    if induced
                    else pattern_edge and not board_edge
                )
                if incompatible:
                    compatible = False
                    break
            if not compatible:
                continue
            mapping[pattern_node] = board_node
            used.add(board_node)
            answer = search(position + 1)
            if answer is not None:
                return answer
            used.remove(board_node)
            del mapping[pattern_node]
        return None

    return search(0)


def authoritative_territory_mapping_for_graph(
    graph: Any,
    *,
    allow_edge_preserving_non_induced_fallback: bool = False,
) -> Dict[str, Any]:
    nodes, edges = rcv._graph_nodes_edges(graph)
    board = _board_adjacency()
    direct_possible = set(nodes).issubset(board)
    if direct_possible:
        direct_edges = {
            (min(left, right), max(left, right))
            for left in nodes
            for right in board[left]
            if right in nodes and left < right
        }
        if direct_edges == set(edges):
            mapping = {int(node): int(node) for node in nodes}
            kind = "real_board_induced_graph"
        else:
            mapping = _find_board_embedding(graph, induced=True)
            kind = "synthetic_topology_induced_board_isomorphism"
    else:
        mapping = _find_board_embedding(graph, induced=True)
        kind = "synthetic_topology_induced_board_isomorphism"
    induced = mapping is not None
    if mapping is None and allow_edge_preserving_non_induced_fallback:
        mapping = _find_board_embedding(graph, induced=False)
        kind = "synthetic_topology_edge_preserving_non_induced_display_mapping"
    if mapping is None:
        return {
            "status": "unavailable_no_board_embedding",
            "mapping_kind": None,
            "node_to_board_node": {},
            "node_to_name": {},
            "induced": False,
            "all_benchmark_edges_authoritative_adjacencies": False,
            "extra_board_edges": (),
        }
    inverse = {board_node: graph_node for graph_node, board_node in mapping.items()}
    mapped_board_nodes = set(mapping.values())
    extra_edges = []
    pattern_edges = set(edges)
    for left_board in sorted(mapped_board_nodes):
        for right_board in sorted(board[left_board]):
            if right_board not in mapped_board_nodes or left_board >= right_board:
                continue
            graph_edge = tuple(sorted((inverse[left_board], inverse[right_board])))
            if graph_edge not in pattern_edges:
                extra_edges.append(graph_edge)
    all_edges_valid = all(mapping[right] in board[mapping[left]] for left, right in edges)
    return {
        "status": "complete",
        "mapping_kind": kind,
        "node_to_board_node": dict(sorted(mapping.items())),
        "node_to_name": {node: territory_name_for_node(board_node) for node, board_node in sorted(mapping.items())},
        "induced": bool(induced),
        "all_benchmark_edges_authoritative_adjacencies": bool(all_edges_valid),
        "extra_board_edges": tuple(sorted(extra_edges)),
    }


def _pairwise_distribution_metrics(
    *,
    reference_distribution: Mapping[Any, float],
    candidate_distribution: Mapping[Any, float],
    graph: Any,
    initial_state: Any,
    partition_signature: Sequence[Sequence[int]],
    state_aware_max_support_size: Optional[int],
) -> Dict[str, Any]:
    nodes, edges = rcv._graph_nodes_edges(graph)
    descriptors = rcv.describe_cross_region_interaction_structure(
        graph=graph,
        initial_state=initial_state,
        partition_signature=partition_signature,
    )
    general = dcm.compare_distributions(reference_distribution, candidate_distribution)
    state_aware = {
        name: asdict(
            dcm.risk_state_wasserstein_distance(
                reference_distribution,
                candidate_distribution,
                state_distance_config=profile,
                max_support_size=state_aware_max_support_size,
            )
        )
        for name, profile in dcm.STATE_DISTANCE_PROFILES.items()
    }
    node = dcm.compare_node_marginals(
        reference_distribution,
        candidate_distribution,
        initial_state=initial_state,
        partition_boundary_nodes=descriptors.get("partition_boundary_nodes", ()),
        articulation_nodes=descriptors.get("articulation_nodes", ()),
    )
    centrality = rcv._betweenness_centrality(nodes, edges)
    maximum_centrality = max(centrality.values(), default=0.0)
    key_nodes = tuple(node_id for node_id, value in centrality.items() if value == maximum_centrality)
    events = dcm.compare_strategic_events(
        reference_distribution,
        candidate_distribution,
        initial_state=initial_state,
        edges=edges,
        at_least_k_values=tuple(range(1, min(3, len(nodes)) + 1)),
        articulation_nodes=descriptors.get("articulation_nodes", ()),
        key_territories=key_nodes,
    )
    summaries = dcm.compare_strategic_summaries(
        reference_distribution,
        candidate_distribution,
        initial_state=initial_state,
        edges=edges,
    )
    dependence = (
        dcm.compare_cross_region_dependence(
            reference_distribution,
            candidate_distribution,
            initial_state=initial_state,
            regions=partition_signature,
        )
        if len(partition_signature) >= 2
        else {
            "status": "insufficient_regions",
            "maximum_absolute_covariance_error": 0.0,
            "maximum_absolute_joint_success_error": 0.0,
        }
    )
    return {
        **general,
        "state_aware": state_aware,
        "node_marginals": node,
        "strategic_events": events,
        "strategic_summaries": summaries,
        "cross_region_dependence": dependence,
        "interaction_descriptors": descriptors,
    }


def _metric_value(metrics: Mapping[str, Any], name: str) -> Optional[float]:
    if name == "balanced_wasserstein":
        value = metrics.get("state_aware", {}).get("balanced", {}).get("distance")
    elif name == "ownership_error":
        value = metrics.get("node_marginals", {}).get("mean_absolute_ownership_probability_error")
    elif name == "troop_error":
        value = metrics.get("node_marginals", {}).get("mean_absolute_expected_troop_error")
    elif name == "conquest_error":
        value = metrics.get("strategic_events", {}).get("complete_conquest", {}).get("absolute_error")
    elif name == "no_gain_error":
        value = metrics.get("strategic_events", {}).get("no_territory_gained", {}).get("absolute_error")
    elif name == "covariance_error":
        value = metrics.get("cross_region_dependence", {}).get("maximum_absolute_covariance_error")
    else:
        value = metrics.get(name)
    return None if value is None else float(value)


def classify_error_sources(
    *,
    topology_family: str,
    mc1_candidate_identity: Any,
    exact_candidate_identity: Any,
    mc1_vs_full: Mapping[str, Any],
    exact_regional_vs_full: Mapping[str, Any],
    mc1_vs_exact_regional: Mapping[str, Any],
    thresholds: ErrorClassificationThresholds = ErrorClassificationThresholds(),
) -> Tuple[str, ...]:
    labels: List[str] = []
    mc_tv = float(mc1_vs_exact_regional.get("total_variation", 0.0))
    structural_tv = float(exact_regional_vs_full.get("total_variation", 0.0))
    total_tv = float(mc1_vs_full.get("total_variation", 0.0))
    if mc_tv >= thresholds.candidate_selection_tv and structural_tv < thresholds.structural_tv:
        labels.append("mc_candidate_selection_error")
    if structural_tv >= thresholds.structural_tv and mc_tv < thresholds.candidate_selection_tv:
        labels.append("structural_regional_error")
    if structural_tv >= thresholds.structural_tv and mc_tv >= thresholds.candidate_selection_tv:
        labels.append("both_candidate_and_structural_error")
    if structural_tv <= thresholds.near_exact_tv:
        labels.append("near_exact_after_exact_selection")
    if str(topology_family) == "double_front" and structural_tv >= thresholds.severe_tv:
        labels.append("double_front_structural_failure")
    wasserstein = _metric_value(exact_regional_vs_full, "balanced_wasserstein")
    if structural_tv >= thresholds.structural_tv and wasserstein is not None and wasserstein <= thresholds.low_wasserstein:
        labels.append("high_tv_low_wasserstein")
    event_errors = [
        float(row.get("absolute_error", 0.0))
        for row in exact_regional_vs_full.get("strategic_events", {}).values()
        if isinstance(row, Mapping)
    ]
    if max(event_errors, default=0.0) >= thresholds.high_strategic_event_error:
        labels.append("high_strategic_event_error")
    node_metrics = exact_regional_vs_full.get("node_marginals", {})
    largest = node_metrics.get("largest_error_node")
    largest_row = node_metrics.get("per_node", {}).get(largest, {})
    ownership = float(node_metrics.get("maximum_ownership_probability_error", 0.0))
    troop = float(node_metrics.get("maximum_expected_troop_error", 0.0))
    if largest_row.get("is_partition_boundary") and ownership >= thresholds.high_ownership_error:
        labels.append("ownership_boundary_error")
    if ownership <= thresholds.low_ownership_error and troop >= thresholds.high_troop_error:
        labels.append("troop_only_error")
    identities_equal = mc1_candidate_identity == exact_candidate_identity
    if not identities_equal and mc_tv <= thresholds.near_exact_tv:
        labels.append("candidate_changed_distribution_similar")
    if identities_equal and mc_tv > thresholds.distribution_identity_tolerance:
        labels.append("candidate_same_distribution_different")
    if not labels and total_tv <= thresholds.near_exact_tv:
        labels.append("near_exact_after_exact_selection")
    return tuple(labels)


class V2ValidationStore:
    LAYOUT = (
        "benchmark_records",
        "candidate_evaluations",
        "distributions",
        "comparison_tables",
        "casebook",
        "tractability_expansion/raw_results",
        "reports",
        "failures",
        "checkpoints",
    )

    def __init__(self, output_dir: Path | str, *, config: Mapping[str, Any], resume: bool) -> None:
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in self.LAYOUT:
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.fingerprint = _stable_digest(
            {"format": V2_FORMAT_VERSION, "schema": V2_SCHEMA_VERSION, "config": self.config}
        )
        manifest_path = self.root / "manifest.json"
        if resume and manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self.manifest.get("config_fingerprint") != self.fingerprint:
                raise ValueError("V2 validation resume configuration mismatch")
        else:
            self.manifest = {
                "format_version": V2_FORMAT_VERSION,
                "schema_version": V2_SCHEMA_VERSION,
                "config_fingerprint": self.fingerprint,
                "completed_benchmark_ids": [],
                "failed_benchmark_ids": [],
                "layout": list(self.LAYOUT),
            }
            _atomic_json(self.root / "config.json", self.config)
            self._write_manifest()

    def _write_manifest(self) -> None:
        self.manifest["updated_at"] = rcv._utc_now()
        _atomic_json(self.root / "manifest.json", self.manifest)

    @property
    def completed_ids(self) -> set[str]:
        return {str(value) for value in self.manifest.get("completed_benchmark_ids", ())}

    def save_completed(
        self,
        *,
        record: RegionalCompoundingValidationV2Record,
        selection: ExactRegionalCandidateSelectionResult,
    ) -> None:
        benchmark_id = record.benchmark_id
        _atomic_pickle(self.root / "benchmark_records" / f"{benchmark_id}.pkl", record)
        _atomic_pickle(self.root / "candidate_evaluations" / f"{benchmark_id}.pkl", selection)
        _atomic_pickle(
            self.root / "distributions" / f"{benchmark_id}.pkl",
            {
                "full_exact": record.full_exact_distribution,
                "regional_mc1": record.regional_mc1_distribution,
                "regional_exact": record.regional_exact_distribution,
            },
        )
        completed = self.completed_ids
        completed.add(benchmark_id)
        self.manifest["completed_benchmark_ids"] = sorted(completed)
        failed = {str(value) for value in self.manifest.get("failed_benchmark_ids", ())}
        failed.discard(benchmark_id)
        self.manifest["failed_benchmark_ids"] = sorted(failed)
        self._write_manifest()
        failure_path = self.root / "failures" / f"{benchmark_id}.json"
        if failure_path.exists():
            failure_path.unlink()
        _atomic_json(
            self.root / "checkpoints" / "completed_benchmarks.json",
            {"completed_benchmark_ids": sorted(completed), "updated_at": rcv._utc_now()},
        )

    def save_failure(self, benchmark_id: str, exc: BaseException) -> None:
        _atomic_json(
            self.root / "failures" / f"{benchmark_id}.json",
            {
                "benchmark_id": benchmark_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recorded_at": rcv._utc_now(),
            },
        )
        failed = {str(value) for value in self.manifest.get("failed_benchmark_ids", ())}
        failed.add(str(benchmark_id))
        self.manifest["failed_benchmark_ids"] = sorted(failed)
        self._write_manifest()


def load_v2_records(output_dir: Path | str) -> Tuple[RegionalCompoundingValidationV2Record, ...]:
    records: List[RegionalCompoundingValidationV2Record] = []
    for path in sorted((Path(output_dir) / "benchmark_records").glob("*.pkl")):
        with path.open("rb") as handle:
            value = pickle.load(handle)
        records.append(
            value
            if isinstance(value, RegionalCompoundingValidationV2Record)
            else RegionalCompoundingValidationV2Record(**dict(value))
        )
    return tuple(records)


def _complexity_summary(selection: ExactRegionalCandidateSelectionResult) -> Dict[str, Any]:
    rows = []
    for item in selection.candidate_evaluations:
        composition = dict(item.diagnostics.get("composition", {}))
        rows.append(
            {
                "candidate_identity": item.candidate_identity,
                "number_of_regions": int(composition.get("number_of_regions", len(item.partition_signature or ()))),
                "regional_support_sizes": tuple(composition.get("regional_support_sizes", ())),
                "raw_cartesian_product_size": item.raw_cartesian_product_size,
                "actual_raw_expansions": item.raw_cartesian_expansions,
                "unique_states_after_each_region": item.unique_states_after_each_region,
                "duplicates_merged_after_each_region": tuple(
                    composition.get("duplicates_merged_after_each_region", ())
                ),
                "final_unique_support_size": item.successor_support_size,
                "compression_ratio": composition.get("compression_ratio"),
                "regional_distribution_preparation_seconds": composition.get(
                    "regional_distribution_preparation_seconds", 0.0
                ),
                "outcome_application_seconds": composition.get("outcome_application_seconds", 0.0),
                "hashing_merging_seconds": composition.get("hashing_merging_seconds", 0.0),
                "normalization_seconds": composition.get("normalization_seconds", 0.0),
                "composition_runtime_seconds": item.composition_runtime_seconds,
            }
        )
    products = [int(row["raw_cartesian_product_size"]) for row in rows if row["raw_cartesian_product_size"] is not None]
    supports = [int(row["final_unique_support_size"]) for row in rows if row["final_unique_support_size"] is not None]
    return {
        "candidate_count": len(rows),
        "candidate_complexities": tuple(rows),
        "maximum_raw_cartesian_product_size": max(products, default=0),
        "maximum_final_unique_support_size": max(supports, default=0),
        "total_actual_raw_expansions": sum(int(row["actual_raw_expansions"]) for row in rows),
    }


def _candidate_for_identity(
    candidates: Sequence[bgr.PartitionPolicyCandidate], identity: Any
) -> Optional[bgr.PartitionPolicyCandidate]:
    return next((candidate for candidate in candidates if _candidate_identity(candidate) == identity), None)


def evaluate_cached_v1_case_for_v2(
    *,
    case: rcv.BenchmarkCase,
    full_exact: rcv.ExactFullGraphReferenceResult,
    regional_mc1: rcv.RegionalApproximationResult,
    v1_record: rcv.RegionalCompoundingValidationRecord,
    library_dir: Path | str,
    composition_limits: Optional[Mapping[str, Any]] = None,
    state_aware_max_support_size: Optional[int] = 300,
    thresholds: ErrorClassificationThresholds = ErrorClassificationThresholds(),
) -> Tuple[RegionalCompoundingValidationV2Record, ExactRegionalCandidateSelectionResult]:
    started = time.perf_counter()
    graph = case.graph_spec.graph()
    candidates = tuple(regional_mc1.retained_candidates)
    region_query_cache = agop.RegionQueryResultCache(
        max_entries=None,
        profile_timings=False,
        cache_library_resources=True,
    )
    with rcv._temporary_board_state(case.initial_state, graph.nodes()) as (players, base_global_state):
        def global_evaluator(state: GlobalState) -> Tuple[float, float, float]:
            ranked = bgr.rank_battle_graph_partitions(
                players=players,
                battle_graph=graph,
                combat_libraries_base=Path(library_dir),
                max_partitions=40,
                ranking_variable="battle_expected_attacker_territory_count",
                lookahead_depth=0,
                use_monte_carlo=False,
                policy_option_selection="primary",
                global_state_override=state,
                region_query_cache=region_query_cache,
            )
            return bgr._partition_eval_utility_tuple(ranked.get("best_evaluation"))

        exact_selection = evaluate_regional_candidates_exactly(
            prepared_candidates=candidates,
            initial_global_state=base_global_state,
            ranking_variable="battle_expected_attacker_territory_count",
            evaluation_mode="production_second_stage",
            rollout_steps=0,
            global_evaluator=global_evaluator,
            composition_limits=composition_limits,
            node_indices=tuple(int(node) for node in graph.nodes()),
        )
    exact_selection = replace(
        exact_selection,
        diagnostics={
            **dict(exact_selection.diagnostics),
            "region_query_cache": region_query_cache.diagnostics(),
            "source_candidate_count": len(candidates),
            "source_candidate_set_is_v1_corrected_retained_set": True,
        },
    )

    p_full = dict(full_exact.canonical_optimal_distribution or {})
    p_mc1 = dict(regional_mc1.exact_compounded_distribution or {})
    p_regional_exact = dict(exact_selection.selected_successor_distribution or {})
    mc1_identity = (
        _candidate_identity(regional_mc1.selected_candidate)
        if regional_mc1.selected_candidate is not None
        else None
    )
    exact_identity = exact_selection.canonical_selected_candidate_identity
    exact_candidate = _candidate_for_identity(candidates, exact_identity)
    exact_partition = tuple(exact_selection.selected_partition_signature or ())
    mc1_partition = tuple(regional_mc1.selected_partition_signature or ())
    complete = bool(
        full_exact.status == "exact_complete"
        and regional_mc1.composition_status == "exact_complete"
        and exact_selection.status == "exact_complete"
    )
    if complete:
        mc1_vs_full = _pairwise_distribution_metrics(
            reference_distribution=p_full,
            candidate_distribution=p_mc1,
            graph=graph,
            initial_state=case.initial_state,
            partition_signature=mc1_partition,
            state_aware_max_support_size=state_aware_max_support_size,
        )
        exact_vs_full = _pairwise_distribution_metrics(
            reference_distribution=p_full,
            candidate_distribution=p_regional_exact,
            graph=graph,
            initial_state=case.initial_state,
            partition_signature=exact_partition,
            state_aware_max_support_size=state_aware_max_support_size,
        )
        mc1_vs_exact = _pairwise_distribution_metrics(
            reference_distribution=p_regional_exact,
            candidate_distribution=p_mc1,
            graph=graph,
            initial_state=case.initial_state,
            partition_signature=exact_partition,
            state_aware_max_support_size=state_aware_max_support_size,
        )
        labels = classify_error_sources(
            topology_family=case.graph_spec.topology_family,
            mc1_candidate_identity=mc1_identity,
            exact_candidate_identity=exact_identity,
            mc1_vs_full=mc1_vs_full,
            exact_regional_vs_full=exact_vs_full,
            mc1_vs_exact_regional=mc1_vs_exact,
            thresholds=thresholds,
        )
    else:
        mc1_vs_full = {}
        exact_vs_full = {}
        mc1_vs_exact = {}
        labels = ()

    candidate_agreement = mc1_identity == exact_identity
    partition_agreement = mc1_partition == exact_partition
    policy_option_agreement = (
        tuple(regional_mc1.selected_policy_option_indices)
        == tuple(exact_selection.selected_policy_option_indices or ())
    )
    old_tv = v1_record.distribution_metrics.get("total_variation")
    old_js = v1_record.distribution_metrics.get("jensen_shannon")
    reproduced_tv = mc1_vs_full.get("total_variation")
    reproduced_js = mc1_vs_full.get("jensen_shannon")
    reproduction = {
        "v1_total_variation": old_tv,
        "v2_recomputed_total_variation": reproduced_tv,
        "total_variation_absolute_delta": (
            abs(float(old_tv) - float(reproduced_tv))
            if old_tv is not None and reproduced_tv is not None
            else None
        ),
        "v1_jensen_shannon": old_js,
        "v2_recomputed_jensen_shannon": reproduced_js,
        "jensen_shannon_absolute_delta": (
            abs(float(old_js) - float(reproduced_js))
            if old_js is not None and reproduced_js is not None
            else None
        ),
    }
    territory = authoritative_territory_mapping_for_graph(
        graph,
        allow_edge_preserving_non_induced_fallback=True,
    )
    complexity = _complexity_summary(exact_selection)
    exact_regional_value = (
        rcv.distribution_implied_value(p_regional_exact, initial_state=case.initial_state)
        if p_regional_exact
        else None
    )
    mc1_value = (
        rcv.distribution_implied_value(p_mc1, initial_state=case.initial_state)
        if p_mc1
        else None
    )
    candidate_effect = {
        "candidate_identity_agreement": candidate_agreement,
        "partition_agreement": partition_agreement,
        "policy_option_agreement": policy_option_agreement,
        "mc1_selected_candidate_identity": mc1_identity,
        "exact_regional_selected_candidate_identity": exact_identity,
        "mc1_vs_exact_regional_metrics": mc1_vs_exact,
    }
    structural_effect = {
        "exact_regional_vs_full_metrics": exact_vs_full,
        "full_exact_optimal_value": full_exact.optimal_value,
        "exact_regional_distribution_implied_value": exact_regional_value,
        "distribution_value_gap": rcv.compare_policy_values(
            full_exact.optimal_value, exact_regional_value
        ),
        "distribution_value_gap_is_policy_regret": False,
    }
    record = RegionalCompoundingValidationV2Record(
        benchmark_id=case.benchmark_id,
        v1_benchmark_id=case.benchmark_id,
        graph_signature=(case.graph_spec.nodes, case.graph_spec.edges),
        topology_family=case.graph_spec.topology_family,
        initial_state_signature=case.initial_state_signature,
        territory_mapping=dict(territory.get("node_to_name", {})),
        full_exact_status=full_exact.status,
        regional_mc1_status=regional_mc1.composition_status,
        regional_exact_status=exact_selection.status,
        full_exact_distribution=p_full or None,
        regional_mc1_distribution=p_mc1 or None,
        regional_exact_distribution=p_regional_exact or None,
        regional_mc1_candidate_identity=mc1_identity,
        regional_exact_candidate_identity=exact_identity,
        mc1_vs_full_metrics=mc1_vs_full,
        exact_regional_vs_full_metrics=exact_vs_full,
        mc1_vs_exact_regional_metrics=mc1_vs_exact,
        candidate_selection_effect=candidate_effect,
        structural_approximation_effect=structural_effect,
        exact_candidate_selection_diagnostics=dict(exact_selection.diagnostics),
        composition_complexity_diagnostics=complexity,
        diagnostic_labels=labels,
        runtimes={
            "full_exact_seconds": full_exact.runtime_seconds,
            "regional_mc1_seconds": regional_mc1.runtime_seconds,
            "exact_regional_candidate_selection_seconds": exact_selection.runtime_seconds,
            "v2_total_seconds": float(time.perf_counter() - started),
        },
        diagnostics={
            "v1_reproduction": reproduction,
            "territory_mapping": territory,
            "thresholds": asdict(thresholds),
            "troop_cap": int(case.troop_cap),
            "state_stratum": case.state_stratum,
            "graph_size": case.graph_spec.node_count,
            "attacker_node_count": case.graph_spec.attacker_count,
            "defender_node_count": case.graph_spec.node_count - case.graph_spec.attacker_count,
            "retained_candidate_count": len(candidates),
            "mc1_region_count": len(mc1_partition),
            "exact_regional_region_count": len(exact_partition),
            "regional_mc1_distribution_implied_value": mc1_value,
            "full_exact_policy_count": len(full_exact.optimal_policy_set),
            "full_exact_canonical_policy": full_exact.canonical_optimal_policy,
            "full_exact_policy_trace": full_exact.diagnostics.get("policy_trace", {}),
            "regional_exact_root_actions": tuple(
                ref.root_action for ref in (exact_candidate.region_policy_options if exact_candidate else ())
            ),
            "production_semantics_unchanged": True,
            "true_policy_regret_available": False,
        },
    )
    return record, exact_selection


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_v2_benchmark_from_v1(
    *,
    v1_output_dir: Path | str,
    output_dir: Path | str,
    library_dir: Path | str,
    benchmark_ids: Optional[Sequence[str]] = None,
    composition_limits: Optional[Mapping[str, Any]] = None,
    state_aware_max_support_size: Optional[int] = 300,
    resume: bool = False,
) -> Dict[str, Any]:
    v1_root = Path(v1_output_dir)
    config = {
        "v1_output_dir": str(v1_root),
        "library_dir": str(Path(library_dir)),
        "composition_limits": dict(composition_limits or {}),
        "state_aware_max_support_size": state_aware_max_support_size,
        "candidate_source": "stored_v1_corrected_retained_candidates",
        "candidate_evaluation": "exact_regional_model",
        "global_evaluation": "production_second_stage_zero_step",
        "score_comparison": "strict_lexicographic_no_tolerance",
    }
    store = V2ValidationStore(output_dir, config=config, resume=resume)
    allowed = None if benchmark_ids is None else {str(value) for value in benchmark_ids}
    state_paths = tuple(
        path
        for path in sorted((v1_root / "states").glob("*.pkl"))
        if allowed is None or path.stem in allowed
    )
    completed_now = 0
    skipped = 0
    failures = 0
    for state_path in state_paths:
        benchmark_id = state_path.stem
        if benchmark_id in store.completed_ids:
            skipped += 1
            continue
        try:
            case = _load_pickle(state_path)
            full_exact = _load_pickle(v1_root / "exact_results" / state_path.name)
            regional_mc1 = _load_pickle(v1_root / "approximation_results" / state_path.name)
            v1_record = _load_pickle(v1_root / "comparison_records" / state_path.name)
            record, selection = evaluate_cached_v1_case_for_v2(
                case=case,
                full_exact=full_exact,
                regional_mc1=regional_mc1,
                v1_record=v1_record,
                library_dir=library_dir,
                composition_limits=composition_limits,
                state_aware_max_support_size=state_aware_max_support_size,
            )
            store.save_completed(record=record, selection=selection)
            completed_now += 1
        except Exception as exc:
            store.save_failure(benchmark_id, exc)
            failures += 1
    return {
        "requested": len(state_paths),
        "completed_now": completed_now,
        "skipped_existing": skipped,
        "failures": failures,
        "completed_total": len(store.completed_ids),
        "output_dir": str(Path(output_dir)),
    }


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


def _flat_pair_metrics(metrics: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}_tv": _metric_value(metrics, "total_variation"),
        f"{prefix}_js": _metric_value(metrics, "jensen_shannon"),
        f"{prefix}_wasserstein_balanced": _metric_value(metrics, "balanced_wasserstein"),
        f"{prefix}_ownership_error": _metric_value(metrics, "ownership_error"),
        f"{prefix}_troop_error": _metric_value(metrics, "troop_error"),
        f"{prefix}_conquest_error": _metric_value(metrics, "conquest_error"),
        f"{prefix}_no_gain_error": _metric_value(metrics, "no_gain_error"),
        f"{prefix}_covariance_error": _metric_value(metrics, "covariance_error"),
    }


def _flat_v2_record(record: RegionalCompoundingValidationV2Record) -> Dict[str, Any]:
    return {
        "benchmark_id": record.benchmark_id,
        "topology_family": record.topology_family,
        "graph_size": record.diagnostics.get("graph_size"),
        "troop_cap": record.diagnostics.get("troop_cap"),
        "retained_candidate_count": record.diagnostics.get("retained_candidate_count"),
        "mc1_region_count": record.diagnostics.get("mc1_region_count"),
        "exact_regional_region_count": record.diagnostics.get("exact_regional_region_count"),
        "candidate_identity_agreement": record.candidate_selection_effect.get("candidate_identity_agreement"),
        "partition_agreement": record.candidate_selection_effect.get("partition_agreement"),
        "policy_option_agreement": record.candidate_selection_effect.get("policy_option_agreement"),
        "regional_exact_status": record.regional_exact_status,
        "diagnostic_labels": record.diagnostic_labels,
        "exact_selection_runtime_seconds": record.runtimes.get("exact_regional_candidate_selection_seconds"),
        "unique_successor_states": record.exact_candidate_selection_diagnostics.get("unique_successor_states"),
        "global_evaluation_cache_hits": record.exact_candidate_selection_diagnostics.get("global_evaluation_cache_hits"),
        "global_evaluation_cache_misses": record.exact_candidate_selection_diagnostics.get("global_evaluation_cache_misses"),
        "v1_tv_reproduction_delta": record.diagnostics.get("v1_reproduction", {}).get("total_variation_absolute_delta"),
        **_flat_pair_metrics(record.mc1_vs_full_metrics, "mc1_full"),
        **_flat_pair_metrics(record.exact_regional_vs_full_metrics, "regional_exact_full"),
        **_flat_pair_metrics(record.mc1_vs_exact_regional_metrics, "mc1_regional_exact"),
    }


def _bucket(value: int, boundaries: Sequence[int]) -> str:
    lower = 0
    for upper in boundaries:
        if value <= int(upper):
            return f"{lower}-{int(upper)}"
        lower = int(upper) + 1
    return f"{lower}+"


def summarize_v2_benchmark(output_dir: Path | str) -> Dict[str, Any]:
    root = Path(output_dir)
    records = load_v2_records(root)
    rows = [_flat_v2_record(record) for record in records]
    pair_prefixes = ("mc1_full", "regional_exact_full", "mc1_regional_exact")
    metric_suffixes = (
        "tv",
        "js",
        "wasserstein_balanced",
        "ownership_error",
        "troop_error",
        "conquest_error",
        "no_gain_error",
        "covariance_error",
    )
    aggregate = {
        prefix: {
            suffix: _numeric_summary(row.get(f"{prefix}_{suffix}") for row in rows)
            for suffix in metric_suffixes
        }
        for prefix in pair_prefixes
    }
    by_topology = {}
    for family in sorted({record.topology_family for record in records}):
        family_rows = [row for row in rows if row["topology_family"] == family]
        by_topology[family] = {
            "records": len(family_rows),
            **{
                prefix: {
                    suffix: _numeric_summary(row.get(f"{prefix}_{suffix}") for row in family_rows)
                    for suffix in metric_suffixes
                }
                for prefix in pair_prefixes
            },
        }
    complexities = [
        item
        for record in records
        for item in record.composition_complexity_diagnostics.get("candidate_complexities", ())
    ]
    raw_products = [int(item["raw_cartesian_product_size"]) for item in complexities if item.get("raw_cartesian_product_size") is not None]
    final_supports = [int(item["final_unique_support_size"]) for item in complexities if item.get("final_unique_support_size") is not None]
    compression = [float(item["compression_ratio"]) for item in complexities if item.get("compression_ratio") is not None]
    composition_runtimes = [float(item.get("composition_runtime_seconds", 0.0)) for item in complexities]
    runtime_by_region_count = {}
    for region_count in sorted({int(item.get("number_of_regions", 0)) for item in complexities}):
        subset = [item for item in complexities if int(item.get("number_of_regions", 0)) == region_count]
        runtime_by_region_count[str(region_count)] = {
            "candidates": len(subset),
            "runtime_seconds": _numeric_summary(item.get("composition_runtime_seconds") for item in subset),
            "raw_product": _numeric_summary(item.get("raw_cartesian_product_size") for item in subset),
            "final_support": _numeric_summary(item.get("final_unique_support_size") for item in subset),
        }
    runtime_by_raw_bucket = {}
    for bucket in sorted({_bucket(int(item.get("raw_cartesian_product_size") or 0), (1, 4, 16, 64, 256, 1024)) for item in complexities}):
        subset = [
            item
            for item in complexities
            if _bucket(int(item.get("raw_cartesian_product_size") or 0), (1, 4, 16, 64, 256, 1024)) == bucket
        ]
        runtime_by_raw_bucket[bucket] = _numeric_summary(item.get("composition_runtime_seconds") for item in subset)
    runtime_by_support_bucket = {}
    for bucket in sorted({_bucket(int(item.get("final_unique_support_size") or 0), (1, 4, 16, 64, 256)) for item in complexities}):
        subset = [
            item
            for item in complexities
            if _bucket(int(item.get("final_unique_support_size") or 0), (1, 4, 16, 64, 256)) == bucket
        ]
        runtime_by_support_bucket[bucket] = _numeric_summary(item.get("composition_runtime_seconds") for item in subset)
    labels = sorted({label for record in records for label in record.diagnostic_labels})
    previous_tv1 = [record for record in records if float(record.mc1_vs_full_metrics.get("total_variation", 0.0)) >= 1.0 - 1e-12]
    double_front = [record for record in records if record.topology_family == "double_front"]
    candidate_changed = [
        record
        for record in records
        if not bool(record.candidate_selection_effect.get("candidate_identity_agreement"))
    ]
    structural_tv_deltas = [
        float(record.exact_regional_vs_full_metrics.get("total_variation", 0.0))
        - float(record.mc1_vs_full_metrics.get("total_variation", 0.0))
        for record in records
    ]
    selection_results = []
    for record in records:
        path = root / "candidate_evaluations" / f"{record.benchmark_id}.pkl"
        if path.exists():
            selection_results.append(_load_pickle(path))
    severe_double_front = [
        record
        for record in double_front
        if float(record.exact_regional_vs_full_metrics.get("total_variation", 0.0)) >= 0.9
    ]

    def descriptor_count(name: str) -> int:
        return sum(
            bool(record.exact_regional_vs_full_metrics.get("interaction_descriptors", {}).get(name))
            for record in severe_double_front
        )

    summary = {
        "format_version": V2_FORMAT_VERSION,
        "schema_version": V2_SCHEMA_VERSION,
        "generated_at": rcv._utc_now(),
        "record_count": len(records),
        "status_histogram": {
            status: sum(record.regional_exact_status == status for record in records)
            for status in sorted({record.regional_exact_status for record in records})
        },
        "candidate_agreement": {
            "identity_agreement_count": sum(bool(record.candidate_selection_effect.get("candidate_identity_agreement")) for record in records),
            "partition_agreement_count": sum(bool(record.candidate_selection_effect.get("partition_agreement")) for record in records),
            "policy_option_agreement_count": sum(bool(record.candidate_selection_effect.get("policy_option_agreement")) for record in records),
            "records": len(records),
            "identity_changed_count": len(candidate_changed),
            "changed_but_distribution_equal_count": sum(
                float(record.mc1_vs_exact_regional_metrics.get("total_variation", 0.0)) <= 1e-10
                for record in candidate_changed
            ),
            "material_distribution_change_tv_ge_0_05_count": sum(
                float(record.mc1_vs_exact_regional_metrics.get("total_variation", 0.0)) >= 0.05
                for record in records
            ),
            "exact_tie_state_count": sum(
                len(selection.exact_best_candidate_identities) > 1
                for selection in selection_results
                if selection.status == "exact_complete"
            ),
            "maximum_exact_tie_count": max(
                (len(selection.exact_best_candidate_identities) for selection in selection_results),
                default=0,
            ),
        },
        "structural_tv_change_from_mc1_selection": {
            "delta_exact_minus_mc1": _numeric_summary(structural_tv_deltas),
            "improved_count": sum(value < -1e-10 for value in structural_tv_deltas),
            "unchanged_count": sum(abs(value) <= 1e-10 for value in structural_tv_deltas),
            "worsened_count": sum(value > 1e-10 for value in structural_tv_deltas),
        },
        "pairwise_metrics": aggregate,
        "by_topology": by_topology,
        "diagnostic_label_histogram": {label: sum(label in record.diagnostic_labels for record in records) for label in labels},
        "v1_reproduction": {
            "tv_delta": _numeric_summary(row.get("v1_tv_reproduction_delta") for row in rows),
            "all_within_1e_10": all(
                row.get("v1_tv_reproduction_delta") is not None
                and float(row["v1_tv_reproduction_delta"]) <= 1e-10
                for row in rows
            ) if rows else True,
        },
        "previous_tv1_failures": {
            "count": len(previous_tv1),
            "remain_tv1_after_exact_selection": sum(
                float(record.exact_regional_vs_full_metrics.get("total_variation", 0.0)) >= 1.0 - 1e-12
                for record in previous_tv1
            ),
            "remain_severe_tv_ge_0_9": sum(
                float(record.exact_regional_vs_full_metrics.get("total_variation", 0.0)) >= 0.9
                for record in previous_tv1
            ),
        },
        "double_front": {
            "records": len(double_front),
            "candidate_changed": sum(
                not bool(record.candidate_selection_effect.get("candidate_identity_agreement"))
                for record in double_front
            ),
            "structural_tv": _numeric_summary(
                record.exact_regional_vs_full_metrics.get("total_variation") for record in double_front
            ),
            "mc1_tv": _numeric_summary(record.mc1_vs_full_metrics.get("total_variation") for record in double_front),
            "severe_after_exact_selection": len(severe_double_front),
            "severe_descriptor_counts": {
                "shared_troop_source_present": descriptor_count("shared_troop_source_present"),
                "sequence_opening_present": descriptor_count("sequence_opening_present"),
                "articulation_present": sum(
                    bool(record.exact_regional_vs_full_metrics.get("interaction_descriptors", {}).get("articulation_nodes"))
                    for record in severe_double_front
                ),
            },
        },
        "exact_candidate_selection_runtime_seconds": _numeric_summary(
            record.runtimes.get("exact_regional_candidate_selection_seconds") for record in records
        ),
        "global_state_reuse": {
            "raw_successor_state_requests": sum(
                int(record.exact_candidate_selection_diagnostics.get("raw_successor_state_requests", 0))
                for record in records
            ),
            "unique_successor_states": sum(
                int(record.exact_candidate_selection_diagnostics.get("unique_successor_states", 0))
                for record in records
            ),
            "cache_hits": sum(
                int(record.exact_candidate_selection_diagnostics.get("global_evaluation_cache_hits", 0))
                for record in records
            ),
            "cache_misses": sum(
                int(record.exact_candidate_selection_diagnostics.get("global_evaluation_cache_misses", 0))
                for record in records
            ),
        },
        "composition_complexity": {
            "candidate_evaluations": len(complexities),
            "raw_cartesian_product": _numeric_summary(raw_products),
            "final_unique_support": _numeric_summary(final_supports),
            "compression_ratio": _numeric_summary(compression),
            "composition_runtime_seconds": _numeric_summary(composition_runtimes),
            "runtime_by_region_count": runtime_by_region_count,
            "runtime_by_raw_product_bucket": runtime_by_raw_bucket,
            "runtime_by_final_support_bucket": runtime_by_support_bucket,
            "regional_support_size": _numeric_summary(
                support
                for item in complexities
                for support in item.get("regional_support_sizes", ())
            ),
            "total_duplicate_states_merged": sum(
                sum(int(value) for value in item.get("duplicates_merged_after_each_region", ()))
                for item in complexities
            ),
            "candidates_with_duplicate_state_merging": sum(
                any(int(value) > 0 for value in item.get("duplicates_merged_after_each_region", ()))
                for item in complexities
            ),
        },
        "policy_regret": {
            "available": False,
            "reason": "regional V2 candidates do not encode a complete contingent full-graph policy",
            "distribution_value_gap_is_not_policy_regret": True,
        },
    }
    _atomic_json(root / "reports" / "benchmark_summary.json", summary)
    write_benchmark_report(root, summary=summary)
    write_exact_composition_complexity_report(root, summary=summary)
    if rows:
        path = root / "comparison_tables" / "benchmark_records.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _json_ready(value) for key, value in row.items()})
        temporary.replace(path)
    return summary


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _report_number(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.{int(digits)}g}"


def write_benchmark_report(
    output_dir: Path | str,
    *,
    summary: Optional[Mapping[str, Any]] = None,
) -> Path:
    root = Path(output_dir)
    data = dict(summary or json.loads((root / "reports" / "benchmark_summary.json").read_text(encoding="utf-8")))
    agreement = data.get("candidate_agreement", {})
    tv_change = data.get("structural_tv_change_from_mc1_selection", {})
    pairwise = data.get("pairwise_metrics", {})
    reuse = data.get("global_state_reuse", {})
    previous = data.get("previous_tv1_failures", {})
    double_front = data.get("double_front", {})
    lines = [
        "# Regional Compounding Validation V2",
        "",
        "## Scope",
        "",
        f"- Completed benchmark records: `{data.get('record_count', 0)}`.",
        f"- V1 MC1 metric reproduction within 1e-10: `{data.get('v1_reproduction', {}).get('all_within_1e_10')}`.",
        "- Candidate source: the persisted corrected-mode retained candidate set; no tolerance or candidate/policy-combination cap was introduced.",
        "- Production routing and Stage A/B/C/D behavior were not modified.",
        "",
        "## Candidate Selection",
        "",
        f"- MC1 and exact regional candidate identity agreement: `{agreement.get('identity_agreement_count')}/{agreement.get('records')}`.",
        f"- Candidate identity changed: `{agreement.get('identity_changed_count')}`; changed but distribution-equal: `{agreement.get('changed_but_distribution_equal_count')}`.",
        f"- Material MC1-to-exact-regional distribution changes (TV >= 0.05): `{agreement.get('material_distribution_change_tv_ge_0_05_count')}`.",
        f"- States with exact best-candidate ties: `{agreement.get('exact_tie_state_count')}`; maximum tie count: `{agreement.get('maximum_exact_tie_count')}`.",
        f"- Full-reference TV improved / unchanged / worsened after exact regional selection: `{tv_change.get('improved_count')}` / `{tv_change.get('unchanged_count')}` / `{tv_change.get('worsened_count')}`.",
        f"- Mean exact-minus-MC1 full-reference TV delta: `{_report_number(tv_change.get('delta_exact_minus_mc1', {}).get('mean'))}`.",
        "",
        "## Error Decomposition",
        "",
        "| Comparison | Mean TV | Mean JS | Mean balanced Wasserstein | Mean ownership error | Mean troop error | Mean conquest error | Mean covariance error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = (
        ("MC1 regional vs full exact", "mc1_full"),
        ("Exact regional vs full exact", "regional_exact_full"),
        ("MC1 regional vs exact regional", "mc1_regional_exact"),
    )
    for label, key in labels:
        row = pairwise.get(key, {})
        lines.append(
            "| " + " | ".join(
                (
                    label,
                    _report_number(row.get("tv", {}).get("mean")),
                    _report_number(row.get("js", {}).get("mean")),
                    _report_number(row.get("wasserstein_balanced", {}).get("mean")),
                    _report_number(row.get("ownership_error", {}).get("mean")),
                    _report_number(row.get("troop_error", {}).get("mean")),
                    _report_number(row.get("conquest_error", {}).get("mean")),
                    _report_number(row.get("covariance_error", {}).get("mean")),
                )
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Double Front",
            "",
            f"- Double-front records: `{double_front.get('records')}`; candidate changed in `{double_front.get('candidate_changed')}`.",
            f"- Mean TV, MC1 vs full / exact regional vs full: `{_report_number(double_front.get('mc1_tv', {}).get('mean'))}` / `{_report_number(double_front.get('structural_tv', {}).get('mean'))}`.",
            f"- Previous TV=1 failures: `{previous.get('count')}`; still TV=1: `{previous.get('remain_tv1_after_exact_selection')}`; still TV >= 0.9: `{previous.get('remain_severe_tv_ge_0_9')}`.",
            f"- Severe descriptor counts: `{double_front.get('severe_descriptor_counts')}`.",
            "- Attack-order switching, conditional policy switching, and survivor redistribution are not inferred because a contingent exact policy DAG is unavailable.",
            "",
            "## Cache Reuse",
            "",
            f"- Raw successor-state requests: `{reuse.get('raw_successor_state_requests')}`.",
            f"- Unique successor states evaluated: `{reuse.get('unique_successor_states')}`.",
            f"- Global evaluation cache hits / misses: `{reuse.get('cache_hits')}` / `{reuse.get('cache_misses')}`.",
            "",
            "## Topology Means",
            "",
            "| Topology | Records | MC1-full TV | Exact-regional-full TV | MC1-exact-regional TV |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, topology in sorted(data.get("by_topology", {}).items()):
        lines.append(
            f"| {family} | {topology.get('records')} | "
            f"{_report_number(topology.get('mc1_full', {}).get('tv', {}).get('mean'))} | "
            f"{_report_number(topology.get('regional_exact_full', {}).get('tv', {}).get('mean'))} | "
            f"{_report_number(topology.get('mc1_regional_exact', {}).get('tv', {}).get('mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Exact regional candidate selection removes Monte Carlo candidate-selection noise but does not remove the regional decomposition or independence assumptions.",
            "",
            "The comparison between the exact regional-model distribution and the full-graph exact distribution isolates structural approximation error more cleanly than the previous MC=1 benchmark.",
            "",
            "A distribution-value gap is not called policy regret unless the regional candidate can be evaluated as a complete contingent policy under the exact full-graph dynamics.",
            "",
            "Concrete good, bad, and mixed examples use authoritative project territory names and preserve the actual graph adjacency.",
            "",
            "The mapping audit distinguishes induced embeddings from edge-preserving non-induced display mappings and lists every extra board edge for the latter.",
            "",
            "Production routing is not changed in this task; the expanded exact-solver results are used to recommend a later exact-first routing policy.",
            "",
            "Stage A regeneration, Stage B retraining, and Stage E remain blocked until these results have been reviewed.",
        ]
    )
    path = root / "reports" / "benchmark_report.md"
    _write_markdown(path, lines)
    return path


def write_exact_composition_complexity_report(
    output_dir: Path | str,
    *,
    summary: Optional[Mapping[str, Any]] = None,
) -> Path:
    root = Path(output_dir)
    data = dict(summary or json.loads((root / "reports" / "benchmark_summary.json").read_text(encoding="utf-8")))
    complexity = data.get("composition_complexity", {})
    runtime = complexity.get("composition_runtime_seconds", {})
    raw = complexity.get("raw_cartesian_product", {})
    final = complexity.get("final_unique_support", {})
    compression = complexity.get("compression_ratio", {})
    supports = complexity.get("regional_support_size", {})
    lines = [
        "# Exact Composition Complexity",
        "",
        f"- Candidate compositions: `{complexity.get('candidate_evaluations')}`.",
        f"- Regional support size median / maximum: `{_report_number(supports.get('median'))}` / `{_report_number(supports.get('maximum'))}`.",
        f"- Raw Cartesian product median / maximum: `{_report_number(raw.get('median'))}` / `{_report_number(raw.get('maximum'))}`.",
        f"- Final support median / maximum: `{_report_number(final.get('median'))}` / `{_report_number(final.get('maximum'))}`.",
        f"- Compression ratio median / maximum: `{_report_number(compression.get('median'))}` / `{_report_number(compression.get('maximum'))}`.",
        f"- Duplicate states merged: `{complexity.get('total_duplicate_states_merged')}` across `{complexity.get('candidates_with_duplicate_state_merging')}` candidates.",
        f"- Composition runtime mean / median / p90 / maximum seconds: `{_report_number(runtime.get('mean'))}` / `{_report_number(runtime.get('median'))}` / `{_report_number(runtime.get('p90'))}` / `{_report_number(runtime.get('maximum'))}`.",
        "",
        "## By Region Count",
        "",
        "| Regions | Candidates | Median raw product | Maximum raw product | Median final support | Median runtime seconds |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for region_count, row in sorted(complexity.get("runtime_by_region_count", {}).items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {region_count} | {row.get('candidates')} | "
            f"{_report_number(row.get('raw_product', {}).get('median'))} | "
            f"{_report_number(row.get('raw_product', {}).get('maximum'))} | "
            f"{_report_number(row.get('final_support', {}).get('median'))} | "
            f"{_report_number(row.get('runtime_seconds', {}).get('median'))} |"
        )
    lines.extend(
        [
            "",
            "## Measured Explanation",
            "",
            "Exact composition was cheap because candidates contained only one to three disjoint regions and their regional supports produced small Cartesian products. Applying and hashing labelled outcomes was correspondingly cheap.",
            "",
            "Duplicate-state merging did not explain the benchmark speed: no candidate merged duplicate assembled states, and final support equalled the theoretical raw product throughout this run.",
            "",
            "The expensive portion of exact regional candidate selection was production second-stage evaluation of unique complete successor states, not distribution composition.",
        ]
    )
    path = root / "reports" / "exact_composition_complexity.md"
    _write_markdown(path, lines)
    return path


def _name(mapping: Mapping[int, str], node: int) -> str:
    if int(node) not in mapping:
        raise KeyError(f"Casebook has no authoritative territory mapping for benchmark node {node}")
    return str(mapping[int(node)])


def _format_state_rows(signature: Any, mapping: Mapping[int, str]) -> str:
    rows = [
        f"{_name(mapping, node)}={owner}{troops}"
        for node, owner, troops in dcm.canonical_risk_state(signature)
    ]
    return ", ".join(rows)


def _top_states_markdown(
    distribution: Optional[Mapping[Any, float]], mapping: Mapping[int, str], *, limit: int = 5
) -> List[str]:
    if not distribution:
        return ["- unavailable"]
    return [
        f"- `{float(probability):.8f}`: {_format_state_rows(state, mapping)}"
        for state, probability in sorted(
            distribution.items(), key=lambda item: (-float(item[1]), repr(item[0]))
        )[: int(limit)]
    ]


def _regional_root_actions(candidate: Optional[bgr.PartitionPolicyCandidate], mapping: Mapping[int, str]) -> Tuple[str, ...]:
    if candidate is None:
        return ()
    rows = []
    for ref in candidate.region_policy_options:
        action = ref.root_action
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            rows.append(f"region {tuple(ref.region_nodes)}: unavailable")
            continue
        source = ref.mapping.get(int(action[0]))
        target = ref.mapping.get(int(action[1]))
        if source is None or target is None:
            rows.append(f"region {tuple(ref.region_nodes)}: unmapped local action {action!r}")
        else:
            rows.append(f"{_name(mapping, source)} -> {_name(mapping, target)}")
    return tuple(rows)


def render_casebook_case(
    *,
    record: RegionalCompoundingValidationV2Record,
    case: rcv.BenchmarkCase,
    full_exact: rcv.ExactFullGraphReferenceResult,
    regional_mc1: rcv.RegionalApproximationResult,
    exact_selection: ExactRegionalCandidateSelectionResult,
) -> str:
    mapping = {int(node): str(name) for node, name in record.territory_mapping.items()}
    if set(case.graph_spec.nodes) - set(mapping):
        raise ValueError("Casebook rendering requires a complete authoritative territory mapping")
    exact_candidate = _candidate_for_identity(
        tuple(regional_mc1.retained_candidates), exact_selection.canonical_selected_candidate_identity
    )
    territory_audit = record.diagnostics.get("territory_mapping", {})
    lines = [
        f"## {record.benchmark_id}",
        "",
        f"- Topology: `{record.topology_family}`",
        f"- Graph size / troop cap: `{case.graph_spec.node_count}` / `{case.troop_cap}`",
        f"- Regional candidates: `{record.diagnostics.get('retained_candidate_count')}`",
        f"- MC1 regions / exact-regional regions: `{record.diagnostics.get('mc1_region_count')}` / `{record.diagnostics.get('exact_regional_region_count')}`",
        f"- Territory mapping: `{territory_audit.get('mapping_kind')}`; induced=`{territory_audit.get('induced')}`",
        "",
        "### Initial territories",
        "",
    ]
    lines.extend(
        f"- {_name(mapping, node)}: owner `{owner}`, troops `{troops}`"
        for node, owner, troops in case.initial_state_signature
    )
    lines.extend(["", "### Edges", ""])
    lines.extend(
        f"- {_name(mapping, left)} - {_name(mapping, right)}"
        for left, right in case.graph_spec.edges
    )
    if territory_audit.get("extra_board_edges"):
        lines.extend(
            [
                "",
                "> Mapping audit: this is edge-preserving but not induced. The authoritative board also has the following edges between mapped territories, which are not part of the synthetic benchmark topology:",
            ]
        )
        lines.extend(
            f"> - {_name(mapping, left)} - {_name(mapping, right)}"
            for left, right in territory_audit.get("extra_board_edges", ())
        )
    lines.extend(["", "### Exact-regional selected partition", ""])
    for index, region in enumerate(exact_selection.selected_partition_signature or (), start=1):
        state = {node: (owner, troops) for node, owner, troops in case.initial_state_signature}
        attackers = [_name(mapping, node) for node in region if state[node][0] == "A"]
        defenders = [_name(mapping, node) for node in region if state[node][0] == "D"]
        lines.append(
            f"- Region {index}: {', '.join(_name(mapping, node) for node in region)}; "
            f"attackers={attackers}; defenders={defenders}"
        )
    lines.extend(
        [
            "",
            "### Candidate selections",
            "",
            f"- MC1 candidate identity: `{record.regional_mc1_candidate_identity!r}`",
            f"- MC1 partition: `{regional_mc1.selected_partition_signature!r}`",
            f"- MC1 policy-option indices: `{regional_mc1.selected_policy_option_indices!r}`",
            f"- MC1 second-stage score: `{regional_mc1.selected_candidate_second_stage_value!r}`",
            f"- Exact-regional candidate identity: `{record.regional_exact_candidate_identity!r}`",
            f"- Exact-regional partition: `{exact_selection.selected_partition_signature!r}`",
            f"- Exact-regional policy-option indices: `{exact_selection.selected_policy_option_indices!r}`",
            f"- Exact-regional expected score: `{exact_selection.selected_exact_expected_score!r}`",
            f"- Candidate identity agreement: `{record.candidate_selection_effect.get('candidate_identity_agreement')}`",
            "",
            "### Root actions",
            "",
            f"- Full exact canonical root action: `{full_exact.canonical_optimal_policy[1] if full_exact.canonical_optimal_policy else None}`",
            f"- Full contingent trace: `{full_exact.diagnostics.get('policy_trace', {}).get('conditional_trace_status', 'unavailable')}`",
        ]
    )
    lines.extend(f"- MC1 regional root action: {value}" for value in _regional_root_actions(regional_mc1.selected_candidate, mapping))
    lines.extend(f"- Exact-regional root action: {value}" for value in _regional_root_actions(exact_candidate, mapping))
    lines.extend(
        [
            "",
            "### Full exact reference",
            "",
            f"- Optimal value: `{full_exact.optimal_value!r}`",
            f"- Root-optimal policy count: `{len(full_exact.optimal_policy_set)}`",
            "",
            "Top full-exact successor states:",
        ]
    )
    lines.extend(_top_states_markdown(record.full_exact_distribution, mapping))
    lines.extend(["", "Top exact-regional successor states:"])
    lines.extend(_top_states_markdown(record.regional_exact_distribution, mapping))
    lines.extend(["", "Top MC1-regional successor states:"])
    lines.extend(_top_states_markdown(record.regional_mc1_distribution, mapping))
    lines.extend(["", "### Pairwise metrics", ""])
    for title, metrics in (
        ("MC1 regional vs full exact", record.mc1_vs_full_metrics),
        ("Exact regional vs full exact", record.exact_regional_vs_full_metrics),
        ("MC1 regional vs exact regional", record.mc1_vs_exact_regional_metrics),
    ):
        node_metrics = metrics.get("node_marginals", {})
        largest = node_metrics.get("largest_error_node")
        per_node = node_metrics.get("per_node", {}).get(largest, {})
        lines.extend(
            [
                f"#### {title}",
                "",
                f"- TV / JS / mass overlap: `{metrics.get('total_variation')}` / `{metrics.get('jensen_shannon')}` / `{metrics.get('probability_mass_overlap')}`",
                f"- Balanced Wasserstein: `{_metric_value(metrics, 'balanced_wasserstein')}`",
                f"- Mean/max ownership error: `{node_metrics.get('mean_absolute_ownership_probability_error')}` / `{node_metrics.get('maximum_ownership_probability_error')}`",
                f"- Mean/max troop error: `{node_metrics.get('mean_absolute_expected_troop_error')}` / `{node_metrics.get('maximum_expected_troop_error')}`",
                f"- Largest-error territory: `{_name(mapping, largest) if largest in mapping else None}`; boundary=`{per_node.get('is_partition_boundary')}`",
                f"- Conquest / no-gain error: `{_metric_value(metrics, 'conquest_error')}` / `{_metric_value(metrics, 'no_gain_error')}`",
                f"- Maximum covariance error: `{_metric_value(metrics, 'covariance_error')}`",
                "",
            ]
        )
    lines.extend(
        [
            "### Measured interpretation",
            "",
            f"Diagnostic labels: `{record.diagnostic_labels!r}`. These labels are generated from the versioned thresholds in the record; no contingent policy behavior is inferred from terminal distributions.",
            "",
        ]
    )
    return "\n".join(lines)


def _choose_casebook_records(
    records: Sequence[RegionalCompoundingValidationV2Record],
) -> Dict[str, Tuple[RegionalCompoundingValidationV2Record, ...]]:
    mapped = [
        record
        for record in records
        if record.diagnostics.get("territory_mapping", {}).get(
            "all_benchmark_edges_authoritative_adjacencies"
        )
    ]
    selected_good: List[RegionalCompoundingValidationV2Record] = []

    def optional_metric(record: RegionalCompoundingValidationV2Record, key: str) -> float:
        value = _metric_value(record.exact_regional_vs_full_metrics, key)
        return math.inf if value is None else float(value)

    def add_best(pool: Sequence[RegionalCompoundingValidationV2Record]) -> None:
        for item in sorted(
            pool,
            key=lambda record: (
                float(record.exact_regional_vs_full_metrics.get("total_variation", math.inf)),
                optional_metric(record, "balanced_wasserstein"),
                record.benchmark_id,
            ),
        ):
            if item not in selected_good:
                selected_good.append(item)
                return

    add_best([record for record in mapped if record.topology_family in {"bridge", "two_dense"}])
    add_best([record for record in mapped if int(record.diagnostics.get("exact_regional_region_count", 0)) == 1])
    add_best([record for record in mapped if int(record.diagnostics.get("exact_regional_region_count", 0)) >= 2])
    add_best(mapped)
    good = tuple(selected_good[:3])
    bad = tuple(
        sorted(
            [record for record in mapped if record.topology_family == "double_front"],
            key=lambda record: (
                -float(record.exact_regional_vs_full_metrics.get("total_variation", -1.0)),
                record.benchmark_id,
            ),
        )[:3]
    )
    mixed_pool = [
        record
        for record in mapped
        if (
            "high_tv_low_wasserstein" in record.diagnostic_labels
            or (
                float(record.exact_regional_vs_full_metrics.get("total_variation", 0.0)) >= 0.20
                and float(_metric_value(record.exact_regional_vs_full_metrics, "ownership_error") or 0.0) <= 0.05
            )
        )
        and record not in good
        and record not in bad
    ]
    if len(mixed_pool) < 2:
        mixed_pool.extend(
            record
            for record in sorted(
                mapped,
                key=lambda item: -float(item.exact_regional_vs_full_metrics.get("total_variation", 0.0)),
            )
            if record not in mixed_pool and record not in good and record not in bad
        )
    return {"good": good, "bad_double_front": bad, "mixed": tuple(mixed_pool[:2])}


def generate_v2_casebooks(
    *,
    output_dir: Path | str,
    v1_output_dir: Path | str,
) -> Dict[str, Any]:
    root = Path(output_dir)
    v1_root = Path(v1_output_dir)
    records = load_v2_records(root)
    selected = _choose_casebook_records(records)
    files = {
        "good": "good_cases.md",
        "bad_double_front": "bad_double_front_cases.md",
        "mixed": "mixed_cases.md",
    }
    manifest_rows = []
    for category, category_records in selected.items():
        sections = [f"# {category.replace('_', ' ').title()}", ""]
        for record in category_records:
            benchmark_id = record.benchmark_id
            case = _load_pickle(v1_root / "states" / f"{benchmark_id}.pkl")
            full_exact = _load_pickle(v1_root / "exact_results" / f"{benchmark_id}.pkl")
            regional_mc1 = _load_pickle(v1_root / "approximation_results" / f"{benchmark_id}.pkl")
            exact_selection = _load_pickle(root / "candidate_evaluations" / f"{benchmark_id}.pkl")
            sections.append(
                render_casebook_case(
                    record=record,
                    case=case,
                    full_exact=full_exact,
                    regional_mc1=regional_mc1,
                    exact_selection=exact_selection,
                )
            )
            manifest_rows.append(
                {
                    "category": category,
                    "benchmark_id": benchmark_id,
                    "territory_mapping": record.territory_mapping,
                    "territory_mapping_audit": record.diagnostics.get("territory_mapping", {}),
                    "diagnostic_labels": record.diagnostic_labels,
                }
            )
        path = root / "casebook" / files[category]
        temporary = path.with_suffix(".md.tmp")
        temporary.write_text("\n\n".join(sections), encoding="utf-8")
        temporary.replace(path)
    _atomic_json(root / "casebook" / "all_selected_cases.json", manifest_rows)
    return {
        "selected_counts": {category: len(values) for category, values in selected.items()},
        "files": {category: str(root / "casebook" / name) for category, name in files.items()},
        "mapping_constraint": "every rendered benchmark edge maps to an authoritative Risk board adjacency",
        "induced_mapping_not_always_available": any(
            not bool(row["territory_mapping_audit"].get("induced")) for row in manifest_rows
        ),
    }


def write_double_front_report(output_dir: Path | str) -> Dict[str, Any]:
    root = Path(output_dir)
    records = [record for record in load_v2_records(root) if record.topology_family == "double_front"]
    rows = []
    lines = ["# Double-Front Analysis", ""]
    for record in sorted(records, key=lambda item: item.benchmark_id):
        row = {
            "benchmark_id": record.benchmark_id,
            "graph_size": record.diagnostics.get("graph_size"),
            "troop_cap": record.diagnostics.get("troop_cap"),
            "initial_state": record.initial_state_signature,
            "territory_mapping": record.territory_mapping,
            "retained_candidate_count": record.diagnostics.get("retained_candidate_count"),
            "mc1_candidate": record.regional_mc1_candidate_identity,
            "exact_regional_candidate": record.regional_exact_candidate_identity,
            "candidate_changed": not bool(record.candidate_selection_effect.get("candidate_identity_agreement")),
            "mc1_full_tv": record.mc1_vs_full_metrics.get("total_variation"),
            "regional_exact_full_tv": record.exact_regional_vs_full_metrics.get("total_variation"),
            "mc1_regional_exact_tv": record.mc1_vs_exact_regional_metrics.get("total_variation"),
            "shared_troop_source_present": record.exact_regional_vs_full_metrics.get("interaction_descriptors", {}).get("shared_troop_source_present"),
            "sequence_opening_present": record.exact_regional_vs_full_metrics.get("interaction_descriptors", {}).get("sequence_opening_present"),
            "articulation_nodes": record.exact_regional_vs_full_metrics.get("interaction_descriptors", {}).get("articulation_nodes"),
            "boundary_edges": record.exact_regional_vs_full_metrics.get("interaction_descriptors", {}).get("partition_boundary_edges"),
            "diagnostic_labels": record.diagnostic_labels,
        }
        rows.append(row)
        lines.extend(
            [
                f"## {record.benchmark_id}",
                "",
                f"- Graph size / cap / candidates: `{row['graph_size']}` / `{row['troop_cap']}` / `{row['retained_candidate_count']}`",
                f"- Candidate changed: `{row['candidate_changed']}`",
                f"- TV MC1-full / exact-regional-full / MC1-exact-regional: `{row['mc1_full_tv']}` / `{row['regional_exact_full_tv']}` / `{row['mc1_regional_exact_tv']}`",
                f"- Shared source / sequence opening / articulations: `{row['shared_troop_source_present']}` / `{row['sequence_opening_present']}` / `{row['articulation_nodes']}`",
                f"- Boundary edges: `{row['boundary_edges']}`",
                f"- Labels: `{row['diagnostic_labels']}`",
                "",
            ]
        )
    previous_tv1 = [row for row in rows if float(row.get("mc1_full_tv") or 0.0) >= 1.0 - 1e-12]
    severe = [row for row in rows if float(row.get("regional_exact_full_tv") or 0.0) >= 0.9]
    report = {
        "records": len(rows),
        "candidate_changed": sum(bool(row["candidate_changed"]) for row in rows),
        "previous_tv1_failures": len(previous_tv1),
        "previous_tv1_now_below_0_9": sum(float(row.get("regional_exact_full_tv") or 0.0) < 0.9 for row in previous_tv1),
        "previous_tv1_remaining_at_or_above_0_9": sum(float(row.get("regional_exact_full_tv") or 0.0) >= 0.9 for row in previous_tv1),
        "severe_after_exact_selection": len(severe),
        "severe_descriptor_counts": {
            "shared_troop_source_present": sum(bool(row.get("shared_troop_source_present")) for row in severe),
            "sequence_opening_present": sum(bool(row.get("sequence_opening_present")) for row in severe),
            "articulation_present": sum(bool(row.get("articulation_nodes")) for row in severe),
            "candidate_changed": sum(bool(row.get("candidate_changed")) for row in severe),
        },
        "rows": rows,
        "mechanism_interpretation": "Only measured topology descriptors and root-action metadata are used; conditional policy switching and survivor redistribution are not inferred from terminal distributions.",
    }
    lines.extend(
        [
            "## Summary",
            "",
            f"- Previous TV=1 failures: `{report['previous_tv1_failures']}`",
            f"- Below TV 0.9 after exact selection: `{report['previous_tv1_now_below_0_9']}`",
            f"- Remaining at or above TV 0.9: `{report['previous_tv1_remaining_at_or_above_0_9']}`",
            f"- All severe exact-regional cases: `{report['severe_after_exact_selection']}`",
            f"- Severe-case measured descriptors: `{report['severe_descriptor_counts']}`",
            "",
            report["mechanism_interpretation"],
        ]
    )
    path = root / "reports" / "double_front_analysis.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    _atomic_json(root / "comparison_tables" / "double_front_cases.json", report)
    return report


def _process_rss_bytes() -> Optional[int]:
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if success else None
    except Exception:
        return None


def _expansion_graph_spec(family: str, node_count: int) -> rcv.BenchmarkGraphSpec:
    normalized = "two_dense" if family == "two_dense_subgraphs" else str(family)
    attacker_count = max(2, int(node_count) // 2)
    edges = rcv._topology_edges(normalized, int(node_count), attacker_count)
    return rcv.BenchmarkGraphSpec(
        graph_id=f"expansion_{family}_n{int(node_count)}_{_stable_digest(edges)[:10]}",
        topology_family=str(family),
        node_count=int(node_count),
        attacker_count=attacker_count,
        edges=edges,
        descriptors={"tractability_expansion": True},
    )


def _expansion_states(
    spec: rcv.BenchmarkGraphSpec,
    *,
    troop_cap: int,
    random_seed: int,
) -> Tuple[Tuple[str, Tuple[Tuple[int, str, int], ...]], ...]:
    cap = int(troop_cap)
    nodes = spec.nodes
    attackers = set(nodes[: spec.attacker_count])
    border_attackers = sorted(
        {
            left if left in attackers else right
            for left, right in spec.edges
            if (left in attackers) != (right in attackers)
        }
    )
    seed = int.from_bytes(
        hashlib.sha256(
            f"{int(random_seed)}|{spec.graph_id}|{cap}".encode("ascii")
        ).digest()[:8],
        "big",
    )
    rng = random.Random(seed)

    def rows(kind: str) -> Tuple[Tuple[int, str, int], ...]:
        answer = []
        for node in nodes:
            owner = "A" if node in attackers else "D"
            if kind == "attacker_favored":
                troops = cap if owner == "A" else 1
            elif kind == "defender_favored":
                troops = 1 if owner == "A" else cap
            elif kind == "high_legal_action_count":
                troops = cap if owner == "A" else max(1, cap // 2)
            elif kind == "low_legal_action_count":
                troops = 1
                if owner == "D":
                    troops = cap
                elif border_attackers and node == border_attackers[0]:
                    troops = min(cap, 2)
            else:
                low = max(1, cap // 2)
                troops = rng.randint(low, cap)
            answer.append((int(node), owner, int(troops)))
        return tuple(answer)

    strata = (
        "balanced",
        "attacker_favored",
        "defender_favored",
        "high_legal_action_count",
        "low_legal_action_count",
    )
    return tuple((stratum, rows(stratum)) for stratum in strata)


def _legal_action_count(
    edges: Sequence[Tuple[int, int]], initial_state: Sequence[Tuple[int, str, int]]
) -> int:
    state = {int(node): (str(owner), int(troops)) for node, owner, troops in initial_state}
    count = 0
    for left, right in edges:
        for source, target in ((left, right), (right, left)):
            if state[source][0] == "A" and state[source][1] > 1 and state[target][0] == "D":
                count += 1
    return count


def _expansion_cells(
    *,
    include_optional: bool,
) -> Tuple[Tuple[int, int], ...]:
    cells = [(9, 3), (9, 4), (9, 5), (10, 3), (10, 4), (8, 6), (8, 7)]
    if include_optional:
        cells.extend(((9, 6), (10, 5)))
    return tuple(cells)


def run_exact_tractability_expansion(
    *,
    output_dir: Path | str,
    topology_families: Sequence[str] = (
        "double_front",
        "bridge",
        "chain",
        "cycle",
        "star",
        "articulation",
        "two_dense_subgraphs",
    ),
    include_optional_cells: bool = False,
    random_seed: int = 20260718,
    max_runtime_seconds: Optional[float] = 10.0,
    max_states: Optional[int] = 250000,
    max_cache_entries: Optional[int] = 500000,
    max_memory_estimate_bytes: Optional[int] = 512 * 1024 * 1024,
    max_policy_options: Optional[int] = 32,
    resume: bool = False,
) -> Dict[str, Any]:
    root = Path(output_dir) / "tractability_expansion"
    raw_root = root / "raw_results"
    raw_root.mkdir(parents=True, exist_ok=True)
    config = {
        "cells": _expansion_cells(include_optional=include_optional_cells),
        "topology_families": tuple(topology_families),
        "state_strata": (
            "balanced",
            "attacker_favored",
            "defender_favored",
            "high_legal_action_count",
            "low_legal_action_count",
        ),
        "random_seed": int(random_seed),
        "max_runtime_seconds": max_runtime_seconds,
        "max_states": max_states,
        "max_cache_entries": max_cache_entries,
        "max_memory_estimate_bytes": max_memory_estimate_bytes,
        "max_policy_options": max_policy_options,
        "memory_measurement": "Windows process RSS plus deterministic cache estimate",
    }
    fingerprint = _stable_digest(config)
    config_path = root / "config.json"
    if resume and config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != fingerprint:
            old_cells = {tuple(int(value) for value in cell) for cell in old.get("cells", ())}
            new_cells = {tuple(int(value) for value in cell) for cell in config["cells"]}
            old_comparable = {key: value for key, value in old.items() if key not in {"cells", "fingerprint"}}
            new_comparable = _json_ready(
                {key: value for key, value in config.items() if key != "cells"}
            )
            if old_comparable != new_comparable or not old_cells.issubset(new_cells):
                raise ValueError("Tractability expansion resume configuration mismatch")
            _atomic_json(config_path, {**config, "fingerprint": fingerprint})
    else:
        _atomic_json(config_path, {**config, "fingerprint": fingerprint})
    requested = 0
    completed_now = 0
    skipped = 0
    for node_count, cap in _expansion_cells(include_optional=include_optional_cells):
        for family in topology_families:
            spec = _expansion_graph_spec(str(family), int(node_count))
            for stratum, initial_state in _expansion_states(
                spec, troop_cap=int(cap), random_seed=int(random_seed)
            ):
                requested += 1
                benchmark_id = "exact_expansion_" + _stable_digest(
                    (spec.graph_id, int(cap), stratum, initial_state)
                )[:24]
                path = raw_root / f"{benchmark_id}.pkl"
                if resume and path.exists():
                    skipped += 1
                    continue
                rss_before = _process_rss_bytes()
                result = rcv.solve_full_graph_exact_reference(
                    graph=spec.graph(),
                    initial_state=initial_state,
                    include_all_optimal_policies=True,
                    max_runtime_seconds=max_runtime_seconds,
                    max_states=max_states,
                    max_cache_entries=max_cache_entries,
                    max_memory_estimate_bytes=max_memory_estimate_bytes,
                    max_policy_options=max_policy_options,
                )
                rss_after = _process_rss_bytes()
                stats = dict(result.diagnostics.get("solver_stats", {}))
                status = "completed" if result.status == "exact_complete" else result.status
                record = ExactTractabilityExpansionRecord(
                    benchmark_id=benchmark_id,
                    graph_signature=(spec.nodes, spec.edges),
                    topology_family=str(family),
                    graph_size=int(node_count),
                    attacker_node_count=spec.attacker_count,
                    defender_node_count=int(node_count) - spec.attacker_count,
                    troop_cap=int(cap),
                    state_stratum=stratum,
                    initial_state_signature=initial_state,
                    initial_legal_action_count=_legal_action_count(spec.edges, initial_state),
                    status=status,
                    limit_reached=(result.status if result.status.endswith("_limit") else None),
                    runtime_seconds=result.runtime_seconds,
                    states_evaluated=result.states_evaluated,
                    value_cache_entries=int(result.diagnostics.get("value_cache_size", result.states_evaluated)),
                    distribution_cache_entries=int(result.diagnostics.get("distribution_cache_size", 0)),
                    total_cache_entries=int(result.diagnostics.get("total_cache_entries", 0)),
                    cache_hits=result.cache_hits,
                    actions_evaluated=result.actions_evaluated,
                    combat_lookups=int(stats.get("combat_lookup_misses", 0)),
                    terminal_support_size=len(result.canonical_optimal_distribution),
                    optimal_policy_count=len(result.optimal_policy_set),
                    estimated_cache_bytes=int(result.diagnostics.get("estimated_cache_bytes", 0)),
                    rss_before_bytes=rss_before,
                    rss_after_bytes=rss_after,
                    rss_delta_bytes=(
                        int(rss_after - rss_before)
                        if rss_before is not None and rss_after is not None
                        else None
                    ),
                    diagnostics={
                        "exact_status": result.status,
                        "error": result.diagnostics.get("error"),
                        "solver_stats": stats,
                        "memory_estimate_method": result.diagnostics.get("memory_estimate_method"),
                        "configured_limits": {
                            "max_runtime_seconds": max_runtime_seconds,
                            "max_states": max_states,
                            "max_cache_entries": max_cache_entries,
                            "max_memory_estimate_bytes": max_memory_estimate_bytes,
                            "max_policy_options": max_policy_options,
                        },
                    },
                )
                _atomic_pickle(path, record)
                completed_now += 1
    summary = summarize_exact_tractability_expansion(output_dir)
    return {
        "requested": requested,
        "completed_now": completed_now,
        "skipped_existing": skipped,
        "records_total": summary.get("record_count", 0),
        "summary_path": str(root / "summary.json"),
    }


def load_tractability_expansion_records(
    output_dir: Path | str,
) -> Tuple[ExactTractabilityExpansionRecord, ...]:
    records = []
    for path in sorted((Path(output_dir) / "tractability_expansion" / "raw_results").glob("*.pkl")):
        value = _load_pickle(path)
        records.append(
            value
            if isinstance(value, ExactTractabilityExpansionRecord)
            else ExactTractabilityExpansionRecord(**dict(value))
        )
    return tuple(records)


def summarize_exact_tractability_expansion(output_dir: Path | str) -> Dict[str, Any]:
    root = Path(output_dir) / "tractability_expansion"
    records = load_tractability_expansion_records(output_dir)
    cells = {}
    grouped = itertools.groupby(
        sorted(records, key=lambda row: (row.graph_size, row.troop_cap, row.topology_family)),
        key=lambda row: (row.graph_size, row.troop_cap, row.topology_family),
    )
    for key, group in grouped:
        rows = list(group)
        cells[repr(key)] = {
            "records": len(rows),
            "completed": sum(row.status == "completed" for row in rows),
            "completion_rate": sum(row.status == "completed" for row in rows) / len(rows),
            "status_histogram": {
                status: sum(row.status == status for row in rows)
                for status in sorted({row.status for row in rows})
            },
            "runtime_seconds": _numeric_summary(row.runtime_seconds for row in rows),
            "states_evaluated": _numeric_summary(row.states_evaluated for row in rows),
            "estimated_cache_bytes": _numeric_summary(row.estimated_cache_bytes for row in rows),
            "rss_delta_bytes": _numeric_summary(row.rss_delta_bytes for row in rows),
            "terminal_support_size": _numeric_summary(row.terminal_support_size for row in rows),
            "initial_legal_action_count": _numeric_summary(row.initial_legal_action_count for row in rows),
        }
    boundaries = {}
    for node_count, cap in sorted({(row.graph_size, row.troop_cap) for row in records}):
        subset = [row for row in records if row.graph_size == node_count and row.troop_cap == cap]
        boundaries[repr((node_count, cap))] = {
            "records": len(subset),
            "all_completed": bool(subset) and all(row.status == "completed" for row in subset),
            "completion_rate": sum(row.status == "completed" for row in subset) / len(subset),
            "status_histogram": {
                status: sum(row.status == status for row in subset)
                for status in sorted({row.status for row in subset})
            },
            "runtime_seconds": _numeric_summary(row.runtime_seconds for row in subset),
            "states_evaluated": _numeric_summary(row.states_evaluated for row in subset),
            "estimated_cache_bytes": _numeric_summary(row.estimated_cache_bytes for row in subset),
            "maximum_runtime_seconds": max((row.runtime_seconds for row in subset), default=None),
            "maximum_states_evaluated": max((row.states_evaluated for row in subset), default=None),
            "maximum_estimated_cache_bytes": max((row.estimated_cache_bytes for row in subset), default=None),
        }
    fully_validated = [
        (node_count, cap)
        for (node_count, cap) in sorted(
            {(row.graph_size, row.troop_cap) for row in records}
        )
        if boundaries[repr((node_count, cap))]["all_completed"]
    ]
    summary = {
        "generated_at": rcv._utc_now(),
        "record_count": len(records),
        "completed": sum(row.status == "completed" for row in records),
        "status_histogram": {
            status: sum(row.status == status for row in records)
            for status in sorted({row.status for row in records})
        },
        "cells": cells,
        "node_cap_boundaries": boundaries,
        "fully_completed_node_cap_cells": tuple(sorted(fully_validated)),
        "overall_runtime_seconds": _numeric_summary(row.runtime_seconds for row in records),
        "overall_states_evaluated": _numeric_summary(row.states_evaluated for row in records),
        "overall_estimated_cache_bytes": _numeric_summary(row.estimated_cache_bytes for row in records),
        "overall_rss_delta_bytes": _numeric_summary(row.rss_delta_bytes for row in records),
        "memory_measurement": (
            "Windows process RSS and deterministic cache estimate"
            if any(row.rss_delta_bytes is not None for row in records)
            else "deterministic cache estimate only; direct process RSS was unavailable"
        ),
        "limit_records": [
            _json_ready(row)
            for row in records
            if row.status != "completed"
        ],
        "largest_completed_states": [
            _json_ready(row)
            for row in sorted(
                (item for item in records if item.status == "completed"),
                key=lambda item: (item.states_evaluated, item.runtime_seconds),
                reverse=True,
            )[:20]
        ],
        "boundary_interpretation": "A node/cap cell is validated only when every configured topology and all five state strata completed under the configured limits.",
    }
    _atomic_json(root / "summary.json", summary)
    write_tractability_report(output_dir, summary=summary)
    return summary


def write_tractability_report(
    output_dir: Path | str,
    *,
    summary: Optional[Mapping[str, Any]] = None,
) -> Path:
    output_root = Path(output_dir)
    root = output_root / "tractability_expansion"
    data = dict(summary or json.loads((root / "summary.json").read_text(encoding="utf-8")))
    lines = [
        "# Exact Tractability Expansion",
        "",
        f"- Attempted states: `{data.get('record_count')}`.",
        f"- Exact completions: `{data.get('completed')}`.",
        f"- Status histogram: `{data.get('status_histogram')}`.",
        f"- Memory measurement: `{data.get('memory_measurement')}`.",
        f"- Fully completed node/cap cells: `{data.get('fully_completed_node_cap_cells')}`.",
        "",
        "A cell is unconditionally validated only when all seven configured topology families and all five state strata completed under every configured safeguard.",
        "",
        "## Node-Cap Cells",
        "",
        "| Nodes | Cap | Complete | Completion | Median runtime | P90 runtime | Max runtime | Median states | Max states | Max estimated bytes |",
        "| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    boundaries = data.get("node_cap_boundaries", {})
    boundary_rows = []
    for key, row in boundaries.items():
        node_count, cap = tuple(int(value) for value in key.strip("()").split(","))
        boundary_rows.append((node_count, cap, row))
    for node_count, cap, row in sorted(boundary_rows):
        lines.append(
            f"| {node_count} | {cap} | {row.get('all_completed')} | "
            f"{_report_number(row.get('completion_rate'))} | "
            f"{_report_number(row.get('runtime_seconds', {}).get('median'))} | "
            f"{_report_number(row.get('runtime_seconds', {}).get('p90'))} | "
            f"{_report_number(row.get('maximum_runtime_seconds'))} | "
            f"{_report_number(row.get('states_evaluated', {}).get('median'))} | "
            f"{_report_number(row.get('maximum_states_evaluated'))} | "
            f"{_report_number(row.get('maximum_estimated_cache_bytes'))} |"
        )
    lines.extend(["", "## Safeguard Stops", ""])
    limit_records = data.get("limit_records", ())
    if not limit_records:
        lines.append("- None.")
    else:
        for row in limit_records:
            lines.append(
                f"- `{row.get('graph_size')} nodes, cap {row.get('troop_cap')}, "
                f"{row.get('topology_family')}, {row.get('state_stratum')}`: "
                f"status `{row.get('status')}`, runtime `{_report_number(row.get('runtime_seconds'))}` seconds, "
                f"states `{row.get('states_evaluated')}`, estimated cache bytes `{row.get('estimated_cache_bytes')}`."
            )
    lines.extend(
        [
            "",
            "## Largest Completed States",
            "",
            "| Nodes | Cap | Topology | Stratum | Runtime | States | Support | Policies | Estimated bytes |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in data.get("largest_completed_states", ())[:10]:
        lines.append(
            f"| {row.get('graph_size')} | {row.get('troop_cap')} | {row.get('topology_family')} | "
            f"{row.get('state_stratum')} | {_report_number(row.get('runtime_seconds'))} | "
            f"{row.get('states_evaluated')} | {row.get('terminal_support_size')} | "
            f"{row.get('optimal_policy_count')} | {row.get('estimated_cache_bytes')} |"
        )
    lines.extend(
        [
            "",
            "## Routing Recommendation",
            "",
            "Retain the prior full-exact boundary through eight nodes and cap five. Add the fully validated non-rectangular cells: eight nodes cap six, nine nodes through cap five, and ten nodes through cap four.",
            "",
            "Treat eight-cap-seven, nine-cap-six, and ten-cap-five as bounded exact attempts with fallback, not unconditional exact cells; their difficult double-front states reached the runtime safeguard.",
            "",
            "All 50 v1 benchmark states are inside the full-exact boundary. Their problematic double-front cases can therefore be solved as full exact graphs in a later routing change.",
            "",
            "Production routing is not changed in this task; the expanded exact-solver results are used to recommend a later exact-first routing policy.",
        ]
    )
    path = output_root / "reports" / "tractability_report.md"
    _write_markdown(path, lines)
    return path


def exact_first_route_recommendation(
    *,
    graph_size: int,
    troop_cap: int,
    topology_family: str,
    expansion_summary: Optional[Mapping[str, Any]] = None,
) -> str:
    n, cap = int(graph_size), int(troop_cap)
    if n <= 8 and cap <= 5:
        return "full_exact"
    boundary = dict((expansion_summary or {}).get("node_cap_boundaries", {})).get(repr((n, cap)))
    if boundary and boundary.get("all_completed"):
        return "full_exact"
    if str(topology_family) == "double_front" and boundary and float(boundary.get("completion_rate", 0.0)) >= 0.8:
        return "full_exact_high_risk_with_configured_limits"
    if n <= 10:
        return "exact_regional_candidate_evaluation"
    return "regional_approximation_or_new_method"


def build_exact_first_routing_analysis(
    *,
    output_dir: Path | str,
    v1_output_dir: Path | str,
) -> Dict[str, Any]:
    root = Path(output_dir)
    tract_path = root / "tractability_expansion" / "summary.json"
    expansion = json.loads(tract_path.read_text(encoding="utf-8")) if tract_path.exists() else {}
    routes = []
    for path in sorted((Path(v1_output_dir) / "states").glob("*.pkl")):
        case = _load_pickle(path)
        route = exact_first_route_recommendation(
            graph_size=case.graph_spec.node_count,
            troop_cap=case.troop_cap,
            topology_family=case.graph_spec.topology_family,
            expansion_summary=expansion,
        )
        routes.append(
            {
                "benchmark_id": case.benchmark_id,
                "graph_size": case.graph_spec.node_count,
                "troop_cap": case.troop_cap,
                "topology_family": case.graph_spec.topology_family,
                "recommended_route": route,
            }
        )
    histogram = {
        route: sum(row["recommended_route"] == route for row in routes)
        for route in sorted({row["recommended_route"] for row in routes})
    }
    analysis = {
        "production_routing_changed": False,
        "baseline_validated_boundary": {"maximum_nodes": 8, "maximum_troop_cap": 5},
        "expanded_fully_completed_node_cap_cells": expansion.get("fully_completed_node_cap_cells", ()),
        "v1_benchmark_records": len(routes),
        "route_histogram": histogram,
        "full_exact_fraction": (
            sum(row["recommended_route"].startswith("full_exact") for row in routes) / len(routes)
            if routes
            else None
        ),
        "routes": routes,
        "recommendation": "Use full exact solving only inside cells with complete topology-and-strata validation; use exact regional candidate evaluation next, and retain regional approximation for larger weakly coupled cases.",
        "stage_a_sample_inspected": False,
        "stage_a_reason": "Final Stage A generation remains blocked and no Stage A inputs were reconstructed for routing in this task.",
    }
    _atomic_json(root / "tractability_expansion" / "routing_analysis.json", analysis)
    return analysis


def validate_v2_output(output_dir: Path | str) -> Dict[str, Any]:
    root = Path(output_dir)
    errors = []
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        errors.append("missing manifest.json")
        completed = set()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = {str(value) for value in manifest.get("completed_benchmark_ids", ())}
        if manifest.get("format_version") != V2_FORMAT_VERSION:
            errors.append("format version mismatch")
    records = load_v2_records(root)
    ids = {record.benchmark_id for record in records}
    if ids != completed:
        errors.append("manifest IDs do not match v2 records")
    for record in records:
        for label, distribution in (
            ("full_exact", record.full_exact_distribution),
            ("regional_mc1", record.regional_mc1_distribution),
            ("regional_exact", record.regional_exact_distribution),
        ):
            if distribution is not None and abs(sum(float(value) for value in distribution.values()) - 1.0) > 1e-8:
                errors.append(f"{record.benchmark_id}: {label} probability mass is not one")
        delta = record.diagnostics.get("v1_reproduction", {}).get("total_variation_absolute_delta")
        if delta is not None and float(delta) > 1e-10:
            errors.append(f"{record.benchmark_id}: v1 TV reproduction delta={delta}")
        names = list(record.territory_mapping.values())
        if len(names) != len(set(names)):
            errors.append(f"{record.benchmark_id}: duplicate territory names")
        audit = record.diagnostics.get("territory_mapping", {})
        if record.territory_mapping and not audit.get("all_benchmark_edges_authoritative_adjacencies"):
            errors.append(f"{record.benchmark_id}: mapped benchmark edge is not authoritative adjacency")
    validation = {
        "valid": not errors,
        "errors": errors,
        "record_count": len(records),
        "completed_manifest_count": len(completed),
        "failure_file_count": len(list((root / "failures").glob("*.json"))),
        "validated_at": rcv._utc_now(),
    }
    _atomic_json(root / "validation.json", validation)
    return validation
