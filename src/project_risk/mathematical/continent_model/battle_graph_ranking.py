from __future__ import annotations

import itertools
import hashlib
import json
import random
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Sequence, Set, Tuple, Optional, Literal, FrozenSet
import warnings

import numpy as np

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState, NodeState
from project_risk.mathematical.continent_model.approximate_graph_outcome_probabilities import (
    STAR_ONLY_PATTERNS, 
    query_region_from_libraries,
    _is_coverage_failure,
    _is_query_viability_failure)

import logging
log = logging.getLogger("risk.ranking")


RankingVariable = Literal[
    "battle_expected_attacker_territory_count",
    "battle_expected_attacker_troop_count",
    "battle_expected_attacker_conquest_probability",
    "expected_territories",
    "expected_troops",
    "conquest_probability",
]

PolicyOptionSelection = Literal[
    "primary",
    "best_local",
    "best_territories",
    "best_troops",
    "best_conquest",
]


def _normalize_ranking_variable(ranking_variable: str) -> str:
    """
    Map external / experiment-facing ranking variable names to the internal
    PartitionEvaluation field names used for scoring.

    Supported inputs:
      - "battle_expected_attacker_territory_count"  -> "expected_territories"
      - "battle_expected_attacker_troop_count"      -> "expected_troops"
      - "battle_expected_attacker_conquest_probability" -> "conquest_probability"

    Backwards compatible:
      - "expected_territories", "expected_troops", "conquest_probability"
    """
    rv = str(ranking_variable)

    mapping = {
        "battle_expected_attacker_territory_count": "expected_territories",
        "battle_expected_attacker_troop_count": "expected_troops",
        "battle_expected_attacker_conquest_probability": "conquest_probability",
    }

    if rv in mapping:
        return mapping[rv]

    # Back-compat / internal names
    if rv in {"expected_territories", "expected_troops", "conquest_probability"}:
        return rv

    # If you prefer hard-fail instead of fallback, change this to raise ValueError.
    return "expected_territories"


def _normalize_policy_option_selection(policy_option_selection: Any) -> str:
    """Small local wrapper around the AGOP policy-option selector."""
    return agop.normalize_policy_option_selection(policy_option_selection)


def _query_region_selected_option(
    *,
    combat_libraries_base: Path,
    global_state: GlobalState,
    global_edges,
    region_nodes: Sequence[int],
    policy_option_selection: Any = "primary",
    ranking_variable: Any = "expected_territories",
    debug: bool = False,
    debug_limit: int = 20,
    query_cache: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Query a region and ask AGOP to select the compatibility payload.

    The result still contains the full result["policy_options_v2"] list, but
    result["payload"] / result["outcomes_v2"] reflect the selected option.
    """
    return agop.query_region_from_libraries(
        combat_libraries_base=combat_libraries_base,
        global_state=global_state,
        global_edges=global_edges,
        region_nodes=region_nodes,
        debug=debug,
        debug_limit=debug_limit,
        policy_option_selection=policy_option_selection,
        ranking_variable=ranking_variable,
        query_cache=query_cache,
    )


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass
class PartitionEvaluation:
    """
    Aggregated metrics for a single partition of regions.

    Metrics are normalized to the FULL battle graph (not just the
    union of regions):

      - expected_new_territories:
            Expected number of *new* territories acquired in the union
            of regions (sum over regions, from probability tables).

      - expected_lost_troops:
            Expected number of attacker troops lost in those regions.
            (Initial troops in regions minus expected final troops in
             regions.)

      - regional_product_conquest_probability:
            Product of region-level conquest probabilities (probability
            of conquering *each region* in the partition). This is the
            "independent regions" approximation.

      - expected_territories:
            Expected total number of territories owned by the attacker
            on the full battle graph after the attack:
                current_territories + expected_new_territories

      - expected_troops:
            Expected total number of attacker troops on the full
            battle graph after the attack:
                current_troops - expected_lost_troops

      - conquest_probability:
            For partitions that cover the entire battle graph:
                conquest_probability = regional_product_conquest_probability
            For subset partitions (backup mode):
                conquest_probability = 0.0

    The `covers_all_nodes` flag indicates whether this partition is a
    full partition of the battle graph (True) or only covers a subset
    (False).
    """

    partition: List[Dict[str, Any]]
    covers_all_nodes: bool

    # Regional sums
    expected_new_territories: float
    expected_lost_troops: float
    regional_product_conquest_probability: float

    # Full-graph metrics
    expected_territories: float
    expected_troops: float
    conquest_probability: float
    region_policy_options: Tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RegionPolicyOptionRef:
    region_nodes: tuple
    attacker_nodes: tuple
    defender_nodes: tuple
    pattern: tuple
    row_label: str
    option_index: int
    option_count: int
    root_action: Any
    split_metadata: Any
    payload: Dict[str, Any]
    utility_tuple: Tuple[float, ...]
    distribution_payload: Dict[str, Any]
    mapping: Dict[int, int]
    topology_signature: Tuple[Tuple[int, int], ...] = ()
    policy_option_mode: Optional[str] = None


@dataclass
class PartitionPolicyCandidate:
    partition_index: int
    partition_regions: tuple
    region_policy_options: tuple
    first_stage_utility: Tuple[float, ...]
    first_stage_score: Tuple[float, ...]
    mc_mean_second_stage_utility: Optional[Tuple[float, ...]] = None
    mc_std_second_stage_utility: Optional[Tuple[float, ...]] = None
    mc_mean_score: Optional[float] = None
    mc_num_scenarios: int = 0
    mc_final_state_counts: Optional[dict] = None
    mc_final_state_sequence: Optional[tuple] = None
    mc_top_final_states: Optional[list] = None
    diagnostics: Optional[dict] = None


@dataclass
class TwoStagePartitionPolicyResult:
    selected_candidate: Optional[PartitionPolicyCandidate]
    first_stage_optimal_candidates: tuple
    all_candidate_count: int
    first_stage_best_utility: Optional[Tuple[float, ...]]
    mc_scenarios: int
    diagnostics: Dict[str, Any]


@dataclass
class TwoStagePreparedCandidates:
    global_state: GlobalState
    battle_nodes: Tuple[int, ...]
    global_edges: Tuple[Tuple[int, int], ...]
    partitions_full: Tuple[Any, ...]
    working_partitions: Tuple[Any, ...]
    all_candidates: Tuple[PartitionPolicyCandidate, ...]
    retained_candidates: Tuple[PartitionPolicyCandidate, ...]
    best_utility: Optional[Tuple[float, ...]]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class PreparedRegionalPolicyOption:
    key: Tuple[Any, ...]
    region_nodes: Tuple[int, ...]
    attacker_nodes: Tuple[int, ...]
    defender_nodes: Tuple[int, ...]
    pattern: Tuple[int, int]
    row_label: str
    policy_option_index: int
    root_action: Any
    normalized_distribution: Tuple[Tuple[Any, float], ...]
    distribution_signatures: Tuple[Any, ...]
    cumulative_probabilities: Tuple[float, ...]
    owners_by_outcome: Tuple[Tuple[int, ...], ...]
    troops_by_outcome: Tuple[Tuple[int, ...], ...]
    mapping: Tuple[Tuple[int, int], ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedPartitionAssembly:
    partition_signature: Tuple[Tuple[int, ...], ...]
    ordered_region_signatures: Tuple[Tuple[int, ...], ...]
    regional_node_mappings: Tuple[Tuple[Tuple[int, int], ...], ...]
    battle_node_order: Tuple[int, ...]
    unchanged_state_template: GlobalState
    merge_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateSelectionCheckpointResult:
    """One cumulative candidate-selection checkpoint."""

    mc_samples: int
    selected_candidate_index: Optional[int]
    selected_candidate_identity: Any
    selected_partition_signature: Any
    selected_policy_option_indices: Tuple[int, ...]
    best_score_mean: Tuple[float, ...]
    best_score_std: Tuple[float, ...]
    runner_up_candidate_index: Optional[int]
    runner_up_candidate_identity: Any
    runner_up_score_mean: Optional[Tuple[float, ...]]
    runner_up_score_std: Optional[Tuple[float, ...]]
    best_runner_up_gap: Optional[Tuple[float, ...]]
    candidate_rank_order: Tuple[int, ...]
    candidate_identities: Tuple[Any, ...]
    top_candidate_indices: Tuple[int, ...]
    runtime_increment_seconds: float
    runtime_cumulative_seconds: float
    unique_global_states_increment: int
    unique_global_states_cumulative: int
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class NestedCandidateSelectionResumeState:
    """Pickle-safe cumulative state used to continue at a later checkpoint."""

    schema_version: str
    base_seed: int
    state_identity: Any
    candidate_identities: Tuple[Any, ...]
    completed_scenarios: int
    candidate_utilities: Tuple[Tuple[Tuple[float, ...], ...], ...]
    candidate_state_sequences: Tuple[Tuple[Any, ...], ...]
    sampled_regional_outcomes: Mapping[Any, int]
    global_evaluation_cache: Mapping[Any, Tuple[float, ...]]
    unique_global_signatures: Tuple[Any, ...]
    checkpoints: Tuple[CandidateSelectionCheckpointResult, ...]
    runtime_cumulative_seconds: float
    counters: Mapping[str, int]


@dataclass(frozen=True)
class NestedCandidateSelectionResult:
    checkpoints: Tuple[CandidateSelectionCheckpointResult, ...]
    final_selected_candidate_index: Optional[int]
    final_selected_candidate_identity: Any
    final_checkpoint_samples: int
    stopped_early: bool
    stopping_reason: str
    evaluated_candidates: Tuple[PartitionPolicyCandidate, ...]
    resume_state: Optional[NestedCandidateSelectionResumeState]
    diagnostics: Mapping[str, Any]

    @property
    def selected_candidate(self) -> Optional[PartitionPolicyCandidate]:
        index = self.final_selected_candidate_index
        if index is None or index < 0 or index >= len(self.evaluated_candidates):
            return None
        return self.evaluated_candidates[index]


@dataclass(frozen=True)
class ExactCoverAnalysis:
    required_cover_nodes: Tuple[int, ...]
    battle_graph_nodes: Tuple[int, ...]
    active_ad_edge_nodes: Tuple[int, ...]
    active_ad_edges: Tuple[Tuple[int, int], ...]
    supported_region_signatures: Tuple[Tuple[int, ...], ...]
    supported_regions_per_node: Mapping[int, int]
    nodes_in_no_supported_region: Tuple[int, ...]
    production_full_cover_signatures: Tuple[Tuple[Tuple[int, ...], ...], ...]
    brute_force_full_cover_signatures: Tuple[Tuple[Tuple[int, ...], ...], ...]
    production_found_full_cover: bool
    brute_force_found_full_cover: bool
    production_bruteforce_agree: bool
    maximum_disjoint_coverage_count: int
    maximum_disjoint_coverage_ratio: float
    best_partial_cover_signature: Optional[Tuple[Tuple[int, ...], ...]]
    uncovered_nodes_in_best_partial_cover: Tuple[int, ...]
    active_node_cover_exists: Optional[bool]
    active_edge_cover_exists: Optional[bool]
    overlap_boundary_cover_exists: Optional[bool]
    context_only_nodes: Tuple[int, ...]
    diagnostics: Mapping[str, Any]


def _canonical_supported_region_signatures(
    *,
    required_nodes: Collection[int],
    supported_region_signatures: Sequence[Sequence[int]],
) -> Tuple[Tuple[int, ...], ...]:
    required = frozenset(int(n) for n in required_nodes)
    return tuple(
        sorted(
            {
                tuple(sorted({int(n) for n in region}))
                for region in supported_region_signatures
                if region
                and frozenset(int(n) for n in region).issubset(required)
            }
        )
    )


def enumerate_exact_region_covers_reference(
    *,
    required_nodes: Collection[int],
    supported_region_signatures: Sequence[Sequence[int]],
    max_covers: Optional[int] = None,
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    """Independent deterministic exact-cover reference for diagnostics."""
    required = frozenset(int(n) for n in required_nodes)
    if not required:
        return (tuple(),)
    regions = _canonical_supported_region_signatures(
        required_nodes=required,
        supported_region_signatures=supported_region_signatures,
    )
    region_sets = tuple(frozenset(region) for region in regions)
    by_node = {
        node: tuple(i for i, region in enumerate(region_sets) if node in region)
        for node in sorted(required)
    }
    if any(not by_node[node] for node in required):
        return tuple()

    covers: Set[Tuple[Tuple[int, ...], ...]] = set()

    def visit(covered: frozenset[int], chosen: Tuple[int, ...]) -> None:
        if max_covers is not None and len(covers) >= int(max_covers):
            return
        if covered == required:
            covers.add(tuple(sorted(regions[i] for i in chosen)))
            return

        uncovered = required - covered
        choices = {
            node: tuple(
                i for i in by_node[node]
                if region_sets[i].isdisjoint(covered)
            )
            for node in uncovered
        }
        pivot = min(uncovered, key=lambda node: (len(choices[node]), int(node)))
        for region_index in choices[pivot]:
            visit(covered | region_sets[region_index], chosen + (region_index,))

    visit(frozenset(), tuple())
    ordered = tuple(sorted(covers, key=lambda cover: (len(cover), cover)))
    if max_covers is not None:
        return ordered[: max(0, int(max_covers))]
    return ordered


def find_maximum_disjoint_region_coverage(
    *,
    required_nodes: Collection[int],
    supported_region_signatures: Sequence[Sequence[int]],
    max_solutions: Optional[int] = None,
) -> Dict[str, Any]:
    """Find deterministic maximum node coverage by pairwise-disjoint regions."""
    required = frozenset(int(n) for n in required_nodes)
    regions = _canonical_supported_region_signatures(
        required_nodes=required,
        supported_region_signatures=supported_region_signatures,
    )
    region_sets = tuple(frozenset(region) for region in regions)
    by_node = {
        node: tuple(i for i, region in enumerate(region_sets) if node in region)
        for node in sorted(required)
    }
    best_count = -1
    best_covers: Set[Tuple[Tuple[int, ...], ...]] = set()

    def record(covered: frozenset[int], chosen: Tuple[int, ...]) -> None:
        nonlocal best_count, best_covers
        count = len(covered)
        cover = tuple(sorted(regions[i] for i in chosen))
        if count > best_count:
            best_count = count
            best_covers = {cover}
        elif count == best_count:
            best_covers.add(cover)

    def visit(
        covered: frozenset[int],
        excluded: frozenset[int],
        chosen: Tuple[int, ...],
    ) -> None:
        if len(required) - len(excluded) < best_count:
            return
        remaining = required - covered - excluded
        compatible = tuple(
            i for i, region in enumerate(region_sets)
            if region.isdisjoint(covered) and region.isdisjoint(excluded)
        )
        if not remaining or not compatible:
            record(covered, chosen)
            return

        choices = {
            node: tuple(i for i in by_node[node] if i in compatible)
            for node in remaining
        }
        pivot = min(remaining, key=lambda node: (len(choices[node]), int(node)))
        for region_index in choices[pivot]:
            visit(
                covered | region_sets[region_index],
                excluded,
                chosen + (region_index,),
            )
        visit(covered, excluded | {pivot}, chosen)

    visit(frozenset(), frozenset(), tuple())
    if best_count < 0:
        best_count = 0
        best_covers = {tuple()}
    ordered = tuple(sorted(best_covers, key=lambda cover: (len(cover), cover)))
    if max_solutions is not None:
        ordered = ordered[: max(0, int(max_solutions))]
    uncovered = tuple(
        tuple(sorted(required - {n for region in cover for n in region}))
        for cover in ordered
    )
    return {
        "maximum_covered_node_count": int(best_count),
        "maximum_coverage_ratio": float(best_count / len(required)) if required else 1.0,
        "best_partial_cover_signatures": ordered,
        "uncovered_node_sets": uncovered,
        "best_partial_cover_signature": ordered[0] if ordered else None,
        "uncovered_nodes_in_best_partial_cover": uncovered[0] if uncovered else tuple(),
    }


def analyze_active_edge_coverage(
    *,
    active_ad_edges: Collection[Tuple[int, int]],
    supported_region_signatures: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    edges = tuple(sorted({tuple(sorted((int(u), int(v)))) for u, v in active_ad_edges}))
    regions = tuple(frozenset(int(n) for n in region) for region in supported_region_signatures)
    counts = {
        edge: sum(1 for region in regions if set(edge).issubset(region))
        for edge in edges
    }
    missing = tuple(edge for edge in edges if counts[edge] == 0)
    values = tuple(counts.values())
    return {
        "active_edge_coverage_complete": not missing,
        "active_edges_in_no_supported_region": missing,
        "supported_region_count_by_active_edge": counts,
        "minimum_active_edge_region_count": min(values) if values else 0,
        "maximum_active_edge_region_count": max(values) if values else 0,
    }


def analyze_overlap_cover_possibility(
    *,
    required_nodes: Collection[int],
    supported_region_signatures: Sequence[Sequence[int]],
    role_by_node: Mapping[int, str],
    overlap_mode: str,
) -> Dict[str, Any]:
    if overlap_mode not in ("any_overlap", "attacker_only_overlap", "same_role_overlap"):
        raise ValueError(f"Unsupported overlap_mode={overlap_mode!r}")
    required = frozenset(int(n) for n in required_nodes)
    regions = _canonical_supported_region_signatures(
        required_nodes=required,
        supported_region_signatures=supported_region_signatures,
    )
    region_sets = tuple(frozenset(region) for region in regions)
    by_node = {
        node: tuple(i for i, region in enumerate(region_sets) if node in region)
        for node in sorted(required)
    }
    best: Optional[Tuple[Tuple[int, ...], ...]] = None
    best_counts: Dict[int, int] = {}
    visited: Set[frozenset[int]] = set()

    def allowed_overlap(region_index: int, selected: frozenset[int]) -> bool:
        overlap = set()
        for other_index in selected:
            overlap.update(region_sets[region_index] & region_sets[other_index])
        if not overlap or overlap_mode == "any_overlap":
            return True
        roles = {str(role_by_node[int(n)]) for n in overlap}
        if overlap_mode == "attacker_only_overlap":
            return roles == {"A"}
        return len(roles) <= 1

    def visit(selected: frozenset[int], counts: Dict[int, int]) -> None:
        nonlocal best, best_counts
        if selected in visited:
            return
        visited.add(selected)
        if best is not None and len(selected) >= len(best):
            return
        covered = frozenset(n for n, count in counts.items() if count > 0)
        if covered == required:
            cover = tuple(sorted(regions[i] for i in selected))
            if best is None or (len(cover), cover) < (len(best), best):
                best = cover
                best_counts = dict(counts)
            return
        uncovered = required - covered
        pivot = min(uncovered, key=lambda node: (len(by_node.get(node, ())), int(node)))
        for region_index in by_node.get(pivot, ()):
            if region_index in selected or not allowed_overlap(region_index, selected):
                continue
            new_counts = dict(counts)
            for node in region_sets[region_index]:
                new_counts[int(node)] = new_counts.get(int(node), 0) + 1
            visit(selected | {region_index}, new_counts)

    visit(frozenset(), {})
    shared_counts = {
        int(node): int(count)
        for node, count in sorted(best_counts.items())
        if count > 1
    }
    return {
        "overlap_mode": str(overlap_mode),
        "complete_node_coverage_possible": best is not None,
        "cover_signature": best,
        "shared_nodes": tuple(shared_counts),
        "shared_node_use_counts": shared_counts,
        "shared_attacker_nodes": tuple(n for n in shared_counts if role_by_node.get(n) == "A"),
        "shared_defender_nodes": tuple(n for n in shared_counts if role_by_node.get(n) == "D"),
    }


def _graph_nodes_tuple(graph: Any) -> Tuple[int, ...]:
    try:
        nodes = graph.nodes()
    except TypeError:
        nodes = graph.nodes
    return tuple(sorted(int(n) for n in nodes))


def _graph_edges_tuple(graph: Any) -> Tuple[Tuple[int, int], ...]:
    try:
        edges = graph.edges()
    except TypeError:
        edges = graph.edges
    return tuple(sorted({tuple(sorted((int(u), int(v)))) for u, v in edges}))


def derive_partition_cover_universes(
    *,
    battle_graph,
    global_state: GlobalState,
    full_graph=None,
    commitment_map=None,
) -> Dict[str, Tuple[int, ...]]:
    """Derive diagnostic cover universes from the conflict-graph contract."""
    del full_graph, commitment_map
    nodes = _graph_nodes_tuple(battle_graph)
    edges = _graph_edges_tuple(battle_graph)
    production = agop.partition_required_graph_nodes(battle_graph, edges=edges)
    active_edges = tuple(
        edge
        for edge in edges
        if agop.normalize_owner_to_combat_role(global_state.nodes[edge[0]].owner)
        != agop.normalize_owner_to_combat_role(global_state.nodes[edge[1]].owner)
    )
    active_endpoints = tuple(sorted({n for edge in active_edges for n in edge}))
    universes: Dict[str, Tuple[int, ...]] = {
        "production_current": tuple(production),
        "active_ad_endpoints": active_endpoints,
    }

    adjacency = {n: set() for n in nodes}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    remaining = set(nodes)
    active_components = set()
    while remaining:
        start = min(remaining)
        stack = [start]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            remaining.discard(node)
            stack.extend(adjacency.get(node, ()))
        if any(u in component and v in component for u, v in active_edges):
            active_components.update(component)
    active_component_nodes = tuple(sorted(active_components))
    if active_component_nodes != tuple(production):
        universes["active_combat_components"] = active_component_nodes

    # In a conflict-only battle graph, every production node is an endpoint of
    # a current A-D edge and can lose troops or change ownership.
    if tuple(production) == active_endpoints:
        universes["transition_required"] = tuple(production)
    return universes


def classify_partition_context_nodes(
    *,
    battle_graph,
    global_state: GlobalState,
    full_graph=None,
    commitment_map=None,
) -> Dict[str, Any]:
    universes = derive_partition_cover_universes(
        battle_graph=battle_graph,
        global_state=global_state,
        full_graph=full_graph,
        commitment_map=commitment_map,
    )
    production = set(universes["production_current"])
    active = set(universes["active_ad_endpoints"])
    future = set(universes.get("active_combat_components", universes["production_current"]))
    unclassified = tuple(sorted(production - future))
    return {
        "currently_active_combat_nodes": tuple(sorted(active)),
        "future_reachable_combat_nodes": tuple(sorted(future)),
        "immutable_context_nodes": tuple(),
        "unclassified_nodes": unclassified,
        "context_only_nodes": tuple(),
        "diagnostics": {
            "reason": "battle_graph is conflict-only; no node is proven immutable context",
            "transition_required_is_proven": "transition_required" in universes,
        },
    }


def analyze_exact_cover_compatibility(
    *,
    battle_graph,
    global_state: GlobalState,
    supported_regions: Sequence[Any],
    required_cover_nodes: Optional[Collection[int]] = None,
    max_reference_covers: Optional[int] = None,
    full_graph=None,
    commitment_map=None,
    run_overlap_diagnostics: bool = True,
    **kwargs,
) -> ExactCoverAnalysis:
    del kwargs
    battle_nodes = _graph_nodes_tuple(battle_graph)
    edges = _graph_edges_tuple(battle_graph)
    universes = derive_partition_cover_universes(
        battle_graph=battle_graph,
        global_state=global_state,
        full_graph=full_graph,
        commitment_map=commitment_map,
    )
    required = tuple(sorted(
        int(n) for n in (
            required_cover_nodes
            if required_cover_nodes is not None
            else universes["production_current"]
        )
    ))
    raw_signatures = []
    for region in supported_regions:
        if isinstance(region, Mapping):
            raw_signatures.append(region.get("region_nodes", ()))
        else:
            raw_signatures.append(region)
    signatures = _canonical_supported_region_signatures(
        required_nodes=required,
        supported_region_signatures=raw_signatures,
    )
    role_by_node = {
        int(n): agop.normalize_owner_to_combat_role(global_state.nodes[int(n)].owner)
        for n in battle_nodes
    }
    active_edges = tuple(
        edge for edge in edges
        if role_by_node[edge[0]] != role_by_node[edge[1]]
    )
    active_nodes = tuple(sorted({n for edge in active_edges for n in edge}))
    support_count = {
        node: sum(1 for region in signatures if node in region)
        for node in required
    }
    nodes_in_none = tuple(node for node in required if support_count[node] == 0)

    production_covers = agop.enumerate_disjoint_exact_region_covers(
        required_nodes=required,
        supported_region_signatures=signatures,
        max_covers=max_reference_covers,
    )
    reference_covers = enumerate_exact_region_covers_reference(
        required_nodes=required,
        supported_region_signatures=signatures,
        max_covers=max_reference_covers,
    )
    maximum = find_maximum_disjoint_region_coverage(
        required_nodes=required,
        supported_region_signatures=signatures,
        max_solutions=max_reference_covers,
    )
    active_node_covers = enumerate_exact_region_covers_reference(
        required_nodes=active_nodes,
        supported_region_signatures=signatures,
        max_covers=1,
    ) if active_nodes else tuple()
    edge_diag = analyze_active_edge_coverage(
        active_ad_edges=active_edges,
        supported_region_signatures=signatures,
    )
    overlap_diags = {}
    if run_overlap_diagnostics:
        overlap_diags = {
            mode: analyze_overlap_cover_possibility(
                required_nodes=required,
                supported_region_signatures=signatures,
                role_by_node=role_by_node,
                overlap_mode=mode,
            )
            for mode in ("any_overlap", "attacker_only_overlap", "same_role_overlap")
        }
    context = classify_partition_context_nodes(
        battle_graph=battle_graph,
        global_state=global_state,
        full_graph=full_graph,
        commitment_map=commitment_map,
    )
    per_node = {
        node: {
            "owner_role": role_by_node[node],
            "supported_region_count": int(support_count.get(node, 0)),
            "supported_region_signatures": tuple(region for region in signatures if node in region),
            "active_ad_degree": sum(1 for edge in active_edges if node in edge),
            "battle_degree": sum(1 for edge in edges if node in edge),
        }
        for node in required
    }
    best_partial = maximum.get("best_partial_cover_signature")
    uncovered = tuple(maximum.get("uncovered_nodes_in_best_partial_cover", ()) or ())
    overlap_possible = None
    if run_overlap_diagnostics:
        overlap_possible = bool(overlap_diags["any_overlap"]["complete_node_coverage_possible"])
    return ExactCoverAnalysis(
        required_cover_nodes=required,
        battle_graph_nodes=battle_nodes,
        active_ad_edge_nodes=active_nodes,
        active_ad_edges=active_edges,
        supported_region_signatures=signatures,
        supported_regions_per_node=support_count,
        nodes_in_no_supported_region=nodes_in_none,
        production_full_cover_signatures=production_covers,
        brute_force_full_cover_signatures=reference_covers,
        production_found_full_cover=bool(production_covers),
        brute_force_found_full_cover=bool(reference_covers),
        production_bruteforce_agree=production_covers == reference_covers,
        maximum_disjoint_coverage_count=int(maximum["maximum_covered_node_count"]),
        maximum_disjoint_coverage_ratio=float(maximum["maximum_coverage_ratio"]),
        best_partial_cover_signature=best_partial,
        uncovered_nodes_in_best_partial_cover=uncovered,
        active_node_cover_exists=bool(active_node_covers) if active_nodes else None,
        active_edge_cover_exists=bool(edge_diag["active_edge_coverage_complete"]) if active_edges else None,
        overlap_boundary_cover_exists=overlap_possible,
        context_only_nodes=tuple(context["context_only_nodes"]),
        diagnostics={
            "cover_universes": universes,
            "per_node_support": per_node,
            "minimum_supported_regions_per_required_node": min(support_count.values()) if support_count else 0,
            "maximum_supported_regions_per_required_node": max(support_count.values()) if support_count else 0,
            "active_edge_coverage": edge_diag,
            "overlap_coverage": overlap_diags,
            "context_classification": context,
            "reference_cover_limit": max_reference_covers,
        },
    )


def _region_node_tuple(region: Any) -> Tuple[int, ...]:
    if isinstance(region, dict):
        nodes = region.get("region_nodes", ())
    else:
        nodes = getattr(region, "region_nodes", region)
    return tuple(sorted(int(x) for x in nodes))


def canonical_partition_signature(partition: Any) -> Tuple[Tuple[int, ...], ...]:
    """
    Canonical node-set-only signature for a partition.

    Region order, node order, topology hashes, and policy options are ignored.
    """
    regions = tuple(sorted(_region_node_tuple(region) for region in (partition or ())))
    seen: Set[int] = set()
    for region in regions:
        rset = set(region)
        if len(rset) != len(region):
            raise ValueError(f"Partition region contains duplicate nodes: {region!r}")
        overlap = seen & rset
        if overlap:
            raise ValueError(f"Partition regions overlap on nodes: {tuple(sorted(overlap))!r}")
        seen.update(rset)
    return regions


def partition_node_universe(partition_signature: Sequence[Sequence[int]]) -> FrozenSet[int]:
    nodes: Set[int] = set()
    for region in partition_signature:
        rset = {int(x) for x in region}
        overlap = nodes & rset
        if overlap:
            raise ValueError(f"Partition signature regions overlap on nodes: {tuple(sorted(overlap))!r}")
        nodes.update(rset)
    return frozenset(nodes)


def is_strict_exact_coarsening(coarser_partition: Any, finer_partition: Any) -> bool:
    q_regions = [frozenset(r) for r in canonical_partition_signature(coarser_partition)]
    p_regions = [frozenset(r) for r in canonical_partition_signature(finer_partition)]
    if partition_node_universe(q_regions) != partition_node_universe(p_regions):
        return False
    if len(q_regions) >= len(p_regions):
        return False
    return all(any(p_region <= q_region for q_region in q_regions) for p_region in p_regions)


def filter_maximal_supported_partitions(
    partitions: Sequence[Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    signature_to_partition: Dict[Tuple[Tuple[int, ...], ...], Any] = {}
    input_signatures: List[Tuple[Tuple[int, ...], ...]] = []
    for partition in partitions or ():
        sig = canonical_partition_signature(partition)
        input_signatures.append(sig)
        if sig not in signature_to_partition:
            signature_to_partition[sig] = partition

    unique_items = sorted(signature_to_partition.items(), key=lambda kv: kv[0])
    dominated: Dict[Tuple[Tuple[int, ...], ...], List[Tuple[Tuple[int, ...], ...]]] = {}
    for p_sig, p in unique_items:
        for q_sig, q in unique_items:
            if p_sig == q_sig:
                continue
            if is_strict_exact_coarsening(q, p):
                dominated.setdefault(p_sig, []).append(q_sig)

    maximal_items = [(sig, p) for sig, p in unique_items if sig not in dominated]
    dominated_records = []
    for p_sig in sorted(dominated):
        dominators = tuple(sorted(dominated[p_sig]))
        dominated_records.append(
            {
                "dominated_partition_signature": p_sig,
                "dominating_coarsening_signatures": dominators,
                "dominating_coarsening_count": int(len(dominators)),
            }
        )

    diagnostics = {
        "num_partitions_input": int(len(partitions or ())),
        "num_unique_partitions": int(len(unique_items)),
        "num_dominated_partitions_removed": int(len(dominated)),
        "num_maximal_partitions": int(len(maximal_items)),
        "input_partition_signatures": tuple(input_signatures),
        "maximal_partition_signatures": tuple(sig for sig, _ in maximal_items),
        "dominated_partition_records": tuple(dominated_records),
    }
    return [p for _, p in maximal_items], diagnostics

@dataclass
class TwoWaveScenarioResult:
    """
    Metrics for ONE Monte Carlo scenario in the two-wave model.
    All metrics are for the FULL original battle graph B.
    """
    expected_territories: float
    expected_new_territories: float
    expected_troops: float
    expected_troop_loss: float
    conquest_probability: float  # 0 or CP_wave2 if B_k' already all-A


@dataclass
class TwoWaveEvaluation:
    """
    Aggregated two-wave lookahead metrics for a single base partition.

    - base_partition: the partition on the initial battle graph (wave1).
    - base_evaluation: the existing one-wave PartitionEvaluation.
    - scenarios: list of per-scenario results (Monte Carlo samples).
    - coverage_wave1: approximate probability mass captured by the
      kept absorbing states in wave1 regional distributions
      (product of per-region coverages).
    - metrics_*: Monte Carlo averages over scenarios.
    """
    base_partition: List[Dict[str, Any]]
    base_evaluation: Optional["PartitionEvaluation"]


    scenarios: List[TwoWaveScenarioResult]
    coverage_wave1: float

    expected_territories: float
    expected_new_territories: float
    expected_troops: float
    expected_troop_loss: float
    conquest_probability: float


@dataclass
class Wave1ScenarioMicrostate:
    """
    One Monte Carlo sample of the post-wave-1 global state for a given
    base partition on a fixed battle graph.

    This is a "microstate": full GlobalState after resolving all regions
    in the partition once using the small-graph libraries.
    """
    global_state_after_wave1: GlobalState


# ---------------------------------------------------------------------
# Helpers: battle-graph-level basics
# ---------------------------------------------------------------------


def _battle_graph_nodes(battle_graph) -> List[int]:
    """
    Extract node list from a networkx-like graph.
    """
    try:
        nodes_iter = battle_graph.nodes()
    except TypeError:
        nodes_iter = battle_graph.nodes
    return list(nodes_iter)


def _battle_graph_edges(battle_graph) -> List[Tuple[int, int]]:
    """
    Extract edges from a networkx-like graph.
    """
    return agop.edges_from_battle_graph(battle_graph)


def _current_battle_graph_totals(
    global_state: GlobalState,
    battle_nodes: Sequence[int],
) -> Tuple[int, int]:
    """
    Compute current_territories and current_troops for the attacker
    (players[0]) restricted to the battle graph nodes.
    """
    current_territories = 0
    current_troops = 0
    for idx in battle_nodes:
        node = global_state.nodes[idx]
        if node.owner == "A" and node.troops > 0:
            current_territories += 1
            current_troops += node.troops
    return current_territories, current_troops


def _attacker_has_any_troops_on_battle_graph(
    global_state: GlobalState,
    battle_nodes: Sequence[int],
) -> bool:
    """
    Quick check if attacker has any troops on the battle graph at all.
    """
    for idx in battle_nodes:
        node = global_state.nodes[idx]
        if node.owner == "A" and node.troops > 0:
            return True
    return False





def _snapshot_board_state() -> Dict[int, Tuple[Optional[Players.Player], int]]:
    """
    Take a snapshot of the Board owners/troops so we can mutate and restore.
    """
    snap: Dict[int, Tuple[Optional[Players.Player], int]] = {}
    for idx, terr in Board.node_to_territory_dict.items():
        owner = getattr(terr, "_owner", None)
        troops = getattr(terr, "_troops", 0)
        snap[idx] = (owner, troops)
    return snap


def _restore_board_state(snapshot: Dict[int, Tuple[Optional[Players.Player], int]]) -> None:
    """
    Restore a snapshot taken by _snapshot_board_state.
    """
    for idx, (owner, troops) in snapshot.items():
        terr = Board.node_to_territory_dict.get(idx)
        if terr is None:
            continue
        terr._owner = owner
        terr._troops = troops


def _apply_global_state_to_board(global_state: GlobalState,
                                 players: Sequence["Players.Player"]) -> None:
    attacker = players[0] if players else None
    defender = players[1] if len(players) > 1 else None

    for idx, terr in Board.node_to_territory_dict.items():
        if idx < 0 or idx >= len(global_state.nodes):
            continue

        node = global_state.nodes[idx]

        if node.troops <= 0:
            terr._owner = None
            terr._troops = 0
            continue

        if node.owner == "A":
            terr._owner = attacker
            terr._troops = int(node.troops)
        elif node.owner == "D":
            terr._owner = defender
            terr._troops = int(node.troops)
        else:
            terr._owner = None
            terr._troops = int(node.troops)


def _ad_conflict_edges(curr_state: GlobalState, curr_battle_graph):
    try:
        edges = list(curr_battle_graph.edges())
    except TypeError:
        edges = list(curr_battle_graph.edges)

    out = []
    for u, v in edges:
        ou = curr_state.nodes[u].owner
        ov = curr_state.nodes[v].owner
        if ou in ("A", "D") and ov in ("A", "D") and ou != ov:
            out.append((u, v))
    return out


# ---------------------------------------------------------------------
# Fallback: best partial partitions (sub-partitions)
# ---------------------------------------------------------------------


def _build_candidate_regions_for_partial_partitions(
    players: Sequence["Players.Player"],
    battle_graph,
) -> List[Dict[str, Any]]:
    """
    Rebuild the candidate regions (valid small graphs) exactly as in
    approximate_graph_outcome_probabilities.partition_continent_battle_graph_into_valid_small_graphs,
    but without requiring an exact cover.

    Returns a list of region dicts:
        {
            "region_nodes":   tuple(...),
            "attacker_nodes": tuple(...),
            "defender_nodes": tuple(...),
            "pattern":        (nA, nD),
        }
    """
    if not players:
        return []

    attacker = players[0]
    attacker_name = attacker._name

    try:
        nodes_iter = battle_graph.nodes()
    except TypeError:
        nodes_iter = battle_graph.nodes
    all_nodes: List[Any] = list(nodes_iter)

    if not all_nodes:
        return []

    candidate_regions: List[Dict[str, Any]] = []

    max_size = max(a + d for a, d in agop.ALLOWED_PATTERNS)
    min_size = 2  # smallest pattern is (1, 1)

    for size in range(min_size, max_size + 1):
        # all size-k subsets of nodes
        from itertools import combinations

        for subset in combinations(all_nodes, size):
            owner_names = [agop._get_node_owner_name(n) for n in subset]

            # exclude unowned territories
            if any(o is None for o in owner_names):
                continue

            attacker_nodes = [
                n for n, o in zip(subset, owner_names) if o == attacker_name
            ]
            defender_nodes = [
                n for n, o in zip(subset, owner_names)
                if o is not None and o != attacker_name
            ]

            nA = len(attacker_nodes)
            nD = len(defender_nodes)

            if (nA, nD) not in agop.ALLOWED_PATTERNS:
                continue
            if nA == 0 or nD == 0:
                continue
            if not agop._is_connected_subset(battle_graph, subset):
                continue

            candidate_regions.append(
                {
                    "region_nodes":   tuple(subset),
                    "attacker_nodes": tuple(attacker_nodes),
                    "defender_nodes": tuple(defender_nodes),
                    "pattern":        (nA, nD),
                }
            )

    return candidate_regions


def _find_best_partial_partitions(
    players: Sequence["Players.Player"],
    battle_graph,
    max_partitions: int,
) -> List[List[Dict[str, Any]]]:
    """
    Fallback when no full partitions exist:

        - Build candidate regions (valid small graphs).
        - Consider disjoint sets of regions (partitions of a subset of nodes).
        - Among all such sets, keep those that maximize:

              (total_defender_nodes_covered, total_attacker_nodes_covered)

        - Return up to `max_partitions` of these "best" sub-partitions.

    These are *sub-partitions* of the battle graph: they cover only a
    subset of nodes, but each region individually satisfies the same
    validity criteria as in the full-partition function.
    """
    candidate_regions = _build_candidate_regions_for_partial_partitions(
        players=players,
        battle_graph=battle_graph,
    )

    if not candidate_regions:
        return []

    try:
        nodes_iter = battle_graph.nodes()
    except TypeError:
        nodes_iter = battle_graph.nodes
    all_nodes: Set[Any] = set(nodes_iter)

    # Precompute which regions contain which nodes
    regions_by_node: Dict[Any, List[int]] = {n: [] for n in all_nodes}
    for i, region in enumerate(candidate_regions):
        for n in region["region_nodes"]:
            if n in regions_by_node:
                regions_by_node[n].append(i)

    # If some node has no region containing it, it's still okay here
    # since this is a partial-cover search.

    # Precompute owner names per node
    attacker = players[0]
    attacker_name = attacker._name
    node_owner_name: Dict[Any, Optional[str]] = {
        n: agop._get_node_owner_name(n) for n in all_nodes
    }

    best_defenders = -1
    best_attackers = -1
    best_partitions_indices: List[List[int]] = []

    def record_partition(covered: Set[Any], chosen_indices: List[int]) -> None:
        nonlocal best_defenders, best_attackers, best_partitions_indices

        # Compute coverage metrics
        defenders_covered = 0
        attackers_covered = 0
        for n in covered:
            owner = node_owner_name.get(n)
            if owner is None:
                continue
            if owner == attacker_name:
                attackers_covered += 1
            else:
                defenders_covered += 1

        cov_key = (defenders_covered, attackers_covered)
        best_key = (best_defenders, best_attackers)

        if cov_key > best_key:
            best_defenders, best_attackers = cov_key
            best_partitions_indices = [chosen_indices.copy()]
        elif cov_key == best_key and cov_key != (0, 0):
            # Keep up to max_partitions of equally good sub-partitions
            if len(best_partitions_indices) < max_partitions:
                best_partitions_indices.append(chosen_indices.copy())

    # Backtracking to explore disjoint sets of regions
    def backtrack(covered: Set[Any], chosen_indices: List[int]) -> None:
        # Always treat the current selection as a candidate partition
        if chosen_indices:
            record_partition(covered, chosen_indices)

        # Choose next uncovered node that still has possible regions
        remaining_nodes = sorted(all_nodes - covered)
        if not remaining_nodes:
            return

        next_node = remaining_nodes[0]
        region_indices = regions_by_node.get(next_node, [])
        if not region_indices:
            # This node cannot be covered by any region; we simply stop
            # extending here (partial cover).
            return

        for region_idx in region_indices:
            region = candidate_regions[region_idx]
            region_nodes_set = set(region["region_nodes"])
            if region_nodes_set & covered:
                continue

            backtrack(
                covered | region_nodes_set,
                chosen_indices + [region_idx],
            )

    backtrack(set(), [])

    if not best_partitions_indices:
        return []

    # Convert index partitions to region-list partitions
    partitions: List[List[Dict[str, Any]]] = []
    for indices in best_partitions_indices:
        part = [candidate_regions[i] for i in indices]
        partitions.append(part)

    return partitions


# ---------------------------------------------------------------------
# Helper function for region partitioning
# ---------------------------------------------------------------------


def _attempt_region_refinement_to_library_covered_subregions(
    region_nodes: Sequence[int],
    global_state: GlobalState,
    global_edges,
    combat_libraries_base: Path,
) -> Optional[List[Dict[str, Any]]]:
    """
    Try to replace a single out-of-coverage region (region_nodes) with
    a set of smaller regions that:

      - are disjoint,
      - whose union == region_nodes,
      - each block is connected in the battle graph,
      - each block has nA>0 and nD>0,
      - each block's (nA, nD) is in agop.ALLOWED_PATTERNS,
      - each block is covered by a small-graph library
        (agop.query_region_from_libraries succeeds).

    Currently this is primarily aimed at the problematic 5-node
    (3A,2D)/(2A,3D) patterns, but it will gracefully return None
    for other cases.

    Returns:
      list of region dicts (same schema as other regions in partitions)
      or None if no suitable sub-partition could be found.
    """
    region_nodes = tuple(region_nodes)
    n_nodes = len(region_nodes)
    if n_nodes != 5:
        return None

    # Build adjacency restricted to region_nodes from global_edges
    region_set = set(region_nodes)
    adjacency: Dict[int, Set[int]] = {n: set() for n in region_nodes}
    try:
        edges_iter = global_edges
    except TypeError:
        edges_iter = global_edges
    for u, v in edges_iter:
        if u in region_set and v in region_set:
            adjacency[u].add(v)
            adjacency[v].add(u)

    # Helper: check connectivity of a block via simple BFS
    def _is_connected_block(block: Sequence[int]) -> bool:
        block = list(block)
        if not block:
            return False
        if len(block) == 1:
            return True
        block_set = set(block)
        start = block[0]
        visited = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adjacency.get(x, ()):
                if y in block_set and y not in visited:
                    visited.add(y)
                    stack.append(y)
        return visited == block_set

    # Owner classification on the region (from global_state)
    def _owners_on_nodes(nodes: Sequence[int]) -> Tuple[List[int], List[int]]:
        attackers: List[int] = []
        defenders: List[int] = []
        for idx in nodes:
            node = global_state.nodes[idx]
            if node.owner == "A" and node.troops > 0:
                attackers.append(idx)
            elif node.owner == "D" and node.troops > 0:
                defenders.append(idx)
        return attackers, defenders

    # Check that this region is of the problematic pattern (3,2) or (2,3)
    attackers_all, defenders_all = _owners_on_nodes(region_nodes)
    nA_all = len(attackers_all)
    nD_all = len(defenders_all)
    if (nA_all, nD_all) not in {(3, 2), (2, 3)}:
        return None

    # --- Generate set partitions of region_nodes ---

    def _set_partitions(elems: List[int]):
        """
        Yield all set partitions of elems as lists of blocks (each block is a list).
        Bell(5) = 52, so brute force is fine.
        """
        if not elems:
            yield []
            return
        first, *rest = elems
        for partition in _set_partitions(rest):
            # Put 'first' in an existing block
            for i in range(len(partition)):
                new_block = partition[i] + [first]
                yield partition[:i] + [new_block] + partition[i + 1 :]
            # Or start a new block
            yield [[first]] + partition

    elems = list(region_nodes)

    for partition in _set_partitions(elems):
        # We don't want the trivial one-block partition (the original region)
        if len(partition) <= 1:
            continue

        candidate_subregions: List[Dict[str, Any]] = []
        ok_partition = True

        for block in partition:
            block_nodes = tuple(block)

            # Connectivity
            if not _is_connected_block(block_nodes):
                ok_partition = False
                break

            # Owners and pattern
            attackers, defenders = _owners_on_nodes(block_nodes)
            nA = len(attackers)
            nD = len(defenders)

            if nA == 0 or nD == 0:
                ok_partition = False
                break
            if (nA, nD) not in agop.ALLOWED_PATTERNS:
                ok_partition = False
                break

            # Check library coverage for this block
            try:
                # Note: we only care that this does NOT raise the "no library" error.
                _ = agop.query_region_from_libraries(
                    combat_libraries_base=combat_libraries_base,
                    global_state=global_state,
                    global_edges=global_edges,
                    region_nodes=block_nodes,
                )
            except ValueError as e:
                msg = str(e)
                if "No library found for nA=" in msg or "with maxA>=" in msg:
                    ok_partition = False
                    break
                else:
                    # Unexpected error -> propagate
                    raise
            # Build region dict for this block
            candidate_subregions.append(
                {
                    "region_nodes": block_nodes,
                    "attacker_nodes": tuple(attackers),
                    "defender_nodes": tuple(defenders),
                    "pattern": (nA, nD),
                }
            )

        if ok_partition:
            warnings.warn(
                "Refined out-of-coverage 5-node region into "
                f"{len(candidate_subregions)} library-covered subregions. "
                "Results for this macro-state may be approximate.",
                RuntimeWarning,
            )
            return candidate_subregions

    # No suitable sub-partition found
    return None



# ---------------------------------------------------------------------
# Metrics per region / partition
# ---------------------------------------------------------------------


def _v2_owner_to_char(x: int) -> str:
    # Convention: 0=D, 1=A
    return "A" if int(x) == 1 else "D"

def _v2_overlay_outcome_into_global_nodes(
    nodes_after: list,
    *,
    owners_row: np.ndarray,   # (M,)
    troops_row: np.ndarray,   # (M,)
    mapping: Dict[int, int],  # local_canonical_idx -> global_idx
) -> None:
    for local_idx in range(int(owners_row.shape[0])):
        global_idx = mapping.get(local_idx)
        if global_idx is None:
            continue
        owner = _v2_owner_to_char(int(owners_row[local_idx]))
        troops = int(troops_row[local_idx])
        nodes_after[global_idx] = NodeState(owner, troops)

def _v2_compute_metrics_from_arrays(
    *,
    p: np.ndarray,                 # (N,)
    owners: np.ndarray,            # (N,M)
    troops: np.ndarray,            # (N,M)
    initial_owner_is_defender: np.ndarray,  # (M,) bool, in local canonical order
) -> tuple[float, float, float]:
    """
    Returns:
      expected_new_territories, expected_final_attacker_troops, p_conquer
    """
    # new territories: nodes that were defender initially, end attacker with troops>0
    end_attacker = (owners == 1) & (troops > 0)
    new_terr = (end_attacker & initial_owner_is_defender[None, :]).sum(axis=1).astype(np.float64)

    # final attacker troops in region
    final_att_troops = (troops * (owners == 1)).sum(axis=1).astype(np.float64)

    # conquer: all nodes attacker with troops>0? (keep your conquer definition here)
    is_conquer = (owners == 1).all(axis=1).astype(np.float64)

    p64 = p.astype(np.float64, copy=False)
    return (
        float(np.dot(p64, new_terr)),
        float(np.dot(p64, final_att_troops)),
        float(np.dot(p64, is_conquer)),
    )

def _evaluate_partition_metrics(
    partition_regions: Sequence[Dict[str, Any]],
    global_state: GlobalState,
    global_edges,
    battle_nodes: Sequence[int],
    combat_libraries_base: Path,
    current_territories: int,
    current_troops: int,
    *,
    policy_option_selection: Any = "primary",
    ranking_variable: Any = "expected_territories",
    region_query_cache: Optional[Any] = None,
) -> Optional[PartitionEvaluation]:
    """
    Compute regional and full-graph metrics for one partition.

    PATCHES IN THIS VERSION
    -----------------------
    1) Uses query_region_from_libraries()'s returned mapping (local_canonical_idx -> global_idx)
       instead of recomputing mapping via reindex_region_nodes(). This guarantees consistency
       with isolate pruning, topology canonicalization, and star-only gating.

    2) Supports BOTH:
         - Legacy V1 distributions: result["probabilities"] = {col_label: p, ...}
         - New V2 precomputed-metrics payload: result["outcomes_v2"] with arrays:
              p, is_conquered, new_territories, final_attacker_troops

       If outcomes_v2 is present, we use it (fast path) and DO NOT decode col_labels.

    3) Treats sentinel/empty distributions consistently:
         - If probabilities == {} (including prob_table==0.0 sentinel) OR outcomes_v2 empty
           => "no effect":
              expected_new_region = 0
              expected_final_troops_region = attacker_troops_in_region
              p_conquer_region = 0

    4) Coverage failures are "zero effect" and include:
         - missing library file
         - missing row label
         - topology gating / pattern missing errors surfaced as ValueError
    """
    if not partition_regions:
        return None

    # Union of region nodes and full-battle coverage flag
    region_nodes_union: Set[int] = set()
    for r in partition_regions:
        region_nodes_union.update(r["region_nodes"])
    covers_all_nodes = set(battle_nodes) == region_nodes_union

    # Precompute initial attacker troops in all regions (union)
    initial_attacker_troops_in_regions = 0
    for idx in region_nodes_union:
        node = global_state.nodes[idx]
        if node.owner == "A" and node.troops > 0:
            initial_attacker_troops_in_regions += node.troops

    expected_new_territories = 0.0
    expected_final_troops_in_regions = 0.0
    regional_product_conquest_probability = 1.0
    region_policy_options: List[Dict[str, Any]] = []

    # Normalize edges iterable once
    try:
        edges_iter = global_edges
    except TypeError:
        edges_iter = global_edges
    edges_list = list(edges_iter)

    for region in partition_regions:
        region_nodes = region["region_nodes"]

        # Initial attacker troops in THIS region (used if we have no probs)
        attacker_troops_in_region = 0
        for idx in region_nodes:
            node = global_state.nodes[idx]
            if node.owner == "A" and node.troops > 0:
                attacker_troops_in_region += node.troops

        # Query probabilities for this region
        try:
            result = _query_region_selected_option(
                combat_libraries_base=combat_libraries_base,
                global_state=global_state,
                global_edges=edges_list,
                region_nodes=region_nodes,
                policy_option_selection=policy_option_selection,
                ranking_variable=ranking_variable,
                query_cache=region_query_cache,
            )
        except (FileNotFoundError, ValueError) as e:
            msg = str(e)
            coverage_like = (
                isinstance(e, FileNotFoundError)
                or "No library found for nA=" in msg
                or "not found in library" in msg
                or "Row label" in msg
                or "not found in prob_table" in msg
                or "STAR_ONLY pattern" in msg
                or "Region has no internal edges" in msg
            )
            if not coverage_like:
                raise

            expected_new_region = 0.0
            expected_final_troops_region = float(attacker_troops_in_region)
            p_conquer_region = 0.0

        else:
            region_policy_options.append(
                {
                    "region_nodes": tuple(result.get("region_nodes_effective", tuple(region_nodes))),
                    "attacker_nodes": tuple(region.get("attacker_nodes", ())),
                    "defender_nodes": tuple(region.get("defender_nodes", ())),
                    "pattern": result.get("pattern", region.get("pattern")),
                    "row_label": result.get("row_label"),
                    "policy_option_count": result.get("policy_option_count"),
                    "selected_policy_option_index": result.get("selected_policy_option_index"),
                    "selected_policy_option_selection": result.get("selected_policy_option_selection"),
                    "selected_policy_option_split_metadata": result.get("selected_policy_option_split_metadata"),
                    "policy_option_summaries": tuple(result.get("policy_option_summaries", ()) or ()),
                }
            )
            outcomes_v2 = result.get("outcomes_v2", None)

            # ----------------------------
            # V2 fast path: precomputed metrics
            # ----------------------------
            if outcomes_v2 is not None:
                p_arr = outcomes_v2.get("p", None)
                is_conq = outcomes_v2.get("is_conquered", None)
                new_terr = outcomes_v2.get("new_territories", None)
                final_att = outcomes_v2.get("final_attacker_troops", None)

                # Treat missing/empty as no effect
                if p_arr is None or is_conq is None or new_terr is None or final_att is None:
                    expected_new_region = 0.0
                    expected_final_troops_region = float(attacker_troops_in_region)
                    p_conquer_region = 0.0
                else:
                    # p_arr etc can be numpy arrays or lists; rely on Python/numpy ops safely
                    # We do explicit float() wrapping for totals.
                    try:
                        # expected new territories in region
                        expected_new_region = float((p_arr * new_terr).sum())  # numpy
                    except Exception:
                        expected_new_region = float(sum(float(p) * float(nt) for p, nt in zip(p_arr, new_terr)))

                    try:
                        # expected final attacker troops in region
                        expected_final_troops_region = float((p_arr * final_att).sum())
                    except Exception:
                        expected_final_troops_region = float(
                            sum(float(p) * float(ft) for p, ft in zip(p_arr, final_att))
                        )

                    try:
                        # probability region conquered
                        p_conquer_region = float((p_arr * is_conq).sum())
                    except Exception:
                        p_conquer_region = float(
                            sum(float(p) * (1.0 if bool(ic) else 0.0) for p, ic in zip(p_arr, is_conq))
                        )

                    # Defensive: if arrays were empty or invalid => no effect
                    if not np.isfinite(expected_new_region):
                        expected_new_region = 0.0
                    if not np.isfinite(expected_final_troops_region):
                        expected_final_troops_region = float(attacker_troops_in_region)
                    if not np.isfinite(p_conquer_region):
                        p_conquer_region = 0.0

            # ----------------------------
            # Legacy path: dict-of-label probabilities
            # ----------------------------
            else:
                prob_row: Dict[str, float] = result.get("probabilities", {}) or {}
                mapping: Dict[int, int] = result.get("mapping", {}) or {}

                if not prob_row or not mapping:
                    expected_new_region = 0.0
                    expected_final_troops_region = float(attacker_troops_in_region)
                    p_conquer_region = 0.0
                else:
                    expected_new_region = 0.0
                    expected_final_troops_region = 0.0
                    p_conquer_region = 0.0

                    for col_label, p in prob_row.items():
                        p = float(p)
                        if p <= 0.0:
                            continue

                        local_end_state = agop.global_state_from_row_label(col_label)

                        new_territories = 0
                        final_troops_region = 0

                        # mapping: local_canonical_idx -> global_idx
                        for local_idx, node_after in enumerate(local_end_state.nodes):
                            global_idx = mapping.get(local_idx)
                            if global_idx is None:
                                continue

                            node_before = global_state.nodes[global_idx]

                            # New territory: was defender, ends attacker
                            if (
                                node_before.owner == "D"
                                and node_after.owner == "A"
                                and node_after.troops > 0
                            ):
                                new_territories += 1

                            # Final attacker troops in region
                            if node_after.owner == "A" and node_after.troops > 0:
                                final_troops_region += node_after.troops

                        expected_new_region += p * float(new_territories)
                        expected_final_troops_region += p * float(final_troops_region)

                        if agop.is_conquered_state_label(col_label):
                            p_conquer_region += p

        # Accumulate regional contributions
        expected_new_territories += float(expected_new_region)
        expected_final_troops_in_regions += float(expected_final_troops_region)
        regional_product_conquest_probability *= float(p_conquer_region)

    # Compute expected lost troops in regions
    expected_lost_troops = float(
        initial_attacker_troops_in_regions - expected_final_troops_in_regions
    )
    if expected_lost_troops < 0.0:
        expected_lost_troops = 0.0

    # Full-graph metrics
    expected_territories = float(current_territories + expected_new_territories)
    expected_troops = float(current_troops - expected_lost_troops)

    conquest_probability = (
        float(regional_product_conquest_probability) if covers_all_nodes else 0.0
    )

    return PartitionEvaluation(
        partition=list(partition_regions),
        covers_all_nodes=covers_all_nodes,
        expected_new_territories=float(expected_new_territories),
        expected_lost_troops=float(expected_lost_troops),
        regional_product_conquest_probability=float(regional_product_conquest_probability),
        expected_territories=float(expected_territories),
        expected_troops=float(expected_troops),
        conquest_probability=float(conquest_probability),
        region_policy_options=tuple(region_policy_options),
    )



@dataclass
class RegionOutcome:
    """
    One absorbing outcome for a region, suitable for scenario building.

    Supports:
      - V1: local_state is populated (GlobalState)
      - V2: owners_row/troops_row are populated (arrays for one outcome)

    mapping:
        dict[local_index -> global_index] mapping for this region.
    """
    probability: float
    mapping: Dict[int, int]
    col_label: str  # debug/inspection

    # V1
    local_state: Optional[GlobalState] = None

    # V2 (one outcome row)
    owners_row: Optional[np.ndarray] = None   # shape (M,)
    troops_row: Optional[np.ndarray] = None   # shape (M,)
    owner_codes: Optional[Dict[str, int]] = None  # e.g. {"A":0,"D":1}
    format_version: int = 1




@dataclass
class RegionOutcomeSet:
    """
    Collection of outcomes for one region plus coverage statistics.
    """
    region_nodes: Tuple[int, ...]
    outcomes: List[RegionOutcome]
    total_mass: float       # sum of all prob_row entries (≈1.0)
    kept_mass: float        # sum of probabilities of kept outcomes
    mapping: Dict[int, int] # local_index -> global_index


# ---------------------------------------------------------------------
# Wave1 absorbing distributions for a base partition
# ---------------------------------------------------------------------


def _node_state_from_owner_code(owner_code: int, troops: int) -> NodeState:
    # Owner encoding: 0 = D, 1 = A
    # If you ever introduce neutral, handle here.
    if troops <= 0:
        # Keep semantics consistent with your board rules:
        # if troops==0, ownership doesn't matter much, but we keep D for determinism.
        return NodeState("D", 0)
    return NodeState("A", int(troops)) if int(owner_code) == 1 else NodeState("D", int(troops))



def _collect_wave1_region_distributions(
    partition_regions: Sequence[Dict[str, Any]],
    global_state: GlobalState,
    global_edges,
    combat_libraries_base: Path,
    min_state_prob: float = 0.0,
    max_end_states_per_region: Optional[int] = None,
    *,
    policy_option_selection: Any = "primary",
    ranking_variable: Any = "expected_territories",
    # Legacy knobs (kept for call-compatibility). Debug output is controlled by log_config.
    debug: bool = False,
    debug_limit: int = 20,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Collect per-region outcome distributions for wave-1 sampling.

    Debug output is emitted via the standard logging system (logger: ``risk.sampler``).
    The ``debug`` argument is retained for backward compatibility but does not
    control output; use ``log_config.set_debug_switches({'sampler': True})`` and
    ``setup_logging(file_level='DEBUG')`` instead.
    """
    log = logging.getLogger("risk.sampler")

    regions_info: List[Dict[str, Any]] = []
    coverage_wave1 = 1.0

    # Debug gating is controlled by log_config (RiskSubsystemFilter).
    do_dbg = log.isEnabledFor(logging.DEBUG)

    # Robust edge extraction
    try:
        edges_iter = global_edges
    except TypeError:
        edges_iter = global_edges
    edges_list = list(edges_iter)

    if do_dbg:
        log.debug(
            "[collect_dist] regions=%d min_state_prob=%s max_end_states_per_region=%s libraries_base=%s",
            len(partition_regions),
            min_state_prob,
            max_end_states_per_region,
            combat_libraries_base,
        )

    for r_i, region in enumerate(partition_regions):
        region_nodes = region.get("region_nodes", ())
        attacker_nodes = region.get("attacker_nodes", ()) or ()
        defender_nodes = region.get("defender_nodes", ()) or ()
        pattern = region.get("pattern", None)

        if do_dbg:
            try:
                rn = len(region_nodes)
                an = len(attacker_nodes)
                dn = len(defender_nodes)
            except Exception:
                rn, an, dn = -1, -1, -1
            log.debug("[collect_dist] region %d: nodes=%s A=%s D=%s pattern=%s", r_i, rn, an, dn, pattern)

        # ----------------------------------------------------------
        # Skip invalid / non-combat regions (nA==0 or nD==0)
        # ----------------------------------------------------------
        if (len(attacker_nodes) == 0) or (len(defender_nodes) == 0):
            if do_dbg:
                reason = "no_attacker" if len(attacker_nodes) == 0 else "no_defender"
                log.debug("[collect_dist]   SKIP region %d: invalid combat region (%s) pattern=%s", r_i, reason, pattern)

            regions_info.append(
                {
                    "region_nodes": tuple(region_nodes),
                    "mapping": {},
                    "payload": {},
                    "skipped": True,
                    "skip_reason": "nA0_or_nD0",
                }
            )
            continue

        try:
            result = _query_region_selected_option(
                combat_libraries_base=combat_libraries_base,
                global_state=global_state,
                global_edges=edges_list,
                region_nodes=region_nodes,
                # query_region_from_libraries also logs via risk.query
                debug=False,
                debug_limit=debug_limit,
                policy_option_selection=policy_option_selection,
                ranking_variable=ranking_variable,
            )
        except (FileNotFoundError, ValueError) as e:
            # Region not covered => record empty.
            coverage_wave1 *= 0.0
            if do_dbg:
                log.debug("[collect_dist]   QUERY_FAIL region %d: %s: %s", r_i, type(e).__name__, e)
            regions_info.append(
                {
                    "region_nodes": tuple(region_nodes),
                    "mapping": {},
                    "payload": {},
                    "error": str(e),
                }
            )
            continue

        mapping: Dict[int, int] = result.get("mapping", {}) or {}
        payload: Dict[str, Any] = result.get("payload", {}) or {}
        effective_nodes = tuple(result.get("region_nodes_effective", tuple(region_nodes)))

        # Adapter (robustness): accept outcomes_v2 if payload missing
        if (not payload) and (result.get("outcomes_v2") is not None):
            payload = result["outcomes_v2"] or {}

        if do_dbg:
            log.debug(
                "[collect_dist]   query_ok region %d: effective_nodes=%d mapping_size=%d payload_keys=%s",
                r_i,
                len(effective_nodes),
                len(mapping),
                sorted(list(payload.keys())) if payload else [],
            )

        if not payload or not mapping:
            coverage_wave1 *= 0.0
            regions_info.append({"region_nodes": effective_nodes, "mapping": mapping, "payload": {}})
            continue

        if "_legacy_prob_row" in payload:
            raise NotImplementedError(
                "Hard switch to v2 chunked storage: legacy probability rows are not supported. "
                "Rebuild the relevant libraries in v2 format."
            )

        p = np.asarray(payload.get("p", []), dtype=np.float64)
        owners = np.asarray(payload.get("owners", []))
        troops = np.asarray(payload.get("troops", []))

        if p.size == 0:
            coverage_wave1 *= 0.0
            regions_info.append({"region_nodes": effective_nodes, "mapping": mapping, "payload": {}})
            continue

        total_mass = float(p.sum())
        if total_mass <= 0.0:
            coverage_wave1 *= 0.0
            regions_info.append({"region_nodes": effective_nodes, "mapping": mapping, "payload": {}})
            continue

        idx = np.arange(p.size)

        idx_after_prob = idx
        if min_state_prob > 0.0:
            idx_after_prob = idx_after_prob[p[idx_after_prob] >= float(min_state_prob)]
            if idx_after_prob.size == 0:
                coverage_wave1 *= 0.0
                regions_info.append({"region_nodes": effective_nodes, "mapping": mapping, "payload": {}})
                continue

        idx_after_cap = idx_after_prob
        if max_end_states_per_region is not None and idx_after_cap.size > int(max_end_states_per_region):
            idx_after_cap = idx_after_cap[np.argsort(p[idx_after_cap])[::-1][: int(max_end_states_per_region)]]

        kept_mass = float(p[idx_after_cap].sum())
        if kept_mass <= 0.0:
            coverage_wave1 *= 0.0
            regions_info.append({"region_nodes": effective_nodes, "mapping": mapping, "payload": {}})
            continue

        p2 = (p[idx_after_cap] / kept_mass).astype(np.float32)
        owners2 = owners[idx_after_cap]
        troops2 = troops[idx_after_cap]

        # store/refresh CDF for fast sampling
        cdf = np.cumsum(p2, dtype=np.float64).astype(np.float32)
        payload2 = dict(payload)
        payload2["p"] = p2
        payload2["owners"] = owners2
        payload2["troops"] = troops2
        payload2["cdf"] = cdf
        payload2["selected_policy_option_index"] = result.get("selected_policy_option_index", payload2.get("option_id", 0))
        payload2["selected_policy_option_selection"] = result.get("selected_policy_option_selection", _normalize_policy_option_selection(policy_option_selection))
        payload2["policy_option_count"] = result.get("policy_option_count", len(result.get("policy_options_v2", []) or []))

        coverage_factor = (kept_mass / total_mass)
        coverage_wave1 *= coverage_factor

        if do_dbg:
            try:
                pmax = float(np.max(p2)) if p2.size else 0.0
            except Exception:
                pmax = float("nan")
            log.debug(
                "[collect_dist]   KEEP region %d: n_raw=%d n_after_prob=%d n_after_cap=%d "
                "total_mass=%.6f kept_mass=%.6f coverage_factor=%.6f p_max=%.6f",
                r_i,
                int(idx.size),
                int(idx_after_prob.size),
                int(idx_after_cap.size),
                total_mass,
                kept_mass,
                coverage_factor,
                pmax,
            )

        regions_info.append(
            {
                "region_nodes": effective_nodes,
                "mapping": mapping,
                "payload": payload2,
            }
        )

    if do_dbg:
        log.debug("[collect_dist] DONE: coverage_wave1=%.6f", float(coverage_wave1))

    return regions_info, float(coverage_wave1)

def sample_wave1_microstates_for_partition(
    players: Sequence["Players.Player"],
    continent_name: str,
    partition_regions: Sequence[Dict[str, Any]],
    global_state: GlobalState,
    battle_graph,
    combat_libraries_base: Path,
    num_scenarios: int = 50,
    min_state_prob: float = 0.0,
    max_end_states_per_region: Optional[int] = None,
    *,
    rng: Optional["random.Random"] = None,
    policy_option_selection: Any = "primary",
    ranking_variable: Any = "expected_territories",
    # Legacy knobs (kept for call-compatibility). Debug output is controlled by log_config.
    debug: bool = True,
    debug_limit: int = 20,
) -> Tuple[List[Wave1ScenarioMicrostate], float]:
    """
    Wave-1 sampler.

    All debug output is emitted via the standard logging system (logger: ``risk.sampler``).
    The ``debug`` argument is retained for backward compatibility but does not
    control output; use ``log_config.set_debug_switches({'sampler': True})`` and
    ``setup_logging(file_level='DEBUG')`` instead.
    """
    import random
    import numpy as np

    log = logging.getLogger("risk.sampler")

    # Trivial empties
    if not partition_regions:
        return [], 0.0

    # Debug gating is controlled by log_config (RiskSubsystemFilter).
    do_dbg = log.isEnabledFor(logging.DEBUG)
    # Robust edge extraction
    try:
        edges_iter = battle_graph.edges()
    except TypeError:
        edges_iter = battle_graph.edges
    global_edges = list(edges_iter)

    def _reg_shape(reg: Dict[str, Any]) -> str:
        try:
            rn = len(reg.get("region_nodes", ()) or ())
            an = len(reg.get("attacker_nodes", ()) or ())
            dn = len(reg.get("defender_nodes", ()) or ())
            pat = reg.get("pattern", None)
            return f"nodes={rn} A={an} D={dn} pattern={pat}"
        except Exception:
            return "shape=?"

    # Battle graph diagnostics
    if do_dbg:
        try:
            bn = len(list(_battle_graph_nodes(battle_graph)))
        except Exception:
            bn = -1
        try:
            be = int(battle_graph.number_of_edges()) if hasattr(battle_graph, "number_of_edges") else -1
        except Exception:
            be = -1

        log.debug(
            "[sampler] continent=%s regions=%d battle_nodes=%s battle_edges=%s num_scenarios=%s "
            "min_state_prob=%s max_end_states_per_region=%s libraries_base=%s",
            continent_name,
            len(partition_regions),
            bn,
            be,
            num_scenarios,
            min_state_prob,
            max_end_states_per_region,
            combat_libraries_base,
        )
        for i, reg in enumerate(partition_regions):
            log.debug("[sampler] region %d: %s", i, _reg_shape(reg))

    regions_info, coverage_wave1 = _collect_wave1_region_distributions(
        partition_regions=partition_regions,
        global_state=global_state,
        global_edges=global_edges,
        combat_libraries_base=combat_libraries_base,
        min_state_prob=min_state_prob,
        max_end_states_per_region=max_end_states_per_region,
        policy_option_selection=policy_option_selection,
        ranking_variable=ranking_variable,
        debug=False,
        debug_limit=debug_limit,
    )

    # Harden coverage semantics and empty cases
    if not regions_info:
        if do_dbg:
            log.debug("[sampler] RETURNING EMPTY: regions_info is empty.")
        return [], 0.0

    try:
        cov = float(coverage_wave1)
    except Exception:
        cov = 0.0
    cov = 0.0 if cov < 0.0 else (1.0 if cov > 1.0 else cov)
    coverage_wave1 = cov

    if do_dbg:
        log.debug("[sampler] coverage_wave1=%.6f regions_info=%d", float(coverage_wave1), len(regions_info))

        for i, r_info in enumerate(regions_info):
            payload = (r_info.get("payload", {}) or {})
            mapping = (r_info.get("mapping", {}) or {})
            has_payload = int(bool(payload))
            has_mapping = int(bool(mapping))

            payload_type = "none"
            if payload:
                if "_legacy_prob_row" in payload:
                    payload_type = "legacy"
                elif ("p" in payload) or ("owners" in payload) or ("troops" in payload):
                    payload_type = "v2"
                else:
                    payload_type = "unknown"

            log.debug(
                "[sampler] region_info %d: has_payload=%d has_mapping=%d payload_type=%s mapping_size=%d",
                i,
                has_payload,
                has_mapping,
                payload_type,
                len(mapping),
            )

            try:
                if payload_type == "legacy":
                    prob_row = payload.get("_legacy_prob_row", {}) or {}
                    n_labels = len(prob_row)
                    sprob = float(sum(float(v) for v in prob_row.values())) if prob_row else 0.0
                    pmax = float(max((float(v) for v in prob_row.values()), default=0.0))
                    log.debug("[sampler]   legacy: n_labels=%d sum_prob=%.6f max_prob=%.6f", n_labels, sprob, pmax)
                elif payload_type == "v2":
                    p = np.asarray(payload.get("p", []), dtype=np.float64)
                    cdf = payload.get("cdf")
                    cdf_arr = np.asarray(cdf, dtype=np.float64) if cdf is not None else None
                    psum = float(np.sum(p)) if p.size else 0.0
                    pmax = float(np.max(p)) if p.size else 0.0
                    log.debug(
                        "[sampler]   v2: p_size=%d cdf_size=%s p_sum=%.6f p_max=%.6f",
                        int(p.size),
                        (int(cdf_arr.size) if cdf_arr is not None else "None"),
                        psum,
                        pmax,
                    )
            except Exception as e:
                log.debug("[sampler]   (payload stats failed: %s: %s)", type(e).__name__, e)

    # "coverage cannot lie" rule
    if all(not (r.get("payload") or {}) for r in regions_info):
        if do_dbg:
            log.debug("[sampler] RETURNING EMPTY: all regions had empty payloads -> forcing coverage=0.0")
        return [], 0.0

    if rng is None:
        rng = random.Random()

    scenarios: List[Wave1ScenarioMicrostate] = []

    max_overlay_checks = 5
    overlay_checks_done = 0
    max_delta_checks = 5
    delta_checks_done = 0

    for scen_draw in range(num_scenarios):
        nodes_after_wave1 = list(global_state.nodes)

        for r_info_idx, r_info in enumerate(regions_info):
            payload: Dict[str, Any] = r_info.get("payload", {}) or {}
            mapping: Dict[int, int] = r_info.get("mapping", {}) or {}

            if not payload or not mapping:
                continue

            # Legacy sampling
            if "_legacy_prob_row" in payload:
                prob_row: Dict[str, float] = payload["_legacy_prob_row"] or {}
                if not prob_row:
                    continue

                labels = sorted(prob_row.keys())
                probs = [float(prob_row[lbl]) for lbl in labels]

                s = float(sum(probs))
                if s <= 0.0:
                    continue
                if abs(s - 1.0) > 1e-6:
                    inv = 1.0 / s
                    probs = [p * inv for p in probs]

                x = rng.random()
                cum = 0.0
                chosen_label = labels[-1]
                for lbl, pval in zip(labels, probs):
                    cum += pval
                    if x <= cum:
                        chosen_label = lbl
                        break

                local_end_state = agop.global_state_from_row_label(chosen_label)
                for local_idx, node_after in enumerate(local_end_state.nodes):
                    global_idx = mapping.get(local_idx)
                    if global_idx is None:
                        continue
                    nodes_after_wave1[global_idx] = node_after
                continue

            # V2 sampling
            p = np.asarray(payload.get("p", []), dtype=np.float64)
            owners = np.asarray(payload.get("owners", []))
            troops = np.asarray(payload.get("troops", []))
            if p.size == 0:
                continue

            cdf = payload.get("cdf")
            if cdf is None:
                cdf = np.cumsum(p, dtype=np.float64)
            cdf = np.asarray(cdf, dtype=np.float64)

            x = rng.random()
            k = int(np.searchsorted(cdf, x, side="right"))
            if k >= int(p.size):
                k = int(p.size) - 1

            if do_dbg and overlay_checks_done < max_overlay_checks:
                overlay_checks_done += 1
                try:
                    owners_row = owners[k]
                    troops_row = troops[k]

                    local_len = len(mapping)
                    owners_len = len(owners_row) if hasattr(owners_row, "__len__") else -1
                    troops_len = len(troops_row) if hasattr(troops_row, "__len__") else -1

                    min_key = min(mapping.keys()) if mapping else None
                    max_key = max(mapping.keys()) if mapping else None

                    log.debug(
                        "[sampler] overlay_check scen=%d region_info=%d k=%d local_len=%d owners_len=%s troops_len=%s map_key_range=(%s,%s)",
                        scen_draw,
                        r_info_idx,
                        k,
                        local_len,
                        owners_len,
                        troops_len,
                        min_key,
                        max_key,
                    )
                    if owners_len != -1 and local_len != owners_len:
                        log.warning(
                            "[sampler][WARN] owners_len != local_len (owners_len=%s, local_len=%s) -> overlay likely broken",
                            owners_len,
                            local_len,
                        )
                    if troops_len != -1 and local_len != troops_len:
                        log.warning(
                            "[sampler][WARN] troops_len != local_len (troops_len=%s, local_len=%s) -> overlay likely broken",
                            troops_len,
                            local_len,
                        )
                except Exception as e:
                    log.debug("[sampler] overlay_check failed: %s: %s", type(e).__name__, e)

            _v2_overlay_outcome_into_global_nodes(
                nodes_after_wave1,
                owners_row=owners[k],
                troops_row=troops[k],
                mapping=mapping,
            )

        if do_dbg and delta_checks_done < max_delta_checks:
            delta_checks_done += 1
            try:
                changed = 0
                changed_idxs: List[int] = []
                for i, (a, b) in enumerate(zip(global_state.nodes, nodes_after_wave1)):
                    if (a.owner != b.owner) or (int(a.troops) != int(b.troops)):
                        changed += 1
                        if len(changed_idxs) < 10:
                            changed_idxs.append(i)

                log.debug(
                    "[sampler] scenario_delta scen=%d: changed_nodes=%d sample_changed_idxs=%s",
                    scen_draw,
                    changed,
                    changed_idxs,
                )
                if changed_idxs:
                    i0 = changed_idxs[0]
                    a0 = global_state.nodes[i0]
                    b0 = nodes_after_wave1[i0]
                    log.debug(
                        "[sampler]   example_change node=%d: %s%d -> %s%d",
                        i0,
                        a0.owner,
                        int(a0.troops),
                        b0.owner,
                        int(b0.troops),
                    )
            except Exception as e:
                log.debug("[sampler] scenario_delta check failed: %s: %s", type(e).__name__, e)

        scenarios.append(
            Wave1ScenarioMicrostate(
                global_state_after_wave1=GlobalState(nodes=tuple(nodes_after_wave1))
            )
        )

    if not scenarios:
        return [], 0.0

    return scenarios, float(coverage_wave1)

def _collect_region_outcomes_for_partition(
    global_state: GlobalState,
    global_edges,
    region: Dict[str, Any],
    combat_libraries_base: Path,
    min_state_prob: float,
    max_end_states_per_region: int,
    enable_state_capping: bool = True,
    state_cap_coverage_threshold: float = 0.9,
    *,
    policy_option_selection: Any = "primary",
    ranking_variable: Any = "expected_territories",
) -> RegionOutcomeSet:
    """
    Final consolidated version: supports V1 and V2 payloads without downstream changes.
    """
    region_nodes = tuple(region["region_nodes"])

    try:
        edges_iter = global_edges
    except TypeError:
        edges_iter = global_edges
    edges_list = list(edges_iter)

    try:
        result = _query_region_selected_option(
            combat_libraries_base=combat_libraries_base,
            global_state=global_state,
            global_edges=edges_list,
            region_nodes=region_nodes,
            policy_option_selection=policy_option_selection,
            ranking_variable=ranking_variable,
        )
    except (FileNotFoundError, ValueError) as e:
        msg = str(e)
        coverage_like = (
            isinstance(e, FileNotFoundError)
            or "No library found for nA=" in msg
            or "not found in library" in msg
            or "Row label" in msg
            or "not found in prob_table" in msg
        )
        if not coverage_like:
            raise

        return RegionOutcomeSet(region_nodes=region_nodes, outcomes=[], total_mass=0.0, kept_mass=0.0, mapping={})

    mapping: Dict[int, int] = result.get("mapping", {}) or {}
    payload: Dict[str, Any] = result.get("payload", {}) or {}

    if not payload or not mapping:
        return RegionOutcomeSet(region_nodes=region_nodes, outcomes=[], total_mass=0.0, kept_mass=0.0, mapping=mapping)

    # Legacy
    if "_legacy_prob_row" in payload:
        prob_row: Dict[str, float] = payload["_legacy_prob_row"] or {}
        if not prob_row:
            return RegionOutcomeSet(region_nodes=region_nodes, outcomes=[], total_mass=0.0, kept_mass=0.0, mapping=mapping)

        items = [(lbl, float(p)) for lbl, p in prob_row.items() if float(p) > 0.0]
        items.sort(key=lambda kv: kv[1], reverse=True)
        total_mass = float(sum(p for _, p in items))
        if total_mass <= 0.0:
            return RegionOutcomeSet(region_nodes=region_nodes, outcomes=[], total_mass=0.0, kept_mass=0.0, mapping=mapping)

        if min_state_prob and min_state_prob > 0.0:
            items = [(lbl, p) for (lbl, p) in items if p >= float(min_state_prob)]

        if not items:
            return RegionOutcomeSet(region_nodes=region_nodes, outcomes=[], total_mass=total_mass, kept_mass=0.0, mapping=mapping)

        kept_mass = float(sum(p for _, p in items))

        # Optional dynamic capping (legacy)
        if enable_state_capping and max_end_states_per_region and len(items) > max_end_states_per_region:
            top_k = items[:max_end_states_per_region]
            top_k_mass = float(sum(p for _, p in top_k))
            coverage_after_thresh = (top_k_mass / kept_mass) if kept_mass > 0 else 0.0
            if coverage_after_thresh >= float(state_cap_coverage_threshold):
                items = top_k
                kept_mass = top_k_mass

        outcomes: List[RegionOutcome] = []
        for col_label, p in items:
            local_state = agop.global_state_from_row_label(col_label)
            outcomes.append(RegionOutcome(local_state=local_state, probability=float(p), mapping=mapping, col_label=col_label))

        return RegionOutcomeSet(region_nodes=region_nodes, outcomes=outcomes, total_mass=total_mass, kept_mass=float(kept_mass), mapping=mapping)

    # V2 arrays payload
    p = np.asarray(payload.get("p", []), dtype=np.float64)
    owners = np.asarray(payload.get("owners", []))
    troops = np.asarray(payload.get("troops", []))

    if p.size == 0:
        return RegionOutcomeSet(region_nodes=region_nodes, outcomes=[], total_mass=0.0, kept_mass=0.0, mapping=mapping)

    # Build items as indices + probs
    idx = np.arange(p.size)
    items = idx[p > 0.0]
    total_mass = float(p[items].sum()) if items.size else 0.0
    if total_mass <= 0.0:
        return RegionOutcomeSet(region_nodes=region_nodes, outcomes=[], total_mass=0.0, kept_mass=0.0, mapping=mapping)

    if min_state_prob and min_state_prob > 0.0:
        items = items[p[items] >= float(min_state_prob)]

    if items.size == 0:
        return RegionOutcomeSet(region_nodes=region_nodes, outcomes=[], total_mass=total_mass, kept_mass=0.0, mapping=mapping)

    # sort by prob desc
    items = items[np.argsort(p[items])[::-1]]
    kept_mass = float(p[items].sum())

    if enable_state_capping and max_end_states_per_region and items.size > int(max_end_states_per_region):
        top_k = items[: int(max_end_states_per_region)]
        top_k_mass = float(p[top_k].sum())
        coverage_after_thresh = (top_k_mass / kept_mass) if kept_mass > 0 else 0.0
        if coverage_after_thresh >= float(state_cap_coverage_threshold):
            items = top_k
            kept_mass = top_k_mass

    # For compatibility with existing RegionOutcomeSet, we still return RegionOutcome objects,
    # but we avoid label parsing by constructing local GlobalState directly from arrays.
    outcomes: List[RegionOutcome] = []
    for k in items.tolist():
        nodes_local = []
        for m in range(int(owners.shape[1])):
            nodes_local.append(NodeState(_v2_owner_to_char(int(owners[k, m])), int(troops[k, m])))
        local_state = GlobalState(nodes=tuple(nodes_local))
        outcomes.append(RegionOutcome(local_state=local_state, probability=float(p[k]), mapping=mapping, col_label=f"@idx:{k}"))

    return RegionOutcomeSet(region_nodes=region_nodes, outcomes=outcomes, total_mass=float(total_mass), kept_mass=float(kept_mass), mapping=mapping)


def _monte_carlo_lookahead_for_partition(
    players: Sequence["Players.Player"],
    battle_graph,
    combat_libraries_base: Path,
    base_global_state: GlobalState,
    base_eval: "PartitionEvaluation",
    partition_regions: Sequence[Dict[str, Any]],
    ranking_variable: str,
    n_scenarios: int,
    min_state_prob: float,
    max_end_states_per_region: int,
    rng: Optional[random.Random] = None,
    enable_state_capping: bool = True,
    state_cap_coverage_threshold: float = 0.9,
    *,
    policy_option_selection: Any = "primary",
) -> Tuple["PartitionEvaluation", Dict[str, float]]:
    if rng is None:
        rng = random.Random()

    board_snapshot = _snapshot_board_state()

    try:
        try:
            edges_iter = battle_graph.edges()
        except TypeError:
            edges_iter = battle_graph.edges
        global_edges = list(edges_iter)

        region_sets: List[RegionOutcomeSet] = []
        regional_coverages: List[float] = []

        for region in partition_regions:
            rset = _collect_region_outcomes_for_partition(
                global_state=base_global_state,
                global_edges=global_edges,
                region=region,
                combat_libraries_base=combat_libraries_base,
                min_state_prob=min_state_prob,
                max_end_states_per_region=max_end_states_per_region,
                enable_state_capping=enable_state_capping,
                state_cap_coverage_threshold=state_cap_coverage_threshold,
                policy_option_selection=policy_option_selection,
                ranking_variable=ranking_variable,
            )
            region_sets.append(rset)

            if rset.total_mass > 0.0:
                regional_coverages.append(rset.kept_mass / rset.total_mass)

        joint_coverage = float(np.prod(regional_coverages)) if regional_coverages else 0.0
        lost_coverage = 1.0 - joint_coverage
        avg_regional_coverage = float(np.mean(regional_coverages)) if regional_coverages else 0.0

        coverage_info = {
            "joint_coverage": joint_coverage,
            "lost_coverage": lost_coverage,
            "avg_regional_coverage": avg_regional_coverage,
            "num_regions": len(partition_regions),
        }

        if all(len(rset.outcomes) == 0 for rset in region_sets):
            return base_eval, coverage_info

        # Precompute per-region discrete distributions (normalized)
        region_weights: List[Optional[List[float]]] = []
        for rset in region_sets:
            if not rset.outcomes:
                region_weights.append(None)
                continue
            probs = np.array([float(o.probability) for o in rset.outcomes], dtype=float)
            s = float(probs.sum())
            if s <= 0.0:
                region_weights.append([1.0 / len(probs)] * len(probs))
            else:
                region_weights.append((probs / s).tolist())

        # --- Monte Carlo sampling over joint outcomes ---
        acc_expected_new_territories = 0.0
        acc_expected_lost_troops = 0.0
        acc_regional_product_conquest_probability = 0.0
        acc_expected_territories = 0.0
        acc_expected_troops = 0.0
        acc_conquest_probability = 0.0

        def _overlay_outcome_into_nodes(
            nodes: List[NodeState],
            *,
            outcome: RegionOutcome,
            mapping: Dict[int, int],
        ) -> None:
            # V1: local_state exists
            if outcome.format_version == 1 and outcome.local_state is not None:
                for local_idx, node_after in enumerate(outcome.local_state.nodes):
                    global_idx = mapping.get(local_idx)
                    if global_idx is None:
                        continue
                    nodes[global_idx] = node_after
                return

            # V2: owners_row/troops_row exists
            if outcome.format_version == 2 and outcome.owners_row is not None and outcome.troops_row is not None:
                owners_row = outcome.owners_row
                troops_row = outcome.troops_row
                owner_codes = outcome.owner_codes or {}

                a_code = owner_codes.get("A", 0)
                d_code = owner_codes.get("D", 1)

                for local_idx, global_idx in mapping.items():
                    li = int(local_idx)
                    oc = int(owners_row[li])
                    tr = int(troops_row[li])

                    # owner decode (default A=0, D=1)
                    owner = "A" if oc == int(a_code) else "D" if oc == int(d_code) else ("A" if oc == 0 else "D")
                    nodes[global_idx] = NodeState(owner, tr)
                return

            # If neither representation is present, do nothing.
            return

            # legacy path
            local_state = outcome.local_state
            if local_state is None:
                return
            for local_idx, node_after in enumerate(local_state.nodes):
                global_idx = mapping.get(local_idx)
                if global_idx is None:
                    continue
                nodes[global_idx] = node_after

        for _ in range(n_scenarios):
            chosen_aligned: List[Optional[RegionOutcome]] = []
            for rset, weights in zip(region_sets, region_weights):
                if not rset.outcomes or not weights:
                    chosen_aligned.append(None)
                    continue
                idx = rng.choices(range(len(rset.outcomes)), weights=weights, k=1)[0]
                chosen_aligned.append(rset.outcomes[idx])

            nodes_after = list(base_global_state.nodes)
            for rset, out in zip(region_sets, chosen_aligned):
                if out is None:
                    continue
                if not rset.mapping:
                    continue
                _overlay_outcome_into_nodes(nodes_after, outcome=out, mapping=rset.mapping)

            scenario_state = GlobalState(nodes=tuple(nodes_after))

            _apply_global_state_to_board(scenario_state, players)

            scenario_result = rank_battle_graph_partitions(
                players=players,
                battle_graph=battle_graph,
                combat_libraries_base=combat_libraries_base,
                max_partitions=40,
                ranking_variable=ranking_variable,
                lookahead_depth=0,
                min_state_prob=min_state_prob,
                max_end_states_per_region=max_end_states_per_region,
                enable_state_capping=enable_state_capping,
                state_cap_coverage_threshold=state_cap_coverage_threshold,
            )

            scenario_best: Optional["PartitionEvaluation"] = scenario_result.get("best_evaluation")  # type: ignore
            if scenario_best is None:
                continue

            acc_expected_new_territories += float(scenario_best.expected_new_territories)
            acc_expected_lost_troops += float(scenario_best.expected_lost_troops)
            acc_regional_product_conquest_probability += float(scenario_best.regional_product_conquest_probability)
            acc_expected_territories += float(scenario_best.expected_territories)
            acc_expected_troops += float(scenario_best.expected_troops)
            acc_conquest_probability += float(scenario_best.conquest_probability)

        factor = (joint_coverage / float(n_scenarios)) if n_scenarios > 0 else 0.0

        second_expected_new_territories = acc_expected_new_territories * factor
        second_expected_lost_troops = acc_expected_lost_troops * factor
        second_regional_product_conquest_probability = acc_regional_product_conquest_probability * factor
        second_expected_territories = acc_expected_territories * factor
        second_expected_troops = acc_expected_troops * factor
        second_conquest_probability = acc_conquest_probability * factor

        combined = PartitionEvaluation(
            partition=list(partition_regions),
            covers_all_nodes=base_eval.covers_all_nodes,
            expected_new_territories=float(base_eval.expected_new_territories + second_expected_new_territories),
            expected_lost_troops=float(base_eval.expected_lost_troops + second_expected_lost_troops),
            regional_product_conquest_probability=float(
                base_eval.regional_product_conquest_probability + second_regional_product_conquest_probability
            ),
            expected_territories=float(base_eval.expected_territories + second_expected_territories),
            expected_troops=float(base_eval.expected_troops + second_expected_troops),
            conquest_probability=float(base_eval.conquest_probability + second_conquest_probability),
            region_policy_options=tuple(getattr(base_eval, "region_policy_options", ()) or ()),
        )

        return combined, coverage_info

    finally:
        _restore_board_state(board_snapshot)


# ---------------------------------------------------------------------
# Special evaluation function for macro experiments
# ---------------------------------------------------------------------

def evaluate_all_partitions_for_state(
    players: Sequence["Players.Player"],
    battle_graph,
    combat_libraries_base: Path,
    max_partitions: int = 40,
) -> Dict[str, Any]:
    """
    Evaluate *all* full battle-graph partitions for the current Board state,
    without ranking or lookahead.

    This is intended for analysis of how partition structure (e.g. number of
    regions) relates to expected metrics, holding the underlying state fixed.

    Returns
    -------
    dict with keys:
        - "global_state"         : GlobalState at the root
        - "battle_nodes"         : list[int] node indices in the battle graph
        - "partitions_full"      : list[list[region dict]] full partitions
        - "evaluations"          : list[PartitionEvaluation] in same order
    """
    # Build global state and battle nodes once
    global_state: GlobalState = agop.build_global_state_for_board(players)

    try:
        nodes_iter = battle_graph.nodes()
    except TypeError:
        nodes_iter = battle_graph.nodes
    battle_nodes = list(nodes_iter)

    # Get full partitions
    partitions_full: List[List[Dict[str, Any]]] = (
        agop.partition_continent_battle_graph_into_valid_small_graphs(
            players=players,
            continent_battle_graph=battle_graph,
            max_partitions=max_partitions,
        )
    )

    # If no full partitions, return early
    if not partitions_full:
        return {
            "global_state": global_state,
            "battle_nodes": battle_nodes,
            "partitions_full": [],
            "evaluations": [],
        }

    # Precompute edges for evaluation
    try:
        edges_iter = battle_graph.edges()
    except TypeError:
        edges_iter = battle_graph.edges
    global_edges = list(edges_iter)

    # Current attacker totals on battle graph
    current_territories = sum(
        1
        for idx in battle_nodes
        if global_state.nodes[idx].owner == "A"
        and global_state.nodes[idx].troops > 0
    )
    current_troops = sum(
        global_state.nodes[idx].troops
        for idx in battle_nodes
        if global_state.nodes[idx].owner == "A"
        and global_state.nodes[idx].troops > 0
    )

    evaluations: List[PartitionEvaluation] = []

    for partition_regions in partitions_full:
        ev = _evaluate_partition_metrics(
            partition_regions=partition_regions,
            global_state=global_state,
            global_edges=global_edges,
            battle_nodes=battle_nodes,
            combat_libraries_base=combat_libraries_base,
            current_territories=current_territories,
            current_troops=current_troops,
            policy_option_selection=policy_option_selection,
            ranking_variable=ranking_variable,
        )
        if ev is not None:
            evaluations.append(ev)

    return {
        "global_state": global_state,
        "battle_nodes": battle_nodes,
        "partitions_full": partitions_full,
        "evaluations": evaluations,
    }


# ---------------------------------------------------------------------
# Ranking main functions
# ---------------------------------------------------------------------


def local_utility_tuple_from_option_payload(option_payload: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Mirror CompactExactTopologySolver utility_mode='local' without no-gain:
    (expected_new_territories, expected_final_attacker_troops, p_local_conquest).
    """
    local_value = option_payload.get("local_value")
    if isinstance(local_value, dict):
        if "raw_value" in local_value:
            raw = tuple(float(x) for x in local_value.get("raw_value", ()) or ())
            if len(raw) >= 3:
                return (raw[0], raw[1], raw[2])
        keys = ("expected_new_territories", "expected_final_attacker_troops", "p_local_conquest")
        if all(k in local_value for k in keys):
            return tuple(float(local_value[k]) for k in keys)  # type: ignore[return-value]

    p = np.asarray(option_payload.get("p", []), dtype=np.float64)
    new_terr = np.asarray(option_payload.get("new_territories", []), dtype=np.float64)
    final_att = np.asarray(option_payload.get("final_attacker_troops", []), dtype=np.float64)
    is_conq = np.asarray(option_payload.get("is_conquered", []), dtype=np.float64)
    if p.size == 0:
        return (0.0, 0.0, 0.0)
    return (
        float(np.dot(p, new_terr)),
        float(np.dot(p, final_att)),
        float(np.dot(p, is_conq)),
    )


def compose_partition_policy_utility(region_option_refs: Sequence[RegionPolicyOptionRef]) -> Tuple[float, float, float]:
    expected_new = 0.0
    expected_troops = 0.0
    conquest_product = 1.0
    for ref in region_option_refs:
        u = tuple(float(x) for x in ref.utility_tuple)
        expected_new += u[0] if len(u) > 0 else 0.0
        expected_troops += u[1] if len(u) > 1 else 0.0
        conquest_product *= u[2] if len(u) > 2 else 0.0
    return (float(expected_new), float(expected_troops), float(conquest_product))


def _utility_equal(
    left: Sequence[float],
    right: Sequence[float],
    tolerances: Optional[Tuple[float, ...]],
) -> bool:
    if len(left) != len(right):
        return False
    if tolerances is None:
        return tuple(left) == tuple(right)
    for i, (a, b) in enumerate(zip(left, right)):
        tol = float(tolerances[i]) if i < len(tolerances) else float(tolerances[-1])
        if abs(float(a) - float(b)) > tol:
            return False
    return True


def query_region_policy_options_from_libraries(
    *,
    combat_libraries_base: Path,
    global_state: GlobalState,
    global_edges,
    region: Dict[str, Any],
    ranking_variable: Any = "expected_territories",
) -> Tuple[RegionPolicyOptionRef, ...]:
    """Return every normalized policy option for a region without collapsing to primary/best_local."""
    region_nodes = tuple(region.get("region_nodes", ()))
    result = agop.query_region_from_libraries(
        combat_libraries_base=combat_libraries_base,
        global_state=global_state,
        global_edges=global_edges,
        region_nodes=region_nodes,
        debug=False,
        policy_option_selection="primary",
        ranking_variable=ranking_variable,
    )
    policy_options = tuple(result.get("policy_options_v2", ()) or ())
    out: List[RegionPolicyOptionRef] = []
    for i, payload in enumerate(policy_options):
        payload2 = dict(payload)
        dist_payload = {
            "p": payload2.get("p"),
            "owners": payload2.get("owners"),
            "troops": payload2.get("troops"),
            "cdf": payload2.get("cdf"),
            "is_conquered": payload2.get("is_conquered"),
            "new_territories": payload2.get("new_territories"),
            "final_attacker_troops": payload2.get("final_attacker_troops"),
        }
        out.append(
            RegionPolicyOptionRef(
                region_nodes=tuple(result.get("region_nodes_effective", region_nodes)),
                attacker_nodes=tuple(region.get("attacker_nodes", ())),
                defender_nodes=tuple(region.get("defender_nodes", ())),
                pattern=tuple(result.get("pattern", region.get("pattern", ()))),
                row_label=str(result.get("row_label")),
                option_index=int(payload2.get("option_id", i)),
                option_count=len(policy_options),
                root_action=payload2.get("root_action"),
                split_metadata=payload2.get("split_metadata"),
                payload=payload2,
                utility_tuple=local_utility_tuple_from_option_payload(payload2),
                distribution_payload=dist_payload,
                mapping=dict(result.get("mapping", {}) or {}),
                topology_signature=tuple(
                    tuple(int(x) for x in edge)
                    for edge in (result.get("graph_edges_reindexed", ()) or ())
                ),
                policy_option_mode=payload2.get("policy_option_mode"),
            )
        )
    return tuple(out)


def expand_partition_policy_candidates(
    *,
    partition_index: int,
    partition_regions: Sequence[Dict[str, Any]],
    combat_libraries_base: Path,
    global_state: GlobalState,
    global_edges,
    ranking_variable: Any = "expected_territories",
    max_policy_combos_per_partition: Optional[int] = 256,
) -> Tuple[Tuple[PartitionPolicyCandidate, ...], Dict[str, Any]]:
    region_options: List[Tuple[RegionPolicyOptionRef, ...]] = []
    diagnostics: Dict[str, Any] = {
        "candidate_cap_hit": False,
        "num_candidates_before_cap": 0,
        "num_candidates_after_cap": 0,
        "region_option_counts": [],
    }
    for region in partition_regions:
        opts = query_region_policy_options_from_libraries(
            combat_libraries_base=combat_libraries_base,
            global_state=global_state,
            global_edges=global_edges,
            region=region,
            ranking_variable=ranking_variable,
        )
        if not opts:
            return (), diagnostics
        region_options.append(opts)
        diagnostics["region_option_counts"].append(len(opts))

    total = 1
    for opts in region_options:
        total *= len(opts)
    diagnostics["num_candidates_before_cap"] = total
    cap = total if max_policy_combos_per_partition is None else int(max_policy_combos_per_partition)

    candidates: List[PartitionPolicyCandidate] = []
    for combo_i, combo in enumerate(itertools.product(*region_options)):
        if combo_i >= cap:
            diagnostics["candidate_cap_hit"] = True
            break
        utility = compose_partition_policy_utility(combo)
        candidates.append(
            PartitionPolicyCandidate(
                partition_index=int(partition_index),
                partition_regions=tuple(dict(r) for r in partition_regions),
                region_policy_options=tuple(combo),
                first_stage_utility=utility,
                first_stage_score=utility,
                diagnostics={"combo_index": combo_i},
            )
        )
    diagnostics["num_candidates_after_cap"] = len(candidates)
    return tuple(candidates), diagnostics


def _candidate_utility_key(candidate: PartitionPolicyCandidate) -> Tuple[float, ...]:
    return tuple(float(x) for x in candidate.first_stage_utility)


def _within_partition_local_tolerance(
    utility: Sequence[float],
    best_utility: Sequence[float],
    *,
    utility_abs_tolerance: Optional[float],
    utility_rel_tolerance: Optional[float],
) -> bool:
    u = tuple(float(x) for x in utility)
    best = tuple(float(x) for x in best_utility)
    if u == best:
        return True
    if utility_abs_tolerance is None and utility_rel_tolerance is None:
        return False
    active = 0
    u0 = float(u[active]) if len(u) > active else 0.0
    b0 = float(best[active]) if len(best) > active else 0.0
    absolute_ok = utility_abs_tolerance is not None and u0 >= b0 - float(utility_abs_tolerance)
    relative_ok = (
        utility_rel_tolerance is not None
        and u0 >= b0 - float(utility_rel_tolerance) * max(1.0, abs(b0))
    )
    return bool(absolute_ok or relative_ok)


def retain_partition_local_utility_candidates(
    candidates: Sequence[PartitionPolicyCandidate],
    *,
    utility_abs_tolerance: Optional[float] = None,
    utility_rel_tolerance: Optional[float] = None,
    max_candidates_per_partition: Optional[int] = None,
) -> Tuple[List[PartitionPolicyCandidate], Dict[str, Any]]:
    grouped: Dict[Tuple[Tuple[int, ...], ...], List[PartitionPolicyCandidate]] = {}
    for candidate in candidates or ():
        sig = canonical_partition_signature(candidate.partition_regions)
        grouped.setdefault(sig, []).append(candidate)

    retained: List[PartitionPolicyCandidate] = []
    before_by_partition: Dict[Tuple[Tuple[int, ...], ...], int] = {}
    after_by_partition: Dict[Tuple[Tuple[int, ...], ...], int] = {}
    best_utilities: Dict[Tuple[Tuple[int, ...], ...], Tuple[float, ...]] = {}

    cap = None if max_candidates_per_partition is None else max(1, int(max_candidates_per_partition))
    for sig in sorted(grouped):
        group = sorted(grouped[sig], key=lambda c: (-float(c.first_stage_utility[0]), tuple(-float(x) for x in c.first_stage_utility[1:]), int(c.partition_index), c.diagnostics.get("combo_index", 0) if isinstance(c.diagnostics, dict) else 0))
        before_by_partition[sig] = int(len(group))
        best = max(group, key=_candidate_utility_key).first_stage_utility
        best_utilities[sig] = tuple(float(x) for x in best)
        kept = [
            c
            for c in group
            if _within_partition_local_tolerance(
                c.first_stage_utility,
                best,
                utility_abs_tolerance=utility_abs_tolerance,
                utility_rel_tolerance=utility_rel_tolerance,
            )
        ]
        kept = sorted(kept, key=lambda c: (_candidate_utility_key(c), -int(c.diagnostics.get("combo_index", 0) if isinstance(c.diagnostics, dict) else 0)), reverse=True)
        if cap is not None:
            kept = kept[:cap]
        if not kept and group:
            kept = [max(group, key=_candidate_utility_key)]
        after_by_partition[sig] = int(len(kept))
        retained.extend(kept)

    diagnostics = {
        "num_policy_candidates_before_partition_local_utility": int(len(candidates or ())),
        "num_policy_candidates_after_partition_local_utility": int(len(retained)),
        "policy_candidates_before_by_partition": before_by_partition,
        "policy_candidates_after_by_partition": after_by_partition,
        "partition_best_utilities": best_utilities,
        "utility_abs_tolerance": None if utility_abs_tolerance is None else float(utility_abs_tolerance),
        "utility_rel_tolerance": None if utility_rel_tolerance is None else float(utility_rel_tolerance),
        "max_candidates_per_partition": None if max_candidates_per_partition is None else int(max_candidates_per_partition),
    }
    return retained, diagnostics


def _classify_region_query_failure(e: Exception) -> str:
    msg = str(e)
    if "Unknown owner representation" in msg:
        return "unrecognized_owner_label"
    if "owner_source_mismatch" in msg:
        return "owner_source_mismatch"
    if "owner_role_normalization_failed" in msg:
        return "owner_role_normalization_failed"
    if isinstance(e, FileNotFoundError) or "No library found" in msg or "get_prob_table" in msg:
        return "missing_topology_library"
    if "exceed" in msg.lower() or "cap" in msg.lower():
        return "troop_cap_exceeded"
    if "Row label" in msg and "not found" in msg:
        return "missing_library_row"
    if "no internal edges" in msg or "not a battle-valid region" in msg or "frontier edge" in msg:
        return "invalid_or_disconnected_region"
    if "STAR_ONLY" in msg or "topology" in msg or "mismatch" in msg:
        return "role_or_topology_mismatch"
    if agop._is_coverage_failure(e):
        return "missing_topology_library"
    if agop._is_query_viability_failure(e):
        return "other"
    return "other"


def _add_rejection_example(
    examples: Dict[str, List[Dict[str, Any]]],
    reason: str,
    *,
    region_nodes: Sequence[int],
    pattern: Optional[Tuple[int, int]],
    global_state: Optional[GlobalState] = None,
    role_summary: Optional[Dict[str, Any]] = None,
    limit: int = 5,
) -> None:
    bucket = examples.setdefault(reason, [])
    if len(bucket) >= int(limit):
        return
    troop_row = None
    if global_state is not None:
        troop_row = tuple(int(global_state.nodes[int(n)].troops) for n in region_nodes)
    example = {
        "region_nodes": tuple(int(n) for n in region_nodes),
        "pattern": None if pattern is None else tuple(int(x) for x in pattern),
        "troop_row": troop_row,
        "reason": str(reason),
    }
    if role_summary:
        example["raw_owner_values"] = tuple(role_summary.get("raw_owner_values", ()) or ())
        normalized = role_summary.get("normalized_roles")
        example["normalized_roles"] = None if normalized is None else tuple(normalized)
        example["role_kind"] = role_summary.get("role_kind")
        example["normalization_error"] = role_summary.get("normalization_error")
    bucket.append(example)


def diagnose_supported_partition_coverage(
    *,
    players: Sequence["Players.Player"],
    battle_graph,
    global_state: Optional[GlobalState] = None,
    library_dir: Path | str = "small_graph_libraries",
    max_partitions: int = 40,
    policy_option_selection: Any = "primary",
    ranking_variable: Any = "battle_expected_attacker_territory_count",
    run_expensive_cover_diagnostics: bool = False,
    **ranking_kwargs,
) -> Dict[str, Any]:
    """Explain supported full-cover availability using the current partition/query path."""
    del policy_option_selection, ranking_variable, ranking_kwargs
    global_state = global_state or agop.build_global_state_for_board(players)
    try:
        nodes_iter = battle_graph.nodes()
    except TypeError:
        nodes_iter = battle_graph.nodes
    all_nodes = list(nodes_iter)
    try:
        edges_iter = battle_graph.edges()
    except TypeError:
        edges_iter = battle_graph.edges
    edges = tuple((int(u), int(v)) for u, v in edges_iter)
    active_edges = 0
    for u, v in edges:
        if str(global_state.nodes[int(u)].owner) != str(global_state.nodes[int(v)].owner):
            active_edges += 1
    attacker_nodes_all = [int(n) for n in all_nodes if str(global_state.nodes[int(n)].owner) == "A"]
    defender_nodes_all = [int(n) for n in all_nodes if str(global_state.nodes[int(n)].owner) == "D"]

    diag: Dict[str, Any] = {
        "battle_node_count": int(len(all_nodes)),
        "battle_edge_count": int(len(edges)),
        "attacker_node_count": int(len(attacker_nodes_all)),
        "defender_node_count": int(len(defender_nodes_all)),
        "active_attacker_defender_edge_count": int(active_edges),
        "has_active_combat": bool(active_edges > 0),
        "num_region_subsets_considered": 0,
        "num_supported_regions": 0,
        "supported_region_signatures": tuple(),
        "supported_pattern_counts": {},
        "region_rejection_counts": {
            "unsupported_pattern": 0,
            "missing_topology_library": 0,
            "troop_cap_exceeded": 0,
            "missing_library_row": 0,
            "invalid_or_disconnected_region": 0,
            "role_or_topology_mismatch": 0,
            "unrecognized_owner_label": 0,
            "owner_source_mismatch": 0,
            "owner_role_normalization_failed": 0,
            "other": 0,
        },
        "region_rejection_examples": {},
    }
    if not all_nodes or active_edges <= 0:
        diag.update(
            {
                "num_supported_full_covers": 0,
                "supported_full_cover_signatures": tuple(),
                "num_unique_supported_partitions": 0,
                "num_dominated_partitions_removed": 0,
                "num_maximal_partitions": 0,
                "maximal_partition_signatures": tuple(),
                "full_cover_status": "no_active_combat",
                "nodes_appearing_in_any_supported_region": tuple(),
                "nodes_never_covered_by_supported_region": tuple(int(n) for n in all_nodes),
                "maximum_nodes_covered_by_any_disjoint_region_collection": 0,
                "best_partial_coverage_ratio": 0.0,
                "best_partial_partition_signature": None,
                "partial_partition_used": False,
                "partition_cover_universe_mode": "battle_graph_nonisolated_nodes",
                "required_cover_nodes": tuple(),
                "excluded_context_nodes": tuple(),
                "supported_region_count_by_node": {},
                "production_exact_cover_found": False,
                "reference_exact_cover_found": False,
                "production_reference_agree": True,
                "maximum_disjoint_coverage_count": 0,
                "maximum_disjoint_coverage_ratio": 0.0,
                "uncovered_nodes_in_best_partial_cover": tuple(),
                "active_edge_coverage_complete": None,
                "overlap_cover_possible": None,
                "run_expensive_cover_diagnostics": bool(run_expensive_cover_diagnostics),
            }
        )
        return diag

    pruned_nodes = list(agop.partition_required_graph_nodes(battle_graph, edges=edges))
    all_node_set = set(pruned_nodes)
    global_edges = tuple(edges)
    supported_regions: List[Dict[str, Any]] = []
    rejection_counts = diag["region_rejection_counts"]
    rejection_examples: Dict[str, List[Dict[str, Any]]] = {}
    max_size = max(a + d for a, d in agop.ALLOWED_PATTERNS)
    min_size = 2

    def _has_frontier_edge(att_nodes: Sequence[Any], def_nodes: Sequence[Any]) -> bool:
        return any(
            agop.graph_has_edge_compatible(battle_graph, u, v, edges=global_edges)
            for u in att_nodes
            for v in def_nodes
        )

    for size in range(min_size, max_size + 1):
        for subset in itertools.combinations(pruned_nodes, size):
            diag["num_region_subsets_considered"] += 1
            role_summary = agop.region_combat_role_summary(region_nodes=subset, global_state=global_state)
            if role_summary.get("normalization_error"):
                reason = "unrecognized_owner_label"
                rejection_counts[reason] += 1
                _add_rejection_example(
                    rejection_examples,
                    reason,
                    region_nodes=subset,
                    pattern=None,
                    global_state=global_state,
                    role_summary=role_summary,
                )
                continue
            role_map = role_summary["role_map"]
            attacker_nodes = [n for n in subset if role_map[int(n)] == "A"]
            defender_nodes = [n for n in subset if role_map[int(n)] == "D"]
            pattern = tuple(role_summary["pattern"])
            if pattern[0] + pattern[1] != len(subset):
                reason = "owner_role_normalization_failed"
                rejection_counts[reason] += 1
                _add_rejection_example(
                    rejection_examples,
                    reason,
                    region_nodes=subset,
                    pattern=pattern,
                    global_state=global_state,
                    role_summary=role_summary,
                )
                continue
            if pattern not in agop.ALLOWED_PATTERNS or pattern[0] == 0 or pattern[1] == 0:
                reason = "unsupported_pattern"
                rejection_counts[reason] += 1
                _add_rejection_example(
                    rejection_examples,
                    reason,
                    region_nodes=subset,
                    pattern=pattern,
                    global_state=global_state,
                    role_summary=role_summary,
                )
                continue
            if not agop._is_connected_subset(battle_graph, subset) or not _has_frontier_edge(attacker_nodes, defender_nodes):
                reason = "invalid_or_disconnected_region"
                rejection_counts[reason] += 1
                _add_rejection_example(
                    rejection_examples,
                    reason,
                    region_nodes=subset,
                    pattern=pattern,
                    global_state=global_state,
                    role_summary=role_summary,
                )
                continue
            region = {
                "region_nodes": tuple(subset),
                "attacker_nodes": tuple(attacker_nodes),
                "defender_nodes": tuple(defender_nodes),
                "pattern": pattern,
            }
            try:
                agop.query_region_from_libraries(
                    combat_libraries_base=library_dir,
                    global_state=global_state,
                    global_edges=global_edges,
                    region_nodes=subset,
                    debug=False,
                )
            except Exception as e:
                reason = _classify_region_query_failure(e)
                rejection_counts[reason] += 1
                _add_rejection_example(
                    rejection_examples,
                    reason,
                    region_nodes=subset,
                    pattern=pattern,
                    global_state=global_state,
                    role_summary=role_summary,
                )
                continue
            supported_regions.append(region)

    supported_signatures = tuple(sorted(tuple(int(n) for n in r["region_nodes"]) for r in supported_regions))
    pattern_counts: Dict[Tuple[int, int], int] = {}
    for r in supported_regions:
        pat = tuple(int(x) for x in r.get("pattern", (0, 0)))
        pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
    diag["num_supported_regions"] = int(len(supported_regions))
    diag["supported_region_signatures"] = supported_signatures
    diag["supported_pattern_counts"] = dict(sorted(pattern_counts.items()))
    diag["region_rejection_examples"] = {k: tuple(v) for k, v in sorted(rejection_examples.items())}

    cover_analysis = analyze_exact_cover_compatibility(
        battle_graph=battle_graph,
        global_state=global_state,
        supported_regions=supported_regions,
        required_cover_nodes=all_node_set,
        max_reference_covers=max_partitions,
        run_overlap_diagnostics=bool(run_expensive_cover_diagnostics),
    )

    partitions_full = agop.partition_continent_battle_graph_into_valid_small_graphs(
        players=players,
        continent_battle_graph=battle_graph,
        max_partitions=max_partitions,
        combat_libraries_base=Path(library_dir),
    )
    maximal, filter_diag = filter_maximal_supported_partitions(partitions_full) if partitions_full else ([], {
        "num_unique_partitions": 0,
        "num_dominated_partitions_removed": 0,
        "num_maximal_partitions": 0,
        "maximal_partition_signatures": tuple(),
    })
    diag.update(
        {
            "num_supported_full_covers": int(len(partitions_full)),
            "supported_full_cover_signatures": tuple(canonical_partition_signature(p) for p in partitions_full),
            "num_unique_supported_partitions": int(filter_diag.get("num_unique_partitions", 0)),
            "num_dominated_partitions_removed": int(filter_diag.get("num_dominated_partitions_removed", 0)),
            "num_maximal_partitions": int(filter_diag.get("num_maximal_partitions", len(maximal))),
            "maximal_partition_signatures": tuple(filter_diag.get("maximal_partition_signatures", ()) or ()),
        }
    )

    nodes_covered = {n for region in cover_analysis.supported_region_signatures for n in region}
    nodes_never = cover_analysis.nodes_in_no_supported_region
    best_partial_sig = cover_analysis.best_partial_cover_signature
    ratio = cover_analysis.maximum_disjoint_coverage_ratio
    if partitions_full:
        status = "supported_full_cover"
    elif not supported_regions:
        status = "no_supported_regions"
    elif nodes_never:
        status = "incomplete_supported_region_coverage"
    elif cover_analysis.production_found_full_cover or cover_analysis.brute_force_found_full_cover:
        status = "production_region_enumeration_mismatch"
    elif not cover_analysis.production_bruteforce_agree:
        status = "exact_cover_diagnostic_disagreement"
    else:
        status = "supported_regions_but_no_exact_cover"
    analysis_diag = dict(cover_analysis.diagnostics)
    per_node_support = dict(analysis_diag.get("per_node_support", {}) or {})
    rejection_reasons_by_node: Dict[int, Dict[str, int]] = {int(n): {} for n in all_node_set}
    for reason, examples in diag["region_rejection_examples"].items():
        for example in examples:
            for node in example.get("region_nodes", ()):
                if int(node) in rejection_reasons_by_node:
                    bucket = rejection_reasons_by_node[int(node)]
                    bucket[str(reason)] = int(bucket.get(str(reason), 0)) + 1
    active_edge_diag = dict(analysis_diag.get("active_edge_coverage", {}) or {})
    overlap_diag = dict(analysis_diag.get("overlap_coverage", {}) or {})
    diag.update(
        {
            "full_cover_status": status,
            "nodes_appearing_in_any_supported_region": tuple(sorted(nodes_covered)),
            "nodes_never_covered_by_supported_region": nodes_never,
            "partition_cover_universe_mode": "battle_graph_nonisolated_nodes",
            "required_cover_nodes": cover_analysis.required_cover_nodes,
            "excluded_context_nodes": cover_analysis.context_only_nodes,
            "supported_region_count_by_node": dict(cover_analysis.supported_regions_per_node),
            "per_node_support_diagnostics": per_node_support,
            "rejected_region_reasons_by_node_from_examples": rejection_reasons_by_node,
            "minimum_supported_regions_per_required_node": analysis_diag.get(
                "minimum_supported_regions_per_required_node", 0
            ),
            "maximum_supported_regions_per_required_node": analysis_diag.get(
                "maximum_supported_regions_per_required_node", 0
            ),
            "production_exact_cover_found": cover_analysis.production_found_full_cover,
            "reference_exact_cover_found": cover_analysis.brute_force_found_full_cover,
            "production_reference_agree": cover_analysis.production_bruteforce_agree,
            "production_exact_cover_signatures": cover_analysis.production_full_cover_signatures,
            "reference_exact_cover_signatures": cover_analysis.brute_force_full_cover_signatures,
            "maximum_nodes_covered_by_any_disjoint_region_collection": cover_analysis.maximum_disjoint_coverage_count,
            "maximum_disjoint_coverage_count": cover_analysis.maximum_disjoint_coverage_count,
            "maximum_disjoint_coverage_ratio": ratio,
            "best_partial_coverage_ratio": ratio,
            "best_partial_partition_signature": best_partial_sig,
            "uncovered_nodes_in_best_partial_cover": cover_analysis.uncovered_nodes_in_best_partial_cover,
            "active_edge_coverage_complete": active_edge_diag.get("active_edge_coverage_complete"),
            "active_edges_in_no_supported_region": active_edge_diag.get(
                "active_edges_in_no_supported_region", tuple()
            ),
            "overlap_cover_possible": (
                overlap_diag.get("any_overlap", {}).get("complete_node_coverage_possible")
                if run_expensive_cover_diagnostics
                else None
            ),
            "overlap_cover_requires_overlap": (
                bool(overlap_diag.get("any_overlap", {}).get("complete_node_coverage_possible"))
                and not cover_analysis.brute_force_found_full_cover
                if run_expensive_cover_diagnostics
                else None
            ),
            "overlap_cover_diagnostics": overlap_diag,
            "context_node_classification": analysis_diag.get("context_classification", {}),
            "partial_partition_used": bool(best_partial_sig and not partitions_full),
            "run_expensive_cover_diagnostics": bool(run_expensive_cover_diagnostics),
        }
    )
    return diag


def prepare_two_stage_partition_policy_candidates(
    *,
    players: Sequence["Players.Player"],
    battle_graph,
    combat_libraries_base: Path,
    max_partitions: int = 40,
    ranking_variable: str = "battle_expected_attacker_territory_count",
    first_stage_value_tolerances: Optional[Tuple[float, ...]] = None,
    max_policy_combos_per_partition: Optional[int] = 256,
    max_total_partition_policy_candidates: Optional[int] = None,
    partition_candidate_selection_mode: str = "maximal_per_partition_utility",
    utility_abs_tolerance: Optional[float] = None,
    utility_rel_tolerance: Optional[float] = None,
    max_candidates_per_partition: Optional[int] = None,
    run_expensive_cover_diagnostics: bool = False,
    coverage_diagnostics: Optional[Mapping[str, Any]] = None,
) -> TwoStagePreparedCandidates:
    if partition_candidate_selection_mode not in ("legacy_global_utility", "maximal_per_partition_utility"):
        raise ValueError(f"Unsupported partition_candidate_selection_mode={partition_candidate_selection_mode!r}")
    global_state = agop.build_global_state_for_board(players)
    try:
        battle_nodes = tuple(int(x) for x in battle_graph.nodes())
    except TypeError:
        battle_nodes = tuple(int(x) for x in battle_graph.nodes)
    try:
        global_edges = tuple((int(u), int(v)) for u, v in battle_graph.edges())
    except TypeError:
        global_edges = tuple((int(u), int(v)) for u, v in battle_graph.edges)

    partitions_full = agop.partition_continent_battle_graph_into_valid_small_graphs(
        players=players,
        continent_battle_graph=battle_graph,
        max_partitions=max_partitions,
        combat_libraries_base=combat_libraries_base,
    )
    if partition_candidate_selection_mode == "maximal_per_partition_utility" and partitions_full:
        working_partitions_list, partition_filter_diag = filter_maximal_supported_partitions(partitions_full)
    else:
        working_partitions_list = partitions_full or _find_best_partial_partitions(players, battle_graph, max_partitions)
        if partitions_full:
            _, partition_filter_diag = filter_maximal_supported_partitions(partitions_full)
        else:
            partition_filter_diag = {
                "num_unique_partitions": 0,
                "num_dominated_partitions_removed": 0,
                "num_maximal_partitions": 0,
                "input_partition_signatures": tuple(),
                "maximal_partition_signatures": tuple(),
                "dominated_partition_records": tuple(),
            }

    supported_region_signatures = {
        tuple(sorted(int(node) for node in (region.get("region_nodes", ()) or ())))
        for partition in (partitions_full or working_partitions_list or ())
        for region in partition
        if region.get("region_nodes")
    }

    all_candidates: List[PartitionPolicyCandidate] = []
    diagnostics: Dict[str, Any] = {
        "partition_candidate_selection_mode": str(partition_candidate_selection_mode),
        "partition_count": len(working_partitions_list),
        "candidate_cap_hit": False,
        "partition_diagnostics": [],
        "num_supported_full_covers": int(len(partitions_full or ())),
        "num_unique_supported_partitions": int(partition_filter_diag.get("num_unique_partitions", 0)),
        "num_dominated_partitions_removed": int(partition_filter_diag.get("num_dominated_partitions_removed", 0)),
        "num_maximal_partitions": int(partition_filter_diag.get("num_maximal_partitions", 0)),
        "num_supported_regions": int(len(supported_region_signatures)),
        "supported_partition_signatures": tuple(partition_filter_diag.get("input_partition_signatures", ()) or ()),
        "maximal_partition_signatures": tuple(partition_filter_diag.get("maximal_partition_signatures", ()) or ()),
        "dominated_partition_records": tuple(partition_filter_diag.get("dominated_partition_records", ()) or ()),
        "utility_abs_tolerance": None if utility_abs_tolerance is None else float(utility_abs_tolerance),
        "utility_rel_tolerance": None if utility_rel_tolerance is None else float(utility_rel_tolerance),
        "max_candidates_per_partition": None if max_candidates_per_partition is None else int(max_candidates_per_partition),
    }
    supplied_coverage = dict(coverage_diagnostics or {})
    if run_expensive_cover_diagnostics and not supplied_coverage:
        supplied_coverage = diagnose_supported_partition_coverage(
            players=players,
            battle_graph=battle_graph,
            global_state=global_state,
            library_dir=combat_libraries_base,
            max_partitions=max_partitions,
            ranking_variable=ranking_variable,
            run_expensive_cover_diagnostics=True,
        )
    coverage_defaults = {
        "partition_cover_universe_mode": "battle_graph_nonisolated_nodes",
        "required_cover_nodes": agop.partition_required_graph_nodes(
            battle_graph,
            edges=global_edges,
        ),
        "excluded_context_nodes": tuple(),
        "num_supported_regions": int(len(supported_region_signatures)),
        "nodes_never_covered_by_supported_region": tuple(),
        "supported_region_count_by_node": {},
        "production_exact_cover_found": bool(partitions_full),
        "reference_exact_cover_found": None,
        "production_reference_agree": None,
        "maximum_disjoint_coverage_count": 0,
        "maximum_disjoint_coverage_ratio": 0.0,
        "best_partial_partition_signature": None,
        "uncovered_nodes_in_best_partial_cover": tuple(),
        "active_edge_coverage_complete": None,
        "overlap_cover_possible": None,
    }
    for key, default in coverage_defaults.items():
        diagnostics[key] = supplied_coverage.get(key, default)
    diagnostics["run_expensive_cover_diagnostics"] = bool(run_expensive_cover_diagnostics)
    for part_idx, partition_regions in enumerate(working_partitions_list):
        candidates, diag = expand_partition_policy_candidates(
            partition_index=part_idx,
            partition_regions=partition_regions,
            combat_libraries_base=combat_libraries_base,
            global_state=global_state,
            global_edges=global_edges,
            ranking_variable=ranking_variable,
            max_policy_combos_per_partition=max_policy_combos_per_partition,
        )
        diagnostics["partition_diagnostics"].append(diag)
        diagnostics["candidate_cap_hit"] = bool(diagnostics["candidate_cap_hit"] or diag.get("candidate_cap_hit"))
        for c in candidates:
            if (
                partition_candidate_selection_mode == "legacy_global_utility"
                and max_total_partition_policy_candidates is not None
                and len(all_candidates) >= int(max_total_partition_policy_candidates)
            ):
                diagnostics["candidate_cap_hit"] = True
                break
            all_candidates.append(c)

    if not all_candidates:
        diagnostics["all_candidate_count"] = 0
        diagnostics["first_stage_optimal_count"] = 0
        diagnostics["second_stage_candidate_partition_signatures"] = tuple()
        return TwoStagePreparedCandidates(
            global_state=global_state,
            battle_nodes=battle_nodes,
            global_edges=global_edges,
            partitions_full=tuple(partitions_full or ()),
            working_partitions=tuple(working_partitions_list or ()),
            all_candidates=tuple(),
            retained_candidates=tuple(),
            best_utility=None,
            diagnostics=diagnostics,
        )

    if partition_candidate_selection_mode == "legacy_global_utility":
        best_utility = max(c.first_stage_utility for c in all_candidates)
        retained = tuple(
            c for c in all_candidates
            if _utility_equal(c.first_stage_utility, best_utility, first_stage_value_tolerances)
        )
        diagnostics["num_policy_candidates_before_partition_local_utility"] = int(len(all_candidates))
        diagnostics["num_policy_candidates_after_partition_local_utility"] = int(len(retained))
        diagnostics["policy_candidates_before_by_partition"] = {}
        diagnostics["policy_candidates_after_by_partition"] = {}
        diagnostics["partition_best_utilities"] = {}
    else:
        retained_list, local_diag = retain_partition_local_utility_candidates(
            all_candidates,
            utility_abs_tolerance=utility_abs_tolerance,
            utility_rel_tolerance=utility_rel_tolerance,
            max_candidates_per_partition=max_candidates_per_partition,
        )
        diagnostics.update(local_diag)
        retained = tuple(retained_list)
        best_utility = max((c.first_stage_utility for c in retained), default=None)
        if max_total_partition_policy_candidates is not None:
            diagnostics["max_total_partition_policy_candidates_ignored"] = int(max_total_partition_policy_candidates)
    diagnostics["all_candidate_count"] = int(len(all_candidates))
    diagnostics["first_stage_optimal_count"] = int(len(retained))
    diagnostics["second_stage_candidate_partition_signatures"] = tuple(
        canonical_partition_signature(c.partition_regions) for c in retained
    )
    return TwoStagePreparedCandidates(
        global_state=global_state,
        battle_nodes=battle_nodes,
        global_edges=global_edges,
        partitions_full=tuple(partitions_full or ()),
        working_partitions=tuple(working_partitions_list or ()),
        all_candidates=tuple(all_candidates),
        retained_candidates=retained,
        best_utility=best_utility,
        diagnostics=diagnostics,
    )


def _canonical_stable_value(value: Any) -> Any:
    """Convert nested metadata to deterministic, pickle-safe primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.generic):
        return _canonical_stable_value(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return ("float", "nan")
        if np.isposinf(value):
            return ("float", "inf")
        if np.isneginf(value):
            return ("float", "-inf")
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, GlobalState):
        return canonical_two_stage_global_state_signature(value)
    if isinstance(value, Mapping):
        items = [
            (_canonical_stable_value(key), _canonical_stable_value(item))
            for key, item in value.items()
        ]
        return tuple(sorted(items, key=lambda pair: repr(pair[0])))
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_stable_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_canonical_stable_value(item) for item in value), key=repr))
    if isinstance(value, bytes):
        return ("bytes_sha256", hashlib.sha256(value).hexdigest())
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            str(value.dtype),
            tuple(int(x) for x in value.shape),
            _stable_array_digest(value),
        )
    return ("repr", type(value).__module__, type(value).__qualname__, repr(value))


def _stable_json_bytes(value: Any) -> bytes:
    canonical = _canonical_stable_value(value)
    return json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _stable_array_digest(value: Any) -> str:
    arr = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii", errors="backslashreplace"))
    digest.update(repr(tuple(int(x) for x in arr.shape)).encode("ascii"))
    if arr.dtype.hasobject:
        digest.update(_stable_json_bytes(arr.tolist()))
    else:
        digest.update(np.ascontiguousarray(arr).tobytes(order="C"))
    return digest.hexdigest()


def _regional_distribution_digest(ref: RegionPolicyOptionRef) -> str:
    digest = hashlib.sha256()
    for name in ("p", "owners", "troops"):
        digest.update(name.encode("ascii"))
        digest.update(_stable_array_digest(ref.distribution_payload.get(name, ())).encode("ascii"))
    return digest.hexdigest()


def canonical_region_policy_option_key(
    region_option_ref: RegionPolicyOptionRef,
) -> Tuple[Any, ...]:
    """Canonical identity for one exact regional policy and distribution."""
    ref = region_option_ref
    mode = ref.policy_option_mode
    if mode is None and isinstance(ref.payload, Mapping):
        mode = ref.payload.get("policy_option_mode")
    return (
        "region_policy_option_v1",
        tuple(sorted(int(x) for x in ref.region_nodes)),
        tuple(sorted(int(x) for x in ref.attacker_nodes)),
        tuple(sorted(int(x) for x in ref.defender_nodes)),
        tuple(int(x) for x in ref.pattern),
        tuple(tuple(int(x) for x in edge) for edge in ref.topology_signature),
        tuple(sorted((int(local), int(global_idx)) for local, global_idx in ref.mapping.items())),
        str(ref.row_label),
        int(ref.option_index),
        None if mode is None else str(mode),
        _canonical_stable_value(ref.root_action),
        _canonical_stable_value(ref.split_metadata),
        _regional_distribution_digest(ref),
    )


def prepare_regional_policy_option(
    ref: RegionPolicyOptionRef,
    *,
    key: Optional[Tuple[Any, ...]] = None,
) -> PreparedRegionalPolicyOption:
    """Normalize one option's arrays and sampling data exactly once."""
    option_key = key or canonical_region_policy_option_key(ref)
    probabilities = np.asarray(ref.distribution_payload.get("p", ()), dtype=np.float64).reshape(-1)
    if probabilities.size:
        total = float(np.sum(probabilities))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"Regional option has invalid probability total {total!r}")
        normalized = probabilities / total
    else:
        normalized = probabilities

    owners = np.asarray(ref.distribution_payload.get("owners", ()))
    troops = np.asarray(ref.distribution_payload.get("troops", ()))
    if owners.ndim == 1 and owners.size:
        owners = owners.reshape(1, -1)
    if troops.ndim == 1 and troops.size:
        troops = troops.reshape(1, -1)
    if normalized.size and (owners.ndim != 2 or troops.ndim != 2):
        raise ValueError("Regional option owners/troops arrays must be two-dimensional")
    if normalized.size and (owners.shape[0] < normalized.size or troops.shape[0] < normalized.size):
        raise ValueError("Regional option outcome arrays do not cover every probability row")

    owners_rows = tuple(
        tuple(int(x) for x in owners[i].tolist())
        for i in range(int(normalized.size))
    )
    troops_rows = tuple(
        tuple(int(x) for x in troops[i].tolist())
        for i in range(int(normalized.size))
    )
    signatures = tuple(
        (owners_rows[i], troops_rows[i])
        for i in range(int(normalized.size))
    )
    cdf = tuple(float(x) for x in np.cumsum(normalized, dtype=np.float64))
    normalized_distribution = tuple(
        (signatures[i], float(normalized[i]))
        for i in range(int(normalized.size))
    )
    metadata = {
        "option_count": int(ref.option_count),
        "split_metadata": _canonical_stable_value(ref.split_metadata),
        "topology_signature": tuple(ref.topology_signature),
        "policy_option_mode": ref.policy_option_mode,
    }
    return PreparedRegionalPolicyOption(
        key=option_key,
        region_nodes=tuple(int(x) for x in ref.region_nodes),
        attacker_nodes=tuple(int(x) for x in ref.attacker_nodes),
        defender_nodes=tuple(int(x) for x in ref.defender_nodes),
        pattern=tuple(int(x) for x in ref.pattern),
        row_label=str(ref.row_label),
        policy_option_index=int(ref.option_index),
        root_action=_canonical_stable_value(ref.root_action),
        normalized_distribution=normalized_distribution,
        distribution_signatures=signatures,
        cumulative_probabilities=cdf,
        owners_by_outcome=owners_rows,
        troops_by_outcome=troops_rows,
        mapping=tuple(sorted((int(k), int(v)) for k, v in ref.mapping.items())),
        metadata=metadata,
    )


def prepare_unique_regional_policy_options(
    candidates: Sequence[PartitionPolicyCandidate],
) -> Tuple[Dict[Tuple[Any, ...], PreparedRegionalPolicyOption], Dict[str, Any]]:
    """Prepare every exact regional option referenced by candidates once."""
    prepared: Dict[Tuple[Any, ...], PreparedRegionalPolicyOption] = {}
    references = 0
    for candidate in candidates:
        for ref in candidate.region_policy_options:
            references += 1
            key = canonical_region_policy_option_key(ref)
            if key not in prepared:
                prepared[key] = prepare_regional_policy_option(ref, key=key)
    unique = len(prepared)
    return prepared, {
        "num_candidate_region_references": int(references),
        "num_unique_region_options": int(unique),
        "reference_to_unique_ratio": float(references) / float(unique) if unique else 0.0,
    }


def derive_stable_regional_sample_seed(
    *,
    base_seed: int,
    state_identity: Any,
    regional_option_key: Tuple[Any, ...],
    scenario_index: int,
) -> int:
    """Derive a process- and order-independent regional sample seed."""
    payload = (
        "regional_sample_seed_v1",
        int(base_seed),
        _canonical_stable_value(state_identity),
        _canonical_stable_value(regional_option_key),
        int(scenario_index),
    )
    digest = hashlib.sha256(_stable_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sample_prepared_regional_option(
    prepared: PreparedRegionalPolicyOption,
    *,
    base_seed: int,
    state_identity: Any,
    scenario_index: int,
    regional_sample_plan: Optional[Mapping[Tuple[Any, int], Any]] = None,
) -> int:
    """Sample an outcome with a stable option/scenario seed or explicit plan."""
    plan_key = (prepared.key, int(scenario_index))
    if regional_sample_plan is not None and plan_key in regional_sample_plan:
        planned = regional_sample_plan[plan_key]
        if isinstance(planned, Mapping):
            planned = planned.get("outcome_index")
        outcome_index = int(planned)
    elif not prepared.cumulative_probabilities:
        outcome_index = 0
    else:
        seed = derive_stable_regional_sample_seed(
            base_seed=int(base_seed),
            state_identity=state_identity,
            regional_option_key=prepared.key,
            scenario_index=int(scenario_index),
        )
        draw = random.Random(seed).random()
        outcome_index = int(
            np.searchsorted(
                np.asarray(prepared.cumulative_probabilities, dtype=np.float64),
                draw,
                side="left",
            )
        )
    if not prepared.distribution_signatures:
        return 0
    return min(max(0, outcome_index), len(prepared.distribution_signatures) - 1)


def _partition_signature_for_candidate(
    candidate: PartitionPolicyCandidate,
) -> Tuple[Tuple[int, ...], ...]:
    signature = canonical_partition_signature(candidate.partition_regions)
    if signature:
        return signature
    return tuple(
        tuple(sorted(int(x) for x in ref.region_nodes))
        for ref in candidate.region_policy_options
    )


def prepare_partition_assembly_plans(
    candidates: Sequence[PartitionPolicyCandidate],
    *,
    base_global_state: GlobalState,
    battle_nodes: Sequence[int],
) -> Tuple[Dict[Tuple[Tuple[int, ...], ...], PreparedPartitionAssembly], Dict[str, Any]]:
    plans: Dict[Tuple[Tuple[int, ...], ...], PreparedPartitionAssembly] = {}
    for candidate in candidates:
        signature = _partition_signature_for_candidate(candidate)
        if signature in plans:
            continue
        plans[signature] = PreparedPartitionAssembly(
            partition_signature=signature,
            ordered_region_signatures=tuple(
                tuple(sorted(int(x) for x in ref.region_nodes))
                for ref in candidate.region_policy_options
            ),
            regional_node_mappings=tuple(
                tuple(sorted((int(k), int(v)) for k, v in ref.mapping.items()))
                for ref in candidate.region_policy_options
            ),
            battle_node_order=tuple(sorted(int(x) for x in battle_nodes)),
            unchanged_state_template=base_global_state,
            merge_metadata={"partition_index": int(candidate.partition_index)},
        )
    return plans, {"num_partition_plans": int(len(plans))}


def canonical_two_stage_global_state_signature(
    state: GlobalState,
    *,
    node_indices: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[int, str, int], ...]:
    """Canonical complete state key for successor evaluation caching."""
    indices = range(len(state.nodes)) if node_indices is None else sorted({int(x) for x in node_indices})
    return tuple(
        (int(idx), str(state.nodes[int(idx)].owner), int(state.nodes[int(idx)].troops))
        for idx in indices
    )


def _candidate_stable_order_key(candidate: PartitionPolicyCandidate) -> Tuple[Any, ...]:
    combo_index = 0
    if isinstance(candidate.diagnostics, Mapping):
        combo_index = int(candidate.diagnostics.get("combo_index", 0) or 0)
    return (
        int(candidate.partition_index),
        combo_index,
        _partition_signature_for_candidate(candidate),
        tuple(canonical_region_policy_option_key(ref) for ref in candidate.region_policy_options),
    )


_SECOND_STAGE_TIMING_NAMES = (
    "candidate_preparation",
    "regional_option_preparation",
    "regional_distribution_sampling",
    "partition_assembly_preparation",
    "global_state_assembly",
    "global_state_signature_creation",
    "global_state_evaluation",
    "candidate_score_aggregation",
    "diagnostic_construction",
    "total_second_stage",
)


class _SecondStageProfiler:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.timings = {name: 0.0 for name in _SECOND_STAGE_TIMING_NAMES}

    def start(self) -> Optional[float]:
        return time.perf_counter() if self.enabled else None

    def stop(self, name: str, started: Optional[float]) -> None:
        if self.enabled and started is not None:
            self.timings[name] += float(time.perf_counter() - started)


_GLOBAL_EVALUATOR_TIMING_NAMES = (
    "state_copy_or_snapshot",
    "apply_state_to_board",
    "graph_reconstruction",
    "full_graph_or_commitment_reconstruction",
    "partition_enumeration",
    "regional_library_queries",
    "partition_evaluation",
    "ranking_selection",
    "restore_board",
    "diagnostics",
    "total",
)

_GLOBAL_EVALUATOR_COUNT_NAMES = (
    "calls",
    "battle_graph_builds",
    "battle_graph_reads",
    "partition_enumerations",
    "regional_query_requests",
    "regional_query_cache_hits",
    "regional_query_cache_misses",
    "supported_regions",
    "partitions_evaluated",
    "pure_state_override_calls",
    "board_state_builds",
    "board_snapshots",
    "board_state_applies",
    "board_state_restores",
)


def _ensure_global_evaluator_profile(profile: Dict[str, Any]) -> None:
    profile["enabled"] = True
    timings = profile.setdefault("global_evaluator_timings_seconds", {})
    counts = profile.setdefault("global_evaluator_counts", {})
    for name in _GLOBAL_EVALUATOR_TIMING_NAMES:
        timings.setdefault(name, 0.0)
    for name in _GLOBAL_EVALUATOR_COUNT_NAMES:
        counts.setdefault(name, 0)
    profile.setdefault(
        "notes",
        {
            "battle_graph_source": "supplied static battle graph; no graph rebuild in ranker",
            "full_graph_or_commitment_reconstruction": "not performed by rank_battle_graph_partitions",
            "timing_partitioning": "regional query time is subtracted from enclosing partition phases",
        },
    )


def _accumulate_global_evaluator_external_timing(
    profile: Optional[Dict[str, Any]],
    *,
    timing_name: str,
    elapsed_seconds: float,
    count_name: Optional[str] = None,
) -> None:
    if profile is None:
        return
    _ensure_global_evaluator_profile(profile)
    timings = profile["global_evaluator_timings_seconds"]
    timings[timing_name] = float(timings.get(timing_name, 0.0)) + float(elapsed_seconds)
    timings["total"] = float(timings.get("total", 0.0)) + float(elapsed_seconds)
    if count_name is not None:
        counts = profile["global_evaluator_counts"]
        counts[count_name] = int(counts.get(count_name, 0)) + 1


class _GlobalEvaluatorProfiler:
    """Aggregate exact evaluator timings across one top-level two-stage call."""

    def __init__(
        self,
        profile: Optional[Dict[str, Any]],
        query_cache: Optional[Any],
    ) -> None:
        self.profile = profile
        self.query_cache = query_cache
        self.enabled = profile is not None
        self.timings = {name: 0.0 for name in _GLOBAL_EVALUATOR_TIMING_NAMES}
        self.counts = {name: 0 for name in _GLOBAL_EVALUATOR_COUNT_NAMES}
        self.total_started = time.perf_counter() if self.enabled else None
        self.initial_query_diagnostics = self._query_diagnostics()

    def start(self) -> Optional[float]:
        return time.perf_counter() if self.enabled else None

    def stop(self, name: str, started: Optional[float]) -> float:
        if not self.enabled or started is None:
            return 0.0
        elapsed = float(time.perf_counter() - started)
        self.timings[name] += elapsed
        return elapsed

    def _query_diagnostics(self) -> Dict[str, Any]:
        if not self.enabled or self.query_cache is None:
            return {}
        diagnostics = getattr(self.query_cache, "diagnostics", None)
        return dict(diagnostics() if callable(diagnostics) else {})

    def query_seconds(self) -> float:
        return float(self._query_diagnostics().get("total_request_seconds", 0.0) or 0.0)

    def stop_excluding_queries(
        self,
        name: str,
        started: Optional[float],
        query_seconds_before: float,
    ) -> None:
        elapsed = self.stop(name, started)
        if not self.enabled:
            return
        query_elapsed = max(0.0, self.query_seconds() - float(query_seconds_before))
        nested_query_elapsed = min(elapsed, query_elapsed)
        self.timings[name] -= nested_query_elapsed
        self.timings["regional_library_queries"] += nested_query_elapsed

    def increment(self, name: str, amount: int = 1) -> None:
        if self.enabled:
            self.counts[name] += int(amount)

    def finish(self) -> None:
        if not self.enabled or self.profile is None:
            return
        if self.total_started is not None:
            self.timings["total"] += float(time.perf_counter() - self.total_started)
        final_query_diagnostics = self._query_diagnostics()
        for field, count_name in (
            ("requests", "regional_query_requests"),
            ("hits", "regional_query_cache_hits"),
            ("misses", "regional_query_cache_misses"),
        ):
            self.counts[count_name] += int(final_query_diagnostics.get(field, 0) or 0) - int(
                self.initial_query_diagnostics.get(field, 0) or 0
            )
        self.counts["calls"] += 1
        _ensure_global_evaluator_profile(self.profile)
        aggregate_timings = self.profile["global_evaluator_timings_seconds"]
        aggregate_counts = self.profile["global_evaluator_counts"]
        for name, value in self.timings.items():
            aggregate_timings[name] = float(aggregate_timings.get(name, 0.0)) + float(value)
        for name, value in self.counts.items():
            aggregate_counts[name] = int(aggregate_counts.get(name, 0)) + int(value)


def _state_signature(global_state: GlobalState, battle_nodes: Sequence[int]) -> Tuple[Tuple[int, str, int], ...]:
    return canonical_two_stage_global_state_signature(global_state, node_indices=battle_nodes)


def _sample_region_option_outcome(ref: RegionPolicyOptionRef, rng: random.Random) -> int:
    p = np.asarray(ref.distribution_payload.get("p", []), dtype=np.float64)
    if p.size == 0:
        return 0
    cdf = ref.distribution_payload.get("cdf")
    if cdf is None:
        cdf_arr = np.cumsum(p)
    else:
        cdf_arr = np.asarray(cdf, dtype=np.float64)
    x = rng.random()
    idx = int(np.searchsorted(cdf_arr, x, side="left"))
    return min(idx, int(p.size) - 1)


def _apply_candidate_sample_to_global_state(
    base_state: GlobalState,
    candidate: PartitionPolicyCandidate,
    rng: random.Random,
) -> GlobalState:
    nodes = list(base_state.nodes)
    for ref in candidate.region_policy_options:
        outcome_idx = _sample_region_option_outcome(ref, rng)
        owners = np.asarray(ref.distribution_payload.get("owners"))
        troops = np.asarray(ref.distribution_payload.get("troops"))
        if owners.size == 0 or troops.size == 0:
            continue
        for local_idx, global_idx in ref.mapping.items():
            owner_val = int(owners[outcome_idx, int(local_idx)])
            owner = "A" if owner_val == 1 else "D"
            nodes[int(global_idx)] = NodeState(owner, int(troops[outcome_idx, int(local_idx)]))
    return GlobalState(nodes=tuple(nodes))


def _partition_eval_utility_tuple(ev: Optional[PartitionEvaluation]) -> Tuple[float, float, float]:
    if ev is None:
        return (0.0, 0.0, 0.0)
    return (
        float(ev.expected_new_territories),
        float(ev.expected_troops),
        float(ev.conquest_probability),
    )


def _mean_utility(values: Sequence[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    return tuple(float(sum(v[i] for v in values)) / float(len(values)) for i in range(3))  # type: ignore[return-value]


def _std_utility(
    values: Sequence[Tuple[float, float, float]],
    mean: Sequence[float],
) -> Tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    return tuple(
        float(
            np.sqrt(
                sum((float(value[i]) - float(mean[i])) ** 2 for value in values)
                / float(len(values))
            )
        )
        for i in range(3)
    )  # type: ignore[return-value]


def _assemble_prepared_candidate_state(
    *,
    base_global_state: GlobalState,
    candidate: PartitionPolicyCandidate,
    option_keys: Sequence[Tuple[Any, ...]],
    outcome_indices: Sequence[int],
    prepared_options: Mapping[Tuple[Any, ...], PreparedRegionalPolicyOption],
    assembly_plan: Optional[PreparedPartitionAssembly],
) -> GlobalState:
    template = assembly_plan.unchanged_state_template if assembly_plan is not None else base_global_state
    nodes = list(template.nodes)
    for ref, option_key, outcome_index in zip(
        candidate.region_policy_options,
        option_keys,
        outcome_indices,
    ):
        prepared = prepared_options[option_key]
        if not prepared.owners_by_outcome or not prepared.troops_by_outcome:
            continue
        index = min(max(0, int(outcome_index)), len(prepared.owners_by_outcome) - 1)
        owners_row = prepared.owners_by_outcome[index]
        troops_row = prepared.troops_by_outcome[index]
        for local_idx, global_idx in prepared.mapping:
            if local_idx >= len(owners_row) or local_idx >= len(troops_row):
                continue
            owner = "A" if int(owners_row[local_idx]) == 1 else "D"
            nodes[int(global_idx)] = NodeState(owner, int(troops_row[local_idx]))
    return GlobalState(nodes=tuple(nodes))


def _cache_can_store(cache: Mapping[Any, Any], limit: Optional[int]) -> bool:
    return limit is None or len(cache) < max(0, int(limit))


def canonical_partition_policy_candidate_identity(
    candidate: PartitionPolicyCandidate,
) -> Tuple[Any, ...]:
    """Return a semantic identity that is independent of candidate list order."""
    regional_options = tuple(
        sorted(
            (
                tuple(sorted(int(node) for node in ref.region_nodes)),
                canonical_region_policy_option_key(ref),
            )
            for ref in candidate.region_policy_options
        )
    )
    return (
        "partition_policy_candidate_v1",
        _partition_signature_for_candidate(candidate),
        regional_options,
    )


def _candidate_policy_option_indices(
    candidate: PartitionPolicyCandidate,
) -> Tuple[int, ...]:
    ordered = sorted(
        candidate.region_policy_options,
        key=lambda ref: tuple(sorted(int(node) for node in ref.region_nodes)),
    )
    return tuple(int(ref.option_index) for ref in ordered)


def _score_tuple_difference(
    higher: Optional[Sequence[float]],
    lower: Optional[Sequence[float]],
) -> Optional[Tuple[float, ...]]:
    if higher is None or lower is None:
        return None
    width = min(len(higher), len(lower))
    return tuple(float(higher[index]) - float(lower[index]) for index in range(width))


def _deterministic_spearman(
    lower_order: Sequence[Any],
    higher_order: Sequence[Any],
) -> Optional[float]:
    common = set(lower_order) & set(higher_order)
    if not common:
        return None
    if len(common) == 1:
        return 1.0
    lower_rank = {
        identity: rank
        for rank, identity in enumerate(item for item in lower_order if item in common)
    }
    higher_rank = {
        identity: rank
        for rank, identity in enumerate(item for item in higher_order if item in common)
    }
    squared = sum(
        (int(lower_rank[identity]) - int(higher_rank[identity])) ** 2
        for identity in common
    )
    count = len(common)
    return float(1.0 - (6.0 * squared) / float(count * (count * count - 1)))


def compare_candidate_selection_checkpoints(
    lower: CandidateSelectionCheckpointResult,
    higher: CandidateSelectionCheckpointResult,
) -> Dict[str, Any]:
    """Compare winner, rank, and score stability between nested checkpoints."""
    lower_ranked = tuple(
        lower.candidate_identities[index] for index in lower.candidate_rank_order
    )
    higher_ranked = tuple(
        higher.candidate_identities[index] for index in higher.candidate_rank_order
    )

    def overlap(k: int) -> Tuple[int, float]:
        left = set(lower_ranked[:k])
        right = set(higher_ranked[:k])
        intersection = len(left & right)
        union = len(left | right)
        return int(intersection), float(intersection / union) if union else 1.0

    top3_count, top3_fraction = overlap(3)
    top5_count, top5_fraction = overlap(5)
    return {
        "lower_mc_samples": int(lower.mc_samples),
        "higher_mc_samples": int(higher.mc_samples),
        "candidate_changed": bool(
            lower.selected_candidate_identity != higher.selected_candidate_identity
        ),
        "partition_changed": bool(
            lower.selected_partition_signature != higher.selected_partition_signature
        ),
        "policy_changed": bool(
            tuple(lower.selected_policy_option_indices)
            != tuple(higher.selected_policy_option_indices)
        ),
        "candidate_set_identical": bool(
            set(lower.candidate_identities) == set(higher.candidate_identities)
        ),
        "top_3_overlap": int(top3_count),
        "top_3_overlap_fraction": float(top3_fraction),
        "top_5_overlap": int(top5_count),
        "top_5_overlap_fraction": float(top5_fraction),
        "rank_correlation": _deterministic_spearman(lower_ranked, higher_ranked),
        "best_score_difference": _score_tuple_difference(
            higher.best_score_mean, lower.best_score_mean
        ),
        "best_score_std_difference": _score_tuple_difference(
            higher.best_score_std, lower.best_score_std
        ),
        "runner_up_score_difference": _score_tuple_difference(
            higher.runner_up_score_mean, lower.runner_up_score_mean
        ),
        "runner_up_score_std_difference": _score_tuple_difference(
            higher.runner_up_score_std, lower.runner_up_score_std
        ),
        "gap_difference": _score_tuple_difference(
            higher.best_runner_up_gap, lower.best_runner_up_gap
        ),
    }


def _paired_candidate_score_diagnostics(
    best_values: Sequence[Tuple[float, ...]],
    runner_values: Optional[Sequence[Tuple[float, ...]]],
    *,
    active_ranking_component: int = 0,
) -> Dict[str, Any]:
    if runner_values is None:
        return {
            "active_ranking_component": int(active_ranking_component),
            "scenario_count": 0,
            "mean_paired_difference": None,
            "standard_deviation": None,
            "standard_error": None,
            "minimum": None,
            "maximum": None,
            "fraction_positive": None,
            "scenario_score_differences": tuple(),
            "scenario_score_tuple_differences": tuple(),
        }
    tuple_differences = tuple(
        tuple(float(a) - float(b) for a, b in zip(best, runner))
        for best, runner in zip(best_values, runner_values)
    )
    if not tuple_differences:
        return _paired_candidate_score_diagnostics(
            best_values,
            None,
            active_ranking_component=active_ranking_component,
        )
    component = min(
        max(0, int(active_ranking_component)),
        max(0, len(tuple_differences[0]) - 1),
    )
    active = tuple(float(value[component]) for value in tuple_differences)
    mean = float(np.mean(active))
    std = float(np.std(active))
    return {
        "active_ranking_component": int(component),
        "scenario_count": int(len(active)),
        "mean_paired_difference": mean,
        "standard_deviation": std,
        "standard_error": float(std / np.sqrt(len(active))),
        "minimum": float(min(active)),
        "maximum": float(max(active)),
        "fraction_positive": float(sum(value > 0.0 for value in active) / len(active)),
        "scenario_score_differences": active,
        "scenario_score_tuple_differences": tuple_differences,
    }


def _coerce_nested_candidate_resume_state(
    value: Optional[Any],
) -> Optional[NestedCandidateSelectionResumeState]:
    if value is None or isinstance(value, NestedCandidateSelectionResumeState):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        payload["checkpoints"] = tuple(
            item
            if isinstance(item, CandidateSelectionCheckpointResult)
            else CandidateSelectionCheckpointResult(**dict(item))
            for item in payload.get("checkpoints", ())
        )
        return NestedCandidateSelectionResumeState(**payload)
    raise TypeError("resume_state must be a mapping or NestedCandidateSelectionResumeState")


def evaluate_candidates_at_nested_checkpoints(
    *,
    prepared_candidates: TwoStagePreparedCandidates,
    players: Sequence["Players.Player"],
    battle_graph,
    combat_libraries_base: Path,
    ranking_variable: str,
    checkpoints: Sequence[int],
    base_seed: int,
    selection_mode: str = "fixed",
    fixed_sample_count: Optional[int] = None,
    max_samples: Optional[int] = None,
    stability_required_consecutive: int = 2,
    score_gap_abs_threshold: Optional[float] = None,
    score_gap_rel_threshold: Optional[float] = None,
    resume_state: Optional[Any] = None,
    checkpoint_callback: Optional[
        Callable[
            [CandidateSelectionCheckpointResult, NestedCandidateSelectionResumeState],
            None,
        ]
    ] = None,
    global_state_utility_evaluator: Optional[
        Callable[[GlobalState], Sequence[float]]
    ] = None,
    regional_sample_plan: Optional[Mapping[Tuple[Any, int], Any]] = None,
    profile_second_stage: bool = False,
    max_cached_global_states: Optional[int] = None,
    max_cached_regional_samples: Optional[int] = None,
    max_cached_region_queries: Optional[int] = None,
) -> NestedCandidateSelectionResult:
    """Evaluate every retained candidate once per nested scenario checkpoint."""
    mode = str(selection_mode)
    if mode not in {"fixed", "adaptive_checkpoints"}:
        raise ValueError(
            f"Unknown candidate selection mode {selection_mode!r}; expected "
            "'fixed' or 'adaptive_checkpoints'."
        )
    points = tuple(int(value) for value in checkpoints)
    if not points or any(value < 1 for value in points):
        raise ValueError("candidate checkpoints must contain positive sample counts")
    if any(right <= left for left, right in zip(points, points[1:])):
        raise ValueError("candidate checkpoints must be strictly increasing")
    if int(stability_required_consecutive) < 1:
        raise ValueError("stability_required_consecutive must be >= 1")

    if mode == "fixed":
        final_limit = int(fixed_sample_count or points[-1])
        if final_limit < 1:
            raise ValueError("fixed_sample_count must be >= 1")
        evaluation_points = tuple(value for value in points if value <= final_limit)
        if final_limit not in evaluation_points:
            evaluation_points = evaluation_points + (final_limit,)
    else:
        final_limit = int(max_samples or points[-1])
        if final_limit < points[-1]:
            raise ValueError("max_samples must be >= the final candidate checkpoint")
        evaluation_points = points
        if final_limit not in evaluation_points:
            evaluation_points = evaluation_points + (final_limit,)

    candidates = tuple(
        sorted(
            prepared_candidates.retained_candidates,
            key=canonical_partition_policy_candidate_identity,
        )
    )
    candidate_identities = tuple(
        canonical_partition_policy_candidate_identity(candidate)
        for candidate in candidates
    )
    if len(set(candidate_identities)) != len(candidate_identities):
        raise ValueError("Retained candidates contain duplicate canonical identities")
    if not candidates:
        return NestedCandidateSelectionResult(
            checkpoints=tuple(),
            final_selected_candidate_index=None,
            final_selected_candidate_identity=None,
            final_checkpoint_samples=0,
            stopped_early=False,
            stopping_reason="no_valid_candidates",
            evaluated_candidates=tuple(),
            resume_state=None,
            diagnostics={
                "candidate_count": 0,
                "candidate_identities": tuple(),
                "all_candidates_retained_at_every_checkpoint": True,
            },
        )
    if len(candidates) == 1:
        return NestedCandidateSelectionResult(
            checkpoints=tuple(),
            final_selected_candidate_index=0,
            final_selected_candidate_identity=candidate_identities[0],
            final_checkpoint_samples=0,
            stopped_early=True,
            stopping_reason="single_candidate",
            evaluated_candidates=candidates,
            resume_state=None,
            diagnostics={
                "candidate_count": 1,
                "candidate_identities": candidate_identities,
                "all_candidates_retained_at_every_checkpoint": True,
                "global_candidate_evaluations_skipped": True,
            },
        )

    state_identity = canonical_two_stage_global_state_signature(
        prepared_candidates.global_state
    )
    resumed = _coerce_nested_candidate_resume_state(resume_state)
    if resumed is not None:
        if resumed.schema_version != "nested_candidate_selection_resume_v1":
            raise ValueError("Unsupported nested candidate resume schema")
        if int(resumed.base_seed) != int(base_seed):
            raise ValueError("Candidate-selection resume base seed mismatch")
        if resumed.state_identity != state_identity:
            raise ValueError("Candidate-selection resume state identity mismatch")
        if tuple(resumed.candidate_identities) != candidate_identities:
            raise ValueError("Candidate-selection resume candidate identity mismatch")
        completed_scenarios = int(resumed.completed_scenarios)
        utility_records = [list(values) for values in resumed.candidate_utilities]
        state_sequences = [list(values) for values in resumed.candidate_state_sequences]
        sampled_regional_outcomes = dict(resumed.sampled_regional_outcomes)
        global_evaluation_cache = dict(resumed.global_evaluation_cache)
        unique_global_signatures = set(resumed.unique_global_signatures)
        checkpoint_results = list(resumed.checkpoints)
        prior_runtime = float(resumed.runtime_cumulative_seconds)
        counters = dict(resumed.counters)
    else:
        completed_scenarios = 0
        utility_records = [[] for _ in candidates]
        state_sequences = [[] for _ in candidates]
        sampled_regional_outcomes: Dict[Any, int] = {}
        global_evaluation_cache: Dict[Any, Tuple[float, ...]] = {}
        unique_global_signatures: Set[Any] = set()
        checkpoint_results: List[CandidateSelectionCheckpointResult] = []
        prior_runtime = 0.0
        counters = {
            "regional_sample_requests": 0,
            "regional_samples_generated": 0,
            "regional_sample_cache_hits": 0,
            "global_states_assembled": 0,
            "global_evaluation_calls": 0,
            "global_evaluation_cache_hits": 0,
        }
    if completed_scenarios > final_limit:
        raise ValueError("Resume state is beyond the requested final sample count")

    candidate_option_keys = tuple(
        tuple(canonical_region_policy_option_key(ref) for ref in candidate.region_policy_options)
        for candidate in candidates
    )
    prepared_options, option_diagnostics = prepare_unique_regional_policy_options(candidates)
    assembly_plans, assembly_diagnostics = prepare_partition_assembly_plans(
        candidates,
        base_global_state=prepared_candidates.global_state,
        battle_nodes=prepared_candidates.battle_nodes,
    )
    region_query_cache = agop.RegionQueryResultCache(
        max_entries=max_cached_region_queries,
        profile_timings=bool(profile_second_stage),
        cache_library_resources=True,
    )
    global_evaluator_profile: Optional[Dict[str, Any]] = (
        {} if profile_second_stage else None
    )
    call_started = time.perf_counter()
    stopped_reason: Optional[str] = None

    def make_resume_state(runtime_cumulative: float) -> NestedCandidateSelectionResumeState:
        return NestedCandidateSelectionResumeState(
            schema_version="nested_candidate_selection_resume_v1",
            base_seed=int(base_seed),
            state_identity=state_identity,
            candidate_identities=candidate_identities,
            completed_scenarios=int(completed_scenarios),
            candidate_utilities=tuple(
                tuple(tuple(float(x) for x in value) for value in values)
                for values in utility_records
            ),
            candidate_state_sequences=tuple(tuple(values) for values in state_sequences),
            sampled_regional_outcomes=dict(sampled_regional_outcomes),
            global_evaluation_cache=dict(global_evaluation_cache),
            unique_global_signatures=tuple(sorted(unique_global_signatures)),
            checkpoints=tuple(checkpoint_results),
            runtime_cumulative_seconds=float(runtime_cumulative),
            counters=dict(counters),
        )

    for checkpoint_samples in evaluation_points:
        if checkpoint_samples <= completed_scenarios:
            continue
        previous_unique_count = (
            checkpoint_results[-1].unique_global_states_cumulative
            if checkpoint_results
            else 0
        )
        previous_runtime = (
            checkpoint_results[-1].runtime_cumulative_seconds
            if checkpoint_results
            else 0.0
        )
        for scenario_index in range(completed_scenarios, checkpoint_samples):
            for candidate_index, candidate in enumerate(candidates):
                option_keys = candidate_option_keys[candidate_index]
                outcome_indices: List[int] = []
                for option_key in option_keys:
                    counters["regional_sample_requests"] += 1
                    sample_key = (option_key, int(scenario_index))
                    if sample_key in sampled_regional_outcomes:
                        outcome_index = int(sampled_regional_outcomes[sample_key])
                        counters["regional_sample_cache_hits"] += 1
                    else:
                        outcome_index = sample_prepared_regional_option(
                            prepared_options[option_key],
                            base_seed=int(base_seed),
                            state_identity=state_identity,
                            scenario_index=int(scenario_index),
                            regional_sample_plan=regional_sample_plan,
                        )
                        counters["regional_samples_generated"] += 1
                        if _cache_can_store(
                            sampled_regional_outcomes, max_cached_regional_samples
                        ):
                            sampled_regional_outcomes[sample_key] = int(outcome_index)
                    outcome_indices.append(int(outcome_index))

                partition_signature = _partition_signature_for_candidate(candidate)
                sampled_state = _assemble_prepared_candidate_state(
                    base_global_state=prepared_candidates.global_state,
                    candidate=candidate,
                    option_keys=option_keys,
                    outcome_indices=outcome_indices,
                    prepared_options=prepared_options,
                    assembly_plan=assembly_plans.get(partition_signature),
                )
                counters["global_states_assembled"] += 1
                complete_signature = canonical_two_stage_global_state_signature(
                    sampled_state
                )
                unique_global_signatures.add(complete_signature)
                if complete_signature in global_evaluation_cache:
                    utility = tuple(global_evaluation_cache[complete_signature])
                    counters["global_evaluation_cache_hits"] += 1
                else:
                    if global_state_utility_evaluator is not None:
                        raw_utility = global_state_utility_evaluator(sampled_state)
                        utility = tuple(float(value) for value in raw_utility)
                    else:
                        ranked = rank_battle_graph_partitions(
                            players=players,
                            battle_graph=battle_graph,
                            combat_libraries_base=combat_libraries_base,
                            max_partitions=40,
                            ranking_variable=ranking_variable,
                            lookahead_depth=0,
                            use_monte_carlo=False,
                            policy_option_selection="primary",
                            global_state_override=sampled_state,
                            region_query_cache=region_query_cache,
                            global_evaluator_profile=global_evaluator_profile,
                        )
                        utility = _partition_eval_utility_tuple(
                            ranked.get("best_evaluation")
                        )
                    if len(utility) != 3:
                        raise ValueError(
                            "Global state utility evaluator must return three score components"
                        )
                    counters["global_evaluation_calls"] += 1
                    if _cache_can_store(global_evaluation_cache, max_cached_global_states):
                        global_evaluation_cache[complete_signature] = tuple(utility)
                utility_records[candidate_index].append(tuple(utility))
                state_sequences[candidate_index].append(
                    _state_signature(sampled_state, prepared_candidates.battle_nodes)
                )
        completed_scenarios = int(checkpoint_samples)

        means = tuple(_mean_utility(values) for values in utility_records)
        standard_deviations = tuple(
            _std_utility(values, means[index])
            for index, values in enumerate(utility_records)
        )
        rank_order = list(range(len(candidates)))
        rank_order.sort(key=lambda index: means[index], reverse=True)
        best_index = int(rank_order[0])
        runner_index = int(rank_order[1]) if len(rank_order) > 1 else None
        best_candidate = candidates[best_index]
        runner_candidate = candidates[runner_index] if runner_index is not None else None
        gap = (
            _score_tuple_difference(means[best_index], means[runner_index])
            if runner_index is not None
            else None
        )
        active_component = next(
            (
                index
                for index, value in enumerate(gap or ())
                if abs(float(value)) > 1e-15
            ),
            0,
        )
        paired = _paired_candidate_score_diagnostics(
            utility_records[best_index],
            utility_records[runner_index] if runner_index is not None else None,
            active_ranking_component=active_component,
        )

        for index, candidate in enumerate(candidates):
            counts: Dict[Any, int] = {}
            for signature in state_sequences[index]:
                counts[signature] = counts.get(signature, 0) + 1
            candidate.mc_mean_second_stage_utility = means[index]
            candidate.mc_std_second_stage_utility = standard_deviations[index]
            candidate.mc_mean_score = float(means[index][0])
            candidate.mc_num_scenarios = int(completed_scenarios)
            candidate.mc_final_state_counts = counts
            candidate.mc_final_state_sequence = tuple(state_sequences[index])

        runtime_cumulative = float(
            prior_runtime + (time.perf_counter() - call_started)
        )
        checkpoint_result = CandidateSelectionCheckpointResult(
            mc_samples=int(completed_scenarios),
            selected_candidate_index=best_index,
            selected_candidate_identity=candidate_identities[best_index],
            selected_partition_signature=_partition_signature_for_candidate(best_candidate),
            selected_policy_option_indices=_candidate_policy_option_indices(best_candidate),
            best_score_mean=means[best_index],
            best_score_std=standard_deviations[best_index],
            runner_up_candidate_index=runner_index,
            runner_up_candidate_identity=(
                candidate_identities[runner_index] if runner_index is not None else None
            ),
            runner_up_score_mean=(means[runner_index] if runner_index is not None else None),
            runner_up_score_std=(
                standard_deviations[runner_index] if runner_index is not None else None
            ),
            best_runner_up_gap=gap,
            candidate_rank_order=tuple(rank_order),
            candidate_identities=candidate_identities,
            top_candidate_indices=tuple(rank_order[:5]),
            runtime_increment_seconds=float(runtime_cumulative - previous_runtime),
            runtime_cumulative_seconds=runtime_cumulative,
            unique_global_states_increment=int(
                len(unique_global_signatures) - previous_unique_count
            ),
            unique_global_states_cumulative=int(len(unique_global_signatures)),
            diagnostics={
                "candidate_count": int(len(candidates)),
                "candidate_sample_counts": tuple(
                    len(values) for values in utility_records
                ),
                "candidate_identities": candidate_identities,
                "paired_best_runner_up": paired,
                "all_candidates_retained": True,
            },
        )
        checkpoint_results.append(checkpoint_result)

        current_resume = make_resume_state(runtime_cumulative)
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint_result, current_resume)

        if mode == "fixed" and completed_scenarios >= final_limit:
            stopped_reason = "fixed_sample_count_reached"
            break
        if mode == "adaptive_checkpoints":
            selected_identities = [
                item.selected_candidate_identity for item in checkpoint_results
            ]
            required = int(stability_required_consecutive)
            stable = bool(
                len(selected_identities) >= required
                and len(set(selected_identities[-required:])) == 1
            )
            gap_passes = True
            active_gap = (
                float(gap[active_component])
                if gap is not None and gap
                else float("inf")
            )
            if score_gap_abs_threshold is not None:
                gap_passes = gap_passes and active_gap >= float(score_gap_abs_threshold)
            if score_gap_rel_threshold is not None and runner_index is not None:
                scale = max(
                    abs(float(means[best_index][active_component])),
                    abs(float(means[runner_index][active_component])),
                    1e-12,
                )
                gap_passes = gap_passes and (
                    active_gap / scale >= float(score_gap_rel_threshold)
                )
            thresholds_configured = (
                score_gap_abs_threshold is not None
                or score_gap_rel_threshold is not None
            )
            if completed_scenarios < final_limit and stable and gap_passes:
                stopped_reason = (
                    "stable_candidate_and_gap"
                    if thresholds_configured
                    else "stable_candidate"
                )
                break
            if completed_scenarios >= final_limit:
                stopped_reason = "maximum_samples_reached"
                break

    if not checkpoint_results:
        raise RuntimeError("Candidate selection produced no checkpoints")
    final_checkpoint = checkpoint_results[-1]
    final_runtime = float(
        prior_runtime + (time.perf_counter() - call_started)
    )
    final_resume = make_resume_state(final_runtime)
    comparisons = tuple(
        compare_candidate_selection_checkpoints(lower, higher)
        for lower, higher in zip(checkpoint_results, checkpoint_results[1:])
    )
    candidate_sets_unchanged = all(
        tuple(item.candidate_identities) == candidate_identities
        for item in checkpoint_results
    )
    return NestedCandidateSelectionResult(
        checkpoints=tuple(checkpoint_results),
        final_selected_candidate_index=final_checkpoint.selected_candidate_index,
        final_selected_candidate_identity=final_checkpoint.selected_candidate_identity,
        final_checkpoint_samples=int(final_checkpoint.mc_samples),
        stopped_early=bool(final_checkpoint.mc_samples < final_limit),
        stopping_reason=stopped_reason or "maximum_samples_reached",
        evaluated_candidates=candidates,
        resume_state=final_resume,
        diagnostics={
            "candidate_count": int(len(candidates)),
            "candidate_identities": candidate_identities,
            "checkpoint_comparisons": comparisons,
            "all_candidates_retained_at_every_checkpoint": bool(
                candidate_sets_unchanged
            ),
            "candidate_set_sizes": tuple(
                len(item.candidate_identities) for item in checkpoint_results
            ),
            "prepared_regional_options": option_diagnostics,
            "prepared_partition_assemblies": assembly_diagnostics,
            "counters": dict(counters),
            "region_query_cache": region_query_cache.diagnostics(),
            "global_evaluator_profile": global_evaluator_profile,
            "resume_reuses_completed_scenarios": True,
        },
    )


def _monte_carlo_tiebreak_partition_policy_candidates(
    *,
    candidates: Sequence[PartitionPolicyCandidate],
    players: Sequence["Players.Player"],
    battle_graph,
    combat_libraries_base: Path,
    base_global_state: GlobalState,
    battle_nodes: Sequence[int],
    ranking_variable: str,
    two_stage_mc_scenarios: int = 100,
    two_stage_mc_seed: Optional[int] = None,
    track_empirical_final_distribution: bool = True,
    max_tracked_final_states: int = 100,
    second_stage_execution_mode: str = "legacy",
    second_stage_sampling_mode: str = "legacy_sequential_rng",
    profile_second_stage: bool = False,
    regional_sample_plan: Optional[Mapping[Tuple[Any, int], Any]] = None,
    max_cached_global_states: Optional[int] = None,
    max_cached_regional_samples: Optional[int] = None,
    max_cached_region_queries: Optional[int] = None,
    second_stage_diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[PartitionPolicyCandidate], Tuple[PartitionPolicyCandidate, ...]]:
    execution_mode = str(second_stage_execution_mode)
    sampling_mode = str(second_stage_sampling_mode)
    if execution_mode not in {"legacy", "optimized_reuse"}:
        raise ValueError(
            f"Unknown second_stage_execution_mode={second_stage_execution_mode!r}; "
            "expected 'legacy' or 'optimized_reuse'."
        )
    if sampling_mode not in {"legacy_sequential_rng", "stable_region_option_scenarios"}:
        raise ValueError(
            f"Unknown second_stage_sampling_mode={second_stage_sampling_mode!r}; "
            "expected 'legacy_sequential_rng' or 'stable_region_option_scenarios'."
        )

    profiler = _SecondStageProfiler(profile_second_stage)
    total_started = profiler.start()
    scenarios = max(0, int(two_stage_mc_scenarios))
    base_seed = int(two_stage_mc_seed or 0)
    state_identity = canonical_two_stage_global_state_signature(base_global_state)

    candidate_started = profiler.start()
    candidate_entries = []
    for candidate in candidates:
        option_keys = tuple(
            canonical_region_policy_option_key(ref)
            for ref in candidate.region_policy_options
        )
        combo_index = 0
        if isinstance(candidate.diagnostics, Mapping):
            combo_index = int(candidate.diagnostics.get("combo_index", 0) or 0)
        candidate_entries.append(
            (
                candidate,
                option_keys,
                _partition_signature_for_candidate(candidate),
                (
                    int(candidate.partition_index),
                    combo_index,
                    _partition_signature_for_candidate(candidate),
                    option_keys,
                ),
            )
        )
    stable_sampling = bool(
        sampling_mode == "stable_region_option_scenarios"
        or regional_sample_plan is not None
    )
    if stable_sampling:
        candidate_entries.sort(key=lambda item: item[3])
    profiler.stop("candidate_preparation", candidate_started)

    regional_started = profiler.start()
    prepared_options, option_diag = prepare_unique_regional_policy_options(
        tuple(entry[0] for entry in candidate_entries)
    )
    profiler.stop("regional_option_preparation", regional_started)

    assembly_started = profiler.start()
    partition_plans: Dict[Tuple[Tuple[int, ...], ...], PreparedPartitionAssembly] = {}
    partition_diag = {"num_partition_plans": 0}
    if execution_mode == "optimized_reuse":
        partition_plans, partition_diag = prepare_partition_assembly_plans(
            tuple(entry[0] for entry in candidate_entries),
            base_global_state=base_global_state,
            battle_nodes=battle_nodes,
        )
    profiler.stop("partition_assembly_preparation", assembly_started)

    sequential_rng = random.Random(two_stage_mc_seed)
    sampled_regional_outcomes: Dict[Tuple[Tuple[Any, ...], int], int] = {}
    global_evaluation_cache: Dict[
        Tuple[Tuple[int, str, int], ...], Tuple[float, float, float]
    ] = {}
    global_evaluator_profile: Optional[Dict[str, Any]] = (
        {} if profile_second_stage else None
    )
    if execution_mode == "optimized_reuse":
        region_query_cache = agop.RegionQueryResultCache(
            max_entries=max_cached_region_queries,
            profile_timings=profile_second_stage,
            cache_library_resources=True,
        )
    elif profile_second_stage:
        # Observe every legacy query without retaining or reusing any result.
        region_query_cache = agop.RegionQueryResultCache(
            max_entries=0,
            profile_timings=True,
            cache_library_resources=False,
        )
    else:
        region_query_cache = None
    unique_regional_sample_keys: Set[Tuple[Tuple[Any, ...], int]] = set()
    unique_global_signatures: Set[Tuple[Tuple[int, str, int], ...]] = set()
    physical_regional_samples = 0
    regional_sample_cache_hits = 0
    global_evaluation_calls = 0
    global_evaluation_cache_hits = 0
    global_states_assembled = 0
    skipped_global_cache_stores = 0
    skipped_regional_cache_stores = 0

    board_snapshot = None
    if execution_mode == "legacy":
        snapshot_started = time.perf_counter() if profile_second_stage else None
        board_snapshot = _snapshot_board_state()
        if snapshot_started is not None:
            _accumulate_global_evaluator_external_timing(
                global_evaluator_profile,
                timing_name="state_copy_or_snapshot",
                elapsed_seconds=time.perf_counter() - snapshot_started,
                count_name="board_snapshots",
            )
    evaluated: List[PartitionPolicyCandidate] = []
    try:
        for candidate, option_keys, partition_signature, _ in candidate_entries:
            utilities: List[Tuple[float, float, float]] = []
            counts: Dict[Tuple[Tuple[int, str, int], ...], int] = {}
            state_sequence: List[Tuple[Tuple[int, str, int], ...]] = []
            for scenario_index in range(scenarios):
                sample_started = profiler.start()
                outcome_indices: List[int] = []
                for ref, option_key in zip(candidate.region_policy_options, option_keys):
                    sample_key = (option_key, int(scenario_index))
                    unique_regional_sample_keys.add(sample_key)
                    if (
                        execution_mode == "optimized_reuse"
                        and stable_sampling
                        and sample_key in sampled_regional_outcomes
                    ):
                        outcome_index = sampled_regional_outcomes[sample_key]
                        regional_sample_cache_hits += 1
                    else:
                        if stable_sampling:
                            outcome_index = sample_prepared_regional_option(
                                prepared_options[option_key],
                                base_seed=base_seed,
                                state_identity=state_identity,
                                scenario_index=scenario_index,
                                regional_sample_plan=regional_sample_plan,
                            )
                        else:
                            outcome_index = _sample_region_option_outcome(ref, sequential_rng)
                        physical_regional_samples += 1
                        if execution_mode == "optimized_reuse" and stable_sampling:
                            if _cache_can_store(
                                sampled_regional_outcomes,
                                max_cached_regional_samples,
                            ):
                                sampled_regional_outcomes[sample_key] = int(outcome_index)
                            else:
                                skipped_regional_cache_stores += 1
                    outcome_indices.append(int(outcome_index))
                profiler.stop("regional_distribution_sampling", sample_started)

                state_started = profiler.start()
                sampled_state = _assemble_prepared_candidate_state(
                    base_global_state=base_global_state,
                    candidate=candidate,
                    option_keys=option_keys,
                    outcome_indices=outcome_indices,
                    prepared_options=prepared_options,
                    assembly_plan=partition_plans.get(partition_signature),
                )
                global_states_assembled += 1
                profiler.stop("global_state_assembly", state_started)

                signature_started = profiler.start()
                complete_signature = canonical_two_stage_global_state_signature(sampled_state)
                unique_global_signatures.add(complete_signature)
                profiler.stop("global_state_signature_creation", signature_started)

                evaluation_started = profiler.start()
                if (
                    execution_mode == "optimized_reuse"
                    and complete_signature in global_evaluation_cache
                ):
                    utility = global_evaluation_cache[complete_signature]
                    global_evaluation_cache_hits += 1
                else:
                    if execution_mode == "legacy":
                        apply_started = time.perf_counter() if profile_second_stage else None
                        _apply_global_state_to_board(sampled_state, players)
                        if apply_started is not None:
                            _accumulate_global_evaluator_external_timing(
                                global_evaluator_profile,
                                timing_name="apply_state_to_board",
                                elapsed_seconds=time.perf_counter() - apply_started,
                                count_name="board_state_applies",
                            )
                        result = rank_battle_graph_partitions(
                            players=players,
                            battle_graph=battle_graph,
                            combat_libraries_base=combat_libraries_base,
                            max_partitions=40,
                            ranking_variable=ranking_variable,
                            lookahead_depth=0,
                            use_monte_carlo=False,
                            policy_option_selection="primary",
                            region_query_cache=region_query_cache,
                            global_evaluator_profile=global_evaluator_profile,
                        )
                    else:
                        result = rank_battle_graph_partitions(
                            players=players,
                            battle_graph=battle_graph,
                            combat_libraries_base=combat_libraries_base,
                            max_partitions=40,
                            ranking_variable=ranking_variable,
                            lookahead_depth=0,
                            use_monte_carlo=False,
                            policy_option_selection="primary",
                            global_state_override=sampled_state,
                            region_query_cache=region_query_cache,
                            global_evaluator_profile=global_evaluator_profile,
                        )
                    utility = _partition_eval_utility_tuple(result.get("best_evaluation"))
                    global_evaluation_calls += 1
                    if execution_mode == "optimized_reuse":
                        if _cache_can_store(
                            global_evaluation_cache,
                            max_cached_global_states,
                        ):
                            global_evaluation_cache[complete_signature] = utility
                        else:
                            skipped_global_cache_stores += 1
                profiler.stop("global_state_evaluation", evaluation_started)
                utilities.append(utility)
                if track_empirical_final_distribution:
                    sig = _state_signature(sampled_state, battle_nodes)
                    state_sequence.append(sig)
                    counts[sig] = counts.get(sig, 0) + 1

            aggregation_started = profiler.start()
            mean_u = _mean_utility(utilities)
            std_u = _std_utility(utilities, mean_u)
            candidate.mc_mean_second_stage_utility = mean_u
            candidate.mc_std_second_stage_utility = std_u
            candidate.mc_mean_score = float(mean_u[0])
            candidate.mc_num_scenarios = int(scenarios)
            candidate.mc_final_state_counts = counts if track_empirical_final_distribution else None
            candidate.mc_final_state_sequence = (
                tuple(state_sequence) if track_empirical_final_distribution else None
            )
            if track_empirical_final_distribution:
                top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: int(max_tracked_final_states)]
                candidate.mc_top_final_states = [
                    {"signature": sig, "count": count, "prob_hat": float(count) / float(max(1, scenarios))}
                    for sig, count in top
                ]
            candidate_diag = dict(candidate.diagnostics or {})
            candidate_diag.update(
                {
                    "second_stage_execution_mode": execution_mode,
                    "second_stage_sampling_mode": sampling_mode,
                    "second_stage_unique_final_states": int(len(counts)),
                }
            )
            candidate.diagnostics = candidate_diag
            evaluated.append(candidate)
            profiler.stop("candidate_score_aggregation", aggregation_started)
    finally:
        if board_snapshot is not None:
            restore_started = time.perf_counter() if profile_second_stage else None
            _restore_board_state(board_snapshot)
            if restore_started is not None:
                _accumulate_global_evaluator_external_timing(
                    global_evaluator_profile,
                    timing_name="restore_board",
                    elapsed_seconds=time.perf_counter() - restore_started,
                    count_name="board_state_restores",
                )

    if not evaluated:
        if second_stage_diagnostics is not None:
            second_stage_diagnostics.update(
                {
                    "second_stage_execution_mode": execution_mode,
                    "second_stage_sampling_mode": sampling_mode,
                    "global_evaluator_profile": global_evaluator_profile,
                }
            )
        return None, tuple()
    selected = max(evaluated, key=lambda c: tuple(c.mc_mean_second_stage_utility or (0.0, 0.0, 0.0)))

    diagnostic_started = profiler.start()
    references = int(option_diag.get("num_candidate_region_references", 0))
    regional_requests = int(references * scenarios)
    region_query_diag = (
        region_query_cache.diagnostics()
        if region_query_cache is not None
        else {
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "failure_hits": 0,
            "stores": 0,
            "skipped_stores": 0,
            "max_entries": max_cached_region_queries,
            "profile_timings": False,
            "requests": 0,
            "total_request_seconds": 0.0,
            "hit_request_seconds": 0.0,
            "miss_request_seconds": 0.0,
        }
    )
    counts_diag = {
        "num_candidates": int(len(candidate_entries)),
        "num_partitions": int(len({_partition_signature_for_candidate(c) for c in candidates})),
        "num_candidate_region_references": references,
        "num_unique_region_options": int(option_diag.get("num_unique_region_options", 0)),
        "num_scenarios": int(scenarios),
        "regional_sample_requests": regional_requests,
        "unique_regional_samples": int(len(unique_regional_sample_keys)),
        "regional_samples_generated": int(physical_regional_samples),
        "regional_sample_cache_hits": int(regional_sample_cache_hits),
        "global_states_assembled": int(global_states_assembled),
        "unique_global_state_signatures": int(len(unique_global_signatures)),
        "global_evaluation_calls": int(global_evaluation_calls),
        "global_evaluation_cache_hits": int(global_evaluation_cache_hits),
        "regional_sample_cache_entries": int(len(sampled_regional_outcomes)),
        "global_evaluation_cache_entries": int(len(global_evaluation_cache)),
        "skipped_regional_sample_cache_stores": int(skipped_regional_cache_stores),
        "skipped_global_evaluation_cache_stores": int(skipped_global_cache_stores),
        "region_query_cache_entries": int(region_query_diag.get("entries", 0)),
        "region_query_cache_hits": int(region_query_diag.get("hits", 0)),
        "region_query_cache_misses": int(region_query_diag.get("misses", 0)),
    }
    reuse_diag = {
        "state_level_reuse": {
            "base_state_prepared_once": True,
            "global_evaluator_context_prepared_once": execution_mode == "optimized_reuse",
            "region_query_reuse_enabled": execution_mode == "optimized_reuse",
            "region_query_cache": region_query_diag,
        },
        "partition_level_reuse": {
            "num_partition_plans": int(partition_diag.get("num_partition_plans", 0)),
        },
        "regional_option_level_reuse": {
            "num_unique_options": int(option_diag.get("num_unique_region_options", 0)),
            "candidate_region_references": references,
            "reference_to_unique_ratio": float(option_diag.get("reference_to_unique_ratio", 0.0)),
        },
        "scenario_level_reuse": {
            "regional_sample_requests_without_reuse": regional_requests,
            "regional_samples_generated": int(physical_regional_samples),
            "regional_samples_reused": int(max(0, regional_requests - physical_regional_samples)),
        },
        "global_state_level_reuse": {
            "assembled_states": int(global_states_assembled),
            "unique_states": int(len(unique_global_signatures)),
            "evaluation_cache_hits": int(global_evaluation_cache_hits),
        },
    }
    profiler.stop("diagnostic_construction", diagnostic_started)
    profiler.stop("total_second_stage", total_started)
    profile_diag = {
        "enabled": bool(profile_second_stage),
        "execution_mode": execution_mode,
        "sampling_mode": sampling_mode,
        "timings_seconds": dict(profiler.timings),
        "counts": counts_diag,
        "global_evaluator_profile": global_evaluator_profile,
    }
    if second_stage_diagnostics is not None:
        second_stage_diagnostics.update(
            {
                "second_stage_execution_mode": execution_mode,
                "second_stage_sampling_mode": sampling_mode,
                "second_stage_profile": profile_diag,
                "second_stage_reuse": reuse_diag,
                "global_evaluator_profile": global_evaluator_profile,
                "second_stage_cache_limits": {
                    "max_cached_global_states": max_cached_global_states,
                    "max_cached_regional_samples": max_cached_regional_samples,
                    "max_cached_region_queries": max_cached_region_queries,
                },
            }
        )
    return selected, tuple(evaluated)


def rank_battle_graph_partition_policy_candidates_two_stage(
    players: Sequence["Players.Player"],
    battle_graph,
    combat_libraries_base: Path,
    max_partitions: int = 40,
    ranking_variable: str = "battle_expected_attacker_territory_count",
    *,
    first_stage_value_tolerances: Optional[Tuple[float, ...]] = None,
    max_policy_combos_per_partition: Optional[int] = 256,
    max_total_partition_policy_candidates: Optional[int] = None,
    partition_candidate_selection_mode: str = "maximal_per_partition_utility",
    utility_abs_tolerance: Optional[float] = None,
    utility_rel_tolerance: Optional[float] = None,
    max_candidates_per_partition: Optional[int] = None,
    two_stage_mc_scenarios: int = 100,
    two_stage_mc_seed: Optional[int] = None,
    track_empirical_final_distribution: bool = True,
    max_tracked_final_states: int = 100,
    run_expensive_cover_diagnostics: bool = False,
    second_stage_execution_mode: str = "optimized_reuse",
    second_stage_sampling_mode: str = "stable_region_option_scenarios",
    profile_second_stage: bool = False,
    regional_sample_plan: Optional[Mapping[Tuple[Any, int], Any]] = None,
    max_cached_global_states: Optional[int] = None,
    max_cached_regional_samples: Optional[int] = None,
    max_cached_region_queries: Optional[int] = None,
) -> TwoStagePartitionPolicyResult:
    total_two_stage_started = time.perf_counter()
    if second_stage_execution_mode not in {"legacy", "optimized_reuse"}:
        raise ValueError(
            f"Unknown second_stage_execution_mode={second_stage_execution_mode!r}; "
            "expected 'legacy' or 'optimized_reuse'."
        )
    if second_stage_sampling_mode not in {
        "legacy_sequential_rng",
        "stable_region_option_scenarios",
    }:
        raise ValueError(
            f"Unknown second_stage_sampling_mode={second_stage_sampling_mode!r}; "
            "expected 'legacy_sequential_rng' or 'stable_region_option_scenarios'."
        )
    first_stage_started = time.perf_counter()
    prepared = prepare_two_stage_partition_policy_candidates(
        players=players,
        battle_graph=battle_graph,
        combat_libraries_base=combat_libraries_base,
        max_partitions=max_partitions,
        ranking_variable=ranking_variable,
        first_stage_value_tolerances=first_stage_value_tolerances,
        max_policy_combos_per_partition=max_policy_combos_per_partition,
        max_total_partition_policy_candidates=max_total_partition_policy_candidates,
        partition_candidate_selection_mode=partition_candidate_selection_mode,
        utility_abs_tolerance=utility_abs_tolerance,
        utility_rel_tolerance=utility_rel_tolerance,
        max_candidates_per_partition=max_candidates_per_partition,
        run_expensive_cover_diagnostics=run_expensive_cover_diagnostics,
    )
    diagnostics = prepared.diagnostics
    diagnostics["first_stage_runtime_seconds"] = float(
        time.perf_counter() - first_stage_started
    )
    diagnostics["second_stage_execution_mode"] = str(second_stage_execution_mode)
    diagnostics["second_stage_sampling_mode"] = str(second_stage_sampling_mode)
    diagnostics["profile_second_stage"] = bool(profile_second_stage)
    all_candidates = prepared.all_candidates
    optimal = prepared.retained_candidates
    best_utility = prepared.best_utility

    if not all_candidates:
        diagnostics["second_stage_runtime_seconds"] = 0.0
        diagnostics["total_two_stage_runtime_seconds"] = float(
            time.perf_counter() - total_two_stage_started
        )
        return TwoStagePartitionPolicyResult(None, (), 0, None, int(two_stage_mc_scenarios), diagnostics)

    second_stage_started = time.perf_counter()
    if len(optimal) == 1 and int(two_stage_mc_scenarios) <= 0:
        selected = optimal[0]
        evaluated_optimal = optimal
    elif len(optimal) == 1 and int(two_stage_mc_scenarios) > 0:
        selected, evaluated_optimal = _monte_carlo_tiebreak_partition_policy_candidates(
            candidates=optimal,
            players=players,
            battle_graph=battle_graph,
            combat_libraries_base=combat_libraries_base,
            base_global_state=prepared.global_state,
            battle_nodes=prepared.battle_nodes,
            ranking_variable=ranking_variable,
            two_stage_mc_scenarios=two_stage_mc_scenarios,
            two_stage_mc_seed=two_stage_mc_seed,
            track_empirical_final_distribution=track_empirical_final_distribution,
            max_tracked_final_states=max_tracked_final_states,
            second_stage_execution_mode=second_stage_execution_mode,
            second_stage_sampling_mode=second_stage_sampling_mode,
            profile_second_stage=profile_second_stage,
            regional_sample_plan=regional_sample_plan,
            max_cached_global_states=max_cached_global_states,
            max_cached_regional_samples=max_cached_regional_samples,
            max_cached_region_queries=max_cached_region_queries,
            second_stage_diagnostics=diagnostics,
        )
    else:
        selected, evaluated_optimal = _monte_carlo_tiebreak_partition_policy_candidates(
            candidates=optimal,
            players=players,
            battle_graph=battle_graph,
            combat_libraries_base=combat_libraries_base,
            base_global_state=prepared.global_state,
            battle_nodes=prepared.battle_nodes,
            ranking_variable=ranking_variable,
            two_stage_mc_scenarios=two_stage_mc_scenarios,
            two_stage_mc_seed=two_stage_mc_seed,
            track_empirical_final_distribution=track_empirical_final_distribution,
            max_tracked_final_states=max_tracked_final_states,
            second_stage_execution_mode=second_stage_execution_mode,
            second_stage_sampling_mode=second_stage_sampling_mode,
            profile_second_stage=profile_second_stage,
            regional_sample_plan=regional_sample_plan,
            max_cached_global_states=max_cached_global_states,
            max_cached_regional_samples=max_cached_regional_samples,
            max_cached_region_queries=max_cached_region_queries,
            second_stage_diagnostics=diagnostics,
        )

    diagnostics["selected_partition_signature"] = (
        canonical_partition_signature(selected.partition_regions) if selected is not None else None
    )
    diagnostics["second_stage_runtime_seconds"] = float(
        time.perf_counter() - second_stage_started
    )
    diagnostics["total_two_stage_runtime_seconds"] = float(
        time.perf_counter() - total_two_stage_started
    )
    return TwoStagePartitionPolicyResult(
        selected_candidate=selected,
        first_stage_optimal_candidates=tuple(evaluated_optimal),
        all_candidate_count=len(all_candidates),
        first_stage_best_utility=best_utility,
        mc_scenarios=int(two_stage_mc_scenarios),
        diagnostics=diagnostics,
    )

def rank_battle_graph_partitions(
    players: Sequence["Players.Player"],
    battle_graph,
    combat_libraries_base: Path,
    max_partitions: int = 40,
    ranking_variable: str = "battle_expected_attacker_territory_count",
    lookahead_depth: int = 0,
    use_monte_carlo: bool = True,
    monte_carlo_scenarios: int = 100,
    min_state_prob: float = 1e-4,
    max_end_states_per_region: int = 5,
    enable_state_capping: bool = True,
    state_cap_coverage_threshold: float = 0.9,
    *,
    policy_option_selection: Any = "primary",
    global_state_override: Optional[GlobalState] = None,
    region_query_cache: Optional[Any] = None,
    global_evaluator_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fully patched ranker:

      - No STAR_ONLY prefilter.
      - Adds library viability probe (cached).
      - Treats all normal query failures as "not viable" (skip partition), not rank_exception.
      - Only re-raises truly unexpected exceptions.

    This should eliminate mass 'rank_exception' collapses.
    """
    import random

    evaluator_profiler = _GlobalEvaluatorProfiler(
        global_evaluator_profile,
        region_query_cache,
    )

    def _finish_result(result: Dict[str, Any]) -> Dict[str, Any]:
        diagnostics_started = evaluator_profiler.start()
        finalized = dict(result)
        evaluator_profiler.stop("diagnostics", diagnostics_started)
        evaluator_profiler.finish()
        return finalized

    rv_internal = _normalize_ranking_variable(ranking_variable)

    state_started = evaluator_profiler.start()
    if global_state_override is None:
        global_state = agop.build_global_state_for_board(players)
        evaluator_profiler.increment("board_state_builds")
    else:
        global_state = global_state_override
        evaluator_profiler.increment("pure_state_override_calls")
    evaluator_profiler.stop("state_copy_or_snapshot", state_started)

    graph_started = evaluator_profiler.start()
    try:
        nodes_iter = battle_graph.nodes()
    except TypeError:
        nodes_iter = battle_graph.nodes
    battle_nodes = list(nodes_iter)

    try:
        edges_iter = battle_graph.edges()
    except TypeError:
        edges_iter = battle_graph.edges
    global_edges = list(edges_iter)
    evaluator_profiler.increment("battle_graph_reads")
    evaluator_profiler.stop("graph_reconstruction", graph_started)

    # If no edges, no battle frontier
    try:
        if hasattr(battle_graph, "number_of_edges") and battle_graph.number_of_edges() == 0:
            return _finish_result({
                "best_evaluation": None,
                "best_partition": None,
                "partitions_full": [],
                "working_partitions": [],
                "evaluated_partitions": [],
                "lookahead_coverage": {},
            })
    except Exception:
        pass

    partition_started = evaluator_profiler.start()
    partition_query_seconds = evaluator_profiler.query_seconds()
    partitions_full = agop.partition_continent_battle_graph_into_valid_small_graphs(
        players=players,
        continent_battle_graph=battle_graph,
        max_partitions=max_partitions,
        combat_libraries_base=combat_libraries_base,
        global_state_override=global_state,
        query_cache=region_query_cache,
    )

    if partitions_full:
        working_partitions: List[List[Dict[str, Any]]] = partitions_full
    else:
        working_partitions = _find_best_partial_partitions(
            players=players,
            battle_graph=battle_graph,
            max_partitions=max_partitions,
        )
    evaluator_profiler.increment("partition_enumerations")
    evaluator_profiler.stop_excluding_queries(
        "partition_enumeration",
        partition_started,
        partition_query_seconds,
    )

    evaluated_partitions: List[PartitionEvaluation] = []
    lookahead_coverage_by_partition: Dict[int, Dict[str, float]] = {}

    if not working_partitions:
        return _finish_result({
            "best_evaluation": None,
            "best_partition": None,
            "partitions_full": partitions_full,
            "working_partitions": working_partitions,
            "evaluated_partitions": evaluated_partitions,
            "lookahead_coverage": lookahead_coverage_by_partition,
        })

    rng = random.Random()

    current_territories = sum(
        1
        for idx in battle_nodes
        if global_state.nodes[idx].owner == "A" and global_state.nodes[idx].troops > 0
    )
    current_troops = sum(
        global_state.nodes[idx].troops
        for idx in battle_nodes
        if global_state.nodes[idx].owner == "A" and global_state.nodes[idx].troops > 0
    )

    # -----------------------------
    # Library viability probe (cached)
    # -----------------------------
    region_viable_cache: Dict[Tuple[Any, ...], bool] = {}

    def _region_nodes_key(region: Dict[str, Any]) -> Tuple[Any, ...]:
        rn = region.get("region_nodes", ())
        return tuple(rn) if isinstance(rn, (list, tuple)) else (rn,)

    def _is_region_viable(region: Dict[str, Any]) -> bool:
        key = _region_nodes_key(region)
        if key in region_viable_cache:
            return region_viable_cache[key]

        try:
            query_region_from_libraries(
                combat_libraries_base=combat_libraries_base,
                global_state=global_state,
                global_edges=global_edges,
                region_nodes=key,
                debug=False,
                query_cache=region_query_cache,
            )
        except Exception as e:
            # Treat both coverage failures and general viability failures as "not viable"
            if _is_coverage_failure(e) or _is_query_viability_failure(e):
                region_viable_cache[key] = False
                return False
            raise  # unexpected bug
        else:
            region_viable_cache[key] = True
            evaluator_profiler.increment("supported_regions")
            return True

    def _partition_is_viable(partition_regions: Sequence[Dict[str, Any]]) -> bool:
        return all(_is_region_viable(reg) for reg in partition_regions)

    best_eval: Optional[PartitionEvaluation] = None
    best_partition: Optional[List[Dict[str, Any]]] = None

    depth = 1 if lookahead_depth and lookahead_depth > 0 else 0

    for part_idx, partition_regions in enumerate(working_partitions):
        evaluation_started = evaluator_profiler.start()
        evaluation_query_seconds = evaluator_profiler.query_seconds()
        if not _partition_is_viable(partition_regions):
            evaluator_profiler.stop_excluding_queries(
                "partition_evaluation",
                evaluation_started,
                evaluation_query_seconds,
            )
            continue

        base_eval = _evaluate_partition_metrics(
            partition_regions=partition_regions,
            global_state=global_state,
            global_edges=global_edges,
            battle_nodes=battle_nodes,
            combat_libraries_base=combat_libraries_base,
            current_territories=current_territories,
            current_troops=current_troops,
            policy_option_selection=policy_option_selection,
            ranking_variable=ranking_variable,
            region_query_cache=region_query_cache,
        )
        if base_eval is None:
            evaluator_profiler.stop_excluding_queries(
                "partition_evaluation",
                evaluation_started,
                evaluation_query_seconds,
            )
            continue

        eval_used = base_eval

        if depth == 1 and use_monte_carlo and monte_carlo_scenarios > 0:
            combined_eval, coverage_info = _monte_carlo_lookahead_for_partition(
                players=players,
                battle_graph=battle_graph,
                combat_libraries_base=combat_libraries_base,
                base_global_state=global_state,
                base_eval=base_eval,
                partition_regions=partition_regions,
                ranking_variable=ranking_variable,
                n_scenarios=monte_carlo_scenarios,
                min_state_prob=min_state_prob,
                max_end_states_per_region=max_end_states_per_region,
                rng=rng,
                enable_state_capping=enable_state_capping,
                state_cap_coverage_threshold=state_cap_coverage_threshold,
                policy_option_selection=policy_option_selection,
            )
            eval_used = combined_eval
            lookahead_coverage_by_partition[part_idx] = coverage_info

        evaluator_profiler.increment("partitions_evaluated")
        evaluator_profiler.stop_excluding_queries(
            "partition_evaluation",
            evaluation_started,
            evaluation_query_seconds,
        )

        ranking_started = evaluator_profiler.start()
        evaluated_partitions.append(eval_used)

        def score(ev: PartitionEvaluation) -> Tuple[float, float, float]:
            if rv_internal == "expected_territories":
                return (ev.expected_territories, ev.conquest_probability, ev.expected_troops)
            elif rv_internal == "expected_troops":
                return (ev.expected_troops, ev.expected_territories, ev.conquest_probability)
            elif rv_internal == "conquest_probability":
                return (ev.conquest_probability, ev.expected_territories, ev.expected_troops)
            return (ev.expected_territories, ev.conquest_probability, ev.expected_troops)

        if best_eval is None or score(eval_used) > score(best_eval):
            best_eval = eval_used
            best_partition = list(partition_regions)
        evaluator_profiler.stop("ranking_selection", ranking_started)

    return _finish_result({
        "best_evaluation": best_eval,
        "best_partition": best_partition,
        "partitions_full": partitions_full,
        "working_partitions": working_partitions,
        "evaluated_partitions": evaluated_partitions,
        "lookahead_coverage": lookahead_coverage_by_partition,
    })





def evaluate_partition_two_wave_lookahead(
    players: Sequence["Players.Player"],
    continent_name: str,
    partition_regions: Sequence[Dict[str, Any]],
    global_state: GlobalState,
    battle_graph,
    combat_libraries_base: Path,
    ranking_variable: str = "expected_territories",
    max_partitions_wave2: int = 40,
    num_scenarios: int = 50,
    min_state_prob: float = 0.0,
    max_end_states_per_region: Optional[int] = None,
    *,
    rng: Optional["random.Random"] = None,
    policy_option_selection: Any = "primary",
) -> TwoWaveEvaluation:
    import random

    if rng is None:
        rng = random.Random()

    try:
        nodes_iter = battle_graph.nodes()
    except TypeError:
        nodes_iter = battle_graph.nodes
    battle_nodes_initial = list(nodes_iter)
    battle_nodes_initial_set = set(battle_nodes_initial)

    try:
        edges_iter = battle_graph.edges()
    except TypeError:
        edges_iter = battle_graph.edges
    global_edges = list(edges_iter)

    current_territories = 0
    current_troops = 0
    for idx in battle_nodes_initial:
        node = global_state.nodes[idx]
        if node.owner == "A" and node.troops > 0:
            current_territories += 1
            current_troops += node.troops

    regions_info, coverage_wave1 = _collect_wave1_region_distributions(
        partition_regions=partition_regions,
        global_state=global_state,
        global_edges=global_edges,
        combat_libraries_base=combat_libraries_base,
        min_state_prob=min_state_prob,
        max_end_states_per_region=max_end_states_per_region,
        policy_option_selection=policy_option_selection,
        ranking_variable=ranking_variable,
        debug=False
    )

    if all(not (r.get("prob_row") or {}) and not (r.get("dist_v2") or {}) for r in regions_info):
        return TwoWaveEvaluation(
            base_partition=list(partition_regions),
            base_evaluation=None,
            scenarios=[],
            coverage_wave1=float(coverage_wave1),
            expected_territories=float(current_territories),
            expected_new_territories=0.0,
            expected_troops=float(current_troops),
            expected_troop_loss=0.0,
            conquest_probability=0.0,
        )

    scenarios: List[TwoWaveScenarioResult] = []

    board_snapshot = _snapshot_board_state()

    try:
        for _ in range(num_scenarios):
            nodes_after_wave1 = list(global_state.nodes)

            for r_info in regions_info:
                mapping: Dict[int, int] = (r_info.get("mapping") or {})

                # ----------------------------
                # V2 array sampling path
                # ----------------------------
                if int(r_info.get("format_version", 1) or 1) == 2 and isinstance(r_info.get("dist_v2"), dict):
                    dist = r_info["dist_v2"]
                    if not mapping:
                        continue

                    p = np.asarray(dist.get("p"))
                    owners = np.asarray(dist.get("owners"))
                    troops = np.asarray(dist.get("troops"))

                    if p.size == 0:
                        continue

                    # Use stored cdf if present, else build
                    cdf = dist.get("cdf")
                    if cdf is None:
                        cdf = np.cumsum(p, dtype=np.float64)
                    else:
                        cdf = np.asarray(cdf)

                    if cdf.size == 0:
                        continue
                    total = float(cdf[-1])
                    if total <= 0.0:
                        continue

                    x = rng.random() * total
                    k = int(np.searchsorted(cdf, x, side="right"))
                    if k >= int(p.size):
                        k = int(p.size) - 1

                    owners_row = owners[k, :]
                    troops_row = troops[k, :]
                    for local_idx in range(int(len(owners_row))):
                        global_idx = mapping.get(local_idx)
                        if global_idx is None:
                            continue
                        nodes_after_wave1[global_idx] = _node_state_from_owner_code(
                            int(owners_row[local_idx]), int(troops_row[local_idx])
                        )
                    continue

                # ----------------------------
                # Legacy label sampling path
                # ----------------------------
                prob_row: Dict[str, float] = (r_info.get("prob_row") or {})
                if not prob_row or not mapping:
                    continue

                labels = sorted(prob_row.keys())
                probs = [float(prob_row[lbl]) for lbl in labels]

                s = float(sum(probs))
                if s <= 0.0:
                    continue
                if abs(s - 1.0) > 1e-6:
                    inv = 1.0 / s
                    probs = [p * inv for p in probs]

                x = rng.random()
                cum = 0.0
                chosen_label = labels[-1]
                for lbl, pval in zip(labels, probs):
                    cum += pval
                    if x <= cum:
                        chosen_label = lbl
                        break

                local_end_state = agop.global_state_from_row_label(chosen_label)

                for local_idx, node_after in enumerate(local_end_state.nodes):
                    global_idx = mapping[local_idx]
                    nodes_after_wave1[global_idx] = node_after

            global_state_after_wave1 = GlobalState(nodes=tuple(nodes_after_wave1))

            _apply_global_state_to_board(global_state_after_wave1, players)
            battle_graph_k = agop.build_continent_battle_graph(continent_name, players)

            try:
                nodes_iter_k = battle_graph_k.nodes()
            except TypeError:
                nodes_iter_k = battle_graph_k.nodes
            battle_nodes_k = list(nodes_iter_k)
            battle_nodes_k_set = set(battle_nodes_k)

            frozen_nodes = list(battle_nodes_initial_set - battle_nodes_k_set)

            terr_frozen = 0
            troops_frozen = 0
            all_frozen_are_A = True
            for idx in frozen_nodes:
                node = global_state_after_wave1.nodes[idx]
                if node.owner == "A" and node.troops > 0:
                    terr_frozen += 1
                    troops_frozen += node.troops
                else:
                    all_frozen_are_A = False

            rank_result_k = rank_battle_graph_partitions(
                players=players,
                battle_graph=battle_graph_k,
                combat_libraries_base=combat_libraries_base,
                max_partitions=max_partitions_wave2,
                ranking_variable=ranking_variable,
            )

            best_eval_k = rank_result_k.get("best_evaluation")

            if best_eval_k is None:
                expected_territories_k = float(terr_frozen)
                expected_troops_k = float(troops_frozen)
                expected_new_territories_k = expected_territories_k - float(current_territories)
                expected_troop_loss_k = float(current_troops) - expected_troops_k
                conquest_probability_k = 0.0
            else:
                expected_territories_k = float(terr_frozen) + float(best_eval_k.expected_territories)
                expected_troops_k = float(troops_frozen) + float(best_eval_k.expected_troops)

                expected_new_territories_k = expected_territories_k - float(current_territories)
                expected_troop_loss_k = float(current_troops) - expected_troops_k

                conquest_probability_k = float(best_eval_k.conquest_probability) if all_frozen_are_A else 0.0

            scenarios.append(
                TwoWaveScenarioResult(
                    expected_territories=float(expected_territories_k),
                    expected_new_territories=float(expected_new_territories_k),
                    expected_troop_loss=float(expected_troop_loss_k),
                    expected_troops=float(expected_troops_k),
                    conquest_probability=float(conquest_probability_k),
                )
            )

    finally:
        _restore_board_state(board_snapshot)

    if scenarios:
        n = float(len(scenarios))
        exp_territories = sum(s.expected_territories for s in scenarios) / n
        exp_new_territories = sum(s.expected_new_territories for s in scenarios) / n
        exp_troops = sum(s.expected_troops for s in scenarios) / n
        exp_troop_loss = sum(s.expected_troop_loss for s in scenarios) / n
        exp_conquest_p = sum(s.conquest_probability for s in scenarios) / n
    else:
        exp_territories = float(current_territories)
        exp_new_territories = 0.0
        exp_troops = float(current_troops)
        exp_troop_loss = 0.0
        exp_conquest_p = 0.0

    return TwoWaveEvaluation(
        base_partition=list(partition_regions),
        base_evaluation=None,
        scenarios=scenarios,
        coverage_wave1=float(coverage_wave1),
        expected_territories=float(exp_territories),
        expected_new_territories=float(exp_new_territories),
        expected_troops=float(exp_troops),
        expected_troop_loss=float(exp_troop_loss),
        conquest_probability=float(exp_conquest_p),
    )


def rank_battle_graph_partitions_with_lookahead(
    players: Sequence["Players.Player"],
    continent_name: str,
    combat_libraries_base: Path,
    max_partitions_wave1: int = 40,
    ranking_variable: str = "battle_expected_attacker_territory_count",
    max_partitions_wave2: int = 40,
    num_scenarios: int = 50,
    min_state_prob: float = 0.0,
    max_end_states_per_region: Optional[int] = None,
    enable_state_capping: bool = True,
    state_cap_coverage_threshold: float = 0.9,
    *,
    policy_option_selection: Any = "primary",
) -> Dict[str, Any]:
    """
    Convenience wrapper:

      1) Build initial battle_graph for continent_name.
      2) Run rank_battle_graph_partitions on it (wave1).
      3) Take best partition and run two-wave Monte Carlo lookahead
         using evaluate_partition_two_wave_lookahead.

    Parameters
    ----------
    ranking_variable : {"expected_territories",
                        "expected_troops",
                        "battle_expected_attacker_conquest_probability"}
        Passed through to both the base ranking and the wave-2 ranking.
    """
    # Build initial battle graph & global state
    battle_graph = agop.build_continent_battle_graph(continent_name, players)
    global_state = agop.build_global_state_for_board(players)

    base_result = rank_battle_graph_partitions(
        players=players,
        battle_graph=battle_graph,
        combat_libraries_base=combat_libraries_base,
        max_partitions=max_partitions_wave1,
        ranking_variable=ranking_variable,
        policy_option_selection=policy_option_selection,
    )

    best_eval = base_result.get("best_evaluation")
    if best_eval is None:
        return {
            "base_result": base_result,
            "lookahead_result": None,
        }

    base_partition = best_eval.partition

    lookahead = evaluate_partition_two_wave_lookahead(
        players=players,
        continent_name=continent_name,
        partition_regions=base_partition,
        global_state=global_state,
        battle_graph=battle_graph,
        combat_libraries_base=combat_libraries_base,
        ranking_variable=ranking_variable,
        max_partitions_wave2=max_partitions_wave2,
        num_scenarios=num_scenarios,
        min_state_prob=min_state_prob,
        max_end_states_per_region=max_end_states_per_region,
        policy_option_selection=policy_option_selection,
    )



    # Attach the base_evaluation into the lookahead result for convenience
    lookahead.base_evaluation = best_eval

    return {
        "base_result": base_result,
        "lookahead_result": lookahead,
    }
