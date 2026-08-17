from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import math
import os
import pickle
import time
import traceback as traceback_module
import uuid

import networkx as nx
import numpy as np
import pandas as pd

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.continent_model import battle_graph_ranking as bgr
from project_risk.mathematical.transition_prediction_ml import generate_data_ML as gdm
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState, NodeState
from project_risk.mathematical.transition_prediction_ml.transition_distribution_stage_a_v2 import (
    _global_state_from_raw_signature,
    _graph_from_signature,
    inspect_transition_library_metadata,
    load_stage_a_v2_grouped_examples,
)


TARGET_GENERATION_VERSION = (
    "transition_targets_v3_separate_selection_and_target_sampling"
)
STAGE_A_SCHEMA_VERSION = "stage_a_v3_schema_1"
CALIBRATION_FORMAT_VERSION = "transition_target_sampling_calibration_v1"
DEFAULT_CALIBRATION_OUTPUT_DIR = Path(
    "transition_target_sampling_calibration_v1_20260717"
)
DEFAULT_SOURCE_STAGE_A_V2_DIR = Path(
    "transition_distribution_data_v2_corrected_mc5_pilot_20260717"
)
DEFAULT_CALIBRATION_STATE_TARGETS = {
    "Australia": 30,
    "South America": 30,
    "North America": 50,
    "Europe": 30,
    "Africa": 30,
    "Asia": 50,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    return repr(value)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_seed(payload: Any) -> int:
    return int(_stable_digest(payload)[:16], 16) % (2 ** 32)


def _strictly_increasing_positive(values: Sequence[int], field_name: str) -> None:
    if not values or any(int(value) < 1 for value in values):
        raise ValueError(f"{field_name} must contain positive sample counts")
    if any(int(right) <= int(left) for left, right in zip(values, values[1:])):
        raise ValueError(f"{field_name} must be strictly increasing")


@dataclass(frozen=True)
class TransitionSamplingV3Config:
    """Two-phase Stage A configuration with explicit, non-ambiguous counts."""

    candidate_selection_mode: str = "fixed"
    candidate_selection_mc_samples: Optional[int] = 20
    candidate_selection_checkpoints: Tuple[int, ...] = (5, 10, 20, 40)
    candidate_selection_max_samples: int = 80
    candidate_stability_required_consecutive: int = 2
    candidate_score_gap_abs_threshold: Optional[float] = None
    candidate_score_gap_rel_threshold: Optional[float] = None
    candidate_selection_base_seed: int = 42

    target_distribution_mc_samples: Optional[int] = 50
    target_distribution_checkpoints: Tuple[int, ...] = (20, 50, 100)
    target_distribution_base_seed: int = 42050
    nested_scenarios: bool = True

    # Compatibility fallback only. Saved v2 configurations continue to use
    # TransitionDistributionConfig and are never silently reinterpreted.
    two_stage_mc_samples: Optional[int] = None

    max_partitions: int = 40
    ranking_variable: str = "battle_expected_attacker_territory_count"
    combat_libraries_base: Path | str = "small_graph_libraries"
    first_stage_value_tolerances: Optional[Tuple[float, ...]] = None
    max_policy_combos_per_partition: Optional[int] = None
    partition_candidate_selection_mode: str = "maximal_per_partition_utility"
    utility_abs_tolerance: Optional[float] = None
    utility_rel_tolerance: Optional[float] = None
    max_candidates_per_partition: Optional[int] = None
    profile_second_stage: bool = True

    target_generation_version: str = TARGET_GENERATION_VERSION
    stage_a_schema_version: str = STAGE_A_SCHEMA_VERSION
    resume: bool = True
    checkpoint_every_state: bool = True

    def __post_init__(self) -> None:
        if self.candidate_selection_mode not in {"fixed", "adaptive_checkpoints"}:
            raise ValueError(
                f"Unknown candidate_selection_mode={self.candidate_selection_mode!r}"
            )
        _strictly_increasing_positive(
            self.candidate_selection_checkpoints,
            "candidate_selection_checkpoints",
        )
        _strictly_increasing_positive(
            self.target_distribution_checkpoints,
            "target_distribution_checkpoints",
        )
        if int(self.candidate_selection_max_samples) < int(
            self.candidate_selection_checkpoints[-1]
        ):
            raise ValueError(
                "candidate_selection_max_samples must be >= the final checkpoint"
            )
        if self.resolved_candidate_selection_mc_samples < 1:
            raise ValueError("candidate_selection_mc_samples must be >= 1")
        if self.resolved_target_distribution_mc_samples < 1:
            raise ValueError("target_distribution_mc_samples must be >= 1")
        if int(self.candidate_stability_required_consecutive) < 1:
            raise ValueError("candidate_stability_required_consecutive must be >= 1")
        if not bool(self.nested_scenarios):
            raise ValueError("Stage A v3 currently requires nested_scenarios=True")
        if self.partition_candidate_selection_mode not in {
            "legacy_global_utility",
            "maximal_per_partition_utility",
        }:
            raise ValueError(
                "Unknown partition_candidate_selection_mode="
                f"{self.partition_candidate_selection_mode!r}"
            )

    @property
    def resolved_candidate_selection_mc_samples(self) -> int:
        value = self.candidate_selection_mc_samples
        if value is None:
            value = self.two_stage_mc_samples
        if value is None:
            raise ValueError(
                "candidate_selection_mc_samples is omitted and no legacy "
                "two_stage_mc_samples fallback was supplied"
            )
        return int(value)

    @property
    def resolved_target_distribution_mc_samples(self) -> int:
        value = self.target_distribution_mc_samples
        if value is None:
            value = self.two_stage_mc_samples
        if value is None:
            raise ValueError(
                "target_distribution_mc_samples is omitted and no legacy "
                "two_stage_mc_samples fallback was supplied"
            )
        return int(value)


def stage_a_v3_config_fingerprint(config: TransitionSamplingV3Config) -> str:
    payload = asdict(config)
    payload.pop("resume", None)
    payload.pop("checkpoint_every_state", None)
    payload["resolved_candidate_selection_mc_samples"] = (
        config.resolved_candidate_selection_mc_samples
    )
    payload["resolved_target_distribution_mc_samples"] = (
        config.resolved_target_distribution_mc_samples
    )
    payload["two_phase_sampling_semantics"] = True
    return _stable_digest(payload)


def stage_a_v3_target_fingerprint(config: TransitionSamplingV3Config) -> str:
    payload = asdict(config)
    for key in ("resume", "checkpoint_every_state", "profile_second_stage"):
        payload.pop(key, None)
    payload["resolved_candidate_selection_mc_samples"] = (
        config.resolved_candidate_selection_mc_samples
    )
    payload["resolved_target_distribution_mc_samples"] = (
        config.resolved_target_distribution_mc_samples
    )
    payload["two_phase_sampling_semantics"] = True
    return _stable_digest(payload)


def input_state_fingerprint(
    *,
    initial_global_state_signature: Any,
    battle_graph_signature: Any,
    full_graph_signature: Any,
) -> str:
    return _stable_digest(
        {
            "initial_global_state_signature": initial_global_state_signature,
            "battle_graph_signature": battle_graph_signature,
            "full_graph_signature": full_graph_signature,
        }
    )


def canonical_stage_a_v3_example_id(
    *,
    input_state_id: str,
    input_fingerprint: str,
    semantic_target_fingerprint: str,
) -> str:
    digest = _stable_digest(
        {
            "input_state_id": str(input_state_id),
            "input_state_fingerprint": str(input_fingerprint),
            "semantic_target_fingerprint": str(semantic_target_fingerprint),
            "target_generation_version": TARGET_GENERATION_VERSION,
        }
    )
    return f"stage_a_v3_{digest[:24]}"


def derive_stage_a_v3_phase_seed(
    *, base_seed: int, example_id: str, phase: str
) -> int:
    return _stable_seed(
        {
            "kind": "stage_a_v3_phase_seed_v1",
            "base_seed": int(base_seed),
            "example_id": str(example_id),
            "phase": str(phase),
        }
    )


@dataclass(frozen=True)
class TargetDistributionCheckpointResult:
    target_distribution_mc_samples: int
    successor_state_counts: Mapping[Any, int]
    successor_state_probabilities: Mapping[Any, float]
    state_sequence: Tuple[Any, ...]
    node_marginals: Mapping[int, Mapping[str, float]]
    strategic_summaries: Mapping[str, float]
    runtime_increment_seconds: float
    runtime_cumulative_seconds: float
    unique_successor_state_count: int
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class SelectedCandidateTargetResumeState:
    schema_version: str
    base_seed: int
    state_identity: Any
    selected_candidate_identity: Any
    completed_scenarios: int
    successor_state_counts: Mapping[Any, int]
    state_sequence: Tuple[Any, ...]
    sampled_regional_outcomes: Mapping[Any, int]
    checkpoints: Tuple[TargetDistributionCheckpointResult, ...]
    runtime_cumulative_seconds: float


@dataclass(frozen=True)
class SelectedCandidateDistributionResult:
    selected_candidate_identity: Any
    checkpoints: Tuple[TargetDistributionCheckpointResult, ...]
    target_distribution_mc_samples: int
    successor_state_counts: Mapping[Any, int]
    successor_state_probabilities: Mapping[Any, float]
    state_sequence: Tuple[Any, ...]
    node_marginals: Mapping[int, Mapping[str, float]]
    strategic_summaries: Mapping[str, float]
    runtime_seconds: float
    unique_successor_state_count: int
    resume_state: SelectedCandidateTargetResumeState
    diagnostics: Mapping[str, Any]


def _probabilities_from_counts(counts: Mapping[Any, int]) -> Dict[Any, float]:
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        return {}
    return {
        signature: float(count) / float(total)
        for signature, count in counts.items()
        if int(count) > 0
    }


def _strategic_summaries_from_distribution(
    probabilities: Mapping[Any, float],
    *,
    full_graph_nodes: Sequence[int],
) -> Dict[str, float]:
    expected_territories = 0.0
    expected_troops = 0.0
    conquest_probability = 0.0
    full_nodes = {int(node) for node in full_graph_nodes}
    for signature, probability in probabilities.items():
        state = {
            int(node): (str(owner), int(troops))
            for node, owner, troops in signature
        }
        attacker_nodes = {
            node
            for node in full_nodes
            if state.get(node, ("D", 0))[0] == "A"
        }
        expected_territories += float(probability) * len(attacker_nodes)
        expected_troops += float(probability) * sum(
            state.get(node, ("D", 0))[1] for node in attacker_nodes
        )
        conquest_probability += float(probability) * float(
            bool(full_nodes) and attacker_nodes == full_nodes
        )
    return {
        "expected_attacker_territories": float(expected_territories),
        "expected_attacker_troops": float(expected_troops),
        "conquest_probability": float(conquest_probability),
    }


def _coerce_target_resume_state(
    value: Optional[Any],
) -> Optional[SelectedCandidateTargetResumeState]:
    if value is None or isinstance(value, SelectedCandidateTargetResumeState):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        payload["checkpoints"] = tuple(
            item
            if isinstance(item, TargetDistributionCheckpointResult)
            else TargetDistributionCheckpointResult(**dict(item))
            for item in payload.get("checkpoints", ())
        )
        return SelectedCandidateTargetResumeState(**payload)
    raise TypeError("resume_state must be a mapping or SelectedCandidateTargetResumeState")


def sample_selected_candidate_successor_distribution(
    *,
    selected_candidate: bgr.PartitionPolicyCandidate,
    initial_global_state: GlobalState,
    full_graph,
    sample_count: int,
    base_seed: int,
    example_id: str,
    checkpoints: Optional[Sequence[int]] = None,
    battle_nodes: Optional[Sequence[int]] = None,
    scenario_start_index: int = 0,
    existing_counts: Optional[Mapping[Any, int]] = None,
    resume_state: Optional[Any] = None,
    checkpoint_callback: Optional[
        Callable[
            [TargetDistributionCheckpointResult, SelectedCandidateTargetResumeState],
            None,
        ]
    ] = None,
    regional_sample_plan: Optional[Mapping[Tuple[Any, int], Any]] = None,
) -> SelectedCandidateDistributionResult:
    """Sample only an already selected policy; no global ranking occurs here."""
    final_count = int(sample_count)
    if final_count < 1:
        raise ValueError("sample_count must be >= 1")
    raw_points = tuple(int(value) for value in (checkpoints or (final_count,)))
    _strictly_increasing_positive(raw_points, "target_distribution_checkpoints")
    points = tuple(value for value in raw_points if value <= final_count)
    if final_count not in points:
        points = points + (final_count,)

    candidate_identity = bgr.canonical_partition_policy_candidate_identity(
        selected_candidate
    )
    full_nodes = tuple(sorted(int(node) for node in full_graph.nodes()))
    resolved_battle_nodes = tuple(
        sorted(
            {int(node) for node in (battle_nodes or ())}
            or {
                int(node)
                for ref in selected_candidate.region_policy_options
                for node in ref.region_nodes
            }
        )
    )
    state_identity = (
        "selected_candidate_target_sampling_v1",
        str(example_id),
        bgr.canonical_two_stage_global_state_signature(initial_global_state),
        candidate_identity,
    )
    resumed = _coerce_target_resume_state(resume_state)
    if resumed is not None:
        if resumed.schema_version != "selected_candidate_target_resume_v1":
            raise ValueError("Unsupported selected-candidate target resume schema")
        if int(resumed.base_seed) != int(base_seed):
            raise ValueError("Target-sampling resume base seed mismatch")
        if resumed.state_identity != state_identity:
            raise ValueError("Target-sampling resume state identity mismatch")
        if resumed.selected_candidate_identity != candidate_identity:
            raise ValueError("Target-sampling resume candidate identity mismatch")
        completed = int(resumed.completed_scenarios)
        counts = dict(resumed.successor_state_counts)
        state_sequence = list(resumed.state_sequence)
        sampled_outcomes = dict(resumed.sampled_regional_outcomes)
        checkpoint_results = list(resumed.checkpoints)
        prior_runtime = float(resumed.runtime_cumulative_seconds)
    else:
        completed = int(scenario_start_index)
        counts = {
            signature: int(count)
            for signature, count in (existing_counts or {}).items()
            if int(count) > 0
        }
        if completed and sum(counts.values()) != completed:
            raise ValueError(
                "existing_counts must sum to scenario_start_index when resuming "
                "without a resume_state"
            )
        state_sequence: List[Any] = []
        sampled_outcomes: Dict[Any, int] = {}
        checkpoint_results: List[TargetDistributionCheckpointResult] = []
        prior_runtime = 0.0
    if completed > final_count:
        raise ValueError("Target resume state is beyond sample_count")

    option_keys = tuple(
        bgr.canonical_region_policy_option_key(ref)
        for ref in selected_candidate.region_policy_options
    )
    prepared_options, option_diagnostics = bgr.prepare_unique_regional_policy_options(
        (selected_candidate,)
    )
    assembly_plans, assembly_diagnostics = bgr.prepare_partition_assembly_plans(
        (selected_candidate,),
        base_global_state=initial_global_state,
        battle_nodes=resolved_battle_nodes,
    )
    partition_signature = bgr._partition_signature_for_candidate(selected_candidate)
    call_started = time.perf_counter()

    def make_resume(runtime_cumulative: float) -> SelectedCandidateTargetResumeState:
        return SelectedCandidateTargetResumeState(
            schema_version="selected_candidate_target_resume_v1",
            base_seed=int(base_seed),
            state_identity=state_identity,
            selected_candidate_identity=candidate_identity,
            completed_scenarios=int(completed),
            successor_state_counts=dict(counts),
            state_sequence=tuple(state_sequence),
            sampled_regional_outcomes=dict(sampled_outcomes),
            checkpoints=tuple(checkpoint_results),
            runtime_cumulative_seconds=float(runtime_cumulative),
        )

    for checkpoint_count in points:
        if checkpoint_count <= completed:
            continue
        previous_runtime = (
            checkpoint_results[-1].runtime_cumulative_seconds
            if checkpoint_results
            else 0.0
        )
        for scenario_index in range(completed, checkpoint_count):
            outcome_indices: List[int] = []
            for option_key in option_keys:
                sample_key = (option_key, int(scenario_index))
                if sample_key not in sampled_outcomes:
                    sampled_outcomes[sample_key] = bgr.sample_prepared_regional_option(
                        prepared_options[option_key],
                        base_seed=int(base_seed),
                        state_identity=state_identity,
                        scenario_index=int(scenario_index),
                        regional_sample_plan=regional_sample_plan,
                    )
                outcome_indices.append(int(sampled_outcomes[sample_key]))
            successor = bgr._assemble_prepared_candidate_state(
                base_global_state=initial_global_state,
                candidate=selected_candidate,
                option_keys=option_keys,
                outcome_indices=outcome_indices,
                prepared_options=prepared_options,
                assembly_plan=assembly_plans.get(partition_signature),
            )
            signature = bgr.canonical_two_stage_global_state_signature(
                successor, node_indices=full_nodes
            )
            state_sequence.append(signature)
            counts[signature] = counts.get(signature, 0) + 1
        completed = int(checkpoint_count)
        probabilities = _probabilities_from_counts(counts)
        marginals = gdm.derive_node_marginals_from_successor_distribution(
            successor_state_counts=counts,
            full_graph=full_graph,
            initial_global_state=initial_global_state,
        )
        strategic = _strategic_summaries_from_distribution(
            probabilities, full_graph_nodes=full_nodes
        )
        runtime_cumulative = float(
            prior_runtime + (time.perf_counter() - call_started)
        )
        checkpoint_result = TargetDistributionCheckpointResult(
            target_distribution_mc_samples=int(completed),
            successor_state_counts=dict(counts),
            successor_state_probabilities=probabilities,
            state_sequence=tuple(state_sequence),
            node_marginals=marginals,
            strategic_summaries=strategic,
            runtime_increment_seconds=float(runtime_cumulative - previous_runtime),
            runtime_cumulative_seconds=runtime_cumulative,
            unique_successor_state_count=int(len(counts)),
            diagnostics={
                "selected_candidate_only": True,
                "global_candidate_evaluation_calls": 0,
                "global_partition_ranking_calls": 0,
                "regional_sample_cache_entries": int(len(sampled_outcomes)),
            },
        )
        checkpoint_results.append(checkpoint_result)
        current_resume = make_resume(runtime_cumulative)
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint_result, current_resume)

    if not checkpoint_results:
        raise RuntimeError("Selected-candidate target sampling produced no checkpoints")
    final = checkpoint_results[-1]
    runtime_seconds = float(prior_runtime + (time.perf_counter() - call_started))
    final_resume = make_resume(runtime_seconds)
    return SelectedCandidateDistributionResult(
        selected_candidate_identity=candidate_identity,
        checkpoints=tuple(checkpoint_results),
        target_distribution_mc_samples=int(final.target_distribution_mc_samples),
        successor_state_counts=dict(final.successor_state_counts),
        successor_state_probabilities=dict(final.successor_state_probabilities),
        state_sequence=tuple(final.state_sequence),
        node_marginals=dict(final.node_marginals),
        strategic_summaries=dict(final.strategic_summaries),
        runtime_seconds=runtime_seconds,
        unique_successor_state_count=int(final.unique_successor_state_count),
        resume_state=final_resume,
        diagnostics={
            "selected_candidate_only": True,
            "candidate_prepared_once": True,
            "global_candidate_evaluation_calls": 0,
            "global_partition_ranking_calls": 0,
            "prepared_regional_options": option_diagnostics,
            "prepared_partition_assemblies": assembly_diagnostics,
            "target_seed_namespace": "selected_candidate_target_sampling_v1",
            "nested_scenarios": True,
        },
    )


def _top_probability_states(
    probabilities: Mapping[Any, float], k: int
) -> Tuple[Any, ...]:
    return tuple(
        signature
        for signature, _ in sorted(
            probabilities.items(), key=lambda item: (-float(item[1]), item[0])
        )[: int(k)]
    )


def compare_target_distribution_checkpoints(
    lower: TargetDistributionCheckpointResult,
    higher: TargetDistributionCheckpointResult,
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Measure empirical target convergence for one fixed selected candidate."""
    left = dict(lower.successor_state_probabilities)
    right = dict(higher.successor_state_probabilities)
    union = set(left) | set(right)
    intersection = set(left) & set(right)
    total_variation = 0.5 * sum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0)))
        for key in union
    )
    js_divergence = 0.0
    for key in union:
        p = float(left.get(key, 0.0))
        q = float(right.get(key, 0.0))
        midpoint = 0.5 * (p + q)
        if p > 0.0:
            js_divergence += 0.5 * p * math.log(p / midpoint)
        if q > 0.0:
            js_divergence += 0.5 * q * math.log(q / midpoint)

    left_top = _top_probability_states(left, top_k)
    right_top = _top_probability_states(right, top_k)
    marginal_nodes = sorted(
        {int(node) for node in lower.node_marginals}
        | {int(node) for node in higher.node_marginals}
    )
    ownership_differences = []
    troop_differences = []
    for node in marginal_nodes:
        left_marginal = lower.node_marginals.get(
            node, lower.node_marginals.get(str(node), {})  # type: ignore[arg-type]
        )
        right_marginal = higher.node_marginals.get(
            node, higher.node_marginals.get(str(node), {})  # type: ignore[arg-type]
        )
        ownership_differences.append(
            abs(
                float(left_marginal.get("p_attacker_final", 0.0))
                - float(right_marginal.get("p_attacker_final", 0.0))
            )
        )
        troop_differences.append(
            abs(
                float(left_marginal.get("expected_troops", 0.0))
                - float(right_marginal.get("expected_troops", 0.0))
            )
        )
    strategic_keys = sorted(
        set(lower.strategic_summaries) | set(higher.strategic_summaries)
    )
    strategic_differences = {
        f"{key}_difference": float(higher.strategic_summaries.get(key, 0.0))
        - float(lower.strategic_summaries.get(key, 0.0))
        for key in strategic_keys
    }
    top_union = set(left_top) | set(right_top)
    top_intersection = set(left_top) & set(right_top)
    return {
        "lower_target_distribution_mc_samples": int(
            lower.target_distribution_mc_samples
        ),
        "higher_target_distribution_mc_samples": int(
            higher.target_distribution_mc_samples
        ),
        "total_variation_distance": float(total_variation),
        "jensen_shannon_divergence": float(js_divergence),
        "support_intersection_size": int(len(intersection)),
        "support_union_size": int(len(union)),
        "support_intersection_over_union": float(
            len(intersection) / len(union) if union else 1.0
        ),
        "probability_mass_overlap": float(
            sum(min(float(left.get(key, 0.0)), float(right.get(key, 0.0))) for key in union)
        ),
        "top_1_agreement": bool(left_top[:1] == right_top[:1]),
        "top_k": int(top_k),
        "top_k_overlap_count": int(len(top_intersection)),
        "top_k_overlap_fraction": float(
            len(top_intersection) / len(top_union) if top_union else 1.0
        ),
        "maximum_node_ownership_difference": float(
            max(ownership_differences, default=0.0)
        ),
        "mean_node_ownership_difference": float(
            np.mean(ownership_differences) if ownership_differences else 0.0
        ),
        "maximum_expected_troop_difference": float(
            max(troop_differences, default=0.0)
        ),
        "mean_expected_troop_difference": float(
            np.mean(troop_differences) if troop_differences else 0.0
        ),
        "strategic_summary_differences": strategic_differences,
        "runtime_increment_seconds": float(higher.runtime_increment_seconds),
        "comparison_kind": "same_selected_candidate_target_sampling",
    }


def decompose_selection_and_target_instability(
    *,
    lower_candidate_identity: Any,
    higher_candidate_identity: Any,
    target_distribution_comparison: Optional[Mapping[str, Any]] = None,
    target_difference_tolerance: float = 1e-12,
) -> Dict[str, Any]:
    candidate_changed = lower_candidate_identity != higher_candidate_identity
    target_changed = bool(
        target_distribution_comparison
        and float(
            target_distribution_comparison.get("total_variation_distance", 0.0)
            or 0.0
        )
        > float(target_difference_tolerance)
    )
    if candidate_changed and target_changed:
        classification = "candidate_selection_and_target_sampling_changed"
    elif candidate_changed:
        classification = "candidate_selection_changed"
    elif target_changed:
        classification = "same_candidate_target_sampling_changed"
    else:
        classification = "stable_candidate_and_target"
    return {
        "candidate_selection_changed": bool(candidate_changed),
        "same_candidate_empirical_target_changed": bool(
            target_changed and not candidate_changed
        ),
        "target_distribution_changed": bool(target_changed),
        "instability_classification": classification,
    }


def _checkpoint_payload(
    checkpoint: bgr.CandidateSelectionCheckpointResult,
) -> Dict[str, Any]:
    return {
        "candidate_selection_mc_samples": int(checkpoint.mc_samples),
        "selected_candidate_index": checkpoint.selected_candidate_index,
        "selected_candidate_identity": checkpoint.selected_candidate_identity,
        "selected_partition_signature": checkpoint.selected_partition_signature,
        "selected_policy_option_indices": checkpoint.selected_policy_option_indices,
        "best_score_mean": checkpoint.best_score_mean,
        "best_score_std": checkpoint.best_score_std,
        "runner_up_candidate_index": checkpoint.runner_up_candidate_index,
        "runner_up_candidate_identity": checkpoint.runner_up_candidate_identity,
        "runner_up_score_mean": checkpoint.runner_up_score_mean,
        "runner_up_score_std": checkpoint.runner_up_score_std,
        "best_runner_up_gap": checkpoint.best_runner_up_gap,
        "candidate_rank_order": checkpoint.candidate_rank_order,
        "top_candidate_indices": checkpoint.top_candidate_indices,
        "runtime_increment_seconds": checkpoint.runtime_increment_seconds,
        "runtime_cumulative_seconds": checkpoint.runtime_cumulative_seconds,
        "unique_global_states_increment": checkpoint.unique_global_states_increment,
        "unique_global_states_cumulative": checkpoint.unique_global_states_cumulative,
        "diagnostics": dict(checkpoint.diagnostics),
    }


def build_stage_a_v3_grouped_row(
    *,
    input_metadata: Mapping[str, Any],
    config: TransitionSamplingV3Config,
    example_id: str,
    input_state_id: str,
    input_fingerprint: str,
    semantic_target_fingerprint: str,
    selection_result: bgr.NestedCandidateSelectionResult,
    target_result: SelectedCandidateDistributionResult,
    selected_candidate: bgr.PartitionPolicyCandidate,
) -> Dict[str, Any]:
    final_selection = (
        selection_result.checkpoints[-1] if selection_result.checkpoints else None
    )
    selected_partition = bgr._partition_signature_for_candidate(selected_candidate)
    selected_policy_indices = tuple(
        int(ref.option_index)
        for ref in sorted(
            selected_candidate.region_policy_options,
            key=lambda ref: tuple(sorted(int(node) for node in ref.region_nodes)),
        )
    )
    row = {
        "example_id": str(example_id),
        "input_state_id": str(input_state_id),
        "source_stage_a_v2_example_id": input_metadata.get("example_id"),
        "input_state_fingerprint": str(input_fingerprint),
        "semantic_target_fingerprint": str(semantic_target_fingerprint),
        "config_fingerprint": stage_a_v3_config_fingerprint(config),
        "target_generation_version": TARGET_GENERATION_VERSION,
        "stage_a_schema_version": STAGE_A_SCHEMA_VERSION,
        "transition_example_status": "ok",
        "continent_name": input_metadata.get("continent_name"),
        "attack_perspective": input_metadata.get("attack_perspective"),
        "state_id": input_metadata.get("state_id"),
        "initial_global_state_signature": input_metadata.get(
            "initial_global_state_signature"
        ),
        "initial_full_graph_signature": input_metadata.get(
            "initial_full_graph_signature"
        ),
        "battle_graph_signature": input_metadata.get("battle_graph_signature"),
        "full_graph_signature": input_metadata.get("full_graph_signature"),
        "battle_graph_nodes": input_metadata.get("battle_graph_nodes"),
        "full_graph_nodes": input_metadata.get("full_graph_nodes"),
        "battle_node_count": input_metadata.get("battle_node_count"),
        "attacker_node_count": input_metadata.get("attacker_node_count"),
        "defender_node_count": input_metadata.get("defender_node_count"),
        "attacker_troop_total": input_metadata.get("attacker_troop_total"),
        "defender_troop_total": input_metadata.get("defender_troop_total"),
        "candidate_count_category": gdm._candidate_count_category(
            len(selection_result.evaluated_candidates)
        ),
        "battle_node_count_category": input_metadata.get("battle_node_count_category"),
        "num_maximal_partitions": input_metadata.get("num_maximal_partitions"),
        "num_retained_second_stage_candidates": int(
            len(selection_result.evaluated_candidates)
        ),
        "macro_features": input_metadata.get("macro_features", {}),
        "candidate_selection_mode": str(config.candidate_selection_mode),
        "candidate_selection_checkpoints": tuple(
            _checkpoint_payload(checkpoint)
            for checkpoint in selection_result.checkpoints
        ),
        "candidate_selection_final_samples": int(
            selection_result.final_checkpoint_samples
        ),
        "candidate_selection_stopped_early": bool(selection_result.stopped_early),
        "candidate_selection_stopping_reason": str(
            selection_result.stopping_reason
        ),
        "candidate_selection_base_seed": int(config.candidate_selection_base_seed),
        "selected_candidate_identity": selection_result.final_selected_candidate_identity,
        "selected_partition_signature": (
            final_selection.selected_partition_signature
            if final_selection is not None
            else selected_partition
        ),
        "selected_policy_option_indices": (
            final_selection.selected_policy_option_indices
            if final_selection is not None
            else selected_policy_indices
        ),
        "target_distribution_mc_samples": int(
            target_result.target_distribution_mc_samples
        ),
        "target_distribution_base_seed": int(config.target_distribution_base_seed),
        "full_graph_successor_state_counts": dict(
            target_result.successor_state_counts
        ),
        "full_graph_successor_state_probabilities": dict(
            target_result.successor_state_probabilities
        ),
        "candidate_selection_diagnostics": {
            "candidate_count": int(len(selection_result.evaluated_candidates)),
            "checkpoint_comparisons": tuple(
                bgr.compare_candidate_selection_checkpoints(lower, higher)
                for lower, higher in zip(
                    selection_result.checkpoints,
                    selection_result.checkpoints[1:],
                )
            ),
            **dict(selection_result.diagnostics),
        },
        "target_distribution_diagnostics": {
            "checkpoint_sample_counts": tuple(
                int(checkpoint.target_distribution_mc_samples)
                for checkpoint in target_result.checkpoints
            ),
            "checkpoint_comparisons": tuple(
                compare_target_distribution_checkpoints(lower, higher)
                for lower, higher in zip(
                    target_result.checkpoints,
                    target_result.checkpoints[1:],
                )
            ),
            **dict(target_result.diagnostics),
        },
        "node_marginals": dict(target_result.node_marginals),
        "strategic_summaries": dict(target_result.strategic_summaries),
        "target_sampling_runtime_seconds": float(target_result.runtime_seconds),
        "unique_successor_state_count": int(
            target_result.unique_successor_state_count
        ),
    }
    return row


def _resolved_v3_example_metadata(
    *,
    input_metadata: Mapping[str, Any],
    global_state: GlobalState,
    battle_graph,
    full_graph,
    config: TransitionSamplingV3Config,
) -> Tuple[str, str, str, str]:
    input_state_id = str(
        input_metadata.get("input_state_id")
        or input_metadata.get("example_id")
        or _stable_digest(bgr.canonical_two_stage_global_state_signature(global_state))
    )
    initial_signature = input_metadata.get(
        "initial_global_state_signature",
        bgr.canonical_two_stage_global_state_signature(global_state),
    )
    battle_signature = input_metadata.get(
        "battle_graph_signature", gdm.canonical_graph_signature(battle_graph)
    )
    full_signature = input_metadata.get(
        "full_graph_signature", gdm.canonical_graph_signature(full_graph)
    )
    input_fingerprint = input_state_fingerprint(
        initial_global_state_signature=initial_signature,
        battle_graph_signature=battle_signature,
        full_graph_signature=full_signature,
    )
    target_fingerprint = stage_a_v3_target_fingerprint(config)
    example_id = canonical_stage_a_v3_example_id(
        input_state_id=input_state_id,
        input_fingerprint=input_fingerprint,
        semantic_target_fingerprint=target_fingerprint,
    )
    return input_state_id, input_fingerprint, target_fingerprint, example_id


def collect_transition_distribution_example_v3_for_state(
    *,
    players: Sequence["Players.Player"],
    battle_graph,
    full_graph,
    global_state: GlobalState,
    input_metadata: Mapping[str, Any],
    config: TransitionSamplingV3Config,
    selection_resume_state: Optional[Any] = None,
    target_resume_state: Optional[Any] = None,
    selection_checkpoint_callback: Optional[Callable[..., None]] = None,
    target_checkpoint_callback: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    """Run the production preparation path, then the two separate MC phases."""
    input_state_id, input_fingerprint, target_fingerprint, example_id = (
        _resolved_v3_example_metadata(
            input_metadata=input_metadata,
            global_state=global_state,
            battle_graph=battle_graph,
            full_graph=full_graph,
            config=config,
        )
    )
    gdm.apply_global_state_to_board(global_state, players)
    prepared = bgr.prepare_two_stage_partition_policy_candidates(
        players=players,
        battle_graph=battle_graph,
        combat_libraries_base=Path(config.combat_libraries_base),
        max_partitions=int(config.max_partitions),
        ranking_variable=str(config.ranking_variable),
        first_stage_value_tolerances=config.first_stage_value_tolerances,
        max_policy_combos_per_partition=config.max_policy_combos_per_partition,
        partition_candidate_selection_mode=config.partition_candidate_selection_mode,
        utility_abs_tolerance=config.utility_abs_tolerance,
        utility_rel_tolerance=config.utility_rel_tolerance,
        max_candidates_per_partition=config.max_candidates_per_partition,
    )
    candidate_seed = derive_stage_a_v3_phase_seed(
        base_seed=int(config.candidate_selection_base_seed),
        example_id=example_id,
        phase="candidate_selection",
    )
    if config.candidate_selection_mode == "fixed":
        fixed_count = int(config.resolved_candidate_selection_mc_samples)
        candidate_points = tuple(
            value
            for value in config.candidate_selection_checkpoints
            if int(value) <= fixed_count
        )
        if not candidate_points:
            candidate_points = (fixed_count,)
        selection_result = bgr.evaluate_candidates_at_nested_checkpoints(
            prepared_candidates=prepared,
            players=players,
            battle_graph=battle_graph,
            combat_libraries_base=Path(config.combat_libraries_base),
            ranking_variable=str(config.ranking_variable),
            checkpoints=candidate_points,
            base_seed=candidate_seed,
            selection_mode="fixed",
            fixed_sample_count=fixed_count,
            stability_required_consecutive=int(
                config.candidate_stability_required_consecutive
            ),
            resume_state=selection_resume_state,
            checkpoint_callback=selection_checkpoint_callback,
            profile_second_stage=bool(config.profile_second_stage),
        )
    else:
        selection_result = bgr.evaluate_candidates_at_nested_checkpoints(
            prepared_candidates=prepared,
            players=players,
            battle_graph=battle_graph,
            combat_libraries_base=Path(config.combat_libraries_base),
            ranking_variable=str(config.ranking_variable),
            checkpoints=config.candidate_selection_checkpoints,
            base_seed=candidate_seed,
            selection_mode="adaptive_checkpoints",
            max_samples=int(config.candidate_selection_max_samples),
            stability_required_consecutive=int(
                config.candidate_stability_required_consecutive
            ),
            score_gap_abs_threshold=config.candidate_score_gap_abs_threshold,
            score_gap_rel_threshold=config.candidate_score_gap_rel_threshold,
            resume_state=selection_resume_state,
            checkpoint_callback=selection_checkpoint_callback,
            profile_second_stage=bool(config.profile_second_stage),
        )
    selected_candidate = selection_result.selected_candidate
    if selected_candidate is None:
        raise RuntimeError(
            f"Candidate selection failed: {selection_result.stopping_reason}"
        )

    target_count = int(config.resolved_target_distribution_mc_samples)
    target_points = tuple(
        value
        for value in config.target_distribution_checkpoints
        if int(value) <= target_count
    )
    if not target_points:
        target_points = (target_count,)
    target_seed = derive_stage_a_v3_phase_seed(
        base_seed=int(config.target_distribution_base_seed),
        example_id=example_id,
        phase="selected_candidate_target_distribution",
    )
    target_result = sample_selected_candidate_successor_distribution(
        selected_candidate=selected_candidate,
        initial_global_state=global_state,
        full_graph=full_graph,
        sample_count=target_count,
        checkpoints=target_points,
        base_seed=target_seed,
        example_id=example_id,
        battle_nodes=prepared.battle_nodes,
        resume_state=target_resume_state,
        checkpoint_callback=target_checkpoint_callback,
    )
    return build_stage_a_v3_grouped_row(
        input_metadata=input_metadata,
        config=config,
        example_id=example_id,
        input_state_id=input_state_id,
        input_fingerprint=input_fingerprint,
        semantic_target_fingerprint=target_fingerprint,
        selection_result=selection_result,
        target_result=target_result,
        selected_candidate=selected_candidate,
    )


def _numeric_or_zero(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def select_stage_a_v3_calibration_states(
    grouped_examples_df: pd.DataFrame,
    *,
    states_per_continent: Mapping[str, int],
    random_seed: int,
    include_top_candidate_outliers: int = 5,
    one_example_id: Optional[str] = None,
) -> pd.DataFrame:
    """Deterministically stratify ordinary states while forcing hard outliers in."""
    if grouped_examples_df.empty:
        return grouped_examples_df.copy()
    frame = grouped_examples_df.copy()
    if "example_id" not in frame or frame["example_id"].astype(str).duplicated().any():
        raise ValueError("Calibration input requires unique example_id values")
    if one_example_id is not None:
        result = frame[frame["example_id"].astype(str) == str(one_example_id)].copy()
        if result.empty:
            raise KeyError(f"Unknown calibration example {one_example_id!r}")
        result["calibration_selection_reasons"] = [("explicit_example",)]
        return result.reset_index(drop=True)

    candidate_field = "num_retained_second_stage_candidates"
    if candidate_field not in frame:
        frame[candidate_field] = frame.get("two_stage_all_candidate_count", 0)
    if "battle_node_count" not in frame:
        frame["battle_node_count"] = frame.get("battle_graph_nodes", pd.Series([()] * len(frame))).map(
            lambda value: len(value or ())
        )
    if "num_maximal_partitions" not in frame:
        frame["num_maximal_partitions"] = 0
    if "attacker_troop_total" not in frame:
        frame["attacker_troop_total"] = 0
    if "defender_troop_total" not in frame:
        frame["defender_troop_total"] = 0
    if "target_generation_runtime_seconds" not in frame:
        frame["target_generation_runtime_seconds"] = 0.0
    if "previous_candidate_selection_changed" not in frame:
        frame["previous_candidate_selection_changed"] = False
    frame["candidate_count_category"] = frame[candidate_field].map(
        lambda value: gdm._candidate_count_category(int(_numeric_or_zero(value)))
    )
    frame["battle_node_count_category"] = frame["battle_node_count"].map(
        lambda value: gdm._battle_node_count_category(int(_numeric_or_zero(value)))
    )
    frame["_selection_score"] = frame["example_id"].astype(str).map(
        lambda example_id: _stable_seed(
            {
                "kind": "stage_a_v3_calibration_selection",
                "random_seed": int(random_seed),
                "example_id": example_id,
            }
        )
    )
    selected_ids: set[str] = set()
    reasons: Dict[str, set[str]] = {}

    def select(example_id: str, reason: str) -> None:
        selected_ids.add(str(example_id))
        reasons.setdefault(str(example_id), set()).add(str(reason))

    for continent_name, desired_raw in states_per_continent.items():
        desired = max(0, int(desired_raw))
        continent = frame[frame["continent_name"].astype(str) == str(continent_name)].copy()
        if desired == 0 or continent.empty:
            continue
        desired = min(desired, len(continent))
        previous_changes = continent[
            continent["previous_candidate_selection_changed"].fillna(False).astype(bool)
        ].sort_values(
            [candidate_field, "example_id"], ascending=[False, True]
        )
        for example_id in previous_changes.head(desired)["example_id"].astype(str):
            select(example_id, "previous_mc_selection_change")

        current = {item for item in selected_ids if item in set(continent["example_id"].astype(str))}
        remaining_slots = max(0, desired - len(current))
        outlier_count = min(
            max(0, int(include_top_candidate_outliers)), remaining_slots
        )
        outliers = continent.sort_values(
            [
                candidate_field,
                "num_maximal_partitions",
                "battle_node_count",
                "attacker_troop_total",
                "defender_troop_total",
                "target_generation_runtime_seconds",
                "example_id",
            ],
            ascending=[False, False, False, False, False, False, True],
        )
        for example_id in outliers[~outliers["example_id"].astype(str).isin(current)].head(
            outlier_count
        )["example_id"].astype(str):
            select(example_id, "difficulty_outlier")
            current.add(example_id)

        strata = [
            group.sort_values(["_selection_score", "example_id"])
            for _, group in continent.groupby(
                ["candidate_count_category", "battle_node_count_category"],
                sort=True,
            )
        ]
        cursor = 0
        while len(current) < desired and strata:
            progressed = False
            for group in strata:
                available = group[
                    ~group["example_id"].astype(str).isin(current)
                ]
                if available.empty:
                    continue
                example_id = str(available.iloc[cursor % len(available)]["example_id"])
                select(example_id, "difficulty_stratum")
                current.add(example_id)
                progressed = True
                if len(current) >= desired:
                    break
            if not progressed:
                break
            cursor += 1
        if len(current) < desired:
            remaining = continent[
                ~continent["example_id"].astype(str).isin(current)
            ].sort_values(["_selection_score", "example_id"])
            for example_id in remaining.head(desired - len(current))["example_id"].astype(str):
                select(example_id, "deterministic_fill")
                current.add(example_id)

    result = frame[frame["example_id"].astype(str).isin(selected_ids)].copy()
    result["calibration_selection_seed"] = int(random_seed)
    result["calibration_selection_reasons"] = result["example_id"].astype(str).map(
        lambda example_id: tuple(sorted(reasons.get(example_id, {"deterministic_fill"})))
    )
    return result.drop(columns=["_selection_score"]).sort_values(
        ["continent_name", "example_id"]
    ).reset_index(drop=True)


def collect_additional_calibration_input_states(
    *,
    states_per_continent: Mapping[str, int],
    random_seed: int,
    output_path: Optional[Path | str] = None,
    max_attempts_multiplier: int = 20,
    state_generator: Optional[Callable[..., Any]] = None,
) -> pd.DataFrame:
    """Collect raw reconstruction payloads only; no MC targets are generated."""
    if state_generator is None:
        from project_risk.mathematical.transition_prediction_ml.state_generators import ml_full_graph_state_generator

        state_generator = ml_full_graph_state_generator
    snapshot = {
        int(index): (territory._owner, int(territory._troops))
        for index, territory in Board.node_to_territory_dict.items()
    }
    rows = []
    try:
        for continent_name, desired_raw in states_per_continent.items():
            desired = max(0, int(desired_raw))
            successes = 0
            attempt_index = 0
            max_attempts = max(1, desired * int(max_attempts_multiplier))
            while successes < desired and attempt_index < max_attempts:
                seed = _stable_seed(
                    {
                        "kind": "stage_a_v3_raw_calibration_state",
                        "random_seed": int(random_seed),
                        "continent_name": str(continent_name),
                        "attempt_index": int(attempt_index),
                    }
                )
                rng = np.random.default_rng(seed)
                territory_ratio = float(rng.uniform(0.2, 0.8))
                troops_ratio = float(rng.uniform(0.5, 2.0))
                constraints = gdm.ExperimentConstraints(
                    continent_name=str(continent_name),
                    max_attacker_troops_per_node=5,
                    max_defender_troops_per_node=5,
                )
                attempt_index += 1
                try:
                    players, battle_graph, full_graph = state_generator(
                        territory_ratio,
                        troops_ratio,
                        constraints,
                        rng,
                    )
                    if int(battle_graph.number_of_edges()) <= 0:
                        continue
                    global_state = agop.build_global_state_for_board(players)
                    initial_global = bgr.canonical_two_stage_global_state_signature(
                        global_state
                    )
                    initial_full = gdm.lift_battle_signature_to_full_graph_signature(
                        battle_signature=tuple(),
                        initial_global_state=global_state,
                        full_graph=full_graph,
                    )
                    battle_signature = gdm.canonical_graph_signature(battle_graph)
                    full_signature = gdm.canonical_graph_signature(full_graph)
                    raw_fingerprint = input_state_fingerprint(
                        initial_global_state_signature=initial_global,
                        battle_graph_signature=battle_signature,
                        full_graph_signature=full_signature,
                    )
                    example_id = f"raw_calibration_{raw_fingerprint[:24]}"
                    battle_nodes = tuple(sorted(int(node) for node in battle_graph.nodes()))
                    attacker_nodes = [
                        node
                        for node in battle_nodes
                        if global_state.nodes[node].owner == "A"
                    ]
                    defender_nodes = [
                        node
                        for node in battle_nodes
                        if global_state.nodes[node].owner == "D"
                    ]
                    rows.append(
                        {
                            "example_id": example_id,
                            "input_state_id": example_id,
                            "continent_name": str(continent_name),
                            "attack_perspective": "P1_as_attacker",
                            "state_id": int(successes),
                            "attempt_index": int(attempt_index - 1),
                            "state_generation_seed": int(seed),
                            "initial_global_state_signature": initial_global,
                            "initial_full_graph_signature": initial_full,
                            "battle_graph_signature": battle_signature,
                            "full_graph_signature": full_signature,
                            "battle_graph_nodes": battle_nodes,
                            "full_graph_nodes": tuple(
                                sorted(int(node) for node in full_graph.nodes())
                            ),
                            "battle_node_count": int(len(battle_nodes)),
                            "attacker_node_count": int(len(attacker_nodes)),
                            "defender_node_count": int(len(defender_nodes)),
                            "attacker_troop_total": int(
                                sum(global_state.nodes[node].troops for node in attacker_nodes)
                            ),
                            "defender_troop_total": int(
                                sum(global_state.nodes[node].troops for node in defender_nodes)
                            ),
                            "num_retained_second_stage_candidates": 0,
                            "num_maximal_partitions": 0,
                            "target_generation_runtime_seconds": 0.0,
                            "previous_candidate_selection_changed": False,
                            "raw_input_state_only": True,
                            "macro_features": {
                                "target_territory_ratio": territory_ratio,
                                "target_troops_ratio": troops_ratio,
                            },
                        }
                    )
                    successes += 1
                except Exception:
                    continue
    finally:
        for index, (owner, troops) in snapshot.items():
            territory = Board.node_to_territory_dict[index]
            territory._owner = owner
            territory._troops = troops
    frame = pd.DataFrame(rows)
    if output_path is not None:
        _atomic_write_pickle(Path(output_path), frame)
    return frame


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def mark_stage_a_v2_dataset_provisional(
    source_stage_a_v2_dir: Path | str,
) -> Dict[str, Any]:
    """Add one external status file; no v2 manifest or data chunk is rewritten."""
    root = Path(source_stage_a_v2_dir)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"No Stage A v2 manifest found in {root}")
    chunk_paths = sorted(root.glob("*/grouped_examples/chunk_*.pkl"))
    chunk_fingerprint = _stable_digest(
        [(str(path.relative_to(root)), _file_sha256(path)) for path in chunk_paths]
    )
    status = {
        "dataset_status": "provisional_mc5",
        "eligible_for_stage_b_training": False,
        "reason": (
            "candidate-selection and target-distribution sampling sensitivity "
            "under calibration"
        ),
        "preserved_uses": (
            "input-state reconstruction",
            "benchmarking",
            "calibration selection",
            "eventual deterministic target regeneration",
        ),
        "data_chunks_rewritten": False,
        "grouped_chunk_count": int(len(chunk_paths)),
        "grouped_chunk_fingerprint": chunk_fingerprint,
        "status_recorded_at": _utc_now(),
    }
    _atomic_write_json(root / "dataset_status.json", status)
    return status


def _safe_example_filename(example_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(example_id)
    )
    return safe[:120] or _stable_digest(example_id)[:24]


class StageAV3CalibrationStore:
    """Small-file calibration store with phase-specific resumable checkpoints."""

    DIRECTORY_NAMES = (
        "input_state_bank",
        "selection_checkpoints",
        "selected_candidate_targets",
        "grouped_rows",
        "candidate_comparisons",
        "target_distribution_comparisons",
        "failures",
        "checkpoints",
    )

    def __init__(
        self,
        *,
        output_dir: Path | str,
        source_stage_a_v2_dir: Path | str,
        config: TransitionSamplingV3Config,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.source_stage_a_v2_dir = Path(source_stage_a_v2_dir)
        self.config = config
        self.config_fingerprint = stage_a_v3_config_fingerprint(config)
        self.target_fingerprint = stage_a_v3_target_fingerprint(config)
        self._initialize()

    def _initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in self.DIRECTORY_NAMES:
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / "config.json"
        config_payload = {
            "calibration_format_version": CALIBRATION_FORMAT_VERSION,
            "target_generation_version": TARGET_GENERATION_VERSION,
            "stage_a_schema_version": STAGE_A_SCHEMA_VERSION,
            "config_fingerprint": self.config_fingerprint,
            "semantic_target_fingerprint": self.target_fingerprint,
            "configuration": asdict(self.config),
            "resolved_candidate_selection_mc_samples": (
                self.config.resolved_candidate_selection_mc_samples
            ),
            "resolved_target_distribution_mc_samples": (
                self.config.resolved_target_distribution_mc_samples
            ),
        }
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != self.config_fingerprint:
                raise ValueError("Calibration resume configuration fingerprint mismatch")
        else:
            _atomic_write_json(config_path, config_payload)
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            _atomic_write_json(
                manifest_path,
                {
                    "calibration_format_version": CALIBRATION_FORMAT_VERSION,
                    "target_generation_version": TARGET_GENERATION_VERSION,
                    "stage_a_schema_version": STAGE_A_SCHEMA_VERSION,
                    "source_stage_a_v2_dir": str(
                        self.source_stage_a_v2_dir.resolve()
                    ),
                    "config_fingerprint": self.config_fingerprint,
                    "semantic_target_fingerprint": self.target_fingerprint,
                    "created_at": _utc_now(),
                    "completed_example_ids": [],
                    "failed_example_ids": [],
                    "layout": tuple(self.DIRECTORY_NAMES),
                },
            )

    def _path(self, directory: str, example_id: str, suffix: str = ".pkl") -> Path:
        return (
            self.output_dir
            / directory
            / f"{_safe_example_filename(example_id)}{suffix}"
        )

    def load_manifest(self) -> Dict[str, Any]:
        return json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))

    def save_input_state_bank(self, frame: pd.DataFrame) -> None:
        selected_path = self.output_dir / "input_state_bank" / "selected_states.pkl"
        if selected_path.exists():
            existing = pd.read_pickle(selected_path)
            frame = pd.concat([existing, frame], ignore_index=True, sort=False)
            if "example_id" in frame:
                frame = frame.drop_duplicates(subset=["example_id"], keep="last")
            frame = frame.sort_values(
                [field for field in ("continent_name", "example_id") if field in frame]
            ).reset_index(drop=True)
        _atomic_write_pickle(
            selected_path, frame
        )
        _atomic_write_json(
            self.output_dir / "input_state_bank" / "selected_states_summary.json",
            {
                "rows": int(len(frame)),
                "continents": {
                    str(key): int(value)
                    for key, value in frame.groupby("continent_name").size().items()
                }
                if not frame.empty
                else {},
            },
        )

    def save_selection_resume(
        self, example_id: str, resume_state: bgr.NestedCandidateSelectionResumeState
    ) -> None:
        _atomic_write_pickle(
            self._path("checkpoints", f"{example_id}_selection"), resume_state
        )

    def load_selection_resume(self, example_id: str) -> Optional[Any]:
        path = self._path("checkpoints", f"{example_id}_selection")
        return pickle.load(path.open("rb")) if path.exists() else None

    def save_target_resume(
        self, example_id: str, resume_state: SelectedCandidateTargetResumeState
    ) -> None:
        _atomic_write_pickle(
            self._path("checkpoints", f"{example_id}_target"), resume_state
        )

    def load_target_resume(self, example_id: str) -> Optional[Any]:
        path = self._path("checkpoints", f"{example_id}_target")
        return pickle.load(path.open("rb")) if path.exists() else None

    def save_selection_result(
        self, example_id: str, result: bgr.NestedCandidateSelectionResult
    ) -> None:
        _atomic_write_pickle(self._path("selection_checkpoints", example_id), result)

    def load_selection_result(
        self, example_id: str
    ) -> Optional[bgr.NestedCandidateSelectionResult]:
        path = self._path("selection_checkpoints", example_id)
        return pickle.load(path.open("rb")) if path.exists() else None

    def save_target_result(
        self, example_id: str, result: SelectedCandidateDistributionResult
    ) -> None:
        _atomic_write_pickle(self._path("selected_candidate_targets", example_id), result)

    def load_target_result(
        self, example_id: str
    ) -> Optional[SelectedCandidateDistributionResult]:
        path = self._path("selected_candidate_targets", example_id)
        return pickle.load(path.open("rb")) if path.exists() else None

    def save_completed_example(
        self,
        *,
        example_id: str,
        row: Mapping[str, Any],
        candidate_comparisons: Sequence[Mapping[str, Any]],
        target_comparisons: Sequence[Mapping[str, Any]],
    ) -> None:
        _atomic_write_pickle(self._path("grouped_rows", example_id), dict(row))
        _atomic_write_json(
            self._path("candidate_comparisons", example_id, ".json"),
            tuple(candidate_comparisons),
        )
        _atomic_write_json(
            self._path("target_distribution_comparisons", example_id, ".json"),
            tuple(target_comparisons),
        )
        manifest = self.load_manifest()
        completed = set(manifest.get("completed_example_ids", ()))
        completed.add(str(example_id))
        manifest["completed_example_ids"] = sorted(completed)
        manifest["updated_at"] = _utc_now()
        _atomic_write_json(self.output_dir / "manifest.json", manifest)
        _atomic_write_json(
            self.output_dir / "checkpoints" / "completed_states.json",
            {
                "completed_example_ids": sorted(completed),
                "completed_count": int(len(completed)),
                "updated_at": _utc_now(),
            },
        )

    def save_failure(self, example_id: str, failure: Mapping[str, Any]) -> None:
        _atomic_write_json(
            self._path("failures", example_id, ".json"), dict(failure)
        )
        manifest = self.load_manifest()
        failures = set(manifest.get("failed_example_ids", ()))
        failures.add(str(example_id))
        manifest["failed_example_ids"] = sorted(failures)
        manifest["updated_at"] = _utc_now()
        _atomic_write_json(self.output_dir / "manifest.json", manifest)


def load_stage_a_v3_grouped_rows(output_dir: Path | str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(output_dir).glob("grouped_rows/*.pkl")):
        with path.open("rb") as handle:
            rows.append(pickle.load(handle))
    return pd.DataFrame(rows)


def _selection_is_stable_at_final_configured_checkpoint(
    result: bgr.NestedCandidateSelectionResult,
    *,
    required_consecutive: int,
) -> bool:
    identities = [
        checkpoint.selected_candidate_identity for checkpoint in result.checkpoints
    ]
    required = int(required_consecutive)
    return bool(
        len(identities) >= required
        and len(set(identities[-required:])) == 1
    )


def _checkpoint_by_samples(
    checkpoints: Sequence[Any], sample_count: int, field_name: str
) -> Optional[Any]:
    for checkpoint in checkpoints:
        if int(getattr(checkpoint, field_name)) == int(sample_count):
            return checkpoint
    return None


def run_stage_a_v3_calibration_example(
    *,
    input_example: Mapping[str, Any],
    config: TransitionSamplingV3Config,
    store: StageAV3CalibrationStore,
    global_state_utility_evaluator: Optional[
        Callable[[GlobalState], Sequence[float]]
    ] = None,
) -> Dict[str, Any]:
    """Run one resumable calibration state through fixed checkpoints and targets."""
    global_state = _global_state_from_raw_signature(
        input_example.get("initial_global_state_signature")
    )
    battle_graph = _graph_from_signature(input_example.get("battle_graph_signature"))
    full_graph = _graph_from_signature(input_example.get("full_graph_signature"))
    players = [Players.Player("A"), Players.Player("D")]
    gdm.apply_global_state_to_board(global_state, players)
    input_state_id, input_fingerprint_value, target_fingerprint, example_id = (
        _resolved_v3_example_metadata(
            input_metadata=input_example,
            global_state=global_state,
            battle_graph=battle_graph,
            full_graph=full_graph,
            config=config,
        )
    )
    if example_id in set(store.load_manifest().get("completed_example_ids", ())):
        row_path = store._path("grouped_rows", example_id)
        if row_path.exists():
            with row_path.open("rb") as handle:
                return {
                    "status": "already_completed",
                    "example_id": example_id,
                    "row": pickle.load(handle),
                }

    selection_result = store.load_selection_result(example_id)
    if selection_result is None:
        prepared = bgr.prepare_two_stage_partition_policy_candidates(
            players=players,
            battle_graph=battle_graph,
            combat_libraries_base=Path(config.combat_libraries_base),
            max_partitions=int(config.max_partitions),
            ranking_variable=str(config.ranking_variable),
            first_stage_value_tolerances=config.first_stage_value_tolerances,
            max_policy_combos_per_partition=config.max_policy_combos_per_partition,
            partition_candidate_selection_mode=config.partition_candidate_selection_mode,
            utility_abs_tolerance=config.utility_abs_tolerance,
            utility_rel_tolerance=config.utility_rel_tolerance,
            max_candidates_per_partition=config.max_candidates_per_partition,
        )
        configured_points = tuple(int(value) for value in config.candidate_selection_checkpoints)
        initial_limit = int(configured_points[-1])
        candidate_seed = derive_stage_a_v3_phase_seed(
            base_seed=int(config.candidate_selection_base_seed),
            example_id=example_id,
            phase="candidate_selection",
        )

        def selection_callback(
            checkpoint: bgr.CandidateSelectionCheckpointResult,
            resume: bgr.NestedCandidateSelectionResumeState,
        ) -> None:
            del checkpoint
            store.save_selection_resume(example_id, resume)

        selection_result = bgr.evaluate_candidates_at_nested_checkpoints(
            prepared_candidates=prepared,
            players=players,
            battle_graph=battle_graph,
            combat_libraries_base=Path(config.combat_libraries_base),
            ranking_variable=str(config.ranking_variable),
            checkpoints=configured_points,
            base_seed=candidate_seed,
            selection_mode="fixed",
            fixed_sample_count=initial_limit,
            stability_required_consecutive=int(
                config.candidate_stability_required_consecutive
            ),
            resume_state=store.load_selection_resume(example_id),
            checkpoint_callback=selection_callback,
            global_state_utility_evaluator=global_state_utility_evaluator,
            profile_second_stage=bool(config.profile_second_stage),
        )
        stable_at_initial_limit = _selection_is_stable_at_final_configured_checkpoint(
            selection_result,
            required_consecutive=int(config.candidate_stability_required_consecutive),
        )
        final_two_differ = bool(
            len(selection_result.checkpoints) >= 2
            and selection_result.checkpoints[-2].selected_candidate_identity
            != selection_result.checkpoints[-1].selected_candidate_identity
        )
        needs_maximum = bool(final_two_differ or not stable_at_initial_limit)
        if (
            needs_maximum
            and selection_result.selected_candidate is not None
            and int(config.candidate_selection_max_samples) > initial_limit
        ):
            extended_points = configured_points + (
                int(config.candidate_selection_max_samples),
            )
            selection_result = bgr.evaluate_candidates_at_nested_checkpoints(
                prepared_candidates=prepared,
                players=players,
                battle_graph=battle_graph,
                combat_libraries_base=Path(config.combat_libraries_base),
                ranking_variable=str(config.ranking_variable),
                checkpoints=extended_points,
                base_seed=candidate_seed,
                selection_mode="fixed",
                fixed_sample_count=int(config.candidate_selection_max_samples),
                stability_required_consecutive=int(
                    config.candidate_stability_required_consecutive
                ),
                resume_state=selection_result.resume_state,
                checkpoint_callback=selection_callback,
                global_state_utility_evaluator=global_state_utility_evaluator,
                profile_second_stage=bool(config.profile_second_stage),
            )
        store.save_selection_result(example_id, selection_result)

    selected_candidate = selection_result.selected_candidate
    if selected_candidate is None:
        raise RuntimeError(
            f"Calibration candidate selection failed: {selection_result.stopping_reason}"
        )
    target_result = store.load_target_result(example_id)
    if target_result is None:
        target_points = tuple(int(value) for value in config.target_distribution_checkpoints)
        target_count = int(target_points[-1])
        target_seed = derive_stage_a_v3_phase_seed(
            base_seed=int(config.target_distribution_base_seed),
            example_id=example_id,
            phase="selected_candidate_target_distribution",
        )

        def target_callback(
            checkpoint: TargetDistributionCheckpointResult,
            resume: SelectedCandidateTargetResumeState,
        ) -> None:
            del checkpoint
            store.save_target_resume(example_id, resume)

        target_result = sample_selected_candidate_successor_distribution(
            selected_candidate=selected_candidate,
            initial_global_state=global_state,
            full_graph=full_graph,
            sample_count=target_count,
            checkpoints=target_points,
            base_seed=target_seed,
            example_id=example_id,
            battle_nodes=tuple(int(node) for node in battle_graph.nodes()),
            resume_state=store.load_target_resume(example_id),
            checkpoint_callback=target_callback,
        )
        store.save_target_result(example_id, target_result)

    row = build_stage_a_v3_grouped_row(
        input_metadata=input_example,
        config=config,
        example_id=example_id,
        input_state_id=input_state_id,
        input_fingerprint=input_fingerprint_value,
        semantic_target_fingerprint=target_fingerprint,
        selection_result=selection_result,
        target_result=target_result,
        selected_candidate=selected_candidate,
    )
    row["calibration_design"] = "nested_5_10_20_40_conditional_max"
    row["original_stage_a_v2_example_id"] = input_example.get("example_id")
    candidate_comparisons = tuple(
        bgr.compare_candidate_selection_checkpoints(lower, higher)
        for lower, higher in zip(
            selection_result.checkpoints, selection_result.checkpoints[1:]
        )
    )
    target_comparisons = tuple(
        compare_target_distribution_checkpoints(lower, higher)
        for lower_index, lower in enumerate(target_result.checkpoints)
        for higher in target_result.checkpoints[lower_index + 1 :]
    )
    target_first_to_final = (
        target_comparisons[-1] if target_comparisons else None
    )
    row["instability_decomposition"] = decompose_selection_and_target_instability(
        lower_candidate_identity=(
            selection_result.checkpoints[0].selected_candidate_identity
            if selection_result.checkpoints
            else selection_result.final_selected_candidate_identity
        ),
        higher_candidate_identity=selection_result.final_selected_candidate_identity,
        target_distribution_comparison=target_first_to_final,
    )
    store.save_completed_example(
        example_id=example_id,
        row=row,
        candidate_comparisons=candidate_comparisons,
        target_comparisons=target_comparisons,
    )
    return {
        "status": "completed",
        "example_id": example_id,
        "selection_result": selection_result,
        "target_result": target_result,
        "row": row,
        "candidate_comparisons": candidate_comparisons,
        "target_comparisons": target_comparisons,
    }


def _merge_previous_selection_change_evidence(
    frame: pd.DataFrame, source_stage_a_v2_dir: Path
) -> pd.DataFrame:
    result = frame.copy()
    result["previous_candidate_selection_changed"] = False
    comparison_path = source_stage_a_v2_dir / "calibration_mc20" / "comparisons.pkl"
    if not comparison_path.exists():
        return result
    comparisons = pd.read_pickle(comparison_path)
    if (
        comparisons.empty
        or "base_example_id" not in comparisons
        or "candidate_selection_changed" not in comparisons
    ):
        return result
    changed = set(
        comparisons[
            comparisons["candidate_selection_changed"].astype(bool)
        ]["base_example_id"].astype(str)
    )
    result["previous_candidate_selection_changed"] = result["example_id"].astype(
        str
    ).isin(changed)
    return result


def run_transition_target_sampling_calibration(
    *,
    source_stage_a_v2_dir: Path | str = DEFAULT_SOURCE_STAGE_A_V2_DIR,
    output_dir: Path | str = DEFAULT_CALIBRATION_OUTPUT_DIR,
    config: Optional[TransitionSamplingV3Config] = None,
    states_per_continent: Optional[Mapping[str, int]] = None,
    random_seed: int = 42030,
    include_top_candidate_outliers: int = 5,
    one_example_id: Optional[str] = None,
    additional_input_states: Optional[pd.DataFrame] = None,
    max_wall_seconds: Optional[float] = None,
    validate_output: bool = True,
) -> Dict[str, Any]:
    config = config or TransitionSamplingV3Config()
    source = Path(source_stage_a_v2_dir)
    marker = mark_stage_a_v2_dataset_provisional(source)
    base_examples = load_stage_a_v2_grouped_examples(source)
    base_examples = _merge_previous_selection_change_evidence(base_examples, source)
    if additional_input_states is not None and not additional_input_states.empty:
        base_examples = pd.concat(
            [base_examples, additional_input_states], ignore_index=True, sort=False
        )
    targets = dict(states_per_continent or DEFAULT_CALIBRATION_STATE_TARGETS)
    selected = select_stage_a_v3_calibration_states(
        base_examples,
        states_per_continent=targets,
        random_seed=int(random_seed),
        include_top_candidate_outliers=int(include_top_candidate_outliers),
        one_example_id=one_example_id,
    )
    store = StageAV3CalibrationStore(
        output_dir=output_dir,
        source_stage_a_v2_dir=source,
        config=config,
    )
    store.save_input_state_bank(selected)
    started = time.perf_counter()
    completed_this_run = 0
    skipped_completed = 0
    failures_this_run = 0
    stopped_for_wall_time = False
    per_example = []
    for input_example in selected.to_dict(orient="records"):
        if max_wall_seconds is not None and (
            time.perf_counter() - started >= float(max_wall_seconds)
        ):
            stopped_for_wall_time = True
            break
        item_started = time.perf_counter()
        provisional_id = str(input_example.get("example_id"))
        try:
            result = run_stage_a_v3_calibration_example(
                input_example=input_example,
                config=config,
                store=store,
            )
            if result["status"] == "already_completed":
                skipped_completed += 1
            else:
                completed_this_run += 1
            per_example.append(
                {
                    "source_example_id": provisional_id,
                    "calibration_example_id": result["example_id"],
                    "status": result["status"],
                    "runtime_seconds": float(time.perf_counter() - item_started),
                }
            )
        except Exception as exc:
            failures_this_run += 1
            failure_id = _stable_digest(
                {"source_example_id": provisional_id, "config": store.config_fingerprint}
            )[:24]
            store.save_failure(
                failure_id,
                {
                    "source_example_id": provisional_id,
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback_module.format_exc(),
                    "runtime_seconds": float(time.perf_counter() - item_started),
                    "recorded_at": _utc_now(),
                },
            )
            per_example.append(
                {
                    "source_example_id": provisional_id,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "message": str(exc),
                    "runtime_seconds": float(time.perf_counter() - item_started),
                }
            )
    summary = summarize_transition_target_sampling_calibration(output_dir)
    validation = (
        validate_transition_target_sampling_calibration(output_dir, strict=False)
        if validate_output
        else None
    )
    status = {
        "status": (
            "in_progress_with_checkpoint"
            if stopped_for_wall_time
            else "completed_requested_states"
        ),
        "source_stage_a_v2_dir": str(source.resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "selected_states": int(len(selected)),
        "completed_this_run": int(completed_this_run),
        "skipped_completed": int(skipped_completed),
        "failures_this_run": int(failures_this_run),
        "stopped_for_wall_time": bool(stopped_for_wall_time),
        "runtime_seconds": float(time.perf_counter() - started),
        "v2_dataset_status": marker,
        "examples": per_example,
        "summary": summary,
        "validation": validation,
    }
    _atomic_write_json(Path(output_dir) / "calibration_status.json", status)
    return status


def _numeric_summary(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    numeric = [
        float(value)
        for value in values
        if value is not None and math.isfinite(_numeric_or_zero(value))
    ]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(len(numeric)),
        "mean": float(np.mean(numeric)),
        "median": float(np.median(numeric)),
        "p90": float(np.percentile(numeric, 90)),
        "maximum": float(max(numeric)),
    }


def recommend_candidate_selection_configuration(
    selection_results: Sequence[bgr.NestedCandidateSelectionResult],
    *,
    stability_required_consecutive: int = 2,
) -> Dict[str, Any]:
    fixed_counts = (10, 20, 40)
    fixed_agreement: Dict[int, Dict[str, Any]] = {}
    for sample_count in fixed_counts:
        agreements = []
        for result in selection_results:
            if not result.checkpoints:
                continue
            reference = result.checkpoints[-1]
            checkpoint = _checkpoint_by_samples(
                result.checkpoints, sample_count, "mc_samples"
            )
            if checkpoint is not None:
                agreements.append(
                    checkpoint.selected_candidate_identity
                    == reference.selected_candidate_identity
                )
        fixed_agreement[sample_count] = {
            "comparisons": int(len(agreements)),
            "agreement_fraction": (
                float(sum(agreements) / len(agreements)) if agreements else None
            ),
        }

    adaptive_samples = []
    adaptive_disagreements = []
    required = int(stability_required_consecutive)
    for result in selection_results:
        if not result.checkpoints:
            adaptive_samples.append(0)
            adaptive_disagreements.append(False)
            continue
        stop_checkpoint = result.checkpoints[-1]
        for index in range(required - 1, len(result.checkpoints)):
            window = result.checkpoints[index - required + 1 : index + 1]
            if len({item.selected_candidate_identity for item in window}) == 1:
                stop_checkpoint = result.checkpoints[index]
                break
        adaptive_samples.append(int(stop_checkpoint.mc_samples))
        adaptive_disagreements.append(
            stop_checkpoint.selected_candidate_identity
            != result.checkpoints[-1].selected_candidate_identity
        )
    disagreement_fraction = (
        float(sum(adaptive_disagreements) / len(adaptive_disagreements))
        if adaptive_disagreements
        else None
    )
    highest_samples = max(
        (
            result.checkpoints[-1].mc_samples
            for result in selection_results
            if result.checkpoints
        ),
        default=None,
    )
    adaptive_is_provisionally_supported = bool(
        adaptive_samples
        and disagreement_fraction == 0.0
        and highest_samples is not None
        and float(np.mean(adaptive_samples)) < float(highest_samples)
    )
    return {
        "reference_interpretation": (
            "The highest available checkpoint is a calibration reference, not truth."
        ),
        "fixed_count_agreement": fixed_agreement,
        "adaptive_mean_samples": (
            float(np.mean(adaptive_samples)) if adaptive_samples else None
        ),
        "adaptive_p90_samples": (
            float(np.percentile(adaptive_samples, 90)) if adaptive_samples else None
        ),
        "adaptive_max_samples": max(adaptive_samples, default=None),
        "adaptive_disagreement_with_highest_checkpoint": disagreement_fraction,
        "recommended_for_review": {
            "mode": (
                "adaptive_checkpoints"
                if adaptive_is_provisionally_supported
                else "fixed"
            ),
            "candidate_selection_mc_samples": (
                None if adaptive_is_provisionally_supported else highest_samples
            ),
            "checkpoints": tuple(
                sorted(
                    {
                        checkpoint.mc_samples
                        for result in selection_results
                        for checkpoint in result.checkpoints
                    }
                )
            ),
            "reason": (
                "Adaptive stopping matched every highest-checkpoint winner while "
                "using fewer scenarios on average."
                if adaptive_is_provisionally_supported
                else "Calibration does not yet justify stopping below the highest "
                "available checkpoint."
            ),
        },
    }


def analyze_target_distribution_sample_sizes(
    target_results: Sequence[SelectedCandidateDistributionResult],
) -> Dict[str, Any]:
    comparisons_by_count: Dict[int, List[Dict[str, Any]]] = {20: [], 50: []}
    for result in target_results:
        if not result.checkpoints:
            continue
        reference = result.checkpoints[-1]
        for sample_count in tuple(comparisons_by_count):
            checkpoint = _checkpoint_by_samples(
                result.checkpoints,
                sample_count,
                "target_distribution_mc_samples",
            )
            if checkpoint is not None and checkpoint is not reference:
                comparisons_by_count[sample_count].append(
                    compare_target_distribution_checkpoints(checkpoint, reference)
                )
    evidence: Dict[int, Dict[str, Any]] = {}
    for sample_count, comparisons in comparisons_by_count.items():
        evidence[sample_count] = {
            "comparisons": int(len(comparisons)),
            "tv_to_highest": _numeric_summary(
                item["total_variation_distance"] for item in comparisons
            ),
            "js_to_highest": _numeric_summary(
                item["jensen_shannon_divergence"] for item in comparisons
            ),
            "top_state_agreement_fraction": (
                float(sum(item["top_1_agreement"] for item in comparisons) / len(comparisons))
                if comparisons
                else None
            ),
            "strategic_summary_absolute_error": _numeric_summary(
                abs(float(value))
                for item in comparisons
                for value in item["strategic_summary_differences"].values()
            ),
        }
    fifty = evidence.get(50, {})
    fifty_tv = dict(fifty.get("tv_to_highest", {}) or {})
    fifty_top = fifty.get("top_state_agreement_fraction")
    recommend_fifty = bool(
        fifty.get("comparisons", 0)
        and fifty_tv.get("p90") is not None
        and float(fifty_tv["p90"]) <= 0.10
        and fifty_top is not None
        and float(fifty_top) >= 0.90
    )
    return {
        "reference_interpretation": (
            "The highest target checkpoint is an empirical reference, not an exact distribution."
        ),
        "sample_count_evidence": evidence,
        "recommended_for_review": {
            "target_distribution_mc_samples": 50 if recommend_fifty else 100,
            "reason": (
                "Target50 met the provisional TV and top-state agreement thresholds."
                if recommend_fifty
                else "Current evidence does not justify reducing the target sample count below 100."
            ),
        },
    }


def _agreement_summary_for_results(
    records: Sequence[Tuple[Mapping[str, Any], bgr.NestedCandidateSelectionResult]],
) -> Dict[str, Any]:
    pair_names = ((5, 10), (10, 20), (20, 40), (40, 80))
    summary: Dict[str, Any] = {}
    for lower_count, higher_count in pair_names:
        comparisons = []
        for _, result in records:
            lower = _checkpoint_by_samples(result.checkpoints, lower_count, "mc_samples")
            higher = _checkpoint_by_samples(result.checkpoints, higher_count, "mc_samples")
            if lower is not None and higher is not None:
                comparisons.append(
                    bgr.compare_candidate_selection_checkpoints(lower, higher)
                )
        key = f"mc{lower_count}_vs_mc{higher_count}"
        summary[key] = {
            "comparisons": int(len(comparisons)),
            "winner_agreement_fraction": (
                float(
                    sum(not item["candidate_changed"] for item in comparisons)
                    / len(comparisons)
                )
                if comparisons
                else None
            ),
            "partition_agreement_fraction": (
                float(
                    sum(not item["partition_changed"] for item in comparisons)
                    / len(comparisons)
                )
                if comparisons
                else None
            ),
            "policy_agreement_fraction": (
                float(
                    sum(not item["policy_changed"] for item in comparisons)
                    / len(comparisons)
                )
                if comparisons
                else None
            ),
            "top_3_overlap": _numeric_summary(
                item["top_3_overlap_fraction"] for item in comparisons
            ),
            "top_5_overlap": _numeric_summary(
                item["top_5_overlap_fraction"] for item in comparisons
            ),
            "rank_correlation": _numeric_summary(
                item["rank_correlation"] for item in comparisons
            ),
        }
    return summary


def _target_comparison_summary(
    records: Sequence[Tuple[Mapping[str, Any], SelectedCandidateDistributionResult]],
) -> Dict[str, Any]:
    pairs = ((20, 50), (50, 100), (20, 100))
    summary: Dict[str, Any] = {}
    for lower_count, higher_count in pairs:
        comparisons = []
        for _, result in records:
            lower = _checkpoint_by_samples(
                result.checkpoints,
                lower_count,
                "target_distribution_mc_samples",
            )
            higher = _checkpoint_by_samples(
                result.checkpoints,
                higher_count,
                "target_distribution_mc_samples",
            )
            if lower is not None and higher is not None:
                comparisons.append(compare_target_distribution_checkpoints(lower, higher))
        key = f"target{lower_count}_vs_target{higher_count}"
        summary[key] = {
            "comparisons": int(len(comparisons)),
            "total_variation": _numeric_summary(
                item["total_variation_distance"] for item in comparisons
            ),
            "jensen_shannon": _numeric_summary(
                item["jensen_shannon_divergence"] for item in comparisons
            ),
            "support_overlap": _numeric_summary(
                item["support_intersection_over_union"] for item in comparisons
            ),
            "probability_mass_overlap": _numeric_summary(
                item["probability_mass_overlap"] for item in comparisons
            ),
            "top_state_agreement_fraction": (
                float(sum(item["top_1_agreement"] for item in comparisons) / len(comparisons))
                if comparisons
                else None
            ),
            "maximum_node_ownership_difference": _numeric_summary(
                item["maximum_node_ownership_difference"] for item in comparisons
            ),
            "maximum_expected_troop_difference": _numeric_summary(
                item["maximum_expected_troop_difference"] for item in comparisons
            ),
            "strategic_summary_absolute_difference": _numeric_summary(
                abs(float(value))
                for item in comparisons
                for value in item["strategic_summary_differences"].values()
            ),
        }
    return summary


def refresh_calibration_paired_score_diagnostics(
    output_dir: Path | str,
) -> Dict[str, int]:
    """Refresh active-component paired diagnostics in already-written v3 results."""
    root = Path(output_dir)
    refreshed = 0
    skipped = 0
    for path in sorted(root.glob("selection_checkpoints/*.pkl")):
        with path.open("rb") as handle:
            result = pickle.load(handle)
        resume = result.resume_state
        if resume is None or not resume.candidate_utilities:
            row_path = root / "grouped_rows" / path.name
            if row_path.exists():
                with row_path.open("rb") as handle:
                    row = pickle.load(handle)
                row["num_retained_second_stage_candidates"] = int(
                    len(result.evaluated_candidates)
                )
                row["candidate_count_category"] = gdm._candidate_count_category(
                    len(result.evaluated_candidates)
                )
                _atomic_write_pickle(row_path, row)
            skipped += 1
            continue
        checkpoints = []
        for checkpoint in result.checkpoints:
            best_index = checkpoint.selected_candidate_index
            runner_index = checkpoint.runner_up_candidate_index
            if best_index is None:
                checkpoints.append(checkpoint)
                continue
            active_component = next(
                (
                    index
                    for index, value in enumerate(
                        checkpoint.best_runner_up_gap or ()
                    )
                    if abs(float(value)) > 1e-15
                ),
                0,
            )
            paired = bgr._paired_candidate_score_diagnostics(
                resume.candidate_utilities[int(best_index)][
                    : int(checkpoint.mc_samples)
                ],
                (
                    resume.candidate_utilities[int(runner_index)][
                        : int(checkpoint.mc_samples)
                    ]
                    if runner_index is not None
                    else None
                ),
                active_ranking_component=active_component,
            )
            diagnostics = dict(checkpoint.diagnostics)
            diagnostics["paired_best_runner_up"] = paired
            checkpoints.append(replace(checkpoint, diagnostics=diagnostics))
        new_checkpoints = tuple(checkpoints)
        new_resume = replace(resume, checkpoints=new_checkpoints)
        updated = replace(
            result, checkpoints=new_checkpoints, resume_state=new_resume
        )
        _atomic_write_pickle(path, updated)
        resume_path = root / "checkpoints" / f"{path.stem}_selection.pkl"
        if resume_path.exists():
            _atomic_write_pickle(resume_path, new_resume)
        row_path = root / "grouped_rows" / path.name
        if row_path.exists():
            with row_path.open("rb") as handle:
                row = pickle.load(handle)
            row["candidate_selection_checkpoints"] = tuple(
                _checkpoint_payload(checkpoint) for checkpoint in new_checkpoints
            )
            row["num_retained_second_stage_candidates"] = int(
                len(updated.evaluated_candidates)
            )
            row["candidate_count_category"] = gdm._candidate_count_category(
                len(updated.evaluated_candidates)
            )
            _atomic_write_pickle(row_path, row)
        refreshed += 1
    return {"refreshed_results": int(refreshed), "skipped_results": int(skipped)}


def summarize_transition_target_sampling_calibration(
    output_dir: Path | str,
) -> Dict[str, Any]:
    root = Path(output_dir)
    rows = load_stage_a_v3_grouped_rows(root)
    metadata_by_id = {
        str(row["example_id"]): row for row in rows.to_dict(orient="records")
    }
    selection_records = []
    for path in sorted(root.glob("selection_checkpoints/*.pkl")):
        with path.open("rb") as handle:
            result = pickle.load(handle)
        grouped_path = root / "grouped_rows" / path.name
        if grouped_path.exists():
            with grouped_path.open("rb") as handle:
                matching = pickle.load(handle)
        else:
            matching = {}
        selection_records.append((matching, result))
    target_records = []
    for path in sorted(root.glob("selected_candidate_targets/*.pkl")):
        with path.open("rb") as handle:
            result = pickle.load(handle)
        grouped_path = root / "grouped_rows" / path.name
        if grouped_path.exists():
            with grouped_path.open("rb") as handle:
                matching = pickle.load(handle)
        else:
            matching = {}
        target_records.append((matching, result))

    overall_selection = _agreement_summary_for_results(selection_records)
    continents = sorted(
        {
            str(metadata.get("continent_name"))
            for metadata, _ in selection_records
            if metadata.get("continent_name") is not None
        }
    )
    per_continent = {
        continent: {
            "candidate_selection": _agreement_summary_for_results(
                [
                    record
                    for record in selection_records
                    if str(record[0].get("continent_name")) == continent
                ]
            ),
            "target_distribution": _target_comparison_summary(
                [
                    record
                    for record in target_records
                    if str(record[0].get("continent_name")) == continent
                ]
            ),
        }
        for continent in continents
    }
    candidate_strata = sorted(
        {
            str(metadata.get("candidate_count_category"))
            for metadata, _ in selection_records
            if metadata.get("candidate_count_category") is not None
        }
    )
    per_candidate_stratum = {
        stratum: _agreement_summary_for_results(
            [
                record
                for record in selection_records
                if str(record[0].get("candidate_count_category")) == stratum
            ]
        )
        for stratum in candidate_strata
    }
    selection_results = [result for _, result in selection_records]
    target_results = [result for _, result in target_records]
    sample_counts = [
        int(result.final_checkpoint_samples) for result in selection_results
    ]
    checkpoint_runtimes: Dict[int, List[float]] = {}
    gap_values: Dict[int, List[float]] = {}
    for result in selection_results:
        for checkpoint in result.checkpoints:
            checkpoint_runtimes.setdefault(int(checkpoint.mc_samples), []).append(
                float(checkpoint.runtime_increment_seconds)
            )
            if checkpoint.best_runner_up_gap:
                active_component = next(
                    (
                        index
                        for index, value in enumerate(
                            checkpoint.best_runner_up_gap
                        )
                        if abs(float(value)) > 1e-15
                    ),
                    0,
                )
                gap_values.setdefault(int(checkpoint.mc_samples), []).append(
                    float(checkpoint.best_runner_up_gap[active_component])
                )
    target_runtimes: Dict[int, List[float]] = {}
    for result in target_results:
        for checkpoint in result.checkpoints:
            target_runtimes.setdefault(
                int(checkpoint.target_distribution_mc_samples), []
            ).append(float(checkpoint.runtime_increment_seconds))
    selection_by_example = {
        str(metadata.get("example_id")): result
        for metadata, result in selection_records
        if metadata.get("example_id") is not None
    }
    target_by_example = {
        str(metadata.get("example_id")): result
        for metadata, result in target_records
        if metadata.get("example_id") is not None
    }
    instability_records = []
    for example_id, metadata in metadata_by_id.items():
        selection_result = selection_by_example.get(str(example_id))
        target_result = target_by_example.get(str(example_id))
        if selection_result is None or target_result is None:
            continue
        candidate_changed = any(
            lower.selected_candidate_identity != higher.selected_candidate_identity
            for lower, higher in zip(
                selection_result.checkpoints,
                selection_result.checkpoints[1:],
            )
        )
        target_comparison = (
            compare_target_distribution_checkpoints(
                target_result.checkpoints[0], target_result.checkpoints[-1]
            )
            if len(target_result.checkpoints) >= 2
            else None
        )
        target_changed = bool(
            target_comparison
            and float(target_comparison["total_variation_distance"]) > 1e-12
        )
        final_checkpoint = (
            selection_result.checkpoints[-1]
            if selection_result.checkpoints
            else None
        )
        final_gap = None
        if final_checkpoint is not None and final_checkpoint.best_runner_up_gap:
            active_index = next(
                (
                    index
                    for index, value in enumerate(
                        final_checkpoint.best_runner_up_gap
                    )
                    if abs(float(value)) > 1e-15
                ),
                0,
            )
            final_gap = float(final_checkpoint.best_runner_up_gap[active_index])
        instability_records.append(
            {
                "example_id": str(example_id),
                "source_example_id": metadata.get("source_stage_a_v2_example_id"),
                "continent_name": metadata.get("continent_name"),
                "candidate_count_category": metadata.get("candidate_count_category"),
                "candidate_selection_changed": bool(candidate_changed),
                "target_distribution_changed": bool(target_changed),
                "same_candidate_target_sampling_changed": bool(
                    target_changed and not candidate_changed
                ),
                "target20_vs_target100_tv": (
                    float(target_comparison["total_variation_distance"])
                    if target_comparison
                    else None
                ),
                "candidate_count": int(
                    metadata.get("num_retained_second_stage_candidates", 0) or 0
                ),
                "num_maximal_partitions": int(
                    metadata.get("num_maximal_partitions", 0) or 0
                ),
                "battle_node_count": int(metadata.get("battle_node_count", 0) or 0),
                "troop_total": int(metadata.get("attacker_troop_total", 0) or 0)
                + int(metadata.get("defender_troop_total", 0) or 0),
                "final_active_score_gap": final_gap,
                "selection_runtime_seconds": (
                    float(final_checkpoint.runtime_cumulative_seconds)
                    if final_checkpoint is not None
                    else 0.0
                ),
            }
        )

    def instability_group(field: str) -> Dict[str, Any]:
        values = sorted({str(item.get(field)) for item in instability_records})
        grouped = {}
        for value in values:
            subset = [
                item for item in instability_records if str(item.get(field)) == value
            ]
            grouped[value] = {
                "states": int(len(subset)),
                "candidate_selection_change_fraction": float(
                    sum(item["candidate_selection_changed"] for item in subset)
                    / len(subset)
                ),
                "same_candidate_target_change_fraction": float(
                    sum(item["same_candidate_target_sampling_changed"] for item in subset)
                    / len(subset)
                ),
                "target20_vs_target100_tv": _numeric_summary(
                    item["target20_vs_target100_tv"] for item in subset
                ),
            }
        return grouped

    changed_records = [
        item for item in instability_records if item["candidate_selection_changed"]
    ]
    stable_records = [
        item for item in instability_records if not item["candidate_selection_changed"]
    ]
    difficulty_relationships = {}
    for field in (
        "candidate_count",
        "num_maximal_partitions",
        "battle_node_count",
        "troop_total",
        "final_active_score_gap",
        "selection_runtime_seconds",
    ):
        difficulty_relationships[field] = {
            "selection_changed": _numeric_summary(
                item[field] for item in changed_records
            ),
            "selection_stable": _numeric_summary(item[field] for item in stable_records),
        }
    outliers = []
    for metadata, result in selection_records:
        if not result.checkpoints:
            continue
        changed = any(
            lower.selected_candidate_identity != higher.selected_candidate_identity
            for lower, higher in zip(result.checkpoints, result.checkpoints[1:])
        )
        if changed or int(result.final_checkpoint_samples) >= 80:
            outliers.append(
                {
                    "example_id": metadata.get("example_id"),
                    "source_example_id": metadata.get("source_stage_a_v2_example_id"),
                    "continent_name": metadata.get("continent_name"),
                    "candidate_count": dict(result.diagnostics).get("candidate_count"),
                    "candidate_selection_changed": bool(changed),
                    "final_samples": int(result.final_checkpoint_samples),
                }
            )
    summary = {
        "calibration_format_version": CALIBRATION_FORMAT_VERSION,
        "target_generation_version": TARGET_GENERATION_VERSION,
        "stage_a_schema_version": STAGE_A_SCHEMA_VERSION,
        "completed_rows": int(len(rows)),
        "continents": (
            {str(key): int(value) for key, value in rows.groupby("continent_name").size().items()}
            if not rows.empty and "continent_name" in rows
            else {}
        ),
        "candidate_selection": overall_selection,
        "candidate_selection_by_continent": per_continent,
        "candidate_selection_by_candidate_count_stratum": per_candidate_stratum,
        "adaptive_stopping_sample_counts": {
            str(value): int(sample_counts.count(value))
            for value in sorted(set(sample_counts))
        },
        "fraction_reaching_mc80": (
            float(sum(value >= 80 for value in sample_counts) / len(sample_counts))
            if sample_counts
            else None
        ),
        "best_runner_up_gap_by_checkpoint": {
            str(key): _numeric_summary(values) for key, values in gap_values.items()
        },
        "runtime_by_candidate_selection_checkpoint": {
            str(key): _numeric_summary(values)
            for key, values in checkpoint_runtimes.items()
        },
        "target_distribution": _target_comparison_summary(target_records),
        "instability_decomposition": {
            "states": int(len(instability_records)),
            "candidate_selection_changed_states": int(len(changed_records)),
            "candidate_selection_stable_states": int(len(stable_records)),
            "same_candidate_target_sampling_changed_states": int(
                sum(
                    item["same_candidate_target_sampling_changed"]
                    for item in instability_records
                )
            ),
            "both_candidate_and_target_changed_states": int(
                sum(
                    item["candidate_selection_changed"]
                    and item["target_distribution_changed"]
                    for item in instability_records
                )
            ),
            "by_continent": instability_group("continent_name"),
            "by_candidate_count_stratum": instability_group(
                "candidate_count_category"
            ),
            "difficulty_relationships": difficulty_relationships,
        },
        "runtime_by_target_sample_count": {
            str(key): _numeric_summary(values)
            for key, values in target_runtimes.items()
        },
        "candidate_selection_decision_analysis": recommend_candidate_selection_configuration(
            selection_results
        ),
        "target_distribution_sample_size_analysis": analyze_target_distribution_sample_sizes(
            target_results
        ),
        "outlier_states": outliers[:50],
        "target_distribution_outlier_states": sorted(
            instability_records,
            key=lambda item: -float(item.get("target20_vs_target100_tv") or 0.0),
        )[:20],
        "generated_at": _utc_now(),
    }
    if root.exists():
        _atomic_write_json(root / "summary.json", summary)
    return summary


def validate_stage_a_v3_grouped_row(row: Mapping[str, Any]) -> List[str]:
    errors = []
    required = (
        "example_id",
        "input_state_id",
        "target_generation_version",
        "stage_a_schema_version",
        "candidate_selection_mode",
        "candidate_selection_checkpoints",
        "candidate_selection_final_samples",
        "candidate_selection_stopped_early",
        "candidate_selection_stopping_reason",
        "selected_candidate_identity",
        "selected_partition_signature",
        "selected_policy_option_indices",
        "target_distribution_mc_samples",
        "full_graph_successor_state_counts",
        "full_graph_successor_state_probabilities",
        "candidate_selection_diagnostics",
        "target_distribution_diagnostics",
        "node_marginals",
        "strategic_summaries",
    )
    for field in required:
        if field not in row:
            errors.append(f"missing field {field}")
    if "mc_samples" in row:
        errors.append("ambiguous top-level mc_samples field is forbidden")
    if row.get("target_generation_version") != TARGET_GENERATION_VERSION:
        errors.append("wrong target_generation_version")
    if row.get("stage_a_schema_version") != STAGE_A_SCHEMA_VERSION:
        errors.append("wrong stage_a_schema_version")
    counts = dict(row.get("full_graph_successor_state_counts", {}) or {})
    probabilities = dict(
        row.get("full_graph_successor_state_probabilities", {}) or {}
    )
    expected = int(row.get("target_distribution_mc_samples", 0) or 0)
    if not counts:
        errors.append("empty successor-state counts")
    if sum(int(value) for value in counts.values()) != expected:
        errors.append("successor-state counts do not sum to target sample count")
    if probabilities and not math.isclose(
        sum(float(value) for value in probabilities.values()),
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        errors.append("successor-state probabilities do not sum to one")
    if set(counts) != set(probabilities):
        errors.append("count and probability supports differ")
    return errors


def validate_transition_target_sampling_calibration(
    output_dir: Path | str,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    root = Path(output_dir)
    errors = []
    for filename in ("config.json", "manifest.json"):
        if not (root / filename).exists():
            errors.append(f"missing {filename}")
    rows = load_stage_a_v3_grouped_rows(root) if root.exists() else pd.DataFrame()
    row_errors = {}
    for row in rows.to_dict(orient="records"):
        current = validate_stage_a_v3_grouped_row(row)
        if current:
            row_errors[str(row.get("example_id"))] = current
    errors.extend(
        f"{example_id}: {message}"
        for example_id, messages in row_errors.items()
        for message in messages
    )
    validation = {
        "valid": not errors,
        "row_count": int(len(rows)),
        "row_error_count": int(len(row_errors)),
        "errors": errors,
        "validated_at": _utc_now(),
    }
    if root.exists():
        _atomic_write_json(root / "validation.json", validation)
    if strict and errors:
        raise ValueError("Stage A v3 calibration validation failed: " + "; ".join(errors[:10]))
    return validation
