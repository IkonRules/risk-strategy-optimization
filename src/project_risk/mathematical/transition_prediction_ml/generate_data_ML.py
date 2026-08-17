from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Literal
import networkx as nx
import warnings
import traceback

import numpy as np
import pandas as pd

from project_risk.game_simulation import Board
from project_risk.mathematical.continent_model import battle_graph_ranking as bgr
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState, NodeState
from project_risk.game_simulation import Players
import logging

try:
    from scipy.stats import skew as _scipy_skew
except ImportError:  # pragma: no cover - exercised only in minimal runtimes
    _scipy_skew = None

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - sequential runs do not need joblib
    Parallel = None
    delayed = None


# ---------------------------------------------------------------------
# Logging (configured by the *caller*, e.g. train_ML)
# ---------------------------------------------------------------------
log_runner = logging.getLogger("risk.runner")
log_rollout = logging.getLogger("risk.rollout")
log_battle_graph = logging.getLogger("risk.battle_graph")
log_partition = logging.getLogger("risk.partition")
log_ranking = logging.getLogger("risk.ranking")
log_query = logging.getLogger("risk.query")
log_sampler = logging.getLogger("risk.sampler")

# ---------------------------------------------------------------------
# Types & configs
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)

# Which *battle outcome* metric to use when ranking partitions
# (these correspond to attributes on PartitionEvaluation objects).
RankingVariable = Literal[
    "battle_expected_attacker_territory_count",
    "battle_expected_attacker_troop_count",
    "battle_expected_attacker_conquest_probability",
]

PolicyOptionSelection = Literal[
    "primary",
    "best_local",
    "best_territories",
    "best_troops",
    "best_conquest",
]


TARGET_GENERATION_VERSION = "transition_targets_v2_corrected_partition_mc5"
STAGE_A_CODE_CONFIGURATION_VERSION = "stage_a_v2_schema_1"


@dataclass
class ExperimentConstraints:
    """
    Constraints / knobs for the state generator.

    This is deliberately generic. You can add more fields as needed
    (e.g. max_nodes, min_degree, etc.) and pass them through to your
    generator function.

    Semantics
    ---------
    - max_attacker_troops_per_node / max_defender_troops_per_node:

        Baseline *global* caps used when constructing the initial state.
        These should never exceed the capacities supported by the
        small-graph libraries, but the *actual* effective cap depends on
        the region (small graph) size and the attacker/defender pattern.

        In particular, the exact finite libraries are now typically built with
        cap 7 per node for regular small-graph patterns. The state generator /
        region-construction logic should enforce this by taking the minimum of:

            - these baseline constraints, and
            - the pattern-based library caps in create_library.

        For example (pseudo-code inside the generator):

            effective_cap_attacker = max_attacker_troops_per_node
            if region_size == 5 and pattern in {(3, 2), (2, 3)}:
                effective_cap_attacker = min(effective_cap_attacker, 3)
            else:
                effective_cap_attacker = min(effective_cap_attacker, 5)

        Similarly for defenders.

    - max_nodes:
        Optional upper bound on the number of territories in the sampled
        battle graph / continent (can be ignored by the generator).

    - continent_name:
        Which continent the experiment focuses on (used to build the
        continent and full graphs).
    """
    # Baseline caps; effective caps are further clamped by library limits
    max_attacker_troops_per_node: int = 10
    max_defender_troops_per_node: int = 10

    max_nodes: Optional[int] = None
    continent_name: str = "North America"



@dataclass
class ExperimentConfig:
    """
    Configuration for a macro-state experiment.

    - territory_ratios:
        target attacker territory shares you want to sample (macro-level).
    - troops_ratios:
        target attacker-vs-defender troops ratios you want to sample.
    - samples_per_combo:
        how many random states per (territory_ratio, troops_ratio).
    - max_partitions:
        passed to rank_battle_graph_partitions (limit fallback sub-partitions).
    - ranking_variable:
        which *battle outcome* metric to rank partitions by
        (see RankingVariable).
    - random_seed:
        for reproducible randomness.
    - constraints:
        generic constraints object passed to the state generator.
    - evaluation_mode:
        "one_wave" or "two_wave" lookahead depth for the partition ranking.

    PATCH:
    - rollout_steps:
        How many times to repeat:
            (partition -> sample -> update global state -> rebuild battle graph)
        when producing the final "end state" labels.

        Defaults to 2 so the ML dataset trains on "one wave, two rollout" end states
        when evaluation_mode="one_wave" (greedy partition selection).
    """
    territory_ratios: Sequence[float]
    troops_ratios: Sequence[float]
    samples_per_combo: int = 10
    max_partitions: int = 40
    ranking_variable: RankingVariable = "battle_expected_attacker_territory_count"
    # How state-set multi-policy rows are collapsed when a downstream consumer
    # still expects a single distribution. "primary" preserves old behavior.
    policy_option_selection: PolicyOptionSelection = "primary"
    random_seed: Optional[int] = None
    constraints: ExperimentConstraints = field(default_factory=ExperimentConstraints)

    # Partition selection policy (greedy vs lookahead)
    evaluation_mode: Literal["one_wave", "two_wave"] = "one_wave"

    # NEW: rollout depth for label generation (default = 2)
    rollout_steps: int = 2

    # Debug/testing only: normal dataset generation records generator failures
    # as skips, while smoke tests can opt into fail-fast tracebacks.
    raise_state_generator_exceptions: bool = False
    include_state_generator_traceback: bool = False


@dataclass(frozen=True)
class TransitionDistributionConfig:
    """
    Additive Stage-A config for two-stage transition-distribution datasets.

    This is intentionally separate from ExperimentConfig so the existing
    deterministic node-row training path keeps its current semantics.
    """
    # Corrected Stage-A V2 production settings. The legacy aliases below are
    # retained only so existing helper-level callers remain source compatible.
    two_stage_mc_samples: int = 5
    two_stage_mc_seed: int = 42
    two_stage_mc_scenarios: Optional[int] = None
    two_stage_rng_seed: Optional[int] = None
    max_partitions: int = 40
    ranking_variable: RankingVariable = "battle_expected_attacker_territory_count"
    combat_libraries_base: Path | str = "small_graph_libraries"
    max_policy_combos_per_partition: Optional[int] = None
    first_stage_value_tolerances: Optional[Tuple[float, ...]] = None
    partition_candidate_selection_mode: str = "maximal_per_partition_utility"
    utility_abs_tolerance: Optional[float] = None
    utility_rel_tolerance: Optional[float] = None
    max_candidates_per_partition: Optional[int] = None
    second_stage_execution_mode: str = "optimized_reuse"
    second_stage_sampling_mode: str = "stable_region_option_scenarios"
    target_generation_version: str = TARGET_GENERATION_VERSION
    code_configuration_version: str = STAGE_A_CODE_CONFIGURATION_VERSION
    evaluation_mode: str = "one_wave"
    rollout_steps: int = 1
    checkpoint_enabled: bool = True
    checkpoint_every_examples: int = 1
    output_chunk_size: int = 100
    resume: bool = True
    calibration_mc_samples: Optional[int] = 20
    calibration_fraction: float = 0.10
    calibration_seed: int = 42020
    min_state_prob: float = 0.0
    max_top_final_states: int = 25
    include_full_graph_successor_signatures: bool = True
    include_node_marginal_rows: bool = True
    profile_second_stage: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        if self.resolved_two_stage_mc_samples < 1:
            raise ValueError("two_stage_mc_samples must be >= 1")
        if self.calibration_mc_samples is not None and int(self.calibration_mc_samples) < 1:
            raise ValueError("calibration_mc_samples must be None or >= 1")
        if not 0.0 <= float(self.calibration_fraction) <= 1.0:
            raise ValueError("calibration_fraction must be between 0 and 1")
        if int(self.output_chunk_size) < 1:
            raise ValueError("output_chunk_size must be >= 1")
        if int(self.checkpoint_every_examples) < 1:
            raise ValueError("checkpoint_every_examples must be >= 1")
        if int(self.rollout_steps) < 1:
            raise ValueError("rollout_steps must be >= 1")
        if self.partition_candidate_selection_mode not in {
            "legacy_global_utility",
            "maximal_per_partition_utility",
        }:
            raise ValueError(
                "Unknown partition_candidate_selection_mode="
                f"{self.partition_candidate_selection_mode!r}"
            )
        if self.second_stage_execution_mode not in {"legacy", "optimized_reuse"}:
            raise ValueError(
                f"Unknown second_stage_execution_mode={self.second_stage_execution_mode!r}"
            )
        if self.second_stage_sampling_mode not in {
            "legacy_sequential_rng",
            "stable_region_option_scenarios",
        }:
            raise ValueError(
                f"Unknown second_stage_sampling_mode={self.second_stage_sampling_mode!r}"
            )

    @property
    def resolved_two_stage_mc_samples(self) -> int:
        if self.two_stage_mc_scenarios is not None:
            return int(self.two_stage_mc_scenarios)
        return int(self.two_stage_mc_samples)

    @property
    def resolved_two_stage_mc_seed(self) -> int:
        if self.two_stage_rng_seed is not None:
            return int(self.two_stage_rng_seed)
        return int(self.two_stage_mc_seed)


@dataclass
class MacroTargets:
    """
    Optional targets for macro-level metrics. Any field that is None is ignored.
    You can extend this with more metrics as needed.
    """
    target_territory_ratio: Optional[float] = None
    target_troops_ratio: Optional[float] = None

    battle_attacker_territory_ratio: Optional[float] = None
    battle_attacker_available_troops_ratio: Optional[float] = None

    full_attacker_territory_ratio: Optional[float] = None
    full_attacker_troops_ratio: Optional[float] = None

    battle_total_territory_count: Optional[int] = None
    battle_total_troops_count: Optional[int] = None


@dataclass
class MacroTolerances:
    """
    Allowed deviations from the targets. You can tune these.
    """
    target_territory_ratio: float = 0.01
    target_troops_ratio: float = 0.01

    battle_attacker_territory_ratio: float = 0.01
    battle_attacker_available_troops_ratio: float = 0.01

    full_attacker_territory_ratio: float = 0.01
    full_attacker_troops_ratio: float = 0.01

    battle_total_territory_count: int = 0
    battle_total_troops_count: int = 0



from dataclasses import dataclass, field, asdict
import time
import math
import numpy as np


# ---------------------------------------------------------------------
# Progress + summary helpers (non-invasive)
# ---------------------------------------------------------------------

@dataclass
class RollingStats:
    """Online stats for a stream of float values (mean/std/min/max + p50 approx via reservoir)."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_v: float = float("inf")
    max_v: float = float("-inf")

    # cheap approximate quantiles by reservoir sample
    reservoir: list = field(default_factory=list)
    reservoir_max: int = 5000

    def add(self, x: Optional[float]) -> None:
        if x is None:
            return
        try:
            x = float(x)
        except Exception:
            return
        if not math.isfinite(x):
            return

        self.n += 1
        # Welford
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

        if x < self.min_v:
            self.min_v = x
        if x > self.max_v:
            self.max_v = x

        # reservoir sample (simple: keep first N, then random replace)
        if len(self.reservoir) < self.reservoir_max:
            self.reservoir.append(x)
        else:
            # replace with probability reservoir_max/n
            j = np.random.randint(0, self.n)
            if j < self.reservoir_max:
                self.reservoir[j] = x

    def std(self) -> float:
        if self.n < 2:
            return float("nan")
        return float(math.sqrt(self.m2 / self.n))

    def quantiles(self, qs: Sequence[float] = (0.05, 0.5, 0.95)) -> Dict[str, float]:
        if not self.reservoir:
            return {f"p{int(q*100):02d}": float("nan") for q in qs}
        arr = np.asarray(self.reservoir, dtype=float)
        out = {}
        for q in qs:
            out[f"p{int(q*100):02d}"] = float(np.quantile(arr, q))
        return out

    def summary(self, name: str) -> Dict[str, Any]:
        q = self.quantiles()
        return {
            f"{name}_n": self.n,
            f"{name}_mean": float(self.mean) if self.n else float("nan"),
            f"{name}_std": self.std(),
            f"{name}_min": float(self.min_v) if self.n else float("nan"),
            f"{name}_max": float(self.max_v) if self.n else float("nan"),
            **{f"{name}_{k}": v for k, v in q.items()},
        }


@dataclass
class RunProgress:
    """
    Lightweight in-memory progress tracker.

    Designed to be used inside run_node_transition_experiment with a few calls
    at natural checkpoints (attempted unit, skip, success, etc.).
    """
    # planned work (upper bound)
    planned_units: int
    planned_states_upper: int  # typically planned_units * 2 (P1 + P2)

    # counters
    units_attempted: int = 0
    units_completed: int = 0  # "completed" = got through partition selection (whether or not rows exist)
    states_success_p1: int = 0
    states_success_p2: int = 0
    states_skipped: int = 0
    rows_emitted: int = 0

    # skip reasons
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    # timing
    t0: float = field(default_factory=time.time)
    last_print_t: float = field(default_factory=time.time)

    # running stats
    full_node_count: RollingStats = field(default_factory=RollingStats)
    battle_node_count: RollingStats = field(default_factory=RollingStats)
    regions_per_partition: RollingStats = field(default_factory=RollingStats)

    coverage_wave1: RollingStats = field(default_factory=RollingStats)
    attacker_holds_final: RollingStats = field(default_factory=RollingStats)
    captured: RollingStats = field(default_factory=RollingStats)
    is_battle_node: RollingStats = field(default_factory=RollingStats)
    final_troops: RollingStats = field(default_factory=RollingStats)

    # optional: store last-seen identifiers for nicer prints
    last_target_pair: Optional[Tuple[float, float]] = None
    last_state_id: Optional[int] = None

    def mark_unit_attempted(self, *, target_territory_ratio: float, target_troops_ratio: float, state_id: int) -> None:
        self.units_attempted += 1
        self.last_target_pair = (float(target_territory_ratio), float(target_troops_ratio))
        self.last_state_id = int(state_id)

    def mark_unit_completed(self) -> None:
        self.units_completed += 1

    def mark_skip(self, reason: str) -> None:
        self.states_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def add_graph_sizes(self, *, full_nodes: int, battle_nodes: int, regions: Optional[int] = None) -> None:
        self.full_node_count.add(full_nodes)
        self.battle_node_count.add(battle_nodes)
        if regions is not None:
            self.regions_per_partition.add(regions)

    def add_rows_stats(
        self,
        node_rows: Sequence[Dict[str, Any]],
        *,
        perspective: str,
    ) -> None:
        # perspective success count
        if perspective == "P1_as_attacker":
            self.states_success_p1 += 1
        elif perspective == "P2_as_attacker":
            self.states_success_p2 += 1

        self.rows_emitted += len(node_rows)

        # sample a small subset to keep overhead low if rows are huge
        # (still accurate enough for a run summary)
        if not node_rows:
            return
        step = max(1, len(node_rows) // 2000)  # sample at most ~2000 rows/state
        for r in node_rows[::step]:
            self.coverage_wave1.add(r.get("coverage_wave1"))
            self.attacker_holds_final.add(r.get("attacker_holds_final"))
            self.captured.add(r.get("captured"))
            self.is_battle_node.add(r.get("is_battle_node"))
            self.final_troops.add(r.get("final_troops"))

    def maybe_print_checkpoint(self, *, every_units: int = 5) -> None:
        """Print a progress line every `every_units` attempted units."""
        if every_units <= 0:
            return
        if self.units_attempted % every_units != 0:
            return
        self.print_checkpoint()

    def print_checkpoint(self) -> None:
        """Log a periodic progress checkpoint (INFO-level)."""
        elapsed = time.time() - self.t0
        units_pct = (100.0 * self.units_attempted / self.planned_units) if self.planned_units > 0 else float("nan")

        states_success_total = self.states_success_p1 + self.states_success_p2
        states_pct = (100.0 * states_success_total / self.planned_states_upper) if self.planned_states_upper > 0 else float("nan")

        units_per_min = (self.units_attempted / elapsed) * 60.0 if elapsed > 0 else float("nan")
        rows_per_sec = (self.rows_emitted / elapsed) if elapsed > 0 else float("nan")

        tt = self.last_target_pair
        tt_str = f"targets=(terr={tt[0]:.2f}, troops={tt[1]:.2f})" if tt else "targets=(?)"

        log_runner.info(
            "run_node_transition_experiment progress attempted_units=%d/%d (%.1f%%) "
            "success_states=%d/%d (%.1f%%) skips=%d rows=%d rate=%.2f units/min %.1f rows/s "
            "elapsed=%.1f min state_id=%s %s",
            self.units_attempted, self.planned_units, units_pct,
            states_success_total, self.planned_states_upper, states_pct,
            self.states_skipped, self.rows_emitted, units_per_min, rows_per_sec,
            elapsed / 60.0, self.last_state_id, tt_str,
        )

        if self.skip_reasons:
            top = sorted(self.skip_reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_str = ", ".join([f"{k}={v}" for k, v in top])
            log_runner.info("run_node_transition_experiment top_skip_reasons %s", top_str)

    def build_summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.t0
        states_success_total = self.states_success_p1 + self.states_success_p2

        summary: Dict[str, Any] = {
            "planned_units": int(self.planned_units),
            "planned_states_upper": int(self.planned_states_upper),
            "units_attempted": int(self.units_attempted),
            "units_completed": int(self.units_completed),
            "states_success_total": int(states_success_total),
            "states_success_p1": int(self.states_success_p1),
            "states_success_p2": int(self.states_success_p2),
            "states_skipped": int(self.states_skipped),
            "rows_emitted": int(self.rows_emitted),
            "elapsed_seconds": float(elapsed),
            "units_per_min": float((self.units_attempted / elapsed) * 60.0) if elapsed > 0 else float("nan"),
            "rows_per_sec": float(self.rows_emitted / elapsed) if elapsed > 0 else float("nan"),
            "skip_reasons": dict(self.skip_reasons),
        }

        # rolling stats
        summary.update(self.full_node_count.summary("full_node_count"))
        summary.update(self.battle_node_count.summary("battle_node_count"))
        summary.update(self.regions_per_partition.summary("regions_per_partition"))

        summary.update(self.coverage_wave1.summary("coverage_wave1"))
        summary.update(self.attacker_holds_final.summary("attacker_holds_final"))
        summary.update(self.captured.summary("captured"))
        summary.update(self.is_battle_node.summary("is_battle_node"))
        summary.update(self.final_troops.summary("final_troops"))

        return summary

    def print_summary(self) -> None:
        """Log an end-of-run summary (INFO-level)."""
        s = self.build_summary()

        log_runner.info("=== NODE TRANSITION EXPERIMENT SUMMARY ===")
        log_runner.info(
            "Planned units=%d Attempted=%d Completed=%d | Planned states (upper)=%d Success total=%d (P1=%d, P2=%d) | Skipped=%d | Rows=%d | Elapsed=%.2f min | Throughput=%.2f units/min %.1f rows/s",
            s["planned_units"], s["units_attempted"], s["units_completed"],
            s["planned_states_upper"], s["states_success_total"], s["states_success_p1"], s["states_success_p2"],
            s["states_skipped"], s["rows_emitted"], s["elapsed_seconds"] / 60.0,
            s["units_per_min"], s["rows_per_sec"],
        )

        if s.get("skip_reasons"):
            for k, v in sorted(s["skip_reasons"].items(), key=lambda kv: kv[1], reverse=True):
                log_runner.info("skip_reason %s=%d", k, v)

        def _fmt_stat(prefix: str) -> str:
            return (
                f"{prefix}: n={s.get(prefix+'_n', 0)} "
                f"mean={s.get(prefix+'_mean', float('nan')):.4f} "
                f"p05={s.get(prefix+'_p05', float('nan')):.4f} "
                f"p50={s.get(prefix+'_p50', float('nan')):.4f} "
                f"p95={s.get(prefix+'_p95', float('nan')):.4f} "
                f"min={s.get(prefix+'_min', float('nan')):.4f} "
                f"max={s.get(prefix+'_max', float('nan')):.4f}"
            )

        log_runner.info("Graph sizes | %s", _fmt_stat("full_node_count"))
        log_runner.info("Graph sizes | %s", _fmt_stat("battle_node_count"))
        log_runner.info("Graph sizes | %s", _fmt_stat("regions_per_partition"))

        log_runner.info("Sampling quality | %s", _fmt_stat("coverage_wave1"))

        log_runner.info("Outcome stats | %s", _fmt_stat("attacker_holds_final"))
        log_runner.info("Outcome stats | %s", _fmt_stat("captured"))
        log_runner.info("Outcome stats | %s", _fmt_stat("is_battle_node"))
        log_runner.info("Outcome stats | %s", _fmt_stat("final_troops"))
        log_runner.info("=== END SUMMARY ===")

def make_run_progress(config: Any) -> RunProgress:
    """
    Create a RunProgress object using only fields that already exist on config.

    Minimal invasiveness: call this once at the top of run_node_transition_experiment.
    """
    planned_units = (
        len(getattr(config, "territory_ratios", []))
        * len(getattr(config, "troops_ratios", []))
        * int(getattr(config, "samples_per_combo", 1))
    )
    planned_states_upper = planned_units * 2  # P1 + P2
    return RunProgress(planned_units=planned_units, planned_states_upper=planned_states_upper)



# ---------------------------------------------------------------------
# Build the object "Full Graph"
# ---------------------------------------------------------------------


def build_full_graph(continent_name: str) -> nx.Graph:
    """
    Build the *full graph* for a continent.

    Definition:
      Full graph = all territories in the continent
                   + all neighbors of those territories
                   (regardless of owner or troop count).

    This represents the static topology relevant for the continent,
    i.e., the maximal region that can influence battles in that continent.
    """
    if continent_name not in Board.continent_territory_dict:
        raise ValueError(f"Unknown continent: {continent_name}")

    continent_territories = Board.continent_territory_dict[continent_name]

    full_node_indices = set(t._index for t in continent_territories)

    # Include all neighbors of continent nodes
    for terr in continent_territories:
        for neigh in terr._neighbors:
            full_node_indices.add(neigh._index)

    # Build induced graph over full_node_indices
    G = nx.Graph()
    for idx in full_node_indices:
        G.add_node(idx)

    for idx in full_node_indices:
        terr = Board.node_to_territory_dict[idx]
        for neigh in terr._neighbors:
            if neigh._index in full_node_indices:
                G.add_edge(idx, neigh._index)

    return G



# ---------------------------------------------------------------------
# Macro-variable computation helpers
# ---------------------------------------------------------------------


def compute_territory_ratio(
    global_state: GlobalState,
    node_indices: Sequence[int],
) -> float:
    """
    Compute the *attacker territory share* over a given set of nodes.

    territory_ratio =
        (# attacker-owned territories with > 0 troops on these nodes)
        / (total # nodes in node_indices)

    Typical use:
      - node_indices = battle graph nodes
        => "battle_realized_attacker_territory_ratio"

    But the function itself is generic: it just uses the provided nodes.
    """
    node_indices = list(node_indices)
    if not node_indices:
        return np.nan

    attacker_territory_count = 0
    for idx in node_indices:
        node = global_state.nodes[idx]
        if node.owner == "A" and node.troops > 0:
            attacker_territory_count += 1

    return attacker_territory_count / len(node_indices)



def _battle_graph_nodes(battle_graph) -> List[int]:
    """
    Extract a list of node indices from a networkx-like battle_graph.
    """
    try:
        nodes_iter = battle_graph.nodes()
    except TypeError:
        nodes_iter = battle_graph.nodes
    return list(nodes_iter)


def _graph_nodes(graph) -> Tuple[int, ...]:
    try:
        nodes_iter = graph.nodes()
    except TypeError:
        nodes_iter = graph.nodes
    return tuple(sorted(int(x) for x in nodes_iter))


def normalize_state_signature(signature) -> Tuple[Tuple[int, str, int], ...]:
    """
    Normalize a state signature into sorted primitive tuples:
        ((node_id, owner, troops), ...)
    """
    if signature is None:
        return tuple()
    out = []
    if isinstance(signature, Mapping):
        items = signature.items()
        for node_id, val in items:
            if isinstance(val, Mapping):
                owner = val.get("owner")
                troops = val.get("troops")
            else:
                owner, troops = val
            out.append((int(node_id), str(owner), max(1, int(troops))))
    else:
        for item in signature:
            node_id, owner, troops = item
            out.append((int(node_id), str(owner), max(1, int(troops))))
    return tuple(sorted(out, key=lambda x: x[0]))


def canonical_graph_signature(graph) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    """Return a stable node/edge signature for a networkx-like graph."""
    nodes = _graph_nodes(graph)
    try:
        edges_iter = graph.edges()
    except TypeError:
        edges_iter = graph.edges
    edges = tuple(
        sorted(
            {
                tuple(sorted((int(u), int(v))))
                for u, v in edges_iter
            }
        )
    )
    return nodes, edges


def _stable_hash_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return {"__float__": "nan"}
        if np.isposinf(value):
            return {"__float__": "inf"}
        if np.isneginf(value):
            return {"__float__": "-inf"}
        return float(value)
    if isinstance(value, np.generic):
        return _stable_hash_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        entries = [
            (_stable_hash_value(key), _stable_hash_value(item))
            for key, item in value.items()
        ]
        entries.sort(
            key=lambda pair: json.dumps(
                pair[0], sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
        )
        return {"__mapping__": entries}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_stable_hash_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return items
    return repr(value)


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        _stable_hash_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def transition_distribution_config_fingerprint(
    config: TransitionDistributionConfig,
) -> str:
    """Fingerprint semantic and output-layout settings, excluding resume controls."""
    payload = asdict(config)
    for key in ("resume", "checkpoint_enabled", "checkpoint_every_examples"):
        payload.pop(key, None)
    payload["resolved_two_stage_mc_samples"] = config.resolved_two_stage_mc_samples
    payload["resolved_two_stage_mc_seed"] = config.resolved_two_stage_mc_seed
    return _stable_digest(payload)


def transition_distribution_target_fingerprint(
    config: TransitionDistributionConfig,
) -> str:
    """Fingerprint only settings capable of changing one generated target."""
    payload = asdict(config)
    for key in (
        "resume",
        "checkpoint_enabled",
        "checkpoint_every_examples",
        "output_chunk_size",
        "calibration_mc_samples",
        "calibration_fraction",
        "calibration_seed",
    ):
        payload.pop(key, None)
    payload["resolved_two_stage_mc_samples"] = config.resolved_two_stage_mc_samples
    payload["resolved_two_stage_mc_seed"] = config.resolved_two_stage_mc_seed
    return _stable_digest(payload)


def canonical_transition_example_id(
    *,
    continent_name: str,
    perspective: str,
    initial_full_graph_signature,
    battle_graph_signature,
    commitment_signature,
    target_generation_version: str,
    two_stage_mc_seed: int,
    generation_config_fingerprint: Optional[str] = None,
) -> str:
    payload = {
        "continent_name": str(continent_name),
        "perspective": str(perspective),
        "initial_full_graph_signature": normalize_state_signature(
            initial_full_graph_signature
        ),
        "battle_graph_signature": battle_graph_signature,
        "commitment_signature": commitment_signature,
        "target_generation_version": str(target_generation_version),
        "two_stage_mc_seed": int(two_stage_mc_seed),
        "generation_config_fingerprint": generation_config_fingerprint,
    }
    return "transition_example_" + _stable_digest(payload)


def derive_transition_target_seed(
    *,
    base_seed: int,
    example_id: str,
    mc_samples: int,
    target_generation_version: str,
) -> int:
    digest = _stable_digest(
        {
            "base_seed": int(base_seed),
            "example_id": str(example_id),
            "mc_samples": int(mc_samples),
            "target_generation_version": str(target_generation_version),
        }
    )
    return int(digest[:16], 16) % (2 ** 32)


def signature_to_node_state_map(signature) -> Dict[int, Tuple[str, int]]:
    return {
        int(node_id): (str(owner), max(1, int(troops)))
        for node_id, owner, troops in normalize_state_signature(signature)
    }


def node_state_map_to_signature(node_state_map) -> Tuple[Tuple[int, str, int], ...]:
    return normalize_state_signature(node_state_map)


def lift_battle_signature_to_full_graph_signature(
    *,
    battle_signature,
    initial_global_state: GlobalState,
    full_graph,
) -> Tuple[Tuple[int, str, int], ...]:
    """
    Lift a battle-node outcome signature to the full continent-local graph.

    Battle nodes use the sampled outcome. Non-battle full-graph nodes are
    preserved from the initial state. Node 0 is included only if full_graph
    contains it.
    """
    battle_map = signature_to_node_state_map(battle_signature)
    out: Dict[int, Tuple[str, int]] = {}
    for node_id in _graph_nodes(full_graph):
        if node_id in battle_map:
            owner, troops = battle_map[node_id]
        else:
            node = initial_global_state.nodes[int(node_id)]
            owner, troops = str(node.owner), int(node.troops)
        out[int(node_id)] = (str(owner), max(1, int(troops)))
    return node_state_map_to_signature(out)


def build_full_graph_successor_distribution_from_mc_counts(
    *,
    mc_final_state_counts: Mapping[Any, int],
    initial_global_state: GlobalState,
    full_graph,
) -> Dict[Tuple[Tuple[int, str, int], ...], int]:
    successor_counts: Dict[Tuple[Tuple[int, str, int], ...], int] = {}
    for battle_signature, count in (mc_final_state_counts or {}).items():
        n = int(count)
        if n <= 0:
            continue
        lifted = lift_battle_signature_to_full_graph_signature(
            battle_signature=battle_signature,
            initial_global_state=initial_global_state,
            full_graph=full_graph,
        )
        successor_counts[lifted] = int(successor_counts.get(lifted, 0) + n)
    return successor_counts


def derive_node_marginals_from_successor_distribution(
    *,
    successor_state_counts: Mapping[Any, int],
    full_graph,
    initial_global_state: Optional[GlobalState] = None,
) -> Dict[int, Dict[str, float]]:
    nodes = _graph_nodes(full_graph)
    total = float(sum(float(v) for v in (successor_state_counts or {}).values()))
    if total <= 0:
        return {
            int(node_id): {
                "p_attacker_final": 0.0,
                "p_defender_final": 0.0,
                "expected_troops": 0.0,
                "expected_troops_if_attacker": 0.0,
                "expected_troops_if_defender": 0.0,
                "p_changed_owner": 0.0,
            }
            for node_id in nodes
        }

    acc: Dict[int, Dict[str, float]] = {
        int(node_id): {
            "a_count": 0.0,
            "d_count": 0.0,
            "troops_sum": 0.0,
            "a_troops_sum": 0.0,
            "d_troops_sum": 0.0,
            "changed_count": 0.0,
        }
        for node_id in nodes
    }
    node_set = set(nodes)
    for signature, count in successor_state_counts.items():
        n = float(count)
        state_map = signature_to_node_state_map(signature)
        for node_id in node_set:
            if node_id not in state_map:
                continue
            owner, troops = state_map[node_id]
            troops_f = float(troops)
            row = acc[int(node_id)]
            row["troops_sum"] += n * troops_f
            if owner == "A":
                row["a_count"] += n
                row["a_troops_sum"] += n * troops_f
            elif owner == "D":
                row["d_count"] += n
                row["d_troops_sum"] += n * troops_f
            if initial_global_state is not None:
                if str(initial_global_state.nodes[int(node_id)].owner) != owner:
                    row["changed_count"] += n

    out: Dict[int, Dict[str, float]] = {}
    denom = float(total)
    for node_id in nodes:
        row = acc[int(node_id)]
        a_count = row["a_count"]
        d_count = row["d_count"]
        out[int(node_id)] = {
            "p_attacker_final": float(a_count / denom),
            "p_defender_final": float(d_count / denom),
            "expected_troops": float(row["troops_sum"] / denom),
            "expected_troops_if_attacker": float(row["a_troops_sum"] / a_count) if a_count > 0 else 0.0,
            "expected_troops_if_defender": float(row["d_troops_sum"] / d_count) if d_count > 0 else 0.0,
            "p_changed_owner": float(row["changed_count"] / denom) if initial_global_state is not None else 0.0,
        }
    return out


def top_k_successor_states_from_counts(
    successor_state_counts: Mapping[Any, int],
    *,
    k: int = 25,
) -> List[Dict[str, Any]]:
    total = int(sum(int(v) for v in (successor_state_counts or {}).values()))
    if total <= 0 or int(k) <= 0:
        return []
    normalized = [
        (normalize_state_signature(sig), int(count))
        for sig, count in successor_state_counts.items()
        if int(count) > 0
    ]
    normalized.sort(key=lambda item: (-item[1], item[0]))
    return [
        {
            "signature": sig,
            "count": int(count),
            "prob_hat": float(count) / float(total),
        }
        for sig, count in normalized[: int(k)]
    ]


def _to_pickle_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(_to_pickle_safe(x) for x in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _to_pickle_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_to_pickle_safe(x) for x in value)
    return value


def compute_attacker_troop_distribution(
    global_state: GlobalState,
    node_indices: Sequence[int],
) -> np.ndarray:
    """
    Returns an array of troop counts on ATTACKER-owned nodes
    (owner == 'A') in the given node set.

    Empty array if attacker owns nothing on these nodes.

    Typical use:
      - node_indices = battle graph nodes
        => attacker troop distribution on the battle graph.
    """
    troops: List[float] = []
    for idx in node_indices:
        node = global_state.nodes[idx]
        if node.owner == "A" and node.troops > 0:
            troops.append(node.troops)
    return np.array(troops, dtype=float)


def compute_troops_cv(troops_array: np.ndarray) -> float:
    """
    Coefficient of Variation (CV) of a troop distribution.

    CV = std / mean

    Intended use:
      - pass attacker troop counts across some node set.
    Returns np.nan if undefined (len < 2 or mean <= 0).
    """
    if len(troops_array) < 2:
        return np.nan
    mean = troops_array.mean()
    if mean <= 0:
        return np.nan
    return float(troops_array.std(ddof=0) / mean)


def compute_troops_gini(troops_array: np.ndarray) -> float:
    """
    Computes the Gini coefficient for a troop distribution.

    G =
      sum_i sum_j |t_i - t_j| / (2 * n^2 * mean)

    Intended use:
      - pass attacker troop counts across some node set.

    Returns np.nan if undefined (n == 0 or mean <= 0).
    """
    n = len(troops_array)
    if n == 0:
        return np.nan
    mean = troops_array.mean()
    if mean <= 0:
        return np.nan

    diffs = np.abs(troops_array.reshape(-1, 1) - troops_array.reshape(1, -1))
    gini = diffs.sum() / (2 * n * n * mean)
    return float(gini)


def compute_troops_skew(troops_array: np.ndarray) -> float:
    """
    Skewness of a troop distribution.

    Intended use:
      - pass attacker troop counts across some node set.

    Returns np.nan if len < 3.
    """
    if len(troops_array) < 3:
        return np.nan
    arr = np.asarray(troops_array, dtype=np.float64)
    if _scipy_skew is not None:
        return float(_scipy_skew(arr))

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std <= 0.0:
        return 0.0
    centered = (arr - mean) / std
    return float(np.mean(centered ** 3))



def compute_full_graph_metrics(full_graph, players):
    """
    Full-graph metrics for the current state (before any battle filtering).

    Returns counts, ratios, troop distribution, and topology metrics with
    explicit 'full_' and 'realized_' naming.

    Key semantics:
      - 'full_*'   -> defined on the full graph (continent + neighbors)
      - 'realized' -> refers to the current pre-battle state (not expected)

    IMPORTANT:
      - full_realized_attacker_territory_ratio = A_territories / total_territories
      - full_realized_attacker_troops_ratio    = A_troops / total_troops
    """
    player1 = players[0]

    full_nodes = list(full_graph.nodes())
    full_node_count = len(full_nodes)

    attacker_territory_count = 0
    defender_territory_count = 0

    attacker_troop_count = 0
    defender_troop_count = 0

    attacker_troops_list = []

    for idx in full_nodes:
        terr = Board.node_to_territory_dict[idx]
        owner = terr._owner
        troops = terr._troops

        if owner is player1:
            attacker_territory_count += 1
            attacker_troop_count += troops
            attacker_troops_list.append(troops)
        else:
            defender_territory_count += 1
            defender_troop_count += troops

    # Total troops on full graph
    full_troop_count = attacker_troop_count + defender_troop_count

    # Territory ratio: attacker share of total territories
    if full_node_count > 0:
        full_realized_attacker_territory_ratio = (
            attacker_territory_count / full_node_count
        )
    else:
        full_realized_attacker_territory_ratio = np.nan

    # Troop ratio: attacker share of total troops (NOT A/D)
    if full_troop_count > 0:
        full_realized_attacker_troops_ratio = (
            attacker_troop_count / full_troop_count
        )
    else:
        full_realized_attacker_troops_ratio = np.nan

    # Troop distribution metrics (attacker-only on full graph)
    attacker_troops_arr = np.array(attacker_troops_list, dtype=float)
    full_realized_attacker_troops_distribution_cv = compute_troops_cv(
        attacker_troops_arr
    )
    full_realized_attacker_troops_distribution_gini = compute_troops_gini(
        attacker_troops_arr
    )

    # Topology metrics on full graph
    degrees = [deg for _, deg in full_graph.degree()]
    full_realized_topology_degree_mean = float(np.mean(degrees)) if degrees else np.nan
    full_realized_topology_degree_variance = float(np.var(degrees)) if degrees else np.nan

    try:
        full_realized_topology_diameter = nx.diameter(full_graph)
    except Exception:
        full_realized_topology_diameter = np.nan

    full_realized_topology_component_count = nx.number_connected_components(full_graph)

    metrics = {
        # Node/edge counts
        "full_realized_total_territory_count": full_node_count,
        "full_realized_topology_edge_count": full_graph.number_of_edges(),

        # Territory counts
        "full_realized_attacker_territory_count": attacker_territory_count,
        "full_realized_defender_territory_count": defender_territory_count,

        # Territory ratios
        "full_realized_attacker_territory_ratio": full_realized_attacker_territory_ratio,

        # Troop counts
        "full_realized_attacker_troops_count": attacker_troop_count,
        "full_realized_defender_troops_count": defender_troop_count,
        "full_realized_total_troops_count": full_troop_count,

        # Troop ratio (attacker share of total troops)
        "full_realized_attacker_troops_ratio": full_realized_attacker_troops_ratio,

        # Attacker troop distribution on full graph
        "full_realized_attacker_troops_distribution_cv": (
            full_realized_attacker_troops_distribution_cv
        ),
        "full_realized_attacker_troops_distribution_gini": (
            full_realized_attacker_troops_distribution_gini
        ),

        # Topology
        "full_realized_topology_degree_mean": full_realized_topology_degree_mean,
        "full_realized_topology_degree_variance": full_realized_topology_degree_variance,
        "full_realized_topology_diameter": full_realized_topology_diameter,
        "full_realized_topology_component_count": full_realized_topology_component_count,
    }

    return metrics



def compute_effectiveness_metrics(full_graph, battle_graph, players):
    """
    Compare full graph to battle graph:
      - How much of the deployment is actually used in the current battle?
      - How far away are reserves (nodes not in the battle graph) on the full graph?

    Returns metrics with explicit naming:
      - battle_realized_effectiveness_* :
            "how much of full is active in battle?"
      - full_realized_effectiveness_reserve_distance_* :
            distances from reserve nodes to battle nodes, measured on the full graph.
    """
    player1 = players[0]

    full_nodes = set(full_graph.nodes())
    battle_nodes = set(battle_graph.nodes())
    reserve_nodes = full_nodes - battle_nodes  # nodes not in battle graph

    # Basic node effectiveness: fraction of full nodes that are in the battle graph
    if len(full_nodes) > 0:
        battle_realized_effectiveness_node_ratio = len(battle_nodes) / len(full_nodes)
    else:
        battle_realized_effectiveness_node_ratio = np.nan

    # Troop effectiveness: fraction of attacker troops on full graph that are in battle graph
    full_attacker_troop_count = 0
    battle_attacker_troop_count = 0

    for idx in full_nodes:
        terr = Board.node_to_territory_dict[idx]
        if terr._owner is player1:
            full_attacker_troop_count += terr._troops

    for idx in battle_nodes:
        terr = Board.node_to_territory_dict[idx]
        if terr._owner is player1:
            battle_attacker_troop_count += terr._troops

    battle_realized_effectiveness_attacker_troops_ratio = (
        battle_attacker_troop_count / full_attacker_troop_count
        if full_attacker_troop_count > 0
        else np.nan
    )

    # Reserve distance metrics: how far reserves are from the active battle region
    if reserve_nodes and battle_nodes:
        # dist_to_battle[n] = min_{b in battle_nodes} dist(n, b)
        dist_to_battle = dict(nx.multi_source_dijkstra_path_length(full_graph, sources=battle_nodes))

        distances = []
        for r in reserve_nodes:
            d = dist_to_battle.get(r, np.inf)  # inf if disconnected
            distances.append(d)

        distances = np.array(distances, dtype=float)
        finite = distances[np.isfinite(distances)]

        if finite.size > 0:
            mean = float(np.mean(finite))
            full_realized_effectiveness_reserve_distance_mean = mean
            full_realized_effectiveness_reserve_distance_min = float(np.min(finite))
            full_realized_effectiveness_reserve_distance_max = float(np.max(finite))
            full_realized_effectiveness_reserve_distance_cv = (
                float(np.std(finite) / mean) if mean > 0 else np.nan
            )
        else:
            # all reserves disconnected from battle
            full_realized_effectiveness_reserve_distance_mean = np.inf
            full_realized_effectiveness_reserve_distance_min = np.inf
            full_realized_effectiveness_reserve_distance_max = np.inf
            full_realized_effectiveness_reserve_distance_cv = np.nan
    else:
        full_realized_effectiveness_reserve_distance_mean = np.nan
        full_realized_effectiveness_reserve_distance_min = np.nan
        full_realized_effectiveness_reserve_distance_max = np.nan
        full_realized_effectiveness_reserve_distance_cv = np.nan

    return {
        "battle_realized_effectiveness_node_ratio": battle_realized_effectiveness_node_ratio,
        "battle_realized_effectiveness_attacker_troops_ratio": battle_realized_effectiveness_attacker_troops_ratio,
        "full_realized_effectiveness_reserve_distance_mean": full_realized_effectiveness_reserve_distance_mean,
        "full_realized_effectiveness_reserve_distance_min": full_realized_effectiveness_reserve_distance_min,
        "full_realized_effectiveness_reserve_distance_max": full_realized_effectiveness_reserve_distance_max,
        "full_realized_effectiveness_reserve_distance_cv": full_realized_effectiveness_reserve_distance_cv,
    }


def summarize_region_policy_options(best_eval: Any, policy_option_selection: Any) -> Dict[str, Any]:
    """
    Compact row-friendly diagnostics from PartitionEvaluation.region_policy_options.

    These columns are diagnostic only; existing ML consumers can ignore them.
    """
    raw = getattr(best_eval, "region_policy_options", ()) if best_eval is not None else ()
    if raw is None:
        raw = ()

    normalized = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "region_nodes": tuple(item.get("region_nodes", ()) or ()),
                "attacker_nodes": tuple(item.get("attacker_nodes", ()) or ()),
                "defender_nodes": tuple(item.get("defender_nodes", ()) or ()),
                "pattern": item.get("pattern"),
                "row_label": item.get("row_label"),
                "policy_option_count": item.get("policy_option_count"),
                "selected_policy_option_index": item.get("selected_policy_option_index"),
                "selected_policy_option_selection": item.get("selected_policy_option_selection"),
                "selected_policy_option_split_metadata": item.get("selected_policy_option_split_metadata"),
                "policy_option_summaries": tuple(item.get("policy_option_summaries", ()) or ()),
            }
        )

    counts = []
    selected_indices = []
    selected_selections = []
    row_labels = []
    for item in normalized:
        try:
            counts.append(int(item.get("policy_option_count") or 0))
        except Exception:
            counts.append(0)
        idx = item.get("selected_policy_option_index")
        if idx is not None:
            try:
                selected_indices.append(int(idx))
            except Exception:
                selected_indices.append(idx)
        selected_selections.append(item.get("selected_policy_option_selection"))
        row_labels.append(item.get("row_label"))

    max_count = max(counts) if counts else 0
    return {
        "policy_option_selection": str(policy_option_selection),
        "best_partition_region_policy_options": tuple(normalized),
        "max_policy_option_count": int(max_count),
        "num_multi_option_regions": int(sum(1 for c in counts if c > 1)),
        "selected_policy_option_indices": tuple(selected_indices),
        "selected_policy_option_selections": tuple(selected_selections),
        "policy_option_row_labels": tuple(row_labels),
    }




def collect_node_outcome_rows_for_state(
    state_id: int,
    players: Sequence["Players.Player"],
    battle_graph,
    full_graph,
    global_state: GlobalState,
    partition_regions: Optional[Sequence[Dict[str, Any]]],
    macro_features: Dict[str, Any],
    continent_name: str,
    combat_libraries_base: Path,
    num_scenarios: int = 50,
    min_state_prob: float = 0.0,
    max_end_states_per_region: Optional[int] = None,
    # -----------------------------
    # PATCH: multi-step rollout
    # -----------------------------
    rollout_steps: int = 1,
    # If partition_regions is None, we can choose the partition on step 1 too.
    max_partitions: int = 40,
    ranking_variable: RankingVariable = "battle_expected_attacker_territory_count",
    policy_option_selection: PolicyOptionSelection = "primary",
    evaluation_mode: Literal["one_wave", "two_wave"] = "one_wave",
    # For "two_wave" partition-ranking lookahead (optional)
    num_scenarios_two_wave: int = 50,
    min_state_prob_two_wave: float = 0.0,
    max_end_states_per_region_two_wave: Optional[int] = None,
    # -----------------------------
    # PATCH: debug controls
    # -----------------------------
    debug: bool = True,
    debug_battle_graph_limit: int = 100,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if rollout_steps < 1:
        return rows

    try:
        full_nodes_iter = full_graph.nodes()
    except TypeError:
        full_nodes_iter = full_graph.nodes
    full_nodes = list(full_nodes_iter)

    initial_battle_nodes = set(_battle_graph_nodes(battle_graph))
    initial_nodes = global_state.nodes

    # -----------------------------
    # Helpers
    # -----------------------------
    def _partition_summary(part_regs: Optional[Sequence[Dict[str, Any]]]) -> str:
        if not part_regs:
            return "None/empty"
        try:
            reg_lens = [len(r.get("region_nodes", ())) for r in part_regs]
            patterns = [r.get("pattern", None) for r in part_regs]
            return f"regions={len(part_regs)} lens={reg_lens} patterns={patterns}"
        except Exception:
            return f"regions={len(part_regs)} (summary failed)"

    def _battle_graph_edges(curr_battle_graph):
        try:
            return list(curr_battle_graph.edges())
        except TypeError:
            return list(curr_battle_graph.edges)

    # -----------------------------
    # PATCH: "has battle" == has at least one edge
    # -----------------------------
    def _battle_graph_has_battle(
        curr_battle_graph,
        *,
        debug: bool = True,
        debug_tag: str = "",
    ) -> bool:
        """Return True iff the current battle graph still has at least one A–D edge.

        Authoritative stop condition: *edges > 0* (node count can be 0/1 in edge cases).
        Debug diagnostics go through logging, not print().
        """
        try:
            n_edges = int(curr_battle_graph.number_of_edges())
            has_battle = n_edges > 0

            if debug:
                try:
                    n_nodes = int(curr_battle_graph.number_of_nodes())
                except Exception:
                    n_nodes = -1
                tag = f" {debug_tag}" if debug_tag else ""
                log_battle_graph.debug(
                    "[battle_graph_has_battle]%s nodes=%s edges=%s has_battle=%d",
                    tag, n_nodes, n_edges, int(has_battle),
                    extra={"state_id": state_id},
                )

            return has_battle

        except Exception:
            # Fallback for graph-like objects without number_of_edges()
            try:
                edges_iter = curr_battle_graph.edges()
            except TypeError:
                edges_iter = curr_battle_graph.edges

            for _ in edges_iter:
                if debug:
                    tag = f" {debug_tag}" if debug_tag else ""
                    log_battle_graph.debug(
                        "[battle_graph_has_battle]%s edges>=1 (iter fallback) has_battle=1",
                        tag,
                        extra={"state_id": state_id},
                    )
                return True

            if debug:
                tag = f" {debug_tag}" if debug_tag else ""
                log_battle_graph.debug(
                    "[battle_graph_has_battle]%s edges=0 (iter fallback) has_battle=0",
                    tag,
                    extra={"state_id": state_id},
                )
            return False

    def _fallback_partition_edge_pairs(curr_state: GlobalState, curr_battle_graph, *, max_regions: int = 50):
        """
        NEW fallback: build a partition out of many tiny 2-node regions (one per battle edge),
        which are typically (1,1) patterns and have libraries.
        This avoids the STAR_ONLY (1,6) / (1,k) traps caused by "all battle nodes in one region".
        """
        edges = _battle_graph_edges(curr_battle_graph)
        if not edges:
            return None

        part: List[Dict[str, Any]] = []
        used = 0

        for u, v in edges:
            if used >= int(max_regions):
                break

            ou = getattr(curr_state.nodes[u], "owner", None)
            ov = getattr(curr_state.nodes[v], "owner", None)
            if ou not in ("A", "D") or ov not in ("A", "D"):
                continue
            if ou == ov:
                continue

            region_nodes = (int(u), int(v))
            attacker_nodes = tuple(i for i in region_nodes if curr_state.nodes[i].owner == "A")
            defender_nodes = tuple(i for i in region_nodes if curr_state.nodes[i].owner == "D")
            pat = (len(attacker_nodes), len(defender_nodes))

            if pat[0] == 0 or pat[1] == 0:
                continue

            part.append(
                {
                    "region_nodes": region_nodes,
                    "attacker_nodes": attacker_nodes,
                    "defender_nodes": defender_nodes,
                    "pattern": pat,
                }
            )
            used += 1

        return part or None

    def _fallback_partition_single_region(curr_state: GlobalState, curr_battle_graph):
        """
        OLD fallback (kept but demoted): single region containing ALL battle nodes.
        This is often STAR_ONLY-invalid and is now only used if edge-pair fallback fails.
        """
        bnodes = list(_battle_graph_nodes(curr_battle_graph))
        if not bnodes:
            return None
        attacker_nodes = tuple(i for i in bnodes if curr_state.nodes[i].owner == "A")
        defender_nodes = tuple(i for i in bnodes if curr_state.nodes[i].owner == "D")
        pat = (len(attacker_nodes), len(defender_nodes))
        return [{
            "region_nodes": tuple(bnodes),
            "attacker_nodes": attacker_nodes,
            "defender_nodes": defender_nodes,
            "pattern": pat,
        }]

    def _rank_partition_for_current_board_and_players(curr_battle_graph):
        if evaluation_mode not in ("one_wave", "two_wave"):
            raise ValueError(
                f"Unknown evaluation_mode={evaluation_mode!r}. "
                "Expected 'one_wave' or 'two_wave'."
            )

        if evaluation_mode == "two_wave":
            bundle = bgr.rank_battle_graph_partitions_with_lookahead(
                players=players,
                continent_name=continent_name,
                combat_libraries_base=combat_libraries_base,
                max_partitions_wave1=max_partitions,
                ranking_variable=ranking_variable,
                max_partitions_wave2=max_partitions,
                num_scenarios=num_scenarios_two_wave,
                policy_option_selection=policy_option_selection,
                min_state_prob=min_state_prob_two_wave,
                max_end_states_per_region=max_end_states_per_region_two_wave,
            )
            base_result = bundle.get("base_result", {}) or {}
            lookahead_eval = bundle.get("lookahead_result", None)
            partitions_full = base_result.get("partitions_full", []) or []
            best_eval = lookahead_eval if lookahead_eval is not None else base_result.get("best_evaluation")

            if best_eval is None:
                return None

            if not partitions_full:
                best_partition = base_result.get("best_partition", None)
                working_partitions = base_result.get("working_partitions", []) or []
                if best_partition is not None:
                    partitions_full = [best_partition]
                elif working_partitions:
                    partitions_full = [working_partitions[0]]

            if not partitions_full:
                return None

            part_idx = getattr(best_eval, "partition_index", 0)
            if part_idx < 0 or part_idx >= len(partitions_full):
                part_idx = 0
            return partitions_full[part_idx]

        # one_wave ranking
        try:
            rank_result = bgr.rank_battle_graph_partitions(
                players=players,
                battle_graph=curr_battle_graph,
                combat_libraries_base=combat_libraries_base,
                max_partitions=max_partitions,
                ranking_variable=ranking_variable,
                lookahead_depth=0,
                policy_option_selection=policy_option_selection,
            )
        except Exception:
            return None

        best_eval = rank_result.get("best_evaluation", None)
        partitions_full = rank_result.get("partitions_full", []) or []

        if best_eval is None:
            return None

        if not partitions_full:
            best_partition = rank_result.get("best_partition", None)
            working_partitions = rank_result.get("working_partitions", []) or []
            if best_partition is not None:
                partitions_full = [best_partition]
            elif working_partitions:
                partitions_full = [working_partitions[0]]

        if not partitions_full:
            return None

        part_idx = getattr(best_eval, "partition_index", 0)
        if part_idx < 0 or part_idx >= len(partitions_full):
            part_idx = 0
        return partitions_full[part_idx]

    # -----------------------------
    # PATCH: battle graph rebuild with debug tagging
    # -----------------------------
    def _rebuild_battle_graph_from_board(*, dbg_tag: str = ""):
        """Rebuild the battle graph from the current Board state.

        IMPORTANT: All diagnostics go through logging. We do *not* re-run the builder
        with debug=True because that would bypass the centralized logging system.
        """
        if continent_name is None:
            return battle_graph

        G = agop.build_continent_battle_graph(
            continent_name,
            players,
            debug=False,
            debug_tag=dbg_tag,
            debug_limit=int(debug_battle_graph_limit),
        )

        try:
            n_edges = int(G.number_of_edges())
        except Exception:
            n_edges = -1

        if debug and n_edges == 0:
            log_battle_graph.debug(
                "[battle_graph_rebuild] %s edges=0",
                dbg_tag,
                extra={"state_id": state_id},
            )

        return G

    # -----------------------------
    # Board restore guard
    # -----------------------------
    try:
        try:
            apply_global_state_to_board(global_state, players)
        except Exception:
            pass

        # ------------------------------------------------------
        # Step 1: choose partition (if not provided) and sample N scenarios
        # ------------------------------------------------------
        if partition_regions is None:
            chosen = _rank_partition_for_current_board_and_players(battle_graph)
            if chosen is None:
                chosen = _fallback_partition_edge_pairs(global_state, battle_graph, max_regions=max_partitions)
            if chosen is None:
                chosen = _fallback_partition_single_region(global_state, battle_graph)
            if chosen is None:
                return rows
            partition_regions_step1 = chosen
        else:
            partition_regions_step1 = partition_regions

        scenarios, coverage_wave1 = bgr.sample_wave1_microstates_for_partition(
            players=players,
            continent_name=continent_name,
            partition_regions=partition_regions_step1,
            global_state=global_state,
            battle_graph=battle_graph,
            combat_libraries_base=combat_libraries_base,
            num_scenarios=num_scenarios,
            min_state_prob=min_state_prob,
            max_end_states_per_region=max_end_states_per_region,
            policy_option_selection=policy_option_selection,
            ranking_variable=ranking_variable,
            debug=bool(debug),
        )

        # NEW: if sampling returned empty, retry with edge-pair fallback before no-op
        if not scenarios and partition_regions is None:
            alt = _fallback_partition_edge_pairs(global_state, battle_graph, max_regions=max_partitions)
            if alt is not None and alt != partition_regions_step1:
                scenarios, coverage_wave1 = bgr.sample_wave1_microstates_for_partition(
                    players=players,
                    continent_name=continent_name,
                    partition_regions=alt,
                    global_state=global_state,
                    battle_graph=battle_graph,
                    combat_libraries_base=combat_libraries_base,
                    num_scenarios=num_scenarios,
                    min_state_prob=min_state_prob,
                    max_end_states_per_region=max_end_states_per_region,
                    debug=bool(debug),
                )
                partition_regions_step1 = alt

        # ------------------------------------------------------
        # DEBUG PRINTS (Step 1): coverage + noop fraction
        # ------------------------------------------------------
        try:
            noop_flags = [int(getattr(s, "used_noop_scenario", 0)) for s in (scenarios or [])]
            noop_frac = (float(np.mean(noop_flags)) if noop_flags else float("nan"))
            log_sampler.debug(
                f"[sample_diag] state_id={state_id} n={len(scenarios) if scenarios else 0} "
                f"coverage={float(coverage_wave1):.6f} "
                f"noop_frac={noop_frac:.3f} "
                f"partition={_partition_summary(partition_regions_step1)} "
                f"min_state_prob={min_state_prob} max_end_states_per_region={max_end_states_per_region}"
            )
        except Exception:
            pass

        used_noop_scenario = 0

        # -----------------------------
        # If sampler returns nothing, emit a single "no-op" scenario
        # -----------------------------
        if not scenarios:
            used_noop_scenario = 1
            coverage_wave1 = 0.0

            try:
                bn = len(initial_battle_nodes)
                log_sampler.warning(
                    f"[collect_node_outcome_rows_for_state][noop] state_id={state_id} "
                    f"evaluation_mode={evaluation_mode} rollout_steps={rollout_steps} "
                    f"battle_nodes={bn} partition={_partition_summary(partition_regions_step1)} "
                    f"num_scenarios={num_scenarios} min_state_prob={min_state_prob} "
                    f"max_end_states_per_region={max_end_states_per_region}"
                )
            except Exception:
                pass

            final_states_per_scenario = [global_state]
            coverage_by_step: Dict[int, float] = {1: float(coverage_wave1)}

        else:
            final_states_per_scenario = []
            coverage_by_step = {1: float(coverage_wave1)}

            for scen_idx, scen in enumerate(scenarios):
                curr_global_state: GlobalState = scen.global_state_after_wave1
                curr_battle_graph = battle_graph

                for step in range(2, rollout_steps + 1):
                    try:
                        apply_global_state_to_board(curr_global_state, players)
                    except Exception:
                        pass

                    # -----------------------------
                    # PATCH: rebuild battle graph from board (debug-enabled)
                    # -----------------------------
                    try:
                        curr_battle_graph = _rebuild_battle_graph_from_board(
                            dbg_tag=f"state={state_id} scen={scen_idx} step={step}"
                        )
                    except Exception:
                        curr_battle_graph = battle_graph

                    # -----------------------------
                    # PATCH: stop early iff no edges (no battle), not based on node count
                    # -----------------------------
                    bnodes_k = list(_battle_graph_nodes(curr_battle_graph))
                    if not _battle_graph_has_battle(
                        curr_battle_graph,
                        debug=bool(debug),
                        debug_tag=f"state={state_id} scen={scen_idx} step={step}",
                    ):
                        try:
                            # extra details: edges + nodes (edges authoritative)
                            try:
                                n_edges_k = int(curr_battle_graph.number_of_edges())
                            except Exception:
                                n_edges_k = -1
                            log_rollout.debug(
                                f"[sample_diag][rollout_stop] state_id={state_id} scen={scen_idx} step={step} "
                                f"reason=no_battle battle_nodes={len(bnodes_k)} battle_edges={n_edges_k}"
                            )
                        except Exception:
                            pass
                        break

                    part_regions_k = _rank_partition_for_current_board_and_players(curr_battle_graph)
                    if part_regions_k is None:
                        part_regions_k = _fallback_partition_edge_pairs(
                            curr_global_state, curr_battle_graph, max_regions=max_partitions
                        )
                    if part_regions_k is None:
                        part_regions_k = _fallback_partition_single_region(curr_global_state, curr_battle_graph)

                    if part_regions_k is None:
                        try:
                            log_rollout.debug(
                                f"[sample_diag][rollout_stop] state_id={state_id} scen={scen_idx} step={step} "
                                f"reason=no_partition battle_nodes={len(bnodes_k)}"
                            )
                        except Exception:
                            pass
                        break

                    cont_scenarios, coverage_k = bgr.sample_wave1_microstates_for_partition(
                        players=players,
                        continent_name=continent_name,
                        partition_regions=part_regions_k,
                        global_state=curr_global_state,
                        battle_graph=curr_battle_graph,
                        combat_libraries_base=combat_libraries_base,
                        num_scenarios=1,
                        min_state_prob=min_state_prob,
                        max_end_states_per_region=max_end_states_per_region,
                        debug=bool(debug),
                    )

                    # NEW: if continuation sampling fails, retry edge-pair fallback once
                    if not cont_scenarios:
                        alt_k = _fallback_partition_edge_pairs(
                            curr_global_state, curr_battle_graph, max_regions=max_partitions
                        )
                        if alt_k is not None and alt_k != part_regions_k:
                            cont_scenarios, coverage_k = bgr.sample_wave1_microstates_for_partition(
                                players=players,
                                continent_name=continent_name,
                                partition_regions=alt_k,
                                global_state=curr_global_state,
                                battle_graph=curr_battle_graph,
                                combat_libraries_base=combat_libraries_base,
                                num_scenarios=1,
                                min_state_prob=min_state_prob,
                                max_end_states_per_region=max_end_states_per_region,
                                debug=bool(debug),
                            )
                            part_regions_k = alt_k

                    coverage_by_step[step] = float(coverage_k)

                    try:
                        log_rollout.debug(
                            f"[sample_diag][rollout] state_id={state_id} scen={scen_idx} step={step} "
                            f"n={len(cont_scenarios) if cont_scenarios else 0} "
                            f"coverage={float(coverage_k):.6f} "
                            f"partition={_partition_summary(part_regions_k)}"
                        )
                    except Exception:
                        pass

                    if not cont_scenarios:
                        try:
                            log_rollout.debug(
                                f"[sample_diag][rollout_stop] state_id={state_id} scen={scen_idx} step={step} "
                                f"reason=no_samples coverage={float(coverage_k):.6f} "
                                f"partition={_partition_summary(part_regions_k)}"
                            )
                        except Exception:
                            pass
                        break

                    curr_global_state = cont_scenarios[0].global_state_after_wave1

                final_states_per_scenario.append(curr_global_state)

        # ------------------------------------------------------
        # Emit rows: use final state after the last rollout step
        # ------------------------------------------------------
        for scenario_id, final_state in enumerate(final_states_per_scenario):
            final_nodes = final_state.nodes

            for node_idx in full_nodes:
                node_init: NodeState = initial_nodes[node_idx]
                node_final: NodeState = final_nodes[node_idx]

                initial_owner = node_init.owner
                initial_troops = int(node_init.troops)

                raw_final_owner = node_final.owner
                raw_final_troops = int(node_final.troops)

                if raw_final_owner in ("A", "D"):
                    effective_final_owner = raw_final_owner
                else:
                    effective_final_owner = initial_owner

                final_troops_actual = raw_final_troops
                if final_troops_actual < 1:
                    final_troops_actual = 1

                attacker_holds_final = int(effective_final_owner == "A")
                captured = int(
                    attacker_holds_final
                    and not (initial_owner == "A" and initial_troops > 0)
                )

                row: Dict[str, Any] = {
                    "state_id": state_id,
                    "scenario_id": scenario_id,
                    "node_index": int(node_idx),

                    "initial_owner": initial_owner,
                    "initial_troops": initial_troops,
                    "final_owner": effective_final_owner,
                    "final_troops": final_troops_actual,
                    "attacker_holds_final": attacker_holds_final,
                    "captured": captured,

                    "is_battle_node": int(node_idx in initial_battle_nodes),

                    "coverage_wave1": float(coverage_by_step.get(1, coverage_wave1)),
                    "rollout_steps": int(rollout_steps),

                    "used_noop_scenario": int(used_noop_scenario),
                }

                for step, cov in coverage_by_step.items():
                    if step >= 2:
                        row[f"coverage_wave{step}"] = float(cov)

                row.update(macro_features)
                rows.append(row)

        return rows

    finally:
        try:
            apply_global_state_to_board(global_state, players)
        except Exception:
            pass





# A state generator takes target macro-variables + config and returns
# (players, battle_graph) according to your own logic.
StateGenerator = Callable[
    [float, float, ExperimentConstraints, np.random.Generator],
    Tuple[Sequence["Players.Player"], Any, Any]  # players, battle_graph, full_graph
]


# ---------------------------------------------------------------------
# Create dataset for ML training
# ---------------------------------------------------------------------

def apply_global_state_to_board(global_state: GlobalState, players: Sequence["Players.Player"]) -> None:
    """
    Write the contents of a GlobalState back into Board.node_to_territory_dict,
    mapping:
      owner "A" -> players[0]
      owner "D" -> players[1]
      anything else -> neutral (None)

    IMPORTANT:
      We iterate over Board.node_to_territory_dict keys (territory indices),
      not enumerate(global_state.nodes), because the board indices are not
      guaranteed to be 0..N-1 but GlobalState is constructed to align with
      those indices.
    """
    pA = players[0]
    pD = players[1]

    for idx, terr in Board.node_to_territory_dict.items():
        # Safety: skip if the global_state doesn't have this index
        if idx < 0 or idx >= len(global_state.nodes):
            continue

        node = global_state.nodes[idx]

        if node.owner == "A":
            terr._owner = pA
        elif node.owner == "D":
            terr._owner = pD
        else:
            terr._owner = None

        terr._troops = int(node.troops)


def swap_roles_in_global_state(global_state: GlobalState) -> GlobalState:
    """
    Return a new GlobalState where owners 'A' and 'D' are swapped.

    Used to generate a second, symmetric training view in which the
    defender (original 'D') is treated as the 'A' attacker.
    """
    new_nodes = []
    for node in global_state.nodes:
        if node.owner == "A":
            new_nodes.append(NodeState(owner="D", troops=node.troops))
        elif node.owner == "D":
            new_nodes.append(NodeState(owner="A", troops=node.troops))
        else:
            new_nodes.append(NodeState(owner=node.owner, troops=node.troops))
    return GlobalState(nodes=tuple(new_nodes))



def compute_local_node_features_map(
    global_state: GlobalState,
    full_graph,
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-node local neighborhood features from full_graph + global_state.

    PATCH:
      - Define frontier using full_graph adjacency (not battle_graph).
      - Replace exploding ratio features with stable, bounded, log-scaled versions:
          * pressure_ratio  := clamp( log1p(enemy_sum) - log1p(friendly_sum), [-10, 10] )
          * local_balance   := clamp( log1p(friendly_sum + troops_i) - log1p(enemy_sum), [-10, 10] )
        These retain ordering information without 1e-6 blowups.
      - Keep the original raw neighbor counts/sums/maxes as-is (they are bounded already).
    """
    import math

    # Determine node list
    try:
        nodes_iter = full_graph.nodes()
    except TypeError:
        nodes_iter = full_graph.nodes
    nodes = list(nodes_iter)

    # Neighbor accessor
    def _neighbors(i):
        try:
            return list(full_graph.neighbors(i))
        except Exception:
            # fallback: networkx-like adj dict
            try:
                return list(full_graph.adj[i].keys())
            except Exception:
                return []

    def _clamp(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    feats: Dict[int, Dict[str, float]] = {}

    for i in nodes:
        owner_i = global_state.nodes[i].owner
        troops_i = float(global_state.nodes[i].troops)

        neighs = _neighbors(i)
        deg = float(len(neighs))

        enemy_ct = 0.0
        friendly_ct = 0.0
        enemy_sum = 0.0
        friendly_sum = 0.0
        enemy_max = 0.0
        friendly_max = 0.0

        for j in neighs:
            owner_j = global_state.nodes[j].owner
            tj = float(global_state.nodes[j].troops)

            if owner_j != owner_i:
                enemy_ct += 1.0
                enemy_sum += tj
                if tj > enemy_max:
                    enemy_max = tj
            else:
                friendly_ct += 1.0
                friendly_sum += tj
                if tj > friendly_max:
                    friendly_max = tj

        # Frontier is adjacency to any enemy in the FULL graph
        is_frontier = 1.0 if enemy_ct > 0.0 else 0.0

        # Stable log-scaled "pressure" signals (bounded)
        # NOTE: These are differences of logs, not raw ratios, to prevent blow-ups.
        pressure_logdiff = math.log1p(enemy_sum) - math.log1p(friendly_sum)
        balance_logdiff = math.log1p(friendly_sum + troops_i) - math.log1p(enemy_sum)

        pressure_ratio = _clamp(pressure_logdiff, -10.0, 10.0)
        local_balance = _clamp(balance_logdiff, -10.0, 10.0)

        feats[int(i)] = {
            "deg_full": deg,
            "enemy_neighbor_count": enemy_ct,
            "friendly_neighbor_count": friendly_ct,
            "enemy_neighbor_troops_sum": enemy_sum,
            "friendly_neighbor_troops_sum": friendly_sum,
            "max_enemy_neighbor_troops": enemy_max,
            "max_friendly_neighbor_troops": friendly_max,
            "is_frontier_node": is_frontier,
            "pressure_ratio": pressure_ratio,
            "local_balance": local_balance,
        }

    return feats


def _build_transition_macro_features(
    *,
    target_territory_ratio: float,
    target_troops_ratio: float,
    global_state: GlobalState,
    players: Sequence["Players.Player"],
    battle_graph,
    full_graph,
    continent_name: Optional[str],
    attack_perspective: str,
) -> Dict[str, Any]:
    battle_node_indices = _battle_graph_nodes(battle_graph)
    battle_total_territory_count = len(battle_node_indices)
    battle_realized_attacker_territory_ratio = compute_territory_ratio(global_state, battle_node_indices)

    battle_initial_attacker_territory_count = 0
    battle_initial_attacker_troops_count = 0
    battle_attacker_available = 0
    battle_total_troops_count = 0
    for idx in battle_node_indices:
        node = global_state.nodes[int(idx)]
        battle_total_troops_count += int(node.troops)
        if node.owner == "A":
            battle_initial_attacker_territory_count += 1
            battle_initial_attacker_troops_count += int(node.troops)
            if int(node.troops) > 1:
                battle_attacker_available += int(node.troops) - 1

    if battle_initial_attacker_troops_count > 0:
        battle_realized_attacker_available_troops_ratio = battle_attacker_available / battle_initial_attacker_troops_count
    else:
        battle_realized_attacker_available_troops_ratio = 0.0

    battle_attacker_troops_array = compute_attacker_troop_distribution(global_state, battle_node_indices)
    battle_realized_attacker_troops_distribution_cv = compute_troops_cv(battle_attacker_troops_array)
    battle_realized_attacker_troops_distribution_gini = compute_troops_gini(battle_attacker_troops_array)

    full_nodes = list(_graph_nodes(full_graph))
    full_attacker_total = sum(
        int(global_state.nodes[int(idx)].troops)
        for idx in full_nodes
        if global_state.nodes[int(idx)].owner == "A"
    )
    if full_attacker_total > 0:
        full_attacker_available_troops_ratio = battle_attacker_available / full_attacker_total
    else:
        full_attacker_available_troops_ratio = 0.0

    macro_features: Dict[str, Any] = {
        "target_territory_ratio": float(target_territory_ratio),
        "target_troops_ratio": float(target_troops_ratio),
        "battle_realized_attacker_territory_ratio": float(battle_realized_attacker_territory_ratio),
        "battle_realized_attacker_available_troops_ratio": float(battle_realized_attacker_available_troops_ratio),
        "full_realized_attacker_available_troops_ratio": float(full_attacker_available_troops_ratio),
        "battle_initial_attacker_territory_count": int(battle_initial_attacker_territory_count),
        "battle_initial_attacker_troops_count": int(battle_initial_attacker_troops_count),
        "battle_total_territory_count": int(battle_total_territory_count),
        "battle_total_troops_count": int(battle_total_troops_count),
        "battle_realized_attacker_troops_distribution_cv": float(battle_realized_attacker_troops_distribution_cv),
        "battle_realized_attacker_troops_distribution_gini": float(battle_realized_attacker_troops_distribution_gini),
        "attack_perspective": str(attack_perspective),
        "continent_name": continent_name,
    }
    macro_features.update(compute_full_graph_metrics(full_graph, players))
    macro_features.update(compute_effectiveness_metrics(full_graph, battle_graph, players))
    return macro_features


def _region_option_tuple(candidate: Any, attr: str) -> Tuple[Any, ...]:
    vals = []
    for ref in getattr(candidate, "region_policy_options", ()) or ():
        vals.append(_to_pickle_safe(getattr(ref, attr, None)))
    return tuple(vals)


def _raw_global_state_signature(global_state: GlobalState) -> Tuple[Tuple[int, str, int], ...]:
    return tuple(
        (int(node_id), str(node.owner), int(node.troops))
        for node_id, node in enumerate(global_state.nodes)
    )


def _candidate_count_category(count: int) -> str:
    value = int(count)
    if value <= 2:
        return "low_1_2"
    if value <= 10:
        return "moderate_3_10"
    if value <= 30:
        return "high_11_30"
    return "very_high_gt_30"


def _battle_node_count_category(count: int) -> str:
    value = int(count)
    if value <= 5:
        return "small_le_5"
    if value <= 10:
        return "medium_6_10"
    return "large_gt_10"


def collect_transition_distribution_example_for_state(
    *,
    state_id: int,
    players: Sequence["Players.Player"],
    battle_graph,
    full_graph,
    global_state: GlobalState,
    macro_features: Dict[str, Any],
    continent_name: str,
    combat_libraries_base: Path | str,
    config: TransitionDistributionConfig,
    attack_perspective: str = "P1_as_attacker",
    example_id: Optional[str] = None,
    target_seed: Optional[int] = None,
    config_fingerprint: Optional[str] = None,
    target_config_fingerprint: Optional[str] = None,
    library_metadata_summary: Optional[Mapping[str, Any]] = None,
    state_generation_seed: Optional[int] = None,
    attempt_index: Optional[int] = None,
    commitment_signature: Any = None,
) -> Dict[str, Any]:
    """
    Build one grouped Stage-A transition-distribution example.

    The two-stage ranker returns battle-node-scoped MC signatures. This helper
    lifts those signatures to full_graph signatures by preserving every
    non-battle full-graph node from the initial state.
    """
    target_started = time.perf_counter()
    initial_full_signature = lift_battle_signature_to_full_graph_signature(
        battle_signature=tuple(),
        initial_global_state=global_state,
        full_graph=full_graph,
    )
    battle_signature = canonical_graph_signature(battle_graph)
    full_graph_signature = canonical_graph_signature(full_graph)
    fingerprint = config_fingerprint or transition_distribution_config_fingerprint(config)
    target_fingerprint = (
        target_config_fingerprint or transition_distribution_target_fingerprint(config)
    )
    resolved_example_id = example_id or canonical_transition_example_id(
        continent_name=continent_name,
        perspective=attack_perspective,
        initial_full_graph_signature=initial_full_signature,
        battle_graph_signature=battle_signature,
        commitment_signature=commitment_signature,
        target_generation_version=config.target_generation_version,
        two_stage_mc_seed=config.resolved_two_stage_mc_seed,
        generation_config_fingerprint=target_fingerprint,
    )
    resolved_target_seed = (
        int(target_seed)
        if target_seed is not None
        else int(config.resolved_two_stage_mc_seed)
    )
    mc_samples = int(config.resolved_two_stage_mc_samples)
    library_summary = dict(library_metadata_summary or {})
    base_metadata = {
        "example_id": str(resolved_example_id),
        "target_generation_version": str(config.target_generation_version),
        "code_configuration_version": str(config.code_configuration_version),
        "config_fingerprint": str(fingerprint),
        "target_config_fingerprint": str(target_fingerprint),
        "state_id": int(state_id),
        "continent_name": str(continent_name),
        "attack_perspective": str(attack_perspective),
        "attempt_index": None if attempt_index is None else int(attempt_index),
        "state_generation_seed": (
            None if state_generation_seed is None else int(state_generation_seed)
        ),
        "target_seed": int(resolved_target_seed),
        "partition_candidate_selection_mode": str(config.partition_candidate_selection_mode),
        "utility_abs_tolerance": config.utility_abs_tolerance,
        "utility_rel_tolerance": config.utility_rel_tolerance,
        "max_candidates_per_partition": config.max_candidates_per_partition,
        "max_policy_combos_per_partition": config.max_policy_combos_per_partition,
        "second_stage_execution_mode": str(config.second_stage_execution_mode),
        "second_stage_sampling_mode": str(config.second_stage_sampling_mode),
        "two_stage_mc_samples": int(mc_samples),
        "two_stage_mc_seed": int(config.resolved_two_stage_mc_seed),
        "two_stage_mc_base_seed": int(config.resolved_two_stage_mc_seed),
        "two_stage_mc_target_seed": int(resolved_target_seed),
        "ranking_variable": str(config.ranking_variable),
        "evaluation_mode": str(config.evaluation_mode),
        "rollout_steps": int(config.rollout_steps),
        "library_dir": str(Path(combat_libraries_base).resolve()),
        "library_format": library_summary.get("library_format"),
        "policy_option_mode": library_summary.get("policy_option_mode"),
        "max_policy_options_per_row": library_summary.get("max_policy_options_per_row"),
        "max_options_per_state": library_summary.get("max_options_per_state"),
        "max_leaf_split_depth": library_summary.get("max_leaf_split_depth"),
        "library_metadata_inconsistent": bool(library_summary.get("inconsistent", False)),
        "library_metadata_summary": _to_pickle_safe(library_summary),
        "initial_full_graph_signature": initial_full_signature,
        "initial_global_state_signature": _raw_global_state_signature(global_state),
        "battle_graph_signature": battle_signature,
        "full_graph_signature": full_graph_signature,
        "commitment_signature": _to_pickle_safe(commitment_signature),
    }

    try:
        apply_global_state_to_board(global_state, players)
        result = bgr.rank_battle_graph_partition_policy_candidates_two_stage(
            players=players,
            battle_graph=battle_graph,
            combat_libraries_base=Path(combat_libraries_base),
            max_partitions=int(config.max_partitions),
            ranking_variable=config.ranking_variable,
            first_stage_value_tolerances=config.first_stage_value_tolerances,
            max_policy_combos_per_partition=config.max_policy_combos_per_partition,
            partition_candidate_selection_mode=config.partition_candidate_selection_mode,
            utility_abs_tolerance=config.utility_abs_tolerance,
            utility_rel_tolerance=config.utility_rel_tolerance,
            max_candidates_per_partition=config.max_candidates_per_partition,
            two_stage_mc_scenarios=int(mc_samples),
            two_stage_mc_seed=int(resolved_target_seed),
            track_empirical_final_distribution=True,
            max_tracked_final_states=int(config.max_top_final_states),
            second_stage_execution_mode=config.second_stage_execution_mode,
            second_stage_sampling_mode=config.second_stage_sampling_mode,
            profile_second_stage=bool(config.profile_second_stage),
        )
    except Exception as e:
        return {
            **base_metadata,
            "two_stage_used": True,
            "transition_example_status": "ranker_error",
            "transition_example_error": f"{type(e).__name__}: {e}",
            "target_generation_runtime_seconds": float(time.perf_counter() - target_started),
            "macro_features": _to_pickle_safe(macro_features),
        }

    selected = getattr(result, "selected_candidate", None)
    two_stage_diag = getattr(result, "diagnostics", {}) or {}
    if selected is None:
        return {
            **base_metadata,
            "two_stage_used": True,
            "two_stage_all_candidate_count": int(getattr(result, "all_candidate_count", 0) or 0),
            "partition_candidate_selection_mode": two_stage_diag.get("partition_candidate_selection_mode", config.partition_candidate_selection_mode),
            "utility_abs_tolerance": config.utility_abs_tolerance,
            "utility_rel_tolerance": config.utility_rel_tolerance,
            "max_candidates_per_partition": config.max_candidates_per_partition,
            "num_maximal_partitions": int(two_stage_diag.get("num_maximal_partitions", 0) or 0),
            "num_dominated_partitions_removed": int(two_stage_diag.get("num_dominated_partitions_removed", 0) or 0),
            "transition_example_status": "no_candidate",
            "transition_example_error": None,
            "target_generation_runtime_seconds": float(time.perf_counter() - target_started),
            "macro_features": _to_pickle_safe(macro_features),
        }

    battle_counts = {
        normalize_state_signature(sig): int(count)
        for sig, count in (getattr(selected, "mc_final_state_counts", None) or {}).items()
        if int(count) > 0
    }
    full_counts = build_full_graph_successor_distribution_from_mc_counts(
        mc_final_state_counts=battle_counts,
        initial_global_state=global_state,
        full_graph=full_graph,
    )
    node_marginals = derive_node_marginals_from_successor_distribution(
        successor_state_counts=full_counts,
        full_graph=full_graph,
        initial_global_state=global_state,
    )
    top_full = top_k_successor_states_from_counts(
        full_counts,
        k=int(config.max_top_final_states),
    )
    total_successor_samples = int(sum(int(value) for value in full_counts.values()))
    full_probabilities = {
        signature: float(count) / float(total_successor_samples)
        for signature, count in full_counts.items()
    } if total_successor_samples > 0 else {}

    second_stage_profile = dict(two_stage_diag.get("second_stage_profile", {}) or {})
    second_stage_counts = dict(second_stage_profile.get("counts", {}) or {})
    second_stage_timings = dict(second_stage_profile.get("timings_seconds", {}) or {})
    global_profile = dict(second_stage_profile.get("global_evaluator_profile", {}) or {})
    global_timings = dict(global_profile.get("global_evaluator_timings_seconds", {}) or {})

    battle_nodes = tuple(sorted(int(x) for x in _battle_graph_nodes(battle_graph)))
    attacker_nodes = tuple(
        node for node in battle_nodes if str(global_state.nodes[node].owner) == "A"
    )
    defender_nodes = tuple(
        node for node in battle_nodes if str(global_state.nodes[node].owner) == "D"
    )
    retained_count = int(
        two_stage_diag.get(
            "first_stage_optimal_count",
            len(getattr(result, "first_stage_optimal_candidates", ()) or ()),
        )
        or 0
    )
    runtime_seconds = float(time.perf_counter() - target_started)

    return {
        **base_metadata,
        "full_graph_nodes": _graph_nodes(full_graph),
        "battle_graph_nodes": battle_nodes,
        "battle_graph_edges": battle_signature[1],
        "full_graph_edges": full_graph_signature[1],
        "two_stage_used": True,
        "two_stage_all_candidate_count": int(getattr(result, "all_candidate_count", 0) or 0),
        "two_stage_first_stage_tie_count": int(len(getattr(result, "first_stage_optimal_candidates", ()) or ())),
        "two_stage_first_stage_best_utility": _to_pickle_safe(getattr(result, "first_stage_best_utility", None)),
        "two_stage_selected_first_stage_utility": _to_pickle_safe(getattr(selected, "first_stage_utility", None)),
        "two_stage_mc_num_scenarios": int(getattr(selected, "mc_num_scenarios", 0) or 0),
        "two_stage_mc_mean_second_stage_utility": _to_pickle_safe(getattr(selected, "mc_mean_second_stage_utility", None)),
        "two_stage_mc_mean_score": _to_pickle_safe(getattr(selected, "mc_mean_score", None)),
        "partition_candidate_selection_mode": two_stage_diag.get("partition_candidate_selection_mode", config.partition_candidate_selection_mode),
        "utility_abs_tolerance": config.utility_abs_tolerance,
        "utility_rel_tolerance": config.utility_rel_tolerance,
        "max_candidates_per_partition": config.max_candidates_per_partition,
        "num_supported_full_covers": int(two_stage_diag.get("num_supported_full_covers", 0) or 0),
        "num_supported_regions": int(two_stage_diag.get("num_supported_regions", 0) or 0),
        "num_unique_supported_partitions": int(two_stage_diag.get("num_unique_supported_partitions", 0) or 0),
        "num_maximal_partitions": int(two_stage_diag.get("num_maximal_partitions", 0) or 0),
        "num_dominated_partitions_removed": int(two_stage_diag.get("num_dominated_partitions_removed", 0) or 0),
        "num_policy_candidates_before_local_utility": int(
            two_stage_diag.get("num_policy_candidates_before_partition_local_utility", 0) or 0
        ),
        "num_policy_candidates_after_local_utility": int(
            two_stage_diag.get("num_policy_candidates_after_partition_local_utility", retained_count) or 0
        ),
        "num_retained_second_stage_candidates": int(retained_count),
        "num_unique_regional_options": int(second_stage_counts.get("num_unique_region_options", 0) or 0),
        "num_regional_sample_requests": int(second_stage_counts.get("regional_sample_requests", 0) or 0),
        "num_unique_regional_samples": int(second_stage_counts.get("unique_regional_samples", 0) or 0),
        "num_successor_states_assembled": int(second_stage_counts.get("global_states_assembled", 0) or 0),
        "num_unique_successor_states_evaluated": int(
            second_stage_counts.get("unique_global_state_signatures", 0) or 0
        ),
        "num_global_state_cache_hits": int(second_stage_counts.get("global_evaluation_cache_hits", 0) or 0),
        "regional_query_cache_hits": int(second_stage_counts.get("region_query_cache_hits", 0) or 0),
        "regional_query_cache_misses": int(second_stage_counts.get("region_query_cache_misses", 0) or 0),
        "target_generation_runtime_seconds": runtime_seconds,
        "first_stage_runtime_seconds": (
            None
            if two_stage_diag.get("first_stage_runtime_seconds") is None
            else float(two_stage_diag.get("first_stage_runtime_seconds"))
        ),
        "second_stage_runtime_seconds": (
            float(two_stage_diag.get("second_stage_runtime_seconds"))
            if two_stage_diag.get("second_stage_runtime_seconds") is not None
            else (
                float(second_stage_timings.get("total_second_stage"))
                if second_stage_timings.get("total_second_stage") is not None
                else None
            )
        ),
        "regional_query_runtime_seconds": (
            None
            if global_timings.get("regional_library_queries") is None
            else float(global_timings.get("regional_library_queries"))
        ),
        "selected_partition_signature": _to_pickle_safe(two_stage_diag.get("selected_partition_signature")),
        "selected_region_option_indices": _region_option_tuple(selected, "option_index"),
        "selected_region_option_counts": _region_option_tuple(selected, "option_count"),
        "selected_region_row_labels": _region_option_tuple(selected, "row_label"),
        "selected_region_root_actions": _region_option_tuple(selected, "root_action"),
        "selected_region_split_metadata": _region_option_tuple(selected, "split_metadata"),
        "battle_node_successor_state_counts": battle_counts,
        "full_graph_successor_state_counts": full_counts if bool(config.include_full_graph_successor_signatures) else {},
        "full_graph_successor_state_probabilities": full_probabilities if bool(config.include_full_graph_successor_signatures) else {},
        "target_distribution_sample_count": int(total_successor_samples),
        "successor_support_size": int(len(full_counts)),
        "top_full_graph_successor_states": top_full,
        "node_marginals": node_marginals,
        "battle_node_count": int(len(battle_nodes)),
        "attacker_node_count": int(len(attacker_nodes)),
        "defender_node_count": int(len(defender_nodes)),
        "attacker_troop_total": int(sum(int(global_state.nodes[node].troops) for node in attacker_nodes)),
        "defender_troop_total": int(sum(int(global_state.nodes[node].troops) for node in defender_nodes)),
        "candidate_count_category": _candidate_count_category(retained_count),
        "battle_node_count_category": _battle_node_count_category(len(battle_nodes)),
        "second_stage_profile": _to_pickle_safe(second_stage_profile),
        "second_stage_reuse": _to_pickle_safe(two_stage_diag.get("second_stage_reuse", {})),
        "macro_features": _to_pickle_safe(macro_features),
        "transition_example_status": "ok",
        "transition_example_error": None,
    }


def _node_marginal_rows_from_example(example: Mapping[str, Any], initial_global_state: GlobalState) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    macro = dict(example.get("macro_features", {}) or {})
    battle_nodes = set(int(x) for x in (example.get("battle_graph_nodes", ()) or ()))
    for node_idx, marg in (example.get("node_marginals", {}) or {}).items():
        i = int(node_idx)
        node = initial_global_state.nodes[i]
        row = {
            "example_id": example.get("example_id"),
            "target_generation_version": example.get("target_generation_version"),
            "config_fingerprint": example.get("config_fingerprint"),
            "target_seed": example.get("target_seed"),
            "two_stage_mc_samples": example.get("two_stage_mc_samples"),
            "state_id": int(example.get("state_id", -1)),
            "continent_name": example.get("continent_name"),
            "attack_perspective": example.get("attack_perspective"),
            "node_index": i,
            "initial_owner": str(node.owner),
            "initial_troops": int(node.troops),
            "p_attacker_final": float(marg.get("p_attacker_final", 0.0)),
            "p_defender_final": float(marg.get("p_defender_final", 0.0)),
            "expected_troops": float(marg.get("expected_troops", 0.0)),
            "expected_troops_if_attacker": float(marg.get("expected_troops_if_attacker", 0.0)),
            "expected_troops_if_defender": float(marg.get("expected_troops_if_defender", 0.0)),
            "p_changed_owner": float(marg.get("p_changed_owner", 0.0)),
            "is_battle_node": int(i in battle_nodes),
        }
        row.update(macro)
        rows.append(row)
    return rows


def _run_one_transition_distribution_unit(
    *,
    state_id: int,
    target_territory_ratio: float,
    target_troops_ratio: float,
    config: ExperimentConfig,
    transition_config: TransitionDistributionConfig,
    state_generator: StateGenerator,
    combat_libraries_base: Path | str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_seed = 0 if config.random_seed is None else int(config.random_seed)
    rng = np.random.default_rng(base_seed + int(state_id))
    examples: List[Dict[str, Any]] = []
    marginal_rows: List[Dict[str, Any]] = []
    continent_name = str(getattr(config.constraints, "continent_name", ""))

    try:
        players, battle_graph, full_graph = state_generator(
            target_territory_ratio,
            target_troops_ratio,
            config.constraints,
            rng,
        )
    except Exception as e:
        if bool(getattr(config, "raise_state_generator_exceptions", False)):
            raise
        examples.append(
            {
                "state_id": int(state_id),
                "continent_name": continent_name,
                "attack_perspective": "P1_as_attacker",
                "two_stage_used": True,
                "transition_example_status": "state_generator_error",
                "transition_example_error": f"{type(e).__name__}: {e}",
                "macro_features": {},
            }
        )
        return examples, marginal_rows

    global_state = agop.build_global_state_for_board(players)
    apply_global_state_to_board(global_state, players)
    macro = _build_transition_macro_features(
        target_territory_ratio=target_territory_ratio,
        target_troops_ratio=target_troops_ratio,
        global_state=global_state,
        players=players,
        battle_graph=battle_graph,
        full_graph=full_graph,
        continent_name=continent_name,
        attack_perspective="P1_as_attacker",
    )
    ex = collect_transition_distribution_example_for_state(
        state_id=state_id,
        players=players,
        battle_graph=battle_graph,
        full_graph=full_graph,
        global_state=global_state,
        macro_features=macro,
        continent_name=continent_name,
        combat_libraries_base=combat_libraries_base,
        config=transition_config,
        attack_perspective="P1_as_attacker",
    )
    examples.append(ex)
    if ex.get("transition_example_status") == "ok" and bool(transition_config.include_node_marginal_rows):
        marginal_rows.extend(_node_marginal_rows_from_example(ex, global_state))

    global_state_swapped = swap_roles_in_global_state(global_state)
    players_swapped = [players[1], players[0]]
    apply_global_state_to_board(global_state_swapped, players_swapped)
    if continent_name:
        battle_graph_swapped = agop.build_continent_battle_graph(continent_name, players_swapped, debug=False)
    else:
        battle_graph_swapped = battle_graph
    macro_swapped = _build_transition_macro_features(
        target_territory_ratio=target_territory_ratio,
        target_troops_ratio=target_troops_ratio,
        global_state=global_state_swapped,
        players=players_swapped,
        battle_graph=battle_graph_swapped,
        full_graph=full_graph,
        continent_name=continent_name,
        attack_perspective="P2_as_attacker",
    )
    ex_swapped = collect_transition_distribution_example_for_state(
        state_id=state_id,
        players=players_swapped,
        battle_graph=battle_graph_swapped,
        full_graph=full_graph,
        global_state=global_state_swapped,
        macro_features=macro_swapped,
        continent_name=continent_name,
        combat_libraries_base=combat_libraries_base,
        config=transition_config,
        attack_perspective="P2_as_attacker",
    )
    examples.append(ex_swapped)
    if ex_swapped.get("transition_example_status") == "ok" and bool(transition_config.include_node_marginal_rows):
        marginal_rows.extend(_node_marginal_rows_from_example(ex_swapped, global_state_swapped))
    return examples, marginal_rows


def run_transition_distribution_experiment(
    *,
    config: ExperimentConfig,
    transition_config: TransitionDistributionConfig,
    state_generator: StateGenerator,
    combat_libraries_base: Path | str,
    n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stage-A transition-distribution dataset generation.

    Returns:
      - examples_df: one grouped row per generated state/perspective.
      - node_marginals_df: one derived row per full-graph node per example.
    """
    jobs: List[Tuple[int, float, float]] = []
    state_id = 0
    for target_territory_ratio in config.territory_ratios:
        for target_troops_ratio in config.troops_ratios:
            for _ in range(int(config.samples_per_combo)):
                jobs.append((state_id, float(target_territory_ratio), float(target_troops_ratio)))
                state_id += 1

    if int(n_jobs) == 1:
        chunks = [
            _run_one_transition_distribution_unit(
                state_id=sid,
                target_territory_ratio=ttr,
                target_troops_ratio=tpr,
                config=config,
                transition_config=transition_config,
                state_generator=state_generator,
                combat_libraries_base=combat_libraries_base,
            )
            for sid, ttr, tpr in jobs
        ]
    else:
        if Parallel is None or delayed is None:
            raise RuntimeError("joblib is required for run_transition_distribution_experiment(n_jobs > 1).")
        chunks = Parallel(n_jobs=int(n_jobs))(
            delayed(_run_one_transition_distribution_unit)(
                state_id=sid,
                target_territory_ratio=ttr,
                target_troops_ratio=tpr,
                config=config,
                transition_config=transition_config,
                state_generator=state_generator,
                combat_libraries_base=combat_libraries_base,
            )
            for sid, ttr, tpr in jobs
        )

    examples: List[Dict[str, Any]] = []
    marginal_rows: List[Dict[str, Any]] = []
    for ex_rows, node_rows in chunks:
        examples.extend(ex_rows)
        marginal_rows.extend(node_rows)
    return pd.DataFrame(examples), pd.DataFrame(marginal_rows)


def _run_one_unit_node_transition(
    *,
    state_id: int,
    target_territory_ratio: float,
    target_troops_ratio: float,
    config: ExperimentConfig,
    state_generator: StateGenerator,
    combat_libraries_base: Path,
    num_scenarios_wave1: int,
    rollout_steps: int,
) -> tuple[list[dict], dict]:
    """
    Runs exactly ONE (territory_ratio, troops_ratio, sample) unit and returns:
      - rows: list of per-node rows (includes both P1-as-attacker and P2-as-attacker augmentations)
      - diag: lightweight diagnostics for summary / debugging
    """
    # Deterministic per-unit RNG (important for reproducibility)
    rng = np.random.default_rng(int(config.random_seed) + int(state_id))

    # local diagnostic counters
    diag = {
        "state_id": state_id,
        "skipped": None,   # string reason if skipped
        "rows_p1": 0,
        "rows_p2": 0,
        "battle_nodes_p1": None,
        "battle_nodes_p2": None,
        "full_nodes": None,
    }

    # --- local helper (copied verbatim from your function) ---
    def _choose_partition_regions_from_rank_result(
        *,
        best_eval: Any,
        partitions_full: list[Any],
        best_partition: Any,
        working_partitions: list[Any],
    ) -> tuple[Optional[Any], str, bool]:
        if best_eval is None:
            return None, "none", False

        if partitions_full:
            part_idx = getattr(best_eval, "partition_index", 0)
            if not isinstance(part_idx, int):
                part_idx = 0
            if part_idx < 0 or part_idx >= len(partitions_full):
                part_idx = 0
            return partitions_full[part_idx], "partitions_full", True

        if best_partition is not None:
            return best_partition, "best_partition", False

        if working_partitions:
            return working_partitions[0], "working_partitions", False

        return None, "none", False

    # ----------------------------
    # Begin: your unit logic
    # ----------------------------
    try:
        players, battle_graph, full_graph = state_generator(
            target_territory_ratio,
            target_troops_ratio,
            config.constraints,
            rng,
        )
    except Exception as e:
        if bool(getattr(config, "raise_state_generator_exceptions", False)):
            raise
        diag["skipped"] = "state_generator_exception"
        diag["state_generator_exception_type"] = type(e).__name__
        diag["state_generator_exception_message"] = str(e)
        if bool(getattr(config, "include_state_generator_traceback", False)):
            diag["state_generator_traceback"] = traceback.format_exc()
        return [], diag

    global_state = agop.build_global_state_for_board(players)

    # Sync Board to canonical state
    try:
        apply_global_state_to_board(global_state, players)
    except Exception:
        pass

    battle_node_indices = _battle_graph_nodes(battle_graph)
    battle_total_territory_count = len(battle_node_indices)

    # Track full node count
    try:
        full_node_count = len(list(full_graph.nodes()))
    except Exception:
        full_node_count = 0
    diag["full_nodes"] = full_node_count
    diag["battle_nodes_p1"] = battle_total_territory_count

    # Full graph metrics + effectiveness metrics
    full_metrics = compute_full_graph_metrics(full_graph, players)
    effectiveness_metrics = compute_effectiveness_metrics(full_graph, battle_graph, players)

    # Battle graph realized metrics
    battle_realized_attacker_territory_ratio = compute_territory_ratio(global_state, battle_node_indices)

    battle_initial_attacker_territory_count = 0
    battle_initial_attacker_troops_count = 0
    battle_attacker_available = 0
    battle_total_troops_count = 0

    for idx in battle_node_indices:
        node = global_state.nodes[idx]
        battle_total_troops_count += node.troops
        if node.owner == "A":
            battle_initial_attacker_territory_count += 1
            battle_initial_attacker_troops_count += node.troops
            if node.troops > 1:
                battle_attacker_available += node.troops - 1

    if battle_initial_attacker_troops_count > 0:
        battle_realized_attacker_available_troops_ratio = battle_attacker_available / battle_initial_attacker_troops_count
    else:
        battle_realized_attacker_available_troops_ratio = 0.0

    battle_attacker_troops_array = compute_attacker_troop_distribution(global_state, battle_node_indices)
    battle_realized_attacker_troops_distribution_cv = compute_troops_cv(battle_attacker_troops_array)
    battle_realized_attacker_troops_distribution_gini = compute_troops_gini(battle_attacker_troops_array)

    # Full-graph "mobilization" metric
    full_nodes = list(full_graph.nodes())
    full_attacker_total = sum(
        global_state.nodes[idx].troops
        for idx in full_nodes
        if global_state.nodes[idx].owner == "A"
    )
    if full_attacker_total > 0:
        full_attacker_available_troops_ratio = battle_attacker_available / full_attacker_total
    else:
        full_attacker_available_troops_ratio = 0.0

    # Partition ranking (canonical)
    continent_name = getattr(config.constraints, "continent_name", None)

    eval_mode = getattr(config, "evaluation_mode", "two_wave")
    policy_option_selection = getattr(config, "policy_option_selection", "primary")
    if eval_mode not in ("one_wave", "two_wave"):
        raise ValueError(f"Unknown evaluation_mode={eval_mode!r}")

    num_scenarios_two_wave = getattr(config, "num_scenarios_two_wave", 50)
    min_state_prob_two_wave = getattr(config, "min_state_prob_two_wave", 0.0)
    max_end_states_two_wave = getattr(config, "max_end_states_per_region_two_wave", None)

    partition_regions = None
    partition_source = "none"
    partition_is_full = False

    try:
        if eval_mode == "two_wave":
            if continent_name is None:
                raise ValueError("evaluation_mode='two_wave' requires constraints.continent_name to be set.")

            lookahead_bundle = bgr.rank_battle_graph_partitions_with_lookahead(
                players=players,
                continent_name=continent_name,
                combat_libraries_base=combat_libraries_base,
                max_partitions_wave1=config.max_partitions,
                ranking_variable=config.ranking_variable,
                max_partitions_wave2=config.max_partitions,
                num_scenarios=num_scenarios_two_wave,
                policy_option_selection=policy_option_selection,
                min_state_prob=min_state_prob_two_wave,
                max_end_states_per_region=max_end_states_two_wave,
            )

            base_result = lookahead_bundle.get("base_result", {}) or {}
            lookahead_eval = lookahead_bundle.get("lookahead_result", None)
            best_eval = lookahead_eval if lookahead_eval is not None else base_result.get("best_evaluation", None)

            partitions_full = base_result.get("partitions_full", []) or []
            best_partition = base_result.get("best_partition", None)
            working_partitions = base_result.get("working_partitions", []) or []

            partition_regions, partition_source, partition_is_full = _choose_partition_regions_from_rank_result(
                best_eval=best_eval,
                partitions_full=partitions_full,
                best_partition=best_partition,
                working_partitions=working_partitions,
            )
        else:
            rank_result = bgr.rank_battle_graph_partitions(
                players=players,
                battle_graph=battle_graph,
                combat_libraries_base=combat_libraries_base,
                max_partitions=config.max_partitions,
                ranking_variable=config.ranking_variable,
                lookahead_depth=0,
                policy_option_selection=policy_option_selection,
            )

            best_eval = rank_result.get("best_evaluation", None)
            partitions_full = rank_result.get("partitions_full", []) or []
            best_partition = rank_result.get("best_partition", None)
            working_partitions = rank_result.get("working_partitions", []) or []

            partition_regions, partition_source, partition_is_full = _choose_partition_regions_from_rank_result(
                best_eval=best_eval,
                partitions_full=partitions_full,
                best_partition=best_partition,
                working_partitions=working_partitions,
            )
    except Exception:
        diag["skipped"] = "rank_exception"
        return [], diag

    if partition_regions is None:
        diag["skipped"] = "no_partition"
        return [], diag

    # Macro features (P1 attacker)
    macro_features: dict[str, Any] = {
        "target_territory_ratio": float(target_territory_ratio),
        "target_troops_ratio": float(target_troops_ratio),
        "battle_realized_attacker_territory_ratio": float(battle_realized_attacker_territory_ratio),
        "battle_realized_attacker_available_troops_ratio": float(battle_realized_attacker_available_troops_ratio),
        "full_realized_attacker_available_troops_ratio": float(full_attacker_available_troops_ratio),
        "battle_initial_attacker_territory_count": battle_initial_attacker_territory_count,
        "battle_initial_attacker_troops_count": battle_initial_attacker_troops_count,
        "battle_total_territory_count": battle_total_territory_count,
        "battle_total_troops_count": battle_total_troops_count,
        "battle_realized_attacker_troops_distribution_cv": float(battle_realized_attacker_troops_distribution_cv),
        "battle_realized_attacker_troops_distribution_gini": float(battle_realized_attacker_troops_distribution_gini),
        "attack_perspective": "P1_as_attacker",
        "continent_name": continent_name,
        "partition_source": partition_source,
        "partition_is_full": bool(partition_is_full),
    }
    macro_features.update(full_metrics)
    macro_features.update(effectiveness_metrics)
    macro_features.update(summarize_region_policy_options(best_eval, policy_option_selection))

    # Collect P1 rows
    try:
        node_rows = collect_node_outcome_rows_for_state(
            state_id=state_id,
            players=players,
            battle_graph=battle_graph,
            full_graph=full_graph,
            global_state=global_state,
            partition_regions=partition_regions,
            macro_features=macro_features,
            continent_name=continent_name,
            combat_libraries_base=combat_libraries_base,
            num_scenarios=num_scenarios_wave1,
            min_state_prob=min_state_prob_two_wave,
            max_end_states_per_region=max_end_states_two_wave,
            rollout_steps=rollout_steps,
            max_partitions=config.max_partitions,
            ranking_variable=config.ranking_variable,
            policy_option_selection=policy_option_selection,
            evaluation_mode=eval_mode,
            num_scenarios_two_wave=num_scenarios_two_wave,
            min_state_prob_two_wave=min_state_prob_two_wave,
            max_end_states_per_region_two_wave=max_end_states_two_wave,
        )
    except Exception:
        diag["skipped"] = "collect_rows_exception_p1"
        return [], diag

    if not node_rows:
        diag["skipped"] = "no_scenarios_p1"
        return [], diag

    local_map = compute_local_node_features_map(global_state, full_graph)
    for r in node_rows:
        ni = int(r.get("node_index"))
        r.update(local_map.get(ni, {}))

    rows_out: list[dict] = []
    rows_out.extend(node_rows)
    diag["rows_p1"] = len(node_rows)

    # ----------------------------
    # Symmetric augmentation P2
    # ----------------------------
    global_state_swapped = swap_roles_in_global_state(global_state)
    players_swapped = [players[1], players[0]]

    try:
        # mutate Board into swapped perspective
        apply_global_state_to_board(global_state_swapped, players_swapped)

        if continent_name is None:
            battle_graph_swapped = battle_graph
            battle_node_indices_swapped = battle_node_indices
        else:
            battle_graph_swapped = agop.build_continent_battle_graph(continent_name, players_swapped, debug=True)
            battle_node_indices_swapped = _battle_graph_nodes(battle_graph_swapped)

        diag["battle_nodes_p2"] = len(battle_node_indices_swapped)

        battle_realized_attacker_territory_ratio_swapped = compute_territory_ratio(
            global_state_swapped, battle_node_indices_swapped
        )

        battle_initial_attacker_territory_count_swapped = 0
        battle_initial_attacker_troops_count_swapped = 0
        battle_attacker_available_swapped = 0
        battle_total_troops_count_swapped = 0

        for idx in battle_node_indices_swapped:
            node_sw = global_state_swapped.nodes[idx]
            battle_total_troops_count_swapped += node_sw.troops
            if node_sw.owner == "A":
                battle_initial_attacker_territory_count_swapped += 1
                battle_initial_attacker_troops_count_swapped += node_sw.troops
                if node_sw.troops > 1:
                    battle_attacker_available_swapped += node_sw.troops - 1

        if battle_initial_attacker_troops_count_swapped > 0:
            battle_realized_attacker_available_troops_ratio_swapped = (
                battle_attacker_available_swapped / battle_initial_attacker_troops_count_swapped
            )
        else:
            battle_realized_attacker_available_troops_ratio_swapped = 0.0

        battle_attacker_troops_array_swapped = compute_attacker_troop_distribution(
            global_state_swapped, battle_node_indices_swapped
        )
        battle_realized_attacker_troops_distribution_cv_swapped = compute_troops_cv(
            battle_attacker_troops_array_swapped
        )
        battle_realized_attacker_troops_distribution_gini_swapped = compute_troops_gini(
            battle_attacker_troops_array_swapped
        )

        full_nodes_swapped = list(full_graph.nodes())
        full_attacker_total_swapped = sum(
            global_state_swapped.nodes[idx].troops
            for idx in full_nodes_swapped
            if global_state_swapped.nodes[idx].owner == "A"
        )
        if full_attacker_total_swapped > 0:
            full_attacker_available_troops_ratio_swapped = (
                battle_attacker_available_swapped / full_attacker_total_swapped
            )
        else:
            full_attacker_available_troops_ratio_swapped = 0.0

        full_metrics_swapped = compute_full_graph_metrics(full_graph, players_swapped)
        effectiveness_metrics_swapped = compute_effectiveness_metrics(full_graph, battle_graph_swapped, players_swapped)

        # Partition ranking swapped
        partition_regions_swapped = None
        partition_source_swapped = "none"
        partition_is_full_swapped = False

        try:
            if eval_mode == "two_wave":
                lookahead_bundle_swapped = bgr.rank_battle_graph_partitions_with_lookahead(
                    players=players_swapped,
                    continent_name=continent_name,
                    combat_libraries_base=combat_libraries_base,
                    max_partitions_wave1=config.max_partitions,
                    ranking_variable=config.ranking_variable,
                    max_partitions_wave2=config.max_partitions,
                    num_scenarios=num_scenarios_two_wave,
                    policy_option_selection=policy_option_selection,
                    min_state_prob=min_state_prob_two_wave,
                    max_end_states_per_region=max_end_states_two_wave,
                )

                base_result_swapped = lookahead_bundle_swapped.get("base_result", {}) or {}
                lookahead_eval_swapped = lookahead_bundle_swapped.get("lookahead_result", None)
                best_eval_swapped = (
                    lookahead_eval_swapped
                    if lookahead_eval_swapped is not None
                    else base_result_swapped.get("best_evaluation", None)
                )

                partitions_full_swapped = base_result_swapped.get("partitions_full", []) or []
                best_partition_swapped = base_result_swapped.get("best_partition", None)
                working_swapped = base_result_swapped.get("working_partitions", []) or []

                partition_regions_swapped, partition_source_swapped, partition_is_full_swapped = (
                    _choose_partition_regions_from_rank_result(
                        best_eval=best_eval_swapped,
                        partitions_full=partitions_full_swapped,
                        best_partition=best_partition_swapped,
                        working_partitions=working_swapped,
                    )
                )
            else:
                rank_result_swapped = bgr.rank_battle_graph_partitions(
                    players=players_swapped,
                    battle_graph=battle_graph_swapped,
                    combat_libraries_base=combat_libraries_base,
                    max_partitions=config.max_partitions,
                    ranking_variable=config.ranking_variable,
                    lookahead_depth=0,
                    policy_option_selection=policy_option_selection,
                )

                best_eval_swapped = rank_result_swapped.get("best_evaluation", None)
                partitions_full_swapped = rank_result_swapped.get("partitions_full", []) or []
                best_partition_swapped = rank_result_swapped.get("best_partition", None)
                working_swapped = rank_result_swapped.get("working_partitions", []) or []

                partition_regions_swapped, partition_source_swapped, partition_is_full_swapped = (
                    _choose_partition_regions_from_rank_result(
                        best_eval=best_eval_swapped,
                        partitions_full=partitions_full_swapped,
                        best_partition=best_partition_swapped,
                        working_partitions=working_swapped,
                    )
                )
        except Exception:
            # Don't kill the whole unit; just omit P2 rows
            return rows_out, diag

        if partition_regions_swapped is None:
            return rows_out, diag

        macro_features_swapped: dict[str, Any] = {
            "target_territory_ratio": float(target_territory_ratio),
            "target_troops_ratio": float(target_troops_ratio),
            "battle_realized_attacker_territory_ratio": float(battle_realized_attacker_territory_ratio_swapped),
            "battle_realized_attacker_available_troops_ratio": float(
                battle_realized_attacker_available_troops_ratio_swapped
            ),
            "full_realized_attacker_available_troops_ratio": float(full_attacker_available_troops_ratio_swapped),
            "battle_initial_attacker_territory_count": battle_initial_attacker_territory_count_swapped,
            "battle_initial_attacker_troops_count": battle_initial_attacker_troops_count_swapped,
            "battle_total_territory_count": len(battle_node_indices_swapped),
            "battle_total_troops_count": battle_total_troops_count_swapped,
            "battle_realized_attacker_troops_distribution_cv": float(battle_realized_attacker_troops_distribution_cv_swapped),
            "battle_realized_attacker_troops_distribution_gini": float(
                battle_realized_attacker_troops_distribution_gini_swapped
            ),
            "attack_perspective": "P2_as_attacker",
            "continent_name": continent_name,
            "partition_source": partition_source_swapped,
            "partition_is_full": bool(partition_is_full_swapped),
        }
        macro_features_swapped.update(full_metrics_swapped)
        macro_features_swapped.update(effectiveness_metrics_swapped)
        macro_features_swapped.update(summarize_region_policy_options(best_eval_swapped, policy_option_selection))

        node_rows_swapped = collect_node_outcome_rows_for_state(
            state_id=state_id,
            players=players_swapped,
            battle_graph=battle_graph_swapped,
            full_graph=full_graph,
            global_state=global_state_swapped,
            partition_regions=partition_regions_swapped,
            macro_features=macro_features_swapped,
            continent_name=continent_name,
            combat_libraries_base=combat_libraries_base,
            num_scenarios=num_scenarios_wave1,
            min_state_prob=min_state_prob_two_wave,
            max_end_states_per_region=max_end_states_two_wave,
            rollout_steps=rollout_steps,
            max_partitions=config.max_partitions,
            ranking_variable=config.ranking_variable,
            policy_option_selection=policy_option_selection,
            evaluation_mode=eval_mode,
            num_scenarios_two_wave=num_scenarios_two_wave,
            min_state_prob_two_wave=min_state_prob_two_wave,
            max_end_states_per_region_two_wave=max_end_states_two_wave,
        )

        if not node_rows_swapped:
            return rows_out, diag

        local_map_sw = compute_local_node_features_map(global_state_swapped, full_graph)
        for r in node_rows_swapped:
            ni = int(r.get("node_index"))
            r.update(local_map_sw.get(ni, {}))

        rows_out.extend(node_rows_swapped)
        diag["rows_p2"] = len(node_rows_swapped)

        return rows_out, diag

    finally:
        # ALWAYS restore Board
        try:
            apply_global_state_to_board(global_state, players)
        except Exception:
            pass


def run_node_transition_experiment(
    config: ExperimentConfig,
    state_generator: StateGenerator,
    combat_libraries_base: Path,
    num_scenarios_wave1: Optional[int] = None,
    rollout_steps: int = 2,
    # -------------------------
    # NEW: parallelism controls
    # -------------------------
    n_jobs: int = 1,                 # 1 = sequential (old behavior)
    verbose: int = 5,                # joblib verbosity
) -> pd.DataFrame:

    rows: list[dict[str, Any]] = []

    # Keep your existing progress tracker for sequential mode
    progress = make_run_progress(config)
    print_every_units = int(getattr(config, "print_every_units", 5))

    eval_mode = getattr(config, "evaluation_mode", "two_wave")
    policy_option_selection = getattr(config, "policy_option_selection", "primary")
    if eval_mode not in ("one_wave", "two_wave"):
        raise ValueError(
            f"Unknown evaluation_mode={eval_mode!r}. "
            "Expected 'one_wave' or 'two_wave'."
        )

    num_scenarios_two_wave = getattr(config, "num_scenarios_two_wave", 50)
    if num_scenarios_wave1 is None:
        num_scenarios_wave1 = num_scenarios_two_wave

    # one-time library diagnostics (unchanged)
    try:
        libs_exists = bool(combat_libraries_base.exists())
        libs_files = 0
        if libs_exists and combat_libraries_base.is_dir():
            libs_files = sum(1 for _ in combat_libraries_base.rglob("*") if _.is_file())
        log_runner.info(
            "[run_node_transition_experiment][diag] "
            f"eval_mode={eval_mode} rollout_steps={rollout_steps} "
            f"num_scenarios_wave1={num_scenarios_wave1} "
            f"combat_libraries_base={combat_libraries_base} exists={libs_exists} files={libs_files}"
        )
        log_runner.info(
            "[run_node_transition_experiment][diag] "
            f"max_partitions={config.max_partitions} "
            f"ranking_variable={config.ranking_variable} "
            f"policy_option_selection={getattr(config, 'policy_option_selection', 'primary')} "
            f"continent_name={getattr(config.constraints, 'continent_name', None)!r}"
        )
    except Exception:
        pass

    # Build job list (each job corresponds to one "unit")
    jobs: list[tuple[int, float, float]] = []
    state_id = 0
    for target_territory_ratio in config.territory_ratios:
        for target_troops_ratio in config.troops_ratios:
            for _ in range(config.samples_per_combo):
                jobs.append((state_id, float(target_territory_ratio), float(target_troops_ratio)))
                state_id += 1

    # -------------------------
    # Sequential mode (old behavior)
    # -------------------------
    if int(n_jobs) <= 1:
        for state_id, ttr, tpr in jobs:
            progress.mark_unit_attempted(
                target_territory_ratio=float(ttr),
                target_troops_ratio=float(tpr),
                state_id=state_id,
            )
            progress.maybe_print_checkpoint(every_units=print_every_units)

            unit_rows, diag = _run_one_unit_node_transition(
                state_id=state_id,
                target_territory_ratio=ttr,
                target_troops_ratio=tpr,
                config=config,
                state_generator=state_generator,
                combat_libraries_base=combat_libraries_base,
                num_scenarios_wave1=int(num_scenarios_wave1),
                rollout_steps=int(rollout_steps),
            )

            if diag.get("skipped"):
                if diag.get("skipped") == "state_generator_exception":
                    log_runner.error(
                        "state_generator_exception state_id=%s type=%s message=%s",
                        diag.get("state_id"),
                        diag.get("state_generator_exception_type"),
                        diag.get("state_generator_exception_message"),
                    )
                    if diag.get("state_generator_traceback"):
                        log_runner.error("%s", diag.get("state_generator_traceback"))
                progress.mark_skip(str(diag["skipped"]))
                continue

            progress.mark_unit_completed()

            if unit_rows:
                rows.extend(unit_rows)
                # mimic your old stats update roughly:
                # (we can't perfectly split P1/P2 row lists here without extra tagging,
                #  but progress is mostly informational anyway)
                progress.add_rows_stats(unit_rows, perspective="mixed")

        progress.print_summary()
        return pd.DataFrame(rows)

    # -------------------------
    # Parallel mode (processes)
    # -------------------------
    if Parallel is None or delayed is None:
        raise ImportError(
            "joblib is required for run_node_transition_experiment(n_jobs > 1). "
            "Install joblib or run with n_jobs=1."
        )

    out = Parallel(n_jobs=int(n_jobs), prefer="processes", verbose=int(verbose))(
        delayed(_run_one_unit_node_transition)(
            state_id=state_id,
            target_territory_ratio=ttr,
            target_troops_ratio=tpr,
            config=config,
            state_generator=state_generator,
            combat_libraries_base=combat_libraries_base,
            num_scenarios_wave1=int(num_scenarios_wave1),
            rollout_steps=int(rollout_steps),
        )
        for (state_id, ttr, tpr) in jobs
    )

    # Flatten and summarize
    skipped = {}
    first_skip_detail: Dict[str, Any] = {}
    total_rows = 0
    for unit_rows, diag in out:
        if diag.get("skipped"):
            skipped[diag["skipped"]] = skipped.get(diag["skipped"], 0) + 1
            first_skip_detail.setdefault(str(diag["skipped"]), diag)
            continue
        if unit_rows:
            rows.extend(unit_rows)
            total_rows += len(unit_rows)

    print("\n=== Parallel run summary ===")
    print(f"Units attempted: {len(jobs)}")
    print(f"Units skipped:   {sum(skipped.values())}")
    if skipped:
        print("Skip reasons:")
        for k, v in sorted(skipped.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k}: {v}")
            detail = first_skip_detail.get(k, {})
            if k == "state_generator_exception" and detail:
                print(
                    "    first exception: "
                    f"{detail.get('state_generator_exception_type')}: "
                    f"{detail.get('state_generator_exception_message')}"
                )
    print(f"Total rows:      {total_rows}")
    print("=== End summary ===\n")

    return pd.DataFrame(rows)
