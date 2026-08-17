
# full_board_simulation_ML.py
"""
Full-board multi-continent ML simulation.

Why this module exists
----------------------
Your existing `predict_future_states_ML.simulate_multi_turns_ML(...)` is *continent-scoped*:
  - it uses ONE continent battle_graph/full_graph
  - it computes reinforcements with a continent "synthetic" rule
  - it implicitly treats outside-adjacent troops as available for that continent

Now that you're simulating the entire board, you need:
  1) Per-continent ML model bundles (you already have them under models/node_level_models__<slug>.joblib)
  2) Board-level reinforcement computation (from actual/predicted full-board state)
  3) A rigorous "commitment" policy so an outside-adjacent node is only counted once.

This module provides a *first working implementation* of (1)-(3) that is:
  - deterministic (given the same state),
  - easy to swap out later (commitment_policy / utility hooks),
  - conservative: only affects which nodes are included in each continent's battle graph.

IMPORTANT
---------
This module assumes your patched `approximate_graph_outcome_probabilities.build_continent_battle_graph`
supports `commitment_map=...` to prevent cross-continent double counting of outside-adjacent nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import math
import joblib
import numpy as np


import logging
from project_risk.infrastructure.log_config import get_logger
from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop

from project_risk.mathematical.transition_prediction_ml.generate_data_ML import build_full_graph, apply_global_state_to_board
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState

# Reuse your proven per-turn mechanics (transition + allocation helpers)
from project_risk.mathematical.transition_prediction_ml.predict_future_states_ML import (
    compute_macro_features_from_global_state,
    apply_expectations_as_state,
    reallocate_troops_within_friendly_components,
    allocate_reinforcements_greedy_cheapest,
    sample_and_apply_transition_distribution_for_continent,
)
from project_risk.mathematical.transition_prediction_ml.transition_distribution_ML import load_transition_distribution_models_by_continent

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Commitment:
    """
    A commitment map associates an *outside* node with exactly one continent
    for the purpose of building continent battle graphs in a single scenario/turn.

    Example:
        commitment_by_node[17] = "Europe"
    """
    commitment_by_node: Dict[int, str]


@dataclass(frozen=True)
class TransitionParticle:
    state: GlobalState
    weight: float
    path_id: int
    parent_path_id: Optional[int] = None
    turn: int = 0
    diagnostics: Tuple[Dict[str, Any], ...] = ()



def apply_reinforcement_allocation(
    global_state: GlobalState,
    alloc: Dict[int, int],
) -> GlobalState:
    """Return a new GlobalState with reinforcement allocation applied.

    `alloc` is {node_index: troops_added}. Nodes are assumed to already be owned
    by the player receiving reinforcements; this function does not change owners.
    """
    if not alloc:
        return global_state
    new_nodes=list(global_state.nodes)
    for idx, add in alloc.items():
        i=int(idx)
        a=int(add)
        if a<=0:
            continue
        n=new_nodes[i]
        new_nodes[i]=type(n)(owner=n.owner, troops=int(n.troops)+a)
    return GlobalState(nodes=tuple(new_nodes))

# ---------------------------------------------------------------------
# Debug / audit helpers
# ---------------------------------------------------------------------

def snapshot_global_state(
    global_state: GlobalState,
    *,
    max_nodes: int = 42,
) -> Dict[str, Any]:
    """Return a compact snapshot of the full-board GlobalState.

    Designed for logging/debugging (not for ML).
    """
    A_nodes = []
    D_nodes = []
    troops = {}
    for i, ns in enumerate(global_state.nodes):
        owner = getattr(ns, "owner", None)
        t = int(getattr(ns, "troops", 0))
        troops[int(i)] = t
        if owner == "A":
            A_nodes.append(int(i))
        elif owner == "D":
            D_nodes.append(int(i))

    # sample a few nodes for readability
    def _sample(lst):
        return lst[:max(0, min(len(lst), 12))]

    return {
        "A_terr": int(len(A_nodes)),
        "D_terr": int(len(D_nodes)),
        "A_nodes_sample": _sample(A_nodes),
        "D_nodes_sample": _sample(D_nodes),
        "troops_total_A": int(sum(troops[i] for i in A_nodes)),
        "troops_total_D": int(sum(troops[i] for i in D_nodes)),
        "troops_sample": {i: troops[i] for i in list(sorted(troops.keys()))[:min(max_nodes, len(troops))]},
    }


def snapshot_continent_ownership(global_state: GlobalState) -> Dict[str, Any]:
    """Per-continent ownership + whether a continent is fully owned."""
    out: Dict[str, Any] = {}
    for cont, terrs in Board.continent_territory_dict.items():
        nodes = [int(t._index) for t in terrs]
        A = sum(1 for i in nodes if global_state.nodes[i].owner == "A")
        D = sum(1 for i in nodes if global_state.nodes[i].owner == "D")
        all_A = bool(nodes) and (A == len(nodes))
        all_D = bool(nodes) and (D == len(nodes))
        out[str(cont)] = {
            "n": int(len(nodes)),
            "A": int(A),
            "D": int(D),
            "all_A": all_A,
            "all_D": all_D,
        }
    return out


def log_commitment(commitment: Commitment, *, turn: int) -> None:
    log = get_logger("risk.commitment", step=turn)
    if not log.isEnabledFor(logging.DEBUG):
        return
    by_cont: Dict[str, int] = {}
    for n, c in (commitment.commitment_by_node or {}).items():
        by_cont[str(c)] = by_cont.get(str(c), 0) + 1
    top = sorted(by_cont.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    log.debug("commitment nodes=%d by_continent(top10)=%s", len(commitment.commitment_by_node or {}), top)

# ---------------------------------------------------------------------
# 1) Model loading
# ---------------------------------------------------------------------

def _slugify_continent(name: str) -> str:
    s = (name or "").strip().lower()
    import re
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_continent"


def list_all_continents() -> List[str]:
    return list(Board.continent_territory_dict.keys())


def load_models_by_continent(
    models_dir: Path | str = "models",
    *,
    strict: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load the per-continent model bundles produced by train_ML.

    Expected file naming:
        models/node_level_models__<slug>.joblib
    """
    models_dir = Path(models_dir)
    out: Dict[str, Dict[str, Any]] = {}

    missing: List[str] = []
    for cont in list_all_continents():
        slug = _slugify_continent(cont)
        p = models_dir / f"node_level_models__{slug}.joblib"
        if not p.exists():
            missing.append(cont)
            continue
        out[cont] = joblib.load(p)

        # stability / reproducibility: force single-threading if supported
        for k in ("capture_model_attacker", "troop_model_attacker", "troop_model_defender"):
            m = out[cont].get(k, None)
            if hasattr(m, "n_jobs"):
                try:
                    m.n_jobs = 1
                except Exception:
                    pass

    if missing and strict:
        raise FileNotFoundError(
            "Missing per-continent model files for: "
            f"{missing}. Looked in {models_dir.resolve()}."
        )
    return out


# ---------------------------------------------------------------------
# 2) Board-level reinforcements
# ---------------------------------------------------------------------

def _all_node_indices() -> List[int]:
    # Board.node_to_territory_dict is dict[int, Territory]
    return [int(i) for i in Board.node_to_territory_dict.keys()]


def compute_board_reinforcements_risklike(
    global_state: GlobalState,
    *,
    min_base: int = 3,
    per_territories_div: int = 3,
    continent_bonus: Optional[Dict[str, int]] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Risk-like reinforcements computed from the FULL board state.

    Base rule (classic Risk):
        base = max(min_base, floor(territories_owned / per_territories_div))

    Plus continent bonuses if a player fully owns a continent.

    Returns (A_reinf, D_reinf, diag).
    """
    if continent_bonus is None:
        # You can customize these; defaults are placeholders if you haven't encoded bonuses in Board.
        continent_bonus = {}

    total_A = 0
    total_D = 0
    for idx in _all_node_indices():
        owner = global_state.nodes[idx].owner
        if owner == "A":
            total_A += 1
        elif owner == "D":
            total_D += 1
    # IMPORTANT: 0 territories => 0 reinforcements
    if total_A <= 0:
        A_base = 0
    else:
        A_base = max(int(min_base), int(total_A // per_territories_div))

    if total_D <= 0:
        D_base = 0
    else:
        D_base = max(int(min_base), int(total_D // per_territories_div))

    A_bonus = 0
    D_bonus = 0
    for cont, terrs in Board.continent_territory_dict.items():
        nodes = [t._index for t in terrs]
        # fully owned?
        if nodes and all(global_state.nodes[int(i)].owner == "A" for i in nodes):
            A_bonus += int(continent_bonus.get(cont, 0))
        if nodes and all(global_state.nodes[int(i)].owner == "D" for i in nodes):
            D_bonus += int(continent_bonus.get(cont, 0))
    # Safety: eliminated player cannot receive continent bonuses
    if A_base == 0:
        A_bonus = 0
    if D_base == 0:
        D_bonus = 0

    diag = {
        "A_territories_total": int(total_A),
        "D_territories_total": int(total_D),
        "A_base": int(A_base),
        "D_base": int(D_base),
        "A_bonus": int(A_bonus),
        "D_bonus": int(D_bonus),
    }
    return int(A_base + A_bonus), int(D_base + D_bonus), diag


def global_state_signature(
    state: GlobalState,
    *,
    node_indices: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[int, str, int], ...]:
    if node_indices is None:
        nodes = [int(i) for i in Board.node_to_territory_dict.keys()]
    else:
        nodes = [int(i) for i in node_indices]
    return tuple(
        (int(i), str(state.nodes[int(i)].owner), max(1, int(state.nodes[int(i)].troops)))
        for i in sorted(nodes)
        if 0 <= int(i) < len(state.nodes)
    )


def normalize_particle_weights(particles: Sequence[TransitionParticle]) -> List[TransitionParticle]:
    particles = list(particles or [])
    if not particles:
        return []
    total = float(sum(max(0.0, float(p.weight)) for p in particles))
    if total <= 0.0:
        w = 1.0 / float(len(particles))
        return [TransitionParticle(p.state, w, p.path_id, p.parent_path_id, p.turn, p.diagnostics) for p in particles]
    return [
        TransitionParticle(p.state, max(0.0, float(p.weight)) / total, p.path_id, p.parent_path_id, p.turn, p.diagnostics)
        for p in particles
    ]


def initialize_particle_population(
    *,
    initial_global_state: GlobalState,
    population_size: int,
    start_path_id: int = 0,
) -> List[TransitionParticle]:
    population_size = int(population_size)
    if population_size <= 0:
        raise ValueError("population_size must be >= 1")
    w = 1.0 / float(population_size)
    return [
        TransitionParticle(
            state=initial_global_state,
            weight=w,
            path_id=int(start_path_id) + i,
            parent_path_id=None,
            turn=0,
            diagnostics=(),
        )
        for i in range(population_size)
    ]


def particle_effective_sample_size(particles: Sequence[TransitionParticle]) -> float:
    particles = normalize_particle_weights(particles)
    if not particles:
        return 0.0
    denom = float(sum(float(p.weight) ** 2 for p in particles))
    return 0.0 if denom <= 0.0 else float(1.0 / denom)


def resample_particles_to_fixed_population(
    particles: Sequence[TransitionParticle],
    *,
    population_size: int,
    rng: Optional[np.random.Generator] = None,
    mode: str = "systematic",
    next_path_id: int = 0,
    turn: Optional[int] = None,
) -> Tuple[List[TransitionParticle], int]:
    population_size = int(population_size)
    if population_size <= 0:
        raise ValueError("population_size must be >= 1")
    particles = normalize_particle_weights(particles)
    if not particles:
        raise ValueError("Cannot resample an empty particle distribution")
    probs = np.asarray([float(p.weight) for p in particles], dtype=float)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("Cannot resample particles with zero total weight")
    probs = probs / total
    gen = rng if rng is not None else np.random.default_rng()

    if mode == "systematic":
        u0 = float(gen.random()) / float(population_size)
        positions = u0 + (np.arange(population_size, dtype=float) / float(population_size))
        cumulative = np.cumsum(probs)
        idxs = np.searchsorted(cumulative, positions, side="left")
        idxs = np.minimum(idxs, len(particles) - 1)
    elif mode == "multinomial":
        idxs = gen.choice(len(particles), size=population_size, replace=True, p=probs)
    else:
        raise ValueError(f"Unsupported population resampling mode={mode!r}")

    out: List[TransitionParticle] = []
    w = 1.0 / float(population_size)
    next_id = int(next_path_id)
    for idx in idxs:
        parent = particles[int(idx)]
        out.append(
            TransitionParticle(
                state=parent.state,
                weight=w,
                path_id=next_id,
                parent_path_id=int(parent.path_id),
                turn=int(turn if turn is not None else parent.turn),
                diagnostics=parent.diagnostics,
            )
        )
        next_id += 1
    return out, next_id


def merge_duplicate_particles(
    particles: Sequence[TransitionParticle],
    *,
    node_indices: Optional[Sequence[int]] = None,
) -> List[TransitionParticle]:
    grouped: Dict[Tuple[Tuple[int, str, int], ...], TransitionParticle] = {}
    weights: Dict[Tuple[Tuple[int, str, int], ...], float] = {}
    for p in particles or ():
        sig = global_state_signature(p.state, node_indices=node_indices)
        weights[sig] = weights.get(sig, 0.0) + float(p.weight)
        prev = grouped.get(sig)
        if prev is None or float(p.weight) > float(prev.weight):
            grouped[sig] = p
    out = []
    for sig, p in grouped.items():
        out.append(TransitionParticle(p.state, weights[sig], p.path_id, p.parent_path_id, p.turn, p.diagnostics))
    return normalize_particle_weights(out)


def prune_or_resample_particles(
    particles: Sequence[TransitionParticle],
    *,
    max_particles: int,
    rng: Optional[np.random.Generator] = None,
    mode: str = "top_weight",
) -> List[TransitionParticle]:
    particles = normalize_particle_weights(particles)
    if not particles or int(max_particles) <= 0:
        return []
    max_particles = int(max_particles)
    if len(particles) <= max_particles:
        return particles
    if mode == "top_weight":
        kept = sorted(particles, key=lambda p: (-float(p.weight), int(p.path_id)))[:max_particles]
        return normalize_particle_weights(kept)
    if mode == "multinomial":
        gen = rng if rng is not None else np.random.default_rng()
        probs = np.asarray([float(p.weight) for p in particles], dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(particles)) / len(particles)
        idxs = gen.choice(len(particles), size=max_particles, replace=True, p=probs)
        out = []
        for new_i, idx in enumerate(idxs):
            p = particles[int(idx)]
            out.append(TransitionParticle(p.state, 1.0 / max_particles, new_i, p.path_id, p.turn, p.diagnostics))
        return out
    raise ValueError(f"Unknown particle resample mode={mode!r}")


def global_state_single_owner(state: GlobalState) -> Optional[str]:
    owners = {
        str(state.nodes[int(i)].owner)
        for i in Board.node_to_territory_dict.keys()
        if 0 <= int(i) < len(state.nodes)
    }
    owners = {o for o in owners if o in ("A", "D")}
    if len(owners) == 1:
        return next(iter(owners))
    return None


def summarize_particle_distribution(particles: Sequence[TransitionParticle]) -> Dict[str, Any]:
    particles = normalize_particle_weights(particles)
    totals = {
        "n_particles": int(len(particles)),
        "weight_sum": float(sum(float(p.weight) for p in particles)),
        "A_territories_expected": 0.0,
        "D_territories_expected": 0.0,
        "A_troops_expected": 0.0,
        "D_troops_expected": 0.0,
        "top_particle_weight": float(max((float(p.weight) for p in particles), default=0.0)),
        "unique_state_count": int(len({global_state_signature(p.state) for p in particles})),
    }
    for p in particles:
        w = float(p.weight)
        for idx in _all_node_indices():
            node = p.state.nodes[int(idx)]
            if node.owner == "A":
                totals["A_territories_expected"] += w
                totals["A_troops_expected"] += w * int(node.troops)
            elif node.owner == "D":
                totals["D_territories_expected"] += w
                totals["D_troops_expected"] += w * int(node.troops)
    return totals


def _commitment_counts(commitment: Commitment) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _, cont in (commitment.commitment_by_node or {}).items():
        out[str(cont)] = int(out.get(str(cont), 0) + 1)
    return out


def _simple_allocate_reinforcements_owned_nodes(
    state: GlobalState,
    *,
    owner: str,
    reinforcements: int,
) -> Tuple[GlobalState, Dict[str, Any]]:
    """Deterministic fallback allocation: stack on the owned node with most enemy neighbors."""
    reinforcements = int(reinforcements)
    if reinforcements <= 0:
        return state, {"allocation_policy": "simple_particle_fallback", "alloc_sum": 0, "alloc_nodes": 0}
    candidates: List[Tuple[int, int, int]] = []
    for idx in _all_node_indices():
        node = state.nodes[int(idx)]
        if node.owner != owner:
            continue
        terr = Board.node_to_territory_dict[int(idx)]
        enemy_neighbors = 0
        for nb in terr._neighbors:
            j = int(nb._index)
            if 0 <= j < len(state.nodes) and state.nodes[j].owner != owner:
                enemy_neighbors += 1
        candidates.append((enemy_neighbors, int(node.troops), int(idx)))
    if not candidates:
        return state, {"allocation_policy": "simple_particle_fallback", "alloc_sum": 0, "alloc_nodes": 0}
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    target = int(candidates[0][2])
    new_state = apply_reinforcement_allocation(state, {target: reinforcements})
    return new_state, {
        "allocation_policy": "simple_particle_fallback",
        "alloc_sum": int(reinforcements),
        "alloc_nodes": 1,
        "alloc": {target: int(reinforcements)},
    }


def apply_full_board_post_combat_mechanics_for_particle(
    *,
    state: GlobalState,
    players: Sequence["Players.Player"],
    commitment: Commitment,
    models_by_continent: Optional[Dict[str, Dict[str, Any]]] = None,
    continent_order: Optional[Sequence[str]] = None,
    apply_reallocation: bool = True,
    apply_reinforcements: bool = True,
) -> Tuple[GlobalState, Dict[str, Any]]:
    """
    Particle-only post-combat mechanics mirroring the deterministic full-board order:
      1. compute board reinforcements
      2. reallocate within friendly components, continent by continent
      3. allocate reinforcements boardwide

    If node-level models are unavailable, reinforcement placement falls back to
    a simple deterministic owned-frontier allocation. Combat is not affected.
    """
    continents = tuple(str(c) for c in (continent_order if continent_order is not None else list_all_continents()))
    curr = state
    reinf_A, reinf_D, reinf_diag = compute_board_reinforcements_risklike(curr)
    reallocation_diag: Dict[str, Any] = {
        "applied": bool(apply_reallocation),
        "continents": {},
        "model_guided_continents": 0,
    }
    allocation_diag: Dict[str, Any] = {
        "applied": bool(apply_reinforcements),
        "model_guided": False,
        "fallback_simple": False,
    }

    if apply_reallocation:
        for cont in continents:
            models = (models_by_continent or {}).get(cont)
            if models is None:
                reallocation_diag["continents"][cont] = {"skipped": "missing_node_model"}
                continue
            full_graph = build_full_graph(cont)
            apply_global_state_to_board(curr, players)
            battle_graph = agop.build_continent_battle_graph(
                cont,
                players,
                debug=False,
                commitment_map=getattr(commitment, "commitment_by_node", None),
            )
            macro = compute_macro_features_from_global_state(curr, battle_graph, full_graph)
            before_sig = global_state_signature(curr)
            curr, diag_a = reallocate_troops_within_friendly_components(
                global_state=curr,
                battle_graph=battle_graph,
                full_graph=full_graph,
                models_bundle=models,
                macro_features=macro,
                owner="A",
                attack_perspective="P1_as_attacker",
            )
            curr, diag_d = reallocate_troops_within_friendly_components(
                global_state=curr,
                battle_graph=battle_graph,
                full_graph=full_graph,
                models_bundle=models,
                macro_features=macro,
                owner="D",
                attack_perspective="P1_as_attacker",
            )
            reallocation_diag["model_guided_continents"] += 1
            reallocation_diag["continents"][cont] = {
                "owner_A": diag_a,
                "owner_D": diag_d,
                "state_changed": bool(before_sig != global_state_signature(curr)),
            }

    if apply_reinforcements:
        if models_by_continent:
            curr2, alloc = allocate_reinforcements_boardwide(
                global_state=curr,
                players=players,
                models_by_continent=models_by_continent,
                commitment=commitment,
                reinf_A=int(reinf_A),
                reinf_D=int(reinf_D),
            )
            curr = curr2
            allocation_diag.update(alloc)
            allocation_diag["model_guided"] = True
            allocation_diag["allocation_policy"] = "model_guided_existing_helper"
        else:
            curr, alloc_a = _simple_allocate_reinforcements_owned_nodes(curr, owner="A", reinforcements=int(reinf_A))
            curr, alloc_d = _simple_allocate_reinforcements_owned_nodes(curr, owner="D", reinforcements=int(reinf_D))
            allocation_diag.update({"A_alloc": alloc_a, "D_alloc": alloc_d})
            allocation_diag["fallback_simple"] = True
            allocation_diag["allocation_policy"] = "simple_particle_fallback"

    diag = {
        "turn_mechanics_applied": True,
        "turn_mechanics_mode": "match_deterministic_full_board_ML",
        "reallocation_applied": bool(apply_reallocation),
        "reinforcements_applied": bool(apply_reinforcements),
        "A_reinforcements": int(reinf_A),
        "D_reinforcements": int(reinf_D),
        "reallocation_diag": reallocation_diag,
        "reinforcement_diag": reinf_diag,
        "allocation_diag": allocation_diag,
        "commitment_n_nodes": int(len(commitment.commitment_by_node or {})),
        "commitment_by_continent": _commitment_counts(commitment),
    }
    return curr, diag


# ---------------------------------------------------------------------
# 3) Commitment policy
# ---------------------------------------------------------------------

def _continent_node_set(continent: str) -> set[int]:
    return {int(t._index) for t in Board.continent_territory_dict[continent]}


def _outside_adjacent_attacker_nodes(continent: str, players: Sequence["Players.Player"]) -> set[int]:
    """
    Matches the *idea* in your battle_graph builder:
      attacker nodes are allowed to be either:
        - inside the continent
        - OR just outside but adjacent to continent and owned by current attacker
    """
    attacker = players[0]
    cont_terrs = Board.continent_territory_dict[continent]
    out: set[int] = set()
    for terr in cont_terrs:
        for neigh in terr._neighbors:
            if neigh._continent != continent and getattr(neigh, "_owner", None) is attacker:
                out.add(int(neigh._index))
    return out


def build_commitment_greedy_frontier(
    players: Sequence["Players.Player"],
    *,
    only_nodes_with_troops_gt1: bool = True,
) -> Commitment:
    """
    Deterministic commitment rule (first pass, conservative):

    For each outside-adjacent attacker node that touches >= 1 continent,
    commit it to exactly one continent:

      score(continent) = number of *enemy* neighbors inside that continent

    Tie-break: max score, then lexicographically smallest continent name.

    This prevents the same outside node from being counted by multiple continent graphs.
    """
    attacker = players[0]

    # Candidate outside nodes: those owned by attacker that neighbor a continent node.
    # We construct by scanning all continents (cheap at this scale).
    node_to_conts: Dict[int, List[str]] = {}
    for cont in list_all_continents():
        for n in _outside_adjacent_attacker_nodes(cont, players):
            node_to_conts.setdefault(int(n), []).append(cont)

    commitment: Dict[int, str] = {}
    for node_idx, conts in node_to_conts.items():
        terr = Board.node_to_territory_dict[node_idx]
        if getattr(terr, "_owner", None) is not attacker:
            continue
        troops = int(getattr(terr, "_troops", 0))
        if only_nodes_with_troops_gt1 and troops <= 1:
            continue

        best_cont = None
        best_score = None

        for cont in conts:
            cont_set = _continent_node_set(cont)
            score = 0
            for neigh in terr._neighbors:
                if int(neigh._index) not in cont_set:
                    continue
                neigh_owner = getattr(neigh, "_owner", None)
                if neigh_owner is not None and neigh_owner is not attacker:
                    score += 1

            key = (score, cont)
            if best_score is None or key > (best_score, best_cont):
                best_score = score
                best_cont = cont

        if best_cont is not None:
            commitment[node_idx] = best_cont

    return Commitment(commitment_by_node=commitment)


# ---------------------------------------------------------------------
# 4) One full-board "turn" (one attacker acts across all continents)
# ---------------------------------------------------------------------

def _swap_roles_in_state(global_state: GlobalState) -> GlobalState:
    # Lightweight swap: A<->D in node.owner.
    nodes = []
    for n in global_state.nodes:
        if n.owner == "A":
            nodes.append(type(n)(owner="D", troops=n.troops))
        elif n.owner == "D":
            nodes.append(type(n)(owner="A", troops=n.troops))
        else:
            nodes.append(n)
    return GlobalState(nodes=tuple(nodes))


def apply_combat_expectations_all_continents(
    global_state: GlobalState,
    players: Sequence["Players.Player"],
    models_by_continent: Dict[str, Dict[str, Any]],
    *,
    commitment: Commitment,
    attacker_is_player1: bool,
) -> GlobalState:
    """
    Apply ML transition sequentially continent-by-continent.

    Notes:
      - This is a *design choice* (operator splitting): each continent uses the current
        state as input, then writes back its expectation update. This is deterministic.
      - The commitment map is computed once per turn, so outside-adjacent nodes aren't double-counted.
    """
    import logging
    from project_risk.infrastructure.log_config import get_logger

    log_roll = get_logger("risk.rollout")

    state = global_state

    # Perspective: your ML models assume "A" is the attacker in features.
    # If current attacker is player2, we swap view for inference, then swap back.
    persp_state = state if attacker_is_player1 else _swap_roles_in_state(state)
    persp_players = players if attacker_is_player1 else [players[1], players[0]]

    # Sync Board to perspective view (battle_graph builder reads Board)
    apply_global_state_to_board(persp_state, persp_players)

    for cont in list_all_continents():
        models = models_by_continent.get(cont)
        if models is None:
            # allow skipping if strict=False in loader
            continue

        full_graph = build_full_graph(cont)

        # Build battle graph WITH commitment_map (patched in agop)
        battle_graph = agop.build_continent_battle_graph(
            cont,
            persp_players,
            debug=False,
            commitment_map=getattr(commitment, 'commitment_by_node', None),
        )
        if log_roll.isEnabledFor(logging.DEBUG):
            snap_pre = snapshot_continent_ownership(persp_state).get(str(cont), {})
            ctx_map = getattr(commitment, "commitment_by_node", None) or {}
            committed_to_cont = sum(1 for _, cname in ctx_map.items() if cname == cont)
            log_roll.debug(
                "[combat] cont=%s pre=%s committed_outside_nodes=%d battle_nodes=%d edges=%d",
                cont,
                snap_pre,
                int(committed_to_cont),
                int(battle_graph.number_of_nodes()),
                int(battle_graph.number_of_edges()),
            )


        macro = compute_macro_features_from_global_state(persp_state, battle_graph, full_graph)

        persp_state = apply_expectations_as_state(
            global_state=persp_state,
            battle_graph=battle_graph,
            full_graph=full_graph,
            models_bundle=models,
            macro_features=macro,
            attack_perspective="P1_as_attacker",  # stable inference schema
        )

        if log_roll.isEnabledFor(logging.DEBUG):
            snap_post = snapshot_continent_ownership(persp_state).get(str(cont), {})
            log_roll.debug('[combat] cont=%s post=%s', cont, snap_post)

        # Keep Board in sync after each continent update (important for later continents)
        apply_global_state_to_board(persp_state, persp_players)

    # Convert back to canonical view
    return persp_state if attacker_is_player1 else _swap_roles_in_state(persp_state)


# ---------------------------------------------------------------------
# 5) Allocate reinforcements across continents
# ---------------------------------------------------------------------

def _continent_friendly_nodes(state: GlobalState, continent: str, owner: str) -> List[int]:
    nodes = [int(t._index) for t in Board.continent_territory_dict[continent]]
    return [i for i in nodes if state.nodes[int(i)].owner == owner]


def allocate_reinforcements_boardwide(
    global_state: GlobalState,
    players: Sequence["Players.Player"],
    models_by_continent: Dict[str, Dict[str, Any]],
    *,
    commitment: Commitment,
    reinf_A: int,
    reinf_D: int,
) -> Tuple[GlobalState, Dict[str, Any]]:
    """
    Allocate A and D reinforcements across the whole board, but in a continent-aware way.

    Default policy:
      - Split each player's reinforcements proportionally by "frontier pressure" across continents:
            pressure(cont, owner) = number of friendly continent nodes that are frontier nodes
      - Within each continent: use your existing greedy-cheapest allocator on that continent's graphs.

    This keeps your existing learned capture model in the loop while scaling to full board.
    """
    diag: Dict[str, Any] = {"A_alloc": {}, "D_alloc": {}}
    state = global_state

    # Helper: compute a simple pressure score from current state + full_graph local features
    from project_risk.mathematical.transition_prediction_ml.generate_data_ML import compute_local_node_features_map

    def _pressure_for(cont: str, owner: str) -> int:
        full_graph = build_full_graph(cont)
        local = compute_local_node_features_map(state, full_graph)
        cont_nodes = [int(t._index) for t in Board.continent_territory_dict[cont]]
        return int(sum(1 for i in cont_nodes if state.nodes[int(i)].owner == owner and local.get(int(i), {}).get("is_frontier_node", 0) >= 0.5))

    def _split(total: int, owner: str) -> Dict[str, int]:
        pressures = {c: _pressure_for(c, owner) for c in list_all_continents()}
        s = sum(pressures.values())
        if total <= 0:
            return {c: 0 for c in pressures}
        if s <= 0:
            # fallback: split evenly across continents where owner has at least 1 node
            eligible = [c for c in list_all_continents() if any(state.nodes[int(t._index)].owner == owner for t in Board.continent_territory_dict[c])]
            if not eligible:
                return {c: 0 for c in pressures}
            base = total // len(eligible)
            rem = total - base * len(eligible)
            out = {c: (base + (1 if i < rem else 0)) if c in eligible else 0 for i, c in enumerate(sorted(eligible))}
            return out

        # proportional integer split with deterministic remainder assignment
        raw = {c: total * pressures[c] / s for c in pressures}
        floor_alloc = {c: int(math.floor(raw[c])) for c in pressures}
        rem = total - sum(floor_alloc.values())
        # distribute remainder to largest fractional parts
        frac = sorted(((raw[c] - floor_alloc[c], c) for c in pressures), reverse=True)
        out = dict(floor_alloc)
        for k in range(rem):
            out[frac[k][1]] += 1
        return out

    split_A = _split(int(reinf_A), "A")
    split_D = _split(int(reinf_D), "D")

    diag["A_split"] = split_A
    diag["D_split"] = split_D

    # Allocate continent by continent (A then D)
    for owner, split in (("A", split_A), ("D", split_D)):
        for cont, n_reinf in split.items():
            n_reinf = int(n_reinf)
            if n_reinf <= 0:
                continue
            models = models_by_continent.get(cont)
            if models is None:
                continue

            full_graph = build_full_graph(cont)

            # Ensure Board reflects current canonical state for battle_graph construction
            apply_global_state_to_board(state, players)

            battle_graph = agop.build_continent_battle_graph(
                cont,
                players,
                debug=False,
                commitment_map=getattr(commitment, 'commitment_by_node', None),
            )

            macro = compute_macro_features_from_global_state(state, battle_graph, full_graph)

            # Candidate nodes restricted to continent nodes owned by `owner`
            cand = _continent_friendly_nodes(state, cont, owner)

            state2, alloc = allocate_reinforcements_greedy_cheapest(
                global_state=state,
                battle_graph=battle_graph,
                full_graph=full_graph,
                models_bundle=models,
                macro_features=macro,
                num_reinforcements=n_reinf,
                owner=owner,
                candidate_nodes=cand,
                attack_perspective="P1_as_attacker",
            )
            state = state2
            diag[f"{owner}_alloc"][cont] = {"n": n_reinf, "alloc_nodes": int(len(alloc)), "alloc_sum": int(sum(alloc.values()))}

    return state, diag


def simulate_multi_turns_full_board_transition_particles(
    *,
    initial_global_state: GlobalState,
    players: Sequence["Players.Player"],
    max_turns: int = 10,
    models_dir: Path | str = "models",
    transition_models_by_continent: Optional[Dict[str, Dict[str, Any]]] = None,
    particle_budget: int = 100,
    samples_per_particle_per_continent: int = 1,
    resample_mode: str = "top_weight",
    commitment_policy: str = "greedy_frontier",
    update_scope: str = "battle_plus_committed_outside",
    continent_order: Optional[Sequence[str]] = None,
    fallback_mode: str = "skip_continent",
    stop_when_one_player_owns_board: bool = False,
    random_seed: Optional[int] = None,
    return_diagnostics: bool = True,
    apply_turn_mechanics: bool = True,
    turn_mechanics_mode: str = "match_deterministic_full_board_ML",
    apply_reallocation: bool = True,
    apply_reinforcements: bool = True,
    node_models_by_continent: Optional[Dict[str, Dict[str, Any]]] = None,
    node_models_dir: Path | str = "models",
    load_node_models_for_turn_mechanics: bool = True,
    population_mode: str = "single_trajectory",
    population_size: Optional[int] = None,
    population_resample_mode: str = "systematic",
    compress_after_turn: bool = True,
) -> Dict[str, Any]:
    """
    Stochastic full-board rollout using Stage-C transition-distribution models.

    This is additive and does not replace `simulate_multi_turns_full_board_ML`.

    In fixed_population mode, compressed particle weights are empirical Monte
    Carlo frequencies after merging identical full-board outcomes. They
    approximate P(full board state after turn t) under the transition models,
    commitment policy, turn mechanics, sequential continent processing, and
    selected population size. These are approximate probabilities, not exact
    probabilities.
    """
    if int(samples_per_particle_per_continent) != 1:
        raise NotImplementedError("Stage D currently supports samples_per_particle_per_continent=1 only.")
    if int(particle_budget) <= 0:
        raise ValueError("particle_budget must be >= 1")
    if fallback_mode not in ("skip_continent", "error"):
        raise ValueError(f"Unsupported fallback_mode={fallback_mode!r}")
    if commitment_policy != "greedy_frontier":
        raise ValueError(f"Unsupported commitment_policy={commitment_policy!r}")
    if turn_mechanics_mode != "match_deterministic_full_board_ML":
        raise ValueError(f"Unsupported turn_mechanics_mode={turn_mechanics_mode!r}")
    if population_mode not in ("single_trajectory", "fixed_population"):
        raise ValueError(f"Unsupported population_mode={population_mode!r}")
    resolved_population_size = int(population_size if population_size is not None else particle_budget)
    if resolved_population_size <= 0:
        raise ValueError("population_size must be >= 1")

    models = transition_models_by_continent
    if models is None:
        models = load_transition_distribution_models_by_continent(models_dir=models_dir, strict=(fallback_mode == "error"))
    node_models = node_models_by_continent
    if apply_turn_mechanics and node_models is None and load_node_models_for_turn_mechanics:
        if hasattr(joblib, "load"):
            node_models = load_models_by_continent(node_models_dir, strict=False)
        else:
            node_models = {}

    continents = tuple(str(c) for c in (continent_order if continent_order is not None else list_all_continents()))
    rng = np.random.default_rng(random_seed)
    if population_mode == "fixed_population":
        particles = initialize_particle_population(
            initial_global_state=initial_global_state,
            population_size=resolved_population_size,
            start_path_id=0,
        )
        next_path_id = resolved_population_size
    else:
        particles = [
            TransitionParticle(state=initial_global_state, weight=1.0, path_id=0, parent_path_id=None, turn=0, diagnostics=())
        ]
        next_path_id = 1
    history: List[Dict[str, Any]] = []

    for turn in range(1, int(max_turns) + 1):
        attacker_is_player1 = (turn % 2 == 1)
        attacker_owner = "A" if attacker_is_player1 else "D"
        compressed_before = len(particles)
        ess_before = particle_effective_sample_size(particles)
        if population_mode == "fixed_population":
            if turn == 1:
                working_particles = particles
                working_population_size = len(working_particles)
            else:
                working_particles, next_path_id = resample_particles_to_fixed_population(
                    particles,
                    population_size=resolved_population_size,
                    rng=rng,
                    mode=population_resample_mode,
                    next_path_id=next_path_id,
                    turn=turn - 1,
                )
                working_population_size = len(working_particles)
        else:
            working_particles = particles
            working_population_size = len(working_particles)
        before = len(working_particles)
        new_particles: List[TransitionParticle] = []
        turn_fallbacks = 0
        turn_continent_diags: List[Dict[str, Any]] = []

        for particle in working_particles:
            if global_state_single_owner(particle.state) is not None:
                combined_diags = particle.diagnostics + (
                    {
                        "turn": int(turn),
                        "terminal_absorbing": True,
                        "continent_diags": tuple(),
                        "mechanics_diag": {
                            "turn_mechanics_applied": False,
                            "turn_mechanics_mode": str(turn_mechanics_mode),
                            "terminal_absorbing": True,
                        },
                    },
                )
                new_particles.append(
                    TransitionParticle(
                        state=particle.state,
                        weight=float(particle.weight),
                        path_id=next_path_id,
                        parent_path_id=int(particle.path_id),
                        turn=int(turn),
                        diagnostics=combined_diags if return_diagnostics else (),
                    )
                )
                next_path_id += 1
                continue
            perspective_state = particle.state if attacker_is_player1 else _swap_roles_in_state(particle.state)
            perspective_players = players if attacker_is_player1 else [players[1], players[0]]
            apply_global_state_to_board(perspective_state, perspective_players)

            commitment = build_commitment_greedy_frontier(perspective_players)
            state_work = perspective_state
            particle_diags: List[Dict[str, Any]] = []

            for cont in continents:
                model = (models or {}).get(cont)
                if model is None:
                    turn_fallbacks += 1
                    diag = {
                        "path_id": int(particle.path_id),
                        "continent_name": cont,
                        "fallback": "missing_model",
                        "fallback_mode": fallback_mode,
                    }
                    turn_continent_diags.append(diag)
                    particle_diags.append(diag)
                    if fallback_mode == "error":
                        raise FileNotFoundError(f"Missing transition-distribution model for continent {cont!r}")
                    continue

                apply_global_state_to_board(state_work, perspective_players)
                full_graph = build_full_graph(cont)
                battle_graph = agop.build_continent_battle_graph(
                    cont,
                    perspective_players,
                    debug=False,
                    commitment_map=getattr(commitment, "commitment_by_node", None),
                )
                try:
                    state_work, diag = sample_and_apply_transition_distribution_for_continent(
                        global_state=state_work,
                        battle_graph=battle_graph,
                        full_graph=full_graph,
                        model_bundle=model,
                        continent_name=cont,
                        commitment_map=getattr(commitment, "commitment_by_node", None),
                        update_scope=update_scope,
                        attack_perspective="P1_as_attacker",
                        rng=rng,
                    )
                    diag = dict(diag)
                    diag["path_id"] = int(particle.path_id)
                    diag["fallback"] = None
                except Exception as e:
                    turn_fallbacks += 1
                    diag = {
                        "path_id": int(particle.path_id),
                        "continent_name": cont,
                        "fallback": "transition_error",
                        "fallback_mode": fallback_mode,
                        "error": f"{type(e).__name__}: {e}",
                    }
                    if fallback_mode == "error":
                        raise
                turn_continent_diags.append(diag)
                particle_diags.append(diag)

            mechanics_diag = {
                "turn_mechanics_applied": False,
                "turn_mechanics_mode": str(turn_mechanics_mode),
                "reallocation_applied": False,
                "reinforcements_applied": False,
                "A_reinforcements": 0,
                "D_reinforcements": 0,
                "reallocation_diag": {},
                "reinforcement_diag": {},
                "allocation_diag": {},
                "commitment_n_nodes": int(len(commitment.commitment_by_node or {})),
                "commitment_by_continent": _commitment_counts(commitment),
            }
            if apply_turn_mechanics:
                state_work, mechanics_diag = apply_full_board_post_combat_mechanics_for_particle(
                    state=state_work,
                    players=perspective_players,
                    commitment=commitment,
                    models_by_continent=node_models,
                    continent_order=continents,
                    apply_reallocation=bool(apply_reallocation),
                    apply_reinforcements=bool(apply_reinforcements),
                )

            canonical_state = state_work if attacker_is_player1 else _swap_roles_in_state(state_work)
            combined_diags = particle.diagnostics + (
                {
                    "turn": int(turn),
                    "continent_diags": tuple(particle_diags),
                    "mechanics_diag": mechanics_diag,
                },
            )
            new_particles.append(
                TransitionParticle(
                    state=canonical_state,
                    weight=float(particle.weight),
                    path_id=next_path_id,
                    parent_path_id=int(particle.path_id),
                    turn=int(turn),
                    diagnostics=combined_diags if return_diagnostics else (),
                )
            )
            next_path_id += 1

        if compress_after_turn:
            merged_particles = merge_duplicate_particles(new_particles)
        else:
            merged_particles = normalize_particle_weights(new_particles)
        support_after_merge = len(merged_particles)
        duplicate_trajectories_merged = max(0, len(new_particles) - support_after_merge)
        ess_after_merge = particle_effective_sample_size(merged_particles)
        particles = prune_or_resample_particles(merged_particles, max_particles=int(particle_budget), rng=rng, mode=resample_mode)
        support_after_pruning = len(particles)
        particles = normalize_particle_weights(particles)
        summary = summarize_particle_distribution(particles)
        mechanics_diags = [
            d.get("mechanics_diag", {})
            for p in new_particles
            for d in (p.diagnostics[-1:] if p.diagnostics else ())
        ]
        total_mech_weight = float(sum(float(p.weight) for p in new_particles)) or 1.0
        reinf_A_exp = float(
            sum(float(p.weight) * float((p.diagnostics[-1].get("mechanics_diag", {}) if p.diagnostics else {}).get("A_reinforcements", 0)) for p in new_particles)
            / total_mech_weight
        )
        reinf_D_exp = float(
            sum(float(p.weight) * float((p.diagnostics[-1].get("mechanics_diag", {}) if p.diagnostics else {}).get("D_reinforcements", 0)) for p in new_particles)
            / total_mech_weight
        )
        history.append(
            {
                "turn": int(turn),
                "attacker_owner": attacker_owner,
                "population_mode": str(population_mode),
                "working_population_size": int(working_population_size),
                "compressed_support_before_resampling": int(compressed_before),
                "compressed_support_after_turn": int(support_after_merge),
                "compressed_support_after_pruning": int(support_after_pruning),
                "ess_before_population_resampling": float(ess_before),
                "ess_after_turn_merge": float(ess_after_merge),
                "duplicate_trajectories_merged": int(duplicate_trajectories_merged),
                "n_particles_before": int(before),
                "n_particles_after": int(len(particles)),
                "particle_weight_sum": float(summary["weight_sum"]),
                "top_particle_weight": float(summary["top_particle_weight"]),
                "top_state_weights": tuple(
                    float(p.weight)
                    for p in sorted(particles, key=lambda p: (-float(p.weight), int(p.path_id)))[:5]
                ),
                "num_fallbacks": int(turn_fallbacks),
                "turn_mechanics_applied": bool(apply_turn_mechanics),
                "turn_mechanics_mode": str(turn_mechanics_mode),
                "A_reinforcements_expected": reinf_A_exp,
                "D_reinforcements_expected": reinf_D_exp,
                "continent_diags": turn_continent_diags if return_diagnostics else [],
                "mechanics_diags": mechanics_diags if return_diagnostics else [],
                "A_territories_expected": float(summary["A_territories_expected"]),
                "D_territories_expected": float(summary["D_territories_expected"]),
                "A_troops_expected": float(summary["A_troops_expected"]),
                "D_troops_expected": float(summary["D_troops_expected"]),
                "summary": summary,
            }
        )

        if stop_when_one_player_owns_board and particles:
            if all(global_state_single_owner(p.state) is not None for p in particles):
                break

    return {
        "particles": particles,
        "history": history,
        "config": {
            "max_turns": int(max_turns),
            "particle_budget": int(particle_budget),
            "samples_per_particle_per_continent": int(samples_per_particle_per_continent),
            "resample_mode": str(resample_mode),
            "commitment_policy": str(commitment_policy),
            "update_scope": str(update_scope),
            "continent_order": continents,
            "fallback_mode": str(fallback_mode),
            "random_seed": random_seed,
            "apply_turn_mechanics": bool(apply_turn_mechanics),
            "turn_mechanics_mode": str(turn_mechanics_mode),
            "apply_reallocation": bool(apply_reallocation),
            "apply_reinforcements": bool(apply_reinforcements),
            "load_node_models_for_turn_mechanics": bool(load_node_models_for_turn_mechanics),
            "population_mode": str(population_mode),
            "population_size": int(resolved_population_size),
            "population_resample_mode": str(population_resample_mode),
            "compress_after_turn": bool(compress_after_turn),
        },
    }


# ---------------------------------------------------------------------
# 6) Public entry point: multi-turn full-board simulation
# ---------------------------------------------------------------------

def simulate_multi_turns_full_board_ML(
    initial_global_state: GlobalState,
    players: Sequence["Players.Player"],
    *,
    models_dir: Path | str = "models",
    max_turns: int = 10,
    stop_when_one_player_owns_board: bool = False,
    commitment_policy: str = "greedy_frontier",
    # --- Strategic planning hooks (optional) ---
    offense_target_continent: Optional[str] = None,
    offense_x_per_territory: float = 1.0,
    offense_continent_bonus: Optional[Dict[str, float]] = None,
    offense_verify_topk: int = 3,
    offense_candidate_nodes: int = 25,
    offense_beam_commitment: int = 6,

) -> List[Dict[str, Any]]:
    """
    Simulate multiple turns on the FULL board.

    Turn semantics:
      - Turn 1: player1 attacks (A)
      - Turn 2: player2 attacks (D)  [implemented via perspective swap during combat]
      - repeat...

    Per turn:
      1) Build commitment map for current attacker (so outside-adjacent troops are used once)
      2) Apply combat expectations across all continents sequentially
      3) Compute board-level reinforcements for both players
      4) Reallocate existing troops within friendly components (per continent, conservative)
      5) Allocate new reinforcements boardwide using continent-aware split policy

    Returns a history list of dictionaries (one per turn).
    """
    models_by_continent = load_models_by_continent(models_dir, strict=True)

    history: List[Dict[str, Any]] = []
    state = initial_global_state

    for turn in range(1, int(max_turns) + 1):
        attacker_is_player1 = (turn % 2 == 1)

        log_runner = get_logger('risk.runner', step=turn)
        if log_runner.isEnabledFor(logging.DEBUG):
            log_runner.debug('TURN_START attacker_is_player1=%s', attacker_is_player1)
            log_runner.debug('state_before=%s', snapshot_global_state(state))
            log_runner.debug('continents_before=%s', snapshot_continent_ownership(state))

        # Keep Board synced to canonical state before commitment building (reads Board)
        apply_global_state_to_board(state, players)

        if commitment_policy == "greedy_frontier":
            commitment = build_commitment_greedy_frontier(players if attacker_is_player1 else [players[1], players[0]])
        else:
            raise ValueError(f"Unknown commitment_policy={commitment_policy!r}")

        log_commitment(commitment, turn=turn)

        # 1) Combat expectations (continent-by-continent)
        state_after_combat = apply_combat_expectations_all_continents(
            global_state=state,
            players=players,
            models_by_continent=models_by_continent,
            commitment=commitment,
            attacker_is_player1=attacker_is_player1,
        )

        # 2) Reinforcements from actual board state (canonical)
        reinf_A, reinf_D, reinf_diag = compute_board_reinforcements_risklike(state_after_combat)

        # 3) Light troop reallocation inside each continent (optional but keeps behavior similar)
        # NOTE: This is conservative: it only reallocates within friendly components, continent by continent.
        #       You can later replace with a global movement model.
        state_realloc = state_after_combat
        for cont in list_all_continents():
            models = models_by_continent.get(cont)
            if models is None:
                continue
            full_graph = build_full_graph(cont)

            apply_global_state_to_board(state_realloc, players)
            battle_graph = agop.build_continent_battle_graph(
                cont,
                players,
                debug=False,
                commitment_map=getattr(commitment, 'commitment_by_node', None),
            )

            macro = compute_macro_features_from_global_state(state_realloc, battle_graph, full_graph)

            # Realloc A then D (canonical)
            state_realloc, _ = reallocate_troops_within_friendly_components(
                global_state=state_realloc,
                battle_graph=battle_graph,
                full_graph=full_graph,
                models_bundle=models,
                macro_features=macro,
                owner="A",
                attack_perspective="P1_as_attacker",
            )
            state_realloc, _ = reallocate_troops_within_friendly_components(
                global_state=state_realloc,
                battle_graph=battle_graph,
                full_graph=full_graph,
                models_bundle=models,
                macro_features=macro,
                owner="D",
                attack_perspective="P1_as_attacker",
            )

        # 4) Allocate reinforcements boardwide with continent split
        #
        # Optional: If offense_target_continent is set, we treat the CURRENT ATTACKER as
        # executing a continent-focused "max effort" plan for its reinforcements.
        # This is an early infrastructure hook for strategic decision comparison.
        if offense_target_continent is not None:
            from offense_planner import (
                OffensiveUtilityWeights,
                OffensePlannerConfig,
                plan_max_effort_offense_for_continent,
            )

            weights = OffensiveUtilityWeights(
                x_per_territory=float(offense_x_per_territory),
                continent_bonus=(offense_continent_bonus or {}),
            )
            cfg = OffensePlannerConfig(
                topk_candidate_nodes=int(offense_candidate_nodes),
                beam_width_commitment=int(offense_beam_commitment),
                verify_topk=int(offense_verify_topk),
            )

            # Decide which side is the "planner" this turn.
            planner_owner = 'A' if attacker_is_player1 else 'D'

            # Planner picks a commitment + allocation for the current attacker only.
            plan = plan_max_effort_offense_for_continent(
                target_continent=str(offense_target_continent),
                global_state=state_realloc,
                players=players,
                models_by_continent=models_by_continent,
                owner=planner_owner,
                reinforcements=(reinf_A if planner_owner == 'A' else reinf_D),
                weights=weights,
                config=cfg,
            )

            # Apply allocation for planner side (other side allocated by default routine).
            if planner_owner == 'A':
                state_planned = apply_reinforcement_allocation(state_realloc, plan.reinf_alloc)
                # Allocate defender reinforcements using the existing heuristic
                state_next, alloc_diag = allocate_reinforcements_boardwide(
                    global_state=state_planned,
                    players=players,
                    models_by_continent=models_by_continent,
                    commitment=plan.commitment,
                    reinf_A=0,
                    reinf_D=reinf_D,
                )
                alloc_diag['offense_plan'] = plan.to_dict()
            else:
                state_planned = apply_reinforcement_allocation(state_realloc, plan.reinf_alloc)
                state_next, alloc_diag = allocate_reinforcements_boardwide(
                    global_state=state_planned,
                    players=players,
                    models_by_continent=models_by_continent,
                    commitment=plan.commitment,
                    reinf_A=reinf_A,
                    reinf_D=0,
                )
                alloc_diag['offense_plan'] = plan.to_dict()
        else:
            state_next, alloc_diag = allocate_reinforcements_boardwide(
                global_state=state_realloc,
                players=players,
                models_by_continent=models_by_continent,
                commitment=commitment,
                reinf_A=reinf_A,
                reinf_D=reinf_D,
            )


        # 5) History snapshot
        # Quick board totals for reporting
        A_terr = 0
        D_terr = 0
        A_troops = 0
        D_troops = 0
        for idx in _all_node_indices():
            node = state_next.nodes[idx]
            if node.owner == "A":
                A_terr += 1
                A_troops += int(node.troops)
            elif node.owner == "D":
                D_terr += 1
                D_troops += int(node.troops)

        history.append(
            {
                "turn": int(turn),
                "attacker_is_player1": bool(attacker_is_player1),
                "A_territories": int(A_terr),
                "D_territories": int(D_terr),
                "A_troops_total": int(A_troops),
                "D_troops_total": int(D_troops),
                "A_reinforcements": int(reinf_A),
                "D_reinforcements": int(reinf_D),
                "reinforcement_diag": reinf_diag,
                "allocation_diag": alloc_diag,
                "commitment_n_nodes": int(len(commitment.commitment_by_node)),
            }
        )

        state = state_next

        if stop_when_one_player_owns_board:
            if D_terr == 0 or A_terr == 0:
                break

    return history
