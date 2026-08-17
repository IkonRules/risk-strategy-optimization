# predict_future_states_ML.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Optional, Any, Tuple
from collections import defaultdict


import logging

# Subsystem loggers (verbosity controlled by your log_config switchboard)
log_runner = logging.getLogger("risk.runner")
log_rollout = logging.getLogger("risk.rollout")
log_predict = logging.getLogger("risk.predict")
log_battle_graph = logging.getLogger("risk.battle_graph")
log_sampler = logging.getLogger("risk.sampler")


import numpy as np
import pandas as pd
import joblib
import networkx as nx

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.libraries.create_library import lowest_lib_node_max
from project_risk.mathematical.transition_prediction_ml.generate_data_ML import (
    _battle_graph_nodes, compute_full_graph_metrics, 
    compute_territory_ratio, compute_effectiveness_metrics,
    compute_attacker_troop_distribution, compute_troops_cv,
    compute_troops_gini, apply_global_state_to_board, 
    swap_roles_in_global_state, compute_local_node_features_map)

from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState, NodeState

from project_risk.mathematical.transition_prediction_ml.transition_distribution_ML import (
    predict_successor_distribution_from_example,
    sample_successor_signatures_from_distribution,
)
from project_risk.mathematical.transition_prediction_ml.generate_data_ML import normalize_state_signature, signature_to_node_state_map




# ----------------------------------------------------------------------
# 1) Helpers for computing macro state
# ----------------------------------------------------------------------

def _build_macro_features_for_state(
    players: Sequence["Players.Player"],
    battle_graph: Any,
    full_graph: Any,
) -> Tuple[GlobalState, Dict[str, Any]]:
    """
    Recompute the SAME macro features used when training the node models.
    This mirrors the logic in train_ML.
    *** USEd OLDER CODE LOGIC, USED ONLY FOR ONE STEP PREDICTION ***
    """

    global_state = agop.build_global_state_for_board(players)

    battle_node_indices = _battle_graph_nodes(battle_graph)
    battle_total_territory_count = len(battle_node_indices)

    # --- Full-graph + effectiveness metrics ---
    full_metrics = compute_full_graph_metrics(full_graph, players)
    effectiveness_metrics = compute_effectiveness_metrics(
        full_graph, battle_graph, players
    )

    # --- Battle realized territory ratio ---
    battle_realized_attacker_territory_ratio = compute_territory_ratio(
        global_state, battle_node_indices
    )

    # --- Battle-level available troops, counts, etc. (same as run_macro_experiment) ---
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
        battle_realized_attacker_available_troops_ratio = (
            battle_attacker_available / battle_initial_attacker_troops_count
        )
    else:
        battle_realized_attacker_available_troops_ratio = 0.0

    # Battle distribution metrics
    battle_attacker_troops_array = compute_attacker_troop_distribution(
        global_state, battle_node_indices
    )
    battle_realized_attacker_troops_distribution_cv = compute_troops_cv(
        battle_attacker_troops_array
    )
    battle_realized_attacker_troops_distribution_gini = compute_troops_gini(
        battle_attacker_troops_array
    )

    # Full-graph "mobilization" metric (as in run_macro_experiment)
    full_nodes = list(full_graph.nodes())
    full_attacker_total = sum(
        global_state.nodes[idx].troops
        for idx in full_nodes
        if global_state.nodes[idx].owner == "A"
    )
    if full_attacker_total > 0:
        full_attacker_available_troops_ratio = (
            battle_attacker_available / full_attacker_total
        )
    else:
        full_attacker_available_troops_ratio = 0.0

    macro_features: Dict[str, Any] = {
        # battle-level
        "battle_realized_attacker_territory_ratio": float(
            battle_realized_attacker_territory_ratio
        ),
        "battle_realized_attacker_available_troops_ratio": float(
            battle_realized_attacker_available_troops_ratio
        ),
        "battle_initial_attacker_territory_count":
            battle_initial_attacker_territory_count,
        "battle_initial_attacker_troops_count":
            battle_initial_attacker_troops_count,
        "battle_total_territory_count": battle_total_territory_count,
        "battle_total_troops_count": battle_total_troops_count,
        "battle_realized_attacker_troops_distribution_cv":
            float(battle_realized_attacker_troops_distribution_cv),
        "battle_realized_attacker_troops_distribution_gini":
            float(battle_realized_attacker_troops_distribution_gini),

        # full-graph
        "full_realized_attacker_available_troops_ratio": float(
            full_attacker_available_troops_ratio
        ),
    }

    macro_features.update(full_metrics)
    macro_features.update(effectiveness_metrics)

    return global_state, macro_features


def _build_node_feature_df_from_state(
    global_state: GlobalState,
    battle_graph,
    full_graph,
    macro_features: Dict[str, Any],
    models_bundle: Dict[str, Any],
    *,
    attack_perspective: str = "P1_as_attacker",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build node-level feature DF + X for the CURRENT global_state.

    Compatibility notes:
      - Training included the categorical feature `attack_perspective` (e.g., P1_as_attacker / P2_as_attacker).
        For inference we must provide it consistently, otherwise the one-hot columns will be all zeros.
      - Uses the same `feature_cols` that were used when training the ML models.
      - Sanitizes NaN/inf so sklearn models don't choke.

    Local neighborhood features are injected via `compute_local_node_features_map`.
    """
    feature_cols = models_bundle["feature_cols"]

    try:
        full_nodes_iter = full_graph.nodes()
    except TypeError:
        full_nodes_iter = full_graph.nodes
    full_nodes = list(full_nodes_iter)

    battle_nodes = set(_battle_graph_nodes(battle_graph))

    # Local node features map (frontier pressure etc.)
    local_map = compute_local_node_features_map(global_state, full_graph)

    rows: list[dict[str, Any]] = []
    for node_idx in full_nodes:
        node: NodeState = global_state.nodes[node_idx]
        row: Dict[str, Any] = {
            "node_index": int(node_idx),
            "attack_perspective": attack_perspective,
            "initial_owner": node.owner,          # "A" or "D"
            "initial_troops": int(node.troops),   # troops in the current global_state
            "is_battle_node": int(node_idx in battle_nodes),
        }

        # Local features (safe defaults)
        lf = local_map.get(int(node_idx), {})
        row.update(lf)

        # Macro features identical for all nodes in the state
        row.update(macro_features)

        rows.append(row)

    df_nodes = pd.DataFrame(rows)

    # One-hot encode categorical features used during training
    df_nodes = pd.get_dummies(
        df_nodes,
        columns=["initial_owner", "attack_perspective"],
        prefix=["init_owner", "attack_perspective"],
    )

    # Ensure all training features exist
    for col in feature_cols:
        if col not in df_nodes.columns:
            df_nodes[col] = 0.0

    # sanitize
    df_nodes[feature_cols] = df_nodes[feature_cols].replace([np.inf, -np.inf], np.nan)
    df_nodes[feature_cols] = df_nodes[feature_cols].fillna(0.0)

    X = df_nodes[feature_cols].to_numpy(dtype=float)
    return df_nodes, X



def apply_expectations_as_state(
    global_state: GlobalState,
    battle_graph,
    full_graph,
    models_bundle: Dict[str, Any],
    macro_features: Dict[str, Any],
    *,
    attack_perspective: str = "P1_as_attacker",
    # -----------------------------
    # PATCH: rich diagnostics
    # -----------------------------
    continent_name: Optional[str] = None,
    state_id: Optional[int] = None,
    step: Optional[int] = None,
    scen: Optional[int] = None,
    log_top_k: int = 12,
    return_diag: bool = False,
) -> Any:
    """
    Use the ML models to produce a deterministic *next* GlobalState from the CURRENT global_state.

    Updates are restricted to contestable nodes only:
      - contestable = is_frontier_node OR is_battle_node
        (is_frontier_node comes from compute_local_node_features_map; if missing, we fall back to
         battle_graph owner-different edges.)

    This prevents interior nodes from drifting for free across turns.

    `attack_perspective` must match the training convention, otherwise the one-hot columns for
    perspective will be all zeros at inference.
    """
    capture_model_att = models_bundle["capture_model_attacker"]
    troop_model_att = models_bundle["troop_model_attacker"]
    troop_model_def = models_bundle["troop_model_defender"]

    df_nodes, X = _build_node_feature_df_from_state(
        global_state=global_state,
        battle_graph=battle_graph,
        full_graph=full_graph,
        macro_features=macro_features,
        models_bundle=models_bundle,
        attack_perspective=attack_perspective,
    )

    # Apply models
    p_A = capture_model_att.predict_proba(X)[:, 1]
    E_A = np.clip(troop_model_att.predict(X), 1.0, None)
    E_D = np.clip(troop_model_def.predict(X), 1.0, None)

    # Build node_index -> row position map (0..n-1)
    if "node_index" not in df_nodes.columns:
        raise KeyError("df_nodes missing required column 'node_index'.")

    node_index_arr = df_nodes["node_index"].to_numpy(dtype=int)
    node_to_rowpos: Dict[int, int] = {int(n): int(i) for i, n in enumerate(node_index_arr)}

    # Determine contestable update mask
    if "is_frontier_node" in df_nodes.columns:
        frontier_mask = df_nodes["is_frontier_node"].to_numpy(dtype=float) >= 0.5
        frontier_nodes = set(node_index_arr[frontier_mask].tolist())
    else:
        frontier_nodes = set()
        try:
            for u, v in battle_graph.edges:
                if global_state.nodes[u].owner != global_state.nodes[v].owner:
                    frontier_nodes.add(int(u))
                    frontier_nodes.add(int(v))
        except Exception:
            pass

    if "is_battle_node" in df_nodes.columns:
        battle_mask = df_nodes["is_battle_node"].to_numpy(dtype=float) >= 0.5
        battle_nodes = set(node_index_arr[battle_mask].tolist())
    else:
        battle_nodes = set(_battle_graph_nodes(battle_graph))

    contestable_nodes = frontier_nodes | battle_nodes

    if log_predict.isEnabledFor(logging.DEBUG):
        log_predict.debug(
            "apply_expectations attack_perspective=%s contestable=%d (frontier=%d battle=%d) nodes=%d",
            attack_perspective,
            len(contestable_nodes),
            len(frontier_nodes),
            len(battle_nodes),
            len(node_index_arr),
        )

    # Apply deterministic transition ONLY on contestable nodes
    new_nodes = list(global_state.nodes)
    for node_idx in contestable_nodes:
        r = node_to_rowpos.get(int(node_idx))
        if r is None:
            continue

        pA = float(p_A[r])
        if pA >= 0.5:
            owner = "A"
            troops_expected = float(E_A[r])
        else:
            owner = "D"
            troops_expected = float(E_D[r])

        troops_int = max(1, int(round(troops_expected)))
        new_nodes[int(node_idx)] = NodeState(owner=owner, troops=troops_int)

    # -----------------------------
    # PATCH: diff / integrity diagnostics
    # -----------------------------
    try:
        log_ctx = get_logger("risk.expectation", state_id=state_id, scen=scen, step=step)
    except Exception:
        log_ctx = log_predict

    owner_changes = 0
    troops_changes = 0
    illegal_changes = 0
    troop_delta_sum = 0

    # Track largest deltas (abs) among contestable nodes
    top_deltas: List[Tuple[int, int, str, str]] = []  # (abs_delta, node_idx, old_owner, new_owner)

    for node_idx in contestable_nodes:
        try:
            old = global_state.nodes[int(node_idx)]
            new = new_nodes[int(node_idx)]
        except Exception:
            continue

        if old.owner != new.owner:
            owner_changes += 1
        if old.troops != new.troops:
            troops_changes += 1
            d = int(new.troops) - int(old.troops)
            troop_delta_sum += d
            top_deltas.append((abs(d), int(node_idx), str(old.owner), str(new.owner)))

    # Ensure no non-contestable node changed (should never happen)
    try:
        contestable_set = set(int(x) for x in contestable_nodes)
        for i in range(len(new_nodes)):
            if i in contestable_set:
                continue
            if global_state.nodes[i].owner != new_nodes[i].owner or global_state.nodes[i].troops != new_nodes[i].troops:
                illegal_changes += 1
                if illegal_changes <= 10:
                    log_ctx.warning(
                        "[apply_expectations] non-contestable node changed! node=%d old=(%s,%d) new=(%s,%d)",
                        i,
                        global_state.nodes[i].owner,
                        global_state.nodes[i].troops,
                        new_nodes[i].owner,
                        new_nodes[i].troops,
                    )
    except Exception:
        pass

    if log_ctx.isEnabledFor(logging.DEBUG):
        top_deltas.sort(key=lambda x: x[0], reverse=True)
        top_deltas = top_deltas[: max(0, int(log_top_k))]
        log_ctx.debug(
            "[apply_expectations] cont=%s attack_perspective=%s contestable=%d owner_changes=%d troop_changes=%d "
            "troop_delta_sum=%d illegal_changes=%d top_abs_deltas=%s",
            continent_name,
            attack_perspective,
            len(contestable_nodes),
            owner_changes,
            troops_changes,
            troop_delta_sum,
            illegal_changes,
            top_deltas,
        )

    new_state = GlobalState(nodes=tuple(new_nodes))
    diag = {
        "continent_name": continent_name,
        "attack_perspective": attack_perspective,
        "contestable_nodes": int(len(contestable_nodes)),
        "frontier_nodes": int(len(frontier_nodes)),
        "battle_nodes": int(len(battle_nodes)),
        "owner_changes": int(owner_changes),
        "troops_changes": int(troops_changes),
        "troop_delta_sum": int(troop_delta_sum),
        "illegal_changes": int(illegal_changes),
        "top_abs_deltas": top_deltas,
    }

    return (new_state, diag) if return_diag else new_state




# ----------------------------------------------------------------------
# 3) Troop allocation
# ----------------------------------------------------------------------

def compute_synthetic_continent_reinforcements(
    continent_expected_attacker_territory_count: float,
    continent_realized_total_territory_count: float,
    base_attacker: int = 1,
    base_defender: int = 1,
    continent_bonus: int = 5,
) -> Tuple[int, int]:
    """
    Smooth, Risk-like fractional continent reinforcement rule.

    Each player gets:

        reinforcement = base + round(continent_bonus * (territories_owned / total_cont))

    This avoids sharp thresholds from floor(att_terr / 3) and scales naturally
    with partial continent control.
    """

    total_cont = max(1.0, continent_realized_total_territory_count)
    att_terr = max(0.0, continent_expected_attacker_territory_count)
    def_terr = max(0.0, total_cont - att_terr)

    att_bonus = round(continent_bonus * (att_terr / total_cont))
    def_bonus = round(continent_bonus * (def_terr / total_cont))

    att_reinf = base_attacker + att_bonus
    def_reinf = base_defender + def_bonus

    return att_reinf, def_reinf


def allocate_reinforcements_greedy_cheapest(
    global_state: GlobalState,
    battle_graph,
    full_graph,
    models_bundle: Dict[str, Any],
    macro_features: Dict[str, Any],
    num_reinforcements: int,
    owner: str = "A",
    candidate_nodes: Optional[Sequence[int]] = None,
    topk_candidates: Optional[int] = None,   # optional pruning knob
    *,
    attack_perspective: str = "P1_as_attacker",
    max_troops_cap: int = lowest_lib_node_max,                 # <-- NEW: deprioritize already-large stacks
) -> Tuple[GlobalState, Dict[int, int]]:
    """
    Cheap greedy reinforcement allocation, but vectorized:
      - Per reinforcement unit: evaluate ALL candidate nodes in ONE predict_proba call.
      - Only mutates the "initial_troops" feature.
      - Macro features held constant (same approximation as before).

    PATCH:
      - Use POSITIONAL row indices (0..n-1) when indexing numpy arrays (X, p_owner_base).
      - Keep a robust node_index -> rowpos mapping independent of df_nodes.index labels.
      - Deprioritize capped nodes: if any candidates are below max_troops_cap, restrict
        allocation to those. If all candidates are capped, fall back to original set.
    """
    from project_risk.infrastructure.log_config import get_logger

    log = get_logger("risk.synthetic")
    log_ml = get_logger("risk.ml")

    if num_reinforcements <= 0:
        log.debug("allocate: owner=%s reinf=%d -> noop", owner, num_reinforcements)
        return global_state, {}

    capture_model_att = models_bundle["capture_model_attacker"]
    feature_cols = models_bundle["feature_cols"]

    # Build node DF + base X for current state (once)
    df_nodes, X = _build_node_feature_df_from_state(
        global_state=global_state,
        battle_graph=battle_graph,
        full_graph=full_graph,
        macro_features=macro_features,
        models_bundle=models_bundle,
        attack_perspective=attack_perspective,
    )

    # Locate the single feature we change
    try:
        troops_col_idx = feature_cols.index("initial_troops")
    except ValueError:
        raise ValueError("'initial_troops' not found in feature_cols. Cannot allocate reinforcements.")

    # ------------------------------------------------------------
    # PATCH: build node_index -> ROW POSITION map (0..n-1)
    # ------------------------------------------------------------
    if "node_index" not in df_nodes.columns:
        raise KeyError("df_nodes missing required column 'node_index'.")

    node_to_rowpos: Dict[int, int] = {}
    node_index_arr = df_nodes["node_index"].to_numpy()
    for rowpos, node_idx in enumerate(node_index_arr):
        node_to_rowpos[int(node_idx)] = int(rowpos)

    full_nodes_set = set(node_to_rowpos.keys())

    # Determine candidate nodes (restrict to nodes present in df_nodes)
    if candidate_nodes is None:
        candidate_nodes = [
            idx for idx in full_nodes_set
            if global_state.nodes[int(idx)].owner == owner
        ]
        cand_source = "all_friendly_in_df_nodes"
    else:
        candidate_nodes = [int(i) for i in candidate_nodes if int(i) in full_nodes_set]
        cand_source = "provided_candidate_nodes"

    if not candidate_nodes:
        log.debug("allocate: owner=%s reinf=%d -> no candidates (source=%s)", owner, num_reinforcements, cand_source)
        return global_state, {}

    orig_candidate_count = len(candidate_nodes)

    # ------------------------------------------------------------
    # NEW: Deprioritize already-large stacks by filtering if possible
    # ------------------------------------------------------------
    filtered_by_cap = 0
    if max_troops_cap is not None:
        below_cap = [n for n in candidate_nodes if global_state.nodes[int(n)].troops < max_troops_cap]
        if below_cap:
            filtered_by_cap = len(candidate_nodes) - len(below_cap)
            candidate_nodes = below_cap
            if not candidate_nodes:
                log.debug(
                    "allocate: owner=%s reinf=%d -> all candidates filtered by cap=%s (orig=%d)",
                    owner, num_reinforcements, str(max_troops_cap), orig_candidate_count
                )
                return global_state, {}

    # Candidate row POSITIONS (safe for numpy indexing)
    cand_rows = np.array([node_to_rowpos[n] for n in candidate_nodes], dtype=int)

    # Optional pruning: keep only top-k candidates by current p(owner)
    pA_base = capture_model_att.predict_proba(X)[:, 1]
    p_owner_base = pA_base if owner == "A" else (1.0 - pA_base)

    if topk_candidates is not None and topk_candidates < len(cand_rows):
        scores = p_owner_base[cand_rows]
        keep = np.argsort(scores)[-topk_candidates:]  # keep best current positions
        cand_rows = cand_rows[keep]
        candidate_nodes = [candidate_nodes[i] for i in keep.tolist()]

    log_ml.debug(
        "allocate: owner=%s reinf=%d cand_source=%s orig_cand=%d after_cap=%d cap_filtered=%d after_topk=%d topk=%s persp=%s",
        owner, num_reinforcements, cand_source, orig_candidate_count, (orig_candidate_count - filtered_by_cap),
        filtered_by_cap, len(candidate_nodes), str(topk_candidates), attack_perspective
    )

    # Track allocations per candidate index
    extra = np.zeros(len(candidate_nodes), dtype=int)

    # Cache base troops for candidates from X (fast)
    base_troops = X[cand_rows, troops_col_idx].astype(int)

    # Current p(owner) per candidate (starts at baseline)
    current_p = p_owner_base[cand_rows].astype(float)

    # Greedy: each step batch-evaluates "+1 troop on each candidate"
    steps_taken = 0
    stopped_reason = "exhausted_budget"
    for _ in range(num_reinforcements):
        X_batch = X[cand_rows].copy()
        X_batch[:, troops_col_idx] = base_troops + extra + 1

        pA_plus1 = capture_model_att.predict_proba(X_batch)[:, 1]
        p_owner_plus1 = pA_plus1 if owner == "A" else (1.0 - pA_plus1)

        gains = p_owner_plus1 - current_p
        j = int(np.argmax(gains))
        best_gain = float(gains[j])

        if best_gain <= 0.0:
            stopped_reason = "no_positive_gain"
            break

        extra[j] += 1
        current_p[j] = p_owner_plus1[j]
        steps_taken += 1

    # Apply allocations to the actual state
    total_added = int(extra.sum())
    if total_added == 0:
        log.debug(
            "allocate: owner=%s reinf=%d -> no allocation (steps=%d reason=%s)",
            owner, num_reinforcements, steps_taken, stopped_reason
        )
        return global_state, {}

    new_nodes = list(global_state.nodes)
    alloc_dict: Dict[int, int] = {}

    for node_idx, add in zip(candidate_nodes, extra.tolist()):
        if add > 0:
            node = new_nodes[node_idx]
            new_nodes[node_idx] = NodeState(owner=node.owner, troops=node.troops + add)
            alloc_dict[node_idx] = add

    # Summarize allocations (top-k)
    top_alloc = sorted(alloc_dict.items(), key=lambda kv: kv[1], reverse=True)[:5]
    log.debug(
        "allocate: owner=%s reinf=%d steps=%d reason=%s total_added=%d alloc_k=%d top=%s",
        owner, num_reinforcements, steps_taken, stopped_reason, total_added, len(alloc_dict), str(top_alloc)
    )

    return GlobalState(nodes=tuple(new_nodes)), alloc_dict


def reallocate_troops_within_friendly_components(
    global_state: GlobalState,
    battle_graph,
    full_graph,
    models_bundle: Dict[str, Any],
    macro_features: Dict[str, Any],
    owner: str = "A",
    max_troops_cap: int = lowest_lib_node_max,   # used for smarter leftover dumping + candidate filtering
    *,
    attack_perspective: str = "P1_as_attacker",  # NEW: keeps inference feature schema consistent with training
) -> Tuple[GlobalState, Dict[int, int]]:
    from project_risk.infrastructure.log_config import get_logger

    log = get_logger("risk.synthetic")

    friendly_nodes = [
        idx for idx, node in enumerate(global_state.nodes)
        if node.owner == owner
    ]

    G_friendly = full_graph.subgraph(friendly_nodes).copy()
    components = list(nx.connected_components(G_friendly))

    new_nodes = list(global_state.nodes)
    movement: Dict[int, int] = {idx: 0 for idx in range(len(new_nodes))}

    MAX_GREEDY_POOL = 30
    TOPK_CANDIDATES = 10

    log.debug(
        "realloc: owner=%s friendly_n=%d comps=%d cap=%s persp=%s",
        owner, len(friendly_nodes), len(components), str(max_troops_cap), attack_perspective
    )

    for ci, comp in enumerate(components):
        comp_nodes = sorted(int(i) for i in comp)

        # 1) Build pool by dropping each friendly node in component to 1 troop
        pool = 0
        stripped_nodes = 0
        for node_idx in comp_nodes:
            node = new_nodes[node_idx]
            if node.owner != owner:
                continue
            if node.troops > 1:
                surplus = node.troops - 1
                pool += surplus
                stripped_nodes += 1
                movement[node_idx] -= surplus
                new_nodes[node_idx] = NodeState(owner=node.owner, troops=1)

        if pool <= 0:
            log.debug("realloc: comp=%d size=%d -> no pool", ci, len(comp_nodes))
            continue

        greedy_pool = min(pool, MAX_GREEDY_POOL)
        leftover = pool - greedy_pool

        baseline_state = GlobalState(nodes=tuple(new_nodes))

        # 2) Allocate (capped) pool inside component
        baseline_after_alloc, alloc = allocate_reinforcements_greedy_cheapest(
            global_state=baseline_state,
            battle_graph=battle_graph,
            full_graph=full_graph,
            models_bundle=models_bundle,
            macro_features=macro_features,
            num_reinforcements=greedy_pool,
            owner=owner,
            candidate_nodes=comp_nodes,
            topk_candidates=min(TOPK_CANDIDATES, len(comp_nodes)),
            max_troops_cap=max_troops_cap,
            attack_perspective=attack_perspective,
        )

        new_nodes = list(baseline_after_alloc.nodes)
        for node_idx, added in alloc.items():
            movement[node_idx] += added

        # 3) Dump leftover somewhere deterministic/cheap, but avoid creating huge stacks
        dump_node = None
        if leftover > 0:
            under = [n for n in comp_nodes if new_nodes[n].troops < max_troops_cap]
            if under:
                dump_node = min(under, key=lambda n: new_nodes[n].troops)
            else:
                dump_node = min(comp_nodes, key=lambda n: new_nodes[n].troops)

            node = new_nodes[dump_node]
            new_nodes[dump_node] = NodeState(owner=node.owner, troops=node.troops + leftover)
            movement[dump_node] += leftover

        # Per-component summary
        top_alloc = sorted(alloc.items(), key=lambda kv: kv[1], reverse=True)[:5]
        log.debug(
            "realloc: comp=%d size=%d stripped=%d pool=%d greedy_pool=%d leftover=%d alloc_sum=%d top_alloc=%s dump_node=%s",
            ci, len(comp_nodes), stripped_nodes, pool, greedy_pool, leftover, int(sum(alloc.values())),
            str(top_alloc), str(dump_node)
        )

    # Final summary: top movements
    nonzero_moves = [(i, d) for i, d in movement.items() if d != 0]
    top_moves = sorted(nonzero_moves, key=lambda kv: abs(kv[1]), reverse=True)[:8]
    net = int(sum(movement.values()))
    log.debug(
        "realloc: owner=%s moved_nodes=%d net=%d top_moves=%s",
        owner, len(nonzero_moves), net, str(top_moves)
    )

    return GlobalState(nodes=tuple(new_nodes)), movement




# ----------------------------------------------------------------------
# 3) Multi turn predictions
# ----------------------------------------------------------------------

def compute_macro_features_from_global_state(
    global_state: GlobalState,
    battle_graph,
    full_graph,
) -> Dict[str, Any]:
    """
    Compute the SAME macro features (pre-combat) that are used for ML training,
    but using an existing GlobalState instead of Players/Board.

    This is used in the multi-turn ML loop, where GlobalState is the
    source of truth and we no longer rely on Board/Players.

    It mirrors:
      - battle-level metrics from run_node_transition_experiment
      - full-graph counts/ratios + attacker troop distribution
      - full-graph topology metrics (degree stats, components, diameter, edge count)
    """
    # ------------------------------
    # Battle graph indices
    # ------------------------------
    battle_node_indices = _battle_graph_nodes(battle_graph)
    battle_total_territory_count = len(battle_node_indices)

    # ------------------------------
    # Full graph node list
    # ------------------------------
    try:
        full_nodes_iter = full_graph.nodes()
    except TypeError:
        full_nodes_iter = full_graph.nodes
    full_nodes = list(full_nodes_iter)
    full_total_territory_count = len(full_nodes)

    # ------------------------------
    # Battle-level aggregates
    # ------------------------------
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

    # Battle realized attacker territory ratio
    if battle_total_territory_count > 0:
        battle_realized_attacker_territory_ratio = (
            battle_initial_attacker_territory_count / battle_total_territory_count
        )
    else:
        battle_realized_attacker_territory_ratio = 0.0

    # Battle realized attacker available troops ratio
    if battle_initial_attacker_troops_count > 0:
        battle_realized_attacker_available_troops_ratio = (
            battle_attacker_available / battle_initial_attacker_troops_count
        )
    else:
        battle_realized_attacker_available_troops_ratio = 0.0

    # Battle attacker troop distribution metrics (CV, Gini)
    battle_attacker_troops_array = compute_attacker_troop_distribution(
        global_state, battle_node_indices
    )
    battle_realized_attacker_troops_distribution_cv = compute_troops_cv(
        battle_attacker_troops_array
    )
    battle_realized_attacker_troops_distribution_gini = compute_troops_gini(
        battle_attacker_troops_array
    )

    # ------------------------------
    # Full-graph aggregates
    # ------------------------------
    full_attacker_territory_count = 0
    full_defender_territory_count = 0
    full_attacker_troops_count = 0
    full_defender_troops_count = 0
    full_total_troops_count = 0

    attacker_troops_list = []

    for idx in full_nodes:
        node = global_state.nodes[idx]
        full_total_troops_count += node.troops
        if node.owner == "A":
            full_attacker_territory_count += 1
            full_attacker_troops_count += node.troops
            attacker_troops_list.append(node.troops)
        elif node.owner == "D":
            full_defender_territory_count += 1
            full_defender_troops_count += node.troops

    # Full realized attacker territory ratio
    if full_total_territory_count > 0:
        full_realized_attacker_territory_ratio = (
            full_attacker_territory_count / full_total_territory_count
        )
    else:
        full_realized_attacker_territory_ratio = 0.0

    # Full realized attacker troops ratio (attacker / total troops)
    if full_total_troops_count > 0:
        full_realized_attacker_troops_ratio = (
            full_attacker_troops_count / full_total_troops_count
        )
    else:
        full_realized_attacker_troops_ratio = 0.0

    # Full-graph "mobilization" metric (same logic as before)
    if full_attacker_troops_count > 0:
        full_attacker_available_troops_ratio = (
            battle_attacker_available / full_attacker_troops_count
        )
    else:
        full_attacker_available_troops_ratio = 0.0

    # Full attacker troop distribution metrics
    attacker_troops_arr = np.array(attacker_troops_list, dtype=float)
    full_realized_attacker_troops_distribution_cv = compute_troops_cv(
        attacker_troops_arr
    )
    full_realized_attacker_troops_distribution_gini = compute_troops_gini(
        attacker_troops_arr
    )

    # ------------------------------
    # Full-graph topology metrics
    # ------------------------------
    degrees = [deg for _, deg in full_graph.degree()]
    if degrees:
        full_realized_topology_degree_mean = float(np.mean(degrees))
        full_realized_topology_degree_variance = float(np.var(degrees))
    else:
        full_realized_topology_degree_mean = 0.0
        full_realized_topology_degree_variance = 0.0

    # NEW: edge count (match training metric name)
    try:
        full_realized_topology_edge_count = int(full_graph.number_of_edges())
    except Exception:
        full_realized_topology_edge_count = 0

    try:
        full_realized_topology_diameter = nx.diameter(full_graph)
    except Exception:
        full_realized_topology_diameter = 0.0

    full_realized_topology_component_count = nx.number_connected_components(full_graph)

    # ------------------------------
    # Assemble macro_features dict
    # ------------------------------
    macro_features: Dict[str, Any] = {
        # battle-level
        "battle_realized_attacker_territory_ratio": float(
            battle_realized_attacker_territory_ratio
        ),
        "battle_realized_attacker_available_troops_ratio": float(
            battle_realized_attacker_available_troops_ratio
        ),
        "battle_initial_attacker_territory_count": battle_initial_attacker_territory_count,
        "battle_initial_attacker_troops_count": battle_initial_attacker_troops_count,
        "battle_total_territory_count": battle_total_territory_count,
        "battle_total_troops_count": battle_total_troops_count,
        "battle_realized_attacker_troops_distribution_cv": float(
            battle_realized_attacker_troops_distribution_cv
        ),
        "battle_realized_attacker_troops_distribution_gini": float(
            battle_realized_attacker_troops_distribution_gini
        ),

        # full-graph core ratios
        "full_realized_attacker_territory_ratio": float(
            full_realized_attacker_territory_ratio
        ),
        "full_realized_attacker_troops_ratio": float(
            full_realized_attacker_troops_ratio
        ),
        "full_realized_attacker_available_troops_ratio": float(
            full_attacker_available_troops_ratio
        ),

        # full-graph counts
        "full_realized_attacker_territory_count": full_attacker_territory_count,
        "full_realized_defender_territory_count": full_defender_territory_count,
        "full_realized_total_territory_count": full_total_territory_count,
        "full_realized_attacker_troops_count": full_attacker_troops_count,
        "full_realized_defender_troops_count": full_defender_troops_count,
        "full_realized_total_troops_count": full_total_troops_count,

        # full-graph attacker troop distribution
        "full_realized_attacker_troops_distribution_cv": float(
            full_realized_attacker_troops_distribution_cv
        ),
        "full_realized_attacker_troops_distribution_gini": float(
            full_realized_attacker_troops_distribution_gini
        ),

        # topology
        "full_realized_topology_degree_mean": full_realized_topology_degree_mean,
        "full_realized_topology_degree_variance": full_realized_topology_degree_variance,
        "full_realized_topology_component_count": full_realized_topology_component_count,
        "full_realized_topology_diameter": full_realized_topology_diameter,
        "full_realized_topology_edge_count": full_realized_topology_edge_count,  # <-- NEW
    }

    return macro_features


def _graph_nodes_tuple(graph) -> Tuple[int, ...]:
    try:
        nodes_iter = graph.nodes()
    except TypeError:
        nodes_iter = graph.nodes
    return tuple(sorted(int(x) for x in nodes_iter))


def build_transition_distribution_example_from_state(
    *,
    global_state: GlobalState,
    battle_graph,
    full_graph,
    continent_name: str,
    attack_perspective: str = "P1_as_attacker",
    state_id: Optional[int] = None,
    macro_features: Optional[dict] = None,
) -> dict:
    """
    Build the grouped example schema expected by the Stage-B transition model.

    The caller should pass `global_state` in active-player perspective:
    "A" is the active attacker and "D" is the defender. Stage D full-board
    rollout will handle player-2 turns by using the existing perspective swap
    helpers before and after these single-continent calls.
    """
    full_nodes = _graph_nodes_tuple(full_graph)
    battle_nodes = _graph_nodes_tuple(battle_graph)
    if macro_features is None:
        macro_features = compute_macro_features_from_global_state(global_state, battle_graph, full_graph)

    signature = tuple(
        (int(node_id), str(global_state.nodes[int(node_id)].owner), max(1, int(global_state.nodes[int(node_id)].troops)))
        for node_id in full_nodes
    )
    return {
        "state_id": None if state_id is None else int(state_id),
        "continent_name": str(continent_name),
        "attack_perspective": str(attack_perspective),
        "full_graph_nodes": full_nodes,
        "battle_graph_nodes": battle_nodes,
        "initial_full_graph_signature": normalize_state_signature(signature),
        "macro_features": dict(macro_features or {}),
        "transition_example_status": "live_inference",
        "transition_example_error": None,
    }


def predict_successor_distribution_for_continent_state(
    *,
    global_state: GlobalState,
    battle_graph,
    full_graph,
    model_bundle: dict,
    continent_name: str,
    attack_perspective: str = "P1_as_attacker",
    k: Optional[int] = None,
    macro_features: Optional[dict] = None,
) -> dict:
    example = build_transition_distribution_example_from_state(
        global_state=global_state,
        battle_graph=battle_graph,
        full_graph=full_graph,
        continent_name=continent_name,
        attack_perspective=attack_perspective,
        macro_features=macro_features,
    )
    successor_distribution = predict_successor_distribution_from_example(
        model_bundle=model_bundle,
        example=example,
        k=k,
    )
    return {
        "continent_name": str(continent_name),
        "attack_perspective": str(attack_perspective),
        "successor_distribution": successor_distribution,
        "example": example,
        "model_type": model_bundle.get("model_type"),
        "schema_version": model_bundle.get("schema_version"),
    }


def sample_successor_signatures_for_continent_state(
    *,
    global_state: GlobalState,
    battle_graph,
    full_graph,
    model_bundle: dict,
    continent_name: str,
    n_samples: int,
    attack_perspective: str = "P1_as_attacker",
    k: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    macro_features: Optional[dict] = None,
) -> List[Tuple[Tuple[int, str, int], ...]]:
    pred = predict_successor_distribution_for_continent_state(
        global_state=global_state,
        battle_graph=battle_graph,
        full_graph=full_graph,
        model_bundle=model_bundle,
        continent_name=continent_name,
        attack_perspective=attack_perspective,
        k=k,
        macro_features=macro_features,
    )
    return sample_successor_signatures_from_distribution(
        pred["successor_distribution"],
        n_samples=int(n_samples),
        rng=rng,
    )


def apply_successor_signature_to_global_state(
    *,
    global_state: GlobalState,
    successor_signature,
    update_nodes: Optional[Sequence[int]] = None,
) -> GlobalState:
    signature_map = signature_to_node_state_map(successor_signature)
    if update_nodes is None:
        allowed = set(signature_map.keys())
    else:
        allowed = set(int(x) for x in update_nodes) & set(signature_map.keys())

    new_nodes = list(global_state.nodes)
    for node_id in sorted(allowed):
        if node_id < 0 or node_id >= len(new_nodes):
            continue
        owner, troops = signature_map[int(node_id)]
        owner = "A" if str(owner) == "A" else "D"
        new_nodes[int(node_id)] = NodeState(owner=owner, troops=max(1, int(troops)))
    return GlobalState(nodes=tuple(new_nodes))


def continent_transition_update_nodes(
    *,
    full_graph,
    battle_graph,
    commitment_map: Optional[dict[int, str]],
    continent_name: str,
    update_scope: str = "battle_plus_committed_outside",
) -> Tuple[int, ...]:
    full_nodes = set(_graph_nodes_tuple(full_graph))
    battle_nodes = set(_graph_nodes_tuple(battle_graph))
    if update_scope == "full_graph":
        return tuple(sorted(full_nodes))
    if update_scope == "battle_graph":
        return tuple(sorted(battle_nodes))
    if update_scope != "battle_plus_committed_outside":
        raise ValueError(f"Unknown update_scope={update_scope!r}")
    if commitment_map is None:
        return tuple(sorted(battle_nodes))
    committed = {
        int(node_id)
        for node_id in full_nodes - battle_nodes
        if commitment_map.get(int(node_id)) == continent_name
    }
    return tuple(sorted((battle_nodes | committed) & full_nodes))


def sample_and_apply_transition_distribution_for_continent(
    *,
    global_state: GlobalState,
    battle_graph,
    full_graph,
    model_bundle: dict,
    continent_name: str,
    commitment_map: Optional[dict[int, str]] = None,
    update_scope: str = "battle_plus_committed_outside",
    attack_perspective: str = "P1_as_attacker",
    k: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    macro_features: Optional[dict] = None,
) -> Tuple[GlobalState, dict]:
    pred = predict_successor_distribution_for_continent_state(
        global_state=global_state,
        battle_graph=battle_graph,
        full_graph=full_graph,
        model_bundle=model_bundle,
        continent_name=continent_name,
        attack_perspective=attack_perspective,
        k=k,
        macro_features=macro_features,
    )
    dist = pred["successor_distribution"]
    samples = sample_successor_signatures_from_distribution(dist, n_samples=1, rng=rng)
    sampled = samples[0] if samples else tuple()
    update_nodes = continent_transition_update_nodes(
        full_graph=full_graph,
        battle_graph=battle_graph,
        commitment_map=commitment_map,
        continent_name=continent_name,
        update_scope=update_scope,
    )
    new_state = apply_successor_signature_to_global_state(
        global_state=global_state,
        successor_signature=sampled,
        update_nodes=update_nodes,
    )
    top_prob = max((float(p) for p in dist.values()), default=0.0)
    diag = {
        "continent_name": str(continent_name),
        "attack_perspective": str(attack_perspective),
        "update_scope": str(update_scope),
        "updated_nodes": tuple(int(x) for x in update_nodes),
        "sampled_signature": sampled,
        "successor_distribution_support": int(len(dist)),
        "successor_distribution_top_prob": float(top_prob),
    }
    return new_state, diag





# ----------------------------------------------------------------------
# Diagnostics helpers (lightweight summaries for debugging/attribution)
# ----------------------------------------------------------------------

def _summarize_alloc_dict(alloc: Dict[int, int] | None) -> Dict[str, Any]:
    """Summarize an allocation dict {node_idx: added_troops}."""
    if not alloc:
        return {"alloc_sum": 0, "alloc_n_nodes": 0, "alloc_top1": 0, "alloc_top1_share": 0.0}
    vals = list(alloc.values())
    total = int(sum(vals))
    top1 = int(max(vals)) if vals else 0
    n_nodes = int(sum(1 for v in vals if v != 0))
    share = float(top1 / total) if total > 0 else 0.0
    return {"alloc_sum": total, "alloc_n_nodes": n_nodes, "alloc_top1": top1, "alloc_top1_share": share}


def _summarize_movement_dict(mov: Dict[int, int] | None) -> Dict[str, Any]:
    """Summarize a movement dict {node_idx: delta_troops}. Uses absolute movement magnitude."""
    if not mov:
        return {"moved_sum_abs": 0, "moved_nodes": 0, "moved_top_abs": 0}
    vals = [int(v) for v in mov.values() if v != 0]
    if not vals:
        return {"moved_sum_abs": 0, "moved_nodes": 0, "moved_top_abs": 0}
    moved_sum_abs = int(sum(abs(v) for v in vals))
    moved_nodes = int(len(vals))
    moved_top_abs = int(max(abs(v) for v in vals))
    return {"moved_sum_abs": moved_sum_abs, "moved_nodes": moved_nodes, "moved_top_abs": moved_top_abs}


def _candidate_counts_for_owner(
    state: GlobalState,
    full_graph,
    *,
    owner: str,
    max_troops_cap: int = lowest_lib_node_max,
    node_subset: Optional[Sequence[int]] = None,
) -> Dict[str, int]:
    """
    Approximate candidate pool size used by allocate_reinforcements_greedy_cheapest.

    IMPORTANT:
    - If `node_subset` is provided, counts are restricted to that subset (e.g., continent nodes),
      which matches continent-scoped simulations.
    """
    if node_subset is not None:
        nodes = [int(i) for i in node_subset]
    else:
        try:
            full_nodes_iter = full_graph.nodes()
        except TypeError:
            full_nodes_iter = full_graph.nodes
        nodes = [int(i) for i in list(full_nodes_iter)]

    friendly = [i for i in nodes if state.nodes[int(i)].owner == owner]
    if max_troops_cap is None:
        under_cap = friendly
    else:
        under_cap = [i for i in friendly if state.nodes[int(i)].troops < max_troops_cap]

    return {
        "cand_friendly": int(len(friendly)),
        "cand_under_cap": int(len(under_cap)),
    }

def simulate_multi_turns_ML(
    initial_global_state: GlobalState,
    battle_graph,  # kept for API symmetry; may be stale across turns
    full_graph,
    players: Sequence["Players.Player"],
    continent_name: str,
    models_path: Path | str = "node_level_models.joblib",
    max_turns: int = 10,
    stop_when_one_player_owns: bool = True,
) -> List[Dict[str, Any]]:
    """
    Multi-turn ML simulation with ALTERNATING attack turns.

    Compatibility patches:
      - No direct print() output; uses subsystem loggers.
      - Passes `attack_perspective` consistently so inference matches training.
      - Battle-graph and board-sync diagnostics are DEBUG-level.

    Added diagnostics:
      - Records continent-level troop totals + troops-per-territory for A and D each turn,
        so reversals (e.g., D has fewer territories but far more troops) are visible in output.
    """
    models_bundle = joblib.load(models_path)

    # Subsystem loggers for deep diagnostics (gated by log_config switches)
    log_synthetic = logging.getLogger("risk.synthetic")
    log_ml = logging.getLogger("risk.ml")


    # Force single-threading for stability / reproducibility
    for k in ("capture_model_attacker", "troop_model_attacker", "troop_model_defender"):
        m = models_bundle.get(k, None)
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1

    # Precompute continent nodes (stop condition + reinforcements)
    continent_territories = Board.continent_territory_dict[continent_name]
    continent_nodes = [t._index for t in continent_territories]
    total_cont = len(continent_nodes)

    history: List[Dict[str, Any]] = []
    global_state = initial_global_state

    continent_node_set = set(continent_nodes)

    def _true_frontier_edges(state_for_truth: GlobalState) -> int:
        edges = 0
        for u, v in full_graph.edges:
            if u in continent_node_set and v in continent_node_set:
                if state_for_truth.nodes[u].owner != state_for_truth.nodes[v].owner:
                    edges += 1
        return edges

    def _debug_battle_graphs(*, tag: str, state_for_truth: GlobalState, static_bg, fresh_bg) -> None:
        if not log_battle_graph.isEnabledFor(logging.DEBUG):
            return

        static_nodes = set(static_bg.nodes) if static_bg is not None else set()
        fresh_nodes = set(fresh_bg.nodes) if fresh_bg is not None else set()
        static_edges = static_bg.number_of_edges() if static_bg is not None else 0
        fresh_edges = fresh_bg.number_of_edges() if fresh_bg is not None else 0

        true_frontier_edges = _true_frontier_edges(state_for_truth)

        only_in_fresh = fresh_nodes - static_nodes
        only_in_static = static_nodes - fresh_nodes

        log_battle_graph.debug(
            "[%s] true_frontier_edges=%d static_bg(nodes=%d edges=%d) fresh_bg(nodes=%d edges=%d) diff(fresh_only=%d static_only=%d)",
            tag,
            true_frontier_edges,
            len(static_nodes),
            static_edges,
            len(fresh_nodes),
            fresh_edges,
            len(only_in_fresh),
            len(only_in_static),
        )

        if true_frontier_edges > 0 and len(fresh_nodes) == 0:
            log_battle_graph.warning("[%s] true frontier exists but FRESH battle_graph is empty", tag)
        if true_frontier_edges > 0 and len(static_nodes) == 0:
            log_battle_graph.warning("[%s] true frontier exists but STATIC battle_graph is empty", tag)

    def _continent_stats_canonical(state: GlobalState) -> Dict[str, Any]:
        """Continent-only totals in CANONICAL view (A=attacker label, D=defender label)."""
        owners = [state.nodes[i].owner for i in continent_nodes]
        troops = [state.nodes[i].troops for i in continent_nodes]

        a_terr = sum(1 for o in owners if o == "A")
        d_terr = sum(1 for o in owners if o == "D")

        a_troops = sum(t for o, t in zip(owners, troops) if o == "A")
        d_troops = sum(t for o, t in zip(owners, troops) if o == "D")

        a_tpt = a_troops / max(a_terr, 1)
        d_tpt = d_troops / max(d_terr, 1)

        return {
            "attacker_territories": a_terr,
            "defender_territories": d_terr,
            "attacker_troops_total": a_troops,
            "defender_troops_total": d_troops,
            "attacker_troops_per_territory": a_tpt,
            "defender_troops_per_territory": d_tpt,
            "all_A": (a_terr == total_cont),
            "all_D": (d_terr == total_cont),
        }

    # Initial continent ownership/troops (canonical)
    init_stats = _continent_stats_canonical(global_state)

    log_runner.info(
        "simulate_multi_turns start continent=%s A_terr=%d D_terr=%d",
        continent_name,
        init_stats["attacker_territories"],
        init_stats["defender_territories"],
    )

    history.append(
        {
            "turn": -1,
            "attacker_label_this_turn": None,
            "attacker_territories": init_stats["attacker_territories"],
            "defender_territories": init_stats["defender_territories"],
            "attacker_troops_total": init_stats["attacker_troops_total"],
            "defender_troops_total": init_stats["defender_troops_total"],
            "attacker_troops_per_territory": init_stats["attacker_troops_per_territory"],
            "defender_troops_per_territory": init_stats["defender_troops_per_territory"],
            "attacker_reinforcements": 0,
            "defender_reinforcements": 0,
            "movement_A_reallocation": {},
            "movement_D_reallocation": {},
            "movement_A_reinforcement": {},
            "movement_D_reinforcement": {},
            "alloc_A_sum": 0,
            "alloc_A_n_nodes": 0,
            "alloc_A_top1": 0,
            "alloc_A_top1_share": 0.0,
            "alloc_D_sum": 0,
            "alloc_D_n_nodes": 0,
            "alloc_D_top1": 0,
            "alloc_D_top1_share": 0.0,
            "realloc_A_moved_sum_abs": 0,
            "realloc_A_moved_nodes": 0,
            "realloc_D_moved_sum_abs": 0,
            "realloc_D_moved_nodes": 0,
            "candA_friendly": 0,
            "candA_under_cap": 0,
            "candD_friendly": 0,
            "candD_under_cap": 0,
            "all_A": init_stats["all_A"],
            "all_D": init_stats["all_D"],
        }
    )

    current_attacker = "A"

    for turn in range(max_turns):
        log_rollout.info("turn=%d attacker=%s", turn, current_attacker)

        # 1) Perspective view for the CURRENT attacker
        if current_attacker == "A":
            perspective_state = global_state
            perspective_players = players
            attack_perspective = "P1_as_attacker"
        else:
            perspective_state = swap_roles_in_global_state(global_state)
            perspective_players = [players[1], players[0]]
            attack_perspective = "P2_as_attacker"

        # 2) Apply perspective_state to Board and build battle_graph (fresh each turn)
        apply_global_state_to_board(perspective_state, perspective_players)

        if log_battle_graph.isEnabledFor(logging.DEBUG):
            mism_owner = []
            mism_troops = []
            pA = perspective_players[0]
            pD = perspective_players[1]
            for i in continent_nodes:
                terr = Board.node_to_territory_dict[i]
                expected_owner_obj = pA if perspective_state.nodes[i].owner == "A" else pD
                if terr._owner is not expected_owner_obj:
                    mism_owner.append((i, terr._owner._name if terr._owner is not None else None, expected_owner_obj._name))
                if terr._troops != perspective_state.nodes[i].troops:
                    mism_troops.append((i, terr._troops, perspective_state.nodes[i].troops))
                if len(mism_owner) >= 5 and len(mism_troops) >= 5:
                    break
            if mism_owner or mism_troops:
                log_battle_graph.debug(
                    "board_sync mism_owner=%d mism_troops=%d sample_owner=%s sample_troops=%s",
                    len(mism_owner),
                    len(mism_troops),
                    mism_owner[:3],
                    mism_troops[:3],
                )

        battle_graph_persp = agop.build_continent_battle_graph(continent_name, perspective_players)

        _debug_battle_graphs(
            tag=f"Turn {turn} PRE (attacker={current_attacker})",
            state_for_truth=perspective_state,
            static_bg=battle_graph,
            fresh_bg=battle_graph_persp,
        )

        # 3) Macro features (PRE-combat) in the perspective view
        macro_pre = compute_macro_features_from_global_state(perspective_state, battle_graph_persp, full_graph)

        # 4) ML expectations -> post-combat state in perspective view
        state_after_combat_persp = apply_expectations_as_state(
            global_state=perspective_state,
            battle_graph=battle_graph_persp,
            full_graph=full_graph,
            models_bundle=models_bundle,
            macro_features=macro_pre,
            attack_perspective=attack_perspective,
            continent_name=continent_name,
            state_id=None,
            step=int(turn),
            scen=None,
        )

        # 5) Convert back to CANONICAL view (A=player1, D=player2)
        state_after_combat = (
            state_after_combat_persp
            if current_attacker == "A"
            else swap_roles_in_global_state(state_after_combat_persp)
        )

        # 6) Continent stats in CANONICAL view (for logging/stop/reinf)
        cont_stats = _continent_stats_canonical(state_after_combat)
        att_terr_cont = cont_stats["attacker_territories"]
        def_terr_cont = cont_stats["defender_territories"]
        all_A = cont_stats["all_A"]
        all_D = cont_stats["all_D"]

        # 7) Compute reinforcements (symmetric rule)
        att_reinf, def_reinf = compute_synthetic_continent_reinforcements(
            continent_expected_attacker_territory_count=float(att_terr_cont),
            continent_realized_total_territory_count=float(total_cont),
        )

        # 8) Post-combat macro features (canonical) for allocation policies
        apply_global_state_to_board(state_after_combat, players)
        battle_graph_after = agop.build_continent_battle_graph(continent_name, players)

        _debug_battle_graphs(
            tag=f"Turn {turn} POST (canonical)",
            state_for_truth=state_after_combat,
            static_bg=battle_graph,
            fresh_bg=battle_graph_after,
        )

        macro_post = compute_macro_features_from_global_state(state_after_combat, battle_graph_after, full_graph)

        # 9) Reallocate old troops (both players, canonical)
        # NOTE: pass attack_perspective so inference feature schema matches training.
        state_after_realloc_A, movement_A_realloc = reallocate_troops_within_friendly_components(
            global_state=state_after_combat,
            battle_graph=battle_graph_after,
            full_graph=full_graph,
            models_bundle=models_bundle,
            macro_features=macro_post,
            owner="A",
            attack_perspective="P1_as_attacker",
        )
        state_after_realloc, movement_D_realloc = reallocate_troops_within_friendly_components(
            global_state=state_after_realloc_A,
            battle_graph=battle_graph_after,
            full_graph=full_graph,
            models_bundle=models_bundle,
            macro_features=macro_post,
            owner="D",
            attack_perspective="P1_as_attacker",
        )

        # 10) Allocate new reinforcements (both players, canonical)
        state_after_reinf_A, movement_A_reinf = allocate_reinforcements_greedy_cheapest(
            global_state=state_after_realloc,
            battle_graph=battle_graph_after,
            full_graph=full_graph,
            models_bundle=models_bundle,
            macro_features=macro_post,
            num_reinforcements=att_reinf,
            owner="A",
            candidate_nodes=[i for i in continent_nodes if state_after_realloc.nodes[int(i)].owner == "A"],
            attack_perspective="P1_as_attacker",
        )
        state_next, movement_D_reinf = allocate_reinforcements_greedy_cheapest(
            global_state=state_after_reinf_A,
            battle_graph=battle_graph_after,
            full_graph=full_graph,
            models_bundle=models_bundle,
            macro_features=macro_post,
            num_reinforcements=def_reinf,
            owner="D",
            candidate_nodes=[i for i in continent_nodes if state_after_reinf_A.nodes[int(i)].owner == "D"],
            attack_perspective="P1_as_attacker",
        )

        # 11) Log history

        # ---- Diagnostics: summarize synthetic actions (allocation / reallocation) ----
        allocA = _summarize_alloc_dict(movement_A_reinf)
        allocD = _summarize_alloc_dict(movement_D_reinf)
        movA = _summarize_movement_dict(movement_A_realloc)
        movD = _summarize_movement_dict(movement_D_realloc)

        # Candidate pool sizes (approximation of allocator's default candidate set)
        candA = _candidate_counts_for_owner(state_after_realloc, full_graph, owner="A", max_troops_cap=lowest_lib_node_max)
        candD = _candidate_counts_for_owner(state_after_reinf_A, full_graph, owner="D", max_troops_cap=lowest_lib_node_max)

        if log_synthetic.isEnabledFor(logging.DEBUG):
            log_synthetic.debug(
                "turn=%d synth_summary reinf(A/D)=%d/%d alloc_top1_share(A/D)=%.2f/%.2f alloc_n_nodes(A/D)=%d/%d moved_sum_abs(A/D)=%d/%d cand_under_cap(A/D)=%d/%d",
                turn,
                int(att_reinf),
                int(def_reinf),
                float(allocA["alloc_top1_share"]),
                float(allocD["alloc_top1_share"]),
                int(allocA["alloc_n_nodes"]),
                int(allocD["alloc_n_nodes"]),
                int(movA["moved_sum_abs"]),
                int(movD["moved_sum_abs"]),
                int(candA["cand_under_cap"]),
                int(candD["cand_under_cap"]),
            )
        # (store the continent stats from *post-combat* canonical state)
        history.append(
            {
                "turn": turn,
                "attacker_label_this_turn": current_attacker,
                "attacker_territories": att_terr_cont,
                "defender_territories": def_terr_cont,
                "attacker_troops_total": cont_stats["attacker_troops_total"],
                "defender_troops_total": cont_stats["defender_troops_total"],
                "attacker_troops_per_territory": cont_stats["attacker_troops_per_territory"],
                "defender_troops_per_territory": cont_stats["defender_troops_per_territory"],
                "attacker_reinforcements": att_reinf,
                "defender_reinforcements": def_reinf,
                "movement_A_reallocation": movement_A_realloc,
                "movement_D_reallocation": movement_D_realloc,
                "movement_A_reinforcement": movement_A_reinf,
                "movement_D_reinforcement": movement_D_reinf,
                "alloc_A_sum": int(allocA["alloc_sum"]),
                "alloc_A_n_nodes": int(allocA["alloc_n_nodes"]),
                "alloc_A_top1": int(allocA["alloc_top1"]),
                "alloc_A_top1_share": float(allocA["alloc_top1_share"]),
                "alloc_D_sum": int(allocD["alloc_sum"]),
                "alloc_D_n_nodes": int(allocD["alloc_n_nodes"]),
                "alloc_D_top1": int(allocD["alloc_top1"]),
                "alloc_D_top1_share": float(allocD["alloc_top1_share"]),
                "realloc_A_moved_sum_abs": int(movA["moved_sum_abs"]),
                "realloc_A_moved_nodes": int(movA["moved_nodes"]),
                "realloc_D_moved_sum_abs": int(movD["moved_sum_abs"]),
                "realloc_D_moved_nodes": int(movD["moved_nodes"]),
                "candA_friendly": int(candA["cand_friendly"]),
                "candA_under_cap": int(candA["cand_under_cap"]),
                "candD_friendly": int(candD["cand_friendly"]),
                "candD_under_cap": int(candD["cand_under_cap"]),
                "all_A": all_A,
                "all_D": all_D,
            }
        )

        # 12) Advance state and alternate attacker for next turn
        global_state = state_next
        current_attacker = "D" if current_attacker == "A" else "A"

        # 13) Stop condition
        if stop_when_one_player_owns and (all_A or all_D):
            log_runner.info("stop_condition met at turn=%d all_A=%s all_D=%s", turn, all_A, all_D)
            break

    return history

# ----------------------------------------------------------------------
# Public inference helpers (for strategic planning / utility search)
# ----------------------------------------------------------------------

def predict_capture_probabilities(
    global_state: GlobalState,
    battle_graph,
    full_graph,
    models_bundle: Dict[str, Any],
    macro_features: Dict[str, Any],
    *,
    attack_perspective: str = "P1_as_attacker",
) -> Dict[int, float]:
    """Return node_index -> P(node owned by attacker 'A' after combat).

    This is a thin public wrapper used by strategy/utility search. It reuses the
    exact feature construction path that `apply_expectations_as_state` uses.

    Notes:
      - The returned probabilities are those of `capture_model_attacker`.
      - Caller is responsible for choosing the correct `attack_perspective`
        convention consistent with training.
    """
    capture_model_att = models_bundle["capture_model_attacker"]

    df_nodes, X = _build_node_feature_df_from_state(
        global_state=global_state,
        battle_graph=battle_graph,
        full_graph=full_graph,
        macro_features=macro_features,
        models_bundle=models_bundle,
        attack_perspective=attack_perspective,
    )

    p_A = capture_model_att.predict_proba(X)[:, 1]

    if "node_index" not in df_nodes.columns:
        raise KeyError("df_nodes missing required column 'node_index'.")

    node_idx = df_nodes["node_index"].to_numpy(dtype=int)
    return {int(n): float(p) for n, p in zip(node_idx.tolist(), p_A.tolist())}
