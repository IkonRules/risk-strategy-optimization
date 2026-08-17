from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import math

import numpy as np
import pandas as pd

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.transition_prediction_ml.generate_data_ML import (
    apply_global_state_to_board,
    build_full_graph,
    normalize_state_signature,
    signature_to_node_state_map,
)
from project_risk.mathematical.transition_prediction_ml.predict_future_states_ML import build_transition_distribution_example_from_state
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState
from project_risk.mathematical.transition_prediction_ml.transition_distribution_ML import (
    predict_successor_distribution_from_example,
    sample_successor_signatures_from_distribution,
    train_transition_distribution_model,
)
from project_risk.mathematical.full_board_model.full_board_simulation_ML import (
    global_state_signature,
    simulate_multi_turns_full_board_transition_particles,
)


Signature = Tuple[Tuple[int, str, int], ...]


def normalize_successor_distribution(distribution: Mapping[Any, float | int]) -> Dict[Signature, float]:
    items: Dict[Signature, float] = {}
    for sig, mass in (distribution or {}).items():
        w = float(mass)
        if w <= 0.0 or not math.isfinite(w):
            continue
        norm_sig = normalize_state_signature(sig)
        items[norm_sig] = items.get(norm_sig, 0.0) + w
    total = float(sum(items.values()))
    if total <= 0.0:
        raise ValueError("successor distribution is empty or has zero positive mass")
    return {sig: float(w / total) for sig, w in sorted(items.items(), key=lambda kv: kv[0])}


def distribution_union_support(p: Mapping[Any, float], q: Mapping[Any, float]) -> Tuple[Signature, ...]:
    pp = normalize_successor_distribution(p)
    qq = normalize_successor_distribution(q)
    return tuple(sorted(set(pp) | set(qq)))


def total_variation_distance(true_distribution: Mapping[Any, float], predicted_distribution: Mapping[Any, float]) -> float:
    p = normalize_successor_distribution(true_distribution)
    q = normalize_successor_distribution(predicted_distribution)
    support = tuple(sorted(set(p) | set(q)))
    tv = 0.5 * sum(abs(float(p.get(s, 0.0)) - float(q.get(s, 0.0))) for s in support)
    return float(min(1.0, max(0.0, tv)))


def jensen_shannon_divergence(
    true_distribution: Mapping[Any, float],
    predicted_distribution: Mapping[Any, float],
    *,
    log_base: float = 2.0,
) -> float:
    p = normalize_successor_distribution(true_distribution)
    q = normalize_successor_distribution(predicted_distribution)
    support = tuple(sorted(set(p) | set(q)))
    log_denom = math.log(float(log_base))

    def _kl(a: Mapping[Signature, float], b: Mapping[Signature, float]) -> float:
        total = 0.0
        for s in support:
            av = float(a.get(s, 0.0))
            if av <= 0.0:
                continue
            bv = float(b.get(s, 0.0))
            total += av * (math.log(av / bv) / log_denom)
        return total

    m = {s: 0.5 * (float(p.get(s, 0.0)) + float(q.get(s, 0.0))) for s in support}
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(min(1.0, max(0.0, js)))


def _sorted_distribution_items(distribution: Mapping[Any, float]) -> List[Tuple[Signature, float]]:
    dist = normalize_successor_distribution(distribution)
    return sorted(dist.items(), key=lambda kv: (-float(kv[1]), kv[0]))


def distribution_support_metrics(
    true_distribution: Mapping[Any, float],
    predicted_distribution: Mapping[Any, float],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    p = normalize_successor_distribution(true_distribution)
    q = normalize_successor_distribution(predicted_distribution)
    true_support = set(p)
    pred_support = set(q)
    top_k = max(1, int(top_k))
    true_top = _sorted_distribution_items(p)
    pred_top = _sorted_distribution_items(q)
    true_top_set = {sig for sig, _ in true_top[:top_k]}
    pred_top_set = {sig for sig, _ in pred_top[:top_k]}
    top_union = true_top_set | pred_top_set
    return {
        "true_support_size": int(len(true_support)),
        "predicted_support_size": int(len(pred_support)),
        "support_intersection_size": int(len(true_support & pred_support)),
        "predicted_mass_on_true_support": float(sum(q.get(s, 0.0) for s in true_support)),
        "true_mass_on_predicted_support": float(sum(p.get(s, 0.0) for s in pred_support)),
        "true_top1_probability": float(true_top[0][1]) if true_top else 0.0,
        "predicted_top1_probability": float(pred_top[0][1]) if pred_top else 0.0,
        "top1_signature_match": bool(true_top and pred_top and true_top[0][0] == pred_top[0][0]),
        "top_k_signature_overlap_count": int(len(true_top_set & pred_top_set)),
        "top_k_signature_jaccard": float(len(true_top_set & pred_top_set) / len(top_union)) if top_union else 0.0,
    }


def _initial_owner_map(initial_global_state: Optional[GlobalState], initial_signature: Any = None) -> Dict[int, str]:
    if initial_global_state is not None:
        return {int(i): str(node.owner) for i, node in enumerate(initial_global_state.nodes)}
    if initial_signature is not None:
        return {int(i): str(owner) for i, owner, _ in normalize_state_signature(initial_signature)}
    return {}


def derive_node_marginals_from_weighted_distribution(
    successor_distribution: Mapping[Any, float],
    *,
    node_indices: Optional[Sequence[int]] = None,
    initial_global_state: Optional[GlobalState] = None,
    initial_signature: Any = None,
) -> Dict[int, Dict[str, float]]:
    dist = normalize_successor_distribution(successor_distribution)
    if node_indices is None:
        nodes = sorted({int(i) for sig in dist for i, _, _ in sig})
    else:
        nodes = sorted(int(i) for i in node_indices)
    initial_owners = _initial_owner_map(initial_global_state, initial_signature)
    acc = {
        i: {
            "a": 0.0,
            "d": 0.0,
            "troops": 0.0,
            "a_troops": 0.0,
            "d_troops": 0.0,
            "changed": 0.0,
        }
        for i in nodes
    }
    for sig, prob in dist.items():
        state_map = signature_to_node_state_map(sig)
        for node in nodes:
            if node not in state_map:
                continue
            owner, troops = state_map[node]
            row = acc[node]
            row["troops"] += float(prob) * float(troops)
            if owner == "A":
                row["a"] += float(prob)
                row["a_troops"] += float(prob) * float(troops)
            elif owner == "D":
                row["d"] += float(prob)
                row["d_troops"] += float(prob) * float(troops)
            if node in initial_owners and initial_owners[node] != owner:
                row["changed"] += float(prob)
    out: Dict[int, Dict[str, float]] = {}
    for node in nodes:
        row = acc[node]
        a = float(row["a"])
        d = float(row["d"])
        out[node] = {
            "p_attacker_final": a,
            "p_defender_final": d,
            "expected_troops": float(row["troops"]),
            "expected_troops_if_attacker": float(row["a_troops"] / a) if a > 0.0 else 0.0,
            "expected_troops_if_defender": float(row["d_troops"] / d) if d > 0.0 else 0.0,
            "p_changed_owner": float(row["changed"]) if initial_owners else 0.0,
        }
    return out


def compare_node_marginals(
    true_distribution: Mapping[Any, float],
    predicted_distribution: Mapping[Any, float],
    *,
    node_indices: Sequence[int],
    initial_global_state: Optional[GlobalState] = None,
    initial_signature: Any = None,
) -> Dict[str, float]:
    true_m = derive_node_marginals_from_weighted_distribution(
        true_distribution,
        node_indices=node_indices,
        initial_global_state=initial_global_state,
        initial_signature=initial_signature,
    )
    pred_m = derive_node_marginals_from_weighted_distribution(
        predicted_distribution,
        node_indices=node_indices,
        initial_global_state=initial_global_state,
        initial_signature=initial_signature,
    )

    def _err(field: str) -> Tuple[float, float, float]:
        vals = [float(pred_m[i][field]) - float(true_m[i][field]) for i in node_indices]
        if not vals:
            return 0.0, 0.0, 0.0
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(np.abs(arr))), float(np.sqrt(np.mean(arr * arr))), float(np.max(np.abs(arr)))

    p_mae, p_rmse, p_max = _err("p_attacker_final")
    t_mae, t_rmse, t_max = _err("expected_troops")
    a_mae, _, _ = _err("expected_troops_if_attacker")
    d_mae, _, _ = _err("expected_troops_if_defender")
    c_mae, _, _ = _err("p_changed_owner")
    return {
        "p_attacker_mae": p_mae,
        "p_attacker_rmse": p_rmse,
        "p_attacker_max_abs_error": p_max,
        "expected_troops_mae": t_mae,
        "expected_troops_rmse": t_rmse,
        "expected_troops_max_abs_error": t_max,
        "expected_troops_if_attacker_mae": a_mae,
        "expected_troops_if_defender_mae": d_mae,
        "p_changed_owner_mae": c_mae,
    }


def summarize_successor_distribution_values(
    distribution: Mapping[Any, float],
    *,
    initial_signature,
    continent_name: str,
    full_graph_nodes: Sequence[int],
) -> Dict[str, float]:
    dist = normalize_successor_distribution(distribution)
    initial_map = signature_to_node_state_map(initial_signature)
    full_nodes = tuple(int(i) for i in full_graph_nodes)
    continent_nodes = tuple(int(t._index) for t in Board.continent_territory_dict.get(str(continent_name), ()))
    out = {
        "expected_attacker_territories_full_graph": 0.0,
        "expected_defender_territories_full_graph": 0.0,
        "expected_attacker_troops_full_graph": 0.0,
        "expected_defender_troops_full_graph": 0.0,
        "expected_new_attacker_territories_full_graph": 0.0,
        "expected_lost_attacker_territories_full_graph": 0.0,
        "probability_attacker_owns_entire_continent": 0.0,
        "probability_defender_owns_entire_continent": 0.0,
    }
    for sig, prob in dist.items():
        smap = signature_to_node_state_map(sig)
        a_nodes = []
        d_nodes = []
        a_troops = 0
        d_troops = 0
        new_a = 0
        lost_a = 0
        for node in full_nodes:
            owner, troops = smap.get(node, initial_map.get(node, ("D", 1)))
            init_owner = initial_map.get(node, ("", 0))[0]
            if owner == "A":
                a_nodes.append(node)
                a_troops += int(troops)
                if init_owner != "A":
                    new_a += 1
            elif owner == "D":
                d_nodes.append(node)
                d_troops += int(troops)
                if init_owner == "A":
                    lost_a += 1
        out["expected_attacker_territories_full_graph"] += float(prob) * len(a_nodes)
        out["expected_defender_territories_full_graph"] += float(prob) * len(d_nodes)
        out["expected_attacker_troops_full_graph"] += float(prob) * a_troops
        out["expected_defender_troops_full_graph"] += float(prob) * d_troops
        out["expected_new_attacker_territories_full_graph"] += float(prob) * new_a
        out["expected_lost_attacker_territories_full_graph"] += float(prob) * lost_a
        if continent_nodes and all(smap.get(n, initial_map.get(n, ("D", 1)))[0] == "A" for n in continent_nodes):
            out["probability_attacker_owns_entire_continent"] += float(prob)
        if continent_nodes and all(smap.get(n, initial_map.get(n, ("A", 1)))[0] == "D" for n in continent_nodes):
            out["probability_defender_owns_entire_continent"] += float(prob)
    return {k: float(v) for k, v in out.items()}


def compare_distribution_value_summaries(
    true_distribution: Mapping[Any, float],
    predicted_distribution: Mapping[Any, float],
    *,
    initial_signature,
    continent_name: str,
    full_graph_nodes: Sequence[int],
) -> Dict[str, float]:
    true_s = summarize_successor_distribution_values(
        true_distribution,
        initial_signature=initial_signature,
        continent_name=continent_name,
        full_graph_nodes=full_graph_nodes,
    )
    pred_s = summarize_successor_distribution_values(
        predicted_distribution,
        initial_signature=initial_signature,
        continent_name=continent_name,
        full_graph_nodes=full_graph_nodes,
    )
    out: Dict[str, float] = {}
    for key, true_v in true_s.items():
        pred_v = float(pred_s.get(key, 0.0))
        err = pred_v - float(true_v)
        out[f"{key}_signed_error"] = float(err)
        out[f"{key}_abs_error"] = float(abs(err))
    return out


def _state_id_value(example: Mapping[str, Any]) -> Any:
    return example.get("state_id", None)


def _example_input_key(example: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(example.get("continent_name", "")),
        str(example.get("attack_perspective", "")),
        normalize_state_signature(example.get("initial_full_graph_signature", ()) or ()),
        tuple(sorted(int(x) for x in (example.get("battle_graph_nodes", ()) or ()))),
    )


def _evaluate_distribution_pair(
    *,
    true_distribution: Mapping[Any, float],
    predicted_distribution: Mapping[Any, float],
    example: Mapping[str, Any],
    top_k: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    true_dist = normalize_successor_distribution(true_distribution)
    pred_dist = normalize_successor_distribution(predicted_distribution)
    full_nodes = tuple(int(x) for x in (example.get("full_graph_nodes", ()) or ()))
    initial_signature = normalize_state_signature(example.get("initial_full_graph_signature", ()) or ())
    node_metrics = compare_node_marginals(
        true_dist,
        pred_dist,
        node_indices=full_nodes,
        initial_signature=initial_signature,
    )
    value_metrics = compare_distribution_value_summaries(
        true_dist,
        pred_dist,
        initial_signature=initial_signature,
        continent_name=str(example.get("continent_name", "")),
        full_graph_nodes=full_nodes,
    )
    row: Dict[str, Any] = {
        "state_id": example.get("state_id"),
        "continent_name": example.get("continent_name"),
        "attack_perspective": example.get("attack_perspective"),
        "validation_status": "ok",
        "validation_error": None,
        "total_variation_distance": total_variation_distance(true_dist, pred_dist),
        "jensen_shannon_divergence": jensen_shannon_divergence(true_dist, pred_dist),
    }
    row.update(distribution_support_metrics(true_dist, pred_dist, top_k=top_k))
    row.update(node_metrics)
    row.update(value_metrics)

    true_m = derive_node_marginals_from_weighted_distribution(true_dist, node_indices=full_nodes, initial_signature=initial_signature)
    pred_m = derive_node_marginals_from_weighted_distribution(pred_dist, node_indices=full_nodes, initial_signature=initial_signature)
    node_rows = []
    for node in full_nodes:
        node_rows.append(
            {
                "state_id": example.get("state_id"),
                "continent_name": example.get("continent_name"),
                "attack_perspective": example.get("attack_perspective"),
                "node_index": int(node),
                "predicted_p_attacker": float(pred_m[int(node)]["p_attacker_final"]),
                "true_p_attacker": float(true_m[int(node)]["p_attacker_final"]),
                "predicted_expected_troops": float(pred_m[int(node)]["expected_troops"]),
                "true_expected_troops": float(true_m[int(node)]["expected_troops"]),
            }
        )
    return row, node_rows


def evaluate_transition_distribution_example(
    *,
    example: Mapping[str, Any],
    model_bundle: Mapping[str, Any],
    k: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    try:
        true_dist = normalize_successor_distribution(example.get("full_graph_successor_state_counts", {}) or {})
        pred_dist = predict_successor_distribution_from_example(model_bundle=model_bundle, example=example, k=k)
        row, _ = _evaluate_distribution_pair(
            true_distribution=true_dist,
            predicted_distribution=pred_dist,
            example=example,
            top_k=top_k,
        )
        return row
    except Exception as e:
        return {
            "state_id": example.get("state_id"),
            "continent_name": example.get("continent_name"),
            "attack_perspective": example.get("attack_perspective"),
            "validation_status": "error",
            "validation_error": f"{type(e).__name__}: {e}",
        }


def split_transition_examples_grouped(
    examples_df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    group_mode: str = "state_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if examples_df.empty:
        return examples_df.copy(), examples_df.copy(), {
            "n_rows_total": 0,
            "n_rows_train": 0,
            "n_rows_test": 0,
            "n_groups_train": 0,
            "n_groups_test": 0,
            "exact_input_overlap_count": 0,
            "group_mode": group_mode,
        }
    df = examples_df.reset_index(drop=True).copy()
    if group_mode not in ("state_id", "initial_signature", "state_id_and_signature"):
        raise ValueError(f"Unsupported group_mode={group_mode!r}")

    keys = [_example_input_key(rec) for rec in df.to_dict(orient="records")]
    state_ids = [_state_id_value(rec) for rec in df.to_dict(orient="records")]

    parent = list(range(len(df)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    buckets: Dict[Any, int] = {}
    if group_mode in ("state_id", "state_id_and_signature"):
        for i, sid in enumerate(state_ids):
            key = ("sid", sid)
            if key in buckets:
                union(i, buckets[key])
            else:
                buckets[key] = i
    if group_mode in ("initial_signature", "state_id_and_signature"):
        for i, key_tuple in enumerate(keys):
            key = ("input", key_tuple)
            if key in buckets:
                union(i, buckets[key])
            else:
                buckets[key] = i

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(df)):
        groups[find(i)].append(i)
    group_ids = sorted(groups)
    rng = np.random.default_rng(int(random_state))
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    n_test_groups = max(1, int(round(float(test_size) * len(shuffled)))) if len(shuffled) > 1 else 0
    test_groups = set(shuffled[:n_test_groups])
    test_idx = sorted(i for g in test_groups for i in groups[g])
    train_idx = sorted(i for g in group_ids if g not in test_groups for i in groups[g])
    if not train_idx and test_idx:
        train_idx.append(test_idx.pop())
    if not test_idx and train_idx and len(group_ids) > 1:
        test_idx.append(train_idx.pop())

    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    train_keys = {_example_input_key(r) for r in train.to_dict(orient="records")}
    test_keys = {_example_input_key(r) for r in test.to_dict(orient="records")}
    overlap = train_keys & test_keys
    if overlap:
        keep = [i for i, r in enumerate(test.to_dict(orient="records")) if _example_input_key(r) not in overlap]
        test = test.iloc[keep].reset_index(drop=True)
        test_keys = {_example_input_key(r) for r in test.to_dict(orient="records")}
        overlap = train_keys & test_keys

    summary = {
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "n_groups_train": int(len({_example_input_key(r) if group_mode == "initial_signature" else _state_id_value(r) for r in train.to_dict(orient="records")})),
        "n_groups_test": int(len({_example_input_key(r) if group_mode == "initial_signature" else _state_id_value(r) for r in test.to_dict(orient="records")})),
        "exact_input_overlap_count": int(len(overlap)),
        "group_mode": group_mode,
    }
    return train, test, summary


def _valid_examples_for_validation(examples_df: pd.DataFrame, *, continent_name: str) -> pd.DataFrame:
    df = examples_df.copy()
    if "continent_name" in df.columns:
        df = df[df["continent_name"] == continent_name]
    if "transition_example_status" in df.columns:
        df = df[df["transition_example_status"] == "ok"]
    df = df[df["full_graph_successor_state_counts"].map(lambda v: isinstance(v, Mapping) and bool(v))]
    return df.reset_index(drop=True)


def run_transition_distribution_holdout_validation(
    *,
    examples_df: pd.DataFrame,
    continent_name: str,
    k_neighbors: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    group_mode: str = "state_id_and_signature",
    max_test_examples: Optional[int] = None,
    top_k: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    df = _valid_examples_for_validation(examples_df, continent_name=continent_name)
    if len(df) < 2:
        raise ValueError(f"Need at least two valid examples for holdout validation; got {len(df)}")
    train_df, test_df, split_summary = split_transition_examples_grouped(
        df,
        test_size=test_size,
        random_state=random_state,
        group_mode=group_mode,
    )
    if train_df.empty or test_df.empty:
        raise ValueError(f"Holdout split produced train={len(train_df)} test={len(test_df)} rows")
    if max_test_examples is not None:
        test_df = test_df.head(max(0, int(max_test_examples))).reset_index(drop=True)
    bundle = train_transition_distribution_model(
        train_df,
        continent_name=continent_name,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    rows: List[Dict[str, Any]] = []
    node_rows: List[Dict[str, Any]] = []
    for rec in test_df.to_dict(orient="records"):
        try:
            true_dist = normalize_successor_distribution(rec.get("full_graph_successor_state_counts", {}) or {})
            pred_dist = predict_successor_distribution_from_example(model_bundle=bundle, example=rec)
            row, nodes = _evaluate_distribution_pair(
                true_distribution=true_dist,
                predicted_distribution=pred_dist,
                example=rec,
                top_k=top_k,
            )
            rows.append(row)
            node_rows.extend(nodes)
        except Exception as e:
            rows.append(
                {
                    "state_id": rec.get("state_id"),
                    "continent_name": rec.get("continent_name"),
                    "attack_perspective": rec.get("attack_perspective"),
                    "validation_status": "error",
                    "validation_error": f"{type(e).__name__}: {e}",
                }
            )
    metrics_df = pd.DataFrame(rows)
    node_df = pd.DataFrame(node_rows)
    ok = metrics_df[metrics_df.get("validation_status", "") == "ok"] if not metrics_df.empty else metrics_df

    def _agg(prefix: str, field: str) -> Dict[str, float]:
        if ok.empty or field not in ok.columns:
            return {f"{prefix}_{field}": 0.0}
        vals = ok[field].astype(float).to_numpy()
        return {
            f"mean_{field}": float(np.mean(vals)),
            f"median_{field}": float(np.median(vals)),
            f"p90_{field}": float(np.percentile(vals, 90)),
        }

    summary: Dict[str, Any] = {
        "continent_name": str(continent_name),
        "n_train_examples": int(len(train_df)),
        "n_test_examples": int(len(test_df)),
        "k_neighbors": int(k_neighbors),
        "validation_failure_count": int((metrics_df.get("validation_status", "") != "ok").sum()) if not metrics_df.empty else 0,
        "top1_signature_accuracy": float(ok["top1_signature_match"].astype(float).mean()) if not ok.empty and "top1_signature_match" in ok else 0.0,
        "mean_predicted_mass_on_true_support": float(ok["predicted_mass_on_true_support"].astype(float).mean()) if not ok.empty and "predicted_mass_on_true_support" in ok else 0.0,
        "mean_true_mass_on_predicted_support": float(ok["true_mass_on_predicted_support"].astype(float).mean()) if not ok.empty and "true_mass_on_predicted_support" in ok else 0.0,
        "split_summary": split_summary,
    }
    for field in ("total_variation_distance", "jensen_shannon_divergence", "p_attacker_mae", "expected_troops_mae"):
        summary.update(_agg("", field))
    for field in (
        "expected_attacker_territories_full_graph_abs_error",
        "expected_attacker_troops_full_graph_abs_error",
        "probability_attacker_owns_entire_continent_abs_error",
    ):
        if not ok.empty and field in ok.columns:
            summary[f"mean_{field}"] = float(ok[field].astype(float).mean())
        else:
            summary[f"mean_{field}"] = 0.0
    return metrics_df, node_df, summary, bundle


def build_node_ownership_calibration_table(
    per_node_predictions: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    if per_node_predictions.empty:
        return pd.DataFrame(columns=["bin_lower", "bin_upper", "n", "mean_predicted_p_attacker", "mean_true_p_attacker", "calibration_error", "ownership_brier_score"])
    df = per_node_predictions.copy()
    n_bins = max(1, int(n_bins))
    rows = []
    brier = float(np.mean((df["predicted_p_attacker"].astype(float) - df["true_p_attacker"].astype(float)) ** 2))
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            mask = (df["predicted_p_attacker"] >= lo) & (df["predicted_p_attacker"] <= hi)
        else:
            mask = (df["predicted_p_attacker"] >= lo) & (df["predicted_p_attacker"] < hi)
        sub = df[mask]
        mean_pred = float(sub["predicted_p_attacker"].astype(float).mean()) if not sub.empty else 0.0
        mean_true = float(sub["true_p_attacker"].astype(float).mean()) if not sub.empty else 0.0
        rows.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "n": int(len(sub)),
                "mean_predicted_p_attacker": mean_pred,
                "mean_true_p_attacker": mean_true,
                "calibration_error": float(mean_pred - mean_true) if not sub.empty else 0.0,
                "ownership_brier_score": brier,
            }
        )
    return pd.DataFrame(rows)


def evaluate_particle_sampling_convergence(
    *,
    successor_distribution: Mapping[Any, float],
    population_sizes: Sequence[int] = (50, 100, 500, 1000),
    repetitions: int = 10,
    random_seed: int = 42,
    node_indices: Optional[Sequence[int]] = None,
    initial_signature: Any = None,
) -> pd.DataFrame:
    pred = normalize_successor_distribution(successor_distribution)
    rows = []
    base_seed = int(random_seed)
    for pop in population_sizes:
        for rep in range(int(repetitions)):
            rng = np.random.default_rng(base_seed + int(pop) * 1009 + rep)
            samples = sample_successor_signatures_from_distribution(pred, n_samples=int(pop), rng=rng)
            counts: Dict[Signature, int] = {}
            for sig in samples:
                counts[sig] = counts.get(sig, 0) + 1
            empirical = normalize_successor_distribution(counts)
            row = {
                "population_size": int(pop),
                "repetition": int(rep),
                "total_variation_distance": total_variation_distance(pred, empirical),
                "jensen_shannon_divergence": jensen_shannon_divergence(pred, empirical),
                "support_size": int(len(empirical)),
                "predicted_support_mass_recovered": float(sum(pred.get(s, 0.0) for s in empirical)),
                "top1_signature_match": bool(_sorted_distribution_items(pred)[0][0] == _sorted_distribution_items(empirical)[0][0]),
            }
            if node_indices is not None:
                row.update(
                    compare_node_marginals(
                        pred,
                        empirical,
                        node_indices=tuple(int(i) for i in node_indices),
                        initial_signature=initial_signature,
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_particle_convergence(convergence_df: pd.DataFrame) -> pd.DataFrame:
    if convergence_df.empty:
        return pd.DataFrame()
    metric_cols = [
        c
        for c in convergence_df.columns
        if c
        not in {
            "population_size",
            "repetition",
            "top1_signature_match",
        }
        and pd.api.types.is_numeric_dtype(convergence_df[c])
    ]
    rows = []
    for pop, sub in convergence_df.groupby("population_size", sort=True):
        row = {"population_size": int(pop), "repetitions": int(len(sub))}
        for col in metric_cols:
            vals = sub[col].astype(float).to_numpy()
            row[f"mean_{col}"] = float(np.mean(vals))
            row[f"std_{col}"] = float(np.std(vals))
            row[f"p90_{col}"] = float(np.percentile(vals, 90))
        if "top1_signature_match" in sub.columns:
            row["top1_signature_match_rate"] = float(sub["top1_signature_match"].astype(float).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def validate_one_turn_particle_rollout_against_direct_distribution(
    *,
    initial_global_state: GlobalState,
    players: Sequence["Players.Player"],
    continent_name: str,
    model_bundle: Mapping[str, Any],
    population_size: int = 1000,
    particle_budget: int = 1000,
    random_seed: int = 42,
    update_scope: str = "full_graph",
) -> Dict[str, Any]:
    apply_global_state_to_board(initial_global_state, players)
    full_graph = build_full_graph(continent_name)
    battle_graph = agop.build_continent_battle_graph(continent_name, players, debug=False)
    direct_example = build_transition_distribution_example_from_state(
        global_state=initial_global_state,
        battle_graph=battle_graph,
        full_graph=full_graph,
        continent_name=continent_name,
        attack_perspective="P1_as_attacker",
    )
    direct_dist = normalize_successor_distribution(
        predict_successor_distribution_from_example(model_bundle=model_bundle, example=direct_example)
    )
    result = simulate_multi_turns_full_board_transition_particles(
        initial_global_state=initial_global_state,
        players=players,
        max_turns=1,
        transition_models_by_continent={str(continent_name): dict(model_bundle)},
        particle_budget=int(particle_budget),
        continent_order=(str(continent_name),),
        fallback_mode="error",
        random_seed=int(random_seed),
        apply_turn_mechanics=False,
        population_mode="fixed_population",
        population_size=int(population_size),
        update_scope=update_scope,
    )
    full_nodes = tuple(int(x) for x in model_bundle.get("full_graph_nodes", tuple(sorted(full_graph.nodes()))))
    empirical: Dict[Signature, float] = {}
    for particle in result["particles"]:
        sig = global_state_signature(particle.state, node_indices=full_nodes)
        empirical[sig] = empirical.get(sig, 0.0) + float(particle.weight)
    empirical = normalize_successor_distribution(empirical)
    node_metrics = compare_node_marginals(
        direct_dist,
        empirical,
        node_indices=full_nodes,
        initial_signature=direct_example.get("initial_full_graph_signature"),
    )
    return {
        "total_variation_distance": total_variation_distance(direct_dist, empirical),
        "jensen_shannon_divergence": jensen_shannon_divergence(direct_dist, empirical),
        "p_attacker_mae": float(node_metrics["p_attacker_mae"]),
        "expected_troops_mae": float(node_metrics["expected_troops_mae"]),
        "direct_support_size": int(len(direct_dist)),
        "particle_support_size": int(len(empirical)),
        "population_size": int(population_size),
        "particle_budget": int(particle_budget),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_validation_outputs(
    *,
    output_dir: Path | str,
    continent_name: str,
    per_example_metrics_df: pd.DataFrame,
    per_node_metrics_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    particle_convergence_df: pd.DataFrame,
    summary: Mapping[str, Any],
) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    slug = str(continent_name).strip().lower().replace(" ", "_")
    paths = {
        "per_example_metrics": output / f"transition_holdout_metrics__{slug}.csv",
        "per_node_metrics": output / f"transition_holdout_node_metrics__{slug}.csv",
        "calibration": output / f"transition_calibration__{slug}.csv",
        "particle_convergence": output / f"transition_particle_convergence__{slug}.csv",
        "summary": output / f"transition_validation_summary__{slug}.json",
    }
    per_example_metrics_df.to_csv(paths["per_example_metrics"], index=False)
    per_node_metrics_df.to_csv(paths["per_node_metrics"], index=False)
    calibration_df.to_csv(paths["calibration"], index=False)
    particle_convergence_df.to_csv(paths["particle_convergence"], index=False)
    with paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, sort_keys=True)
    return {k: str(v) for k, v in paths.items()}
