from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math
import pickle
import re

import numpy as np
import pandas as pd

from project_risk.game_simulation import Board
from project_risk.mathematical.transition_prediction_ml.generate_data_ML import (
    derive_node_marginals_from_successor_distribution,
    normalize_state_signature,
    signature_to_node_state_map,
)


Signature = Tuple[Tuple[int, str, int], ...]


@dataclass
class StandardFeatureScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "StandardFeatureScaler":
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=0) if X.size else np.zeros((X.shape[1],), dtype=float)
        scale = X.std(axis=0) if X.size else np.ones((X.shape[1],), dtype=float)
        scale = np.where(scale <= 1e-12, 1.0, scale)
        return cls(mean_=mean.astype(float), scale_=scale.astype(float))

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_


@dataclass
class TransitionDistributionKNNModel:
    continent_name: str
    feature_cols: Tuple[str, ...]
    full_graph_nodes: Tuple[int, ...]
    k_neighbors: int
    distance_metric: str
    scaler: StandardFeatureScaler
    X_train: np.ndarray
    example_records: Tuple[Dict[str, Any], ...]
    eps: float = 1e-9

    def _feature_vector_from_example(self, example: Mapping[str, Any]) -> np.ndarray:
        frame, _ = build_transition_example_feature_frame(pd.DataFrame([dict(example)]), feature_cols=list(self.feature_cols))
        return frame.loc[:, list(self.feature_cols)].to_numpy(dtype=float)

    def _neighbor_indices_and_weights(self, X: np.ndarray, *, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        train = np.asarray(self.X_train, dtype=float)
        if Xs.shape[0] != 1:
            raise ValueError("TransitionDistributionKNNModel expects exactly one query example.")
        if self.distance_metric != "euclidean":
            raise ValueError(f"Unsupported distance_metric={self.distance_metric!r}")
        dist = np.sqrt(((train - Xs[0]) ** 2).sum(axis=1))
        k_eff = max(1, min(int(k if k is not None else self.k_neighbors), len(dist)))
        order = np.argsort(dist, kind="mergesort")
        zero = order[dist[order] <= self.eps]
        if len(zero) > 0:
            idx = zero[:k_eff]
            weights = np.ones(len(idx), dtype=float) / float(len(idx))
            return idx, weights
        idx = order[:k_eff]
        raw = 1.0 / (dist[idx] + self.eps)
        weights = raw / raw.sum()
        return idx, weights

    def predict_distribution(self, features_or_example: Mapping[str, Any] | np.ndarray, *, k: Optional[int] = None) -> Dict[Signature, float]:
        if isinstance(features_or_example, Mapping):
            X = self._feature_vector_from_example(features_or_example)
        else:
            X = np.asarray(features_or_example, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        idx, weights = self._neighbor_indices_and_weights(X, k=k)
        mixed: Dict[Signature, float] = {}
        for train_idx, w in zip(idx, weights):
            rec = self.example_records[int(train_idx)]
            counts = rec.get("full_graph_successor_state_counts", {}) or {}
            total = float(sum(float(v) for v in counts.values()))
            if total <= 0:
                continue
            for sig, count in counts.items():
                norm_sig = normalize_state_signature(sig)
                mixed[norm_sig] = mixed.get(norm_sig, 0.0) + float(w) * (float(count) / total)
        z = float(sum(mixed.values()))
        if z <= 0:
            return {}
        return {sig: float(prob / z) for sig, prob in sorted(mixed.items(), key=lambda kv: kv[0])}

    def sample_successor_states(
        self,
        features_or_example: Mapping[str, Any] | np.ndarray,
        *,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
        k: Optional[int] = None,
    ) -> List[Signature]:
        return sample_successor_signatures_from_distribution(
            self.predict_distribution(features_or_example, k=k),
            n_samples=n_samples,
            rng=rng,
        )


def _slugify_continent(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_continent"


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, bool)) and not isinstance(value, (list, tuple, dict))


def _feature_dict_from_example(example: Mapping[str, Any]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    macro = example.get("macro_features", {}) or {}
    if isinstance(macro, Mapping):
        for key, value in sorted(macro.items(), key=lambda kv: str(kv[0])):
            if _is_numeric_scalar(value):
                f = float(value)
                if math.isfinite(f):
                    feats[f"macro__{key}"] = f

    state_map = signature_to_node_state_map(example.get("initial_full_graph_signature", ()) or ())
    full_nodes = tuple(sorted(int(x) for x in (example.get("full_graph_nodes", ()) or ())))
    battle_nodes = set(int(x) for x in (example.get("battle_graph_nodes", ()) or ()))
    for node_id in full_nodes:
        owner, troops = state_map.get(int(node_id), ("", 0))
        feats[f"node_{node_id}__owner_A"] = 1.0 if owner == "A" else 0.0
        feats[f"node_{node_id}__owner_D"] = 1.0 if owner == "D" else 0.0
        feats[f"node_{node_id}__troops"] = float(troops)
        feats[f"node_{node_id}__is_battle_node"] = 1.0 if int(node_id) in battle_nodes else 0.0

    perspective = str(example.get("attack_perspective", ""))
    feats["perspective__P1_as_attacker"] = 1.0 if perspective == "P1_as_attacker" else 0.0
    feats["perspective__P2_as_attacker"] = 1.0 if perspective == "P2_as_attacker" else 0.0
    return feats


def build_transition_example_feature_frame(
    examples_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    rows = [_feature_dict_from_example(rec) for rec in examples_df.to_dict(orient="records")]
    if feature_cols is None:
        cols = sorted({c for row in rows for c in row.keys()})
    else:
        cols = list(feature_cols)
    data = [{c: float(row.get(c, 0.0)) for c in cols} for row in rows]
    return pd.DataFrame(data, columns=cols), cols


def _valid_examples(examples_df: pd.DataFrame, *, continent_name: str, min_successor_count: int) -> pd.DataFrame:
    if examples_df.empty:
        return examples_df.copy()
    df = examples_df.copy()
    if "transition_example_status" in df.columns:
        df = df[df["transition_example_status"] == "ok"]
    if "continent_name" in df.columns:
        df = df[df["continent_name"] == continent_name]

    def enough_counts(v: Any) -> bool:
        if not isinstance(v, Mapping) or not v:
            return False
        return sum(float(x) for x in v.values()) >= float(min_successor_count)

    df = df[df["full_graph_successor_state_counts"].map(enough_counts)]
    return df.reset_index(drop=True)


def train_transition_distribution_model(
    examples_df: pd.DataFrame,
    *,
    continent_name: str,
    k_neighbors: int = 10,
    distance_metric: str = "euclidean",
    min_successor_count: int = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    df = _valid_examples(examples_df, continent_name=continent_name, min_successor_count=min_successor_count)
    if df.empty:
        raise ValueError(f"No valid transition-distribution examples for continent={continent_name!r}")
    feature_df, feature_cols = build_transition_example_feature_frame(df)
    X = feature_df.to_numpy(dtype=float)
    scaler = StandardFeatureScaler.fit(X)
    X_scaled = scaler.transform(X)
    full_nodes = tuple(sorted(int(x) for x in (df.iloc[0].get("full_graph_nodes") or ())))
    model = TransitionDistributionKNNModel(
        continent_name=str(continent_name),
        feature_cols=tuple(feature_cols),
        full_graph_nodes=full_nodes,
        k_neighbors=int(k_neighbors),
        distance_metric=str(distance_metric),
        scaler=scaler,
        X_train=X_scaled,
        example_records=tuple(df.to_dict(orient="records")),
    )
    support_sizes = [len(row.get("full_graph_successor_state_counts", {}) or {}) for row in model.example_records]
    return {
        "model_type": "transition_distribution_knn_v1",
        "schema_version": "transition_distribution_knn_v1",
        "continent_name": str(continent_name),
        "feature_cols": tuple(feature_cols),
        "full_graph_nodes": full_nodes,
        "k_neighbors": int(k_neighbors),
        "distance_metric": str(distance_metric),
        "scaler": scaler,
        "model": model,
        "n_examples": int(len(df)),
        "training_diagnostics": {
            "random_state": int(random_state),
            "min_successor_count": int(min_successor_count),
            "n_feature_cols": int(len(feature_cols)),
            "avg_successor_support_size": float(np.mean(support_sizes)) if support_sizes else 0.0,
            "max_successor_support_size": int(max(support_sizes)) if support_sizes else 0,
        },
    }


def save_transition_distribution_model_bundle(bundle: Mapping[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib  # type: ignore
        joblib.dump(dict(bundle), path)
    except Exception:
        with path.open("wb") as f:
            pickle.dump(dict(bundle), f)
    return path


def load_transition_distribution_model_bundle(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    try:
        import joblib  # type: ignore
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def train_transition_distribution_model_for_continent(
    *,
    continent: str,
    examples_pickle: Path | str,
    models_dir: Path | str = "models",
    k_neighbors: int = 10,
) -> Dict[str, Any]:
    examples_df = pd.read_pickle(examples_pickle)
    bundle = train_transition_distribution_model(
        examples_df,
        continent_name=str(continent),
        k_neighbors=int(k_neighbors),
    )
    out = Path(models_dir) / f"transition_distribution_models__{_slugify_continent(continent)}.joblib"
    save_transition_distribution_model_bundle(bundle, out)
    return bundle


def train_transition_distribution_models(
    *,
    continents: Optional[Sequence[str]] = None,
    datasets_dir: Path | str = "datasets",
    models_dir: Path | str = "models",
    k_neighbors: int = 10,
) -> Dict[str, Any]:
    selected = list(continents) if continents is not None else list(Board.continent_territory_dict.keys())
    out: Dict[str, Any] = {}
    for continent in selected:
        slug = _slugify_continent(str(continent))
        examples_pickle = Path(datasets_dir) / f"transition_distribution_examples__{slug}.pkl"
        if not examples_pickle.exists():
            out[str(continent)] = {"status": "missing_dataset", "examples_pickle": str(examples_pickle)}
            continue
        bundle = train_transition_distribution_model_for_continent(
            continent=str(continent),
            examples_pickle=examples_pickle,
            models_dir=models_dir,
            k_neighbors=int(k_neighbors),
        )
        out[str(continent)] = {
            "status": "ok",
            "n_examples": int(bundle.get("n_examples", 0)),
            "model_path": str(Path(models_dir) / f"transition_distribution_models__{slug}.joblib"),
        }
    return out


def load_transition_distribution_models_by_continent(
    models_dir: Path | str = "models",
    *,
    strict: bool = True,
) -> Dict[str, Dict[str, Any]]:
    models_dir = Path(models_dir)
    out: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for continent in Board.continent_territory_dict.keys():
        path = models_dir / f"transition_distribution_models__{_slugify_continent(continent)}.joblib"
        if not path.exists():
            missing.append(str(continent))
            continue
        out[str(continent)] = load_transition_distribution_model_bundle(path)
    if missing and strict:
        raise FileNotFoundError(f"Missing transition-distribution model files for: {missing}. Looked in {models_dir.resolve()}.")
    return out


def predict_successor_distribution_from_example(
    *,
    model_bundle: Mapping[str, Any],
    example: Mapping[str, Any],
    k: Optional[int] = None,
) -> Dict[Signature, float]:
    model = model_bundle.get("model")
    if not isinstance(model, TransitionDistributionKNNModel):
        raise TypeError("model_bundle['model'] is not a TransitionDistributionKNNModel")
    return model.predict_distribution(example, k=k)


def sample_successor_signatures_from_distribution(
    successor_distribution: Mapping[Any, float],
    *,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Signature]:
    items = [(normalize_state_signature(sig), float(prob)) for sig, prob in successor_distribution.items() if float(prob) > 0]
    if not items or int(n_samples) <= 0:
        return []
    total = float(sum(prob for _, prob in items))
    probs = np.asarray([prob / total for _, prob in items], dtype=float)
    gen = rng if rng is not None else np.random.default_rng()
    idx = gen.choice(len(items), size=int(n_samples), replace=True, p=probs)
    return [items[int(i)][0] for i in idx]


def _top_signature(distribution: Mapping[Any, float]) -> Optional[Signature]:
    if not distribution:
        return None
    items = [(normalize_state_signature(sig), float(v)) for sig, v in distribution.items()]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items[0][0]


def _support_stats(distribution: Mapping[Any, float]) -> Tuple[int, float]:
    if not distribution:
        return 0, 0.0
    vals = [float(v) for v in distribution.values()]
    return len(vals), max(vals) / sum(vals) if sum(vals) > 0 else 0.0


def _distribution_to_probabilities(counts: Mapping[Any, Any]) -> Dict[Signature, float]:
    total = float(sum(float(v) for v in counts.values()))
    if total <= 0:
        return {}
    return {normalize_state_signature(sig): float(v) / total for sig, v in counts.items()}


def _marginal_errors(true_dist: Mapping[Any, float], pred_dist: Mapping[Any, float], full_graph_nodes: Sequence[int]) -> Tuple[float, float]:
    class TinyGraph:
        def __init__(self, nodes):
            self._nodes = tuple(nodes)
        def nodes(self):
            return list(self._nodes)

    graph = TinyGraph(full_graph_nodes)
    true_m = derive_node_marginals_from_successor_distribution(successor_state_counts=true_dist, full_graph=graph)
    pred_m = derive_node_marginals_from_successor_distribution(successor_state_counts=pred_dist, full_graph=graph)
    if not full_graph_nodes:
        return 0.0, 0.0
    p_err = []
    t_err = []
    for node in full_graph_nodes:
        p_err.append(abs(float(true_m[int(node)]["p_attacker_final"]) - float(pred_m[int(node)]["p_attacker_final"])))
        t_err.append(abs(float(true_m[int(node)]["expected_troops"]) - float(pred_m[int(node)]["expected_troops"])))
    return float(np.mean(p_err)), float(np.mean(t_err))


def evaluate_transition_distribution_model(
    examples_df: pd.DataFrame,
    *,
    continent_name: str,
    k_neighbors: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    df = _valid_examples(examples_df, continent_name=continent_name, min_successor_count=1)
    if len(df) < 2:
        raise ValueError("Need at least two valid examples for evaluation.")
    rng = np.random.default_rng(int(random_state))
    perm = rng.permutation(len(df))
    n_test = max(1, min(len(df) - 1, int(round(float(test_size) * len(df)))))
    test_idx = set(int(i) for i in perm[:n_test])
    train_df = df.iloc[[i for i in range(len(df)) if i not in test_idx]].reset_index(drop=True)
    test_df = df.iloc[[i for i in range(len(df)) if i in test_idx]].reset_index(drop=True)
    bundle = train_transition_distribution_model(
        train_df,
        continent_name=continent_name,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    top_matches = []
    p_errors = []
    troop_errors = []
    true_support = []
    pred_support = []
    true_top_prob = []
    pred_top_prob = []
    full_nodes = tuple(int(x) for x in bundle.get("full_graph_nodes", ()))
    for rec in test_df.to_dict(orient="records"):
        true_dist = _distribution_to_probabilities(rec.get("full_graph_successor_state_counts", {}) or {})
        pred_dist = predict_successor_distribution_from_example(model_bundle=bundle, example=rec)
        top_matches.append(int(_top_signature(true_dist) == _top_signature(pred_dist)))
        pe, te = _marginal_errors(true_dist, pred_dist, full_nodes)
        p_errors.append(pe)
        troop_errors.append(te)
        ts, tt = _support_stats(true_dist)
        ps, pt = _support_stats(pred_dist)
        true_support.append(ts)
        pred_support.append(ps)
        true_top_prob.append(tt)
        pred_top_prob.append(pt)
    return {
        "continent_name": str(continent_name),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "top1_exact_match_rate": float(np.mean(top_matches)) if top_matches else 0.0,
        "mean_abs_error_p_attacker_final": float(np.mean(p_errors)) if p_errors else 0.0,
        "mean_abs_error_expected_troops": float(np.mean(troop_errors)) if troop_errors else 0.0,
        "avg_true_support_size": float(np.mean(true_support)) if true_support else 0.0,
        "avg_pred_support_size": float(np.mean(pred_support)) if pred_support else 0.0,
        "avg_top1_true_prob": float(np.mean(true_top_prob)) if true_top_prob else 0.0,
        "avg_top1_pred_prob": float(np.mean(pred_top_prob)) if pred_top_prob else 0.0,
    }
