"""
train_ML.py

End-to-end setup for the NEW per-node macro→micro approach:

1) Generate node-level transition dataset using Monte Carlo wave-1 outcomes.
2) Train:
     - a capture model: P(attacker holds node after the turn)
     - a troop model:   E[troops on attacker-held node after the turn)
3) Save the dataset and models to disk.

"""

from pathlib import Path

import re
import argparse

import numpy as np
import pandas as pd


from project_risk.game_simulation import Board

from project_risk.mathematical.transition_prediction_ml.state_generators import ml_full_graph_state_generator
from project_risk.mathematical.transition_prediction_ml.generate_data_ML import (
    ExperimentConstraints, ExperimentConfig,
    TransitionDistributionConfig,
    run_node_transition_experiment,
    run_transition_distribution_experiment)
from project_risk.mathematical.libraries.create_library import lowest_lib_node_max

import logging
from project_risk.infrastructure import log_config as lc
from project_risk.mathematical.transition_prediction_ml.transition_distribution_ML import (
    train_transition_distribution_model_for_continent as _train_td_for_continent,
    train_transition_distribution_models as _train_td_models,
)

# ----------------------------------------------------------------------
# 0) Logging settings
# ----------------------------------------------------------------------

lc.set_debug_switches({
    "runner": False,        # progress/info
    "battle_graph": True,
    "sampler": False,
    "query": True,
    "ranking": False,
    "partition": True,
    "rollout": False,
})

# ----------------------------------------------------------------------
# 1) Experiment configuration for node-level dataset
# ----------------------------------------------------------------------

def build_experiment_config() -> ExperimentConfig:
    """
    Build ExperimentConfig for the node-level ML training.

    Key choices:
      - territory_ratios: cover more extremes (attacker from very weak to very strong)
      - troops_ratios:    include cases where attacker is far behind / ahead in troops
      - per-node caps:    slightly higher than before to allow larger stacks

    DEFAULT POLICY + LABEL SEMANTICS (PATCH):
      - evaluation_mode="one_wave"  -> greedy partition selection (1-wave evaluation)
      - rollout_steps=2             -> labels are the end state after TWO rollouts of:
                                      (partition -> sample -> update -> rebuild)
    """
    # Expanded coverage over attacker territory ratios:
    #  - very weak:  0.05, 0.1
    #  - mid:        0.2 .. 0.8
    #  - very strong:0.9, 0.95
    territory_ratios = [
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 0.95,
    ]

    # Expanded coverage over attacker "available troops" ratio:
    #  - clearly behind:   0.1, 0.2, 0.3, 0.4, 0.5
    #  - around parity:    0.7, 1.0
    #  - far ahead:        1.5, 2.0, 2.5, 3.0
    troops_ratios = [
        0.10, 0.20, 0.30, 0.40, 0.50, 0.70,
        1.00, 1.50, 2.00, 2.50, 3.00,
    ]

    config = ExperimentConfig(
        territory_ratios=territory_ratios,
        troops_ratios=troops_ratios,
        samples_per_combo=5,   # you can lower this if total runtime explodes
        max_partitions=30,
        random_seed=42,
        constraints=ExperimentConstraints(
            continent_name="North America",
            max_attacker_troops_per_node=lowest_lib_node_max,
            max_defender_troops_per_node=lowest_lib_node_max,
        ),
        evaluation_mode="one_wave",  # greedy partition selection
        rollout_steps=2,             # <-- PATCH: two-rollout end-state labels by default
    )

    # Optional overrides for MC:
    # config.num_scenarios_two_wave = 50
    # config.min_state_prob_two_wave = 0.0
    # config.max_end_states_per_region_two_wave = None

    return config



# ----------------------------------------------------------------------
# 2) Generate node-level transition dataset
# ----------------------------------------------------------------------

def generate_node_dataset(
    config: ExperimentConfig,
    libraries_base: Path,
    out_pickle: Path,
) -> pd.DataFrame:
    """
    Runs the node-level experiment and saves the resulting DataFrame.

    PATCH:
      - Adds a hard "stop training on junk" gate:
          If used_noop_scenario exists and >95% are noop -> raise RuntimeError.
      - Also prints mean coverage_wave1 and captured_rate when available.
    """
    print("Running node-level transition experiment...")
    df_nodes = run_node_transition_experiment(
        config=config,
        state_generator=ml_full_graph_state_generator,
        combat_libraries_base=libraries_base,
        num_scenarios_wave1=getattr(config, "num_scenarios_two_wave", 50),
        rollout_steps=getattr(config, "rollout_steps", 2),
        n_jobs=4,   # <-- choose based on CPU cores (start with 4–8)
    )

    print(f"Node-level dataset shape: {df_nodes.shape}")

    # --------------------------------------------------------------
    # PATCH: stop training on junk data (noop fallback everywhere)
    # --------------------------------------------------------------
    if "used_noop_scenario" in df_nodes.columns:
        frac_noop = float(df_nodes["used_noop_scenario"].mean())
        print(f"[diag] frac_noop={frac_noop:.4f}")

        if "coverage_wave1" in df_nodes.columns:
            mean_cov = float(df_nodes["coverage_wave1"].mean())
            print(f"[diag] mean_coverage_wave1={mean_cov:.6f}")

        if "captured" in df_nodes.columns:
            captured_rate = float(df_nodes["captured"].mean())
            print(f"[diag] captured_rate={captured_rate:.6f}")

        if frac_noop > 0.95:
            raise RuntimeError(
                "Too many noop scenarios; dataset is not usable for training. "
                "Fix sampling/library lookup first."
            )

    df_nodes.to_pickle(out_pickle)
    print(f"Saved node-level transition results to {out_pickle}")
    return df_nodes


def generate_transition_distribution_dataset(
    config: ExperimentConfig,
    transition_config: TransitionDistributionConfig,
    libraries_base: Path,
    out_examples_pickle: Path,
    out_node_marginals_pickle: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stage-A dataset generation only. This does not train or save any new model.
    """
    examples_df, node_marginals_df = run_transition_distribution_experiment(
        config=config,
        transition_config=transition_config,
        state_generator=ml_full_graph_state_generator,
        combat_libraries_base=libraries_base,
        n_jobs=1,
    )
    examples_df.to_pickle(out_examples_pickle)
    node_marginals_df.to_pickle(out_node_marginals_pickle)
    print(f"Saved transition-distribution examples to {out_examples_pickle}")
    print(f"Saved transition-distribution node marginals to {out_node_marginals_pickle}")
    return examples_df, node_marginals_df


def train_transition_distribution_model_for_continent(
    *,
    continent: str,
    examples_pickle: Path | str,
    models_dir: Path | str = "models",
    k_neighbors: int = 10,
) -> dict:
    return _train_td_for_continent(
        continent=continent,
        examples_pickle=examples_pickle,
        models_dir=models_dir,
        k_neighbors=k_neighbors,
    )


def train_transition_distribution_models(
    *,
    continents: list[str] | tuple[str, ...] | None = None,
    datasets_dir: Path | str = "datasets",
    models_dir: Path | str = "models",
    k_neighbors: int = 10,
) -> dict:
    return _train_td_models(
        continents=continents,
        datasets_dir=datasets_dir,
        models_dir=models_dir,
        k_neighbors=k_neighbors,
    )




# ----------------------------------------------------------------------
# 3) Train per-node models:
#    - capture_model: P(attacker_holds_final == 1)
#    - troop_model:   E[final_troops | attacker_holds_final == 1]
# ----------------------------------------------------------------------

def sanity_check_node_dataset(df_nodes: pd.DataFrame) -> None:
    """
    Print basic sanity statistics for the node-level dataset.

    PATCH:
      - Adds checks for newly introduced local node features used to improve pA realism:
          deg_full, enemy/friendly neighbor counts, troop sums, frontier flag,
          pressure_ratio, local_balance.
      - Reports whether they exist, NaN counts, basic distribution summaries,
        and a quick “all zeros?” diagnostic.
    """
    print("\n=== NODE DATASET SANITY CHECK ===")
    print(f"Total rows: {len(df_nodes)}")
    print(f"Columns: {sorted(df_nodes.columns.tolist())}")

    # 1) Attack perspective counts
    if "attack_perspective" in df_nodes.columns:
        print("\n[attack_perspective]")
        print(df_nodes["attack_perspective"].value_counts(dropna=False))
    else:
        print("\n[attack_perspective] column missing (ok if older dataset)")

    # 2) attacker_holds_final distribution
    if "attacker_holds_final" in df_nodes.columns:
        print("\n[attacker_holds_final counts]")
        print(df_nodes["attacker_holds_final"].value_counts(dropna=False))

        print("\n[attacker_holds_final proportion]")
        print(df_nodes["attacker_holds_final"].value_counts(normalize=True))
    else:
        print("\n[attacker_holds_final] column missing – this should NOT happen.")

    # 3) attacker_holds_final per perspective
    if "attack_perspective" in df_nodes.columns and "attacker_holds_final" in df_nodes.columns:
        print("\n[attacker_holds_final mean per attack_perspective]")
        print(
            df_nodes.groupby("attack_perspective")["attacker_holds_final"]
            .mean()
            .rename("P(attacker holds)")
        )

    # 4) is_battle_node frequency
    if "is_battle_node" in df_nodes.columns:
        print("\n[is_battle_node frequency]")
        print(df_nodes["is_battle_node"].value_counts(dropna=False))
        try:
            print(
                "Proportion of nodes that are battle nodes: "
                f"{df_nodes['is_battle_node'].mean():.3f}"
            )
        except Exception:
            pass
    else:
        print("\n[is_battle_node] column missing – check collect_node_outcome_rows_for_state.")

    # 5) Macro feature basic stats
    macro_cols = [
        "battle_realized_attacker_territory_ratio",
        "battle_realized_attacker_available_troops_ratio",
        "full_realized_attacker_territory_ratio",
        "full_realized_attacker_troops_ratio",
        "full_realized_attacker_available_troops_ratio",
    ]

    existing_macro_cols = [c for c in macro_cols if c in df_nodes.columns]
    if existing_macro_cols:
        print("\n[Macro feature summary]")
        print(
            df_nodes[existing_macro_cols]
            .describe(percentiles=[0.05, 0.5, 0.95])
            .T[["min", "5%", "50%", "95%", "max"]]
        )
    else:
        print("\nNo expected macro feature columns found – check run_node_transition_experiment.")

    # 6) NEW: Local node feature checks
    local_cols = [
        "deg_full",
        "enemy_neighbor_count",
        "friendly_neighbor_count",
        "enemy_neighbor_troops_sum",
        "friendly_neighbor_troops_sum",
        "max_enemy_neighbor_troops",
        "max_friendly_neighbor_troops",
        "is_frontier_node",
        "pressure_ratio",
        "local_balance",
    ]
    existing_local_cols = [c for c in local_cols if c in df_nodes.columns]

    if existing_local_cols:
        print("\n[Local node feature presence]")
        missing_local = [c for c in local_cols if c not in df_nodes.columns]
        if missing_local:
            print(f"  Missing local cols (ok during rollout, not ok for final): {missing_local}")
        else:
            print("  All expected local cols present ✅")

        print("\n[Local feature NaN counts]")
        print(df_nodes[existing_local_cols].isna().sum())

        # quick “all zeros / constant?” diagnostics
        print("\n[Local feature constant/zero diagnostics]")
        diag_rows = []
        for c in existing_local_cols:
            s = df_nodes[c]
            # treat non-numeric gracefully
            try:
                nuniq = int(s.nunique(dropna=True))
                mean = float(s.mean())
                std = float(s.std())
                frac_zero = float((s == 0).mean()) if (s.dtype.kind in "if") else float("nan")
            except Exception:
                nuniq, mean, std, frac_zero = -1, float("nan"), float("nan"), float("nan")
            diag_rows.append((c, nuniq, mean, std, frac_zero))
        # pretty print without pandas dependency beyond what's already present
        for c, nuniq, mean, std, frac_zero in diag_rows:
            print(f"  {c:28s} nunique={nuniq:4d} mean={mean:10.4f} std={std:10.4f} frac_zero={frac_zero:7.3f}")

        # distribution summary
        print("\n[Local feature summary]")
        try:
            print(
                df_nodes[existing_local_cols]
                .describe(percentiles=[0.05, 0.5, 0.95])
                .T[["min", "5%", "50%", "95%", "max"]]
            )
        except Exception as e:
            print(f"  (Could not compute describe() for local features: {e})")
    else:
        print("\n[Local node features] None found. "
              "If you've patched dataset generation, this indicates local features were not injected.")

    # 7) NaN counts in important columns
    important_cols = [
        "attacker_holds_final",
        "final_troops",
        "initial_troops",
        "is_battle_node",
    ]
    important_cols += existing_macro_cols
    important_cols += existing_local_cols

    existing_imp = [c for c in important_cols if c in df_nodes.columns]
    if existing_imp:
        print("\n[NaN counts in important columns]")
        print(df_nodes[existing_imp].isna().sum())
    else:
        print("\nNo important columns found to check NaNs – suspicious.")
    
    if "rollout_steps" in df_nodes.columns:
        print("\n[rollout_steps]")
        print(df_nodes["rollout_steps"].value_counts(dropna=False))

    print("=== END SANITY CHECK ===\n")



def prepare_features(df_nodes: pd.DataFrame):
    """
    Build feature matrices for node-level models.

    PATCH:
      - Adds the hard noop gate here as well (defense-in-depth).
      - Includes local neighborhood / frontier pressure node features.
      - Enforces consistent feature schema (including owner dummies).
      - Sanitizes NaN/inf in features.
    """
    # --------------------------------------------------------------
    # PATCH: stop training on junk data (noop fallback everywhere)
    # --------------------------------------------------------------
    if "used_noop_scenario" in df_nodes.columns:
        frac_noop = float(df_nodes["used_noop_scenario"].mean())
        print(f"[diag] frac_noop={frac_noop:.4f}")

        if "coverage_wave1" in df_nodes.columns:
            mean_cov = float(df_nodes["coverage_wave1"].mean())
            print(f"[diag] mean_coverage_wave1={mean_cov:.6f}")

        if "captured" in df_nodes.columns:
            captured_rate = float(df_nodes["captured"].mean())
            print(f"[diag] captured_rate={captured_rate:.6f}")

        if frac_noop > 0.95:
            raise RuntimeError(
                "Too many noop scenarios; dataset is not usable for training. "
                "Fix sampling/library lookup first."
            )

    df = df_nodes.copy()

    # One-hot encode initial_owner ("A", "D")
    df = pd.get_dummies(df, columns=["initial_owner"], prefix="init_owner")

    # ------------------------------------------------------------------
    # Macro-level features (pre-combat only, no leakage)
    # ------------------------------------------------------------------
    macro_core = [
        "battle_realized_attacker_territory_ratio",
        "battle_realized_attacker_available_troops_ratio",
        "full_realized_attacker_territory_ratio",
        "full_realized_attacker_troops_ratio",
        "full_realized_attacker_available_troops_ratio",
    ]

    macro_distribution = [
        "battle_realized_attacker_troops_distribution_cv",
        "battle_realized_attacker_troops_distribution_gini",
        "full_realized_attacker_troops_distribution_cv",
        "full_realized_attacker_troops_distribution_gini",
    ]

    macro_topology = [
        "full_realized_topology_degree_mean",
        "full_realized_topology_degree_variance",
        "full_realized_topology_component_count",
        "full_realized_topology_diameter",
        # Optional but present in your dataset:
        "full_realized_topology_edge_count",
    ]

    macro_counts = [
        "battle_initial_attacker_territory_count",
        "battle_initial_attacker_troops_count",
        "battle_total_territory_count",
        "battle_total_troops_count",
        "full_realized_attacker_territory_count",
        "full_realized_defender_territory_count",
        "full_realized_total_territory_count",
        "full_realized_attacker_troops_count",
        "full_realized_defender_troops_count",
        "full_realized_total_troops_count",
    ]

    all_macro_candidates = macro_core + macro_distribution + macro_topology + macro_counts
    macro_feature_cols = [c for c in all_macro_candidates if c in df.columns]

    # ------------------------------------------------------------------
    # Node-level features
    # ------------------------------------------------------------------
    node_feature_cols = [
        "initial_troops",
        "is_battle_node",
    ]

    # Local neighborhood / frontier pressure features
    local_node_candidates = [
        "deg_full",
        "enemy_neighbor_count",
        "friendly_neighbor_count",
        "enemy_neighbor_troops_sum",
        "friendly_neighbor_troops_sum",
        "max_enemy_neighbor_troops",
        "max_friendly_neighbor_troops",
        "is_frontier_node",
        "pressure_ratio",
        "local_balance",
    ]
    node_feature_cols.extend([c for c in local_node_candidates if c in df.columns])

    # Add all dummy columns for initial_owner
    # Ensure both expected dummies exist even if one class is absent in a dataset slice.
    for expected_dummy in ("init_owner_A", "init_owner_D"):
        if expected_dummy not in df.columns:
            df[expected_dummy] = 0

    extra_dummies = [c for c in df.columns if c.startswith("init_owner_")]
    node_feature_cols.extend(extra_dummies)

    # Final feature list (preserve order; drop accidental duplicates)
    feature_cols = []
    for c in (macro_feature_cols + node_feature_cols):
        if c not in feature_cols:
            feature_cols.append(c)

    # ------------------------------------------------------------------
    # Ensure all feature columns exist (robustness across datasets)
    # ------------------------------------------------------------------
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # ------------------------------------------------------------------
    # SANITIZE FEATURES: replace inf with NaN, then fill NaN with 0.0
    # ------------------------------------------------------------------
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # ------------------------------------------------------------------
    # Feature matrix for capture model (all nodes)
    # ------------------------------------------------------------------
    X = df[feature_cols].to_numpy(dtype=float)

    # Attacker capture target
    if "attacker_holds_final" not in df.columns:
        raise KeyError("Missing required column 'attacker_holds_final' in df_nodes.")
    y_capture_att = df["attacker_holds_final"].astype(int).to_numpy()

    # Defender capture derived (no neutral): def = 1 - att
    df["defender_holds_final"] = 1 - df["attacker_holds_final"].astype(int)

    # ------------------------------------------------------------------
    # Troop models: attacker-held vs defender-held nodes
    # ------------------------------------------------------------------
    if "final_troops" not in df.columns:
        raise KeyError("Missing required column 'final_troops' in df_nodes.")

    mask_att = df["attacker_holds_final"] == 1
    df_att = df.loc[mask_att].copy()
    X_troops_att = df_att[feature_cols].to_numpy(dtype=float)
    y_troops_att = df_att["final_troops"].astype(int).to_numpy()

    mask_def = df["defender_holds_final"] == 1
    df_def = df.loc[mask_def].copy()
    X_troops_def = df_def[feature_cols].to_numpy(dtype=float)
    y_troops_def = df_def["final_troops"].astype(int).to_numpy()

    return (
        df,
        feature_cols,
        X,
        y_capture_att,
        X_troops_att,
        y_troops_att,
        X_troops_def,
        y_troops_def,
    )



def train_capture_model(X, y_capture, random_state: int = 42):
    """
    Train a simple RandomForest classifier for P(attacker_holds_final == 1).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_capture, test_size=0.2, random_state=random_state, stratify=y_capture
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Basic AUC evaluation
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Capture model ROC-AUC: {auc:.3f}")

    return clf


def train_troop_model(X_troops, y_troops, random_state: int = 42):
    """
    Train a RandomForest regressor for E[final_troops | holder == attacker/defender].
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_troops, y_troops, test_size=0.2, random_state=random_state
    )

    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    print(f"Troop model RMSE: {rmse:.3f}")

    return reg


# ----------------------------------------------------------------------
# 4) Main entry point: run everything
# ----------------------------------------------------------------------

def _slugify_continent(name: str) -> str:
    """Filesystem-safe slug."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_continent"


def list_all_continents() -> list[str]:
    """Source of truth: Board.continent_territory_dict keys."""
    try:
        return list(Board.continent_territory_dict.keys())
    except Exception as e:
        raise RuntimeError("Could not read Board.continent_territory_dict to list continents.") from e



def main(
    *,
    continent: str | None = None,
    libraries_base: Path | str = "small_graph_libraries",
    datasets_dir: Path | str = "datasets",
    models_dir: Path | str = "models",
) -> None:
    import joblib

    libraries_base = Path(libraries_base)

    datasets_dir = Path(datasets_dir)
    models_dir = Path(models_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    base_config = build_experiment_config()

    all_continents = list_all_continents()

    # Choose which continents to train
    if continent is None:
        continents = all_continents
        print(f"Training per-continent models for {len(continents)} continents: {continents}")
    else:
        if continent not in all_continents:
            raise ValueError(
                f"Unknown continent={continent!r}. "
                f"Valid options: {all_continents}"
            )
        continents = [continent]
        print(f"Training SINGLE continent: {continent}")

    for continent_name in continents:
        slug = _slugify_continent(continent_name)

        print("" + "=" * 80)
        print(f"[CONTINENT] {continent_name} (slug={slug})")
        print("=" * 80)

        # Clone config to avoid accidental cross-continent mutation
        config = ExperimentConfig(
            territory_ratios=list(base_config.territory_ratios),
            troops_ratios=list(base_config.troops_ratios),
            samples_per_combo=int(base_config.samples_per_combo),
            max_partitions=int(base_config.max_partitions),
            ranking_variable=base_config.ranking_variable,
            random_seed=base_config.random_seed,
            constraints=ExperimentConstraints(
                continent_name=continent_name,
                max_attacker_troops_per_node=int(base_config.constraints.max_attacker_troops_per_node),
                max_defender_troops_per_node=int(base_config.constraints.max_defender_troops_per_node),
                max_nodes=base_config.constraints.max_nodes,
            ),
            evaluation_mode=base_config.evaluation_mode,
            rollout_steps=int(getattr(base_config, "rollout_steps", 2)),
            policy_option_selection=getattr(base_config, "policy_option_selection", "primary"),
            raise_state_generator_exceptions=bool(getattr(base_config, "raise_state_generator_exceptions", False)),
            include_state_generator_traceback=bool(getattr(base_config, "include_state_generator_traceback", False)),
        )

        out_nodes_pickle = datasets_dir / f"node_transition_results__{slug}.pkl"
        out_models_joblib = models_dir / f"node_level_models__{slug}.joblib"

        df_nodes = generate_node_dataset(
            config=config,
            libraries_base=libraries_base,
            out_pickle=out_nodes_pickle,
        )

        # --------------------------------------------------------------
        # Gate here too (fast fail before any further work)
        # --------------------------------------------------------------
        if "used_noop_scenario" in df_nodes.columns:
            frac_noop = float(df_nodes["used_noop_scenario"].mean())
            print(f"[diag] frac_noop={frac_noop:.4f}")

            if "coverage_wave1" in df_nodes.columns:
                print(f"[diag] mean_coverage_wave1={float(df_nodes['coverage_wave1'].mean()):.6f}")

            if "captured" in df_nodes.columns:
                print(f"[diag] captured_rate={float(df_nodes['captured'].mean()):.6f}")

            if frac_noop > 0.95:
                raise RuntimeError(
                    f"[{continent_name}] Too many noop scenarios; dataset is not usable for training. "
                    "Fix sampling/library lookup first."
                )

        sanity_check_node_dataset(df_nodes)

        (
            df_nodes,
            feature_cols,
            X,
            y_capture_att,
            X_troops_att,
            y_troops_att,
            X_troops_def,
            y_troops_def,
        ) = prepare_features(df_nodes)

        # --- attacker models ---
        print("Training attacker capture model.")
        capture_model_att = train_capture_model(X, y_capture_att)

        print("Training attacker troop model.")
        troop_model_att = train_troop_model(X_troops_att, y_troops_att)

        # --- defender models ---
        print("Training defender troop model.")
        troop_model_def = train_troop_model(X_troops_def, y_troops_def)

        joblib.dump(
            {
                "continent_name": continent_name,
                "capture_model_attacker": capture_model_att,
                "troop_model_attacker": troop_model_att,
                "troop_model_defender": troop_model_def,
                "feature_cols": feature_cols,
            },
            out_models_joblib,
        )
        print(f"Saved node-level models to {out_models_joblib}")
        print(f"Saved node-level dataset to {out_nodes_pickle}")


def _parse_args(continent_name: str):
    ap = argparse.ArgumentParser(description="Train ML models for Risk battle prediction (per-continent).")
    ap.add_argument("--continent", type=str, default=continent_name,
                    help="Train ONLY this continent (exact name from Board.continent_territory_dict). "
                         "If omitted, trains all continents.")
    ap.add_argument("--list-continents", action="store_true",
                    help="Print valid continent names and exit.")
    ap.add_argument("--libraries-base", type=str, default="small_graph_libraries",
                    help="Base folder containing small-graph combat libraries.")
    ap.add_argument("--datasets-dir", type=str, default="datasets",
                    help="Output folder for per-continent datasets.")
    ap.add_argument("--models-dir", type=str, default="models",
                    help="Output folder for per-continent model bundles.")
    return ap.parse_args()


if __name__ == "__main__":
    lc.setup_logging(level="INFO", log_file="run.log")
    continent_name = "Australia"
    args = _parse_args(continent_name)

    if args.list_continents:
        print(list_all_continents())
        raise SystemExit(0)

    main(
        continent=args.continent,
        libraries_base=args.libraries_base,
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir,
    )
