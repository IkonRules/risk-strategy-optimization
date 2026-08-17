from __future__ import annotations

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, List, Union, Tuple
import pandas as pd
import numpy as np
from project_risk.mathematical.small_graph_model.markov_matrix_probabilities import battle_summary
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
    build_absorption_tables_two_player,
    build_absorption_tables_two_player_with_plateau,
    encode_state_label,
    canonicalize_edges_with_roles,
    generate_connected_graphs_n_nodes
    )

import time
import itertools
import shutil
import os

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - library query/single-process builds do not need joblib
    Parallel = None
    delayed = None

# ---------------------------------------------------------------------
# Exact finite compact solver integration (experimental main candidate)
# ---------------------------------------------------------------------
try:
    from project_risk.mathematical.small_graph_model.exact_finite_solver import (
        CompactExactTopologySolver,
        combat_df_for_caps,
    )
except Exception:  # pragma: no cover - keeps legacy plateau builder importable
    CompactExactTopologySolver = None
    combat_df_for_caps = None



'''
This module is made to run on different cores in order to create libraries faster. 
AS OF YET, UNTESTED.
'''




# ---------------------------------------------------------------------
# Global combat_df (2-node combat outcome table)
# ---------------------------------------------------------------------

# You can reuse this combat_df both for precomputing and for lazy-building.
res = battle_summary(15, 15)
combat_df = res["F_df"]


# ---------------------------------------------------------------------
# Graph key / identity helpers
# ---------------------------------------------------------------------



def make_graph_key(edges, num_attacker_nodes: int, num_defender_nodes: int) -> str:
    """
    Create a stable string key for a graph topology + ownership pattern.
    Canonicalizes topology under A/D-preserving permutations so isomorphic
    graphs share the same key.
    """
    canonical_edges_key, _, _ = canonicalize_edges_with_roles(
        edges=sorted(tuple(sorted(e)) for e in edges),
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
    )

    topology = {"edges": list(canonical_edges_key)}
    owner_pattern = {
        "num_attacker_nodes": num_attacker_nodes,
        "num_defender_nodes": num_defender_nodes,
    }
    key_dict = {"topology": topology, "owners": owner_pattern}
    return json.dumps(key_dict, sort_keys=True)



# ---------------------------------------------------------------------
# Save / load helpers (work with BOTH per-graph and multi-graph dicts)
# ---------------------------------------------------------------------


def save_library(library: Dict[str, Any], path: Path | str) -> None:
    """
    Save a library dict to disk using pickle.

    Supports:
      - legacy multi-graph libraries with 'graphs'
      - classic per-graph libraries with 'prob_table' (DataFrame or 0.0)
      - NEW per-graph libraries with 'prob_table_chunked' (disk-backed chunks)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(library, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_library(path: Path | str) -> Dict[str, Any]:
    """
    Load a previously saved library dict from disk.

    Returns
    -------
    library : dict
        Library structure.
    """
    path = Path(path)
    with path.open("rb") as f:
        library = pickle.load(f)
    return library


# ---------------------------------------------------------------------
# File naming / path helpers for PER-GRAPH libraries
# ---------------------------------------------------------------------


BASE_LIB_DIR = Path("small_graph_libraries")


def ensure_dir(path: Path) -> None:
    """
    Ensure that the given directory path exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def graph_filename(
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops: int,
    max_defender_troops: int,
    edges,
) -> str:
    """
    Filename for a specific graph topology and troop caps, for a
    fixed role pattern (nA, nD).

    The filename encodes:
      - nA, nD
      - maxA, maxD
      - a stable 8-hex-digit hash of the JSON graph key (topology+owners)
    """
    key = make_graph_key(edges, num_attacker_nodes, num_defender_nodes)
    # Stable hash via SHA1, first 8 hex chars
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return (
        f"graph_{num_attacker_nodes}A_{num_defender_nodes}D_"
        f"A{max_attacker_troops}_D{max_defender_troops}_{h}.pkl"
    )


def graph_path(
    edges,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops: int,
    max_defender_troops: int,
    base_dir: Path = BASE_LIB_DIR,
) -> Path:
    """
    Full path for a per-graph library file.

    Directory structure:
        {base_dir}/{nA}A_{nD}D/A{maxA}_D{maxD}/
            graph_{nA}A_{nD}D_A{maxA}_D{maxD}_{HASH}.pkl
    """
    subdir = (
        base_dir
        / f"{num_attacker_nodes}A_{num_defender_nodes}D"
        / f"A{max_attacker_troops}_D{max_defender_troops}"
    )
    ensure_dir(subdir)
    filename = graph_filename(
        num_attacker_nodes,
        num_defender_nodes,
        max_attacker_troops,
        max_defender_troops,
        edges,
    )
    return subdir / filename


def _edges_to_norm(e):
    """
    Normalize any of:
      - tuple/list of 2-int tuples
      - set of 2-int tuples
    into: tuple(sorted((min(u,v), max(u,v)) ...))
    """
    if e is None:
        return None
    try:
        return tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in list(e)))
    except Exception:
        return None


# ---------------------------------------------------------------------
# PATCH: parallel worker for plateau table building
# ---------------------------------------------------------------------


def _build_plateau_tables_worker(
    *,
    combat_df,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_A_exact: int,
    max_D_exact: int,
    max_A_extended: int,
    max_D_extended: int,
    high_min_att_troops: int,
    canonical_edges_list_chunk,
    chunk_rows: int,
    combo_folder: Path,
    worker_id: int,
    auto_chunk_threshold_rows: int,
    utility_mode: str = "legacy",
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_split_depth: Optional[int] = 1,
):
    """Build plateau tables for a subset of canonical graphs.

    Safety/Correctness notes
    ------------------------
    - Each worker writes chunked row files to its own unique directory to
      avoid collisions (e.g. _chunked_rows_w00, _chunked_rows_w01, ...).
    - The returned per-graph descriptors include their own chunk_dir, so
      later lookup works regardless of which worker produced the graph.
    """
    chunk_rel_prefix = f"_chunked_rows_w{int(worker_id):02d}"
    chunk_root_dir = combo_folder / chunk_rel_prefix

    tables_extended, plateau_policies = build_absorption_tables_two_player_with_plateau(
        combat_df=combat_df,
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
        max_attacker_troops_exact=max_A_exact,
        max_defender_troops_exact=max_D_exact,
        max_attacker_troops_extended=max_A_extended,
        max_defender_troops_extended=max_D_extended,
        high_min_att_troops=high_min_att_troops,
        edges_list=None,
        canonical_edges_list=canonical_edges_list_chunk,
        output_format="chunked_rows",
        chunk_rows=chunk_rows,
        chunk_root_dir=chunk_root_dir,
        chunk_rel_prefix=chunk_rel_prefix,
        dtype=np.float32,
        auto_chunk_threshold_rows=auto_chunk_threshold_rows,
        utility_mode=utility_mode,
        value_tolerances=value_tolerances,
        include_no_gain_in_value=include_no_gain_in_value,
        multi_policy_options=multi_policy_options,
        policy_option_mode=policy_option_mode,
        max_policy_options_per_row=max_policy_options_per_row,
        max_options_per_state=max_options_per_state,
        max_split_depth=max_split_depth,
    )
    return tables_extended, plateau_policies



# ---------------------------------------------------------------------
# Per-graph probability table access (lazy build or precomputed)
# ---------------------------------------------------------------------


def get_prob_table(
    edges: tuple,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops: int,
    max_defender_troops: int,
    base_dir: Path = BASE_LIB_DIR,
    *,
    # Hard switch: no lazy builds here
    lazy_build: bool = False,
    include_policies: bool = False,
):
    """Hard-switch per-graph probability table lookup (V2 only).

    Returns
    -------
    dict
        V2 chunked descriptor (on-disk row-store descriptor) with keys:
          - format == "chunked_prob_table_v2_rows_v1"
          - chunks : list[str]                 (filenames like "chunk_000123.pkl")
          - row_to_chunk : dict[str,int]
          - chunk_dir : str                    (relative folder next to the .pkl)
          - _base_path : str                   (ABSOLUTE path to the chunk folder!)
          - library_pkl_path : str             (ABSOLUTE .pkl path)

    Notes
    -----
    - In your v2 format, chunk files live under:
        <library_pkl_path.parent> / <chunk_dir> / <chunk_filename>
      So `_base_path` must be the *chunk folder* (parent/chunk_dir), not just parent.
    - DataFrame / 0.0 sentinel libraries are NOT supported.
    - V1 chunk formats are NOT supported.
    - lazy_build=True is NOT supported.
    """
    if lazy_build:
        raise NotImplementedError(
            "Hard switch to v2 chunked storage: lazy_build=True is not supported. "
            "Prebuild libraries with build_libraries_grid_with_plateau(...)."
        )
    if include_policies:
        # Kept for backwards compatibility; not used in v2 table retrieval.
        pass

    # -----------------------------
    # Local helpers (no shadowing)
    # -----------------------------
    def _canon_norm_edges(e_norm: tuple) -> tuple:
        canon_key, _, _ = canonicalize_edges_with_roles(
            edges=list(e_norm),
            num_attacker_nodes=num_attacker_nodes,
            num_defender_nodes=num_defender_nodes,
        )
        # Use GLOBAL normalizer here:
        return _edges_to_norm(canon_key)

    def _finalize_desc(desc: dict, *, pkl_path: Path) -> dict:
        """
        Ensure the returned chunk descriptor is path-resolvable by library_io.

        IMPORTANT:
          desc["chunks"] contains filenames only (e.g. "chunk_000000.pkl")
          so base path must be the directory that contains those chunk files:
              pkl_path.parent / desc["chunk_dir"]
        """
        if not isinstance(desc, dict):
            raise TypeError(f"Expected descriptor dict, got {type(desc)} from {pkl_path}")

        # Always store the .pkl path (useful for debugging and alternative resolution)
        desc.setdefault("library_pkl_path", str(pkl_path.resolve()))

        # Compute chunk folder correctly
        chunk_dir = desc.get("chunk_dir")
        if isinstance(chunk_dir, str) and chunk_dir.strip():
            chunk_folder = (pkl_path.parent / chunk_dir).resolve()
        else:
            # Fallback: if chunk_dir missing (shouldn't happen in v2), use pkl parent
            chunk_folder = pkl_path.parent.resolve()

        # ✅ This is what library_io expects as the folder containing chunk filenames
        desc["_base_path"] = str(chunk_folder)

        return desc

    def _extract_v2_desc(lib: dict, *, pkl_path: Path) -> dict:
        if lib.get("prob_format") != "chunked_rows":
            raise ValueError(
                f"Hard switch: {pkl_path} prob_format must be 'chunked_rows', got {lib.get('prob_format')!r}"
            )

        desc = lib.get("prob_table_chunked")
        if not isinstance(desc, dict):
            raise KeyError(f"Hard switch: {pkl_path} missing 'prob_table_chunked' descriptor dict")

        fmt = desc.get("format")
        if fmt != "chunked_prob_table_v2_rows_v1":
            raise ValueError(
                f"Hard switch: expected prob_table_chunked.format == 'chunked_prob_table_v2_rows_v1', "
                f"got {fmt!r} in {pkl_path}"
            )

        for k in ("chunks", "row_to_chunk", "chunk_dir"):
            if k not in desc:
                raise KeyError(f"Hard switch: descriptor missing required key {k!r} in {pkl_path}")

        return _finalize_desc(desc, pkl_path=pkl_path)

    # -----------------------------
    # Canonicalize request
    # -----------------------------
    req_norm = _edges_to_norm(edges)  # GLOBAL function
    if req_norm is None:
        raise ValueError(f"Invalid edges input: {edges}")

    target_edges = _canon_norm_edges(req_norm)
    if target_edges is None:
        raise ValueError(f"Failed to canonicalize edges input: {edges}")

    # -----------------------------
    # Expected hashed path
    # -----------------------------
    path = graph_path(
        edges=target_edges,
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
        max_attacker_troops=max_attacker_troops,
        max_defender_troops=max_defender_troops,
        base_dir=base_dir,
    )

    # -----------------------------
    # Fast path: exact expected file exists
    # -----------------------------
    if path.exists():
        lib = load_library(path)
        return _extract_v2_desc(lib, pkl_path=path)

    # -----------------------------
    # Fallback: scan folder (hash drift / old canonical storage)
    # -----------------------------
    folder = path.parent
    if folder.exists():
        for pkl in folder.glob("graph_*.pkl"):
            try:
                lib = load_library(pkl)
            except Exception:
                continue

            lib_edges_norm = _edges_to_norm(lib.get("edges"))  # GLOBAL function
            if lib_edges_norm is None:
                continue

            try:
                lib_edges_canon = _canon_norm_edges(lib_edges_norm)
            except Exception:
                continue

            if lib_edges_canon == target_edges:
                return _extract_v2_desc(lib, pkl_path=pkl)

    raise FileNotFoundError(
        f"V2 per-graph library file not found for "
        f"{num_attacker_nodes}A, {num_defender_nodes}D, "
        f"maxA={max_attacker_troops}, maxD={max_defender_troops}, "
        f"edges(canonical)={target_edges}: {path}"
    )




def build_libraries_grid_with_plateau(
    combat_df,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_A_exact: int,
    max_D_exact: int,
    max_A_extended: int,
    max_D_extended: int,
    base_dir: Path = BASE_LIB_DIR,
    overwrite: bool = False,
    include_policies: bool = False,
    high_min_att_troops: int = 3,
    edges_list: Optional[List[Iterable[tuple[int, int]]]] = None,
    canonical_edges_list: Optional[List[Iterable[tuple[int, int]]]] = None,
    *,
    chunk_rows: Optional[int] = None,
    auto_chunk_threshold_rows: int = 250_000,
    # ------------------------------------------------------------
    # PATCH: parallelism controls
    # ------------------------------------------------------------
    n_jobs: int = 1,
    graphs_per_job: Optional[int] = None,
    utility_mode: str = "legacy",
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_split_depth: Optional[int] = 1,
) -> Dict[str, Any]:
    """
    Plateau-aware per-graph library builder.

    HARD SWITCH (v2-only)
    ---------------------
    - Always writes v2 chunked row-store:
        prob_format = "chunked_rows"
        prob_table_chunked["format"] = "chunked_prob_table_v2_rows_v1"
    - Underlying builder is called with output_format="chunked_rows" and MUST
      return v2 descriptors (format == "chunked_prob_table_v2_rows_v1").
    - No DataFrames, no v1 chunk formats, no mixed-mode.

    Returns
    -------
    stats : dict
        Summary stats for this combo.
    """
    total_nodes = num_attacker_nodes + num_defender_nodes

    combo_label = (
        f"nA={num_attacker_nodes}, nD={num_defender_nodes}, "
        f"exact A<={max_A_exact}, D<={max_D_exact}; "
        f"extended A<={max_A_extended}, D<={max_D_extended}"
    )
    print(f"[PLATEAU BUILD] {combo_label}")

    t0 = time.time()

    # ----------------------------
    # Graph count info only
    # ----------------------------
    if canonical_edges_list is not None:
        num_graphs_seen = len(canonical_edges_list)
        graph_source = "canonical_edges_list"
    elif edges_list is not None:
        num_graphs_seen = len(edges_list)
        graph_source = "edges_list"
    else:
        num_graphs_seen = len(generate_connected_graphs_n_nodes(total_nodes))
        graph_source = "enumerated_labelled"

    extended_rows_total = (max_A_extended ** num_attacker_nodes) * (max_D_extended ** num_defender_nodes)

    # Choose chunk_rows default if not provided
    if chunk_rows is None:
        chunk_rows = 2_000 if extended_rows_total >= auto_chunk_threshold_rows else 5_000

    # ----------------------------
    # Determine folder where per-graph PKLs will live
    # ----------------------------
    dummy_edges = [(0, 1)]  # only used to locate the folder; not used for saving
    combo_folder = graph_path(
        edges=dummy_edges,
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
        max_attacker_troops=max_A_extended,
        max_defender_troops=max_D_extended,
        base_dir=base_dir,
    ).parent

    chunk_rel_prefix = "_chunked_rows"
    chunk_root_dir = combo_folder / chunk_rel_prefix

    print(f"[PLATEAU BUILD] output_format=chunked_rows_v2, chunk_rows={chunk_rows}")
    print(f"[PLATEAU BUILD] chunk_root_dir={chunk_root_dir}")

    # ----------------------------
    # Build tables (exact + extension)
    #
    # PATCH: Parallelize by splitting the canonical graph list across processes.
    # We do this safely by giving each worker a unique chunk directory.
    # ----------------------------

    # Step 1: Build a canonical_edges_list_local so we can split it.
    # - If canonical_edges_list is provided: normalize it.
    # - Else if edges_list is provided: canonicalize+dedupe it.
    # - Else: enumerate labelled connected graphs, canonicalize+dedupe.
    canonical_edges_list_local: List[tuple] = []

    if canonical_edges_list is not None:
        for e in canonical_edges_list:
            canon_norm = _edges_to_norm(e)
            if canon_norm is not None:
                canonical_edges_list_local.append(canon_norm)
    else:
        canon_set: Dict[tuple, tuple] = {}
        if edges_list is not None:
            labelled_iter = edges_list
        else:
            labelled_iter = generate_connected_graphs_n_nodes(total_nodes)

        for e in labelled_iter:
            try:
                canon_key, _, _ = canonicalize_edges_with_roles(
                    edges=sorted(tuple(sorted(edge)) for edge in list(e)),
                    num_attacker_nodes=num_attacker_nodes,
                    num_defender_nodes=num_defender_nodes,
                )
            except Exception:
                continue

            canon_norm = _edges_to_norm(canon_key)
            if canon_norm is not None:
                canon_set[canon_norm] = canon_norm

        canonical_edges_list_local = list(canon_set.values())

    num_graphs_canonical_target = len(canonical_edges_list_local)

    # Decide splitting granularity
    if graphs_per_job is None:
        # default: aim for ~2 chunks per worker, but avoid tiny chunks
        if int(n_jobs) > 1:
            graphs_per_job = max(25, (num_graphs_canonical_target // int(n_jobs)) // 2)
        else:
            graphs_per_job = num_graphs_canonical_target

    def _chunk(lst: List[tuple], k: int):
        for i in range(0, len(lst), k):
            yield lst[i : i + k]

    # Sequential fallback
    if int(n_jobs) <= 1 or num_graphs_canonical_target <= int(graphs_per_job):
        tables_extended, plateau_policies = build_absorption_tables_two_player_with_plateau(
            combat_df=combat_df,
            num_attacker_nodes=num_attacker_nodes,
            num_defender_nodes=num_defender_nodes,
            max_attacker_troops_exact=max_A_exact,
            max_defender_troops_exact=max_D_exact,
            max_attacker_troops_extended=max_A_extended,
            max_defender_troops_extended=max_D_extended,
            high_min_att_troops=high_min_att_troops,
            edges_list=None,
            canonical_edges_list=canonical_edges_list_local,
            output_format="chunked_rows",  # hard switch
            chunk_rows=chunk_rows,
            chunk_root_dir=chunk_root_dir,
            chunk_rel_prefix=chunk_rel_prefix,
            dtype=np.float32,
            auto_chunk_threshold_rows=auto_chunk_threshold_rows,
            utility_mode=utility_mode,
            value_tolerances=value_tolerances,
            include_no_gain_in_value=include_no_gain_in_value,
            multi_policy_options=multi_policy_options,
            policy_option_mode=policy_option_mode,
            max_policy_options_per_row=max_policy_options_per_row,
            max_options_per_state=max_options_per_state,
            max_split_depth=max_split_depth,
        )
        
    else:
        if Parallel is None or delayed is None:
            raise ImportError(
                "joblib is required for parallel library builds. "
                "Install joblib or run with n_jobs=1."
            )
        chunks = list(_chunk(canonical_edges_list_local, int(graphs_per_job)))
        print(
            f"[PLATEAU BUILD][parallel] n_jobs={n_jobs}, graphs={num_graphs_canonical_target}, "
            f"graphs_per_job={graphs_per_job}, chunks={len(chunks)}"
        )

        outs = Parallel(n_jobs=int(n_jobs), prefer="processes", verbose=5)(
            delayed(_build_plateau_tables_worker)(
                combat_df=combat_df,
                num_attacker_nodes=num_attacker_nodes,
                num_defender_nodes=num_defender_nodes,
                max_A_exact=max_A_exact,
                max_D_exact=max_D_exact,
                max_A_extended=max_A_extended,
                max_D_extended=max_D_extended,
                high_min_att_troops=high_min_att_troops,
                canonical_edges_list_chunk=chunk_edges,
                chunk_rows=int(chunk_rows),
                combo_folder=combo_folder,
                worker_id=i,
                auto_chunk_threshold_rows=auto_chunk_threshold_rows,
                utility_mode=utility_mode,
                value_tolerances=value_tolerances,
                include_no_gain_in_value=include_no_gain_in_value,
                multi_policy_options=multi_policy_options,
                policy_option_mode=policy_option_mode,
                max_policy_options_per_row=max_policy_options_per_row,
                max_options_per_state=max_options_per_state,
                max_split_depth=max_split_depth,
            )
            for i, chunk_edges in enumerate(chunks)
        )

        tables_extended: Dict[Any, Any] = {}
        plateau_policies: Dict[Any, Any] = {}
        for tables_i, policies_i in outs:
            tables_extended.update(tables_i)
            plateau_policies.update(policies_i)

    num_graphs_canonical = len(tables_extended)
    num_plateau_graphs = len(plateau_policies)

    print(
        f"[PLATEAU BUILD] graph_source={graph_source}, "
        f"graphs_seen={num_graphs_seen}, canonical graphs built={num_graphs_canonical}, "
        f"with plateau={num_plateau_graphs}"
    )

    expected_exact_rows = (max_A_exact ** num_attacker_nodes) * (max_D_exact ** num_defender_nodes)

    num_written = 0
    num_existing = 0
    num_failures = 0
    num_plateau_written = 0

    extended_rows_counts: list[int] = []
    plateau_extended_rows_counts: list[int] = []

    for edges_key, table_obj in tables_extended.items():
        plateau_used = edges_key in plateau_policies

        # ----------------------------
        # Validate v2 descriptor
        # ----------------------------
        if not isinstance(table_obj, dict) or table_obj.get("format") != "chunked_prob_table_v2_rows_v1":
            num_failures += 1
            print(
                f"[ERROR] Hard switch: builder returned non-v2 table for edges={edges_key}: "
                f"type={type(table_obj)}, format={(table_obj.get('format') if isinstance(table_obj, dict) else None)!r}"
            )
            continue

        # ----------------------------
        # Row count for stats
        # ----------------------------
        exact_df = table_obj.get("exact_df")
        n_exact = int(exact_df.shape[0]) if isinstance(exact_df, pd.DataFrame) else 0
        n_ext = int(len(table_obj.get("row_to_chunk", {})))
        n_rows = n_exact + n_ext

        extended_rows = max(n_rows - expected_exact_rows, 0)
        extended_rows_counts.append(extended_rows)
        if plateau_used:
            plateau_extended_rows_counts.append(extended_rows)

        # ----------------------------
        # Per-graph path
        # ----------------------------
        path = graph_path(
            edges=edges_key,
            num_attacker_nodes=num_attacker_nodes,
            num_defender_nodes=num_defender_nodes,
            max_attacker_troops=max_A_extended,
            max_defender_troops=max_D_extended,
            base_dir=base_dir,
        )

        if path.exists() and not overwrite:
            num_existing += 1
            print(f"[SKIP] {edges_key} -> {path.name} (exists, overwrite=False)")
            continue

        try:
            per_graph_lib: Dict[str, Any] = {
                "params": {
                    "num_attacker_nodes": num_attacker_nodes,
                    "num_defender_nodes": num_defender_nodes,
                    "max_attacker_troops_exact": max_A_exact,
                    "max_defender_troops_exact": max_D_exact,
                    "max_attacker_troops": max_A_extended,
                    "max_defender_troops": max_D_extended,
                    "include_policies": include_policies,
                    "high_min_att_troops": high_min_att_troops,
                    "plateau_used": plateau_used,
                    "graph_source": graph_source,
                    "output_format": "chunked_rows_v2",
                    "chunk_rows": chunk_rows,
                    "utility_mode": utility_mode,
                    "value_tolerances": tuple(value_tolerances) if value_tolerances is not None else None,
                    "include_no_gain_in_value": bool(include_no_gain_in_value),
                    "multi_policy_options": bool(multi_policy_options),
                    "policy_option_mode": str(policy_option_mode),
                    "max_policy_options_per_row": None if max_policy_options_per_row is None else int(max_policy_options_per_row),
                    "max_options_per_state": None if max_options_per_state is None else int(max_options_per_state),
                    "max_split_depth": None if max_split_depth is None else int(max_split_depth),
                },
                "edges": edges_key,
                "num_nodes": total_nodes,
                "prob_format": "chunked_rows",
                "prob_table_chunked": table_obj,
            }

            if plateau_used:
                per_graph_lib["plateau_policy"] = plateau_policies[edges_key]

            print(f"[WRITE] edges={edges_key} -> {path}")
            save_library(per_graph_lib, path)

            num_written += 1
            if plateau_used:
                num_plateau_written += 1

        except Exception as e:
            num_failures += 1
            print(f"[ERROR] Failed writing library for edges={edges_key} -> {path}")
            print(f"        {type(e).__name__}: {e}")

    elapsed = time.time() - t0

    def _stats(lst: list[int]) -> Dict[str, float | int]:
        if not lst:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {"min": min(lst), "max": max(lst), "mean": float(sum(lst)) / len(lst)}

    stats = {
        "nA": num_attacker_nodes,
        "nD": num_defender_nodes,
        "maxA_exact": max_A_exact,
        "maxD_exact": max_D_exact,
        "maxA_extended": max_A_extended,
        "maxD_extended": max_D_extended,
        "graph_source": graph_source,
        "num_graphs_seen": num_graphs_seen,
        "num_graphs_canonical": num_graphs_canonical,
        "num_plateau_graphs": num_plateau_graphs,
        "num_libraries_written": num_written,
        "num_plateau_libraries_written": num_plateau_written,
        "num_libraries_existing": num_existing,
        "num_failures": num_failures,
        "expected_exact_rows": expected_exact_rows,
        "extended_rows_stats": _stats(extended_rows_counts),
        "plateau_extended_rows_stats": _stats(plateau_extended_rows_counts),
        "elapsed_seconds": elapsed,
        "output_format": "chunked_rows_v2",
        "chunk_rows": chunk_rows,
        "utility_mode": utility_mode,
        "value_tolerances": tuple(value_tolerances) if value_tolerances is not None else None,
        "include_no_gain_in_value": bool(include_no_gain_in_value),
        "multi_policy_options": bool(multi_policy_options),
        "policy_option_mode": str(policy_option_mode),
        "max_policy_options_per_row": None if max_policy_options_per_row is None else int(max_policy_options_per_row),
        "max_options_per_state": None if max_options_per_state is None else int(max_options_per_state),
        "max_split_depth": None if max_split_depth is None else int(max_split_depth),
        "extended_rows_total": int(extended_rows_total),
        "chunk_root_dir": str(chunk_root_dir),
    }

    print(
        f"[PLATEAU SUMMARY] {combo_label} | "
        f"format=chunked_rows_v2, graphs_seen={num_graphs_seen}, graphs_canon={num_graphs_canonical}, "
        f"plateau_graphs={num_plateau_graphs}, libs_written={num_written}, "
        f"libs_existing={num_existing}, failures={num_failures}, time={elapsed:.1f}s"
    )

    return stats


def build_star_graph(
    nr_edges: int,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_A_exact: int,
    max_D_exact: int,
    max_A_extended: int,
    max_D_extended: int,
    *,
    overwrite: bool = False,
    include_policies: bool = False,
    high_min_att_troops: int = 2,
    chunk_rows: int | None = None,
    auto_chunk_threshold_rows: int = 250_000,
    # PATCH: parallelism controls (mostly irrelevant for single-graph stars)
    n_jobs: int = 1,
    graphs_per_job: Optional[int] = None,
    utility_mode: str = "legacy",
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_split_depth: Optional[int] = 1,
):
    """
    HARD SWITCH (v2-only): build exactly ONE star topology for this (nA,nD) combo.

    Star topology: edges = [(0,1), (0,2), ..., (0,nr_edges)]

    Notes
    -----
    - Always uses v2 chunked row-store via build_libraries_grid_with_plateau().
    - No output_format param here; v2 is enforced downstream.
    """
    # Star topology on labelled nodes (0 is center)
    star_edges = [(0, i) for i in range(1, nr_edges + 1)]

    # Choose a default chunk size if not provided
    if chunk_rows is None:
        extended_rows_total = (max_A_extended ** num_attacker_nodes) * (max_D_extended ** num_defender_nodes)
        chunk_rows = 2_000 if extended_rows_total >= auto_chunk_threshold_rows else 5_000

    return build_libraries_grid_with_plateau(
        combat_df=combat_df,
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
        max_A_exact=max_A_exact,
        max_D_exact=max_D_exact,
        max_A_extended=max_A_extended,
        max_D_extended=max_D_extended,
        base_dir=BASE_LIB_DIR,
        overwrite=overwrite,
        include_policies=include_policies,
        high_min_att_troops=high_min_att_troops,
        edges_list=[star_edges],
        canonical_edges_list=None,
        chunk_rows=chunk_rows,
        auto_chunk_threshold_rows=auto_chunk_threshold_rows,
        n_jobs=n_jobs,
        graphs_per_job=graphs_per_job,
        utility_mode=utility_mode,
        value_tolerances=value_tolerances,
        include_no_gain_in_value=include_no_gain_in_value,
        multi_policy_options=multi_policy_options,
        policy_option_mode=policy_option_mode,
        max_policy_options_per_row=max_policy_options_per_row,
        max_options_per_state=max_options_per_state,
        max_split_depth=max_split_depth,
    )



# The lowest node_max_troop number in the libraries (except 1-1 combat) (change name)
lowest_lib_node_max = 7
lim = lowest_lib_node_max

combos = [
        # nA, nD, maxA_exact, maxD_exact, maxA_extended, maxD_extended
        [1, 1, 10, 10, 10, 10],
        [2, 1, 3, 3, lim, lim],
        [3, 1, 3, 3, lim, lim],
        [4, 1, 3, 3, lim, lim],
        [1, 2, 3, 3, lim, lim],
        [2, 2, 3, 3, lim, lim],
        [3, 2, 3, 3, lim, lim],
        [1, 3, 3, 3, lim, lim],
        [1, 4, 3, 3, lim, lim],
        [2, 3, 3, 3, lim, lim],
    ]

star_combos = [
        # nA, nD, maxA_exact, maxD_exact, maxA_extended, maxD_extended
        [1, 5, 2, 2, lim, lim],
        [1, 6, 2, 2, lim, lim],
        [5, 1, 2, 2, lim, lim],
        [6, 1, 2, 2, lim, lim],

    ]


def build_all_libraries_with_plateau(
    combat_df,
    overwrite: bool = False,
    include_policies: bool = False,
    combos=combos,
    *,
    chunk_rows: int | None = None,
    auto_chunk_threshold_rows: int = 250_000,
    # PATCH: parallelism controls
    n_jobs: int = 1,
    graphs_per_job: Optional[int] = None,
    utility_mode: str = "legacy",
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_split_depth: Optional[int] = 1,
):
    """
    HARD SWITCH (v2-only): Precompute per-graph libraries using plateau extension.

    What this does now
    ------------------
    - Builds ALL connected graphs for each combo in `combos` (enumerated_labelled).
    - Builds ONLY star graphs for `star_combos` via build_star_graph(...).
    - ALWAYS uses v2 chunked row-store (no DataFrames, no v1 chunks, no mixed-mode).

    Parameters
    ----------
    chunk_rows:
        If None, each combo chooses a reasonable default based on extended state-space size.
        If provided, that value is used for every combo (and star combo).
    """
    all_stats: list[dict] = []

    overall_start = time.time()
    print("[BUILD ALL PLATEAU] Starting full library generation (v2-only)...")

    # ---- all topologies for small graph combos ----
    for nA, nD, maxA_exact, maxD_exact, maxA_ext, maxD_ext in combos:
        stats = build_libraries_grid_with_plateau(
            combat_df=combat_df,
            num_attacker_nodes=nA,
            num_defender_nodes=nD,
            max_A_exact=maxA_exact,
            max_D_exact=maxD_exact,
            max_A_extended=maxA_ext,
            max_D_extended=maxD_ext,
            base_dir=BASE_LIB_DIR,
            overwrite=overwrite,
            include_policies=include_policies,
            edges_list=None,
            canonical_edges_list=None,
            chunk_rows=chunk_rows,
            auto_chunk_threshold_rows=auto_chunk_threshold_rows,
            n_jobs=n_jobs,
            graphs_per_job=graphs_per_job,
            utility_mode=utility_mode,
            value_tolerances=value_tolerances,
            include_no_gain_in_value=include_no_gain_in_value,
            multi_policy_options=multi_policy_options,
            policy_option_mode=policy_option_mode,
            max_policy_options_per_row=max_policy_options_per_row,
            max_options_per_state=max_options_per_state,
            max_split_depth=max_split_depth,
        )
        all_stats.append(stats)

    # ---- star-only combos ----
    for nA, nD, maxA_exact, maxD_exact, maxA_ext, maxD_ext in star_combos:
        nr_edges = (nA + nD) - 1

        star_stats = build_star_graph(
            nr_edges=nr_edges,
            num_attacker_nodes=nA,
            num_defender_nodes=nD,
            max_A_exact=maxA_exact,
            max_D_exact=maxD_exact,
            max_A_extended=maxA_ext,
            max_D_extended=maxD_ext,
            overwrite=overwrite,
            include_policies=include_policies,
            high_min_att_troops=2,
            chunk_rows=chunk_rows,
            auto_chunk_threshold_rows=auto_chunk_threshold_rows,
        )
        all_stats.append(star_stats)

    overall_elapsed = time.time() - overall_start

    # ---- Global summary ----
    total_graphs_seen = sum((s.get("num_graphs_seen") or 0) for s in all_stats)
    total_graphs_canon = sum((s.get("num_graphs_canonical") or s.get("num_graphs") or 0) for s in all_stats)

    total_plateau_graphs = sum(s.get("num_plateau_graphs", 0) for s in all_stats)
    total_libs_written = sum(s.get("num_libraries_written", 0) for s in all_stats)
    total_plateau_libs_written = sum(s.get("num_plateau_libraries_written", 0) for s in all_stats)

    total_libs_existing = sum(s.get("num_libraries_existing", 0) for s in all_stats)
    total_failures = sum(s.get("num_failures", 0) for s in all_stats)

    print("\n================== BUILD ALL PLATEAU SUMMARY (v2-only) ==================")
    print(f"Total combos run                 : {len(all_stats)}")
    print(f"Total graphs seen (raw)          : {total_graphs_seen}")
    print(f"Total graphs processed (canon)   : {total_graphs_canon}")
    print(f"Total plateau graphs             : {total_plateau_graphs}")
    print(f"Libraries written (files)        : {total_libs_written}")
    print(f"Libraries skipped (already exist): {total_libs_existing}")
    print(f"Plateau-based libraries written  : {total_plateau_libs_written}")
    print(f"Failures                         : {total_failures}")
    print(f"Overall wall-clock time          : {overall_elapsed:.1f} s")

    print("\nPer-combo recap:")
    for s in all_stats:
        combo_str = (
            f"nA={s.get('nA')}, nD={s.get('nD')}, "
            f"exact A<={s.get('maxA_exact')},D<={s.get('maxD_exact')} "
            f"-> ext A<={s.get('maxA_extended')},D<={s.get('maxD_extended')}"
        )

        graphs_seen = s.get("num_graphs_seen", None)
        graphs_canon = s.get("num_graphs_canonical", s.get("num_graphs", None))
        existing = s.get("num_libraries_existing", 0)
        failures = s.get("num_failures", 0)
        fmt = s.get("output_format", "chunked_rows_v2")

        ext_stats = s.get("extended_rows_stats", {"min": 0, "mean": 0.0, "max": 0})
        print(
            f"  - {combo_str}: "
            f"format={fmt}, seen={graphs_seen}, canon={graphs_canon}, "
            f"plateau_graphs={s.get('num_plateau_graphs')}, "
            f"written={s.get('num_libraries_written')}, existing={existing}, "
            f"plateau_written={s.get('num_plateau_libraries_written')}, failures={failures}, "
            f"ext_rows(min/mean/max)={ext_stats['min']}/{ext_stats['mean']:.1f}/{ext_stats['max']}, "
            f"time={s.get('elapsed_seconds', 0.0):.1f}s"
        )

    print("==========================================================================\n")


# =====================================================================
# Exact finite compact library builder (V2 chunked row-store)
# =====================================================================

exact_finite_combos = [
    # nA, nD, maxA, maxD
    # [1, 1, 10, 10],
    # [2, 1, lim, lim],
    # [3, 1, lim, lim],
    # [4, 1, lim, lim],
    # [1, 2, lim, lim],
    # [2, 2, lim, lim],
    [3, 2, 8, 8],
    # [1, 3, lim, lim],
    # [2, 3, lim, lim],
    # [1, 4, lim, lim],
]


def _edges_key_hash(edges_key: Any) -> str:
    """Stable short hash for canonical edge-key-specific chunk folders."""
    return hashlib.sha1(repr(edges_key).encode("utf-8")).hexdigest()[:10]


def _write_exact_finite_v2_chunk(
    folder: Path,
    chunk_index: int,
    edges_key: Any,
    rows_obj: Dict[str, Dict[str, Any]],
) -> str:
    """Write one V2 exact-finite row chunk and return its filename only."""
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"chunk_{int(chunk_index):06d}.pkl"
    with p.open("wb") as f:
        pickle.dump(
            {"edges_key": edges_key, "format": "v2_rowchunk_v1", "rows": rows_obj},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return p.name


def _canonical_edges_list_for_library_build(
    *,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    edges_list: Optional[List[Iterable[tuple[int, int]]]] = None,
    canonical_edges_list: Optional[List[Iterable[tuple[int, int]]]] = None,
) -> Tuple[List[tuple], str, int]:
    """Return normalized canonical representatives plus source diagnostics.

    This mirrors the graph-selection semantics of the plateau builder:
      1. explicit canonical_edges_list is trusted/normalized;
      2. edges_list is canonicalized and deduplicated;
      3. otherwise all connected labelled graphs are enumerated, canonicalized,
         and deduplicated under A/D-preserving permutations.
    """
    total_nodes = int(num_attacker_nodes) + int(num_defender_nodes)

    if canonical_edges_list is not None:
        out: List[tuple] = []
        for e in canonical_edges_list:
            norm = _edges_to_norm(e)
            if norm is not None:
                out.append(norm)
        return out, "canonical_edges_list", len(canonical_edges_list)

    if edges_list is not None:
        labelled_iter = edges_list
        graph_source = "edges_list"
    else:
        labelled_iter = generate_connected_graphs_n_nodes(total_nodes)
        graph_source = "enumerated_labelled"

    canon_set: Dict[tuple, tuple] = {}
    seen = 0
    for e in labelled_iter:
        seen += 1
        try:
            canon_key, _, _ = canonicalize_edges_with_roles(
                edges=sorted(tuple(sorted(edge)) for edge in list(e)),
                num_attacker_nodes=num_attacker_nodes,
                num_defender_nodes=num_defender_nodes,
            )
        except Exception:
            continue
        norm = _edges_to_norm(canon_key)
        if norm is not None:
            canon_set[norm] = norm

    return list(canon_set.values()), graph_source, seen


def _stats_min_max_mean(values: List[float | int]) -> Dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": float(sum(values)) / float(len(values)),
    }


def _build_exact_finite_graph_library(
    *,
    edges_key: tuple,
    combat_df,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops: int,
    max_defender_troops: int,
    base_dir: Path,
    overwrite: bool,
    chunk_rows: int,
    chunk_rel_prefix: str,
    utility_mode: str,
    value_tolerances: Optional[Tuple[float, ...]],
    include_no_gain_in_value: bool,
    multi_policy_options: bool,
    policy_option_mode: str,
    max_policy_options_per_row: Optional[int],
    max_options_per_state: Optional[int],
    max_leaf_split_depth: Optional[int],
    max_split_depth: Optional[int],
    cache_distributions: bool,
    sort_actions: bool,
) -> Dict[str, Any]:
    """Build exactly one per-graph exact-finite V2 library file."""
    if CompactExactTopologySolver is None:
        raise ImportError(
            "CompactExactTopologySolver could not be imported. Make sure "
            "exact_finite_solver.py is on PYTHONPATH next to create_library.py."
        )

    t0 = time.time()
    total_nodes = int(num_attacker_nodes) + int(num_defender_nodes)
    edges_key = tuple(tuple(sorted((int(u), int(v)))) for u, v in edges_key)

    path = graph_path(
        edges=edges_key,
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
        max_attacker_troops=max_attacker_troops,
        max_defender_troops=max_defender_troops,
        base_dir=base_dir,
    )

    if path.exists() and not overwrite:
        return {
            "status": "existing",
            "edges": edges_key,
            "path": str(path),
            "elapsed_seconds": 0.0,
        }

    combo_folder = path.parent
    edge_hash = _edges_key_hash(edges_key)
    chunk_dir_rel = f"{chunk_rel_prefix}/{edge_hash}"
    chunk_folder = combo_folder / chunk_dir_rel

    solver = CompactExactTopologySolver(
        edges=edges_key,
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
        combat_df=combat_df,
        utility_mode=utility_mode,
        include_no_gain_in_value=include_no_gain_in_value,
        value_tolerances=value_tolerances,
        max_total_troops=(
            int(num_attacker_nodes) * int(max_attacker_troops)
            + int(num_defender_nodes) * int(max_defender_troops)
        ),
        cache_distributions=cache_distributions,
        sort_actions=sort_actions,
    )

    row_to_chunk: Dict[str, int] = {}
    chunk_files: List[str] = []
    chunk_index = 0
    rows_obj: Dict[str, Dict[str, Any]] = {}
    rows_written = 0

    def flush_chunk() -> None:
        nonlocal chunk_index, rows_obj, chunk_files
        if not rows_obj:
            return
        fname = _write_exact_finite_v2_chunk(chunk_folder, chunk_index, edges_key, rows_obj)
        chunk_files.append(fname)
        chunk_index += 1
        rows_obj = {}

    mode = str(policy_option_mode or "root").strip().lower()
    if mode in {"state-set", "bottom_up", "bottom-up"}:
        mode = "state_set"
    if max_leaf_split_depth is None:
        max_leaf_split_depth = max_split_depth
    if max_leaf_split_depth is None:
        max_leaf_split_depth = 1
    if multi_policy_options and mode not in {"root", "state_set"}:
        raise ValueError(
            "Exact finite builder policy_option_mode must be 'root' or 'state_set', "
            f"got {policy_option_mode!r}"
        )

    att_range = range(1, int(max_attacker_troops) + 1)
    def_range = range(1, int(max_defender_troops) + 1)

    for att_troops in itertools.product(att_range, repeat=int(num_attacker_nodes)):
        for def_troops in itertools.product(def_range, repeat=int(num_defender_nodes)):
            row_label = solver.row_label(att_troops, def_troops)
            row_to_chunk[row_label] = chunk_index

            if multi_policy_options:
                if mode == "root":
                    options = solver.root_policy_options(
                        att_troops,
                        def_troops,
                        max_policy_options=max_policy_options_per_row,
                    )
                else:
                    options = solver.state_set_policy_options(
                        att_troops,
                        def_troops,
                        max_policy_options_per_row=max_policy_options_per_row,
                        max_options_per_state=max_options_per_state,
                        max_leaf_split_depth=int(max_leaf_split_depth),
                    )
                payload = solver.policy_options_to_v2_payload(
                    options,
                    policy_option_mode=mode,
                    max_policy_options_per_row=max_policy_options_per_row,
                )
            else:
                result = solver.evaluate_start(att_troops, def_troops)
                payload = solver.dist_to_v2_payload(result.absorbing_dist)

            rows_obj[row_label] = payload
            rows_written += 1
            if len(rows_obj) >= int(chunk_rows):
                flush_chunk()

    flush_chunk()

    graph_stats = solver.stats.as_dict()
    graph_stats.update(
        rows=int(rows_written),
        value_cache_size=len(solver._value_cache),
        dist_cache_size=len(solver._dist_cache),
        combat_cache_size=len(solver._combat_cache),
        state_options_cache_size=len(getattr(solver, "_state_options_cache", {})),
        elapsed_seconds=float(time.time() - t0),
        rows_per_second=(float(rows_written) / max(float(time.time() - t0), 1e-12)),
        chunk_count=len(chunk_files),
    )

    desc: Dict[str, Any] = {
        "format": "chunked_prob_table_v2_rows_v1",
        "exact_df": None,
        "chunks": chunk_files,
        "row_to_chunk": row_to_chunk,
        "chunk_dir": chunk_dir_rel,
        "builder": "compact_exact_finite_v1",
        "multi_policy_options": bool(multi_policy_options),
        "policy_option_mode": str(mode),
        "max_policy_options_per_row": (
            None if max_policy_options_per_row is None else int(max_policy_options_per_row)
        ),
        "max_options_per_state": None if max_options_per_state is None else int(max_options_per_state),
        "max_leaf_split_depth": None if max_leaf_split_depth is None else int(max_leaf_split_depth),
        "max_split_depth": None if max_leaf_split_depth is None else int(max_leaf_split_depth),
        "include_no_gain_in_value": bool(include_no_gain_in_value),
        "exact_finite_stats": graph_stats,
    }

    per_graph_lib: Dict[str, Any] = {
        "params": {
            "num_attacker_nodes": int(num_attacker_nodes),
            "num_defender_nodes": int(num_defender_nodes),
            "max_attacker_troops": int(max_attacker_troops),
            "max_defender_troops": int(max_defender_troops),
            "max_attacker_troops_exact": int(max_attacker_troops),
            "max_defender_troops_exact": int(max_defender_troops),
            "include_policies": bool(multi_policy_options),
            "utility_mode": str(utility_mode),
            "value_tolerances": tuple(value_tolerances) if value_tolerances is not None else None,
            "include_no_gain_in_value": bool(include_no_gain_in_value),
            "multi_policy_options": bool(multi_policy_options),
            "policy_option_mode": str(mode),
            "max_policy_options_per_row": (
                None if max_policy_options_per_row is None else int(max_policy_options_per_row)
            ),
            "max_options_per_state": None if max_options_per_state is None else int(max_options_per_state),
            "max_leaf_split_depth": None if max_leaf_split_depth is None else int(max_leaf_split_depth),
            "max_split_depth": None if max_leaf_split_depth is None else int(max_leaf_split_depth),
            "output_format": "chunked_rows_v2",
            "builder": "compact_exact_finite_v1",
            "chunk_rows": int(chunk_rows),
        },
        "edges": edges_key,
        "num_nodes": total_nodes,
        "prob_format": "chunked_rows",
        "prob_table_chunked": desc,
        "exact_finite_stats": graph_stats,
    }

    save_library(per_graph_lib, path)

    return {
        "status": "written",
        "edges": edges_key,
        "path": str(path),
        **graph_stats,
    }


def _build_exact_finite_worker(
    *,
    worker_id: int,
    edges_chunk: List[tuple],
    combat_df,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops: int,
    max_defender_troops: int,
    base_dir: Path,
    overwrite: bool,
    chunk_rows: int,
    chunk_rel_prefix: str,
    utility_mode: str,
    value_tolerances: Optional[Tuple[float, ...]],
    include_no_gain_in_value: bool,
    multi_policy_options: bool,
    policy_option_mode: str,
    max_policy_options_per_row: Optional[int],
    max_options_per_state: Optional[int],
    max_leaf_split_depth: Optional[int],
    max_split_depth: Optional[int],
    cache_distributions: bool,
    sort_actions: bool,
) -> List[Dict[str, Any]]:
    """Parallel worker: build a subset of canonical exact-finite graph libs."""
    out = []
    # Separate worker prefix avoids accidental chunk-folder collisions if a hash
    # ever collides or if a previous interrupted build left stale chunks.
    worker_prefix = f"{chunk_rel_prefix}_w{int(worker_id):02d}"
    for edges_key in edges_chunk:
        out.append(
            _build_exact_finite_graph_library(
                edges_key=edges_key,
                combat_df=combat_df,
                num_attacker_nodes=num_attacker_nodes,
                num_defender_nodes=num_defender_nodes,
                max_attacker_troops=max_attacker_troops,
                max_defender_troops=max_defender_troops,
                base_dir=base_dir,
                overwrite=overwrite,
                chunk_rows=chunk_rows,
                chunk_rel_prefix=worker_prefix,
                utility_mode=utility_mode,
                value_tolerances=value_tolerances,
                include_no_gain_in_value=include_no_gain_in_value,
                multi_policy_options=multi_policy_options,
                policy_option_mode=policy_option_mode,
                max_policy_options_per_row=max_policy_options_per_row,
                max_options_per_state=max_options_per_state,
                max_leaf_split_depth=max_leaf_split_depth,
                max_split_depth=max_split_depth,
                cache_distributions=cache_distributions,
                sort_actions=sort_actions,
            )
        )
    return out


def build_libraries_grid_exact_finite(
    *,
    num_attacker_nodes: int,
    num_defender_nodes: int,
    max_attacker_troops: int,
    max_defender_troops: int,
    combat_df=None,
    base_dir: Path = BASE_LIB_DIR,
    overwrite: bool = False,
    edges_list: Optional[List[Iterable[tuple[int, int]]]] = None,
    canonical_edges_list: Optional[List[Iterable[tuple[int, int]]]] = None,
    chunk_rows: Optional[int] = None,
    auto_chunk_threshold_rows: int = 250_000,
    n_jobs: int = 1,
    graphs_per_job: Optional[int] = None,
    utility_mode: str = "local",
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_leaf_split_depth: Optional[int] = 1,
    max_split_depth: Optional[int] = None,
    cache_distributions: bool = True,
    sort_actions: bool = True,
) -> Dict[str, Any]:
    """Build exact finite V2 per-graph libraries for one (nA,nD,cap) combo.

    This is the exact finite counterpart to ``build_libraries_grid_with_plateau``.
    It writes the same hard-switch V2 row-store contract:

        per_graph_lib["prob_format"] == "chunked_rows"
        per_graph_lib["prob_table_chunked"]["format"] == "chunked_prob_table_v2_rows_v1"

    Differences from the plateau builder
    ------------------------------------
    - Every row up to ``max_attacker_troops`` / ``max_defender_troops`` is solved
      exactly by ``CompactExactTopologySolver``.
    - One solver/cache is shared across all rows of the same topology.
    - No ``exact_df`` is stored; all rows live in V2 chunks so normal and
      policy-option rows use one retrieval path.
    - ``state_set`` policy options are intentionally not implemented yet in the
      compact solver; use ``multi_policy_options=True, policy_option_mode='root'``
      for root options, or ``multi_policy_options=False`` for single optimal rows.
    """
    if CompactExactTopologySolver is None or combat_df_for_caps is None:
        raise ImportError(
            "exact_finite_solver.py could not be imported. Place it next to "
            "create_library.py before using build_libraries_grid_exact_finite()."
        )

    nA = int(num_attacker_nodes)
    nD = int(num_defender_nodes)
    maxA = int(max_attacker_troops)
    maxD = int(max_defender_troops)
    total_nodes = nA + nD
    total_rows = (maxA ** nA) * (maxD ** nD)

    if chunk_rows is None:
        chunk_rows = 2_000 if total_rows >= int(auto_chunk_threshold_rows) else 5_000
    chunk_rows = int(chunk_rows)

    if combat_df is None:
        combat_df = combat_df_for_caps(
            num_attacker_nodes=nA,
            num_defender_nodes=nD,
            max_attacker_troops=maxA,
            max_defender_troops=maxD,
        )
    if max_leaf_split_depth is None:
        max_leaf_split_depth = max_split_depth
    if max_leaf_split_depth is None:
        max_leaf_split_depth = 1

    canonical_edges_list_local, graph_source, num_graphs_seen = _canonical_edges_list_for_library_build(
        num_attacker_nodes=nA,
        num_defender_nodes=nD,
        edges_list=edges_list,
        canonical_edges_list=canonical_edges_list,
    )
    num_graphs_canonical_target = len(canonical_edges_list_local)

    label = f"nA={nA}, nD={nD}, exact finite A<={maxA}, D<={maxD}"
    print(f"[EXACT FINITE BUILD] {label}")
    print(
        f"[EXACT FINITE BUILD] graph_source={graph_source}, "
        f"graphs_seen={num_graphs_seen}, canonical_target={num_graphs_canonical_target}, "
        f"rows_per_graph={total_rows}, chunk_rows={chunk_rows}"
    )

    t0 = time.time()

    if graphs_per_job is None:
        if int(n_jobs) > 1:
            graphs_per_job = max(1, (num_graphs_canonical_target // int(n_jobs)) // 2)
        else:
            graphs_per_job = max(1, num_graphs_canonical_target)
    graphs_per_job = max(1, int(graphs_per_job))

    def _chunk(lst: List[tuple], k: int):
        for i in range(0, len(lst), k):
            yield lst[i : i + k]

    results: List[Dict[str, Any]] = []
    chunk_rel_prefix = "_exact_finite_rows"

    if int(n_jobs) <= 1 or num_graphs_canonical_target <= graphs_per_job:
        for edges_key in canonical_edges_list_local:
            try:
                r = _build_exact_finite_graph_library(
                    edges_key=edges_key,
                    combat_df=combat_df,
                    num_attacker_nodes=nA,
                    num_defender_nodes=nD,
                    max_attacker_troops=maxA,
                    max_defender_troops=maxD,
                    base_dir=base_dir,
                    overwrite=overwrite,
                    chunk_rows=chunk_rows,
                    chunk_rel_prefix=chunk_rel_prefix,
                    utility_mode=utility_mode,
                    value_tolerances=value_tolerances,
                    include_no_gain_in_value=include_no_gain_in_value,
                    multi_policy_options=multi_policy_options,
                    policy_option_mode=policy_option_mode,
                    max_policy_options_per_row=max_policy_options_per_row,
                    max_options_per_state=max_options_per_state,
                    max_leaf_split_depth=max_leaf_split_depth,
                    max_split_depth=max_split_depth,
                    cache_distributions=cache_distributions,
                    sort_actions=sort_actions,
                )
            except Exception as e:
                r = {
                    "status": "failure",
                    "edges": edges_key,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
                print(f"[ERROR][EXACT FINITE] edges={edges_key}: {type(e).__name__}: {e}")
            results.append(r)
    else:
        if Parallel is None or delayed is None:
            raise ImportError(
                "joblib is required for parallel exact-finite library builds. "
                "Install joblib or run with n_jobs=1."
            )
        chunks = list(_chunk(canonical_edges_list_local, graphs_per_job))
        print(
            f"[EXACT FINITE BUILD][parallel] n_jobs={n_jobs}, "
            f"graphs_per_job={graphs_per_job}, chunks={len(chunks)}"
        )
        outs = Parallel(n_jobs=int(n_jobs), prefer="processes", verbose=5)(
            delayed(_build_exact_finite_worker)(
                worker_id=i,
                edges_chunk=chunk,
                combat_df=combat_df,
                num_attacker_nodes=nA,
                num_defender_nodes=nD,
                max_attacker_troops=maxA,
                max_defender_troops=maxD,
                base_dir=base_dir,
                overwrite=overwrite,
                chunk_rows=chunk_rows,
                chunk_rel_prefix=chunk_rel_prefix,
                utility_mode=utility_mode,
                value_tolerances=value_tolerances,
                include_no_gain_in_value=include_no_gain_in_value,
                multi_policy_options=multi_policy_options,
                policy_option_mode=policy_option_mode,
                max_policy_options_per_row=max_policy_options_per_row,
                max_options_per_state=max_options_per_state,
                max_leaf_split_depth=max_leaf_split_depth,
                max_split_depth=max_split_depth,
                cache_distributions=cache_distributions,
                sort_actions=sort_actions,
            )
            for i, chunk in enumerate(chunks)
        )
        for o in outs:
            results.extend(o)

    elapsed = time.time() - t0

    written = [r for r in results if r.get("status") == "written"]
    existing = [r for r in results if r.get("status") == "existing"]
    failures = [r for r in results if r.get("status") == "failure"]

    seconds = [float(r.get("elapsed_seconds", 0.0)) for r in written]
    rows_per_second = [float(r.get("rows_per_second", 0.0)) for r in written]
    value_cache_sizes = [int(r.get("value_cache_size", 0)) for r in written]
    dist_cache_sizes = [int(r.get("dist_cache_size", 0)) for r in written]
    movement_choices = [int(r.get("movement_choice_evals", 0)) for r in written]
    combat_branches = [int(r.get("combat_outcome_branches", 0)) for r in written]

    stats = {
        "nA": nA,
        "nD": nD,
        "maxA_exact": maxA,
        "maxD_exact": maxD,
        "maxA_extended": maxA,
        "maxD_extended": maxD,
        "graph_source": graph_source,
        "num_graphs_seen": int(num_graphs_seen),
        "num_graphs_canonical": int(num_graphs_canonical_target),
        "num_libraries_written": len(written),
        "num_libraries_existing": len(existing),
        "num_failures": len(failures),
        "rows_per_graph": int(total_rows),
        "total_rows_written": int(len(written) * total_rows),
        "elapsed_seconds": float(elapsed),
        "output_format": "chunked_rows_v2",
        "builder": "compact_exact_finite_v1",
        "chunk_rows": int(chunk_rows),
        "utility_mode": str(utility_mode),
        "value_tolerances": tuple(value_tolerances) if value_tolerances is not None else None,
        "include_no_gain_in_value": bool(include_no_gain_in_value),
        "multi_policy_options": bool(multi_policy_options),
        "policy_option_mode": str(policy_option_mode),
        "max_policy_options_per_row": (
            None if max_policy_options_per_row is None else int(max_policy_options_per_row)
        ),
        "max_options_per_state": None if max_options_per_state is None else int(max_options_per_state),
        "max_leaf_split_depth": None if max_leaf_split_depth is None else int(max_leaf_split_depth),
        "max_split_depth": None if max_leaf_split_depth is None else int(max_leaf_split_depth),
        "seconds_stats": _stats_min_max_mean(seconds),
        "rows_per_second_stats": _stats_min_max_mean(rows_per_second),
        "value_cache_size_stats": _stats_min_max_mean(value_cache_sizes),
        "dist_cache_size_stats": _stats_min_max_mean(dist_cache_sizes),
        "movement_choice_stats": _stats_min_max_mean(movement_choices),
        "combat_branch_stats": _stats_min_max_mean(combat_branches),
        "failures": failures,
    }

    print(
        f"[EXACT FINITE SUMMARY] {label} | "
        f"canon={num_graphs_canonical_target}, written={len(written)}, "
        f"existing={len(existing)}, failures={len(failures)}, "
        f"time={elapsed:.1f}s"
    )

    return stats


def build_all_libraries_exact_finite(
    *,
    overwrite: bool = False,
    combos=exact_finite_combos,
    base_dir: Path = BASE_LIB_DIR,
    chunk_rows: Optional[int] = None,
    auto_chunk_threshold_rows: int = 250_000,
    n_jobs: int = 1,
    graphs_per_job: Optional[int] = None,
    utility_mode: str = "local",
    value_tolerances: Optional[Tuple[float, ...]] = None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_leaf_split_depth: Optional[int] = 1,
    max_split_depth: Optional[int] = None,
    cache_distributions: bool = True,
) -> List[Dict[str, Any]]:
    """Build all configured exact-finite libraries.

    This intentionally does not replace ``build_all_libraries_with_plateau``.
    Use it when the target cap is finite and exact, typically cap 7.
    """
    all_stats: List[Dict[str, Any]] = []
    overall_start = time.time()
    print("[BUILD ALL EXACT FINITE] Starting exact finite library generation...")

    for nA, nD, maxA, maxD in combos:
        cdf = combat_df_for_caps(
            num_attacker_nodes=int(nA),
            num_defender_nodes=int(nD),
            max_attacker_troops=int(maxA),
            max_defender_troops=int(maxD),
        )
        stats = build_libraries_grid_exact_finite(
            num_attacker_nodes=int(nA),
            num_defender_nodes=int(nD),
            max_attacker_troops=int(maxA),
            max_defender_troops=int(maxD),
            combat_df=cdf,
            base_dir=base_dir,
            overwrite=overwrite,
            chunk_rows=chunk_rows,
            auto_chunk_threshold_rows=auto_chunk_threshold_rows,
            n_jobs=n_jobs,
            graphs_per_job=graphs_per_job,
            utility_mode=utility_mode,
            value_tolerances=value_tolerances,
            include_no_gain_in_value=include_no_gain_in_value,
            multi_policy_options=multi_policy_options,
            policy_option_mode=policy_option_mode,
            max_policy_options_per_row=max_policy_options_per_row,
            max_options_per_state=max_options_per_state,
            max_leaf_split_depth=max_leaf_split_depth,
            max_split_depth=max_split_depth,
            cache_distributions=cache_distributions,
        )
        all_stats.append(stats)

    overall_elapsed = time.time() - overall_start
    total_written = sum(int(s.get("num_libraries_written", 0)) for s in all_stats)
    total_existing = sum(int(s.get("num_libraries_existing", 0)) for s in all_stats)
    total_failures = sum(int(s.get("num_failures", 0)) for s in all_stats)

    print("\n================ BUILD ALL EXACT FINITE SUMMARY ================")
    print(f"Total combos run                 : {len(all_stats)}")
    print(f"Libraries written                : {total_written}")
    print(f"Libraries skipped                : {total_existing}")
    print(f"Failures                         : {total_failures}")
    print(f"Overall wall-clock time          : {overall_elapsed:.1f} s")
    print("=================================================================\n")

    return all_stats

# ---------------------------------------------------------------------
# Exact finite star-only library helpers
# ---------------------------------------------------------------------

import inspect
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EXACT_FINITE_STAR_COMBOS: List[Tuple[int, int, int, int]] = [
    # nA, nD, maxA, maxD
    (1, 5, 7, 7),
    (1, 6, 7, 7),
    (5, 1, 7, 7),
    (6, 1, 7, 7),
]


def make_star_edges_exact_finite(
    num_attacker_nodes: int,
    num_defender_nodes: int,
) -> Tuple[Tuple[int, int], ...]:
    """
    Return the single intended star topology for exact finite star libraries.

    Node convention matches the rest of create_library.py:
      - attacker nodes are 0, ..., nA-1
      - defender nodes are nA, ..., nA+nD-1

    Supported star cases:
      - 1A kD: attacker node 0 is the center, connected to every defender leaf
      - kA 1D: defender node nA is the center, connected to every attacker leaf

    This intentionally does NOT enumerate all connected topologies on 6 or 7 nodes.
    """
    nA = int(num_attacker_nodes)
    nD = int(num_defender_nodes)

    if nA <= 0 or nD <= 0:
        raise ValueError(f"nA and nD must be positive, got nA={nA}, nD={nD}.")

    if nA == 1 and nD >= 1:
        return tuple((0, d) for d in range(1, 1 + nD))

    if nD == 1 and nA >= 1:
        defender_center = nA
        return tuple((a, defender_center) for a in range(nA))

    raise ValueError(
        "Star-only exact finite helper expects exactly one side to have one node, "
        f"got nA={nA}, nD={nD}. Use build_libraries_grid_exact_finite for ordinary combos."
    )


def _call_with_supported_kwargs(func, kwargs: Dict[str, Any]):
    """
    Call func while passing only kwargs supported by its current signature.

    This makes the star helper tolerant of small local signature differences while still
    failing clearly if required parameters are missing.
    """
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**supported)


def build_star_libraries_exact_finite(
    *,
    star_combos: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    base_dir=None,
    overwrite: bool = False,
    utility_mode: str = "local",
    value_tolerances=None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_leaf_split_depth: Optional[int] = 1,
    max_split_depth: Optional[int] = None,
    chunk_rows: int = 5000,
    auto_chunk_threshold_rows: int = 250_000,
    n_jobs: int = 1,
    graphs_per_job: Optional[int] = None,
    combat_df=None,
) -> List[Dict[str, Any]]:
    """
    Build exact finite libraries for the star-only 1A-kD and kA-1D cases.

    This writes the same per-graph V2 chunked row-store as build_libraries_grid_exact_finite,
    but passes exactly one topology per combo via canonical_edges_list / edges_list.

    Example
    -------
    stats = build_star_libraries_exact_finite(
        overwrite=False,
        utility_mode="local",
        multi_policy_options=False,
        chunk_rows=5000,
        n_jobs=4,
    )
    """
    if star_combos is None:
        star_combos = DEFAULT_EXACT_FINITE_STAR_COMBOS

    # These names must exist in create_library.py when this helper is pasted there.
    try:
        builder = build_libraries_grid_exact_finite
    except NameError as exc:
        raise RuntimeError(
            "build_star_libraries_exact_finite must be placed after "
            "build_libraries_grid_exact_finite in create_library.py."
        ) from exc

    sig = inspect.signature(builder)
    if "canonical_edges_list" in sig.parameters:
        edge_kw_name = "canonical_edges_list"
    elif "edges_list" in sig.parameters:
        edge_kw_name = "edges_list"
    else:
        raise RuntimeError(
            "build_libraries_grid_exact_finite does not expose canonical_edges_list or edges_list. "
            "Add one of those parameters first, otherwise the star helper cannot restrict the build "
            "to one star topology."
        )

    all_stats: List[Dict[str, Any]] = []
    start_all = time.time()

    print("[EXACT FINITE STAR BUILD] Starting star-only exact finite library generation...")

    for nA, nD, maxA, maxD in star_combos:
        edges = make_star_edges_exact_finite(nA, nD)
        rows_per_graph = int(maxA) ** int(nA) * int(maxD) ** int(nD)

        print(
            f"[EXACT FINITE STAR BUILD] nA={nA}, nD={nD}, "
            f"A<={maxA}, D<={maxD}, rows={rows_per_graph}, edges={edges}"
        )

        kwargs = dict(
            num_attacker_nodes=int(nA),
            num_defender_nodes=int(nD),
            max_attacker_troops=int(maxA),
            max_defender_troops=int(maxD),
            base_dir=base_dir,
            overwrite=overwrite,
            utility_mode=utility_mode,
            value_tolerances=value_tolerances,
            include_no_gain_in_value=include_no_gain_in_value,
            multi_policy_options=multi_policy_options,
            policy_option_mode=policy_option_mode,
            max_policy_options_per_row=max_policy_options_per_row,
            max_options_per_state=max_options_per_state,
            max_leaf_split_depth=max_leaf_split_depth,
            max_split_depth=max_split_depth,
            chunk_rows=chunk_rows,
            auto_chunk_threshold_rows=auto_chunk_threshold_rows,
            n_jobs=n_jobs,
            graphs_per_job=graphs_per_job,
            combat_df=combat_df,
            # exactly one topology
            **{edge_kw_name: [edges]},
        )

        # Avoid passing base_dir=None / combat_df=None if the local signature accepts them but
        # expects a real object. The underlying builder should fall back to its defaults when
        # these are omitted.
        if base_dir is None:
            kwargs.pop("base_dir", None)
        if combat_df is None:
            kwargs.pop("combat_df", None)

        stats = _call_with_supported_kwargs(builder, kwargs)
        if isinstance(stats, dict):
            stats.setdefault("graph_source", "star_only_exact_finite")
            stats.setdefault("star_edges", edges)
        all_stats.append(stats)

    elapsed = time.time() - start_all
    total_written = sum((s or {}).get("num_libraries_written", 0) for s in all_stats if isinstance(s, dict))
    total_existing = sum((s or {}).get("num_libraries_existing", 0) for s in all_stats if isinstance(s, dict))
    total_failures = sum((s or {}).get("num_failures", 0) for s in all_stats if isinstance(s, dict))

    print("\n================ EXACT FINITE STAR SUMMARY ================")
    print(f"Star combos run                 : {len(all_stats)}")
    print(f"Libraries written               : {total_written}")
    print(f"Libraries skipped existing       : {total_existing}")
    print(f"Failures                         : {total_failures}")
    print(f"Wall-clock time                  : {elapsed:.1f} s")
    print("===========================================================\n")

    return all_stats


def build_all_libraries_exact_finite_with_stars(
    *,
    combos=None,
    star_combos: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    build_regular: bool = True,
    build_stars: bool = True,
    base_dir=None,
    overwrite: bool = False,
    utility_mode: str = "local",
    value_tolerances=None,
    include_no_gain_in_value: bool = False,
    multi_policy_options: bool = False,
    policy_option_mode: str = "root",
    max_policy_options_per_row: Optional[int] = 1,
    max_options_per_state: Optional[int] = 2,
    max_leaf_split_depth: Optional[int] = 1,
    max_split_depth: Optional[int] = None,
    chunk_rows: int = 5000,
    auto_chunk_threshold_rows: int = 250_000,
    n_jobs: int = 1,
    graphs_per_job: Optional[int] = None,
    combat_df=None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: build ordinary exact finite combos and then star-only combos.

    Use this instead of putting 1A5D/1A6D/5A1D/6A1D into the ordinary combo list.
    """
    result: Dict[str, Any] = {
        "regular_stats": None,
        "star_stats": None,
    }

    if build_regular:
        try:
            regular_builder = build_all_libraries_exact_finite
        except NameError as exc:
            raise RuntimeError(
                "build_all_libraries_exact_finite_with_stars must be placed after "
                "build_all_libraries_exact_finite in create_library.py."
            ) from exc

        kwargs = dict(
            combos=combos,
            base_dir=base_dir,
            overwrite=overwrite,
            utility_mode=utility_mode,
            value_tolerances=value_tolerances,
            include_no_gain_in_value=include_no_gain_in_value,
            multi_policy_options=multi_policy_options,
            policy_option_mode=policy_option_mode,
            max_policy_options_per_row=max_policy_options_per_row,
            max_options_per_state=max_options_per_state,
            max_leaf_split_depth=max_leaf_split_depth,
            max_split_depth=max_split_depth,
            chunk_rows=chunk_rows,
            auto_chunk_threshold_rows=auto_chunk_threshold_rows,
            n_jobs=n_jobs,
            graphs_per_job=graphs_per_job,
            combat_df=combat_df,
        )
        if combos is None:
            kwargs.pop("combos", None)
        if base_dir is None:
            kwargs.pop("base_dir", None)
        if combat_df is None:
            kwargs.pop("combat_df", None)

        result["regular_stats"] = _call_with_supported_kwargs(regular_builder, kwargs)

    if build_stars:
        result["star_stats"] = build_star_libraries_exact_finite(
            star_combos=star_combos,
            base_dir=base_dir,
            overwrite=overwrite,
            utility_mode=utility_mode,
            value_tolerances=value_tolerances,
            include_no_gain_in_value=include_no_gain_in_value,
            multi_policy_options=multi_policy_options,
            policy_option_mode=policy_option_mode,
            max_policy_options_per_row=max_policy_options_per_row,
            max_options_per_state=max_options_per_state,
            max_leaf_split_depth=max_leaf_split_depth,
            max_split_depth=max_split_depth,
            chunk_rows=chunk_rows,
            auto_chunk_threshold_rows=auto_chunk_threshold_rows,
            n_jobs=n_jobs,
            graphs_per_job=graphs_per_job,
            combat_df=combat_df,
        )

    return result

if __name__ == "__main__":
    from pathlib import Path
    import json
    import time

    # ------------------------------------------------------------------
    # FULL STATE-SET EXACT FINITE BUILD
    # ------------------------------------------------------------------
    # Output is intentionally separate from the validated single-policy
    # small_graph_libraries folder.
    #
    # State-set policy controls:
    #   max_policy_options_per_row = 2
    #   max_options_per_state      = 2
    #   max_leaf_split_depth       = 1
    # ------------------------------------------------------------------

    STATE_SET_BASE_DIR = Path("small_graph_libraries")

    FULL_EXACT_FINITE_COMBOS = [
        # nA, nD, maxA, maxD

        # Keep 1A1D at cap10, as in the historical exact finite setup.
        (1, 1, 10, 10),

        # Regular cap-7 libraries
        (2, 1, 7, 7),
        (3, 1, 7, 7),
        (4, 1, 7, 7),

        (1, 2, 7, 7),
        (2, 2, 7, 7),
        (3, 2, 7, 7),

        (1, 3, 7, 7),
        (2, 3, 7, 7),

        (1, 4, 7, 7),
    ]

    FULL_EXACT_FINITE_STAR_COMBOS = [
        # nA, nD, maxA, maxD
        (1, 5, 7, 7),
        (1, 6, 7, 7),
        (5, 1, 7, 7),
        (6, 1, 7, 7),
    ]

    t0 = time.time()

    result = build_all_libraries_exact_finite_with_stars(
        combos=FULL_EXACT_FINITE_COMBOS,
        star_combos=FULL_EXACT_FINITE_STAR_COMBOS,

        build_regular=True,
        build_stars=True,

        base_dir=STATE_SET_BASE_DIR,
        overwrite=False,

        utility_mode="local",
        value_tolerances=None,
        include_no_gain_in_value=False,

        multi_policy_options=True,
        policy_option_mode="state_set",
        max_policy_options_per_row=2,

        # State-set internal caps
        max_options_per_state=2,
        max_leaf_split_depth=1,
        max_split_depth=1,

        chunk_rows=5000,
        n_jobs=4,
        graphs_per_job=None,
        auto_chunk_threshold_rows=250_000,
        combat_df=None,
    )

    elapsed = time.time() - t0

    print("\n=== FULL STATE-SET EXACT FINITE BUILD COMPLETE ===")
    print(f"Output folder: {STATE_SET_BASE_DIR.resolve()}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print(result)

    # Optional: save a compact JSON summary next to the library folder.
    summary_path = STATE_SET_BASE_DIR / "_full_state_set_build_summary.json"
    STATE_SET_BASE_DIR.mkdir(parents=True, exist_ok=True)

    def _json_default(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, tuple):
            return list(x)
        return str(x)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_dir": str(STATE_SET_BASE_DIR),
                "elapsed_seconds": elapsed,
                "regular_combos": FULL_EXACT_FINITE_COMBOS,
                "star_combos": FULL_EXACT_FINITE_STAR_COMBOS,
                "policy_settings": {
                    "multi_policy_options": True,
                    "policy_option_mode": "state_set",
                    "max_policy_options_per_row": 2,
                    "max_options_per_state": 2,
                    "max_leaf_split_depth": 1,
                    "max_split_depth": 1,
                    "utility_mode": "local",
                    "include_no_gain_in_value": False,
                    "chunk_rows": 5000,
                    "n_jobs": 4,
                },
                "result": result,
            },
            f,
            indent=2,
            default=_json_default,
        )

    print(f"Saved summary: {summary_path.resolve()}")



