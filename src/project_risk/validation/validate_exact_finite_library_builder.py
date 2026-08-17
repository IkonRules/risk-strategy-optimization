"""Smoke-test the exact finite V2 library builder.

Run from the project folder that contains:
    create_library.py
    exact_finite_solver.py
    library_io.py
    small_graph_outcome_probabilities.py
    markov_matrix_probabilities.py

This test builds one tiny exact finite per-graph library in a temporary folder,
loads it through the existing V2 library reader, and compares one non-trivial
row against the legacy exact recursive solver.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import numpy as np

from project_risk.mathematical.libraries.create_library import build_libraries_grid_exact_finite, graph_path, load_library
from project_risk.mathematical.small_graph_model.exact_finite_solver import combat_df_for_caps, compare_rowdicts
from project_risk.mathematical.libraries.library_io import load_graph_library, get_prob_row_payload_from_library
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
    encode_state_label,
    explore_absorbing_states_for_graph_local_objective,
    initial_state_generic,
)


def payload_to_rowdict(payload: dict) -> Dict[str, float]:
    """Decode a normal V2 payload back to {state_label: probability}."""
    p = np.asarray(payload["p"], dtype=float)
    owners = np.asarray(payload["owners"])
    troops = np.asarray(payload["troops"])

    row: Dict[str, float] = {}
    for i in range(len(p)):
        parts = []
        for owner_code, troop in zip(owners[i], troops[i]):
            prefix = "A" if int(owner_code) == 1 else "D"
            parts.append(f"{prefix}{int(troop)}")
        lbl = "(" + ",".join(parts) + ")"
        row[lbl] = row.get(lbl, 0.0) + float(p[i])

    total = sum(row.values())
    if total > 0:
        row = {k: v / total for k, v in row.items() if v > 0.0}
    return row


def main() -> None:
    base_dir = Path("_tmp_exact_finite_builder_test")
    if base_dir.exists():
        shutil.rmtree(base_dir)

    nA, nD = 2, 2
    maxA = maxD = 3
    edges = [(0, 2), (0, 3), (1, 2)]

    combat_df = combat_df_for_caps(
        num_attacker_nodes=nA,
        num_defender_nodes=nD,
        max_attacker_troops=maxA,
        max_defender_troops=maxD,
    )

    stats = build_libraries_grid_exact_finite(
        num_attacker_nodes=nA,
        num_defender_nodes=nD,
        max_attacker_troops=maxA,
        max_defender_troops=maxD,
        combat_df=combat_df,
        base_dir=base_dir,
        overwrite=True,
        edges_list=[edges],
        chunk_rows=25,
        utility_mode="local",
        multi_policy_options=False,
    )

    assert stats["num_libraries_written"] == 1, stats
    assert stats["num_failures"] == 0, stats

    pkl_path = graph_path(
        edges=edges,
        num_attacker_nodes=nA,
        num_defender_nodes=nD,
        max_attacker_troops=maxA,
        max_defender_troops=maxD,
        base_dir=base_dir,
    )
    lib = load_graph_library(pkl_path)

    row_label = "(A3,A2,D2,D3)"
    payload = get_prob_row_payload_from_library(
        lib,
        row_label,
        allow_extrapolation=False,
        num_attacker_nodes=nA,
        library_pkl_path=str(pkl_path),
    )
    assert payload is not None, f"Missing payload for {row_label}"
    assert abs(float(np.asarray(payload["p"]).sum()) - 1.0) < 1e-6
    assert "cdf" in payload

    built_row = payload_to_rowdict(payload)

    reference_start = initial_state_generic((3, 2), (2, 3))
    ref_dist, _val, _policy = explore_absorbing_states_for_graph_local_objective(
        edges=set(tuple(sorted(e)) for e in edges),
        combat_df=combat_df,
        start_state=reference_start,
        num_attacker_nodes=nA,
    )
    ref_row = {encode_state_label(st): float(p) for st, p in ref_dist.items() if float(p) > 0.0}
    s = sum(ref_row.values())
    ref_row = {k: v / s for k, v in ref_row.items()}

    cmp = compare_rowdicts(built_row, ref_row, tol=1e-8)
    assert cmp["ok"], cmp

    print("=== exact finite library builder smoke test ===")
    print("stats:", stats)
    print("library:", pkl_path)
    print("row_label:", row_label)
    print("comparison:", cmp)
    print("payload outcomes:", len(payload["p"]))

    shutil.rmtree(base_dir)


if __name__ == "__main__":
    main()
