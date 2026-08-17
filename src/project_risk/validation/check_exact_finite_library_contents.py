"""
check_exact_finite_library_contents.py

Sanity-check exact finite small-graph libraries written by
build_libraries_grid_exact_finite(...).

What it checks
--------------
1. Per-graph pickle structure.
2. V2 chunked descriptor contract.
3. Chunk files exist and row coverage matches maxA^nA * maxD^nD.
4. A sample of row payloads can be loaded through library_io.
5. Payload probabilities, CDF, owners/troops shapes, and metrics are consistent.
6. Optional: compare a few sampled rows against the old exact recursive solver.

Example
-------
python check_exact_finite_library_contents.py --nA 2 --nD 3 --maxA 7 --maxD 7 --max-graphs 0 --sample-rows 8 --reference-rows 2

Use --reference-rows 0 to skip old-solver comparisons.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from project_risk.mathematical.libraries.library_io import load_graph_library, get_prob_row_payload_from_library
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
    initial_state_generic,
    encode_state_label,
    parse_row_label,
    explore_absorbing_states_for_graph,
    explore_absorbing_states_for_graph_local_objective,
)
from project_risk.mathematical.small_graph_model.exact_finite_solver import combat_df_for_caps


# ---------------------------------------------------------------------
# Row-label helpers
# ---------------------------------------------------------------------

def make_row_label(att_troops: Tuple[int, ...], def_troops: Tuple[int, ...]) -> str:
    return "(" + ",".join([f"A{t}" for t in att_troops] + [f"D{t}" for t in def_troops]) + ")"


def row_label_to_troops(row_label: str, nA: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    owners, troops = parse_row_label(row_label)
    if len(owners) < nA:
        raise ValueError(f"row label too short for nA={nA}: {row_label}")
    return tuple(int(t) for t in troops[:nA]), tuple(int(t) for t in troops[nA:])


def expected_row_labels(
    nA: int,
    nD: int,
    maxA: int,
    maxD: int,
) -> Iterable[str]:
    for att in itertools.product(range(1, maxA + 1), repeat=nA):
        for deff in itertools.product(range(1, maxD + 1), repeat=nD):
            yield make_row_label(tuple(att), tuple(deff))


def sample_row_labels(
    nA: int,
    nD: int,
    maxA: int,
    maxD: int,
    n_random: int,
    rng: random.Random,
) -> List[str]:
    """Deterministic corner cases plus random rows."""
    rows: List[str] = []

    corners = [
        (tuple([1] * nA), tuple([1] * nD)),
        (tuple([maxA] * nA), tuple([maxD] * nD)),
        (tuple([maxA] + [1] * max(0, nA - 1)), tuple([maxD] * nD)),
        (tuple([1] * max(0, nA - 1) + [maxA]) if nA else tuple(), tuple([maxD] * nD)),
        (tuple([maxA] * nA), tuple([1] * nD)),
    ]

    for att, deff in corners:
        if len(att) == nA and len(deff) == nD:
            lbl = make_row_label(att, deff)
            if lbl not in rows:
                rows.append(lbl)

    for _ in range(max(0, int(n_random))):
        att = tuple(rng.randint(1, maxA) for _ in range(nA))
        deff = tuple(rng.randint(1, maxD) for _ in range(nD))
        lbl = make_row_label(att, deff)
        if lbl not in rows:
            rows.append(lbl)

    return rows


# ---------------------------------------------------------------------
# Payload decoding and checks
# ---------------------------------------------------------------------

def _owner_code_to_label(code: Any, troop: int) -> str:
    """Robust owner decoding for both common encodings.

    Current library_io V2 path often uses: 1=A, 2=D, 0=empty.
    Some older helpers use: 1=A, 0=D.

    Since exact finite states should normally have positive troops on all local
    nodes, code 0 with troop>0 is treated as D for backward compatibility.
    """
    c = int(code)
    if int(troop) <= 0:
        return "D"
    if c == 1:
        return "A"
    if c in (0, 2):
        return "D"
    raise ValueError(f"unknown owner code {code!r} for troop={troop}")


def payload_to_rowdict(payload: Dict[str, Any]) -> Dict[str, float]:
    """Convert a single V2 payload with p/owners/troops arrays to {label: p}."""
    p = np.asarray(payload["p"], dtype=np.float64)
    owners = np.asarray(payload["owners"])
    troops = np.asarray(payload["troops"])

    if owners.ndim != 2 or troops.ndim != 2 or owners.shape != troops.shape:
        raise ValueError(f"bad owners/troops shapes: owners={owners.shape}, troops={troops.shape}")
    if owners.shape[0] != p.shape[0]:
        raise ValueError(f"p length {p.shape[0]} does not match outcomes {owners.shape[0]}")

    out: Dict[str, float] = {}
    for i in range(p.shape[0]):
        parts = []
        for j in range(owners.shape[1]):
            t = int(troops[i, j])
            o = _owner_code_to_label(owners[i, j], t)
            parts.append(f"{o}{t}")
        lbl = "(" + ",".join(parts) + ")"
        out[lbl] = out.get(lbl, 0.0) + float(p[i])
    return {k: v for k, v in out.items() if v > 0.0}


def compare_rowdicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, Any]:
    keys = set(a) | set(b)
    max_abs = 0.0
    l1 = 0.0
    worst_key = None
    for k in sorted(keys):
        d = abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0)))
        l1 += d
        if d > max_abs:
            max_abs = d
            worst_key = k
    return {
        "max_abs_diff": max_abs,
        "l1_diff": l1,
        "worst_key": worst_key,
        "num_keys_a": len(a),
        "num_keys_b": len(b),
        "num_union_keys": len(keys),
    }


def validate_single_payload(
    payload: Dict[str, Any],
    *,
    nA: int,
    n_nodes: int,
    prob_tol: float,
    metric_tol: float,
) -> List[str]:
    """Return list of problems for one non-policy-options V2 payload."""
    problems: List[str] = []

    required = ("p", "owners", "troops")
    for k in required:
        if k not in payload:
            problems.append(f"missing payload key {k!r}")
    if problems:
        return problems

    p = np.asarray(payload["p"], dtype=np.float64)
    owners = np.asarray(payload["owners"])
    troops = np.asarray(payload["troops"])

    if p.ndim != 1:
        problems.append(f"p must be 1D, got shape={p.shape}")
    if owners.ndim != 2 or troops.ndim != 2:
        problems.append(f"owners/troops must be 2D, got owners={owners.shape}, troops={troops.shape}")
    elif owners.shape != troops.shape:
        problems.append(f"owners/troops shapes differ: owners={owners.shape}, troops={troops.shape}")
    elif owners.shape[1] != n_nodes:
        problems.append(f"outcome node count {owners.shape[1]} != expected {n_nodes}")
    elif p.shape[0] != owners.shape[0]:
        problems.append(f"p length {p.shape[0]} != outcome count {owners.shape[0]}")

    if problems:
        return problems

    if p.size == 0:
        problems.append("empty probability vector")
        return problems

    if np.any(~np.isfinite(p)):
        problems.append("p contains non-finite values")
    if np.any(p < -prob_tol):
        problems.append(f"p contains negative values below tolerance {prob_tol}")

    s = float(p.sum())
    if abs(s - 1.0) > prob_tol:
        problems.append(f"probabilities sum to {s:.12g}, not 1")

    if "cdf" in payload and payload["cdf"] is not None:
        cdf = np.asarray(payload["cdf"], dtype=np.float64)
        if cdf.shape != p.shape:
            problems.append(f"cdf shape {cdf.shape} != p shape {p.shape}")
        elif abs(float(cdf[-1]) - 1.0) > max(prob_tol, 1e-5):
            problems.append(f"cdf[-1]={float(cdf[-1]):.12g}, not 1")

    # Decode owners robustly to validate metrics.
    decoded_is_A = np.zeros_like(owners, dtype=bool)
    decoded_is_D = np.zeros_like(owners, dtype=bool)
    for i in range(owners.shape[0]):
        for j in range(owners.shape[1]):
            t = int(troops[i, j])
            o = _owner_code_to_label(owners[i, j], t)
            if t > 0 and o == "A":
                decoded_is_A[i, j] = True
            if t > 0 and o == "D":
                decoded_is_D[i, j] = True

    expected_is_conq = (~decoded_is_D.any(axis=1)).astype(np.uint8)
    defender_block = np.zeros_like(decoded_is_A, dtype=bool)
    defender_block[:, int(nA):] = True
    expected_new = np.sum(decoded_is_A & defender_block, axis=1).astype(np.int64)
    expected_final_att = np.sum(troops * decoded_is_A, axis=1).astype(np.int64)

    if "is_conquered" in payload:
        got = np.asarray(payload["is_conquered"]).astype(np.int64)
        if got.shape[0] != p.shape[0] or np.max(np.abs(got - expected_is_conq.astype(np.int64))) > metric_tol:
            problems.append("is_conquered metric mismatch")

    if "new_territories" in payload:
        got = np.asarray(payload["new_territories"]).astype(np.int64)
        if got.shape[0] != p.shape[0] or np.max(np.abs(got - expected_new)) > metric_tol:
            problems.append("new_territories metric mismatch")

    if "final_attacker_troops" in payload:
        got = np.asarray(payload["final_attacker_troops"]).astype(np.int64)
        if got.shape[0] != p.shape[0] or np.max(np.abs(got - expected_final_att)) > metric_tol:
            problems.append("final_attacker_troops metric mismatch")

    return problems


def validate_payload_or_options(
    payload: Dict[str, Any],
    *,
    nA: int,
    n_nodes: int,
    prob_tol: float,
    metric_tol: float,
) -> Tuple[List[str], int]:
    """Validate either a single V2 payload or a policy_options_v2 payload.

    Returns (problems, number_of_option_payloads_checked).
    """
    if isinstance(payload, dict) and payload.get("format") == "policy_options_v2":
        options = payload.get("options")
        if not isinstance(options, list) or not options:
            return ["policy_options_v2 has no options"], 0
        problems: List[str] = []
        checked = 0
        for i, opt in enumerate(options):
            if not isinstance(opt, dict):
                problems.append(f"option {i} is not a dict")
                continue
            checked += 1
            sub = validate_single_payload(
                opt,
                nA=nA,
                n_nodes=n_nodes,
                prob_tol=prob_tol,
                metric_tol=metric_tol,
            )
            problems.extend([f"option {i}: {x}" for x in sub])
        return problems, checked

    return validate_single_payload(
        payload,
        nA=nA,
        n_nodes=n_nodes,
        prob_tol=prob_tol,
        metric_tol=metric_tol,
    ), 1


# ---------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------

def reference_rowdict(
    *,
    row_label: str,
    edges: List[Tuple[int, int]],
    nA: int,
    nD: int,
    maxA: int,
    maxD: int,
    utility_mode: str,
) -> Dict[str, float]:
    att, deff = row_label_to_troops(row_label, nA)
    start = initial_state_generic(att, deff)
    combat_df = combat_df_for_caps(
        num_attacker_nodes=nA,
        num_defender_nodes=nD,
        max_attacker_troops=maxA,
        max_defender_troops=maxD,
    )

    if utility_mode == "local":
        dist, _val, _policy = explore_absorbing_states_for_graph_local_objective(
            edges=edges,
            combat_df=combat_df,
            start_state=start,
            num_attacker_nodes=nA,
        )
    elif utility_mode == "legacy":
        dist, _val, _policy = explore_absorbing_states_for_graph(
            edges=edges,
            combat_df=combat_df,
            start_state=start,
            num_attacker_nodes=nA,
        )
    else:
        raise ValueError(f"unsupported utility_mode={utility_mode!r}")

    return {encode_state_label(st): float(p) for st, p in dist.items() if float(p) > 0.0}


# ---------------------------------------------------------------------
# Library checks
# ---------------------------------------------------------------------

def check_one_library(
    path: Path,
    *,
    nA: int,
    nD: int,
    maxA: int,
    maxD: int,
    sample_rows: int,
    reference_rows: int,
    rng: random.Random,
    prob_tol: float,
    ref_tol: float,
    verbose: bool,
) -> Dict[str, Any]:
    n_nodes = nA + nD
    expected_rows = (maxA ** nA) * (maxD ** nD)
    problems: List[str] = []

    try:
        lib = load_graph_library(path)
    except Exception as e:
        return {"path": str(path), "ok": False, "problems": [f"failed to load library: {type(e).__name__}: {e}"]}

    params = lib.get("params", {}) or {}
    utility_mode = str(params.get("utility_mode", "local"))
    edges = [tuple(e) for e in lib.get("edges", [])]
    desc = lib.get("prob_table_chunked")

    if lib.get("prob_format") != "chunked_rows":
        problems.append(f"prob_format={lib.get('prob_format')!r}, expected 'chunked_rows'")
    if not isinstance(desc, dict):
        problems.append("prob_table_chunked missing/not a dict")
        return {"path": str(path), "ok": False, "problems": problems}
    if desc.get("format") != "chunked_prob_table_v2_rows_v1":
        problems.append(f"descriptor format={desc.get('format')!r}, expected chunked_prob_table_v2_rows_v1")

    chunks = desc.get("chunks", [])
    row_to_chunk = desc.get("row_to_chunk", {})
    base_path = desc.get("_base_path")

    if not isinstance(chunks, list):
        problems.append("chunks is not a list")
        chunks = []
    if not isinstance(row_to_chunk, dict):
        problems.append("row_to_chunk is not a dict")
        row_to_chunk = {}
    if not base_path:
        problems.append("descriptor missing _base_path")
    else:
        folder = Path(base_path)
        if not folder.exists():
            problems.append(f"chunk folder does not exist: {folder}")
        else:
            missing = [c for c in chunks if not (folder / str(c)).exists()]
            if missing:
                problems.append(f"missing {len(missing)} chunk files; first={missing[:3]}")

    exact_df = desc.get("exact_df")
    exact_index = set(exact_df.index) if exact_df is not None and hasattr(exact_df, "index") else set()
    covered_rows = set(row_to_chunk.keys()) | exact_index
    if len(covered_rows) != expected_rows:
        problems.append(f"covered rows={len(covered_rows)}, expected={expected_rows}")

    # Check a few expected row labels are present and loadable.
    rows_to_check = sample_row_labels(nA, nD, maxA, maxD, sample_rows, rng)
    missing_rows = [r for r in rows_to_check if r not in covered_rows]
    if missing_rows:
        problems.append(f"sample rows missing from coverage: {missing_rows[:5]}")

    payloads_checked = 0
    options_checked = 0
    payload_problem_count = 0
    loaded_payloads: Dict[str, Dict[str, Any]] = {}

    for row_label in rows_to_check:
        try:
            payload = get_prob_row_payload_from_library(
                lib,
                row_label,
                allow_extrapolation=False,
                num_attacker_nodes=nA,
                library_pkl_path=str(path),
            )
        except Exception as e:
            problems.append(f"row {row_label}: failed to load payload: {type(e).__name__}: {e}")
            continue

        if payload is None:
            problems.append(f"row {row_label}: payload is None")
            continue

        loaded_payloads[row_label] = payload
        payloads_checked += 1
        subproblems, nopts = validate_payload_or_options(
            payload,
            nA=nA,
            n_nodes=n_nodes,
            prob_tol=prob_tol,
            metric_tol=0.0,
        )
        options_checked += nopts
        if subproblems:
            payload_problem_count += len(subproblems)
            problems.extend([f"row {row_label}: {x}" for x in subproblems[:8]])

    # Optional old-solver comparisons. Only compare single-payload rows or
    # policy_options_v2 rows with exactly one option, because multi-option rows
    # intentionally may preserve several equivalent distributions.
    ref_compared = 0
    ref_failures = 0
    for row_label in rows_to_check[: max(0, int(reference_rows))]:
        payload = loaded_payloads.get(row_label)
        if payload is None:
            continue

        compare_payload = payload
        if isinstance(payload, dict) and payload.get("format") == "policy_options_v2":
            options = payload.get("options") or []
            if len(options) != 1:
                if verbose:
                    print(f"[SKIP REF] {path.name} {row_label}: {len(options)} policy options")
                continue
            compare_payload = options[0]

        try:
            got = payload_to_rowdict(compare_payload)
            ref = reference_rowdict(
                row_label=row_label,
                edges=edges,
                nA=nA,
                nD=nD,
                maxA=maxA,
                maxD=maxD,
                utility_mode=utility_mode,
            )
            cmp = compare_rowdicts(got, ref)
            ref_compared += 1
            if cmp["max_abs_diff"] > ref_tol or cmp["l1_diff"] > max(ref_tol * 10, 1e-6):
                ref_failures += 1
                problems.append(f"row {row_label}: reference mismatch {cmp}")
        except Exception as e:
            ref_failures += 1
            problems.append(f"row {row_label}: reference comparison failed: {type(e).__name__}: {e}")

    ok = not problems
    if verbose or not ok:
        status = "OK" if ok else "FAIL"
        print(
            f"[{status}] {path.name} | rows={len(covered_rows)}/{expected_rows} "
            f"chunks={len(chunks)} payloads={payloads_checked} options={options_checked} "
            f"ref={ref_compared} ref_fail={ref_failures}"
        )
        for p in problems[:12]:
            print(f"  - {p}")
        if len(problems) > 12:
            print(f"  ... {len(problems) - 12} more problems")

    return {
        "path": str(path),
        "ok": ok,
        "problems": problems,
        "num_chunks": len(chunks),
        "covered_rows": len(covered_rows),
        "expected_rows": expected_rows,
        "payloads_checked": payloads_checked,
        "options_checked": options_checked,
        "reference_compared": ref_compared,
        "reference_failures": ref_failures,
        "utility_mode": utility_mode,
        "edges": edges,
    }


def find_graph_libraries(base_dir: Path, nA: int, nD: int, maxA: int, maxD: int) -> List[Path]:
    folder = base_dir / f"{nA}A_{nD}D" / f"A{maxA}_D{maxD}"
    if not folder.exists():
        return []
    return sorted(folder.glob("graph_*.pkl"))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default=Path("small_graph_libraries"))
    ap.add_argument("--nA", type=int, required=True)
    ap.add_argument("--nD", type=int, required=True)
    ap.add_argument("--maxA", type=int, required=True)
    ap.add_argument("--maxD", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=0, help="0 means all graph libraries")
    ap.add_argument("--sample-rows", type=int, default=8, help="random rows per graph, plus deterministic corners")
    ap.add_argument("--reference-rows", type=int, default=2, help="rows per graph to compare against old exact solver; 0 disables")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--prob-tol", type=float, default=1e-5)
    ap.add_argument("--ref-tol", type=float, default=2e-5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    rng = random.Random(int(args.seed))
    paths = find_graph_libraries(args.base_dir, args.nA, args.nD, args.maxA, args.maxD)
    if int(args.max_graphs) > 0:
        paths = paths[: int(args.max_graphs)]

    print("=== exact finite library content check ===")
    print(f"base_dir     : {args.base_dir.resolve()}")
    print(f"combo        : {args.nA}A_{args.nD}D A≤{args.maxA}, D≤{args.maxD}")
    print(f"graphs found : {len(paths)}")
    print(f"sample rows  : {args.sample_rows} random + corners")
    print(f"ref rows     : {args.reference_rows}")

    if not paths:
        print("ERROR: no graph_*.pkl libraries found for this combo.")
        return 2

    t0 = time.time()
    results = []
    for i, path in enumerate(paths, start=1):
        if args.verbose:
            print(f"\n--- graph {i}/{len(paths)} ---")
        results.append(
            check_one_library(
                path,
                nA=args.nA,
                nD=args.nD,
                maxA=args.maxA,
                maxD=args.maxD,
                sample_rows=args.sample_rows,
                reference_rows=args.reference_rows,
                rng=rng,
                prob_tol=args.prob_tol,
                ref_tol=args.ref_tol,
                verbose=args.verbose,
            )
        )

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    n_fail = len(results) - n_ok
    total_payloads = sum(int(r.get("payloads_checked", 0)) for r in results)
    total_ref = sum(int(r.get("reference_compared", 0)) for r in results)
    total_ref_fail = sum(int(r.get("reference_failures", 0)) for r in results)

    print("\n=== summary ===")
    print(f"graphs checked       : {len(results)}")
    print(f"graphs OK            : {n_ok}")
    print(f"graphs failed        : {n_fail}")
    print(f"payload rows checked : {total_payloads}")
    print(f"reference rows       : {total_ref}")
    print(f"reference failures   : {total_ref_fail}")
    print(f"elapsed seconds      : {elapsed:.2f}")

    if n_fail:
        print("\nFailed libraries:")
        for r in results:
            if not r["ok"]:
                print(f"- {Path(r['path']).name}: {len(r['problems'])} problem(s)")
                for p in r["problems"][:4]:
                    print(f"    {p}")
        return 1

    print("\nAll checked libraries passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
