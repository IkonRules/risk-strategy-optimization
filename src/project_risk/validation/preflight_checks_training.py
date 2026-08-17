# preflight_checks_training.py
#
# Purpose:
#   Fast, deterministic pre-flight checks before running train_ML.py.
#   Verifies:
#     1) small_graph_libraries folder exists + looks populated
#     2) Board continents are available
#     3) For EACH continent:
#         - full_graph builds (non-empty)
#         - state generator runs once (ml_full_graph_state_generator)
#         - global_state builds without errors
#         - battle_graph builds and has nodes/edges (or at least nodes)
#     4) Spot-check that at least one small-graph library file can be loaded
#
# Usage:
#   python preflight_checks_training.py
#
# Optional:
#   python preflight_checks_training.py --libraries small_graph_libraries --seed 123 --smoke-continent "North America"

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from project_risk.game_simulation import Board
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.transition_prediction_ml.state_generators import ml_full_graph_state_generator
from project_risk.mathematical.transition_prediction_ml.generate_data_ML import build_full_graph
from project_risk.mathematical.transition_prediction_ml.train_ML import build_experiment_config

# Library IO (spot-check)
from project_risk.mathematical.libraries.create_library import load_library, graph_path, combos, star_combos


def _p(msg: str) -> None:
    print(msg, flush=True)


def _fail(msg: str, *, exit_code: int = 2) -> None:
    _p(f"❌ {msg}")
    sys.exit(exit_code)


def _warn(msg: str) -> None:
    _p(f"⚠️  {msg}")


def _ok(msg: str) -> None:
    _p(f"✅ {msg}")


def _count_files(root: Path, max_depth: int = 4) -> int:
    """
    Rough file count, depth-limited to avoid walking huge trees forever.
    """
    if not root.exists():
        return 0
    root = root.resolve()
    n = 0
    # Depth-limited walk
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            for p in d.iterdir():
                if p.is_file():
                    n += 1
                elif p.is_dir():
                    stack.append((p, depth + 1))
        except Exception:
            continue
    return n


def check_libraries_folder(libraries_base: Path) -> None:
    _p("\n=== Check 1: small_graph_libraries presence ===")
    if not libraries_base.exists():
        _fail(f"Missing libraries folder: {libraries_base}")

    if not libraries_base.is_dir():
        _fail(f"Libraries path exists but is not a directory: {libraries_base}")

    # Heuristic: should have a decent number of files if built
    nfiles = _count_files(libraries_base, max_depth=4)
    if nfiles < 20:
        _warn(
            f"Libraries folder exists but looks sparse (depth-limited file count={nfiles}). "
            "If you just started building libraries this might be OK, otherwise coverage may be missing."
        )
    else:
        _ok(f"Libraries folder exists and looks populated (depth-limited file count={nfiles}).")


def check_board_continents() -> list[str]:
    _p("\n=== Check 2: Board continents ===")
    try:
        conts = list(Board.continent_territory_dict.keys())
    except Exception as e:
        _fail(f"Could not read Board.continent_territory_dict: {e}")

    if not conts:
        _fail("Board.continent_territory_dict is empty.")

    _ok(f"Found {len(conts)} continents: {conts}")
    return conts


def _graph_size(g: Any) -> tuple[int, int]:
    try:
        n = int(g.number_of_nodes())
    except Exception:
        try:
            n = len(list(g.nodes()))
        except Exception:
            n = -1
    try:
        m = int(g.number_of_edges())
    except Exception:
        try:
            m = len(list(g.edges()))
        except Exception:
            m = -1
    return n, m


def check_generator_for_continent(
    *,
    continent_name: str,
    rng: np.random.Generator,
    constraints,
) -> None:
    """
    Runs one sample state generation and validates that the core objects can be built.
    """
    # Pick one target pair (not too extreme) to avoid degenerate graphs
    target_territory_ratio = 0.5
    target_troops_ratio = 1.0

    players, battle_graph, full_graph = ml_full_graph_state_generator(
        target_territory_ratio=target_territory_ratio,
        target_troops_ratio=target_troops_ratio,
        constraints=constraints,
        rng=rng,
    )

    if not players or len(players) < 2:
        raise RuntimeError("State generator returned invalid players list.")

    # Build global state (must work)
    gs = agop.build_global_state_for_board(players)
    if gs is None or not hasattr(gs, "nodes"):
        raise RuntimeError("build_global_state_for_board returned invalid GlobalState.")

    # full_graph sanity
    fn, fm = _graph_size(full_graph)
    if fn <= 0:
        raise RuntimeError("full_graph is empty.")
    # battle_graph sanity (can be small but should usually be non-empty)
    bn, bm = _graph_size(battle_graph)
    if bn < 0:
        raise RuntimeError("battle_graph object does not expose nodes.")
    if bn == 0:
        _warn("battle_graph has 0 nodes (possible if no conflicts / no available attacks).")

    # Also confirm build_full_graph(continent_name) works (independent path)
    fg2 = build_full_graph(continent_name)
    fn2, fm2 = _graph_size(fg2)
    if fn2 <= 0:
        raise RuntimeError("build_full_graph(continent_name) returned empty graph.")

    return  # success


def check_all_continents_state_generation(
    continents: list[str],
    *,
    libraries_base: Path,
    seed: int,
    smoke_continent: str | None = None,
) -> None:
    _p("\n=== Check 3: Per-continent generator + graph construction ===")

    config = build_experiment_config()

    # We'll reuse config constraints but swap continent_name per loop.
    base_constraints = config.constraints

    rng = np.random.default_rng(seed)

    cont_list = continents
    if smoke_continent:
        if smoke_continent not in continents:
            _fail(f"--smoke-continent '{smoke_continent}' is not in Board.continent_territory_dict keys.")
        cont_list = [smoke_continent]
        _warn(f"Smoke mode enabled: only checking continent='{smoke_continent}'.")

    failures: list[tuple[str, str]] = []

    for c in cont_list:
        _p(f"\n--- Continent: {c} ---")
        try:
            # clone constraints shallowly (keep other fields)
            constraints = type(base_constraints)(**vars(base_constraints))
            constraints.continent_name = c

            # quick check: continent exists and has territories
            terrs = Board.continent_territory_dict.get(c, [])
            if not terrs:
                raise RuntimeError("continent_territory_dict has empty territory list.")

            check_generator_for_continent(continent_name=c, rng=rng, constraints=constraints)
            _ok("State generation + global_state + full_graph + battle_graph OK.")
        except Exception as e:
            tb = "".join(traceback.format_exception_only(type(e), e)).strip()
            failures.append((c, tb))
            _p(f"❌ Failed for continent '{c}': {tb}")

    if failures:
        _p("\n=== FAILURES ===")
        for c, err in failures:
            _p(f"- {c}: {err}")
        _fail(f"{len(failures)} continent(s) failed generator/graph checks. Fix these before training.", exit_code=3)

    _ok("All checked continents passed generator/graph checks.")


def spot_check_library_load(libraries_base: Path) -> None:
    _p("\n=== Check 4: Spot-check a small-graph library loads ===")

    # Choose a small pattern that should exist in almost all setups.
    # We'll try a few patterns in order until one loads.
    candidate_patterns = [
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
    ]

    # Build a list of available patterns from combos/star_combos (source of truth in your codebase)
    available_patterns = {(nA, nD) for (nA, nD, *_rest) in combos} | {(nA, nD) for (nA, nD, *_rest) in star_combos}

    tried = []
    for pat in candidate_patterns:
        if pat not in available_patterns:
            continue

        # Find *any* canonical graph file path for that pattern.
        # We can attempt to load the "master lib index" for that pattern,
        # or load a specific graph file if graph_path resolves it.
        try:
            nA, nD = pat
            tried.append(pat)

            # load_library expects a directory; it loads the per-pattern library index.
            lib_obj = load_library(libraries_base, nA, nD)
            if lib_obj is None:
                raise RuntimeError("load_library returned None")

            _ok(f"Loaded library index for pattern (nA={nA}, nD={nD}).")
            return
        except Exception as e:
            _warn(f"Could not load library for pattern {pat}: {e}")

    if not tried:
        _warn("None of the candidate patterns were present in combos/star_combos; cannot spot-check load.")
        return

    _fail(
        "Could not load ANY candidate small-graph library index. "
        "Likely libraries are missing/corrupt or libraries_base is wrong."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-flight checks before running train_ML.py")
    ap.add_argument("--libraries", type=str, default="small_graph_libraries", help="Path to small_graph_libraries folder.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for generator smoke tests.")
    ap.add_argument("--smoke-continent", type=str, default=None, help="Only test one continent by name.")
    args = ap.parse_args()

    libraries_base = Path(args.libraries)

    check_libraries_folder(libraries_base)

    continents = check_board_continents()

    # Spot-check library loading early (fast failure)
    spot_check_library_load(libraries_base)

    # Check generator across continents (or one)
    check_all_continents_state_generation(
        continents,
        libraries_base=libraries_base,
        seed=args.seed,
        smoke_continent=args.smoke_continent,
    )

    _p("\n🎉 Pre-flight checks completed successfully. You are ready to run train_ML.py.\n")


if __name__ == "__main__":
    main()
