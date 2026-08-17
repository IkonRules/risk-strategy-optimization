from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import pickle
from typing import Any, Dict, Iterable, List, Tuple

from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.libraries import create_library
from project_risk.mathematical.libraries import library_io


BASE_DIR = Path("small_graph_libraries")

REGULAR_COMBOS = [
    (1, 1, 10, 10),
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

STAR_COMBOS = [
    (1, 5, 7, 7),
    (1, 6, 7, 7),
    (5, 1, 7, 7),
    (6, 1, 7, 7),
]

EXPECTED_PARAMS = {
    "multi_policy_options": True,
    "policy_option_mode": "state_set",
    "max_policy_options_per_row": 2,
    "max_options_per_state": 2,
    "max_leaf_split_depth": 1,
}


def _norm_edges(edges: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted(tuple(sorted((int(a), int(b)))) for a, b in edges))


def _combo_label(nA: int, nD: int, maxA: int, maxD: int, *, star: bool = False) -> str:
    suffix = " star-only" if star else ""
    return f"{nA}A_{nD}D A{maxA}_D{maxD}{suffix}"


def _combo_folder(nA: int, nD: int, maxA: int, maxD: int) -> Path:
    return BASE_DIR / f"{nA}A_{nD}D" / f"A{maxA}_D{maxD}"


def _expected_path(edges, nA: int, nD: int, maxA: int, maxD: int) -> Path:
    # Avoid create_library.graph_path because it creates directories.
    return _combo_folder(nA, nD, maxA, maxD) / create_library.graph_filename(nA, nD, maxA, maxD, edges)


def _canonical_edges(edges, nA: int, nD: int) -> Tuple[Tuple[int, int], ...]:
    canon, _old_to_new, _new_to_old = create_library.canonicalize_edges_with_roles(
        edges=sorted(tuple(sorted(edge)) for edge in edges),
        num_attacker_nodes=nA,
        num_defender_nodes=nD,
    )
    return _norm_edges(canon)


def _matching_graph_files(folder: Path, edges, nA: int, nD: int) -> List[Path]:
    target = _canonical_edges(edges, nA, nD)
    out: List[Path] = []
    for pkl in sorted(folder.glob("graph_*.pkl")):
        try:
            lib = create_library.load_library(pkl)
            lib_edges = lib.get("edges")
            if lib_edges is None:
                continue
            if _canonical_edges(lib_edges, nA, nD) == target:
                out.append(pkl)
        except Exception:
            continue
    return out


def _resolve_production_graph(edges, nA: int, nD: int, maxA: int, maxD: int, failures: List[str]) -> Path | None:
    folder = _combo_folder(nA, nD, maxA, maxD)
    exact = _expected_path(edges, nA, nD, maxA, maxD)
    matches = _matching_graph_files(folder, edges, nA, nD) if folder.exists() else []

    if exact.exists():
        return exact
    if len(matches) == 1:
        return matches[0]
    if not matches:
        failures.append(f"Missing expected graph: combo={nA}A_{nD}D A{maxA}_D{maxD} edges={edges}")
        return None
    failures.append(
        f"Ambiguous fallback graph resolution: combo={nA}A_{nD}D A{maxA}_D{maxD} "
        f"edges={edges} matches={[str(p) for p in matches]}"
    )
    return None


def _state_value(obj: Any) -> Any:
    if isinstance(obj, bool):
        return obj
    return obj


def _check_state_set_metadata(container: Dict[str, Any], where: str, failures: List[str]) -> None:
    for key, expected in EXPECTED_PARAMS.items():
        actual = container.get(key)
        if _state_value(actual) != expected:
            failures.append(f"{where}: expected {key}={expected!r}, got {actual!r}")
    if container.get("output_format") != "chunked_rows_v2" and where.endswith("params"):
        failures.append(f"{where}: expected output_format='chunked_rows_v2', got {container.get('output_format')!r}")
    if container.get("builder") != "compact_exact_finite_v1" and where.endswith("params"):
        failures.append(f"{where}: expected builder='compact_exact_finite_v1', got {container.get('builder')!r}")


def _option_count(payload: Dict[str, Any]) -> int:
    return len(payload.get("options", []) or [])


def _same_root_non_root_split(payload: Dict[str, Any]) -> bool:
    options = payload.get("options", []) or []
    for i, left in enumerate(options):
        for right in options[i + 1 :]:
            if (
                left.get("root_action") == right.get("root_action")
                and left.get("split_metadata") != right.get("split_metadata")
                and (left.get("split_metadata") or right.get("split_metadata"))
            ):
                return True
    return False


def _chunk_folder(graph_file: Path, desc: Dict[str, Any]) -> Path:
    return graph_file.parent / str(desc.get("chunk_dir"))


def _load_graph_and_validate(
    graph_file: Path,
    *,
    nA: int,
    nD: int,
    maxA: int,
    maxD: int,
    failures: List[str],
) -> Dict[str, Any]:
    label = str(graph_file)
    try:
        lib = create_library.load_library(graph_file)
    except Exception as exc:
        failures.append(f"{label}: failed to load pkl: {exc!r}")
        return {"ok": False}

    if lib.get("prob_format") != "chunked_rows":
        failures.append(f"{label}: expected prob_format='chunked_rows', got {lib.get('prob_format')!r}")

    params = lib.get("params", {}) or {}
    _check_state_set_metadata(params, f"{label} params", failures)

    desc = lib.get("prob_table_chunked")
    if not isinstance(desc, dict):
        failures.append(f"{label}: missing prob_table_chunked descriptor")
        return {"ok": False}
    if desc.get("format") != "chunked_prob_table_v2_rows_v1":
        failures.append(f"{label}: expected V2 descriptor, got {desc.get('format')!r}")

    _check_state_set_metadata(desc, f"{label} descriptor", failures)

    for key in ("chunks", "row_to_chunk", "chunk_dir"):
        if not desc.get(key):
            failures.append(f"{label}: descriptor missing/nonempty required key {key!r}")

    rows_per_graph = int(maxA) ** int(nA) * int(maxD) ** int(nD)
    row_to_chunk = desc.get("row_to_chunk", {}) if isinstance(desc.get("row_to_chunk"), dict) else {}
    if len(row_to_chunk) != rows_per_graph:
        failures.append(f"{label}: row_to_chunk length {len(row_to_chunk)} != expected {rows_per_graph}")

    chunks = list(desc.get("chunks", []) or [])
    chunk_dir = _chunk_folder(graph_file, desc)
    loaded_rows = 0
    hist: Dict[int, int] = defaultdict(int)
    two_option_rows = 0
    same_root_split_rows = 0
    max_option_count = 0
    largest_chunk = (None, 0)
    sample_row_label = None
    sample_row_payload = None
    preferred_row_label = None
    preferred_row_payload = None

    for chunk_name in chunks:
        chunk_path = chunk_dir / str(chunk_name)
        if not chunk_path.exists():
            failures.append(f"{label}: referenced chunk missing: {chunk_path}")
            continue
        size = chunk_path.stat().st_size
        if size > largest_chunk[1]:
            largest_chunk = (str(chunk_path), size)
        try:
            with chunk_path.open("rb") as f:
                chunk = pickle.load(f)
        except Exception as exc:
            failures.append(f"{label}: failed to load chunk {chunk_path}: {exc!r}")
            continue
        if not isinstance(chunk, dict) or chunk.get("format") != "v2_rowchunk_v1":
            failures.append(f"{label}: chunk {chunk_path} has invalid format {getattr(chunk, 'get', lambda _k: None)('format')!r}")
            continue
        rows = chunk.get("rows")
        if not isinstance(rows, dict):
            failures.append(f"{label}: chunk {chunk_path} missing rows dict")
            continue
        loaded_rows += len(rows)
        for row_label, payload in rows.items():
            if not isinstance(payload, dict) or payload.get("format") != "policy_options_v2":
                failures.append(f"{label}: row {row_label} is not policy_options_v2")
                continue
            n_opts = _option_count(payload)
            hist[n_opts] += 1
            max_option_count = max(max_option_count, n_opts)
            if n_opts > 2:
                failures.append(f"{label}: row {row_label} option_count {n_opts} > 2")
            if n_opts == 2:
                two_option_rows += 1
                if preferred_row_label is None:
                    preferred_row_label = row_label
                    preferred_row_payload = payload
            if _same_root_non_root_split(payload):
                same_root_split_rows += 1
                preferred_row_label = row_label
                preferred_row_payload = payload
            if sample_row_label is None:
                sample_row_label = row_label
                sample_row_payload = payload

    if loaded_rows != rows_per_graph:
        failures.append(f"{label}: loaded rows {loaded_rows} != expected {rows_per_graph}")

    return {
        "ok": True,
        "lib": lib,
        "desc": desc,
        "rows": loaded_rows,
        "chunks": len(chunks),
        "hist": dict(sorted(hist.items())),
        "two_option_rows": two_option_rows,
        "same_root_split_rows": same_root_split_rows,
        "max_option_count": max_option_count,
        "largest_chunk": largest_chunk,
        "sample_row_label": preferred_row_label or sample_row_label,
        "sample_row_payload": preferred_row_payload or sample_row_payload,
        "graph_size": graph_file.stat().st_size,
        "chunk_bytes": sum((chunk_dir / str(c)).stat().st_size for c in chunks if (chunk_dir / str(c)).exists()),
    }


def _merge_hist(dst: Dict[int, int], src: Dict[int, int]) -> None:
    for k, v in src.items():
        dst[int(k)] = dst.get(int(k), 0) + int(v)


def _direct_lookup_smoke(
    graph_file: Path,
    row_label: str,
    *,
    nA: int,
    failures: List[str],
) -> bool:
    try:
        lib = create_library.load_library(graph_file)
        payload = library_io.get_prob_row_payload_from_library(
            lib,
            row_label,
            allow_extrapolation=False,
            num_attacker_nodes=nA,
            library_pkl_path=str(graph_file),
        )
        options = library_io.normalize_payload_to_policy_options(payload)
        if payload is None or payload.get("format") != "policy_options_v2":
            failures.append(f"{graph_file}: direct lookup row {row_label} did not return policy_options_v2")
            return False
        if len(options) > 2:
            failures.append(f"{graph_file}: direct lookup option count {len(options)} > 2")
            return False
        agop.select_policy_option_payload(options, selection="primary")
        agop.select_policy_option_payload(options, selection="best_local")
        return True
    except Exception as exc:
        failures.append(f"{graph_file}: direct lookup smoke failed for row {row_label}: {exc!r}")
        return False


def _production_precedence_check(
    edges,
    *,
    nA: int,
    nD: int,
    maxA: int,
    maxD: int,
    expected_file: Path,
    failures: List[str],
) -> bool:
    try:
        desc = create_library.get_prob_table(
            edges=edges,
            num_attacker_nodes=nA,
            num_defender_nodes=nD,
            max_attacker_troops=maxA,
            max_defender_troops=maxD,
            base_dir=BASE_DIR,
            lazy_build=False,
            include_policies=False,
        )
        selected = Path(desc.get("library_pkl_path", "")).resolve()
        if selected != expected_file.resolve():
            failures.append(
                f"Production lookup selected unexpected file for {nA}A_{nD}D A{maxA}_D{maxD}: "
                f"{selected} != {expected_file.resolve()}"
            )
            return False
        _check_state_set_metadata(desc, f"{expected_file} production descriptor", failures)
        return True
    except Exception as exc:
        failures.append(f"Production precedence check failed for {expected_file}: {exc!r}")
        return False


def _summary_json_check(failures: List[str]) -> str:
    candidates = [
        BASE_DIR / "_full_state_set_build_summary.json",
        Path("small_graph_libraries_state_set_cap7_full") / "_full_state_set_build_summary.json",
    ]
    found = [p for p in candidates if p.exists()]
    if not found:
        return "MISSING"
    ok = True
    for path in found:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            settings = data.get("policy_settings", {}) or {}
            expected = {
                "policy_option_mode": "state_set",
                "max_policy_options_per_row": 2,
                "max_options_per_state": 2,
                "max_leaf_split_depth": 1,
                "chunk_rows": 5000,
                "n_jobs": 4,
            }
            for key, value in expected.items():
                if settings.get(key) != value:
                    failures.append(f"{path}: summary policy_settings {key}={settings.get(key)!r}, expected {value!r}")
                    ok = False
        except Exception as exc:
            failures.append(f"{path}: summary JSON load/check failed: {exc!r}")
            ok = False
    return "PASS" if ok else "FAIL"


def _expected_edges_for_combo(nA: int, nD: int, *, star: bool) -> Tuple[List[Tuple[Tuple[int, int], ...]], int]:
    if star:
        return [_norm_edges(create_library.make_star_edges_exact_finite(nA, nD))], 1
    edges, _source, _seen = create_library._canonical_edges_list_for_library_build(
        num_attacker_nodes=nA,
        num_defender_nodes=nD,
        edges_list=None,
        canonical_edges_list=None,
    )
    return [_norm_edges(e) for e in edges], len(edges)


def main() -> None:
    failures: List[str] = []
    combo_reports: List[Dict[str, Any]] = []
    global_hist: Dict[int, int] = {}
    direct_smoke_results: List[Tuple[str, bool]] = []
    precedence_results: List[Tuple[str, bool]] = []

    if not BASE_DIR.exists():
        failures.append(f"Base folder missing: {BASE_DIR}")

    expected_total = 0
    resolved_total = 0
    extra_total = 0
    total_rows = 0
    total_chunks = 0
    total_two_option_rows = 0
    total_same_root_split_rows = 0

    all_combos = [(c, False) for c in REGULAR_COMBOS] + [(c, True) for c in STAR_COMBOS]
    for (nA, nD, maxA, maxD), star in all_combos:
        label = _combo_label(nA, nD, maxA, maxD, star=star)
        combo_failures_before = len(failures)
        folder = _combo_folder(nA, nD, maxA, maxD)
        if not folder.exists():
            failures.append(f"{label}: combo folder missing: {folder}")
            combo_reports.append({"label": label, "expected": 0, "resolved": 0, "failures": len(failures) - combo_failures_before})
            continue

        expected_edges, expected_count = _expected_edges_for_combo(nA, nD, star=star)
        expected_total += expected_count
        all_graph_files = sorted(folder.glob("graph_*.pkl"))
        resolved_files: List[Path] = []
        combo_hist: Dict[int, int] = {}
        combo_rows = 0
        combo_chunks = 0
        combo_two = 0
        combo_same_root = 0
        combo_bytes = 0
        combo_largest_graph = (None, 0)
        combo_largest_chunk = (None, 0)
        combo_smoke_done = False

        for edges in expected_edges:
            resolved = _resolve_production_graph(edges, nA, nD, maxA, maxD, failures)
            if resolved is None:
                continue
            resolved_files.append(resolved)
            scan = _load_graph_and_validate(resolved, nA=nA, nD=nD, maxA=maxA, maxD=maxD, failures=failures)
            if not scan.get("ok"):
                continue

            resolved_total += 1
            combo_rows += int(scan["rows"])
            combo_chunks += int(scan["chunks"])
            combo_two += int(scan["two_option_rows"])
            combo_same_root += int(scan["same_root_split_rows"])
            combo_bytes += int(scan["graph_size"]) + int(scan["chunk_bytes"])
            total_rows += int(scan["rows"])
            total_chunks += int(scan["chunks"])
            total_two_option_rows += int(scan["two_option_rows"])
            total_same_root_split_rows += int(scan["same_root_split_rows"])
            _merge_hist(combo_hist, scan["hist"])
            _merge_hist(global_hist, scan["hist"])
            if int(scan["graph_size"]) > combo_largest_graph[1]:
                combo_largest_graph = (str(resolved), int(scan["graph_size"]))
            if scan["largest_chunk"][1] > combo_largest_chunk[1]:
                combo_largest_chunk = scan["largest_chunk"]

            if not combo_smoke_done and scan.get("sample_row_label"):
                ok = _direct_lookup_smoke(resolved, str(scan["sample_row_label"]), nA=nA, failures=failures)
                direct_smoke_results.append((label, ok))
                combo_smoke_done = True

            if len(precedence_results) < 12:
                ok = _production_precedence_check(
                    edges,
                    nA=nA,
                    nD=nD,
                    maxA=maxA,
                    maxD=maxD,
                    expected_file=resolved,
                    failures=failures,
                )
                precedence_results.append((label, ok))

        resolved_set = {p.resolve() for p in resolved_files}
        extras = [p for p in all_graph_files if p.resolve() not in resolved_set]
        extra_total += len(extras)

        if expected_count != len(resolved_files):
            failures.append(f"{label}: resolved {len(resolved_files)} expected graphs, expected {expected_count}")
        if combo_rows == 0:
            failures.append(f"{label}: zero verified rows")
        non_trivial = (nA + nD) > 2
        if non_trivial and combo_two <= 0:
            failures.append(f"{label}: non-trivial combo has zero two-option rows")

        combo_reports.append(
            {
                "label": label,
                "expected": expected_count,
                "resolved": len(resolved_files),
                "extra": len(extras),
                "rows": combo_rows,
                "chunks": combo_chunks,
                "bytes": combo_bytes,
                "hist": dict(sorted(combo_hist.items())),
                "two": combo_two,
                "same_root": combo_same_root,
                "largest_graph": combo_largest_graph,
                "largest_chunk": combo_largest_chunk,
                "failures": len(failures) - combo_failures_before,
            }
        )

    summary_status = _summary_json_check(failures)
    folder_size = sum(p.stat().st_size for p in BASE_DIR.rglob("*") if p.is_file()) if BASE_DIR.exists() else 0

    print("=== FULL STATE-SET CAP7 LIBRARY VERIFICATION ===")
    print(f"Base folder: {BASE_DIR}")
    print(f"Total size: {folder_size}")
    print(f"Total expected graph files: {expected_total}")
    print(f"Total resolved graph files: {resolved_total}")
    print(f"Total extra graph files: {extra_total}")
    print(f"Total rows: {total_rows}")
    print(f"Total chunks: {total_chunks}")
    print(f"Global option histogram: {dict(sorted(global_hist.items()))}")
    print(f"Rows with 2 options: {total_two_option_rows}")
    print(f"Same-root/non-root split rows: {total_same_root_split_rows}")
    print(f"Failures: {len(failures)}")
    print()
    print("Per-combo:")
    for report in combo_reports:
        print(
            f"  {report['label']}: expected={report['expected']} resolved={report['resolved']} "
            f"extra={report['extra']} rows={report['rows']} chunks={report['chunks']} "
            f"bytes={report['bytes']} hist={report['hist']} two={report['two']} "
            f"same_root={report['same_root']} failures={report['failures']}"
        )
        print(f"    largest_graph={report['largest_graph']}")
        print(f"    largest_chunk={report['largest_chunk']}")
    print()
    print("Direct production lookup smoke tests:")
    for label, ok in direct_smoke_results:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}")
    print()
    print("Production path precedence checks:")
    for label, ok in precedence_results:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}")
    print()
    print(f"Summary JSON: {summary_status}")

    if failures:
        print()
        print("Failure details:")
        for failure in failures[:200]:
            print(f"  - {failure}")
        if len(failures) > 200:
            print(f"  ... {len(failures) - 200} more failures omitted")
        print()
        print("FINAL RESULT: FAIL")
        raise SystemExit(1)

    print()
    print("FINAL RESULT: PASS")


if __name__ == "__main__":
    main()
