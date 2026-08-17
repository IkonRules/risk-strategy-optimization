# library_io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pickle
import pandas as pd
import numpy as np


from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
    global_state_from_row_label, parse_row_label)


# -------------------------
# Chunk cache (simple LRU)
# -------------------------

class ChunkCache:
    def __init__(
        self,
        max_chunks: int = 8,
        max_source_bytes: Optional[int] = None,
    ):
        self.max_chunks = int(max_chunks)
        self.max_source_bytes = (
            None if max_source_bytes is None else max(0, int(max_source_bytes))
        )
        self._cache: dict[tuple[str, str], Any] = {}     # (base_dir, rel_chunk) -> obj
        self._order: list[tuple[str, str]] = []          # LRU order: oldest first
        self._source_bytes: dict[tuple[str, str], int] = {}
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.evictions = 0
        self._seen_keys: set[tuple[str, str]] = set()
        self.peak_entries = 0
        self.peak_cached_source_bytes = 0

    def get(self, base_dir: str, rel_chunk: str):
        k = (base_dir, rel_chunk)
        self._seen_keys.add(k)
        if k not in self._cache:
            self.misses += 1
            return None
        self.hits += 1
        # bump LRU
        try:
            self._order.remove(k)
        except ValueError:
            pass
        self._order.append(k)
        return self._cache[k]

    def put(self, base_dir: str, rel_chunk: str, obj):
        k = (base_dir, rel_chunk)
        if k in self._cache:
            try:
                self._order.remove(k)
            except ValueError:
                pass
        self._cache[k] = obj
        self._order.append(k)
        try:
            self._source_bytes[k] = int((Path(base_dir) / rel_chunk).stat().st_size)
        except OSError:
            self._source_bytes[k] = 0
        self.stores += 1

        while self._order and (
            len(self._order) > self.max_chunks
            or (
                self.max_source_bytes is not None
                and sum(self._source_bytes.values()) > self.max_source_bytes
            )
        ):
            old = self._order.pop(0)
            self._cache.pop(old, None)
            self._source_bytes.pop(old, None)
            self.evictions += 1
        self.peak_entries = max(self.peak_entries, len(self._cache))
        self.peak_cached_source_bytes = max(
            self.peak_cached_source_bytes,
            sum(self._source_bytes.values()),
        )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "entries": int(len(self._cache)),
            "max_chunks": int(self.max_chunks),
            "max_source_bytes": self.max_source_bytes,
            "hits": int(self.hits),
            "misses": int(self.misses),
            "stores": int(self.stores),
            "evictions": int(self.evictions),
            "cached_source_bytes": int(sum(self._source_bytes.values())),
            "unique_chunk_keys_seen": int(len(self._seen_keys)),
            "peak_entries": int(self.peak_entries),
            "peak_cached_source_bytes": int(self.peak_cached_source_bytes),
        }


_chunk_cache = ChunkCache(max_chunks=8)


# -------------------------
# Row extraction helpers (V1: dict-of-probabilities)
# -------------------------

def _row_series_to_dict_nonzero(s: pd.Series) -> Dict[str, float]:
    """
    Convert a row Series to a {col: prob} dict, keeping only non-zeros.
    Handles dense and SparseDtype without densifying unnecessarily.
    """
    # Sparse series
    if isinstance(s.dtype, pd.SparseDtype):
        arr = s.array
        # pandas uses SparseArray for SparseDtype
        if isinstance(arr, pd.arrays.SparseArray):
            idx = arr.sp_index.indices
            vals = arr.sp_values
            cols = s.index
            out: Dict[str, float] = {}
            for j, v in zip(idx, vals):
                if v != 0.0:
                    out[str(cols[j])] = float(v)
            return out

    # Dense fallback
    out: Dict[str, float] = {}
    for k, v in s.items():
        if v != 0.0:
            out[str(k)] = float(v)
    return out


def _row_from_dataframe(df: pd.DataFrame, row_label: str) -> Optional[Dict[str, float]]:
    if row_label not in df.index:
        return None
    return _row_series_to_dict_nonzero(df.loc[row_label])


def _load_chunk_object(
    base_dir: str,
    rel_chunk: str,
    *,
    cache: Optional[ChunkCache] = None,
) -> Any:
    """
    Load a chunk from disk, with LRU caching.
    """
    active_cache = _chunk_cache if cache is None else cache
    cached = active_cache.get(base_dir, rel_chunk)
    if cached is not None:
        return cached

    p = (Path(base_dir) / rel_chunk)
    if not p.exists():
        raise FileNotFoundError(f"Missing chunk file: {p}")

    with p.open("rb") as f:
        obj = pickle.load(f)

    active_cache.put(base_dir, rel_chunk, obj)
    return obj


def _row_from_chunk_payload(obj: Any, row_label: str) -> Optional[Dict[str, float]]:
    """
    V1 chunk payload can be:
      - DataFrame with rows
      - dict with format=rowdict_chunk_v1 containing {"rows": {...}}
      - (optionally) raw dict row_label -> dict
    """
    if isinstance(obj, pd.DataFrame):
        return _row_from_dataframe(obj, row_label)

    if isinstance(obj, dict):
        # preferred format written by your builder
        if obj.get("format") == "rowdict_chunk_v1":
            rows = obj.get("rows", {})
            row = rows.get(row_label)
            if isinstance(row, dict):
                return {str(k): float(v) for k, v in row.items() if v != 0.0}
            return None

        # fallback: direct row_label -> rowdict
        row = obj.get(row_label)
        if isinstance(row, dict):
            return {str(k): float(v) for k, v in row.items() if v != 0.0}
        return None

    raise TypeError(f"Unsupported chunk payload type: {type(obj)}")


def _row_from_chunked_desc(chunked_desc: dict, row_label: str) -> Optional[Dict[str, float]]:
    """
    V1 descriptor expected runtime-ready:
      {
        "format": "chunked_prob_table_v1",
        "exact_df": <DataFrame>,
        "chunks": [<chunk filename>, ...],
        "row_to_chunk": {row_label: chunk_index, ...},
        "_base_path": "<folder containing chunk files>",   # injected by get_prob_table
      }
    """
    if chunked_desc.get("format") != "chunked_prob_table_v1":
        return None

    # 1) exact shortcut
    exact_df = chunked_desc.get("exact_df")
    if isinstance(exact_df, pd.DataFrame):
        r = _row_from_dataframe(exact_df, row_label)
        if r is not None:
            return r

    # 2) extended lookup
    row_to_chunk = chunked_desc.get("row_to_chunk", {})
    if row_label not in row_to_chunk:
        return None

    chunks = chunked_desc.get("chunks", [])
    idx = int(row_to_chunk[row_label])
    if idx < 0 or idx >= len(chunks):
        return None

    base_dir = chunked_desc.get("_base_path")
    if not base_dir:
        # treat as coverage failure (cannot resolve)
        return None

    rel = chunks[idx]
    obj = _load_chunk_object(str(base_dir), str(rel))
    return _row_from_chunk_payload(obj, row_label)


# -------------------------
# V2 (indexed outcomes + arrays) helpers
# -------------------------

def _v2_rowpayload_from_chunk_payload(obj: Any, row_label: str) -> Optional[Dict[str, Any]]:
    """
    V2 chunk payload expected:
      {
        "format": "v2_rowchunk_v1",
        "rows": { row_label: row_payload, ... }
      }

    row_payload is a dict containing arrays:
      "p": (N,) float32
      "owners": (N,M) uint8
      "troops": (N,M) uint16
      optional: "cdf", "is_conquered", "new_territories", "final_attacker_troops", ...
    """
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported V2 chunk payload type: {type(obj)}")

    if obj.get("format") != "v2_rowchunk_v1":
        return None

    rows = obj.get("rows", {})
    row = rows.get(row_label)
    if isinstance(row, dict):
        return row
    return None


def _v2_rowpayload_from_chunked_desc(chunked_desc: dict, row_label: str) -> Optional[Dict[str, Any]]:
    """
    V2 descriptor expected runtime-ready:
      {
        "format": "chunked_prob_table_v2_rows_v1",
        "exact_df": <DataFrame or None>         # (optional) legacy exact rows
        "chunks": [<chunk filename>, ...],
        "row_to_chunk": {row_label: chunk_index, ...},
        "_base_path": "<folder containing chunk files>",   # injected by get_prob_table
      }

    Returns:
      - row payload dict (arrays) if found
      - None if missing
    """
    if chunked_desc.get("format") != "chunked_prob_table_v2_rows_v1":
        return None

    # 1) exact shortcut (if exact_df is still stored as a DataFrame of label-probs)
    exact_df = chunked_desc.get("exact_df")
    if isinstance(exact_df, pd.DataFrame):
        r = _row_from_dataframe(exact_df, row_label)
        if r is not None:
            # NOTE: This is still legacy dict-of-probs (V1 style).
            # We return it as-is; callers can convert via prob_row_dict_labels_to_arrays.
            return {"_legacy_prob_row": r}

    # 2) extended lookup
    row_to_chunk = chunked_desc.get("row_to_chunk", {})
    if row_label not in row_to_chunk:
        return None

    chunks = chunked_desc.get("chunks", [])
    idx = int(row_to_chunk[row_label])
    if idx < 0 or idx >= len(chunks):
        return None

    base_dir = chunked_desc.get("_base_path")
    if not base_dir:
        return None

    rel = chunks[idx]
    obj = _load_chunk_object(str(base_dir), str(rel))
    return _v2_rowpayload_from_chunk_payload(obj, row_label)


# -------------------------
# Public API (existing)
# -------------------------

def get_prob_row_from_prob_table(*args: Any, **kwargs: Any) -> None:
    """Hard switch: V1 row dict access is disabled."""
    raise NotImplementedError(
        "Hard switch to v2 chunked storage: get_prob_row_from_prob_table() is disabled. "
        "Use get_prob_row_payload_from_prob_table() (v2 payload) instead."
    )


def _infer_num_attacker_nodes_from_row_label(rl: str) -> int:
    s = rl.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    nA = 0
    for p in parts:
        if p.startswith("A"):
            nA += 1
        else:
            break
    return nA

def _rowdict_to_v2_payload_with_metrics_from_labels(
    rowdict: dict[str, float],
    *,
    num_attacker_nodes: int,
) -> dict:
    """
    Convert {col_label: p} -> v2 payload with arrays and metrics,
    decoding labels using parse_row_label only.

    owners encoding: 0 empty, 1 attacker, 2 defender
    metrics semantics:
      - is_conquered: all nodes attacker-owned (no defenders remain)
      - new_territories: defender-block nodes that end attacker-owned
      - final_attacker_troops: sum troops on attacker-owned nodes
    """
    items = [(lbl, float(p)) for (lbl, p) in rowdict.items() if float(p) > 0.0]
    items.sort(key=lambda t: t[0])

    if not items:
        # Can't infer M reliably; return empty arrays with M=0
        return {
            "p": np.zeros((0,), dtype=np.float32),
            "owners": np.zeros((0, 0), dtype=np.uint8),
            "troops": np.zeros((0, 0), dtype=np.uint16),
            "is_conquered": np.zeros((0,), dtype=np.uint8),
            "new_territories": np.zeros((0,), dtype=np.int16),
            "final_attacker_troops": np.zeros((0,), dtype=np.int32),
        }

    # Infer M from the first end-state label
    owners0, troops0 = parse_row_label(items[0][0])
    M = len(troops0)
    N = len(items)

    p_arr = np.empty((N,), dtype=np.float32)
    owners_arr = np.empty((N, M), dtype=np.uint8)
    troops_arr = np.empty((N, M), dtype=np.uint16)

    is_conq_arr = np.empty((N,), dtype=np.uint8)
    new_terr_arr = np.empty((N,), dtype=np.int16)
    final_att_arr = np.empty((N,), dtype=np.int32)

    for i, (lbl, p) in enumerate(items):
        p_arr[i] = np.float32(p)

        owners_lbl, troops_lbl = parse_row_label(lbl)
        if len(troops_lbl) != M:
            raise ValueError(f"Inconsistent end-state label length: {lbl} (len={len(troops_lbl)}) expected M={M}")

        any_defender_remaining = False
        new_terr = 0
        final_att = 0

        for j in range(M):
            t = int(troops_lbl[j])
            if t <= 0:
                owners_arr[i, j] = 0
                troops_arr[i, j] = 0
                continue

            o = owners_lbl[j]
            if o == "A":
                owners_arr[i, j] = 1
            elif o == "D":
                owners_arr[i, j] = 2
            else:
                owners_arr[i, j] = 0

            troops_arr[i, j] = np.uint16(max(t, 0))

            if owners_arr[i, j] == 2:
                any_defender_remaining = True

            if owners_arr[i, j] == 1:
                final_att += t
                if j >= num_attacker_nodes:
                    new_terr += 1

        is_conq_arr[i] = np.uint8(0 if any_defender_remaining else 1)
        new_terr_arr[i] = np.int16(new_terr)
        final_att_arr[i] = np.int32(final_att)

    # Renormalize (defensive)
    s = float(p_arr.sum())
    if s > 0.0 and abs(s - 1.0) > 1e-6:
        p_arr = p_arr / np.float32(s)

    return {
        "p": p_arr,
        "owners": owners_arr,
        "troops": troops_arr,
        "is_conquered": is_conq_arr,
        "new_territories": new_terr_arr,
        "final_attacker_troops": final_att_arr,
    }


def _ensure_cdf(payload: dict) -> dict:
    import numpy as np
    if payload is None:
        return payload
    if "cdf" in payload and payload["cdf"] is not None:
        return payload
    p = payload.get("p", None)
    if p is None:
        return payload
    p = np.asarray(p)
    out = dict(payload)
    out["cdf"] = np.cumsum(p)
    return out


def get_prob_row_payload_from_prob_table(
    prob_table_chunked: Dict[str, Any],
    row_label: str,
    *,
    library_pkl_path: str | Path | None = None,
    chunk_cache: Optional[ChunkCache] = None,
) -> Optional[Dict[str, Any]]:
    """
    V2 fetch one row payload from a v2 chunked descriptor.

    PATCH:
      - If row_label is present in exact_df, convert that distribution into a V2 payload
        WITH metrics: is_conquered, new_territories, final_attacker_troops.
      - Otherwise, load from the chunked row-store as before.
      - Always ensure `cdf` exists on the returned payload (for sampling).
      - If an older v2 chunk payload is missing metrics, derive them from owners/troops.

    Chunk folder resolution:
      1) prob_table_chunked["_base_path"] if present
      2) prob_table_chunked["chunk_folder"] if present (legacy)
      3) (Path(library_pkl_path).parent / prob_table_chunked["chunk_dir"]) if provided
    """
    import pickle
    from pathlib import Path
    import numpy as np

    if not isinstance(prob_table_chunked, dict):
        raise TypeError("prob_table_chunked must be a dict descriptor")

    if prob_table_chunked.get("format") != "chunked_prob_table_v2_rows_v1":
        raise ValueError(
            f"Expected v2 chunked descriptor format 'chunked_prob_table_v2_rows_v1', "
            f"got {prob_table_chunked.get('format')!r}"
        )

    REQUIRED_METRICS = ("is_conquered", "new_territories", "final_attacker_troops")

    def _ensure_metrics_from_owners_troops(payload: Dict[str, Any], *, num_attacker_nodes: int) -> Dict[str, Any]:
        """
        Older v2 chunks may only contain: p, owners, troops.
        Compute missing metric arrays deterministically from owners/troops.

        owners encoding: 0 empty, 1 attacker, 2 defender
        semantics:
          - is_conquered: no defenders remain anywhere in the region
          - new_territories: defender-block indices (>= num_attacker_nodes) that are attacker-owned
          - final_attacker_troops: sum troops on attacker-owned nodes
        """
        if payload is None:
            return payload
        if all(k in payload for k in REQUIRED_METRICS):
            return payload
        if "owners" not in payload or "troops" not in payload or "p" not in payload:
            return payload

        owners = np.asarray(payload["owners"])
        troops = np.asarray(payload["troops"])

        if owners.ndim != 2 or troops.ndim != 2 or owners.shape != troops.shape:
            return payload

        att_mask = (owners == 1)
        def_mask = (owners == 2)

        defender_block = np.zeros_like(att_mask, dtype=bool)
        defender_block[:, int(num_attacker_nodes):] = True

        new_territories = np.sum(att_mask & defender_block, axis=1).astype(np.int16)
        is_conquered = (~np.any(def_mask, axis=1)).astype(np.uint8)
        final_attacker_troops = np.sum(troops * att_mask, axis=1).astype(np.int32)

        out = dict(payload)
        out.setdefault("new_territories", new_territories)
        out.setdefault("is_conquered", is_conquered)
        out.setdefault("final_attacker_troops", final_attacker_troops)
        return out

    # ------------------------------------------------------------
    # 0) exact_df path (convert DF distribution -> V2 payload w/ metrics)
    # ------------------------------------------------------------
    exact_df = prob_table_chunked.get("exact_df")
    if exact_df is not None and hasattr(exact_df, "index") and row_label in exact_df.index:
        df_row = exact_df.loc[row_label]
        rowdict = {k: float(v) for k, v in df_row.to_dict().items() if float(v) > 0.0}

        nA = _infer_num_attacker_nodes_from_row_label(row_label)
        payload = _rowdict_to_v2_payload_with_metrics_from_labels(rowdict, num_attacker_nodes=nA)

        # Ensure sampling support
        payload = _ensure_cdf(payload)
        return payload

    # ------------------------------------------------------------
    # 1) chunked path
    # ------------------------------------------------------------
    row_to_chunk = prob_table_chunked.get("row_to_chunk")
    chunks = prob_table_chunked.get("chunks")
    if not isinstance(row_to_chunk, dict) or not isinstance(chunks, list):
        raise KeyError("Descriptor missing row_to_chunk/chunks")

    chunk_idx = row_to_chunk.get(row_label)
    if chunk_idx is None:
        return None
    if not (0 <= int(chunk_idx) < len(chunks)):
        return None

    chunk_filename = chunks[int(chunk_idx)]

    # ---- Resolve base folder for chunks ----
    base_path = prob_table_chunked.get("_base_path")
    if isinstance(base_path, str) and base_path.strip():
        chunk_folder = Path(base_path).resolve()
    else:
        chunk_folder = None

    if chunk_folder is None:
        chunk_folder_str = prob_table_chunked.get("chunk_folder")
        if isinstance(chunk_folder_str, str) and chunk_folder_str.strip():
            chunk_folder = Path(chunk_folder_str).resolve()

    if chunk_folder is None:
        chunk_dir = prob_table_chunked.get("chunk_dir")
        if isinstance(chunk_dir, str) and chunk_dir.strip() and library_pkl_path is not None:
            chunk_folder = (Path(library_pkl_path).resolve().parent / chunk_dir).resolve()

    if chunk_folder is None:
        raise ValueError(
            "Chunk folder could not be resolved. Provide either:\n"
            "  - prob_table_chunked['_base_path'], OR\n"
            "  - prob_table_chunked['chunk_folder'], OR\n"
            "  - library_pkl_path + prob_table_chunked['chunk_dir']\n"
        )

    chunk_path = chunk_folder / chunk_filename
    if not chunk_path.exists():
        return None

    if chunk_cache is None:
        with chunk_path.open("rb") as f:
            chunk_obj = pickle.load(f)
    else:
        chunk_obj = _load_chunk_object(
            str(chunk_folder),
            str(chunk_filename),
            cache=chunk_cache,
        )

    if not isinstance(chunk_obj, dict) or chunk_obj.get("format") != "v2_rowchunk_v1":
        raise ValueError(f"Unexpected chunk file structure in {chunk_path}")

    rows = chunk_obj.get("rows")
    if not isinstance(rows, dict):
        raise ValueError(f"Chunk file missing 'rows' dict in {chunk_path}")

    payload = rows.get(row_label)
    if payload is None:
        return None

    # Attach missing metrics for older v2 libs + ensure sampling support
    nA = _infer_num_attacker_nodes_from_row_label(row_label)
    payload = _ensure_metrics_from_owners_troops(payload, num_attacker_nodes=nA)
    payload = _ensure_cdf(payload)

    return payload



def get_region_distribution_from_prob_table(
    prob_table: dict,
    row_label: str,
) -> Optional[Dict[str, Any]]:
    """Return a normalized region distribution (v2 only)."""
    payload = get_prob_row_payload_from_prob_table(prob_table, row_label)
    if payload is None:
        return None
    return {"format_version": 2, "dist_v2": payload}


def load_graph_library(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a per-graph library and enforce hard-switch invariants (v2 only)."""
    path = Path(path)
    with path.open("rb") as f:
        lib = pickle.load(f)

    if lib.get("prob_format") != "chunked_rows":
        raise ValueError(
            f"Hard switch: {path} prob_format must be 'chunked_rows', got {lib.get('prob_format')!r}"
        )
    desc = lib.get("prob_table_chunked")
    if not isinstance(desc, dict) or desc.get("format") != "chunked_prob_table_v2_rows_v1":
        raise ValueError(
            f"Hard switch: {path} must contain v2 chunked descriptor under 'prob_table_chunked'."
        )
    # Inject _base_path if missing (same rule as create_library)
    if "_base_path" not in desc:
        chunk_dir = desc.get("chunk_dir")
        if isinstance(chunk_dir, str) and chunk_dir.strip():
            desc["_base_path"] = str((path.parent / chunk_dir).resolve())
        else:
            desc["_base_path"] = str(path.parent.resolve())
    lib["prob_table_chunked"] = desc
    return lib


def _rules_from_dict(rules_dict: dict):
    """
    Rehydrate an ExtrapolationRules object from a serialized dict.

    This mirrors the structure produced by `rules_to_dict(...)` in
    extrapolate_config_distributions.py.

    IMPORTANT:
      - Uses lazy imports to avoid circular import with extrapolate_config_distributions,
        since that module imports library_io.
    """
    if not rules_dict:
        return None

    # Lazy import to avoid circular import
    from extrapolate_config_distributions import (
        ExtrapolationRules,
        NodeRule,
        FunctionalSeriesDiagnostics,
        StabilityThresholds,
    )

    thr = rules_dict.get("thresholds", {}) or {}
    thresholds = StabilityThresholds(
        eps_mono=float(thr.get("eps_mono", 1e-10)),
        alpha=float(thr.get("alpha", 0.9)),
        eps_tail=float(thr.get("eps_tail", 2e-3)),
    )

    node_rules = []
    for nr in (rules_dict.get("node_rules", []) or []):
        diagnostics = {}
        for fn, diag in (nr.get("diagnostics", {}) or {}).items():
            diagnostics[fn] = FunctionalSeriesDiagnostics(
                functional=diag.get("functional", fn),
                values=list(diag.get("values", [])),
                deltas=list(diag.get("deltas", [])),
                monotone_ok=bool(diag.get("monotone_ok", False)),
                diminishing_ok=bool(diag.get("diminishing_ok", False)),
                tail_small_ok=bool(diag.get("tail_small_ok", False)),
                stable=bool(diag.get("stable", False)),
            )

        node_rules.append(
            NodeRule(
                node_index=int(nr.get("node_index")),
                stable=bool(nr.get("stable", False)),
                stable_functionals=list(nr.get("stable_functionals", [])),
                diagnostics=diagnostics,
            )
        )

    return ExtrapolationRules(
        version=str(rules_dict.get("version", "unknown")),
        max_exact=int(rules_dict.get("max_exact", 6)),
        base_row_label=str(rules_dict.get("base_row_label", "")),
        thresholds=thresholds,
        node_rules=node_rules,
    )


def get_prob_row_payload_from_library(
    graph_library: dict,
    row_label: str,
    *,
    allow_extrapolation: bool = True,
    num_attacker_nodes: int | None = None,
    library_pkl_path: str | None = None,
) -> dict | None:
    """
    Fetch a V2 row payload for `row_label` from a loaded per-graph library.

    Behavior
    --------
    1) Try exact lookup from the library's chunked prob table (<= max_exact rows).
    2) If missing AND `allow_extrapolation=True` AND the library contains
       `extrapolation_rules`, return an approximate payload computed by the
       extrapolation module (currently: conservative "freeze at max_exact").

    Notes
    -----
    - Returns a standard V2 row payload dict (same format as exact rows).
    - Approximate payloads include a `_approximation` field describing provenance.
    - `num_attacker_nodes` is required for functional extraction/metric patching
      in the extrapolation path. If not supplied, we try to infer it from
      graph_library["params"] using common key variants.

    Parameters
    ----------
    graph_library:
        The object loaded from a per-graph library pickle (e.g. load_graph_library()).
        Must contain `prob_table_chunked` (V2 descriptor).
    row_label:
        Row label string, e.g. "(A2,A1,D3,...)"
    allow_extrapolation:
        If False, only exact lookup is attempted.
    num_attacker_nodes:
        Number of attacker-owned nodes in the topology/ownership context (nA).
        If None, inferred from graph_library["params"] when possible.
    library_pkl_path:
        Path to the library pickle (used by library_io to locate chunk files,
        and by the extrapolation module to resolve base paths consistently).

    Returns
    -------
    dict | None
        V2 row payload, or None if not found and cannot extrapolate.
    """
    # --- Basic sanity / fetch V2 descriptor ---
    prob_table = graph_library.get("prob_table_chunked")
    if prob_table is None:
        # Some older libraries might store exact_df instead; we keep this simple.
        return None

    # --- 1) Exact lookup ---
    payload = get_prob_row_payload_from_prob_table(
        prob_table,
        row_label,
        library_pkl_path=library_pkl_path,
    )
    if payload is not None or not allow_extrapolation:
        return payload

    # --- 2) Extrapolation fallback (if present) ---
    rules_dict = graph_library.get("extrapolation_rules")
    if not rules_dict:
        return None

    # Infer num_attacker_nodes if not provided
    if num_attacker_nodes is None:
        params = graph_library.get("params", {}) or {}
        for key in (
            "num_attacker_nodes",  # current writer key
            "nA",                  # possible legacy key
            "n_attacker",
            "nA_nodes",
        ):
            if key in params:
                try:
                    num_attacker_nodes = int(params[key])
                    break
                except Exception:
                    pass

    if num_attacker_nodes is None:
        raise ValueError(
            "num_attacker_nodes is required for extrapolation but could not be inferred "
            f"from graph_library['params'] keys: {list((graph_library.get('params', {}) or {}).keys())}"
        )

    # Lazy import to avoid circular imports at module import time
    from extrapolate_config_distributions import (
        StabilityThresholds,
        get_approx_row_payload,
    )

    # Rehydrate thresholds (rules object is rebuilt by get_approx_row_payload's caller)
    thr = rules_dict.get("thresholds", {}) or {}
    thresholds = StabilityThresholds(
        eps_mono=float(thr.get("eps_mono", 1e-10)),
        alpha=float(thr.get("alpha", 0.9)),
        eps_tail=float(thr.get("eps_tail", 2e-3)),
    )

    # Rehydrate full ExtrapolationRules via helper (keeps future rule types compatible)
    rules = _rules_from_dict({
        "version": rules_dict.get("version", "freeze_v1"),
        "max_exact": rules_dict.get("max_exact", 6),
        "base_row_label": rules_dict.get("base_row_label", ""),
        "thresholds": {"eps_mono": thresholds.eps_mono, "alpha": thresholds.alpha, "eps_tail": thresholds.eps_tail},
        "node_rules": rules_dict.get("node_rules", []),
    })
    if rules is None:
        return None

    return get_approx_row_payload(
        prob_table,
        row_label,
        rules=rules,
        num_attacker_nodes=num_attacker_nodes,
        library_pkl_path=library_pkl_path,
    )



# ---------------------------------------------------------------------
# Policy-option row helpers
# ---------------------------------------------------------------------

def normalize_payload_to_policy_options(payload):
    """
    Normalize both supported row shapes to a list of policy-option payloads.

    Supported inputs
    ----------------
    1. Old/single-policy V2 row payload:
       {"p": ..., "owners": ..., "troops": ..., ...}
       -> returns [payload_with_option_id_0]

    2. New multi-policy row payload:
       {"format": "policy_options_v2", "options": [...]}
       -> returns payload["options"] with option_id/cdf ensured.
    """
    if payload is None or payload == {}:
        return []

    if isinstance(payload, dict) and payload.get("format") == "policy_options_v2":
        out = []
        for i, opt in enumerate(payload.get("options", []) or []):
            if not isinstance(opt, dict):
                continue
            opt2 = dict(opt)
            opt2.setdefault("option_id", i)
            opt2 = _ensure_cdf(opt2)
            out.append(opt2)
        return out

    opt = dict(payload)
    opt.setdefault("option_id", 0)
    opt.setdefault("root_action", None)
    opt = _ensure_cdf(opt)
    return [opt]


def get_prob_row_policy_options_from_prob_table(
    prob_table_chunked,
    row_label: str,
    *,
    library_pkl_path=None,
):
    """Return a list of policy-option payloads for one row label."""
    payload = get_prob_row_payload_from_prob_table(
        prob_table_chunked,
        row_label,
        library_pkl_path=library_pkl_path,
    )
    return normalize_payload_to_policy_options(payload)
