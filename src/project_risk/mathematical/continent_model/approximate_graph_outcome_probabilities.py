from __future__ import annotations

import logging
import itertools
import time
from pathlib import Path
from typing import Collection, Optional, Sequence, Dict, Any, List, Set, Tuple
import networkx as nx
import pandas as pd
import numpy as np


from project_risk.infrastructure.log_config import get_logger
from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import (
    GlobalState, NodeState, is_successful, canonicalize_edges_with_roles
    )
from project_risk.mathematical.libraries.create_library import (
    get_prob_table, graph_path, BASE_LIB_DIR, load_library, 
    combos, star_combos,
    )

log = logging.getLogger("risk.query")


# ---------------------------------------------------------------------
# Owner-role normalization
# ---------------------------------------------------------------------


def _owner_identity_label(owner: Any) -> Any:
    if owner is None:
        return None
    name = getattr(owner, "_name", None)
    if name is not None:
        return str(name)
    return owner


def owner_matches_player(owner: Any, player: Any) -> bool:
    """
    Return True when an owner value is the given Player by explicit repository
    representations: object identity, object equality, or exact Player._name.
    """
    if owner is None or player is None:
        return False
    if owner is player:
        return True
    try:
        if owner == player:
            return True
    except Exception:
        pass
    owner_name = getattr(owner, "_name", None)
    player_name = getattr(player, "_name", None)
    if owner_name is not None and player_name is not None:
        return str(owner_name) == str(player_name)
    if player_name is not None and isinstance(owner, str):
        return str(owner) == str(player_name)
    return False


def normalize_owner_to_combat_role(
    owner: Any,
    *,
    attacker_player: Any = None,
    defender_player: Any = None,
    attacker_owner_values: Optional[Collection[Any]] = None,
    defender_owner_values: Optional[Collection[Any]] = None,
) -> str:
    """
    Normalize known repository owner representations to canonical combat roles.

    Returns only "A" or "D". Unknown values raise ValueError so they cannot be
    silently omitted from pattern counts.
    """
    if owner in ("A", "D"):
        return str(owner)
    if attacker_owner_values is not None and owner in attacker_owner_values:
        return "A"
    if defender_owner_values is not None and owner in defender_owner_values:
        return "D"
    if owner_matches_player(owner, attacker_player):
        return "A"
    if owner_matches_player(owner, defender_player):
        return "D"
    raise ValueError(f"Unknown owner representation for combat role: {_owner_identity_label(owner)!r}")


def extract_region_combat_roles(
    *,
    region_nodes: Sequence[int],
    global_state: GlobalState,
    attacker_player: Any = None,
    defender_player: Any = None,
) -> Dict[int, str]:
    """
    Return node -> canonical "A"/"D" roles for a region.

    GlobalState is the authoritative combat-state source for exact-library
    query/ranking paths. Player parameters are accepted for callers that pass
    player-backed owner values in small fixtures.
    """
    roles: Dict[int, str] = {}
    for n in region_nodes:
        idx = int(n)
        if idx < 0 or idx >= len(global_state.nodes):
            raise ValueError(f"Node index {idx} is outside GlobalState length {len(global_state.nodes)}")
        raw_owner = getattr(global_state.nodes[idx], "owner", None)
        roles[idx] = normalize_owner_to_combat_role(
            raw_owner,
            attacker_player=attacker_player,
            defender_player=defender_player,
        )
    if len(roles) != len(tuple(region_nodes)):
        raise ValueError(f"Region contains duplicate nodes: {tuple(region_nodes)!r}")
    return roles


def region_combat_role_summary(
    *,
    region_nodes: Sequence[int],
    global_state: GlobalState,
    attacker_player: Any = None,
    defender_player: Any = None,
) -> Dict[str, Any]:
    raw_owner_values = tuple(
        _owner_identity_label(getattr(global_state.nodes[int(n)], "owner", None))
        for n in region_nodes
    )
    try:
        role_map = extract_region_combat_roles(
            region_nodes=region_nodes,
            global_state=global_state,
            attacker_player=attacker_player,
            defender_player=defender_player,
        )
        normalized_roles = tuple(role_map[int(n)] for n in region_nodes)
        nA = sum(1 for r in normalized_roles if r == "A")
        nD = sum(1 for r in normalized_roles if r == "D")
        if nA + nD != len(tuple(region_nodes)):
            raise ValueError("owner_role_normalization_failed")
        if nA > 0 and nD > 0:
            kind = "mixed_A_D_combat_region"
        elif nA > 0:
            kind = "all_A_region"
        else:
            kind = "all_D_region"
        return {
            "raw_owner_values": raw_owner_values,
            "normalized_roles": normalized_roles,
            "role_map": role_map,
            "pattern": (nA, nD),
            "role_kind": kind,
            "normalization_error": None,
        }
    except Exception as e:
        return {
            "raw_owner_values": raw_owner_values,
            "normalized_roles": None,
            "role_map": None,
            "pattern": None,
            "role_kind": "unknown_owner_region",
            "normalization_error": f"{type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------
# Policy-option selection helpers
# ---------------------------------------------------------------------

_POLICY_OPTION_SELECTION_ALIASES = {
    None: "primary",
    "": "primary",
    "first": "primary",
    "default": "primary",
    "primary": "primary",
    "option0": "primary",
    "best": "best_local",
    "best_local": "best_local",
    "best_by_ranking": "best_local",
    "best_territories": "best_territories",
    "expected_territories": "best_territories",
    "battle_expected_attacker_territory_count": "best_territories",
    "best_troops": "best_troops",
    "expected_troops": "best_troops",
    "battle_expected_attacker_troop_count": "best_troops",
    "best_conquest": "best_conquest",
    "conquest_probability": "best_conquest",
    "battle_expected_attacker_conquest_probability": "best_conquest",
}


def normalize_policy_option_selection(selection: Any) -> str:
    """
    Normalize user/module-facing option-selection names.

    The returned value is one of:
      - "primary"          : preserve old behavior; choose option 0.
      - "best_local"       : choose by the supplied ranking_variable.
      - "best_territories" : choose max E[new territories].
      - "best_troops"      : choose max E[final attacker troops].
      - "best_conquest"    : choose max P(region conquered).
    """
    key = None if selection is None else str(selection).strip()
    return _POLICY_OPTION_SELECTION_ALIASES.get(key, str(key))


def _normalize_ranking_variable_for_policy_options(ranking_variable: Any) -> str:
    rv = str(ranking_variable or "expected_territories")
    if rv == "battle_expected_attacker_territory_count":
        return "expected_territories"
    if rv == "battle_expected_attacker_troop_count":
        return "expected_troops"
    if rv == "battle_expected_attacker_conquest_probability":
        return "conquest_probability"
    if rv in {"expected_territories", "expected_troops", "conquest_probability"}:
        return rv
    return "expected_territories"


def policy_option_metrics(option_payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute option-level scalar metrics from a V2 option payload.

    These are local-region metrics. For partition ranking, current full-graph
    totals are added later by battle_graph_ranking; therefore E[new territories]
    is the correct local proxy for expected_territories.
    """
    p = np.asarray(option_payload.get("p", []), dtype=np.float64)
    if p.size == 0:
        return {
            "expected_new_territories": 0.0,
            "expected_final_attacker_troops": 0.0,
            "conquest_probability": 0.0,
        }

    def _dot(key: str) -> float:
        arr = np.asarray(option_payload.get(key, []), dtype=np.float64)
        if arr.size != p.size:
            return 0.0
        return float(np.dot(p, arr))

    return {
        "expected_new_territories": _dot("new_territories"),
        "expected_final_attacker_troops": _dot("final_attacker_troops"),
        "conquest_probability": _dot("is_conquered"),
    }


def _policy_option_score_tuple(
    option_payload: Dict[str, Any],
    *,
    selection: str,
    ranking_variable: Any = "expected_territories",
) -> Tuple[float, float, float, float]:
    """
    Deterministic score for selecting one option from a state-set row.

    The final term is negative option_id so ties are stable and prefer earlier
    options, preserving old behavior when options are metric-equivalent.
    """
    m = policy_option_metrics(option_payload)
    sel = normalize_policy_option_selection(selection)

    if sel == "best_local":
        rv = _normalize_ranking_variable_for_policy_options(ranking_variable)
    elif sel == "best_troops":
        rv = "expected_troops"
    elif sel == "best_conquest":
        rv = "conquest_probability"
    else:
        rv = "expected_territories"

    terr = float(m["expected_new_territories"])
    troops = float(m["expected_final_attacker_troops"])
    conq = float(m["conquest_probability"])

    try:
        opt_id = int(option_payload.get("option_id", 0) or 0)
    except Exception:
        opt_id = 0

    if rv == "expected_troops":
        return (troops, terr, conq, -float(opt_id))
    if rv == "conquest_probability":
        return (conq, terr, troops, -float(opt_id))
    return (terr, conq, troops, -float(opt_id))


def select_policy_option_payload(
    policy_options: Sequence[Dict[str, Any]],
    *,
    selection: Any = "primary",
    ranking_variable: Any = "expected_territories",
) -> Tuple[Dict[str, Any], int, str]:
    """
    Select one payload from a list of V2 policy-option payloads.

    Returns (payload, index, normalized_selection). This is the compatibility
    adapter used by old single-distribution consumers. Modules that want to
    preserve all alternatives should use the full policy_options_v2 list instead.
    """
    opts = list(policy_options or [])
    sel = normalize_policy_option_selection(selection)
    if not opts:
        return {}, -1, sel
    if sel == "primary":
        return opts[0], 0, sel

    if sel not in {"best_local", "best_territories", "best_troops", "best_conquest"}:
        raise ValueError(
            f"Unknown policy_option_selection={selection!r}. Expected one of: "
            "primary, best_local, best_territories, best_troops, best_conquest."
        )

    best_i = 0
    best_key = None
    for i, opt in enumerate(opts):
        key = _policy_option_score_tuple(opt, selection=sel, ranking_variable=ranking_variable)
        if best_key is None or key > best_key:
            best_key = key
            best_i = i
    return opts[best_i], best_i, sel

# ---------------------------------------------------------------------
# Reindexing nodes/edges to match library conventions
# ---------------------------------------------------------------------


def reindex_region_nodes(
    global_state: GlobalState,
    region_nodes: Sequence[int],
) -> Tuple[Dict[int, int], Dict[int, int], List[int], List[int]]:
    """
    Given a global state and a list of node indices in the big graph (region_nodes),
    return:

      - mapping:     new_index -> old_index
      - inv_mapping: old_index -> new_index (for nodes in region)
      - attacker_new: list of new indices of attacker nodes
      - defender_new: list of new indices of defender nodes

    New indices are arranged as:
      [all attackers..., all defenders...]

    where ownership is taken from global_state (owner=='A' => attacker).
    """
    attackers: List[int] = []
    defenders: List[int] = []
    role_map = extract_region_combat_roles(region_nodes=region_nodes, global_state=global_state)
    for idx in region_nodes:
        idx_int = int(idx)
        if role_map[idx_int] == 'A':
            attackers.append(idx_int)
        else:
            defenders.append(idx_int)

    ordered_old = attackers + defenders  # this will define new index order
    mapping: Dict[int, int] = {new: old for new, old in enumerate(ordered_old)}
    inv_mapping: Dict[int, int] = {old: new for new, old in mapping.items()}

    attacker_new = list(range(len(attackers)))
    defender_new = list(range(len(attackers), len(attackers) + len(defenders)))

    return mapping, inv_mapping, attacker_new, defender_new


def reindex_edges_for_region(
    edges,
    inv_mapping: Dict[int, int],
    region_nodes_set: Set[int],
) -> Set[Tuple[int, int]]:
    """
    Given global edges and a region node set, build reindexed edges for the subgraph.

    Parameters
    ----------
    edges : iterable of (u, v)
        Edges from the big graph (global indices).
    inv_mapping : dict[int, int]
        Mapping from old (global) index -> new index for nodes in region.
    region_nodes_set : set[int]
        Set of node indices (global) belonging to this region.

    Returns
    -------
    sub_edges : set[(int, int)]
        Reindexed edges with new indices and u < v.
    """
    sub_edges: Set[Tuple[int, int]] = set()
    for u, v in edges:
        if u in region_nodes_set and v in region_nodes_set:
            u_new = inv_mapping[u]
            v_new = inv_mapping[v]
            if u_new < v_new:
                sub_edges.add((u_new, v_new))
            else:
                sub_edges.add((v_new, u_new))
    return sub_edges


def encode_state_label_from_mapping(
    global_state: GlobalState,
    mapping: Dict[int, int],
) -> str:
    """
    Encode the region's state as a row_label string like "(A3,D2,D1,...)"
    in the "new index space" defined by mapping.

    Parameters
    ----------
    global_state : GlobalState
        Full board state.
    mapping : dict[int, int]
        new_index -> old_index (for region nodes).

    Returns
    -------
    row_label : str
        Label suitable for indexing into a library prob_table.
    """
    new_nodes: List[str] = []
    # new indices 0..k-1
    for new_idx in range(len(mapping)):
        old_idx = mapping[new_idx]
        node = global_state.nodes[old_idx]
        prefix = normalize_owner_to_combat_role(node.owner)
        new_nodes.append(f"{prefix}{node.troops}")
    return "(" + ",".join(new_nodes) + ")"


def encode_library_row_label_from_mapping(
    global_state: GlobalState,
    mapping: Dict[int, int],
    *,
    num_attacker_nodes: int,
) -> str:
    """
    Encode row label in the SAME convention used by library generation:

      (A{t0},...,A{t(nA-1)},D{t(nA)},...,D{t(M-1)})

    i.e. the A/D prefix is determined by POSITION (attacker block first),
    not by global_state.nodes[old_idx].owner.

    This avoids mismatches when runtime owner encoding differs slightly,
    or when partitioning/canonicalization moves nodes around.
    """
    M = len(mapping)
    parts: list[str] = []
    for new_idx in range(M):
        old_idx = mapping[new_idx]
        node = global_state.nodes[old_idx]
        prefix = "A" if new_idx < num_attacker_nodes else "D"
        parts.append(f"{prefix}{int(node.troops)}")
    return "(" + ",".join(parts) + ")"



# ---------------------------------------------------------------------
# Library coverage: patterns + troop caps, with STAR-ONLY special cases
# ---------------------------------------------------------------------


# Patterns for which we have "full topology" libraries (all canonical graphs)
REGULAR_PATTERNS: set[Tuple[int, int]] = {
    (nA, nD) for nA, nD, maxA_exact, maxD_exact, maxA_ext, maxD_ext in combos
}

# Patterns for which we ONLY have star-topology libraries (centered at new index 0)
STAR_ONLY_PATTERNS: set[Tuple[int, int]] = {
    (nA, nD) for nA, nD, maxA_exact, maxD_exact, maxA_ext, maxD_ext in star_combos
}

# Allowed patterns overall (partitioner must additionally enforce star-topology
# when (nA,nD) is in STAR_ONLY_PATTERNS)
ALLOWED_PATTERNS: set[Tuple[int, int]] = REGULAR_PATTERNS | STAR_ONLY_PATTERNS


# Max troop caps used when building the libraries for each pattern
REG_PATTERN_MAX_TROOPS: Dict[Tuple[int, int], Tuple[int, int]] = {
    (nA, nD): (maxA_ext, maxD_ext)
    for nA, nD, maxA_exact, maxD_exact, maxA_ext, maxD_ext in combos
}

STAR_PATTERN_MAX_TROOPS: Dict[Tuple[int, int], Tuple[int, int]] = {
    (nA, nD): (maxA_ext, maxD_ext)
    for nA, nD, maxA_exact, maxD_exact, maxA_ext, maxD_ext in star_combos
}

# For each (nA, nD) pattern, the (maxA, maxD) caps the per-graph libraries were built with.
# NOTE: if a pattern appears in both (shouldn't), STAR overrides REGULAR.
PATTERN_MAX_TROOPS: Dict[Tuple[int, int], Tuple[int, int]] = dict(REG_PATTERN_MAX_TROOPS)
PATTERN_MAX_TROOPS.update(STAR_PATTERN_MAX_TROOPS)


def _available_library_caps_for_pattern(
    combat_libraries_base: Path,
    pattern: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Return available A/D cap folders for a pattern under a library base."""
    pattern_dir = Path(combat_libraries_base) / f"{pattern[0]}A_{pattern[1]}D"
    if not pattern_dir.exists():
        return []

    out: List[Tuple[int, int]] = []
    for child in pattern_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not (name.startswith("A") and "_D" in name):
            continue
        try:
            a_part, d_part = name[1:].split("_D", 1)
            out.append((int(a_part), int(d_part)))
        except Exception:
            continue
    return sorted(set(out))


def _required_troop_caps_for_mapping(
    global_state: GlobalState,
    mapping: Dict[int, int],
    *,
    num_attacker_nodes: int,
) -> Tuple[int, int]:
    max_a = 0
    max_d = 0
    for local_idx, global_idx in mapping.items():
        troops = int(global_state.nodes[global_idx].troops)
        if int(local_idx) < int(num_attacker_nodes):
            max_a = max(max_a, troops)
        else:
            max_d = max(max_d, troops)
    return max_a, max_d


def _select_library_caps_for_pattern(
    combat_libraries_base: Path,
    pattern: Tuple[int, int],
    *,
    required_attacker_troops: int,
    required_defender_troops: int,
) -> Tuple[int, int]:
    """
    Prefer the configured production cap, but allow narrow test library bases
    to provide a smaller cap folder when the queried row fits inside it.
    """
    configured = PATTERN_MAX_TROOPS[pattern]
    configured_dir = (
        Path(combat_libraries_base)
        / f"{pattern[0]}A_{pattern[1]}D"
        / f"A{configured[0]}_D{configured[1]}"
    )
    if configured_dir.exists():
        return configured

    for max_a, max_d in _available_library_caps_for_pattern(combat_libraries_base, pattern):
        if max_a >= int(required_attacker_troops) and max_d >= int(required_defender_troops):
            return (max_a, max_d)

    return configured


# ---------------------------------------------------------------------
# Library query for a given region (per-graph files)
# ---------------------------------------------------------------------

def _is_coverage_failure(e: Exception) -> bool:
    """
    Return True if this exception should be treated as a library coverage issue,
    meaning: try subpartitioning or skip candidate rather than crash training.

    This includes:
      - missing per-graph library file
      - missing chunk files
      - missing row label in either dataframe or chunked store
      - malformed chunk descriptor / manifest
      - chunk read/unpickle errors

    We keep this intentionally conservative: only known library/IO/index failures.
    """
    import pickle

    if isinstance(e, FileNotFoundError):
        return True

    # Chunk files / pickle read issues
    if isinstance(e, (EOFError, OSError, pickle.UnpicklingError)):
        return True

    msg = str(e)

    # Existing coverage messages
    if "No library found for nA=" in msg:
        return True
    if "not found in prob_table index" in msg:
        return True
    if "Row label" in msg and "not found" in msg:
        return True

    # Chunked descriptor / manifest issues that should behave like "coverage"
    if "chunked" in msg and "not found" in msg:
        return True
    if "chunk" in msg and ("missing" in msg or "not found" in msg):
        return True
    if "row_to_chunk" in msg or "chunks" in msg or "_base_path" in msg:
        return True
    if "manifest" in msg:
        return True

    # KeyErrors from missing expected keys in chunk descriptors
    if isinstance(e, KeyError):
        # Most of these happen when a lib/chunk descriptor is incomplete
        return True

    return False



def is_star_edges(edges: set[tuple[int,int]], center: int, n_nodes: int) -> bool:
    """
    True iff edges are exactly a star centered at `center`:
      - center connected to all other nodes
      - no edges among non-center nodes
    """
    expected = set()
    for i in range(n_nodes):
        if i == center:
            continue
        u, v = (center, i) if center < i else (i, center)
        expected.add((u, v))
    return edges == expected




class RegionQueryResultCache:
    """Call-scoped cache for exact regional library queries."""

    def __init__(
        self,
        max_entries: Optional[int] = None,
        *,
        profile_timings: bool = False,
        cache_library_resources: bool = True,
        max_library_tables: int = 64,
        max_library_chunks: int = 24,
        max_library_chunk_source_bytes: int = 192 * 1024 * 1024,
        max_library_rows: int = 4096,
        max_library_row_array_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.max_entries = None if max_entries is None else max(0, int(max_entries))
        self.profile_timings = bool(profile_timings)
        self.cache_library_resources = bool(cache_library_resources)
        self.max_library_tables = max(0, int(max_library_tables))
        self._prob_table_entries: Dict[Tuple[Any, ...], Any] = {}
        self._prob_table_order: List[Tuple[Any, ...]] = []
        self.prob_table_hits = 0
        self.prob_table_misses = 0
        self.prob_table_stores = 0
        self.prob_table_evictions = 0
        self.library_chunk_cache = None
        self.max_library_rows = max(0, int(max_library_rows))
        self.max_library_row_array_bytes = max(0, int(max_library_row_array_bytes))
        self._row_payload_entries: Dict[Tuple[Any, ...], Any] = {}
        self._row_payload_order: List[Tuple[Any, ...]] = []
        self._row_payload_array_bytes: Dict[Tuple[Any, ...], int] = {}
        self._row_payload_seen_keys: Set[Tuple[Any, ...]] = set()
        self.row_payload_hits = 0
        self.row_payload_misses = 0
        self.row_payload_stores = 0
        self.row_payload_evictions = 0
        self.peak_row_payload_entries = 0
        self.peak_row_payload_array_bytes = 0
        if self.cache_library_resources:
            from project_risk.mathematical.libraries.library_io import ChunkCache

            self.library_chunk_cache = ChunkCache(
                max_chunks=max_library_chunks,
                max_source_bytes=max_library_chunk_source_bytes,
            )
        self._entries: Dict[Tuple[Any, ...], Tuple[bool, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.failure_hits = 0
        self.stores = 0
        self.skipped_stores = 0
        self.total_request_seconds = 0.0
        self.hit_request_seconds = 0.0
        self.miss_request_seconds = 0.0

    def lookup(self, key: Tuple[Any, ...]) -> Tuple[bool, Optional[Tuple[bool, Any]]]:
        if key in self._entries:
            self.hits += 1
            entry = self._entries[key]
            if not entry[0]:
                self.failure_hits += 1
            return True, entry
        self.misses += 1
        return False, None

    def store(self, key: Tuple[Any, ...], entry: Tuple[bool, Any]) -> None:
        if key in self._entries:
            self._entries[key] = entry
            return
        if self.max_entries is not None and len(self._entries) >= self.max_entries:
            self.skipped_stores += 1
            return
        self._entries[key] = entry
        self.stores += 1

    def __len__(self) -> int:
        return len(self._entries)

    def lookup_prob_table(self, key: Tuple[Any, ...]) -> Tuple[bool, Any]:
        if not self.cache_library_resources:
            return False, None
        if key not in self._prob_table_entries:
            self.prob_table_misses += 1
            return False, None
        self.prob_table_hits += 1
        try:
            self._prob_table_order.remove(key)
        except ValueError:
            pass
        self._prob_table_order.append(key)
        return True, self._prob_table_entries[key]

    def store_prob_table(self, key: Tuple[Any, ...], value: Any) -> None:
        if not self.cache_library_resources or self.max_library_tables <= 0:
            return
        if key in self._prob_table_entries:
            self._prob_table_entries[key] = value
            return
        self._prob_table_entries[key] = value
        self._prob_table_order.append(key)
        self.prob_table_stores += 1
        while len(self._prob_table_order) > self.max_library_tables:
            old = self._prob_table_order.pop(0)
            self._prob_table_entries.pop(old, None)
            self.prob_table_evictions += 1

    @staticmethod
    def _payload_array_nbytes(value: Any, seen: Optional[Set[int]] = None) -> int:
        seen_ids = set() if seen is None else seen
        value_id = id(value)
        if value_id in seen_ids:
            return 0
        seen_ids.add(value_id)
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(
                RegionQueryResultCache._payload_array_nbytes(item, seen_ids)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return sum(
                RegionQueryResultCache._payload_array_nbytes(item, seen_ids)
                for item in value
            )
        return 0

    def lookup_row_payload(self, key: Tuple[Any, ...]) -> Tuple[bool, Any]:
        if not self.cache_library_resources:
            return False, None
        self._row_payload_seen_keys.add(key)
        if key not in self._row_payload_entries:
            self.row_payload_misses += 1
            return False, None
        self.row_payload_hits += 1
        try:
            self._row_payload_order.remove(key)
        except ValueError:
            pass
        self._row_payload_order.append(key)
        return True, self._row_payload_entries[key]

    def store_row_payload(self, key: Tuple[Any, ...], value: Any) -> None:
        if not self.cache_library_resources or self.max_library_rows <= 0:
            return
        if key in self._row_payload_entries:
            self._row_payload_entries[key] = value
            return
        self._row_payload_entries[key] = value
        self._row_payload_order.append(key)
        self._row_payload_array_bytes[key] = self._payload_array_nbytes(value)
        self.row_payload_stores += 1
        while self._row_payload_order and (
            len(self._row_payload_order) > self.max_library_rows
            or sum(self._row_payload_array_bytes.values()) > self.max_library_row_array_bytes
        ):
            old = self._row_payload_order.pop(0)
            self._row_payload_entries.pop(old, None)
            self._row_payload_array_bytes.pop(old, None)
            self.row_payload_evictions += 1
        self.peak_row_payload_entries = max(
            self.peak_row_payload_entries,
            len(self._row_payload_entries),
        )
        self.peak_row_payload_array_bytes = max(
            self.peak_row_payload_array_bytes,
            sum(self._row_payload_array_bytes.values()),
        )

    def record_request(self, *, hit: bool, elapsed_seconds: float) -> None:
        if not self.profile_timings:
            return
        elapsed = float(elapsed_seconds)
        self.total_request_seconds += elapsed
        if hit:
            self.hit_request_seconds += elapsed
        else:
            self.miss_request_seconds += elapsed

    def diagnostics(self) -> Dict[str, Any]:
        chunk_diagnostics = (
            self.library_chunk_cache.diagnostics()
            if self.library_chunk_cache is not None
            else {
                "entries": 0,
                "max_chunks": 0,
                "max_source_bytes": 0,
                "hits": 0,
                "misses": 0,
                "stores": 0,
                "evictions": 0,
                "cached_source_bytes": 0,
                "unique_chunk_keys_seen": 0,
                "peak_entries": 0,
                "peak_cached_source_bytes": 0,
            }
        )
        return {
            "entries": int(len(self._entries)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "failure_hits": int(self.failure_hits),
            "stores": int(self.stores),
            "skipped_stores": int(self.skipped_stores),
            "max_entries": self.max_entries,
            "profile_timings": bool(self.profile_timings),
            "requests": int(self.hits + self.misses),
            "total_request_seconds": float(self.total_request_seconds),
            "hit_request_seconds": float(self.hit_request_seconds),
            "miss_request_seconds": float(self.miss_request_seconds),
            "library_resource_reuse_enabled": bool(self.cache_library_resources),
            "library_table_entries": int(len(self._prob_table_entries)),
            "library_table_max_entries": int(self.max_library_tables),
            "library_table_hits": int(self.prob_table_hits),
            "library_table_misses": int(self.prob_table_misses),
            "library_table_stores": int(self.prob_table_stores),
            "library_table_evictions": int(self.prob_table_evictions),
            "library_row_entries": int(len(self._row_payload_entries)),
            "library_row_max_entries": int(self.max_library_rows),
            "library_row_max_array_bytes": int(self.max_library_row_array_bytes),
            "library_row_hits": int(self.row_payload_hits),
            "library_row_misses": int(self.row_payload_misses),
            "library_row_stores": int(self.row_payload_stores),
            "library_row_evictions": int(self.row_payload_evictions),
            "library_row_unique_keys_seen": int(len(self._row_payload_seen_keys)),
            "library_row_cached_array_bytes": int(sum(self._row_payload_array_bytes.values())),
            "library_row_peak_entries": int(self.peak_row_payload_entries),
            "library_row_peak_array_bytes": int(self.peak_row_payload_array_bytes),
            "library_chunk_cache": chunk_diagnostics,
        }


def canonical_region_query_cache_key(
    *,
    combat_libraries_base: Path,
    global_state: GlobalState,
    global_edges,
    region_nodes: Sequence[int],
    policy_option_selection: Any,
    ranking_variable: Any,
) -> Tuple[Any, ...]:
    """Identify every input that can change one regional library result."""
    nodes = tuple(sorted({int(node) for node in region_nodes}))
    node_set = set(nodes)
    induced_edges = tuple(
        sorted(
            {
                tuple(sorted((int(u), int(v))))
                for u, v in global_edges
                if int(u) in node_set and int(v) in node_set
            }
        )
    )
    local_state = tuple(
        (node, str(global_state.nodes[node].owner), int(global_state.nodes[node].troops))
        for node in nodes
    )
    return (
        "region_query_v1",
        str(Path(combat_libraries_base)),
        nodes,
        induced_edges,
        local_state,
        normalize_policy_option_selection(policy_option_selection),
        str(ranking_variable),
    )


def _query_region_from_libraries_uncached(
    combat_libraries_base: Path,
    global_state: GlobalState,
    global_edges,
    region_nodes: Sequence[int],
    *,
    debug: bool = True,
    debug_limit: int = 30,
    policy_option_selection: Any = "primary",
    ranking_variable: Any = "expected_territories",
    resource_cache: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Fully patched (STAR_ONLY authority lives here).

    What this function guarantees now:
      - Region is non-empty, has internal edges, and after isolate pruning is still supported.
      - STAR_ONLY patterns are correctly gated WITHOUT assuming center==0.
      - Library table existence + row existence diagnostics remain intact.
      - Returns a dict containing 'payload' (v2) + mapping + metadata.
    """
    from project_risk.mathematical.libraries.library_io import get_prob_row_payload_from_prob_table, normalize_payload_to_policy_options
    from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import parse_row_label

    # --------------------------------------------------------------
    # spam guard
    # --------------------------------------------------------------
    if not hasattr(query_region_from_libraries, "_dbg_count"):
        query_region_from_libraries._dbg_count = 0  # type: ignore[attr-defined]

    do_dbg = False
    if debug and query_region_from_libraries._dbg_count < int(debug_limit):  # type: ignore[attr-defined]
        query_region_from_libraries._dbg_count += 1  # type: ignore[attr-defined]
        do_dbg = True

    DEBUG_REGION_PRUNE = True

    region_nodes = list(region_nodes)
    region_nodes_set: Set[int] = set(region_nodes)

    def _prune_isolates_in_preindexed_space(
        *,
        sub_edges_set: set[tuple[int, int]],
        n_nodes: int,
        mapping_pre: Dict[int, int],
    ) -> tuple[list[int], list[int]]:
        deg = [0] * n_nodes
        for u, v in sub_edges_set:
            deg[u] += 1
            deg[v] += 1

        kept_pre = [i for i, d in enumerate(deg) if d > 0]
        removed_pre = [i for i, d in enumerate(deg) if d == 0]

        if not removed_pre:
            return list(mapping_pre.values()), []

        kept_global = [mapping_pre[i] for i in kept_pre]
        return kept_global, removed_pre

    def _row_key_exists(prob_table_obj: Any, rl: str) -> bool:
        if not isinstance(prob_table_obj, dict):
            return False

        exact_df = prob_table_obj.get("exact_df")
        try:
            if exact_df is not None and hasattr(exact_df, "index") and rl in exact_df.index:
                return True
        except Exception:
            pass

        rtc = prob_table_obj.get("row_to_chunk")
        if isinstance(rtc, dict) and rl in rtc:
            return True

        return False

    def _row_key_location(prob_table_obj: Any, rl: str) -> str:
        if not isinstance(prob_table_obj, dict):
            return "unknown_prob_table_type"
        in_exact = False
        in_rtc = False
        try:
            exact_df = prob_table_obj.get("exact_df")
            in_exact = bool(exact_df is not None and hasattr(exact_df, "index") and rl in exact_df.index)
        except Exception:
            in_exact = False
        try:
            rtc = prob_table_obj.get("row_to_chunk")
            in_rtc = bool(isinstance(rtc, dict) and rl in rtc)
        except Exception:
            in_rtc = False
        if in_exact and in_rtc:
            return "exact_df+row_to_chunk"
        if in_exact:
            return "exact_df"
        if in_rtc:
            return "row_to_chunk"
        return "missing"

    def _generate_row_label_candidates(
        row_label: str,
        *,
        num_attacker_nodes_local: int,
        allow_attacker_zero: bool = False,
    ) -> List[str]:
        out: List[str] = []
        rl0 = str(row_label)
        out.append(rl0)

        try:
            owners, troops = parse_row_label(rl0)
        except Exception:
            return out

        if not owners or len(owners) != len(troops):
            return out

        troops2 = list(troops)
        for i in range(min(num_attacker_nodes_local, len(troops2))):
            if owners[i] == "A":
                t = int(troops2[i])
                t2 = t - 1
                if not allow_attacker_zero and t2 < 1:
                    t2 = 1
                troops2[i] = t2

        parts = [f"{owners[i]}{int(troops2[i])}" for i in range(len(owners))]
        rl_avail = "(" + ",".join(parts) + ")"
        if rl_avail != rl0:
            out.append(rl_avail)

        return out

    # ============================================================
    # 0) Debug header
    # ============================================================
    if do_dbg:
        rn = len(region_nodes)
        log.debug(f"[agop.query] base={combat_libraries_base} region_nodes(n={rn})={sorted(region_nodes)}")

    # ============================================================
    # 1) Initial reindex (A first, D last)
    # ============================================================
    mapping, inv_mapping, attacker_new, defender_new = reindex_region_nodes(global_state, region_nodes)

    num_attacker_nodes = len(attacker_new)
    num_defender_nodes = len(defender_new)
    pattern_before = (num_attacker_nodes, num_defender_nodes)

    if do_dbg:
        log.debug(f"[agop.query] pattern_before={pattern_before}")

    if pattern_before not in PATTERN_MAX_TROOPS:
        if do_dbg:
            log.debug(f"[agop.query] FAIL: pattern_before not supported by PATTERN_MAX_TROOPS: {pattern_before}")
        raise ValueError(f"No library found for nA={num_attacker_nodes}, nD={num_defender_nodes}")

    reqA_lib, reqD_lib = _required_troop_caps_for_mapping(
        global_state,
        mapping,
        num_attacker_nodes=num_attacker_nodes,
    )
    maxA_lib, maxD_lib = _select_library_caps_for_pattern(
        combat_libraries_base,
        pattern_before,
        required_attacker_troops=reqA_lib,
        required_defender_troops=reqD_lib,
    )

    # ============================================================
    # 2) Build region edges (pre-canonical index space)
    # ============================================================
    sub_edges_set = reindex_edges_for_region(global_edges, inv_mapping, region_nodes_set)

    if do_dbg:
        try:
            ne = len(sub_edges_set)
        except Exception:
            ne = -1
        log.debug(f"[agop.query] pre-canonical sub_edges_set size={ne}")

    # Fail early if no internal edges
    if not sub_edges_set:
        if do_dbg:
            log.debug("[agop.query] FAIL: no sub_edges_set from reindex_edges_for_region()")
        raise ValueError("Region has no internal edges (no sub_edges_set).")

    # ============================================================
    # 2b) prune isolates
    # ============================================================
    pruned_region_nodes, removed_pre = _prune_isolates_in_preindexed_space(
        sub_edges_set=sub_edges_set,
        n_nodes=num_attacker_nodes + num_defender_nodes,
        mapping_pre=mapping,
    )

    if removed_pre:
        if DEBUG_REGION_PRUNE:
            removed_global = [mapping[i] for i in removed_pre]
            log.debug("\n[REGION PRUNE] Disconnected region detected")
            log.debug(f"  original region_nodes (global): {sorted(region_nodes)}")
            log.debug(f"  pre-index isolates removed     : {removed_pre}")
            log.debug(f"  removed nodes (global)         : {removed_global}")

        region_nodes = pruned_region_nodes
        region_nodes_set = set(region_nodes)

        mapping, inv_mapping, attacker_new, defender_new = reindex_region_nodes(global_state, region_nodes)
        num_attacker_nodes = len(attacker_new)
        num_defender_nodes = len(defender_new)
        pattern_after = (num_attacker_nodes, num_defender_nodes)

        if DEBUG_REGION_PRUNE:
            log.debug(f"  pruned region_nodes (global)   : {sorted(region_nodes)}")
            log.debug(
                f"  pattern changed                : "
                f"({pattern_before[0]}A,{pattern_before[1]}D) -> "
                f"({pattern_after[0]}A,{pattern_after[1]}D)"
            )

        if pattern_after not in PATTERN_MAX_TROOPS:
            if do_dbg:
                log.debug(f"[agop.query] FAIL: pattern_after not supported: {pattern_after}")
            raise ValueError(
                f"After pruning isolates, no library pattern for nA={num_attacker_nodes}, nD={num_defender_nodes}"
            )

        reqA_lib, reqD_lib = _required_troop_caps_for_mapping(
            global_state,
            mapping,
            num_attacker_nodes=num_attacker_nodes,
        )
        maxA_lib, maxD_lib = _select_library_caps_for_pattern(
            combat_libraries_base,
            pattern_after,
            required_attacker_troops=reqA_lib,
            required_defender_troops=reqD_lib,
        )
        sub_edges_set = reindex_edges_for_region(global_edges, inv_mapping, region_nodes_set)

    if not sub_edges_set:
        if do_dbg:
            log.debug("[agop.query] FAIL: Region has no internal edges after pruning.")
        raise ValueError("Region has no internal edges after pruning.")

    # ============================================================
    # 2c) battle-validity sanity: must contain at least one A–D edge
    # ============================================================
    # This is cheap and protects against accidental A–A/D–D edge pollution upstream.
    if num_attacker_nodes > 0 and num_defender_nodes > 0:
        has_AD = any(
            (u < num_attacker_nodes) != (v < num_attacker_nodes)  # one in A block, one in D block
            for (u, v) in sub_edges_set
        )
        if not has_AD:
            if do_dbg:
                log.debug(f"[agop.query] FAIL: no A–D frontier edge in region (pattern={num_attacker_nodes, num_defender_nodes})")
            raise ValueError("Region has no A–D frontier edge (not a battle-valid region).")

    # ============================================================
    # 3) STAR-ONLY topology gating (pre-canonical) — robust center
    # ============================================================
    pattern_now = (num_attacker_nodes, num_defender_nodes)
    if pattern_now in STAR_ONLY_PATTERNS:
        n_nodes = num_attacker_nodes + num_defender_nodes

        # detect unique star center by degree==n_nodes-1
        deg_local = [0] * n_nodes
        for u, v in sub_edges_set:
            if 0 <= u < n_nodes:
                deg_local[u] += 1
            if 0 <= v < n_nodes:
                deg_local[v] += 1
        centers = [i for i, d in enumerate(deg_local) if d == (n_nodes - 1)]
        ok_star = (len(centers) == 1) and bool(is_star_edges(sub_edges_set, center=centers[0], n_nodes=n_nodes))

        if do_dbg:
            cshow = centers[0] if centers else None
            log.debug(f"[agop.query] STAR_ONLY gating: pattern={pattern_now} ok_star={int(ok_star)} center={cshow} n_nodes={n_nodes}")

        if not ok_star:
            if do_dbg:
                log.debug(f"[agop.query] FAIL: STAR_ONLY topology mismatch for pattern={pattern_now}")
            raise ValueError(f"STAR_ONLY pattern {pattern_now} requires a star; region topology is not a star.")

    # ============================================================
    # 4) Canonicalize topology
    # ============================================================
    canonical_edges_key, perm_old_to_new, perm_new_to_old = canonicalize_edges_with_roles(
        edges=sorted(sub_edges_set),
        num_attacker_nodes=num_attacker_nodes,
        num_defender_nodes=num_defender_nodes,
    )
    sub_edges = canonical_edges_key

    # Update mapping to canonical order: canonical local idx -> global idx
    mapping_canonical: Dict[int, int] = {}
    for canonical_idx, pre_idx in enumerate(perm_new_to_old):
        mapping_canonical[canonical_idx] = mapping[pre_idx]
    mapping = mapping_canonical

    if do_dbg:
        n_edges = len(sub_edges) if hasattr(sub_edges, "__len__") else -1
        log.debug(f"[agop.query] canonical pattern={pattern_now} canonical_edges_len={n_edges}")

    # ============================================================
    # 5) Load probability table
    # ============================================================
    table_cache_key = (
        str(Path(combat_libraries_base)),
        tuple(sub_edges),
        int(num_attacker_nodes),
        int(num_defender_nodes),
        int(maxA_lib),
        int(maxD_lib),
    )
    table_found = False
    prob_table = None
    lookup_prob_table = getattr(resource_cache, "lookup_prob_table", None)
    if callable(lookup_prob_table):
        table_found, prob_table = lookup_prob_table(table_cache_key)

    try:
        if not table_found:
            prob_table = get_prob_table(
                edges=sub_edges,
                num_attacker_nodes=num_attacker_nodes,
                num_defender_nodes=num_defender_nodes,
                max_attacker_troops=maxA_lib,
                max_defender_troops=maxD_lib,
                base_dir=combat_libraries_base,
                lazy_build=False,
                include_policies=False,
            )
            store_prob_table = getattr(resource_cache, "store_prob_table", None)
            if callable(store_prob_table):
                store_prob_table(table_cache_key, prob_table)
    except FileNotFoundError as e:
        if do_dbg:
            log.debug(
                f"[agop.query] FAIL: get_prob_table FileNotFoundError "
                f"pattern={pattern_now} maxA_lib={maxA_lib} maxD_lib={maxD_lib} edges={sub_edges} err={e}"
            )
        raise

    if do_dbg:
        if isinstance(prob_table, dict):
            keys = sorted(list(prob_table.keys()))
            log.debug(f"[agop.query] prob_table type=dict keys={keys[:20]}{'...' if len(keys) > 20 else ''}")
        else:
            log.debug(f"[agop.query] prob_table type={type(prob_table).__name__}")

    # ============================================================
    # 6) Encode state -> row_label
    # ============================================================
    row_label = encode_library_row_label_from_mapping(
        global_state,
        mapping,
        num_attacker_nodes=num_attacker_nodes,
    )
    if do_dbg:
        log.debug(f"[agop.query] row_label={row_label}")

    # ============================================================
    # 7) Fetch row payload with label-resolution fallback
    # ============================================================
    chunk_cache = getattr(resource_cache, "library_chunk_cache", None)
    row_cache_key = table_cache_key + (str(row_label),)
    lookup_row_payload = getattr(resource_cache, "lookup_row_payload", None)
    row_found = False
    payload = None
    if callable(lookup_row_payload):
        row_found, payload = lookup_row_payload(row_cache_key)
    if not row_found:
        payload = get_prob_row_payload_from_prob_table(
            prob_table,
            row_label,
            chunk_cache=chunk_cache,
        )
        store_row_payload = getattr(resource_cache, "store_row_payload", None)
        if callable(store_row_payload):
            store_row_payload(row_cache_key, payload)

    if payload is None:
        candidates = _generate_row_label_candidates(
            row_label,
            num_attacker_nodes_local=num_attacker_nodes,
            allow_attacker_zero=False,
        )

        if do_dbg:
            log.debug(f"[agop.query] row_label MISS. candidates={candidates}")
            for cand in candidates:
                loc = _row_key_location(prob_table, cand)
                log.debug(f"[agop.query]   exists? cand={cand} location={loc}")

        for alt in candidates[1:]:
            if not _row_key_exists(prob_table, alt):
                continue
            alt_cache_key = table_cache_key + (str(alt),)
            alt_found = False
            payload_alt = None
            if callable(lookup_row_payload):
                alt_found, payload_alt = lookup_row_payload(alt_cache_key)
            if not alt_found:
                payload_alt = get_prob_row_payload_from_prob_table(
                    prob_table,
                    alt,
                    chunk_cache=chunk_cache,
                )
                store_row_payload = getattr(resource_cache, "store_row_payload", None)
                if callable(store_row_payload):
                    store_row_payload(alt_cache_key, payload_alt)
            if payload_alt is not None:
                row_label = alt
                payload = payload_alt
                if do_dbg:
                    log.debug(f"[agop.query] row_label RESOLVED via alt={alt}")
                break

    if payload is None:
        if do_dbg:
            log.debug(f"[agop.query] FAIL: Row label not found after candidates. row_label={row_label} edges={sub_edges}")
        raise ValueError(f"Row label {row_label} not found in library for edges={sub_edges}")

    # 0.0 sentinel => {}
    if payload == {}:
        if do_dbg:
            log.debug(f"[agop.query] payload=0.0 sentinel (empty). row_label={row_label}")
        return {
            "probabilities": {},
            "payload": {},
            "outcomes_v2": None,
            "policy_options_v2": [],
            "selected_policy_option_index": -1,
            "selected_policy_option_selection": normalize_policy_option_selection(policy_option_selection),
            "policy_option_count": 0,
            "policy": None,
            "library_path": None,
            "graph_edges_reindexed": sub_edges,
            "row_label": row_label,
            "mapping": mapping,
            "region_nodes_effective": tuple(region_nodes),
            "pattern": (num_attacker_nodes, num_defender_nodes),
        }

    # Legacy hard-switch guard
    if "_legacy_prob_row" in payload:
        raise NotImplementedError(
            "Hard switch to v2 chunked storage: legacy probability rows are not supported. "
            "Rebuild the relevant libraries in v2 format."
        )

    policy_options = normalize_payload_to_policy_options(payload)
    primary_payload, selected_policy_option_index, selected_policy_option_selection = select_policy_option_payload(
        policy_options,
        selection=policy_option_selection,
        ranking_variable=ranking_variable,
    )

    required = ("p", "is_conquered", "new_territories", "final_attacker_troops")
    for opt_i, opt in enumerate(policy_options):
        missing = [k for k in required if k not in opt]
        if missing:
            if do_dbg:
                log.debug(
                    f"[agop.query] FAIL: V2 policy option {opt_i} missing keys={missing}. "
                    f"row_label={row_label} keys={list(opt.keys())}"
                )
            raise ValueError(
                f"V2 policy option {opt_i} for row_label={row_label} missing required keys {missing}. "
                f"Got keys={list(opt.keys())}"
            )

    if do_dbg:
        try:
            import numpy as np
            p = np.asarray(primary_payload.get("p", []), dtype=np.float64)
            psum = float(np.sum(p)) if p.size else 0.0
            pmax = float(np.max(p)) if p.size else 0.0
            log.debug(
                f"[agop.query] V2 payload OK: options={len(policy_options)} "
                f"selected={selected_policy_option_selection}[{selected_policy_option_index}] "
                f"primary_p_size={int(p.size)} p_sum={psum:.6f} p_max={pmax:.6f}"
            )
        except Exception:
            log.debug("[agop.query] V2 payload OK (stats failed)")

    primary_outcomes = {
        "p": primary_payload["p"],
        "is_conquered": primary_payload["is_conquered"],
        "new_territories": primary_payload["new_territories"],
        "final_attacker_troops": primary_payload["final_attacker_troops"],
        "owners": primary_payload.get("owners"),
        "troops": primary_payload.get("troops"),
        "cdf": primary_payload.get("cdf"),
        "option_id": primary_payload.get("option_id", 0),
        "root_action": primary_payload.get("root_action"),
        "local_value": primary_payload.get("local_value"),
        "split_metadata": primary_payload.get("split_metadata"),
    }

    policy_option_summaries = [
        {
            "option_id": opt.get("option_id", i),
            "root_action": opt.get("root_action"),
            "local_value": opt.get("local_value"),
            "split_metadata": opt.get("split_metadata"),
        }
        for i, opt in enumerate(policy_options)
    ]

    return {
        "probabilities": {},
        "payload": primary_outcomes,
        "outcomes_v2": primary_outcomes,
        "policy_options_v2": policy_options,
        "policy_option_summaries": policy_option_summaries,
        "selected_policy_option_index": selected_policy_option_index,
        "selected_policy_option_selection": selected_policy_option_selection,
        "selected_policy_option_split_metadata": primary_payload.get("split_metadata"),
        "policy_option_count": len(policy_options),
        "policy": None,
        "library_path": None,
        "graph_edges_reindexed": sub_edges,
        "row_label": row_label,
        "mapping": mapping,
        "region_nodes_effective": tuple(region_nodes),
        "pattern": (num_attacker_nodes, num_defender_nodes),
        "key": None,
        "path": None,
    }


def query_region_from_libraries(
    combat_libraries_base: Path,
    global_state: GlobalState,
    global_edges,
    region_nodes: Sequence[int],
    *,
    debug: bool = True,
    debug_limit: int = 30,
    policy_option_selection: Any = "primary",
    ranking_variable: Any = "expected_territories",
    query_cache: Optional[Any] = None,
) -> Dict[str, Any]:
    """Query one region, optionally reusing an exact call-scoped result."""
    if query_cache is None:
        return _query_region_from_libraries_uncached(
            combat_libraries_base=combat_libraries_base,
            global_state=global_state,
            global_edges=global_edges,
            region_nodes=region_nodes,
            debug=debug,
            debug_limit=debug_limit,
            policy_option_selection=policy_option_selection,
            ranking_variable=ranking_variable,
        )

    request_started = (
        time.perf_counter()
        if bool(getattr(query_cache, "profile_timings", False))
        else None
    )
    edges = tuple((int(u), int(v)) for u, v in global_edges)
    key = canonical_region_query_cache_key(
        combat_libraries_base=combat_libraries_base,
        global_state=global_state,
        global_edges=edges,
        region_nodes=region_nodes,
        policy_option_selection=policy_option_selection,
        ranking_variable=ranking_variable,
    )
    if hasattr(query_cache, "lookup"):
        found, entry = query_cache.lookup(key)
    else:
        found = key in query_cache
        entry = query_cache.get(key) if found else None
    if found and entry is not None:
        ok, value = entry
        if ok:
            if request_started is not None and hasattr(query_cache, "record_request"):
                query_cache.record_request(
                    hit=True,
                    elapsed_seconds=time.perf_counter() - request_started,
                )
            return dict(value)
        exc_type, exc_args = value
        if request_started is not None and hasattr(query_cache, "record_request"):
            query_cache.record_request(
                hit=True,
                elapsed_seconds=time.perf_counter() - request_started,
            )
        raise exc_type(*exc_args)

    try:
        result = _query_region_from_libraries_uncached(
            combat_libraries_base=combat_libraries_base,
            global_state=global_state,
            global_edges=edges,
            region_nodes=region_nodes,
            debug=debug,
            debug_limit=debug_limit,
            policy_option_selection=policy_option_selection,
            ranking_variable=ranking_variable,
            resource_cache=query_cache,
        )
    except Exception as exc:
        if _is_query_viability_failure(exc):
            entry = (False, (exc.__class__, tuple(exc.args)))
            if hasattr(query_cache, "store"):
                query_cache.store(key, entry)
            else:
                query_cache[key] = entry
        if request_started is not None and hasattr(query_cache, "record_request"):
            query_cache.record_request(
                hit=False,
                elapsed_seconds=time.perf_counter() - request_started,
            )
        raise

    entry = (True, dict(result))
    if hasattr(query_cache, "store"):
        query_cache.store(key, entry)
    else:
        query_cache[key] = entry
    if request_started is not None and hasattr(query_cache, "record_request"):
        query_cache.record_request(
            hit=False,
            elapsed_seconds=time.perf_counter() - request_started,
        )
    return result








# ---------------------------------------------------------------------
# Utilities for debugging / testing
# ---------------------------------------------------------------------


def global_state_from_row_label(row_label: str) -> GlobalState:
    """
    Turn a label like '(A3,D2,D1)' into a GlobalState
    with node 0 = A3, node 1 = D2, node 2 = D1, ...
    """
    assert row_label.startswith("(") and row_label.endswith(")")
    inner = row_label[1:-1]
    parts = inner.split(",") if inner else []
    nodes: List[NodeState] = []
    for p in parts:
        owner = p[0]
        troops = int(p[1:])
        nodes.append(NodeState(owner, troops))
    return GlobalState(nodes=tuple(nodes))


def test_query_region_roundtrip(
    lib_path: str | Path,
    base_dir: Path = BASE_LIB_DIR,
) -> None:
    """
    Sanity-check that query_region_from_libraries reproduces the library's
    probabilities for a small graph when treated as the whole board.

    This version assumes lib_path is a per-graph library file created by
    create_library.build_libraries_grid (i.e. it has a single 'prob_table').
    """
    lib_path = Path(lib_path)
    log.debug(f"Testing per-graph library: {lib_path}")
    library = load_library(lib_path)

    if "prob_table" not in library or not isinstance(library["prob_table"], pd.DataFrame):
        raise RuntimeError(
            "Library does not contain a full prob_table DataFrame; "
            "test_query_region_roundtrip requires a non-compressed table."
        )

    prob_table: pd.DataFrame = library["prob_table"]
    params = library.get("params", {})
    num_attacker_nodes = params.get("num_attacker_nodes")
    num_defender_nodes = params.get("num_defender_nodes")
    num_nodes = library.get("num_nodes")
    edges = set(library.get("edges", []))

    if num_attacker_nodes is None or num_defender_nodes is None or num_nodes is None:
        raise RuntimeError("Per-graph library missing required metadata (params / num_nodes / edges).")

    log.debug(f"  nA={num_attacker_nodes}, nD={num_defender_nodes}, num_nodes={num_nodes}")
    log.debug("  Edges:", edges)

    # 1) Pick one initial state (row) that has some non-zero probabilities
    row_label = None
    for candidate in prob_table.index:
        row = prob_table.loc[candidate]
        if (row > 0).any():
            row_label = candidate
            break

    if row_label is None:
        raise RuntimeError("No row with non-zero probabilities found in table!")

    log.debug("  Testing row_label:", row_label)

    # 2) Build a GlobalState that matches this small graph exactly
    global_state = global_state_from_row_label(row_label)
    global_edges = edges
    region_nodes = list(range(num_nodes))  # [0,1,2,...]

    # 3) Call query_region_from_libraries using the base dir
    result = query_region_from_libraries(
        combat_libraries_base=base_dir,
        global_state=global_state,
        global_edges=global_edges,
        region_nodes=region_nodes,
    )

    log.debug("  query_region_from_libraries used:", result["library_path"])
    log.debug("  row_label from query:", result["row_label"])
    assert result["row_label"] == row_label, "Row label mismatch!"

    # 4) Compare probabilities
    lib_row = prob_table.loc[row_label]

    # Turn into comparable dicts (only non-zero entries)
    lib_probs = {col: float(p) for col, p in lib_row.items() if p > 0}
    q_probs = result["probabilities"]

    # For debug:
    log.debug("  Library non-zero states:", len(lib_probs))
    log.debug("  Query non-zero states  :", len(q_probs))

    # Check that they match numerically
    for col, p_lib in lib_probs.items():
        p_q = q_probs.get(col)
        if p_q is None:
            raise AssertionError(f"Column {col} missing in query result")
        if abs(p_lib - p_q) > 1e-9:
            raise AssertionError(f"Probability mismatch for {col}: lib={p_lib}, query={p_q}")

    log.debug("  Probabilities match!")
    log.debug("OK ✅")


# ---------------------------------------------------------------------
# Continent battle graph construction
# ---------------------------------------------------------------------



def _raw_frontier_stats_for_continent(continent_name: str, player1) -> tuple[int, int]:
    continent_territories = Board.continent_territory_dict[continent_name]
    continent_indices = {t._index for t in continent_territories}

    raw_frontier_edges_in_cont = 0
    raw_attackers_gt1_touch_enemy_in_cont = 0

    for terr in continent_territories:
        if getattr(terr, "_owner", None) is not player1:
            continue
        touches = False
        for neigh in terr._neighbors:
            if neigh._index not in continent_indices:
                continue
            neigh_owner = getattr(neigh, "_owner", None)
            if neigh_owner is not None and neigh_owner is not player1:
                raw_frontier_edges_in_cont += 1
                touches = True
        if touches and getattr(terr, "_troops", 0) > 1:
            raw_attackers_gt1_touch_enemy_in_cont += 1

    return raw_frontier_edges_in_cont, raw_attackers_gt1_touch_enemy_in_cont




def build_continent_battle_graph(
    continent_name: str,
    players: Sequence["Players.Player"],
    *,
    # -----------------------------
    # debug instrumentation (logging is the real gate; these just limit spam)
    # -----------------------------
    debug: bool = True,
    debug_limit: int = 50,
    debug_tag: str = "",
    # -----------------------------
    # NEW: commitment control
    # -----------------------------
    commitment_map: Dict[int, str] | None = None,
) -> nx.Graph:
    """
    Build a *conflict frontier* graph (A–D only) for a given continent.

    Semantics (same as your previous patch intent):
      - Defender nodes are restricted to **inside** the continent.
      - Attacker nodes are:
          * inside-continent attacker-owned territories, plus
          * attacker-owned territories **outside** the continent that are adjacent to it.
      - Viable attacker nodes must have troops > 1 AND touch an enemy **inside** the continent.
      - NEW: if `commitment_map` is provided, an outside-of-continent attacker node is included
            ONLY if commitment_map[node_index] == continent_name. This prevents cross-continent
            double counting of the same outside node.

    Logging:
      - Uses logger ``risk.battle_graph``. DEBUG is gated by log_config.DEBUG_SWITCHES["battle_graph"].
      - Emits structured "why empty" diagnostics and commitment filtering stats.
    """
    # --------------------------------------------------------------
    # spam guard
    # --------------------------------------------------------------
    if not hasattr(build_continent_battle_graph, "_dbg_count"):
        build_continent_battle_graph._dbg_count = 0  # type: ignore[attr-defined]

    do_dbg = False
    if debug and build_continent_battle_graph._dbg_count < int(debug_limit):  # type: ignore[attr-defined]
        build_continent_battle_graph._dbg_count += 1  # type: ignore[attr-defined]
        do_dbg = True

    log_bg = get_logger("risk.battle_graph")

    def _dbg(msg: str) -> None:
        if do_dbg and log_bg.isEnabledFor(logging.DEBUG):
            prefix = f"[battle_graph]{(' ' + debug_tag) if debug_tag else ''}"
            log_bg.debug("%s %s", prefix, msg)

    if continent_name not in Board.continent_territory_dict:
        raise ValueError(f"Unknown continent: {continent_name}")

    if not players:
        raise ValueError("players sequence is empty; need attacker at index 0")

    player1 = players[0]
    continent_territories = Board.continent_territory_dict[continent_name]
    continent_indices: Set[int] = {t._index for t in continent_territories}

    # --------------------------------------------------------------
    # Raw frontier stats restricted to continent (truth on Board)
    # --------------------------------------------------------------
    raw_A_nodes_in_cont = 0
    raw_D_nodes_in_cont = 0
    raw_frontier_edges_in_cont = 0
    raw_attackers_gt1_touch_enemy_in_cont = 0

    for terr in continent_territories:
        owner = getattr(terr, "_owner", None)
        if owner is player1:
            raw_A_nodes_in_cont += 1
        elif owner is not None:
            raw_D_nodes_in_cont += 1

    for terr in continent_territories:
        if getattr(terr, "_owner", None) is not player1:
            continue
        touches_enemy = False
        for neigh in terr._neighbors:
            if neigh._index not in continent_indices:
                continue
            neigh_owner = getattr(neigh, "_owner", None)
            if neigh_owner is not None and neigh_owner is not player1:
                raw_frontier_edges_in_cont += 1
                touches_enemy = True
        if touches_enemy and int(getattr(terr, "_troops", 0)) > 1:
            raw_attackers_gt1_touch_enemy_in_cont += 1

    _dbg(
        f"continent={continent_name} cont_nodes={len(continent_indices)} "
        f"raw_A_nodes={raw_A_nodes_in_cont} raw_D_nodes={raw_D_nodes_in_cont} "
        f"raw_frontier_edges_in_cont={raw_frontier_edges_in_cont} "
        f"raw_attackers_gt1_touch_enemy_in_cont={raw_attackers_gt1_touch_enemy_in_cont}"
    )

    # --------------------------------------------------------------
    # Attacker candidates = continent + outside neighbors owned by attacker
    # --------------------------------------------------------------
    adjacent_attacker_indices: Set[int] = set()

    outside_total = 0
    outside_allowed = 0
    outside_rejected = 0

    for terr in continent_territories:
        for neigh in terr._neighbors:
            if getattr(neigh, "_continent", None) == continent_name:
                continue
            if getattr(neigh, "_owner", None) is not player1:
                continue

            outside_total += 1
            n_idx = int(neigh._index)

            if commitment_map is not None:
                committed_to = commitment_map.get(n_idx)
                if committed_to != continent_name:
                    outside_rejected += 1
                    continue

            outside_allowed += 1
            adjacent_attacker_indices.add(n_idx)

    if commitment_map is not None:
        _dbg(
            f"commitment_filter outside_total={outside_total} "
            f"outside_allowed={outside_allowed} outside_rejected={outside_rejected}"
        )

    attacker_candidates: Set[int] = set(continent_indices) | adjacent_attacker_indices

    # --------------------------------------------------------------
    # Filter viable attackers
    # --------------------------------------------------------------
    rej_not_owner = 0
    rej_troops_le1 = 0
    rej_no_enemy_in_cont = 0
    attacker_indices: Set[int] = set()

    for idx in attacker_candidates:
        terr = Board.node_to_territory_dict[int(idx)]
        if getattr(terr, "_owner", None) is not player1:
            rej_not_owner += 1
            continue
        troops = int(getattr(terr, "_troops", 0))
        if troops <= 1:
            rej_troops_le1 += 1
            continue

        has_enemy_neighbor_in_cont = False
        for neigh in terr._neighbors:
            n_idx = int(neigh._index)
            if n_idx not in continent_indices:
                continue
            neigh_owner = getattr(neigh, "_owner", None)
            if neigh_owner is not None and neigh_owner is not player1:
                has_enemy_neighbor_in_cont = True
                break

        if not has_enemy_neighbor_in_cont:
            rej_no_enemy_in_cont += 1
            continue

        attacker_indices.add(int(idx))

    # --------------------------------------------------------------
    # Defender nodes = inside-continent enemy nodes adjacent to viable attackers
    # --------------------------------------------------------------
    defender_indices: Set[int] = set()
    for a_idx in attacker_indices:
        a_terr = Board.node_to_territory_dict[int(a_idx)]
        for neigh in a_terr._neighbors:
            v = int(neigh._index)
            if v not in continent_indices:
                continue
            v_owner = getattr(neigh, "_owner", None)
            if v_owner is None or v_owner is player1:
                continue
            defender_indices.add(v)

    _dbg(
        f"attacker_candidates={len(attacker_candidates)} attackers_kept={len(attacker_indices)} "
        f"defenders_kept={len(defender_indices)} rej_not_owner={rej_not_owner} "
        f"rej_troops_le1={rej_troops_le1} rej_no_enemy_in_cont={rej_no_enemy_in_cont}"
    )

    if not attacker_indices or not defender_indices:
        _dbg(
            f"RETURN EMPTY: attackers={len(attacker_indices)} defenders={len(defender_indices)} "
            f"(raw_frontier_edges_in_cont={raw_frontier_edges_in_cont}, "
            f"raw_attackers_gt1_touch_enemy_in_cont={raw_attackers_gt1_touch_enemy_in_cont})"
        )
        return nx.Graph()

    # --------------------------------------------------------------
    # Build conflict-only A–D graph
    # --------------------------------------------------------------
    G = nx.Graph()
    for idx in attacker_indices:
        G.add_node(int(idx))
    for idx in defender_indices:
        G.add_node(int(idx))

    for a_idx in attacker_indices:
        a_terr = Board.node_to_territory_dict[int(a_idx)]
        for neigh in a_terr._neighbors:
            v = int(neigh._index)
            if v in defender_indices:
                G.add_edge(int(a_idx), int(v))

    if G.number_of_edges() == 0:
        _dbg(f"RETURN EMPTY: built graph had 0 edges (nodes={G.number_of_nodes()})")
        return nx.Graph()

    # Degree summary
    try:
        degs = [d for _, d in G.degree()]
        degs_sorted = sorted(degs)
        dmin = degs_sorted[0] if degs_sorted else 0
        dmed = degs_sorted[len(degs_sorted) // 2] if degs_sorted else 0
        dmax = degs_sorted[-1] if degs_sorted else 0
        _dbg(
            f"BUILT OK: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
            f"attackers={len(attacker_indices)} defenders={len(defender_indices)} "
            f"deg(min/med/max)=({dmin},{dmed},{dmax})"
        )
    except Exception:
        _dbg("BUILT OK (degree stats failed)")

    return G

def debug_battle_graphs(continent_name, players, static_battle_graph):
    fresh = build_continent_battle_graph(continent_name, players)

    static_nodes = set(static_battle_graph.nodes)
    fresh_nodes  = set(fresh.nodes)

    log.debug("STATIC battle_graph:", len(static_nodes), "nodes,", static_battle_graph.number_of_edges(), "edges")
    log.debug("FRESH  battle_graph:", len(fresh_nodes),  "nodes,", fresh.number_of_edges(), "edges")

    only_in_fresh  = fresh_nodes - static_nodes
    only_in_static = static_nodes - fresh_nodes
    log.debug("Nodes only in fresh :", len(only_in_fresh))
    log.debug("Nodes only in static:", len(only_in_static))

    # If you want: print a few example indices
    if only_in_fresh:
        log.debug("  example fresh-only:", list(only_in_fresh)[:10])
    if only_in_static:
        log.debug("  example static-only:", list(only_in_static)[:10])

    return fresh





# ---------------------------------------------------------------------
# Region partitioning into valid small graphs
# ---------------------------------------------------------------------


def _get_node_owner_name(node: Any) -> str | None:
    """
    Helper: return owner name ('Red', 'Blue', ...) for a territory index or Territory.
    """
    if isinstance(node, Board.Territory):
        terr = node
    else:
        terr = Board.node_to_territory_dict[node]
    owner = getattr(terr, "_owner", None)
    return owner._name if owner is not None else None


def _is_connected_subset(graph, subset_nodes: Tuple[Any, ...]) -> bool:
    """
    Check if the induced subgraph on subset_nodes is connected,
    using graph.neighbors(node).
    """
    subset = set(subset_nodes)
    if not subset:
        return False

    # BFS/DFS over the subset
    start = next(iter(subset))
    visited = {start}
    stack = [start]

    while stack:
        u = stack.pop()
        for v in graph.neighbors(u):
            if v in subset and v not in visited:
                visited.add(v)
                stack.append(v)

    return len(visited) == len(subset)


def graph_has_edge_compatible(
    graph,
    u: Any,
    v: Any,
    *,
    edges: Optional[Collection[Tuple[Any, Any]]] = None,
) -> bool:
    """Return edge membership across NetworkX and the bundled graph shim."""
    has_edge = getattr(graph, "has_edge", None)
    if callable(has_edge):
        return bool(has_edge(u, v))

    edge_iter = edges
    if edge_iter is None:
        try:
            edge_iter = graph.edges()
        except TypeError:
            edge_iter = graph.edges
    target = frozenset((u, v))
    return any(frozenset((a, b)) == target for a, b in edge_iter)


def partition_required_graph_nodes(
    graph,
    *,
    edges: Optional[Collection[Tuple[Any, Any]]] = None,
) -> Tuple[int, ...]:
    """Return the production partition universe: non-isolated graph nodes."""
    try:
        nodes_iter = graph.nodes()
    except TypeError:
        nodes_iter = graph.nodes
    nodes = tuple(int(n) for n in nodes_iter)

    edge_iter = edges
    if edge_iter is None:
        try:
            edge_iter = graph.edges()
        except TypeError:
            edge_iter = graph.edges
    incident = {int(n) for edge in edge_iter for n in edge}
    return tuple(sorted(n for n in nodes if n in incident))


def enumerate_disjoint_exact_region_covers(
    *,
    required_nodes: Collection[int],
    supported_region_signatures: Sequence[Sequence[int]],
    max_covers: Optional[int] = None,
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    """Production exact-cover search over canonical supported node sets."""
    required = frozenset(int(n) for n in required_nodes)
    if not required:
        return (tuple(),)

    regions = tuple(
        sorted(
            {
                tuple(sorted({int(n) for n in region}))
                for region in supported_region_signatures
                if region
                and frozenset(int(n) for n in region).issubset(required)
            }
        )
    )
    region_sets = tuple(frozenset(region) for region in regions)
    by_node: Dict[int, Tuple[int, ...]] = {
        node: tuple(i for i, region in enumerate(region_sets) if node in region)
        for node in sorted(required)
    }
    if any(not by_node[node] for node in required):
        return tuple()

    covers: Set[Tuple[Tuple[int, ...], ...]] = set()

    def backtrack(covered: frozenset[int], chosen: Tuple[int, ...]) -> None:
        if max_covers is not None and len(covers) >= int(max_covers):
            return
        if covered == required:
            covers.add(tuple(sorted(regions[i] for i in chosen)))
            return

        uncovered = required - covered
        viable_by_node = {
            node: tuple(
                i for i in by_node[node]
                if not (region_sets[i] & covered)
            )
            for node in uncovered
        }
        next_node = min(
            uncovered,
            key=lambda node: (len(viable_by_node[node]), int(node)),
        )
        for region_index in viable_by_node[next_node]:
            backtrack(covered | region_sets[region_index], chosen + (region_index,))

    backtrack(frozenset(), tuple())
    ordered = tuple(sorted(covers, key=lambda cover: (len(cover), cover)))
    if max_covers is not None:
        return ordered[: max(0, int(max_covers))]
    return ordered


def _partition_region_nodes_into_library_valid_subregions(
    players: Sequence["Players.Player"],
    continent_battle_graph,
    region_nodes: Sequence[Any],
    global_state: GlobalState,
    global_edges,
    combat_libraries_base: Path = BASE_LIB_DIR,
    *,
    query_cache: Optional[Any] = None,
) -> List[Dict[str, Any]] | None:

    if not players:
        return None

    region_nodes = tuple(region_nodes)
    region_set = set(region_nodes)
    if not region_set:
        return None

    candidate_subregions: List[Dict[str, Any]] = []

    max_size = max(a + d for a, d in ALLOWED_PATTERNS)
    min_size = 2

    for size in range(min_size, min(max_size, len(region_nodes)) + 1):
        for subset in itertools.combinations(region_nodes, size):
            try:
                role_map = extract_region_combat_roles(region_nodes=subset, global_state=global_state)
            except ValueError:
                continue

            attacker_nodes = [n for n in subset if role_map[int(n)] == "A"]
            defender_nodes = [n for n in subset if role_map[int(n)] == "D"]

            nA = len(attacker_nodes)
            nD = len(defender_nodes)

            if (nA, nD) not in ALLOWED_PATTERNS:
                continue
            if nA == 0 or nD == 0:
                continue
            if not _is_connected_subset(continent_battle_graph, subset):
                continue

            # ---- star-only topology gating ----
            if (nA, nD) in STAR_ONLY_PATTERNS:
                mapping, inv_mapping, _, _ = reindex_region_nodes(global_state, subset)
                sub_edges_set = reindex_edges_for_region(global_edges, inv_mapping, set(subset))
                if not is_star_edges(sub_edges_set, center=0, n_nodes=(nA + nD)):
                    continue

            # Library support check for this subregion
            try:
                query_region_from_libraries(
                    combat_libraries_base=combat_libraries_base,
                    global_state=global_state,
                    global_edges=global_edges,
                    region_nodes=subset,
                    query_cache=query_cache,
                )
            except Exception as e:
                if _is_coverage_failure(e):
                    continue
                raise
            else:
                candidate_subregions.append(
                    {
                        "region_nodes":   tuple(subset),
                        "attacker_nodes": tuple(attacker_nodes),
                        "defender_nodes": tuple(defender_nodes),
                        "pattern":        (nA, nD),
                    }
                )

    if not candidate_subregions:
        return None

    # --- Step 2: map each node -> which subregions contain it ---
    subs_by_node: Dict[Any, List[int]] = {n: [] for n in region_set}
    for i, sub in enumerate(candidate_subregions):
        for n in sub["region_nodes"]:
            subs_by_node[n].append(i)

    for n in region_set:
        if not subs_by_node[n]:
            return None

    # --- Step 3: exact cover via backtracking ---
    best_partition: List[Dict[str, Any]] | None = None

    def backtrack(covered: Set[Any], chosen_indices: List[int]) -> None:
        nonlocal best_partition

        if covered == region_set:
            part = [candidate_subregions[i] for i in chosen_indices]
            if best_partition is None or len(part) < len(best_partition):
                best_partition = part
            return

        remaining = sorted(region_set - covered)
        next_node = remaining[0]

        for sub_idx in subs_by_node[next_node]:
            sub = candidate_subregions[sub_idx]
            sub_nodes = set(sub["region_nodes"])
            if sub_nodes & covered:
                continue
            backtrack(covered | sub_nodes, chosen_indices + [sub_idx])

    backtrack(set(), [])
    return best_partition



# -----------------------------
# Helper: classify viability failures (shared by partitioner + ranker)
# -----------------------------
def _is_query_viability_failure(e: Exception) -> bool:
    """
    Return True if this exception means "this region/partition is not queryable from available libraries"
    and should be treated as a normal *viability failure* (skip / fallback), NOT a hard crash.

    IMPORTANT: This is intentionally broad. We only re-raise truly unexpected errors.
    """
    return isinstance(
        e,
        (
            FileNotFoundError,    # missing topology/pattern library file
            ValueError,           # row label missing, star mismatch, no internal edges, etc.
            KeyError,             # missing dict keys in prob_table / payload structures
            NotImplementedError,  # legacy payload encountered under v2 hard-switch
        ),
    )



def partition_continent_battle_graph_into_valid_small_graphs(
    players: Sequence["Players.Player"],
    continent_battle_graph,
    max_partitions: int = 10,
    combat_libraries_base: Path = BASE_LIB_DIR,
    *,
    global_state_override: Optional[GlobalState] = None,
    query_cache: Optional[Any] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Partition the (frontier-only) continent battle graph into small, library-valid regions.

    Fully patched (STAR_ONLY gating REMOVED here; library query is the authority):

      1) Canonical owner labels: attacker is "A" (NOT player._name).
         (_get_node_owner_name must return "A"/"D"/None)
      2) Prune degree-0 nodes before partitioning (cannot participate in any battle region).
      3) Enforce region invariants:
           - (nA, nD) in ALLOWED_PATTERNS
           - nA > 0 and nD > 0
           - connected induced subset
           - contains at least one A–D edge (frontier edge) inside region
      4) NO STAR_ONLY topology gating here.
      5) Library viability gate:
           - If coverage failure: attempt local subpartition.
           - If other *viability failure*: skip region (do NOT crash).
           - If unexpected error: re-raise.

    Returns:
      List of partitions, each a list of region dicts.
    """
    if not players:
        return []

    try:
        nodes_iter = continent_battle_graph.nodes()
    except TypeError:
        nodes_iter = continent_battle_graph.nodes

    all_nodes: List[Any] = list(nodes_iter)
    if not all_nodes:
        return []

    # Prune isolated nodes early (degree==0) using the shared graph protocol.
    all_nodes = list(partition_required_graph_nodes(continent_battle_graph))

    if not all_nodes:
        return []

    all_node_set: Set[Any] = set(all_nodes)

    global_state = global_state_override or build_global_state_for_board(players)
    global_edges = edges_from_battle_graph(continent_battle_graph)

    raw_candidate_regions: List[Dict[str, Any]] = []

    max_size = max(a + d for a, d in ALLOWED_PATTERNS)
    min_size = 2

    def _has_frontier_edge(att_nodes: Sequence[Any], def_nodes: Sequence[Any]) -> bool:
        return any(
            graph_has_edge_compatible(
                continent_battle_graph,
                u,
                v,
                edges=global_edges,
            )
            for u in att_nodes
            for v in def_nodes
        )

    # ------------------------------------------------------------------
    # Enumerate candidate regions
    # ------------------------------------------------------------------
    for size in range(min_size, max_size + 1):
        for subset in itertools.combinations(all_nodes, size):
            try:
                role_map = extract_region_combat_roles(region_nodes=subset, global_state=global_state)
            except ValueError:
                continue

            attacker_nodes = [n for n in subset if role_map[int(n)] == "A"]
            defender_nodes = [n for n in subset if role_map[int(n)] == "D"]

            nA = len(attacker_nodes)
            nD = len(defender_nodes)

            if (nA, nD) not in ALLOWED_PATTERNS:
                continue
            if nA == 0 or nD == 0:
                continue
            if not _is_connected_subset(continent_battle_graph, subset):
                continue
            if not _has_frontier_edge(attacker_nodes, defender_nodes):
                continue

            raw_candidate_regions.append(
                {
                    "region_nodes": tuple(subset),
                    "attacker_nodes": tuple(attacker_nodes),
                    "defender_nodes": tuple(defender_nodes),
                    "pattern": (nA, nD),
                }
            )

    if not raw_candidate_regions:
        return []

    # ------------------------------------------------------------------
    # Library viability gate
    # ------------------------------------------------------------------
    candidate_regions: List[Dict[str, Any]] = []
    for region in raw_candidate_regions:
        region_nodes = region["region_nodes"]

        try:
            query_region_from_libraries(
                combat_libraries_base=combat_libraries_base,
                global_state=global_state,
                global_edges=global_edges,
                region_nodes=region_nodes,
                debug=False,  # avoid spam during partition generation
                query_cache=query_cache,
            )
        except Exception as e:
            if _is_coverage_failure(e):
                subregions = _partition_region_nodes_into_library_valid_subregions(
                    players=players,
                    continent_battle_graph=continent_battle_graph,
                    region_nodes=region_nodes,
                    global_state=global_state,
                    global_edges=global_edges,
                    combat_libraries_base=combat_libraries_base,
                    query_cache=query_cache,
                )
                if subregions is None:
                    continue
                candidate_regions.extend(subregions)
                continue

            # NEW: treat all normal query failures as "region not viable"
            if _is_query_viability_failure(e):
                continue

            # Unexpected: real bug
            raise

        else:
            candidate_regions.append(region)

    if not candidate_regions:
        return []

    # ------------------------------------------------------------------
    # Exact-cover partitions: disjoint regions that cover all nodes
    # ------------------------------------------------------------------
    region_by_signature: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for region in candidate_regions:
        signature = tuple(sorted(int(n) for n in region["region_nodes"]))
        region_by_signature.setdefault(signature, region)

    cover_signatures = enumerate_disjoint_exact_region_covers(
        required_nodes=all_node_set,
        supported_region_signatures=tuple(region_by_signature),
        max_covers=max_partitions,
    )
    if not cover_signatures:
        return []
    return [
        [region_by_signature[signature] for signature in cover]
        for cover in cover_signatures
    ]



# ---------------------------------------------------------------------
# Using libraries to evaluate partitions
# ---------------------------------------------------------------------


def is_conquered_state_label(col_label: str) -> bool:
    """
    Return True if this absorbing state label corresponds to
    a fully conquered region (same logic as is_successful).
    """
    state = global_state_from_row_label(col_label)
    return is_successful(state)


def build_global_state_for_board(players: Sequence["Players.Player"]) -> GlobalState:
    """
    Build a GlobalState over all territories, using:
      - owner 'A' for player1 (players[0])
      - owner 'D' for everyone else

    nodes[i] corresponds to territory with index i, for i in 0..max_index.
    Indices that do not map to a real territory are filled with a dummy D0 node.
    """
    player1 = players[0]

    max_idx = max(Board.node_to_territory_dict.keys())

    nodes: List[NodeState] = []
    for idx in range(max_idx + 1):
        terr = Board.node_to_territory_dict.get(idx, None)
        if terr is None:
            # Dummy node, unused
            nodes.append(NodeState(owner='D', troops=0))
            continue

        owner = getattr(terr, "_owner", None)
        troops = getattr(terr, "_troops", 0)

        if owner is player1:
            owner_char = 'A'
        else:
            owner_char = 'D'

        nodes.append(NodeState(owner=owner_char, troops=troops))

    return GlobalState(nodes=tuple(nodes))


def edges_from_battle_graph(battle_graph) -> List[Tuple[int, int]]:
    """
    Extract edges from a battle_graph (networkx-like), returning a list of (u, v).
    """
    try:
        edges_iter = battle_graph.edges()
    except TypeError:
        edges_iter = battle_graph.edges
    return list(edges_iter)
