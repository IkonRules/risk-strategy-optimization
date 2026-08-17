from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import math
import os
import pickle
import time
import traceback as traceback_module
import uuid

import networkx as nx
import numpy as np
import pandas as pd

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.mathematical.continent_model import approximate_graph_outcome_probabilities as agop
from project_risk.mathematical.transition_prediction_ml import generate_data_ML as gdm
from project_risk.mathematical.small_graph_model.small_graph_outcome_probabilities import GlobalState, NodeState


DEFAULT_STAGE_A_V2_OUTPUT_DIR = Path("transition_distribution_data_v2_corrected_mc5")
STAGE_A_V2_OUTPUT_FORMAT = "transition_distribution_stage_a_v2_chunked_v1"
DEFAULT_TARGET_SUCCESSES = {
    "Australia": 250,
    "South America": 250,
    "North America": 250,
    "Europe": 100,
    "Africa": 100,
    "Asia": 100,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    return repr(value)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _safe_continent_name(continent_name: str) -> str:
    return str(continent_name).strip().replace(" ", "_")


def _stable_int_seed(payload: Any) -> int:
    encoded = json.dumps(
        _json_ready(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return int(hashlib.sha256(encoded).hexdigest()[:16], 16) % (2 ** 32)


def derive_stage_a_state_generation_seed(
    *, base_seed: int, continent_name: str, attempt_index: int
) -> int:
    return _stable_int_seed(
        {
            "kind": "stage_a_v2_state_generation",
            "base_seed": int(base_seed),
            "continent_name": str(continent_name),
            "attempt_index": int(attempt_index),
        }
    )


def _consistent_value(values: Iterable[Any]) -> Any:
    unique = list(dict.fromkeys(values))
    return unique[0] if len(unique) == 1 else None


@lru_cache(maxsize=8)
def inspect_transition_library_metadata(library_dir_string: str) -> Dict[str, Any]:
    library_dir = Path(library_dir_string).resolve()
    graph_files = sorted(library_dir.rglob("graph_*.pkl")) if library_dir.exists() else []
    fields: Dict[str, List[Any]] = {
        "library_format": [],
        "policy_option_mode": [],
        "max_policy_options_per_row": [],
        "max_options_per_state": [],
        "max_leaf_split_depth": [],
    }
    read_errors: List[Dict[str, str]] = []
    for graph_file in graph_files:
        try:
            with graph_file.open("rb") as handle:
                payload = pickle.load(handle)
            params = dict(payload.get("params", {}) or {})
            descriptor = dict(payload.get("prob_table_chunked", {}) or {})
            fields["library_format"].append(
                descriptor.get("format") or payload.get("prob_format")
            )
            for field in (
                "policy_option_mode",
                "max_policy_options_per_row",
                "max_options_per_state",
                "max_leaf_split_depth",
            ):
                fields[field].append(
                    descriptor.get(field, params.get(field, payload.get(field)))
                )
        except Exception as exc:
            read_errors.append(
                {"file": str(graph_file), "error": f"{type(exc).__name__}: {exc}"}
            )

    unique_values = {
        field: sorted(
            {json.dumps(_json_ready(value), sort_keys=True): value for value in values}.values(),
            key=repr,
        )
        for field, values in fields.items()
    }
    inconsistent_fields = tuple(
        field for field, values in unique_values.items() if len(values) > 1
    )
    return {
        "library_dir": str(library_dir),
        "graph_file_count": int(len(graph_files)),
        "graph_files_read": int(len(graph_files) - len(read_errors)),
        "library_format": _consistent_value(fields["library_format"]),
        "policy_option_mode": _consistent_value(fields["policy_option_mode"]),
        "max_policy_options_per_row": _consistent_value(fields["max_policy_options_per_row"]),
        "max_options_per_state": _consistent_value(fields["max_options_per_state"]),
        "max_leaf_split_depth": _consistent_value(fields["max_leaf_split_depth"]),
        "unique_values": unique_values,
        "inconsistent_fields": inconsistent_fields,
        "inconsistent": bool(inconsistent_fields or read_errors),
        "read_error_count": int(len(read_errors)),
        "read_errors": read_errors[:20],
    }


def node_marginal_rows_from_grouped_example(
    example: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    initial = {
        int(node): (str(owner), int(troops))
        for node, owner, troops in (example.get("initial_full_graph_signature", ()) or ())
    }
    battle_nodes = {int(node) for node in (example.get("battle_graph_nodes", ()) or ())}
    macro = dict(example.get("macro_features", {}) or {})
    rows: List[Dict[str, Any]] = []
    for node_key, marginal in (example.get("node_marginals", {}) or {}).items():
        node = int(node_key)
        initial_owner, initial_troops = initial[node]
        row = {
            "example_id": str(example.get("example_id")),
            "base_example_id": example.get("base_example_id"),
            "target_generation_version": example.get("target_generation_version"),
            "config_fingerprint": example.get("config_fingerprint"),
            "target_seed": example.get("target_seed"),
            "two_stage_mc_samples": example.get("two_stage_mc_samples"),
            "state_id": int(example.get("state_id", -1)),
            "continent_name": example.get("continent_name"),
            "attack_perspective": example.get("attack_perspective"),
            "node_index": node,
            "initial_owner": initial_owner,
            "initial_troops": int(initial_troops),
            "p_attacker_final": float(marginal.get("p_attacker_final", 0.0)),
            "p_defender_final": float(marginal.get("p_defender_final", 0.0)),
            "expected_troops": float(marginal.get("expected_troops", 0.0)),
            "expected_troops_if_attacker": float(
                marginal.get("expected_troops_if_attacker", 0.0)
            ),
            "expected_troops_if_defender": float(
                marginal.get("expected_troops_if_defender", 0.0)
            ),
            "p_changed_owner": float(marginal.get("p_changed_owner", 0.0)),
            "is_battle_node": int(node in battle_nodes),
        }
        row.update(macro)
        rows.append(row)
    return rows


class StageAV2ChunkStore:
    """Atomic grouped/node chunk storage with per-continent checkpoints."""

    def __init__(
        self,
        *,
        output_dir: Path | str,
        base_config: gdm.TransitionDistributionConfig,
        library_metadata_summary: Mapping[str, Any],
        phase: Optional[str] = None,
        active_config: Optional[gdm.TransitionDistributionConfig] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.base_config = base_config
        self.active_config = active_config or base_config
        self.phase = phase
        self.data_root = self.output_dir if phase is None else self.output_dir / str(phase)
        self.library_metadata_summary = dict(library_metadata_summary)
        self.base_fingerprint = gdm.transition_distribution_config_fingerprint(base_config)
        self.config_fingerprint = gdm.transition_distribution_config_fingerprint(
            self.active_config
        )
        self.target_fingerprint = gdm.transition_distribution_target_fingerprint(
            self.active_config
        )
        self._initialize()

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def config_path(self) -> Path:
        if self.phase is None:
            return self.output_dir / "config.json"
        return self.data_root / "config.json"

    def _config_payload(self, config: gdm.TransitionDistributionConfig) -> Dict[str, Any]:
        return {
            "target_generation_version": config.target_generation_version,
            "code_configuration_version": config.code_configuration_version,
            "config_fingerprint": gdm.transition_distribution_config_fingerprint(config),
            "target_config_fingerprint": gdm.transition_distribution_target_fingerprint(config),
            "configuration": asdict(config),
            "resolved_two_stage_mc_samples": config.resolved_two_stage_mc_samples,
            "resolved_two_stage_mc_seed": config.resolved_two_stage_mc_seed,
            "library_configuration": self.library_metadata_summary,
        }

    def _initialize(self) -> None:
        output_had_files = self.output_dir.exists() and any(self.output_dir.iterdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base_config_path = self.output_dir / "config.json"
        if self.phase is None and output_had_files and not self.base_config.resume:
            raise FileExistsError(
                f"Refusing to write into existing Stage A V2 output with resume=False: "
                f"{self.output_dir}"
            )
        if base_config_path.exists():
            existing = json.loads(base_config_path.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != self.base_fingerprint:
                raise ValueError(
                    "Refusing Stage A V2 resume: configuration fingerprint mismatch "
                    f"({existing.get('config_fingerprint')} != {self.base_fingerprint})"
                )
        else:
            _atomic_write_json(base_config_path, self._config_payload(self.base_config))

        if self.phase is not None:
            self.data_root.mkdir(parents=True, exist_ok=True)
            if self.config_path.exists():
                if not self.active_config.resume:
                    raise FileExistsError(
                        f"Refusing to reuse existing {self.phase} output with resume=False: "
                        f"{self.data_root}"
                    )
                existing = json.loads(self.config_path.read_text(encoding="utf-8"))
                if existing.get("config_fingerprint") != self.config_fingerprint:
                    raise ValueError(
                        f"Refusing {self.phase} resume: configuration fingerprint mismatch"
                    )
            else:
                _atomic_write_json(
                    self.config_path, self._config_payload(self.active_config)
                )

        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_fingerprint") != self.base_fingerprint:
                raise ValueError("Manifest/configuration fingerprint mismatch")
        else:
            manifest = {
                "format": STAGE_A_V2_OUTPUT_FORMAT,
                "target_generation_version": self.base_config.target_generation_version,
                "code_configuration_version": self.base_config.code_configuration_version,
                "config_fingerprint": self.base_fingerprint,
                "target_config_fingerprint": gdm.transition_distribution_target_fingerprint(
                    self.base_config
                ),
                "created_utc": _utc_now(),
                "last_updated_utc": _utc_now(),
                "continents": {},
                "phases": {},
            }
            _atomic_write_json(self.manifest_path, manifest)

    def _load_manifest(self) -> Dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self, manifest: Mapping[str, Any]) -> None:
        payload = dict(manifest)
        payload["last_updated_utc"] = _utc_now()
        _atomic_write_json(self.manifest_path, payload)

    def _manifest_continents(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        if self.phase is None:
            return manifest.setdefault("continents", {})
        phase_record = manifest.setdefault("phases", {}).setdefault(
            self.phase,
            {
                "config_fingerprint": self.config_fingerprint,
                "target_config_fingerprint": self.target_fingerprint,
                "continents": {},
            },
        )
        if phase_record.get("config_fingerprint") != self.config_fingerprint:
            raise ValueError(f"Manifest fingerprint mismatch for phase {self.phase}")
        return phase_record.setdefault("continents", {})

    def _continent_paths(self, continent_name: str) -> Dict[str, Path]:
        root = self.data_root / _safe_continent_name(continent_name)
        paths = {
            "root": root,
            "grouped": root / "grouped_examples",
            "nodes": root / "node_marginals",
            "failures": root / "failures",
            "checkpoints": root / "checkpoints",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def _ensure_continent_manifest(self, continent_name: str) -> Dict[str, Any]:
        manifest = self._load_manifest()
        continents = self._manifest_continents(manifest)
        safe = _safe_continent_name(continent_name)
        paths = self._continent_paths(continent_name)
        record = continents.setdefault(
            str(continent_name),
            {
                "directory": str(paths["root"].relative_to(self.output_dir)),
                "grouped_chunks": [],
                "node_marginal_chunks": [],
                "failure_files": [],
                "checkpoint": str(
                    (paths["checkpoints"] / "checkpoint.json").relative_to(
                        self.output_dir
                    )
                ),
                "safe_name": safe,
            },
        )
        self._save_manifest(manifest)
        return record

    def _checkpoint_path(self, continent_name: str) -> Path:
        return self._continent_paths(continent_name)["checkpoints"] / "checkpoint.json"

    def _default_checkpoint(self, continent_name: str) -> Dict[str, Any]:
        return {
            "target_generation_version": self.active_config.target_generation_version,
            "config_fingerprint": self.config_fingerprint,
            "target_config_fingerprint": self.target_fingerprint,
            "continent_name": str(continent_name),
            "completed_example_ids": [],
            "completed_records": {},
            "failed_example_ids": [],
            "failure_records": [],
            "expected_no_combat_records": [],
            "current_chunk_index": 0,
            "attempt_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "expected_no_combat_count": 0,
            "cumulative_runtime_seconds": 0.0,
            "last_updated_utc": _utc_now(),
        }

    def _save_checkpoint(self, continent_name: str, checkpoint: Mapping[str, Any]) -> None:
        payload = dict(checkpoint)
        payload["last_updated_utc"] = _utc_now()
        _atomic_write_json(self._checkpoint_path(continent_name), payload)

    def _registered_chunks(self, continent_name: str, key: str) -> List[Path]:
        manifest = self._load_manifest()
        continents = self._manifest_continents(manifest)
        record = continents.get(str(continent_name), {})
        return [self.output_dir / rel for rel in record.get(key, ())]

    def load_checkpoint(self, continent_name: str) -> Dict[str, Any]:
        self._ensure_continent_manifest(continent_name)
        path = self._checkpoint_path(continent_name)
        checkpoint = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else self._default_checkpoint(continent_name)
        )
        if checkpoint.get("config_fingerprint") != self.config_fingerprint:
            raise ValueError(
                f"Refusing resume for {continent_name}: configuration fingerprint mismatch"
            )

        rows_by_id: Dict[str, Mapping[str, Any]] = {}
        paths = self._continent_paths(continent_name)
        registered = self._registered_chunks(continent_name, "grouped_chunks")
        discovered = sorted(paths["grouped"].glob("chunk_*.pkl"))
        for chunk_path in sorted(set(registered + discovered)):
            if not chunk_path.exists():
                continue
            frame = pd.read_pickle(chunk_path)
            for row in frame.to_dict(orient="records"):
                example_id = str(row.get("example_id"))
                if example_id in rows_by_id:
                    raise ValueError(
                        f"Duplicate example_id {example_id} in existing chunks"
                    )
                rows_by_id[example_id] = row

        completed = {str(value) for value in checkpoint.get("completed_example_ids", ())}
        missing = completed - set(rows_by_id)
        if missing:
            raise ValueError(
                f"Checkpoint contains completed IDs missing from outputs: {sorted(missing)[:5]}"
            )
        records = dict(checkpoint.get("completed_records", {}) or {})
        for example_id, row in rows_by_id.items():
            completed.add(example_id)
            records.setdefault(
                example_id,
                {
                    "status": "ok",
                    "target_seed": row.get("target_seed"),
                    "elapsed_seconds": row.get("target_generation_runtime_seconds"),
                    "recovered_from_output": True,
                },
            )
        checkpoint["completed_example_ids"] = sorted(completed)
        checkpoint["completed_records"] = records
        checkpoint["success_count"] = int(len(completed))
        checkpoint["current_chunk_index"] = int(
            len(completed) // int(self.active_config.output_chunk_size)
        )
        self._save_checkpoint(continent_name, checkpoint)
        self.rebuild_node_marginal_chunks(continent_name)
        return checkpoint

    def _register_chunk(
        self, continent_name: str, grouped_path: Path, node_path: Path
    ) -> None:
        manifest = self._load_manifest()
        continents = self._manifest_continents(manifest)
        record = continents[str(continent_name)]
        grouped_rel = str(grouped_path.relative_to(self.output_dir))
        node_rel = str(node_path.relative_to(self.output_dir))
        if grouped_rel not in record["grouped_chunks"]:
            record["grouped_chunks"].append(grouped_rel)
            record["grouped_chunks"].sort()
        if node_rel not in record["node_marginal_chunks"]:
            record["node_marginal_chunks"].append(node_rel)
            record["node_marginal_chunks"].sort()
        self._save_manifest(manifest)

    def record_success(
        self,
        *,
        continent_name: str,
        example: Mapping[str, Any],
        checkpoint: Dict[str, Any],
    ) -> bool:
        example_id = str(example.get("example_id"))
        completed = {str(value) for value in checkpoint.get("completed_example_ids", ())}
        if example_id in completed:
            return False
        paths = self._continent_paths(continent_name)
        chunk_size = int(self.active_config.output_chunk_size)
        chunk_index = int(checkpoint.get("current_chunk_index", 0) or 0)
        while True:
            grouped_path = paths["grouped"] / f"chunk_{chunk_index:06d}.pkl"
            grouped_df = (
                pd.read_pickle(grouped_path)
                if grouped_path.exists()
                else pd.DataFrame()
            )
            if len(grouped_df) < chunk_size:
                break
            chunk_index += 1
        if not grouped_df.empty and example_id in set(grouped_df["example_id"].astype(str)):
            return False
        grouped_df = pd.concat(
            [grouped_df, pd.DataFrame([dict(example)])], ignore_index=True
        )
        node_path = paths["nodes"] / f"chunk_{chunk_index:06d}.pkl"
        node_df = pd.read_pickle(node_path) if node_path.exists() else pd.DataFrame()
        if not node_df.empty and "example_id" in node_df:
            node_df = node_df[node_df["example_id"].astype(str) != example_id]
        node_rows = node_marginal_rows_from_grouped_example(example)
        if node_rows:
            node_df = pd.concat([node_df, pd.DataFrame(node_rows)], ignore_index=True)

        _atomic_write_pickle(grouped_path, grouped_df)
        _atomic_write_pickle(node_path, node_df)
        self._register_chunk(continent_name, grouped_path, node_path)

        completed.add(example_id)
        checkpoint["completed_example_ids"] = sorted(completed)
        records = dict(checkpoint.get("completed_records", {}) or {})
        records[example_id] = {
            "status": "ok",
            "output_chunk": str(grouped_path.relative_to(self.output_dir)),
            "target_seed": int(example.get("target_seed", 0)),
            "elapsed_seconds": float(
                example.get("target_generation_runtime_seconds", 0.0) or 0.0
            ),
            "attempt_index": example.get("attempt_index"),
        }
        checkpoint["completed_records"] = records
        checkpoint["success_count"] = int(len(completed))
        checkpoint["current_chunk_index"] = int(chunk_index)
        self._save_checkpoint(continent_name, checkpoint)
        return True

    def record_failure(
        self,
        *,
        continent_name: str,
        failure: Mapping[str, Any],
        checkpoint: Dict[str, Any],
        expected_no_combat: bool = False,
    ) -> Path:
        paths = self._continent_paths(continent_name)
        attempt_index = int(failure.get("attempt_index", checkpoint.get("attempt_count", 0)) or 0)
        example_id = str(failure.get("example_id") or "no_example_id")
        kind = "expected" if expected_no_combat else "failure"
        path = paths["failures"] / (
            f"{kind}_{attempt_index:08d}_{example_id[-12:]}.json"
        )
        _atomic_write_json(path, dict(failure))

        manifest = self._load_manifest()
        continents = self._manifest_continents(manifest)
        record = continents[str(continent_name)]
        rel = str(path.relative_to(self.output_dir))
        if rel not in record["failure_files"]:
            record["failure_files"].append(rel)
            record["failure_files"].sort()
        self._save_manifest(manifest)

        if expected_no_combat:
            values = list(checkpoint.get("expected_no_combat_records", ()) or ())
            if rel not in {str(item.get("file")) for item in values}:
                values.append({"example_id": failure.get("example_id"), "file": rel})
            checkpoint["expected_no_combat_records"] = values
            checkpoint["expected_no_combat_count"] = int(len(values))
        else:
            records = list(checkpoint.get("failure_records", ()) or ())
            if rel not in {str(item.get("file")) for item in records}:
                records.append({"example_id": failure.get("example_id"), "file": rel})
            checkpoint["failure_records"] = records
            failed = list(checkpoint.get("failed_example_ids", ()) or ())
            if failure.get("example_id") is not None:
                failed.append(str(failure.get("example_id")))
            checkpoint["failed_example_ids"] = sorted(set(failed))
            checkpoint["failure_count"] = int(len(records))
        self._save_checkpoint(continent_name, checkpoint)
        return path

    def finish_attempt(
        self,
        continent_name: str,
        checkpoint: Dict[str, Any],
        attempt_index: int,
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        previous_count = int(checkpoint.get("attempt_count", 0) or 0)
        next_count = max(previous_count, int(attempt_index) + 1)
        checkpoint["attempt_count"] = next_count
        if next_count > previous_count and elapsed_seconds is not None:
            checkpoint["cumulative_runtime_seconds"] = float(
                checkpoint.get("cumulative_runtime_seconds", 0.0) or 0.0
            ) + max(0.0, float(elapsed_seconds))
        self._save_checkpoint(continent_name, checkpoint)

    def rebuild_node_marginal_chunks(self, continent_name: str) -> None:
        paths = self._continent_paths(continent_name)
        for grouped_path in sorted(paths["grouped"].glob("chunk_*.pkl")):
            grouped_df = pd.read_pickle(grouped_path)
            rows: List[Dict[str, Any]] = []
            for example in grouped_df.to_dict(orient="records"):
                rows.extend(node_marginal_rows_from_grouped_example(example))
            node_path = paths["nodes"] / grouped_path.name
            _atomic_write_pickle(node_path, pd.DataFrame(rows))
            self._register_chunk(continent_name, grouped_path, node_path)


def load_stage_a_v2_grouped_examples(
    output_dir: Path | str,
    *,
    phase: Optional[str] = None,
) -> pd.DataFrame:
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if phase is None:
        continents = manifest.get("continents", {})
    else:
        continents = manifest.get("phases", {}).get(str(phase), {}).get("continents", {})
    frames = []
    for record in continents.values():
        for rel in record.get("grouped_chunks", ()):
            path = root / rel
            if path.exists():
                frames.append(pd.read_pickle(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_corrected_stage_a_v2_production_config(
    config: gdm.TransitionDistributionConfig,
) -> None:
    expected = {
        "partition_candidate_selection_mode": "maximal_per_partition_utility",
        "second_stage_execution_mode": "optimized_reuse",
        "second_stage_sampling_mode": "stable_region_option_scenarios",
    }
    for field, required in expected.items():
        actual = getattr(config, field)
        if actual != required:
            raise ValueError(f"Stage A V2 requires {field}={required!r}, got {actual!r}")
    for field in (
        "utility_abs_tolerance",
        "utility_rel_tolerance",
        "max_candidates_per_partition",
        "max_policy_combos_per_partition",
    ):
        if getattr(config, field) is not None:
            raise ValueError(f"Stage A V2 production forbids {field}")


def _graph_from_signature(signature: Any) -> nx.Graph:
    nodes, edges = signature
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(int(node))
    for u, v in edges:
        graph.add_edge(int(u), int(v))
    return graph


def _global_state_from_raw_signature(signature: Any) -> GlobalState:
    records = tuple(signature or ())
    if not records:
        return GlobalState(nodes=tuple())
    max_node = max(int(node) for node, _, _ in records)
    nodes = [NodeState(owner="D", troops=0) for _ in range(max_node + 1)]
    for node, owner, troops in records:
        nodes[int(node)] = NodeState(owner=str(owner), troops=int(troops))
    return GlobalState(nodes=tuple(nodes))


def _initial_full_state_as_global(example: Mapping[str, Any]) -> GlobalState:
    records = tuple(example.get("initial_full_graph_signature", ()) or ())
    max_node = max((int(node) for node, _, _ in records), default=0)
    nodes = [NodeState(owner="D", troops=0) for _ in range(max_node + 1)]
    for node, owner, troops in records:
        nodes[int(node)] = NodeState(owner=str(owner), troops=int(troops))
    return GlobalState(nodes=tuple(nodes))


def validate_grouped_transition_example_v2(
    example: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    errors: List[str] = []
    counts = dict(example.get("full_graph_successor_state_counts", {}) or {})
    probabilities = dict(
        example.get("full_graph_successor_state_probabilities", {}) or {}
    )
    full_nodes = tuple(int(node) for node in (example.get("full_graph_nodes", ()) or ()))
    battle_nodes = {int(node) for node in (example.get("battle_graph_nodes", ()) or ())}
    initial = {
        int(node): (str(owner), int(troops))
        for node, owner, troops in (example.get("initial_full_graph_signature", ()) or ())
    }
    expected_samples = int(example.get("two_stage_mc_samples", 0) or 0)

    if not example.get("example_id"):
        errors.append("missing example_id")
    if not counts:
        errors.append("empty successor-state distribution")
    total = 0
    for signature, count in counts.items():
        if not isinstance(count, (int, np.integer)) or int(count) < 0:
            errors.append(f"invalid count for signature {signature!r}")
            continue
        total += int(count)
        records = tuple(signature or ())
        if tuple(int(node) for node, _, _ in records) != full_nodes:
            errors.append("successor signature does not use fixed full-graph node order")
        for node, owner, troops in records:
            if str(owner) not in {"A", "D"}:
                errors.append(f"non-canonical owner {owner!r}")
            if int(troops) < 1:
                errors.append(f"non-positive troop count at node {node}")
            if int(node) not in battle_nodes and initial.get(int(node)) != (
                str(owner),
                int(troops),
            ):
                errors.append(f"non-updated node {node} changed")
    if expected_samples > 0 and total != expected_samples:
        errors.append(
            f"successor counts sum to {total}, expected {expected_samples}"
        )
    if probabilities:
        probability_sum = float(sum(float(value) for value in probabilities.values()))
        if not np.isclose(probability_sum, 1.0, rtol=1e-12, atol=1e-12):
            errors.append(f"successor probabilities sum to {probability_sum}")
        if set(probabilities) != set(counts):
            errors.append("count and probability supports differ")
        elif total > 0:
            for signature, count in counts.items():
                expected = float(count) / float(total)
                if not np.isclose(
                    float(probabilities[signature]), expected, rtol=1e-12, atol=1e-12
                ):
                    errors.append("probability does not match normalized count")
                    break
    else:
        errors.append("missing normalized successor probabilities")

    if counts and full_nodes:
        graph = nx.Graph()
        for node in full_nodes:
            graph.add_node(node)
        for u, v in (example.get("full_graph_edges", ()) or ()):
            graph.add_edge(int(u), int(v))
        derived = gdm.derive_node_marginals_from_successor_distribution(
            successor_state_counts=counts,
            full_graph=graph,
            initial_global_state=_initial_full_state_as_global(example),
        )
        stored = dict(example.get("node_marginals", {}) or {})
        for node, expected in derived.items():
            actual = stored.get(node, stored.get(str(node), {}))
            for field in (
                "p_attacker_final",
                "p_defender_final",
                "expected_troops",
                "expected_troops_if_attacker",
                "expected_troops_if_defender",
                "p_changed_owner",
            ):
                if not np.isclose(
                    float(actual.get(field, float("nan"))),
                    float(expected[field]),
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=False,
                ):
                    errors.append(f"node marginal mismatch: node={node} field={field}")
                    break

    result = {
        "valid": not errors,
        "errors": tuple(dict.fromkeys(errors)),
        "count_sum": int(total),
        "probability_sum": float(sum(float(v) for v in probabilities.values())),
        "support_size": int(len(counts)),
    }
    if strict and errors:
        raise ValueError("Invalid Stage A V2 grouped example: " + "; ".join(result["errors"]))
    return result


def _failure_record(
    *,
    example_id: Optional[str],
    continent_name: str,
    attempt_index: int,
    state_generation_seed: Optional[int],
    target_seed: Optional[int],
    failure_stage: str,
    failure_type: str,
    message: str,
    traceback_text: Optional[str],
    initial_state_signature: Any,
    battle_graph_signature: Any,
    runtime_seconds: float,
) -> Dict[str, Any]:
    return {
        "example_id": example_id,
        "continent_name": str(continent_name),
        "attempt_index": int(attempt_index),
        "state_generation_seed": (
            None if state_generation_seed is None else int(state_generation_seed)
        ),
        "target_seed": None if target_seed is None else int(target_seed),
        "failure_stage": str(failure_stage),
        "failure_type": str(failure_type),
        "message": str(message),
        "traceback": traceback_text,
        "initial_state_signature": _json_ready(initial_state_signature),
        "battle_graph_signature": _json_ready(battle_graph_signature),
        "runtime_seconds": float(runtime_seconds),
        "created_utc": _utc_now(),
    }


def _graph_has_active_combat(graph: Any) -> bool:
    try:
        return int(graph.number_of_edges()) > 0
    except Exception:
        try:
            return bool(list(graph.edges()))
        except TypeError:
            return bool(list(graph.edges))


def _build_example_identity_and_seed(
    *,
    config: gdm.TransitionDistributionConfig,
    config_fingerprint: str,
    target_fingerprint: str,
    continent_name: str,
    perspective: str,
    global_state: GlobalState,
    battle_graph: Any,
    full_graph: Any,
    mc_samples: Optional[int] = None,
) -> Tuple[str, int, Tuple[Tuple[int, str, int], ...]]:
    initial_full_signature = gdm.lift_battle_signature_to_full_graph_signature(
        battle_signature=tuple(),
        initial_global_state=global_state,
        full_graph=full_graph,
    )
    example_id = gdm.canonical_transition_example_id(
        continent_name=continent_name,
        perspective=perspective,
        initial_full_graph_signature=initial_full_signature,
        battle_graph_signature=gdm.canonical_graph_signature(battle_graph),
        commitment_signature=None,
        target_generation_version=config.target_generation_version,
        two_stage_mc_seed=config.resolved_two_stage_mc_seed,
        generation_config_fingerprint=target_fingerprint,
    )
    samples = int(mc_samples or config.resolved_two_stage_mc_samples)
    target_seed = gdm.derive_transition_target_seed(
        base_seed=config.resolved_two_stage_mc_seed,
        example_id=example_id,
        mc_samples=samples,
        target_generation_version=config.target_generation_version,
    )
    return example_id, target_seed, initial_full_signature


def generate_transition_distribution_dataset_v2(
    *,
    output_dir: Path | str = DEFAULT_STAGE_A_V2_OUTPUT_DIR,
    config: Optional[gdm.TransitionDistributionConfig] = None,
    target_successes_by_continent: Optional[Mapping[str, int]] = None,
    max_attempts_multiplier: int = 20,
    random_seed: int = 42,
    state_generator: Optional[Callable[..., Any]] = None,
    example_collector: Callable[..., Mapping[str, Any]] = (
        gdm.collect_transition_distribution_example_for_state
    ),
    max_wall_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    config = config or gdm.TransitionDistributionConfig()
    validate_corrected_stage_a_v2_production_config(config)
    targets = dict(target_successes_by_continent or DEFAULT_TARGET_SUCCESSES)
    if any(int(value) < 0 for value in targets.values()):
        raise ValueError("target successes must be non-negative")
    if int(max_attempts_multiplier) < 1:
        raise ValueError("max_attempts_multiplier must be >= 1")
    if state_generator is None:
        from project_risk.mathematical.transition_prediction_ml.state_generators import ml_full_graph_state_generator

        state_generator = ml_full_graph_state_generator

    library_summary = inspect_transition_library_metadata(
        str(Path(config.combat_libraries_base).resolve())
    )
    store = StageAV2ChunkStore(
        output_dir=output_dir,
        base_config=config,
        library_metadata_summary=library_summary,
    )
    config_fingerprint = store.config_fingerprint
    target_fingerprint = store.target_fingerprint
    run_started = time.perf_counter()
    stopped_for_wall_time = False
    continent_results: Dict[str, Any] = {}

    for continent_name, target_successes in targets.items():
        target_successes = int(target_successes)
        checkpoint = store.load_checkpoint(continent_name)
        max_attempts = max(1, int(max_attempts_multiplier) * max(1, target_successes))
        while (
            int(checkpoint.get("success_count", 0)) < target_successes
            and int(checkpoint.get("attempt_count", 0)) < max_attempts
        ):
            if max_wall_seconds is not None and (
                time.perf_counter() - run_started >= float(max_wall_seconds)
            ):
                stopped_for_wall_time = True
                break
            attempt_index = int(checkpoint.get("attempt_count", 0))
            state_seed = derive_stage_a_state_generation_seed(
                base_seed=int(random_seed),
                continent_name=continent_name,
                attempt_index=attempt_index,
            )
            attempt_started = time.perf_counter()
            rng = np.random.default_rng(state_seed)
            target_territory_ratio = float(rng.uniform(0.2, 0.8))
            target_troops_ratio = float(rng.uniform(0.5, 2.0))
            constraints = gdm.ExperimentConstraints(
                continent_name=str(continent_name),
                max_attacker_troops_per_node=5,
                max_defender_troops_per_node=5,
            )
            try:
                players, battle_graph, full_graph = state_generator(
                    target_territory_ratio,
                    target_troops_ratio,
                    constraints,
                    rng,
                )
                base_global_state = agop.build_global_state_for_board(players)
            except Exception as exc:
                failure = _failure_record(
                    example_id=None,
                    continent_name=continent_name,
                    attempt_index=attempt_index,
                    state_generation_seed=state_seed,
                    target_seed=None,
                    failure_stage="state_generation",
                    failure_type=type(exc).__name__,
                    message=str(exc),
                    traceback_text=traceback_module.format_exc(),
                    initial_state_signature=None,
                    battle_graph_signature=None,
                    runtime_seconds=time.perf_counter() - attempt_started,
                )
                store.record_failure(
                    continent_name=continent_name,
                    failure=failure,
                    checkpoint=checkpoint,
                )
                store.finish_attempt(
                    continent_name,
                    checkpoint,
                    attempt_index,
                    elapsed_seconds=time.perf_counter() - attempt_started,
                )
                continue

            perspectives = ("P1_as_attacker", "P2_as_attacker")
            for perspective_index, perspective in enumerate(perspectives):
                if int(checkpoint.get("success_count", 0)) >= target_successes:
                    break
                if perspective_index == 0:
                    perspective_players = players
                    perspective_state = base_global_state
                    perspective_graph = battle_graph
                    gdm.apply_global_state_to_board(perspective_state, perspective_players)
                else:
                    perspective_state = gdm.swap_roles_in_global_state(base_global_state)
                    perspective_players = [players[1], players[0]]
                    gdm.apply_global_state_to_board(perspective_state, perspective_players)
                    perspective_graph = agop.build_continent_battle_graph(
                        continent_name, perspective_players, debug=False
                    )

                example_id, target_seed, initial_full_signature = (
                    _build_example_identity_and_seed(
                        config=config,
                        config_fingerprint=config_fingerprint,
                        target_fingerprint=target_fingerprint,
                        continent_name=continent_name,
                        perspective=perspective,
                        global_state=perspective_state,
                        battle_graph=perspective_graph,
                        full_graph=full_graph,
                    )
                )
                if example_id in set(checkpoint.get("completed_example_ids", ())):
                    continue
                if not _graph_has_active_combat(perspective_graph):
                    failure = _failure_record(
                        example_id=example_id,
                        continent_name=continent_name,
                        attempt_index=attempt_index,
                        state_generation_seed=state_seed,
                        target_seed=target_seed,
                        failure_stage="battle_graph_construction",
                        failure_type="expected_no_active_combat",
                        message="no_active_combat",
                        traceback_text=None,
                        initial_state_signature=initial_full_signature,
                        battle_graph_signature=gdm.canonical_graph_signature(
                            perspective_graph
                        ),
                        runtime_seconds=time.perf_counter() - attempt_started,
                    )
                    store.record_failure(
                        continent_name=continent_name,
                        failure=failure,
                        checkpoint=checkpoint,
                        expected_no_combat=True,
                    )
                    continue

                macro = gdm._build_transition_macro_features(
                    target_territory_ratio=target_territory_ratio,
                    target_troops_ratio=target_troops_ratio,
                    global_state=perspective_state,
                    players=perspective_players,
                    battle_graph=perspective_graph,
                    full_graph=full_graph,
                    continent_name=continent_name,
                    attack_perspective=perspective,
                )
                collection_started = time.perf_counter()
                try:
                    example = dict(
                        example_collector(
                            state_id=attempt_index,
                            players=perspective_players,
                            battle_graph=perspective_graph,
                            full_graph=full_graph,
                            global_state=perspective_state,
                            macro_features=macro,
                            continent_name=continent_name,
                            combat_libraries_base=Path(config.combat_libraries_base),
                            config=config,
                            attack_perspective=perspective,
                            example_id=example_id,
                            target_seed=target_seed,
                            config_fingerprint=config_fingerprint,
                            target_config_fingerprint=target_fingerprint,
                            library_metadata_summary=library_summary,
                            state_generation_seed=state_seed,
                            attempt_index=attempt_index,
                            commitment_signature=None,
                        )
                    )
                except Exception as exc:
                    failure = _failure_record(
                        example_id=example_id,
                        continent_name=continent_name,
                        attempt_index=attempt_index,
                        state_generation_seed=state_seed,
                        target_seed=target_seed,
                        failure_stage="second_stage_evaluation",
                        failure_type=type(exc).__name__,
                        message=str(exc),
                        traceback_text=traceback_module.format_exc(),
                        initial_state_signature=initial_full_signature,
                        battle_graph_signature=gdm.canonical_graph_signature(
                            perspective_graph
                        ),
                        runtime_seconds=time.perf_counter() - collection_started,
                    )
                    store.record_failure(
                        continent_name=continent_name,
                        failure=failure,
                        checkpoint=checkpoint,
                    )
                    continue

                status = str(example.get("transition_example_status"))
                if status != "ok":
                    stage = (
                        "partition_preparation"
                        if status == "no_candidate"
                        else "second_stage_evaluation"
                    )
                    failure = _failure_record(
                        example_id=example_id,
                        continent_name=continent_name,
                        attempt_index=attempt_index,
                        state_generation_seed=state_seed,
                        target_seed=target_seed,
                        failure_stage=stage,
                        failure_type=status,
                        message=str(example.get("transition_example_error") or status),
                        traceback_text=None,
                        initial_state_signature=initial_full_signature,
                        battle_graph_signature=gdm.canonical_graph_signature(
                            perspective_graph
                        ),
                        runtime_seconds=float(
                            example.get("target_generation_runtime_seconds", 0.0) or 0.0
                        ),
                    )
                    store.record_failure(
                        continent_name=continent_name,
                        failure=failure,
                        checkpoint=checkpoint,
                    )
                    continue

                try:
                    validate_grouped_transition_example_v2(example, strict=True)
                except Exception as exc:
                    failure = _failure_record(
                        example_id=example_id,
                        continent_name=continent_name,
                        attempt_index=attempt_index,
                        state_generation_seed=state_seed,
                        target_seed=target_seed,
                        failure_stage="target_packaging",
                        failure_type=type(exc).__name__,
                        message=str(exc),
                        traceback_text=traceback_module.format_exc(),
                        initial_state_signature=initial_full_signature,
                        battle_graph_signature=gdm.canonical_graph_signature(
                            perspective_graph
                        ),
                        runtime_seconds=time.perf_counter() - collection_started,
                    )
                    store.record_failure(
                        continent_name=continent_name,
                        failure=failure,
                        checkpoint=checkpoint,
                    )
                    continue
                store.record_success(
                    continent_name=continent_name,
                    example=example,
                    checkpoint=checkpoint,
                )

            store.finish_attempt(
                continent_name,
                checkpoint,
                attempt_index,
                elapsed_seconds=time.perf_counter() - attempt_started,
            )

        success_count = int(checkpoint.get("success_count", 0) or 0)
        if success_count >= target_successes:
            status = "completed"
        elif stopped_for_wall_time:
            status = "in_progress_with_checkpoint"
        else:
            status = "stopped_max_attempts"
        continent_results[str(continent_name)] = {
            "status": status,
            "target_successes": int(target_successes),
            "success_count": success_count,
            "attempt_count": int(checkpoint.get("attempt_count", 0) or 0),
            "failure_count": int(checkpoint.get("failure_count", 0) or 0),
            "expected_no_combat_count": int(
                checkpoint.get("expected_no_combat_count", 0) or 0
            ),
            "cumulative_runtime_seconds": float(
                checkpoint.get("cumulative_runtime_seconds", 0.0) or 0.0
            ),
            "checkpoint": str(store._checkpoint_path(continent_name)),
        }
        if stopped_for_wall_time:
            break

    result = {
        "target_generation_version": config.target_generation_version,
        "config_fingerprint": config_fingerprint,
        "target_config_fingerprint": target_fingerprint,
        "output_dir": str(Path(output_dir)),
        "status": (
            "completed"
            if all(item["status"] == "completed" for item in continent_results.values())
            and len(continent_results) == len(targets)
            else "in_progress_with_checkpoint"
        ),
        "runtime_seconds": float(time.perf_counter() - run_started),
        "continents": continent_results,
        "library_metadata_summary": library_summary,
    }
    _atomic_write_json(Path(output_dir) / "generation_status.json", result)
    return result


def select_transition_target_calibration_examples(
    grouped_examples_df: pd.DataFrame,
    *,
    fraction: float,
    random_seed: int,
    minimum_per_continent: int = 5,
    include_top_candidate_outliers: int = 5,
) -> pd.DataFrame:
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if grouped_examples_df.empty or float(fraction) == 0.0:
        return grouped_examples_df.iloc[0:0].copy()
    frame = grouped_examples_df.copy()
    if frame["example_id"].astype(str).duplicated().any():
        raise ValueError("calibration input contains duplicate example IDs")
    if "candidate_count_category" not in frame:
        frame["candidate_count_category"] = frame[
            "num_retained_second_stage_candidates"
        ].map(gdm._candidate_count_category)
    if "battle_node_count_category" not in frame:
        frame["battle_node_count_category"] = frame["battle_node_count"].map(
            gdm._battle_node_count_category
        )
    frame["_selection_score"] = frame["example_id"].astype(str).map(
        lambda example_id: _stable_int_seed(
            {
                "kind": "stage_a_v2_calibration_selection",
                "seed": int(random_seed),
                "example_id": example_id,
            }
        )
    )
    selected: set[str] = set()

    outlier_count = min(max(0, int(include_top_candidate_outliers)), len(frame))
    outlier_ids: set[str] = set()
    if outlier_count:
        outliers = frame.sort_values(
            [
                "num_retained_second_stage_candidates",
                "battle_node_count",
                "target_generation_runtime_seconds",
                "example_id",
            ],
            ascending=[False, False, False, True],
        ).head(outlier_count)
        outlier_ids = set(outliers["example_id"].astype(str))
        selected.update(outlier_ids)

    for continent_name, continent_df in frame.groupby("continent_name", sort=True):
        desired = min(
            len(continent_df),
            max(
                int(minimum_per_continent),
                int(math.ceil(float(fraction) * len(continent_df))),
            ),
        )
        continent_selected = {
            example_id
            for example_id in selected
            if example_id in set(continent_df["example_id"].astype(str))
        }
        strata = list(
            continent_df.groupby(
                ["candidate_count_category", "battle_node_count_category"],
                sort=True,
            )
        )
        candidates_by_stratum = [
            group.sort_values(["_selection_score", "example_id"])
            for _, group in strata
        ]
        cursor = 0
        while len(continent_selected) < desired and candidates_by_stratum:
            progress = False
            for group in candidates_by_stratum:
                available = group[
                    ~group["example_id"].astype(str).isin(continent_selected)
                ]
                if available.empty:
                    continue
                example_id = str(available.iloc[cursor % len(available)]["example_id"])
                continent_selected.add(example_id)
                progress = True
                if len(continent_selected) >= desired:
                    break
            if not progress:
                break
            cursor += 1
        if len(continent_selected) < desired:
            remaining = continent_df[
                ~continent_df["example_id"].astype(str).isin(continent_selected)
            ].sort_values(["_selection_score", "example_id"])
            continent_selected.update(
                remaining.head(desired - len(continent_selected))["example_id"].astype(str)
            )
        selected.update(continent_selected)

    result = frame[frame["example_id"].astype(str).isin(selected)].copy()
    result["calibration_selection_seed"] = int(random_seed)
    result["calibration_selection_fraction"] = float(fraction)
    result["calibration_selected_as_candidate_outlier"] = result[
        "example_id"
    ].astype(str).isin(outlier_ids)
    return result.drop(columns=["_selection_score"]).sort_values(
        ["continent_name", "example_id"]
    ).reset_index(drop=True)


def _distribution_probabilities(example: Mapping[str, Any]) -> Dict[Any, float]:
    probabilities = dict(
        example.get("full_graph_successor_state_probabilities", {}) or {}
    )
    if probabilities:
        total = float(sum(float(value) for value in probabilities.values()))
        return {
            signature: float(value) / total
            for signature, value in probabilities.items()
            if float(value) > 0.0
        }
    counts = dict(example.get("full_graph_successor_state_counts", {}) or {})
    total = float(sum(float(value) for value in counts.values()))
    return {
        signature: float(value) / total
        for signature, value in counts.items()
        if total > 0.0 and float(value) > 0.0
    }


def _node_marginal_map(example: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    return {
        int(node): dict(values)
        for node, values in (example.get("node_marginals", {}) or {}).items()
    }


def _strategic_distribution_summary(example: Mapping[str, Any]) -> Dict[str, float]:
    probabilities = _distribution_probabilities(example)
    battle_nodes = {int(node) for node in (example.get("battle_graph_nodes", ()) or ())}
    expected_attacker_territories = 0.0
    expected_attacker_troops = 0.0
    expected_defender_troops = 0.0
    conquest_probability = 0.0
    for signature, probability in probabilities.items():
        state = {int(node): (str(owner), int(troops)) for node, owner, troops in signature}
        expected_attacker_territories += probability * sum(
            1 for node in battle_nodes if state[node][0] == "A"
        )
        expected_attacker_troops += probability * sum(
            state[node][1] for node in battle_nodes if state[node][0] == "A"
        )
        expected_defender_troops += probability * sum(
            state[node][1] for node in battle_nodes if state[node][0] == "D"
        )
        if battle_nodes and all(state[node][0] == "A" for node in battle_nodes):
            conquest_probability += probability
    return {
        "expected_attacker_owned_territories": float(expected_attacker_territories),
        "expected_attacker_troop_total": float(expected_attacker_troops),
        "expected_defender_troop_total": float(expected_defender_troops),
        "probability_of_local_conquest": float(conquest_probability),
    }


def compare_mc5_mc20_transition_targets(
    mc5_example: Mapping[str, Any],
    mc20_example: Mapping[str, Any],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    left = _distribution_probabilities(mc5_example)
    right = _distribution_probabilities(mc20_example)
    union = set(left) | set(right)
    intersection = set(left) & set(right)
    tv = 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in union)
    js = 0.0
    for key in union:
        p = float(left.get(key, 0.0))
        q = float(right.get(key, 0.0))
        m = 0.5 * (p + q)
        if p > 0.0:
            js += 0.5 * p * math.log(p / m)
        if q > 0.0:
            js += 0.5 * q * math.log(q / m)

    left_top = [
        key for key, _ in sorted(left.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]
    right_top = [
        key for key, _ in sorted(right.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]
    left_marginals = _node_marginal_map(mc5_example)
    right_marginals = _node_marginal_map(mc20_example)
    marginal_nodes = sorted(set(left_marginals) | set(right_marginals))
    ownership_differences = [
        abs(
            float(left_marginals.get(node, {}).get("p_attacker_final", 0.0))
            - float(right_marginals.get(node, {}).get("p_attacker_final", 0.0))
        )
        for node in marginal_nodes
    ]
    troop_differences = [
        abs(
            float(left_marginals.get(node, {}).get("expected_troops", 0.0))
            - float(right_marginals.get(node, {}).get("expected_troops", 0.0))
        )
        for node in marginal_nodes
    ]
    left_strategic = _strategic_distribution_summary(mc5_example)
    right_strategic = _strategic_distribution_summary(mc20_example)
    strategic_differences = {
        key + "_difference": float(right_strategic[key] - left_strategic[key])
        for key in left_strategic
    }
    selected_partition_equal = (
        mc5_example.get("selected_partition_signature")
        == mc20_example.get("selected_partition_signature")
    )
    selected_policy_equal = (
        tuple(mc5_example.get("selected_region_option_indices", ()) or ())
        == tuple(mc20_example.get("selected_region_option_indices", ()) or ())
    )
    left_runtime = float(mc5_example.get("target_generation_runtime_seconds", 0.0) or 0.0)
    right_runtime = float(mc20_example.get("target_generation_runtime_seconds", 0.0) or 0.0)
    return {
        "base_example_id": str(mc5_example.get("example_id")),
        "calibration_example_id": str(mc20_example.get("example_id")),
        "continent_name": mc5_example.get("continent_name"),
        "candidate_count_category": mc5_example.get("candidate_count_category"),
        "battle_node_count_category": mc5_example.get("battle_node_count_category"),
        "total_variation_distance": float(tv),
        "jensen_shannon_divergence": float(js),
        "support_intersection_size": int(len(intersection)),
        "support_union_size": int(len(union)),
        "mc20_mass_on_mc5_support": float(sum(right.get(key, 0.0) for key in left)),
        "mc5_mass_on_mc20_support": float(sum(left.get(key, 0.0) for key in right)),
        "top1_state_agreement": bool(left_top[:1] == right_top[:1]),
        "top_k": int(top_k),
        "top_k_state_overlap_count": int(len(set(left_top) & set(right_top))),
        "top_k_state_overlap_fraction": float(
            len(set(left_top) & set(right_top)) / max(1, len(set(left_top) | set(right_top)))
        ),
        "maximum_ownership_probability_difference": float(max(ownership_differences, default=0.0)),
        "mean_ownership_probability_difference": float(np.mean(ownership_differences)) if ownership_differences else 0.0,
        "maximum_expected_troop_difference": float(max(troop_differences, default=0.0)),
        "mean_expected_troop_difference": float(np.mean(troop_differences)) if troop_differences else 0.0,
        "mc5_strategic_summary": left_strategic,
        "mc20_strategic_summary": right_strategic,
        **strategic_differences,
        "selected_partition_equal": bool(selected_partition_equal),
        "selected_policy_equal": bool(selected_policy_equal),
        "candidate_selection_changed": bool(
            not selected_partition_equal or not selected_policy_equal
        ),
        "sampled_successor_distribution_changed": bool(tv > 1e-12),
        "mc5_runtime_seconds": left_runtime,
        "mc20_runtime_seconds": right_runtime,
        "runtime_ratio_mc20_over_mc5": (
            float(right_runtime / left_runtime) if left_runtime > 0.0 else None
        ),
    }


def run_transition_target_calibration_v2(
    *,
    output_dir: Path | str,
    config: gdm.TransitionDistributionConfig,
    minimum_per_continent: int = 5,
    include_top_candidate_outliers: int = 5,
    example_collector: Callable[..., Mapping[str, Any]] = (
        gdm.collect_transition_distribution_example_for_state
    ),
    max_wall_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    if config.calibration_mc_samples is None:
        raise ValueError("calibration_mc_samples is None")
    base_examples = load_stage_a_v2_grouped_examples(output_dir)
    selected = select_transition_target_calibration_examples(
        base_examples,
        fraction=float(config.calibration_fraction),
        random_seed=int(config.calibration_seed),
        minimum_per_continent=int(minimum_per_continent),
        include_top_candidate_outliers=int(include_top_candidate_outliers),
    )
    calibration_config = replace(
        config,
        two_stage_mc_samples=int(config.calibration_mc_samples),
        two_stage_mc_scenarios=None,
        two_stage_rng_seed=None,
    )
    validate_corrected_stage_a_v2_production_config(calibration_config)
    library_summary = inspect_transition_library_metadata(
        str(Path(config.combat_libraries_base).resolve())
    )
    store = StageAV2ChunkStore(
        output_dir=output_dir,
        base_config=config,
        library_metadata_summary=library_summary,
        phase="calibration_mc20",
        active_config=calibration_config,
    )
    started = time.perf_counter()
    completed = 0
    failures = 0
    stopped = False

    checkpoints = {
        continent: store.load_checkpoint(continent)
        for continent in sorted(set(selected.get("continent_name", ())))
    }
    for base_example in selected.to_dict(orient="records"):
        if max_wall_seconds is not None and time.perf_counter() - started >= float(
            max_wall_seconds
        ):
            stopped = True
            break
        continent_name = str(base_example["continent_name"])
        checkpoint = checkpoints[continent_name]
        global_state = _global_state_from_raw_signature(
            base_example.get("initial_global_state_signature")
        )
        battle_graph = _graph_from_signature(base_example.get("battle_graph_signature"))
        full_graph = _graph_from_signature(base_example.get("full_graph_signature"))
        players = [Players.Player("A"), Players.Player("D")]
        gdm.apply_global_state_to_board(global_state, players)
        calibration_id, target_seed, initial_full_signature = (
            _build_example_identity_and_seed(
                config=calibration_config,
                config_fingerprint=store.config_fingerprint,
                target_fingerprint=store.target_fingerprint,
                continent_name=continent_name,
                perspective=str(base_example.get("attack_perspective")),
                global_state=global_state,
                battle_graph=battle_graph,
                full_graph=full_graph,
            )
        )
        if calibration_id in set(checkpoint.get("completed_example_ids", ())):
            completed += 1
            continue
        item_started = time.perf_counter()
        try:
            calibrated = dict(
                example_collector(
                    state_id=int(base_example.get("state_id", -1)),
                    players=players,
                    battle_graph=battle_graph,
                    full_graph=full_graph,
                    global_state=global_state,
                    macro_features=dict(base_example.get("macro_features", {}) or {}),
                    continent_name=continent_name,
                    combat_libraries_base=Path(config.combat_libraries_base),
                    config=calibration_config,
                    attack_perspective=str(base_example.get("attack_perspective")),
                    example_id=calibration_id,
                    target_seed=target_seed,
                    config_fingerprint=store.config_fingerprint,
                    target_config_fingerprint=store.target_fingerprint,
                    library_metadata_summary=library_summary,
                    state_generation_seed=base_example.get("state_generation_seed"),
                    attempt_index=base_example.get("attempt_index"),
                    commitment_signature=base_example.get("commitment_signature"),
                )
            )
            calibrated["base_example_id"] = str(base_example.get("example_id"))
            calibrated["calibration_example_id"] = calibration_id
            calibrated["calibration_mc_samples"] = int(
                calibration_config.resolved_two_stage_mc_samples
            )
            if calibrated.get("transition_example_status") != "ok":
                raise RuntimeError(
                    calibrated.get("transition_example_error")
                    or calibrated.get("transition_example_status")
                )
            validate_grouped_transition_example_v2(calibrated, strict=True)
            store.record_success(
                continent_name=continent_name,
                example=calibrated,
                checkpoint=checkpoint,
            )
            completed += 1
        except Exception as exc:
            failure = _failure_record(
                example_id=calibration_id,
                continent_name=continent_name,
                attempt_index=int(base_example.get("attempt_index", 0) or 0),
                state_generation_seed=base_example.get("state_generation_seed"),
                target_seed=target_seed,
                failure_stage="second_stage_evaluation",
                failure_type=type(exc).__name__,
                message=str(exc),
                traceback_text=traceback_module.format_exc(),
                initial_state_signature=initial_full_signature,
                battle_graph_signature=base_example.get("battle_graph_signature"),
                runtime_seconds=time.perf_counter() - item_started,
            )
            store.record_failure(
                continent_name=continent_name,
                failure=failure,
                checkpoint=checkpoint,
            )
            failures += 1
        checkpoint["attempt_count"] = int(checkpoint.get("attempt_count", 0) or 0) + 1
        store._save_checkpoint(continent_name, checkpoint)

    calibration_examples = load_stage_a_v2_grouped_examples(
        output_dir, phase="calibration_mc20"
    )
    base_by_id = {
        str(row["example_id"]): row for row in base_examples.to_dict(orient="records")
    }
    comparisons = []
    for calibrated in calibration_examples.to_dict(orient="records"):
        base_example = base_by_id.get(str(calibrated.get("base_example_id")))
        if base_example is not None:
            comparisons.append(
                compare_mc5_mc20_transition_targets(base_example, calibrated)
            )
    comparison_df = pd.DataFrame(comparisons)
    phase_root = Path(output_dir) / "calibration_mc20"
    _atomic_write_pickle(phase_root / "comparisons.pkl", comparison_df)
    _atomic_write_csv(phase_root / "comparisons.csv", comparison_df)
    result = {
        "status": "in_progress_with_checkpoint" if stopped else "completed",
        "selected_examples": int(len(selected)),
        "completed_examples": int(len(calibration_examples)),
        "failures_this_run": int(failures),
        "comparison_pairs": int(len(comparison_df)),
        "runtime_seconds": float(time.perf_counter() - started),
        "output_dir": str(phase_root),
    }
    _atomic_write_json(phase_root / "calibration_status.json", result)
    return result


def _numeric_summary(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    numeric_values: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            numeric_values.append(numeric)
    clean = np.asarray(numeric_values, dtype=float)
    if clean.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "p90": float(np.quantile(clean, 0.90)),
        "p95": float(np.quantile(clean, 0.95)),
        "max": float(np.max(clean)),
    }


def _reuse_rates(frame: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    def rate(numerator: str, denominator: str, *, invert: bool = False) -> List[float]:
        values = []
        for row in frame.to_dict(orient="records"):
            num = float(row.get(numerator, 0.0) or 0.0)
            den = float(row.get(denominator, 0.0) or 0.0)
            if den > 0.0:
                values.append((den - num) / den if invert else num / den)
        return values

    query_rates = []
    for row in frame.to_dict(orient="records"):
        hits = float(row.get("regional_query_cache_hits", 0.0) or 0.0)
        misses = float(row.get("regional_query_cache_misses", 0.0) or 0.0)
        if hits + misses > 0.0:
            query_rates.append(hits / (hits + misses))
    return {
        "regional_sample_reuse_rate": _numeric_summary(
            rate(
                "num_unique_regional_samples",
                "num_regional_sample_requests",
                invert=True,
            )
        ),
        "complete_state_cache_hit_rate": _numeric_summary(
            rate("num_global_state_cache_hits", "num_successor_states_assembled")
        ),
        "regional_query_cache_hit_rate": _numeric_summary(query_rates),
    }


def _difficulty_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    total = max(1, len(frame))
    output: Dict[str, Any] = {}
    for category, group in frame.groupby("candidate_count_category", sort=True):
        output[str(category)] = {
            "count": int(len(group)),
            "fraction": float(len(group) / total),
            "runtime_seconds": _numeric_summary(
                group["target_generation_runtime_seconds"].tolist()
            ),
            "successor_support_size": _numeric_summary(
                group["successor_support_size"].tolist()
            ),
        }
    return output


def _frame_summary(
    frame: pd.DataFrame,
    *,
    attempts: int,
    expected_no_combat: int,
    failures: int,
    cumulative_runtime_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    runtimes = frame.get("target_generation_runtime_seconds", pd.Series(dtype=float))
    successful_runtime_total = float(
        pd.to_numeric(runtimes, errors="coerce").fillna(0.0).sum()
    )
    cumulative_runtime = (
        None
        if cumulative_runtime_seconds is None
        else max(0.0, float(cumulative_runtime_seconds))
    )
    runtime_total = (
        cumulative_runtime
        if cumulative_runtime is not None and cumulative_runtime > 0.0
        else successful_runtime_total
    )
    example_outcomes = len(frame) + int(expected_no_combat) + int(failures)
    return {
        "state_generation_attempts": int(attempts),
        "successful_examples": int(len(frame)),
        "expected_no_combat_states": int(expected_no_combat),
        "failures": int(failures),
        "example_success_rate": float(len(frame) / example_outcomes) if example_outcomes else 0.0,
        "total_runtime_seconds": runtime_total,
        "successful_target_runtime_seconds": successful_runtime_total,
        "examples_per_hour": float(3600.0 * len(frame) / runtime_total) if runtime_total > 0 else None,
        "runtime_seconds": _numeric_summary(runtimes.tolist()),
        "maximal_partition_count": _numeric_summary(
            frame.get("num_maximal_partitions", pd.Series(dtype=float)).tolist()
        ),
        "retained_candidate_count": _numeric_summary(
            frame.get(
                "num_retained_second_stage_candidates", pd.Series(dtype=float)
            ).tolist()
        ),
        "unique_regional_option_count": _numeric_summary(
            frame.get("num_unique_regional_options", pd.Series(dtype=float)).tolist()
        ),
        "successor_support_size": _numeric_summary(
            frame.get("successor_support_size", pd.Series(dtype=float)).tolist()
        ),
        "cache_and_reuse": _reuse_rates(frame),
        "difficulty_strata": _difficulty_summary(frame) if not frame.empty else {},
    }


def _load_checkpoint_counts(
    output_dir: Path, manifest: Mapping[str, Any], *, phase: Optional[str] = None
) -> Dict[str, Dict[str, int]]:
    continents = (
        manifest.get("continents", {})
        if phase is None
        else manifest.get("phases", {}).get(phase, {}).get("continents", {})
    )
    output = {}
    for continent, record in continents.items():
        checkpoint_path = output_dir / record["checkpoint"]
        checkpoint = (
            json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint_path.exists()
            else {}
        )
        output[str(continent)] = {
            "attempt_count": int(checkpoint.get("attempt_count", 0) or 0),
            "failure_count": int(checkpoint.get("failure_count", 0) or 0),
            "expected_no_combat_count": int(
                checkpoint.get("expected_no_combat_count", 0) or 0
            ),
            "cumulative_runtime_seconds": float(
                checkpoint.get("cumulative_runtime_seconds", 0.0) or 0.0
            ),
        }
    return output


def summarize_transition_distribution_dataset_v2(
    *, output_dir: Path | str
) -> Dict[str, Any]:
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    examples = load_stage_a_v2_grouped_examples(root)
    counts = _load_checkpoint_counts(root, manifest)
    per_continent = {}
    example_continents = (
        set(examples["continent_name"].astype(str))
        if not examples.empty and "continent_name" in examples
        else set()
    )
    for continent_name in sorted(set(counts) | example_continents):
        group = (
            examples[examples["continent_name"].astype(str) == str(continent_name)]
            if not examples.empty and "continent_name" in examples
            else pd.DataFrame()
        )
        count = counts.get(str(continent_name), {})
        per_continent[str(continent_name)] = _frame_summary(
            group,
            attempts=count.get("attempt_count", 0),
            expected_no_combat=count.get("expected_no_combat_count", 0),
            failures=count.get("failure_count", 0),
            cumulative_runtime_seconds=count.get("cumulative_runtime_seconds"),
        )
    overall_counts = {
        key: sum(item.get(key, 0) for item in counts.values())
        for key in (
            "attempt_count",
            "failure_count",
            "expected_no_combat_count",
            "cumulative_runtime_seconds",
        )
    }
    summary = {
        "target_generation_version": manifest.get("target_generation_version"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "output_dir": str(root),
        "overall": _frame_summary(
            examples,
            attempts=overall_counts["attempt_count"],
            expected_no_combat=overall_counts["expected_no_combat_count"],
            failures=overall_counts["failure_count"],
            cumulative_runtime_seconds=overall_counts["cumulative_runtime_seconds"],
        ),
        "per_continent": per_continent,
    }

    comparison_path = root / "calibration_mc20" / "comparisons.pkl"
    if comparison_path.exists():
        comparisons = pd.read_pickle(comparison_path)
    else:
        comparisons = pd.DataFrame()
    if comparisons.empty:
        calibration = {"pair_count": 0}
    else:
        calibration = {
            "pair_count": int(len(comparisons)),
            "selection_change_fraction": float(
                comparisons["candidate_selection_changed"].astype(float).mean()
            ),
            "total_variation_distance": _numeric_summary(
                comparisons["total_variation_distance"].tolist()
            ),
            "jensen_shannon_divergence": _numeric_summary(
                comparisons["jensen_shannon_divergence"].tolist()
            ),
            "maximum_ownership_probability_difference": _numeric_summary(
                comparisons["maximum_ownership_probability_difference"].tolist()
            ),
            "mean_ownership_probability_difference": _numeric_summary(
                comparisons["mean_ownership_probability_difference"].tolist()
            ),
            "maximum_expected_troop_difference": _numeric_summary(
                comparisons["maximum_expected_troop_difference"].tolist()
            ),
            "mean_expected_troop_difference": _numeric_summary(
                comparisons["mean_expected_troop_difference"].tolist()
            ),
            "expected_attacker_owned_territories_difference": _numeric_summary(
                comparisons[
                    "expected_attacker_owned_territories_difference"
                ].tolist()
            ),
            "expected_attacker_troop_total_difference": _numeric_summary(
                comparisons["expected_attacker_troop_total_difference"].tolist()
            ),
            "expected_defender_troop_total_difference": _numeric_summary(
                comparisons["expected_defender_troop_total_difference"].tolist()
            ),
            "probability_of_local_conquest_difference": _numeric_summary(
                comparisons["probability_of_local_conquest_difference"].tolist()
            ),
            "mc20_mass_on_mc5_support": _numeric_summary(
                comparisons["mc20_mass_on_mc5_support"].tolist()
            ),
            "mc5_mass_on_mc20_support": _numeric_summary(
                comparisons["mc5_mass_on_mc20_support"].tolist()
            ),
            "top1_state_agreement_fraction": float(
                comparisons["top1_state_agreement"].astype(float).mean()
            ),
            "top_k_state_overlap_fraction": _numeric_summary(
                comparisons["top_k_state_overlap_fraction"].tolist()
            ),
            "runtime_ratio_mc20_over_mc5": _numeric_summary(
                comparisons["runtime_ratio_mc20_over_mc5"].tolist()
            ),
        }
    summary["calibration"] = calibration

    outlier_columns = [
        "example_id",
        "continent_name",
        "attack_perspective",
        "num_retained_second_stage_candidates",
        "successor_support_size",
        "target_generation_runtime_seconds",
        "candidate_count_category",
    ]
    summary["outliers"] = {
        "runtime": examples.sort_values(
            "target_generation_runtime_seconds", ascending=False
        ).head(10)[outlier_columns].to_dict(orient="records")
        if not examples.empty
        else [],
        "candidate_count": examples.sort_values(
            "num_retained_second_stage_candidates", ascending=False
        ).head(10)[outlier_columns].to_dict(orient="records")
        if not examples.empty
        else [],
        "successor_support_size": examples.sort_values(
            "successor_support_size", ascending=False
        ).head(10)[outlier_columns].to_dict(orient="records")
        if not examples.empty
        else [],
        "calibration_tv": comparisons.sort_values(
            "total_variation_distance", ascending=False
        ).head(10).to_dict(orient="records")
        if not comparisons.empty
        else [],
        "calibration_strategic": comparisons.assign(
            _strategic_abs=comparisons[
                [
                    "expected_attacker_owned_territories_difference",
                    "expected_attacker_troop_total_difference",
                    "expected_defender_troop_total_difference",
                    "probability_of_local_conquest_difference",
                ]
            ].abs().max(axis=1)
        ).sort_values("_strategic_abs", ascending=False).head(10).drop(
            columns=["_strategic_abs"]
        ).to_dict(orient="records")
        if not comparisons.empty
        else [],
    }
    _atomic_write_json(root / "summary.json", summary)
    return summary


def validate_transition_distribution_dataset_v2(
    *,
    output_dir: Path | str,
    strict: bool = True,
) -> Dict[str, Any]:
    root = Path(output_dir)
    errors: List[str] = []
    config_path = root / "config.json"
    manifest_path = root / "manifest.json"
    if not config_path.exists():
        errors.append("missing config.json")
    if not manifest_path.exists():
        errors.append("missing manifest.json")
    if errors:
        result = {"valid": False, "errors": tuple(errors)}
        if strict:
            raise ValueError("; ".join(errors))
        return result

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fingerprint = config_payload.get("config_fingerprint")
    try:
        reconstructed_config = gdm.TransitionDistributionConfig(
            **dict(config_payload.get("configuration", {}) or {})
        )
        computed_fingerprint = gdm.transition_distribution_config_fingerprint(
            reconstructed_config
        )
        if str(computed_fingerprint) != str(expected_fingerprint):
            errors.append("config.json fingerprint does not match its configuration")
    except Exception as exc:
        errors.append(f"config.json configuration cannot be reconstructed: {exc}")
    if manifest.get("config_fingerprint") != expected_fingerprint:
        errors.append("manifest configuration fingerprint differs from config.json")
    if manifest.get("format") != STAGE_A_V2_OUTPUT_FORMAT:
        errors.append("unexpected manifest format")

    all_manifest_files: List[Path] = []
    manifest_sections = [manifest.get("continents", {})] + [
        phase.get("continents", {})
        for phase in manifest.get("phases", {}).values()
    ]
    for section in manifest_sections:
        for record in section.values():
            for field in (
                "grouped_chunks",
                "node_marginal_chunks",
                "failure_files",
            ):
                all_manifest_files.extend(root / rel for rel in record.get(field, ()))
            if record.get("checkpoint"):
                all_manifest_files.append(root / record["checkpoint"])
            record_root = root / str(record.get("directory", ""))
            for directory_name, manifest_key in (
                ("grouped_examples", "grouped_chunks"),
                ("node_marginals", "node_marginal_chunks"),
            ):
                discovered = {
                    str(path.relative_to(root))
                    for path in (record_root / directory_name).glob("chunk_*.pkl")
                }
                registered = {str(value) for value in record.get(manifest_key, ())}
                orphaned = sorted(discovered - registered)
                if orphaned:
                    errors.append(
                        f"unregistered {directory_name} chunks: {orphaned[:5]}"
                    )
    manifest_paths = [str(path) for path in all_manifest_files]
    if len(manifest_paths) != len(set(manifest_paths)):
        errors.append("manifest lists one or more files more than once")
    missing_files = [str(path) for path in all_manifest_files if not path.exists()]
    if missing_files:
        errors.append(f"manifest files missing: {missing_files[:5]}")

    base_examples = load_stage_a_v2_grouped_examples(root)
    if not base_examples.empty and base_examples["example_id"].astype(str).duplicated().any():
        errors.append("duplicate grouped example IDs")
    base_ids = set(base_examples.get("example_id", pd.Series(dtype=str)).astype(str))
    for example in base_examples.to_dict(orient="records"):
        validation = validate_grouped_transition_example_v2(example, strict=False)
        if not validation["valid"]:
            errors.append(
                f"invalid grouped example {example.get('example_id')}: {validation['errors']}"
            )
        if str(example.get("config_fingerprint")) != str(expected_fingerprint):
            errors.append(f"row config fingerprint mismatch: {example.get('example_id')}")

    fixed_nodes: Dict[str, Tuple[int, ...]] = {}
    for example in base_examples.to_dict(orient="records"):
        continent = str(example.get("continent_name"))
        nodes = tuple(int(node) for node in (example.get("full_graph_nodes", ()) or ()))
        if continent in fixed_nodes and fixed_nodes[continent] != nodes:
            errors.append(f"full-graph node order differs within {continent}")
        fixed_nodes.setdefault(continent, nodes)

    for continent, record in manifest.get("continents", {}).items():
        checkpoint_path = root / record["checkpoint"]
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("config_fingerprint") != expected_fingerprint:
                errors.append(f"checkpoint fingerprint mismatch: {continent}")
            completed = set(str(value) for value in checkpoint.get("completed_example_ids", ()))
            continent_ids = set(
                base_examples.loc[
                    base_examples["continent_name"].astype(str) == str(continent),
                    "example_id",
                ].astype(str)
            ) if not base_examples.empty else set()
            if not completed.issubset(continent_ids):
                errors.append(f"checkpoint completed IDs missing from {continent} outputs")

        node_frames = [
            pd.read_pickle(root / rel)
            for rel in record.get("node_marginal_chunks", ())
            if (root / rel).exists()
        ]
        node_df = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
        if not node_df.empty:
            if node_df.duplicated(subset=["example_id", "node_index"]).any():
                errors.append(f"duplicate node-marginal rows in {continent}")
            grouped_continent = (
                base_examples[
                    base_examples["continent_name"].astype(str) == str(continent)
                ]
                if not base_examples.empty and "continent_name" in base_examples
                else pd.DataFrame()
            )
            expected_pairs = {
                (str(row["example_id"]), int(node))
                for row in grouped_continent.to_dict(orient="records")
                for node in (row.get("full_graph_nodes", ()) or ())
            }
            actual_pairs = {
                (str(row["example_id"]), int(row["node_index"]))
                for row in node_df.to_dict(orient="records")
            }
            if expected_pairs != actual_pairs:
                errors.append(f"node-marginal coverage differs from grouped rows in {continent}")
            grouped_by_id = {
                str(row["example_id"]): row
                for row in grouped_continent.to_dict(orient="records")
            }
            for node_row in node_df.to_dict(orient="records"):
                example = grouped_by_id.get(str(node_row.get("example_id")))
                if example is None:
                    continue
                node = int(node_row["node_index"])
                marginal = dict(example.get("node_marginals", {}) or {}).get(
                    node,
                    dict(example.get("node_marginals", {}) or {}).get(str(node), {}),
                )
                for field in (
                    "p_attacker_final",
                    "p_defender_final",
                    "expected_troops",
                    "expected_troops_if_attacker",
                    "expected_troops_if_defender",
                    "p_changed_owner",
                ):
                    if not np.isclose(
                        float(node_row.get(field, float("nan"))),
                        float(marginal.get(field, float("nan"))),
                        rtol=1e-10,
                        atol=1e-10,
                        equal_nan=False,
                    ):
                        errors.append(
                            f"node-marginal value mismatch in {continent}: "
                            f"example={node_row.get('example_id')} node={node} field={field}"
                        )
                        break

    calibration_phase = manifest.get("phases", {}).get("calibration_mc20")
    calibration_examples = (
        load_stage_a_v2_grouped_examples(root, phase="calibration_mc20")
        if calibration_phase is not None
        else pd.DataFrame()
    )
    calibration_ids = set(
        calibration_examples.get("example_id", pd.Series(dtype=str)).astype(str)
    )
    phase_fingerprint = None
    if calibration_phase is not None:
        phase_config_path = root / "calibration_mc20" / "config.json"
        if not phase_config_path.exists():
            errors.append("missing calibration_mc20/config.json")
        else:
            phase_config = json.loads(phase_config_path.read_text(encoding="utf-8"))
            phase_fingerprint = phase_config.get("config_fingerprint")
            if str(calibration_phase.get("config_fingerprint")) != str(
                phase_fingerprint
            ):
                errors.append("calibration manifest/config fingerprint mismatch")
            try:
                reconstructed_phase_config = gdm.TransitionDistributionConfig(
                    **dict(phase_config.get("configuration", {}) or {})
                )
                if str(
                    gdm.transition_distribution_config_fingerprint(
                        reconstructed_phase_config
                    )
                ) != str(phase_fingerprint):
                    errors.append(
                        "calibration config fingerprint does not match its configuration"
                    )
            except Exception as exc:
                errors.append(f"calibration configuration cannot be reconstructed: {exc}")

        if not calibration_examples.empty and calibration_examples[
            "example_id"
        ].astype(str).duplicated().any():
            errors.append("duplicate calibration example IDs")
        for example in calibration_examples.to_dict(orient="records"):
            validation = validate_grouped_transition_example_v2(example, strict=False)
            if not validation["valid"]:
                errors.append(
                    f"invalid calibration example {example.get('example_id')}: {validation['errors']}"
                )
            if str(example.get("config_fingerprint")) != str(phase_fingerprint):
                errors.append("calibration row fingerprint mismatch")
            if str(example.get("base_example_id")) not in base_ids:
                errors.append("calibration row links to missing MC5 example")

        for continent, record in calibration_phase.get("continents", {}).items():
            checkpoint_path = root / record["checkpoint"]
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if str(checkpoint.get("config_fingerprint")) != str(phase_fingerprint):
                    errors.append(
                        f"calibration checkpoint fingerprint mismatch: {continent}"
                    )
                completed = {
                    str(value)
                    for value in checkpoint.get("completed_example_ids", ())
                }
                if not completed.issubset(calibration_ids):
                    errors.append(
                        f"calibration checkpoint IDs missing from {continent} outputs"
                    )

            node_frames = [
                pd.read_pickle(root / rel)
                for rel in record.get("node_marginal_chunks", ())
                if (root / rel).exists()
            ]
            node_df = (
                pd.concat(node_frames, ignore_index=True)
                if node_frames
                else pd.DataFrame()
            )
            grouped_continent = (
                calibration_examples[
                    calibration_examples["continent_name"].astype(str)
                    == str(continent)
                ]
                if not calibration_examples.empty
                and "continent_name" in calibration_examples
                else pd.DataFrame()
            )
            expected_pairs = {
                (str(row["example_id"]), int(node))
                for row in grouped_continent.to_dict(orient="records")
                for node in (row.get("full_graph_nodes", ()) or ())
            }
            actual_pairs = {
                (str(row["example_id"]), int(row["node_index"]))
                for row in node_df.to_dict(orient="records")
            }
            if expected_pairs != actual_pairs:
                errors.append(
                    f"calibration node-marginal coverage differs in {continent}"
                )

    comparison_path = root / "calibration_mc20" / "comparisons.pkl"
    if comparison_path.exists():
        comparisons = pd.read_pickle(comparison_path)
        if not comparisons.empty:
            if comparisons.duplicated(
                subset=["base_example_id", "calibration_example_id"]
            ).any():
                errors.append("duplicate MC5/MC20 comparison links")
            if not set(comparisons["base_example_id"].astype(str)).issubset(base_ids):
                errors.append("comparison links to missing MC5 examples")
            if not set(
                comparisons["calibration_example_id"].astype(str)
            ).issubset(calibration_ids):
                errors.append("comparison links to missing MC20 examples")

    result = {
        "valid": not errors,
        "errors": tuple(dict.fromkeys(errors)),
        "manifest_file_count": int(len(all_manifest_files)),
        "base_example_count": int(len(base_examples)),
        "calibration_example_count": int(len(calibration_examples)),
        "unique_base_example_ids": int(len(base_ids)),
        "fixed_full_graph_nodes_by_continent": fixed_nodes,
        "config_fingerprint": expected_fingerprint,
    }
    _atomic_write_json(root / "validation.json", result)
    if strict and errors:
        raise ValueError("Stage A V2 dataset validation failed: " + "; ".join(result["errors"]))
    return result
