from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from project_risk.mathematical.transition_prediction_ml import generate_data_ML as gdm
from project_risk.mathematical.transition_prediction_ml.transition_distribution_stage_a_v2 import (
    DEFAULT_STAGE_A_V2_OUTPUT_DIR,
    DEFAULT_TARGET_SUCCESSES,
    generate_transition_distribution_dataset_v2,
    run_transition_target_calibration_v2,
    summarize_transition_distribution_dataset_v2,
    validate_transition_distribution_dataset_v2,
)


def _optional_float(value: str) -> Optional[float]:
    return None if value.strip().lower() == "none" else float(value)


def _optional_int(value: str) -> Optional[int]:
    return None if value.strip().lower() == "none" else int(value)


def _flatten_names(values: Optional[Iterable[str]]) -> List[str]:
    names: List[str] = []
    for value in values or ():
        names.extend(item.strip() for item in str(value).split(",") if item.strip())
    return names


def _parse_target_overrides(values: Optional[Iterable[str]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in _flatten_names(values):
        if "=" not in item:
            raise ValueError(
                "Target overrides must use CONTINENT=COUNT, for example Australia=10"
            )
        continent, count = item.rsplit("=", 1)
        result[continent.strip()] = int(count)
    return result


def _load_saved_config(output_dir: Path) -> gdm.TransitionDistributionConfig:
    path = output_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"No Stage A V2 config found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = dict(payload.get("configuration", {}) or {})
    return gdm.TransitionDistributionConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate corrected, versioned Stage A transition targets."
    )
    parser.add_argument("--continents", nargs="+", help="Names or comma-separated names")
    parser.add_argument("--target-successes", type=int)
    parser.add_argument(
        "--target-successes-by-continent",
        nargs="+",
        metavar="CONTINENT=COUNT",
    )
    parser.add_argument("--max-attempts-multiplier", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-wall-seconds", type=float)

    parser.add_argument("--two-stage-mc-samples", type=int, default=5)
    parser.add_argument("--two-stage-mc-seed", type=int, default=42)
    parser.add_argument(
        "--second-stage-execution-mode", default="optimized_reuse"
    )
    parser.add_argument(
        "--second-stage-sampling-mode", default="stable_region_option_scenarios"
    )
    parser.add_argument(
        "--partition-candidate-selection-mode",
        default="maximal_per_partition_utility",
    )
    parser.add_argument("--utility-abs-tolerance", type=_optional_float, default=None)
    parser.add_argument("--utility-rel-tolerance", type=_optional_float, default=None)
    parser.add_argument(
        "--max-candidates-per-partition", type=_optional_int, default=None
    )
    parser.add_argument("--max-partitions", type=int, default=40)
    parser.add_argument("--combat-libraries-base", default="small_graph_libraries")

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STAGE_A_V2_OUTPUT_DIR)
    parser.add_argument("--output-chunk-size", type=int, default=100)
    parser.add_argument("--checkpoint-every-examples", type=int, default=1)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--calibration-mc-samples", type=_optional_int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-seed", type=int, default=42020)
    parser.add_argument("--calibration-minimum-per-continent", type=int, default=5)
    parser.add_argument("--calibration-top-candidate-outliers", type=int, default=5)
    parser.add_argument("--calibration-max-wall-seconds", type=float)

    parser.add_argument("--validate-output", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _build_config(args: argparse.Namespace) -> gdm.TransitionDistributionConfig:
    return gdm.TransitionDistributionConfig(
        two_stage_mc_samples=args.two_stage_mc_samples,
        two_stage_mc_seed=args.two_stage_mc_seed,
        max_partitions=args.max_partitions,
        combat_libraries_base=args.combat_libraries_base,
        partition_candidate_selection_mode=args.partition_candidate_selection_mode,
        utility_abs_tolerance=args.utility_abs_tolerance,
        utility_rel_tolerance=args.utility_rel_tolerance,
        max_candidates_per_partition=args.max_candidates_per_partition,
        second_stage_execution_mode=args.second_stage_execution_mode,
        second_stage_sampling_mode=args.second_stage_sampling_mode,
        output_chunk_size=args.output_chunk_size,
        checkpoint_every_examples=args.checkpoint_every_examples,
        resume=args.resume,
        calibration_mc_samples=args.calibration_mc_samples,
        calibration_fraction=args.calibration_fraction,
        calibration_seed=args.calibration_seed,
    )


def _target_map(args: argparse.Namespace) -> Dict[str, int]:
    continents = _flatten_names(args.continents) or list(DEFAULT_TARGET_SUCCESSES)
    unknown = sorted(set(continents) - set(DEFAULT_TARGET_SUCCESSES))
    if unknown:
        raise ValueError(f"Unknown continents: {unknown}")
    targets = {
        continent: (
            int(args.target_successes)
            if args.target_successes is not None
            else int(DEFAULT_TARGET_SUCCESSES[continent])
        )
        for continent in continents
    }
    overrides = _parse_target_overrides(args.target_successes_by_continent)
    unknown_overrides = sorted(set(overrides) - set(DEFAULT_TARGET_SUCCESSES))
    if unknown_overrides:
        raise ValueError(f"Unknown target-override continents: {unknown_overrides}")
    for continent, count in overrides.items():
        if continent in targets:
            targets[continent] = int(count)
    return targets


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    results: Dict[str, object] = {"output_dir": str(output_dir)}

    if args.validation_only:
        results["validation"] = validate_transition_distribution_dataset_v2(
            output_dir=output_dir, strict=True
        )
        results["summary"] = summarize_transition_distribution_dataset_v2(
            output_dir=output_dir
        )
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
        return 0
    if args.summary_only:
        results["summary"] = summarize_transition_distribution_dataset_v2(
            output_dir=output_dir
        )
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
        return 0

    if args.calibration_only:
        config = _load_saved_config(output_dir)
    else:
        config = _build_config(args)
        results["configuration"] = asdict(config)
        results["generation"] = generate_transition_distribution_dataset_v2(
            output_dir=output_dir,
            config=config,
            target_successes_by_continent=_target_map(args),
            max_attempts_multiplier=args.max_attempts_multiplier,
            random_seed=args.random_seed,
            max_wall_seconds=args.max_wall_seconds,
        )

    if args.run_calibration or args.calibration_only:
        results["calibration"] = run_transition_target_calibration_v2(
            output_dir=output_dir,
            config=config,
            minimum_per_continent=args.calibration_minimum_per_continent,
            include_top_candidate_outliers=args.calibration_top_candidate_outliers,
            max_wall_seconds=args.calibration_max_wall_seconds,
        )

    results["summary"] = summarize_transition_distribution_dataset_v2(
        output_dir=output_dir
    )
    if args.validate_output:
        results["validation"] = validate_transition_distribution_dataset_v2(
            output_dir=output_dir, strict=True
        )
    print(json.dumps(results, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
