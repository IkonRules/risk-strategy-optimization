# Portfolio Build Manifest

Development-only provenance record for the public portfolio extraction.

## Preservation boundary

- Original research archive: `<research-archive>/project_risk`
- Curated public repository: `<public-repository>/risk-strategy-optimization`
- Build date: 2026-08-13
- Rule: the original archive is read-only source material. No source, data,
  generated library, validation output, Git metadata, or historical artifact is
  modified or removed during this extraction.

## Selected original files (recorded before copying)

| Original path | Public path | Original SHA-256 | Reason for inclusion |
|---|---|---|---|
| `markov_matrix_probabilities.py` | `risk_strategy/markov_matrix_probabilities.py` | `B41F5C03E290B982331B7D56AF8EA40E9EE8A7D98376DABCAEB1BC0A380BEE09` | Absorbing Markov whole-battle transition kernel. |
| `small_graph_outcome_probabilities.py` | `risk_strategy/small_graph_outcome_probabilities.py` | `42F26A0517E844D8DED05844DB5673E32E27D655C0993C403E0101E873DD6BF3` | Graph state, legal-action, movement, utility, canonicalization, and reference semantics required by the compact solver. |
| `exact_finite_solver.py` | `risk_strategy/exact_finite_solver.py` | `C8358C401079B618B5442BC11EC1B231E7E5A121044BF7F4A81978A7FF5A566D` | Compact shared-cache exact finite-state optimizer. |
| `exact_policy_dag.py` | `risk_strategy/exact_policy_dag.py` | `327A9B6496A6511D1B2253FB10F78A0CE074DD67E712A15CADC3AD797A6EE1DB` | Validation-oriented exact policy DAG and explicit optimal-tie representation. |
| `distribution_comparison_metrics.py` | `risk_strategy/distribution_comparison_metrics.py` | `C8AAFC40950C11EC51BF64A80D1C32E871FDAD31A5ED26EB8653F80216BBFE83` | Total variation and other distribution-level validation metrics. |
| `<authoritative-context>/MODEL_DEVELOPMENT_HISTORY.md` | `docs/MODEL_DEVELOPMENT_HISTORY.md` | `72F3C6349993FA41E34711E1C00314C859D53749A370DE0023C046558A77AE4B` | Authoritative multi-year methodological history. |
| `<authoritative-context>/MODEL_DEVELOPMENT_HISTORY_2025-11_2025-12.md` | `docs/MODEL_DEVELOPMENT_HISTORY_2025-11_2025-12.md` | `FF1796909A82FC157A337D0DD99AC8810021C0D5091FD80B162868A252907BB2` | Authoritative reconstruction of the macro-statistical phase. |
| `regional_compounding_validation_v1/report.md` | `validation/historical/regional_compounding_validation_v1.md` | `2DE26941E591D3FDB29E0B80349D4B5982CC14E6BB3B1A3C5FD4EAA4E9DA07A6` | Compact saved report supporting exact tractability, exact-vs-Monte-Carlo, and bridge/double-front findings. |
| `regional_compounding_validation_v2_exact_candidate_selection/reports/benchmark_report.md` | `validation/historical/regional_compounding_validation_v2.md` | `4D7B7FDF2075B8683E2ED78FFD269DCBE69DDE0C813E65C3A4D40D30917A39A6` | Compact saved report supporting exact candidate-selection and structural-error findings. |
| `exact_policy_dag_branching_validation_v1/reports/exact_tie_distribution_report.md` | `validation/historical/exact_tie_distribution_report.md` | `7122E0753E91FB04F82D3F852BDD2808B2D4159A4130808350D888EE3153D690` | Compact saved table supporting policy-tie distribution findings. |
| `exact_policy_dag_branching_validation_v1/reports/policy_dag_summary.json` | `validation/historical/policy_dag_summary.json` | `F2C8DF12198D4D3FCC7EEAB1163988F9770DE4F4A7227AF426C0F62462D97E14` | Machine-readable saved summary supporting the 120-record integrity, canonical invariance, and maximum tied-policy TV findings. |
| `exact_policy_dag_branching_validation_v1/reports/double_front_macro_region_report.md` | `validation/historical/double_front_macro_region_report.md` | `C0E408309634CA9D75F74B38BD07907FC8B62F29B069438BE2A00062056F048D` | Compact saved report supporting the coupled macro-region experiment and its full-graph limitation. |

## Explicitly excluded from the selected code boundary

`create_library.py`, `library_io.py`, the production-like regional path, ML
pipelines, and full-board modules are not needed by the small exact example or
public tests. Their methods and results are documented, but copying them would
expand dependencies and expose partially integrated or artifact-dependent
paths. Generated libraries, datasets, model binaries, raw validation records,
third-party PDFs, commercial artwork, caches, backups, and bulk historical code
are excluded.

## Post-copy change classification

| Public file | Public SHA-256 | Classification |
|---|---|---|
| `risk_strategy/markov_matrix_probabilities.py` | `B41F5C03E290B982331B7D56AF8EA40E9EE8A7D98376DABCAEB1BC0A380BEE09` | Unchanged. |
| `risk_strategy/small_graph_outcome_probabilities.py` | `70D834563C46468946E0323473CFC4F1EC5F9E86E1BE42362CCD264905C5E7DD` | Import/path portability only: one absolute intra-project import became a relative package import. |
| `risk_strategy/exact_finite_solver.py` | `8598242A046795FBC640B5E4D4815CA84C20C12C96C7BC48B7108C1441E0C156` | Import/path portability only: two intra-project imports became relative package imports. |
| `risk_strategy/exact_policy_dag.py` | `C49B36B39539363C13F60A9540F8F084130001D23B042C4AA05BCBEC2F6D834D` | Import/path portability only: one intra-project import became a relative package import. |
| `risk_strategy/distribution_comparison_metrics.py` | `C8AAFC40950C11EC51BF64A80D1C32E871FDAD31A5ED26EB8653F80216BBFE83` | Unchanged. |
| `docs/MODEL_DEVELOPMENT_HISTORY.md` | `777BECEA32A118BF641D1AD7BBC37EF8FC5E139FD57FEDBA5DE9F2ECCA98418C` | Documentation/privacy cleanup: two private conversation identifiers were removed; methodological text was not rewritten. |
| `docs/MODEL_DEVELOPMENT_HISTORY_2025-11_2025-12.md` | `FF1796909A82FC157A337D0DD99AC8810021C0D5091FD80B162868A252907BB2` | Unchanged. |
| `validation/historical/*` | Individual hashes above | Unchanged copies. |

No genuine behavioural change was made to the copied mathematical core. The
new `risk_strategy/demo.py` is packaging/example code that calls the original
solver and policy-DAG interfaces; it does not replace or simplify them.

## Clean-environment verification

Verified on 2026-08-13 in a fresh temporary Python 3.12 virtual environment:

- installed only `requirements.txt` (`numpy==2.5.2`, `pandas==2.3.3`, and their
  transitive dependencies);
- `python examples/run_exact_example.py`: pass;
- `python -m unittest discover -s tests -v`: 13/13 pass;
- `python -m compileall -q risk_strategy examples tests validation`: pass;
- `python -m pip check`: no broken requirements;
- no generated library, dataset, model, PDF, artwork, or original-project path
  was required.

The temporary virtual environment and generated bytecode caches were removed
after verification.
