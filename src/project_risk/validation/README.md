# Reusable validation source

This cross-cutting package contains **CURRENT_REUSABLE** and
**HISTORICAL_REUSABLE** metrics and checks for exact solvers, policy libraries,
regional composition, transition distributions, and training readiness. Main
modules include `distribution_comparison_metrics.py`, `exact_policy_dag.py`,
the regional compounding validators, library validators,
`transition_distribution_validation.py`, and `preflight_checks_training.py`.
Some checks consume generated libraries, datasets, or model outputs; expensive
suites and generated results are not distributed. Compact public checks remain
under the repository-level `tests/` and `validation/` directories.
