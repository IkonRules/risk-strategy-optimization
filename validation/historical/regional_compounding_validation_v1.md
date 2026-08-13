# Regional Compounding Accuracy Validation v1

## Scope and status

This additive framework compares the current production regional-compounding path with an explicit compact exact full-graph reference. Production partition-selection semantics remain `maximal_per_partition_utility` with exact equality, no utility tolerance, no candidate cap, and no policy-combination cap.

All 50 focused benchmark records completed the exact reference, production approximation, and exact selected-candidate composition. The output validator reports zero errors and zero failure files.

## Files changed

- `exact_finite_solver.py`: additive runtime/state limits and controlled `ExactSolverLimitReached` status.
- `distribution_comparison_metrics.py`: distribution, state-aware transport, marginal, strategic, and dependence metrics.
- `regional_compounding_validation.py`: exact composition/reference boundaries, production approximation adapter, benchmark generation, persistence, summaries, and validation.
- `run_regional_compounding_validation.py`: tractability, benchmark, summary, validate, filtering, resume, and limit CLI.
- `_regional_validation_test_helpers.py` and 18 focused `test_*` scripts listed in the task.

## Solver capabilities and boundaries

The compact exact solver uses the existing finite battle and movement semantics. It supports `local` and `legacy` utilities, shared value/distribution caches, tied root-policy options, and root/state-set library policy modes. The validation reference preserves all tied root actions with canonical optimal continuation. It does not enumerate all combinations of downstream tie-resolved policies, and records that limitation explicitly.

The full-reference tractability pilot tested up to 8 nodes, 4 attacker plus 4 defender nodes, and troop cap 5. The solver is not hard-coded to that boundary, but larger cases were not validated here. Distribution caching can consume substantial memory as reachable-state and absorbing-support counts grow; direct process RSS was not measured.

## Public validation APIs

- `compose_selected_candidate_distribution_exact(...)` prepares each regional option once, multiplies exact regional outcomes region by region, and merges duplicate global states after every region.
- Composition statuses are `exact_complete`, `unique_state_limit`, `cartesian_expansion_limit`, `runtime_limit`, `invalid_region_overlap`, `probability_error`, and `unsupported_payload`. Incomplete results are never silently pruned or renormalized.
- `solve_full_graph_exact_reference(...)` returns the optimal lexicographic value, canonical policy/distribution, tied-root policy set/distributions, cache counters, action counts, limits, and diagnostics.
- `evaluate_policy_under_exact_full_graph_model(...)` exactly evaluates an explicit `exact_root_policy_v1` policy with canonical optimal continuation.
- `evaluate_regional_compounding_approximation(...)` calls the corrected production candidate preparation and second-stage selection path, then exact-composes the selected candidate.

Regional V2 payloads contain terminal distributions but not a complete full-graph state-to-action policy. Therefore exact lifted-policy regret is unavailable for all 50 focused records. `distribution_value_primary_gap` is reported separately and is not labeled policy regret. The exact-policy evaluation API itself is tested and reproduces the exact reference value/distribution for a representable policy.

## Metrics

The framework records TV, natural-log JS divergence, mass overlap, support overlap, top-k overlap, and exact min-cost-flow Wasserstein distance under ownership-dominant, balanced, and troop-dominant profiles. The per-node distance is the mean of weighted owner mismatch plus weighted troop difference divided by a fixed comparison-wide troop scale.

It also records node ownership/troop marginals, troop PMFs, strategic event probabilities, one-dimensional strategic-summary distributions, expectations, variances, tails, cross-region joint probabilities, covariance, correlation, conditional probabilities, and successful-region-count distributions.

Topology descriptors include articulations, partition-boundary nodes/edges, active A-D boundary edges, shared potential troop sources, defender paths that create sequence openings, attacker/defender cross-region paths, and cycles crossing boundaries. Exact action alternation is marked unavailable because compact payloads do not preserve full traces.

## Tractability pilot

- Grid: nodes 6/7/8, caps 3/4/5, 8 topology families, 5 deterministic state strata per cell.
- Cases/cells: 360 cases across 72 cells.
- Exact completions: 360/360 under 5 seconds and 100,000 evaluated states.
- Worst observed cell: 8-node cap-5 `double_front`.
- Worst exact runtime: 0.783527 seconds.
- Maximum states evaluated: 15,518.
- Maximum canonical absorbing support: 1,617.
- Maximum tied-root policy count: 4.

This pilot ran the exact reference only. Production approximation and composition runtimes were measured in the 50-state focused benchmark. The practical validated exact boundary is 8 nodes at cap 5 for this suite, not a proof that all 8-node cap-5 topologies and states are feasible.

## Exact composition versus target Monte Carlo

Real selected candidate: `regional_benchmark_8f8adbe9c6e1c87ef8176b4c`, exact support 9.

| Samples | TV | JS | Mean ownership error | Mean troop error |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.044445 | 0.002148 | 0.007888 | 0.005000 |
| 1,000 | 0.017603 | 0.000547 | 0.002278 | 0.003833 |
| 10,000 | 0.003499 | 0.000018 | 0.000295 | 0.000517 |

Exact composition took 0.000496 seconds. The nested 10,000-sample run took 4.008617 seconds. The synthetic convergence test also passed.

## Focused benchmark

The 50 records include a 45-cell grid over 6/7/8 nodes, caps 3/4/5, and cycle/bridge/double-front/articulation/sequence-opening families, one cap-2 smoke state, and four cap-3 chain/star/tree/two-dense cases. Candidate selection used one Monte Carlo scenario to bound runtime; target composition was exact. Policy-selection conclusions are therefore provisional and candidate-selection stability remains a separate concern.

| Metric | Mean | Median | p90 | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distribution-value primary gap | 0.160191 | 0.000000004 | 0.602383 | 1.037603 | 1.543085 |
| TV | 0.257805 | 0.000000008 | 1.000000 | 1.000000 | 1.000000 |
| JS | 0.170086 | 0.000000000 | 0.693147 | 0.693147 | 0.693147 |
| Balanced Wasserstein | 0.054081 | 0.000000001 | 0.238482 | 0.318353 | 0.364709 |
| Mean ownership marginal error | 0.024788 | 0.000000001 | 0.095587 | 0.158745 | 0.197613 |
| Mean troop marginal error | 0.124902 | 0.000000002 | 0.367720 | 0.627870 | 1.122663 |
| Conquest probability error | 0.011023 | 0.000000 | 0.012863 | 0.079033 | 0.258168 |
| No-gain probability error | 0.065940 | 0.000000001 | 0.261548 | 0.448901 | 0.882667 |
| Territory expectation error | 0.159715 | 0.000000004 | 0.601573 | 1.034933 | 1.542053 |
| Attacker-troop expectation error | 0.416306 | 0.000000004 | 1.482316 | 2.395975 | 3.314323 |
| Maximum covariance error | 0.006096 | 0.000000 | 0.022476 | 0.044426 | 0.060901 |

Exact reference runtime averaged 0.007490 seconds. Production approximation runtime averaged 5.722942 seconds and reached 53.922276 seconds. Exact composition averaged 0.000440 seconds, reached 0.001626 seconds, and had maximum support 88 and maximum raw expansion count 99.

Thirty-one of 50 states have TV at most 0.05. Thirteen match an exact-optimal root-policy distribution within TV `1e-10`. Fifteen have TV at least 0.20. Seven have disjoint support and TV 1.0; all seven are double-front cases.

## Group patterns

- Bridge, 9 states: mean TV 0.006117, max 0.055049.
- Articulation, 9 states: mean TV 0.153758, median near zero, max 0.657179.
- Sequence-opening, 9 states: mean TV 0.120644, median near zero, max 0.856860.
- Cycle, 9 states: mean TV 0.265401, median 0.132012, max 0.856860.
- Double-front, 10 states: mean TV 0.797696, median 1.0, mean distribution-value gap 0.710014, max 1.543085.
- The single chain/star/tree/two-dense additions are near exact, but one case per family is not enough for a family-level conclusion.

Mean TV by graph size is 0.212494 for 6 nodes, 0.233768 for 7, and 0.342257 for 8. Mean TV by cap is 0.145582 at cap 3, 0.248443 at cap 4, and 0.426502 at cap 5; suite composition differs by cap, so this is descriptive.

One-region cases, 30 states, have mean TV 0.084156. Two-region cases, 20 states, have mean TV 0.518278. TV correlation is 0.543 with region count, 0.636 with retained-candidate count, 0.401 with tied-root policy count, 0.306 with cap, and 0.134 with graph size. These are descriptive associations, not causal estimates. Articulation and sequence-opening group flags are strongly confounded with region count in this suite.

Among the 15 high-TV states, 12 place the largest node error on a partition boundary. Maximum cross-region covariance error is 0.060901. This shows measurable missing dependence in some coupled cases, but covariance alone is smaller than the full joint-support discrepancy.

## Outlier

Worst value-gap state: `regional_benchmark_daebef5997a634f72b38f4af`.

- Graph/state: 8-node cap-5 double front; `(A1,A1,A5,A3,D4,D2,D1,D4)`.
- Selected partition: `(1,2,3,5,6)` plus `(4,7,8)`; 9 retained candidates.
- Exact tied root actions: `(4,8)` and `(3,7)`.
- Exact value: `(1.656337, 5.724084, 0.002274)`.
- Compounded distribution-implied value: `(0.113251, 8.113251, 0.0)`.
- Exact/approximate supports: 37/6 with zero intersection, TV 1.0.
- Three graph edges cross the selected partition. The largest node error is on boundary node 7.

The likely cause is not generic articulation structure. The selected regional policy cannot represent the full graph's cross-region sequence and troop-flow choices, and independently composing its regional terminal outcomes produces a support disjoint from the exact global policy. One-sample candidate selection may amplify this, so a matched higher-sample rerun is required before attributing the entire gap to partition structure.

## Primary questions

1. Optimal policy selection cannot yet be measured exactly for regional V2 candidates because no liftable full state-to-action policy is stored. The terminal distribution-value gap is often near zero but has a large upper tail, and MC=1 selection makes this provisional.
2. Strategic events are often accurate but have important outliers: mean/max conquest error is 0.011023/0.258168, no-gain error 0.065940/0.882667, and key-territory capture error 0.067382/0.754557.
3. Median node marginal errors are near numerical zero, but maxima reach 0.197613 ownership probability and 1.122663 expected troops per node.
4. The complete joint distribution is not reliably reproduced on coupled cases: mean TV is 0.257805 and seven states have disjoint supports.
5. Boundary concentration is present in the high-error subset: 12/15 high-TV cases have their largest node error on a partition boundary. Articulation conclusions are confounded and not causal.
6. The exact model shows dependence absent from independent composition in some cases, with covariance error up to 0.060901 and larger joint-support differences.
7. Error is associated with more regions and retained candidates, but sparse candidate-count groups and topology confounding prevent a causal interpretation.
8. The approximation is most reliable here for bridge/weakly coupled and one-region cases.
9. It is least reliable for double fronts, cross-boundary cycles, shared or sequential troop-flow opportunities, and cases where the exact opening action crosses the selected partition logic.
10. Exact selected-candidate composition is easily tractable in this focused suite and is a strong replacement for target-distribution Monte Carlo after candidate selection.

## Recommendations

Use exact selected-candidate composition for future Stage A labels when its explicit limits complete. Preserve a controlled fallback status for cases that exceed limits; do not silently prune or renormalize. This removes target Monte Carlo noise without changing the regional-independence model.

Do not treat the regional approximation as globally reliable. It is promising for weakly coupled/one-region cases but requires guardrails, larger exact regions or explicit global handling for double fronts and sequence-sensitive partitions. Before a policy-quality conclusion, rerun representative hard cases with stable candidate-selection checkpoints and add a liftable full-policy representation if true policy regret is required.

## Verification

- `py_compile`: pass.
- New validation tests: 18/18 pass.
- Exact finite solver: 1/1 pass.
- Two-stage regressions: 20/20 pass.
- Partition-selection and exact-cover regressions: 33/33 pass.
- Owner-role regressions: 6/6 pass.
- Stage A v2 regressions: 13/13 pass.
- Stage A v3 regressions: 14/14 pass.
- Standard-ranking regression: pass.
- Saved output validation: 50 records, zero failures, valid.

## Required interpretations

No single distribution metric is treated as a complete measure of approximation quality.

Policy regret, strategic-event accuracy, node marginals, full joint-distribution similarity, and cross-region dependence are evaluated separately.

Exact selected-candidate composition enumerates the same regional-independence model currently estimated by Monte Carlo; it does not introduce additional independence assumptions.

The explicit full-graph reference is used to measure the accuracy of the regional-independence and partition-policy approximation itself.

Large discrepancies are interpreted according to which tests fail and which graph structures produce them.

Stage A regeneration, Stage B retraining, and Stage E remain blocked until the benchmark results have been reviewed.
