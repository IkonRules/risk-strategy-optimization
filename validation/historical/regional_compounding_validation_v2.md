# Regional Compounding Validation V2

## Scope

- Completed benchmark records: `50`.
- V1 MC1 metric reproduction within 1e-10: `True`.
- Candidate source: the persisted corrected-mode retained candidate set; no tolerance or candidate/policy-combination cap was introduced.
- Production routing and Stage A/B/C/D behavior were not modified.

## Candidate Selection

- MC1 and exact regional candidate identity agreement: `35/50`.
- Candidate identity changed: `15`; changed but distribution-equal: `4`.
- Material MC1-to-exact-regional distribution changes (TV >= 0.05): `11`.
- States with exact best-candidate ties: `29`; maximum tie count: `7`.
- Full-reference TV improved / unchanged / worsened after exact regional selection: `3` / `42` / `5`.
- Mean exact-minus-MC1 full-reference TV delta: `0.00971087`.

## Error Decomposition

| Comparison | Mean TV | Mean JS | Mean balanced Wasserstein | Mean ownership error | Mean troop error | Mean conquest error | Mean covariance error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MC1 regional vs full exact | 0.257805 | 0.170086 | 0.0540807 | 0.024788 | 0.124902 | 0.0110233 | 0.00609561 |
| Exact regional vs full exact | 0.267516 | 0.176767 | 0.0588707 | 0.0298447 | 0.123879 | 0.0112497 | 0.00743208 |
| MC1 regional vs exact regional | 0.0954659 | 0.0619787 | 0.0194373 | 0.0115436 | 0.0300664 | 0.000226386 | 0.00400698 |

## Double Front

- Double-front records: `10`; candidate changed in `5`.
- Mean TV, MC1 vs full / exact regional vs full: `0.797696` / `0.799118`.
- Previous TV=1 failures: `7`; still TV=1: `7`; still TV >= 0.9: `7`.
- Severe descriptor counts: `{'shared_troop_source_present': 0, 'sequence_opening_present': 8, 'articulation_present': 0}`.
- Attack-order switching, conditional policy switching, and survivor redistribution are not inferred because a contingent exact policy DAG is unavailable.

## Cache Reuse

- Raw successor-state requests: `7311`.
- Unique successor states evaluated: `2710`.
- Global evaluation cache hits / misses: `4601` / `2710`.

## Topology Means

| Topology | Records | MC1-full TV | Exact-regional-full TV | MC1-exact-regional TV |
| --- | ---: | ---: | ---: | ---: |
| articulation | 9 | 0.153758 | 0.1918 | 0.208206 |
| bridge | 9 | 0.00611656 | 3.30575e-09 | 0.00611656 |
| chain | 1 | 0 | 0 | 0 |
| cycle | 9 | 0.265401 | 0.285845 | 0.0470298 |
| double_front | 10 | 0.797696 | 0.799118 | 0.242112 |
| sequence_opening | 9 | 0.120644 | 0.120644 | 0 |
| star | 1 | 4.76837e-09 | 4.76837e-09 | 0 |
| tree | 1 | 1.2035e-08 | 1.2035e-08 | 0 |
| two_dense | 1 | 0 | 0 | 0 |

## Interpretation

Exact regional candidate selection removes Monte Carlo candidate-selection noise but does not remove the regional decomposition or independence assumptions.

The comparison between the exact regional-model distribution and the full-graph exact distribution isolates structural approximation error more cleanly than the previous MC=1 benchmark.

A distribution-value gap is not called policy regret unless the regional candidate can be evaluated as a complete contingent policy under the exact full-graph dynamics.

Concrete good, bad, and mixed examples use authoritative project territory names and preserve the actual graph adjacency.

The mapping audit distinguishes induced embeddings from edge-preserving non-induced display mappings and lists every extra board edge for the latter.

Production routing is not changed in this task; the expanded exact-solver results are used to recommend a later exact-first routing policy.

Stage A regeneration, Stage B retraining, and Stage E remain blocked until these results have been reviewed.
