# Double-Front Macro-Region Results

The combined macro-region is the union of the prior independent exact-cover regions. In these stored double-front benchmarks that union is the complete graph, so its exact solve is identical to the full exact solve and safely reuses the same policy DAG.

| Benchmark | Full/macro root | Independent roots | Macro equals full | Full-macro TV | Macro-independent TV | Outcome-dependent switch | Sequence opening | Cross-partition followup |
| --- | --- | --- | :---: | ---: | ---: | :---: | :---: | :---: |
| regional_benchmark_536940d4a51038a5a12f3ec1 | `(1, 4)` | `((1, 4), (3, 6))` | True | 0 | 0.991175 | True | True | True |
| regional_benchmark_7de096e6084abbc6badf0cca | `(2, 5)` | `(None, (2, 5))` | True | 0 | 1 | True | True | True |
| regional_benchmark_aede86e9dd3dc84b7adf5930 | `(2, 5)` | `((1, 4), (3, 6))` | True | 0 | 1 | True | True | True |
| regional_benchmark_b86948934a74bed925904eb8 | `(1, 5)` | `((1, 5), (4, 8))` | True | 0 | 1 | False | True | True |
| regional_benchmark_d15e8a271c34a173a29c8ae9 | `(3, 6)` | `((1, 4), (2, 5))` | True | 0 | 1 | True | True | True |
| regional_benchmark_daebef5997a634f72b38f4af | `(3, 7)` | `(None, (4, 8))` | True | 0 | 1 | True | True | True |
| regional_benchmark_e7207e413ffdbf1bc390816c | `(3, 6)` | `(None, None)` | True | 0 | 1 | False | True | True |
| regional_benchmark_ecc97bd375bca2f43d6f49f7 | `(1, 5)` | `((1, 5), (4, 8))` | True | 0 | 1 | True | True | True |

Deeper policy branching can represent contingent decisions within one exact graph or macro-region, but it cannot repair interactions between regions that remain independently composed.
