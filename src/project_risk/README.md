# Project Risk research source

This package is the authoritative reusable research implementation. The compact
runnable example in `examples/run_exact_example.py` imports this package
directly.

- `game_simulation/` is the original mutable board/player platform.
- `mathematical/` contains the graph-scale strategy pipeline.
- `infrastructure/` contains narrowly shared support code.
- `validation/` contains reusable scientific checks, not generated results.

The package spans local tactical graphs, continent-scale battle graphs, and the
complete multi-turn board. Its exact core is current reusable code; the later
transition and full-board routes include historical reusable and active
experimental components. Large exact libraries, datasets, trained models, and
experiment outputs are intentionally not distributed.
