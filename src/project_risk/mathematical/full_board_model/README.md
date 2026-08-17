# Full-board model

**ACTIVE_EXPERIMENTAL** complete-board mathematical rollouts add player
alternation, multiple continents, shared-front commitments, reinforcement,
allocation, reallocation, and fortification. Main modules are
`full_board_state_generators.py`, `full_board_simulation_ML.py`,
`full_board_simulation_GT.py`, and `strategy_policy_gt.py`.

`full_board_simulation_ML.py` contains both a historical deterministic node-marginal route and a later joint-state particle route. `full_board_simulation_GT.py` currently uses historical node-level learned transitions. Neither is the latest exact-first continent architecture integrated end to end. Required model bundles are not distributed.

The resulting multi-turn states feed `strategic_evaluation/` and optional demo
rendering. This layer does not execute turns through the original
`SimulationEngine`.
