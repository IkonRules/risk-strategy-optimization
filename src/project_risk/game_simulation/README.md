# Original game simulation

This layer models explicit territories, players, rules, turns, combat, reinforcement, and movement on the mutable board. It is parallel to the later `GlobalState` mathematical pipeline; no completed adapter installs the advanced mathematical strategy as a player policy.

Main modules are `Board.py`, `Players.py`, `SimulationFunctions.py`, and
`SimulationEngine.py`. The source is **HISTORICAL_REUSABLE**: imports are
supported, but a complete-game run is not claimed as publicly validated because
the archived engine and rule-helper interfaces show development drift. The
mathematical pipeline reuses some representations but does not consume turns
through this engine. Rendering needs optional Pillow and caller-supplied
artwork; no generated model artifact is required for core imports.
