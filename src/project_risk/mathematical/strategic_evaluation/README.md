# Strategic evaluation

At full-board outcome scale, `utility_terminal.py` is **CURRENT_REUSABLE**
terminal evaluation for compatible start/end states from exact, regional, ML,
or rollout producers. It requires no trained model itself.
`game_theory_commitment.py` is **ACTIVE_EXPERIMENTAL**: it consumes the current
GT rollout, enumerates commitment profiles, and constructs payoff tables. It
does not solve or select a Nash equilibrium, and rollout use requires the
relevant learned-model artifacts.
