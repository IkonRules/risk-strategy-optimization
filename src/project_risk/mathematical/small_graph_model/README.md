# Small graph model

**CURRENT_REUSABLE** exact local model for elementary stochastic combat, legal
graph actions, finite-state dynamic programming, and policy-specific joint
terminal distributions. Its main modules are
`markov_matrix_probabilities.py`, `small_graph_outcome_probabilities.py`, and
`exact_finite_solver.py`. It needs NumPy and pandas but no generated policy
library. The `libraries/` layer consumes its exact solutions offline.
