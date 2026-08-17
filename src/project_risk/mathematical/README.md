# Mathematical strategy model

The mathematical pipeline is organized by scale:

`small_graph_model -> libraries -> continent_model -> transition_prediction_ml -> full_board_model -> strategic_evaluation`

Validation is cross-cutting. The exact-first direction is current, but it is not yet integrated end to end through the experimental full-board rollouts.

The layers progress from exact local tactical graphs, through generated policy
libraries and continent-scale composition, to experimental transition and
complete-board rollouts. Their outputs ultimately feed strategic evaluation.
Maturity and artifact requirements are recorded in each layer README.
