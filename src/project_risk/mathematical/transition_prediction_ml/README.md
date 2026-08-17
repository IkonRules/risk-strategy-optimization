# Transition prediction ML

This layer models a scoped active-player transition, approximately `P(S' | S, policy)`, rather than a complete alternating game.

- `generate_data_ML.py`, `state_generators.py`, `train_ML.py`, and
  `predict_future_states_ML.py` retain the **HISTORICAL_REUSABLE** node-level
  Random Forest route and reusable transition primitives.
- `transition_distribution_ML.py` is the **ACTIVE_EXPERIMENTAL** joint
  successor-signature training/inference route using
  `TransitionDistributionKNNModel`.
- Stage A v2 is **HISTORICAL_REUSABLE** target/dataset infrastructure; Stage A
  v3 is **ACTIVE_EXPERIMENTAL** candidate-selection and target-sampling
  calibration. Neither is a deployed inference model.

No datasets or trained bundles are distributed, and some paths require exact
policy libraries. `predict_future_states_ML.py` also retains a legacy
continent-scoped multi-turn helper. Predicted successor distributions are
consumed experimentally by `full_board_model/`.
