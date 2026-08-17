# Exact demo

The runnable [`examples/run_exact_example.py`](../../examples/run_exact_example.py)
is deliberately small, but it imports the real Project Risk implementation. It
does not contain or depend on a second copy of the mathematical source.

The example exercises the exact combat and finite-state solver under
[`src/project_risk/mathematical/small_graph_model/`](../../src/project_risk/mathematical/small_graph_model/)
and the policy-DAG and distribution utilities under
[`src/project_risk/validation/`](../../src/project_risk/validation/). It requires
no generated production policy library or trained model.
