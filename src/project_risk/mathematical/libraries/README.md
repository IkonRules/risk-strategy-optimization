# Exact policy libraries

**CURRENT_REUSABLE** offline generation and runtime lookup code for canonical
small-graph policies. `canonicalize_graphs.py`, `create_library.py`, and
`library_io.py` convert exact local solutions into indexed policy-specific
terminal distributions consumed by `continent_model/`. Production libraries
are generated artifacts, occupy many gigabytes, and are intentionally excluded.
Tests build only tiny temporary fixtures.

Two lazy compatibility calls in `library_io.py` refer to an older extrapolation module that was not present in the inspected source tree; current exact-finite paths do not require it.
