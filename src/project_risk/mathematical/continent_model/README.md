# Continent model

**CURRENT_REUSABLE** continent-scale battle-graph infrastructure for extracting
supported regions, querying exact libraries, constructing covers, retaining
policy alternatives, and ranking downstream consequences. Its main modules are
`approximate_graph_outcome_probabilities.py` and `battle_graph_ranking.py`.
“Large” normally means continent scale, not the complete 42-territory board.
Runtime queries require separately generated exact libraries; selected
successor distributions feed data generation, transition modelling, and later
strategic evaluation.
