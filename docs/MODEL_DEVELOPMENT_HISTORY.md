# Model Development History

> **Document status:** Historical technical reconstruction, consolidated 2026-08-17
> **Covered period:** approximately July 2024 through 18 July 2026
> **Purpose:** One chronological record of what Project Risk tried, what the evidence showed, and why the modelling direction changed

## 1. Purpose and scope of this history

This document reconstructs the development of Project Risk from its early board
simulation and search experiments to the current exact-first research direction.
It records the project in chronological terms: hypothesis, implementation,
experiment, observed result, limitation, and the next modelling direction. It is
not a description of only the current architecture; `MODELLING_APPROACH.md`
provides the shorter conceptual account, while `architecture.md` describes how
the public source is organized now.

The surviving material is incomplete. Early chronology is reconstructed from 
file dates, archived programs, notebooks, generated reports, and later recovered 
conversation evidence. 

The following evidence labels are used:

- **[CHAT]** A claim supported by a preserved conversation or by a later
  recovered summary of dated conversations. For November–December 2025, the
  underlying individual conversations were not directly available; claims from
  that period are therefore identified as recovered chat evidence rather than
  primary source-code evidence.
- **[REPO]** A claim directly supported by surviving code, notebooks, file
  metadata, datasets, saved output, or generated reports.
- **[INFERENCE]** A plausible interpretation of sequence, intention, or cause
  that is not stated directly in the surviving evidence.
- **[GAP]** A fact that cannot be recovered reliably from the available
  material.

When evidence classes differ, the weaker relevant classification is retained.
Numerical results are included only where a conversation, saved summary, or
report supports them. File modification dates are chronological anchors, not
proof of the exact date on which an idea first appeared.

Two further limits matter throughout this history. First, several modelling
tracks overlapped rather than replacing one another cleanly. Second, later
successes should not be projected backward: the 2025 statistical work was a
genuine modelling direction, not a deliberately staged prelude to the later
exact solver.

## 2. 2024 — Initial simulation and search methods

### 2.1 Board and combat simulation

The oldest dated artifacts show Project Risk beginning with explicit board and
combat simulation. Territories were nodes, adjacency was represented by a
matrix, and an attack was legal when source and target were adjacent, had
different owners, and the source held more than one troop. Setup, ownership,
troop placement, movement, and simulated combat were represented directly.
**[REPO]**

By July–August 2024, isolated battles could be expanded recursively into event
chains with associated probabilities. One archived filename describes a
simulation program with “too many operations,” direct evidence that path
expansion had already become a computational concern. The available material
does not preserve a formal bottleneck analysis from that point. **[REPO; GAP]**

### 2.2 Parallel search and sampling experiments

Several approaches were tested during August–December 2024:

- an SMC program using 100,000 scenarios and a strongest-node attack heuristic;
- an MCTS implementation with selection, expansion, simulation, backpropagation,
  and 10,000 iterations;
- Monte Carlo programs using 100,000 simulations, including an
  expected-value-oriented variant;
- M1–M4 families of simulation and analysis data;
- small-subgraph classification and lookup of precomputed expected outcomes or
  success coefficients; and
- Pareto analysis across several objectives. **[REPO]**

These artifacts establish that sampling, tree search, local motifs, lookup
tables, and multi-objective reasoning were explored. They do not establish one
clean replacement sequence, and the reasons for discontinuing SMC or MCTS are
not recorded. Sampling cost, duplicated work, and loss of distributional
structure are plausible contributors, but assigning those motives would be an
inference. **[INFERENCE; GAP]**

### 2.3 What the early phase contributed

The early work established the enduring problem representation: Risk combat is
a stochastic decision process on a labelled graph, and an attack changes both
troops and the set of future legal actions. It also exposed the tension that
would recur throughout the project: broad simulation can produce outcomes, but
strategy optimization requires a state representation and transition mechanism
that can be reused recursively.

## 3. Early 2025 — Explicit state trees and graph-state reasoning

### 3.1 Choice and outcome branches

A March 2025 notebook, `Explicit approach/Risk_Model.ipynb`, reframed the
problem as alternating choice and stochastic-outcome branches. It compared two
directions: first simulate probability distributions, or calculate explicit
probabilities and then simulate from them. The notebook chose the second.
**[REPO]**

The notebook's most consequential observation was that a repeated state did not
need a separate copy of its entire future tree. A branch could point to an
already defined successor structure. This was an early state-sharing idea: an
explosive event tree could become a directed acyclic graph evaluated with
memoization. **[REPO; INFERENCE]**

One example for battle state `(4,3)` reported a state-tree vector
`[1406, 1776]`. The notebook does not make both components sufficiently clear
for the pair to be reused as a general benchmark. **[REPO; GAP]**

### 3.2 Repeated states as a modelling issue

This phase shifted attention from paths to states. Different event sequences can
arrive at the same owner/troop configuration, and future value depends on that
configuration rather than on how it was reached. The later dynamic-programming
solver made this principle explicit, but the notebook is evidence that the
structural idea preceded the modern implementation.

The chronology between March and November 2025 is incomplete. No surviving
commit history records when each intermediate prototype was started or retired.
The next well-evidenced phase is the macro-statistical transition work beginning
in November. **[GAP]**

## 4. November–December 2025 — Statistical state compression and machine learning

### 4.1 The macro-state hypothesis

During November–December 2025, Project Risk pursued a macro-level statistical
model as a serious strategy for making future-state prediction tractable. The
hypothesis was:

\[
\text{complex board state}
\longrightarrow
\text{small vector of strategic descriptors}
\longrightarrow
\text{expected future outcome}.
\]

A Risk position is a graph-labelled **microstate**: every territory has an
owner and troop count, edges define adjacency, and the arrangement determines
borders, legal attacks, and future options. A **macrostate** is an aggregate
summary such as troop share, territory share, troop concentration, graph degree,
or reserve distance. An **outcome variable** is the aggregate quantity the
statistical model attempts to predict. Surviving fragments include names such
as `expected_new_territories_ratio` and `expected_territories_ratio`, but the
exact target attached to every historical coefficient is not recoverable.
**[CHAT; REPO; GAP]**

The starting predictors were `territory_ratio` and `troops_ratio`. Territory
share represented spatial control; troop share represented material strength;
and their interaction could express that the value of troops changes with the
amount of territory being defended or attacked. This compression was attractive
because direct modelling of ownership, placement, borders, and legal sequences
appeared combinatorially difficult at the time. **[CHAT]**

### 4.2 Manual success factors and controlled simulation

Archived utility fragments preserve a manually designed probability layer. It
started from each player's fraction of continent troops, multiplied attacking
players by a default attacker bonus of `1.2`, renormalized the strengths, and
mapped them to conquest, remaining, or eradication outcomes. Other variants
used squared strengths or competing-risk-like intensities. Their precise
November chronology is unknown because the surviving files were archived in
January 2026. **[REPO; GAP]**

Recovered history dates the explicit “macro-state transition model” to
2025-11-27 and names a missing module, `macro_state_experiments.py`. Its design
survives in a later `ExperimentConfig`: target territory ratios, target troop
ratios, repeated samples for each combination, graph and troop constraints, a
random seed, and a selected outcome variable. The intended experiment grid was

\[
(r_T,r_A) \in
\{\text{target territory ratios}\}
\times
\{\text{target troop ratios}\}.
\]

For each grid point, the system generated legal states, ran repeated
simulations, calculated realized features, and assembled analysis rows.
**[CHAT; REPO]**

Requested ratios could not always be realized exactly. Territory counts are
discrete, troop counts are integers, each owned territory requires at least one
troop, and node caps restrict feasible allocations. The generator therefore
converted targets into legal counts, clamped infeasible troop ratios, and
recorded both target and realized values. This avoided treating generation
constraints as unexplained regression noise. **[REPO]**

The number of simulations per macro row, the complete macro dataframe schema,
the original grid dimensions, and the original sample size are not recoverable.
Later defaults must not be projected backward. **[GAP]**

Full-board setup and rendering were operating by 2025-11-03, when surviving
before/after images were written. Those images confirm simulation capability,
not the later regression design. **[REPO]**

### 4.3 Multiple regression and the first encouraging result

The work began with interpretable statistical baselines rather than complex
machine learning. A surviving OLS helper uses `numpy.linalg.lstsq`, optionally
adds an intercept, drops incomplete rows, and returns coefficients, predictions,
residuals, sample size, and conventional \(R^2\). **[REPO]**

The recovered late-November result was approximately

\[
\widehat y \approx 0.0026
  + 0.344\,\text{troops ratio}
  + 0.823\,\text{territory ratio},
\]

with \(R^2\approx0.952\). The exact response variable cannot be recovered with
confidence; it was likely territory-related, but that should not be stated as
fact. **[CHAT; GAP]**

| Quantity | Recovered value | Qualification |
|---|---:|---|
| Troop-ratio coefficient | approximately `0.344` | Recovered chat summary |
| Territory-ratio coefficient | approximately `0.823` | Recovered chat summary |
| Intercept | approximately `0.0026` | Recovered chat summary |
| \(R^2\) | approximately `0.952` | Recovered chat summary |
| Response variable | unknown | Not recoverable with confidence |

The result established that two aggregate ratios explained much broad outcome
variation. It did not establish that the linear surface was correctly specified
or that the two ratios were sufficient for continued simulation. Structured
residuals, rather than the headline \(R^2\), drove the next iteration.

### 4.4 Residual analysis, interactions, and response geometry

The model sequence continued through territory-by-troop interactions,
quadratic or polynomial terms, and transformed predictors. The purpose was to
capture curvature, saturation, and changes in the marginal troop effect across
territory-control levels. Exact coefficients and improvements from these
intermediate models are not preserved. **[CHAT; GAP]**

Archived code retains systematic transformations:

| Transformation | Historical column |
|---|---|
| Log | `log_realized_troops_ratio` |
| Log | `log_expected_new_territories_ratio` |
| Log | `log_troops_cv` |
| Square root | `sqrt_realized_territory_ratio` |
| Square root | `sqrt_expected_new_territories_ratio` |
| Square root | `sqrt_troops_gini` |
| Logit | `logit_realized_territory_ratio` |

The project then implemented a binomial/quasi-binomial GLM for bounded
proportion outcomes:

\[
\operatorname{logit}(p_i)
=\beta_0+\sum_j\beta_j x_{ij}.
\]

The surviving helper accepts a proportion response and denominator, clips exact
zero/one values, uses a binomial family and logit link, supplies the denominator
as trial weights, and can estimate a Pearson-chi-square scale for
overdispersion. It also reports response-scale residuals and an explicitly
nonstandard SSE-based \(R^2\)-like diagnostic. Coefficients, deviance,
dispersion, AIC, and sample size were not preserved. **[REPO; GAP]**

This was a substantive correction to response geometry: territory shares are
bounded and occur in discrete fractions. It still left the conditional mean too
rigid, and the residuals retained systematic curvature.

### 4.5 GAM and spline modelling

On 2025-12-02, quadratic terms still left a U- or arc-shaped residual pattern.
The project moved to a binomial generalized additive model, reportedly a
`pyGAM.LogisticGAM`, with smooth territory and troop effects and a tensor
interaction conceptually like

\[
\operatorname{logit}(p)
=f_1(\text{territory ratio})
+f_2(\text{troops ratio})
+f_{12}(\text{territory ratio},\text{troops ratio}).
\]

Splines were attractive because advantage could saturate near extremes, be
steeper near balance, and interact differently across control regimes. The
reported diagnostics included heatmaps, response surfaces, troop-effect curves
at fixed territory ratios, residual plots, and direct GLM/GAM comparisons.
Residuals were described as materially flatter than for the GLM. Remaining
bands were attributed partly to discrete territory denominators and Monte Carlo
noise. **[CHAT]**

The original GAM source, plots, spline count, basis order, smoothing penalty,
optimizer settings, convergence information, explained deviance, and held-out
performance are missing. No numerical improvement over the GLM can be claimed.
**[GAP]**

GAM addressed functional-form error within the macro formulation. It did not
change what information entered or left the model.

### 4.6 Expansion of the feature representation

From approximately 2025-12-05 through 2025-12-08, feature engineering expanded
beyond two force-balance ratios. Surviving successor code and recovered
conversation evidence support the following families:

- **Force balance:** target and realized territory ratios, troop ratios, and
  available-troop ratios on the battle and context graphs.
- **Counts and scale:** attacker/defender territory counts, troop counts, total
  territories, total troops, and battle-graph sizes.
- **Deployment distribution:** coefficient of variation and Gini concentration
  for troop placement. A skewness helper survives, but its use in a historical
  fitted model is not proven.
- **Topology:** edge count, mean and variance of degree, diameter, and connected
  components. Recovered conversations mention clustering, but the historical
  definition is missing.
- **Deployment effectiveness:** the fraction of context nodes active in battle,
  the fraction of attacker troops deployed into the battle graph, and reserve
  distance summaries.
- **Frontier and local pressure:** enemy/friendly neighbour counts and troops,
  frontier status, and bounded local-balance signals. These belong primarily to
  the later hybrid node representation rather than the original GAM.

These additions distinguished nominal strength from usable strength. Two
players with equal total troops may differ because one has forces on the active
front while the other's reserves are several edges away. They also show that
the modelling effort did not simply cycle through regressions; it progressively
tested which structural information broad ratios omitted. **[CHAT; REPO]**

Not every feature in successor code can be claimed as a December GAM input.
Clustering and some frontier measures are mentioned but not reconstructable;
effectiveness/reserve columns appear in later datasets but not in the preserved
January Random Forest feature list. **[GAP]**

### 4.7 The representation-loss result

The decisive limitation was not another missing polynomial term. The macro
model estimated something like

\[
E[Y_{t+1}\mid M(S_t)],
\]

where \(M(S_t)\) is a compressed summary. Recursive game simulation needs a
usable representation of

\[
P(S_{t+1}\mid S_t,\text{policy}).
\]

Two states can share territory ratio, troop ratio, Gini, node count, and edge
count while differing in connectivity, frontier location, reserve access,
legal attacks, and stopping points. A predicted future territory ratio cannot
identify which concrete territories changed owner or where troops remain.

This many-to-one compression cannot be inverted by adding flexibility to the
regression function. A GAM can improve the conditional macro mean; it cannot
recover node identity or troop placement that was discarded before fitting.
The central result of the phase was therefore:

> A good predictor of broad advantage is not automatically a generative model
> of legal future game states.

This finding should not be rewritten as if the macro route was never serious.
It was a genuine modelling direction that achieved strong aggregate fit,
developed a richer strategic feature vocabulary, and then failed an important
downstream requirement: constructing the next legal state.

### 4.8 The 2025-12-08 macro-to-micro transition

On 2025-12-08, the proposed supervision changed. Instead of immediately
collapsing Monte Carlo future `GlobalState` objects to one expectation, the
pipeline would retain node-level labels such as

\[
P(\text{attacker holds node }i\text{ after the transition}\mid S_t)
\]

and conditional troop counts. Near-term successor code materializes, for each
simulated final state and node:

- initial and final owner;
- initial and final troops;
- `attacker_holds_final`;
- `captured`;
- battle-node membership;
- shared macro descriptors; and
- later local and regional diagnostics. **[CHAT; REPO]**

Macro features were retained as global context and combined with local node
features. This was a natural continuation of the evidence: broad balance still
mattered, but output identity had to be preserved if another turn was to be
simulated.

### 4.9 January 2026 corroboration and evaluation caveats

Artifacts from 2026-01-08 and 2026-01-09 corroborate implementation of the
December transition. A sequential data generator described node experiments as
the successor to `run_macro_experiment`, and a training file called itself the
“NEW per-node macro→micro approach.” Datasets and Random Forest bundles were
saved for all six continents. **[REPO]**

| Continent | Node rows | Capture ROC-AUC | Attacker-held troop RMSE | Defender-held troop RMSE |
|---|---:|---:|---:|---:|
| North America | 228,600 | 0.992 | 0.547 | 0.396 |
| Africa | 93,000 | 0.993 | 0.573 | 0.408 |
| Asia | 315,000 | 0.995 | 0.536 | 0.330 |
| South America | 7,800 | 0.988 | 0.913 | 0.400 |
| Europe | 198,900 | 0.994 | 0.586 | 0.383 |
| Australia | 7,750 | 0.985 | 1.007 | 0.518 |

The successor training code used a fixed 20% row-level test split, 200 trees,
and random state 42. These scores demonstrate that the pipeline trained and
predicted its saved node targets. They do not establish calibration, legal
joint-state generation, generalization to unseen graph regimes, or multi-turn
strategy quality. Because many rows derive from related initial states and
successors, later documentation treats the ungrouped evaluation as potentially
optimistic and calls for grouped validation. **[REPO; INFERENCE]**

The node-wise formulation also retained a deeper limitation: independently
predicted node marginals need not form a jointly legal or solver-observed board.
That issue motivated the later move to distributions over whole successor
states. The December transition was still a meaningful advance; it preserved
more identity than a scalar macro target and supplied the infrastructure for
the next phase.

The old GAM/statistical subsystem was removed as a production dependency after
this transition, though fragments of OLS, transformations, and GLM code were
archived. The phase remained part of the project's intellectual lineage: it
established controlled generation, target-versus-realized accounting,
simulation-derived supervision, graph-aware features, residual discipline, and
the requirement that an output representation match the next decision stage.

## 5. Exact combat and small-graph policy modelling

### 5.1 Absorbing Markov combat

By June 2026, the active system contained an exact combat kernel in
`markov_matrix_probabilities.py`. It models one fully fought Risk battle as a
finite absorbing Markov chain. Transient states `(a,d)` contain remaining
attacking and defending armies; absorbing states `(0,d)` and `(a,0)` represent
attacker or defender elimination. **[REPO]**

Partitioning the transition matrix into transient-to-transient block `Q` and
transient-to-absorbing block `R` gives the fundamental matrix

\[
N=(I-Q)^{-1}
\]

and absorption probabilities

\[
F=NR.
\]

The strategic solver therefore does not simulate individual dice rounds. For a
chosen attack it reads the complete battle-outcome distribution from `F_df`,
starting at `(source_troops - 1, defender_troops)`. This eliminated local Monte
Carlo noise and reduced strategic tree depth without approximating the combat
mechanics represented by the table. **[CHAT; REPO]**

The code's dice probabilities derive from a table attributed there to Osborne.
The public history records that provenance but does not claim that external
paper as a project-authored result.

### 5.2 Concrete graph states and legal actions

`small_graph_outcome_probabilities.py` represents a position as
`GlobalState(nodes=(NodeState(owner,troops), ...))`. A legal action `(u,v)`
requires attacker ownership at `u`, more than one troop at `u`, defender
ownership at adjacent `v`, and the relevant graph edge. A state is absorbing
when one side is gone or no legal attack remains. **[REPO]**

Troop movement after conquest is part of the decision process. Depending on
remaining hostile neighbours, the solver can compare a minimum move with
pushing all but one troop forward; in other cases movement is forced. This made
the output a concrete legal state rather than only a battle win/loss statistic.

### 5.3 Utility and exact recursion

Several objectives appeared over the project and should not be conflated. A
legacy utility prioritized total success probability and then attacker troops
on former defender nodes. The later context-independent local objective became
lexicographic:

1. expected newly conquered territories;
2. expected remaining attacker troops; and
3. probability of complete local conquest.

An optional `include_no_gain` variant inserts the negative probability of
winning no new territory before the troop component. The comparison uses
tolerances and lexicographic ordering rather than an arbitrary weighted sum.
**[CHAT; REPO]**

For every state, the recursive solver:

1. generates legal battles and movement choices;
2. obtains the complete exact combat distribution;
3. maps each outcome to a successor graph state;
4. solves or reuses every successor value;
5. aggregates utility and terminal-state probabilities; and
6. selects optimal actions while preserving ties where requested.

Each transition reduces the remaining tactical problem, so the reachable state
graph is finite for a fixed topology and troop cap. The implementation is a
memoized Bellman-style dynamic program over a finite DAG, not an approximate
infinite-horizon model. **[CHAT; REPO]**

This exact state representation addressed the same information problem exposed
by the 2025 macro phase from another direction: the next calculation receives a
legal joint state and full terminal distribution, not a decoded scalar
expectation.

## 6. Exact policy libraries and computational scaling

### 6.1 Role-preserving canonicalization

Small combat graphs repeat under different node labels. Canonicalization treats
attacker nodes as permutable among attacker nodes and defender nodes among
defender nodes, without allowing the roles to mix. A canonical representative
can be solved once and mapped back to each labelled query graph. **[CHAT; REPO]**

This produced three practical gains:

- topologically equivalent graphs were not solved repeatedly;
- many troop rows on one topology shared a solver cache; and
- libraries could use canonical topology and row labels as stable identities.

`canonicalize_graphs.py` maintains topology mappings;
`create_library.py` enumerates and solves rows; and `library_io.py` loads
indexed/chunked payloads at runtime.

### 6.2 From values to policy-aware distributions

The library was not merely a table of scalar utilities. Modern rows store sparse
or vectorized policy-specific terminal distributions: probability vector `p`,
owner and troop arrays, row-label mappings, and policy metadata. An inspection
reported 2,158 payloads, all in `policy_options_v2` form, with no older
`exact_df` payloads in that inspected set. **[CHAT; REPO]**

Storage evolved through monolithic tables, compact vector rows, chunked files,
and indexed lookup. This separated offline exact computation from runtime use:
the exact solver generated reusable policy distributions once, while the
continent-scale model normally queried those artifacts rather than resolving
every region online.

### 6.3 Solver and library milestones

Early compact-solver verification on a 2A2D graph with cap 3 and 81 rows matched
the reference solver exactly. Runtime was `0.011 s` versus `0.074 s`, a `6.72×`
speedup; another local comparison reported about `7.9×`. A cap-7 build with
2,401 rows took about `0.64 s`. **[CHAT]**

Reachable-state measurements were far below loose combinatorial bounds:

- maximum 3,816 against a bound of 81,900, with median 3,085;
- maximum 5,440 against a bound of 212,520, with median 4,437.

Five-node full-distribution runs at cap 7 often took roughly one to four
seconds. **[CHAT]**

A 2A3D library covered 98 canonical graphs and 16,807 rows per graph:
1,647,086 rows total. It built in 986 seconds with no failures. A checker later
sampled 2,450 rows across all 98 graphs in 146.53 seconds without finding an
error. Parallelization operated across topologies so each worker could reuse a
topology-local cache. **[CHAT]**

The bottleneck shifted from computation to storage as richer policy alternatives
were retained. The heaviest 3A2D `state_set` libraries were estimated at about
2 GB for cap 7 and 5 GB for cap 8; one cap-8 build was reported at roughly 20
minutes. These production artifacts are not part of the public repository.

## 7. Plateau, motif, and decomposition experiments

### 7.1 Plateau extrapolation

The plateau approach attempted to extend known policies to higher troop counts.
If the best root action appeared stable, the system could evaluate a fixed
policy template instead of solving every state exactly. **[CHAT]**

On 2026-06-16, inspection found an implementation defect: the plateau builder
accepted multi-policy and `state_set` options but did not forward them to the
lower-level builder. A run could therefore appear correctly configured while
not storing the intended policy alternatives. **[CHAT]**

The larger limitation remained after that defect was identified. Stable root
action does not imply stable full policy or global value. Movement choices,
newly opened fronts, and later decisions can change even when the first action
does not. Plateau behavior could be a heuristic or compression device, but not
evidence of exactness by itself.

### 7.2 Puzzle and motif composition

Another idea was to solve two- or three-node motifs and combine them into a
larger “puzzle.” The investigation established an important distinction:

- a solved motif is safe as an exact transition operator or macro-action under
  a higher-level solver;
- the motif's locally optimal policy need not be globally optimal once its
  boundary touches a larger graph; and
- a solver restricted to macro-actions is exact relative to that restricted
  action set, not necessarily exact for the original game. **[CHAT]**

Local or root stability was therefore insufficient to establish a globally
valid policy. This limitation anticipated the later regional result: exact
components do not guarantee exact composition when cross-component decisions
remain coupled.

### 7.3 Why the track changed

On 2026-06-17, the assumed computational trade-off changed. Relevant troop caps
were reported as normally around 7 and not expected above 10, and empirical
reachable states were much smaller than worst-case bounds. Rather than rely on
plateau or motif extrapolation simply because full exact solving looked
combinatorially intimidating, the project built and measured a compact exact
finite solver. Section 10 records that tractability re-evaluation in detail.

## 8. Regional large-graph modelling

### 8.1 From local libraries to continent-scale battle graphs

The next scaling strategy decomposed a larger active battle graph—often roughly
continent-scale—into supported small regions. Each region was canonicalized,
looked up in an exact policy library, and mapped back to the labelled context
graph. `approximate_graph_outcome_probabilities.py` handled extraction,
canonicalization, coverage, and state mapping;
`battle_graph_ranking.py` constructed and evaluated region/policy combinations.
**[CHAT; REPO]**

In this historical code, a variable named `full_graph` often means the complete
context graph for one continent transition, not the full 42-territory world.
The later full-board rollout is a distinct scale.

### 8.2 Policy combinations and two-stage ranking

Each region could expose several locally optimal policy alternatives. The
implemented design became:

1. query every region for relevant policy alternatives;
2. retain each policy's own terminal distribution;
3. form partition-policy combinations;
4. rank them with a locally consistent lexicographic utility;
5. preserve exact ties; and
6. evaluate remaining candidates on a reconstructed global successor and the
   next battle wave. **[CHAT]**

The second stage initially used Monte Carlo. The result therefore had to retain
the winner, all first-stage-optimal candidates, policy identities, and sampled
successor counts. A root action alone was not sufficient because policies with
the same first action could differ later and produce different distributions.

### 8.3 Precision-first partition semantics

On 2026-07-15, partition selection was corrected. If one larger exact region and
several smaller regions cover the same nodes, the larger supported region
retains at least as much coupling information. A finer partition should not win
merely because its independently composed utility looks better. **[CHAT]**

The revised rule generated supported full covers, removed partitions dominated
by exact coarsening, retained maximal non-comparable covers, and used cut edges,
region count, size patterns, and concentration only as diagnostics. Two
implementation problems were also corrected: inconsistent ownership
normalization between `A/D` and player identities, and use of the wrong required
node universe in exact-cover logic. After correction, coverage succeeded for
57/57 unbiased active states and 56/56 library-compatible states. **[CHAT]**

### 8.4 Composition versus decomposition

The regional work separated two questions that had previously been easy to
conflate:

- **Composition:** Given regional distributions, can their Cartesian product be
  assembled exactly and cheaply?
- **Decomposition:** Is treating those regions as independent a faithful model
  of the full battle process?

Exact composition proved easy. Decomposition was the substantive modelling
assumption. A perfectly assembled product distribution can still be wrong if
one region's outcome opens attacks, changes stopping, or redirects troops in
another region.

## 9. Machine-learning successor-state work

### 9.1 Historical node-level Random Forest route

The macro-to-micro work led to continent-specific node-level models. A Random
Forest classifier predicted capture/attacker ownership, while regressors
predicted troop counts conditional on the final holder. Global macro features
were repeated for each node and combined with initial owner/troops, battle
membership, local neighbours, frontier status, and pressure measures.
**[REPO]**

The January results in Section 4 show that this route was operational. They do
not prove joint legality: independently accurate node marginals may combine into
a board that no legal battle sequence could produce. This was the same
representation principle encountered in the macro phase, now at a finer scale.

### 9.2 Joint successor-state distributions

On 2026-07-10, the target shifted again—from independent node predictions to a
probability distribution over complete successor signatures. The desired object
was approximately

\[
P(S'\mid S,\text{active-player transition or policy}),
\]

at continent scale. Concrete signatures preserved correlated owners and troops
from the same simulated outcome. **[CHAT; REPO]**

The later experimental pipeline contained:

- **Stage A:** grouped examples from an initial signature to
  `full_graph_successor_state_counts`, top states, node marginals, and candidate
  diagnostics;
- **Stage B:** `TransitionDistributionKNNModel`, using standardized numerical
  and macro/node features, Euclidean nearest neighbours, and inverse-distance
  or equal weighting to mix empirical whole-state distributions;
- **Stage C:** continent-scale live inference, sampling one concrete successor
  signature and merging it into a `GlobalState`;
- **Stage D:** bounded-particle full-board rollout with alternating player
  perspective and a commitment map preventing external troops from being
  double-counted;
- **Stage D.1:** reinforcement, reallocation, and turn mechanics; and
- **Stage D.2:** fixed-population sequential Monte Carlo with state merging,
  normalized weights, and systematic resampling. **[CHAT; REPO]**

This was retrieval/KNN over joint empirical states, not a neural network or
reinforcement-learning system. Stage A v2/v3 generated and calibrated targets;
they were not deployed inference models.

### 9.3 Particle checks and Stage A instability

Simple systematic-resampling checks preserved expected mass:

- `0.75/0.25` produced `75/25`;
- `0.80/0.20` produced `80/20`; and
- a true `0.70/0.30` distribution produced about `0.698/0.302` over 500
  trajectories. **[CHAT]**

Candidate-selection Monte Carlo and target-distribution Monte Carlo were later
separated. Reusing region queries, sampling plans, state assembly, and canonical
evaluation reduced one 50-candidate runtime from 486.73 seconds to 96.21
seconds. A matched MC20 comparison improved from 220.875 seconds to 11.407
seconds (`19.36×`) with equivalent output. **[CHAT]**

Speed did not solve stability. A provisional Stage A v2 run produced 506 rows,
39 no-combat rows, no failures, an integrity pass, 7,031,889 bytes, and runtime
382 seconds. Yet MC5 versus MC20 changed candidate selection in 45.5% of rows;
five specifically examined North America rows changed in every case. Median
target TV was 1.0, maximum expected-troop shift 5.4, and maximum conquest
probability change 0.70. The dataset was not approved for training. **[CHAT]**

Stage A v3 calibration still encountered candidate sets of 86, 79, and 25 with
target TV around 0.73–0.74. MC80/MC100 was discussed as a conservative review
budget, not established as a final solution. These experiments led to exact
candidate selection and exact composition, which removed sampling error and
made the remaining structural error easier to see.

The planned Stage E evaluation—ownership marginal MAE, troop MAE, utility error,
top-state accuracy/mass, TV, Jensen–Shannon, calibration, and multi-turn
divergence—had not begun in the last report. **[CHAT; REPO]**

## 10. Exact finite solver and re-evaluation of tractability

### 10.1 Compact shared-cache solver

The compact exact solver developed from 2026-06-17 used one solver per topology,
a cache shared across troop rows, packed integer states, precomputed adjacency
and combat rows, and separate value solving from terminal-distribution
reconstruction. Policy selection could therefore operate on compact values and
materialize the larger distribution only when needed. **[CHAT; REPO]**

The exact finite track and ML track overlapped chronologically. They should not
be read as a simple sequence in which one was completed before the other. The
exact solver served library construction and later became a reference against
which regional and learned approximations could be assessed.

### 10.2 Empirical tractability frontier

A later grid attempted 315 exact cases and completed 311. Four stopped at a
10-second runtime limit; none stopped because of state count, cache size, or
memory. Median runtime was 0.0105 seconds, p90 0.0366 seconds, and maximum
10.026 seconds. Estimated cache use had median 10 KB and maximum 24.99 MB.
**[CHAT]**

The provisional conservative full-exact boundary was:

- 8 nodes at cap 6 or below;
- 9 nodes at cap 5 or below; and
- 10 nodes at cap 4 or below.

The next cells—8/cap 7, 9/cap 6, and 10/cap 5—were treated as bounded fallback
cases. All 50 states in the focused regional benchmark fell inside the
conservative exact boundary. **[CHAT]**

This changed the architecture more than a further optimization would have.
Loose state-count bounds had made exact computation look impractical, but
reachable states, caching, and measured runtimes showed that many supposedly
large cases were cheaper to solve exactly than to approximate regionally.

## 11. Policy ties and distributional identity

### 11.1 From one policy to `state_set`

Policy representation evolved through four stages:

1. **Single canonical policy:** deterministically break ties at each state.
2. **Context-independent local-objective policy:** keep library rows reusable
   without importing a larger graph's objective.
3. **Root policy alternatives:** retain tied first actions with optimal
   continuation beneath each.
4. **`state_set` alternatives:** retain policies that share a root action but
   diverge at later tied states. **[CHAT; REPO]**

The `state_set` implementation used a shared options cache and compact policy
objects internally, then flattened them to `policy_options_v2`. Split depth was
defined from the leaves: it controlled how far from policy termination
alternative tied decisions were exposed, not how deeply the canonical solver
optimized.

Tests on 2026-07-08 included a cap-4 set of 256 rows with option histogram
`{1: 157, 2: 99}` and a case in which two alternatives shared root action
`[1,2]` but differed below the root. All 16 canonical 2A2D cap-7 topologies took
31.38 seconds; one 3A2D cap-7 topology took 99.11 seconds. **[CHAT]**

### 11.2 Exact policy DAG validation

The 2026-07-18 policy-DAG report covered 16 unique full-graph cases and eight
macro-region cases, with all 120 depth records complete. It distinguished the
canonical full-depth exact solution from optional export of alternative
exact-tied decisions at selected leaf-split depths. **[REPO]**

Canonical invariance held across export depth: zero value changes, zero
canonical distribution changes, and maximum numerical TV
`6.67e-17`. Export settings therefore did not alter the optimal canonical
solution.

| Export mode | Mean DAG nodes | Max DAG nodes | Mean branching points | Max branching points | Cases with differing tied distributions |
|---|---:|---:|---:|---:|---:|
| Canonical depth 0 | 121.9 | 502 | 0 | 0 | 0 |
| Exact ties depth 1 | — | — | 0.4375 | 1 | 0 |
| Exact ties depth 2 | — | — | 2.75 | 18 | 3 |
| Exact ties depth 3 | — | — | 12.125 | 129 | 6 |
| Unrestricted | 171.25 | 940 | 42.06 | 370 | 5 |

Across all records, 14 cases had materially different successor distributions
among exactly tied policies. Maximum sampled pairwise TV was `0.18507376`.
Equal objective value therefore did not imply distributional identity. This
created an unresolved choice for ML labels: canonical policy, a defined mixture,
strategic tie-breaking, or explicit policy identity. **[REPO]**

## 12. Regional validation and structural failure modes

### 12.1 Full-exact reference versus regional composition

On 2026-07-17, the project built a controlled validation chain: solve the full
graph exactly, solve the regional approximation, compose regional distributions
exactly, and compare. Exact references succeeded for 360/360 cases across 6–8
nodes and caps 3–5. One 8-node, 4A4D, cap-5 boundary run had worst exact time
0.784 seconds, 15,518 states, and support 1,617. **[CHAT]**

For the focused 50-case benchmark:

| Measure | Result |
|---|---:|
| Exact/reference/approximation/composition completion | 50/50 |
| Mean TV | 0.2578047 |
| Median TV | approximately `8.4e-9` |
| p90 TV | 1.0 |
| Cases with TV ≤ 0.05 | 31/50 |
| Mean Jensen–Shannon | 0.1701 |
| Mean balanced Wasserstein | 0.05408 |
| Maximum balanced Wasserstein | 0.36471 |
| One-region mean TV | 0.0842 |
| Two-region mean TV | 0.5183 |

The result was bimodal rather than uniformly mediocre. Bridge structures had
mean TV 0.0061; double-front structures had mean TV 0.7977; all seven TV=1
cases were double fronts. **[CHAT]**

Runtime reinforced the diagnosis. Regional approximation averaged 5.72 seconds
and reached 53.92 seconds, while full exact averaged 0.00749 seconds and reached
0.06382 seconds for the focused states. Exact product composition averaged
0.000440 seconds. A 10,000-sample target Monte Carlo took 4.01 seconds and still
sat at TV 0.00350 from the exact regional product, which took 0.000496 seconds
in the paired comparison. **[CHAT]**

Exact composition could therefore remove target-sampling noise nearly for free.
It could not repair the independence assumption used to produce the regional
inputs.

### 12.2 Exact candidate selection

On 2026-07-18, candidate-selection Monte Carlo was replaced by exact evaluation
under the same lexicographic semantics. Exact ties were preserved and canonical
identity supplied deterministic default ordering. **[CHAT]**

Across 50 records:

- MC1 versus full exact had mean TV 0.2578;
- exact regional versus full exact had mean TV 0.2675;
- MC1 versus exact regional had mean TV 0.0955;
- candidate identity changed in 15/50 cases, 11 materially;
- partition agreement was 35/50;
- policy-option agreement was 50/50; and
- 29/50 cases contained exact ties, with maximum seven.

Relative to full-reference TV, three cases improved, 42 were unchanged, and
five worsened. Exact selection averaged 8.99 seconds, with median 1.39, p90
24.58, and maximum 148.57 seconds. **[CHAT]**

Removing candidate noise did not remove the severe errors. Approximate
structure-specific TV remained near zero for bridge, chain, star, tree, and
two-dense cases; 0.1206 for sequence-opening, 0.1918 for articulation, 0.2858
for cycle, and 0.7991 for double-front cases. All seven earlier TV=1 cases
remained 1. Eight of ten double-front cases were severe, and every severe case
had sequence opening.

### 12.3 Structural coupling, not sampling noise

Exact composition across 229 candidates was trivial: one to three regions,
median regional support 5, maximum 41, median final support 8, maximum 304, and
mean runtime 0.000344 seconds. Global second-stage evaluation, not composition,
dominated runtime. **[CHAT]**

The central failure mode was structural. Outcomes in one region could open
cross-partition attacks, change stopping behavior, or redirect the next front.
Independent regional products discard those conditional relationships. More
Monte Carlo samples estimate the wrong factored model more precisely; they do
not restore the missing coupling.

In eight severe macro-region checks, the selected macro-region was the complete
graph, so full-vs-macro TV was zero and macro beat independent regional in 8/8.
Macro-vs-independent mean TV was 0.9988969. All eight unrestricted cases had
sequence opening, cross-partition follow-up, and outcome-dependent stopping;
six had outcome-dependent front switching. Because the tested macro-region was
the full graph, the experiment did not prove that a smaller sufficient
macro-region can always be identified. **[REPO; INFERENCE]**

## 13. Current exact-first direction

### 13.1 Validation-led routing order

The combined tractability and regional-validation evidence reversed the earlier
regional-first assumption. The current research direction is:

```text
full exact where empirically tractable
    -> otherwise retain strongly coupled structure in an exact macro-region
    -> use exact regional composition only where weak coupling is justified
    -> use bounded joint-state approximation only when necessary
```

In short:

> Preserve exactness first, preserve coupling second, approximate only when
> necessary.

This order eliminates cheap error sources before accepting harder ones. Exact
combat removes dice-sampling error; exact candidate evaluation removes selection
noise; exact composition removes product-sampling error; what remains can then
be attributed to state-space approximation or structural decomposition.

### 13.2 Current direction versus integrated system

As of the last documented research state on 2026-07-18, the following were
supported:

- full exact was practical for substantially more cases than assumed;
- exact composition was effectively free at tested scale;
- independent regional decomposition failed structurally in sequence-coupled
  cases;
- canonical policy-DAG export and canonical invariance were validated;
- exactly tied policies could have different successor distributions; and
- a full/coupled macro representation corrected the eight severe benchmark
  cases that were tested.

The following were not complete:

- production routing had not been switched to exact-first;
- no general algorithm selected the smallest sufficient coupled macro-region;
- Stage A had not been regenerated under the corrected routing and tie policy;
- the joint-state model had not been retrained on corrected targets;
- Stage E distributional and multi-turn validation had not started;
- the exact-first continent model had not been integrated through the
  full-board rollout; and
- no adapter installed the latest mathematical strategy as a player strategy
  in the original `SimulationEngine` environment. **[REPO]**

The exact-first architecture is therefore a validated research direction, not a
claim that one final end-to-end Risk system is implemented.

## 14. Open research questions

### 14.1 Exact routing and macro-region selection

- How should a production-safe tractability predicate combine node count,
  attacker/defender balance, troop cap, topology, runtime, and memory evidence?
- Can sequence openings, cross-partition follow-ups, outcome-dependent stopping,
  and front switching identify the smallest sufficient coupled macro-region?
- How stable are the measured frontiers across hardware, implementation changes,
  and the complete intended rule set?

### 14.2 Conditions for safe regional composition

- Can weak coupling be expressed as a usable sufficient condition rather than a
  post hoc diagnostic?
- Which bridge, cut, stopping, or reachable-action properties guarantee that
  exact regional products preserve the relevant full-graph distribution?
- How should cases near the boundary be validated against full exact references?

### 14.3 Policy identity

- Should exact policy ties produce a canonical deterministic label, a specified
  mixture, a strategic tie-break, or an explicit policy input?
- How much do tied distributions change next-wave utility, not only pairwise TV?
- How large does the full policy DAG become beyond the 16 validated cases?
- How should near-optimal, rather than exactly tied, policies be represented?

### 14.4 ML data and sequential validation

- After routing and tie conventions are fixed, can a small full-exact gold
  dataset validate the joint-state model before large Stage A generation?
- How does KNN/retrieval compare with simple baselines and the older node-level
  Random Forests under grouped, legal joint-state metrics?
- Do good one-step TV, calibration, and marginal errors remain stable over
  alternating-player multi-turn rollout?
- Which model inputs preserve enough policy and coupling context without
  recreating the entire exact state space?

### 14.5 Full-game integration

- Which reinforcement, continent, troop-movement, stopping, fortification, and
  player-alternation rules belong in each layer?
- How should tactical battle utility interact with strategic full-board goals?
- Can the mathematical strategy be adapted safely into the original player and
  simulation platform despite the historical interface drift?
- What evidence would be required before describing the system as a complete
  multi-player Risk strategy rather than a collection of validated modelling
  components?

### 14.6 Remaining historical gaps and reproducibility

- The exact ordering and retirement reasons for several 2024 prototypes remain
  unknown.
- The complete November macro dataframe, simulation count, and historical GAM
  artifacts are missing.
- The exact response attached to the recovered \(R^2\approx0.952\) regression
  is unknown.
- GAM hyperparameters and quantitative fit comparisons are unavailable.
- There is no later report proving completion of exact-first production routing,
  corrected Stage A targets, model retraining, or Stage E.

Future benchmarks should record code snapshot, parameters, seed, hardware, and
output artifact together. The limited historical Git record made this
reconstruction unnecessarily difficult.

The history's central thread is not a smooth march toward one predetermined
algorithm. It is a repeated correction of what information the next decision
stage must retain: simulated values became explicit distributions; event paths
became shared states; labels became canonical graphs; root choices became full
policies; macro expectations became concrete successor states; node marginals
became joint distributions; and assumed intractability gave way to measured
exact-first routing. The project remains unfinished, but its current direction
is grounded in explicit failure analysis rather than retrospective narrative.
