# Risk Strategy Optimization

> A multi-year computational modelling project exploring how strategies can be
> evaluated and optimized in a stochastic graph-based game.

Risk provides a useful experimental setting for studying a broader computational
problem:

> **How should a decision-maker choose between stochastic action sequences when
> every outcome changes the state of the graph and therefore the decisions that
> become available next?**

The project began as an attempt to build a mathematical framework for evaluating
strategies in the board game Risk. Over several years it developed into a larger
investigation involving graph theory, absorbing Markov chains, dynamic
programming, Monte Carlo simulation, statistical modelling, machine learning,
graph canonicalization and precomputed policy libraries.

The project is not built around one final algorithm. Instead, it has progressed
through a sequence of modelling problems:

- how should strategic progress and utility be defined?
- how can stochastic combat outcomes be represented?
- how can optimal tactical decisions be computed exactly on small graphs?
- how can those exact solutions be reused efficiently during full-game
  simulation?
- what information can be compressed when exact enumeration becomes too
  expensive?
- how can approximations be validated against exact reference solutions?

Several approaches were developed, tested and later revised as their limitations
became clearer.

The current direction is **exact-first**: preserve exact state transitions and
strategic coupling wherever computation permits, and introduce approximation
only beyond that boundary.

---

## The modelling idea

A Risk board can be represented naturally as a graph.

- territories are nodes;
- borders are edges;
- ownership and troop counts define the state;
- combat creates stochastic state transitions;
- conquest changes which actions become possible next.

This means that predicting whether one battle succeeds is only the smallest part
of the problem.

A tactical policy must account for sequences such as:

```text
current graph state
        ↓
choose an attack
        ↓
stochastic battle outcome
        ↓
new ownership and troop placement
        ↓
new legal actions
        ↓
choose again
```

If every possible action and outcome could be explored indefinitely, the problem
could in principle be solved through a complete stochastic game tree.

The practical challenge is state-space growth.

Much of Project Risk has therefore revolved around one question:

> **What information or computation can be discarded without removing
> information needed by the next strategic decision?**

This question connects the project's exact, statistical and machine-learning
phases.

For a more detailed account of how the modelling ideas developed, see
[`docs/MODELLING_APPROACH.md`](docs/MODELLING_APPROACH.md).

---

## From combat probabilities to exact tactical policies

The lowest level of the model treats combat between two hostile territories as
a finite absorbing Markov chain.

`markov_matrix_probabilities.py` computes the probability distribution over all
possible terminal troop configurations for a battle.

These distributions become the stochastic transition kernel for larger graph
problems.

When several hostile edges are available, the problem becomes sequential:
the player must decide which battle to initiate, observe its stochastic outcome
and then choose again from the resulting state.

The exact small-graph solver handles this through memoized dynamic programming
over the reachable state DAG.

The local objective is compared lexicographically using:

1. expected newly conquered territories;
2. expected final attacker troops;
3. probability of complete local conquest.

The solver returns not only an optimal value but a **joint probability
distribution over concrete terminal graph states**.

That distinction is important because two locally equal policies may leave
troops on different nodes and therefore create different future tactical
possibilities.

---

## Exact policy libraries

One of the central ideas in the project is that exact tactical solutions can be
**precomputed and reused**.

A full-board simulation may encounter equivalent local combat structures many
times. Re-solving the same stochastic dynamic-programming problem at every
occurrence would be wasteful.

Project Risk therefore builds libraries of exact solutions for supported small
graphs.

The offline process is approximately:

```text
graph topology
    ↓
role-preserving canonicalization
    ↓
all supported troop configurations
    ↓
exact dynamic-programming solution
    ↓
optimal policy or tied policy alternatives
    ↓
joint terminal-state distributions
    ↓
indexed policy library
```

At runtime the process is reversed:

```text
board region
    ↓
canonicalize the local graph
    ↓
identify troop configuration
    ↓
query the precomputed library
    ↓
recover one or more exact policy distributions
    ↓
map the result back to the full board
```

The library infrastructure evolved considerably as the project grew.

Earlier representations relied on more explicit matrices, DataFrames and
dictionary structures. Later versions moved toward compact vectorized payloads,
graph-level indexes and chunked storage so that millions of solved states could
be queried efficiently.

The current research archive contains policy libraries spanning thousands of
generated files and many gigabytes of exact state distributions. They are not
included in this public repository, but the architecture and generation methods
are documented.

See [`docs/architecture.md`](docs/architecture.md) for the full system design.

---

## Scaling beyond small graphs

Precomputed exact policies solve only part of the problem.

A large active battle graph may exceed the topology or troop ranges covered by
the exact libraries.

Several approaches have been investigated for this boundary.

| Stage | Main idea | What it revealed |
|---|---|---|
| **Macro statistical modelling** | Compress board states into strategic descriptors and predict broad outcomes using regression, GLMs and GAMs. | Strong broad predictive signal does not reconstruct a legal concrete successor state. |
| **Node-level machine learning** | Predict ownership and troop outcomes for individual nodes using global and local features. | Accurate node marginals do not necessarily combine into one legal joint board state. |
| **Plateau / local motif methods** | Reuse apparently stable policies or compose previously solved local structures. | Stable root actions do not imply stable full policies; local optimality depends on context. |
| **Regional decomposition** | Partition large graphs into exactly solved small regions and combine their policy distributions. | Works well when regions are weakly coupled but can fail badly when actions in one region open or alter another. |
| **Exact-first architecture** | Measure exact tractability first and preserve strongly coupled structure before approximating. | Approximation is often needed later than originally assumed. |

These phases are not simply discarded prototypes. Each changed how the later
model was represented.

---

## Regional strategy modelling

The most substantial implemented large-graph route uses the exact policy
libraries as building blocks.

A full battle graph is covered by supported smaller regions. Each region is
queried against the exact library, producing one or more policy-specific
successor distributions.

The ranking layer then considers combinations of regional policies.

```text
full battle graph
        ↓
supported regional covers
        ↓
exact policy-library queries
        ↓
partition-policy candidates
        ↓
local utility ranking
        ↓
second-wave evaluation
        ↓
approximate global policy
```

An important feature is that multiple locally optimal policies are not always
collapsed immediately.

Two policies can have the same local objective value while placing surviving
troops differently. Those placements may affect the next attack wave.

The second-stage model therefore samples regional successor states, reconstructs
the resulting full board and evaluates the next tactical wave.

This was intended to recover some of the interaction lost by considering
regions separately.

---

## Validation changed the architecture

A major part of the project has been comparing approximations against exact
reference solutions rather than assuming that intuitively reasonable
decompositions are valid.

Several results changed the direction of the model.

| Experiment | Result | Modelling implication |
|---|---:|---|
| Exact tractability pilot | 360/360 tested cases completed; worst runtime `0.783527 s` | Exact solving was practical over a wider region than loose combinatorial bounds suggested. |
| Exact composition vs 10,000-sample Monte Carlo | `0.000496 s` vs `4.008617 s`; MC TV error `0.003499` | Once regional distributions are known, Monte Carlo composition may add unnecessary cost and noise. |
| Weakly coupled bridge cases | mean TV `0.006117` | Regional decomposition can be highly accurate when coupling is weak. |
| Strongly coupled double-front cases | mean TV `0.797696` | Independent regions can lose essential sequence dependence. |
| Exact regional candidate selection | changed 15/50 selections, but all seven previous TV=1 failures remained | Better candidate ranking does not repair a structurally invalid decomposition. |
| Exact tied-policy study | different equally valued policies produced materially different successor distributions | Equal utility does not imply an interchangeable transition model. |

The important conclusion is not that regional modelling is unusable.

It is that **the validity of decomposition depends on graph coupling**.

Likewise, the tractability experiments suggested that graphs should not be sent
to approximation simply because they appear large under a loose combinatorial
bound.

See [`docs/validation.md`](docs/validation.md) for conditions, provenance and
caveats.

---

## Current direction: exact first

The emerging architecture now prefers the richest tractable representation:

```mermaid
flowchart TD
    G["Active combat graph"]
    F{"Full exact within empirical budget?"}
    X["Full exact solve"]

    C{"Coupled exact region feasible?"}
    M["Exact coupled macro-region"]

    W{"Weak coupling established?"}
    R["Exact regional policies + composition"]

    A["Bounded joint-state approximation"]

    J["Policy-aware successor-state distribution"]

    G --> F
    F -- Yes --> X
    F -- No --> C
    C -- Yes --> M
    C -- No --> W
    W -- Yes --> R
    W -- No --> A

    X --> J
    M --> J
    R --> J
    A --> J
```

The ordering is intentional:

**preserve exactness first, preserve coupling second, approximate only when
necessary.**

This router is a research conclusion and target architecture rather than one
fully integrated production implementation.

---

## Project architecture

The broader research system is approximately:

```text
full board / player state
        ↓
active battle graph
        ↓
        ├── full exact solve where tractable
        │
        └── library-backed regional reasoning
                 ↓
          canonical graph queries
                 ↓
          exact policy distributions
                 ↓
          partition-policy ranking
                 ↓
          downstream evaluation
        ↓
joint successor-state distribution
        ↓
simulation / next turn / training data
```

The principal exact-library route connects:

```text
markov_matrix_probabilities.py
        ↓
small_graph_outcome_probabilities.py
        ↓
exact_finite_solver.py
        ↓
create_library.py
        ↓
generated exact policy libraries
        ↓
library_io.py
        ↓
approximate_graph_outcome_probabilities.py
        ↓
battle_graph_ranking.py
        ↓
simulation / data generation
```

For a detailed module-level description, see
[`docs/architecture.md`](docs/architecture.md).

---

## Historical modelling branches

The repository documentation also preserves earlier stages of the project.

The macro-statistical phase explored regression, GLMs and spline-based GAMs over
strategic state descriptors.

The node-level machine-learning phase used simulation-generated data and
Random Forest models to predict node ownership and troop outcomes.

Later work moved toward complete joint successor-state distributions rather than
independent node predictions.

Plateau extrapolation, policy compression, local motif composition and several
generations of regional ranking were also explored.

These approaches are documented because the project developed largely through
the cycle:

```text
hypothesis
    ↓
implementation
    ↓
experiment
    ↓
validation
    ↓
limitation discovered
    ↓
change in representation or architecture
```

The chronological evidence is retained in
[`docs/MODEL_DEVELOPMENT_HISTORY.md`](docs/MODEL_DEVELOPMENT_HISTORY.md), the
consolidated record of the statistical/ML phase, exact-policy development,
regional modelling, and the later exact-first direction.

---

## Research source and reproducible demo

The repository now exposes two deliberately separate surfaces:

**Actual research source:** [`src/project_risk/`](src/project_risk/)

```text
game_simulation/
mathematical/
    small_graph_model/
    libraries/
    continent_model/
    transition_prediction_ml/
    full_board_model/
    strategic_evaluation/
infrastructure/
validation/
```

**Runnable example:** [`examples/run_exact_example.py`](examples/run_exact_example.py)

**Demo documentation:** [`demo/exact_demo/`](demo/exact_demo/)

**Optional presentation:** [`demo/visualization/`](demo/visualization/)

The source tree contains recognizable public copies of the reusable research
implementation. The compact example imports and exercises that authoritative
`project_risk` source directly; it is orchestration, not a second mathematical
implementation.

The exact-first architecture is the current research direction, but it is not
yet integrated through the complete full-board multi-turn pipeline. In
particular, the experimental GT rollout still consumes historical node-level
learned transitions, while the later joint-state work remains a separate
experimental route.

---

## Reproducible exact example

Some end-to-end research workflows depend on generated policy libraries,
trained models, datasets, and historical experiment outputs that are not
distributed in a compact public repository.

To make the mathematical core directly inspectable, this repository includes a
small self-contained exact example.

It contains one attacker node with four troops and two adjacent one-troop
defender nodes.

Run:

```bash
python -m pip install -e .
python examples/run_exact_example.py
```

Expected output includes:

```text
Lexicographic value: (1.496647620, 2.878787620, 0.580098240)
Canonical optimal opening attack: (0, 1)
Exactly tied optimal opening attacks: 2
Terminal support: 4 states
Probability mass: 1.000000000000
```

The example demonstrates:

- absorbing Markov combat transitions;
- exact finite-state dynamic programming;
- graph state semantics;
- optimal policy selection;
- preservation of exact policy ties;
- joint terminal-state distributions.

The two symmetric optimal openings have the same local value but different
labelled successor distributions.

The demo should be viewed as a **reproducible window into the mathematical
core**, not as the complete Project Risk model.

---

## Repository guide

- [`docs/MODELLING_APPROACH.md`](docs/MODELLING_APPROACH.md)
  The main conceptual account: how the modelling problem developed, why
  different approaches were tried and what their limitations revealed.

- [`docs/architecture.md`](docs/architecture.md)
  Architecture of the complete research system, including full-board
  simulation, exact solving, policy libraries, regional reasoning and
  experimental ML branches.

- [`docs/validation.md`](docs/validation.md)
  Validation results, evidence classes, experimental conditions and caveats.

- [`docs/MODEL_DEVELOPMENT_HISTORY.md`](docs/MODEL_DEVELOPMENT_HISTORY.md)
  Detailed chronological reconstruction of the project's modelling
  development, including the statistical/ML phase, exact-policy development,
  regional modelling and the later exact-first direction.

- [`src/project_risk/`](src/project_risk/)
  Reusable source representing the original simulator and the layered research
  system.

- [`demo/`](demo/)
  Guides and optional presentation code kept separate from research source.

- [`examples/`](examples/)
  Small exact demonstration using the authoritative `project_risk` package.

- [`tests/`](tests/)
  Public combat, solver, canonicalization, policy and distribution tests.

- [`validation/`](validation/)
  Reproducible public validation plus selected retained historical reports.

---

## Public repository versus research archive

The original research archive remains substantially larger than this public
repository. Reusable source for the original simulator, exact/library,
continent, transition-prediction, full-board, strategic-evaluation and selected
validation layers is now included under `src/project_risk/`.

The public extraction still excludes:

- generated exact policy libraries;
- trained model bundles and large datasets;
- research output/checkpoint trees;
- one-off experiment runners and temporary diagnostics;
- third-party papers and commercial board artwork.

The generated small-graph libraries alone occupy many gigabytes and are tightly
coupled to their serialization format and code version.

They are therefore documented rather than distributed.

The compact exact demo remains separately runnable. Source inclusion does not
imply that every experimental route has the generated artifacts needed for an
end-to-end run.

---

## Scope and limitations

Project Risk is a modelling research project, not a complete implementation of
optimal play for the commercial board game.

Important limitations include:

- the tactical objective is a modelling choice rather than a universal
  definition of optimal Risk play;
- combat uses a fixed local battle policy and an inherited probability table
  rounded to three decimals;
- exact state enumeration remains combinatorial;
- the current regional route is reliable only where its coupling assumptions
  are defensible;
- historical node-level ML results require stronger grouped validation;
- the general smallest sufficient coupled region remains an open problem;
- the exact-first router has not yet been integrated into one complete
  multi-turn production pipeline.

These limitations are part of the modelling problem rather than hidden
implementation details.

---

## Why this project

The most interesting part of Project Risk has not been finding one final
algorithm.

It has been discovering what information a useful stochastic strategy model has
to preserve.

A model may predict broad strategic advantage accurately while being unable to
produce the next legal board state.

A collection of accurate node predictions may fail to represent one coherent
joint outcome.

Several exact local policies may have equal utility while leading to different
future opportunities.

Several exact regional models may each be correct while their independent
composition is wrong.

These observations led to the principle that now guides the project:

> **A representation is sufficient only if the information it discards is also
> irrelevant to the decisions that consume its output.**

That principle is what connects the project's statistical, machine-learning,
dynamic-programming and graph-modelling phases.
