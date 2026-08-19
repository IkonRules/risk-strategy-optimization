# Modelling Approach

Project Risk is an attempt to build a mathematical framework for evaluating
strategy in a stochastic, graph-based game.

The board game Risk provides a useful environment for this problem. The state of
the game can be represented naturally as a graph: territories are nodes, borders
are edges, ownership and troop counts define the state of each node, and combat
introduces stochastic transitions between states. Every attack changes the graph
on which future decisions are made. A conquest may open a new front, close
another, move troops to a different part of the graph, or make a previously
impossible action available.

The central problem is therefore to determine which sequence of actions should
be preferred when actions have stochastic outcomes and those outcomes change the
set of decisions available afterwards.

A complete solution could in theory be obtained by enumerating every legal
action, every stochastic outcome, and every subsequent action until the game
terminates. The limitation is computational. As the number of territories,
troop configurations and possible action sequences increases, explicit
enumeration becomes increasingly expensive.

Much of the project has consequently revolved around one recurring question:

> **How much of the full decision problem can be simplified, compressed or
> precomputed without discarding information that matters for later decisions?**

Different parts of the model answer this question at different scales. The
current repository reflects those scales directly:

```text
small_graph_model
    -> libraries
    -> continent_model
    -> transition_prediction_ml
    -> full_board_model
    -> strategic_evaluation
```

This document follows that pipeline rather than the chronology in which every
idea was implemented. The project has been revisited repeatedly over several
years, and many components were replaced, rebuilt or reinterpreted after later
experiments exposed weaknesses in earlier assumptions. In particular, some of
the oldest architectural ideas in the project are the small-graph policy model
and the decomposition of larger graphs into smaller regions. Components that
appear earlier in the pipeline today, such as the current Markov combat kernel,
were introduced later as better implementations of problems that earlier
versions already attempted to solve.

The purpose here is therefore not to reconstruct a strict sequence of commits.
It is to explain how the present modelling pipeline works, why each layer
exists, which approximations were introduced at that layer, and what later
validation showed about them.

---

## 1. Strategic objectives and evaluation framework

Before searching for an optimal strategy, optimality has to be defined.

Winning the complete game is the obvious ultimate objective, but it is not a
particularly useful quantity for evaluating every intermediate decision. A
tactical choice may improve a player's position substantially without producing
an immediate game win. Strategic progress must therefore be represented through
quantities that can be evaluated at shorter horizons.

At a high level, the project treats acquisition and retention of territory as
the basis of strategic progress. Continents provide a natural higher-level
structure because control of a complete continent has strategic value beyond
the individual territories it contains. This eventually motivated a
game-theoretic framework in which players can be viewed as committing to
strategic goals, such as sets of continents to pursue, and the outcomes of those
commitments can be evaluated through a utility function.

This game-theoretic evaluation layer came relatively late in the project.
Earlier versions evaluated outcomes by prioritizing one quantity at a time or
by identifying Pareto-optimal outcomes across several objectives. The later
framework provided a more coherent way to express preferences across different
levels of the model.

This also introduced an important distinction between **strategic actions** and
**tactical actions**.

A strategic action describes what a player is trying to achieve over a longer
part of the game. A tactical action is a concrete battlefield decision: which
territory to attack, from where, and how to continue after the outcome becomes
known.

The distinction is useful because the two levels operate at very different
scales. Enumerating arbitrary combinations of individual territories as
strategic objectives creates an enormous choice space even before stochastic
combat is considered. Continents provide a coarser strategic representation. At
the tactical level, however, the identity of individual territories and their
connections cannot be discarded because they determine which actions are legal.

This leads to two separate modelling tasks.

The first is **prediction**:

> Given a state and a set of actions, what states may follow and with what
> probabilities?

The second is **evaluation**:

> Given a possible outcome, how desirable is that outcome?

An optimizer requires both. A sophisticated utility function cannot compensate
for an incorrect transition model, and a perfect transition model cannot choose
between actions unless preferences are defined.

The rest of the modelling pipeline is primarily concerned with constructing
useful transition models at increasing graph scales. The final strategic
evaluation layer then uses those transitions to compare higher-level strategic
commitments.

---

## 2. Small-graph model

The small-graph model is the tactical core of the project. It asks how a player
should act when the relevant combat situation contains only a limited number of
territories and explicit calculation is feasible.

The basic object is a graph whose nodes contain ownership and troop counts. A
policy specifies what attack to make in each reachable state, and combat makes
the transition between states stochastic.

The small-graph layer therefore combines two problems:

1. calculate the stochastic result of one node-to-node battle; and
2. choose the sequence of battles that gives the preferred distribution of
   terminal graph states.

### 2.1 Node-to-node combat

The smallest stochastic building block is combat between two adjacent hostile
territories.

The project used several ways of computing these battle outcomes over time. The
current implementation uses an absorbing Markov chain, which replaced earlier
and less efficient explicit battle calculations.

Under a fixed combat policy, a battle can be represented as a finite absorbing
Markov chain. A transient state records the number of attacking and defending
troops still able to participate in the battle. Dice outcomes create
probabilistic transitions between these states. Eventually either the defender
is eliminated or the attacker can no longer continue, producing an absorbing
state.

Writing the transition matrix in canonical absorbing form,

$$
P =
\begin{bmatrix}
Q & R \\
0 & I
\end{bmatrix},
$$

the fundamental matrix is

$$
N = (I - Q)^{-1},
$$

and the probabilities of absorption in each terminal state are

$$
F = NR.
$$

A row of $F$ gives considerably more information than a single probability of
victory. It gives the complete probability distribution over the possible
terminal troop configurations of the battle.

This distribution is the elementary stochastic transition kernel used by the
higher-level tactical solver.

The implementation is contained primarily in
`markov_matrix_probabilities.py`. The approach was inspired by *Markov Chains
for the RISK Board Game Revisited* by Jason A. Osborne.

Two simplifying assumptions are important. Combat is treated as continuing
until either the defender is eliminated or the attacker can no longer proceed,
and the maximum number of dice permitted by the model is used. These are
modelling assumptions that reduce the node-to-node action space. They should not
be interpreted as a claim that the same behaviour must always be globally
optimal in every complete board position.

The architectural advantage is that the higher-level solver does not need to
branch over every individual dice roll. Once it chooses an attack, the Markov
model supplies the complete distribution of terminal outcomes for that battle.

### 2.2 From battles to sequential policies

Adding a third territory changes the nature of the problem.

In a two-node battle, once the combat assumptions are fixed there is essentially
one tactical course of action. On a graph with several hostile edges, the player
must choose which battle to initiate. The result of that battle then changes the
set of actions available next.

The problem has therefore moved from a probability calculation to a sequential
decision problem.

A state consists of the ownership and troop count of every node in the active
combat graph. From a state, the solver enumerates all legal attacks. For each
attack, the node-to-node combat model provides the possible battle outcomes and
their probabilities. Each outcome produces a new graph state, from which
another decision may have to be made.

For a state $s$, a legal action $a$, and possible successor states $s'$,
the reasoning is Bellman-like:

$$
V(s) =
\max_{a \in A(s)}
\sum_{s'}
P(s' \mid s, a) V(s').
$$

Conceptually this creates a game tree. In practice, many different paths arrive
at the same ownership-and-troop configuration. The calculation is therefore
better understood as a finite directed acyclic graph of states in which repeated
successors can be reused.

The current exact implementation is represented primarily by
`small_graph_outcome_probabilities.py` and the more computationally efficient
`exact_finite_solver.py`.

### 2.3 Local utility and optimal policy value

The solver requires a local objective for comparing tactical outcomes.

The current small-graph objective prioritizes three quantities
lexicographically:

1. expected number of newly conquered territories;
2. expected number of final attacker troops; and
3. probability of complete local conquest.

The ordering is deliberate. Territorial progress comes first because a policy
that simply preserves troops by avoiding combat should not dominate a policy
that achieves the tactical objective. Once territorial outcomes are equal,
conserving troops becomes valuable. Full local conquest provides an additional
preference when the first two quantities are also equal.

This is not intended to represent the complete strategic utility of the game.
It is a context-independent local objective that makes exact small-graph
solutions reusable. Higher layers can then evaluate the resulting states in a
broader strategic context.

### 2.4 Why an optimal value is not enough

An important complication appears when several policies have the same local
value.

Suppose two policies conquer the same expected number of territories, leave the
same expected number of attacker troops and have the same probability of local
conquest. Under the local objective they are equally good.

They need not, however, be strategically equivalent.

One policy may leave most surviving troops on a node adjacent to a new front,
while another leaves them farther away. The local objective may assign the same
value to both distributions even though the next large-graph decision sees very
different opportunities.

For this reason, the model moved away from representing a solved state only by
an optimal scalar value or even by one optimal action. A useful solution must
preserve the **distribution of concrete successor states** associated with the
policy.

This also means that multiple equally valued policies may need to be retained.

Early implementations preserved different optimal root actions. The assumption
was that different opening actions would often produce the most strategically
different terminal troop distributions. Later `state_set` representations
allowed policies to differ at tied decisions deeper in the state graph even
when the opening action was identical. The policy-DAG work extended the same
idea by explicitly representing tied optimal choices at several depths.

The purpose is not to multiply policies for its own sake. It is to avoid
discarding downstream-relevant information simply because several policies are
equal under the current local objective.

The general lesson is:

> **Equality under the current objective does not imply equality as a transition
> to the next decision problem.**

The information passed forward by the small-graph solver therefore matters as
much as the scalar criterion used to select a policy.

### 2.5 Scaling with troop counts

Small graphs create two different kinds of scaling pressure: the graph can
contain more nodes, or the existing nodes can contain more troops. These
problems eventually required different solutions.

Increasing the number of troops while keeping the topology small was the first
scaling problem addressed directly inside the small-graph model.

Early explicit policy calculations became expensive at troop counts well below
the range the full game might realistically produce. The practical objective
was therefore to support roughly seven to ten troops per node, including edge
cases, even though the older implementation could explicitly solve only much
smaller troop ranges efficiently.

#### Plateau approximation

The first solution was to approximate policies above the tractable range.

The most promising hypothesis was that optimal policies might reach a
**plateau** as troop counts increased. If the preferred action or policy
structure stabilized, then exact policies could be computed only up to some
threshold and extrapolated above it.

The intuition was that once sufficiently many troops were present, adding one
more troop might not change which attacks were available. If the action set had
stabilized and the same continuation remained optimal, perhaps the policy itself
would also stabilize.

This was attractive because it converted an expensive exact calculation into a
lookup-and-extrapolation problem.

The hypothesis was ultimately too weak.

Stable action availability did not imply stable optimal policies. Topology,
post-conquest troop movement, newly opened fronts and later stochastic outcomes
could change the best continuation even if the opening action appeared stable.
Several attempts were made to identify useful patterns in the high-troop
policies, but the resulting rules were either too complicated to be practical
or too crude to be trusted.

The important historical point is that the plateau approximation **predated**
the later use of memoization and dynamic programming as the practical solution
to the troop-scaling problem. It was introduced because the explicit solver of
the time was too inefficient at higher troop counts.

#### Efficient explicit computation

The eventual solution was not a better extrapolation rule. It was a better
exact solver.

Once repeated successor states were recognized and reused, memoization and
dynamic programming avoided recalculating the same continuation many times. The
later compact finite solver pushed this further with a representation designed
specifically for repeated exact evaluation: compact state encoding, shared
caches, precomputed graph and combat information, and separation of value
calculation from terminal-distribution reconstruction.

These changes increased the tractable troop range enough that the plateau
approximation was no longer necessary for the small graphs the project intended
to support.

This is an important distinction in the overall architecture:

> **Troop-count scaling on small graphs was first treated as an approximation
> problem, but was ultimately solved primarily as an exact-computation
> problem.**

The remaining large-graph scaling problem is different. It is driven mainly by
the number of interacting nodes and therefore appears at the
`continent_model` layer rather than inside `small_graph_model`.

---

## 3. Exact policy libraries

Solving one small graph exactly is useful, but a larger simulation may encounter
the same tactical structure many times. Re-solving every encounter online would
waste the work already done by the small-graph solver.

This motivated one of the central architectural ideas in the project:
**precompute policies for small combat graphs and reuse them whenever the same
situation appears in a larger graph.**

The library idea predates the current `exact_finite_solver`. Earlier libraries
were built from earlier versions of the small-graph calculation. As the exact
solver improved, the same library architecture could be rebuilt with larger
troop ranges, more graph topologies and richer policy outputs.

The development of the small-graph solver and the library system was therefore
iterative rather than strictly sequential: better solvers made better libraries
possible, while the requirements of large libraries exposed new computational
and storage bottlenecks in the solver.

### 3.1 Precomputation as an interface

A library is built for a pattern containing a given number of attacker and
defender nodes. For each supported graph topology, initial troop configurations
up to chosen caps are evaluated. The resulting policy-specific terminal
distributions are stored so that later calculations can query them directly.

At runtime the intended flow is approximately:

$$
\text{large-graph state}
\rightarrow
\text{local combat region}
\rightarrow
\text{canonical graph}
\rightarrow
\text{troop configuration}
\rightarrow
\text{precomputed policy distribution}.
$$

This turns repeated tactical optimization into an amortized-computation problem:
solve the expensive stochastic decision problem once, then reuse the result many
times.

The library is therefore not merely storage. It is the interface between the
exact small-graph model and the large-graph approximation model.

### 3.2 Graph canonicalization

A major source of redundant computation is node labelling.

Two graphs may have identical topology and ownership structure but different
territory names or local node numbers. Solving both independently duplicates the
same decision problem.

The model therefore canonicalizes graphs under relabellings that preserve the
attacker and defender roles. Attacker nodes may be permuted among attacker
nodes and defender nodes among defender nodes, but the roles themselves are not
interchanged.

The canonical graph becomes the library identity. When a region is queried from
a larger graph, the concrete labels are mapped to the canonical
representation, the corresponding policy distribution is retrieved, and the
result can then be mapped back to the original labelled graph.

Canonicalization serves two purposes at once:

- it reduces the number of graph topologies that need to be solved; and
- it allows one exact solution to be reused in many concrete positions.

### 3.3 Policy-aware library rows

The library output also became richer over time.

A row cannot always be represented adequately by one optimal scalar value or
one policy. As described in the small-graph section, tied local policies may
produce different successor distributions.

The later library representations therefore retain policy-specific terminal
distributions and, where appropriate, multiple exact policy alternatives.
`state_set` and later policy-option formats preserve alternatives that are equal
under the current objective but differ in how they distribute ownership and
troops across terminal states.

This allows a downstream large-graph model to decide whether those differences
matter in the broader context.

### 3.4 From large tables to compact probability stores

Earlier implementations represented exact outcomes through relatively verbose
tables, matrices, dictionaries and DataFrames. As the number of topologies,
troop configurations and policy alternatives increased, this became expensive
both in memory and on disk.

Later versions moved toward compact vectorized payloads. An outcome distribution
can be stored through aligned arrays containing probabilities, owner states,
troop counts and derived metrics. Rows are grouped and chunked so that runtime
queries do not require loading one enormous object.

The current library pipeline separates generation from consumption.
`create_library.py` enumerates and solves the canonical graph/state space and
writes the generated libraries. `library_io.py` defines the format-aware lookup
boundary and loads the relevant data for a queried state.

This design made it possible to build libraries containing millions of solved
initial states. One recorded `2A3D` build covered 98 canonical topologies and
1,647,086 troop configurations. As richer policy representations were retained,
some production libraries grew to several gigabytes.

The bottleneck therefore shifted over time. Once exact computation became
faster, **representation and storage could become limiting factors before the
underlying dynamic programming did**.

### 3.5 Extending coverage with structurally restricted graphs

Library coverage is not determined only by node count. Some ownership patterns
are difficult because enumerating every possible topology and troop
configuration becomes expensive, even though particular graph families within
that pattern may be easy.

This motivated restricted library families for special cases, particularly
star-topology graphs. A star graph has a strongly constrained topology, which
greatly reduces the number of structures that need to be represented. Such
libraries can therefore extend exact coverage to otherwise awkward
attacker/defender count combinations without pretending that every topology of
the same size has been solved.

These special cases later became useful not only for runtime coverage but also
for improving the range of large-graph states that could be used as training
examples in the transition-prediction work.

---

## 4. Continent / large-graph model

The second scaling problem is fundamentally different from increasing troop
counts on a fixed small graph.

When the number of interacting nodes grows, the number of possible topologies,
ownership states, legal attack sequences, movement choices and stochastic
branches grows rapidly. Precomputing every possible large graph is not practical
in the same way as precomputing a bounded family of small graphs.

The project therefore introduced a decomposition idea very early: **represent a
large combat graph as a collection of smaller regions that can be solved or
queried individually**.

This partitioning idea is one of the architectural backbones of Project Risk.
It predates the current library and exact-solver implementations. Over time,
however, it became increasingly library-backed: once exact small-graph policies
were available, the large-graph model could use them as reusable regional
building blocks.

The current public implementation of this layer is represented primarily by
`approximate_graph_outcome_probabilities.py` and
`battle_graph_ranking.py`.

### 4.1 Partitioning the large graph

The basic large-graph hypothesis is:

> If a large combat graph can be partitioned into smaller regions for which
> sufficiently accurate policies are available, perhaps the large-graph policy
> can be approximated by combining those regional policies.

The process begins by constructing an active battle graph from the current
state. Candidate small regions are then identified subject to constraints such
as connectivity, attacker/defender composition and library support.

A valid partition covers the relevant battle graph with disjoint supported
regions. For each region, the concrete subgraph is canonicalized, the
appropriate library row is located and one or more policy-specific successor
distributions are retrieved.

This gives the first large-graph approximation:

$$
\text{large combat graph}
\rightarrow
\text{small exact-supported regions}
\rightarrow
\text{regional policy distributions}.
$$

### 4.2 Enumerating partitions and regional policies

A large graph can often be partitioned in several different ways.

The model therefore does not commit immediately to one arbitrary decomposition.
It can enumerate admissible partitions and, where a region contains multiple
locally optimal policies, enumerate combinations of regional policy
alternatives.

The simplest ranking approach was to combine the regional expected outcomes
into an aggregate utility for the large graph. This made it possible to compare
partitions and policy combinations using quantities such as expected territorial
gain, remaining troops and conquest probability.

This approach was useful but exposed a central weakness: a sum or product of
regional utilities can look attractive even when the partition itself has
removed dependencies that matter to the larger tactical problem.

### 4.3 Monte Carlo look-ahead and re-partitioning

The next development was to look beyond the first regional wave.

For a candidate partition-policy combination, the model samples one outcome
from each region's policy distribution and assembles those outcomes into a
concrete successor state of the large graph. That sampled successor is not
treated as belonging permanently to the old partition.

Instead, it becomes a **new large-graph state**.

The battle graph is rebuilt from that state and partitioned again. Region
boundaries can therefore move between waves. Nodes that were placed in separate
regions in the first partition may appear in the same region after a conquest
opens a new connection, while previously interacting regions may disappear or
split.

The approximate look-ahead therefore became conceptually:

$$
\text{large state}_t
\rightarrow
\text{partition}_t
\rightarrow
\text{sample regional outcomes}
\rightarrow
\text{large state}_{t+1}
\rightarrow
\text{new partition}_{t+1}.
$$

Monte Carlo simulation was used to repeat this process across many sampled
successors and estimate the downstream value of each candidate.

This was a substantial improvement over a one-wave aggregation because it gave
regional outcomes a route to influence the structure of the next decision
problem.

### 4.4 Why fewer, larger exact regions are preferred

Partition choice eventually required a principle beyond simply comparing the
aggregated utility of every possible partition.

Suppose the same set of large-graph nodes can be represented either by one
larger exact-supported region or by several smaller exact-supported regions.
The finer partition discards interactions between those smaller regions before
utility is calculated. If the larger region can be solved as one unit, there is
no precision advantage in splitting it first and then allowing the sum of the
smaller regional utilities to compete with the coupled solution.

The current default candidate preparation therefore uses a **maximal supported
partition** principle. A finer partition is treated as dominated when a strict
exact coarsening covers the same node universe with fewer regions and every
region in the finer partition is contained within a region of the coarser one.
Such dominated partitions are filtered before the later utility comparison.

This is stronger than merely giving large regions a small ranking bonus. It
expresses a modelling preference:

> **Preserve as much exact coupling as the available region library permits
> before comparing approximate regional utilities.**

Among non-dominated partitions, utility and downstream evaluation still matter.
But the model should not prefer an artificially fine decomposition simply
because the factorized utility appears better after dependencies have already
been removed.

### 4.5 Composition versus decomposition

The regional model contains two distinct approximation questions.

The first is **composition**:

> Given several regional successor distributions, how should they be combined
> into a distribution over the larger graph?

The second is **decomposition**:

> Was it valid to treat those regions as separate stochastic components in the
> first place?

These questions turned out to have very different answers.

Once the regional distributions are known, exact Cartesian composition can be
cheap. Monte Carlo sampling of the product distribution introduced unnecessary
sampling noise in cases where exact composition was already computationally
trivial.

But exact composition does not validate the decomposition assumption. A
perfectly calculated product of regional distributions can still be the wrong
large-graph distribution if the true process contains dependencies between the
regions.

### 4.6 Testing the decomposition assumption

The regional approximation was eventually tested directly against full exact
solutions.

For weakly connected bridge-like cases, regional composition reproduced the
full exact distribution very closely. Across nine retained bridge cases, the
mean total-variation distance was approximately

$$
0.0061.
$$

In double-front cases, the behaviour was very different. Across ten cases, mean
total-variation distance was approximately

$$
0.798,
$$

and several cases had total variation equal to one.

The difference was structural.

In strongly coupled graphs, success in one region may open an attack into
another. The active front may depend on the realized stochastic outcome.
Stopping in one branch may leave troops available elsewhere. A partition chosen
before these events occur may therefore impose an independence structure that
the real tactical process does not have.

Making candidate selection more exact changed some choices but did not remove
the severe failures. Improving the ranking procedure could not restore
dependencies that had already been discarded when the graph was decomposed.

The key result was therefore:

> **Exact composition cannot repair an invalid independence assumption.**

Regional decomposition was not rejected. It remains useful when interactions
between regions are weak enough. What changed was the assumption that
partitioning should automatically be the default whenever a graph exceeded some
predefined small size.

### 4.7 Re-evaluating exact tractability

At roughly the same stage, improvements to the exact solver made it possible to
test larger graphs directly.

Compact state representation, shared caches, precomputed combat rows and
separation of value solving from distribution reconstruction reduced the
practical cost of explicit calculation. More importantly, measured reachable
state spaces were often much smaller than loose combinatorial upper bounds had
suggested.

In one tractability study, all 360 tested cases covering graphs with six to
eight nodes and troop caps from three to five completed within the specified
resource budget, with the worst recorded runtime below one second. A broader
315-cell experiment completed 311 cases under a ten-second stop; the remaining
four cases stopped because of runtime rather than memory or a state-count limit.

These results do not make the large-graph problem non-combinatorial.

They change the order in which approximation should be considered.

Earlier development often began from the question:

> The graph is getting large; how can it be partitioned or approximated?

The later question became:

> Is approximation actually necessary for this graph?

### 4.8 The current large-graph hybrid model

The present research direction is therefore best described as a
**large-graph hybrid model**.

Conceptually, its preferred routing is:

$$
\text{full exact}
\rightarrow
\text{coupled exact region}
\rightarrow
\text{weakly coupled regional composition}
\rightarrow
\text{approximate transition model}.
$$

The first choice is full exact calculation whenever empirical resource limits
make it practical.

If the complete graph is too expensive, the next objective is to preserve
strongly interacting nodes inside the largest useful exact-supported region
rather than immediately forcing them into independent pieces.

Regional composition is then appropriate where the discarded coupling is weak
enough to justify it.

Only beyond that boundary should the model fall back on a learned or otherwise
approximate transition mechanism.

This hybrid logic is the current conclusion of the large-graph research, not a
claim that a final automatic production router has already been implemented.
Determining the smallest sufficient coupled region is itself an open modelling
problem, and the exact-first logic has not yet been propagated through every
later layer of the full-board system.

---

## 5. Transition prediction and machine learning

The statistical and machine-learning work is best understood as a
**surrogate-modelling branch of the large-graph pipeline**, not as an unrelated
alternative to partitioning.

The partition-based large-graph model can generate strategic transitions, but
it is expensive. It must construct candidate regions, evaluate regional policy
combinations and, in later versions, use Monte Carlo look-ahead and
re-partitioning to estimate downstream effects.

The natural next question was therefore:

> Can the mapping learned from this expensive transition process be generalized,
> so that later simulations do not need to repeat the full optimization every
> time?

A large collection of initial states could be generated, solved by the
partition/library/Monte Carlo model, and then used as supervised training data
for a cheaper predictive model.

This relationship is central to the interpretation of both the statistical and
machine-learning phases:

$$
\text{initial large-graph state}
\rightarrow
\text{expensive partition-based model}
\rightarrow
\text{generated target}
\rightarrow
\text{statistical or ML surrogate}.
$$

The surrogate was therefore intended to learn the behaviour of the large-graph
solver, not to replace the underlying strategic definition of the problem.

### 5.1 Describing the board statistically

The first surrogate represented the board through a small number of strategic
descriptors.

A complex state could be summarized using quantities such as territory balance,
troop balance, troop concentration, graph topology, reserve distance and the
proportion of forces actively participating in the battle.

The idea was that the expensive large-graph process might contain stable
macro-level relationships that could be learned statistically. If so, expected
future outcomes could be predicted directly from those descriptors instead of
repeatedly performing partition enumeration and Monte Carlo evaluation.

The statistical work developed in stages.

Initial multiple regression models showed that simple variables such as troop
and territory ratios already contained substantial predictive information. One
historical two-variable fit reached approximately

$$
R^2 \approx 0.952.
$$

The high fit was encouraging, but the residuals were not structureless.
Interactions and polynomial terms were introduced, followed by models more
appropriate for bounded outcomes. Generalized linear models were explored, and
eventually generalized additive models with splines were used to represent
nonlinear relationships without imposing one global polynomial form.

The same phase produced increasingly rich feature engineering: coefficients of
variation and Gini measures for troop concentration, graph statistics, measures
of active versus reserve forces, and distances between reserves and the current
front.

The statistical models became increasingly good at describing broad strategic
outcomes generated by the large-graph process.

The problem was the type of output.

A macro model estimates something like

$$
E[Y_{t+1} \mid M(S_t)],
$$

where $M(S_t)$ is a compressed description of the current state.

A recursive simulation does not merely require an expected number of
territories or an expected troop balance. It requires a concrete successor
state, or a distribution over concrete successor states:

$$
P(S_{t+1} \mid S_t, \pi).
$$

Two board states can have almost identical territory ratios, troop ratios, Gini
coefficients and coarse topology while having different borders, different
legal attacks and different possible continuations.

No increase in regression flexibility can reconstruct information that was
discarded before fitting the model.

The macro-statistical approach therefore failed primarily as a **transition
representation**, not as an exercise in statistical prediction. It demonstrated
that broad strategic advantage could be learned, but broad advantage was not
enough to construct the next legal state.

The simulation-to-data infrastructure and feature engineering developed during
this phase were retained and became inputs to the next approach.

### 5.2 Predicting the board node by node

The next model increased the output resolution.

Instead of asking only what proportion of territory the attacker might control,
the expensive large-graph model generated concrete successor `GlobalState`
objects. Those states could then be converted into supervised labels for each
territory.

The learning task became questions such as:

> What is the probability that the attacker controls this node after the
> transition?

and

> How many troops are expected to remain on this node given its final owner?

Macro variables from the statistical phase could be retained as global context
and combined with local features such as initial ownership, troop count,
neighbouring ownership, neighbouring troop strength and frontier position.

Random Forest classifiers and regressors were trained on these labels.
Historically, the continent-specific models achieved strong predictive metrics,
including capture ROC-AUC values around $0.985$–$0.995$.

Those results should be interpreted carefully.

The Random Forests were relatively accurate at reproducing the targets they
were given. The larger problem was that the **training targets themselves were
not yet sufficiently valid representations of optimal large-graph play**.

Several sources of target error were later identified:

- the small-graph policies used older objective functions that were later
  revised;
- higher troop counts could rely on the plateau approximation before its
  limitations were understood;
- the validity of the large-graph partition approximation had not yet been
  tested against full exact references;
- some attacker/defender count edge cases were poorly represented because
  suitable small-graph coverage was missing; and
- the historical train/test evaluation used row-level random splitting, which
  could overstate generalization when related node rows originated from related
  scenarios.

The most useful interpretation is therefore:

> **The node-level model showed high predictive performance relative to its
> generated labels, while the validity of those labels as a representation of
> optimal play was still uncertain.**

This distinction is important. The RF model was not simply a failed predictor.
It successfully learned a transition generator whose own assumptions were later
shown to require correction.

### 5.3 Edge-case coverage and restricted graph families

One practical source of biased training data was insufficient coverage of
unbalanced local patterns, for example many attacker nodes against few defender
nodes or the reverse.

Some of these graphs had too many possible topology/state combinations to
precompute naively at the desired troop caps, even though particular structural
special cases were tractable.

Star-topology libraries were one response. Because a star graph restricts the
possible edge structure, exact policies can be precomputed for otherwise awkward
node-count patterns without enumerating every graph of that size.

This did not solve every training-data issue, but it increased the range of
large-graph states for which the target generator could rely on explicit
small-graph policies rather than extrapolation or missing coverage.

### 5.4 The joint-state problem

Even if every node prediction were individually accurate, a deeper
representation problem remained.

Independent node predictions describe marginal probabilities. They do not
necessarily describe one legal board state.

A model might assign individually plausible capture probabilities to several
nodes that cannot all be captured in the same battle sequence, or combine troop
outcomes that originate from mutually exclusive paths.

The node-level model therefore recovered **state identity** but not **joint
dependence**.

This is the same information problem encountered earlier at a different scale:

- the macro model discarded node identity;
- the node model restored identity but could still discard the dependencies that
  make a collection of node outcomes one coherent state.

The desired transition target became a distribution over complete successor
signatures rather than independent node marginals.

### 5.5 Joint successor-state prediction

The later experimental pipeline therefore shifted toward learning

$$
P(S' \mid S, \text{policy or active-player transition})
$$

over complete successor states.

Concrete simulated successors preserve correlated ownership and troop outcomes
because every node in a signature comes from the same realized transition.

The current experimental implementation uses joint-state datasets and a
retrieval/KNN-style transition distribution rather than independent Random
Forest marginals. The surrounding Stage A infrastructure generates and
calibrates target successor distributions; it is not itself the deployed
prediction model.

The purpose of this work is to provide a transition kernel that can be sampled
repeatedly in a multi-turn simulation without producing internally inconsistent
boards.

However, the project deliberately stopped short of treating the current
joint-state pipeline as a finished replacement for the RF model.

Before retraining the surrogate, the target generator itself must be corrected.
Otherwise the project risks repeating the earlier mistake: learning a flawed
large-graph approximation with high predictive accuracy and then confusing
agreement with that approximation for strategic validity.

The current priority is therefore:

1. improve and validate the large-graph hybrid model;
2. define stable policy/tie semantics for target generation;
3. generate corrected successor-state distributions; and only then
4. train and validate the joint-state transition model.

---

## 6. Full-board multi-turn model

A one-step transition model becomes useful for full-game analysis only when its
output can be passed to the next player and the process repeated.

The first workable multi-turn model was built on the node-level Random Forest
transition predictor.

Conceptually:

$$
S_t
\rightarrow
\text{player A transition}
\rightarrow
S_{t+1}
\rightarrow
\text{player B transition}
\rightarrow
S_{t+2}
\rightarrow \cdots
$$

This created a new modelling layer. A full turn contains more than combat, and a
full board contains more than one continent-scale battle graph.

The public implementation of this stage is collected under
`full_board_model/`.

### 6.1 From one combat transition to a turn

A continent-scale transition predictor primarily approximates the result of an
active combat process.

A multi-turn board simulator must also account for mechanics that change the
state between combat phases, including:

- reinforcement allocation;
- redistribution of troops;
- fortification or reallocation;
- competing continent objectives;
- shared frontier nodes;
- alternating player perspectives; and
- the fact that resources committed to one objective cannot be reused
  simultaneously somewhere else.

These mechanics form a separate approximation layer around the learned combat
transition.

### 6.2 Strategic commitments

The full-board model introduces strategic commitments at a coarser level than
individual tactical attacks.

A player may, for example, commit resources toward one or more continent
objectives. That commitment affects reinforcement and allocation decisions and
determines which parts of the board are emphasized during the subsequent
rollout.

This creates a useful separation of responsibilities:

- the tactical/transition layers estimate how combat changes states;
- the full-board layer chains those transitions while managing resources and
  player turns; and
- the strategic evaluation layer compares the outcomes of different
  commitments.

### 6.3 Historical RF-based rollout

The first multi-turn implementation used the historical node-level RF models to
predict the next state at each transition.

This was enough to demonstrate the architecture of repeated player turns and to
connect continent-level prediction with board-level strategic mechanics.

It also inherited the weaknesses of the RF target generator.

A multi-turn simulation can only be as valid as the one-step transitions it
chains together. Errors in small-graph objectives, plateau approximation,
regional decomposition or node-level independence do not disappear when the
transition is repeated; they can accumulate over several turns.

The RF-based multi-turn system should therefore be understood as a valuable
architectural prototype rather than a validated final game simulator.

### 6.4 Joint-state particle rollout

The later joint-state work was intended to replace independent node predictions
with samples from complete successor-state distributions.

This naturally led to a particle-based multi-turn architecture. Instead of
propagating one deterministic expected board, the simulator can maintain a
bounded collection of possible board states, sample or weight successors, merge
equivalent states where useful, and continue through alternating player turns.

The advantage is conceptual as much as statistical: uncertainty is propagated
as a distribution over coherent boards rather than collapsed into independent
expected node values.

This route remains experimental. Its transition targets still depend on the
large-graph model, and that model is currently being revised toward the
large-graph hybrid architecture described above.

### 6.5 Current integration gap

The full-board layer therefore contains two modelling generations:

1. a historical RF-based rollout that made multi-turn simulation operational;
   and
2. a later joint-state/particle direction intended to provide a more coherent
   stochastic transition model.

The most recent exact-first and coupling-aware conclusions from the
`continent_model` layer have not yet been integrated end to end through this
full-board pipeline.

That is an important current boundary of the project. The preferred development
order is to validate the large-graph transition generator first and only then
retrain and reconnect the learned transition and multi-turn layers.

---

## 7. Strategic evaluation implementation

The conceptual role of utility was introduced at the beginning of this
document. The concrete strategic evaluation modules sit at the other end of the
pipeline.

Once a multi-turn rollout produces a terminal or horizon state, that outcome
must be scored in a way that allows strategic commitments to be compared.

The public repository separates this into reusable terminal utility evaluation
and experimental commitment-profile analysis.

### 7.1 Terminal utility

`utility_terminal.py` evaluates the strategic quality of a compatible resulting
state.

Conceptually, this module is independent of how the state was produced. A state
could come from an exact solver, a partition-based approximation, a learned
transition model or a multi-turn rollout; the terminal evaluator consumes the
result rather than generating the transition itself.

This makes utility a useful boundary between prediction and evaluation.

### 7.2 Commitment profiles and payoff matrices

`game_theory_commitment.py` operates at the strategic level.

A commitment profile specifies the higher-level objectives pursued by the
players. For a given profile, the full-board rollout generates resulting states,
and terminal utility converts those states into payoffs.

Repeating the process over combinations of commitments produces a payoff matrix:

$$
\text{commitment profile}
\rightarrow
\text{multi-turn rollout}
\rightarrow
\text{terminal state}
\rightarrow
\text{utility}
\rightarrow
\text{payoff}.
$$

The module then compares the strategic consequences of different commitment
combinations.

The term *game theory* should therefore be interpreted narrowly in the current
implementation. The module constructs and evaluates strategic payoff structures;
it does not currently solve for or select a Nash equilibrium, and it does not
replace the tactical policy optimization performed by the lower layers.

---

## 8. What the modelling approaches taught us

Although the repository is now organized as a pipeline, the project did not
develop as one clean top-down implementation.

Different approximations failed for different reasons, but the failures share a
common theme: information was discarded at one stage and later turned out to be
needed by the stage that consumed its output.

The small-graph plateau approximation attempted to discard explicit high-troop
policy calculation. Better exact computation showed that this approximation was
unnecessary for the intended small-graph range.

Representing a solved small graph by one value or one policy could discard
distributional differences between equally valued policies.

Regional decomposition preserved exact local policy distributions but could
discard sequence dependence between regions.

Macro-statistical prediction compressed the large graph so strongly that the
identity of the successor state was lost.

Node-level machine learning restored local identity but predicted nodes
independently, discarding joint dependence.

Multi-turn rollout then made each of these representation choices more
consequential because the output of one transition became the input to another.

The general principle that emerged is:

> **Information may be discarded only when it is irrelevant not merely to the
> current objective, but also to the decisions that consume the model's output.**

This principle explains why concrete successor-state distributions became
increasingly central to the project.

A distribution is not retained simply for statistical completeness. It is an
interface between decision stages.

The same principle also explains the current preference for larger exact
regions in the large-graph model. If a dependency can still affect future
actions, splitting it away before optimization is not a harmless
simplification.

---

## 9. Current state and open problems

The strongest mature components of the project are currently:

- the node-to-node stochastic combat kernel;
- exact finite small-graph policy solving;
- policy-aware successor distributions;
- graph canonicalization;
- generation and lookup of exact small-graph libraries; and
- the infrastructure for composing those local solutions inside larger combat
  graphs.

The large-graph model is the main current research frontier.

Validation has shown that regional decomposition can be highly accurate for
weakly coupled structures and seriously wrong for strongly coupled ones.
Improvements to the exact solver have simultaneously shown that some graphs
previously assumed to require approximation can instead be solved explicitly.

The resulting large-graph hybrid direction is therefore:

$$
\text{solve exactly when practical}
\rightarrow
\text{preserve strongly coupled exact regions}
\rightarrow
\text{partition only where coupling is weak}
\rightarrow
\text{approximate only when necessary}.
$$

Several open problems follow directly from that design.

### 9.1 Exact-routing boundary

The practical exact boundary depends on node count, attacker/defender balance,
troop caps, topology, implementation and hardware. It is therefore better
treated empirically than as one fixed maximum graph size.

A robust production router still needs a principled way to decide when full
exact calculation is affordable.

### 9.2 Selecting sufficient coupled regions

For graphs beyond the full-exact boundary, the model still needs a general
method for finding the smallest region that preserves strategically important
coupling.

Sequence openings, cross-partition follow-up attacks, outcome-dependent stopping
and front switching are known warning signs, but they are not yet a complete
selection rule.

### 9.3 Policy identity and training labels

Exactly tied small-graph policies can produce different successor
distributions. A downstream training pipeline must therefore define whether the
target represents:

- one canonical policy;
- a specified mixture of tied policies;
- a strategic tie-break; or
- explicit policy identity.

This issue must be resolved before large new transition datasets are generated.

### 9.4 Rebuilding the learned transition model

The historical RF models demonstrated that large-graph behaviour can be learned
with high predictive accuracy relative to generated labels, but those labels
were based on assumptions that were later revised.

The joint-state transition model should therefore be retrained only after the
large-graph hybrid target generator has been corrected and validated.

Its evaluation should focus not only on marginal node metrics but on the
distribution of complete successor states and on how errors accumulate across
several consecutive turns.

### 9.5 Full-board integration

The latest large-graph hybrid logic is not yet integrated through the
full-board multi-turn system.

A complete next stage would connect:

```text
validated large-graph transition generator
    -> joint-state surrogate
    -> multi-turn particle rollout
    -> strategic commitment evaluation
```

and compare the resulting strategic conclusions against exact or otherwise
trusted references wherever those references remain tractable.

---

Project Risk is therefore not one solver followed by a collection of discarded
experiments. It is a layered attempt to determine where exact stochastic
reasoning is feasible, where approximation is necessary, and what information
must survive the boundary between one decision problem and the next.

The question that continues to organize the project is the same one that
motivated it from the beginning:

> **What representation is detailed enough to support the next strategic
> decision while remaining computationally tractable?**
