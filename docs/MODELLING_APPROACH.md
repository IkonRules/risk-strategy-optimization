# Modelling Approach

Project Risk is an attempt to build a mathematical framework for evaluating
strategy in a stochastic, graph-based game.

The board game Risk provides a useful environment for this problem. The state
of the game can be represented naturally as a graph: territories are nodes,
borders are edges, ownership and troop counts define the state of each node,
and combat introduces stochastic transitions between states. At the same time,
every successful attack changes the graph on which future decisions are made.
A conquest may open a new front, close another, move troops to a different
part of the graph, or make a previously impossible action available.

The central problem is therefore not simply to estimate whether an individual
attack will succeed. It is to determine which sequence of actions should be
preferred when actions have stochastic outcomes and those outcomes change the
set of decisions available afterwards.

A complete solution could in principle be obtained by enumerating every legal
action, every stochastic outcome, and every subsequent action until the game
terminates. The difficulty is computational. As the number of territories,
troop configurations and possible action sequences increases, explicit
enumeration becomes increasingly expensive.

Much of the project has consequently revolved around one recurring question:

> **How much of the full decision problem can be simplified, compressed or
> precomputed without discarding information that matters for later decisions?**

Different stages of the project have answered that question in different ways.
Some approaches proved useful, some exposed limitations in the representation
being used, and several ideas developed for earlier models remain important
components of the current system.

---

## 1. From strategy to a mathematical objective

Before searching for an optimal strategy, optimality has to be defined.

Winning the complete game is the obvious ultimate objective, but it is not a
particularly useful quantity for evaluating every intermediate decision.
Strategic progress must therefore be represented through quantities that can be
evaluated at shorter horizons.

At a high level, the project treats acquisition and retention of territory as
the basis of strategic progress. Continents provide a natural higher-level
structure because control of a complete continent has strategic value beyond
the individual territories it contains. This motivated an early game-theoretic
framework in which players could be viewed as committing to strategic goals,
such as sets of continents to pursue, and the outcomes of those commitments
could be evaluated through a utility function.

This also introduced an important distinction between **strategic actions** and
**tactical actions**.

A strategic action describes what a player is trying to achieve over a longer
part of the game. A tactical action is a concrete battlefield decision: which
territory to attack, from where, and how to continue after the outcome becomes
known.

The distinction is useful because the two levels operate at very different
scales. Enumerating arbitrary combinations of individual territories as
strategic objectives creates an enormous choice space even before stochastic
combat is considered. Continents provide a coarser strategic representation.
At the tactical level, however, the identity of individual territories and
their connections cannot be discarded because they determine which actions
are legal.

This leads to two separate modelling tasks.

The first is **prediction**:

> Given a state and a set of actions, what states may follow and with what
> probabilities?

The second is **evaluation**:

> Given a possible outcome, how desirable is that outcome?

An optimizer requires both. A sophisticated utility function cannot compensate
for an incorrect transition model, and a perfect transition model cannot
choose between actions unless preferences are defined.

This distinction between predicting states and evaluating states became
increasingly important as the project developed.

---

## 2. The smallest stochastic building block: node-to-node combat

The first prediction problem is combat between two adjacent hostile
territories.

Under a fixed combat policy, a battle can be represented as a finite absorbing
Markov chain. A transient state records the number of attacking and defending
troops still able to participate in the battle. Dice outcomes create
probabilistic transitions between these states. Eventually either the defender
is eliminated or the attacker can no longer continue, producing an absorbing
state.

Writing the transition matrix in canonical absorbing form,

\[
P =
\begin{bmatrix}
Q & R \\
0 & I
\end{bmatrix},
\]

the fundamental matrix is

\[
N=(I-Q)^{-1},
\]

and the probabilities of absorption in each terminal state are

\[
F=NR.
\]

The resulting row of \(F\) gives considerably more information than a single
probability of victory. It gives the probability distribution over all possible
terminal troop configurations for the battle.

This distribution becomes the elementary stochastic transition kernel for the
rest of the model.

The implementation is contained primarily in
`markov_matrix_probabilities.py`.

Two simplifying assumptions are important. Combat is treated as continuing
until either the defender is eliminated or the attacker can no longer proceed,
and the maximum number of dice permitted by the model is used. These
assumptions make the tactical action space manageable at the node-to-node
level. They are modelling assumptions rather than claims that the same behaviour
must always be globally optimal in every complete Risk position.

The important architectural idea is that the higher-level solver does not need
to branch over every individual dice roll. Once it chooses an attack, the
Markov model supplies the complete distribution of outcomes for that battle.

---

## 3. From battles to policies on small graphs

Adding a third territory changes the nature of the problem.

In a two-node battle, once the combat assumptions are fixed there is
essentially one tactical course of action. On a graph with several hostile
edges, a player must choose which battle to initiate. The result of that battle
then determines what actions are available next.

The model has therefore moved from a probability problem to a sequential
decision problem.

A state consists of the ownership and troop count of every node in the active
combat graph. From a state, the solver enumerates all legal attacks. For each
attack, the Markov combat model provides the complete set of possible battle
outcomes and their probabilities. Each outcome produces a new graph state, from
which another decision may have to be made.

Conceptually, this creates a game tree. In practice, many branches reach the
same state, so the structure is better represented as a directed acyclic graph
of states. Solving the same successor state repeatedly would be wasteful, and
the model therefore uses memoization and dynamic programming to reuse previously
computed results.

For a state \(s\), a legal action \(a\), and possible successor states \(s'\),
the reasoning is Bellman-like:

\[
V(s)
=
\max_{a \in A(s)}
\sum_{s'}
P(s' \mid s,a)V(s').
\]

Terminal outcomes are evaluated, their values are propagated backwards through
the state graph, and the action producing the preferred expected outcome is
selected at each decision point.

The local objective currently prioritizes three quantities lexicographically:

1. expected number of newly conquered territories;
2. expected number of final attacker troops;
3. probability of complete local conquest.

The ordering is deliberate. Territorial progress comes first because a policy
that simply preserves troops by avoiding combat should not dominate a policy
that actually achieves the strategic objective. Once territorial outcomes are
equal, conserving troops becomes valuable. Full conquest then provides an
additional preference when the first two criteria are equal.

This exact small-graph problem is represented primarily in
`small_graph_outcome_probabilities.py` and, in its more computationally efficient
form, `exact_finite_solver.py`.

---

## 4. Why an optimal value is not enough

An important complication appears when several policies have the same expected
local value.

Suppose two policies conquer the same expected number of territories, leave the
same expected number of attacker troops and have the same probability of local
conquest. Under the local objective they are equally good.

They need not, however, be strategically equivalent.

One policy may leave most surviving troops on a border node while another
leaves them farther from the next front. Locally, that difference may not
change the value at all. In the next battle it may completely change the
available actions.

For this reason, the model increasingly moved away from representing a solved
state only by an optimal value or even by one optimal action. A useful solution
must preserve the **distribution of concrete successor states** associated with
the policy.

This also means that multiple equally valued policies may need to be retained.

Early versions preserved different optimal root actions. Later `state_set`
representations allowed policy alternatives to differ further down the decision
graph even when the opening action was identical. The policy-DAG work extended
this idea further by explicitly representing tied optimal decisions at multiple
depths.

The general lesson is important for the rest of the project:

> **Equality under the current objective does not imply equality as a transition
> to the next decision problem.**

The information passed forward by the model therefore matters just as much as
the scalar criterion used to select policies.

---

## 5. Turning exact solutions into reusable strategy libraries

Solving one small graph exactly is useful, but a complete game simulation may
encounter the same tactical structure many times.

This motivated one of the central architectural ideas in Project Risk:
**precompute exact policies for small combat graphs and reuse them whenever the
same situation appears during simulation.**

The expensive dynamic programming is therefore moved largely offline.

A library is built for a pattern containing a given number of attacker and
defender nodes. For each supported graph topology, every initial troop
configuration up to a chosen cap is evaluated. The resulting policy-specific
terminal distributions are stored so that the full game simulation can later
query them directly.

At runtime the intended flow is approximately:

\[
\text{board state}
\rightarrow
\text{active battle region}
\rightarrow
\text{canonical graph}
\rightarrow
\text{troop configuration}
\rightarrow
\text{precomputed policy distribution}.
\]

This turns repeated tactical optimization into an amortized-computation
problem: solve the expensive stochastic decision problem once, then reuse the
solution many times.

### Graph canonicalization

A major source of redundant computation is node labelling.

Two graphs may have identical topology and ownership structure but different
territory names or local node numbers. Solving both independently would
duplicate the same decision problem.

The model therefore canonicalizes graphs under relabellings that preserve the
attacker and defender roles. Attacker nodes may be permuted among attacker
nodes and defender nodes among defender nodes, but the roles themselves are
never exchanged.

The canonical graph becomes the library identity. When a region is queried
during a full-board simulation, the concrete territory labels are mapped to the
canonical representation, the corresponding policy is retrieved, and the
result can subsequently be mapped back to the original board.

Canonicalization therefore serves two roles at once. It reduces the number of
topologies that must be solved, and it allows solutions calculated independently
of the full board to be reused in concrete labelled positions.

### From large tables to compact probability stores

The storage system itself has evolved substantially.

Earlier implementations represented exact outcomes through relatively verbose
tables, matrices, dictionaries and DataFrames. As the number of graph
topologies, troop configurations and policy alternatives increased, this became
expensive both in memory and storage.

Later versions moved toward compact vectorized payloads. An outcome
distribution is represented by aligned arrays containing probabilities, owner
states, troop counts and derived metrics. Multiple policy alternatives for one
initial state can be grouped while retaining a complete independently
sampleable distribution for each policy.

Rows are then stored in bounded chunks rather than one enormous object.

The current library pipeline separates generation from consumption.
`create_library.py` enumerates and solves the canonical graph/state space and
writes the generated libraries. `library_io.py` defines the format-aware lookup
boundary and loads only the chunk containing the required state. The compact
exact solver supplies the policies during library construction; it is not
normally rerun every time a library-backed region is encountered during
simulation.

This design made it possible to build libraries containing millions of solved
initial states. One recorded `2A3D` build covered 98 canonical topologies and
1,647,086 troop configurations. As policy representations became richer, the
generated libraries themselves eventually grew to many gigabytes.

That growth produced another important insight: after computational improvements
had made many exact solutions feasible, **representation and storage could
become limiting factors before the underlying dynamic programming did**.

The library system is therefore not merely an implementation detail. It is the
mechanism by which exact small-graph reasoning is intended to become usable as
part of a larger strategy system.

---

## 6. The scaling problem

Exact small-graph policies do not remove the fundamental combinatorial problem.

As graphs grow, so do the number of possible topologies, ownership states, troop
configurations, legal attack sequences, movement choices, stochastic branches
and potentially tied policies.

The project has explored several ways of crossing this boundary.

These approaches were not independent experiments added arbitrarily to the
project. Each attempted to discard or approximate some part of the full state
space while retaining enough information to make useful strategic decisions.

Their limitations gradually clarified what information the next stage of the
model actually requires.

---

## 7. First approximation: describing the board statistically

One early approach was to replace the detailed board state with a smaller set of
strategic descriptors.

A complex position could be summarized using variables such as territory
balance, troop balance, troop concentration, graph topology, reserve distance
and the proportion of forces actually participating in the active battle.

The hope was that these macro variables could predict the expected outcome of a
larger strategic interaction without explicitly modelling every tactical path.

The statistical work developed in stages.

Initial multiple regression models showed that simple variables such as troop
and territory ratios already contained substantial predictive information. One
historical two-variable fit reached approximately

\[
R^2 \approx 0.952.
\]

The high fit was encouraging, but the residuals were not structureless.
Interactions and polynomial terms were introduced, followed by models more
appropriate for bounded outcomes. Generalized linear models were explored, and
eventually generalized additive models with splines were used to represent
nonlinear relationships without imposing one global polynomial form.

This phase also produced increasingly rich feature engineering: coefficients of
variation and Gini measures for troop concentration, graph statistics, measures
of active versus reserve forces, and distances between reserves and the current
front.

The statistical models became better at describing broad strategic outcomes.

But a more fundamental problem remained.

A macro model estimates something like

\[
E[Y_{t+1}\mid M(S_t)],
\]

where \(M(S_t)\) is a compressed description of the current board.

The next tactical decision does not merely require an expected number of
territories or an expected troop balance. It requires a concrete state, or a
distribution over concrete states:

\[
P(S_{t+1}\mid S_t,\pi).
\]

Two board positions can have almost identical territory ratios, troop ratios,
Gini coefficients and coarse topology while having completely different
borders and legal attack sequences.

No increase in regression flexibility can reconstruct information that was
removed before the model was fitted.

The main result of the macro phase was therefore not that regression “failed”.
It identified the difference between predicting **strategic advantage** and
predicting a **usable state transition**.

The feature engineering and simulation-to-data infrastructure developed during
this period were retained and became inputs to the next modelling approach.

---

## 8. Second approximation: predicting the board node by node

The natural response to the macro-state limitation was to predict outcomes at a
finer resolution.

Instead of asking only what proportion of territory the attacker might control,
Monte Carlo simulations could generate concrete future `GlobalState` objects.
Those states could then provide supervised labels for individual territories.

The learning problem became questions such as:

> What is the probability that the attacker controls this node after combat?

and

> How many troops are expected to remain on this node given its final owner?

Macro variables from the previous phase could now be combined with local
information such as node ownership, troop count, neighbouring ownership,
neighbouring troop strength and frontier position.

Random Forest models produced very strong historical predictive metrics, with
recorded capture ROC-AUC values around 0.985–0.995 for the continent-specific
models.

Two limitations subsequently became important.

First, the evaluation used a random row-level train/test split. Several node rows
may originate from related game states, so the historical scores may be
optimistic compared with a state- or scenario-grouped evaluation.

The more important limitation was again representational.

Independent predictions for each node provide marginal probabilities. They do
not necessarily describe one possible board.

A model might assign individually plausible probabilities to several captures
that cannot occur together, or combine troop outcomes arising from mutually
exclusive battle sequences.

The output therefore lacked the joint dependency structure required for the
next turn.

This led to experiments in which the target became a distribution over complete
successor-state signatures rather than independent node outcomes. That work
progressed into joint-state datasets and prototype multi-turn sampling, but it
did not reach a final validated transition model.

The node-level phase nevertheless made the representation problem clearer:
moving from macro outcomes to local predictions recovered state identity, but
**state identity without joint dependence was still insufficient**.

---

## 9. Extending exact solutions instead: plateau and local motifs

Another family of approaches attempted to avoid statistical prediction and
instead extend exact small-graph behaviour beyond the explicitly solved range.

One idea was that policies might reach a plateau as troop counts increase. If
the optimal action stabilised, perhaps the same policy structure could be reused
for higher troop configurations without explicitly solving every state.

A related idea treated previously solved small graphs as puzzle pieces or
macro-actions from which larger strategies might be assembled.

Both ideas were motivated by a sensible observation: nearby tactical problems
often have very similar solutions.

The difficulty is that similarity at the root does not guarantee equivalence of
the complete policy.

The same opening attack can lead to different subsequent decisions as troop
counts and stochastic outcomes change. Likewise, a policy that is optimal when
a small graph is considered in isolation need not remain optimal when its border
is connected to the rest of the board.

These experiments clarified the distinction between **reusing an exact
transition** and **assuming that a locally optimal policy remains globally
optimal**.

A solved local graph can safely serve as a transition operator or candidate
macro-action. Its policy cannot automatically be treated as context-free proof
of global optimality.

---

## 10. Regional decomposition: composing exact local policies

The most substantial attempt to scale the exact-library approach was regional
decomposition.

The basic hypothesis was attractive:

> If a large combat graph can be partitioned into smaller regions for which exact
> policies are already available, perhaps the large-graph policy can be
> approximated by combining those regional policies.

This allowed the project to reuse the expensive work already stored in the
small-graph libraries.

The resulting pipeline became considerably more elaborate than simply solving
each region independently.

A board state is converted to an active battle graph. Supported partitions are
generated from regions represented by the exact libraries. For each region, the
concrete graph is canonicalized, the appropriate troop row is located and one
or more policy-specific successor distributions are retrieved.

Candidate partitions and policy combinations are then compared.

Because several locally optimal policies may exist in the same region, the
ranking layer can retain combinations of policy alternatives rather than
collapsing each region immediately to one choice. Local expected utility is used
to remove clearly inferior candidates.

The remaining problem is interaction between regions.

A policy that is locally equivalent at the end of the first wave may leave
troops in positions that make very different second attacks possible. The model
therefore introduced a second stage: sample successor states from the regional
distributions, reconstruct the resulting global board, redraw the active battle
regions and evaluate the next wave.

Monte Carlo simulation was used to estimate this downstream effect.

This became a serious production-like route through the research system:

\[
\text{board state}
\rightarrow
\text{battle graph}
\rightarrow
\text{partition}
\rightarrow
\text{library lookup}
\rightarrow
\text{policy combinations}
\rightarrow
\text{second-wave evaluation}.
\]

The architecture also demonstrates why the exact policy libraries are central
to the complete project. The exact solver operates offline to create reusable
local strategy distributions; the regional layer consumes those distributions
when evaluating larger board states.

---

## 11. Testing the decomposition assumption

Regional decomposition created a testable modelling assumption:

> Local exactness is sufficient for global approximation when interactions
> between regions are weak enough.

Validation showed that this is sometimes true and sometimes very false.

For weakly connected bridge-like cases, regional composition reproduced the
full exact distribution extremely well. Across nine retained bridge cases, the
mean total-variation distance was approximately

\[
0.0061.
\]

In double-front cases, the behaviour was completely different. Across ten
cases, mean total-variation distance was approximately

\[
0.798,
\]

and several cases had total variation equal to one.

The difference was structural.

In strongly coupled graphs, success in one region may open an attack into
another. The active front may depend on the realized stochastic outcome.
Stopping in one branch may leave troops available elsewhere. A regional
partition chosen before these events occur may therefore impose an independence
structure that the real tactical process does not have.

Two separate issues were also identified.

One was how regional distributions were **composed**. Monte Carlo sampling
introduced unnecessary noise once the local distributions were already known.
Replacing that step with exact Cartesian composition was both faster and free
from sampling error in the tested cases.

The second issue was whether the graph should have been **decomposed at all**.

Making candidate selection exact changed a substantial number of choices, but
the severe double-front failures remained. Improving the ranking algorithm did
not restore dependencies that had already been discarded when the graph was
partitioned.

This was an important distinction:

> **Exact composition cannot repair an invalid independence assumption.**

Regional decomposition was therefore not rejected completely. It remains a
useful approximation for sufficiently weakly coupled structures. What changed
was the idea that it should automatically be the default whenever a graph
exceeded some assumed small exact size.

---

## 12. Re-evaluating the boundary of exact computation

The decomposition experiments coincided with improvements to the exact solver.

State packing, shared caches, canonicalization, precomputed combat rows,
separation of value calculation from distribution reconstruction and more
efficient library formats all reduced the practical cost of exact solving.

The measured state spaces also turned out to be much smaller than loose
combinatorial upper bounds had suggested.

In one tractability study, all 360 tested cases covering graphs with six to
eight nodes and troop caps from three to five completed within the specified
resource budget; the worst recorded runtime was below one second. A broader
315-cell experiment completed 311 cases under a ten-second stop, with the four
remaining cases stopped by runtime rather than memory or state-count limits.

These results do not make the full problem non-combinatorial.

They do change the engineering decision.

If a graph can be solved exactly within a reasonable measured budget, there is
little reason to introduce an approximation whose independence assumptions then
have to be justified and validated.

This produced a reversal in the preferred routing logic.

Earlier development had implicitly asked:

> The graph is getting large; how can we approximate it?

The later model asks first:

> Is approximation actually necessary for this graph?

---

## 13. The emerging exact-first architecture

The current direction is therefore an **exact-first** architecture.

For an active combat graph, the preferred hierarchy is conceptually:

\[
\text{full exact}
\rightarrow
\text{coupled exact region}
\rightarrow
\text{weakly coupled regional composition}
\rightarrow
\text{joint-state approximation}.
\]

The full graph should be solved exactly whenever empirical resource limits allow
it.

If it is too large, the next objective is not immediately to partition it into
independent pieces. Strongly interacting parts of the graph should remain inside
one exact region if possible.

Independent regional composition becomes appropriate only when the discarded
coupling is sufficiently weak.

Beyond that boundary, approximation or learned joint successor-state
distributions remain possible fallbacks.

This architecture is not yet implemented as one final automatic router.
Determining the smallest sufficient coupled region is itself an open modelling
problem, and the later joint-state machine-learning pipeline still requires
further development and validation.

The exact-first direction should therefore be understood as the current
conclusion of the modelling work rather than as a completed end-to-end system.

---

## 14. How the pieces fit together

The project can be viewed as several modelling layers connected by the type of
information they produce.

At the lowest level, the Markov combat model supplies exact stochastic outcomes
for an individual battle.

The exact graph solver combines those battle distributions with tactical
decision-making and returns optimal policy distributions on tractable local
graphs.

Canonicalization and caching reduce repeated computation, while the library
system converts these local solutions into reusable strategy data that can be
queried repeatedly from full-board simulation.

For larger battle graphs, the regional system attempts to combine library-backed
local policies while accounting for downstream interactions. Validation of this
layer determines when the approximation is defensible.

The statistical and machine-learning work occupies a complementary branch of
the same problem. It investigates whether larger transitions can be predicted
without explicit enumeration, and in doing so has progressively moved from
aggregate outcomes toward complete joint-state distributions.

The full conceptual flow is therefore closer to:

\[
\text{strategic objective}
\rightarrow
\text{utility}
\rightarrow
\text{state-transition model}
\rightarrow
\text{exact combat}
\rightarrow
\text{exact local policies}
\rightarrow
\text{precomputed policy libraries}
\rightarrow
\text{large-graph decision problem}
\rightarrow
\text{validated approximation or further exact solving}.
\]

The project is not one solver followed by several discarded experiments.

It is a sequence of increasingly demanding questions about what a useful
strategy model must preserve.

---

## 15. What the failed approaches taught us

The different approximation attempts eventually revealed variations of the same
problem.

Macro-statistical models discarded the identity of the future state.

Independent node models restored local identity but discarded joint dependence.

Regional decomposition preserved exact local distributions but could discard
cross-region sequence dependence.

Representing a solved state by one optimal value or one policy could discard
distributional differences between equally valued policies.

These limitations suggest a more general principle:

> **Information may be discarded only when it is irrelevant not merely to the
> current objective, but also to the decisions that consume the model's output.**

This principle explains why successor-state distributions became increasingly
central to the project.

The purpose of retaining a distribution is not simply statistical completeness.
The distribution is the interface between one decision stage and the next.

---

## 16. Current state of the project

The strongest mature components of the model are the stochastic combat kernel,
the exact finite graph solver, canonicalization, policy-aware successor
distributions and the infrastructure for generating and querying exact
small-graph libraries.

The library-backed regional pipeline represents the most substantial
implemented attempt to use those exact local solutions inside larger battle
graphs and full-board simulation. Its limitations are now better understood
because it has been compared against full exact references.

The machine-learning branch has produced useful feature engineering and strong
historical predictive results, but a final validated joint transition model has
not yet replaced the exact/regional route.

The most recent work therefore shifts the emphasis from finding ever more
powerful approximations toward measuring where approximation is actually
necessary and preserving exact stochastic structure whenever computation makes
that possible.

A small self-contained exact example is included in the public repository so
that the mathematical core can be inspected and reproduced without the large
generated policy libraries. It should be viewed as a demonstration of one
component of the larger system rather than as a representation of the complete
Project Risk architecture.

---

## 17. Open problems

The project still leaves several substantial questions open.

The exact tractability boundary depends on graph structure, troop limits,
hardware and implementation details and is therefore better treated
empirically than as a fixed node-count rule.

For graphs beyond that boundary, a general method is still needed for
identifying the smallest region that preserves strategically important
coupling.

Policy ties also remain relevant whenever successor distributions are used as
training targets or passed to later decision stages. A downstream system must
either preserve policy identity or define a principled convention for choosing
between equally valued policies.

Finally, a learned transition model should ultimately be evaluated on its
ability to reproduce coherent joint successor states over several consecutive
decision stages, rather than only on marginal prediction metrics.

These problems continue the same question that motivated the project from the
beginning:

> **What representation is detailed enough to support the next strategic
> decision while remaining computationally tractable?**