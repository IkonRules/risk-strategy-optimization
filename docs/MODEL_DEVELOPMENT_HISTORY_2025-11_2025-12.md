# Project Risk: Macro-Statistical Modelling Phase, November–December 2025

> **Document status:** Historical technical reconstruction  
> **Prepared:** 2026-08-13  
> **Primary period:** 2025-11-01 through 2025-12-31  
> **Purpose:** Internal project history, future Codex context, and source material for a public portfolio narrative  
> **Important scope rule:** This phase is described in its own terms. It is not retroactively rewritten as an early version of the later exact finite solver or regional decomposition architecture.

## Evidence conventions

The surviving repository is not a complete historical archive. Its Git history contains only a single `Initial commit` dated 2026-01-09, while most project files are untracked in that repository snapshot. Several historical modules were deliberately removed or reduced to fragments when the project moved to node-level machine learning. This document therefore distinguishes four evidence classes:

- **Direct repository evidence:** surviving source code, file metadata, datasets, saved console output, model bundles, or images.
- **Recovered chat evidence:** a later ChatGPT history search that identified and summarized dated Risk conversations from November–December 2025. The original individual conversations were not directly available in this reconstruction pass, so quoted technical details are attributed to the recovered summary rather than treated as source-code evidence.
- **Strong reconstruction:** a conclusion supported jointly by the recovered chat chronology and surviving successor code.
- **Uncertain:** a detail that cannot be recovered reliably. In such cases this document says so explicitly.

The principal recovered chat dates are 2025-11-27, 2025-12-02, a range of 2025-12-05 through 2025-12-08, and 2025-12-08. Files modified on 2026-01-08 and training artefacts written on 2026-01-09 are used only as near-term corroboration of the transition that began in December.

---

## 1. Executive Summary

During November–December 2025, Project Risk investigated a macro-level statistical approach to forecasting strategic outcomes. The central hypothesis was that a complex Risk position could be compressed into a relatively small set of aggregate descriptors—initially territory control and troop strength—and that those descriptors could predict the expected result of a future battle or turn. This was attractive because the full board state was combinatorial: ownership, troop placement, borders, and legal attack sequences produced far more configurations than could conveniently be modelled directly at that time. A low-dimensional macro state offered a plausible route from simulation to a usable transition model. **[Recovered chat evidence; supported by later code]**

The work did not begin with sophisticated machine learning. It began with manually designed success factors and statistical diagnostics. Surviving archived code uses continent troop shares as a base success signal, applies a fixed attacker advantage, and maps the resulting strengths into conquer/remain/eradicated outcomes. In parallel, the macro experiment framework generated controlled states over territory- and troop-ratio targets, ran repeated simulations, stored realized rather than merely requested ratios, and assembled pandas-style analysis tables. **[Direct repository evidence for the surviving mechanisms; recovered chat evidence for their November chronology]**

An early multiple linear regression appeared promising. The recovered history reports an approximate model

\[
\widehat y \approx 0.0026
  + 0.344\,\text{troops ratio}
  + 0.823\,\text{territory ratio},
\]

with \(R^2 \approx 0.952\). The exact response variable associated with this result is not recoverable from the surviving repository or the summarized history. It was likely a future territory-related aggregate, but that cannot be stated as fact. More importantly, the high aggregate fit did not end the investigation: residuals retained systematic structure, showing that the simple linear form was misspecified. **[Recovered chat evidence; response variable uncertain]**

The project then added interactions, quadratic or polynomial terms, and transformations. Surviving archived code confirms logarithmic, square-root, and logit transformations for territory ratios, troop ratios, troop-distribution CV, Gini, and expected territory outcomes. A quasi-binomial GLM with a logit link and an explicit trial denominator was also implemented. The GLM treated a territory outcome as a proportion, used the corresponding number of territories as binomial weights, and optionally estimated overdispersion using a Pearson-chi-square scale. **[Direct repository evidence]**

On 2025-12-02, the project moved to generalized additive models because quadratic terms still left a pronounced U- or arc-shaped residual pattern. The recovered conversations identify a `fit_binomial_gam` function using `pyGAM` and `LogisticGAM`, with spline terms for territory and troop ratios and a tensor-product interaction conceptually of the form `te(0, 1)`. Diagnostics included residual plots, heatmaps, response surfaces, and troop-effect curves at fixed territory ratios. GAM residuals were reportedly flatter than GLM residuals, but remaining bands were attributed partly to discrete territory denominators and Monte Carlo noise. The original GAM source, plots, smoothing parameters, and numerical fit summaries are no longer present. **[Recovered chat evidence; missing repository artefacts]**

From approximately 2025-12-05 to 2025-12-08, feature engineering expanded. The surviving pipeline computes force-balance ratios, troop-distribution CV and Gini, full-graph edge and degree statistics, diameter, component count, the fraction of the full graph active in battle, the fraction of attacker troops actually deployed into the battle graph, and reserve-to-battle distances. A skewness helper also survives, although the preserved successor pipeline does not place skewness in the final feature dictionary. The recovered conversations additionally mention clustering and frontier-related measures; their exact historical definitions are not recoverable. **[Direct repository evidence plus recovered chat evidence]**

The decisive limitation was representational rather than purely statistical. Even a well-fitted function from macro descriptors to an expected global outcome could not construct the concrete next Risk state required for recursive simulation and strategy optimization. Two positions can share the same territory ratio, troop ratio, Gini, and even coarse topology statistics while differing in which territories are owned, where the border lies, where reserves sit, which attacks are legal, and what tactical options become available next. A GAM can model nonlinear conditional means, but it cannot recreate information that was discarded before fitting. **[Strong reconstruction, explicitly described in recovered conversations]**

On 2025-12-08, the modelling target changed. Rather than predict only a global expectation, Monte Carlo-generated future `GlobalState` objects would supply supervised labels at node level: whether the attacker holds or captures a node and the troop count on that node conditional on the final holder. Macro variables were retained as shared state-level features and combined with local node features. A near-term successor file explicitly calls this the “NEW per-node macro→micro approach,” and the January 2026 datasets contain `attacker_holds_final`, `captured`, and `final_troops` labels alongside the inherited macro and topology columns. **[Recovered chat evidence; directly corroborated by successor code and data]**

This phase should therefore not be described as a series of failed regressions. It established the simulation-to-data pipeline, exposed nonlinearities through residual analysis, produced a reusable vocabulary of strategic features, and—most importantly—identified the information-loss boundary between predicting an aggregate and simulating a legal successor state. That distinction directly motivated the later node/state-level machine-learning work and supplied many of its inputs. The later exact and regional solvers addressed different questions, but they inherited the same discipline: preserve the state and distributional information required by the next decision stage.

---

## 2. Starting Point: The Macro-State Hypothesis

### 2.1 The modelling problem

A Risk position is naturally a graph-labelled microstate. Each territory has an owner and troop count; edges determine adjacency; ownership patterns determine borders; troop placement determines which attacks are legal; and stochastic combat changes both the graph labelling and the future action space. Even before considering multiple players and reinforcement decisions, the number of distinct positions is very large.

The macro-state hypothesis proposed a compression:

\[
\text{complex board state}
\longrightarrow
\text{small vector of strategic descriptors}
\longrightarrow
\text{expected future outcome}.
\]

The recovered 2025 history identifies `territory_ratio` and `troops_ratio` as the initial core variables. They are strategically plausible:

- Territory share measures spatial control and the number of positions from which a player may attack or defend.
- Troop strength measures the material available to convert that control into conquest or resistance.
- Their interaction can express cases in which troops matter differently when a player controls few versus many territories.

### 2.2 Microstate, macrostate, and outcome

These terms must remain separate.

**Microstate** means the concrete state required to continue the game: node-by-node owners and troop counts, embedded in the relevant graph. In later code this is represented by `GlobalState` containing `NodeState(owner, troops)` entries.

**Macrostate** means an aggregate summary calculated from the microstate. Examples include attacker territory share, attacker troop share, Gini concentration of attacker troops, mean graph degree, or mean reserve distance.

**Outcome variable** means the response that the statistical model attempts to predict. Surviving fragments refer to names such as `expected_new_territories_ratio` and `expected_territories_ratio`. The recovered history also describes future territorial control, troop outcomes, and conquest probability as targets explored in this phase. It is not possible to associate every historical coefficient or diagnostic with a single response variable.

### 2.3 Manually designed success factors

The archived `Gammalt/UtilityFunction.py` and the fragmentary `old_functions.py` preserve a manual success-probability layer. Its key ideas were:

- start from each player's fraction of troops on a continent;
- multiply attacking players by a fixed `attack_prob_bonus`, defaulting to `1.2`;
- renormalize those weighted strengths;
- map the resulting success signals into probabilities of conquest, remaining, or eradication;
- combine those outcomes with a hand-designed continent utility based on territory count and reinforcement value.

Several outcome-allocation variants survive. One uses quadratic attacker strength, \(q_i=p_i^2\), and distributes remaining eradication risk inversely to player strength. Another uses competing-risk-like attack intensities and a no-conquest mass. These functions show the design philosophy of the phase: formulate interpretable global factors, encode plausible strategic effects, and inspect whether they reproduce simulated outcomes.

The exact creation dates of these particular function variants are not recoverable. Their files were last modified on 2026-01-08, likely during the cleanup that archived the old subsystem. They are therefore evidence of the approach, not exact proof that every variant existed on a specific November date.

---

## 3. Data Generation

### 3.1 Controlled macro experiments

The recovered history dates the explicit “macro-state transition model” to 2025-11-27 and names a module `macro_state_experiments.py`. That file is not present in the current repository. However, its interface survives almost verbatim in the later `ExperimentConfig`, whose docstring still says “Configuration for a macro-state experiment” and defines:

- a sequence of target territory ratios;
- a sequence of target troop ratios;
- `samples_per_combo` for repeated random states at each ratio pair;
- constraints on graph and troop generation;
- a random seed;
- a selected outcome/ranking variable.

The conceptual experimental grid was therefore:

\[
(r_T, r_A) \in
\{\text{target territory ratios}\}
\times
\{\text{target troop ratios}\},
\]

with repeated state generation and simulation at every grid point.

### 3.2 Target versus realized values

The generator did not assume that a requested ratio was exactly attainable. Territory counts are discrete, troop counts are integer-valued, every owned territory requires at least one troop, and per-node troop caps can make a requested troop ratio infeasible. The surviving `state_generators.py` therefore:

- converts a target territory ratio to an integer number of attacker-held nodes;
- keeps at least one attacker and one defender in non-degenerate generated states;
- calculates the maximum feasible number of “available” attacker troops under node caps;
- clamps the requested troop ratio to that feasible range;
- distributes extra attacker troops subject to the cap;
- rebuilds both the full graph and the active battle graph.

The analysis pipeline records both `target_territory_ratio` / `target_troops_ratio` and realized quantities. This distinction matters for regression. A model trained on requested ratios alone would absorb discretization and feasibility effects as unexplained noise; realized ratios describe the state that was actually simulated.

The exact November implementation may have differed from the surviving successor generator, particularly in whether control was imposed only on a continent or on the continent-plus-neighbours full graph. The target/realized distinction itself is supported by both recovered chat evidence and later code.

### 3.3 Simulation and repeated outcomes

The macro approach required labels that were not analytically available for full board positions. Simulation supplied them. The repository contains a `SimulationEngine.run_many(...)` helper explicitly documented for Monte Carlo analysis, and the 2025 history describes repeated Monte Carlo outcomes being collected into pandas data.

For each controlled initial state, the intended data flow was approximately:

1. choose a target territory/troop-ratio pair;
2. generate a random legal state approximating those targets;
3. calculate the realized macro descriptors;
4. simulate the battle or turn repeatedly;
5. aggregate outcomes such as future territory share, territorial gain, troop outcome, or conquest probability;
6. append one analysis row per initial state, or later one supervised row per node and sampled successor state.

The number of Monte Carlo simulations per macro-state row in November–December 2025 is **not recoverable from the available repository history**. Later successor code uses configurable scenario counts, but those later defaults must not be projected backward.

### 3.4 Surviving visual evidence

`After_Setup.png` and `After_Game.png`, both written on 2025-11-03, show rendered full-board states with player-coloured troop counts before and after a simulation. They do not establish the regression methodology, but they confirm that full-board setup, simulation, and state rendering were already operating before the documented macro-statistical work began later that month.

### 3.5 Dataframe construction

The missing `macro_state_experiments.py` reportedly assembled pandas datasets and regression diagnostics. Near-term successor code retains the same dataframe-oriented pattern. `run_node_transition_experiment(...)` explicitly says it is “Like `run_macro_experiment`, but instead of returning one row per state” it returns rows for every node and simulated successor. This is direct evidence that a one-row-per-macro-state experiment preceded the node-level dataset.

The original macro dataframe's complete column list and sample size are **not recoverable from the available repository history**.

---

## 4. Initial Statistical Models

### 4.1 Ordinary least squares / multiple regression

The surviving `fit_linear_regression(...)` helper implements OLS through `numpy.linalg.lstsq`. It:

- accepts an arbitrary list of predictors;
- optionally adds an intercept;
- drops incomplete rows;
- returns coefficients, fitted values, residuals, sample size, and conventional \(R^2\).

The recovered chat history reports the following early result:

| Quantity | Recovered value | Evidence qualification |
|---|---:|---|
| Troop-ratio coefficient | approximately `0.344` | Recovered chat summary |
| Territory-ratio coefficient | approximately `0.823` | Recovered chat summary |
| Intercept | approximately `0.0026` | Recovered chat summary |
| \(R^2\) | approximately `0.952` | Recovered chat summary |
| Response variable | Not recoverable with confidence | The summary warns that the historical response was not identified |

This appears to have been a two-predictor multiple linear regression, not multiple logistic regression. Logistic/binomial modelling entered as a later step when the response was explicitly treated as a bounded proportion.

The result was useful even though it was not sufficient. It established that territory and troop ratios captured a large part of broad outcome variation. It also created a baseline against which nonlinear models could be judged.

### 4.2 Interaction models

The recovered history states that the next models added a territory-by-troop interaction. The strategic logic is strong: the marginal value of additional troops need not be constant across territory-control levels. A concentrated army controlling few entry points and the same army spread across a large frontier may produce different outcomes.

The exact interaction formula, coefficients, fit statistics, and response variable are **not recoverable from the available repository history**. No surviving source file contains the historical fitted interaction model.

### 4.3 Binomial and quasi-binomial GLM

The archived `fit_binomial_glm(...)` is a much stronger direct artefact. It models a proportion outcome with a known denominator using `statsmodels`:

\[
\operatorname{logit}(p_i)
= \beta_0 + \sum_j \beta_j x_{ij}.
\]

The implementation:

- accepts a proportion response `y_col`;
- accepts a `denominator_col`, for example `total_battle_territories`;
- clips exact 0 and 1 outcomes slightly to avoid logit singularities;
- uses a binomial family with a logit link;
- passes the denominator as `var_weights`;
- optionally fits `scale="X2"` as a quasi-binomial-style overdispersion correction;
- returns response-scale residuals;
- calculates an SSE-based \(R^2\)-like diagnostic, explicitly not presented as a standard GLM goodness-of-fit statistic.

The docstring uses `expected_territories_ratio` as its example response. This is evidence of the intended target class, not proof that every historical GLM used precisely that column.

No preserved GLM coefficient table, deviance, dispersion estimate, AIC, or sample size was found.

### 4.4 Summary of models recoverable from the period

| Model | Target | Predictors/form | Recoverable performance | Status of evidence |
|---|---|---|---|---|
| Manual success factor | Conquer/remain/eradicated probabilities | Continent troop share, attacker flag, fixed bonus; several allocation rules | No calibrated historical metric | Surviving archived code; exact date uncertain |
| OLS / multiple regression | Aggregate future outcome; exact response uncertain | Territory ratio and troop ratio | \(R^2\approx0.952\); coefficients above | Recovered dated chat summary plus OLS helper |
| Interaction regression | Aggregate future outcome | Territory ratio, troop ratio, interaction | Not recoverable | Recovered chat only |
| Polynomial/quadratic regression | Aggregate/proportion outcome | Squared and interaction terms | Residual curvature remained | Recovered chat only |
| Binomial/quasi-binomial GLM | Territory-related proportion | Logit-linear predictors with explicit territory denominator | Exact metrics not recoverable | Surviving archived code plus recovered chat |
| GAM / spline model | Territory-related probability or expected ratio | Smooth territory and troop effects plus tensor interaction | Residuals reportedly flatter than GLM; no numeric metric retained | Recovered dated chat summary; original source removed |

---

## 5. Residual Diagnostics and Nonlinearity

### 5.1 Why \(R^2\approx0.952\) did not settle the problem

A high \(R^2\) answers a limited question: how much variation in the observed response is reproduced by the fitted linear surface on that dataset. It does not show that:

- the conditional mean is linear;
- residuals are structureless;
- variance is constant;
- predictions behave correctly near probability bounds;
- the selected features are a sufficient state representation;
- or the model can generate a legal successor board.

The recovered history says the early regression's residuals displayed systematic structure. This was the trigger for additional terms rather than a reason to declare the macro model complete.

### 5.2 Curvature

By 2025-12-02, quadratic terms still left a U- or arc-shaped residual pattern. That is direct recovered-chat evidence of unresolved nonlinearity. A single polynomial degree may bend the surface globally but cannot easily represent different local slopes, saturation, thresholds, or asymmetric effects near the 0/1 boundaries.

### 5.3 Interaction structure

The move toward territory-by-troop interactions indicates that the effect of force balance varied with map control. Recovered diagnostics included troop-effect curves conditional on fixed territory ratios, a direct way to inspect whether one predictor's effect changes over the other predictor's range.

### 5.4 Discrete banding

Territory ratios are discrete fractions whose denominators depend on graph size: for a graph with \(n\) relevant territories, observed ratios lie on multiples of \(1/n\). This can produce visible residual bands even when the conditional mean is modelled flexibly. The recovered conversations explicitly associated some remaining GAM bands with discrete denominators.

### 5.5 Monte Carlo noise

Simulation-derived outcomes add finite-sample variance. If each macro state is evaluated with a limited number of rollouts, the estimated proportion or mean fluctuates around its true value. The recovered history cites Monte Carlo noise as another contributor to residual texture after the GAM fit.

The historical simulation count and a formal decomposition of residual variance into model error versus Monte Carlo error were not recovered.

### 5.6 Heteroscedasticity

No surviving source or recovered summary explicitly reports a heteroscedasticity test or a conclusion about heteroscedasticity. The move to binomial/quasi-binomial modelling is compatible with recognizing non-constant variance in bounded proportions, but it should not be described as proof that a particular heteroscedasticity diagnostic was run.

---

## 6. Polynomial, Interaction, and Transformed Models

### 6.1 Polynomial and interaction attempts

The recovered history places interaction and quadratic/polynomial modelling between the initial OLS fit and the 2025-12-02 GAM pivot. Their purpose was to capture:

- curvature in territory advantage;
- curvature or saturation in troop advantage;
- a territory × troop interaction;
- different marginal troop effects under different control levels.

They improved flexibility but did not eliminate structured residuals. The exact polynomial degrees, selected terms, coefficients, and fit improvements are **not recoverable from the available repository history**.

### 6.2 Surviving transformations

`old_functions.py` preserves an `add_transformations(...)` function with the following engineered columns:

| Transformation | Historical column |
|---|---|
| Log | `log_realized_troops_ratio` |
| Log | `log_expected_new_territories_ratio` |
| Log | `log_troops_cv` |
| Square root | `sqrt_realized_territory_ratio` |
| Square root | `sqrt_expected_new_territories_ratio` |
| Square root | `sqrt_troops_gini` |
| Logit | `logit_realized_territory_ratio` |

Small epsilon values were added before logs, and bounded variables were clipped before square-root or logit transformations. These transformations demonstrate that the modelling effort went beyond adding one quadratic term: it explored link-scale and variance-stabilizing representations tailored to ratios and concentration measures.

### 6.3 Remaining problem

Transformations and finite polynomial bases change functional form, but they do not solve either of the phase's two deeper issues:

1. the response surface may have local nonlinear structure that a fixed basis captures awkwardly;
2. the macro feature vector may not contain enough information to identify the actual successor state.

The first issue motivated splines. The second eventually motivated node/state-level labels.

---

## 7. GAM and Spline Modelling

### 7.1 Dated pivot

The recovered conversations identify 2025-12-02 as a clear pivot from GLM/polynomial models to generalized additive models. The reason was empirical: residual curvature remained after simpler nonlinear terms were added.

### 7.2 Recovered implementation

The historical function was reportedly named `fit_binomial_gam` and used `pyGAM`'s `LogisticGAM`. Its conceptual form was:

\[
\operatorname{logit}(p)
= f_1(\text{territory ratio})
+ f_2(\text{troops ratio})
+ f_{12}(\text{territory ratio},\text{troops ratio}),
\]

where the main effects were spline smooths and the joint effect could be represented by a tensor term such as `te(0, 1)`.

The original file is absent, so the exact code expression, number of splines, spline order, smoothing penalty, optimizer settings, grid search, and convergence statistics are **not recoverable from the available repository history**.

### 7.3 Why splines were attractive

Splines matched the structure of the problem better than a single global polynomial:

- strategic advantage can saturate near complete troop or territory dominance;
- effects near balance may be steeper than effects at the extremes;
- discrete graph constraints can create local bends;
- the interaction surface may change shape across different control regimes;
- smoothness regularization can limit overfitting while allowing local curvature.

### 7.4 Recovered diagnostics

The history search reports that the GAM work included:

- two-dimensional heatmaps;
- response surfaces;
- curves showing the troop effect at fixed territory ratios;
- residual plots;
- direct residual comparison with GLM results.

GAM residuals were described as materially flatter than GLM residuals. Remaining bands were interpreted as a combination of discrete denominator effects and Monte Carlo noise.

No historical plot files survive in the repository. No exact residual variance, explained deviance, AIC, accuracy, calibration score, or held-out metric can be reported.

### 7.5 What the GAM solved—and what it did not

The GAM addressed functional-form misspecification. It could learn a smooth nonlinear conditional mean without forcing the analyst to choose one polynomial degree globally.

It did **not** solve state reconstruction. Its output remained a macro probability or expected aggregate. Even a perfectly estimated macro response surface would not identify which territories changed owner or where troops remained. GAM was therefore a useful diagnostic and predictive advance within the macro formulation, but not a complete transition model for recursive strategy simulation.

---

## 8. Expansion of Macro Feature Engineering

The feature inventory below separates variables that survive in code from variables mentioned only in recovered conversations and from later node-level additions.

### 8.1 Force balance

| Feature | Meaning | Surviving status |
|---|---|---|
| `target_territory_ratio` | Requested attacker share when generating a controlled state | Present in experiment/generator interfaces |
| `target_troops_ratio` | Requested attacker available-troop strength relative to defender troops in the generator | Present; clamped to feasible range |
| `battle_realized_attacker_territory_ratio` | Attacker-owned nodes divided by nodes in the battle graph | Computed and later used as ML feature |
| `battle_realized_attacker_available_troops_ratio` | Attacker's mobile troops, based on troops above the one-troop floor, divided by initial attacker troops in battle | Computed and later used |
| `full_realized_attacker_territory_ratio` | Attacker territories divided by all territories in the full graph | Computed and later used |
| `full_realized_attacker_troops_ratio` | Attacker troops divided by all troops in the full graph | Computed and later used |
| `full_realized_attacker_available_troops_ratio` | Battle-available attacker troops divided by all attacker troops in the full graph | Computed and later used |

The coexistence of target and realized values is a methodological contribution of the macro phase: controlled experiment design was retained without pretending discrete generated states exactly matched continuous targets.

### 8.2 Counts and scale variables

The later feature pipeline preserves:

- attacker and defender territory counts;
- total territory count;
- attacker and defender troop counts;
- total troop count;
- battle attacker territory and troop counts;
- battle total territories and troops.

These variables distinguish, for example, a 0.5 territory share on a four-node graph from 0.5 on a twelve-node graph. Equal ratios with different denominators can have different uncertainty, discretization, and tactical meaning.

### 8.3 Distributional statistics

| Feature | Strategic intuition | Evidence qualification |
|---|---|---|
| Troop CV | Relative dispersion of attacker troop stacks; distinguishes even deployment from uneven deployment | Computed for battle and full graph; later used in ML |
| Troop Gini | Concentration/inequality of attacker troop placement | Computed for battle and full graph; later used in ML |
| Troop skewness | Whether deployment has a long tail of unusually large stacks | Helper survives, but the preserved successor code does not add skewness to the final feature dictionary or training list |
| Mean/variance | Means and variances appear for graph degree and reserve-distance summaries; a separate troop-variance feature is not preserved | Direct code evidence only where named |

Skewness is therefore a historically explored or prepared feature, not a confirmed input to the January node-level models.

### 8.4 Spatial and topological structure

The full graph is constructed as a continent plus neighbouring territories, retaining the static local topology around the active strategic area. Surviving metrics include:

- edge count;
- mean degree;
- degree variance;
- graph diameter;
- number of connected components.

The recovered 2025 conversations also mention full-graph and battle-graph topology metrics, clustering, and frontier-related quantities. The current code does not preserve an explicit clustering-coefficient feature or a historical battle-graph degree-feature set, so their exact definitions and whether they entered the fitted GAM are **not recoverable from the available repository history**.

### 8.5 Deployment and reserves

`compute_effectiveness_metrics(...)` preserves a particularly important conceptual extension beyond raw force totals:

- `battle_realized_effectiveness_node_ratio`: fraction of full-graph nodes that are active battle nodes;
- `battle_realized_effectiveness_attacker_troops_ratio`: fraction of all attacker troops in the full graph that are actually located in the battle graph;
- reserve distance mean, minimum, maximum, and CV: shortest-path distance from nodes outside the battle graph to the active battle region.

These features encode the difference between nominal strength and usable strength. Two players may have the same total troops, but the one whose forces are several edges from the front has less immediate influence on the current battle.

The January node-level dataset contains these effectiveness columns. The preserved `train_ML.py` feature-selection list does not include them in the Random Forest feature matrix, so they should be described as computed dataset descriptors, not confirmed trained inputs for that saved model generation.

### 8.6 Frontier and local pressure

The node-level successor later adds:

- full-graph node degree;
- enemy and friendly neighbour counts;
- enemy and friendly neighbouring troop sums;
- maximum enemy and friendly neighbouring stacks;
- `is_frontier_node`;
- a bounded log-difference pressure signal;
- a bounded log-difference local balance signal.

These are not purely macro features. They represent the transition from one shared state vector to a hybrid design in which global descriptors are repeated for every node and combined with node-specific tactical context.

### 8.7 Strategic and partition descriptors

Later dataset rows contain `partition_source`, `partition_is_full`, and coverage diagnostics. These belong to the later regional strategy-solving pipeline and should not be projected back into the original November macro model. They demonstrate that the dataframe infrastructure remained extensible as the architecture changed.

### 8.8 Features not verified

No surviving evidence supports claiming that a separately named reinforcement-concentration feature was fitted during November–December. Clustering is mentioned in recovered conversations, but its exact implementation is missing. Any public summary should use “topology and deployment-concentration features” rather than list unverifiable historical formulas.

---

## 9. Core Failure Mode of the Macro Approach

### 9.1 Aggregate fit is not a transition model

The macro model estimated something like:

\[
E[Y_{t+1}\mid M(S_t)],
\]

where \(S_t\) is the full microstate and \(M(S_t)\) is its macro summary. Recursive simulation needs a draw from, or at least a usable representation of,

\[
P(S_{t+1}\mid S_t, \text{policy}),
\]

not merely an expected aggregate \(Y_{t+1}\).

### 9.2 A concrete information-loss example

Consider two states with the same:

- attacker territory ratio `0.5`;
- attacker troop ratio `0.6`;
- troop Gini;
- full-graph node and edge counts.

In State A, attacker territories form one connected block, the strongest stack borders the weakest defender node, and reserves are one edge behind the front. In State B, attacker territories are split into two components, the strongest stack is isolated from the relevant front, and multiple exposed border nodes must each retain troops.

The macro vector may be identical or very similar. The legal actions, capture sequence, stopping points, and next-turn reinforcement opportunities are not. A predicted future territory ratio cannot determine which concrete territories changed owner.

### 9.3 Why this breaks recursive simulation

To simulate another turn, the program must know:

- which player owns each node;
- how many troops remain on each node;
- which nodes are adjacent across opposing ownership;
- which attacks are legal;
- where reserves can move;
- what the next policy's feature values are.

An expected macro outcome cannot be reliably “decoded” into that information. Many microstates map to the same macrostate, and those microstates have different successor distributions. The compression is many-to-one, so no more flexible regression function can reverse it without additional state information.

### 9.4 Why GAM could not fix the failure

GAM improved the map from macro inputs to macro outputs. It did not change the inputs or outputs. This is a representation problem, not merely a bias-variance or functional-form problem.

That distinction is the major methodological result of the phase:

> A good predictor of broad advantage is not automatically a generative model of legal future game states.

---

## 10. Transition Toward Node-Level / State-Level Learning

### 10.1 The 2025-12-08 transition

The recovered history identifies 2025-12-08 as the point at which future Monte Carlo-generated `GlobalState` objects were proposed as labels rather than being collapsed immediately to one global expectation.

The new supervised questions were approximately:

\[
P(\text{attacker holds node } i \text{ after the turn}\mid S_t)
\]

and

\[
E[\text{troops at node } i \mid \text{final holder}, S_t].
\]

### 10.2 Surviving label definitions

The near-term successor pipeline materializes, for every simulated final `GlobalState` and every node:

- `initial_owner`;
- `initial_troops`;
- `final_owner`;
- `final_troops`;
- `attacker_holds_final`;
- `captured`, defined as attacker-held finally but not already attacker-held initially;
- `is_battle_node`;
- the shared macro descriptors of the initial state;
- later local and regional diagnostics.

This is direct repository evidence that the conceptual macro-to-micro transition became executable.

### 10.3 Hybrid global and local features

The macro features were not discarded. `train_ML.py` groups them into:

- `macro_core` ratios;
- `macro_distribution` CV and Gini measures;
- `macro_topology` degree, components, diameter, and edge count;
- `macro_counts` for territories and troops.

Those shared state features are combined with node-specific inputs such as initial troops, owner, battle membership, neighbouring forces, frontier status, and local pressure. This architecture directly reflects the lesson of the macro phase: broad strategic context is useful, but it must be attached to a representation that preserves local identity.

### 10.4 Why this was a natural next step

Simulation already generated concrete future states. Converting them into supervised node rows required less information loss than collapsing them into a single ratio. It also made the model's output actionable: a set of node predictions could be assembled into an approximate next state and fed into another turn.

This later node-wise design still had an important limitation: independently predicted node marginals do not guarantee a jointly legal or solver-observed state. That problem motivated the still later distribution-over-whole-successor-state architecture in 2026. It does not invalidate the December transition; it shows the same preservation principle being applied more strictly.

---

## 11. Relationship to Later Machine Learning

### 11.1 Infrastructure retained from the macro phase

The macro work contributed several reusable components:

1. **Controlled state generation.** Target ratio grids and repeated samples became the coverage design for supervised datasets.
2. **Target-versus-realized accounting.** Generated states were described by what actually occurred, not only what was requested.
3. **Feature computation.** Ratios, concentration statistics, topology, active-force measures, and reserve distances became candidate ML predictors.
4. **Dataframe pipelines.** The one-row-per-state analysis pattern was extended to one-row-per-node-per-successor.
5. **Simulation-derived labels.** Monte Carlo outcomes supplied supervision where closed-form full-state labels were unavailable.
6. **Diagnostics.** Residual plots and response surfaces evolved into hold-out metrics, sanity checks, class-balance checks, and later distributional validation.
7. **Representation awareness.** The failure of macro decoding made state identity an explicit design requirement.

### 11.2 Near-term January 2026 corroboration

The following artefacts fall just outside the requested period. They are included because they verify that the December transition was implemented, not because they are December results.

On 2026-01-08:

- `generate_data_ML_(sequential).py` was saved with the description that node transition experiments were “Like `run_macro_experiment`, but instead of returning one row per state” returned node rows;
- the archived `old_functions.py` retained OLS, transformations, and quasi-binomial GLM fragments while the active statistical subsystem was being removed;
- a successor `train_ML.py` described itself as the “NEW per-node macro→micro approach.”

On 2026-01-09, saved datasets and model bundles were produced for all six continents. The training code used Random Forest classifiers/regressors, a fixed 20% test split, 200 trees, and random state 42. These settings belong to the successor implementation, not the original macro regressions.

| Continent | Node rows | Capture ROC-AUC | Attacker-held troop RMSE | Defender-held troop RMSE |
|---|---:|---:|---:|---:|
| North America | 228,600 | 0.992 | 0.547 | 0.396 |
| Africa | 93,000 | 0.993 | 0.573 | 0.408 |
| Asia | 315,000 | 0.995 | 0.536 | 0.330 |
| South America | 7,800 | 0.988 | 0.913 | 0.400 |
| Europe | 198,900 | 0.994 | 0.586 | 0.383 |
| Australia | 7,750 | 0.985 | 1.007 | 0.518 |

The output file `datasets/summaries.txt` records 60 dataframe columns and the displayed metrics. These values confirm that the macro features entered a functioning node-level training pipeline. They do **not** prove that the learned model generalized to unseen graph regimes, was calibrated, produced legal joint states, or solved multi-turn strategy optimization.

### 11.3 What was later removed

The recovered December history says the old GAM/statsmodels/plotting subsystem was explicitly judged removable from production after the node-level transition. The current repository supports this: no `macro_state_experiments.py`, `fit_binomial_gam`, `LogisticGAM`, or historical GAM plots remain, while incomplete OLS/GLM fragments survive in `old_functions.py`.

The statistical work was removed as a production dependency, not erased from the project's intellectual lineage.

---

## 12. Approaches Tested and Rejected or Superseded

| Approach | Why it was tried | What worked | Why it was insufficient | What was retained |
|---|---|---|---|---|
| Manual success factors | Create interpretable global conquest probabilities before enough training data existed | Encoded troop strength, attacker advantage, and continent value in transparent formulas | Hand-set relationships were not empirically sufficient and compressed tactical structure | Strategic feature intuition; explicit utility/outcome decomposition |
| OLS / multiple regression | Establish whether troop and territory ratios explain broad outcomes | Very high recovered \(R^2\approx0.952\); simple interpretable baseline | Structured residuals; bounded/nonlinear response; no successor microstate | Baseline, coefficient interpretation, residual workflow |
| Interaction regression | Let troop effects vary with territory control | Captured some joint structure | Exact improvement unavailable; residual patterns remained | Interaction thinking later reappeared in GAM tensor term and hybrid features |
| Polynomial/quadratic regression | Model curvature without leaving parametric regression | Increased flexibility | U-/arc-shaped residual structure remained; global polynomials awkward near boundaries | Evidence that nonlinear effects mattered |
| Log/sqrt/logit transformations | Match scale to ratios, concentration, and skewed variables | Surviving code shows systematic feature transformation | Transformations alone could not eliminate missing structure or information loss | Feature-engineering discipline |
| Binomial/quasi-binomial GLM | Respect probability bounds, denominators, and overdispersion | Correctly framed proportion outcomes with logit and trial weights | Linear predictor remained too rigid; still predicted aggregates | Denominator-aware modelling and response-scale diagnostics |
| GAM with splines | Learn smooth local nonlinearities and interactions | Flatter residuals than GLM; useful response surfaces | No legal next-state reconstruction; some residual bands/noise remained | Nonlinear effect understanding; visualization and diagnostic methods |
| Macro-state transition model | Make long-run simulation tractable through low-dimensional states | Useful broad predictors and reusable macro/topology features | Many-to-one compression destroyed node identity, borders, troop placement, and future action space | State generator, dataframe pipeline, macro features, Monte Carlo supervision |

“Superseded” is more accurate than “failed” for most rows. Each method answered a narrower question and exposed the next modelling requirement.

---

## 13. Chronological Timeline

| Date or period | Development | Evidence strength |
|---|---|---|
| 2025-11-03 | Full-board `After_Setup.png` and `After_Game.png` were generated, confirming operational state setup, simulation, and rendering before the macro regression phase | Direct repository timestamp and visual inspection |
| 2025-11-27 | Earliest clearly recovered Risk macro discussion: “macro-state transition model,” controlled territory/troop ratios, `macro_state_experiments.py`, pandas data, and regression diagnostics | Recovered dated chat summary; original module missing |
| Late Nov 2025 | Initial two-variable regression produced approximately \(R^2=0.952\) with coefficients around 0.344, 0.823, and intercept 0.0026 | Recovered chat summary; response variable uncertain |
| Late Nov–early Dec 2025 | Residual structure motivated territory×troop interactions, quadratic/polynomial terms, and transformed predictors | Recovered chat; transformations survive in archived code |
| 2025-12-02 | Quadratic terms still left a U-/arc-shaped residual pattern; pivot to `pyGAM`/`LogisticGAM`, spline main effects, and tensor interaction | Recovered dated chat summary |
| Early Dec 2025 | Heatmaps, response surfaces, conditional troop-effect curves, and GLM-vs-GAM residual comparisons | Recovered chat; plot files missing |
| Approximately 2025-12-05 to 2025-12-08 | Macro feature set expanded toward topology, CV, Gini, skewness, reserves, deployment effectiveness, and structural/frontier measures | Recovered chat; many definitions survive in successor code |
| 2025-12-08 | Explicit transition toward Monte Carlo future `GlobalState` labels and node-level targets: hold/capture probability and conditional troop count | Recovered dated chat summary; strongly corroborated by successor code |
| Later Dec 2025, exact date unavailable | Discussion concluded that old GAM/statsmodels/plotting subsystem could be removed after transition to node-level ML | Recovered chat summary |
| 2026-01-08, corroboration | OLS/GLM fragments archived in `old_functions.py`; sequential node-data code explicitly references predecessor `run_macro_experiment` | Direct repository timestamp and code |
| 2026-01-09, corroboration | Six continent-specific node datasets and Random Forest model bundles written; metrics saved in `datasets/summaries.txt` | Direct repository output |

No exact dates should be assigned to the individual OLS, interaction, polynomial, and GLM commits beyond the ordering above.

---

## 14. Numerical and Experimental Milestones

### 14.1 Numbers attributable to the November–December macro phase

| Number | Context | Confidence |
|---|---|---|
| `2025-11-27` | Earliest clearly recovered macro-state transition discussion | High within recovered chat history |
| `2025-12-02` | GAM/spline pivot after residual curvature | High within recovered chat history |
| `2025-12-08` | Node/`GlobalState` label transition | High within recovered chat history |
| \(R^2\approx0.952\) | Early two-ratio linear regression | Recovered, but response variable uncertain |
| `0.344` | Approximate troop-ratio coefficient | Recovered, response uncertain |
| `0.823` | Approximate territory-ratio coefficient | Recovered, response uncertain |
| `0.0026` | Approximate intercept | Recovered, response uncertain |
| `1.2` | Default attacker bonus in surviving manual success-factor code | Direct code; exact introduction date uncertain |

### 14.2 Numbers not recoverable for the macro phase

The following are not available and should not be invented:

- macro dataframe sample size;
- ratio-grid dimensions;
- simulations per macro state;
- GLM coefficients, dispersion, deviance, or AIC;
- polynomial-model \(R^2\) or residual reduction;
- GAM spline count, smoothing penalty, explained deviance, or held-out performance;
- numeric GLM-to-GAM residual improvement;
- runtimes for the November–December experiments.

### 14.3 Post-period corroborating numbers

The six January node-level row counts and Random Forest metrics are listed in Section 11. They verify the successor pipeline, not the quality of the original macro models. The January configuration visible in code used an 11-by-11 target-ratio grid and five generated states per combination, but that configuration was saved after the target had changed to node-level prediction. It must not be reported as the November macro experiment design.

---

## 15. Lessons Learned

### 15.1 High fit is not sufficient-state evidence

The early \(R^2\) showed that two ratios described broad outcome variation. It did not show that those ratios formed a Markov-sufficient state for continued simulation.

### 15.2 Residual analysis drove method development

The project did not stop at a headline fit statistic. Structured residuals triggered interactions, nonlinear terms, denominator-aware GLMs, and spline models. This is a substantive modelling strength.

### 15.3 Correct response geometry matters

Territory shares are bounded and have discrete denominators. Binomial/quasi-binomial models were better aligned with that geometry than unconstrained OLS, even though they did not solve the full problem.

### 15.4 Flexible functions cannot recover discarded information

GAM can approximate a nonlinear macro response surface. It cannot infer a unique border configuration or troop placement from a many-to-one summary.

### 15.5 Graph topology and deployment location matter

Troop totals do not measure immediate strategic availability. Degree structure, components, frontier exposure, the fraction of troops in the active battle, and reserve distances add strategically meaningful information.

### 15.6 Aggregate features can remain useful predictors

Abandoning macro output prediction did not require abandoning macro inputs. Global balance and topology features became context for local node predictions.

### 15.7 Simulation can create supervision

When analytical full-state labels are unavailable, controlled simulation can generate training examples. The later pipeline operationalized this by retaining every sampled future `GlobalState` and converting it to node outcomes.

### 15.8 The output representation must match the downstream task

If the next component must simulate another turn, the model must output or sample a state representation compatible with legal game dynamics. A scalar expected value is inadequate regardless of its predictive fit.

---

## 16. Bridge to the Later Project

The historical bridge should be described in stages:

```text
manual macro success factors
    ↓
controlled macro-state experiments
    ↓
OLS, interactions, transformations, binomial GLM
    ↓
GAM and spline response surfaces
    ↓
richer force, concentration, topology, and reserve features
    ↓
recognition that aggregate outcomes cannot reconstruct the next state
    ↓
simulation-generated GlobalState labels
    ↓
hybrid macro + local node-level supervised learning
    ↓
later whole-successor-state distribution modelling
    ↓
later exact and regional strategy-solving work
```

The link to later exact/regional modelling is meaningful but indirect. The November–December work did not yet implement the later exact finite Bellman-style solver, canonical policy libraries, or validated regional composition architecture. It contributed instead:

- a graph-aware vocabulary for state description;
- controlled state generators;
- simulation and dataframe infrastructure;
- an insistence on residual and out-of-sample diagnostics;
- an understanding that the next decision stage needs concrete state information;
- a reason to retain distributions or state-level labels rather than only scalar expectations.

Those lessons later reappeared when independent regional approximations were found to lose cross-region sequence information. The specific mathematics differed, but the modelling principle was the same: compression is safe only when it preserves what downstream decisions depend on.

### Does the evidence support treating this phase as a meaningful predecessor?

**Yes, with a precise qualification.**

The evidence strongly supports treating the November–December 2025 macro/statistical phase as a direct predecessor to the later machine-learning work. The dated history identifies the macro experiment, GAM phase, feature expansion, and node-label pivot; surviving code retains the OLS/GLM fragments, feature calculators, controlled ratio generators, and explicit “macro→micro” successor design; January datasets demonstrate that those features and labels entered trained node models.

The evidence also supports treating it as a conceptual predecessor to the later exact/regional work, but not as an implementation ancestor of the exact finite solver. Its contribution was the identification of representation loss and the construction of graph/topology features and simulation data infrastructure. The later exact/regional architecture was a separate methodological development that addressed global policy and distribution fidelity more directly.

---

## 17. Portfolio Interpretation

### Recommended public narrative

> Early Project Risk experiments used controlled simulation, multiple regression, binomial GLMs, and spline-based GAMs to identify macro-level strategic predictors such as force balance, troop concentration, topology, and reserve accessibility. These models captured broad outcome structure—one early linear fit reached approximately \(R^2=0.95\)—but residual analysis and recursive simulation exposed a deeper limitation: aggregate forecasts could not reconstruct the legal node-level successor state. The resulting feature engineering and simulation pipeline were therefore repurposed for supervised node/state-level prediction and informed the project's later distributional and exact modelling work.

### Short CV/GitHub bullet

> Developed a simulation-based statistical modelling pipeline for Risk, progressing from interpretable macro regressions and spline GAMs to graph-aware node/state-level learning after identifying information loss in aggregate transition forecasts.

### Claims to avoid

Do not claim that:

- the \(R^2\approx0.952\) model's exact response variable is known;
- the GAM had a specific accuracy, deviance, spline count, or smoothing parameter;
- the macro model generated legal successor boards;
- the January Random Forest metrics validate multi-turn strategy quality;
- the November–December phase already contained the later exact finite or regional decomposition architecture;
- every feature found in successor code was definitely fitted in the December GAM.

The strongest portfolio story is not “many models failed.” It is:

> The project used statistical diagnostics to distinguish functional-form error from state-representation error, then changed the prediction target when added flexibility could no longer solve the downstream problem.

---

## Appendix A. Sources Inspected

### Historical conversation evidence

- ChatGPT conversation `CV tips för statistikexamen`, especially the recovered-history response following the user's request to find the 2025 macro/spline/regression phase.
- The response reports historical Risk conversations dated 2025-11-27, 2025-12-02, 2025-12-05 through 2025-12-08, and 2025-12-08.
- The underlying individual 2025 conversations were not directly retrievable in this pass; the later recovered summary is therefore treated as a secondary historical source.

### Historical or archived scripts

- `old_functions.py` — incomplete archive containing manual probability logic, OLS, transformations, and quasi-binomial GLM.
- `Gammalt/GameAnalysis.py` — continent troop/territory counts and shares.
- `Gammalt/UtilityFunction.py` — manual success factors, attacker bonus, outcome allocation, and continent utility.
- `Gammalt/NashEquilibria.py` — later/adjacent use of the manual probability and utility layer in strategy-profile enumeration.
- `SimulationEngine.py` — repeated-game Monte Carlo interface.
- `SimulationFunctions.py` — full-board simulation and rendering support.
- `After_Setup.png`, `After_Game.png` — 2025-11-03 state-rendering artefacts.

### Successor scripts used to verify continuity

- `generate_data_ML_(sequential).py` — 2026-01-08 near-term snapshot retaining macro feature computation and explicitly describing node rows as the successor to `run_macro_experiment`.
- `state_generators.py` — target and realized ratio generation, full-graph state generation, and macro constraints.
- `train_ML.py` and `tillfällig_kopia/train_ML.py` — hybrid macro/local feature selection and node-level Random Forest targets.
- `predict_future_states_ML.py` — rebuilds the same macro features at inference and combines them with local node features.
- `generate_data_ML.py` — evolved pipeline retaining the historical macro feature families.
- `datasets/summaries.txt` — 2026-01-09 node dataset columns and saved model metrics.
- `datasets/node_transition_results__*.pkl` and `models/node_level_models__*.joblib` — dated successor artefacts; inspected by metadata and accompanying saved summaries, not treated as December results.

### Missing historical sources

- `macro_state_experiments.py` — named in recovered conversations but absent from the current repository.
- Original `fit_binomial_gam` implementation.
- Historical GAM/GLM plots, heatmaps, response surfaces, residual figures, and model summaries.
- Original macro-state dataframe and its experiment metadata.

---

## Appendix B. Important Uncertainties

1. The exact response variable for the \(R^2\approx0.952\) regression is not recoverable.
2. The exact macro experiment grid, sample size, and simulations per state are not recoverable.
3. Exact dates for OLS, interaction, polynomial, and GLM iterations are not recoverable beyond their approximate order.
4. GAM hyperparameters, smoothing selection, coefficients, and numeric performance are not recoverable.
5. Clustering and early frontier metrics are mentioned in recovered chats, but their original formulas are missing.
6. Skewness was implemented as a helper, but use in the historical fitted models is not proven.
7. Effectiveness and reserve-distance columns were generated in the successor dataset, but the preserved January training feature list did not use them.
8. The manual success-factor variants survive in files archived on 2026-01-08; their precise November–December version chronology is unknown.
9. The date on which the GAM/statsmodels/plotting subsystem was actually deleted is not recoverable; only the late-December cleanup decision and January archive fragments remain.
10. The relationship to the later exact/regional architecture is conceptual and infrastructural, not a direct code lineage from GAM to exact finite solving.

