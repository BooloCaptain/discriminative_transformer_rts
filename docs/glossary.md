# Glossary

A catalogue of the nomenclature used in this project — in the code, in the artifacts, and in
the documentation. It is a *reference*, not a narrative: the intent is that a term met
anywhere (a table header, a population name, an artifact key, a doc paragraph) can be looked up
here and resolved to one meaning.

Organisation is by the layer a term belongs to, because a word's meaning here is almost always
"which layer is speaking". Several words are used in more than one layer on purpose and a few
are genuinely overloaded; the last section lists those explicitly.

| Where a term is defined | Authoritative record |
|---|---|
| the study's question, scope, protocol | `docs/plan.md` |
| the dataset half (contract, accessors, features) | `docs/refactor.md` |
| the experiment layer (roles, axes, cells) | `docs/experiment.md` |
| the measured numbers | `docs/implementation.md` |
| status and next steps | `docs/handoff.md` |
| the execution plan for the two gaps | `docs/plan_next_steps.md` |
| the code | the module named beside each entry |

---

## 0. Standard terminology: project term → standard term

> **Adopted 2026-10-02, and applied throughout the project.** Every local coinage below has
> been migrated: the *code identifiers*, the *recorded artifact keys and names*, the *file and
> cache names*, and the *prose*. A recorded artifact is the specification only in the sense
> that it pins down what a change must not move by accident; a deliberate vocabulary change is
> exactly the case where the artifacts must move with it, so they were regenerated under the
> new names rather than preserved.
>
> - **What was renamed:** domain identifiers (`Population` → `Subset`, `Selector` → `Ranker`,
>   `Axis`/`Element` → `Factor`/`Level`, `Knobs` → `Controls`, `Cell` → `DesignPoint`,
>   `Unmeasured` → `Undefined`, `Warning` → `Diagnostic`, `MATERIAL` → `INPUTS`), the feature
>   column and family names, the averaging-subset names, the artifact and cache file names, and
>   the recorded JSON keys.
> - **What was deliberately kept:** ordinary English ("runs", "arm" as a figure of speech,
>   "history" as the *data* property, "signal" as in "lexical signal", "covered" as an
>   adjective), numpy's `axis=` keyword, and `k` as the retrieval cut-off symbol.
> - **Not renamed, and why:** *killing test*, *fault*, *failure*, *mutant*, *killed*,
>   *survived*, *recall*, *precision*, *BM25* and *suite* were already the standard words. The
>   point of the exercise was to remove the collisions, not to churn the settled vocabulary.

### 0.1 What each term became

Two rules governed the mapping: a term is migrated only when it is (a) a local coinage or (b) a
word that *collides* with a standard term meaning something else; and the standard term is
preferred for new code while the old name survives wherever a recorded number depends on it.

The references the mapping rests on: IEEE 610.12 (software engineering terminology);
Avizienis et al. 2004 (**fault–error–failure cycle**: a fault may cause an error, which may
propagate to a failure at the system boundary); Rothermel & Harrold 1996 and Yoo & Harman 2010
(RTS: **minimisation, selection, prioritisation** — three distinct techniques); Elbaum et al.
2002 (prioritisation, **fault-revealing test**, APFD); Jia & Harman 2011 and Ammann & Offutt
(mutation: mutant, killed, **killing test**, mutation score, RIP model, **higher-order
mutant**); Manning et al. (IR: **recall@k**, **precision@k**, **cut-off k**); Montgomery and
Box–Hunter–Hunter (DOE: **factor**, **level**, **control variable**, **design point**,
**blocking**, **preregistration**).

### 0.2 Decisions on the proposed terms

| your suggestion | verdict |
|---|---|
| Test pool → **test suite** | **Agreed.** "Pool" is not used in the SE literature; "suite" is the standard term. |
| Killing test → "catching test" | **Keep "killing test".** It is the established term in mutation testing, and "catching test" is not attested anywhere. Add **fault-revealing test** as the alias for real-change (non-mutation) contexts. |
| Ran test → "observed test" | **Use "executed test".** "Observed" is a poor fit twice over: this project already uses *observed* vs *imposed* ordering, and in statistics "observed" contrasts with "expected". "Executed" is the SE standard. |
| Per-change budget → "per-change test budget" | **Use "selection budget"**, and split the two things it currently names: the **budget fraction** (b) and the **cut-off** (k). |
| k → b? | **No — keep k.** k is the standard symbol for a retrieval cut-off (`precision@k`, `recall@k`, "precision at cut-off k"). b would read as the budget fraction, which is the other parameter. Rename the *fraction* to b and leave k alone. |
| Suite reduction → better description | **Rename to "selection ratio"**, and say what it is: 1 − k/n, the fraction of the suite *not run for this change*. Critically, it collides with the SE term **test suite reduction** (TSR), which is a *different technique* — permanently removing redundant tests. |
| Fault-bearing "makes sense" | **Agreed**; add the aliases **fault-revealing change** / **detectable change**, and **killed mutant** for a mutation audience. |
| Structural funnel, recurrence, boundary, bridge rate, lexical bridge | **All agreed** — each is a local metaphor with a standard counterpart; see §0.1. |
| Candidate set/mode → ? | **"Candidate set" is standard** (the eligible tests). Rename the *mode* to **candidate policy**, since it selects between whole-suite and coverage-restricted selection. |
| Primitive → "primitive attribute"? | **"Primitive operation"** / **raw accessor** — an attribute would be data, and these are operations. Safer still: name the split **required operations vs derived attributes**. |
| Declaration → more descriptive | **Agreed**; say what it declares — **capability descriptor** / **metadata declaration**. |

### 0.1 Problem vocabulary (§1)

| project term | recommended | why | note |
|---|---|---|---|
| **Test pool** | **test suite** | Standard term for the whole set of tests. | Keep `test_pool` in code. |
| **Change** | **change** / **modification** | Already standard ("change impact analysis", "change-based RTS"). | — |
| **Killing test** | **killing test** (mutation) / **fault-revealing test** (general SE) | Both are established; they belong to different literatures. | `killing_tests(c)` stays. |
| **Ran test** | **executed test** | SE standard; unambiguous against "not executed". | Replaces "ran". |
| **Fault** | **fault** | IEEE 610.12 / Avizienis: the defect. | — |
| **Failure** | **failure** / **test failure** | The *event* at the boundary, not the change. | Say "test failure" when the outcome is meant. |
| **Fault-bearing** | **fault-revealing change** / **detectable change** (mutation: **killed mutant**) | Describes the property (the suite can reveal it), not just the presence of a fault. | — |
| **Per-change budget** | **selection budget**; **budget fraction** b; **cut-off** k | Standard in RTS evaluation and IR. | See the k-vs-b verdict above. |
| **Recall** | **recall** (alias **fault-detection rate**) | Standard; Rothermel & Harrold's *inclusiveness* is the same quantity. | — |
| **F-measure** | **F1 score** | Standard name for the balanced form. | — |
| **Suite reduction** | **selection ratio** (per-change) | Described above; avoids the TSR collision. | — |
| **Structural funnel** | **coverage-and-naming filter** (a static candidate filter) | SE calls this static/coverage-based test selection; "funnel" is a pipeline metaphor. | — |
| **Usual suspect** | **the test with prior (file,test) failure history** | The concept is failure history; there is no standard noun. | Keep "usual suspect" as informal prose only. |
| **Recurrence** | **pair co-occurrence frequency** | Maps onto SE **co-change** / **change coupling**. | — |
| **Boundary** | **execution boundary** (this work: **host–target boundary**) | The test process and the code under test are on different sides of it. | "Hardware-in-the-loop" only if hardware is in fact involved. |
| **Bridge rate** | **lexical-overlap rate** | The NLP/IR term for shared-token fraction. | — |
| **Lexical bridge** | **lexical-overlap signal** (SE: **traceability link**) | Its absence is a missing traceability link between change and test. | — |
| **Co-change proxy** | **surrogate label** / **proxy ground truth** | Standard ML term for a stand-in label. | — |
| **Recall@budget** | **recall@k** (k = cut-off) | IR standard. | — |

### 0.2 Dataset contract (§2)

| project term | recommended | why |
|---|---|---|
| **Primitive** | **required operation** / **raw accessor** | Operations, not attributes. The contract's split becomes *required operations* vs *derived attributes*. |
| **Declaration** | **capability descriptor** / **metadata declaration** | Says what is declared rather than "that something is". |
| **Derived feature / accessor** | **derived attribute** (ML: **engineered feature**) | Standard in both halves. |
| **Capability** | **capability** (optional supported material) | Already standard. |
| **Requirement** | **requirement** (alias **dependency**) | Already standard. |
| **Material** | **inputs** | Plainer. |
| **Ordering: observed / imposed** | **natural (temporal) order** / **synthetic order** | "Imposed" is vivid but non-standard; "synthetic" is the usual contrast to "natural". |
| **Test unit** | **test granularity** | Standard. |
| **Semantics** | **annotations** / **notes** | Avoids the overloaded "semantics". |
| **Unmeasured** | **undefined** / **not applicable** | Standard for a value that cannot exist; "unmeasured" reads as "not yet measured". |
| **Warning** | **diagnostic** | Standard compiler/linter vocabulary. |
| **Source / sample generator** | **data source** / **record parser (adapter)** | Standard. |
| **Derived / pooled dataset** | **view / wrapper dataset** / **aggregate dataset** | Standard composition names. |
| **Bundle** | **composite change** (mutation: **higher-order mutant**) | A multi-site change; in mutation terms a higher-order mutant. |
| **Distractor** | **distractor** (the object: **survived mutant**) | "Distractor" is standard ML; name the thing itself too. |
| **Signal mutant** | **target mutant** / **focal mutant** | Standard. |

### 0.3 Derived quantities (§3)

| project term | recommended |
|---|---|
| **labels** | **label matrix** |
| **runs** | **execution matrix** |
| **covered** | **coverage sets** |
| **fault_mask / fault_idx** | **detectable-change mask / index** |
| **candidates** | **candidate set** (per-change eligible tests) — already standard |
| **candidate mode** | **candidate policy** (whole-suite vs coverage-restricted) |
| **pair_counts** | **pair co-occurrence counts** |
| **sparse_mask** | **low-co-occurrence mask** |
| **change_id / coverage_key** | **change identity / coverage key** |

### 0.4 Populations (§4)

| project term | recommended | why |
|---|---|---|
| **Population** | **averaging subset** (alias **evaluation subset**) | "Population" is standard in statistics but overloaded here — it means the subset a metric is averaged over, not the whole dataset. |
| **rows** | **evaluation set** | — |
| **fault_bearing** | **detectable-change set** | — |
| **no_prior_failure / starved** | **cold-start subset** | "Cold start" is the standard ML term for exactly this regime. |
| **low_pair_recurrence / sparse** | **low-co-occurrence subset** | — |
| **starved141** | **cache-covered subset** | Named for provenance; say so. |

### 0.5 Splits and evaluation (§5)

| project term | recommended |
|---|---|
| **Split** | **train/test split** (temporal variant: **time-based split**) |
| **Evaluation window** | **held-out set** (alias **test fold**) |
| **Effective ordering** | **effective order** |
| **Probe budget** | **reporting budget** |
| **BudgetResult / Evaluation** | **metric sweep / evaluation record** |
| **n** | **sample size** |
| **Power limit** | **statistical power limit** — already standard |
| **±0.02** | **reproducibility floor** (alias **numerical noise floor**) |

### 0.6 Features (§6)

| project term | recommended | why |
|---|---|---|
| **FeatureBlock / FeatureGroup** | **feature set / feature group** | Standard ML. |
| **Family** | **feature group** | "Family" is local. |
| **Structured block** | **hand-crafted tabular feature set** | Standard ML contrast with learned representations. |
| **Traceability** (family) | **proximity & naming heuristics** | "Traceability" is a term of art for requirements→code links; this family is path/naming proximity. |
| **Intrinsic** (family) | **static size features** | — |
| **History features** | **temporal features** | Standard. |
| **covers_function** | **function coverage** | — |
| **filename_stem_match** | **filename match** | — |
| **path_distance** | **path proximity** | — |
| **change_size** | **code churn** (lines changed) | Standard SE term. |
| **test_failure_rate_cum** | **cumulative failure rate** | — |
| **test_last_failure_age** | **recency of last failure** | — |
| **n_covering_tests / coverage_rank_prior** | **coverage-set size / coverage-set-size prior** | — |

### 0.7 Models and selectors (§7)

| project term | recommended |
|---|---|
| **Selector** | **ranker** (ML) / **RTS technique** (SE) |
| **Score** | **relevance score** (or **priority**, test-prioritisation framing) |
| **Context** | **run context** |
| **Orientation** | **query–document assignment** (IR standard) |
| **Instruction variant** | **prompt template** |
| **Placement control** | **prompt-position control** |
| **Dilution** | **prompt position effect** (IR: **position bias**) |
| **Mirror arm** | **feature-augmented prompt arm** |
| **Importances** | **feature importances** — already standard |

### 0.8 Experiment layer (§8)

| project term | recommended | why |
|---|---|---|
| **Role** | **factor** | DOE standard: a factor is an input varied by design. |
| **Axis** | **factor levels** | — |
| **Element** | **level** (or **variant**) | — |
| **Knob** | **control variable** (alias **hyperparameter**) | DOE: a variable held constant to prevent confounding. |
| **Cell** | **design point** (alias **treatment combination**) | DOE: "design points" are combinations of factor settings. |
| **Comparison** | **contrast** (alias **paired comparison**) | DOE standard. |
| **Group** | **comparison block** | DOE "blocking" — grouping similar units. |
| **Pairable / row-changing** | **paired factor / row-changing factor** | — |
| **Tier** | **cost tier** | Fine as-is. |
| **Poisoned element** | **undefined level** | — |
| **Unmeasured cell** | **missing cell** | Standard: missing data. |
| **Binding** | **build context** | — |
| **Report** | **run record** | — |

### 0.9 Studies and renderers (§9)

| project term | recommended | why |
|---|---|---|
| **Arm** | **experimental condition** (or **configuration**) | "Arm" is clinical-trials usage; it is understood but imprecise here. |
| **Driver** | **imperative runner** | The contrast with "renderer" is imperative vs declarative. |
| **Renderer** | **artifact reporter** | It runs a declared arm and writes the artifact — "renderer" implies graphics. |
| **Reading** | **derived metric** (post-hoc analysis) | — |
| **Rung / ladder** | **ablation level** / **ablation study** | Standard ML. |
| **Panel** | **analysis figure** | — |
| **Gate / T0** | **quality gate** | Standard. |
| **Falsifier** | **falsification criterion** / **preregistered prediction** | DOE standard: **preregistration**. |
| **De-lexicalisation** | **identifier obfuscation** | Descriptive and standard. |

### 0.10 Sources, mutation and the SUT (§10)

| project term | recommended | why |
|---|---|---|
| **Mutant / killed / survived / mutation score** | unchanged | Already the standard vocabulary. |
| **Trampoline** | **mutant dispatch** (implementation); the concept is the **mutant schema** | Keep "trampoline" only in code comments. |
| **Verdict** | **mutation verdict** | — |
| **Outcome** | **test outcome** (per mutant,test) | — |
| **Label source** | **label source** (alias **oracle**) | The labels are the oracle. |
| **Deselected test** | **excluded test** | — |
| **Out-of-coverage fault** | **coverage-invisible fault** | States the property: its revealing test does not cover the change. |
| **Full-suite relabelling** | **retest-all relabelling** | SE standard: **retest all**. |
| **Boundary-broken corpus** | **cross-boundary corpus** | — |

### 0.11 Artifacts, keys and conventions (§11)

| project term | recommended | why |
|---|---|---|
| **Artifact (as specification)** | **golden file** | The recorded output that a change must reproduce is a *golden file*. |
| **Leaf-by-leaf verification** | **golden-file (snapshot) comparison** | Standard names: golden-master, snapshot, approval, characterization testing. |
| **Documented addition list** | **accepted-delta list** | — |
| **Score cache** | **precomputed score cache** | Standard: cached scores. |

### 0.12 Cost and statistics (§12)

| project term | recommended |
|---|---|
| **delta** | **paired effect size** (or **paired difference**) |
| **CPU / GPU arm** | **resource class** |
| **Bootstrap stream is positional** | **RNG draw order** |
| **`--pilot N`** | **temporal prefix sample** (state it is not a random sample) |
| **b0.05** | **budget fraction 0.05** (cut-off k reported alongside) |

### 0.13 The three collisions worth memorising

Renames are cheap; *collisions* corrupt meaning. The three to fix first:

1. **suite reduction** — here: the fraction of the suite skipped for one change. In SE: **test
   suite reduction / minimisation**, a technique that permanently removes redundant tests.
   Different technique, same words. Use **selection ratio**.
2. **population** — here: the subset a metric averages over. In statistics: the whole universe
   sampled from. Use **averaging subset**.
3. **traceability** — here: filename/path proximity heuristics. In SE: the existence of a link
   between requirements, code and tests. Use **proximity & naming heuristics** for the feature
   family, and reserve "traceability" for the property the study is really about.

---

## 1. The study's problem vocabulary

The words that describe the RTS problem itself, independent of how anything is implemented.

| Term | Meaning |
|---|---|
| **RTS** | Regression Test Selection. Given a change, rank/pick the subset of the existing suite to run. The study asks whether a semantic reranker can do this better than cheap structure. |
| **SUT** | System Under Test. Here `marshmallow` at a pinned commit, cloned to `sut/marshmallow`. Also used for any project a dataset is built over. |
| **Change** | The unit a ranker ranks tests for: one candidate modification. For the mutation dataset one change is one *mutant*; for BugsInPy it is one bug-inducing commit. `Dataset.changes`. |
| **Change id** | Stable identity of a change. `Dataset.change_id(c)`; keys the change index **and** the score caches. Distinct from `coverage_key` (see §3 and §11). |
| **Diff text** | The change's unified diff. The archetypal *required operation*: nothing generic can obtain it, because extraction differs per source. `Dataset.diff_text(c)`. |
| **Test id** | One test, as a pytest node id (`tests/test_x.py::test_y`). `TestId = str`. |
| **Test suite** | The full set of tests a dataset offers. `Dataset.test_suite()`; `test_ids`/`test_index` are the index-alignment convention every matrix obeys. |
| **Candidate set** | The subset of the pool that is *rankable* for a given change — a per-`(change, test)` pair mask. Not the same as the pool, not the same as the subset (§2, §4). |
| **Candidate policy** | `full` (every test, the realistic RTS setting), `coverage_restricted` (only tests covering the change, plus that change's killing tests), `own` (a corpus where each change has its own pool). `accessors.candidate_sets(ds, mode)`. |
| **Killing test** | A test that fails against the change — i.e. that detects the fault. `Dataset.killing_tests(c)`. This is the **label**. Mutmut calls the corresponding change "killed". |
| **Executed test** | A test that was *actually executed* for the change. `Dataset.executed_tests(c)`. Not the same as "passed": a test that never ran is absent from both sets, and the distinction separates "passed" from "never ran". |
| **Fault** | A change that **can be caught**, i.e. has at least one killing test. `accessors.fault_mask(ds)` / `fault_idx(ds)`. A fault is *caught* if any killing test is selected for it. |
| **Failure** | A test's *failing outcome*. Used for history: `cumulative_failure_rate`, "failure history", `(file, test)` pair failure counts. A fault is a change; a failure is an event in a test's past — the two are different kinds of thing. |
| **Detectable** | A change with ≥1 killing test. The default averaging subset name. |
| **Selection budget** | For every change, `k = ceil(budget × n_candidates_for_that_change)` tests are selected. A fraction of *that change's own* suite, never one global count. |
| **Recall** | Fraction of faults caught: caught faults / faults. The discriminating metric here. |
| **Precision** | Selected tests that are killing tests, over selected tests. Tiny for everyone in this benchmark because there is ~1 killer per fault while a budget selects dozens of tests. |
| **F-measure** | Harmonic mean of recall and precision. |
| **Selection ratio** | `1 − k/n_tests`. Identical for every ranker by construction, because the budget fixes `k`: it is a property of the budget, not the model. |
| **Recall@k** | The reported curve shape: recall as a function of the budget. `plan.md` commits to reporting a **cost-effectiveness curve** (time saved vs faults missed) once real data arrives. |
| **Mutation testing** | Generating "changes" by mutating source and observing which tests fail. The synthetic-history engine of this study. |
| **Mutation score** | Fraction of mutants killed (87.2% here). |
| **Coverage-and-naming filter** | The cheap narrowing `function_coverage ∧ filename_match`, which cuts the suite to a median of ~9 candidates. Finding the filter is most of the task; the residual ranking is small. |
| **Usual suspect** | The test that a file's history says normally catches changes there. History-based methods find usual suspects; a semantic model is the only thing that could find an *unusual* one. Panel C's mechanism. |
| **Recurrence** | How often a `(file, test)` pair reappears across changes. `reporting.recurrence(ds)`; median 159× here, i.e. the synthetic history over-repeats pairs versus real evolution. |
| **Boundary** | A process/execution boundary the changed code sits behind (Python tests → serial/socket → firmware). The target regime: it kills coverage, filename matching, identifier overlap and history at once. |
| **Lexical-overlap rate** | The fraction of changes whose killing test shares tokens with the change text. The gate (`T0`) that must be measured *before* building a text-model experiment: if the bridge is absent, no text model can work. |
| **Lexical-overlap signal** | The shared-vocabulary shortcut BM25 depends on. |
| **Co-change proxy** | For MicroPython, where no test-execution labels exist: BM25 recall of the file a change co-changes with, used as a stand-in for real failing tests. |

---

## 2. The dataset half

`rts/data/` — what a dataset supplies, what it declares, and the values those declarations
speak in.

### 2.1 The contract

`rts/data/contract.py` defines **the vocabulary and the interface**; it defines no statistic.

| Term | Meaning |
|---|---|
| **Dataset** | The unit of evaluation: supplies changes, a test pool, and the outcome relation between them, plus the inputs to rank tests for a change. A plain value — constructing one has no side effects, two may coexist in a process, reading one has no side effect on another. `rts/data/contract.py:Dataset`. |
| **Required operation** | A dataset-specific supply whose *extraction* cannot be generic: `name`, `changes`, `files(c)`, `diff_text(c)`, `killing_tests(c)`, `executed_tests(c)`, `test_suite()`, `test_source(t)` (plus optional `coverage(c)`, `durations()`, `own_candidate_pool()`, `order_seed`, `change_id(c)`, `coverage_key(c)`). |
| **Metadata declaration** | A machine-readable answer to a question the harness asks: `capabilities()`, `ordering()`, `test_granularity()`, `annotations()`, `source_counts()`, `integrity_notes()`. |
| **Derived feature / accessor** | A harness function over the required operations, supplied once and inherited by every dataset, so an identical statistic means an identical quantity across datasets. Lives in `rts/data/accessors.py`, `rts/features/`, `rts/data/subsets.py` — *not* on the contract. |
| **Capability** | Optional inputs a dataset may or may not have: `coverage`, `durations`. Absence is a fact, not a zero column. `Capability` enum. |
| **Requirement** | What a computation needs in order to be *defined at all*: `labels`, `diff_text`, `coverage`, `durations`. `Requirement` enum. A capability is a requirement of the same spelling; `requirement_for()` is the only capability→requirement map. |
| **Policy** | A requirement about *order* rather than inputs: `ordering.natural`, `ordering.effective`. Kept separate from `Requirement` so one id does not carry three meanings. |
| **Material** | The named inputs a computation reads, catalogued once in `accessors.MATERIAL`. Requirements are *derived* from the inputs a computation names, so a need cannot drift from the code that reads it. `requirements_for(names)`. |
| **Ordering** | `natural` (the source carries a real sequence) or `synthetic` (it does not). Declared, never inferred. Enforced by `Ordering` enum. |
| **Order seed** | The seeded permutation applied to an *synthetic* dataset before computing temporal features. Evaluation configuration, not a property of the data. `Dataset.order_seed`. |
| **Test unit** | What one test id denotes: `module`, `class`, `function`, `case`. The harness-checkable half of the annotations split; used for pooling compatibility checks. `Granularity` enum. |
| **Semantics** | Free-form human-readable qualifications, keyed by attribute, that the enums necessarily flatten. `Dataset.annotations()`. |
| **Undefined** | A *state*, not a value: a quantity that cannot be defined (rather than merely being zero or absent), carrying the requirement that was not met. Propagates; a undefined design point is reported, never dropped. `contract.Undefined`. |
| **Diagnostic** | A *value*, not a log line: a structured record (`code`, `requirement`, `note`, `scope`) returned by the computation that produced it and travelling with its result. `contract.Diagnostic` / `contract.Diagnostics`. Known codes include `feature.history_on_imposed_order`, `split.shuffles_observed_order`, `split.shuffles`, `split.contiguous_prefix_on_imposed_order`, `dataset.multi_file_changes_flattened`, `dataset.history_off_for_shuffled_run`. |
| **Integrity note** | A divergence a dataset declares about itself (e.g. a pool whose constituents disagree on `test_granularity`). `Dataset.integrity_notes()`; surfaced by `reporting.audit`. |
| **`cached(key, factory)`** | The one memo mechanism on the contract, so the derived accessors share a single cache rather than nine hand-written copies. |

### 2.2 Concrete datasets and composition

`rts/data/datasets.py`, `rts/data/sources.py`, `rts/data/mutmut.py`, `rts/data/composition.py`.

| Term | Meaning |
|---|---|
| **Source** | The raw inputs for one SUT or revision, holding no module-level state. `MutmutSource(labels=, sut=)`, `BugsInPySource(project)`. |
| **Sample generator** | Turns source records into samples (changes with killing/ran tests, plus coverage). One pair of generators per data format. |
| **MarshmallowDataset** | The mutation-testing dataset over `marshmallow`. `datasets.marshmallow(labels=, sut=, order_seed=)`. |
| **BugsInPyDataset** | One per project (8 of them); one dataset per project because suites, pools and meanings are separate. `datasets.bugsinpy(project)`. |
| **DerivedDataset** | A dataset defined as a transformation of another (bundle, filtered subset, relabelled variant). |
| **BundleDataset** | A derived dataset that merges several base changes into one sample. |
| **PooledDataset / PooledChange** | Several datasets evaluated as one. Pooled changes are wrappers carrying `.dataset`/`.change_id`/`.own`/`.key`; identity is namespaced, never keyed by `id()`. |
| **Namespace** | `dataset::nodeid`, resolving test-id collisions across pooled projects. A dataset name may not itself contain `::`. |
| **Mixed test units** | A pool whose constituents disagree about what a test is. Reported, not refused: `PooledDataset.mixed_test_units()`. |

---

## 3. Derived quantities and accessors

`rts/data/accessors.py` — free functions over the contract. The load-bearing ones:

| Term | Meaning |
|---|---|
| **labels** | `[n_changes, n_tests]` uint8 matrix, 1 where the test failed against the change. The evaluation contract's label matrix. Read-only. |
| **runs** | `[n_changes, n_tests]` uint8 matrix, 1 where the test was actually executed. Separates "passed" from "never ran". |
| **coverage_sets** | Tests covering each change, index-aligned. Requires the `coverage` capability. |
| **fault_mask / fault_idx** | Boolean per change "has a killing test" / the indices of those changes. |
| **test_fault_idx** | Detectable changes *inside a given evaluation window*. |
| **change_paths** | One path per change (the first, when a change touches several). Does not warn — flattening is reported by `multi_file_changes` / `audit`. |
| **multi_file_changes** | Indices of changes touching more than one file, and therefore flattened by the single-file derived features. |
| **change_index** | Change id → row. |
| **candidates** | The `(change, test)` eligibility mask, per candidate mode (§1). |
| **candidate_counts** | Per-change count of rankable pairs; what `budget_k` multiplies. |
| **pair_counts** | How often each `(file, test)` combination recurs across changes. Requires `coverage`. |
| **pair_failures / pair_runs** | Failure counts / run counts for a `(file, test)` pair. What the cold_start and no-prior-failure subsets read. |
| **sparse_mask** | Changes whose every `(file, test)` pair recurs at most N times. The sparsity proxy. |
| **inputs** | The catalogue (`accessors.MATERIAL`) of named inputs a computation may read; resolves to a mapping handed to a subset predicate. |
| **change_id vs coverage_key** | `change_id` is stable identity and keys the change index *and* the score caches; `coverage_key` keys the coverage map (for mutmut it is the function key **without** the `__mutmut_N` suffix). Conflating them mis-maps caches. |

---

## 4. Populations

`rts/data/subsets.py`. A **subset** is a named subset of the **evaluation window**
(`rows`) that a metric is *averaged over*. Three sets are deliberately distinct:

- **candidates** — per `(change, test)`: which pairs are rankable;
- **rows** — per change: which changes are in the evaluation window at all;
- **subset** — a named subset of `rows` that the average is taken over.

Requirements are *derived* from the inputs a subset names; **unavailable is not empty** —
a subset the dataset cannot support returns `Undefined`, never an empty array. The registry
is a value (`SubsetRegistry`), composable with `+`; the study's own set is `STUDY`.

| Subset name | Meaning |
|---|---|
| `detectable` | Changes with ≥1 killing test. The default averaging subset. |
| `no_prior_failure` | Killing `(file, test)` pairs with no earlier failure (`max_failures=1`). |
| `cold_start` | The fixed-threshold data-cold_start proxy (`max_failures=2`). |
| `cold_start<N>` | Parameterised form, `subsets.cold_start(N)` — counts include the change itself, so `cold_start1` means no prior failure history at all. `cold_start2` = 43 held-out faults, `cold_start5` = 141 (under *mutmut* labels). |
| `cache_covered` | Not a threshold: the held-out changes **the ladder's SemIf cache covers**, named for provenance. Its subset definition is the cache's coverage; when the cache is absent it is *unavailable*, not smaller. |
| `held_out` | Every held-out fault-bearing change ("the whole evaluation window"). |
| `low_pair_recurrence` | Changes whose `(file, test)` pairs rarely recur (`max_pair_count=1`). |
| `low_cooccurrence<N>` | Parameterised sparsity proxy, `subsets.low_pair_recurrence(N)`; the recorded condition uses `low_cooccurrence80` and `low_cooccurrence160`. |

> **Starved vs low_cooccurrence** — two different proxies that are easy to confuse. *Starved* keys on the
> **failure history** of a killing pair (a data-cold_start deployment). *Sparse* keys on the
> **recurrence of coverage pairs** (a proxy for real evolution). The low_cooccurrence condition is a separate
> `Experiment` because its budget grid differs — a control variable is constant across a run.

---

## 5. Splits and the evaluation contract

`rts/data/splits.py`, `rts/evaluate.py`.

| Term | Meaning |
|---|---|
| **Split** | A partition of a dataset's canonical order into a **train prefix** and a **test tail**. A value, not a dataset method: a dataset cannot report which of its changes were held out. `splits.make_split(ds, train_fraction, shuffle, seed)`. Default 80/20 → 2121 train / 530 held-out. |
| **Evaluation window** | Alias for the test tail — `Split.test_idx`. Metrics may only be averaged over rows the split did not hold out as training data; `splits.require_in_window` is the one guard. |
| **Effective ordering** | The dataset's ordering, *unless* the split shuffles, in which case it is synthetic and temporal features are off by default. `Split.effective_ordering`. |
| **Context** | Everything a ranker may read: `ds`, `features` (a `FeatureMatrix`), `split`, `bm25`, `extras`, `seed`. The split is in the context so every ranker in a run sees the same one. `model/rankers.py:Context`. |
| **Budget** | See §1. `budget_k(budget, n_candidates)`. |
| **Probe budget** | The single budget at which a `Contrast` reports its paired delta (0.05 here). |
| **BudgetResult / Evaluation** | One row of the sweep / the whole sweep with the subset it was averaged over and any caveat. `Evaluation.results is None` exactly when it is undefined. |
| **UnmeasuredPopulation** | Raised by `evaluate()` when a measurement is requested from a subset that cannot exist; use `evaluate_rows()` to get the `Undefined` back instead. |
| **Curve from arrays / sweep_matrices** | The metric sweep expressed over matrices `(scores, labels, candidates, rows)`, so a bundle condition carrying its own label matrix is evaluated by exactly the same code. The cumulative-argsort trick yields every budget at once. |
| **Paired bootstrap** | The delta between two models computed over the *same* rows, with a bootstrap CI and p-value. Pairable across `model` and `features`; not across the row-changing roles (§8). |
| **n (in a table header)** | The number of averaged rows the delta was computed over (e.g. n=43, n=141). |
| **Power limit** | A contrast that can never reach significance at any feasible n on this benchmark (the SemIf-vs-BM25 difference of 0.037). Reported as a limit, not as evidence of equivalence. |
| **±0.02** | The bf16 non-reproducibility floor: batched scoring is not bitwise reproducible, so **do not read differences below ~0.03 as findings**. |

---

## 6. Features

`rts/features/`. A feature set is a declared value, not control flow.

| Term | Meaning |
|---|---|
| **FeatureBlock** | A declared set of columns: name, note, and an ordered tuple of groups. `STRUCTURED` (15 columns), the bundle block. |
| **FeatureGroup** | Columns produced together by one function, declaring the `needs` it reads. |
| **FeatureColumn** | One column: `name` and its `family`. |
| **FeatureMatrix** | The built result: `.X`, `.columns`, `.column(name)`, `.index(name)`, `.keep`/`.without`, `.audit`, `.undefined`, `.diagnostics`. Replaced the bare `(X, names)` pair so a ranker asks for a column *by name* and fails loudly on a rename rather than reading whatever moved into that slot. |
| **Family** | A cross-cutting label used for ablation: `coverage`, `traceability`, `history` (plus `intrinsic` for size/duration columns). Declared per *column*, because a family can take one column from each of two groups. |
| **`without_families(...)`** | Withholds columns by family. Ladder ablation levels are built this way; a typo in a family name raises rather than quietly ablating nothing. |
| **Undefined column** | A column whose group the dataset cannot support: measured-but-zeroed, with the reason in the matrix audit. Rungs rely on this, so a block's requirements must not gate the design point. |
| **Structured block** | The 15 columns the classical rankers read, in recorded order. |

The 15 structured columns, in order (group · family):

1. `function_coverage` — coverage · whether the change's code is observable by the test
2. `n_covering_tests` — coverage
3. `coverage_rank_prior` — coverage
4. `path_proximity` — traceability · directory-tree distance
5. `tests_per_file` — traceability
6. `filename_match` — traceability (formerly mislabelled `module_name_in_test_file` — the value is a filename-*stem* match)
7. `test_duration` — intrinsic · wall-clock, hardware-dependent (available but not comparable)
8. `test_lines` — intrinsic
9. `test_tokens` — intrinsic
10. `code_churn` — intrinsic · added + removed lines
11. `change_added_lines` — intrinsic
12. `change_removed_lines` — intrinsic
13. `cumulative_failure_rate` — history · cumulative, over strictly earlier changes
14. `test_runs_cum` — history
15. `failure_recency` — history

| Term | Meaning |
|---|---|
| **`changed_lines` / `removed_lines`** | Added / removed lines parsed from a unified diff (the `+++`/`---` headers excluded). |
| **`change_query_text`** | The change side of a pair: added + removed lines. What BM25 and the reranker read as the query. |
| **History features** | The three cumulative columns. **Off by default for an `synthetic` dataset**, because a cumulative feature over an arbitrary order does not merely fail to be interpretable, it manufactures a leak. Enabling them is an explicit experiment, reported as a spread over seeds. |
| **Bundle block** | Features of a *set* of base changes presented as one change: how much of the bundle each test covers, whether any member's filename matches, the minimum path distance, and aggregates of the base's size/duration/history columns. Reaches the base's columns through `FeatureMatrix` by name, not by position. |
| **BM25** | Okapi BM25 between the changed lines and the test source, change text as query, tests as documents. The cheapest competing explanation for a transformer win, therefore a required baseline. `features.text.BM25Scorer`, `features.text.tokenize`. |
| **KEYWORDS** | Python keywords/boilerplate tokens dropped by `tokenize` (they appear in every test and would swamp the lexical signal). |
| **Embedding baseline** | Mean-pooled `microsoft/codebert-base`, L2-normalised, cosine between change text and test source (P3). |

---

## 7. Models and rankers

`rts/model/`. Every ranker implements `scores(ctx) -> [n_changes, n_tests]`; higher is better;
evaluation takes the top `k` per change. `Ranker.requirements()` declares the inputs beyond
the dataset contract that the ranker reads (`artifact:<path>` for a cache, or a `Requirement`).

| Ranker name | Meaning |
|---|---|
| `random` | Uniform random scores; the floor. `RandomSelector(seed=None)` means "the run's seed". |
| `recency` | Select the tests that failed most recently. Degenerate by construction here — reported as a limitation of the setup, not a finding. |
| `failure_rate` | Cumulative per-test failure rate. |
| `coverage` | Tests covering the mutated function, preferring smaller coverage sets. Mutmut's own association. |
| `structural_rule` | Hand-built: covered ∧ filename-match, then shortest test first. A three-line rule that explains most of the achievable recall. |
| `bm25_lexical` | The BM25 baseline. |
| `bm25_change_shuffled` / `bm25_test_shuffled` / `bm25_both_shuffled` | The **shuffle ablations**: re-score the same pairs with the change text (or test text, or both) shuffled across changes. If recall barely drops, the signal is a change-independent test prior rather than a match. |
| `XGBoost` variants | See the naming grammar below. |
| `semif_reranker` / `semif_textonly` | SemIf in reranker mode, reading a score cache; text-only (no structured features in the prompt). |
| `semif_direct_pairwise` | Direct mode adapted to two options (P1). |
| `embed_codebert` | The code-embedding baseline (P3). |
| `rankaverage_xgb_semif` | Fitted-free rank average of SemIf and a tree (P5's cheap condition). `models.RankAverageSelector`, `models.normalised_rank`. |
| `CachedScores` / `ProducedScores` | Selectors that read a precomputed score artifact (the general form; `SemIfSelector` subclasses `CachedScores`). |

**XGBoost naming grammar.** Parts combine, so the name states the feature set:

- `struct` = with the **history** columns; `static` = **no history**;
- `_lex` = **BM25** added as an extra column;
- `nocov` = **coverage** columns excluded;
- e.g. `xgboost_static_nocov_lex` = no history, no coverage, but BM25.

| Term | Meaning |
|---|---|
| **SemIf** | The discriminative transformer under test: `Qwen/Qwen3-Reranker-4B` wrapped by the pinned SemIf adapter. Raw yes/no log-odds readout (the repo's normalization destroys cross-change comparability). |
| **Reranker mode** | Each `(change, test)` pair scored independently, so scores share one global scale. The main condition. |
| **Direct mode** | The model states a criterion and reads native next-token logits on option letters — a *decision*, not a relevance rating. A change of task formulation. |
| **Windowed direct mode** | The repo's native 2–16-option form. Untested: a window is only comparable within itself, and a multi-topic window would hurt for dilution reasons. |
| **Orientation** | Which side is Query and which is Document: `change_query` (change as Query, test as Document) or `test_query`. Rerankers are asymmetric, so this is a real design choice; only tested at n=10, so untested at adequate n. |
| **Instruction variant** | A prompt wording for the reranker (P2): `default` (reference), `execution` (ask about observable runtime behaviour), `fault` (ask for the prediction directly), `retrieval` (the **negative control**: lean into the native topical-relevance prior), `terse` (shortest question). |
| **Placement control** | Where in the prompt the structured features are put (§5.5): `textonly` (reference), `after_document`, `informative` (only varying features), `placebo` (same field names, every value `n/a`), `shuffled` (real values, decorrelated), `full` (all 15, before the content). |
| **Dilution** | The measured effect that ~200 tokens of *any* text before the content costs ~0.21 recall — a **position** effect, not a content effect. "Query dilution" is the related degradation when a bundle concatenates unrelated edits. |
| **Mirror condition** | Feeding the structured features into the prompt. Its apparent "degradation" was a position artifact. |
| **Score cache** | A resumable JSONL of pair scores, keyed on `(dataset, features, model, split)` plus the model seed. Subset is *not* in the key. Scoring is a one-time cost per condition; everything downstream is CPU-only. |
| **Pairs/s** | Reranker throughput: 30.1 at batch 8 (best found); direct mode 1.3. Cost ≈ pairs / 30 seconds for the reranker. |
| **Importances** | XGBoost feature importances, recorded only on the design point that actually scored. **Unstable — trust ablations, not importances.** |

---

## 8. The experiment layer

`rts/experiment/`. An experiment is a *declaration*; running it is separate.

| Term | Meaning |
|---|---|
| **Role** | A slot in a design point: exactly five — `dataset`, `features`, `model`, `subset`, `split`. A new *level* is cheap and is what most new dimensions actually are; a new *role* is a deliberate kernel change. |
| **Factor** | A named, ordered set of variants of one role. `Factor("model", (...))`. Factor builders are functions (not constants) wherever levels hold state, so two runs cannot share one trained model. |
| **Level** | One variant of one role: a name, a builder (`make`), a cost tier, an estimated seconds, a note, and optionally an applicability predicate. `Level`. |
| **`constant(name, value, ...)`** | The idiom for a level whose value is already built. |
| **Binding** | What a level is handed when built: the environment, the design point's resolved dataset, and (only while applicability is checked) the design point's factor names. A value is built **once per run** and shared across design points, so `make` must not read `factors`. |
| **Knob** | A run-level constant: **a choice that is free** — the harness cannot derive the right answer from anything else. Controls do not multiply into design points and are recorded once: `seed`, `budgets`, `n_bootstrap`, `n_bootstrap_paired`, `candidates`, `model_seed`. The layer's one rule: *anything the harness can derive must not be an option.* |
| **Environment** | What a run offers: the controls, an `out_dir`, a `caches` name→path map, and free-form `shared` values a caller can inject. |
| **DesignPoint** | One point in the product over the five factors plus the controls. Its `key` is `"dataset=..\|features=..\|model=..\|subset=..\|split=.."`; `factors` are the level names by role. |
| **Undefined design point** | A design point reported as undefined rather than dropped, for exactly one of five reasons: level unavailable, subset unavailable, model requirement unresolved, tier not enabled, applicability. Dropping is what turns "we asked and could not answer" into a silently halved contrast. |
| **Tier** | A coarse cost class — `cpu` or `gpu`. A design point's tier is its *most expensive* level's; a cache-reading design point is still `cpu` because features and evaluation always compute (which is why a third `cache` tier was removed). |
| **`applies`** | An level's applicability predicate, for variants meaningful only in combination with a particular level of **another** role (a cache covering one subset and not another). Asked per design point; `factors` are populated only here. |
| **`Factor.map(fn)`** | Applies a *shared option* to every level of a factor. An option that cannot be expressed for a level yields a **poisoned level** which is kept, so its design point is reported undefined rather than vanishing. |
| **Poisoned level** | An level an option could not be expressed for; kept so the resulting undefined design point appears as a finding. |
| **Contrast** | A paired delta against a reference level of one role, at a probe budget. `Contrast(role, reference, probe_budget)`. |
| **Pairable role** | `model`, `features` — varying these changes scores, not the quantity being averaged. |
| **Row-changing role** | `dataset`, `subset`, `split` — a delta across these compares different subsets and is **refused at construction** rather than producing a plausible number. |
| **Group** | The set of design points sharing the other four roles; a design point is paired against the reference level's design point in its own group. |
| **Requirement resolution** | `artifact:<path>` resolves by file existence; anything else resolves against the design point's dataset. An unrecognised spelling raises (the vocabulary is closed). |
| **`run(exp, out_dir=, controls=, shared=, caches=, tiers=)`** | The measurement. Elements and score matrices are memoised per run. |
| **Report / DesignPointResult** | What a run records: per design point the key, factors, tier, seconds, ranker, subset, n_rows, n_changes, split, dataset declaration, feature audit, `diagnostics` (derivation) and `audit` (dataset+split) held **separately**, results, importances. The artifact also carries the undefined design points and the contrasts. |

---

## 9. Studies, conditions and renderers

`rts/studies/` (the declarations) and `rts/render/` (the drivers).

| Term | Meaning |
|---|---|
| **Arm** | A complete `Experiment` value. |
| **Headline condition** | The full ranker set over the full candidate set; reproduces `results_full.json`. `study_condition()`. |
| **Sparse condition** | The same rankers over rarely-recurring corners; a *separate experiment* because it reports three budgets where the headline condition reports six. `low_cooccurrence_condition()`. |
| **Ladder condition** | The traceability-loss ladder; reproduces `ladder.json`. |
| **Variation condition** | One of the P1–P5 sub-experiments; reproduces sections of `variations.json`. |
| **Driver** | The *legacy* name for a hand-written sweep module: control flow in code that re-implemented the same sequence (build dataset → split → context → loop → table). Five of them: `pipeline`, `ladder`, `variations`, `bugsinpy`, plus `panels`/`figures`. `rts/bundles.py` is the last driver outside `rts/render`. |
| **Renderer** | A module under `rts/render/` that runs a *declared* condition from `studies` and writes the legacy artifact shape. A renderer contains **no experiment logic**: `pipeline.py`, `ladder.py`, `variations.py`, `bugsinpy.py`. |
| **Reading** | A quantity *computed from a report* rather than recorded by the kernel — an interpretation, kept apart from the data so the table can be recomputed from the layer's own records. `studies/readings.py`; the example is `semif_margins`, the ladder's headline quantity. |
| **Reading (panel)** | Also used for `panels.py`, which the design deliberately keeps off the layer: a panel is a *reading*, not a sweep, and its deciles are dataset-derived subsets plus a per-*row* availability filter that `applies` (per-*design point*) cannot express. |
| **Rung** | One step of a ladder: a feature block with families withheld. |
| **Traceability-loss ladder** | Removes feature families cumulatively and asks whether SemIf crosses the classical baselines as traceability is withdrawn. Rungs `L0_all`, `L1_nohistory`, `L2_nocoverage`, `L3_notrace`. A ablation level name states what is **unavailable** at that ablation level. |
| **Bundle / complexity ladder** | Broadens the *change* while holding the *answer* fixed. Rungs `0`–`5` by `(n_distractors, cross_file, distractor_kind)`. |
| **Distractor** | A change bundled alongside the signal mutant. `survived` (no killing tests, so the union label stays exact), `killed` (ablation level 4's control: inflates the killer count and makes RTS easier), `coherent` (ablation level 5: chosen for relatedness rather than at random). |
| **Union annotations** | A bundle is "caught" if a selected test is sensitive to any edit in it. |
| **Signal mutant** | The killed mutant a bundle is built around; the bundle's label is exactly its kill set. |
| **Panel A / Panel C** | The sparsity panels. A: recall against change-subset sparsity (faults binned into equal-count deciles by killing-pair failure history). C: the mechanism — the structural-funnel fraction and what the structural models actually recover per bin. |
| **Probe** | A throwaway measurement script under `scripts/probes/`, and generally a measurement taken to decide whether to build something (e.g. the MicroPython bridge probe, the T0 gate). |
| **P1 … P6** | The ranked proposals from the SemIf analysis: **P1** direct mode (change of task formulation); **P2** instruction/prompt sweep; **P3** code-specialised embedding baseline; **P4** model substitution; **P5** SemIf score as an XGBoost feature (redundancy test); **P6** prompt-side dilution mitigation (subsumed by P2). |
| **Gate / T0** | A pre-committed check that must pass before an expensive condition is run (T0 = measure the bridge rate before building any text-model experiment). |
| **Pre-registered falsifier** | A prediction stated in advance whose failure discredits the hypothesis — e.g. "the reranker's margin over bag-of-words should *grow* with change complexity"; it shrinks and inverts, and the falsifier failed. |
| **De-lexicalisation** | The retired Gap-2 manipulation: rename identifiers/obfuscate vocabulary in the diff only (A1), on both sides (A2), or including test names (A3). Removed BM25's signal but left the coverage tree immune. |

---

## 10. Sources, mutation and the SUT

| Term | Meaning |
|---|---|
| **mutmut** | The mutation framework (3.8.0). Chosen over `cosmic-ray` because it runs only the tests associated with the mutated function. Mutates only inside functions with a narrow operator set. |
| **Mutant** | One generated mutation; one change. |
| **Killed / Survived** | A mutant killed by ≥1 test / by none. Zero no-tests here. |
| **Trampoline** | mutmut's generated dispatch around each function: one `__mutmut_orig` block plus one block per mutant; diffing them yields the change text. |
| **`MUTANT_UNDER_TEST`** | The env var mutmut's trampolines read per call to select the active mutant. Non-mutant sentinel values (`""`, `mutant_generation`, `fail`, `stats`) must be excluded. |
| **Verdict** | mutmut's per-mutant judgement (killed/survived), derived from exit codes. |
| **Outcome** | A per-`(mutant, test)` result written by the harness plugin: `{mutant, nodeid, when, outcome}`. No mutation framework records these, so the harness adds the hook. |
| **Label source** | Which outcome log supplies the labels: `mutmut` (mutmut's own selection; the default, reproducing every documented number) or `full` (the full 1190-test suite run per mutant). Selected with `--labels {mutmut,full}` / `RTS_LABELS`. |
| **Deselected test** | Three mutmut-incompatible tests kept out of the run (`--deselect`); left in they fabricate "killed" labels. |
| **Out-of-coverage fault** | A fault whose real killer does not cover the changed function — structurally invisible to mutmut's labels. 5.5% (128/2327) under full-suite labels. Also called an **indirect fault**. |
| **Full-suite relabelling** | Gap-1 fix: run the whole suite per mutant and rebuild labels from the outcome log. ~5 min wall-clock here. |
| **marshmallow** | The SUT: pure Python, minimal deps, pytest suite of 1190 tests, 98% coverage, logic-dense validation code. |
| **SemIf** | The discriminative transformer under test (an open-source pretrained model wrapping frozen Qwen checkpoints). |
| **BugsInPy** | Real multi-project data: real bugs with known `failing_tests`. No coverage at all. |
| **MicroPython** | The boundary-broken corpus: Python tests driving C firmware over a subprocess/pty boundary. Labels are a co-change proxy. |
| **Zephyr / SWE-bench / Defects4J** | Candidate corpora named as alternatives for scale, a second language (Java), or genuine per-test failure history. |

---

## 11. Artifacts, keys and file conventions

A recorded artifact is the **specification, not a cache**: if a change moves a number, either the
change is wrong or the movement is a finding to argue and record — never absorbed.

| Term | Meaning |
|---|---|
| **Artifact** | A recorded result under `artifacts/`. The recorded shape is what renderers must reproduce. |
| **Leaf-by-leaf verification** | The acceptance gate: regenerate an artifact and compare **every leaf**, not a sample of fields. `scripts/verify_experiment_layer.py` (every condition, ~30 min) and `scripts/check_fast.py` (the cheap part, ~50 s). |
| **Documented addition list** | Leaves a migration is allowed to add; an unexpected new leaf fails the gate. |
| **Results artifacts** | `results_full.json`, `results_covered.json` (headline condition, by candidate mode); `ladder.json`; `bugsinpy_results.json`; `variations.json`; `bundles_*.json`. |
| **Score caches** | `semif_scores*.jsonl` (resumable JSONL), `embed_scores.npy`. Naming records the condition: e.g. `semif_scores_starved2_full.jsonl`, `semif_scores_instr_execution_starved5.jsonl`, `semif_scores_ladder141_full.jsonl`. |
| **Variation section keys** | `full_starved`, `full_starved5`, `full_starved_seeds`, `p2`, `p3`, `p5` (all rendered from declared conditions) plus `p5_trained` and `p1_direct`, which keep their original implementations. |
| **Verdict/outcome key** | `module.x_func__mutmut_N` — last dot separates module from function; `xǁ` marks mangled nesting. |
| **Coverage key** | The same but **without** the `__mutmut_N` suffix. |
| **Span key** | File-local. |
| **Cache key (scores)** | `(dataset, features, model, split)` **plus the effective model seed**. Subset is not in the key (it only restricts which rows a metric averages). |
| **`b0.05`** | Budget 0.05, shorthand. |

---

## 12. Cost, statistics and shorthand

| Term | Meaning |
|---|---|
| **CPU / GPU condition** | Whether the condition needs the GPU. SemIf caches are keyed on text pairs, so feature/pool manipulations are CPU-only. |
| **`delta`** | A paired difference in recall (or the quantity named) between a model and a reference, at a given budget. Reported as `value [lo, hi] p=…`. |
| **Bonferroni correction** | Family-wise correction applied to the P5/P2 significance claims; the two surviving p-values do not survive it. |
| **Bootstrap stream is positional** | Consequence of the RNG: drawing random baselines over a pooled matrix instead of per-bug moves *every* random number; forcing two sweeps into one run moved the low_cooccurrence condition's interval bounds. The RNG consumption pattern is part of a recorded number. |
| **`--pilot N`** | The first N held-out faults by index — a temporal slice, not a random sample. |
| **`--exclude-scored`** | Score only the pairs a previous cache does not already contain (used to extend `cold_start2` → `cold_start5`). |
| **`--tiers cpu`** | Run only the CPU design points; the rest are reported undefined rather than run. |
| **Bridge / gate / falsifier** | See §1 and §9. |

---

## 13. Names that are easy to confuse

| Pair | The distinction |
|---|---|
| **fault vs failure** | A fault is a *change* that has a killing test. A failure is a *test outcome*; "failure history" is a count of past test failures. |
| **killing test vs ran test** | Killing = the label (test failed against the change). Ran = what was executed. A missing pair is "not selected", never "passed". |
| **candidate vs row vs subset** | Candidates = rankable pairs (per change,test). Rows = changes in the evaluation window. Subset = a named subset of rows the metric averages over. |
| **cold_start vs low_cooccurrence** | Starved keys on failure history; low_cooccurrence keys on coverage-pair recurrence. Different proxies, different conditions. |
| **`change_id` vs `coverage_key`** | Change identity (and score-cache key) versus coverage-map key. Mutmut coverage keys drop `__mutmut_N`. |
| **`natural` vs `synthetic` ordering** | Whether the sequence is real or harness-synthetic. Also `effective_ordering`, which a shuffling split downgrades. |
| **`mutmut` vs `full` labels** | The label source: mutmut's selected tests versus the full suite. |
| **Capability vs Requirement vs Policy** | Material the dataset has, inputs a computation needs (derived from the inputs it names), and a requirement about *order*. |
| **struct vs static (XGBoost)** | `struct` includes the history columns; `static` excludes them. |
| **undefined design point vs undefined column** | A design point that could not be measured (reported in `undefined`), versus a feature column the dataset could not support (measured-but-zeroed, audit in the matrix). |
| **unavailable vs empty** | A subset the dataset cannot support is `Undefined`, not a smaller or empty set. |
| **driver vs renderer** | A driver *contains* the sweep as control flow; a renderer *runs a declared condition* and writes the artifact. |
| **L0–L3 ablation levels vs bundle ablation levels 0–5** | Two different ladders: traceability loss (feature families withheld) versus change complexity (bundling). |
| **`cache_covered` vs `cold_start5`** | `cache_covered` (ladder) is "the changes the cache covers", named for provenance — under corrected labels the cold_start filter no longer defines a subset of that size. |
