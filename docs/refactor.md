# Refactoring the RTS harness: the dataset interface

Design doc, not implemented. Scope is **datasets only**; the model/selector half is
deliberately deferred (§9). Companions: `plan.md`, `implementation.md`, `plan_next_steps.md`.

The goal is modularity and testability — every dataset behind one contract, so that a new SUT
or a new change source is one class to write, and so that datasets can be compared and combined
without special cases.

## 1. A dataset

A **dataset** supplies changes, a test pool, and the outcome relation between them, together
with the material needed to rank tests for a change. It is the unit of evaluation, and a plain
value: constructing one has no side effects, and two may coexist in a process.

**Granularity is maximal.** One dataset is the smallest independently meaningful unit of
evaluation — one source, one test pool, one sequence of changes. For a single SUT that is the
SUT; for a multi-project corpus it is one project, because a project's suite is what tests are
run against and pooled evaluation is a decision an experiment makes rather than a property of
the data. All datasets obey the same contract, so they compose by construction: pooling,
partitioning and pairing are all just iteration.

## 2. The contract

The harness declares the interface; each dataset implements it. The interface is cut in two,
and that is the load-bearing decision.

**Primitives** are dataset-specific, because their *extraction* is. `diff_text` is the
archetype: a mutation is reconstructed by diffing generated source, a real change is a git
patch, and a derived change is a concatenation of others. Nothing generic can obtain it.

**Derived features** are harness functions over the primitives. Every dataset inherits them, so
an identical statistic means an identical quantity across datasets.

### 2.1 Primitives

    name
    changes()               -> list[Change]
    files(change)           -> tuple[str, ...]
    diff_text(change)       -> str
    killing_tests(change)   -> frozenset[TestId]
    ran_tests(change)       -> frozenset[TestId]
    test_pool()             -> list[TestId]
    test_source(test)       -> str

plus the optional primitives of 2.3.

`files` is plural because a change may touch several files, and a singular accessor would have
to lie about it. `killing_tests` is the label; `ran_tests` is what was executed. They are not
the same thing, and the distinction is what separates "passed" from "never ran".

### 2.2 Derived features

Supplied by the harness, not implemented per dataset:

    changed_lines, removed_lines, change_size      <- parsed from diff_text
    change_query_text                              <- added + removed lines
    test_n_lines, test_n_tokens                    <- from test_source
    n_tests_in_test_file                           <- from test_pool
    path_distance, filename_stem_match             <- from files x test files
    failure_rate_cum, runs_cum, last_failure_age   <- from labels and ran, over the sequence

This places an obligation on `diff_text`: it must be a unified diff, because the harness
recovers added and removed lines from it. A source with a native patch returns it unchanged; a
source that synthesizes its changes constructs the diff. Both satisfy one contract, which is why
`change_size` is a single function rather than one per dataset — and that is precisely what
makes a cross-dataset comparison meaningful.

A derived feature may be overridden only for a genuine semantic difference, and the override
should be recorded. Silent divergence is how the same statistic quietly stops meaning the same
thing.

### 2.3 Capabilities

Two things are neither derivable nor universally available:

    capabilities()          -> frozenset[str]           # subset of {"coverage", "durations"}
    coverage(change)        -> frozenset[TestId]        # only if "coverage"
    durations()             -> dict[TestId, float]      # only if "durations"

A dataset declares what it has, and the harness requests only that. Absence becomes an explicit,
visible fact rather than an all-zero column that a model will happily split on. Coverage is the
feature that must not be faked: where the test process cannot observe the changed code, the
dataset has no coverage and must say so.

A capability reports **availability, not comparability**. Test durations are hardware-dependent,
so a dataset that has them is still not comparable on them with one measured elsewhere — the
duration is a property of the machine as much as of the test. Where a feature is meaningful only
relative to other datasets, presence is necessary and not sufficient.

### 2.4 Ordering

History features are derivable from any dataset, but they are only *interpretable* when the
change sequence is real. A dataset therefore declares:

    ordering()              -> "observed" | "imposed"

`observed` means the source carries a real sequence. `imposed` means it does not. There is no
separate "none": a sequence is always needed to materialise cumulative features, and absence of
a temporal one is precisely what `imposed` states.

**The canonical order is always deterministic**, temporal or not, so any run reproduces without a
seed. When ordering is `observed`, the canonical order *is* the temporal order — the dataset may
not keep one order for storage and another for history. When it is `imposed`, the harness may
additionally apply a **seeded permutation** before computing history features; that seed is
evaluation configuration, not a property of the data (§5).

**History features are off by default for an `imposed` dataset.** Not out of timidity: a
cumulative feature over an arbitrary order does not merely fail to be interpretable, it
manufactures a leak. Where a (file, test) pair recurs, an arbitrary order lets a change's
"prior failures" include failures that are prior to it in no causal sense, which partly encodes
the label. A real order bounds the feature to genuinely earlier events. So enabling history on
an `imposed` dataset is an explicit experiment, and it is reported as a spread over several
seeds rather than as one number — the seed's variance is a measurement, not a nuisance.

Ordering is **declared, never inferred** — including by a derived dataset, which states its own
rather than inheriting one automatically. A transformation that maps each derived sample to
exactly one point in its base's sequence may declare `observed`; one that does not, or that draws
its parts from anywhere in the base, is `imposed`. The harness cannot verify either claim, so the
declaration carries the responsibility.

**Order is part of a dataset's identity.** For an `imposed` dataset a permutation is free — every
order is equally valid, and a seed is only a reproducibility knob. For an `observed` dataset it is
not: replacing its order discards the property that made it `observed`, so the result is a
*different dataset* declaring `imposed`, not a variant of the original. An experiment wanting a
non-temporal view of a temporal source therefore defines a new dataset rather than reordering one
in place.

### 2.5 Declared semantics

Two datasets can satisfy the same contract while meaning different things by it. The harness
therefore gets, for every attribute it reasons about, a machine-readable value **and** a
human-readable note:

    test_unit()             -> TestUnit        # "module" | "class" | "function" | "case"
    semantics()             -> dict[str, str]  # free-form notes, keyed by attribute

`test_unit` says what one `TestId` denotes. It is the harness-checkable half, used for
compatibility checks, and it necessarily flattens: a parametrization may be collapsed, a "test"
may be a whole scenario script. `semantics()` carries those qualifications, and the enum exists so
that the note is never the only record.

The pattern is general, not special to `TestId`. An enum is **preferred wherever the harness has a
question to ask** — something it compares, checks, or gates a behaviour on — and the matching note
is **always** present. Where there is no such question, the note alone suffices: an enum nothing
reads adds a decision without adding a capability. Adding an attribute therefore means deciding
whether the harness asks anything about it, and writing the note either way.

## 3. Assembling a dataset

A concrete dataset is three pieces of internal machinery behind the contract:

- a **source** — the raw material for one SUT or revision, holding no module-level state;
- a **sample generator** — turns source records into samples: changes with their killing and
  ran tests, plus coverage where available. It may wrap, filter or *transform*;
- the **dataset** — implements the primitives over those samples, and declares its capabilities,
  ordering and semantics.

**One pair of generators per data format; one dataset per evaluation unit.** The eight BugsInPy
projects are eight datasets, because their suites, pools and meanings are separate — but they
share a single source class and a single sample generator, since the projects ship the same
schema and the generator differs only by which project it is pointed at.

Transformation is not a special case. Bundling several changes into one sample is a sample
generator over another dataset, expressed as a dataset that wraps a dataset rather than as a
parallel code path. The same shape covers holding out a file, taking one change per function,
and any other reshaping an experiment needs.

## 4. Composition and comparison

Because datasets are values behind one contract:

- **Compare.** Every statistic computed from the contract — derived features, declared
  capabilities, outcome spread — is comparable across datasets, since the derived features are
  the same functions. Comparing datasets means iterating them and tabulating.
- **Pool.** Several datasets can be combined into one evaluation population. Test ids need
  namespacing, because two projects may both contain `tests/test_utils.py::test_x`;
  `dataset::nodeid` resolves it, provided a dataset name contains no `::`. What namespacing does
  *not* fix is meaning: pooling datasets whose `test_unit` (§2.5) differs is defensible only when
  the difference is immaterial, and is warned about rather than refused. The split needs no
  reconciling here, because it belongs to evaluation (§5).
- **Derive.** A dataset may be a function of others — a bundle, a filtered subset, a relabelled
  variant — so experimental manipulations are datasets too.

## 5. The evaluation contract

Evaluation owns the split and the averaging population, and needs from a dataset only a
quadruple:

    (scores, labels, candidates, rows) -> recall / hits

A dataset provides `labels`, `candidates` and its `rows`; a selector provides `scores`.
Evaluation therefore never depends on the dataset type, which is what lets a wrapped dataset, a
pooled dataset, or a bundle carrying its own label matrix be evaluated by exactly the same code.

Three sets are deliberately distinct, because conflating them loses information:

- **candidates** — per (change, test) pair: which pairs are rankable.
- **rows** — per change: which changes are in the evaluation window.
- **population** — a named subset of `rows` that a metric is averaged over.

Recall over all changes and recall over fault-bearing changes are different quantities, so the
population is part of the result rather than an implicit choice buried inside it. Fault-bearing
changes are the default, and are the population every number has silently used so far; study-
specific populations — no prior failure history, low pair recurrence — are passed in as named
predicates and recorded beside the metric.

A population is to `rows` what a derived feature is to pairs: a harness function over the
contract, defined once and applicable to any dataset. That constrains what a predicate may read.
"Has no prior failure history" is expressible from `killing_tests` and the order; "low pair
recurrence" needs `coverage`. A population therefore declares its requirements, exactly as a
derived feature does, and where a dataset cannot meet them the population is **unavailable** —
never silently empty (§6). This is the sense in which population restriction is independent of
the dataset: the dataset supplies the contract, and everything else is computed from it.

The split is a fraction, a shuffle flag and a seed, applied to the dataset's canonical order.
`shuffle=False` takes a contiguous prefix — a well-defined operation on any dataset, so it only
warns when the ordering is `imposed`. `shuffle=True` permutes first, and is permitted on any
dataset at the user's risk, because it discards the temporal reading that history features rest
on.

One rule ties the split to the ordering: **the effective ordering of a run is the dataset's
ordering, unless the split shuffles, in which case it is `imposed`.** Shuffling an `observed`
dataset therefore switches history features off by the §2.4 default, with no second switch to
forget.

That rule and the pooling rule in §4 are both **advisory**: the harness warns and records rather
than refusing. The purpose of a declaration is to make a mismatch visible in the result, not to
prevent an experiment that knows what it is doing.

## 6. Unmeasured values and warnings

Three rules apply to everything, not only to populations.

**Unmeasured is a state, not a value, and it propagates.** A quantity that cannot be defined —
rather than merely being zero or absent — stays *unmeasured*, and carries the reason it could not
be defined, naming the requirement that was not met. The reason follows the §2.5 pattern: a
checkable requirement id, plus a note. Anything computed from it is itself unmeasured, so a
meaningless number never reaches a table, a figure, or a comparison.

The failure mode to avoid is specific. An average over an empty set and an average over a
population that cannot exist are indistinguishable once both are `0.0`, and reporting the second
as the first is not a rounding error but a false statement about the data. Whether a quantity is
measured is therefore part of every quantity, rather than something recovered by inspecting it
afterwards.

**Information is lost only at a boundary, deliberately and once.** Unmeasured survives through
computation as long as it can, and conversion to whatever a consumer needs happens at the edge
where that consumer is served. The representation is therefore *not* one global choice between a
sentinel and absence — it is chosen per boundary, because the right answer depends on the
consumer.

- An artifact records it as `null` beside its reason, or omits the entry and lists it in an
  unmeasured registry; either way the reason survives.
- A table or figure shows it as unavailable on the side where it is unavailable, rather than
  dropping the row. A comparison is never silently halved: that a question was asked and could not
  be answered is itself a finding, and removing it destroys the finding.
- A model input, which usually cannot represent the state at all, is the last boundary and the only
  place a lossy coercion is legitimate. There it becomes something the model can ingest — and since
  zero is itself meaningful, the coercion is recorded, in practice as a companion indicator rather
  than a silent substitution.

**Warnings are structured and propagated**, whether or not anything currently acts on them. A
declaration that is only printed is invisible to the layers that should be able to act on it — the
figures layer, for instance, is exactly where a comparison the warnings call incomparable should
be declined. Recording a warning is not conditional on a consumer existing; anticipating one is
the reason to record it at all.

## 7. Testability

The contract makes a dataset cheap to fake: a fixture-backed dataset returning ten changes over
three tests is complete and valid, and two of them can exist at once. This requires that no
dataset state live at module level — no pinned paths, no mutable label source, no cache keyed by
a bare node id. Test-time datasets then need neither a real checkout nor a real test run.

## 8. Keeping the existing numbers intact

The documented numbers are the deliverable, so the refactor is verified against them rather than
trusted.

- Record the current `results_full.json` and `ladder.json` before starting; re-run and diff
  after each slice.
- Reproduce existing outputs exactly under existing defaults, so "unchanged" is checkable rather
  than asserted.
- Remove study-wide global state last, and deliberately. It is the highest-risk change: every
  selector, feature and evaluation must agree on it, which is why it was global to begin with.

## 9. Out of scope

The model half — selectors, their context object, feature-family ablation, model inputs — is
deferred. It is closer to a real interface already, and it can land independently.

## 10. Remaining open question

What is the *scope* of propagation — does an unmeasured input invalidate only the quantities that
read it, or the whole run? An undefined population plainly should not invalidate an unrelated
metric on the same dataset, but where that boundary sits is a choice.

## 11. What was implemented

Status: implemented on branch `refactor/dataset-contract`. This section records how the design
above was realised, and every place where implementing it changed the design.

**Modules.** The contract lives in `rts/dataset.py` -- decomposed by §14, which supersedes this
paragraph; the contract is now `rts/data/contract.py`: `TestUnit`, `Ordering`, `Capabilities`,
`Unmeasured`, `Warning`/`Warnings`, `Split`, `Population`, the `Dataset` ABC, the derived-feature
functions, the composition helpers (`pool`, `PooledDataset`, `namespace`) and the `describe`/`save`
boundaries. Sources are in `rts/data/sources.py` (`MutmutSource`, `BugsInPySource`); concrete datasets in
`rts/data/datasets.py` (`MarshmallowDataset`, `BugsInPyDataset`, `BundleDataset`, `DerivedDataset`).

**Derived features are inherited, not implemented per dataset.** `Dataset` supplies `labels`,
`ran`, `change_paths`, `covered`, `test_index`, `fault_idx`, `candidates`, `pair_counts`,
`sparse_mask`, `pair_history_counts`, `change_index`, the default `split`, `describe` and
`describe_starved` once, and `rts.dataset.structured_features` builds the model-input tensor from
the primitives. A concrete dataset implements only the seven primitives plus its declarations, so
the eight BugsInPy datasets are eight constructors over one source and one generator.

**Granularity.** The eight BugsInPy projects are eight datasets, and the arm's headline numbers come
from `pool(...)` of all eight — which is what makes the arm a *pooled population* rather than one
flat dataset. Because test ids are namespaced, the pooled per-project breakdown (§13) is free.

**Bundling is a dataset.** `BundleDataset` wraps a base dataset; `rts.bundles` keeps only the rung
definitions (a study choice) and reads everything else through the wrapper. Its `diff_text` is a
concatenation, which the design did not anticipate: the wrapper declares that, because it is a
genuine semantic divergence from "one valid unified diff" — line structure is preserved, so the size
features are exact, but the value is not itself a well-formed diff.

**The label source is a source property.** `config.LABELS`, `config.set_labels`,
`config.outcomes_file` and `artifacts.ALL_TEST_IDS` are gone. Agreement between selectors, features
and evaluation is now achieved by handing one `MutmutSource` to every consumer of a run, which
cannot disagree with itself and does not prevent a second source existing beside it. `config.SUT`
survives only as the *default* a constructor may use; `source.py` takes the checkout as a parameter
and caches per `(checkout, file)` rather than per bare node id.

**Two things the contract forced apart.** `change_key` was doing two jobs. `change_id` is a change's
stable identity and keys `change_index` *and* the score caches; `coverage_key` is what a coverage map
is indexed by. For mutmut these differ (`marshmallow.utils.x_f` vs
`marshmallow.utils.x_f__mutmut_1`) and conflating them mis-maps every lookup. Separately, the generic
`describe()` was reading `change.killed`, a mutmut-only attribute — the fixture dataset in
`tests/stub_dataset.py` has a `killing` field and exposed it immediately. Source-specific counts now
come from a `source_counts()` hook, so `describe()` cannot become source-specific by accident.

**§10's open question, answered in the one place it was forced.** Propagation is scoped to the
quantities that read the unmeasured input. An unavailable population produces an `Unmeasured`
`Evaluation` carrying the unmet requirement, and `evaluate()` raises `UnmeasuredPopulation` rather
than returning an empty average; every other metric on the same dataset is unaffected. This is the
narrow reading, and it is the one that keeps "a question was asked and could not be answered" a
finding rather than a run-level failure.

## 12. Verification

The documented numbers are the deliverable, so the refactor was verified against them rather than
trusted. `artifacts/results_full.json` and `artifacts/ladder.json` were copied aside before any edit
and diffed field by field afterwards; `artifacts/bugsinpy_results.json` and
`artifacts/variations.json` likewise.

`results_full.json` — every arm present in both files is **identical**, including the bootstrap
intervals. The regenerated report has three arms the recorded file lacked
(`xgboost_static_nocov`, `xgboost_static_nocov_lex`, `semif_reranker`). That is **pre-existing
drift**, not a regression: `git show HEAD:artifacts/results_full.json` lacks them too, so the
artifact was written before `models.default_selectors` gained those arms. Because regenerating it is
the only way to diff the arms at all, the refreshed artifact is committed alongside the refactor,
with the three new arms and the four new top-level keys (`labels`, `population`,
`dataset_declaration`, `warnings`) called out rather than absorbed silently.

`results_covered.json` — same shape of result as `results_full.json`: every arm present in both is
**identical**, and the regenerated report adds the same three selector arms. This is the arm where
the `covered` mask makes `covers_function` constant, so it is the one place the feature tensor is
deliberately degenerate; that it reproduces exactly is the check that the capability refactor did not
quietly change what "coverage is present" means.

`ladder.json` — **exact**. Zero value differences across both label sources, all four rungs, both
populations, every selector and every SemIf margin. The only differences are three additive keys per
label source: `dataset_declaration`, `warnings`, and `populations_unmeasured`, the last of which is
empty here because the SemIf cache the `starved141` population depends on is present. The ladder is
the arm this design was most likely to disturb, since it zeroes feature *families* by column name and
those names are now owned by the contract rather than by `rts.features`.

`bugsinpy_results.json` — **exact**, with two additive keys: `dataset_declaration` and
`per_project_recall_at_0.05`. Adding the second is only possible because the eight projects became
eight datasets; the pooled SemIf-vs-BM25 verdict (0.211 vs 0.225 at b0.05, p=0.854) is unchanged, and
the breakdown shows the pooled tie holds per project rather than being carried by one. Re-deriving
the T0 bridge audit through the new datasets also reproduces it exactly (87.3% share a token, 2.66 vs
2.03 mean shared tokens).

Verification found two defects in the refactor itself, both invisible without diffing:

* the `random` baseline was drawn over the whole 71×3000 matrix instead of per bug over its own pool,
  which moved every random-baseline number and, through them, the SemIf-vs-random comparison. RNG
  *consumption pattern* is part of a recorded number; a rewrite that changes it changes the result;
* `mean_k` was reported as `BudgetResult.k`, which is rounded, instead of as the mean of the
  per-change counts.

Both are fixed, and both are the reason §8 asks for a diff rather than an assertion.

## 13. Tests, and what remains

`tests/` holds 36 tests in two files. `tests/test_dataset_contract.py` pins the *contract* against a
fixture-backed dataset (four changes over three tests, one flag per capability) — that a dataset is
cheap to fake, that two coexist with different declarations, that an absent capability raises rather
than returning zeros and simultaneously marks its columns unmeasured, that history is off by default
on an imposed order and on by default on an observed one, that an unavailable population is
unmeasured rather than empty, that pooling namespaces and warns on mixed `test_unit`, and that a
bundle's derived `change_size` equals the sum over its members.
`tests/test_marshmallow_reproduction.py` checks the arm against the recorded artifact, including that
`describe()` still equals `results_full.json["dataset"]` exactly.

Known gaps, stated rather than implied:

* The **model half is still out of scope** (§9), so `models.Context` still carries a
  pre-materialised `X` rather than requesting named derived features. Selectors now receive the
  dataset's warnings with the features, which is the minimum the §6 argument requires of them.
* **Durations are declared but not otherwise policed.** §2.3's comparability caveat (a duration is a
  property of the machine as much as of the test) is recorded in `MarshmallowDataset.semantics` and
  nowhere enforced.
* `rts.render.variations`, `rts.analysis`, `rts.bundles` and `rts.model.semif_runner` were migrated for
  construction and primitives only; their experiment logic, and therefore their numbers, were left
  alone. None of them is re-verified here beyond importing, because each needs a GPU arm or a
  multi-hour cache to re-run in full.
* The **`starved` population and the ladder's `starved141`** are declared, and the latter is now
  unmeasured rather than empty if its SemIf cache is absent — but §5's population vocabulary is only
  as complete as the populations actually written down. `no_prior_failure`, `starved` and
  `low_pair_recurrence` are the three that exist.

## 14. Second pass: decomposition

§11–13 implemented the contract but left it in one 1357-line module, with feature assembly, the
population vocabulary and evaluation configuration inside it. A review of that result found the
separation-of-concerns problem and a set of specific defects. This section records what changed and,
where a proposal was declined, why.

### The layout

| module | lines | responsibility |
|---|---|---|
| `data/contract.py` | 465 | primitives, declarations, `Capability`/`Requirement`/`Policy`, `Unmeasured`, `Warnings` |
| `data/accessors.py` | 393 | derived accessors as free functions, and the single `MATERIAL` catalogue |
| `data/splits.py` | 120 | `Split` and `make_split` — evaluation configuration |
| `features/block.py` | 375 | `FeatureBlock`, `FeatureGroup`, `FeatureMatrix` |
| `features/derived.py` | 269 | one function per quantity, pure over its input |
| `features/structured.py` | 184 | the 15-column block, declared as data |
| `features/bundle.py` | 181 | the bundle block, declared as data |
| `features/text.py` | 173 | tokenizer and BM25 |
| `data/populations.py` | 266 | populations, predicates, registry value |
| `data/composition.py` | 236 | namespacing, pooling, derived datasets |
| `data/reporting.py` | 174 | audit, describe, save |
| `data/datasets.py` | 522 | the concrete datasets |

The table names the modules as they stand after the later move into packages (see the
review): the dataset half is `rts/data/`, the model half `rts/model/`, and the drivers
`rts/render/`.

`rts/dataset.py` and `rts/features.py` are gone; the one-letter `dataset`/`datasets` hazard the first
pass introduced is gone with them.

### The division that mattered most

A **derived feature** is one quantity computed from the contract. A **feature block** is an ordered,
named *selection* of them assembled into a tensor for a model input. Colocating those is why the first
pass could not add a feature without editing an assembly function that also had to know about
capability gating, history policy and column order. They are now separate, and adding a column is:
add the computation to `features/derived.py`, add a `FeatureColumn` to a group in the block that wants
it. `tests/test_extensions.py` does exactly that and asserts the result, without touching `rts/`.

`FeatureBlock.build` returns a `FeatureMatrix` rather than a bare `(X, names)` tuple. That retired nine
separate `{n: i for i, n in enumerate(names)}` reconstructions across the package and made a renamed
column fail loudly instead of reading its neighbour.

### Gating is derived, not asserted

A group or a population names the *material* it reads (`needs=("coverage",)`);
`accessors.MATERIAL` — one catalogue, not the three parallel spellings the first pass had — maps that
onto requirements. So a requirement set cannot disagree with the code that reads it. Declaring too
little used to raise a capability error from inside a predicate instead of reporting unmeasured;
declaring too much reported a working computation unavailable. Both are now unrepresentable.

Where a caller supplies material the contract does not know about — the bundle block's members and its
base's features — the block declares it as external and raises if it is not supplied, because a
caller-supplied material is a promise rather than a capability.

### Declined: decorator registration

A proposal to register features and populations with `@feature`/`@population` decorators was declined.
The reasons, in order of weight: a decorator registry is module-level state, which is what the first
pass spent its effort removing, and it makes availability a property of the import graph so it cannot
express "dataset A has this, dataset B does not" — the case that needs it least and the one the
contract exists to express. Column order would become import order. The metadata a column needs (unit,
capability gate, dtype, broadcast rule, family) does not fit an annotation. And the proposal does not
address the defect that motivated it, since a decorated predicate still hand-writes its requirements.

What the proposal wanted was extension ergonomics, and that is delivered: an explicit tuple of columns
in the block's own module is one line more than a decorator and has none of those costs. A name→
implementation registry remains legitimate at the *experiment* layer, and exists there as
`PopulationRegistry`, an immutable value that composes with `+`.

### Defects fixed

* **Warnings were attached to the wrong object.** `structured_features` mutated `ds.warnings`, so a
  dataset accumulated a record of how a harness function had been *called*. Warnings are now returned
  by the computation that produced them (`FeatureMatrix.warnings`) or derived by
  `reporting.audit(ds, split)`, and `Dataset` has no `warnings` attribute at all. The symptom was a
  test whose assertion was `... or True` — a tautology that passed unconditionally; it is gone, and
  `test_warnings_are_returned_not_stored` asserts the absence of the attribute instead.
* **`describe` and `save` baked in a default split**, publishing evaluation configuration as dataset
  metadata and putting it in the recorded artifacts. Both now require a `Split`. That is why
  `results_full.json` gains a `split` key.
* **`no_prior_failure` and `starved_mask`** were two implementations of one predicate that had already
  drifted (only one supported `max_runs`). There is one implementation, with a declared population and
  a parameterised helper as two entry points onto it.
* **Nine copies of the memoisation idiom**, one of which (`PooledDataset`) omitted all of them, so
  inherited `n_changes` rebuilt the entire change list to take a length. There is now one
  `Dataset.cached` mechanism, and the pooled dataset uses it.
* **`PooledDataset` keyed its change→owner map by `id(change)`**, which worked only because the map
  held strong references. Pooled changes are now `PooledChange` values carrying their dataset and id,
  so identity is namespaced and stable — two projects with a bug numbered `3` cannot be confused.
* **`labels`, `runs` and `fault_mask` handed out mutable internals.** They are frozen on construction.
* **An expensive `test_source` was read twice per test.** `accessors.test_source` memoises per dataset
  instance, which is the opposite of the module-level node-id cache the design forbids.
* **`REQ_ORDERING_OBSERVED` carried three meanings.** A `Policy` vocabulary now separates the
  dataset's order (`OBSERVED_ORDER`) from the run's (`EFFECTIVE_ORDER`), so a shuffle warning no longer
  claims the dataset lacks a real order.
* **A second feature generator** lived in `rts/bundles.py`, invisible to the structured block and
  coupled to it by string lookups. It is `features/bundle.py` now, declared like any other block, and
  `bundles.bundle_arrays` is a thin wrapper over it.
* **Six back-compat shims** were deleted.
* **The ladder ablated by comparing column names against a list in its own module.** A rename would
  have silently stopped ablating anything, with no error and a plausible number. A rung is now the
  block with a family withheld, which takes the same unmeasured path a genuinely absent capability
  takes and reports what it withheld.

### The ablation bug this exposed

The prediction above was not hypothetical. `TRACEABILITY_FEATURES` named
``("filename_stem_match", "path_distance", "n_tests_in_file")``, and ``n_tests_in_file`` **is not a
column** -- the real name is ``n_tests_in_test_file``. A name-based ablation ignores a name that
matches nothing, so the L3 rung zeroed two of the three traceability columns and kept
``n_tests_in_test_file`` live while reporting itself as traceability-free. The comment above that
tuple in the old ``models.py`` even documents this hazard for a *different* name, which is what makes
it a good example of the class of bug rather than an isolated typo.

Under the historical `mutmut` labels the leaked column was worth up to **+0.042 recall at b0.10** to
the strongest tree (`starved141` L3 `xgboost_struct_lex`: 0.645 → 0.603), and +0.035 at b0.05
(0.496 → 0.461). The corrected `full`-label figures are in ``implementation.md`` §12.2.

Three things follow, and they are the argument for the design rather than an aside:

1. **The ablation was silently wrong for as long as it existed**, and no test noticed because nothing
   checked that a family name matched a column. Withholding is strict now: the same typo raises.
2. **The numbers it produced were plausible.** L3 looked like a clean monotone degradation, which is
   exactly why "L2 == L3, so coverage is the whole effect" survived review. A plausible number from a
   broken ablation is worse than an error.
3. `tests/test_extensions.py` now pins the traceability family to its three real columns and asserts
   that each rung withholds exactly the columns its families name.

Note that this is a change to a *documented* number, and the only one the decomposition produced:
every other ladder rung, both pipeline artifacts and the BugsInPy artifact reproduce exactly. The
corrected L3 rows are recorded in ``implementation.md`` §12.2.


### Verification

`ladder.json` is **exact**: zero value differences across both label sources, all four rungs, both
populations, every selector and every SemIf margin. `results_full.json` and `results_covered.json` are
**exact** over every section present in both. `bugsinpy_results.json` is **byte-identical**, including
the T0 audit gate (87.3% share a token, 2.66 vs 2.03). 66 tests pass.

The only artifact differences are additive keys (`split`, `audit`) and one renamed warning scope
(`structured_features` → `features:history`), each of which is the intended consequence of a fix
above.


