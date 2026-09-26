# The experiment layer: a declared sweep, and the run that measures it

Design doc and implementation plan. Companions: `plan.md` (study design),
`refactor.md` (the dataset contract), `plan_next_steps.md` (the agenda),
`implementation.md` (measured results).

Scope is **the experiment layer only** — what an experiment *is*, how its dimensions are
declared, and what running one produces. The dataset half already exists (`refactor.md`);
this document does not revisit it. Status: implemented on branch `feature/experiment-layer`;
verification in §11.

## Purpose

The goal is to be able to write, in a Python module, a description of an experiment —

```python
# studies.py

def study_arm(label="mutmut", *, candidates="full") -> Experiment:
    return Experiment(
        name=f"study.{label}.{candidates}",
        datasets=dataset_axis((label,)),
        features=structured_feature_axis(),
        models=model_axis(),                       # .without("semif_reranker") for CPU only
        populations=population_axis(("fault_bearing",)),
        splits=split_axis(),
        knobs=Knobs(candidates=candidates),
        comparisons=(Comparison(ROLE_MODEL, "coverage", 0.05),),
        history=True,
    )


# my_experiment.py
report = run(study_arm(), out_dir="artifacts/study")
```

— and get back the tables, without writing a driver. Today that is not possible: the sweep
exists, but it exists **as control flow in five hand-written drivers** rather than as data.

Axis builders are functions rather than module constants wherever the elements hold state, so
that two runs cannot share one trained model object. The `Experiment` value itself is immutable
and safe to hold as a constant.

## 1. What is already declarative, and what is not

The refactor made three of the four things an experiment sweeps over into plain values with
declarations, and it did so with a consistent idiom. It did not build the layer that composes
them. The current position:

| role | is it a value? | does it declare availability? |
|---|---|---|
| dataset | yes — `Dataset`, with `capabilities()`/`ordering()`/`semantics()` | yes, `available_requirements()` |
| feature set | yes — `FeatureBlock`, declared as data | yes — `needs=` → derived requirements |
| population | yes — `Population`, a value | yes — `needs=` → `unavailable(ds)` |
| split | yes — `Split` | n/a |
| **model** | yes — `Selector` | **no: `Selector` declares only a name** |

So the missing half is the *model* declaration, which `refactor.md` §9 deferred and §13
records as a known gap. Everything else is composition.

The sweep itself lives in `pipeline.run`, `ladder.run_label_source`, `variations` (six
sub-experiments), `bugsinpy.main`, `analysis.run` and `figures.run`. Each one re-implements
the same sequence — build dataset, make split, build context, enumerate selectors, loop over
populations/rungs/variants, render a table, write a bespoke payload — and each keeps the
study's choices as its own module constants (`pipeline.HISTORY`, `pipeline.POPULATION`,
`ladder.RUNGS`, `ladder.BUDGETS`, `variations.N_BOOTSTRAP`, `models.default_selectors`).

**The clearest symptom.** "A named variant of one input to the measurement" exists three
times, spelled three ways:

* `ladder.RUNGS` — feature families withheld;
* `bundles.make_bundles` + `datasets.bundles` — bundle rungs;
* `semif_runner.INSTRUCTION_VARIANTS` — prompt wording.

Each grew its own cache keying, its own population and its own table rendering. Three
implementations of one concept is the argument that the axis abstraction is not speculative.

## 2. The five roles

An experiment is a sweep over **five roles**, and a genuinely new role is a kernel change:

    dataset, features, model, population, split

**Why not an open list of axes.** An arbitrary axis would have to say what it feeds, and the
kernel would have to know how to consume it; without that, an "axis" is a bag of values whose
meaning lives in the driver — which is the problem being solved. Naming the five roles fixes
the meaning of each and makes the kernel a fixed pipeline. The extension story is then honest:
**new elements are cheap and are what most new dimensions actually are; new roles are a
deliberate kernel change.**

That distinction is not a limitation in practice, because every factor the study has
conceived already reduces to one of the five:

| factor people reach for | role it actually is |
|---|---|
| rung / feature ablation | `features` — a `FeatureBlock` with families withheld |
| bundle rung | `dataset` — a dataset derived from a base |
| instruction variant (P2) | `model` — a `SemIfSelector` reading a different cache |
| direct mode (P1), embedding baseline (P3) | `model` — a selector reading a different cache |
| candidate pool mode | a knob (§4) |
| starvation threshold | a `population` |
| label source | a `dataset` |

## 3. Axis and Element

An **axis** is a named, ordered set of variants of one input. An **element** is one variant.

```python
Axis("model", (
    constant("coverage", models.CoverageSelector(), tier="cpu", estimated_seconds=0.1),
    constant("semif_reranker", models.SemIfSelector(), tier="cache", estimated_seconds=0.0),
))
```

`Element.make` is `Callable[[Binding], Any]` — arbitrary Python, so an element can close over
whatever it needs, including constructing a derived dataset:

```python
constant("bundles_r2", datasets.bundles(BASE, rung=2, seed=cfg.SEED))
```

No special machinery is needed for "a dataset that is a function of another dataset", because
`make` is already a function. The idiomatic form hoists the base to a module constant so it is
built once and shared.

**Two authorities, deliberately separated.**

* The **element** declares its name, its cost tier and its estimated seconds. These are
  presentation and scheduling facts about *this run's* use of a thing.
* The **value** declares its requirements — `Selector.requirements()`,
  `FeatureGroup.needs`, `Population.needs`, `Dataset.capabilities()`. These are facts about
  the thing itself, and they must live where the material is read, for the reason
  `accessors.MATERIAL` exists: a need written beside the code that reads it cannot drift from
  it.

`Axis.map(fn)` applies a **shared option** to every element of an axis — the thing a
dimension-level option actually is. `fn` returns an `Element`, or an `Unmeasured` when the
option cannot be expressed for that element; the inapplicable element is then *kept* as a
poisoned element rather than dropped, so the cell it would have produced is reported
unmeasured instead of silently vanishing. `XGBoostSelector` can express "exclude the history
family"; `SemIfSelector` cannot, because it reads text pairs rather than columns — and that
difference should show up as a finding in the report, not as a missing row.

**A third thing an element may declare: applicability.** Availability is normally asked of one
role's value at a time, which cannot express a variant that is only meaningful in combination
with a particular element of *another* role. That case is real and was found while wiring the
ladder: its SemIf cache was built over the changes `starved141` selects, and
`semif.load_scores` fills every *uncached* pair with a sentinel rather than reporting that it
has no score — so evaluating SemIf on the other population produces a number that looks like a
measurement and is not. `ladder.py` avoided this by not tabulating SemIf there;
`Element.applies` declares it, which keeps the cell in the report as a finding with a reason.
The parameter is a `Binding` carrying the cell's factor names, so the predicate reads the other
role's element rather than reaching for it. Those names are populated **only** for this check: a
value is built once per run and shared across every cell that selects it, so an element whose
`make` read them would silently be reused for cells it does not describe. Keeping the two
bindings distinct makes that a mechanical guarantee rather than a convention.

## 4. Knobs, and the rule about what may not be configured

Only roles multiply into cells. Everything else is a **knob**: one value for the whole run,
recorded once.

```python
Knobs(seed=..., budgets=..., n_bootstrap=..., candidates="full", n_bootstrap_paired=None)
```

`n_bootstrap_paired` exists because a table's confidence interval and a paired test's p-value
are two quantities with two precision needs, and the recorded ladder used different counts for
them (1000 and 2000). Leaving it `None` means "the same as `n_bootstrap`". `Knobs` otherwise
holds only what is free.

A knob is not merely "something global". It is specifically **a choice that is free** — the
harness cannot derive the right answer from anything else. The rule that follows, and it is
the one design rule this layer adds:

> **Anything the harness can derive must not be an option.**

`history` is the worked example. Whether the cumulative history columns are present is not a
free choice: it is implied by the dataset's `ordering()` and by whether the split shuffles
(`splits.Split.effective_ordering`), and the harness already *warns* when a caller overrides
it. So `Experiment.history` exists as `None | bool`, is named an **override** rather than an
option, defaults to `None` (derive it), and when set produces the existing
`feature.history_on_imposed_order` warning which travels with the cell. Stating it as an
ordinary option would make the default silent and the override invisible.

The same rule rules out the two placements this design was originally asked about:

* **features on the dataset** would force every model in a run to share one feature set, which
  is exactly what the ladder must not do — it holds the dataset fixed and varies features.
* **features on the model** would make a model responsible for a block's capability gating,
  which the block already derives from the dataset's declarations.

The feature block *is* the interface between the two, so it is its own role. `XGBoostSelector`
already reads this way: it selects columns from a block it is handed (`exclude=...`); it does
not define one. `SemIfSelector.scores` ignores `ctx.features` entirely, which is the proof
that the role is independent.

## 5. Cells, and unmeasured cells

A **cell** is one point in the product over the five axes plus the run's knobs. Its key is the
tuple of element names, so it is stable, ordered, and readable.

The product is **not** materialised blindly. Each cell is either measured or carries an
`Unmeasured` naming the requirement that stopped it, and an unmeasured cell is **reported**,
never dropped. This is the load-bearing difference from `itertools.product`: dropping is what
turns "we asked and could not answer" into a silently halved comparison, and §6 of
`refactor.md` exists to prevent exactly that.

Five distinct things produce an unmeasured cell, and they are different findings:

1. **element unavailable** — an element that cannot be built (a poisoned element from §3);
2. **population unavailable** — the dataset cannot support it (`Population.unavailable`);
3. **model unavailable** — a requirement of the selector's cannot be resolved (§6);
4. **tier not enabled** — the cell's cost tier was excluded from this run (§8).
5. **applicability** — the element declares it does not apply to this cell, which is how an
   interaction between two roles is expressed (§3).

An *unmeasured column* is a fifth and different thing: a feature block whose group the dataset
cannot support is measured-but-zeroed, and the `FeatureMatrix` audit travels in the cell
record. Rungs of the ladder rely on this, so a block's requirements must not gate the cell.

## 6. Availability

Requirements are strings. Two kinds, and the resolution happens in exactly one place:

* `artifact:<path>` — a file the element reads (a score cache). Resolved by existence.
* anything else — a `contract.Requirement` value (`coverage`, `durations`, `labels`,
  `diff_text`), resolved against the cell's dataset.

An unrecognised string raises, for the same reason `has_capability("coverge")` raises: the
vocabulary is closed and a quiet `False` would gate a whole block off.

`Selector.requirements()` is the new half. `SemIfSelector` returns
`("artifact:artifacts/semif_scores.jsonl",)`; `XGBoostSelector` returns one per
`extra_score_files` entry; the baselines return `()`. A missing cache therefore becomes an
unmeasured cell carrying the path, rather than a `FileNotFoundError` string swallowed by the
driver's `except` (which is what `pipeline` does today, into a `skipped` dict).

## 7. Comparability

A comparison is a paired delta against a reference element of one role, at a probe budget.
Pairing is only meaningful between cells that share rows, population, split, candidates and
labels, so the roles split:

* **pairable** — `model`, `features`. Varying these changes scores, not the quantity being
  averaged.
* **row-changing** — `dataset`, `population`, `split`. A delta across these compares different
  populations and is not a paired quantity.

`Comparison(role, reference, probe_budget)` therefore **refuses a row-changing role at
construction** rather than producing a plausible number. Grouping is automatic: cells are
grouped by the other four roles, and each cell is paired against the reference element's cell
in its own group. A group whose reference cell is unmeasured is recorded with the reason.

This is what replaces `pipeline`'s single hard-coded reference and the `starved141` population
in `ladder.py` — which exists *solely* to make pairing possible, and is a workaround for this
not having been answered once at the layer above.

## 8. Cost tiers and execution order

An element declares a `tier`; cells execute in `Experiment.tier_order` (default
`("cpu", "gpu")`) and `run(..., tiers=("cpu",))` excludes the rest as unmeasured rather than
running them.

**The vocabulary is deliberately coarse.** It says whether measuring a cell needs a GPU, which
is the one cost distinction this study acts on — the CPU-only traceability ladder before gated
expensive work. It is not a cost model: `estimated_seconds` carries the finer number where an
element has one, and now that these are recorded per cell, cost sits next to the result, which
is the direction `plan.md` already commits to (a cost-effectiveness curve rather than
recall@budget).

A cell's tier is its most expensive element's, so a cell that reads a precomputed score cache
but still assembles features and sweeps metrics is `cpu` — which is correct: the cache makes
the *model* cheap, not the cell. An earlier revision had a `cache` tier as well; it was
removed because no cell could ever be it, since feature assembly and evaluation always compute.

No scheduler is built. Ordering, filtering and recording are enough to answer "what is cheap
here" without inventing a cost model whose units would not be comparable across datasets.

## 9. The artifact

One uniform payload per run, rather than a bespoke dict per driver:

    experiment, note, environment (knobs, out_dir, caches, shared),
    axes (element declarations, including whether each declares applicability),
    n_cells, seconds,
    cells   [ key, factors, tier, estimated_seconds, measured: true,
              selector, population, n_rows, n_changes, split, dataset_declaration,
              features (FeatureMatrix.audit), warnings, results, seconds, importances ]
    unmeasured [ key, factors, tier, estimated_seconds, measured: false,
                 unmeasured: true, requirement, note ]
    comparisons [ role, reference, probe_budget, group, cell, reference_cell, delta, lo, hi, p_value, n ]

`cells` carries the measured ones and `unmeasured` the rest, so the holes are as visible as the
numbers. The `dataset_declaration`, `split` and `features` entries are per cell rather than per
run, because in a sweep they genuinely vary — the ladder's rungs differ in exactly that field.
`importances` is populated only on the cell that actually scored, for the reason in §10.
field.

## 10. Declined

* **An external config language (YAML/TOML).** Elements are objects with behaviour *and*
  declarations, so a config must name and construct Python objects. A data format would need a
  mirrored registry plus a schema and would still need Python for any new element. Python is
  also what `plan.md` and the report already read.
* **A blind `itertools.product`.** It discards §5's unmeasured cells and §7's comparability.
* **A decorator or module-level registry.** Declined in `refactor.md` §14 and still declined,
  for the reasons given there (module-level state; availability becomes a property of the
  import graph; column order becomes import order). `Axis` is a value, like
  `PopulationRegistry`, and a study's axes are a module of values, like `populations.STUDY`.
* **An open list of axes.** §2.
* **Reusing the elements' objects across cells.** The run materialises each element once and
  shares the value, which is what makes a 48-cell grid affordable (the dataset is built once).
  It is sound because these are values (`refactor.md` §2). The same reasoning extends to scores:
  a score matrix is a function of the *context* — dataset, features, model, split — so it is
  memoised by that key, and `population` is deliberately not part of it, because a population
  restricts which rows a metric averages rather than which pairs get scored. Without that, every
  population sweep would pay for a second identical training run.
* **Reading importances back off a selector after the run.** A stateful selector's
  `importances_` describes its most recent call, and the element is shared, so a later cell in
  another context would have overwritten it. They are captured on the cell that actually scored
  and are empty on cells that reused a matrix.
* **A named-artifact indirection in `Environment.caches`.** The first sketch let a requirement
  be written `artifact:<name>` and resolved through a run-supplied alias map, so a run could
  move where artifacts are read from. It was cut back to a plain path→path redirect: an element
  that reads a file needs a real path anyway, so the alias only ever bought a level of
  indirection between a selector and the cache it was constructed with. `Environment.shared`
  covers the injection need — a stub dataset in a test, one base dataset shared by several
  derived elements — without a second mechanism.
* **A separate `Comparison.only` filter for "this comparison is defined only on this
  population".** It was the first attempt at the SemIf-cache problem above. `Element.applies`
  subsumes it and is more general, and the comparison machinery then needs no special case at
  all: a group whose reference cell is unmeasured is recorded with the reason.

## 11. Verification

The documented numbers are the deliverable, so the layer is checked against them rather than
trusted. `scripts/verify_experiment_layer.py` runs the arms through the layer and compares field
by field — exactly where the recorded artifact is unrounded, and at the artifact's own rounding
where it is rounded (the ladder rounds recall to 4 dp, the study arm records full precision).

**The study arm is exact.** `studies.study_arm()` against `artifacts/results_full.json`: 651
comparisons — 12 selectors × 6 budgets × 9 fields (`budget`, `k`, `recall`, `precision`,
`f_measure`, `suite_reduction`, both interval bounds, `n_faults`), plus the split fields — with
**0 mismatches**. That the `random` row reproduces exactly is the load-bearing part: `refactor.md`
§12 records a rewrite that moved every random-baseline number by changing the RNG *consumption
pattern*, so an exact random row is evidence that the layer consumes the RNG identically rather
than merely closely.

**Both ladder arms are exact at the artifact's precision.** `studies.ladder_arm(label)` against
`artifacts/ladder.json[label]` for `mutmut` and `full`: recall at 4 dp, `k`, `n_faults` and the
`withheld` column list for every selector in every population in every rung; the paired deltas,
their intervals and their p-values for every baseline against SemIf; and the headline SemIf
margins with their argmax.

Together: **2613 comparisons, 0 mismatches**, and a non-zero exit status if that ever changes. The
study arm contributes 651 of them and the two ladder arms the remaining 1962. Each ladder run
measures 68 of its 72 cells in 157 s (`mutmut`) and 344 s (`full`).

The four unmeasured cells in each ladder run are exactly SemIf against the population its cache
does not cover — the `Element.applies` case from §3 — and each carries the reason in the report
rather than being silently absent, which is what `ladder.py` did.

That the ladder verified is the stronger result, because it exercises what the study arm cannot:
four feature blocks in one run (so the matrix cache is keyed correctly), two populations per rung
(so a population is a factor and not a constant), a comparison whose reference is unavailable for
one of its two populations, and the split between the table intervals' resample count and the
paired test's.

**Not covered here.** `results_full.json`'s `ablations`, `sparse_arm` and `recurrence` sections are
BM25-shuffle probes and the sparse population; the layer can express both — a `LexicalSelector`
variant and a `population_axis` entry — and they belong to the driver migration in §13 rather than
to a first pass. `results_covered.json` is likewise declared (`ARMS["study.covered"]`) but only
the `full` variant is verified.

## 12. What was implemented

**Modules.**

| module | responsibility |
|---|---|
| `rts/experiment.py` | `Axis`, `Element`, `Knobs`, `Environment`, `Binding`, `Cell`, `Comparison`, `Experiment`, `run`, `RunReport` |
| `rts/studies.py` | the study's axes as builders, the named arms, and `semif_margins` as a *reading* of a report |
| `scripts/verify_experiment_layer.py` | the reproduction check of §11 |
| `tests/test_experiment.py` | 20 tests, all over the stub dataset |

`rts/models.py` gained one method — `Selector.requirements()` — which is the whole of the
model-half declaration §1 identified as missing. The only other change to an existing module is
removing a parameter from `ladder.build_populations` that its body never read.

**Decisions implementation forced**, as opposed to the ones §1–§10 anticipated:

1. **The paired set is the population's rows, not the evaluation window.** The first version
   computed per-change hits over `split.test_idx`, which for the ladder's `starved141` group
   would have paired 464 changes where the metric was averaged over 141. A comparison pairs two
   cells inside one group, and the group names a population, so hits are computed over exactly
   the rows the group's metric used.
2. **`Element.applies`, for an interaction between two roles.** §3 has it: the ladder's SemIf
   cache covers one population, and `semif.load_scores` gives the other a sentinel rather than
   saying it has nothing. This was not foreseen; it was found while wiring a real arm, which is
   the argument for wiring real arms during the build rather than after it.
3. **`Knobs.n_bootstrap_paired`.** The recorded ladder used 1000 resamples for its table
   intervals and 2000 for its paired p-values. Two quantities, two precision needs; making them
   one knob would have made exact reproduction impossible and would have hidden the choice.
4. **The `cache` cost tier was removed.** A cell's tier is its most expensive element's, and
   feature assembly and metric evaluation always compute — so no cell could ever have been
   `cache`. The vocabulary is now `("cpu", "gpu")`, which is the distinction the study acts on.
5. **`Environment.caches` was demoted to a path→path redirect**, and `Environment.shared` is the
   injection seam. The alias-map version (§10) only ever inserted a layer between a selector and
   the cache it was constructed with.
6. **Derived datasets needed no machinery at all.** `make` is arbitrary Python, so a bundle rung
   is an element whose builder constructs its base; `_bundle_dataset` in `studies.py` is the
   whole implementation. The one discipline is that a builder reading the run's seed does so
   through the binding — `order_seed=b.knobs.seed` — rather than closing over `config.SEED`,
   because otherwise a knob override would silently not reach the dataset.
7. **Elements are materialised once per run and shared across cells**, which is what makes the
   12-cell study arm ~110 s rather than twelve dataset builds. Sound because these are values.

**One hazard checked deliberately.** `refactor.md` §12 records that a rewrite moved every
`random`-baseline number by changing the RNG *consumption pattern*. The layer does not touch that
pattern: it calls the same `accessors.candidates` once per dataset element and the same
`evaluate.evaluate_rows` with the same arguments, so the random baseline reproduces exactly
(§11) rather than approximately.

## 13. What remains

* **Migrating the drivers.** `pipeline`, `ladder`, `variations`, `bundles`, `bugsinpy` and
  `analysis` should become `Experiment` values once §11 passes, and `figures` should read the
  uniform payload. `studies.py` currently restates the ladder's selector list and rung
  handling, which is duplication that exists only until those drivers go.
* **`results_covered.json`.** The layer's `Knobs.candidates` covers it, and the arm is declared
  (`ARMS["study.covered"]`), but only the `full` variant is verified so far.
* **Producing score caches is still outside the layer.** A `semif_runner` GPU arm is a
  *precondition* of a model element, not a cell. Modelling production as well as consumption
  would be a second feature (a build graph over artifacts); the layer currently treats caches
  as given, which is what they are.
* **`test_unit`/semantics checks across a pooled dataset** are declared but not policed
  (`refactor.md` §13); the experiment layer does not change that.
* **Cost is recorded, not modelled.** No budget-limited execution.
* **`splits.in_window` is not enforced.** A population's rows are taken from the split's
  evaluation window by construction, but a custom population predicate that named rows outside
  it would not be caught. Worth a check if a population is ever built from something other than
  the held-out set.
