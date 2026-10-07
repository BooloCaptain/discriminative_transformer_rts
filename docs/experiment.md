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

def study_condition(label="mutmut", *, candidates="full") -> Experiment:
    return Experiment(
        name=f"study.{label}.{candidates}",
        datasets=dataset_axis((label,)),
        features=structured_feature_axis(),
        models=model_axis(),                       # .without("semif_reranker") for CPU only
        averaging subsets=subset_axis(("detectable",)),
        splits=split_axis(),
        control variables=Controls(candidates=candidates),
        contrasts=(Contrast(ROLE_MODEL, "coverage", 0.05),),
        history=True,
    )


# my_experiment.py
report = run(study_condition(), out_dir="artifacts/study")
```

— and get back the tables, without writing a imperative runner. Today that is not possible: the sweep
exists, but it exists **as control flow in five hand-written drivers** rather than as data.

Factor builders are functions rather than module constants wherever the levels hold state, so
that two runs cannot share one trained model object. The `Experiment` value itself is immutable
and safe to hold as a constant.

## 1. What is already declarative, and what is not

The refactor made three of the four things an experiment sweeps over into plain values with
declarations, and it did so with a consistent idiom. It did not build the layer that composes
them. The current position:

| role | is it a value? | does it declare availability? |
|---|---|---|
| dataset | yes — `Dataset`, with `capabilities()`/`ordering()`/`annotations()` | yes, `available_requirements()` |
| feature set | yes — `FeatureBlock`, declared as data | yes — `needs=` → derived requirements |
| averaging subset | yes — `Averaging subset`, a value | yes — `needs=` → `unavailable(ds)` |
| split | yes — `Split` | n/a |
| **model** | yes — `Ranker` | **no: `Ranker` declares only a name** |

So the missing half is the *model* declaration, which `refactor.md` §9 deferred and §13
records as a known gap. Everything else is composition.

The sweep itself lives in `pipeline.run`, `ladder.run_label_source`, `variations` (six
sub-experiments), `bugsinpy.main`, `panels.run` and `figures.run`. Each one re-implements
the same sequence — build dataset, make split, build context, enumerate rankers, loop over
averaging subsets/ablation levels/variants, render a table, write a bespoke payload — and each keeps the
study's choices as its own module constants (`pipeline.HISTORY`, `pipeline.POPULATION`,
`ladder.RUNGS`, `ladder.BUDGETS`, `variations.N_BOOTSTRAP`, `models.default_rankers`).

**The clearest symptom.** "A named variant of one input to the measurement" exists three
times, spelled three ways:

* `ladder.RUNGS` — feature families withheld;
* `bundles.make_bundles` + `datasets.bundles` — bundle ablation levels;
* `semif_runner.INSTRUCTION_VARIANTS` — prompt wording.

Each grew its own cache keying, its own averaging subset and its own table rendering. Three
implementations of one concept is the argument that the factor abstraction is not speculative.

## 2. The five roles

An experiment is a sweep over **five roles**, and a genuinely new role is a kernel change:

    dataset, features, model, averaging subset, split

**Why not an open list of factors.** An arbitrary factor would have to say what it feeds, and the
kernel would have to know how to consume it; without that, an "factor" is a bag of values whose
meaning lives in the imperative runner — which is the problem being solved. Naming the five roles fixes
the meaning of each and makes the kernel a fixed pipeline. The extension story is then honest:
**new levels are cheap and are what most new dimensions actually are; new roles are a
deliberate kernel change.**

That distinction is not a limitation in practice, because every factor the study has
conceived already reduces to one of the five:

| factor people reach for | role it actually is |
|---|---|
| ablation level / feature ablation | `features` — a `FeatureBlock` with families withheld |
| bundle ablation level | `dataset` — a dataset derived from a base |
| prompt template (P2) | `model` — a `SemIfSelector` reading a different cache |
| direct mode (P1), embedding baseline (P3) | `model` — a ranker reading a different cache |
| candidate pool mode | a control variable (§4) |
| starvation threshold | a `averaging subset` |
| label source | a `dataset` |

## 3. Factor and Level

An **factor** is a named, ordered set of variants of one input. An **level** is one variant.

```python
Factor("model", (
    constant("coverage", models.CoverageSelector(), tier="cpu", estimated_seconds=0.1),
    constant("semif_reranker", models.SemIfSelector(), tier="cache", estimated_seconds=0.0),
))
```

`Level.make` is `Callable[[Build context], Any]` — arbitrary Python, so a level can close over
whatever it needs, including constructing a derived dataset:

```python
constant("bundles_r2", datasets.bundles(BASE, ablation level=2, seed=cfg.SEED))
```

No special machinery is needed for "a dataset that is a function of another dataset", because
`make` is already a function. The idiomatic form hoists the base to a module constant so it is
built once and shared.

**Two authorities, deliberately separated.**

* The **level** declares its name, its cost tier and its estimated seconds. These are
  presentation and scheduling facts about *this run's* use of a thing.
* The **value** declares its requirements — `Ranker.requirements()`,
  `FeatureGroup.needs`, `Averaging subset.needs`, `Dataset.capabilities()`. These are facts about
  the thing itself, and they must live where the inputs is read, for the reason
  `accessors.MATERIAL` exists: a need written beside the code that reads it cannot drift from
  it.

`Factor.map(fn)` applies a **shared option** to every level of a factor — the thing a
dimension-level option actually is. `fn` returns an `Level`, or an `undefined` when the
option cannot be expressed for that level; the inapplicable level is then *kept* as a
poisoned level rather than dropped, so the design point it would have produced is reported
undefined instead of silently vanishing. `XGBoostSelector` can express "exclude the history
family"; `SemIfSelector` cannot, because it reads text pairs rather than columns — and that
difference should show up as a finding in the report, not as a missing row.

**A third thing a level may declare: applicability.** Availability is normally asked of one
role's value at a time, which cannot express a variant that is only meaningful in combination
with a particular level of *another* role. That case is real and was found while wiring the
ladder: its SemIf cache was built over the changes `cache_covered` selects, and
`semif.load_scores` fills every *uncached* pair with a sentinel rather than reporting that it
has no score — so evaluating SemIf on the other averaging subset produces a number that looks like a
measurement and is not. `ladder.py` avoided this by not tabulating SemIf there;
`Level.applies` declares it, which keeps the design point in the report as a finding with a reason.
The parameter is a `Build context` carrying the design point's factor names, so the predicate reads the other
role's level rather than reaching for it. Those names are populated **only** for this check: a
value is built once per run and shared across every design point that selects it, so a level whose
`make` read them would silently be reused for design points it does not describe. Keeping the two
bindings distinct makes that a mechanical guarantee rather than a convention.

## 4. Controls, and the rule about what may not be configured

Only roles multiply into design points. Everything else is a **control variable**: one value for the whole run,
recorded once.

```python
Controls(seed=..., budgets=..., n_bootstrap=..., candidates="full", n_bootstrap_paired=None)
```

`n_bootstrap_paired` exists because a table's confidence interval and a paired test's p-value
are two quantities with two precision needs, and the recorded ladder used different counts for
them (1000 and 2000). Leaving it `None` means "the same as `n_bootstrap`". `Controls` otherwise
holds only what is free.

A control variable is not merely "something global". It is specifically **a choice that is free** — the
harness cannot derive the right answer from anything else. The rule that follows, and it is
the one design rule this layer adds:

> **Anything the harness can derive must not be an option.**

`history` is the worked example. Whether the cumulative temporal columns are present is not a
free choice: it is implied by the dataset's `ordering()` and by whether the split shuffles
(`splits.Split.effective_ordering`), and the harness already *warns* when a caller overrides
it. So `Experiment.history` exists as `None | bool`, is named an **override** rather than an
option, defaults to `None` (derive it), and when set produces the existing
`feature.history_on_imposed_order` warning which travels with the design point. Stating it as an
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

## 5. Cells, and undefined design points

A **design point** is one point in the product over the five factors plus the run's control variables. Its key is the
tuple of level names, so it is stable, ordered, and readable.

The product is **not** materialised blindly. Each design point is either measured or carries an
`undefined` naming the requirement that stopped it, and a undefined design point is **reported**,
never dropped. This is the load-bearing difference from `itertools.product`: dropping is what
turns "we asked and could not answer" into a silently halved contrast, and §6 of
`refactor.md` exists to prevent exactly that.

Five distinct things produce a undefined design point, and they are different findings:

1. **level unavailable** — a level that cannot be built (a poisoned level from §3);
2. **averaging subset unavailable** — the dataset cannot support it (`Averaging subset.unavailable`);
3. **model unavailable** — a requirement of the ranker's cannot be resolved (§6);
4. **tier not enabled** — the design point's cost tier was excluded from this run (§8).
5. **applicability** — the level declares it does not apply to this design point, which is how an
   interaction between two roles is expressed (§3).

An *undefined column* is a fifth and different thing: a feature block whose group the dataset
cannot support is measured-but-zeroed, and the `FeatureMatrix` audit travels in the design point
record. Rungs of the ladder rely on this, so a block's requirements must not gate the design point.

## 6. Availability

Requirements are strings. Two kinds, and the resolution happens in exactly one place:

* `artifact:<path>` — a file the level reads (a precomputed score cache). Resolved by existence.
* anything else — a `contract.Requirement` value (`coverage`, `durations`, `labels`,
  `diff_text`), resolved against the design point's dataset.

An unrecognised string raises, for the same reason `has_capability("coverge")` raises: the
vocabulary is closed and a quiet `False` would gate a whole block off.

`Ranker.requirements()` is the new half. `SemIfSelector` returns
`("artifact:artifacts/semif_scores.jsonl",)`; `XGBoostSelector` returns one per
`extra_score_files` entry; the baselines return `()`. A missing cache therefore becomes an
undefined design point carrying the path, rather than a `FileNotFoundError` string swallowed by the
imperative runner's `except` (which is what `pipeline` does today, into a `skipped` dict).

## 7. Comparability

A contrast is a paired delta against a reference level of one role, at a probe budget.
Pairing is only meaningful between design points that share rows, averaging subset, split, candidates and
labels, so the roles split:

* **pairable** — `model`, `features`. Varying these changes scores, not the quantity being
  averaged.
* **row-changing** — `dataset`, `averaging subset`, `split`. A delta across these compares different
  averaging subsets and is not a paired quantity.

`Contrast(role, reference, probe_budget)` therefore **refuses a row-changing role at
construction** rather than producing a plausible number. Grouping is automatic: design points are
grouped by the other four roles, and each design point is paired against the reference level's design point
in its own group. A group whose reference design point is undefined is recorded with the reason.

This is what replaces `pipeline`'s single hard-coded reference and the `cache_covered` averaging subset
in `ladder.py` — which exists *solely* to make pairing possible, and is a workaround for this
not having been answered once at the layer above.

## 8. Cost tiers and execution order

An level declares a `tier`; design points execute in `Experiment.tier_order` (default
`("cpu", "gpu")`) and `run(..., tiers=("cpu",))` excludes the rest as undefined rather than
running them.

**The vocabulary is deliberately coarse.** It says whether measuring a design point needs a GPU, which
is the one cost distinction this study acts on — the CPU-only traceability ladder before gated
expensive work. It is not a cost model: `estimated_seconds` carries the finer number where an
level has one, and now that these are recorded per design point, cost sits next to the result, which
is the direction `plan.md` already commits to (a cost-effectiveness curve rather than
recall@k).

A design point's tier is its most expensive level's, so a design point that reads a precomputed precomputed score cache
but still assembles features and sweeps metrics is `cpu` — which is correct: the cache makes
the *model* cheap, not the design point. An earlier revision had a `cache` tier as well; it was
removed because no design point could ever be it, since feature assembly and evaluation always compute.

No scheduler is built. Ordering, filtering and recording are enough to answer "what is cheap
here" without inventing a cost model whose units would not be comparable across datasets.

## 9. The artifact

One uniform payload per run, rather than a bespoke dict per imperative runner:

    experiment, note, environment (control variables, out_dir, caches, shared),
    factors (level declarations, including whether each declares applicability),
    n_cells, seconds,
    design points   [ key, factors, tier, estimated_seconds, measured: true,
              ranker, averaging subset, n_rows, n_changes, split, dataset_declaration,
              features (FeatureMatrix.audit), diagnostics, audit, results, seconds, importances ]
    undefined [ key, factors, tier, estimated_seconds, measured: false,
                 undefined: true, requirement, note ]
    contrasts [ role, reference, probe_budget, group, design point, reference_cell, delta, lo, hi, p_value, n ]

`design points` carries the measured ones and `undefined` the rest, so the holes are as visible as the
numbers. The `dataset_declaration`, `split` and `features` entries are per design point rather than per
run, because in a sweep they genuinely vary — the ladder's ablation levels differ in exactly that field.
`importances` is populated only on the design point that actually scored, for the reason in §10.
field.

## 10. Declined

* **An external config language (YAML/TOML).** Elements are objects with behaviour *and*
  declarations, so a config must name and construct Python objects. A data format would need a
  mirrored registry plus a schema and would still need Python for any new level. Python is
  also what `plan.md` and the report already read.
* **A blind `itertools.product`.** It discards §5's undefined design points and §7's comparability.
* **A decorator or module-level registry.** Declined in `refactor.md` §14 and still declined,
  for the reasons given there (module-level state; availability becomes a property of the
  import graph; column order becomes import order). `Factor` is a value, like
  `SubsetRegistry`, and a study's factors are a module of values, like `averaging subsets.STUDY`.
* **An open list of factors.** §2.
* **Reusing the levels' objects across design points.** The run materialises each level once and
  shares the value, which is what makes a 48-design point grid affordable (the dataset is built once).
  It is sound because these are values (`refactor.md` §2). The same reasoning extends to scores:
  a score matrix is a function of the *context* — dataset, features, model, split — so it is
  memoised by that key, and `averaging subset` is deliberately not part of it, because a averaging subset
  restricts which rows a metric averages rather than which pairs get scored. Without that, every
  averaging subset sweep would pay for a second identical training run.
* **Derived metric importances back off a ranker after the run.** A stateful ranker's
  `importances_` describes its most recent call, and the level is shared, so a later design point in
  another context would have overwritten it. They are captured on the design point that actually scored
  and are empty on design points that reused a matrix.
* **A named-artifact indirection in `Environment.caches`.** The first sketch let a requirement
  be written `artifact:<name>` and resolved through a run-supplied alias map, so a run could
  move where artifacts are read from. It was cut back to a plain path→path redirect: a level
  that reads a file needs a real path anyway, so the alias only ever bought a level of
  indirection between a ranker and the cache it was constructed with. `Environment.shared`
  covers the injection need — a stub dataset in a test, one base dataset shared by several
  derived levels — without a second mechanism.
* **A separate `Contrast.only` filter for "this contrast is defined only on this
  averaging subset".** It was the first attempt at the SemIf-cache problem above. `Level.applies`
  subsumes it and is more general, and the contrast machinery then needs no special case at
  all: a group whose reference design point is undefined is recorded with the reason.

## 11. Verification

The documented numbers are the deliverable, so the conditions are checked against them rather than
trusted, and both are checked the same way: run the declared experiment, **render** the artifact
the imperative runner writes, and compare every leaf of the payload.

    study condition (mutmut, full)      1581 recorded leaves   0 mismatches
    study condition (mutmut, coverage_restricted)   1581 recorded leaves   0 mismatches
    ladder condition (mutmut)            967 recorded leaves   0 mismatches
    ladder condition (full)              968 recorded leaves   0 mismatches

    together                     5097 contrasts        0 mismatches, 0 additions

Comparing the rendered payload rather than the design points is deliberately the stronger test: it
exercises the declarations, the sweep, the metric sweep, the paired contrasts, the dataset
statistics and the artifact reporter at once. A missing leaf is a failure and a differing leaf is a
failure; an unexpected leaf is a failure too unless its path is on the addition list in
`scripts/verify_experiment_layer.py`, so a renamed key cannot pass and a silent extra cannot
either. The earlier version of this check compared selected fields and would have missed a
artifact reporter that dropped a whole section.

That the `random` row reproduces exactly is the load-bearing part. `refactor.md` §12 records a
rewrite that moved every random-baseline number by changing the RNG *consumption pattern*, so an
exact random row is evidence that the layer consumes the RNG identically rather than merely
closely — and the same hazard appeared again during this work. Sweeping the low-co-occurrence condition's three
budgets in the same pass as the headline condition's six moved its *interval bounds*, because a
bootstrap stream is positional. The point estimates were unaffected. Since `budgets` is a control variable by
design, the low-co-occurrence condition is a second experiment, and the two share one precomputed score cache so the corners
cost only their metric sweeps.

The four undefined design points in each ladder run are exactly SemIf against the averaging subset its cache
does not cover — the `Level.applies` case from §3 — and each carries the reason in the report
rather than being silently absent, which is what the old imperative runner did by not tabulating it.

That the ladder verified is the stronger result, because it exercises what the study condition cannot:
four feature blocks in one run (so the matrix cache is keyed correctly), two averaging subsets per ablation level
(so a averaging subset is a factor and not a constant), a contrast whose reference is unavailable for
one of its two averaging subsets, and the split between the table intervals' resample count and the
paired test's. And it is exact in the strongest sense available: regenerating
`artifacts/ladder.json` with the migrated artifact reporter produces a **byte-identical file**.

**The variation sections are exact too, and two of them caught real defects.** Each of the six
migrated sections is rendered and compared leaf by leaf:

    cold_start        1821 recorded leaves   0 mismatches
    cold_start5       1821 recorded leaves   0 mismatches
    cold_start_seeds   263 recorded leaves   0 mismatches
    p2                   853 recorded leaves   0 mismatches
    p3                   690 recorded leaves   0 mismatches
    p5                  1042 recorded leaves   0 mismatches

    together            6490 contrasts        0 mismatches

Neither defect would have been visible anywhere else, which is the argument for verifying against
whole rendered artifacts rather than against the conditions that already work:

* **the score key did not carry the model seed**, so the four refits in `cold_start_seeds` would
  have reused the first fit and reported four identical deltas (§12);
* **the embedding baseline's cache is a `.npy` matrix, not a scored-pair log**, and reading it with
  the default loader raises rather than returning a wrong matrix -- lucky, and only because the two
  cache formats are distinguishable. A silent version of this is easy to imagine, and the loader
  is now an explicit part of the level.

## 12. What was implemented

**Modules.**

| module | responsibility |
|---|---|
| `rts/experiment/` | declaration (`Factor`, `Level`, `Controls`, `Environment`, `Build context`, `Design point`, `Contrast`, `Experiment`), report (`DesignPointResult`, `RunReport`), and run (`run` and the measurement it drives) |
| `rts/studies/` | the study's choices as values: `factors`, `conditions`, `ladder`, `variations`, `readings`, `bugsinpy` |
| `scripts/verify_experiment_layer.py` | the reproduction check of §11 |
| `tests/test_experiment.py` | 31 tests, all over the stub dataset |

The layer was later split from one 1227-line module into the three named above, so that "what
an experiment *is*" (``declaration``), "what a run records" (``report``) and "how it is
measured" (``run``) are separate files, with the package docstring still carrying the
five-role design. It is a partition rather than a rewrite: every top-level statement moved
verbatim, including its decorators and the comments above it, and the package re-exports the
same public names -- so ``rts.experiment.run``, ``ROLE_MODEL`` and the rest are the same
objects as before. The paragraph below describes the *first* pass only; later passes changed
more of the existing modules (the drivers in §13, and the restructure that moved the dataset
and model halves into ``rts/data/`` and ``rts/model/``).

`rts/model/rankers.py` gained one method — `Ranker.requirements()` — which is the whole of the
model-half declaration §1 identified as missing. The only other change to an existing module is
removing a parameter from `ladder.build_subsets` that its body never read.

**Decisions implementation forced**, as opposed to the ones §1–§10 anticipated:

1. **The paired set is the averaging subset's rows, not the evaluation window.** The first version
   computed per-change hits over `split.test_idx`, which for the ladder's `cache_covered` group
   would have paired 464 changes where the metric was averaged over 141. A contrast pairs two
   design points inside one group, and the group names a averaging subset, so hits are computed over exactly
   the rows the group's metric used.
2. **`Level.applies`, for an interaction between two roles.** §3 has it: the ladder's SemIf
   cache covers one averaging subset, and `semif.load_scores` gives the other a sentinel rather than
   saying it has nothing. This was not foreseen; it was found while wiring a real condition, which is
   the argument for wiring real conditions during the build rather than after it.
3. **`Controls.n_bootstrap_paired`.** The recorded ladder used 1000 resamples for its table
   intervals and 2000 for its paired p-values. Two quantities, two precision needs; making them
   one control variable would have made exact reproduction impossible and would have hidden the choice.
4. **The `cache` cost tier was removed.** A design point's tier is its most expensive level's, and
   feature assembly and metric evaluation always compute — so no design point could ever have been
   `cache`. The vocabulary is now `("cpu", "gpu")`, which is the distinction the study acts on.
5. **`Environment.caches` was demoted to a path→path redirect**, and `Environment.shared` is the
   injection seam. The alias-map version (§10) only ever inserted a layer between a ranker and
   the cache it was constructed with.
6. **Derived datasets needed no machinery at all.** `make` is arbitrary Python, so a bundle ablation level
   is a level whose builder constructs its base; `_bundle_dataset` in `rts/studies/factors.py` is
   the whole implementation. The one discipline is that a builder reading the run's seed does so
   through the binding — `order_seed=b.control variables.seed` — rather than closing over `config.SEED`,
   because otherwise a control variable override would silently not reach the dataset. The BugsInPy condition later
   confirmed the shape: a project selection is a parameter of the dataset *level*, so a corpus
   of three projects is a different experiment rather than a filter on the results.
7. **Elements are materialised once per run and shared across design points**, which is what makes the
   12-design point study condition ~110 s rather than twelve dataset builds. Sound because these are values.

**One hazard checked deliberately.** `refactor.md` §12 records that a rewrite moved every
`random`-baseline number by changing the RNG *consumption pattern*. The layer does not touch that
pattern: it calls the same `accessors.candidates` once per dataset level and the same
`evaluate.evaluate_rows` with the same arguments, so the random baseline reproduces exactly
(§11) rather than approximately.

**Second pass: the drivers.** §13 asked for the drivers to become `Experiment` values. Two have.
`pipeline.py` and `ladder.py` now contain no experiment logic: they run a declared condition and render
the artifact shape their recorded numbers, the `implementation.md` tables and the figures are
written against. That shape is kept deliberately and the sweep is what moved, so no documented
number moves. `ladder.py` went from 347 lines of sweep to 213 lines of rendering.

Two things the migration needed from the layer, both capability rather than convenience:

* **`run(..., scores=...)`**, which lets a caller share score matrices *between* runs. A study
  whose conditions report different budget sets cannot be one run, because `budgets` is a control variable; without
  sharing, the second condition rebuilds the dataset and retrains every model. The key is the design point's
  context -- dataset level, dataset name, features level, model level, split level, split
  shape, history policy, candidate mode -- so reuse is sound rather than incidental.
* **`models.Context.seed`**, so a run-level seed override reaches the models. Before it, `--seed`
  moved the split and the ordering but left `RandomSelector` and `XGBoostSelector` at their
  constructor defaults: a control variable that lies about what it changed. Both now take `None` to mean "the
  run's seed", as the BM25 shuffle controls do.

And two things it removed from a imperative runner:

* **the BM25 shuffle controls are model levels** (`LexicalSelector(shuffle_changes=...)`), so
  `results` and `ablations` are two slices of one grid with one provenance instead of a second
  loop with its own bookkeeping -- and the shuffle seed now comes from the run rather than from a
  imperative runner argument;
* **`reporting.recurrence(ds)`** and **`reporting.describe(ds, split)`** are dataset statistics the
  imperative runner computed inline. They live with the other reporting functions now, because they are facts
  about the data rather than about a sweep.

**Third pass: the variation study.** Six of `variations.py`'s seven sections are declared in
`rts/studies` and rendered from reports. Four capabilities were needed, and each turned out to be
the general form of something that already existed:

* **`models.CachedScores`** -- a score matrix read from a file, with the cache declared as a
  *requirement*. The instruction wordings, the direct-mode cache and the embedding matrix are all
  this; `SemIfSelector` is now a two-line subclass of it. It also means an absent cache is an
  undefined design point with the path rather than the `FileNotFoundError`-and-skip the imperative runner did. The
  *loader* is part of the level, because a cache is not always a scored-pair log: the embedding
  baseline is a whole matrix saved with `numpy.save`.
* **`models.RankAverageSelector`** -- the fitted-free combination of two rankers' normalised rank
  positions. The cold-start condition and the redundancy test both need it, and it is what makes the
  redundancy question answerable without fitting anything on the rows being evaluated.
* **`averaging subsets.cold-start(max_failures)`** -- the cold-start proxy as a declared averaging subset for one
  threshold, named for it. Two thresholds are two averaging subsets, and the caches are named the same
  way, so the two line up.
* **`Controls.model_seed`** -- see below; the one place the five-role design was asked for something
  it did not have.

**A defect the seed sweep exposed, which is the kind that produces plausible numbers.** The score
cache was keyed on the model's *level name* but not on its seed, so four refits of one level
would have silently reused the first fit and reported four identical deltas. That is exactly the
shape of the ablation bug `refactor.md` §14 records -- a name that matched nothing, a plausible
number, no error -- and it would not have been visible in the conditions already verified, because none
of them varies a model seed. `Controls.effective_model_seed` is now part of the key.

**`cold_start_seeds` is a multi-seed run, not a factor.** The question is whether the correction
rests on one lucky fit, so the same two trees are refit under four *model* seeds. Only the model
seed varies -- `Controls.seed` fixes the synthetic order and the split, and the caches are keyed to that
order -- which is why the two seeds are separate control variables rather than one.

The layer also stopped conflating two vocabularies: a design point records `diagnostics` (the *derivation's*
caveats -- a family withheld, history on a synthetic order) separately from `audit` (what the
dataset and split say about trusting a number at all). The ladder artifact needs them apart, and
merging them had made the distinction unavailable to a consumer.

**The artifacts were regenerated, not just checked.** Running the migrated drivers rewrites the
tracked artifacts, which is the end-to-end proof that the renderers work and the only way to keep
code and artifact in step:

* `artifacts/ladder.json` is **byte-identical** -- the migrated `ladder.py` writes exactly what the
  old imperative runner wrote.
* `artifacts/results_full.json` and `results_coverage_restricted.json` each gain **seven lines**, all of them
  the `paired_vs_reference.bm25_both_shuffled` entry. That is the one place where the layer
  computes more than the imperative runner did: it pairs *every* model in the group, and the old imperative runner
  paired only the two shuffle controls it had computed. Dropping the third to match the old key
  set would discard a computed number for cosmetic parity, so it is kept and called out here --
  the same treatment `refactor.md` §12 gave the three ranker conditions `results_full.json` had been
  missing. Every other leaf is unchanged, which the payload contrast in §11 shows leaf by leaf.

## 13. What remains

> Status, ordering and cost for these items are in **`handoff.md` §2**, which is the
> handoff for the harness. This section is the design-level statement of what is missing and why;
> that one is the work record and is kept current.

* **`variations` is six-sevenths migrated.** The cold-start conditions, the seed refit, the instruction
  sweep, the embedding baseline and the redundancy test are declared in `rts/studies` and rendered
  by `rts/render/variations.py`. Two sections are not:
  * **`p5_trained`** needs an evaluation window *inside* the held-out tail -- a `Split` whose train
    and test are both drawn from the tail -- plus a NaN convention for unscored pairs in
    `XGBoostSelector`'s extra columns. Both are small; both change what an existing concept means,
    so they are a deliberate step rather than a migration detail.
  * **`p1_direct` cannot be reproduced at all.** Its cache
    (`semif_direct_cold_start2_coverage_restricted.jsonl`) is absent from the artifacts, so the section is
    recorded but not reproducible, and only the layer's *behaviour* differs today: it would report
    a undefined design point with the path instead of raising. The code is kept as the record of what
    ran.
* **`bugsinpy` is migrated; `bundles` is not.** The real-label condition is declared in
  `rts/studies/bugsinpy.py` and rendered byte-identically by `rts/render/bugsinpy.py`. It needed three
  capabilities, each added to a *value* rather than to the layer: a per-change candidate pool
  (`Dataset.own_candidate_pool`), per-change-scope rankers (a BM25 fitted per change over that
  change's own documents, and a per-change random draw), and a cache loader keyed by position
  within a bug's own pool. `bundles` still needs a bundle-ablation level dataset factor and a bundle feature
  block; `studies._bundle_dataset` already shows the derived-dataset pattern as a three-line
  level, so its dataset side is cheap and the block side is the work.
* **`panels` is settled: off the layer on purpose.** It was `rts/analysis.py`, it had no factors or
  conditions, and it sits beside `reporting` and `figures` as an artifact reader. Its two non-design point
  properties are why it is not a condition, and both are stated in its docstring: the deciles are cut
  from the *dataset*, and each model is evaluated on the subset of a bin it actually has scores
  for -- a per-*row* filter, where `Level.applies` is per-*design point*. Evaluating the unscored rows
  instead would report `semif.load_scores`'s sentinel as a measurement.
* **`rts/studies` is a package** (`factors`, `conditions`, `ladder`, `variations`, `readings`,
  `bugsinpy`), re-exporting every public name so no call site changed. The split was scripted as
  an anchor-based partition that asserts the pieces rebuild the original file byte for byte, so a
  mis-anchored cut cannot silently drop a function.
* **`figures.py` is untouched and needs nothing.** It reads `artifacts/variations.json`, whose
  shape the artifact reporter reproduces, so the figures follow the payload without change. It is an
  artifact reader like `panels.py`, and that is the stated boundary: a *figure* is a reading of
  recorded numbers, not a sweep.
* **A `RunReport` now carries what a artifact reporter needs, but not the dataset object.** A run records,
  once per (dataset, split), the dataset's declaration, its description under the split it used
  and the pair-recurrence statistic, plus each averaging subset's size before and after the fault
  filter. `pipeline.render` and `ladder.run_label_source` read those instead of rebuilding the
  dataset, and `variations` reads its averaging subset sizes from the report, so no artifact reporter can
  describe a dataset the run did not measure. What is still *not* on the report is the dataset
  itself: a value a artifact reporter needs as an object (the ladder's averaging subsets, a condition's
  `own_candidate_pool`) it rebuilds, which is cheap and permitted (`refactor.md` §7), and which
  keeps the report serialisable.
* **Producing score caches is a design point now, for the caches that read as requirements.**
  `semif_runner.score_context` is the context-driven entry point and `models.ProducedScores` is
  the ranker whose `requirements()` is empty on purpose: declaring the cache would make the
  design point undefined before it could produce it. `python -m rts.studies semif.produce --tiers gpu`
  scores and writes; without the GPU tier the design point is reported undefined. The BugsInPy cache is
  the exception and is still produced by `python -m rts.render.bugsinpy --stage semif`: it uses a record
  format nothing else reads, so it would need its own production level.
* **`test_granularity`/annotations across a pooled dataset** are declared and now **policed**: the BugsInPy
  pool's constituents agree on `test_granularity`, and a test asserts both that and that the pool emits
  no `pool.mixed_test_unit` note -- so a warning that fired unconditionally, or one that never
  fired, would fail.
* **Cost is recorded, not modelled.** No budget-limited execution.
* **`splits.in_window` is enforced.** `Split` checks its own partition invariant, and
  `evaluate_rows`/`evaluate` refuse rows outside the split's window when one is given. The layer
  passes its split, so the guard holds there by construction -- which is the point: it exists for
  the callers that pass rows *directly*, which is what evaluating inside the held-out tail does.
