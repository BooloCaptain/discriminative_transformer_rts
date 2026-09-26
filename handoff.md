# Handoff

Status of the study and of the harness: what is complete, what is left, and in what order.

This was section 11 and section 15 of `implementation.md`, which had grown into a design record,
a results record and a status record at once. The measured results stay there; the status lives
here. Section references like "§12" or "§5.7" inside this document point into `implementation.md`.

Two agendas, kept separate because they are different kinds of work and were previously entangled
in one file:

- **the study** -- what to measure next, and what would change the verdict (section 1)
- **the harness** -- the machinery those measurements are made with (section 2)

---

## 1. The study: what to measure next

### Bottom line

**SemIf does not win in any regime tested, and it never becomes competitive.** The corrected
answer is a clean no rather than the equivocal one the study previously reported.

| regime | SemIf | best classical | verdict |
|---|---|---|---|
| full held-out, 464 faults, covered | 0.306 | 0.700 XGBoost | loses 2.3x |
| starved `failures <= 5`, 141 faults, **full suite** | 0.681 | **0.972** XGBoost | **loses 1.4x** |
| starved `failures <= 2`, 43 faults, **full suite** | 0.674 | **0.953** XGBoost | **loses 1.4x** |
| starved `failures <= 2`, 43 faults, covered | 0.442 | 0.256 XGBoost | "won" 1.7x — **artifact of the mask** |
| 6 mutations, cross-file | 0.190 | 0.525 XGBoost | loses |
| 6 mutations, coherent | 0.160 | 0.555 XGBoost | loses, and BM25 overtakes it |

(all at budget 0.05)

| lever | outcome |
|---|---|
| **P5** SemIf as a feature | Near-redundant; equal-weight blending hurts (-0.226, p<0.0001); a trained column adds +0.08 to +0.10 at the two smallest budgets only, and fails correction. |
| **P2** instruction sweep, 5 wordings | Clean null, 16/16 non-significant, at both starvation thresholds. |
| **P3** code-specialised embedding | Below BM25 in every regime (0.071 vs 0.504 starved/full). Not an architecture problem. |
| **P1** direct mode, pairwise | Significantly worse than the reranker at every budget (-0.163 to -0.209); lands level with the cheap tree. |
| **P6** dilution mitigation | Subsumed by P2 — the dilution is a position effect, not a content effect. |

### Completed

Synthetic change history from mutmut (2651 mutants, 2311 killed, exact per-test labels);
evaluation harness (per-change budgets, temporal split, paired bootstrap, shuffle ablations,
sparsity sweep, complexity ladder, coherent bundles, embedding baseline, direct-mode arm);
four SemIf reranker arms over all 530 held-out changes plus four diagnostic controls; the
full-suite starved arm at n=43 and n=141; a history×coverage decomposition and a four-seed
robustness check; five instruction wordings; and figures for the correction, the mechanism and
the levers. New machinery: `rts/semif_runner.py` (instruction variants, full/starved/train-prefix
arms, `--exclude-scored`), `rts/direct_runner.py`, `rts/embed.py`, `rts/variations.py`,
`rts/figures.py`, `scripts/run_variation_arms.sh`.

### Next steps, in priority order

> The same agenda, in a form meant to be read on its own, is in `plan.md` under **Next steps**.
> Keep the two in sync if either changes.
>
> **Superseded in part by `plan_next_steps.md`, which is the detailed execution plan.** The
> target regime is now specified as a test suite driving an embedded system across a boundary,
> and that plan retires proposal 2 (de-lexicalisation) — it varied vocabulary while holding
> structure fixed, and the coverage-bearing tree is measurably unaffected. It replaces it with a
> CPU-only traceability-loss ladder, and promotes real data (proposals 4-5) to the load-bearing
> workstream.

**Harness work is separate from study work.** The agenda above is about *what to measure next*.
The work needed to extend or finish the measurement machinery is in **section 2** below.

The five proposals below come from a review of how this benchmark diverges from the target
setting (long-running integration tests of embedded systems). The important observation is
that there are **two independent gaps**, and the second is more fundamental than any feature
manipulation:

**Gap 1 — the label set is defined by coverage** (§10). Feature manipulations cannot reach
this; the labels have to change.

**Gap 2 — the features that win are exactly the ones that do not transfer.** Coverage,
filename matching and identifier overlap are all artefacts of a co-located, instrumented,
conventionally-named unit-test suite. Embedded integration suites are usually none of those.

1. **Full-suite relabelling (~5 min wall-clock, CPU).** Run the full 1190-test suite for every
   mutant instead of mutmut's median of 5 selected tests, and rebuild labels from the outcome
   log. Yields: how many faults actually have out-of-coverage killers, an "indirect fault"
   evaluation subset, and an honest ceiling for the structural funnel. This is a correctness
   fix for the current study as much as a new experiment — the existing label set is circular
   with respect to the coverage feature that dominates every result.
2. **De-lexicalisation ladder (~1 h GPU, 3 arms).** Rename the changed symbol and its locals in
   the *diff* only; then obfuscate both sides consistently; then obfuscate test names too.
   This severs the shared-vocabulary bridge that BM25 depends on and is **the one manipulation
   expected to favour a text model** — all four negative results so far left that bridge
   intact. Predictions: BM25 drops sharply; SemIf drops less but still loses to the coverage +
   BM25 tree. If SemIf cannot beat that tree here, the semantic hypothesis is dead in a way
   nothing so far establishes.
3. **Traceability-loss manipulations (~1 h CPU, no GPU).** Any manipulation that changes only
   features or the candidate pool needs no re-scoring, because the SemIf caches are keyed on
   text pairs. In order of how directly each targets the embedded setting:
   - **coverage coarsening** — recompute `covers_function` at module granularity, dilate it
     with k random coverers, and drop coverage for a random 50% of tests ("partial
     instrumentation"). Targets the measured winner directly.
   - **time budgets** — give tests heavy-tailed runtimes and select under a seconds budget
     rather than a count, which is what a practitioner actually optimises.
   - **coarse test units** — group by test class/fixture (coherent, unlike random grouping) so
     that no single filename matches. Requires a token-budget-matched control (the built but
     unrun `bundle_text(token_budget=)`) because a ~12k-token multi-topic window would hurt
     the reranker for dilution reasons unrelated to semantics.
4. **Real commit history on marshmallow (~10 min).** The suite is 0.75 s, so the suite can be
   run at a sampled set of real revisions, replacing the imposed random order with the real
   commit graph and giving genuine cumulative failure/coverage history. This is the cheapest
   available non-synthetic step and it directly addresses the least realistic property of the
   starved arm (synthetic history over-repeats `(file, test)` pairs ~159x). Expectation: the
   history features look weaker, not stronger.
5. **BugsInPy — real multi-project data.** 493 real bugs across 17 Python projects with known
   `failing_tests`, i.e. non-synthetic changes and labels, several failing tests per change,
   and more than one SUT. Needs per-project environments and test commands but no GPU for the
   classical side. Alternatives for scale or a second language: SWE-bench `FAIL_TO_PASS` (2294
   instances, 12 repos, but heavy contamination risk and curated test lists), Defects4J (Java),
   and CI corpora (TravisTorrent, Bears, GitBug-Java) for genuine per-test failure history.

Two framing changes are needed once real data arrives: report a **cost-effectiveness curve**
(time saved vs faults missed) rather than recall@budget, and account for the mostly-harmless
changes that dominate real history. What cannot be simulated on this SUT is hardware coupling,
non-determinism and cross-compilation; label noise (flipping a small fraction of outcomes) is a
cheap partial proxy for flakiness only.

Lower priority, carried over: the **16-option windowed direct mode** (the formulation the repo
actually designed — blocked on throughput, and the dilution evidence is against it), and the
**`after_document` re-run on all 464 faults** (~74 min) to make the full-set fairness
comparison citable rather than inferred.

### What would change the verdict

- A prompt or framing change lifting SemIf above 0.700 on the full held-out set. **Tested and
  refuted** (P2, P1).
- A code-specialised model clearly beating BM25. **Refuted for the cheapest member of that
  family** (P3); a modern instruction-tuned embedding is untested but would need a wide margin.
- SemIf carrying signal the structured features lack. **Largely refuted** (P5): +0.08 at b0.05
  on the richest feature set, not surviving correction.
- **De-lexicalisation showing BM25 collapsing while SemIf holds** (proposal 2) — the sharpest
  remaining test of the semantic hypothesis, and the only one with a stated reason to favour
  the transformer.
- **Out-of-coverage faults being a large fraction of the population** (proposal 1) — would mean
  every result in this document is measured on an easier, structurally-biased subset.
- Real data (proposals 4-5) showing multiple failing tests per change shifting the balance, or
  a different project behaving differently.

---

## 2. The harness: the experiment layer and what remains

State at the time of writing: branch `feature/experiment-layer`, 111 tests, clean tree.
`experiment.md` is the **design** record -- what an experiment is, why it has five roles, what was
declined and why. This section is the **work** record: what is migrated, what is not, and in what
order the rest should be done. Where the two overlap, `experiment.md` is authoritative for design
and this section for status.

### The gate every migration must pass

A migration is accepted only when the driver's artifact is **regenerated and compared leaf by
leaf** against the recorded one:

    python scripts/verify_experiment_layer.py                  # study, ladder, variations
    python scripts/verify_experiment_layer.py ladder --labels full
    python scripts/verify_experiment_layer.py variations

It exits non-zero on a missing leaf, a differing leaf, or an unexpected leaf that is not on the
documented addition list. Three migrations have passed it:

| artifact | result |
|---|---|
| `results_full.json`, `results_covered.json` | 1581 leaves each, exact; refreshed for one documented addition |
| `ladder.json`, both label sources | 967 and 968 leaves, exact; regeneration is **byte-identical** |
| `variations.json`, six of seven sections | 6490 leaves, exact; **nothing changed** |

So the recorded artifacts are the specification, not a sanity check. If a migration moves a
number, either the migration is wrong or the movement is a finding to be argued and recorded --
never absorbed. `refactor.md` §12 and `experiment.md` §12 both record cases where a "plausible"
number came from something broken, and the second of those (a score cache keyed on an element's
name but not its seed) was found only because a whole artifact was regenerated rather than a
sample of fields compared.

### What is already migrated

| module | state |
|---|---|
| `rts/pipeline.py` | renders `results_{full,covered}.json` from a declared arm |
| `rts/ladder.py` | renders `ladder.json` from a declared arm |
| `rts/variations.py` | six of seven sections rendered; `p5_trained` and `p1_direct` keep their original code |
| `rts/studies.py` | the declarations: axes, arms, and the readings that turn a report back into a table |
| `rts/experiment.py` | the layer itself |

### The remaining work, in priority order

**1. `p5_trained` -- the only variation section with a stated reason to be next.** It is P5 as
originally proposed (SemIf as an extra XGBoost *column*), where `p5_redundancy` is the fitted-free
proxy. Two things block it, both small and both a change to what an existing concept means: a split
whose evaluation window sits *inside* the held-out tail (constructible as a `Split` value, but it
is the first split that is neither a prefix nor a shuffle), and the NaN convention for unscored
pairs in `XGBoostSelector`'s extra columns. Cost: about an hour. Gate: regenerate the section,
expect 0 leaves, and note that the `_semif` arms' numbers depend on the NaN rule.

**2. `bundles` -- the complexity ladder.** One of the study's headline negative results (SemIf
loses on 6-mutation bundles, and loses to BM25 on the coherent ones), currently outside the layer.
The dataset side is nearly free -- `studies._bundle_dataset` already shows a derived dataset as a
three-line element, and `features/bundle.py` is a declared block -- so the work is the rung
definitions as a dataset axis, the pool mode as an element parameter, and a renderer for
`bundles_cpu_signal.json` / `bundles_cpu_union.json` plus `figures/complexity_ladder.png`. Gate:
regenerate both artifacts.

**3. `bugsinpy` -- the real-data arm.** `bugsinpy_results.json` is a documented artifact and the
only real-label data in the study. Needs a dataset element for the pooled corpus (eight projects,
one dataset each, already composed by `datasets.bugsinpy_pooled`) and a renderer for its bespoke
content: the T0 bridge audit and the per-project recall breakdown. Gate: regenerate
`bugsinpy_results.json`; the pooled SemIf-vs-BM25 tie at b0.05 is the number to watch.

**4. Split `rts/studies.py` into a package.** It is now about 1150 lines holding three catalogues
-- the headline arms, the ladder's declarations, the variation arms. Splitting is mechanical
(`rts/studies/{axes,arms,ladder,variations,readings}.py`) and would make the variation arms
findable without reading the whole file. No behaviour change, so the verification should show 0
differences anywhere.

**5. Decide the fate of `analysis.py`.** It is a *derivative* producer rather than an experiment:
it has no axes and no arms, it reads the recorded score caches and the dataset, and it writes the
sparsity-sweep panels under `artifacts/figures/` (`panels_budget0.01.csv`,
`panels_budget0.05.csv`, `panels_summary.json`) that §5.4's figures come from. So it does not
belong on the layer -- there is nothing to sweep. The open question is whether its panels should
be derived from the layer's reports (which carry the per-cell tables) rather than recomputed from
the raw caches, and the honest answer is that it is cheaper to leave it alone than to decide.
Worth a line in `experiment.md` §13 either way, because "analysis and figures read artifacts"
deserves to be a stated boundary rather than an omission.

**6. `p1_direct` cannot be reproduced.** Its cache (`semif_direct_starved2_covered.jsonl`) is
absent from `artifacts/`, which is why it is the one section that is recorded but unverifiable.
Either regenerate the scores (8,329 pairs at the measured ~1.25 pairs/s, so roughly 104 minutes)
and then migrate it as an ordinary cached-score element, or accept the recorded numbers as final
and drop the code path. Leaving it as-is is the worst option, because it looks runnable.

### Smaller items, deliberately not ordered

- **Cache production is outside the layer.** A `semif_runner` GPU arm is a *precondition* of a
  model element rather than a cell, so the study's most expensive step is invisible to the layer
  and a stale cache is indistinguishable from a fresh one. Modelling production as well as
  consumption is a build graph over artifacts -- a second feature, not a migration.
- **`RunReport` has no consumer.** `python -m rts.studies` writes the layer's own uniform payload,
  and nothing reads it; every reader still reads a legacy shape from a renderer. Either point the
  figures at it, or accept it as a debugging aid and say so.
- **Cost is recorded, not modelled.** Elements carry an optional `estimated_seconds` that is
  mostly unset, and tiers only distinguish `cpu` from `gpu`. There is no budget-limited execution.
- **`splits.in_window` is not enforced.** A population's rows come from the evaluation window by
  construction, but a predicate naming rows outside it would not be caught.
- **`test_unit` and semantics across a pooled dataset** are declared but not policed, carried over
  from `refactor.md` §13.
- **`Element.applies` is used exactly once** (SemIf against the ladder population its cache does
  not cover). It earns its place, but a second user would confirm the shape.
- **`history=True` is set explicitly by every arm**, so the derived default is never exercised by
  the study. Worth confirming the default is the one a new arm should get.

### How to add an arm or a section

The recipe the five migrations followed, for the next one:

1. Declare the sweep in `rts/studies.py`: axes of *values*, `Knobs` for what is genuinely free, and
   one `Comparison` per (reference, probe budget) -- a comparison is defined by one budget.
2. If a value needs a capability the harness lacks, add it **to the value** (`Selector`,
   `Population`, `Dataset`) rather than to the arm. Every capability this work needed turned out
   to be the general form of something that already existed: a cached score matrix, a fitted-free
   combination of two selectors, a parameterised population.
3. Render the artifact in the driver, and keep experiment logic out of the renderer. If the
   driver needs the dataset itself (for `describe` or `recurrence`), rebuild it -- datasets are
   values and two may coexist.
4. Add the arm to `scripts/verify_experiment_layer.py`, run it, and expect **0 additions**.
5. Pin the arm's element names against the recorded artifact's keys in `tests/test_studies.py`, so
   a rename cannot pass quietly.
