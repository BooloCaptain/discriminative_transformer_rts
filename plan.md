# Purpose
Evaluate if discriminative transformer models like SemIf can compete with traditional discriminative machine learning for regression test selection. SemIf is an open source pretrained discriminative transfomer (https://github.com/TheoLeeCJ/SemIf-OpenJev). This is a pre-study feasibility test.

Tooling, harness details, and measured verification results: see `implementation.md`.

# Scope
- Run a mutation testing framework on a Python open source library to generate synthetic changes with attached test result data to build a synthetic change history. Keep non-killed mutants and no-op mutants.
- Start the synthetic change history at a single revision (N=1, HEAD), using the full generated mutant population. History-wide mutation requires installing dependencies and obtaining a green suite at every historical revision, which is a reproducibility project of its own and the most likely way the probe stalls. Prove the pipeline at HEAD first, then extend to N revisions.
- Extract a standard set of structured features for the rts task, such as diff metadata, failure rate, and change to test comparative features such as test coverage and directory path distance etc.
- Compute all history-derived structured features (failure rate, coverage, co-occurrence) cumulatively, using only changes and test outcomes strictly before the change's own timestamp.
- Extract unstructured features such as diff content, changed file content, and test content.
- Train XGBoost on the structured features for the rts task.
- Compare the performance of XGBoost to statistical baselines such as failure rate and recency (recency will prove useless because of the structure of the synthetic data) as well as test coverage.
- Add a lexical baseline: BM25 / token overlap between the changed lines and the test text, scored per (change, test) pair. This is the cheapest plausible explanation for a transformer win, so SemIf must clearly beat it for the semantic hypothesis to survive.
- Compare the preformance of XGBoost and the baselines against a discriminative transformer model that uses the unstructured features to measure semantic relevance between each change and test to rank them.
- Run evaluation for both an unfiltered set of changes, as well as one where chnages are filtered to be sparse --- meaning a file - test combination only re-occurs rarely --- which mirrors real software evolution.
- Accepted limitation: the pinned mutation framework cannot generate complex multi-line mutants. See `implementation.md` for the operator set, and for why the change-shuffle ablation and the lexical baseline compensate.

# Evaluation
- Evaluate using a configurable temporal split (default 80/20). Ensure no temporal leakage.
- Give each selection model a per-change budget of number of selected tests (a fixed fraction of that change's own suite, not one global budget). A global budget lets a constant-selection strategy score well; a per-change budget neutralizes a change-independent test prior.
- Evaluate a sweep of budgets (0.01 to 0.2)
- A fault is caught if any failing test was selected for the change that caused the fault.
- Evaluate recall of caught faults (caught faults / faults).
- Evaluate precision, F-measure, and suite-reduction.
- Report paired bootstrap CIs.
- Change-shuffle ablation: re-score the same (change, test) pairs with the change text shuffled across changes (or blanked), and separately with the test text shuffled. If recall barely drops when the change is shuffled, the model is exploiting a change-independent test prior and the headline result is an artifact. If recall collapses, the model is genuinely conditioning on the change.

# Undecided
- Query/document roles for the reranker: which side is the change and which is the test. Rerankers are asymmetric, so this is a real design choice.
- Precise definition and threshold for the "sparse" change filter (granularity, window, cutoff).

# Resolution status

Both items were carried into the SemIf work and are now partly settled. See
`implementation.md` for the numbers.

**Sparse filter.** Implemented as `dataset.starved_mask(max_failures=...)` keyed on the
killing `(file, test)` pair's *failure* count. Thresholds 2 (43 held-out faults) and 5
(141) are the practical ones; adding the `max_runs` half of the "run once or twice"
condition collapses the sample to 27/15/8 faults and is not usable. Note the threshold
was chosen after seeing the data, which the implementation document flags.

**Query/document roles.** Still open. The original comparison was n=10 and favoured
`test_query`, which is why every cached arm uses `change_query` only after that pilot;
the planned re-test at adequate n was dropped for time. This matters less than it
looked: P2's instruction sweep shows the model is insensitive to how the question is
phrased, and P1 tested a different formulation (direct mode) rather than the
orientation. If the orientation is ever re-tested it should be done on the full
candidate set, since the covered mask is now known to distort the comparison.

**Superseded by the results.** The "sparse change filter" arm was the study's main hope
for finding a regime where a semantic model wins. It does not: re-scoring the starved
population against the full 1187-test suite shows the classical selectors ahead at every
budget, and the larger population moves the two soft cells further against SemIf. See
section 5.7 of `implementation.md`.

# Next steps

> Detailed execution plan: `plan_next_steps.md`. **Read it before this section** — it revises
> the agenda below. The target regime has been sharpened to *a test suite driving an embedded
> system across a boundary* (Python tests over serial/socket to firmware), which kills coverage,
> filename matching, identifier overlap and history at once. The plan is now organised around
> (a) a CPU-only traceability-loss ladder that tests the hypothesis on existing data for free,
> and (b) real data — BugsInPy for honest labels, and a boundary-structured corpus for the
> actual regime. **Proposal 2 below (de-lexicalisation) is retired**: it varied vocabulary while
> holding structure fixed, and measurement showed the coverage-bearing tree is unaffected by it.

This is the agenda for the next session. The study's question is answered — SemIf does not
beat the classical selectors in any regime tested, and all four text-side levers failed
(`implementation.md` §6 and §11) — so what follows is the work that could still change that,
or show the benchmark is measuring the wrong thing. It rests on two independent gaps between
this benchmark and the target setting (long-running integration tests of embedded systems).

**Gap 1 — the label set is defined by coverage.** `mutmut` only *runs* the tests covering the
mutated function, so a fault whose real killer does not cover the changed function is
recorded as "never ran" and treated as not failing. The recorded invariant ("0 killing tests
fall outside the coverage set") is therefore true *by construction, not by discovery*. This is
the benchmark's most consequential limitation: it structurally excludes the integration-test
failure mode, and it makes the coverage feature that dominates every result circular with
respect to the labels. Feature manipulations cannot reach this — the labels have to change.

**Gap 2 — the features that win are exactly the ones that do not transfer.** Coverage,
filename matching and identifier overlap are all artefacts of a co-located, instrumented,
conventionally-named unit-test suite. Embedded integration suites are usually none of those.

1. **Full-suite relabelling.** Run the full 1190-test suite for every mutant instead of
   mutmut's median of 5 selected tests, and rebuild the labels from the outcome log.
   **~5 min wall-clock** at 8 workers (2651 mutants × 0.75 s ≈ 33 min CPU). Yields the count
   of faults with out-of-coverage killers, an "indirect fault" evaluation subset, and an
   honest ceiling for the structural funnel. A correctness fix for the current results as
   much as a new experiment.
2. **De-lexicalisation ladder.** Three arms: rename the changed symbol and its locals in the
   *diff* only; then obfuscate both sides consistently; then obfuscate test names too.
   **~1 h GPU.** This severs the shared-vocabulary bridge BM25 depends on and is **the only
   manipulation with a stated reason to favour a text model** — every one of the four
   negative results so far left that bridge intact. Prediction: BM25 collapses; SemIf drops
   less but still loses to the coverage + BM25 tree. If it cannot beat that tree here, the
   semantic hypothesis is dead in a way nothing so far establishes.
3. **Traceability-loss manipulations.** **~1 h CPU, no GPU** — the SemIf caches are keyed on
   (change, test) *text* pairs, so manipulations that change only features or the candidate
   pool need no re-scoring. In order of how directly each targets the embedded setting:
   *coverage coarsening* (recompute `covers_function` at module granularity; dilate it with k
   random coverers; drop coverage for a random 50% of tests, i.e. partial instrumentation);
   *time budgets* (heavy-tailed test runtimes, select under a seconds budget rather than a
   count — what a practitioner actually optimises); *coarse test units* (group by test
   class/fixture, coherent unlike random grouping, so no single filename matches — needs the
   built-but-unrun `bundle_text(token_budget=)` control, because a ~12k-token multi-topic
   window would hurt the reranker for dilution reasons unrelated to semantics).
4. **Real commit history on marshmallow.** **~10 min.** The suite is 0.75 s, so it can be run
   at a sampled set of real revisions, replacing the imposed random order with the real
   commit graph and yielding genuine cumulative failure/coverage history. The cheapest
   available non-synthetic step, and it directly addresses the least realistic property of
   the starved arm (synthetic history over-repeats `(file, test)` pairs ~159×). Expectation:
   the history features look weaker, not stronger.
5. **BugsInPy — real multi-project data.** 493 real bugs across 17 Python projects with known
   `failing_tests`: non-synthetic changes and labels, several failing tests per change, and
   more than one SUT. Needs per-project environments and test commands; no GPU for the
   classical side. Alternatives for scale or a second language: SWE-bench `FAIL_TO_PASS`
   (2294 instances, 12 repos, but heavy pretraining-contamination risk and curated test
   lists), Defects4J (Java), and CI corpora (TravisTorrent, Bears, GitBug-Java) for genuine
   per-test failure history.

**Framing changes real data forces.** Report a *cost-effectiveness curve* (time saved vs
faults missed) rather than recall@budget, and account for the mostly-harmless changes that
dominate real history.

**What cannot be simulated on this SUT.** Hardware coupling, non-determinism and
cross-compilation. Label noise (flipping a small fraction of outcomes) is a cheap partial
proxy for flakiness only.

**Lower priority, carried over.** The 16-option windowed direct mode — the formulation SemIf
was actually designed for, blocked on throughput (1.3 pairs/s without
`flash-linear-attention`/`causal_conv1d`), and the dilution evidence is against it — and the
`after_document` re-run on all 464 faults (~74 min) to make the full-set fairness comparison
citable rather than inferred.
