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
