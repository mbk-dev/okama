# Frontier CPU batching implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` for inline implementation and a bounded independent review.

**Goal:** Implement [okama#127](https://github.com/mbk-dev/okama/issues/127): accelerate repeated portfolio calculations without changing formulas or public APIs.

**Architecture:** Profile actual Monte Carlo and grid methods on fixed offline returns. Select the single-period path if repeated pandas covariance/mean preparation dominates; prepare these statistics once and calculate float64 quadratic forms in bounded NumPy chunks. Keep multi-period rebalancing intact unless profiling and equivalence coverage justify changing it.

**Tech Stack:** Existing Python >=3.11, NumPy, pandas, pytest, Poetry; no new dependencies.

**Spec:** The linked issue is the binding specification.

## Global constraints

Preserve annualized risk, arithmetic return and risk-adjusted approximate CAGR; bounds, label and row order, validation and output schema. No GPU dependency, global BLAS setting or scalar optimizer changes. Measure 128/2048/8192 weights, preparation and construction, timing distributions and peak memory. Use offline data and identical weights for baseline/candidate comparisons.

## Review focus

- Bounds and grid guards must still apply before calculation.
- Mixed-case and invalid `kind` behavior, empty requests, labels and duplicate labels must retain existing semantics.
- Pairwise covariance and column means must preserve pandas missing-data semantics.
- Chunk boundaries, zero-risk/correlated assets and float64 cancellation must agree with scalar reference.
- Multi-period calendar and unrebalanced semantics stay unchanged.

### Task 1: Reproducible profiling and baseline

- [x] Create `analytics/benchmark_frontier_batching.py`: fixed offline asset fixture, actual methods, fixed-weight timing and natural grid/Monte Carlo workloads; capture versions, BLAS, revision, distributions, tracemalloc peak and phase profile.
- [x] Run clean baseline tests and benchmark; retain JSON and profile evidence under `tmp/issue127` until incorporated into report.

### Task 2: Single-period batch computation through TDD

- [x] Add equivalence and edge-case tests in `tests/test_frontier_batching.py` using scalar helper formulas as oracle.
- [x] Verify RED using a real requirement: covariance preparation must be once per request, independent of weight count; spy on real pandas work, do not substitute numerical output.
- [x] Add private `EfficientFrontierSingle._get_portfolio_points(weights_series: pd.Series, kind: str) -> pd.DataFrame`; reuse in both public methods and bound intermediate allocation with a private chunk size.
- [x] Verify GREEN and entire frontier test modules; retain validation and duplicate-label behavior.

### Task 3: Benchmark, report and verification

- [x] Repeat identical benchmark against candidate and record environment, baseline revision, candidate content digest, timings, memory and limitations in `docs/performance/frontier-cpu-batching.md`.
- [x] Obtain read-only independent review; verify findings and fix material issues through TDD.
- [x] Run `poetry run pytest -q`, `poetry run ruff check .`, `poetry run ruff format --check .`; inspect final diff and Git lock state.
- [x] Report measured user impact and exact delivery state. Commits/pushes/releases require separate authorization.

## Execution record

Selected single-period batching after actual-method profiling; multi-period rebalancing was left unchanged under the issue's at-least-one-path criterion. No dependency, public API, commit, push or deployment changes. TDD covariance tests failed at 19/24 preparations and then passed at one preparation. All 547 tests pass (3 skips, 371 existing pandas warnings); frontier modules are warning-free and both Ruff commands pass. Independent completed-diff review found no significant issues. Final report and raw measurement distributions are in `docs/performance/frontier-cpu-batching.md`.
