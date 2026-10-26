# Pairwise Frontier Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Accelerate the first multi-period frontier calculation and repeated pair plots while preserving numerical results and solver behavior.

**Architecture:** Reuse prepared calendar/growth arrays within a numerical calculation, evaluating trial weights with the original arithmetic order. Validate input freshness at calculation boundaries. Cache only the latest pair tables against fresh child-frontier inputs, keeping rendering separate.

**Tech Stack:** Python >=3.11, NumPy, pandas, SciPy SLSQP, joblib, pytest; existing uv environment, no dependencies.

**Spec:** `docs/superpowers/specs/2026-10-26-pair-frontier-acceleration-design.md` (user authorized implementation after reviewing the proposal).

## Global Constraints

- Preserve SLSQP options, target grids, initial guesses, corner handling, 10-decimal weight-cache keys and off-constraint trial arithmetic.
- Preserve scalar fallback for unsupported/exceptional inputs and custom subclass behavior.
- Preserve public signatures, parent common dates, pair bounds/order, labels and Axes output.
- No new parallel layer, global cache, dependency or runtime configuration.
- Compare full results at `rtol=atol=1e-12`; do not relax tolerance to hide a solver change.
- Current task worktree is already isolated; parent is sole writer. Existing scoped dev delivery authorization remains active.

## Review Focus

- Finite-difference weights temporarily do not sum to one: pin their original return series in Task 1.
- Mutable frames/strategy settings can invalidate caches: pin boundary refresh in Task 1 and pair invalidation in Task 2.
- NumPy reductions can change solver trajectories: compare full frontier weights and diagnostics to a frozen reference in Tasks 1/3.
- Pair histories can differ from parent dates or change through source asset objects: build current child inputs and test freshness in Task 2.
- Worker serialization, subclasses, exceptional returns and failed solves must retain behavior: cover Task 1 and measure Task 3.

### Task 1: Prepared multi-period optimizer evaluations

**Files:** `okama/common/helpers/rebalancing.py`, `okama/frontier/multi_period.py`, `tests/test_frontier_multi_preparation.py`, `tests/helpers/multi_optimizer_reference.py`.

**Interfaces:** `_PreparedRebalance.return_ror_single(weights: np.ndarray, first_return: float) -> np.ndarray | None`; frontier private calculation scope retains original cache behavior within a stable input snapshot. Public numerical entry points share the preparation.

- [x] Add a frozen original numerical reference and a real frontier test proving calendar grouping is bounded independently of objective evaluations; run it and verify an assertion failure.
- [x] Add return-series parity for all five periods, off-constraint weights, partial bins and fallback inputs; add full frontier/optimizer parity and freshness tests.
- [x] Implement the single-weight evaluator with elementwise multiplication and the original summation/cumulative multiplication order. Wrap results in the original pandas shape; fall back when unsupported.
- [x] Add calculation-scoped preparation, with nested calls and joblib workers sharing the prepared inputs; compare snapshots at external boundaries and clear affected cached values when changed.
- [x] Run `uv run --no-sync pytest -q tests/test_frontier_multi_preparation.py tests/test_frontier_reb.py tests/test_frontier_multi_batching.py`; expected: all pass. Resolve failures within project retry policy and record deviations.

### Task 2: Reuse pair tables across rendering

**Files:** `okama/frontier/multi_period.py`, `tests/test_frontier_pair_cache.py`.

**Interfaces:** private `_get_pair_frontiers() -> list[pd.DataFrame]` returns the numerical tables used by `plot_pair_ef`; no new public API. Latest validated pair table collection only; no figures cached.

- [x] Add a real repeated-plot test proving the second render performs no new optimization and produces identical lines; verify it fails before implementation.
- [x] Test changed child return data, common dates, bounds, points and period invalidate affected results; labels/figure size re-render without calculation; custom subclass behavior remains uncached.
- [x] Extract numerical pair construction and cache against freshly resolved child inputs. Reuse parent common dates and original child constructor semantics. Keep internal tables private from caller mutation.
- [x] Run `uv run --no-sync pytest -q tests/test_frontier_pair_cache.py tests/test_frontier_multi_preparation.py tests/test_frontier_reb.py`; expected: all pass.

### Task 3: Measurements, review and delivery

**Files:** `analytics/benchmark_pair_frontier.py`, `docs/performance/pair-frontier-acceleration.md`, accompanying JSON, summary/index/spec status.

- [x] Compare baseline `48a0e7c` with current complete cold pair calculations and warm repeated plots on offline ready assets; record source hashes, median/p95, output parity, peak RSS, environment and parallelism. Include 4/8/12 assets, 120/600 months and 12/80 points, using a bounded representative matrix rather than silently exhausting every Cartesian product.
- [x] Run full `uv run --no-sync pytest -q` and both Ruff commands; expected: no test failures and both Ruff checks pass. Report pre-existing warnings accurately.
- [x] Obtain an independent read-only final review after current routing admission, verify findings and record resolution.
- [x] Update durable results and the existing web capacity task with measured benefits and limits; do not change website limits in this library task.
- [ ] Commit/push reviewed changes, merge into dev after exact-head CI, synchronize owned task/default checkouts, archive scratch and verify no active Git lock.
