# Calendar rebalancing and multi-period batching plan

> **For agentic workers:** Use `superpowers:executing-plans` for inline TDD with independent final review.

**Goal:** Implement the remaining calendar and multi-period repeated-weight hot paths with measured complete-method gains.
**Architecture:** Exact resample boundaries plus array period loops for calendar wealth; once-prepared EF growth plus memory-bounded independent-weight chunks for frontier sampling.
**Tech Stack:** Poetry Python >=3.11, existing NumPy/pandas, pytest and Ruff.
**Spec:** `docs/superpowers/specs/2026-10-26-calendar-batching-design.md`.

## Review focus

- Calendar checks occur at period ends; partial starts, half-years, zero thresholds and event priority remain compatible.
- NaN/gap/zero-growth/unsupported dtype and subclass cases preserve scalar behavior and errors.
- Multi-period grouping occurs once; chunk size changes neither numbers nor schema and memory scales with the configured element bound.
- Short-history CAGR, zero-risk cancellation, labels/duplicates, bounds and no-cache-growth remain compatible.
- Earlier frontier/ListMaker/conditional implementations remain intact.

### Task 1: Calendar Rebalance

- [x] Extend the checked-in frozen reference with original calendar/EF methods, independent of Git at test time.
- [x] Add RED for per-period container growth and full calendar Result equivalence across periods/thresholds/metadata.
- [x] Extract original calendar branch as fallback; implement finite contiguous monthly NumPy period loop with original boundaries/arithmetic.
- [x] Verify GREEN including partial years, event timing, zero thresholds, gaps/NaNs/zero wealth, weights and subclass fallback.

### Task 2: Multi-period frontier batching

- [x] Add RED for one-time grouping of repeated weights and scalar-equivalent public output across all periods/methods.
- [x] Add private prepared EF array representation and chunked return calculation.
- [x] Route Monte Carlo/grid through a shared private point builder with scalar exceptional/low-risk fallbacks.
- [x] Verify periods, bounds/labels/order, short histories, chunk limits, low-risk and unsupported-data behavior without growing weight caches.

### Task 3: Acceptance

- [x] Benchmark actual calendar and frontier public methods against frozen baseline on identical offline data; retain distributions, memory and source digests.
- [x] Run affected tests and full suite, both Ruff commands and independent review under subscription admission.
- [x] Record results/limits, check Git locks and archive owned scratch evidence. Keep changes local without publishing.

## Execution evidence

Completed with strict RED/GREEN for container growth, repeated grouping, low-risk cancellation and high-mean cancellation. Final full suite: 710 passed, 3 skipped; both Ruff checks passed. Independent read-only Grok Build review found no blocking issues after an initial review timeout. Actual paired measurements and limits: [performance report](../../performance/calendar-batching.md). Source digests match measured code. Changes remain local; publication was not authorized. Scratch evidence archived outside the worktree after verification; Git locks checked without altering foreign locks.
