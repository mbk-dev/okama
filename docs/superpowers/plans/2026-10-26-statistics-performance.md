# GMV and rolling CAGR implementation plan

> Use `superpowers:executing-plans` for inline TDD; parent is sole writer and obtains independent final review.

**Goal:** Remove repeated statistics/solves in GMV and per-window Python CAGR callbacks.
**Architecture:** Per-solve pandas statistics and single-result reuse; separate rolling CAGR helper with guarded logarithmic path and original callback fallback.
**Tech Stack:** Existing Poetry Python >=3.11, NumPy/pandas/SciPy, pytest and Ruff. No dependencies.
**Spec:** `docs/superpowers/specs/2026-10-26-statistics-performance-design.md`.

## Review focus

Retain pairwise NaN covariance and exact multiplication order; bounds and unsuccessful-solver errors; mutation freshness; zero/negative/infinite growth and product underflow/overflow; short/invalid windows and complete metadata; generic callbacks and existing real/inflation semantics.

### GMV
- [x] Freeze original four GMV properties in a checked-in test reference.
- [x] Write RED for covariance preparation once and one solve per derived result; verify assertion failures.
- [x] Prepare covariance/means locally and reuse derived results without changing objective or settings.
- [x] Verify original weights/results, missing returns, changed data/bounds and failure behavior.

### Rolling CAGR
- [x] Write RED for eliminating callbacks in complete public AssetList/Portfolio methods.
- [x] Add explicit guarded `Frame.get_rolling_cagr` and route CAGR call sites only.
- [x] Verify scalar equivalence across windows/types, missing values, losses/overflow, real returns, metadata and unchanged validation/generic callbacks.

### Acceptance
- [x] Measure actual public calls against frozen source; retain distributions and source digests.
- [x] Run full pytest, both Ruff checks, diff check and independent final review.
- [x] Record final results, archive owned scratch and check Git locks. Keep changes local.

## Execution evidence

RED verified 8 assertions for repeated covariance/solves and callbacks. GREEN and original-scalar/edge comparisons passed. Final full suite: 825 passed, 3 skipped, 174 existing warnings. Both Ruff commands and diff checks passed. Twelve actual-method paired benchmarks, final source/script hashes verified; see [report](../../performance/statistics-performance.md). Grok read-only review completed; sub-year finding rejected with explicit validator and 1–11-window tests. No production changes followed measurement. Initial long-history currency fixture truncation was detected by shape assertion; discarded timings and corrected complete rerun retained in evidence. Owned scratch archived externally with SHA verification; Git locks checked. No delivery authorized; all earlier local work retained.
