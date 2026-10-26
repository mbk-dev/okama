# Remaining frontier and workflow implementation plan

> Use `superpowers:executing-plans` for inline TDD with a final independent review.

**Goal:** Accelerate remaining frontier optimizer statistics and measure complete asset-to-portfolio CPU work.
**Architecture:** Per-solve pandas covariance/means, unchanged solver math and settings; frozen prior-method reference and staged actual public workflow benchmarks.
**Tech Stack:** Existing Poetry Python >=3.11, NumPy/pandas/SciPy, pytest and Ruff; no new dependencies.
**Spec:** `docs/superpowers/specs/2026-10-26-frontier-workflow-design.md`.

## Review focus

Keep tangency geometric and arithmetic risk conventions distinct; preserve solver calls/settings, failure/validation order, NaN handling and target conversions; retain corner fallback and label semantics; statistics cannot leak across changed inputs; distinguish warmed caches, CSV fixture loading and network time in measurements.

### Implementation
- [x] Freeze original three optimizer methods in `tests/helpers/optimizer_reference.py` without test-time Git dependency.
- [x] Write RED for repeated covariance/means in `minimize_risk`, tangency and return-range optimization; verify real assertion failures.
- [x] Prepare invariant pandas statistics locally and use unchanged objective/constraint formulas.
- [x] Verify original weights/metrics across labels, modes, targets, missing returns, corner fallback, bounds and errors.

### Complete workflow and acceptance
- [x] Benchmark actual public methods and complete CSV-to-ready-assets-to-frontier-to-Portfolio workflow, keeping original formulas and stage timings.
- [x] Run full pytest, both Ruff commands and diff check; obtain and reconcile independent read-only review.
- [x] Write measured report and remaining bottlenecks with limitations; archive owned scratch and verify Git locks. Retain changes locally.

## Execution evidence

Five RED assertions verified repeated statistics; affected suite GREEN. Final full suite: 874 passed, 3 skipped, 174 existing warnings. Both Ruff commands and diff check passed. Independent `grok-4.6-build` read-only review found no blocking issues; existing case sensitivity was correctly retained. Ten incremental actual-method/complete-CPU cases passed with zero measured output difference, final production/reference/script digests match. The initial workflow mismatch came from random Portfolio symbols; corrected fixture and complete rerun retained.

Ruling: add supplemental current-candidate live API observations to cover real symbol loading, separate from paired CPU timing — HTTP variance and parallelism make their timings unsuitable as a before/after comparison. Final observation completed successfully, documented request date differences prevent a false claim of identical USD return requests. See [report](../../performance/frontier-workflow.md). Evidence archived externally with SHA receipt; Git locks checked. All changes remain local and undelivered; no commit or external mutation.
