# Conditional Rebalance implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` for inline implementation and a final independent review.

**Goal:** Accelerate actual threshold rebalancing with complete Result equivalence.
**Architecture:** Dispatch standard monthly float64 data to an array loop; retain the pandas fallback and the threshold-check hook.
**Tech Stack:** Existing Poetry Python >=3.11, NumPy, pandas and pytest; no dependencies.
**Spec:** `docs/superpowers/specs/2026-10-26-rebalance-conditional-design.md`.

## Review focus

- Events happen at the previous calendar month of the actual rebalance and prefer absolute over relative reasons.
- Strict threshold boundaries, zero targets/balances and NaNs retain existing behavior.
- Per-asset optional outputs, empty outputs, dtype/index/column metadata and initial wealth must match.
- Subclasses/nonstandard/indexed weights retain the old path.
- Calendar/EF calculations and earlier delivered local changes remain untouched.

### Task 1: Threshold check

- [x] Add direct ndarray-check and scalar-reference coverage in `tests/test_rebalance_performance.py`.
- [x] Observe RED for missing ndarray support and implement the minimal equivalent check branch.
- [x] Verify GREEN with existing threshold-hook tests.

### Task 2: Conditional wealth loop

- [x] Add a failing constant-container-work test and complete Result equivalence tests against frozen Git source.
- [x] Add array dispatch and preallocated loop; preserve the original pandas function as fallback.
- [x] Verify hand-calculated event timing/priority, exact boundaries, gaps, zeros/NaNs, asset outputs, metadata, hooks/subclasses and fallback.
- [x] Run affected Rebalance/Portfolio modules and the full suite.

### Task 3: Acceptance

- [x] Benchmark actual public methods with portfolio, asset and event outputs against frozen source on identical offline inputs; retain timings and source digests.
- [x] Independently review the completed diff through the subscription policy and verify findings.
- [x] Run both Ruff commands, inspect diff/locks and archive owned scratch evidence.
- [x] Write the measured report and report local delivery state without publication.

## Execution record and rulings

- Direct ndarray checks and bounded Series-construction requirements were RED, then GREEN. Full Result comparisons caught the version-dependent string dtype; inferred dtype now matches the original implementation.
- Ruling: retain the pandas path for float32/extended/indexed weights and unusual input/subclass cases. A float32-weight regression test proved that conversion before initial allocation changes values.
- Ruling: check the original scalar reference into test helpers rather than reading Git history at test time, preserving compatibility with shallow CI checkouts and source distributions. The analytical benchmark still loads the exact Git baseline.
- Representative 600x16 absolute-threshold complete asset result: 372.192 ms to 7.324 ms; tracked peak 2.350 MiB to 0.182 MiB. Full outputs agree, with zero recorded portfolio-wealth differences.
- Independent Grok review found no significant defects. Full suite: 599 passed, 3 skipped, 200 existing warnings; both Ruff commands and diff checks pass. No external mutations or delivery operations.
