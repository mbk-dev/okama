# PortfolioDCF and MonteCarlo implementation plan

> Use `superpowers:executing-plans` inline with TDD; one writing parent and independent read-only review.

**Goal:** Implement every audited acceleration while preserving financial behavior and mutable-input correctness.
**Architecture:** Fresh owned arrays, guarded historical dispatch, paired MC outputs and a latest-draw revision/snapshot shared by derived caches, local statistical reuse and guarded shortcuts.
**Tech Stack:** Existing uv Python>=3.11, NumPy/pandas/SciPy, pytest/Ruff. No dependencies.
**Spec:** `docs/superpowers/specs/2026-10-26-dcf-mc-acceleration-design.md`.

## Review focus

Copy ownership and caller mutations; scalar index/name/dtype/error semantics; partial calendar bins and inflation; CWD/VDS and custom strategy overrides; public same-object invalidation; cached return matrix in-place edits/settings; covariance-free survival day convention; bootstrap ties/NaN/seed; IRR extreme/multi-sign/custom-guess fallback; paired caches/snapshot memory; actual public-call rather than kernel-only gains.

- [x] Freeze original numerical helpers and relevant high-level DCF/MC methods independently of runtime Git.
- [x] Stage A RED/GREEN: owned-array copies, seeded draws and output isolation.
- [x] Stage B RED/GREEN: guarded historical engine, cache preservation and batched asset histories, with full reference/edge coverage.
- [x] Stage C RED/GREEN: mutable-return revision, joint MC results, survival dates, post-plot caches and guarded IRR.
- [x] Stage D RED/GREEN: local fitting, safe CAGR cache, vector quantiles and bootstrap tails.
- [x] Actual baseline/candidate complete-method benchmarks, peak allocations and source hashes.
- [x] Full pytest, both Ruff commands, independent review and reconciled findings.

Delivery gate: scoped commits, dev PR CI/exact-SHA merge, synchronization, owned scratch archive and final Git lock check. The associated PR and external execution receipt record completion of this gate.
