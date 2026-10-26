# Scoped API request reuse plan

> Use `superpowers:executing-plans` for inline TDD and independent final review.

**Goal:** Reuse HTTP connections within logical loading operations without persistent stale metadata or time-series caching.
**Architecture:** ContextVar-owned bounded metadata and thread-local sessions; automatic constructor decorators; explicit borrowed scope in ListMaker workers; optional outer public context.
**Tech Stack:** Existing requests/stdlib contextvars/threading, uv Python>=3.11. No dependencies.
**Spec:** `docs/superpowers/specs/2026-10-26-api-request-scope-design.md`.

- [x] Write RED for session reuse within a real Asset constructor and scoped metadata behavior.
- [x] Implement scope/session lifetime and API integration with unchanged transport settings/errors.
- [x] Decorate Asset/ListMaker and propagate only the request scope into ListMaker workers.
- [x] Verify nested ownership, cleanup, threads, bounded metadata, errors/URLs and no return-range cache.
- [x] Measure transport counts and bounded live A/B; run full pytest, both Ruff commands and independent review.
- [x] Commit, pass dev CI, exact-SHA merge and synchronize owned/default checkout; archive evidence and check Git locks.

## Execution evidence

Initial constructor-session RED verified duplicate factory calls; scoped tests and final full-parameter metadata-key RED/GREEN retained. An affected-test harness incorrectly expected uppercase default period and then omitted recorded transport settings; both fixture assumptions were corrected within two repair cycles, with no production change to defaults. Final full suite: 922 passed, 3 skipped, 172 existing warnings; both Ruff checks passed. Independent full scope and focused final-key reviews found no blocking issues. Offline counts11sessions→5automatic/4outer, requests11→10outer; final live three rotating repetitions compared complete outputs exactly. See [report](../../performance/api-request-scope.md). First calculation batch is delivered in PR134; API follow-up is committed and integrated into dev after CI. Cleanup evidence is external; active/foreign worktrees remain protected.
