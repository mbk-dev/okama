# Idle issue implementation plan — 2026-10-26

Development base: `bcd877421262d107b7b35a9a771498802dc39c0b`, `origin/dev`.
The public release branch is `master`; development work targets `dev`.

## Scope and ownership

The owner requested all currently idle GitHub issues and subagents where possible.
GitHub returned seven open issues and no open pull requests. Orca confirms another
live coordinator owns portfolio acceleration (#127); its work is excluded.
This is a set of independent issues, not a new epic. Use isolated parallel tasks,
not the generic epic-wave loop (this repository has no specialization).

| Issue | Outcome | Execution / dependency |
|---|---|---|
| #133 | Typed, warning-free rebalancing event values and monthly index, including empty events | Independent fix and offline regressions |
| #131 | Literal punctuation in namespace search; nullable fields and return formats preserved | Independent fix and offline regressions |
| #132 | Bound-feasible frontier targets, both frontier modes and asymmetric bounds | Independent fix and offline regressions |
| #129 | Every current dependency alert receives a source-backed disposition; compatible ranges fixed | Coordinator; inspect current alerts first |
| #125 | Correct ETF distributions and explicit data-quality handling | Inspect live API and upstream ownership before implementation |
| #126 | Personal mortgage portfolio request | Closed as outside the library scope, per the owner's answer |
| #127 | CPU batching | Excluded: live work in acceleration workspace |

## Acceptance checkpoint

- #131: literal search implemented by an isolated Grok worker and independently
  inspected and integrated by the coordinator.
- #133: event typing implemented by another Grok worker; parent review caught
  an additional conditional-path PeriodIndex warning and removed it with a new
  failing regression before the correction. 110 related tests passed cleanly.
- #132: bounded endpoint optimization implemented with test-first reproduction;
  restrictive symmetric/asymmetric two- and three-asset regressions pass. The
  original fixed-history SPY/GLD API example succeeds in both modes; see
  [the reproduction and result](frontier-bound-targets.md).
- #129: [78-alert advisory ledger and dependency floors](../security/dependency-alerts-2026-10-26.md)
  prepared. The built wheel includes the runtime floors; installed versions are
  outside all 78 captured affected ranges. Remote reconciliation awaits delivery.
- #125: a narrow missing-history warning is implemented; actual data remediation
  remains open in [okama-API #545](https://github.com/mbk-dev/okama-API/issues/545).
- #126: closed as `not_planned` after the owner's explicit answer; readback verified.
- #127: existing active work remains with its original coordinator.

Each full suite on Python 3.11, 3.12, 3.13 and 3.14 passed 922 tests with zero
failures/errors and three opt-in manual benchmark skips. Remaining warnings in
other modules are tracked in [#138](https://github.com/mbk-dev/okama/issues/138);
the DCF portion overlaps the active performance coordinator and is not duplicated.
See [machine-readable validation](2026-10-26-validation.json).

Both Ruff checks and wheel/sdist builds passed. The production/manifest patch
hash still matches the four-version test matrix. An isolated complete tracked
snapshot of newer `origin/dev` (`370204bf7db4901626281b3ce624edbb11f84450`) plus
this patch passed 1,083 tests with three manual benchmark skips and no failures
or errors on Python 3.14. The only overlapping changed file is `CHANGELOG.md`;
both sets of release notes must be retained during integration.

Independent final review used the native Codex reserve after fresh Grok usage
reached 86%, Go quota returned HTTP 403, actual OpenCode free attempts returned
HTTP 403 directly and through the proxy, and an isolated OpenRouter/OpenCode CLI
attempt received an upstream Nvidia HTTP 404. The preceding local context-gate
timeout was retained separately and was not counted as a provider failure.
No paid API fallback was used. The owner subsequently authorized scoped commits,
publication and integration into `dev` through a pull request. Final delivery
and remote alert state are recorded in that pull request and the linked issues.

The independent reviewer reported no actionable findings in the five scoped
changes and completed 22 additional offline three- and four-asset frontier
probes. Checked CAGR discrepancies were below 5.5e-7. Advisory verification was
not repeated by that reviewer; the per-alert ledger remains the coordinator's
evidence. Endpoint optimization is numerical, not a global-optimality guarantee.

## Implementation and acceptance

For each code change: inspect source, reproduce with an offline failing test,
then implement the smallest compatible fix. Workers own separate worktrees and
paths and must not commit, publish, deploy, write to production or redelegate.
Select actual external CLI workers using fresh subscription admission. Disable
browser MCPs, cap numerical threads and retain complete logs outside the repository.
The parent reviews every diff and reruns relevant regressions, the complete suite,
`uv run ruff check .`, and `uv run ruff format --check .`. Check Python 3.11 compatibility.
No lockfile is committed. Dependency changes update both manifests.

Keep issues open until accepted implementation is delivered. Preserve concrete
upstream/input blockers and verify remote dependency alerts after publication.

## Authorized publication acceptance

The owner authorized committing and merging via a PR into `dev`. Subsequent
changes in `dev` through `6af312dccd01ceb0ba942e6dbb0b424a7d68bc23` were merged
without conflicts. Rebalancing and frontier changes remain scoped to their
original fixes; upstream optimizer preparation and pairwise caches are retained.
Both local Python 3.11 and 3.14 suites passed 1,157 tests each with three
manual benchmark skips and no failures/errors. Both Ruff checks passed.
The existing CI matrix supplies the final Python 3.11–3.14 PR checks.

## Split delivery

The owner requires native Codex review inputs at most 500,000 serialized bytes,
OpenRouter for larger inputs up to 1,000,000 bytes, and separate PRs above that
ceiling. The original aggregate candidate is superseded by three complete parts:
search/frontier, rebalancing, and missing-distribution mitigation/dependencies.
The final delivery record lives in the linked GitHub issues and PRs. Historical
pre-split validation above remains evidence for the combined implementation;
each split candidate also receives the required CI and independent acceptance.
