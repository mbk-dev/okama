# Resolve current okama issues

> **For agentic workers:** Use superpowers:subagent-driven-development. Implement each group in its isolated worktree, then review its diff.

**Goal:** Resolve the library-owned gaps in open issues #113–123 after confirming applicability against origin/master 6a2c4a4.
**Architecture:** Keep pandas Series and common-history semantics stable, expose new accessors and reusable FinPlan paths, validate silent data losses, and clarify cash-flow conventions.
**Tech Stack:** Python >=3.11, Poetry, pandas, NumPy, pytest, Ruff.
**Spec:** The GitHub issue bodies at https://github.com/mbk-dev/okama/issues are the requirements; this plan records compatible choices where alternatives are offered.

## Global constraints
- English code/docstrings/documentation, Python 3.11 compatible, no new dependencies.
- RED/GREEN for executable changes; no tests required for docstring-only changes.
- Implementation review precedes publication. The follow-up user request on 2026-10-09 authorizes committing all changes and publishing a new release; upstream API changes remain outside scope.
- Each writer owns one isolated worktree. Reports stay in tmp; parent owns this plan and integration.
- Full pytest and both Ruff commands required on each executable change and the integrated branch.

## Task 1: Asset access and CAGR (#121, #122, #123)
- [x] Add Asset.get_cagr(period=None, real=False) returning a scalar in the asset currency; reuse existing period validation/formulas; define inflation alignment for real results.
- [x] Add monthly close accessor with a KeyError naming requested month and available bounds; retain ordinary close_monthly Series.
- [x] Warn explicitly when inflation shortens an explicitly requested last_date in ListMaker; preserve common-period semantics; clarify get_cagr period/result shape and inflation=False workaround.
- [x] Test full history/trailing CAGR equivalence, before/after close-range errors, clipping warning and untrimmed nominal window.
- [x] Verify full pytest and Ruff; produce report with RED/GREEN evidence and diff.

## Task 2: FinPlan (#113, #115, #116, #118, #119, #120)
- [x] Validate all stage end dates; append optional t0 argument preserving existing positional parameters; report mismatched stage and dates.
- [x] Expose plan-resolved stage month indices; reject out-of-stage TimeSeries keys before simulation and revalidate after strategy edits.
- [x] Log automatic discount-rate fallback and test explicit/inflation/default rates, retaining existing readable property.
- [x] Expose reusable per-stage simulated returns through public API and allow rerunning flows with the supplied paths without redrawing; validate index/shape/numeric content, defensive copies, seed parity.
- [x] Explain cumulative survival shares, opening row and raw-flow verification; retain cash-flow masking default; docs include reading goals and goal-size bisection with reused paths.
- [x] Verify regression tests, full pytest and Ruff; produce report with RED/GREEN evidence and diff.

## Task 3: Cash-flow strategies (#114, #117)
- [x] Default only TimeSeriesStrategy.time_series_discounted_values to True (verbatim nominal forecast amounts). Explicit flags retain meaning.
- [x] Document both forecast/backtest meanings and migration in CHANGELOG.
- [x] Remove the IndexationStrategy withdrawal upper bound based on its own initial investment, retaining numeric validation; don't change unrelated strategy checks.
- [x] Test both conventions and probabilities, new default, and withdrawals larger than the strategy default within FinPlan.
- [x] Verify full pytest and Ruff; produce report with RED/GREEN evidence and diff.

## Task 4: Review and integrate
- [x] Review each worker diff independently against full issue DoD and compatibility constraints.
- [x] Apply reviewed patches to this worktree, resolve shared documentation/tests carefully.
- [x] Run integrated pytest, both Ruff checks and targeted regression evidence.
- [x] Record #125 as upstream-owned and unresolved; do not fabricate payout data or ETF classifications.

## Review focus
- Trailing CAGR boundaries match AssetList and reject invalid periods.
- Return paths retain month/scenario shape and cannot mutate cached data.
- TimeSeries validation catches swapped dictionaries and later in-place edits.
- New defaults have a migration record and explicit flags retain forecast/backtest semantics.
- AssetList warns only for real inflation clipping, not matching date windows.

## Execution result

Implemented locally in branch `новый-релиз`, with source branch/worktree groups
`fix/issues-asset-cagr`, `fix/issues-finplan`, and `fix/issues-cashflow`.
All library-owned gaps in issues #113 through #123 were implemented and
independently reviewed. The first implementation pass left the reviewed changes uncommitted.
The subsequent user request authorizes commits and release publication.

Integrated verification on Python 3.11.17:

- `poetry run pytest -q`: 518 passed, 3 skipped in 18.65 seconds.
- `poetry run ruff check .`: all checks passed.
- `poetry run ruff format --check .`: 92 files already formatted.
- `git diff --check`: clean.
- `docutils.core.publish_doctree` on `docs/finplan.rst`: no warning/error messages.
- Independent final review: no Critical or Important findings.

Issue #125 remains unresolved: the library consumes API distribution data and
cannot supply missing ETF payouts or infer a reliable payout-coverage flag.
No upstream data ingestion or API changes are part of this local patch.

Compatibility decisions: inflation clipping retains the common history and
warns (#122); TimeSeriesStrategy alone defaults to nominal forecast amounts
(#114); flow masking remains the default with a documented raw opt-out (#120);
flow-key validation occurs when a forecast or backtest is selected to preserve
historical ledgers (#113).

Minor deferred test gap: the clipping warning is covered, but there is no new
paired assertion that an inflation=True matching history emits no warning.
The reviewed month-bound condition is correct; this did not block review.

## Release follow-up

The release is 4.0.0 because the default of
`TimeSeriesStrategy.time_series_discounted_values` changes behavior.
The three implementation groups were committed separately after staged secret
checks. Dependencies were refreshed with `poetry update`; no updates were needed.
The release additionally requires unit tests, all notebook examples, local
Sphinx documentation, the supported-Python CI matrix, version-specific hosted
documentation, and PyPI artifact verification.

Release validation recorded before publication:

- Python 3.11.17 unit suite: 518 passed, 3 skipped.
- All 11 example notebooks passed via nbmake with the current Poetry kernel.
- Minimum-Python suite passed; supported-Python CI remains a publication gate.
- Both Ruff checks and all-file pre-commit checks passed.
- Wheel and source archive built successfully; wheel version is 4.0.0 and
  Requires-Python is >=3.11,<4.0.0.
- Hosted documentation and registry publication are verified after pushing
  the release tag; their authoritative records are the external build and
  registry results rather than a mutable local checklist.
