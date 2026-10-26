# ListMaker performance implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` for inline TDD and an independent final review.

**Goal:** Remove measured ListMaker overhead while preserving observable financial results and validation.

**Architecture:** Fast-path ready asset validation and monthly dividend-window arithmetic; preserve existing I/O, conversion and caching paths.

**Tech Stack:** Existing Poetry Python >=3.11, pandas, NumPy, pytest; no dependencies.

**Spec:** `docs/superpowers/specs/2026-10-26-listmaker-performance-design.md`.

## Review focus

- Mixed inputs must retain threaded loading and key/order/identity behavior.
- Validation still rejects insufficient history, including objects without `pl`.
- Calendar windows must not become row windows when dividends or prices have gaps.
- FX conversion, forecast removal, zero prices, net-zero dividends and cache behavior stay compatible.
- Index metadata and unusual-input scalar fallback must remain compatible.

### Task 1: Ready-object validation

- [x] Add tests in `tests/test_list_maker_performance.py` for no pool work on ready objects and for mixed inputs/order/validation.
- [x] Observe RED for the no-pool requirement, with real validation and actual loading behavior.
- [x] Add minimal sequential ready-object branch in `okama/common/make_asset_list.py`; keep symbol loading unchanged.
- [x] Verify GREEN for the new file and existing ListMaker tests.

### Task 2: LTM dividend yield

- [x] Add hand-calculated and frozen-reference tests for calendar windows, price/dividend gaps, FX, signs/NaNs/zero price, forecasts, cache and fallback.
- [x] Observe RED for removal of per-month DateOffset work on normal monthly inputs.
- [x] Extract private `_get_ltm_dividend_yield(dividends: pd.Series, prices: pd.Series) -> pd.Series` with a calendar-aligned rolling fast path and original fallback.
- [x] Verify GREEN for new tests, ListMaker and AssetList modules without new warnings.

### Task 3: Evidence and verification

- [x] Benchmark full ready-object constructors and actual dividend-yield calls against frozen original source with identical offline inputs and warm data. Record distributions, equivalence, versions, revisions/digests and cache limits in a checked-in report.
- [x] Run full tests and both Ruff commands; obtain/verify independent review under subscription routing.
- [x] Inspect final diff and Git locks; archive owned scratch evidence. Report local delivery state, without committing or publishing.

## Execution record and rulings

- Ready-object and DateOffset performance requirements were observed RED and then GREEN.
- Ruling: limit rolling arithmetic to float64 dividends without infinities; preserve scalar behavior for float32, nullable and nonstandard inputs. Numerical tests caught the incompatible rolling cases before completion.
- Ruling: remove the old module-reload workaround and add explicit portfolio/MC/FinPlan test package boundaries. Reproductions proved both class-identity interference and package-fixture leakage; the final full suite passes without weakening validation tests.
- Actual-method frozen-source benchmark, raw distributions and source digests are recorded in `docs/performance/listmaker-optimizations.md`. Grok independent review completed without significant findings; source conclusions were verified by the coordinator.
- Final tests: 568 passed, 3 skipped, 203 existing warnings. Ruff lint/format and diff checks pass. No commits, pushes, tracker mutations or deployments.
