# Conditional Rebalance performance implementation

Implemented 2026-10-26 as the next stage after [ListMaker](listmaker-optimizations.md) and the [opportunities audit](additional-opportunities.md). This change accelerates threshold-based rebalancing with `period="none"`. Calendar rebalancing, unconditional buy-and-hold and multi-weight efficient-frontier calculations keep their existing paths; no speedup is claimed for those paths.

## Implementation and compatibility

`Rebalance._rebalance_by_condition()` now dispatches standard inputs to a chronological NumPy loop with preallocated portfolio wealth and optional asset-wealth arrays. The threshold decision remains chronological; it is not replaced by an independent-row matrix formula. The loop still calls `_check_if_rebalancing_required()`, which now accepts ndarray inputs alongside its existing pandas inputs. Pandas Series/DataFrames are constructed once after computation.

The fast path requires an exact standard `Rebalance` instance, nonempty sorted unique monthly PeriodIndex, float64 return columns and positional float64 weights (or binary integer weights). Float32/extended/indexed/nonstandard weights or return types, duplicate/unsorted indexes and custom subclasses retain the frozen pandas path. The float32-weight fallback is necessary: converting weights before multiplying by the initial investment changes the original rounded allocation. No normalization, clipping, solver, financial formula, public signature or dependency changes were introduced.

Full `Result` compatibility covers:

- Portfolio wealth, optional per-asset amounts, the initial investment point, empty asset outputs and index/column metadata.
- Event dates at the previous calendar month of the actual rebalance, deferred execution until the next observed return row, and absolute-before-relative reason priority.
- Strict `>` comparisons, zero thresholds, zero target weights and balances, missing returns, and missing calendar observations.
- Empty-event numeric dtype and version-appropriate inferred string dtype for nonempty events. The initial implementation's forced object dtype was caught by frozen-source comparisons and corrected.
- Caller return data and read-only weight arrays remain unchanged. Subclass hooks expecting pandas objects continue on the legacy path.

The original pandas implementation remains as `_rebalance_by_condition_pandas()`. Default finite float64 arithmetic is unchanged; `np.nansum()` retains pandas' skip-NaN sum behavior, and invalid divisions used in threshold comparisons retain NaN/inf comparison behavior without new NumPy warnings.

## Measured public methods

The benchmark loads the complete baseline Rebalance module from Git revision `b33ca12bdd81b4107c2de484fff581eda1023ce1` and compares actual public calls against the current implementation. It verifies complete portfolio/assets/events outputs before timing; `return_ror_ts()` and `assets_weights_ts()` are also compared directly. This is a stronger measurement than the audit prototype, which omitted assets and event construction.

Protocol: identical offline monthly float64 inputs, seed 127, 252/600/1200 months and 3/16/32 assets, one warmup pair followed by five timed pairs with alternating baseline/candidate order. Absolute-only (0.05), relative-only (0.1) and combined thresholds are measured, with both optional asset-output settings. Timings include allocations, threshold checks, event construction and returned pandas objects, but exclude input generation and baseline module setup. Rebalance itself has no result cache in these calls.

Representative results for 600 months × 16 assets:

- Portfolio wealth and events, absolute threshold: 301.386 ms → 7.283 ms (41.4x).
- Including per-asset wealth: 372.192 ms → 7.324 ms (50.8x).
- `return_ror_ts()` with combined thresholds: 418.280 ms → 11.742 ms.
- `assets_weights_ts()` with combined thresholds: 606.153 ms → 11.107 ms.

All complete pandas outputs agree at `rtol=atol=1e-12`, including exact event/index/column/dtype checks. Maximum recorded portfolio-wealth absolute difference over the measured cases is 0.

Separate warm-call `tracemalloc` measurement with per-asset output: 2.350 MiB → 0.182 MiB. This is tracked Python/NumPy allocation peak during the call, including returned objects; it is not total process RSS and excludes resident input data/imports. The no-asset-output path does not allocate a periods-by-assets result matrix. Output memory remains O(months × assets) when per-asset wealth is requested.

Environment: Python 3.11.17, NumPy 2.4.6, pandas 3.1.0, AMD Ryzen 7 8845H on Linux. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4` were set only for benchmark processes. Scheduling, CPU frequency and background load on the shared machine remain uncontrolled; these ratios apply to the measured workload, not the entire library or an API request.

Measured/final source SHA-256: `bfe7fc408cdae688a74e23e73abbebadedf86db465b814d3ac844d5e27deb097`. Benchmark script SHA-256: `e3bc3883f838695395d73ef25ccfc04e8c1515ee71f9b97517e7bd74138b3e15`.

## TDD and independent review

Direct array threshold tests first failed because the existing hook required pandas objects. The real container-work test then showed 130 Series constructions for 12 months and 374 for 36 months; the implementation now keeps container work bounded independently of month count. Full financial comparisons use a checked-in scalar reference copied from the original source, so CI tests do **not** depend on Git history availability. Hand-calculated tests independently pin event dates, reason priority, initial/per-asset values and exact threshold boundaries.

Coverage includes absolute/relative/combined/zero thresholds, optional asset output, zero portfolio balances and target weights, NaNs, gaps, binary integer and float32 weights, readonly inputs, nonstandard dtype/index fallbacks and subclass callbacks. Existing Rebalance and Portfolio tests remain part of verification.

Independent read-only Grok review completed in one turn with no significant findings. The coordinator verified the conclusions against the implementation, tests and full-output benchmark comparisons. No worker edits or browser calls occurred.

## Complete timings

Medians in milliseconds; all individual repetitions are in the accompanying JSON.

| Case | Baseline ms | Candidate ms | Ratio |
|---|---:|---:|---:|
| `wealth_abs_252x3_assetsFalse` | 143.555 | 3.953 | 36.3x |
| `wealth_abs_252x3_assetsTrue` | 175.778 | 4.385 | 40.1x |
| `wealth_rel_252x3_assetsFalse` | 148.175 | 4.275 | 34.7x |
| `wealth_rel_252x3_assetsTrue` | 201.193 | 5.122 | 39.3x |
| `wealth_both_252x3_assetsFalse` | 171.693 | 5.787 | 29.7x |
| `wealth_both_252x3_assetsTrue` | 195.521 | 5.165 | 37.9x |
| `wealth_abs_600x16_assetsFalse` | 301.386 | 7.283 | 41.4x |
| `wealth_abs_600x16_assetsTrue` | 372.192 | 7.324 | 50.8x |
| `wealth_rel_600x16_assetsFalse` | 448.836 | 9.159 | 49.0x |
| `wealth_rel_600x16_assetsTrue` | 647.348 | 10.394 | 62.3x |
| `wealth_both_600x16_assetsFalse` | 547.259 | 13.298 | 41.2x |
| `wealth_both_600x16_assetsTrue` | 548.962 | 11.304 | 48.6x |
| `return_ror_ts_600x16` | 418.280 | 11.742 | 35.6x |
| `assets_weights_ts_600x16` | 606.153 | 11.107 | 54.6x |
| `wealth_abs_1200x32_assetsFalse` | 699.160 | 13.632 | 51.3x |
| `wealth_abs_1200x32_assetsTrue` | 840.354 | 13.957 | 60.2x |
| `wealth_rel_1200x32_assetsFalse` | 1134.623 | 19.366 | 58.6x |
| `wealth_rel_1200x32_assetsTrue` | 1352.348 | 19.376 | 69.8x |
| `wealth_both_1200x32_assetsFalse` | 1228.657 | 23.218 | 52.9x |
| `wealth_both_1200x32_assetsTrue` | 1513.926 | 23.126 | 65.5x |

## Reproduction and delivery

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 poetry run python -m analytics.benchmark_rebalance --output tmp/rebalance.json
poetry run pytest -q
poetry run ruff check .
poetry run ruff format --check .
```

See [the benchmark script](https://github.com/mbk-dev/okama/blob/v4.0.1/analytics/benchmark_rebalance.py) and [raw distributions/source digests](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/rebalance-conditional.json). The benchmark requires the named historical source in Git; unit tests use the independent checked-in scalar reference and need no history access.

Earlier local frontier/ListMaker implementations remain intact. No commit, push, merge, dependency update, tracker mutation or deployment has been made.

## Final verification

`poetry run pytest -q`: **599 passed, 3 skipped**, 200 remaining existing pandas deprecation warnings (stage baseline: 568 passed, 3 skipped, 203 warnings). The new performance tests pass without warnings. Both Ruff commands and `git diff --check` pass. The existing baseline input/output contract is retained; no calendar/EF acceleration is included in this stage.
