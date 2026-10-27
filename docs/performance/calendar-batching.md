# Calendar Rebalance and multi-period frontier batching

The implementation accelerates `Rebalance.wealth_ts()` with calendar periods and `okama.frontier.multi_period.EfficientFrontier.get_monte_carlo()` / `get_grid_portfolios()`. Earlier single-period, ListMaker and conditional Rebalance changes remain separate.

## Implementation and compatibility

Calendar wealth uses NumPy compounding over the original pandas resample boundaries. Thresholds are checked only at period ends; skipped rebalances retain the original divide/multiply carry. Events, partial first periods, zero-threshold truthiness, index metadata and optional asset paths are preserved. Complete finite float64 monthly data and compatible positional weights qualify; unsupported dtypes, gaps, missing data and subclasses retain the original pandas implementation.

Repeated multi-period weights share one request-local EF grouping and growth preparation. EF Grouper boundaries remain distinct from calendar resampling, matching the original methods. Chunks contain at most 1,024 portfolios and 262,144 periods-by-portfolios elements per matrix; histories exceeding the latter use scalar calculations. No persistent cache or public configuration is added. The limit bounds each intermediate matrix, not total process memory, prepared inputs or the complete output table.

Nonstandard inputs and exceptional growth use scalar calculations. Individual portfolios with low volatility or numerically sensitive annual-risk subtraction also use the original scalar formulas. Regression tests cover both ordinary low volatility and unusually high means with small variance. Short histories retain object-typed `None` CAGR. Labels, duplicate-label last-wins behavior, bounds, seeded generation order and empty-result validation remain compatible. Sampling does not grow the per-weight optimizer cache.

## Measurement protocol

`analytics/benchmark_calendar_batching.py` loads the original implementations from revision `b33ca12bdd81b4107c2de484fff581eda1023ce1`. Offline seeded data and identical ready assets avoid network variability. Timing covers complete public calls, including weight generation and output construction, with one warm pair and five alternating timed pairs. It compares financial values with relative and absolute tolerance 1e-12 and checks complete result metadata. The frozen test reference is checked in and requires no Git history at test time.

Environment: AMD Ryzen 7 8845H, Python 3.11.17, NumPy 2.4.6, pandas 3.1.0. BLAS/OpenMP threads are limited to one for this command only. Measurements run on a shared host; other activity and load cause dispersion. Results describe these workloads, not a universal whole-library speedup. JSON retains distributions and source digests.

Memory figures use separate warm `tracemalloc` calls and include newly allocated output. Existing inputs/imports and native allocations not traced by Python may be excluded; these are not process RSS measurements. Batching deliberately exchanges additional bounded temporary memory for speed.

## Results

| Workload | Original median | Candidate median | Speedup |
|---|---:|---:|---:|
| Calendar monthly, 600 months × 16 assets, full asset paths | 733.891 ms | 13.415 ms | 54.7× |
| Calendar yearly, 5% absolute threshold, same history | 32.335 ms | 3.127 ms | 10.3× |
| Yearly MC, 252 months × 3 assets, 128 portfolios | 291.295 ms | 2.753 ms | 105.8× |
| Yearly MC, same history, 2,048 portfolios | 4,693.626 ms | 18.020 ms | 260.5× |
| Yearly MC, same history, 8,192 portfolios | 19,795.167 ms | 65.835 ms | 300.7× |
| Yearly grid, same history, step 1/100 (5,151 portfolios) | 12,578.275 ms | 30.520 ms | 412.1× |

All 29 paired workloads passed the equivalence checks. Maximum absolute difference: calendar wealth 2.91e-11 (absolute currency units), frontier metrics/weights 1.915e-14. The test uses both relative and absolute tolerance 1e-12; calendar wealth is much larger in magnitude than returns, so these absolute errors do not imply a failed comparison. Event values and dates are compared exactly.

Monthly calendar full-path allocation peak fell from 3.074 MiB to 0.366 MiB. Yearly MC 2,048-portfolio allocation peak increased from 1.404 MiB to 10.440 MiB; candidate 8,192-portfolio peak was 11.530 MiB. This is the batching memory tradeoff, not a memory reduction for frontier sampling. The table and full distributions in [calendar-batching.json](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/calendar-batching.json) describe actual public methods.

## Verification

Strict TDD included reproducing per-period pandas-container growth, repeated EF grouping, low-risk cancellation and high-mean cancellation before implementing their fixes. Financial references include original scalar methods rather than comparisons only between chunk sizes. Final-code Grok Build review (`grok-4.6-build`, read-only, one turn) reported no blocking findings; an earlier oversized review timed out and is retained as an unsuccessful attempt.

Final verification: `poetry run pytest -q` — **710 passed, 3 skipped, 174 pre-existing pandas warnings** (baseline: 599 passed, 3 skipped, 200 warnings). No test failures. Both `poetry run ruff check .` and `poetry run ruff format --check .` passed. `git diff --check` passed. No dependencies or global settings changed. This development-stage work was subsequently merged in [PR #134](https://github.com/mbk-dev/okama/pull/134) and released in [okama 4.0.1](https://github.com/mbk-dev/okama/releases/tag/v4.0.1).
