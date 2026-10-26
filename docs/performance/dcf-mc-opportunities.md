# PortfolioDCF, helpers and MonteCarlo performance audit

Audited dev revision `789efbb5d661610670598b58250f003fedbd171f`. This is research with scoped benchmark-process prototypes: **no production library implementation is changed by this audit**. Numbers below are opportunities, not released speedups.

Implementation and actual complete-method results are now documented in [DCF/MC acceleration](dcf-mc-acceleration.md). Reproduce this historical prototype audit from `bcd8774`.

## Priority 1: avoid copying newly owned simulation arrays into DataFrames

Targets: `generate_returns_ts()` in [mc.py](https://github.com/mbk-dev/okama/blob/v4.0.1/okama/portfolios/mc.py), and `get_wealth_indexes_fv_with_cashflow_mc()`, `get_cash_flow_fv_mc()` and `zero_wealth_after_first_void()` in [dcf_calculations.py](https://github.com/mbk-dev/okama/blob/v4.0.1/okama/portfolios/dcf_calculations.py).

These functions allocate fresh NumPy results, then construct a DataFrame with its default copy policy. On the installed pandas 3.1.0, an ndarray input is copied by default; the ownership probe confirms that the default constructor does not share its input, while explicit `copy=False` does. A wide MC-pair profile spent about 0.55 of 0.62 seconds in DataFrame construction / `numpy.array`, rather than the simulation engine. This is an observed version/shape-specific bottleneck, not a general claim about every pandas version.

The prototype changes only those four known fresh-array constructor sites to `copy=False`; it never disables copying globally or aliases an externally supplied return frame. It preserves values, labels, seed/stream and public result-copy behavior on measured inputs. The 8,192-path prepared-return pair improves about 8×, but the 128-path case improves only about 6%. Cold generation is measured separately. The warm separate `tracemalloc` allocation peak for the 8,192-path pair falls from 138.15 to 121.42 MiB; this is not RSS, and existing input matrices/imports are excluded. Lower copying cost does not imply an equally large reduction in peak memory.

Before implementing: pin fresh ownership at every site, caller mutation isolation, seeded output/derived CAGR, supported pandas versions, empty/NaN inputs, and the effect of memory layout on downstream reductions. The audit also checks seeded CAGR/percentiles across normal, Student-t and lognormal layouts at 1e-12: all observed differences were zero.

## Priority 2: historical DCF engine and accidental cache invalidation

`PortfolioDCF.wealth_index()` and `cash_flow_ts()` still use scalar historical helpers that construct/update pandas containers month by month or period by period. The existing `_simulate_paths_mc(..., task="backtest")` already supports chronological cash-flow arithmetic for one path. Reusing it on ordinary histories is much cheaper than rebuilding Series in the loop. The monthly IndexationStrategy example is tens of times faster; yearly IndexationStrategy is several times faster (table below). Percentage, VDS and CWD examples are also compared, with no numerical differences in the accepted cases.

This is not a complete compatible replacement yet: preserve inflation columns, partial periods, time-series cash flows, gaps/NaNs, index/name metadata, ruin behavior and strategy subclasses. The first monthly prototype exposed an index-name difference (`Date` versus None); the accepted adapter explicitly preserves the historical monthly/none naming behavior.

`wealth_index_fv_with_assets` is especially expensive: it uses DataFrame.apply to run the historical helper independently for the portfolio and every underlying asset. The array prototype evaluates these columns together, with the same cash-flow rule on each virtual all-in-asset path. For 16 assets plus the portfolio it reduces 1.195 s to 1.605 ms, about 745×. The measured histories have float64 returns, monthly indexed cash flows and no inflation. This is a narrow property-level opportunity, not a claim that all PortfolioDCF methods improve by that factor.

A separate issue occurs in both scalar helpers: they reassign `dcf_object.cashflow_parameters` even when the same strategy is already installed. Its setter clears all DCF caches. Consequently, wealth → cash flow → wealth → cash flow triggers three wealth-helper calls and two cash-flow-helper calls, despite unchanged inputs. A narrow same-object guard inside the helpers reduces that to one of each, about 2.7× for the monthly chain. Do not weaken the public setter's deliberate invalidation semantics; changes in strategy/data and in-place mutable inputs require tests. This opportunity overlaps with replacing the historical engine: their ratios must not be multiplied.

## Priority 3: remaining MonteCarlo work

- `backtesting_error()` resolves distribution parameters twice. For Student-t that means two `scipy.stats.t.fit` calls. One local resolution preserves measured results and nearly halves the method time. Partial parameter tuples and externally changed `ror` need coverage. A global identity-only fit cache is unsafe for mutable Series and is unnecessary for this local fix.
- `_get_cagr_distribution()` recomputes the full path CAGR for each percentile/inverse-percentile query. The measured report calls it four times; preparing one local distribution and evaluating quantiles together calls it once, about 4× faster. This is a multi-query report workload, not a promise of 4× for one `percentile_distribution_cagr()` call. Persistent CAGR caching needs care because `monte_carlo_returns_ts` exposes mutable cached data.
- `plot_qq()` generates its bootstrap sample in one array, but then makes 2,000 Python CVaR calls and repeats quantile work. A row-wise quantile/tail-mask calculation preserves ties and tail membership; it accelerates the kernel about 5× and the complete plot about 1.5×. Bootstrap seed 11 stays unchanged. NaNs, empty tails and plotting spans need regression coverage; compare full operation time rather than reporting the kernel ratio as chart speedup.

## Survival and withdrawal search

`Frame.get_survival_date(DataFrame)` is already vectorized. The subsequent `dates.apply(Date.get_period_length)` in `monte_carlo_survival_period()` still performs Python work per path. Vectorizing that conversion helps, especially after DataFrame copy overhead is removed. Retain the current 365-day year and one-decimal rounding; replacing it with months/12 changes the financial convention.

The combined constructor/date prototype reduces the 2,048-path withdrawal search from about 120 to 50 ms. Both baseline and prototype use four objective evaluations and return identical success, withdrawal and solution tables. This is a measured combination, not a product of separate speedup ratios. `find_the_largest_withdrawals_size()` already has residual caching, bounded evaluations and early stopping; these mechanisms should be retained.

## Lower priority or already optimized

The MC engine is already vectorized across scenarios, random sampling fills a complete matrix, and returns are cached after the first generation. There is no per-path distribution fit to remove. Rolling skewness/kurtosis and DataFrame survival detection already use vectorized pandas/NumPy paths.

The engine computes both wealth and cash flow, but separate public accessors run it twice. A shared output cache can reduce two engine calls to one, yet measured full-method improvement at 8,192 yearly paths is only about 1% while DataFrame copies dominate. It is not the first change to implement.

Other inspected historical methods (`irr`, `survival_date_hist`, `survival_period_hist`) depend on the same wealth/cash-flow helpers, so improving those shared paths can help them; separate end-to-end speedups for these historical wrappers were not measured. Initial-investment PV/FV conversions are already simple scalar arithmetic. `plot_forecast_monte_carlo(backtest=False)` also clears DCF caches after plotting despite not changing the strategy; this is an unmeasured secondary candidate. The backtest=True path temporarily changes the strategy and must invalidate/restore it correctly. Drawing thousands of forecast lines is a distinct visualization cost and is not accelerated by these numerical prototypes.

The IRR helper already batches Newton iterations, uses a good initial guess and has a bracketing fallback. For zero intermediate cash flows, an analytic shortcut makes its kernel tens of times faster, but the complete `monte_carlo_irr()` improves only little in the large measured workload. Prioritize matrix construction first. The shortcut applies only to conventional initial outflow/terminal inflow cases, never arbitrary multi-sign cash flows or custom guesses.

## Representative measurements

| Actual method/workload | Current | Narrow prototype | Ratio |
|---|---:|---:|---:|
| Historical wealth_index_fv_with_assets, 600 months, 3 assets + portfolio | 276.703 ms | 1.567 ms | 176.55× |
| Same property, 16 assets + portfolio | 1,195.442 ms | 1.605 ms | 744.94× |
| MC wealth + cash flow, returns prepared, 360 × 8,192 | 622.003 ms | 76.991 ms | 8.08× |
| Same complete call, cold return generation included | 771.683 ms | 96.806 ms | 7.97× |
| MC wealth + cash flow, 360 × 2,048 | 38.805 ms | 19.761 ms | 1.96× |
| MC wealth + cash flow, 360 × 128 | 2.830 ms | 2.684 ms | 1.05× |
| Historical monthly-indexed wealth, 600 months | 72.736 ms | 1.500 ms | 48.49× |
| Historical yearly-indexed wealth, 600 months | 9.983 ms | 1.389 ms | 7.19× |
| Historical wealth/cash-flow chain, 600 months | 356.251 ms | 139.489 ms | 2.55× |
| Withdrawal solver: fresh-array constructors + date processing | 123.168 ms | 49.821 ms | 2.47× |
| Student-t backtesting_error, 600 observations | 64.096 ms | 32.246 ms | 1.99× |
| One percentile report + three inverse-CAGR queries | 37.091 ms | 9.649 ms | 3.84× |
| Complete plot_qq with 2,000 bootstrap samples | 102.559 ms | 59.783 ms | 1.72× |
| Joint engine output alone, yearly MC | 685.027 ms | 635.030 ms | 1.08× |
| Zero-intermediate-flow IRR kernel only | 43.619 ms | 1.228 ms | 35.51× |
| Same IRR shortcut in complete monte_carlo_irr | 542.162 ms | 499.730 ms | 1.08× |

## Protocol, validation and limits

The [reproduction script](https://github.com/mbk-dev/okama/blob/v4.0.1/analytics/benchmark_dcf_mc_opportunities.py) runs 44 paired cases with one warm pair and five alternating timed pairs, plus separate allocation measurements/profiles and work counters. Ready synthetic assets and mocked currency loading keep all inputs offline. Portfolio histories contain exactly 600 months; MC horizons contain 360 months. Return/bootstrapping seeds are 127 and 11. Normal returns are prepared outside timers where stated; cold-return cases explicitly include their generation. Chart tests use the Agg backend and close figures. Simulation caches are reset or deliberately warmed per workload; all scenarios use fresh private fixtures and restore scoped patches.

Environment: Python 3.14.8, NumPy 2.5.4, pandas 3.1.0 on AMD Ryzen 7 8845H. BLAS/OpenMP are limited to one thread for the command, with joblib capped at four. A shared host has background/scheduling noise. Candidate timings include prototype patch overhead; small cases can show little benefit. No HTTP/GPU/JIT/dependency change is involved.

All accepted pairs compare full values/types/index/column/name metadata with rtol=atol=1e-12. The largest observed absolute difference is 2.887e-15, in the special IRR case. Original formulas, draw streams and solver evaluation budgets are retained on the measured inputs. These guarded prototypes are not full edge-case proofs and do not authorize replacing scalar fallbacks without TDD.

[dcf-mc-opportunities.json](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/dcf-mc-opportunities.json) retains all 44 timing distributions, source/script hashes, profiles and measured work counts. Profile paths are normalized for portability; original raw files and failed exploratory attempts are archived privately. The research script never writes production source; its AST/patch transformations exist only inside the benchmark process.

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 uv run --no-sync python -m analytics.benchmark_dcf_mc_opportunities --output tmp/dcf-mc-audit.json --counts
```

Final checks: `uv run --no-sync pytest -q` — 922 passed, 3 skipped, 174 existing warnings; both Ruff commands and diff check passed. Production files remain byte-identical to the audited revision. An independent read-only MC exploration identified fit/CAGR/bootstrap candidates; the coordinator verified them through source tracing and numerical experiments, rejecting identity-only mutable-data cache proposals. This change publishes only the audit report and reproduction script; none of these new optimization prototypes is implemented in the library.
