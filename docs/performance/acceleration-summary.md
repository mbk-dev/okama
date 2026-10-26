# Delivered calculation speedups and consumer capacity planning

Snapshot: 26 October 2026, earlier library `dev` delivery through `370204bf7db4901626281b3ce624edbb11f84450`, plus pairwise implementation `24d3147`. This is a consolidated record of measured, implemented improvements, including the subsequent pairwise-frontier implementation. It is not a claim that a package release or downstream deployment already contains them.

## Delivery and evidence

- [PR #134](https://github.com/mbk-dev/okama/pull/134): frontier MC/grid batching, ListMaker/dividend preparation, Rebalance, statistics and single-period optimizers; integrated as `31db96c`.
- [PR #135](https://github.com/mbk-dev/okama/pull/135): scoped API connection/metadata reuse; integrated as `789efbb`.
- [PR #137](https://github.com/mbk-dev/okama/pull/137): PortfolioDCF and MonteCarlo implementation; integrated as `370204b` (production change `275d0f3`).
- [Pairwise implementation](pair-frontier-acceleration.md), commit `24d3147`: shared numerical preparation and repeated pair plots.
- [PR #136](https://github.com/mbk-dev/okama/pull/136) preserved the DCF/MC opportunity audit. Its prototype estimates are not additional delivered speedups.

Detailed reports below retain their original measurement-time status. Their references to local/uncommitted changes describe that earlier stage; the PRs above establish subsequent integration into `dev`.

The implementation verification at this snapshot was 1,157 tests passed, 3 skipped, with 33 existing pandas warnings. Ruff lint/format checks and CI on Python 3.11–3.14 passed. Numerical comparisons, fallback coverage, source digests and raw samples are recorded in the individual reports. No production code is changed by this summary.

## Representative complete-call measurements

Times are medians in milliseconds unless explicitly described otherwise. Rows are separate experiments, with different baselines and cache conditions; ratios must not be multiplied or treated as site-wide speedups. Earlier stages used Python 3.11.17 / NumPy 2.4.6; the DCF/MC and pairwise audits used Python 3.14.8 / NumPy 2.5.4. Both used pandas 3.1.0 on the same Ryzen 7 8845H host. See each report for threading, warmup, repetitions and memory methodology.

| Area and measured workload | Before → after (ms) | Speedup | Evidence |
|---|---:|---:|---|
| `EfficientFrontierSingle.get_monte_carlo`, mean, 252 months × 3 assets × 8,192 portfolios | 3,347.399 → 18.535 | 180.6x | [Single-period batching](frontier-cpu-batching.md) |
| Same MC, approximate CAGR | 4,371.255 → 20.481 | 213.4x | [Single-period batching](frontier-cpu-batching.md) |
| `EfficientFrontierSingle.get_grid_portfolios`, mean, 5,151 portfolios | 1,898.721 → 8.251 | 230.1x | [Single-period batching](frontier-cpu-batching.md) |
| `EfficientFrontier.get_monte_carlo`, yearly rebalance, 252 months × 3 assets × 8,192 portfolios | 19,795.167 → 65.835 | 300.7x | [Calendar batching](calendar-batching.md) |
| `EfficientFrontier.get_grid_portfolios`, yearly rebalance, 5,151 portfolios | 12,578.275 → 30.520 | 412.1x | [Calendar batching](calendar-batching.md) |
| `Rebalance`, monthly, full asset paths, 600 months × 16 assets | 733.891 → 13.415 | 54.7x | [Calendar batching](calendar-batching.md) |
| `Rebalance`, yearly with 5% absolute threshold, same size | 32.335 → 3.127 | 10.3x | [Calendar batching](calendar-batching.md) |
| `Rebalance`, relative threshold, full asset paths, 600 × 16 | 647.348 → 10.394 | 62.3x | [Conditional rebalance](rebalance-conditional.md) |
| `ListMaker` construction from 32 ready asset objects | 14.887 → 1.983 | 7.5x | [ListMaker](listmaker-optimizations.md) |
| Cold dividend-yield calculation, 600 months × 16 assets | 4,751.886 → 6.874 | 691.3x | [ListMaker](listmaker-optimizations.md) |
| Annual-risk GMV weights, 32 assets | 166.323 → 11.618 | 14.3x | [Statistics](statistics-performance.md) |
| `AssetList.get_rolling_cagr`, 600 × 16, 60-month window | 253.667 → 1.129 | 224.6x | [Statistics](statistics-performance.md) |
| `Portfolio.get_rolling_cagr`, same history/assets, portfolio returns already cached | 16.471 → 1.174 | 14.0x | [Statistics](statistics-performance.md) |
| Remaining single-period optimizer workflow: 600 × 32, 20 points, tangency portfolio and metrics | 12,173.550 → 956.158 | 12.73x | [Frontier workflow](frontier-workflow.md) |
| Complete live three-asset API workflow, automatic HTTP-session scope | 611.5 → 519.1 | about 1.18x | [API scope](api-request-scope.md) |
| `PortfolioDCF` historical monthly-indexation wealth, 600 months | 101.557 → 4.290 | 23.67x | [DCF/MC](dcf-mc-acceleration.md) |
| Historical DCF with asset wealth, 600 months × 16 assets | 1,487.225 → 2.562 | 580.48x | [DCF/MC](dcf-mc-acceleration.md) |
| MC wealth + cash-flow pair, 360 months × 8,192 simulations, cold / returns pre-generated | 692.072 → 99.792 / 497.997 → 61.917 | 6.94x / 8.04x | [DCF/MC](dcf-mc-acceleration.md) |
| MC survival, same size, wealth already cached | 181.911 → 44.013 | 4.13x | [DCF/MC](dcf-mc-acceleration.md) |
| MC CAGR statistics bundle, same size, returns pre-generated | 27.758 → 8.438 | 3.29x | [DCF/MC](dcf-mc-acceleration.md) |
| MC IRR without intermediate cash flows, same size, cold complete call | 698.126 → 85.190 | 8.19x | [DCF/MC](dcf-mc-acceleration.md) |
| Largest-withdrawal solver, 360 months × 2,048 simulations | 125.442 → 53.826 | 2.33x | [DCF/MC](dcf-mc-acceleration.md) |
| Student-t distribution backtesting | 73.321 → 34.399 | 2.13x | [DCF/MC](dcf-mc-acceleration.md) |
| Full QQ plot, 600 historical observations and 2,000 bootstrap samples | 105.638 → 49.497 | 2.13x | [DCF/MC](dcf-mc-acceleration.md) |
| `EfficientFrontierSingle.plot_pair_ef`, 4 assets / 6 pairs, 120 months, 12 points | 380.880 → 209.429 | 1.82x | [Pairwise audit](pair-frontier-audit.md) |
| `EfficientFrontier.plot_pair_ef`, 4 assets / 6 pairs, 120 months, 12 points, first / repeat | 3,688.564 → 874.400 / 11.235 | 4.22x / 328.3x | [Pairwise implementation](pair-frontier-acceleration.md) |
| Same multi-period plot, 12 assets / 66 pairs, first / repeat | 38,891.992 → 14,228.225 / 83.182 | 2.73x / 467.6x | [Pairwise implementation](pair-frontier-acceleration.md) |
| Same multi-period plot, 4 assets / 6 pairs, 80 requested points, first | 20,387.916 → 5,445.002 | 3.74x | [Pairwise implementation](pair-frontier-acceleration.md) |

## Boundaries that matter when increasing limits

- These are workload-specific library results. Tiny frontier MC calls showed essentially no improvement; the cold 128-simulation DCF pair improved only 1.28x. DCF CAGR statistics improved 3.29x warm but only 1.09x cold. Input preparation, random draws, loading and response rendering remain part of a real request.
- Ready-asset construction and offline dividend calculations exclude network data acquisition. The live API figure has only three rotating samples after warmup and is sensitive to network variance. Automatic scoping reduced session construction (11 to 5), not request count (11). Explicit outer scoping used 4 sessions and 10 requests but was not consistently faster. There is no persistent global time-series cache.
- The single-period optimizer workflow row measures an incremental stage, with earlier batching/ListMaker improvements shared by both versions. It does not establish a 12.73x gain for the multi-period efficient-frontier solver.
- MC/grid batching accelerates evaluation of sampled weights. It does not automatically accelerate constrained optimization of a full efficient frontier or all pairwise frontiers.
- Portfolio Monte Carlo simulates paths over time; frontier Monte Carlo evaluates a cloud of portfolio weights. Their input sizes, memory use and UI costs need separate limits.
- Bounded batches reduce temporary arrays but do not eliminate the output-size cost. Calendar frontier batches cap intermediates at 1,024 portfolios and 262,144 period-by-portfolio entries. Output tables and weights still grow with requested size.
- DCF reuse retains one latest draw snapshot, adding about 22.5 MiB at 360 × 8,192. Joint wealth/cash-flow preparation can also allocate cash flows for a wealth-only request. Measure process peak memory and concurrent workers, not just a microbenchmark allocation peak.
- A grid with `n` assets and step `1/q` has `C(q+n-1, n-1)` portfolios. At 12 assets, step 0.2 gives 4,368 portfolios but step 0.1 gives 352,716. Raising the asset cap and retaining the same step can overwhelm a point budget despite faster evaluation.
- Pair count is `n(n-1)/2`: 12 assets give 66 curves, 20 give 190. The [implemented pairwise stage](pair-frontier-acceleration.md) reuses calendar/growth preparation inside the numerical engine and caches completed pair tables for repeat plots. Its gain is included above, but even the faster first 12-asset calculation takes about 14.2 seconds on one CPU. Service response-cache hits must not be credited with the additional library repeat-plot gain.

## Consumer adoption and capacity acceptance

Before changing a service limit, identify the installed artifact and exact source revision containing the relevant changes; a matching package version string or a merged library PR alone is insufficient. Check compatibility with the consumer's existing version and keep a reproducible rollback target.

Measure cold and warm end-to-end requests separately at current and candidate limits, including joint worst cases: asset count, history length, grid step, requested points, simulation horizon and nested portfolio exposure. Record p50/p95 latency, peak process memory, queue wait/timeouts, failures, result size and concurrency. Test chart rendering and interaction on phones and low-powered laptops; faster server arithmetic does not remove transfer or browser costs.

Choose an explicit capacity policy for frontier MC points, grid point budget/resolution, portfolio MC simulations × horizon, portfolio positions/underlying assets and frontier asset count/pair count. Keep defaults separate from maxima. Align frontend schemas, server validators, contracts, URL compatibility and boundary tests; enforce limits for direct API callers too. Treat mocks as response-shape fixtures, not evidence of numerical capacity. Publish new caps only after measuring the actual adopted artifact.

The remaining research list is in the [performance index](README.md). No site limit, worker count or deployment is changed by these reports.
