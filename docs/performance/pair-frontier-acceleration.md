# Multi-period pairwise frontier acceleration

Implementation: `24d3147`; baseline: `48a0e7c8eb139a8b8b3b0eeb6e6a1dbbcbc8e3aa`. Measured on 26 October 2026. This implements the approved [pairwise proposal](../superpowers/specs/2026-10-26-pair-frontier-acceleration-design.md) and supersedes the no-gain conclusion for the earlier multi-period path in the [historical audit](pair-frontier-audit.md).

## Changes

`EfficientFrontier` prepares calendar groups and cumulative asset growth once for a stable numerical calculation. `minimize_risk`, GMV, tangency and the other nested objectives reuse it through `_get_portfolio_ror_ts`; joblib point workers receive the same prepared state. An external calculation boundary checks a private input snapshot and effective settings, refreshing prepared data, weight results and frontier tables when necessary. The measured grouping regression falls from 56 DataFrame groupings to two for one optimizer call.

The single-weight evaluator preserves elementwise multiplication, reduction order, first-month pandas dot product and off-constraint SLSQP trial weights. Existing 10-decimal weight-cache keys, solver options, starts, target ranges and corner guards remain intact. Invalid/non-finite growth, exceptional wealth, unsupported dtypes/indices and custom frontier classes use the original numerical path. Wide fragmented pandas frames also retain scalar evaluation because consolidating their columns changes rounding; two-column sums contain a single addition and remain safe. Input layout is part of the boundary check.

`EfficientFrontier.plot_pair_ef` now renders tables from `_get_pair_frontiers`. Each call still resolves fresh child inputs, preserving parent dates, pair order, bounds, effective rebalance period and data/FX freshness. Matching pairs reuse the latest stored Risk/CAGR table; changed source assets invalidate only affected pairs. Returned private tables are copied, and figures/child frontier objects are not cached. Changing plot labels or size does not repeat optimization. The cache retains one collection, not a history of configurations.

## Complete-method measurements

All times below are medians. “First” means an empty pair-table cache, with imports and worker pools already warm; it does not mean a cold Python/service process. Parent construction is excluded on both sides. Fresh child construction, optimization and Agg plot/artist creation are included. No canvas drawing, HTTP, JSON transport or browser rendering is measured. Inputs are ready offline assets with NumPy seed127 and ordinary float64 histories.

| Assets / pairs | Months | Requested points | Point workers | Original (s) | Current first (s) | First gain | Current repeat | Repeat gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 / 6 | 120 | 12 | 1 | 3.689 | 0.874 | 4.22x | 11.2 ms | 328.3x |
| 8 / 28 | 120 | 12 | 1 | 15.892 | 4.236 | 3.75x | 41.1 ms | 386.2x |
| 12 / 66 | 120 | 12 | 1 | 38.892 | 14.228 | 2.73x | 83.2 ms | 467.6x |
| 4 / 6 | 600 | 12 | 1 | 3.410 | 1.181 | 2.89x | 15.2 ms | 223.9x |
| 4 / 6 | 120 | 80 | 1 | 20.388 | 5.445 | 3.74x | 13.1 ms | 1560.9x |
| 4 / 6 | 120 | 12 | 4 | 1.754 | 0.498 | 3.52x | 12.7 ms | 137.7x |

For 80 requested points, six pairs improve from 20.39 to 5.45 seconds on one CPU. With four point workers, the 12-point case improves from 1.75 to 0.50 seconds. These are representative cases, not every combination of asset count, history, points and parallelism. A complete 12-asset calculation still takes about 14.2 seconds on one CPU; the first request has not become instantaneous.

One untimed full-output comparison precedes three timed calls per side. Baseline samples were preserved from the initial run; after the review repair, every candidate case was measured again against those hash-verified baseline outputs. Each side runs in an independent process. BLAS/OpenMP are restricted to one thread, and joblib uses the table's explicit worker count. The baseline restores the original frontier module; current loading/common helpers are shared, and original `Rebalance.wealth_ts_ef` / `return_ror_ts_ef` remain unchanged. This isolates this implementation stage rather than comparing entire old environments.

Full frontier tables, all weights, iteration counts, initial-guess diagnostics and plotted coordinates matched at `rtol=atol=1e-12`. Observed maximum difference was zero; serialized full-table hashes also matched in all six cases. The [raw timing samples, descriptive sample p95 values, source hashes and environment](pair-frontier-acceleration.json) are retained. Three samples do not establish a production p95 or a capacity limit. Shared-host scheduling and different Python/NumPy/pandas versions can change gains. Environment: Ryzen 7 8845H, Linux, Python 3.14.8, NumPy 2.5.4, pandas 3.1.0.

## Memory and downstream limits

A separate call samples aggregate resident memory of the benchmark process and its own worker descendants every 10 ms. It includes imports, held validation output and warmed workers; shared pages can be counted multiple times. It is neither PSS nor an isolated allocation delta. The JSON also records main-process lifetime peak RSS.

Representative process-tree peaks: 4 assets / 12 points / one worker, 258.9→259.2 MiB; 12 assets, 266.6→269.0 MiB; 4 assets / 80 points, 281.8→290.3 MiB. Four workers used about 1.3 GiB on both sides (1305.0→1293.0 MiB); this variation is not evidence of a memory reduction. Snapshots, growth arrays and latest pair tables add retained storage. Pair count and retained data still grow as `n(n-1)/2`.

The shared numerical preparation benefits consumers that directly create pair frontiers and read `ef_points`. Repeat-plot cache gains apply to library plotting; a service that already caches pair responses should not count them as another end-to-end speedup. Child construction still reads current inputs, so network-bound repeat requests need not approach the offline millisecond figures. This stage does not change service caps, installed dependency versions or deployments.

## Verification and reproduction

Strict RED→GREEN tests demonstrated repeated grouping, stale mutable-input results and redundant repeated-plot optimization before implementation. Independent review found the fragmented-frame arithmetic edge case; dedicated regressions reproduce it and retain scalar compatibility. Coverage includes all five periods, off-constraint weights, wide reductions, unsupported histories, cache freshness/identity, pair configuration/source changes, copies, subclasses, worker execution and failure unwinding.

Final local suite: **1157 passed, 3 skipped, 33 existing pandas warnings**; no additional warnings. Both Ruff commands and whitespace checks passed. The implementation requires no new dependency or global runtime setting.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg LOKY_MAX_CPU_COUNT=4 \
  uv run --no-sync python -m analytics.benchmark_pair_frontier \
  --output /tmp/pair-frontier.json --scratch /tmp/pair-frontier-raw
```

The repository must contain the baseline Git commit and test/dev dependencies. Default execution measures both sides afresh. The optional `--reuse-baseline` mode requires a manifest pinning the baseline commit and every reused raw file's SHA-256; it is for a same-experiment candidate rerun, not arbitrary historical sample reuse.
