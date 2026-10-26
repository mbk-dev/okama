# Pairwise efficient-frontier audit

Read-only audit of dev `370204bf7db4901626281b3ce624edbb11f84450`, 26 October 2026. This records the state before the subsequent [shared-preparation and repeat-plot implementation](pair-frontier-acceleration.md).

Both `EfficientFrontierSingle.plot_pair_ef()` and `EfficientFrontier.plot_pair_ef()` retain the same executable pair-loop bodies as `b33ca12bdd81b4107c2de484fff581eda1023ce1`. Every call constructs fresh two-asset frontier objects. Instrumentation counted six new objects on a first plot and six more on a second plot of the same four-asset parent. Existing numerical caches inside those child objects are discarded with them.

## Complete-call experiment

Four assets, six pairs, 120 monthly observations and 12 requested frontier points; offline seeded float64 data and ready asset objects. Parent construction is outside timing; pair construction, optimization and Agg plot creation are inside. One warm pair precedes three alternating timed pairs. One joblib/BLAS/OpenMP CPU, shared host; Python 3.14.8, NumPy 2.5.4, pandas 3.1.0. Current ListMaker, loading and common helpers are shared; only frontier and Rebalance implementations are restored from the baseline. This is not an old-environment/site-response comparison.

| Complete call | Original median | Current median | Interpretation |
|---|---:|---:|---|
| `EfficientFrontierSingle.plot_pair_ef()` | 380.880 ms | 209.429 ms | 1.82x; benefits from accelerated inner single-period optimizers |
| `EfficientFrontier.plot_pair_ef()`, yearly rebalance | 3,281.576 ms | 3,819.280 ms | No demonstrated speedup; overlapping samples do not establish a systematic regression |

Original yearly samples were 3.25–4.36 seconds; current samples were 3.19–4.13 seconds. Complete plotted coordinates matched at `rtol=atol=1e-12`, with zero observed difference. [Raw paired samples, object counts and unchanged-body checks](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/pair-frontier-audit.json).

## Why the multi-period path remains slow

`ef_points -> minimize_risk -> _get_portfolio_ror_ts -> Rebalance.return_ror_ts_ef -> wealth_ts_ef` still executes the original numerical path. `wealth_ts_ef` rebuilds calendar grouping and cumulative within-period asset growth for each uncached weight vector. Existing `get_grid_portfolios` / `get_monte_carlo` batching is reached through a different helper, `_get_portfolio_points`.

The separate current profile recorded 1,300 resampler and 3,900 groupby constructions. Risk minimization consumed 6.63 seconds of the 7.06-second instrumented call; pair constructor/ListMaker work was about 0.01 seconds. Profiler timings include instrumentation overhead and must not be compared directly with the timing table. Removing constructors alone would not address the main CPU cost.

Pair count is `n * (n - 1) / 2`: 12 assets produce 66 pairs; 20 assets produce 190. Pairs are processed sequentially, while the numerical frontier already has point-level joblib parallelism. There is no measured justification for multiplying the two parallel layers.

The first optimization should reuse calendar/growth preparation in the numerical engine so it also benefits consumers that instantiate pair frontiers and read `ef_points` directly. Caching completed pair tables is a separate improvement for repeated plots. HTTP and browser rendering were not measured in this experiment.
