# PortfolioDCF and MonteCarlo acceleration

This implements the [DCF/MC audit](dcf-mc-opportunities.md). The measurements below use complete public methods in the implementation, compared with the unmodified source at `bcd877421262d107b7b35a9a771498802dc39c0b`. Production implementation commit: `275d0f3`. They supersede the audit's prototype estimates for these workloads.

## Changes

- `PortfolioDCF.wealth_index()` and `cash_flow_ts()` use the existing array engine for ordinary finite float64 monthly histories. `wealth_index_fv_with_assets` evaluates all assets in one pass. Original scalar implementations handle unsupported indices/dtypes, missing values, strategy subclasses and exceptional rates. Calendar bins, partial periods, inflation and extra cash flows retain their original semantics and metadata.
- Historical helpers avoid reassigning the same strategy and accidentally clearing caches. Explicit public strategy assignment still invalidates caches. Changing `PortfolioDCF.discount_rate` now also invalidates cash-flow caches, preventing stale discounted extra payments.
- `PortfolioDCF.monte_carlo_wealth()` and `monte_carlo_cash_flow()` share one simulation. Fresh owned arrays are wrapped with `copy=False`; caller-owned input arrays are not reused as mutable outputs. Public result copies retain mutation isolation.
- `MonteCarlo` tracks a deep snapshot of its latest mutable return matrix. Both joint DCF results and derived CAGR check values and axes before reuse; setting changes and in-place input edits invalidate results. Unsupported array types retain the uncached derivation path. This snapshot adds memory, quantified below.
- `MonteCarlo.backtesting_error()` resolves distribution parameters once. CAGR queries share the derived distribution and evaluate percentile lists together. `plot_qq()` computes bootstrap lower-tail statistics in a batch, retaining seed 11, inclusive ties and confidence bands.
- `PortfolioDCF.monte_carlo_survival_period()` computes date lengths as an array with the existing 365-day/one-decimal convention. `plot_forecast_monte_carlo(backtest=False)` preserves caches; temporary backtest strategies still restore/invalidate through the public setter.
- `irr_of_cashflow_matrix()` directly solves ordinary initial-outflow/terminal-inflow matrices without intermediate payments. General, extreme and explicit-guess cases retain Newton/brentq. Withdrawal search retains its objective, evaluation budget, bracketing and residual cache.

## Complete-method measurements

Offline deterministic 600-month histories, 360-month forecasts and seed 127. Five alternating paired runs, median wall time; Python 3.14.8, NumPy 2.5.4, pandas 3.1.0, Agg plots, one BLAS thread. The script loads the exact original DCF/calculation/MC modules from Git; it does not patch production candidates. Portfolio data preparation is outside timing. Cold cases clear MC draws and derived caches on every call; warm-return pairs recompute DCF outputs from prepared draws. Survival uses cached wealth; warm CAGR bundles include repeated-query cache reuse. No other tests ran concurrently.

| Workload | Before, ms | After, ms | Speedup |
|---|---:|---:|---:|
| `history_600_index_month` | 101.557 | 4.290 | 23.67x |
| `history_600_index_year` | 20.757 | 5.319 | 3.90x |
| `history_600_percentage_quarter` | 56.454 | 7.890 | 7.15x |
| `history_600_cwd_month` | 89.133 | 4.512 | 19.75x |
| `history_600_vds_year` | 12.064 | 3.129 | 3.86x |
| `history_chain_600` | 427.805 | 7.375 | 58.01x |
| `history_with_16_assets_600` | 1487.225 | 2.562 | 580.48x |
| `mc_pair_360x128_warm_returns` | 3.990 | 2.418 | 1.65x |
| `mc_pair_360x128_cold` | 3.555 | 2.777 | 1.28x |
| `survival_360x128_cached_wealth` | 0.883 | 0.552 | 1.60x |
| `cagr_bundle_360x128_warm_returns` | 1.719 | 0.711 | 2.42x |
| `mc_pair_360x2048_warm_returns` | 43.144 | 16.335 | 2.64x |
| `mc_pair_360x2048_cold` | 51.460 | 19.963 | 2.58x |
| `survival_360x2048_cached_wealth` | 16.922 | 6.562 | 2.58x |
| `cagr_bundle_360x2048_warm_returns` | 7.629 | 2.605 | 2.93x |
| `mc_pair_360x8192_warm_returns` | 497.997 | 61.917 | 8.04x |
| `mc_pair_360x8192_cold` | 692.072 | 99.792 | 6.94x |
| `survival_360x8192_cached_wealth` | 181.911 | 44.013 | 4.13x |
| `cagr_bundle_360x8192_warm_returns` | 27.758 | 8.438 | 3.29x |
| `cagr_bundle_360x8192_cold` | 201.733 | 185.136 | 1.09x |
| `irr_no_intermediate_cashflows_360x8192_cold` | 698.126 | 85.190 | 8.19x |
| `withdrawal_solver_360x2048` | 125.442 | 53.826 | 2.33x |
| `backtesting_error_student_t` | 73.321 | 34.399 | 2.13x |
| `qq_full_plot_600_bootstrap2000` | 105.638 | 49.497 | 2.13x |

These are workload- and dependency-specific results, not universal speedups. In particular, pandas 3.1's ndarray constructor copies made wide matrices expensive; older pandas versions need not show the same gain. The 16-asset result combines 17 scalar histories into one array pass. Small MC runs benefit less. Cold CAGR still pays for random draws and the snapshot; its gain is only 1.09x here, versus about 3.29x for repeated queries.

All measured outputs passed `rtol=atol=1e-12`, including pandas labels/dtypes, withdrawal solution history and QQ lines/confidence bands. Maximum observed IRR difference was 2.89e-15; all other compared values were identical. Tests also cover non-default strategies, partial periods, inflation, NaN/dtype/index fallbacks, mutable inputs/results, cache invalidation and seeded distributions. The complete suite passed with 1083 tests, 3 skips and 33 existing pandas deprecation warnings; both Ruff checks passed. New reference tests locally filter only the known scalar PeriodIndex-insertion deprecation, preserving the frozen reference arithmetic.

## Memory and limits

`tracemalloc` measures allocations during a cold public MC wealth/cash-flow pair, after portfolio construction. It is not process RSS and excludes pre-existing Python/import state. For 360 x 8192, retained allocations increase from 67.58 to 90.27 MiB, and peak allocations from 160.66 to 166.61 MiB. The retained increase is the 22.5 MiB return snapshot. If only wealth is requested, the paired cash-flow cache is also retained. Only the latest draw is snapshotted; there is no unbounded or global cache. Repeated cache accesses scan the return matrix to detect caller edits.

No RNG stream ordering, solver tolerances, discount convention, dependency, GPU/JIT or global runtime setting is changed.

## Reproduce

From a checkout containing the baseline Git commit, with test/dev dependencies installed:

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 \
  uv run --no-sync python -W ignore -m analytics.benchmark_dcf_mc_acceleration \
  --repeats 5 --output /tmp/dcf-mc-acceleration.json
```

[Raw paired samples, allocation measurements and source digests](dcf-mc-acceleration.json). The earlier opportunity-audit script uses prototype patches and should be reproduced at `bcd8774`, before these changes.
