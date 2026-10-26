# Per-solve GMV statistics and rolling CAGR

This stage accelerates `EfficientFrontierSingle.gmv_monthly_weights`, `gmv_annual_weights`, `gmv_monthly` and `gmv_annualized`, plus `AssetList.get_rolling_cagr`, `Portfolio.get_rolling_cagr` and the rolling branch of `AssetList.tracking_difference_annualized`.

## Changes and compatibility

GMV now computes pandas covariance once per solve, and the annual-risk objective computes means once. It retains pandas multiplication order, the square-root risk objective, SLSQP settings, initial guess, bounds and failure message. `gmv_monthly` solves once instead of twice, and `gmv_annualized` reuses one monthly result instead of requesting it three times. Statistics/results are local to each access, so changed returns or bounds are not hidden by a persistent cache. This deliberately keeps more pandas overhead than the earlier exploratory NumPy prototype in order to preserve solver trajectories.

The explicit `Frame.get_rolling_cagr` helper computes rolling sums of logarithms and annualized `expm1` for ordinary float64 data. It uses the same rounded `1 + ror` growth factors as the original product. Generic `Frame.get_rolling_fn` and arbitrary callbacks remain unchanged. Existing inflation/real-return conversion runs before the new helper.

The logarithmic path requires positive finite growth factors (NaN observations remain allowed) and a conservative `max(abs(log(growth))) * window < 600` bound. This bounds every intermediate product, including prefixes, away from overflow/underflow. Otherwise the original product callback runs. Nonpositive growth, infinity, unsupported dtypes, all-missing values and potentially extreme intermediate products thus preserve the original behavior. Window validation, full-window minimum observations, `dropna`, names, index/columns and output types remain compatible. Logarithms change rounding slightly; equivalence is checked with both relative and absolute tolerance 1e-12, rather than bitwise identity.

## Paired measurements

The [benchmark script](../../analytics/benchmark_statistics_performance.py) runs actual public methods on seeded offline return histories and ready assets, excluding object construction and network calls. Original GMV code and original rolling public methods are loaded from revision `b33ca12bdd81b4107c2de484fff581eda1023ce1`. Other previously implemented improvements are shared by both sides. One warm pair precedes five alternating timed pairs. Portfolio timing uses the warmed return series cache for both sides.

Python 3.11.17, NumPy 2.4.6, pandas 3.1.0; AMD Ryzen 7 8845H, shared Linux host. Command-local BLAS/OpenMP limits are one thread. Scheduling and background activity can affect timing; results apply to these workloads, not the whole library. No RSS or allocation-peak experiment is claimed for this stage; rolling intermediates scale with input rows × columns.

| Complete public operation | Original median | Candidate median | Speedup |
|---|---:|---:|---:|
| GMV monthly weights, 3 assets | 2.729 ms | 1.638 ms | 1.7× |
| GMV monthly weights, 32 assets | 48.672 ms | 15.172 ms | 3.2× |
| GMV annual-risk weights, 32 assets | 166.323 ms | 11.618 ms | 14.3× |
| GMV monthly risk/return tuple, 32 assets | 118.538 ms | 19.250 ms | 6.2× |
| GMV annualized risk/return tuple, 32 assets | 291.386 ms | 18.458 ms | 15.8× |
| AssetList rolling CAGR, 252 × 3, window 12 | 22.624 ms | 0.750 ms | 30.2× |
| AssetList rolling CAGR, 600 × 16, window 60 | 253.667 ms | 1.129 ms | 224.6× |
| AssetList rolling CAGR, 6,000 × 16, window 120 | 2,818.963 ms | 3.548 ms | 794.5× |
| Portfolio rolling CAGR, 600 months, 16 underlying assets, window 60 | 16.471 ms | 1.174 ms | 14.0× |

All 12 cases passed complete numerical/metadata comparisons at rtol=atol=1e-12. GMV weights and tuples matched exactly on the measured inputs; maximum absolute difference over the rolling cases was 6.106e-16. Full timing distributions, source digests and script digest are retained in [statistics-performance.json](statistics-performance.json).

The first benchmark attempt detected that the currency fixture truncated the 6,000-month history. Its results were discarded. The final fixture supplies explicit full input dates and asserts the exact return matrix shape; all cases were rerun. The long history is a synthetic scaling exercise, not an assertion that such market history exists.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 poetry run python -m analytics.benchmark_statistics_performance --output tmp/statistics-performance.json
```

## Validation and remaining scope

Strict TDD first reproduced repeated covariance and repeated solves and Python rolling callbacks. Regression coverage compares checked-in original GMV methods and actual original product calculations, including missing returns, bounded weights, mutable input freshness, unsuccessful solves, Series/DataFrame metadata, real/inflation conversion, unsupported dtypes, total/negative growth, infinity, product range and invalid windows. Frozen test references require no Git history at runtime.

Final verification: `poetry run pytest -q` — **825 passed, 3 skipped, 174 pre-existing pandas warnings**. The previous stage had the same warning count; affected new tests are clean. Both `poetry run ruff check .` and `poetry run ruff format --check .` passed; `git diff --check` passed. Measured source and benchmark script digests match the final files.

Independent read-only review ran on `grok-4.6-build`. It raised a possible sub-year-window mismatch; parent verification rejected that finding because `check_rolling_window` rejects all windows 1–11 before either calculation. Explicit tests now verify every such window. No open correctness findings remain after reconciliation. The reviewer executed no tools or writes. Changes remain local and uncommitted. No dependency or global configuration changed. Other scalar optimization objectives, CSV parsing and network/data-loading work remain outside this stage.
