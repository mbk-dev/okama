# Remaining frontier optimizer preparation and complete CPU workflow

This stage accelerates `EfficientFrontierSingle.minimize_risk`, `get_tangency_portfolio` and `optimize_return`, which determines return ranges for complete `ef_points` construction.

## Implementation

`minimize_risk` now prepares pandas covariance and means once per solve. It keeps the square-root objective and the exact pandas multiplication order used by the scalar helper, the target-return constraint, annual-to-monthly conversion, SLSQP method/options/tolerance, bounds, corner-portfolio guard and failure message. The corner guard still computes its own means once; per-call statistics are not shared through persistent state.

Tangency preparation preserves both financial conventions: arithmetic mean uses covariance risk, while geometric CAGR continues to use the actual weighted return series and its sample standard deviation. Asset means are reused in both modes; covariance is prepared only for the arithmetic mode. Returned objective attributes and solver behavior remain unchanged. `optimize_return` prepares means once after validating the requested option, retaining case sensitivity and post-solve risk calculation.

No new optimizer, analytic gradient, variance objective, dependency or public tuning parameter is introduced. `ef_points` still calls the public `minimize_risk` per point; cache invalidation and subclass dispatch are not changed. Further sharing across frontier points is deferred until its benefit and invalidation/dispatch cost are demonstrated.

## Baseline and protocol

The baseline is the immediately preceding local implementation stage, with all earlier batching, GMV, ListMaker, rolling CAGR and Rebalance improvements shared. Checked-in `tests/helpers/optimizer_reference.py` overrides only the three remaining optimizer methods with their original source from revision `b33ca12bdd81b4107c2de484fff581eda1023ce1`. Test references require no Git history at runtime. These are incremental results, not before-all-optimizations timings.

The [benchmark script](https://github.com/mbk-dev/okama/blob/v4.0.1/analytics/benchmark_frontier_workflow.py) uses offline seeded monthly float64 histories, 600 months and 3 or 32 assets. Each complete workflow creates fresh objects and a fresh frontier cache. It parses resident CSV payloads using actual `QueryData.csv_to_series`, constructs ready synthetic asset objects, builds the frontier object, calculates 20 frontier points and geometric tangency weights, creates an annually rebalanced Portfolio and obtains monthly returns, CAGR, annual-risk history, rolling CAGR and rebalancing events. Both sides use the same fixed portfolio symbol so metadata comparisons remain meaningful.

Loading is deliberately bounded to CSV parsing and ready-object creation: HTTP, remote metadata and server response time are excluded from the paired benchmark; supplemental current-candidate live observations are reported separately below. The synthetic ready objects are not a substitute for measuring a production network loading session. No cold-import or peak-memory claim is made. The sum of per-stage medians need not equal the median complete-call time.

One warm pair precedes five alternating timed pairs. Complete output tables, series, keys and metadata are compared with rtol=atol=1e-12; event values are compared exactly. Profiles are separate additional candidate calls after the timing runs and must not be interpreted as timed baseline/candidate pairs. Python 3.11.17, NumPy 2.4.6, pandas 3.1.0 on AMD Ryzen 7 8845H; command-local BLAS/OpenMP threads are limited to one. The shared host has uncontrolled background load; distributions remain in JSON.

## Results

| Complete call | Before this stage | Candidate | Speedup |
|---|---:|---:|---:|
| `minimize_risk_3_assets` | 2.563 ms | 1.262 ms | 2.03× |
| `tangency_cagr_3_assets` | 6.491 ms | 3.485 ms | 1.86× |
| `tangency_mean_3_assets` | 8.517 ms | 3.035 ms | 2.81× |
| `optimize_return_3_assets` | 10.385 ms | 2.850 ms | 3.64× |
| `workflow_600x3_20_points` | 208.956 ms | 97.104 ms | 2.15× |
| `minimize_risk_32_assets` | 370.956 ms | 26.177 ms | 14.17× |
| `tangency_cagr_32_assets` | 1,492.456 ms | 176.741 ms | 8.44× |
| `tangency_mean_32_assets` | 1,338.344 ms | 84.789 ms | 15.78× |
| `optimize_return_32_assets` | 1,342.094 ms | 39.612 ms | 33.88× |
| `workflow_600x32_20_points` | 12,173.550 ms | 956.158 ms | 12.73× |

All 10 cases passed complete output comparisons. The maximum absolute numerical difference was zero on these measured inputs. All earlier performance improvements are shared by both sides. See [frontier-workflow.json](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/frontier-workflow.json) for raw distributions, per-stage timings, separate candidate profiles and measured source/script hashes.

| Pipeline stage, 32 assets | Before this stage | Candidate |
|---|---:|---:|
| `csv_load_and_ready_assets` | 64.507 ms | 77.533 ms |
| `frontier_setup` | 2.532 ms | 2.874 ms |
| `frontier_and_tangency` | 12,058.273 ms | 856.583 ms |
| `portfolio_setup` | 8.870 ms | 7.938 ms |
| `portfolio_metrics` | 6.578 ms | 5.820 ms |

The remaining frontier/tangency stage is about 90% of candidate complete CPU time in the 32-asset case; CSV/object loading is about 8%. Per-stage medians vary independently. Loading code is shared and is not optimized by this stage. The separate profile shows substantial pandas dispatch/conversion and SciPy finite-difference work. It does not justify changing objective math or solver settings without additional accuracy tests.

## Verification

Five RED assertions demonstrated repeated statistics in the original methods. Financial equivalence covers both tangency modes, complete upper/full frontiers, pairwise missing data, bounds and labels, changed inputs/bounds, target conversions/tolerances, solver errors and case validation. Existing forced-corner-fallback regressions passed. Final full suite: **874 passed, 3 skipped, 174 pre-existing pandas warnings**. Both Ruff commands and `git diff --check` passed. Independent read-only `grok-4.6-build` review found no blocking correctness issues; its note about case-sensitive return-optimizer options describes existing tested behavior. The reviewer executed no tools or writes.

The initial complete-workflow benchmark caught mismatched randomly generated portfolio names. Those timings were discarded; the final scenario assigns the same explicit `workflow.PF` symbol and reruns complete metadata comparisons.

## Supplemental live observation

The [live observation script](https://github.com/mbk-dev/okama/blob/v4.0.1/analytics/profile_live_frontier_workflow.py) also ran the standard public `EfficientFrontierSingle(["SPY.US", "BND.US", "GLD.US"], first_date="2005-01", last_date="2025-12", ccy="USD", inflation=False, n_points=20)` path against the API, then geometric tangency and the same Portfolio metrics. Native symbol loading retains the existing parallel asset loader. The shared history was 224 months. On the final current-candidate observation, wall time was **0.940 s**: frontier loading/setup 0.658 s, frontier calculation 0.086 s, tangency 0.006 s, Portfolio setup 0.186 s and metrics 0.005 s. An earlier separate symbol-constructor observation took 0.792 s. These are individual observations, not timing distributions or a paired original/candidate comparison; remote caching and scheduling may differ.

Loading/setup dominated this live observation. Eleven logical `API.connect` calls were recorded, including USD.FX metadata and return-series access during both constructors. Calls may overlap during asset loading, so their summed durations are not total network wall time. The USD return requests were for **different date ranges** (2005-01 to 2025-12 for the frontier; 1913-01-01 to 2100-01-01 for the Portfolio), while metadata calls used the same symbol. Reuse therefore needs explicit range/coverage rules; identical endpoint/symbol alone is insufficient. No API/cache/session behavior was changed in this stage.

[frontier-workflow-live.json](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/frontier-workflow-live.json) retains stage timings, logical request arguments, success flags and script digest. A separate initial serial-ready-Asset observation remains in the private evidence archive; it is not the standard symbol constructor and is not used as a network baseline. The final instrumented observation verified all requests succeeded. Command-local `OKAMA_API_TIMEOUT=3` only bounded this diagnostic.

```bash
OKAMA_API_TIMEOUT=3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poetry run python -m analytics.profile_live_frontier_workflow --output tmp/frontier-workflow-live.json
```

## Remaining bottlenecks and scope

Preparation removes repeated data statistics but leaves SLSQP finite-difference evaluations and the pandas financial arithmetic used to preserve solver trajectories. Network latency is outside the paired CPU experiment; the supplemental live observations are not a before/after network speedup experiment. CPU-only parser changes and FX/session request reuse remain separate opportunities; their value for live usage requires a measured real data-loading session. No unsupported whole-library or live-response speedup is inferred from these CPU timings.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 poetry run python -m analytics.benchmark_frontier_workflow --output tmp/frontier-workflow.json
```

Changes remain local and uncommitted. Earlier local stages are retained.
