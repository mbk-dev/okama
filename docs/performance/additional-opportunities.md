# Additional calculation performance opportunities

Offline audit on 2026-10-26, starting with `Rebalance` and `ListMaker`. This is research and prioritization, not an implementation or release. The existing uncommitted issue-127 changes are preserved; this audit does not edit library files.

## Main findings and suggested order

Times are medians in milliseconds. Ratios compare actual current computations with deliberately narrow experimental alternatives; they are not promised end-to-end library speedups. Preparation is included except where explicitly stated below.

| Order | Candidate | Workload | Current ms | Prototype ms | Ratio | Proposed change |
|---|---|---|---:|---:|---:|---|
| 1 | ListMaker: first LTM dividend yield | 600 months × 16 assets; dividend/price data already loaded | 3260.759 | 2.729 | 1194.7x | rolling sums and aligned vector division |
| 2 | ListMaker: construct from ready objects | 252 months × 32 assets; complete AssetList constructor | 13.518 | 1.470 | 9.2x | skip the thread pool when no I/O is needed |
| 3 | Rebalance: conditional wealth | 600 months × 16 assets; absolute threshold 5% | 248.157 | 2.793 | 88.9x | NumPy loop with preallocated output, construct pandas once |
| 4 | Rebalance: repeated annual weights | 252 months × 3 assets × 2048 weight vectors | 2605.556 | 6.744 | 386.4x | prepare period groups/growth once, then batch weights |
| 5 | Single-period GMV optimizer | 252 months × 32 assets; actual SLSQP call | 64.521 | 5.563 | 11.6x | prepare covariance once per solve |
| 6 | Rolling CAGR helper | 600 months × 16 assets; 60-month window | 240.279 | 0.770 | 312.2x | rolling sum of log growth on an eligible input path |
| 7 | Rebalance: calendar wealth | 600 months × 16 assets; annual calendar, no threshold | 25.721 | 1.477 | 17.4x | reuse existing vectorized EF calculation for wealth |
| 8 | CSV time-series parser | 6000 monthly CSV rows; local parsing only | 7.686 | 4.044 | 1.9x | C engine instead of Python engine where compatible |

Order reflects confidence, reach, implementation risk and effort, rather than sorting by the largest synthetic ratio. Start with ready-object construction and LTM dividend yield for relatively contained wins, then address conditional rebalancing and prepared multi-period calculations with financial equivalence tests. GMV preparation is another contained candidate. The rolling CAGR prototype needs explicit handling of nonpositive growth factors.

## ListMaker

### First dividend-yield calculation

Source: [`ListMaker._assets_dividend_yield()`](../../okama/common/make_asset_list.py#L532), reached by `AssetList.dividend_yield` and dividend summary methods.

The implementation loops over every asset and every price month, repeatedly slices dividends and all preceding prices, and extends a pandas Series one cell at a time. The 600-month × 16-asset profile confirms this overhead; the research prototype uses `rolling(12, min_periods=1).sum()` and vector division by aligned monthly closes.

The timing clears only the computed yield cache. Dividend/price inputs are already resident, so it isolates CPU work rather than network latency. Both paths operate on the same inputs; warm yield-cache access remains approximately a microsecond and will not benefit. This is a first-calculation improvement.

Before implementation: preserve calendar-window boundaries, missing price months, currencies, forecast removal, zero-dividend behavior, signed/cancelling distributions and division by zero. A blind row-based rolling window is insufficient if calendar months are missing. The prototype only covers complete monthly histories and positive base-currency dividends.

### Ready asset objects and thread-pool overhead

Source: [`ListMaker._get_asset_obj_dict()`](../../okama/common/make_asset_list.py#L158).

Every list of two or more assets currently enters `joblib.Parallel(backend="threading")`, including ready Asset-like objects requiring only validation. The experiment replaces the scheduling wrapper with sequential execution of the **same delayed validation function** and returns the same objects in the same order. It also measures the complete actual `AssetList` constructor. For 32 assets, the complete constructor improves from about 13.5 ms to 1.5 ms; this is stronger evidence than timing only the tiny validation kernel.

Keep concurrency for symbol strings that need I/O. Mixed ready objects and strings need order preservation and identical exception propagation. Do not replace `_make_list()` concat: it already collects series and concatenates once.

### Repeated currency data and HTTP setup — counted, not timed

Source: [`ListMaker._make_ror()`](../../okama/common/make_asset_list.py#L322), [`_adjust_price_to_currency_monthly()`](../../okama/common/make_asset_list.py#L377), [`Asset.__init__()`](../../okama/asset.py#L28), [`API.connect()`](../../okama/api/api_methods.py#L33).

For 16 already constructed EUR assets in USD with identical histories, instrumentation observed **16 separate `EURUSD.FX` Asset constructions**, plus the base `USD.FX` construction. A construction-local cache keyed by currency pair and effective date window could reduce the identical FX constructions from 16 to 1. The current `Asset` constructor requests symbol metadata and returns; the inspected query/API methods do not cache these historical calls. `API.connect()` also creates a fresh requests Session and adapters per call, preventing persistent connection-pool reuse across calls.

No real HTTP requests or latency benchmark were performed. The observation demonstrates duplicate work, not a measured network speedup. Any broader cache needs freshness/TTL, size bounds, copy-on-read for mutable pandas objects, error behavior and a thread-safe ownership model. Prefer construction-local deduplication before global caching. Connection reuse needs a separate bounded real-network test and an explicit thread-safety design.

## Rebalance

### Threshold rebalancing

Source: [`_rebalance_by_condition()`](../../okama/common/helpers/rebalancing.py#L216), [`_check_if_rebalancing_required()`](../../okama/common/helpers/rebalancing.py#L251).

Each month creates pandas Series, allocates aligned arithmetic results and extends an initially empty output Series. Threshold checking also constructs target weights and computes both absolute and relative differences even when one threshold is disabled. The profile confirms these operations dominate; the conditional prototype keeps the required chronological loop but performs the arithmetic on NumPy arrays and preallocates wealth output.

The prototype compares **wealth values only**, for `period="none"` and absolute deviation 0.05. It omits the Result wrapper, event labels and per-asset wealth output. Its ratio is an estimate of core-computation potential, not a complete drop-in replacement. Production coverage must include relative thresholds, zero target weights, combined thresholds, exact `>` boundary comparisons, event date/type, monthly/quarterly/half-year/year calendars, nonpositive balances and NaN handling. Array conversion alone is not a reason to change these semantics.

### Repeated weights: prepare once, then batch

Source: [`wealth_ts_ef()`](../../okama/common/helpers/rebalancing.py#L327), [`return_ror_ts_ef()`](../../okama/common/helpers/rebalancing.py#L381), and `EfficientFrontier.get_monte_carlo()` / `get_grid_portfolios()` in [multi_period.py](../../okama/frontier/multi_period.py).

The current EF function is already vectorized **within one portfolio**. The remaining repetition is regrouping the same return matrix, recalculating within-period growth, and building pandas intermediates for every new weight vector. Multi-period Monte Carlo/grid paths call it in a Python loop; scalar SLSQP evaluation can also reuse invariant prepared arrays without batching the optimizer itself.

The prototype prepares period IDs using the exact existing pandas Grouper, computes within-period growth once, and evaluates all weights together. Comparison covers all five current periods (`none`, `month`, `quarter`, `half-year`, `year`), 128 and 2048 vectors, and histories starting in March to include a partial first calendar period. It compares the **return-series calculation**, excluding weight generation, risk/CAGR reductions and final public result construction. The annual 2048-vector kernel improves from about 2.61 s to 6.74 ms on this workload; no equivalent whole-method gain has been measured.

Unlike single-period covariance batching, this prototype materializes periods × portfolios matrices. At 252 × 2048, each float64 matrix alone is about 3.94 MiB; several are live. A production version must chunk and benchmark peak memory, sparse/incomplete calendars, NaNs, zero wealth and all rebalancing modes. The existing per-weight `_ror_cache` already exists; do not propose adding that same cache again. Invariant return-data preparation is the different opportunity.

### Ordinary calendar calculation and reuse of results

Source: [`wealth_ts()`](../../okama/common/helpers/rebalancing.py#L116), [`Portfolio.ror`](../../okama/portfolios/core.py#L389), [`Portfolio.rebalancing_events`](../../okama/portfolios/core.py#L209).

Unconditional calendar `wealth_ts()` still loops over resampled periods even though `wealth_ts_ef()` provides an existing vectorized wealth path. Comparison uses that **existing library method**, adding the initial 1000 point; it verifies portfolio wealth values only. Event series and optional per-asset paths still need a complete compatible implementation.

`Portfolio.ror` already has a fast path for monthly constant weights and caches its result. Do not claim this proposal speeds that existing path. `Portfolio.rebalancing_events`, however, directly recalculates `wealth_ts()` on each access; retaining a compatible shared result could avoid recalculation after return-series evaluation. That cache proposal was found by static tracing and was **not benchmarked**. It needs invalidation for weights, strategy and input changes.

## Additional opportunities outside the starting classes

### Invariant covariance inside scalar optimization

Source: [`EfficientFrontierSingle.gmv_monthly_weights`](../../okama/frontier/single_period.py#L177), [`minimize_risk()`](../../okama/frontier/single_period.py#L597), [`get_tangency_portfolio()`](../../okama/frontier/single_period.py#L321).

Issue 127 accelerates sampled weight sets, but these scalar optimizer paths still call helpers that recompute covariance/means during repeated objective/constraint evaluations. The GMV experiment runs the **actual unchanged SLSQP call**, preparing covariance once and temporarily replacing only its scalar risk helper in the benchmark process. The same starting point, bounds, solver and constraints remain in use, and returned weights agree at the research tolerance. The 32-asset run called covariance **264 times** in the baseline, versus once in the prototype.

Prepare immutable statistics once per solve rather than globally caching potentially mutable return data. Do not silently change the scalar objective to variance or change solver tolerances under a preparation-only optimization. Other optimizer methods were inspected but not quantitatively benchmarked here.

### Rolling CAGR

Source: [`Frame.get_rolling_fn()`](../../okama/common/helpers/helpers.py#L287), [`AssetList.get_rolling_cagr()`](../../okama/asset_list.py#L583), `Portfolio.get_rolling_cagr()`.

The generic helper uses `rolling.apply()` and calls a Python CAGR function for each window/column. For strictly positive growth factors, a specialized CAGR path can use a rolling sum of `log1p` values and `expm1`; the prototype compares that numerical kernel to the actual existing helper. Do not replace the generic function for arbitrary callbacks. Preserve NaN/min-periods rules, output window boundaries, annualization, total losses at -100%, negative growth factors and overflow behavior; these exceptional cases require fallback or explicit equivalent handling. Only returns safely above -1 were measured.

### CSV parser

Source: [`QueryData.csv_to_series()`](../../okama/api/data_queries.py#L21).

The explicit Python CSV engine is slower than pandas' C engine on the measured well-formed single-column monthly series. The parser experiment includes date parsing, PeriodIndex conversion and column squeezing, and checks complete Series equality. This is CPU-only parsing; the improvement is modest for 252 rows and does not measure HTTP latency. Verify empty/header-only payloads, malformed CSV, date formats and supported pandas versions before switching defaults.

## Measurement protocol and limitations

- Python 3.11.17, NumPy 2.4.6, pandas 3.1.0; Ryzen 7 8845H on Linux. Base revision `b33ca12bdd81b4107c2de484fff581eda1023ce1` plus the existing local issue-127 change. The newly explored library methods were not edited.
- Offline synthetic monthly float64 returns, seed 127, start March 1970. Sizes are named in each case. Ready assets use fake objects; currency loading alone is replaced. Corrected ListMaker fixtures cover the full requested 252/600 months, asserted in the script. An initial fixture that truncated pre-1990 data was detected and its affected measurements discarded and rerun.
- One warm pair, then alternating baseline/prototype order. Five repetitions for smaller cases; three for dividend, repeated-weight EF and rolling CAGR cases. All distributions and diagnostic profiles are retained.
- Numerical comparisons use `rtol=atol=1e-11`. CSV additionally checks complete Series equality. Ready-object validation checks object identity and key order. Experimental prototypes are intentionally narrower than public production contracts.
- Benchmark-process-only environment: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4`. No global BLAS settings or services were changed.
- Known pandas deprecation warnings are suppressed only within the exploratory benchmark so warning rendering does not dominate the timing. They are not disabled in the library or test configuration.
- A shared laptop has uncontrolled scheduling/frequency/background load. Ratios indicate priorities, not performance promises. Resident-input timing excludes network and object setup except the explicitly measured full constructor.
- External read-only exploration ended at its finite turn cap without a final report and was not accepted as a completed independent audit. Findings above were verified directly by the coordinator through source tracing and numerical experiments.

## Complete measured cases

| Case | Current ms | Prototype ms | Ratio |
|---|---:|---:|---:|
| `rebalance_conditional_252x3` | 113.455 | 1.171 | 96.9x |
| `rebalance_calendar_month_252x3` | 131.067 | 1.578 | 83.1x |
| `rebalance_calendar_year_252x3` | 11.270 | 1.411 | 8.0x |
| `rebalance_conditional_600x16` | 248.157 | 2.793 | 88.9x |
| `rebalance_calendar_month_600x16` | 388.775 | 1.671 | 232.7x |
| `rebalance_calendar_year_600x16` | 25.721 | 1.477 | 17.4x |
| `rebalance_ef_batch_none_128` | 46.239 | 0.188 | 245.3x |
| `rebalance_ef_batch_none_2048` | 634.464 | 3.896 | 162.8x |
| `rebalance_ef_batch_month_128` | 176.774 | 1.148 | 154.0x |
| `rebalance_ef_batch_month_2048` | 2554.746 | 14.634 | 174.6x |
| `rebalance_ef_batch_quarter_128` | 168.172 | 1.063 | 158.3x |
| `rebalance_ef_batch_quarter_2048` | 2555.278 | 9.519 | 268.4x |
| `rebalance_ef_batch_half-year_128` | 160.551 | 0.732 | 219.5x |
| `rebalance_ef_batch_half-year_2048` | 2562.573 | 6.650 | 385.3x |
| `rebalance_ef_batch_year_128` | 156.486 | 0.737 | 212.2x |
| `rebalance_ef_batch_year_2048` | 2605.556 | 6.744 | 386.4x |
| `listmaker_ready_objects_3` | 11.588 | 0.102 | 113.8x |
| `listmaker_full_init_3` | 12.940 | 0.733 | 17.6x |
| `listmaker_ready_objects_32` | 11.581 | 0.159 | 73.0x |
| `listmaker_full_init_32` | 13.518 | 1.470 | 9.2x |
| `listmaker_ready_objects_128` | 11.612 | 0.403 | 28.8x |
| `listmaker_full_init_128` | 15.988 | 3.913 | 4.1x |
| `listmaker_dividend_yield_252x3` | 272.762 | 1.172 | 232.6x |
| `listmaker_dividend_yield_600x16` | 3260.759 | 2.729 | 1194.7x |
| `query_csv_parser_252` | 1.012 | 0.748 | 1.4x |
| `query_csv_parser_6000` | 7.686 | 4.044 | 1.9x |
| `rolling_cagr_600x16_window60` | 240.279 | 0.770 | 312.2x |
| `gmv_monthly_3_assets` | 7.409 | 1.998 | 3.7x |
| `gmv_monthly_32_assets` | 64.521 | 5.563 | 11.6x |

## Reproduction and evidence

From the repository root, with the existing Poetry environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 poetry run python -m analytics.benchmark_performance_opportunities --output tmp/performance-opportunities.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 poetry run python -m analytics.benchmark_performance_opportunities --optimizers-only --output tmp/optimizer-opportunities.json
```

The optional `--listmaker-only` runs corrected ListMaker and adjacent cases without repeating Rebalance. The script is [analytics/benchmark_performance_opportunities.py](../../analytics/benchmark_performance_opportunities.py); it makes no persistent library changes. Scoped patches are confined to the benchmark process and restored after each measurement.

Evidence: [Rebalance timings and profiles](opportunities/rebalance.json), [ListMaker and adjacent timings/profiles](opportunities/listmaker.json), [GMV solver timings/counts](opportunities/optimizers.json). Each source JSON retains its measurement script digest; the final script additionally includes the fixture correction and optimizer-only runner. Final script SHA-256: `38e41300286abe79e6752a1b258fc33947c128866fe89a907e6ab438e499e427`.

## Verification and delivery

The exploratory runs completed with numerical assertions. `poetry run pytest -q`: **547 passed, 3 skipped, 371 existing pandas warnings**. `poetry run ruff check .`, `poetry run ruff format --check .` and `git diff --check` passed. Library content is unchanged from the completed local issue-127 implementation; no new optimization was applied, no dependency was added, and no commit, tracker mutation or deployment was made.
