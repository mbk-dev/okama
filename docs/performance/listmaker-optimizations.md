# ListMaker performance implementation

Implemented 2026-10-26 after the [offline opportunities audit](additional-opportunities.md). Scope: ready-object construction and the first LTM dividend-yield calculation. Rebalance and the other audited opportunities are not implemented by this change. The earlier local issue-127 frontier optimization is preserved.

## Result

`ListMaker._get_asset_obj_dict()` validates a list of ready Asset-like objects directly instead of starting a joblib thread pool. The same validator handles short histories, key order and duplicate objects. Lists containing symbols requiring loading retain the threaded path.

`ListMaker._assets_dividend_yield` now delegates arithmetic to a private `_get_ltm_dividend_yield()` helper. For normal monthly inputs it creates a complete calendar, fills missing dividend values with zero, calculates trailing twelve-month sums once, then aligns to prices and divides. Missing **price** months do not lengthen the dividend window. FX conversion, forecast removal, zero/net-zero dividend handling and the computed-result cache are unchanged.

The fast path requires sorted unique monthly PeriodIndexes, NumPy numeric prices and float64 dividends without infinities. Other inputs retain the original scalar date-slice calculation. In particular, tests demonstrated that pandas rolling changes infinity handling and float32 accumulation; those cases deliberately use the fallback. Nullable dtypes and duplicate indexes also retain the original behavior.

## Actual-method measurements

One warmup pair followed by five alternating baseline/candidate calls. Timings include the complete inherited AssetList constructor for constructor cases and the full dividend-yield calculation for yield cases. Dividend/price inputs are resident; only the computed yield cache is cleared. Network latency and initial asset downloads are excluded. Synthetic seed 127, float64 monthly data from March 1970; offline currency data covers the complete history (252 or 600 months).

| Case | Baseline ms | Candidate ms | Ratio |
|---|---:|---:|---:|
| `ready_constructor_3` | 13.585 | 0.747 | 18.2x |
| `ready_constructor_32` | 14.887 | 1.983 | 7.5x |
| `ready_constructor_128` | 18.334 | 5.476 | 3.3x |
| `cold_yield_252x3` | 359.726 | 2.090 | 172.1x |
| `cold_yield_600x16` | 4751.886 | 6.874 | 691.3x |

The 600-month × 16-asset first calculation improves from about 4.75 s to 6.87 ms, approximately 691x in this experiment. Cached access is already about 1.7 microseconds and receives no claimed benefit. The full 32-ready-asset constructor improves from 14.9 ms to 1.98 ms, approximately 7.5x. These are measured library paths, not the narrower audit prototypes; all output DataFrames agree at `rtol=atol=1e-12`, including index/column metadata. Maximum absolute numerical difference is 1.04e-17.

Baseline library source is loaded directly from Git revision `b33ca12bdd81b4107c2de484fff581eda1023ce1` into a separate module. The baseline AssetList facade exposes the original inherited ListMaker constructor and dividend property; the candidate uses the actual current `ok.AssetList`. The only data-boundary replacement is offline currency loading. Weights and financial formulas are not changed.

Environment: Python 3.11.17, NumPy 2.4.6, pandas 3.1.0, AMD Ryzen 7 8845H on Linux. `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4` are set only for benchmark processes. Shared-machine scheduling and frequency remain uncontrolled; distributions are retained rather than treating ratios as universal guarantees. Generated output and the dense monthly calendar require storage proportional to history length; no global cache is introduced.

Measured/final `okama/common/make_asset_list.py` SHA-256: `fd9601ea0edc0ced6421850cd3a1057c6b396ff55f7fbc409c19b7168816e800`. Benchmark SHA-256: `41d75098fb1b070f98edf2198aef56e879e5324bb0a371c8d417f16b34fa304b`. No commit, push, release or deployment has been made.

## Tests and review

TDD first demonstrated a pool construction for ready inputs and 48 DateOffset calls for two assets over 24 months. Both requirements now pass without those repeated operations. Tests cover hand-calculated 12-month boundaries, leap-year months, missing dividend/price observations, NaNs and all-NaN windows, zero/net-zero dividends, zero prices, forecasts, FX conversion, cache identity, duplicate labels/objects, mixed loading, short histories, float32/infinity accumulation and nonstandard fallback inputs.

Verification also exposed a pre-existing order-dependent test problem: `tests/test_list_maker.py` reloaded the ListMaker module while AssetList still inherited the earlier class definition. Running that test followed by the AssetList spy test reproduced the failure. The reload was masking another test-isolation defect: Monte Carlo and FinPlan fixtures declared package scope, but their directories lacked Python package boundaries, so patches remained active across other tests. A Monte Carlo test followed by the short-history validator reproduced this leak. Explicit package markers now contain those fixtures; removing the unnecessary reload preserves class identity and keeps the actual validation assertion. No production workaround was added.

Independent read-only review by Grok completed in one turn with no significant findings. The coordinator verified its conclusions against the implementation, tests and retained numerical comparisons. The reviewer made no file/tool changes.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=4 poetry run python -m analytics.benchmark_listmaker --output tmp/listmaker.json
poetry run pytest -q
poetry run ruff check .
poetry run ruff format --check .
```

See [the benchmark script](https://github.com/mbk-dev/okama/blob/v4.0.1/analytics/benchmark_listmaker.py) and [raw timing distributions](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/listmaker-optimizations.json). The script compares frozen baseline and candidate in one process without checking out an older worktree or modifying library methods.

## Final verification

`poetry run pytest -q`: **568 passed, 3 skipped**, 203 remaining existing pandas deprecation warnings (baseline: 547 passed, 3 skipped, 371 warnings). Both Ruff commands pass, as does `git diff --check`. The ListMaker test files pass without warnings; annual dividend summaries and other unchanged modules retain existing warnings. All checks run under the Poetry Python 3.11.17 environment. Changes remain local and uncommitted.
