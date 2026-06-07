# Ex-post Tracking Error for `Portfolio` (#61) — design

Date: 2026-06-07
Status: approved (brainstorming session)

## Background

GitHub issue [#61](https://github.com/mbk-dev/okama/issues/61) asks for an
**ex-post** Tracking Error (TE) method on `Portfolio`. Ex-post TE is the
realized dispersion of the difference between portfolio returns and benchmark
returns, computed from the historical time series (as opposed to the
**ex-ante** TE forecast from weights and a covariance matrix — tracked
separately in [#88](https://github.com/mbk-dev/okama/issues/88)).

Existing machinery:

- `AssetList.tracking_error(rolling_window=None)` (`okama/asset_list.py:1386`)
  computes ex-post TE for assets against the benchmark in the first position
  of the symbols list.
- The math lives in `helpers.Index.tracking_error(ror)`
  (`okama/common/helpers/helpers.py:651`): expanding
  `sqrt(cumsum(d²)/n) · √12`, where `d_t` is the monthly return difference.
  This is a **root-mean-square (RMS)** of differences — it is *not* centered
  around the mean difference, so it mixes the systematic lag (tracking
  difference) into the dispersion measure.
- The classic definition (Hwang & Satchell, *Tracking Error: Ex-Ante versus
  Ex-Post Measures*, 2001, eq. 2) is the **centered sample standard
  deviation with Bessel's correction**:
  `sqrt(1/(T−1) · Σ(d_t − d̄)²) · √12` — pure volatility of deviations.
- `Portfolio` has no benchmark-comparison methods at all. The existing
  workaround is `ok.AssetList([benchmark, pf]).tracking_error()`, since
  `ListMaker` accepts any object with `.symbol` and `.ror` as an asset.

## Goals

- `Portfolio.tracking_error(benchmark, ...)` — direct ex-post TE for a
  portfolio against an explicit benchmark.
- A `method` switch between the historical okama formula (RMS, default) and
  the classic centered std (Hwang & Satchell eq. 2) — in **both**
  `Portfolio.tracking_error` and `AssetList.tracking_error`.
- Docstrings explaining the difference between the two formulas.

## Non-goals / compatibility contract

- Ex-ante TE is out of scope (issue #88).
- `AssetList.tracking_error()` called without `method` keeps producing
  exactly the current values (default `method="rms"` — backward compatible).
- No benchmark attribute on the `Portfolio` constructor; the benchmark is a
  method parameter only.
- TE_MAD (mean absolute deviation variant from the paper) is not implemented.

## API

New method on `Portfolio` (`okama/portfolios/core.py`, near
`get_sharpe_ratio` / `get_sortino_ratio`):

```python
def tracking_error(
    self,
    benchmark,                       # str | Asset | Portfolio (asset-like)
    rolling_window: int | None = None,
    method: Literal["rms", "std"] = "rms",
) -> pd.Series
```

- `benchmark` — ticker string (e.g. `"SP500TR.INDX"`) or an asset-like
  object (`Asset`, `Portfolio`): anything `ListMaker` accepts as an asset.
  Required parameter — `Portfolio` has no "first symbol is the benchmark"
  convention.
- `rolling_window` — same semantics as in `AssetList`: `None` → expanding
  TE, integer N (≥ 12) → rolling TE over N months.
- `method` — `"rms"` (default, historical okama formula) or `"std"`
  (Hwang & Satchell eq. 2).
- Returns an annualized (×√12) `pd.Series` named after `self.symbol`.

Changed method on `AssetList`:

```python
def tracking_error(
    self,
    rolling_window: int | None = None,
    method: Literal["rms", "std"] = "rms",
) -> pd.DataFrame
```

## Architecture (approach A — internal AssetList reuse)

`Portfolio.tracking_error` builds an internal
`AssetList([benchmark, self], ccy=self.currency, inflation=False)` and
delegates to its `tracking_error(rolling_window, method)`, then squeezes the
single-column result into a `pd.Series` named `self.symbol`.

This promotes the documented workaround into a first-class API call and gets
for free from `ListMaker`:

- benchmark return currency conversion into the portfolio currency;
- date alignment (intersection of available histories);
- acceptance of ticker strings and asset-like objects uniformly.

The portfolio enters the internal list as an asset-like object, so its
already-computed `ror` (with its rebalancing strategy) is reused, not
recomputed from scratch.

The rejected alternative (approach B — fetch benchmark `ror` directly,
convert currency and align dates manually) would duplicate `ListMaker`
logic and risk subtle inconsistencies.

## Math (`helpers.Index.tracking_error`)

The helper gains a `method` parameter; `d_t = r_{p,t} − r_{b,t}` are monthly
return differences:

- **`"rms"`** (default — current formula, unchanged):
  expanding `sqrt(cumsum(d²)/n) · √12`. Not centered; mixes volatility with
  the systematic lag.
- **`"std"`** (Hwang & Satchell eq. 2):
  `d.expanding().std() · √12` — centered sample std with Bessel's correction
  (pandas default `ddof=1`). The first expanding point is `NaN` (std of one
  observation is undefined) and is dropped from the output.

Rolling variant goes through the existing `helpers.Index.rolling_fn` with a
partial function carrying `method`.

Existing guards stay for both methods: history < 12 months →
`ShortPeriodLengthError`; fewer than 2 columns → `ValueError`.

## Error handling / edge cases

- Benchmark in another currency → converted to the portfolio currency
  (automatic via the internal `AssetList`).
- Benchmark with shorter history → the period is clipped to the
  intersection (standard `AssetList` semantics); the resulting period is
  visible in the returned Series index.
- Invalid `method` value → `ValueError` listing allowed values.
- Benchmark passed as a `Portfolio` object → works like any asset-like
  object.

## Testing (TDD, RED → GREEN)

- `Portfolio.tracking_error(benchmark=str)` — expanding; `.iloc[-1]` matches
  the reference `ok.AssetList([bench, pf]).tracking_error()` value.
- `method="std"` — value matches a manual
  `(pf.ror − bench.ror).std(ddof=1) · √12` computation over the full period.
- `rolling_window=24` — series length and values.
- Benchmark as an `Asset` object and as a `Portfolio` object.
- `AssetList.tracking_error(method="std")` — new mode; `method="rms"` and
  the no-argument call — regression test that current values are unchanged.
- Errors: invalid `method`, history < 12 months.

## Documentation

- English docstrings explaining the RMS vs centered-std difference: RMS
  includes the systematic lag (tracking difference), centered std measures
  pure volatility of deviations; reference Hwang & Satchell (2001); note the
  measure is ex-post (see #88 for ex-ante).
- Add the new `Portfolio` method to the Sphinx docs (`docs/`), Portfolio
  method list.
