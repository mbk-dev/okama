# Design: `monte_carlo_irr` + seedable shared Monte Carlo draw

Date: 2026-06-03
Status: Approved (brainstorming), pending implementation plan

## Goal

Add `PortfolioDCF.monte_carlo_irr()` — the distribution of money-weighted IRRs
across Monte Carlo forecast paths, the forward-looking counterpart of the
historical `PortfolioDCF.irr()`. It reuses the existing vectorized core
`dcf_calculations.irr_of_cashflow_matrix` (one solve over an `(M+1, mc_number)`
matrix).

This requires a prerequisite fix to the Monte Carlo engine: the random return
draw must be **generated once and shared** by all MC methods, and **seedable**
for reproducibility.

## Background and the problem this solves

`MonteCarlo.monte_carlo_returns_ts` is a `@property` with **no caching**: every
access regenerates fresh randomness (`np.random.normal` / `scipy.stats.*.rvs`).
`PortfolioDCF.monte_carlo_wealth` and `PortfolioDCF.monte_carlo_cash_flow` each
call it independently, so they are built from **different random scenarios**.

- For IRR this is fatal: a path's cash-flow stream and its terminal wealth must
  come from the **same** scenario, otherwise the per-path IRR mixes unrelated
  paths.
- It is also a pre-existing latent bug: `monte_carlo_cash_flow`'s depletion mask
  (`condition = mc_wealth_index == 0`) compares the cash flow of one draw against
  the wealth of a different draw.

Caching the draw fixes both and makes `monte_carlo_irr` a correct, thin consumer.

## Decisions (locked during brainstorming)

- **Approach B** — cache the MC return draw, shared by all MC methods.
- **`seed: int | None = None`** — an optional parameter defaulting to `None`
  (fresh randomness), not a fixed default. Matches numpy/scipy/sklearn
  convention; a fixed default would make every run return identical "random"
  scenarios, hiding the sampling variability that MC exists to show.
- **Seed lives in the MC configuration**, not as a per-method argument: a
  per-method seed could differ between methods and reintroduce the cross-method
  inconsistency Approach B removes. The single cached draw is generated from the
  configured seed.
- **Resampling** (getting a new draw when `seed is None`) is done **only via
  cache invalidation** on MC-parameter change (including re-calling
  `set_mc_parameters` or re-assigning `mc.seed`). No separate `regenerate()`
  method (YAGNI).
- **`monte_carlo_irr` returns `pd.Series`** (one IRR per path, length
  `mc_number`), like `monte_carlo_survival_period`. Nominal-only (IRR is a rate —
  no FV/PV switch, consistent with `irr()`).

## Component 1 — seedable, cached, shared draw (`okama/portfolios/mc.py`)

- Add instance attribute `self._returns_ts_cache: pd.DataFrame | None = None` and
  `self._seed` in `MonteCarlo.__init__` (new `seed: int | None = None` parameter).
- Add a `seed` property + setter. The setter validates (`int` or `None`) and calls
  `self._clear_cf_cache()`.
- Convert `monte_carlo_returns_ts` to generate-once-and-cache:

  ```python
  @property
  def monte_carlo_returns_ts(self) -> pd.DataFrame:
      if self._returns_ts_cache is None:
          self._returns_ts_cache = self._generate_returns_ts()
      return self._returns_ts_cache
  ```

  Move the current generation body into `_generate_returns_ts`, switching to a
  single `rng = np.random.default_rng(self.seed)`:
  - `norm`: `rng.normal(mu, sigma, (period_months, mc_number))`
  - `lognorm`: `scipy.stats.lognorm(...).rvs(size=[...], random_state=rng)`
  - `t`: `scipy.stats.t(...).rvs(size=[...], random_state=rng)`

- **Invalidation:** extend `MonteCarlo._clear_cf_cache` (mc.py, currently clears
  the parent's `_monte_carlo_wealth_fv` / `_monte_carlo_cash_flow_fv`) to also do
  `self._returns_ts_cache = None`. Because the existing setters for
  `distribution`, `distribution_parameters`, `period`, `mc_number` already call
  `_clear_cf_cache`, and the new `seed` setter calls it too, any MC-config change
  regenerates the draw and clears its derivatives in one chokepoint.
- **Do NOT** invalidate `_returns_ts_cache` on cash-flow-strategy changes
  (`CashFlow._clear_cf_cache` in `cashflow_strategies.py`): the draw is
  independent of the strategy. Changing a withdrawal amount must keep the same
  scenarios and only recompute wealth/cash flow.
- Add `seed` to `PortfolioDCF.set_mc_parameters(..., seed=None)` and assign
  `self.mc.seed = seed`.
- Update the `monte_carlo_returns_ts` docstring example to set an explicit `seed`
  so the shown output is deterministic.

## Component 2 — `PortfolioDCF.monte_carlo_irr` (`okama/portfolios/dcf.py`)

Placed next to `monte_carlo_survival_period`.

```python
def monte_carlo_irr(self) -> pd.Series:
    if self.cashflow_parameters is None:
        raise AttributeError("'cashflow_parameters' is not defined.")
    return_ts = self.mc.monte_carlo_returns_ts  # single shared (cached) draw
    cfp = self.cashflow_parameters
    wealth = return_ts.apply(
        dcf_calculations.get_wealth_indexes_fv_with_cashflow,
        axis=0, args=(None, None, cfp, "monte_carlo"),
    )
    wealth = wealth.apply(dcf_calculations.remove_negative_values, axis=0).fillna(0.0)
    cash_flow = return_ts.apply(
        dcf_calculations.get_cash_flow_fv,
        axis=0, args=(self.parent.symbol, cfp, "monte_carlo"),
    )
    # Zero a path's cash flow once its (floored) wealth is depleted — consistent draw.
    cash_flow = cash_flow.where(wealth.reindex(cash_flow.index) != 0, 0.0)
    terminal = wealth.iloc[-1]
    n_months, n_paths = cash_flow.shape
    flows = np.empty((n_months + 1, n_paths), dtype=float)
    flows[0, :] = -cfp.initial_investment
    flows[1:, :] = -cash_flow.to_numpy()
    flows[-1, :] += terminal.reindex(cash_flow.columns).to_numpy()
    irr = dcf_calculations.irr_of_cashflow_matrix(flows, periods_per_year=settings._MONTHS_PER_YEAR)
    return pd.Series(irr, index=cash_flow.columns, name="monte_carlo_irr")
```

Notes:
- The grid matches the historical `irr()` exactly: `initial_investment` at
  exponent 0 (MC start), monthly flows at `1..M` (`M = period * 12`), terminal at
  exponent `M`.
- `wealth` carries the `t0` row (MC start); `wealth.reindex(cash_flow.index)`
  aligns the depletion mask to the `M` forecast months. `terminal = wealth.iloc[-1]`
  is the last forecast month.
- Reuses the tested `dcf_calculations` helpers — the math is not duplicated, only
  re-orchestrated on a single consistent draw. This deliberately does not touch
  `monte_carlo_wealth` / `monte_carlo_cash_flow`.

## Behavioral changes / regression risks

- `mc.monte_carlo_returns_ts` accessed twice now returns the **same** cached draw
  (was a fresh draw each time). More correct, but a public-semantics change → run
  the full `tests/portfolio/mc` suite.
- RNG stream changes (`np.random.normal` → `rng.normal` via `default_rng`), so
  unseeded numeric MC results differ from before. Unseeded tests cannot assert
  exact values, so risk is low; any test asserting fresh-draw-per-access must be
  updated.

## Testing (TDD)

New test module `tests/portfolio/mc/test_monte_carlo_irr.py` (uses the MC test
fixtures / `synthetic_env`; `distribution="norm"`):

1. **MC CAGR oracle (primary):** with no intermediate cash flows
   (`frequency="none"`) and an explicit `seed`, each path's IRR equals that path's
   CAGR: `monte_carlo_irr()` equals `helpers.Frame.get_cagr(mc.monte_carlo_returns_ts)`
   (the same cached draw), to `atol=1e-9`. Parallels the historical CAGR invariant.
2. **Reproducibility:** the same `seed` (set via `set_mc_parameters`) twice yields
   identical Series (`assert_series_equal`).
3. **Shared-draw / caching:** two accesses of `mc.monte_carlo_returns_ts` are
   identical (`assert_frame_equal`); after `monte_carlo_irr()`, `monte_carlo_wealth`
   and `monte_carlo_cash_flow` are built from that same draw (e.g. recompute
   one column and compare).
4. **Cross-check:** for a withdrawal strategy, a few paths' `monte_carlo_irr`
   values match an independent per-column `scipy.optimize.brentq` solve to `1e-8`.
5. **`cashflow_parameters is None`** → `AttributeError`.
6. **Shape:** Series length equals `mc_number`; `name == "monte_carlo_irr"`.

Plus: confirm the existing `tests/portfolio/mc` suite still passes after the
draw-caching change (update any test that assumed fresh-draw-per-access).

## Files touched

- `okama/portfolios/mc.py` — `seed` property + ctor param, cached
  `monte_carlo_returns_ts` + `_generate_returns_ts`, `rng`-based generation,
  extended `_clear_cf_cache`, docstring example seed.
- `okama/portfolios/dcf.py` — `set_mc_parameters(..., seed=None)` and
  `monte_carlo_irr`.
- `tests/portfolio/mc/test_monte_carlo_irr.py` — new test module.

No new third-party dependency: `numpy` + `scipy` (already required) suffice.
The IRR core `irr_of_cashflow_matrix` is reused unchanged.
