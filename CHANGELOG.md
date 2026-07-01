# Changelog

All notable changes to **okama** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.3] - 2026-07

Fixes two efficient-frontier failures where the risk optimizer could not reach a
single-asset corner point — the leftmost (minimum-CAGR) point of the rebalanced
`EfficientFrontier` and the maximum-return point of `EfficientFrontierSingle` — so
both frontiers are drawn for the affected asset sets instead of raising a
`RuntimeError`. The second failure surfaced with the stricter SLSQP solver in
scipy 1.18.

### Fixed

- `EfficientFrontier.minimize_risk` (rebalanced/multi-period frontier) raised
  `RuntimeError: No solution found for target CAGR value: ...` at the leftmost
  frontier point when the target CAGR equalled the minimum-CAGR asset's own CAGR
  and SLSQP failed to converge to that single-asset vertex from the multi-start
  initial guesses. The method now falls back to the deterministic single-asset
  corner portfolio (mirroring the existing guard in `_maximize_risk`), so the
  efficient frontier is drawn for such asset sets instead of failing.
- `EfficientFrontierSingle.minimize_risk` (single-period frontier) raised
  `RuntimeError: No solutions were found` at the maximum-return frontier point
  when the target return equalled a single asset's own mean return and SLSQP
  failed to converge to that single-asset vertex from the equal-weights start
  (surfaced by the stricter SLSQP in scipy 1.18). The method now falls back to
  the deterministic single-asset corner portfolio, so `EfficientFrontierSingle.ef_points`
  is drawn for such asset sets instead of failing.

## [2.2.2] - 2026-06

Adds new analytics — ex-post tracking error for `Portfolio` and an RMS/std
`method` switch for tracking error across `Index`, `AssetList`, and `Portfolio`,
plus inflation-adjusted and price-only drawdown views — and fixes the
`maintain_balance` goals of `find_the_largest_withdrawals_size()` together with
duplicated efficient-frontier points under a thread backend.

### Added

- `Portfolio.tracking_error(benchmark, rolling_window, method)` — ex-post
  tracking error of a portfolio against a benchmark (#61). The benchmark may be
  a string ticker or an asset-like object (`Asset`, `Portfolio`); the method
  delegates to `AssetList`.
- `method` parameter for `AssetList.tracking_error` and the underlying
  `helpers.Index.tracking_error`: `"rms"` (default, legacy uncentered
  root-mean-square) or `"std"` (centered sample standard deviation with
  Bessel's correction).
- `AssetList.real_drawdowns` and `Portfolio.real_drawdowns` — drawdowns of the
  inflation-adjusted wealth index, exposing purchasing-power losses hidden by
  nominal growth (requires `inflation=True`) (#51).
- `AssetList.price_drawdowns` and `Portfolio.price_drawdowns` — drawdowns based
  on close prices not adjusted for dividends, which can differ markedly from the
  total-return `drawdowns` for high-dividend assets (#44).

### Fixed

- `PortfolioDCF.find_the_largest_withdrawals_size()` raised
  `ValueError: target_survival_period must be less than Monte Carlo simulation period`
  for the `maintain_balance_pv` and `maintain_balance_fv` goals on any Monte
  Carlo period ≤ 27, even though those goals never use `target_survival_period`
  and the caller never passed it (#90). The parameter is now validated only for
  the `survival_period` goal.
- `EfficientFrontier.ef_points` produced duplicated right-part points under a
  thread-based joblib backend, because the right-part worker both appended its
  row to the shared list and returned it (#86). The worker now only returns the
  row, matching the left-part worker.
- `Frame.kstest_series` (used by `kstest_for_all_distributions` and the
  distribution-fit properties of `AssetList` / `Portfolio`) raised
  `TypeError: ndtr() takes from 1 to 2 positional arguments but 3 were given`
  with scipy 1.18.0 on Python ≥ 3.12, which routes the named-distribution
  `kstest(..., "norm", args=...)` call through the `ndtr` ufunc. The
  Kolmogorov–Smirnov test now passes a frozen-distribution CDF, which is
  numerically equivalent and compatible with scipy 1.17 and 1.18.

### Docs

- Clarified that `tracking_error` values are decimal fractions, not percentages.
- `PortfolioDCF.find_the_largest_withdrawals_size()` docstring now notes that
  `IndexationStrategy` / `PercentageStrategy` subclasses (e.g.
  `CutWithdrawalsIfDrawdown`, `VanguardDynamicSpending`) are supported.

## [2.2.1] - 2026-06

Fixes the multi-period Efficient Frontier around single-asset corner points —
the right part of the frontier now always terminates at the corner asset
(no dominated "hook", no silently missing right part, pairwise frontiers reach
the asset dots) — and removes pandas 3 deprecation warnings.

### Fixed

- `EfficientFrontier.ef_points` drew a dominated hook near the max-CAGR corner
  (#84): SLSQP started exactly at the optimal vertex of the bounds fails
  spuriously, and the fallback start converged to an interior local maximum.
  `EfficientFrontier._maximize_risk()` now keeps the better of the optimizer
  result and the 100% single-asset portfolio whenever the target CAGR equals an
  asset's own CAGR, so the right part of the frontier ends exactly at the
  corner asset with monotonically increasing risk.
- The right part of `EfficientFrontier.ef_points` could silently disappear
  together with its corner point when the right CAGR span was much narrower
  than the left one (the point-count formula produced an empty target range).
- Pairwise efficient frontiers (`EfficientFrontier.plot_pair_ef()`) stopped
  short of the asset point when the best rebalanced mix beat the asset by less
  than 1% of CAGR (#87). An asset is now considered to be the global max-CAGR
  point only when both its CAGR and its risk match it, and a narrow-but-real
  right CAGR span is drawn instead of being treated as degenerate.
- `EfficientFrontier.plot_pair_ef()` ignored the parent's
  `rebalancing_strategy`, always computing pair frontiers with the default
  yearly rebalancing.
- `Pandas4Warning` on pandas 3 (#85): dropped the deprecated `copy` keyword in
  `symbols_in_namespace()` and `Index.rolling_fn()` (slated for removal in
  pandas 4.0).

### Changed

- The `EfficientFrontier.ef_points` target grid now samples every asset's CAGR
  lying inside the range (previously only the minimum-variance asset's), so the
  frontier polyline passes exactly through single-asset points on the boundary.
  The number of rows in `ef_points` / `mdp_points` can therefore slightly
  exceed `n_points`.

### Docs

- README refreshed: fixed broken images on PyPI, added a hero image, an MCP
  server section and a uv install option.

## [2.2.0] - 2026-06

Makes Monte Carlo cash-flow simulations dramatically faster (vectorized wealth
and cash-flow engines, a Brent-based withdrawal solver — three to four orders
of magnitude per simulation), adds money-weighted IRR (MWRR) for portfolio
cash flows — both on historical data and across Monte Carlo forecast paths —
makes the Monte Carlo return draw cached and reproducible, and fixes three
cash-flow calculation bugs.

### Added
- `Portfolio.dcf.irr()` (`PortfolioDCF.irr`) — nominal annualized money-weighted
  internal rate of return (IRR/MWRR) of the portfolio cash flow over the full
  historical period, honoring the configured `CashFlow` strategy
  (`IndexationStrategy`, `PercentageStrategy`, `VanguardDynamicSpending`,
  `CutWithdrawalsIfDrawdown`, `TimeSeriesStrategy`). With no intermediate cash
  flows it equals `Portfolio.get_cagr()` for the period.
- `Portfolio.dcf.monte_carlo_irr()` (`PortfolioDCF.monte_carlo_irr`) — the
  distribution (`pandas.Series`) of per-path money-weighted IRRs across Monte
  Carlo forecast paths, the forward-looking counterpart of `PortfolioDCF.irr`.
- `irr_of_cashflow_matrix()` in `okama.portfolios.dcf_calculations` — a
  vectorized Newton solver (analytic derivative, `scipy.optimize.brentq`
  fallback) computing IRR for an `(n_periods, n_series)` cash-flow matrix in one
  pass; shared by both `PortfolioDCF.irr` and `PortfolioDCF.monte_carlo_irr`.
- `seed` parameter for reproducible Monte Carlo draws: `MonteCarlo.seed` and the
  new `seed` argument of `PortfolioDCF.set_mc_parameters()`.

### Changed
- The Monte Carlo return draw (`MonteCarlo.monte_carlo_returns_ts`) is now
  generated once and cached — shared by `PortfolioDCF.monte_carlo_wealth`,
  `monte_carlo_cash_flow`, `monte_carlo_survival_period`, `monte_carlo_irr` and
  the CAGR-distribution methods — so all of them see one consistent scenario set
  (previously each access regenerated fresh randomness). The cache is invalidated
  when any Monte Carlo parameter changes (`distribution`,
  `distribution_parameters`, `period`, `mc_number`, `seed`). As a consequence,
  unseeded results of `PortfolioDCF.find_the_largest_withdrawals_size()` and of
  the CAGR-distribution methods shift versus 2.1.1: the bisection in
  `find_the_largest_withdrawals_size` now evaluates every candidate against the
  same scenario set (removing the sampling noise that previously broke its
  monotonicity). Use `set_mc_parameters(..., seed=...)` for reproducible runs.
- `Portfolio.dcf.find_the_largest_withdrawals_size()` is faster: the bisection
  search is replaced with Brent's method (`scipy.optimize.brentq`) on a signed
  goal residual, and both ends of `withdrawals_range` are checked first so the
  solver exits after 1–2 Monte Carlo simulations when the solution lies outside
  the range. The public signature, the `Result` shape and the stopping rule
  (`error_rel < tolerance_rel`) are unchanged; `iter_max` now caps objective
  evaluations (Monte Carlo simulations), including the two range-end checks, so
  the history of intermediate attempts in `Result.solutions` differs from the
  former bisection midpoints. `iter_max` values below 1 are now rejected with a
  `ValueError`.
- Monte Carlo wealth simulation is vectorized: `Portfolio.dcf.monte_carlo_wealth()`
  and everything built on it (`monte_carlo_survival_period()`,
  `plot_forecast_monte_carlo()`,
  `find_the_largest_withdrawals_size()`) now computes all
  random paths in one pass (`get_wealth_indexes_fv_with_cashflow_mc` in
  `okama.portfolios.dcf_calculations`) instead of a per-path pandas `apply`.
  Results are unchanged (pinned by an equivalence-test grid across strategies,
  frequencies and extra cash flows); measured speedup of one full simulation is
  three to four orders of magnitude (×1400 for yearly and ×6800 for monthly
  withdrawal frequencies on 1,000 paths × 30 years). The negative-balance
  masking and the survival-date scan are vectorized as well.
- Monte Carlo cash-flow simulation is vectorized as well:
  `Portfolio.dcf.monte_carlo_cash_flow()` builds its cache with the new
  `get_cash_flow_fv_mc` (one pass for all paths, sharing a core with the
  wealth engine), and `Portfolio.dcf.monte_carlo_irr()` now consumes the
  shared `monte_carlo_wealth`/`monte_carlo_cash_flow` caches instead of two
  per-path computations (measured end-to-end speedup of `monte_carlo_irr()`:
  ×488 on 1,000 paths × 30 years).

### Fixed
- `PortfolioDCF.monte_carlo_cash_flow()` with `remove_if_wealth_index_negative=True`
  previously masked a cash-flow draw against a wealth-index draw generated from
  *different* randomness; with the shared cached draw the depletion mask is now
  consistent per path.
- In periodic-frequency simulations, the first month of a period containing
  extra cash flows (`time_series`) skipped its return in
  `get_wealth_indexes_fv_with_cashflow` (and additionally skipped the cash
  flow in `get_cash_flow_fv` balance tracking). The recursion is now uniform
  for every month; wealth indexes and cash flow series with extra cash flows
  at `year`/`half-year`/`quarter` frequencies change accordingly (#81).
- `VanguardDynamicSpending` `floor_ceiling` limits never bound in wealth-index
  calculations (`wealth_index`, `monte_carlo_wealth` and everything built on
  them): the previous withdrawal was always reported as 0. Wealth indexes for
  VDS strategies with `floor_ceiling` change accordingly and are now
  consistent with `cash_flow_ts` (#82).
- `VanguardDynamicSpending.__init__` bypassed the validating setters for
  `floor_ceiling`, `min_max_annual_withdrawals` and `adjust_min_max`, so
  out-of-contract limits (e.g. a non-negative floor or `min > max`) were
  silently accepted at construction. The constructor now routes through the
  public setters, and both limit setters accept `None` (meaning "limit
  disabled") so the documented defaults remain valid (#83).

### Security
- Dependency floor `idna >= 3.15` to close CVE-2026-45409.

### Docs
- New "IRR — money-weighted return" section in the
  [04 investment portfolios with DCF](https://github.com/mbk-dev/okama/blob/master/examples/04%20investment%20portfolios%20with%20DCF.ipynb)
  notebook demonstrating `Portfolio.dcf.irr()` and
  `Portfolio.dcf.monte_carlo_irr()`.
- The `find_the_largest_withdrawals_size()` docstring example now shows a real,
  seeded solver run (range-end checks followed by Brent steps).

## [2.1.1] - 2026-05

Adds systematic (grid-based) enumeration of portfolio weights on the efficient
frontier as a deterministic alternative to Monte-Carlo sampling, and enriches
the frontier sampling outputs with per-asset weight columns.

### Added
- `EfficientFrontier.get_grid_portfolios()` (multi-period, rebalanced) and
  `EfficientFrontierSingle.get_grid_portfolios()` (single-period) enumerate all
  portfolios whose weights lie on a fixed percentage grid (`step`, default
  `0.10`), respecting per-asset `bounds`. This complements the random
  `get_monte_carlo()` with a reproducible, exhaustive sampling of the
  feasible region.
- `Float.get_grid_weights()` helper in `okama.common.helpers` — a reusable
  generator of all weight vectors summing to 1.0 on a given grid step, honoring
  per-asset bounds. The `step` is validated to lie in `[0.01, 1.0]` and to
  divide 1.0 evenly.
- Per-asset weight columns in the outputs of
  `EfficientFrontier.get_monte_carlo()` and
  `EfficientFrontier.get_grid_portfolios()` (multi-period), matching the
  column layout already produced by `EfficientFrontierSingle.get_monte_carlo()`.

### Tooling
- Pinned `sphinx < 9` for the docs build: the Sphinx 9.x autodoc rewrite raises
  `ValueError: The truth value of a DataFrame is ambiguous` on pandas
  DataFrame class attributes (e.g. in `okama.common.make_asset_list`).
- Bumped the pre-commit `ruff` hook to `v0.15.14` to match the poetry/CI ruff
  version, so it stops re-applying fixes the project ruff already suppresses
  via `# noqa: UP0xx` comments.

## [2.1.0] - 2026-05

Feature release that switches `get_cagr` and `get_cumulative_return` to an
expanding-window definition, makes the okama API endpoint configurable via
environment variables, and ships a batch of correctness fixes across the
helpers, frontier, DCF, macro, and plotting layers.

### Changed
- `AssetList.get_cumulative_return()` and `Portfolio.get_cumulative_return()`
  now return an **expanding** cumulative return series instead of a single
  end-of-period scalar. Notebook
  [03 investment portfolios.ipynb](examples/03%20investment%20portfolios.ipynb)
  updated accordingly.
- `AssetList.get_cagr()` and `Portfolio.get_cagr()` now compute CAGR on an
  expanding window, consistent with `get_cumulative_return()`.

### Added
- Configurable API base URL and request timeout via environment variables
  (`OKAMA_API_URL`, `OKAMA_API_TIMEOUT`) in `okama.settings` and
  `okama.api.api_methods`.

### Fixed
- `Frame.get_semideviation()` (in `okama.common.helpers`) now uses the sample
  mean of returns instead of the population mean, restoring the standard
  semideviation definition; propagated through `AssetList` and `Portfolio`
  consumers.
- `AssetList.recovery_periods` is robust to a `last_date` that is not on a
  month start.
- `EfficientFrontier` / `EfficientFrontierReb` (single- and multi-period
  variants) now raise `RuntimeError` on failed SLSQP optimisation instead of
  silently returning invalid weights.
- `AssetList.plot_assets()` / `Portfolio.plot_assets()` autoscale no longer
  passes the invalid `axis="year"` argument.
- `Inflation.cumulative_inflation` (in `okama.macro`) uses `.iloc[-1]` instead
  of positional `[-1]`, fixing a pandas FutureWarning / lookup bug.
- `PortfolioDCF` discount-rate attribute renamed from the misspelled
  `monlthly_discount_rate` to `monthly_discount_rate`
  (`okama.portfolios.dcf`, `okama.portfolios.dcf_calculations`).
- Helpers producing NaN rows now use `np.nan` in `dict.fromkeys(...)` so
  resulting DataFrames keep float dtype (`okama.common.helpers`,
  consumed by `AssetList` and `Portfolio`).

### Removed
- Dead `Portfolio._clear_cf_cache` method.

### Tooling
- Enabled ruff `UP` (pyupgrade) rules and applied auto-fixes across
  `okama.common.helpers`, `okama.common.helpers.rebalancing`,
  `okama.portfolios.dcf`, and notebook
  [11 rebalancing portfolio.ipynb](examples/11%20rebalancing%20portfolio.ipynb).
- Aligned declared supported Python versions with the `pyproject.toml`
  minimum; `AGENTS.md` now mandates a TDD workflow for production code
  changes.
- `.gitignore` excludes `.env`.

## [2.0.1] - 2026-04

Maintenance-focused release that improves compatibility with `pandas` 3.x and Python 3.14,
refines `PortfolioDCF` cash flow calculations, and makes several plotting and sharing APIs
easier to integrate.

### Added
- `pandas` 3.x and Python 3.14 support across `Asset`, `AssetList`, `Portfolio`,
  `PortfolioDCF`, `MacroABC`, `Rebalance`, and related helpers.
- `PortfolioDCF.plot_forecast_monte_carlo()`, `MonteCarlo.plot_qq()`, and
  `MonteCarlo.plot_hist_fit()` now return matplotlib `Axes` objects.
- `Portfolio.plot_assets()` and `EfficientFrontier.plot_assets()` accept extra
  `matplotlib.pyplot.scatter()` keyword arguments.
- `Portfolio.okamaio_link` serializes the full `Rebalance` configuration
  (`period`, `abs_deviation`, `rel_deviation`).
- `AssetList.get_monthly_geometric_mean_return()` and
  `Portfolio.get_monthly_geometric_mean_return()` — direct monthly geometric
  mean return calculations.
- `EfficientFrontier` caches intermediate calculations to speed up repeated
  optimization runs.

### Fixed
- `PortfolioDCF.wealth_index()` and `PortfolioDCF.cash_flow_ts()` no longer apply
  reverse discounting in present-value mode.
- `PortfolioDCF.wealth_index(discounting="pv")` now discounts inflation correctly.
- `PortfolioDCF.find_the_largest_withdrawals_size()` fixed for Monte Carlo period
  validation; works correctly with `VanguardDynamicSpending` and
  `CutWithdrawalsIfDrawdown`.
- `PortfolioDCF.plot_forecast_monte_carlo()` no longer mutates
  `PortfolioDCF.cashflow_parameters`.
- `MonteCarlo` distribution fitting and plotting: float dtype handling and
  QQ/histogram compatibility.
- `EF._get_gmv_monthly` returns geometric mean (not arithmetic).
- First `Portfolio.ror` value was lost when rebalancing.
- `pandas` compatibility for `DataFrame.applymap()`, `pd.concat(..., copy=...)`,
  and updated `pd.Grouper` frequency aliases.

### Docs
- Updated notebooks: [04 investment portfolios with DCF.ipynb](examples/04%20investment%20portfolios%20with%20DCF.ipynb),
  [10 forecasting.ipynb](examples/10%20forecasting.ipynb),
  [11 rebalancing portfolio.ipynb](examples/11%20rebalancing%20portfolio.ipynb).
- Read the Docs navigation and API pages refreshed for `okama.search()` and
  `okama.symbols_in_namespace()`.
- README: added GitHub and pepy.tech download badges and refreshed the project
  roadmap.

### Tooling

- Migrated linting and formatting from `flake8` + `black` to `ruff check`,
  including GitHub workflows and contributor instructions in `AGENTS.md`.
- Added `.pre-commit-config.yaml` with ruff hooks.
- Added top-level `requirements.txt` mirroring runtime dependencies from
  `pyproject.toml`.
- Added this `CHANGELOG.md` following Keep a Changelog and Semantic Versioning.

## [2.0.0] - 2025-11-27

Major release focused on a rewritten efficient frontier, extended Monte Carlo
engine, and richer DCF cash flow strategies.

### Added
- New `EfficientFrontier` based on `EfficientFrontierReb`, using fast vectorized
  `Rebalance.wealth_ts_ef` / `return_ror_ts_ef` methods.
- `EfficientFrontierReb.get_tangency_portfolio()` and `plot_cml()`.
- Most Diversified Portfolio (MDP) in `EfficientFrontierReb`.
- `MonteCarlo` class: Monte Carlo methods moved from `Portfolio` into a dedicated
  class; parameters for `norm`, `lognorm`, and Student's *t* distributions; all
  distribution calculations consolidated in `tails.py`.
- New cash flow strategy `CutWithdrawalsIfDrawdown` (CWD, formerly CWID) and
  `VanguardDynamicSpending` (VDS) extensions (`adjust_min_max`,
  `adjust_floor_ceiling`, negative-percentage guard).
- `ListMaker.period_length` precision improvements; consistent use of
  `Asset.first_date` / `Asset.last_date`.
- HTTPS for the okama API with SSL verification.
- New `Portfolio.okamaio_link` URL format with full rebalancing-strategy
  parameters.
- Public `okama.search()`, `okama.symbols_in_namespace()`, and
  `okama.namespaces` with docstrings and Read the Docs pages.

### Changed
- `PortfolioDCF.find_the_largest_withdrawals_size()` refactored; no longer
  mutates `CashFlow` parameters.
- Rebalancing sped up by moving `pd.concat` out of the inner loop.
- Mean return for `Portfolio` uses `ror.mean()`.
- CWID renamed to CWD across cash flow strategies.

### Fixed
- DCF discount rate for short histories (< 12 months).
- `Portfolio.dcf.wealth_index(discounting="pv")` correctness.
- `mc._get_params_for_lognormal`, `_target_cagr_range_left`, and several VDS
  floor/ceiling and min/max edge cases.
- Pandas `FutureWarning` on `concat` with empty DataFrame during rebalancing.
- Annual arithmetic mean return compounding.

## [1.5.0] - 2025-06-24

- Feature and maintenance release preceding the 2.0 rewrite of the efficient
  frontier and Monte Carlo subsystems.

## [1.4.4] - 2024-10-10

## [1.4.3] - 2024-10-08

## [1.4.1] - 2024-07-05

## [1.4.0] - 2024-02-23

## [1.3.2] - 2023-12-11

## [1.3.1] - 2023-05-22

## [1.3.0] - 2023-05-02

## [1.2.4] - 2022-10-12

## [1.2.3] - 2022-08-10

## [1.2.2] - 2022-08-09

## [1.2.1] - 2022-07-13

## [1.2.0] - 2022-05-24

## [1.1.6] - 2022-04-15

## [1.1.5] - 2022-04-06

## [1.1.4] - 2022-03-30

## [1.1.3] - 2022-02-14

## [1.1.2] - 2022-01-28

## [1.1.1] - 2021-11-27

## [1.1.0] - 2021-10-11

## [1.0.3] - 2021-09-28

## [1.0.2] - 2021-09-26

## [1.0.1] - 2021-09-01

## [1.0.0] - 2021-08-16

First stable release.

## [0.99] - 2021-04-29

## [0.98] - 2021-03-19

## [0.97] - 2021-01-27

## [0.96] - 2021-01-26

## [0.95] - 2021-01-26

## [0.94] - 2021-01-20

## [0.93] - 2021-01-20

## [0.92] - 2021-01-04

## [0.91] - 2020-12-23
