# Changelog

All notable changes to **okama** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Adds money-weighted IRR (MWRR) for portfolio cash flows — both on historical
data and across Monte Carlo forecast paths — and makes the Monte Carlo return
draw cached and reproducible.

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

### Fixed
- `PortfolioDCF.monte_carlo_cash_flow()` with `remove_if_wealth_index_negative=True`
  previously masked a cash-flow draw against a wealth-index draw generated from
  *different* randomness; with the shared cached draw the depletion mask is now
  consistent per path.

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
