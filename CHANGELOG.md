# Changelog

All notable changes to **okama** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
