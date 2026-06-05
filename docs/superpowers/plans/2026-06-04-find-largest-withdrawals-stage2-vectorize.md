# Stage 2 — Vectorized Monte Carlo wealth engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-path pandas `apply` Monte Carlo wealth computation with a vectorized numpy engine (single Python loop over months, vector ops across all paths), speeding up every `monte_carlo_*` method — and therefore each solver evaluation — by 1–2 orders of magnitude, with behavior pinned to the old engine by an equivalence-test grid.

**Architecture:** A new function `get_wealth_indexes_fv_with_cashflow_mc(ror, cashflow_parameters, discount_rate)` in `okama/portfolios/dcf_calculations.py` computes the whole T×N wealth matrix at once, replicating `get_wealth_indexes_fv_with_cashflow` (monte_carlo task) **exactly, including two known quirks** (see "Replicate, do not fix" below). `PortfolioDCF.monte_carlo_wealth` routes through it; the negative-value masking and the survival-date scan are vectorized too. The old per-column function remains for the backtest path.

**Tech Stack:** numpy, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-04-find-largest-withdrawals-speedup-design.md` (Stage 2 section).

**Branch:** work happens on `dev`.

---

## Context for the engineer (read first)

- Reference implementation (must be replicated bit-faithfully for the `"monte_carlo"` task): `get_wealth_indexes_fv_with_cashflow` in `okama/portfolios/dcf_calculations.py:14-162`. Caller: `PortfolioDCF.monte_carlo_wealth` (`okama/portfolios/dcf.py`, `if self._monte_carlo_wealth_fv.empty:` block) applies it per column with `args=(None, None, self.cashflow_parameters, "monte_carlo")`.
- In MC mode `portfolio_symbol=None`, `inflation_symbol=None`, so the per-column function returns a plain wealth Series (no inflation concat). The output frame: same index as `monte_carlo_returns_ts` **plus one prepended row** at `index[0] - 1` holding `initial_investment`, same columns.
- Strategy dispatch is by the class attribute `NAME`: `"fixed_amount"` (IndexationStrategy), `"fixed_percentage"` (PercentageStrategy), `"time_series"` (TimeSeriesStrategy), `"VDS"` (VanguardDynamicSpending, frequency locked to `"year"`), `"CWD"` (CutWithdrawalsIfDrawdown). Frequencies: `"none"`, `"month"` → "fast branch" (plain monthly loop); `"year"`, `"half-year"`, `"quarter"` → "slow branch" (`resample(rule, convention="start")` periods, regular cash flow applied at period END, scaled by `period_fraction = months_local / (12 / periods_per_year)` for partial periods).
- Extra cash flows: every strategy has `time_series` (a monthly Series built from `time_series_dic`) and `time_series_discounted_values`. For the `monte_carlo` task, if `time_series_discounted_values` is False the values are multiplied by `(1 + monthly_discount_rate) ** arange(T)` where `monthly_discount_rate = (1 + discount_rate) ** (1/12) - 1`.
- CWD reduction: `_crash_threshold_reduction_series` is sorted by threshold **descending** (`make_series_from_list`, `cashflow_strategies.py:998`); the scalar loop picks the FIRST (deepest) threshold with `|drawdown| >= threshold`. Drawdowns come from `helpers.Frame.get_drawdowns` (`helpers.py:429`): `wealth = 1000*(1+ror).cumprod(); dd = (wealth - wealth.cummax())/wealth.cummax()`.
- VDS: `_calculate_withdrawal_size` (`cashflow_strategies.py:714`) with `last_withdrawal=0` reduces to `-clip(balance*|percentage|, min_indexed, max_indexed)` when `min_max_annual_withdrawals` is set (indexed by `(1+indexation)**n` when `adjust_min_max`), else `-balance*|percentage|`; floor/ceiling are inert at `last_withdrawal=0`.
- Tests run via `poetry run pytest` (never bare pytest). Lint: `poetry run ruff check .`. Comments/docstrings in English.
- The working tree may contain UNRELATED dirty files (`examples/07 efficient frontier multi-period.ipynb`, `OPTIMIZATION_RECOMMENDATIONS.md`, `get_grid_portfolios_recommendation.md`). Never stage them; stage only the files named in each task.
- The global `synthetic_env` fixture (`tests/conftest.py:92`) provides mocked assets `A.US`, `B.US`, `IDX.US` over 2020-01..2021-12. `ok.Portfolio(["A.US"], ccy="USD", inflation=False, last_date="2021-06", rebalancing_strategy=ok.Rebalance(period="month"))` truncates history so the MC index starts 2021-07 → partial first/last `resample` periods for the `"year"` rule.

## Replicate, do not fix (two known quirks of the reference engine)

The vectorized engine MUST reproduce these behaviors; both are pre-existing and are reported separately as candidate bugfixes. Do not "improve" them here — the equivalence grid is the ground truth for this stage:

1. **First month of a period containing extra cash flows gets NO return.** In the slow branch, when a resample period has any nonzero extra cash flow, the per-month loop does `k==0: month_balance = period_initial_amount + cashflow_ts_local[date]` — the month's return `r` is skipped (`dcf_calculations.py:111-112`), unlike the no-extra-cash-flow branch (`cumprod` applies returns to all months).
2. **VDS sees `last_withdrawal=0` for every period.** `get_wealth_indexes_fv_with_cashflow` initializes `last_regular_cash_flow = 0` and never updates it (`dcf_calculations.py:30,125`), unlike `get_cash_flow_fv` which does (`dcf_calculations.py:290`). Floor/ceiling limits therefore never bind in wealth-index calculations.

Mark both with code comments in the new engine so a future bugfix finds the sites.

## File structure

- Create: `tests/portfolio/dcf/test_mc_engine_equivalence.py` — old-vs-new equivalence grid.
- Create: `tests/portfolio/dcf/test_mc_engine_benchmark.py` — env-gated manual benchmark.
- Modify: `okama/portfolios/dcf_calculations.py` — new engine + two private vector helpers + `zero_wealth_after_first_void`.
- Modify: `okama/portfolios/dcf.py` — route `monte_carlo_wealth` through the engine.
- Modify: `okama/common/helpers/helpers.py` — vectorize the DataFrame overload of `Frame.get_survival_date`.
- Modify: `CHANGELOG.md`.

---

### Task 1: Equivalence-test grid (RED)

**Files:**
- Create: `tests/portfolio/dcf/test_mc_engine_equivalence.py`

- [ ] **Step 1: Create the test file**

```python
"""Equivalence of the vectorized MC wealth engine with the per-path reference.

`get_wealth_indexes_fv_with_cashflow_mc` must reproduce
`get_wealth_indexes_fv_with_cashflow` (task="monte_carlo") applied per column —
including the period-fraction scaling for partial periods and the known quirks
of the reference implementation (no return in the first month of a period with
extra cash flows; VDS last_withdrawal pinned to 0).
"""

import pandas as pd  # noqa: I001
import pytest
import okama as ok
from okama.portfolios import dcf_calculations


def _make_portfolio(last_date=None):
    kwargs = {"ccy": "USD", "inflation": False, "rebalancing_strategy": ok.Rebalance(period="month")}
    if last_date is not None:
        kwargs["last_date"] = last_date
    return ok.Portfolio(["A.US"], **kwargs)


def _indexation(pf, frequency):
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = frequency
    ind.amount = -300 if frequency == "month" else -1_200
    ind.indexation = 0.05
    return ind


def _percentage(pf, frequency):
    pc = ok.PercentageStrategy(pf)
    pc.initial_investment = 10_000
    pc.frequency = frequency
    pc.percentage = -0.08
    return pc


def _time_series_only(pf, extra_dic):
    ts = ok.TimeSeriesStrategy(pf)
    ts.initial_investment = 10_000
    ts.time_series_dic = extra_dic
    return ts


def _cwd(pf, frequency):
    cwd = ok.CutWithdrawalsIfDrawdown(
        pf,
        initial_investment=10_000,
        amount=-1_000,
        indexation=0.02,
        crash_threshold_reduction=[(0.05, 0.3), (0.15, 1.0)],
    )
    cwd.frequency = frequency
    return cwd


def _vds(pf):
    return ok.VanguardDynamicSpending(
        pf,
        initial_investment=10_000,
        percentage=-0.08,
        indexation=0.0,
        min_max_annual_withdrawals=(500.0, 900.0),
        adjust_min_max=True,
    )


# Each case: (case_id, strategy_builder(pf) -> CashFlow, last_date, mc_period_years, extra_dic)
# extra_dic dates must fall inside the Monte Carlo window (it starts the month
# after the portfolio last_date: 2022-01 for full history, 2021-07 for last_date="2021-06").
CASES = [
    ("indexation_month", lambda pf: _indexation(pf, "month"), None, 3, None),
    ("indexation_year", lambda pf: _indexation(pf, "year"), None, 3, None),
    ("indexation_quarter", lambda pf: _indexation(pf, "quarter"), None, 3, None),
    ("indexation_year_partial_periods", lambda pf: _indexation(pf, "year"), "2021-06", 3, None),
    ("indexation_year_extra_cf", lambda pf: _indexation(pf, "year"), None, 3, {"2022-03": -500, "2023-11": 1_000}),
    ("percentage_month", lambda pf: _percentage(pf, "month"), None, 3, None),
    ("percentage_year", lambda pf: _percentage(pf, "year"), None, 3, None),
    ("percentage_halfyear_partial", lambda pf: _percentage(pf, "half-year"), "2021-06", 3, None),
    ("time_series_only", lambda pf: _time_series_only(pf, {"2022-03": -500, "2022-11": 1_000}), None, 3, None),
    ("cwd_year", lambda pf: _cwd(pf, "year"), None, 3, None),
    ("cwd_month", lambda pf: _cwd(pf, "month"), None, 3, None),
    ("vds_year", lambda pf: _vds(pf), None, 3, None),
    ("vds_year_partial", lambda pf: _vds(pf), "2021-06", 3, None),
]


def _reference_wealth(dcf, params):
    """Old per-column engine output (the ground truth)."""
    return_ts = dcf.mc.monte_carlo_returns_ts
    return return_ts.apply(
        dcf_calculations.get_wealth_indexes_fv_with_cashflow,
        axis=0,
        args=(None, None, params, "monte_carlo"),
    )


@pytest.mark.parametrize(("case_id", "builder", "last_date", "period", "extra_dic"), CASES, ids=[c[0] for c in CASES])
def test_vectorized_engine_matches_per_path_reference(
    synthetic_env, case_id, builder, last_date, period, extra_dic
) -> None:
    pf = _make_portfolio(last_date)
    params = builder(pf)
    if extra_dic is not None:
        params.time_series_dic = extra_dic
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=period, mc_number=8, seed=0)

    reference = _reference_wealth(pf.dcf, params)
    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(
        pf.dcf.mc.monte_carlo_returns_ts, params, pf.dcf.discount_rate
    )

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8)


def test_vectorized_engine_discounted_extra_cashflows(synthetic_env) -> None:
    # time_series_discounted_values=True skips the FV compounding of extra cash flows.
    pf = _make_portfolio()
    params = _percentage(pf, "quarter")
    params.time_series_dic = {"2022-05": -700}
    params.time_series_discounted_values = True
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=8, seed=0)

    reference = _reference_wealth(pf.dcf, params)
    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(
        pf.dcf.mc.monte_carlo_returns_ts, params, pf.dcf.discount_rate
    )

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8)
```

- [ ] **Step 2: Run and verify RED for the right reason**

Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_equivalence.py -v`
Expected: every test FAILS with `AttributeError: module 'okama.portfolios.dcf_calculations' has no attribute 'get_wealth_indexes_fv_with_cashflow_mc'` (missing function — a valid RED reason per the repo TDD rules). If a case fails earlier (e.g. a strategy builder raises), fix the TEST, not the library, and re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/portfolio/dcf/test_mc_engine_equivalence.py
git commit -m "test(dcf): add equivalence grid for the vectorized MC wealth engine

13 strategy/frequency/extra-cashflow cases plus a discounted-values case pin
the per-path reference (get_wealth_indexes_fv_with_cashflow, monte_carlo task)
as ground truth for the upcoming vectorized engine. RED: the engine does not
exist yet.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Implement the vectorized engine (GREEN)

**Files:**
- Modify: `okama/portfolios/dcf_calculations.py`

- [ ] **Step 1: Add the two private vector helpers**

Add after `get_cash_flow_fv` (before `remove_negative_values`) in `okama/portfolios/dcf_calculations.py`:

```python
def _cwd_reduction_factor_matrix(cashflow_parameters: cf.CashFlow, drawdowns: np.ndarray) -> np.ndarray:
    """Per-path CutWithdrawalsIfDrawdown reduction factors (1 - reduction), T x N.

    `_crash_threshold_reduction_series` is sorted by threshold descending and the
    scalar loop picks the first (deepest) threshold with |drawdown| >= threshold.
    Iterating thresholds ascending and overwriting reproduces that choice.
    """
    factors = np.ones_like(drawdowns)
    series = cashflow_parameters._crash_threshold_reduction_series.sort_index(ascending=True)
    for threshold, reduction in series.items():
        factors = np.where(np.abs(drawdowns) >= threshold, 1 - reduction, factors)
    return factors


def _vds_withdrawal_vector(
    cashflow_parameters: cf.CashFlow, balance: np.ndarray, number_of_periods: int
) -> np.ndarray:
    """Vectorized VanguardDynamicSpending withdrawal for one period across paths.

    Mirrors `VanguardDynamicSpending._calculate_withdrawal_size` with
    last_withdrawal == 0, which is what `get_wealth_indexes_fv_with_cashflow`
    effectively passes for every period (it never updates the previous regular
    cash flow — a known divergence from `get_cash_flow_fv`). At
    last_withdrawal == 0 the floor/ceiling limits are inert and only the
    min/max annual bounds apply.
    """
    withdrawal_by_percentage = balance * abs(cashflow_parameters.percentage)
    if cashflow_parameters.min_max_annual_withdrawals is not None:
        min_withdrawal, max_withdrawal = cashflow_parameters.min_max_annual_withdrawals
        if cashflow_parameters.adjust_min_max:
            indexation_factor = (1 + cashflow_parameters.indexation) ** number_of_periods
            min_withdrawal = abs(min_withdrawal) * indexation_factor
            max_withdrawal = abs(max_withdrawal) * indexation_factor
        else:
            min_withdrawal, max_withdrawal = abs(min_withdrawal), abs(max_withdrawal)
        return -np.clip(withdrawal_by_percentage, min_withdrawal, max_withdrawal)
    return -withdrawal_by_percentage


def _resample_slices(index: pd.PeriodIndex, pandas_frequency: str) -> list[tuple[int, int]]:
    """Start/stop integer positions of the groups of `resample(rule, convention="start")`.

    One cheap resample on a marker Series replaces N identical per-path resamples
    and guarantees the exact same grouping as the per-path reference.
    """
    marker = pd.Series(np.zeros(len(index)), index=index)
    slices = []
    position = 0
    for _, group in marker.resample(pandas_frequency, convention="start"):
        if group.shape[0] == 0:
            continue
        slices.append((position, position + group.shape[0]))
        position += group.shape[0]
    return slices
```

Note: if `cashflow_parameters.min_max_annual_withdrawals` is not a public attribute (check the class — it may be `_min_max_annual_withdrawals` with a property), use whatever `_calculate_withdrawal_size` itself reads (`self.min_max_annual_withdrawals` / `self.adjust_min_max` / `self.floor_ceiling`). Mirror the scalar code's attribute access exactly.

- [ ] **Step 2: Add the engine function**

Add right after the helpers:

```python
def get_wealth_indexes_fv_with_cashflow_mc(
    ror: pd.DataFrame,
    cashflow_parameters: cf.CashFlow,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Vectorized Monte Carlo counterpart of `get_wealth_indexes_fv_with_cashflow`.

    Computes wealth index future values (FV) for all random return paths at
    once: a single Python loop over time steps with numpy operations across
    paths, instead of a per-path pandas apply. Implements the "monte_carlo"
    task semantics only: extra cash flows from `time_series` are compounded
    with the discount rate unless `time_series_discounted_values` is True.

    Replicates the per-path reference exactly (equivalence is pinned by
    tests), including two known quirks kept intentionally:

    - the first month of a resample period that contains extra cash flows
      receives the cash flow but not the month's return;
    - VDS withdrawals are computed with last_withdrawal == 0 for every period
      (the reference never updates it, unlike `get_cash_flow_fv`).
    """
    initial_investment = cashflow_parameters.initial_investment
    n_rows, n_cols = ror.shape
    returns = ror.to_numpy(dtype=float)
    frequency = cashflow_parameters.frequency
    periods_per_year = settings.frequency_periods_per_year[frequency]
    amount = getattr(cashflow_parameters, "amount", None)
    indexation_per_period = 0.0
    if hasattr(cashflow_parameters, "indexation") and frequency != "none":
        indexation_per_period = (1 + cashflow_parameters.indexation) ** (1 / periods_per_year) - 1

    # Extra cash flows from `time_series`, aligned to the simulation index.
    extra_cf = np.zeros(n_rows)
    cash_flow_ts = cashflow_parameters.time_series
    if not (cash_flow_ts.empty or (cash_flow_ts == 0).all()):
        aligned = cash_flow_ts.reindex(ror.index).fillna(0).to_numpy(dtype=float)
        if not cashflow_parameters.time_series_discounted_values:
            monthly_discount_rate = (1 + discount_rate) ** (1 / settings._MONTHS_PER_YEAR) - 1
            aligned = aligned * (1.0 + monthly_discount_rate) ** np.arange(n_rows)
        extra_cf = aligned

    # Per-path drawdowns and reduction factors for the CWD strategy
    # (they depend on the return paths only, so they are precomputed once).
    if cashflow_parameters.NAME == "CWD":
        wealth_for_dd = 1000 * np.cumprod(1 + returns, axis=0)
        peaks = np.maximum.accumulate(wealth_for_dd, axis=0)
        drawdowns = (wealth_for_dd - peaks) / peaks
        cwd_factors = _cwd_reduction_factor_matrix(cashflow_parameters, drawdowns)

    wealth = np.empty((n_rows, n_cols))
    balance = np.full(n_cols, float(initial_investment))

    if frequency in ("month", "none"):
        for n in range(n_rows):
            if frequency == "none":
                cashflow: float | np.ndarray = 0.0
            elif cashflow_parameters.NAME == "fixed_amount":
                cashflow = amount * (1 + indexation_per_period) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow = cashflow_parameters.percentage / periods_per_year * balance
            elif cashflow_parameters.NAME == "time_series":
                cashflow = 0.0
            elif cashflow_parameters.NAME == "CWD":
                base_withdrawal = amount * (1 + indexation_per_period) ** n
                cashflow = base_withdrawal * np.where(drawdowns[n] < 0, cwd_factors[n], 1.0)
            else:
                raise ValueError("Wrong cashflow strategy name value.")
            balance = balance * (1 + returns[n]) + cashflow + extra_cf[n]
            wealth[n] = balance
    else:
        months_in_full_period = settings._MONTHS_PER_YEAR / cashflow_parameters.periods_per_year
        for n, (start, stop) in enumerate(_resample_slices(ror.index, cashflow_parameters._pandas_frequency)):
            start_balance = balance
            if np.any(extra_cf[start:stop] != 0):
                # Reference quirk kept intentionally: the first month of a
                # period with extra cash flows gets the cash flow but NOT the
                # month's return.
                for k in range(start, stop):
                    if k == start:
                        balance = balance + extra_cf[k]
                    else:
                        balance = balance * (1 + returns[k]) + extra_cf[k]
                    wealth[k] = balance
            else:
                segment = np.cumprod(1 + returns[start:stop], axis=0) * start_balance
                wealth[start:stop] = segment
                balance = segment[-1]
            # Regular cash flow at the period end.
            if cashflow_parameters.NAME == "fixed_amount":
                cashflow_value: float | np.ndarray = amount * (1 + indexation_per_period) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow_value = cashflow_parameters.percentage / periods_per_year * start_balance
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = _vds_withdrawal_vector(cashflow_parameters, start_balance, n)
            elif cashflow_parameters.NAME == "CWD":
                base_withdrawal = amount * (1 + indexation_per_period) ** n
                cashflow_value = base_withdrawal * np.where(drawdowns[stop - 1] < 0, cwd_factors[stop - 1], 1.0)
            else:
                raise ValueError("Wrong cashflow_method value.")
            cashflow_value = cashflow_value * ((stop - start) / months_in_full_period)
            balance = balance + cashflow_value
            wealth[stop - 1] = balance

    out_index = ror.index.insert(0, ror.index[0] - 1)
    data = np.vstack([np.full((1, n_cols), float(initial_investment)), wealth])
    return pd.DataFrame(data, index=out_index, columns=ror.columns)
```

- [ ] **Step 3: Run the equivalence grid**

Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_equivalence.py -v`
Expected: 14 passed. Debug guidance if a case fails:
- `*_partial_periods` cases → check `_resample_slices` grouping and the `period_fraction` scaling.
- `*_extra_cf` cases → check the first-month-no-return quirk and the discount factors.
- `cwd_*` → check drawdown definition (`1000 * cumprod`) and threshold iteration order.
- `vds_*` → check the min/max indexation exponent (`number_of_periods` = period counter `n`).

- [ ] **Step 4: Run the dcf package and lint**

Run: `poetry run pytest tests/portfolio/dcf/ -q` — all pass.
Run: `poetry run ruff check .` — clean. If the engine trips C901, add `# noqa: C901` on its `def` line with rationale (strategy dispatch table; mirrors the reference structure), like the reference function does.

- [ ] **Step 5: Commit**

```bash
git add okama/portfolios/dcf_calculations.py
git commit -m "perf(dcf): add vectorized Monte Carlo wealth engine

get_wealth_indexes_fv_with_cashflow_mc computes all random paths at once
(one Python loop over months, numpy across paths) instead of a per-path
pandas apply, replicating get_wealth_indexes_fv_with_cashflow
(task=monte_carlo) exactly - including the first-month-no-return quirk for
periods with extra cash flows and VDS last_withdrawal==0, both kept
intentionally and marked with comments. Equivalence is pinned by a 14-case
strategy/frequency/extra-cashflow test grid.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Route `monte_carlo_wealth` through the engine + vectorize negative-value masking

**Files:**
- Modify: `okama/portfolios/dcf_calculations.py` (new `zero_wealth_after_first_void`)
- Modify: `okama/portfolios/dcf.py` (`monte_carlo_wealth` body)
- Modify: `tests/portfolio/dcf/test_mc_engine_equivalence.py` (two guard tests)

- [ ] **Step 1: Add guard tests pinning the public API (they must pass BEFORE and AFTER the routing)**

Append to `tests/portfolio/dcf/test_mc_engine_equivalence.py`:

```python
def test_monte_carlo_wealth_matches_per_path_reference(synthetic_env) -> None:
    # Guard for the routing: the public API output must stay equal to the
    # old per-path computation (passes before and after the engine switch).
    pf = _make_portfolio()
    params = _indexation(pf, "year")
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=8, seed=0)

    reference = _reference_wealth(pf.dcf, params)
    pf.dcf.cashflow_parameters = params  # restore after the reference run side effects
    pf.dcf.cashflow_parameters._clear_cf_cache()
    result = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=True)

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8)


def test_monte_carlo_wealth_zeroes_paths_after_first_void(synthetic_env) -> None:
    # Guard for the vectorized negative-value masking: once a path hits a
    # non-positive value, that month and everything after must read 0.
    pf = _make_portfolio()
    params = _indexation(pf, "year")
    params.amount = -3_000  # large withdrawal: paths void within the horizon
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=8, seed=0)

    fv = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=True)
    masked = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=False)

    expected = fv.apply(dcf_calculations.remove_negative_values, axis=0).fillna(0)
    pd.testing.assert_frame_equal(masked, expected, check_exact=False, rtol=1e-12, atol=1e-8)
    assert (masked.to_numpy() >= 0).all()
```

Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_equivalence.py -v` — 16 passed (guards pass against the CURRENT implementation; they are regression pins, not RED drivers — the routing must not change behavior).

- [ ] **Step 2: Add `zero_wealth_after_first_void` to dcf_calculations.py**

Add right after `remove_negative_values` (keep the old per-Series function — it has other callers; verify with `grep -rn "remove_negative_values" okama/ --include="*.py"`):

```python
def zero_wealth_after_first_void(wealth: pd.DataFrame) -> pd.DataFrame:
    """Vectorized frame counterpart of per-column `remove_negative_values` + `fillna(0)`.

    For every column, the first non-positive value and everything after it are
    replaced with 0. Returns a new DataFrame; the input is not modified.
    """
    values = wealth.to_numpy(dtype=float)
    voided = values <= 0
    has_void = voided.any(axis=0)
    first_void = voided.argmax(axis=0)
    rows = np.arange(values.shape[0])[:, None]
    masked = np.where(has_void & (rows >= first_void), 0.0, values)
    return pd.DataFrame(masked, index=wealth.index, columns=wealth.columns)
```

- [ ] **Step 3: Rewrite the `monte_carlo_wealth` body**

In `okama/portfolios/dcf.py`, replace the body of `monte_carlo_wealth` after the `cashflow_parameters is None` check (keep signature and docstring) with:

```python
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._monte_carlo_wealth_fv.empty:
            return_ts = self.mc.monte_carlo_returns_ts
            self._monte_carlo_wealth_fv = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(
                return_ts, self.cashflow_parameters, self.discount_rate
            )
        if not include_negative_values:
            monte_carlo_wealth_fv = dcf_calculations.zero_wealth_after_first_void(self._monte_carlo_wealth_fv)
        else:
            monte_carlo_wealth_fv = self._monte_carlo_wealth_fv.copy()
        if discounting.lower() == "fv":
            return monte_carlo_wealth_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(monte_carlo_wealth_fv, self.discount_rate)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")
```

Notes:
- This drops the old commented-out lines and the double `.copy()` on the masked path (`zero_wealth_after_first_void` already returns a new frame). The `.copy()` on the `include_negative_values=True` path stays — it protects the cache from caller mutation, same as before.
- The old per-column apply also re-assigned `dcf.cashflow_parameters` inside each column call (a side effect of the reference function); the engine does not — the parameters are already set by the caller.

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/portfolio/dcf/ -q` — all pass (the 16 equivalence/guard tests and every pre-existing `monte_carlo_*`/solver test).
Run: `poetry run pytest -q` — full suite green.

- [ ] **Step 5: Commit**

```bash
git add okama/portfolios/dcf.py okama/portfolios/dcf_calculations.py tests/portfolio/dcf/test_mc_engine_equivalence.py
git commit -m "perf(dcf): route monte_carlo_wealth through the vectorized engine

monte_carlo_wealth now builds the wealth-index cache with
get_wealth_indexes_fv_with_cashflow_mc (one call for all paths) and masks
voided paths with the vectorized zero_wealth_after_first_void instead of a
per-column apply + fillna. Public behavior is unchanged - pinned by two new
guard tests against the per-path reference and by the existing suite.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Vectorize `Frame.get_survival_date` for DataFrames

**Files:**
- Modify: `okama/common/helpers/helpers.py`
- Modify: `tests/portfolio/dcf/test_mc_engine_equivalence.py` (one guard test)

- [ ] **Step 1: Add the guard test (passes before and after)**

Append to `tests/portfolio/dcf/test_mc_engine_equivalence.py`:

```python
def test_survival_dates_frame_matches_per_column_scan(synthetic_env) -> None:
    # Guard: the DataFrame overload of get_survival_date must equal the
    # per-column Series overload for both threshold modes.
    from okama.common.helpers import helpers

    pf = _make_portfolio()
    params = _indexation(pf, "year")
    params.amount = -3_000
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=8, seed=0)
    wealth = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=False)

    for threshold in (0, 0.1):
        frame_result = helpers.Frame.get_survival_date(wealth, pf.dcf.discount_rate, threshold)
        per_column = pd.Series(
            {col: helpers.Frame.get_survival_date(wealth[col], pf.dcf.discount_rate, threshold) for col in wealth},
        )
        pd.testing.assert_series_equal(frame_result, per_column, check_names=False, check_dtype=False)
```

Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_equivalence.py::test_survival_dates_frame_matches_per_column_scan -v` — 1 passed (guard against the current apply-based overload).

- [ ] **Step 2: Replace the DataFrame overload**

In `okama/common/helpers/helpers.py`, replace the DataFrame register of `get_survival_date` (currently `return wealth.apply(func=Frame.get_survival_date, axis=0, args=(discount_rate, threshold))`, around line 362-364) with:

```python
    @get_survival_date.register
    def _(wealth: pd.DataFrame, discount_rate: float, threshold: float = 0) -> pd.Series:
        if threshold > 1 or threshold < 0:
            raise ValueError("threshold must be in range from 0 to 1.")
        values = wealth.to_numpy(dtype=float)
        n_rows = values.shape[0]
        if threshold:
            factors = (1.0 + discount_rate / 12) ** np.arange(n_rows)
            voided = values <= values[0] * factors[:, None] * threshold
        else:
            voided = values <= 0
        has_void = voided.any(axis=0)
        first_void = voided.argmax(axis=0)
        positions = np.where(has_void, first_void, n_rows - 1)
        dates = wealth.index[positions].to_timestamp(freq="M")
        return pd.Series(dates, index=wealth.columns)
```

(The return annotation also changes from the incorrect `pd.Timestamp` to `pd.Series` — the old overload already returned a Series in practice.)

- [ ] **Step 3: Run tests**

Run: `poetry run pytest tests/portfolio/dcf/ tests/helpers/ -q` — all pass (the guard test, `monte_carlo_survival_period` callers, and any direct helpers tests).
Run: `poetry run pytest -q` — full suite green.

- [ ] **Step 4: Commit**

```bash
git add okama/common/helpers/helpers.py tests/portfolio/dcf/test_mc_engine_equivalence.py
git commit -m "perf(helpers): vectorize Frame.get_survival_date for DataFrames

Scan the whole wealth matrix with numpy (any/argmax per column) instead of
a per-column apply; fixes the return annotation to pd.Series. Behavior is
pinned by a guard test against the per-column Series overload.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Manual benchmark (env-gated) + record numbers

**Files:**
- Create: `tests/portfolio/dcf/test_mc_engine_benchmark.py`

- [ ] **Step 1: Create the benchmark module**

```python
"""Manual benchmark: vectorized MC engine vs the per-path reference.

Not part of the regular suite. Run with:

    OKAMA_BENCH=1 poetry run pytest tests/portfolio/dcf/test_mc_engine_benchmark.py -s
"""

import os  # noqa: I001
import time

import pytest
import okama as ok
from okama.portfolios import dcf_calculations

pytestmark = pytest.mark.skipif(not os.environ.get("OKAMA_BENCH"), reason="manual benchmark; set OKAMA_BENCH=1")


@pytest.mark.parametrize("frequency", ["year", "month"])
def test_benchmark_engine_vs_reference(synthetic_env, frequency) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = frequency
    ind.amount = -100 if frequency == "month" else -1_200
    ind.indexation = 0.05
    pf.dcf.cashflow_parameters = ind
    pf.dcf.set_mc_parameters(distribution="norm", period=30, mc_number=1_000, seed=0)
    return_ts = pf.dcf.mc.monte_carlo_returns_ts  # draw once, outside the timers

    started = time.perf_counter()
    reference = return_ts.apply(
        dcf_calculations.get_wealth_indexes_fv_with_cashflow,
        axis=0,
        args=(None, None, ind, "monte_carlo"),
    )
    reference_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(return_ts, ind, pf.dcf.discount_rate)
    engine_seconds = time.perf_counter() - started

    assert result.shape == reference.shape
    assert engine_seconds < reference_seconds
    print(
        f"\n[{frequency}] mc_number=1000, period=30y: reference {reference_seconds:.2f}s, "
        f"engine {engine_seconds:.3f}s, speedup x{reference_seconds / engine_seconds:.0f}"
    )
```

- [ ] **Step 2: Run it and record the numbers**

Run: `OKAMA_BENCH=1 poetry run pytest tests/portfolio/dcf/test_mc_engine_benchmark.py -s`
Expected: 2 passed, printed speedup lines. Record both lines — they go into the Task 6 CHANGELOG wording check and the final report. Also confirm the module is skipped in a normal run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_benchmark.py -q` → 2 skipped.

- [ ] **Step 3: Commit**

```bash
git add tests/portfolio/dcf/test_mc_engine_benchmark.py
git commit -m "test(dcf): add env-gated benchmark for the vectorized MC engine

Skipped unless OKAMA_BENCH=1; times the per-path reference vs the
vectorized engine on 1000 paths x 30 years for yearly and monthly
frequencies and asserts the engine is faster.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: CHANGELOG + final verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append to the `### Changed` section under `## [Unreleased]`**

```markdown
- Monte Carlo wealth simulation is vectorized: `Portfolio.dcf.monte_carlo_wealth()`
  and everything built on it (`monte_carlo_survival_period()`,
  `monte_carlo_cash_flow()`, `plot_forecast_monte_carlo()`,
  `find_the_largest_withdrawals_size()`, `monte_carlo_irr()`) now compute all
  random paths in one pass (`get_wealth_indexes_fv_with_cashflow_mc` in
  `okama.portfolios.dcf_calculations`) instead of a per-path pandas `apply`.
  Results are unchanged (pinned by an equivalence-test grid across strategies,
  frequencies and extra cash flows); typical speedup of one full simulation is
  one to two orders of magnitude. The negative-balance masking and the
  survival-date scan are vectorized as well.
```

If the measured benchmark speedups (Task 5) contradict "one to two orders of magnitude", adjust the sentence to the measured range — the CHANGELOG must not overclaim.

- [ ] **Step 2: Final verification**

Run: `poetry run ruff check .` → `All checks passed!`
Run: `poetry run pytest -q` → all green, no warnings in the summary.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): document the vectorized Monte Carlo wealth engine

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Out of scope (later)

- The backtest path (`wealth_index`, `get_cash_flow_fv`) keeps the per-column reference implementation — migrating it is a separate change.
- The two reference quirks documented above are candidate BUGFIXES to handle separately (their fix must update both the reference and the engine plus the equivalence tests).
- Stage 3 (structural shortcut solvers) gets its own plan after this stage lands.
- Stage 1 leftovers (notebook `examples/04` `iter_max` comments, docstring `solutions` example regeneration) — fold into the Stage 3 / release docs pass.
