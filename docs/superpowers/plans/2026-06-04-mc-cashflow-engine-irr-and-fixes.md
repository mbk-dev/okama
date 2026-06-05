# MC cash-flow engine, IRR routing, and fixes #81/#82 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two reference-engine bugs (GitHub issues #81, #82), add a vectorized Monte Carlo cash-flow engine sharing one core pass with the wealth engine, and route `monte_carlo_cash_flow` / `monte_carlo_irr` through it — completing the vectorization of every `monte_carlo_*` method.

**Architecture:** Bugs are fixed in the per-path reference functions AND the vectorized engine simultaneously (the equivalence grid keeps them in lockstep). Then the wealth engine is refactored into a shared core `_simulate_paths_mc(ror, params, discount_rate) -> (wealth, cash_flow)` with two thin public wrappers; `monte_carlo_cash_flow` and `monte_carlo_irr` route through the wrappers and the existing dcf-level caches.

**Tech Stack:** numpy, pandas, pytest. No new dependencies.

**Context docs:** spec `docs/superpowers/specs/2026-06-04-find-largest-withdrawals-speedup-design.md`; issues https://github.com/mbk-dev/okama/issues/81 and /82; prior plan `docs/superpowers/plans/2026-06-04-find-largest-withdrawals-stage2-vectorize.md`.

**Branch:** `dev`.

---

## Context for the engineer (read first)

- Reference per-path functions in `okama/portfolios/dcf_calculations.py`: `get_wealth_indexes_fv_with_cashflow` (~lines 14-162) and `get_cash_flow_fv` (~165-297). Vectorized wealth engine: `get_wealth_indexes_fv_with_cashflow_mc` + helpers (`_cwd_reduction_factor_matrix`, `_vds_withdrawal_vector`, `_resample_slices`), added below them. Verify all line numbers with grep before editing.
- Equivalence grid: `tests/portfolio/dcf/test_mc_engine_equivalence.py` (19 tests). It pins engine == reference; when a bug is fixed in BOTH simultaneously, the grid stays green.
- **Bug #81:** in both reference functions' slow branch, when a resample period contains nonzero extra cash flows, the in-period loop special-cases `k == 0`: the wealth function applies the cash flow but NOT the month's return; the cash-flow function applies NEITHER. The correct recursion (used by the monthly fast branch and the no-extra-cashflow `cumprod` branch) applies the return every month: `balance = balance * (1 + r) + cash_flow`.
- **Bug #82:** `get_wealth_indexes_fv_with_cashflow` initializes `last_regular_cash_flow = 0` and never updates it, so VDS gets `last_withdrawal=0` every period and its floor/ceiling limits never bind. `get_cash_flow_fv` updates it correctly (`last_regular_cash_flow = cashflow_value` right after the `period_fraction` scaling).
- VDS scalar logic to mirror: `VanguardDynamicSpending._calculate_withdrawal_size` (`okama/portfolios/cashflow_strategies.py:714-802`). Branch order matters: in-range → `-wsbp`; `wsbp > max_final` → `-max_final`; `wsbp < min_final` → `-min_final`. With both limit kinds set: `max_final = ceiling if min < ceiling <= max else max`; `min_final = floor if floor > min else min`. With floor/ceiling only: `min_final = floor_indexed`, `max_final = ceiling_indexed if ceiling_indexed != 0 else wsbp`.
- Reference cash-flow output (`get_cash_flow_fv`): a Series on the ror index (NO prepended initial row). Fast branch: `cs[date] = regular_cashflow + extra_cf[date]` each month. Slow branch: in-period rows carry the (discounted) extra cash flows; the regular period-end cash flow (scaled by `period_fraction`) is ADDED to the period's last month.
- Caches: `PortfolioDCF._monte_carlo_wealth_fv` and `._monte_carlo_cash_flow_fv`; both are cleared by strategy setters (`cashflow_strategies.py:167-172`) and MC setters (`mc.py:219-222`).
- `monte_carlo_cash_flow` (`okama/portfolios/dcf.py:676-686`) still builds its cache with a per-path apply; `monte_carlo_irr` (`dcf.py:864-888`) builds BOTH wealth and cash flow with per-path applies and re-implements the negative masking.
- These bugfixes change user-visible numbers (backtest `wealth_index` and `cash_flow_ts` included) for: periodic frequencies with extra cash flows inside a period (#81), and VDS with `floor_ceiling` (#82). That is the point; CHANGELOG documents it under "Fixed".
- Tests via `poetry run pytest` (never bare pytest); lint `poetry run ruff check .`; comments/docstrings in English. Never stage the unrelated dirty files (`examples/07 efficient frontier multi-period.ipynb`, `OPTIMIZATION_RECOMMENDATIONS.md`, `get_grid_portfolios_recommendation.md`).
- Commit messages referencing the issues use GitHub keywords (`Fixes #81`) so the issues auto-close when `dev` merges to `master`.

## File structure

- Create: `tests/portfolio/dcf/test_dcf_calculation_fixes.py` — hand-computed TDD reproductions for #81/#82 + the VDS scalar↔vector property test.
- Modify: `okama/portfolios/dcf_calculations.py` — both reference functions, the engine (#81/#82 + shared-core refactor), new `get_cash_flow_fv_mc`.
- Modify: `tests/portfolio/dcf/test_mc_engine_equivalence.py` — new VDS grid cases, cash-flow grid, routing guards.
- Modify: `okama/portfolios/dcf.py` — `monte_carlo_cash_flow` and `monte_carlo_irr` routing.
- Modify: `tests/portfolio/dcf/test_mc_engine_benchmark.py` — gated IRR benchmark.
- Modify: `CHANGELOG.md`.

---

### Task 1: Fix #81 — first month of a period with extra cash flows must earn its return

**Files:**
- Create: `tests/portfolio/dcf/test_dcf_calculation_fixes.py`
- Modify: `okama/portfolios/dcf_calculations.py`

- [ ] **Step 1: Write the two failing reproductions (hand-computed, deterministic ror)**

Create `tests/portfolio/dcf/test_dcf_calculation_fixes.py`:

```python
"""Reproductions and regression tests for reference-engine fixes (issues #81, #82).

All tests build deterministic return series by hand and compute the expected
balances with the canonical recursion `balance = balance * (1 + r) + cash_flow`
applied to every month.
"""

import numpy as np  # noqa: I001
import pandas as pd
import pytest
import okama as ok
from okama.portfolios import dcf_calculations


@pytest.fixture()
def pf_single(synthetic_env):
    return ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))


def test_wealth_index_applies_first_month_return_in_period_with_extra_cash_flows(pf_single) -> None:
    # Issue #81 (wealth side): with an extra cash flow anywhere in a resample
    # period, the first month of that period must still earn its return.
    ind = ok.IndexationStrategy(pf_single)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = 0  # no regular withdrawals: isolate the in-period recursion
    ind.indexation = 0.0
    ind.time_series_dic = {"2022-04": -500}
    ind.time_series_discounted_values = True  # keep the extra flow at face value
    pf_single.dcf.cashflow_parameters = ind

    idx = pd.period_range("2022-01", periods=12, freq="M")
    ror = pd.Series(0.01, index=idx)

    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow(ror, None, None, ind, "monte_carlo")

    balance = 10_000.0
    expected = {}
    for date, r in ror.items():
        balance = balance * (1 + r) + (-500.0 if str(date) == "2022-04" else 0.0)
        expected[date] = balance

    assert result.loc[idx[0]] == pytest.approx(expected[idx[0]])  # 10_100, not 10_000
    assert result.loc[idx[-1]] == pytest.approx(expected[idx[-1]])


def test_cash_flow_fv_sizes_next_period_from_fully_compounded_balance(pf_single) -> None:
    # Issue #81 (cash-flow side): the internal balance tracking must include
    # both the first month's return and the extra cash flow, otherwise the
    # next period's percentage withdrawal is sized from a wrong balance.
    pc = ok.PercentageStrategy(pf_single)
    pc.initial_investment = 10_000
    pc.frequency = "year"
    pc.percentage = -0.12
    pc.time_series_dic = {"2022-01": -500}
    pc.time_series_discounted_values = True
    pf_single.dcf.cashflow_parameters = pc

    idx = pd.period_range("2022-01", periods=24, freq="M")
    ror = pd.Series(0.0, index=idx)  # zero returns keep the math exact

    result = dcf_calculations.get_cash_flow_fv(ror, None, pc, "monte_carlo")

    # Year 2022: start 10_000, extra -500 in January, regular withdrawal
    # -0.12 * 10_000 = -1_200 at year end -> 2023 starts at 8_300.
    # Year 2023: regular withdrawal -0.12 * 8_300 = -996 at year end.
    assert result.loc[pd.Period("2023-12", freq="M")] == pytest.approx(-996.0)
```

- [ ] **Step 2: Verify RED**

Run: `poetry run pytest tests/portfolio/dcf/test_dcf_calculation_fixes.py -v`
Expected: both tests FAIL with `AssertionError` on the value comparisons:
- wealth test: first month is `10_000.0` (return skipped) instead of `10_100.0`;
- cash-flow test: `-1_056.0` (the buggy balance tracking drops January's `-500` entirely, so 2023 starts at `8_800` and the withdrawal is `-0.12 * 8_800`; if the actual buggy number differs, record it — the point is it is not `-996.0`).
If a test fails on strategy construction instead, fix the TEST and re-run.

- [ ] **Step 3: Fix both reference functions**

In `okama/portfolios/dcf_calculations.py`, in `get_wealth_indexes_fv_with_cashflow` (slow branch, the block currently reading `if k == 0: month_balance = period_initial_amount + cashflow_ts_local[date]`), replace:

```python
            if (cashflow_ts_local != 0).any():
                period_wealth_index = pd.Series(dtype=float, name=portfolio_symbol)
                for k, (date, r) in enumerate(ror_ts.items()):
                    if k == 0:
                        month_balance = period_initial_amount + cashflow_ts_local[date]
                    else:
                        month_balance = month_balance * (r + 1) + cashflow_ts_local[date]
                    period_wealth_index[date] = month_balance
```

with:

```python
            if (cashflow_ts_local != 0).any():
                period_wealth_index = pd.Series(dtype=float, name=portfolio_symbol)
                month_balance = period_initial_amount
                for date, r in ror_ts.items():
                    month_balance = month_balance * (r + 1) + cashflow_ts_local[date]
                    period_wealth_index[date] = month_balance
```

In `get_cash_flow_fv` (slow branch, the block currently reading `if k == 0: month_balance = period_initial_amount`), replace:

```python
            if (cashflow_ts_local != 0).any():
                period_wealth_index = pd.Series(dtype=float, name=portfolio_symbol)
                for k, (date, r) in enumerate(ror_ts.items()):
                    if k == 0:
                        month_balance = period_initial_amount
                    else:
                        month_balance = month_balance * (r + 1) + cashflow_ts_local[date]
                    period_wealth_index[date] = month_balance
```

with the same uniform loop as above (identical five lines).

- [ ] **Step 4: Fix the vectorized engine and its docs**

In `get_wealth_indexes_fv_with_cashflow_mc` (same file), replace the quirk block:

```python
            if np.any(extra_cf[start:stop] != 0):
                # Reference quirk kept intentionally (GitHub issue #81): the
                # first month of a period with extra cash flows gets the cash
                # flow but NOT the month's return.
                for k in range(start, stop):
                    if k == start:
                        balance = balance + extra_cf[k]
                    else:
                        balance = balance * (1 + returns[k]) + extra_cf[k]
                    wealth[k] = balance
```

with:

```python
            if np.any(extra_cf[start:stop] != 0):
                for k in range(start, stop):
                    balance = balance * (1 + returns[k]) + extra_cf[k]
                    wealth[k] = balance
```

In the engine docstring, delete the bullet "- the first month of a resample period that contains extra cash flows receives the cash flow but not the month's return (GitHub issue #81);" (the quirks list intro sentence is rewritten in Task 2).

- [ ] **Step 5: GREEN + grid + suite**

Run: `poetry run pytest tests/portfolio/dcf/test_dcf_calculation_fixes.py -v` — 2 passed.
Run: `poetry run pytest tests/portfolio/dcf/ -q` — all pass (the equivalence grid compares the fixed reference to the fixed engine: still green).
Run: `poetry run pytest -q` — full suite green. If any pre-existing test pinned the buggy numbers, STOP and report it (do not silently update expectations).
Run: `poetry run ruff check .` — clean.

- [ ] **Step 6: Commit**

```bash
git add okama/portfolios/dcf_calculations.py tests/portfolio/dcf/test_dcf_calculation_fixes.py
git commit -m "fix(dcf): apply the first month's return in periods with extra cash flows

In the slow (periodic-frequency) branch of both per-path reference
functions, a period containing extra cash flows special-cased its first
month: get_wealth_indexes_fv_with_cashflow skipped the month's return and
get_cash_flow_fv skipped both the return and the cash flow in its balance
tracking. The recursion is now uniform (balance = balance*(1+r) + cf) for
every month, matching the monthly fast branch and the no-extra-cash-flow
cumprod branch. The vectorized engine is fixed identically; the
equivalence grid keeps both in lockstep.

Fixes #81

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Fix #82 — VDS must see the previous period's withdrawal

**Files:**
- Modify: `okama/portfolios/dcf_calculations.py`
- Modify: `tests/portfolio/dcf/test_dcf_calculation_fixes.py`
- Modify: `tests/portfolio/dcf/test_mc_engine_equivalence.py`

- [ ] **Step 1: Write the failing reproduction**

Append to `tests/portfolio/dcf/test_dcf_calculation_fixes.py`:

```python
def test_vds_floor_ceiling_binds_in_wealth_index(pf_single) -> None:
    # Issue #82: with floor_ceiling=(0, 0) every withdrawal must equal the
    # previous one. With zero returns: 10_000 -> withdrawal 1_000 (first
    # period, by percentage) -> 9_000 -> withdrawal forced to 1_000 again
    # (not 0.10 * 9_000 = 900) -> 8_000.
    vds = ok.VanguardDynamicSpending(
        pf_single,
        initial_investment=10_000,
        percentage=-0.10,
        floor_ceiling=(0.0, 0.0),
        adjust_floor_ceiling=False,
        indexation=0.0,
    )
    pf_single.dcf.cashflow_parameters = vds

    idx = pd.period_range("2022-01", periods=24, freq="M")
    ror = pd.Series(0.0, index=idx)

    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow(ror, None, None, vds, "monte_carlo")

    assert result.loc[pd.Period("2022-12", freq="M")] == pytest.approx(9_000.0)
    assert result.loc[pd.Period("2023-12", freq="M")] == pytest.approx(8_000.0)
```

Run: `poetry run pytest tests/portfolio/dcf/test_dcf_calculation_fixes.py::test_vds_floor_ceiling_binds_in_wealth_index -v`
Expected: FAIL with `AssertionError` on the second assert — actual `8_100.0` (the second withdrawal was `900` because `last_withdrawal` stayed 0).

- [ ] **Step 2: Fix the reference wealth function**

In `get_wealth_indexes_fv_with_cashflow`, right after the line `cashflow_value *= period_fraction  # adjust cash flow to the period length (months)`, insert (mirroring `get_cash_flow_fv`):

```python
            last_regular_cash_flow = cashflow_value
```

- [ ] **Step 3: Replace `_vds_withdrawal_vector` with the full scalar-faithful tree**

Replace the whole `_vds_withdrawal_vector` function with:

```python
def _vds_withdrawal_vector(
    cashflow_parameters: cf.CashFlow,
    balance: np.ndarray,
    last_withdrawal: float | np.ndarray,
    number_of_periods: int,
) -> np.ndarray:
    """Vectorized VanguardDynamicSpending withdrawal for one period across paths.

    Faithful translation of `VanguardDynamicSpending._calculate_withdrawal_size`
    to numpy, preserving the scalar branch order (in-range -> percentage;
    above max -> max; below min -> min). Scalar/vector parity is pinned by a
    property test.
    """
    withdrawal_by_percentage = balance * abs(cashflow_parameters.percentage)
    last = np.abs(last_withdrawal)
    has_floor_ceiling = cashflow_parameters.floor_ceiling is not None
    has_min_max = cashflow_parameters.min_max_annual_withdrawals is not None
    if has_floor_ceiling:
        floor, ceiling = cashflow_parameters.floor_ceiling
        adjust = (1 + cashflow_parameters.indexation) if cashflow_parameters.adjust_floor_ceiling else 1.0
        floor_indexed = last * adjust * (1 + floor)
        ceiling_indexed = last * adjust * (1 + ceiling)
    if has_min_max:
        min_withdrawal, max_withdrawal = cashflow_parameters.min_max_annual_withdrawals
        indexation_factor = (
            (1 + cashflow_parameters.indexation) ** number_of_periods if cashflow_parameters.adjust_min_max else 1.0
        )
        min_indexed = abs(min_withdrawal) * indexation_factor
        max_indexed = abs(max_withdrawal) * indexation_factor
    if has_floor_ceiling and has_min_max:
        max_final = np.where(
            ceiling_indexed > max_indexed,
            max_indexed,
            np.where((min_indexed < ceiling_indexed) & (ceiling_indexed <= max_indexed), ceiling_indexed, max_indexed),
        )
        min_final = np.where(floor_indexed > min_indexed, floor_indexed, min_indexed)
    elif has_min_max:
        min_final, max_final = min_indexed, max_indexed
    elif has_floor_ceiling:
        min_final = floor_indexed
        max_final = np.where(ceiling_indexed != 0, ceiling_indexed, withdrawal_by_percentage)
    else:
        return -withdrawal_by_percentage
    withdrawal = np.where(
        (min_final <= withdrawal_by_percentage) & (withdrawal_by_percentage <= max_final),
        withdrawal_by_percentage,
        np.where(withdrawal_by_percentage > max_final, max_final, min_final),
    )
    return -withdrawal
```

- [ ] **Step 4: Track the per-path withdrawal state in the engine**

In `get_wealth_indexes_fv_with_cashflow_mc`, in the periodic (slow) branch:

- before the `for n, (start, stop) in enumerate(...)` loop, add:

```python
        last_regular_cashflow: float | np.ndarray = 0.0
```

- replace the VDS branch call:

```python
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = _vds_withdrawal_vector(cashflow_parameters, start_balance, n)
```

with:

```python
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = _vds_withdrawal_vector(
                    cashflow_parameters, start_balance, last_regular_cashflow if n > 0 else 0.0, n
                )
```

- right after `cashflow_value = cashflow_value * ((stop - start) / months_in_full_period)`, add:

```python
            last_regular_cashflow = cashflow_value
```

- In the engine docstring, replace the remaining quirks block (intro sentence + the VDS bullet) with:

```python
    Replicates the per-path reference exactly; equivalence is pinned by tests.
    (The two historical reference quirks were fixed together with this engine —
    see GitHub issues #81 and #82.)
```

- [ ] **Step 5: Add the scalar↔vector property test**

Append to `tests/portfolio/dcf/test_dcf_calculation_fixes.py`:

```python
@pytest.mark.parametrize("floor_ceiling", [None, (-0.025, 0.05)])
@pytest.mark.parametrize("min_max", [None, (500.0, 900.0)])
@pytest.mark.parametrize("adjust_floor_ceiling", [False, True])
def test_vds_withdrawal_vector_matches_scalar(pf_single, floor_ceiling, min_max, adjust_floor_ceiling) -> None:
    vds = ok.VanguardDynamicSpending(
        pf_single,
        initial_investment=10_000,
        percentage=-0.08,
        floor_ceiling=floor_ceiling,
        adjust_floor_ceiling=adjust_floor_ceiling,
        min_max_annual_withdrawals=min_max,
        adjust_min_max=True,
        indexation=0.04,
    )
    balances = np.array([100.0, 4_000.0, 8_000.0, 12_000.0, 20_000.0])
    last_withdrawals = np.array([0.0, -300.0, -700.0, -900.0, -1_500.0])
    for n in (0, 3):
        vector = dcf_calculations._vds_withdrawal_vector(vds, balances, last_withdrawals, n)
        scalar = np.array(
            [
                vds._calculate_withdrawal_size(last_withdrawal=lw, balance=b, number_of_periods=n)
                for b, lw in zip(balances, last_withdrawals)
            ]
        )
        np.testing.assert_allclose(vector, scalar, rtol=1e-12)
```

- [ ] **Step 6: Add floor/ceiling cases to the equivalence grid**

In `tests/portfolio/dcf/test_mc_engine_equivalence.py`, add a builder next to `_vds_indexed`:

```python
def _vds_floor_ceiling(pf):
    return ok.VanguardDynamicSpending(
        pf,
        initial_investment=10_000,
        percentage=-0.08,
        indexation=0.03,
        floor_ceiling=(-0.025, 0.05),
        adjust_floor_ceiling=True,
        min_max_annual_withdrawals=(500.0, 900.0),
        adjust_min_max=True,
    )
```

and two CASES entries after `"vds_year_indexed"`:

```python
    ("vds_year_floor_ceiling", lambda pf: _vds_floor_ceiling(pf), None, 3, None),
    ("vds_year_floor_ceiling_partial", lambda pf: _vds_floor_ceiling(pf), "2021-06", 3, None),
```

- [ ] **Step 7: GREEN + suite**

Run: `poetry run pytest tests/portfolio/dcf/test_dcf_calculation_fixes.py tests/portfolio/dcf/test_mc_engine_equivalence.py -v` — all pass (reproduction now GREEN; grid incl. the two new floor/ceiling cases GREEN — the engine and the fixed reference move together).
Run: `poetry run pytest -q` — full suite green (same stop-and-report rule if some test pinned buggy numbers).
Run: `poetry run ruff check .` — clean.

- [ ] **Step 8: Commit**

```bash
git add okama/portfolios/dcf_calculations.py tests/portfolio/dcf/test_dcf_calculation_fixes.py tests/portfolio/dcf/test_mc_engine_equivalence.py
git commit -m "fix(dcf): VDS floor/ceiling limits now bind in wealth index calculations

get_wealth_indexes_fv_with_cashflow never updated last_regular_cash_flow,
so VanguardDynamicSpending received last_withdrawal=0 every period and its
floor/ceiling limits were inert (unlike get_cash_flow_fv, which updates
it). The reference now records the scaled period cash flow; the vectorized
engine tracks a per-path withdrawal state and uses a full scalar-faithful
vector translation of _calculate_withdrawal_size (parity pinned by a
property test and new floor/ceiling equivalence-grid cases).

Fixes #82

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Shared core + vectorized cash-flow engine

**Files:**
- Modify: `okama/portfolios/dcf_calculations.py`
- Modify: `tests/portfolio/dcf/test_mc_engine_equivalence.py`

- [ ] **Step 1: Add the failing cash-flow grid (RED)**

Append to `tests/portfolio/dcf/test_mc_engine_equivalence.py`:

```python
def _reference_cash_flow(dcf, params):
    """Old per-column cash-flow engine output (the ground truth)."""
    return_ts = dcf.mc.monte_carlo_returns_ts
    return return_ts.apply(
        dcf_calculations.get_cash_flow_fv,
        axis=0,
        args=(None, params, "monte_carlo"),
    )


@pytest.mark.parametrize(("case_id", "builder", "last_date", "period", "extra_dic"), CASES, ids=[c[0] for c in CASES])
def test_vectorized_cash_flow_matches_per_path_reference(
    synthetic_env, case_id, builder, last_date, period, extra_dic
) -> None:
    pf = _make_portfolio(last_date)
    params = builder(pf)
    if extra_dic is not None:
        params.time_series_dic = extra_dic
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=period, mc_number=8, seed=0)

    reference = _reference_cash_flow(pf.dcf, params)
    result = dcf_calculations.get_cash_flow_fv_mc(
        pf.dcf.mc.monte_carlo_returns_ts, params, pf.dcf.discount_rate
    )

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8, check_names=False)
```

Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_equivalence.py -k cash_flow_matches -v`
Expected: all parametrized cases FAIL with `AttributeError: ... no attribute 'get_cash_flow_fv_mc'` (missing function — valid RED). The `_reference_cash_flow` calls themselves must succeed; if a case fails inside the reference, report it instead of dropping the case.

- [ ] **Step 2: Refactor the engine into a shared core and add the wrapper**

In `okama/portfolios/dcf_calculations.py`:

1. Rename the body of `get_wealth_indexes_fv_with_cashflow_mc` into a private core (keep all logic; add cash-flow collection):

```python
def _simulate_paths_mc(  # noqa: C901
    ror: pd.DataFrame,
    cashflow_parameters: cf.CashFlow,
    discount_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One vectorized Monte Carlo pass over all paths.

    Returns ``(wealth, cash_flow)`` matrices of shape (T, N): the wealth index
    (without the prepended initial row) and the total cash flow recorded per
    month (regular strategy cash flow at its due month plus extra cash flows
    from `time_series`), matching `get_wealth_indexes_fv_with_cashflow` and
    `get_cash_flow_fv` (task="monte_carlo") respectively.
    """
```

The core body is the existing engine body with these changes:

- after `wealth = np.empty((n_rows, n_cols))` add:

```python
    cash_flow = np.zeros((n_rows, n_cols))
```

- fast branch: replace the balance update pair with:

```python
            cs_value = cashflow + extra_cf[n]
            cash_flow[n] = cs_value
            balance = balance * (1 + returns[n]) + cs_value
            wealth[n] = balance
```

(`cashflow` stays exactly as computed per strategy above these lines.)

- slow branch: at the top of each period iteration (right after `start_balance = balance`), add:

```python
            cash_flow[start:stop] = extra_cf[start:stop][:, None]
```

and right after the period-end `balance = balance + cashflow_value` / before `wealth[stop - 1] = balance`, add:

```python
            cash_flow[stop - 1] += cashflow_value
```

- the function returns `return wealth, cash_flow` instead of building a DataFrame.

2. Re-create the public wrappers after the core:

```python
def get_wealth_indexes_fv_with_cashflow_mc(
    ror: pd.DataFrame,
    cashflow_parameters: cf.CashFlow,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Vectorized Monte Carlo counterpart of `get_wealth_indexes_fv_with_cashflow`.

    Computes wealth index future values (FV) for all random return paths at
    once; one Python loop over time steps with numpy operations across paths.
    Implements the "monte_carlo" task semantics only: extra cash flows from
    `time_series` are compounded with the discount rate unless
    `time_series_discounted_values` is True. Replicates the per-path reference
    exactly; equivalence is pinned by tests. (The two historical reference
    quirks were fixed together with this engine — see GitHub issues #81, #82.)
    """
    wealth, _ = _simulate_paths_mc(ror, cashflow_parameters, discount_rate)
    n_cols = ror.shape[1]
    out_index = ror.index.insert(0, ror.index[0] - 1)
    data = np.vstack([np.full((1, n_cols), float(cashflow_parameters.initial_investment)), wealth])
    return pd.DataFrame(data, index=out_index, columns=ror.columns)


def get_cash_flow_fv_mc(
    ror: pd.DataFrame,
    cashflow_parameters: cf.CashFlow,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Vectorized Monte Carlo counterpart of `get_cash_flow_fv`.

    Returns the monthly cash flow (regular strategy cash flow plus extra cash
    flows from `time_series`) for all random return paths at once, on the same
    index as `ror` (no prepended initial row). Equivalence with the per-path
    reference is pinned by tests.
    """
    _, cash_flow = _simulate_paths_mc(ror, cashflow_parameters, discount_rate)
    return pd.DataFrame(cash_flow, index=ror.index, columns=ror.columns)
```

(Keep the docstring of the old engine where the core inherits its detailed notes; do not leave two copies of the same long docstring — the core gets the short one above, the wrappers keep the public-facing ones.)

- [ ] **Step 3: GREEN — both grids**

Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_equivalence.py -v` — ALL tests pass: the wealth grid (refactor must not change wealth results) and the new cash-flow grid.
Debug hints: fast-branch `cs_value` must include extra cash flows (reference records `cashflow + cash_flow_ts[date]`); slow-branch in-period rows carry ONLY extra cash flows; the regular period cash flow lands on the period's LAST month, scaled by `period_fraction`.

- [ ] **Step 4: dcf package + suite + lint**

Run: `poetry run pytest tests/portfolio/dcf/ -q` — all pass. `poetry run pytest -q` — green. `poetry run ruff check .` — clean (`# noqa: C901` stays on the core's `def` line with the same rationale).

- [ ] **Step 5: Commit**

```bash
git add okama/portfolios/dcf_calculations.py tests/portfolio/dcf/test_mc_engine_equivalence.py
git commit -m "perf(dcf): add vectorized Monte Carlo cash-flow engine sharing one core pass

The wealth engine body becomes _simulate_paths_mc, one vectorized pass
returning both the wealth and the cash-flow matrices;
get_wealth_indexes_fv_with_cashflow_mc and the new get_cash_flow_fv_mc are
thin wrappers. The cash-flow side replicates get_cash_flow_fv
(task=monte_carlo) and is pinned by a parametrized equivalence grid over
the same strategy/frequency/extra-cashflow cases as the wealth grid.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Route `monte_carlo_cash_flow` and `monte_carlo_irr`

**Files:**
- Modify: `okama/portfolios/dcf.py`
- Modify: `tests/portfolio/dcf/test_mc_engine_equivalence.py`
- Modify: `tests/portfolio/dcf/test_mc_engine_benchmark.py`

- [ ] **Step 1: Add routing guards (must pass BEFORE and AFTER)**

Append to `tests/portfolio/dcf/test_mc_engine_equivalence.py`:

```python
def test_monte_carlo_cash_flow_matches_per_path_reference(synthetic_env) -> None:
    # Guard for the routing: public API output must stay equal to the old
    # per-path computation (passes before and after the engine switch).
    pf = _make_portfolio()
    params = _indexation(pf, "year")
    params.time_series_dic = {"2022-03": -500}
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=8, seed=0)

    reference = _reference_cash_flow(pf.dcf, params)
    params._clear_cf_cache()
    result = pf.dcf.monte_carlo_cash_flow(discounting="fv", remove_if_wealth_index_negative=False)

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8, check_names=False)


def test_monte_carlo_irr_matches_per_path_reference(synthetic_env) -> None:
    # Guard for the routing: IRR distribution must stay equal to the inline
    # per-path construction used by the pre-routing implementation.
    from okama import settings

    pf = _make_portfolio()
    params = _indexation(pf, "year")
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=8, seed=0)
    return_ts = pf.dcf.mc.monte_carlo_returns_ts

    wealth = _reference_wealth(pf.dcf, params)
    wealth = wealth.apply(dcf_calculations.remove_negative_values, axis=0).fillna(0.0)
    cash_flow = _reference_cash_flow(pf.dcf, params)
    cash_flow = cash_flow.where(wealth.reindex(cash_flow.index) != 0, 0.0)
    terminal = wealth.iloc[-1]
    import numpy as np

    flows = np.empty((cash_flow.shape[0] + 1, cash_flow.shape[1]), dtype=float)
    flows[0, :] = -params.initial_investment
    flows[1:, :] = -cash_flow.to_numpy()
    flows[-1, :] += terminal.reindex(cash_flow.columns).to_numpy()
    expected = pd.Series(
        dcf_calculations.irr_of_cashflow_matrix(flows, periods_per_year=settings._MONTHS_PER_YEAR),
        index=cash_flow.columns,
        name="monte_carlo_irr",
    )

    params._clear_cf_cache()
    result = pf.dcf.monte_carlo_irr()

    pd.testing.assert_series_equal(result, expected, rtol=1e-9)
```

(Move the `import numpy as np` / `from okama import settings` lines to the top of the file with the other imports — ruff E402/I001 will flag inline imports; the snippet shows them inline only to make the dependency explicit.)

Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_equivalence.py -k "matches_per_path_reference" -v` — both new guards PASS against the current per-path implementations.

- [ ] **Step 2: Route `monte_carlo_cash_flow`**

In `okama/portfolios/dcf.py`, replace the cache-building block:

```python
        if self._monte_carlo_cash_flow_fv.empty:
            return_ts = self.mc.monte_carlo_returns_ts
            self._monte_carlo_cash_flow_fv = return_ts.apply(
                dcf_calculations.get_cash_flow_fv,
                axis=0,
                args=(
                    self.parent.symbol,  # portfolio_symbol
                    self.cashflow_parameters,
                    "monte_carlo",  # task
                ),
            )
```

with:

```python
        if self._monte_carlo_cash_flow_fv.empty:
            return_ts = self.mc.monte_carlo_returns_ts
            self._monte_carlo_cash_flow_fv = dcf_calculations.get_cash_flow_fv_mc(
                return_ts, self.cashflow_parameters, self.discount_rate
            )
```

- [ ] **Step 3: Route `monte_carlo_irr`**

Replace the body after the `cashflow_parameters is None` check (keep signature and docstring) with:

```python
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        cashflow_parameters = self.cashflow_parameters
        wealth = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
        cash_flow = self.monte_carlo_cash_flow(discounting="fv", remove_if_wealth_index_negative=False)
        # Zero a path's cash flow once its (floored) wealth is depleted, consistent per path.
        cash_flow = cash_flow.where(wealth.reindex(cash_flow.index) != 0, 0.0)
        terminal = wealth.iloc[-1]
        n_months, n_paths = cash_flow.shape
        flows = np.empty((n_months + 1, n_paths), dtype=float)
        flows[0, :] = -cashflow_parameters.initial_investment
        flows[1:, :] = -cash_flow.to_numpy()
        flows[-1, :] += terminal.reindex(cash_flow.columns).to_numpy()
        irr = dcf_calculations.irr_of_cashflow_matrix(flows, periods_per_year=settings._MONTHS_PER_YEAR)
        return pd.Series(irr, index=cash_flow.columns, name="monte_carlo_irr")
```

(The method now reuses the shared dcf-level caches; both `monte_carlo_wealth` and `monte_carlo_cash_flow` consume the same cached draw, so the docstring's shared-draw note stays true.)

- [ ] **Step 4: Add the gated IRR benchmark**

Append to `tests/portfolio/dcf/test_mc_engine_benchmark.py`:

```python
def test_benchmark_monte_carlo_irr(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -500
    ind.indexation = 0.05
    pf.dcf.cashflow_parameters = ind
    pf.dcf.set_mc_parameters(distribution="norm", period=30, mc_number=1_000, seed=0)
    return_ts = pf.dcf.mc.monte_carlo_returns_ts  # draw once, outside the timers

    started = time.perf_counter()
    reference_wealth = return_ts.apply(
        dcf_calculations.get_wealth_indexes_fv_with_cashflow, axis=0, args=(None, None, ind, "monte_carlo")
    )
    reference_cash_flow = return_ts.apply(
        dcf_calculations.get_cash_flow_fv, axis=0, args=(None, ind, "monte_carlo")
    )
    reference_seconds = time.perf_counter() - started
    assert reference_wealth.shape[1] == reference_cash_flow.shape[1]

    ind._clear_cf_cache()
    started = time.perf_counter()
    irr = pf.dcf.monte_carlo_irr()
    routed_seconds = time.perf_counter() - started

    assert irr.shape[0] == 1_000
    assert routed_seconds < reference_seconds
    print(
        f"\n[irr] mc_number=1000, period=30y: per-path matrices {reference_seconds:.2f}s, "
        f"routed monte_carlo_irr {routed_seconds:.3f}s, speedup x{reference_seconds / routed_seconds:.0f}"
    )
```

- [ ] **Step 5: Verify**

Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_equivalence.py -v` — all pass (both guards green AFTER routing).
Run: `poetry run pytest -q` — full suite green (notably `tests/portfolio/mc/test_monte_carlo_irr.py`).
Run: `OKAMA_BENCH=1 poetry run pytest tests/portfolio/dcf/test_mc_engine_benchmark.py -s` — 3 passed; RECORD the printed `[irr]` line for the CHANGELOG and report.
Run: `poetry run pytest tests/portfolio/dcf/test_mc_engine_benchmark.py -q` (no env) — 3 skipped.
Run: `poetry run ruff check .` — clean.

- [ ] **Step 6: Commit**

```bash
git add okama/portfolios/dcf.py tests/portfolio/dcf/test_mc_engine_equivalence.py tests/portfolio/dcf/test_mc_engine_benchmark.py
git commit -m "perf(dcf): route monte_carlo_cash_flow and monte_carlo_irr through the vectorized engines

monte_carlo_cash_flow builds its cache with get_cash_flow_fv_mc (one pass
for all paths); monte_carlo_irr consumes monte_carlo_wealth and
monte_carlo_cash_flow instead of two private per-path applies, sharing the
dcf-level caches and dropping its duplicated negative-masking. Behavior is
pinned by two public-API guards against the per-path references; an
env-gated IRR benchmark records the end-to-end speedup.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: CHANGELOG + final verification

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the entries**

Under `## [Unreleased]` / `### Fixed`, append:

```markdown
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
```

Under `### Changed`, append (insert the measured `[irr]` numbers from Task 4 — do not overclaim):

```markdown
- Monte Carlo cash-flow simulation is vectorized as well:
  `Portfolio.dcf.monte_carlo_cash_flow()` builds its cache with the new
  `get_cash_flow_fv_mc` (one pass for all paths, sharing a core with the
  wealth engine), and `Portfolio.dcf.monte_carlo_irr()` now consumes the
  shared `monte_carlo_wealth`/`monte_carlo_cash_flow` caches instead of two
  per-path computations (measured end-to-end speedup: ×<N> on 1,000 paths ×
  30 years — substitute the benchmark number).
```

- [ ] **Step 2: Final verification**

Run: `poetry run ruff check .` → All checks passed. `poetry run pytest -q` → all green (expect 3 skipped benchmark tests). Report exact counts.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): document #81/#82 fixes and the vectorized cash-flow engine

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Out of scope

- Backtest-path vectorization (`wealth_index`, `cash_flow_ts` per-Series calls) — the references stay; they now agree with the engines on semantics.
- Structural shortcut solvers (former Stage 3) — dropped by decision of 2026-06-04 (the ×1400–6800 engine speedup made them unnecessary).
- Stage 1 docs leftovers (notebook `examples/04` `iter_max` comments, docstring `solutions` table) — release docs pass.
