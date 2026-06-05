# `Portfolio.dcf.irr` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Portfolio.dcf.irr()` returning the annualized nominal money-weighted IRR of the portfolio cash flow over the full historical period, built on a fast vectorized core reusable by Monte Carlo.

**Architecture:** A pure vectorized core `irr_of_cashflow_matrix` (Newton–Raphson over all columns at once, ratio seed, `brentq` fallback) lives in `dcf_calculations.py`. `PortfolioDCF.irr` assembles the investor cash-flow vector from the existing `cash_flow_ts` + `wealth_index` + `initial_investment` and solves the nominal IRR via the core. IRR is a rate, so there is no FV/PV switch (that would be nominal-vs-real, the `get_cagr(real=...)` axis, out of scope here).

**Tech Stack:** Python 3.11+, numpy, scipy (`scipy.optimize.brentq`), pandas, pytest. No new dependency.

**Conventions:** TDD (RED → GREEN), `pytest -q` (or `poetry run pytest -q`), `poetry run ruff check .` must be clean. Tests use the existing `synthetic_env` fixture (mocked data, no network). Conventional commit messages (`feat(dcf): ...`).

---

## File Structure

- `okama/portfolios/dcf_calculations.py` — **modify**: add `from scipy import optimize`, `_irr_initial_guess`, `_irr_brentq_column`, and the public `irr_of_cashflow_matrix`. This file already owns the cash-flow math, so the solver belongs here.
- `okama/portfolios/dcf.py` — **modify**: add `import numpy as np` and the `PortfolioDCF.irr` method.
- `tests/portfolio/dcf/test_irr.py` — **create**: unit tests for the core (pure, deterministic) + integration tests for `dcf.irr` (via `synthetic_env`).

---

## Task 1: Vectorized IRR core + seed helper

**Files:**
- Modify: `okama/portfolios/dcf_calculations.py`
- Test: `tests/portfolio/dcf/test_irr.py` (create)

- [ ] **Step 1: Write the failing core tests**

Create `tests/portfolio/dcf/test_irr.py`:

```python
import numpy as np
import pytest

import okama as ok  # noqa: F401  (used by integration tests added in later tasks)
from okama.portfolios import dcf_calculations
from okama.settings import _MONTHS_PER_YEAR, DEFAULT_DISCOUNT_RATE  # noqa: F401


def test_irr_core_single_in_single_out_matches_closed_form():
    # -1000 at t0, +1200 at t12 on a monthly grid -> annual IRR is exactly 20%.
    cf = np.zeros(13)
    cf[0] = -1000.0
    cf[12] = 1200.0
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=_MONTHS_PER_YEAR)
    assert result[0] == pytest.approx(0.2, abs=1e-9)


def test_irr_core_textbook_per_period_rate():
    # -100, +60, +60 with periods_per_year=1 -> per-period IRR from 100 x^2 - 60 x - 60 = 0.
    cf = np.array([-100.0, 60.0, 60.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    x = (60.0 + np.sqrt(3600.0 + 24000.0)) / 200.0
    assert result[0] == pytest.approx(x - 1.0, abs=1e-9)


def test_irr_core_no_sign_change_returns_nan():
    cf = np.array([-100.0, -50.0, -30.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert np.isnan(result[0])


def test_irr_core_vectorized_columns_are_independent():
    # Two columns solved at once; periods_per_year=1 so annual == per-period rate.
    cf = np.array(
        [
            [-1000.0, -1000.0],
            [0.0, 0.0],
            [1200.0, 900.0],
        ]
    )
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert result[0] == pytest.approx(np.sqrt(1.2) - 1.0, abs=1e-9)
    assert result[1] == pytest.approx(np.sqrt(0.9) - 1.0, abs=1e-9)


def test_irr_core_depleted_partial_recovery_is_negative():
    # Invested 1000, recovered only 300 over the period, terminal 0 -> finite, negative IRR.
    cf = np.array([-1000.0, 100.0, 100.0, 100.0, 0.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert np.isfinite(result[0])
    assert result[0] < 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/portfolio/dcf/test_irr.py -q`
Expected: FAIL with `AttributeError: module 'okama.portfolios.dcf_calculations' has no attribute 'irr_of_cashflow_matrix'`.

- [ ] **Step 3: Add the scipy import**

In `okama/portfolios/dcf_calculations.py`, add to the imports block (after `import numpy as np`):

```python
from scipy import optimize
```

- [ ] **Step 4: Implement the seed helper, the brentq fallback, and the core**

Append to `okama/portfolios/dcf_calculations.py`:

```python
def _irr_initial_guess(cashflows: np.ndarray) -> np.ndarray:
    """
    Cheap per-column periodic-rate seed for Newton IRR iteration.

    For the dominant single-outflow/single-inflow case this equals
    ``(total_in / total_out) ** (1 / horizon) - 1``, i.e. the true periodic rate,
    so Newton converges in ~1 iteration.
    """
    n_periods = cashflows.shape[0]
    horizon = max(n_periods - 1, 1)
    inflows = np.where(cashflows > 0.0, cashflows, 0.0).sum(axis=0)
    outflows = -np.where(cashflows < 0.0, cashflows, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(outflows > 0.0, inflows / outflows, 1.0)
    ratio = np.where(ratio > 0.0, ratio, 1.0)
    return ratio ** (1.0 / horizon) - 1.0


def _irr_brentq_column(cashflow_column: np.ndarray) -> float:
    """Bracketing-solver fallback for a single cash-flow column. Returns NaN if no bracket."""
    t = np.arange(cashflow_column.shape[0], dtype=float)

    def npv(rate: float) -> float:
        return float((cashflow_column * (1.0 + rate) ** (-t)).sum())

    try:
        return optimize.brentq(npv, -1.0 + 1e-9, 1e6, xtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")


def irr_of_cashflow_matrix(
    cashflows: np.ndarray,
    periods_per_year: int = 12,
    guess: Union[np.ndarray, float, None] = None,  # noqa: UP007
    xtol: float = 1e-10,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Annualized effective IRR for each column of a cash-flow matrix.

    Solves, per column, the periodic rate ``r`` with ``sum_t cf[t] / (1 + r) ** t = 0``
    (row index ``t`` = period number, row 0 = t0), then annualizes to
    ``(1 + r) ** periods_per_year - 1``.

    The Newton iteration runs over all columns simultaneously with the analytic
    derivative. Columns that do not converge but have a sign change fall back to a
    bracketing solver. Columns with no sign change (no real root) return NaN.

    Parameters
    ----------
    cashflows : np.ndarray
        Shape ``(n_periods, n_series)`` or ``(n_periods,)``. Each column is one series.
    periods_per_year : int, default 12
        Periods per year used for annualization (12 for monthly grids).
    guess : np.ndarray or float or None, default None
        Periodic-rate seed. If None, a ratio-based seed is used.
    xtol : float, default 1e-10
        Convergence tolerance on the Newton step (scale-free).
    max_iter : int, default 50
        Maximum Newton iterations.

    Returns
    -------
    np.ndarray
        Shape ``(n_series,)``. Annualized effective IRR per column; NaN where no root.
    """
    cf = np.asarray(cashflows, dtype=float)
    if cf.ndim == 1:
        cf = cf[:, None]
    n_periods, n_series = cf.shape
    t = np.arange(n_periods, dtype=float)[:, None]  # (n_periods, 1)

    if guess is None:
        rate = _irr_initial_guess(cf)
    elif np.isscalar(guess):
        rate = np.full(n_series, float(guess))
    else:
        rate = np.asarray(guess, dtype=float).copy()

    eps = 1e-12
    has_sign_change = (cf > 0.0).any(axis=0) & (cf < 0.0).any(axis=0)

    for _ in range(max_iter):
        base = np.where(1.0 + rate <= eps, eps, 1.0 + rate)  # (n_series,)
        disc = base[None, :] ** (-t)                         # (n_periods, n_series)
        f = (cf * disc).sum(axis=0)                          # (n_series,)
        fprime = -(t * cf * disc).sum(axis=0) / base         # (n_series,)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(fprime != 0.0, f / fprime, 0.0)
        rate = rate - step
        rate = np.where(rate <= -1.0 + eps, -1.0 + eps, rate)
        if np.nanmax(np.abs(step)) < xtol:
            break

    # Validate convergence (scale-free residual) and retry stragglers with brentq.
    base = np.where(1.0 + rate <= eps, eps, 1.0 + rate)
    residual = (cf * base[None, :] ** (-t)).sum(axis=0)
    scale = np.abs(cf).sum(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    not_converged = np.abs(residual) / scale > 1e-8
    for j in np.flatnonzero(not_converged & has_sign_change):
        rate[j] = _irr_brentq_column(cf[:, j])

    rate = np.where(has_sign_change, rate, np.nan)
    return (1.0 + rate) ** periods_per_year - 1.0
```

Note: `Union` is already imported at the top of `dcf_calculations.py` (`from typing import Union, ...`).

- [ ] **Step 5: Run the core tests to verify they pass**

Run: `poetry run pytest tests/portfolio/dcf/test_irr.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add okama/portfolios/dcf_calculations.py tests/portfolio/dcf/test_irr.py
git commit -m "feat(dcf): add vectorized irr_of_cashflow_matrix core"
```

---

## Task 2: `PortfolioDCF.irr` (nominal) + CAGR oracle + None guard

**Files:**
- Modify: `okama/portfolios/dcf.py`
- Test: `tests/portfolio/dcf/test_irr.py`

- [ ] **Step 1: Write the failing tests (CAGR oracle + None guard)**

Append to `tests/portfolio/dcf/test_irr.py`:

```python
@pytest.fixture()
def pf_no_inflation(synthetic_env):
    """Two-asset portfolio, monthly rebalancing, no inflation (mocked data)."""
    return ok.Portfolio(
        ["A.US", "B.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month")
    )


def test_irr_equals_cagr_without_intermediate_cashflows(pf_no_inflation):
    # MWRR with a single inflow and single outflow is identically the TWR/CAGR.
    ind = ok.IndexationStrategy(pf_no_inflation)
    ind.initial_investment = 10_000
    ind.frequency = "none"  # no regular cash flow -> only initial_investment and terminal value
    ind.indexation = DEFAULT_DISCOUNT_RATE
    pf_no_inflation.dcf.cashflow_parameters = ind

    expected_cagr = pf_no_inflation.get_cagr().iloc[-1].loc[pf_no_inflation.symbol]
    assert pf_no_inflation.dcf.irr() == pytest.approx(expected_cagr, abs=1e-9)


def test_irr_raises_when_cashflow_parameters_none(pf_no_inflation):
    pf_no_inflation.dcf.cashflow_parameters = None
    with pytest.raises(AttributeError, match=r"'cashflow_parameters' is not defined\."):
        pf_no_inflation.dcf.irr()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/portfolio/dcf/test_irr.py -q -k "cagr or none"`
Expected: FAIL with `AttributeError: 'PortfolioDCF' object has no attribute 'irr'`.

- [ ] **Step 3: Add the numpy import to `dcf.py`**

In `okama/portfolios/dcf.py`, add after `import pandas as pd`:

```python
import numpy as np
```

- [ ] **Step 4: Implement `PortfolioDCF.irr`**

Add this method to the `PortfolioDCF` class in `okama/portfolios/dcf.py` (place it right after `cash_flow_ts`, before `wealth_index_fv_with_assets`):

```python
    def irr(self) -> float:
        """
        Nominal internal rate of return (money-weighted return, MWRR) of the portfolio
        cash flow over the full historical period, honoring the configured cash flow strategy.

        The cash-flow vector (investor perspective) on a monthly grid ``t = 0 .. N``,
        where ``N = portfolio.ror.shape[0]``, is:

        - ``t = 0`` (one month before ``first_date``): ``-initial_investment``
        - ``t = 1 .. N``: ``-cash_flow_ts`` (withdrawals become inflows, contributions outflows)
        - ``t = N`` (``last_date``): additionally ``+`` the terminal wealth index value
          (liquidation; floored at 0, so the terminal withdrawal happens only if there is
          a positive balance left)

        The annualized effective IRR is returned. With no intermediate cash flows the
        result equals ``Portfolio.get_cagr`` for the full period.

        IRR is a rate, so there is no future/present-value variant: discounting the flows
        would yield the *real* rate (the ``get_cagr(real=True)`` axis), not "the IRR in PV".

        Returns
        -------
        float
            Annualized effective IRR. NaN if the cash flow has no sign change
            (no real root, e.g. pure accumulation with a zero terminal value).
            With contributions and withdrawals interleaved the equation may have several
            real roots; the root nearest the solver's seed is returned.

        Examples
        --------
        >>> pf = ok.Portfolio(["SPY.US", "AGG.US"], ccy="USD", last_date="2024-10")
        >>> ind = ok.IndexationStrategy(pf)
        >>> ind.initial_investment = 10_000
        >>> ind.frequency = "year"
        >>> ind.amount = -1_000
        >>> pf.dcf.cashflow_parameters = ind
        >>> pf.dcf.irr()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")

        cash_flow = self.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=True)
        terminal = self.wealth_index(discounting="fv", include_negative_values=False)[
            self.parent.symbol
        ].iloc[-1]
        initial_investment = self.cashflow_parameters.initial_investment
        n_months = self.parent.ror.shape[0]

        flows = np.empty(n_months + 1, dtype=float)
        flows[0] = -initial_investment
        flows[1:] = -cash_flow.reindex(self.parent.ror.index).fillna(0.0).to_numpy()
        flows[-1] += terminal

        return float(
            dcf_calculations.irr_of_cashflow_matrix(flows, periods_per_year=settings._MONTHS_PER_YEAR)[0]
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `poetry run pytest tests/portfolio/dcf/test_irr.py -q -k "cagr or none"`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add okama/portfolios/dcf.py tests/portfolio/dcf/test_irr.py
git commit -m "feat(dcf): add PortfolioDCF.irr with CAGR-equivalence guarantee"
```

---

## Task 3: Cross-check against a bracketing reference for real strategies

**Files:**
- Test: `tests/portfolio/dcf/test_irr.py`

- [ ] **Step 1: Write the failing reference cross-check tests**

Append to `tests/portfolio/dcf/test_irr.py`:

```python
def _reference_irr(dcf_obj):
    """Independent, slow reference: rebuild the vector and solve with scipy.brentq."""
    from scipy import optimize

    cash_flow = dcf_obj.cash_flow_ts("fv", remove_if_wealth_index_negative=True)
    terminal = dcf_obj.wealth_index("fv", include_negative_values=False)[dcf_obj.parent.symbol].iloc[-1]
    initial_investment = dcf_obj.cashflow_parameters.initial_investment
    n_months = dcf_obj.parent.ror.shape[0]

    v = np.empty(n_months + 1, dtype=float)
    v[0] = -initial_investment
    v[1:] = -cash_flow.reindex(dcf_obj.parent.ror.index).fillna(0.0).to_numpy()
    v[-1] += terminal

    t = np.arange(n_months + 1, dtype=float)

    def npv(rate):
        return float((v * (1.0 + rate) ** (-t)).sum())

    monthly = optimize.brentq(npv, -1.0 + 1e-9, 1e6, xtol=1e-12, maxiter=200)
    return (1.0 + monthly) ** _MONTHS_PER_YEAR - 1.0


def test_irr_matches_brentq_reference_indexation(pf_no_inflation):
    ind = ok.IndexationStrategy(pf_no_inflation)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -500
    ind.indexation = DEFAULT_DISCOUNT_RATE
    pf_no_inflation.dcf.cashflow_parameters = ind
    assert pf_no_inflation.dcf.irr() == pytest.approx(_reference_irr(pf_no_inflation.dcf), abs=1e-8)


def test_irr_matches_brentq_reference_percentage(pf_no_inflation):
    pc = ok.PercentageStrategy(pf_no_inflation)
    pc.initial_investment = 50_000
    pc.frequency = "half-year"
    pc.percentage = -0.05
    pf_no_inflation.dcf.cashflow_parameters = pc
    assert pf_no_inflation.dcf.irr() == pytest.approx(_reference_irr(pf_no_inflation.dcf), abs=1e-8)
```

- [ ] **Step 2: Run the tests**

Run: `poetry run pytest tests/portfolio/dcf/test_irr.py -q -k "brentq_reference"`
Expected: PASS — the vectorized core matches the independent bracketing solver to 1e-8 for both strategies.

- [ ] **Step 3: Commit**

```bash
git add tests/portfolio/dcf/test_irr.py
git commit -m "test(dcf): cross-check irr against brentq reference for strategies"
```

---

## Task 4: Full-suite verification and lint

**Files:** none (verification only).

- [ ] **Step 1: Run the full dcf test module**

Run: `poetry run pytest tests/portfolio/dcf/test_irr.py -q`
Expected: PASS (all IRR tests, ~9).

- [ ] **Step 2: Run the wider portfolio suite to catch regressions**

Run: `poetry run pytest tests/portfolio -q`
Expected: PASS, no new failures/warnings introduced by the change.

- [ ] **Step 3: Lint**

Run: `poetry run ruff check .`
Expected: clean. If anything is reported in the changed files, fix it (do not silence real findings). Re-run until clean.

- [ ] **Step 4: Final commit (only if Step 3 required fixes)**

```bash
git add -A
git commit -m "style(dcf): satisfy ruff for irr implementation"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- API `irr()` (nominal, method, no FV/PV) → Task 2. ✓
- Cash-flow vector definition (`-II` at t0, `-cf` at 1..N, `+terminal` at N) → Task 2 implementation + Task 3 reference mirrors it. ✓
- Monthly grid / annualization → core `periods_per_year=_MONTHS_PER_YEAR`. ✓
- CAGR invariant (primary regression test) → Task 2 Step 1. ✓
- Vectorized Newton + ratio seed + brentq fallback + NaN on no sign change → Task 1. ✓
- Edge cases: no sign change (Task 1), depleted/negative (Task 1), `cashflow_parameters is None` (Task 2), `initial_investment=0` covered structurally (`flows[0]=0`) and by the core's general solving — no special-casing exists. ✓
- Shared core for Monte Carlo → `irr_of_cashflow_matrix` accepts an `(N+1, M)` matrix (Task 1 vectorized-columns test proves multi-column). `monte_carlo_irr` itself is out of scope per spec. ✓

**Placeholder scan:** none — every code step contains full code.

**Type/name consistency:** `irr_of_cashflow_matrix`, `_irr_initial_guess`, `_irr_brentq_column` used consistently across Task 1, 2, 3. `irr()` signature (no args) identical in Task 2/3. ✓
