# `monte_carlo_irr` + seedable shared MC draw — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PortfolioDCF.monte_carlo_irr()` (distribution of money-weighted IRRs across Monte Carlo paths) on top of a Monte Carlo engine whose random draw is generated once, shared by all MC methods, and reproducible via an optional `seed`.

**Architecture:** Cache the MC return draw in `MonteCarlo` (Approach B) so `monte_carlo_wealth` / `monte_carlo_cash_flow` see one consistent scenario set; generate it through `np.random.default_rng(seed)`. `monte_carlo_irr` builds an `(M+1, mc_number)` investor cash-flow matrix from that single draw (reusing the tested `dcf_calculations` helpers) and solves it with the existing vectorized core `irr_of_cashflow_matrix`.

**Tech Stack:** Python 3.11+, numpy (`default_rng`), scipy (`stats.*.rvs(random_state=...)`, `optimize.brentq`), pandas, pytest. No new dependency.

**Conventions:** TDD (RED → GREEN), `poetry run pytest -q`, `poetry run ruff check .` clean. Tests live under `tests/portfolio/mc/` (a package-scoped autouse fixture in `tests/portfolio/mc/conftest.py` patches asset loading offline). Conventional commits.

---

## File Structure

- `okama/portfolios/mc.py` — **modify**: `MonteCarlo.__init__` gains `seed`; add `seed` property; cache the draw in `monte_carlo_returns_ts` + extract `_generate_returns_ts` (rng-based); extend `MonteCarlo._clear_cf_cache` to drop the cached draw.
- `okama/portfolios/dcf.py` — **modify**: `set_mc_parameters(..., seed=None)`; add `PortfolioDCF.monte_carlo_irr`.
- `tests/portfolio/mc/test_mc_seed.py` — **create**: seed/caching/invalidation tests.
- `tests/portfolio/mc/test_monte_carlo_irr.py` — **create**: IRR oracle, guard, shape, reproducibility, brentq cross-check.

The IRR solver core `dcf_calculations.irr_of_cashflow_matrix` is reused unchanged.

---

## Task 1: Seedable, cached, shared MC return draw

**Files:**
- Modify: `okama/portfolios/mc.py`
- Modify: `okama/portfolios/dcf.py` (`set_mc_parameters`)
- Test: `tests/portfolio/mc/test_mc_seed.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/portfolio/mc/test_mc_seed.py`:

```python
import numpy as np  # noqa: I001
import pandas as pd
import pytest

import okama as ok


@pytest.fixture()
def pf_mc():
    """Fresh single-asset portfolio per test (offline assets via package conftest)."""
    return ok.Portfolio(
        ["MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf_seed.PF",
    )


def test_monte_carlo_returns_ts_is_cached(pf_mc):
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=None)
    first = pf_mc.dcf.mc.monte_carlo_returns_ts
    second = pf_mc.dcf.mc.monte_carlo_returns_ts
    pd.testing.assert_frame_equal(first, second)


def test_monte_carlo_seed_is_reproducible(pf_mc):
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=42)
    first = pf_mc.dcf.mc.monte_carlo_returns_ts.copy()
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=42)
    second = pf_mc.dcf.mc.monte_carlo_returns_ts
    pd.testing.assert_frame_equal(first, second)


def test_monte_carlo_different_seed_changes_draw(pf_mc):
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=1)
    first = pf_mc.dcf.mc.monte_carlo_returns_ts.copy()
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=2)
    second = pf_mc.dcf.mc.monte_carlo_returns_ts
    assert not np.allclose(first.to_numpy(), second.to_numpy())


def test_monte_carlo_seed_invalid_type_raises(pf_mc):
    with pytest.raises((TypeError, ValueError)):
        pf_mc.dcf.mc.seed = "not-an-int"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/portfolio/mc/test_mc_seed.py -q`
Expected: FAIL — `set_mc_parameters()` has no `seed` parameter yet (`TypeError: ... unexpected keyword argument 'seed'`).

- [ ] **Step 3: Add `seed` + draw cache to `MonteCarlo.__init__`**

In `okama/portfolios/mc.py`, replace the constructor signature and body:

```python
    def __init__(
        self,
        parent: dcf.PortfolioDCF,
        distribution: str = "norm",
        distribution_parameters: Optional[tuple] = None,  # noqa: UP045
        period: int = 25,
        mc_number: int = 100,
    ):
        self.parent = parent
        self._distribution = distribution
        self._distribution_parameters = distribution_parameters
        self._period = period
        self._mc_number = mc_number
        self.ror = self.parent.parent.ror
```

with:

```python
    def __init__(
        self,
        parent: dcf.PortfolioDCF,
        distribution: str = "norm",
        distribution_parameters: Optional[tuple] = None,  # noqa: UP045
        period: int = 25,
        mc_number: int = 100,
        seed: Optional[int] = None,  # noqa: UP045
    ):
        self.parent = parent
        self._distribution = distribution
        self._distribution_parameters = distribution_parameters
        self._period = period
        self._mc_number = mc_number
        self._seed = seed
        self._returns_ts_cache: Optional[pd.DataFrame] = None  # noqa: UP045
        self.ror = self.parent.parent.ror
```

- [ ] **Step 4: Add the `seed` property and extend `_clear_cf_cache`**

In `okama/portfolios/mc.py`, replace the existing `_clear_cf_cache`:

```python
    def _clear_cf_cache(self):
        self.parent._monte_carlo_wealth_fv = pd.DataFrame(dtype=float)
        self.parent._monte_carlo_cash_flow_fv = pd.DataFrame(dtype=float)
```

with (add the `seed` property immediately before it, and drop the cached draw inside it):

```python
    @property
    def seed(self) -> Optional[int]:  # noqa: UP045
        """
        Random seed for Monte Carlo return generation.

        If `None`, each regeneration draws fresh randomness. Set an integer for
        reproducible scenarios. Changing the seed (or any other Monte Carlo
        parameter) invalidates the cached return draw.

        Returns
        -------
        int or None
            The configured random seed.
        """
        return self._seed

    @seed.setter
    def seed(self, seed):
        if seed is not None:
            validators.validate_integer("seed", seed)
        self._clear_cf_cache()
        self._seed = seed

    def _clear_cf_cache(self):
        self.parent._monte_carlo_wealth_fv = pd.DataFrame(dtype=float)
        self.parent._monte_carlo_cash_flow_fv = pd.DataFrame(dtype=float)
        self._returns_ts_cache = None
```

- [ ] **Step 5: Cache the draw and route generation through `default_rng(seed)`**

In `okama/portfolios/mc.py`, the current `monte_carlo_returns_ts` ends with this generation block (keep its long docstring):

```python
        period_months, ts_index = self._forecast_preparation()
        parameters = self.get_parameters_for_distribution()
        match self.distribution:
            case "norm":
                random_returns = np.random.normal(parameters[0], parameters[1], (period_months, self.mc_number))
            case "lognorm":
                random_returns = scipy.stats.lognorm(parameters[0], loc=parameters[1], scale=parameters[2]).rvs(
                    size=[period_months, self.mc_number]
                )
            case "t":
                random_returns = scipy.stats.t(df=parameters[0], loc=parameters[1], scale=parameters[2]).rvs(
                    size=[period_months, self.mc_number]
                )
            case _:
                raise ValueError("Unknown distribution type.")
        return pd.DataFrame(data=random_returns, index=ts_index)
```

Replace that block (the body after the docstring) with a cache check, and add a new `_generate_returns_ts` method right after the property:

```python
        if self._returns_ts_cache is None:
            self._returns_ts_cache = self._generate_returns_ts()
        return self._returns_ts_cache

    def _generate_returns_ts(self) -> pd.DataFrame:
        """Draw one Monte Carlo return matrix using ``np.random.default_rng(self.seed)``."""
        period_months, ts_index = self._forecast_preparation()
        parameters = self.get_parameters_for_distribution()
        rng = np.random.default_rng(self.seed)
        match self.distribution:
            case "norm":
                random_returns = rng.normal(parameters[0], parameters[1], (period_months, self.mc_number))
            case "lognorm":
                random_returns = scipy.stats.lognorm(parameters[0], loc=parameters[1], scale=parameters[2]).rvs(
                    size=[period_months, self.mc_number], random_state=rng
                )
            case "t":
                random_returns = scipy.stats.t(df=parameters[0], loc=parameters[1], scale=parameters[2]).rvs(
                    size=[period_months, self.mc_number], random_state=rng
                )
            case _:
                raise ValueError("Unknown distribution type.")
        return pd.DataFrame(data=random_returns, index=ts_index)
```

Also, in the `monte_carlo_returns_ts` docstring `Examples` block, change the line
`>>> pf.dcf.set_mc_parameters(period=8, mc_number=5000)` to
`>>> pf.dcf.set_mc_parameters(period=8, mc_number=5000, seed=0)` so the shown output is deterministic (the surrounding numeric output stays illustrative; doctests are not collected).

- [ ] **Step 6: Add `seed` to `set_mc_parameters`**

In `okama/portfolios/dcf.py`, the current method is:

```python
    def set_mc_parameters(
        self,
        distribution: str = "norm",
        distribution_parameters: Optional[tuple] = None,  # noqa: UP045
        period: int = 1,
        mc_number: int = 100,
    ):
```

Add a `seed` parameter to the signature:

```python
    def set_mc_parameters(
        self,
        distribution: str = "norm",
        distribution_parameters: Optional[tuple] = None,  # noqa: UP045
        period: int = 1,
        mc_number: int = 100,
        seed: Optional[int] = None,  # noqa: UP045
    ):
```

and at the end of the method body, after `self.mc.mc_number = mc_number`, add:

```python
        self.mc.seed = seed
```

Also add to the method's docstring `Parameters` section (after the `mc_number` entry):

```
        seed : int or None, default None
            Random seed for reproducible Monte Carlo draws. If None, each
            regeneration draws fresh randomness.
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `poetry run pytest tests/portfolio/mc/test_mc_seed.py -q`
Expected: PASS (4 passed).

- [ ] **Step 8: Lint**

Run: `poetry run ruff check okama/portfolios/mc.py okama/portfolios/dcf.py tests/portfolio/mc/test_mc_seed.py`
Expected: clean. Fix any reported issue in these files.

- [ ] **Step 9: Commit**

```bash
git add okama/portfolios/mc.py okama/portfolios/dcf.py tests/portfolio/mc/test_mc_seed.py
git commit -m "feat(mc): cache the Monte Carlo return draw and make it seedable"
```

---

## Task 2: `PortfolioDCF.monte_carlo_irr` + CAGR oracle

**Files:**
- Modify: `okama/portfolios/dcf.py`
- Test: `tests/portfolio/mc/test_monte_carlo_irr.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/portfolio/mc/test_monte_carlo_irr.py`:

```python
import numpy as np  # noqa: I001
import pandas as pd
import pytest

import okama as ok
from okama.common.helpers import helpers


@pytest.fixture()
def pf_mc_irr():
    """Fresh single-asset portfolio per test (offline assets via package conftest)."""
    return ok.Portfolio(
        ["MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf_mcirr.PF",
    )


def test_monte_carlo_irr_equals_cagr_distribution_without_cashflows(pf_mc_irr):
    # With no intermediate cash flows, each path's MWRR equals that path's CAGR.
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=50, seed=42)
    ind = ok.IndexationStrategy(pf_mc_irr)
    ind.initial_investment = 10_000
    ind.frequency = "none"
    pf_mc_irr.dcf.cashflow_parameters = ind

    irr_dist = pf_mc_irr.dcf.monte_carlo_irr()
    # Same cached draw -> deterministic oracle.
    expected = helpers.Frame.get_cagr(pf_mc_irr.dcf.mc.monte_carlo_returns_ts)
    pd.testing.assert_series_equal(irr_dist, expected, check_names=False, atol=1e-9, rtol=0)


def test_monte_carlo_irr_raises_when_cashflow_parameters_none(pf_mc_irr):
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=10, seed=1)
    pf_mc_irr.dcf.cashflow_parameters = None
    with pytest.raises(AttributeError, match=r"'cashflow_parameters' is not defined\."):
        pf_mc_irr.dcf.monte_carlo_irr()


def test_monte_carlo_irr_shape_and_name(pf_mc_irr):
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=25, seed=7)
    ind = ok.IndexationStrategy(pf_mc_irr)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -300
    ind.indexation = 0.0
    pf_mc_irr.dcf.cashflow_parameters = ind

    s = pf_mc_irr.dcf.monte_carlo_irr()
    assert isinstance(s, pd.Series)
    assert len(s) == 25
    assert s.name == "monte_carlo_irr"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/portfolio/mc/test_monte_carlo_irr.py -q`
Expected: FAIL with `AttributeError: 'PortfolioDCF' object has no attribute 'monte_carlo_irr'`.

- [ ] **Step 3: Implement `monte_carlo_irr`**

In `okama/portfolios/dcf.py`, add this method to the `PortfolioDCF` class immediately after `monte_carlo_survival_period`:

```python
    def monte_carlo_irr(self) -> pd.Series:
        """
        Distribution of money-weighted IRRs across Monte Carlo forecast paths.

        For each simulated path the nominal internal rate of return (MWRR) is computed
        over the forecast horizon, honoring the configured cash flow strategy. It is the
        forward-looking counterpart of :meth:`irr`.

        All paths share a single Monte Carlo return draw (see `MonteCarlo.seed` for
        reproducibility). The per-path investor cash-flow vector on a monthly grid
        ``t = 0 .. M`` (``M = mc.period * 12``) is:

        - ``t = 0`` (Monte Carlo start): ``-initial_investment``
        - ``t = 1 .. M``: ``-monte_carlo_cash_flow`` (withdrawals become inflows)
        - ``t = M``: additionally ``+`` the terminal wealth value (floored at 0)

        With no intermediate cash flows each path's IRR equals that path's CAGR.

        Returns
        -------
        Series
            One annualized effective IRR per Monte Carlo path (length ``mc_number``).
            NaN for paths whose cash flow has no sign change.

        Examples
        --------
        >>> pf = ok.Portfolio(["SPY.US", "AGG.US"], ccy="USD")
        >>> pf.dcf.set_mc_parameters(distribution="norm", period=20, mc_number=100, seed=0)
        >>> ind = ok.IndexationStrategy(pf)
        >>> ind.initial_investment = 10_000
        >>> ind.frequency = "year"
        >>> ind.amount = -500
        >>> pf.dcf.cashflow_parameters = ind
        >>> pf.dcf.monte_carlo_irr().quantile([0.1, 0.5, 0.9])
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        return_ts = self.mc.monte_carlo_returns_ts  # single shared (cached) draw
        cashflow_parameters = self.cashflow_parameters
        wealth = return_ts.apply(
            dcf_calculations.get_wealth_indexes_fv_with_cashflow,
            axis=0,
            args=(None, None, cashflow_parameters, "monte_carlo"),
        )
        wealth = wealth.apply(dcf_calculations.remove_negative_values, axis=0).fillna(0.0)
        cash_flow = return_ts.apply(
            dcf_calculations.get_cash_flow_fv,
            axis=0,
            args=(self.parent.symbol, cashflow_parameters, "monte_carlo"),
        )
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/portfolio/mc/test_monte_carlo_irr.py -q`
Expected: PASS (3 passed). If the CAGR oracle fails by a tiny amount, STOP and report actual vs expected — do NOT loosen the tolerance.

- [ ] **Step 5: Lint**

Run: `poetry run ruff check okama/portfolios/dcf.py tests/portfolio/mc/test_monte_carlo_irr.py`
Expected: clean. Fix any reported issue.

- [ ] **Step 6: Commit**

```bash
git add okama/portfolios/dcf.py tests/portfolio/mc/test_monte_carlo_irr.py
git commit -m "feat(dcf): add PortfolioDCF.monte_carlo_irr over Monte Carlo paths"
```

---

## Task 3: Reproducibility + brentq cross-check + full-suite verification

**Files:**
- Test: `tests/portfolio/mc/test_monte_carlo_irr.py`

- [ ] **Step 1: Write the cross-check and reproducibility tests**

Append to `tests/portfolio/mc/test_monte_carlo_irr.py`:

```python
def _reference_irr_column(flows_column):
    """Independent per-path reference: solve the NPV with scipy.brentq."""
    from scipy import optimize

    n = flows_column.shape[0]
    t = np.arange(n, dtype=float)
    lower = -1.0 + max(1e-9, 10.0 ** (-300.0 / max(n - 1, 1)))

    def npv(rate):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return float((flows_column * (1.0 + rate) ** (-t)).sum())

    try:
        monthly = optimize.brentq(npv, lower, 1e6, xtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")
    return (1.0 + monthly) ** 12 - 1.0


def test_monte_carlo_irr_matches_brentq_reference(pf_mc_irr):
    # Build the per-path reference from the (now shared-draw) public MC methods and an
    # independent brentq solve; it must agree with the vectorized monte_carlo_irr.
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=4, mc_number=30, seed=11)
    ind = ok.IndexationStrategy(pf_mc_irr)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -400
    ind.indexation = 0.0
    pf_mc_irr.dcf.cashflow_parameters = ind

    irr_dist = pf_mc_irr.dcf.monte_carlo_irr()

    wealth = pf_mc_irr.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=False)
    cash_flow = pf_mc_irr.dcf.monte_carlo_cash_flow(discounting="fv", remove_if_wealth_index_negative=True)
    terminal = wealth.iloc[-1]
    initial_investment = ind.initial_investment
    n_months, n_paths = cash_flow.shape

    for j in range(n_paths):
        v = np.empty(n_months + 1, dtype=float)
        v[0] = -initial_investment
        v[1:] = -cash_flow.iloc[:, j].to_numpy()
        v[-1] += terminal.iloc[j]
        ref = _reference_irr_column(v)
        got = irr_dist.iloc[j]
        if np.isnan(ref) or np.isnan(got):
            assert np.isnan(ref) and np.isnan(got)
        else:
            assert got == pytest.approx(ref, abs=1e-8)


def test_monte_carlo_irr_reproducible_with_seed(pf_mc_irr):
    ind = ok.IndexationStrategy(pf_mc_irr)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -400
    ind.indexation = 0.0
    pf_mc_irr.dcf.cashflow_parameters = ind

    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=4, mc_number=30, seed=123)
    first = pf_mc_irr.dcf.monte_carlo_irr()
    pf_mc_irr.dcf.set_mc_parameters(distribution="norm", period=4, mc_number=30, seed=123)
    second = pf_mc_irr.dcf.monte_carlo_irr()
    pd.testing.assert_series_equal(first, second)
```

- [ ] **Step 2: Run the new tests**

Run: `poetry run pytest tests/portfolio/mc/test_monte_carlo_irr.py -q -k "brentq_reference or reproducible"`
Expected: PASS (2 passed). A cross-check mismatch indicates a real assembly/solver discrepancy — STOP and report the path index, the `monte_carlo_irr` value, and the reference value; do not loosen the tolerance.

- [ ] **Step 3: Run the full MC suite to catch regressions from the draw-caching change**

Run: `poetry run pytest tests/portfolio/mc -q`
Expected: PASS. The shared/cached draw is a behavioral change. If a pre-existing test fails because it assumed `monte_carlo_returns_ts` returns a fresh draw on each access, update it to the new (correct) shared-draw semantics. If a test fails on exact numeric values that depended on the old `np.random.normal` stream, that test was relying on unseeded randomness — fix it by setting an explicit `seed` in that test, not by weakening assertions. If any failure is not clearly explained by these two causes, STOP and report it.

- [ ] **Step 4: Run the wider portfolio suite + lint**

Run: `poetry run pytest tests/portfolio -q`
Expected: PASS.

Run: `poetry run ruff check .`
Expected: clean. Fix any reported issue (no global disables, no bare `# noqa`).

- [ ] **Step 5: Commit**

```bash
git add tests/portfolio/mc/test_monte_carlo_irr.py
git commit -m "test(dcf): cross-check monte_carlo_irr vs brentq and lock seed reproducibility"
```

If Step 3 required changes to existing MC tests, commit those separately first with a message like `test(mc): adapt existing tests to shared cached MC draw`.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Cached, shared draw → Task 1 (Steps 3–5). ✓
- `seed: int | None = None`, config-level via `set_mc_parameters` + `mc.seed` → Task 1 (Steps 3, 4, 6). ✓
- Invalidation only on MC-config change (not cashflow change) → Task 1 Step 4 extends `MonteCarlo._clear_cf_cache` only; `CashFlow._clear_cf_cache` is untouched, so cashflow changes preserve the draw. ✓
- `monte_carlo_irr -> pd.Series`, nominal, None-guard, NaN on no-sign-change → Task 2 Step 3. ✓
- MC CAGR oracle → Task 2 Step 1. ✓
- Reproducibility + cross-check + full-suite regression check → Task 3. ✓
- Behavioral-change risk (fresh-draw-per-access; RNG stream) → Task 3 Step 3 handles it explicitly. ✓

**Placeholder scan:** none — every code step contains full code.

**Type/name consistency:** `_returns_ts_cache`, `_generate_returns_ts`, `seed`/`_seed`, `monte_carlo_irr`, `set_mc_parameters(..., seed=...)` used consistently across tasks. `monte_carlo_irr` reuses `irr_of_cashflow_matrix(... periods_per_year=settings._MONTHS_PER_YEAR)` exactly as the historical `irr`. ✓
