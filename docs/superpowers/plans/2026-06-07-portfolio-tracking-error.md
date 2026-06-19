# Ex-post Tracking Error for `Portfolio` (#61) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Portfolio.tracking_error(benchmark, rolling_window, method)` (ex-post TE against an explicit benchmark) and a `method="rms"|"std"` formula switch shared with `AssetList.tracking_error`.

**Architecture:** The math lives in `helpers.Index.tracking_error`, which gains a `method` parameter ("rms" = current uncentered RMS formula, default; "std" = centered sample std with Bessel's correction, Hwang & Satchell 2001 eq. 2). `AssetList.tracking_error` plumbs `method` through (expanding directly, rolling via `functools.partial`). `Portfolio.tracking_error` builds an internal `AssetList([benchmark, self], ccy=self.currency, inflation=False)` and delegates — currency conversion and date alignment come from `ListMaker` for free.

**Tech Stack:** Python 3.11+, pandas, numpy, pytest (offline `synthetic_env` fixtures), poetry, ruff.

**Spec:** `docs/superpowers/specs/2026-06-07-portfolio-tracking-error-design.md`

**Reference facts (verified in code):**

- `helpers.Index.tracking_error` — `okama/common/helpers/helpers.py:650-664`; called only from `okama/asset_list.py` (2 call sites: rolling at :1423, expanding at :1427).
- `ShortPeriodLengthError` is defined in `okama/common/error.py:1` and already imported in `helpers.py:13`.
- `ListMaker` accepts any object with `.symbol` and `.ror` as an asset (`okama/common/make_asset_list.py:161`), so `Asset` and `Portfolio` objects work as a benchmark; `ListMaker.currency` property returns the base currency string.
- `okama/portfolios/core.py` has `from __future__ import annotations` (modern union syntax OK) and imports `Optional, List, Union, Tuple, Literal` from typing.
- `okama/asset_list.py` imports `Optional, Union, Tuple` from typing (no `Literal`, no `functools`) and does NOT import portfolios → no import cycle.
- Tests are fully offline: global `synthetic_env` fixture (`tests/conftest.py:92`) provides 24 monthly periods (2020-01..2021-12) for `IDX.US`, `A.US`, `B.US` (all USD) and passes asset-like objects through. Portfolio fixtures (`pf_ab_monthly` etc.) live in `tests/portfolio/test_portfolio.py`.
- Sphinx docs use `autosummary_generate = True` with `docs/_templates/custom-class-template.rst`, which auto-lists all public methods; `docs/stubs/` is NOT tracked by git. **No manual docs edits are needed** — the new method appears in the generated docs automatically.

---

### Task 1: `method` parameter in `helpers.Index.tracking_error`

**Files:**
- Modify: `okama/common/helpers/helpers.py:650-664` (the `Index.tracking_error` static method)
- Test: `tests/test_helpers.py` (append at the end)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_helpers.py`:

```python
# --- Index.tracking_error tests ---


def _make_two_asset_ror(months: int = 24) -> pd.DataFrame:
    """Benchmark in the first column + a fund tracking it with noise."""
    rng = np.random.default_rng(42)
    idx = pd.period_range("2020-01", periods=months, freq="M")
    bench = pd.Series(rng.normal(0.01, 0.03, size=months), index=idx, name="BENCH.INDX")
    fund = pd.Series(bench.values + rng.normal(0.001, 0.01, size=months), index=idx, name="FUND.US")
    return pd.concat([bench, fund], axis=1)


def test_tracking_error_rms_default_matches_legacy_formula():
    """Default method and method='rms' produce the historical uncentered RMS values."""
    ror = _make_two_asset_ror()
    d = ror["FUND.US"] - ror["BENCH.INDX"]
    expected_last = np.sqrt((d**2).sum() / len(d)) * np.sqrt(12)
    result_default = helpers.Index.tracking_error(ror)
    result_rms = helpers.Index.tracking_error(ror, method="rms")
    pd.testing.assert_frame_equal(result_default, result_rms)
    assert result_default["FUND.US"].iloc[-1] == pytest.approx(expected_last)
    assert len(result_default) == len(ror)


def test_tracking_error_std_is_centered_with_bessel_correction():
    """method='std' is the centered sample std of differences (ddof=1), annualized."""
    ror = _make_two_asset_ror()
    d = ror["FUND.US"] - ror["BENCH.INDX"]
    expected_last = d.std(ddof=1) * np.sqrt(12)
    result = helpers.Index.tracking_error(ror, method="std")
    assert result["FUND.US"].iloc[-1] == pytest.approx(expected_last)
    # The first expanding point (std of a single observation) is dropped
    assert len(result) == len(ror) - 1


def test_tracking_error_invalid_method_raises_value_error():
    ror = _make_two_asset_ror()
    with pytest.raises(ValueError, match="method"):
        helpers.Index.tracking_error(ror, method="mad")


def test_tracking_error_short_period_raises_for_both_methods():
    from okama.common.error import ShortPeriodLengthError

    ror = _make_two_asset_ror(months=11)
    for m in ("rms", "std"):
        with pytest.raises(ShortPeriodLengthError):
            helpers.Index.tracking_error(ror, method=m)
```

(`numpy as np`, `pandas as pd`, `pytest`, and `helpers` are already imported at the top of `tests/test_helpers.py`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_helpers.py -q -k tracking_error`
Expected: all 5 new tests FAIL with `TypeError: tracking_error() got an unexpected keyword argument 'method'` (every test passes the `method` kwarg at least once) — the RED reason is the missing `method` parameter, not a typo or import error.

- [ ] **Step 3: Implement the `method` parameter**

In `okama/common/helpers/helpers.py` replace the whole `tracking_error` static method (lines 650-664) with:

```python
    @staticmethod
    def tracking_error(ror: pd.DataFrame, method: str = "rms") -> pd.DataFrame:
        """
        Return expanding tracking error time series for a rate of return time series.

        Assets are compared with the index or another benchmark.
        Index should be in the first position (first column).

        Tracking error is an ex-post measure: it is computed from the realized (historical)
        monthly return differences `d` between each asset and the benchmark.
        Two formulas are available:

        - "rms" (default): expanding root-mean-square of the differences, sqrt(mean(d²)).
          The differences are not centered around their mean, hence the systematic lag
          between an asset and the benchmark (tracking difference) is included in the result.
        - "std": expanding sample standard deviation of the differences with Bessel's
          correction, sqrt(sum((d - mean(d))²) / (n - 1)) — the classic tracking error
          definition (Hwang & Satchell, "Tracking Error: Ex-Ante versus Ex-Post Measures",
          2001, eq. 2) measuring the pure volatility of deviations. The first expanding
          point is dropped (a single observation has no standard deviation).

        The result is annualized for monthly time series (multiplied by sqrt(12)).
        """
        if ror.shape[1] < 2:
            raise ValueError("At least 2 symbols should be provided to calculate Tracking Error.")
        if ror.shape[0] < 12:
            raise ShortPeriodLengthError("Tracking Error is not defined for time periods < 1 year")
        difference = ror.subtract(ror.iloc[:, 0], axis=0)
        difference = difference.drop(difference.columns[0], axis=1)  # drop the first column (stock index data)
        if method == "rms":
            cumsum = difference.pow(2, axis=0).cumsum()
            tracking_error = cumsum.divide((1.0 + np.arange(ror.shape[0])), axis=0).pow(0.5, axis=0)
        elif method == "std":
            tracking_error = difference.expanding().std().dropna(how="all")
        else:
            raise ValueError(f"method must be 'rms' or 'std', got '{method}'.")
        return tracking_error * np.sqrt(12)
```

Note: the refactor moves the "subtract benchmark, drop first column" step before squaring — mathematically identical for "rms" (the benchmark column difference is identically zero), verified by the regression test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_helpers.py -q`
Expected: all PASS (including pre-existing helper tests).

- [ ] **Step 5: Run ruff**

Run: `poetry run ruff check okama/common/helpers/helpers.py tests/test_helpers.py`
Expected: no issues.

- [ ] **Step 6: Commit**

```bash
git add okama/common/helpers/helpers.py tests/test_helpers.py
git commit -m "feat(helpers): add method parameter (rms/std) to Index.tracking_error"
```

---

### Task 2: `method` parameter in `AssetList.tracking_error`

**Files:**
- Modify: `okama/asset_list.py:1` (typing import), `okama/asset_list.py:1386-1427` (the `tracking_error` method); add `from functools import partial` to the imports block
- Test: `tests/asset_list/test_asset_list.py` (append at the end)

- [ ] **Step 1: Write the failing tests**

Append to `tests/asset_list/test_asset_list.py`:

```python
def test_tracking_error_std_method(synthetic_env):
    """method='std' returns the centered sample std (ddof=1) of return differences, annualized."""
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    te = al.tracking_error(method="std")
    assert isinstance(te, pd.DataFrame)
    assert list(te.columns) == ["A.US", "B.US"]
    d = al.assets_ror["A.US"] - al.assets_ror["IDX.US"]
    assert te["A.US"].iloc[-1] == pytest.approx(d.std(ddof=1) * np.sqrt(12))
    # The first expanding point is dropped for the std method
    assert len(te) == len(al.assets_ror) - 1


def test_tracking_error_rms_default_unchanged(synthetic_env):
    """Calling without arguments equals method='rms' and reproduces the legacy formula."""
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    d = al.assets_ror["A.US"] - al.assets_ror["IDX.US"]
    expected_last = np.sqrt((d**2).sum() / len(d)) * np.sqrt(12)
    te_default = al.tracking_error()
    te_rms = al.tracking_error(method="rms")
    pd.testing.assert_frame_equal(te_default, te_rms)
    assert te_default["A.US"].iloc[-1] == pytest.approx(expected_last)


def test_tracking_error_rolling_with_std_method(synthetic_env):
    """Rolling tracking error supports method='std' (window >= 12 months)."""
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    te = al.tracking_error(rolling_window=12, method="std")
    assert isinstance(te, pd.DataFrame)
    assert list(te.columns) == ["A.US", "B.US"]
    assert len(te) > 0
    assert te.notna().all(axis=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/asset_list/test_asset_list.py -q -k tracking_error`
Expected: `test_tracking_error_std_method` and `test_tracking_error_rolling_with_std_method` FAIL with `TypeError: tracking_error() got an unexpected keyword argument 'method'`; `test_tracking_error_rms_default_unchanged` FAILS for the same reason (it also calls `method="rms"`).

- [ ] **Step 3: Implement the parameter**

In `okama/asset_list.py`:

1. Change the typing import (line 1) to:

```python
from typing import Optional, Union, Tuple, Literal  # noqa: I001, UP035
```

2. Add below it (before `import numpy as np`):

```python
from functools import partial
```

3. Replace the `tracking_error` method (lines 1386-1427) with:

```python
    def tracking_error(
        self,
        rolling_window: Optional[int] = None,  # noqa: UP045
        method: Literal["rms", "std"] = "rms",
    ) -> pd.DataFrame:
        """
        Calculate tracking error time series for the rate of return of assets.

        Tracking error is an ex-post measure of how closely the assets follow the benchmark.
        It is computed from the realized monthly return differences between each asset and
        the benchmark, and is annualized (multiplied by sqrt(12)). Tracking error is
        measured in percents.

        Benchmark should be in the first position of the symbols list in AssetList parameters.

        Two formulas are available (`method` parameter):

        - "rms" (default): root-mean-square of the return differences. The differences are
          not centered around their mean, hence the systematic lag between an asset and
          the benchmark (tracking difference) is included in the result.
        - "std": sample standard deviation of the return differences with Bessel's
          correction — the classic tracking error definition (Hwang & Satchell,
          "Tracking Error: Ex-Ante versus Ex-Post Measures", 2001, eq. 2) measuring
          the pure volatility of deviations from the benchmark. The first point of the
          expanding time series is dropped (a single observation has no standard deviation).

        Parameters
        ----------
        rolling_window : int or None, default None
            Size of the moving window in months. Must be at least 12 months.
            If None calculate expanding tracking error.
        method : {"rms", "std"}, default "rms"
            Tracking error formula: "rms" for the uncentered root-mean-square of return
            differences, "std" for the centered sample standard deviation.

        Returns
        -------
        DataFrame
            rolling or expanding tracking error time series for each asset.

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> x = ok.AssetList(["SP500TR.INDX", "SPY.US", "VOO.US"], last_date="2021-01")
        >>> x.tracking_error().plot()
        >>> plt.show()

        To calculate rolling tracking error set `rolling_window` to a number of months (moving window size):

        >>> x.tracking_error(rolling_window=12 * 5, method="std").plot()
        >>> plt.show()
        """
        if rolling_window:
            return helpers.Index.rolling_fn(
                df=self.assets_ror,
                window=rolling_window,
                fn=partial(helpers.Index.tracking_error, method=method),
                window_below_year=False,  # small windows below 12 months are not allowed
            )
        else:
            return helpers.Index.tracking_error(self.assets_ror, method=method)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/asset_list/ -q`
Expected: all PASS (new tests plus the pre-existing `test_tracking_and_index_metrics` regression).

- [ ] **Step 5: Run ruff**

Run: `poetry run ruff check okama/asset_list.py tests/asset_list/test_asset_list.py`
Expected: no issues.

- [ ] **Step 6: Commit**

```bash
git add okama/asset_list.py tests/asset_list/test_asset_list.py
git commit -m "feat: add method parameter (rms/std) to AssetList.tracking_error"
```

---

### Task 3: `Portfolio.tracking_error`

**Files:**
- Modify: `okama/portfolios/core.py:4` (typing import), insert the new method after `get_sortino_ratio` (after line 1646, before the `diversification_ratio` property)
- Test: `tests/portfolio/test_portfolio.py` (append at the end)

- [ ] **Step 1: Write the failing tests**

Append to `tests/portfolio/test_portfolio.py`:

```python
def test_tracking_error_matches_asset_list_workaround(pf_ab_monthly):
    """Portfolio.tracking_error equals the documented AssetList([benchmark, pf]) workaround."""
    te = pf_ab_monthly.tracking_error(benchmark="IDX.US")
    al = ok.AssetList(["IDX.US", pf_ab_monthly], ccy="USD", inflation=False)
    expected = al.tracking_error()[pf_ab_monthly.symbol]
    assert isinstance(te, pd.Series)
    assert te.name == pf_ab_monthly.symbol
    pd.testing.assert_series_equal(te, expected)


def test_tracking_error_std_matches_manual_computation(pf_ab_monthly, synthetic_env):
    """method='std' equals the centered std (ddof=1) of portfolio-vs-benchmark differences."""
    te = pf_ab_monthly.tracking_error(benchmark="IDX.US", method="std")
    diff = pf_ab_monthly.ror - synthetic_env["series"]["IDX.US"]
    assert te.iloc[-1] == pytest.approx(diff.std(ddof=1) * np.sqrt(12))
    # The first expanding point is dropped for the std method
    assert len(te) == len(diff) - 1


def test_tracking_error_rolling_matches_asset_list_workaround(pf_ab_monthly):
    te = pf_ab_monthly.tracking_error(benchmark="IDX.US", rolling_window=12)
    al = ok.AssetList(["IDX.US", pf_ab_monthly], ccy="USD", inflation=False)
    expected = al.tracking_error(rolling_window=12)[pf_ab_monthly.symbol]
    pd.testing.assert_series_equal(te, expected)


def test_tracking_error_with_asset_like_benchmark(pf_ab_monthly, synthetic_env):
    """Benchmark can be an asset-like object (anything with .symbol and .ror)."""
    from tests.helpers.factories import FakeAsset

    bench = FakeAsset("IDX.US", synthetic_env["series"]["IDX.US"], currency="USD")
    te_obj = pf_ab_monthly.tracking_error(benchmark=bench)
    te_str = pf_ab_monthly.tracking_error(benchmark="IDX.US")
    pd.testing.assert_series_equal(te_obj, te_str)


def test_tracking_error_with_portfolio_benchmark(pf_ab_monthly, synthetic_env):
    """Benchmark can be another Portfolio object."""
    bench_pf = ok.Portfolio(["IDX.US"], ccy="USD", inflation=False, symbol="bench.PF")
    te = pf_ab_monthly.tracking_error(benchmark=bench_pf)
    assert isinstance(te, pd.Series)
    assert te.name == pf_ab_monthly.symbol
    diff = pf_ab_monthly.ror - bench_pf.ror
    expected_last = np.sqrt((diff**2).sum() / len(diff)) * np.sqrt(12)
    assert te.iloc[-1] == pytest.approx(expected_last)


def test_tracking_error_invalid_method_raises(pf_ab_monthly):
    with pytest.raises(ValueError, match="method"):
        pf_ab_monthly.tracking_error(benchmark="IDX.US", method="mad")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/portfolio/test_portfolio.py -q -k tracking_error`
Expected: all 6 FAIL with `AttributeError: 'Portfolio' object has no attribute 'tracking_error'`.

- [ ] **Step 3: Implement the method**

In `okama/portfolios/core.py`:

1. Change the typing import (line 4) to:

```python
from typing import Optional, List, Union, Tuple, Literal, Type  # noqa: UP035
```

2. Insert after the `get_sortino_ratio` method (after line 1646, before `@property def diversification_ratio`):

```python
    def tracking_error(
        self,
        benchmark: str | Type,
        rolling_window: int | None = None,
        method: Literal["rms", "std"] = "rms",
    ) -> pd.Series:
        """
        Calculate ex-post tracking error time series of the portfolio against a benchmark.

        Tracking error is an ex-post (backward-looking) measure of how closely the portfolio
        follows the benchmark. It is computed from the realized monthly return differences
        between the portfolio and the benchmark, and is annualized (multiplied by sqrt(12)).
        Tracking error is measured in percents.

        Two formulas are available (`method` parameter):

        - "rms" (default): root-mean-square of the return differences. The differences are
          not centered around their mean, hence the systematic lag between the portfolio
          and the benchmark (tracking difference) is included in the result.
        - "std": sample standard deviation of the return differences with Bessel's
          correction — the classic tracking error definition (Hwang & Satchell,
          "Tracking Error: Ex-Ante versus Ex-Post Measures", 2001, eq. 2) measuring
          the pure volatility of deviations from the benchmark. The first point of the
          expanding time series is dropped (a single observation has no standard deviation).

        The benchmark rate of return is converted to the portfolio base currency, and the
        time period is limited to the intersection of the portfolio and benchmark
        available histories.

        Parameters
        ----------
        benchmark : str, Asset, Portfolio
            Benchmark ticker (e.g. "SP500TR.INDX") or an asset-like object
            (`Asset`, `Portfolio`).
        rolling_window : int or None, default None
            Size of the moving window in months. Must be at least 12 months.
            If None calculate expanding tracking error.
        method : {"rms", "std"}, default "rms"
            Tracking error formula: "rms" for the uncentered root-mean-square of return
            differences, "std" for the centered sample standard deviation.

        Returns
        -------
        Series
            Expanding or rolling annualized tracking error time series.
            The series is named after the portfolio symbol.

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> pf = ok.Portfolio(["SPY.US", "AGG.US"], weights=[0.60, 0.40], last_date="2024-01")
        >>> pf.tracking_error(benchmark="SP500TR.INDX").plot()
        >>> plt.show()

        To calculate rolling tracking error set `rolling_window` to a number of months (moving window size):

        >>> pf.tracking_error(benchmark="SP500TR.INDX", rolling_window=24, method="std").plot()
        >>> plt.show()
        """
        from okama.asset_list import AssetList  # local import to avoid a circular dependency

        al = AssetList([benchmark, self], ccy=self.currency, inflation=False)
        tracking_error = al.tracking_error(rolling_window=rolling_window, method=method)
        return tracking_error[self.symbol]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/portfolio/test_portfolio.py -q`
Expected: all PASS.

- [ ] **Step 5: Run ruff**

Run: `poetry run ruff check okama/portfolios/core.py tests/portfolio/test_portfolio.py`
Expected: no issues.

- [ ] **Step 6: Commit**

```bash
git add okama/portfolios/core.py tests/portfolio/test_portfolio.py
git commit -m "feat: add ex-post tracking_error method to Portfolio (#61)"
```

---

### Task 4: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `poetry run pytest -q`
Expected: all tests PASS, no warnings/errors in the output.

- [ ] **Step 2: Run ruff over the whole repo**

Run: `poetry run ruff check .`
Expected: no issues.

- [ ] **Step 3: Docs check (no action expected)**

The Sphinx docs use `autosummary_generate = True` with `docs/_templates/custom-class-template.rst`, which auto-lists every public method of `Portfolio`; `docs/stubs/` is generated at build time and not tracked by git. Confirm no manual `.rst` edits are required (nothing references method lists by hand). No commit for this step.
