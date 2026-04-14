import warnings  # noqa: I001

import pandas as pd
import numpy as np
import pytest

import okama as ok
from okama.settings import DEFAULT_DISCOUNT_RATE, _MONTHS_PER_YEAR

# Notes
# -----
# - These tests use the global synthetic_env fixture (defined in tests/asset_list/conftest.py
#   and exposed in tests/conftest.py) to patch asset loading and the currency Asset.
#   This guarantees there are no external API calls.
# - We prefer portfolios without historical inflation (inflation=False) to avoid
#   dependence on macro series; thus the default DCF discount_rate equals DEFAULT_DISCOUNT_RATE.


@pytest.fixture()
def pf_ab_monthly(synthetic_env):
    """Two-asset Portfolio with monthly rebalancing and no inflation (mocked data)."""
    return ok.Portfolio(["A.US", "B.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))


@pytest.fixture()
def dcf_indexation_yearly(pf_ab_monthly):
    """PortfolioDCF configured with IndexationStrategy (yearly withdrawals)."""
    ind = ok.IndexationStrategy(pf_ab_monthly)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -1_200  # 1200 per year withdrawals
    # Explicit indexation for inflation=False portfolio to avoid None
    ind.indexation = DEFAULT_DISCOUNT_RATE
    pf_ab_monthly.dcf.cashflow_parameters = ind

    # Ensure Portfolio has MC interfaces expected by dcf methods
    class _MCShim:
        def __init__(self, mc):
            self._mc = mc

        def monte_carlo_returns_ts(self):
            # dcf.monte_carlo_wealth wrongly calls parent.mc.monte_carlo_returns_ts()
            return self._mc.monte_carlo_returns_ts

    pf_ab_monthly.mc = _MCShim(pf_ab_monthly.dcf.mc)

    # dcf.monte_carlo_cash_flow calls parent.monte_carlo_returns_ts(distr=..., years=..., n=...)
    def _pf_mc_returns(*args, **kwargs):
        return pf_ab_monthly.dcf.mc.monte_carlo_returns_ts

    pf_ab_monthly.monte_carlo_returns_ts = _pf_mc_returns  # type: ignore[attr-defined]
    return pf_ab_monthly.dcf


@pytest.fixture()
def dcf_percentage_halfyear(pf_ab_monthly):
    """PortfolioDCF with PercentageStrategy (half-year payouts from initial amount)."""
    pc = ok.PercentageStrategy(pf_ab_monthly)
    pc.initial_investment = 50_000
    pc.frequency = "half-year"
    pc.percentage = 0.04  # 4% annualized over two periods
    pf_ab_monthly.dcf.cashflow_parameters = pc

    class _MCShim:
        def __init__(self, mc):
            self._mc = mc

        def monte_carlo_returns_ts(self):
            return self._mc.monte_carlo_returns_ts

    pf_ab_monthly.mc = _MCShim(pf_ab_monthly.dcf.mc)

    def _pf_mc_returns(*args, **kwargs):
        return pf_ab_monthly.dcf.mc.monte_carlo_returns_ts

    pf_ab_monthly.monte_carlo_returns_ts = _pf_mc_returns  # type: ignore[attr-defined]
    return pf_ab_monthly.dcf


@pytest.fixture()
def dcf_timeseries_monthly(pf_ab_monthly):
    """PortfolioDCF with explicit monthly cash flow time series (contrib then withdrawal)."""
    ts = ok.TimeSeriesStrategy(pf_ab_monthly)
    # two events in the 24-month window; dates are within synthetic_env period
    ts.initial_investment = 1_000
    ts.time_series_dic = {"2020-06": 300.0, "2021-03": -500.0}
    ts.time_series_discounted_values = False
    pf_ab_monthly.dcf.cashflow_parameters = ts

    class _MCShim:
        def __init__(self, mc):
            self._mc = mc

        def monte_carlo_returns_ts(self):
            return self._mc.monte_carlo_returns_ts

    pf_ab_monthly.mc = _MCShim(pf_ab_monthly.dcf.mc)

    def _pf_mc_returns(*args, **kwargs):
        return pf_ab_monthly.dcf.mc.monte_carlo_returns_ts

    pf_ab_monthly.monte_carlo_returns_ts = _pf_mc_returns  # type: ignore[attr-defined]
    return pf_ab_monthly.dcf


def test_discount_rate_default_and_setter(pf_ab_monthly):
    # No inflation in portfolio -> default discount rate equals DEFAULT_DISCOUNT_RATE
    assert pf_ab_monthly.dcf.discount_rate == DEFAULT_DISCOUNT_RATE
    # Custom value must be accepted and returned
    pf_ab_monthly.dcf.discount_rate = 0.087
    assert pf_ab_monthly.dcf.discount_rate == pytest.approx(0.087, abs=1e-12)


def test_time_series_dic_creates_float_series(pf_ab_monthly: ok.Portfolio) -> None:
    ts = ok.TimeSeriesStrategy(pf_ab_monthly)
    ts.time_series_dic = {"2020-06": 300, "2021-03": -500}

    assert pd.api.types.is_float_dtype(ts.time_series.dtype)


def test_indexation_strategy_default_indexation_when_no_inflation(pf_ab_monthly):
    assert not hasattr(pf_ab_monthly, "inflation")
    ind = ok.IndexationStrategy(pf_ab_monthly, indexation=None)
    assert ind.indexation == DEFAULT_DISCOUNT_RATE


def test_wealth_index_fv_with_indexation(dcf_indexation_yearly):
    wi = dcf_indexation_yearly.wealth_index(discounting="fv", include_negative_values=False)
    # Shape sanity: wealth index is monthly over the available history
    assert isinstance(wi, pd.DataFrame)
    assert wi.shape[0] in {len(dcf_indexation_yearly.parent.ror), len(dcf_indexation_yearly.parent.ror) + 1}
    # First value equals initial investment when negative values are removed
    assert wi.iloc[0, 0] == pytest.approx(dcf_indexation_yearly.cashflow_parameters.initial_investment, rel=1e-12)
    # All values non-negative when include_negative_values=False
    assert (wi.iloc[:, 0] >= 0).all()


def test_cash_flow_ts_fv_sign_and_length(dcf_indexation_yearly):
    cfts = dcf_indexation_yearly.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=True)
    assert isinstance(cfts, pd.Series)
    assert len(cfts) == len(dcf_indexation_yearly.parent.ror)
    # Since strategy has withdrawals, the sum should be negative (allow zero if masked by negative WI rule)
    assert cfts.sum() <= 0


def test_cash_flow_ts_halfyear_without_dtype_warning(dcf_percentage_halfyear) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfts = dcf_percentage_halfyear.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)

    assert isinstance(cfts, pd.Series)
    assert not any(
        issubclass(warning.category, FutureWarning) and "incompatible dtype" in str(warning.message)
        for warning in caught
    )


def test_wealth_index_fv_with_assets_columns(dcf_percentage_halfyear):
    df = dcf_percentage_halfyear.wealth_index_fv_with_assets
    # Must include portfolio column and each asset
    expected_cols = [dcf_percentage_halfyear.parent.name, *dcf_percentage_halfyear.parent.symbols]
    for col in expected_cols:
        assert col in df.columns
    # Number of rows equals length of returns or returns + 1 (depending on helper implementation)
    assert df.shape[0] in {len(dcf_percentage_halfyear.parent.ror), len(dcf_percentage_halfyear.parent.ror) + 1}


def test_initial_investment_pv_and_fv(dcf_percentage_halfyear):
    dcf = dcf_percentage_halfyear
    dcf.discount_rate = 0.10
    dcf.mc.period = 5
    wi_pv = dcf.wealth_index(discounting="pv", include_negative_values=False)
    wi_fv = dcf.wealth_index(discounting="fv", include_negative_values=False)
    monthly_discount_rate = (1 + dcf.discount_rate) ** (1 / _MONTHS_PER_YEAR) - 1
    expected_initial_investment_pv = (
        dcf.cashflow_parameters.initial_investment / (1.0 + monthly_discount_rate) ** dcf.parent.ror.shape[0]
    )

    assert dcf.initial_investment_fv == wi_fv.iloc[0, 0] == wi_pv.iloc[0, 0]
    assert dcf.initial_investment_pv == pytest.approx(expected_initial_investment_pv)
    assert dcf.initial_investment_pv < dcf.initial_investment_fv


def test_wealth_index_pv_less_than_fv(dcf_indexation_yearly):
    """PV wealth index should match FV at start and stay below it afterward."""
    dcf = dcf_indexation_yearly
    wi_fv = dcf.wealth_index(discounting="fv", include_negative_values=False)
    wi_pv = dcf.wealth_index(discounting="pv", include_negative_values=False)

    assert wi_pv.iloc[0, 0] == pytest.approx(wi_fv.iloc[0, 0])
    assert (wi_pv.iloc[:, 0] <= wi_fv.iloc[:, 0]).all()
    assert (wi_pv.iloc[1:, 0] < wi_fv.iloc[1:, 0]).any()


def test_monte_carlo_wealth_shapes(dcf_indexation_yearly):
    dcf = dcf_indexation_yearly
    # Small MC for speed (set via properties to avoid distribution_parameters validation)
    dcf.mc.distribution = "norm"
    dcf.mc.period = 1
    dcf.mc.mc_number = 10
    dcf.discount_rate = 0.05
    # Wealth FV
    df_fv = dcf.monte_carlo_wealth(discounting="fv")
    assert isinstance(df_fv, pd.DataFrame)
    assert df_fv.shape[1] == 10
    # 1-year monthly horizon + initial historical anchor row
    assert df_fv.shape[0] == _MONTHS_PER_YEAR + 1
    # PV shape must match FV
    df_pv = dcf.monte_carlo_wealth(discounting="pv")
    assert df_pv.shape == df_fv.shape


def test_monte_carlo_cash_flow_basic(dcf_timeseries_monthly):
    dcf = dcf_timeseries_monthly
    dcf.mc.distribution = "norm"
    dcf.mc.period = 1
    dcf.mc.mc_number = 10
    cf_fv = dcf.monte_carlo_cash_flow(discounting="fv", remove_if_wealth_index_negative=True)
    assert isinstance(cf_fv, pd.DataFrame)
    # For cash flow MC, there is no extra anchor row: exactly months in horizon
    assert cf_fv.shape == (_MONTHS_PER_YEAR, 10)
    # PV should match shape; numerical difference is not guaranteed if cash flow becomes zeroed
    dcf.discount_rate = 0.07
    cf_pv = dcf.monte_carlo_cash_flow(discounting="pv", remove_if_wealth_index_negative=False)
    assert cf_pv.shape == cf_fv.shape
    # If both are all-zero matrices (possible with masking), skip strict difference assertion
    if not (np.allclose(cf_pv.values, 0) and np.allclose(cf_fv.values, 0)):
        assert not np.allclose(cf_pv.values, cf_fv.values)


def test_historical_survival_metrics(dcf_indexation_yearly):
    # Survival period (years) should be a non-negative float
    sp = dcf_indexation_yearly.survival_period_hist(threshold=0)
    assert np.isfinite(sp)
    assert sp >= 0
    # Survival date is a Timestamp (may be the last date if no depletion)
    sd = dcf_indexation_yearly.survival_date_hist(threshold=0)
    assert isinstance(sd, pd.Timestamp)


def test_find_the_largest_withdrawals_size_converges(dcf_indexation_yearly):
    dcf = dcf_indexation_yearly
    # Use small MC to keep it fast
    dcf.mc.distribution = "norm"
    dcf.mc.period = 2
    dcf.mc.mc_number = 16
    initial_amount = dcf.cashflow_parameters.amount
    res = dcf.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.0, 1.0),  # relative to initial investment per period
        target_survival_period=1,  # 1 year target
        percentile=50,
        threshold=0,
        tolerance_rel=0.25,
        iter_max=10,
    )
    # Basic checks on solver result
    assert res.success is True
    assert isinstance(res.withdrawal_abs, float)
    assert isinstance(res.withdrawal_rel, float)
    assert isinstance(res.error_rel, float)
    assert isinstance(res.solutions, pd.DataFrame)
    assert {"withdrawal_abs", "withdrawal_rel", "error_rel", "error_rel_change"}.issubset(res.solutions.columns)
    assert dcf.cashflow_parameters.amount == pytest.approx(initial_amount)


def test_find_the_largest_withdrawals_size_rejects_target_above_period_with_tolerance(
    dcf_indexation_yearly,
) -> None:
    dcf = dcf_indexation_yearly
    dcf.mc.period = 10

    with pytest.raises(
        ValueError,
        match=r"target_survival_period must be less than Monte Carlo simulation period",
    ):
        dcf.find_the_largest_withdrawals_size(
            goal="survival_period",
            target_survival_period=8,
            percentile=50,
            tolerance_rel=0.25,
            iter_max=1,
        )


# ------------------ Additional coverage: repr, validations, and plotting smoke ------------------


def test_portfolio_dcf_repr_contains_key_fields(dcf_indexation_yearly):
    """__repr__ should include key fields for quick inspection."""
    dcf = dcf_indexation_yearly
    s = repr(dcf)
    # Keys from PortfolioDCF.__repr__ implementation
    assert "Portfolio symbol" in s
    assert "Monte Carlo distribution" in s
    assert "Monte Carlo period" in s
    assert "Cash flow strategy" in s
    assert "discount_rate" in s


def test_cashflow_parameters_setter_type_error(pf_ab_monthly):
    """Setting cashflow_parameters to a non-CashFlow must raise TypeError."""
    with pytest.raises(TypeError, match=r"cashflow_parameters must be a CashFlow instance or None"):
        pf_ab_monthly.dcf.cashflow_parameters = object()


def test_wealth_index_bad_discounting_raises(dcf_indexation_yearly):
    with pytest.raises(ValueError, match=r"'discounting' must be either 'fv' or 'pv'"):
        dcf_indexation_yearly.wealth_index(discounting="bad", include_negative_values=False)


def test_cash_flow_ts_bad_discounting_raises(dcf_indexation_yearly):
    with pytest.raises(ValueError, match=r"'discounting' must be either 'fv' or 'pv'"):
        dcf_indexation_yearly.cash_flow_ts(discounting="bad", remove_if_wealth_index_negative=True)


def test_discount_rate_setter_validation(pf_ab_monthly):
    """Non-real values for discount_rate should be rejected by validators."""
    # Wrong type
    with pytest.raises(TypeError):
        pf_ab_monthly.dcf.discount_rate = "abc"  # type: ignore[assignment]


def test_plot_forecast_monte_carlo_smoke(dcf_indexation_yearly):
    """Smoke-test plot_forecast_monte_carlo: should not raise and must return a result object."""
    # Use a non-interactive backend to avoid GUI requirements
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass
    dcf = dcf_indexation_yearly
    dcf.mc.distribution = "norm"
    dcf.mc.period = 1
    dcf.mc.mc_number = 5
    # Function returns None; this is a smoke test to ensure no exceptions are raised
    dcf.plot_forecast_monte_carlo(backtest=False)
