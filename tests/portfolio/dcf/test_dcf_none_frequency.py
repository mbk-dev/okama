import pandas as pd  # noqa: I001
import warnings
import numpy as np
import pytest
import okama as ok
from okama.common.helpers import helpers


@pytest.fixture()
def pf_monthly(synthetic_env):
    """Portfolio with monthly rebalancing and no inflation (mocked data)."""
    return ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))


def test_cashflow_base_none_frequency_no_ts(pf_monthly):
    """Test CashFlow with frequency='none' and no time_series_dic."""
    cf = ok.CashFlow(pf_monthly, frequency="none", initial_investment=1000.0)
    pf_monthly.dcf.cashflow_parameters = cf

    # Wealth index should be just compounding returns
    wi = pf_monthly.dcf.wealth_index(discounting="fv", include_negative_values=True)
    expected_wi = helpers.Frame.get_wealth_indexes(pf_monthly.ror, initial_amount=1000.0)
    # Adjust expected_wi name/format if necessary
    assert np.allclose(wi[pf_monthly.name].values, expected_wi.values)

    # Cash flow should be all zeros
    cfts = pf_monthly.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)
    assert (cfts == 0).all()


def test_cashflow_base_none_frequency_with_ts(pf_monthly):
    """Test CashFlow with frequency='none' AND time_series_dic."""
    mid_date = pf_monthly.ror.index[10]
    ts_dic = {str(mid_date): 500.0}
    cf = ok.CashFlow(pf_monthly, frequency="none", initial_investment=1000.0, time_series_dic=ts_dic)
    pf_monthly.dcf.cashflow_parameters = cf

    cfts = pf_monthly.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)
    # print(f"CFTS: {cfts}")
    # print(f"Mid date: {mid_date}, type: {type(mid_date)}")
    # print(f"Value at mid_date: {cfts[mid_date]}")
    assert cfts[mid_date] == 500.0
    assert cfts.sum() == 500.0

    wi = pf_monthly.dcf.wealth_index(discounting="fv", include_negative_values=True)
    # Check that wealth increased by 500 (plus return) at that point relative to base

    cf_base = ok.CashFlow(pf_monthly, frequency="none", initial_investment=1000.0)
    pf_monthly.dcf.cashflow_parameters = cf_base
    wi_base = pf_monthly.dcf.wealth_index(discounting="fv", include_negative_values=True)

    assert wi.iloc[-1, 0] > wi_base.iloc[-1, 0]


def test_indexation_strategy_none_frequency(pf_monthly):
    """Test IndexationStrategy with frequency='none'."""
    # amount is set but should be IGNORED because frequency is none
    cf = ok.IndexationStrategy(pf_monthly, frequency="none", initial_investment=1000.0, amount=-100.0, indexation=0.0)
    pf_monthly.dcf.cashflow_parameters = cf

    cfts = pf_monthly.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)
    # Should be all zeros because regular withdrawal is ignored
    assert (cfts == 0).all()

    # Now with time_series_dic
    mid_date = pf_monthly.ror.index[10]
    cf.time_series_dic = {str(mid_date): -500.0}

    cfts = pf_monthly.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)
    assert cfts[mid_date] == -500.0
    assert cfts.sum() == -500.0  # Only the extra withdrawal


def test_percentage_strategy_none_frequency(pf_monthly):
    """Test PercentageStrategy with frequency='none'."""
    # percentage is set but should be IGNORED
    cf = ok.PercentageStrategy(pf_monthly, frequency="none", initial_investment=1000.0, percentage=0.1)
    pf_monthly.dcf.cashflow_parameters = cf

    cfts = pf_monthly.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)
    assert (cfts == 0).all()

    # With time_series_dic
    mid_date = pf_monthly.ror.index[5]
    cf.time_series_dic = {str(mid_date): 200.0}

    cfts = pf_monthly.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)
    assert cfts[mid_date] == 200.0


def test_monte_carlo_none_frequency(pf_monthly):
    """Test Monte Carlo simulation works with frequency='none'."""
    cf = ok.CashFlow(pf_monthly, frequency="none", initial_investment=1000.0)
    pf_monthly.dcf.cashflow_parameters = cf
    pf_monthly.dcf.mc.period = 1
    pf_monthly.dcf.mc.mc_number = 10

    # Just check it doesn't crash and returns correct shape
    mc_wealth = pf_monthly.dcf.monte_carlo_wealth(discounting="fv")
    assert isinstance(mc_wealth, pd.DataFrame)

    mc_cf = pf_monthly.dcf.monte_carlo_cash_flow(discounting="fv")
    # Should be all zeros
    assert (mc_cf.values == 0).all()


@pytest.mark.parametrize("frequency", ["none", "month", "quarter", "year"])
@pytest.mark.parametrize("kind", ["wealth", "cashflow"])
def test_scalar_extra_cashflow_preserves_periods_and_values_without_pandas4_warning(pf_monthly, frequency, kind):
    from okama.portfolios import dcf_calculations as calc

    ror = pf_monthly.ror
    deposit_date = ror.index[4]
    if frequency == "none":
        parameters = ok.CashFlow(
            pf_monthly, frequency="none", initial_investment=1000.0, time_series_dic={str(deposit_date): 500.0}
        )
    else:
        parameters = ok.IndexationStrategy(
            pf_monthly,
            frequency=frequency,
            initial_investment=1000.0,
            amount=0.0,
            indexation=0.0,
            time_series_dic={str(deposit_date): 500.0},
        )
    pf_monthly.dcf.cashflow_parameters = parameters
    with warnings.catch_warnings():
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("error", pd.errors.Pandas4Warning)
        if kind == "wealth":
            result = calc._get_wealth_indexes_fv_with_cashflow_scalar(
                ror, pf_monthly.name, None, parameters, "backtest"
            )
        else:
            result = calc._get_cash_flow_fv_scalar(ror, pf_monthly.name, parameters, "backtest")
    assert isinstance(result.index, pd.PeriodIndex)
    assert result.index.name is None
    if kind == "cashflow":
        assert result.index.equals(ror.index.rename(None))
        assert result.loc[deposit_date] == pytest.approx(500.0)
        assert result.sum() == pytest.approx(500.0)
    else:
        assert result.index.tolist() == [ror.index[0] - 1, *ror.index]
        balance = 1000.0
        expected = [balance]
        for date, value in ror.items():
            balance = balance * (1 + value) + (500.0 if date == deposit_date else 0.0)
            expected.append(balance)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-12, atol=1e-12)
