import numpy as np
import pandas as pd
import pytest
from pytest import approx
from pytest import mark
from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_series_equal, assert_frame_equal

import okama as ok

from tests import conftest


def test_initialization_failing():
    with pytest.raises(
        ValueError,
        match=r"Number of tickers \(2\) should be equal to the weights number \(3\)",
    ):
        ok.Portfolio(assets=["MCFTR.INDX", "MCFTR.INDX", "RUB.FX"], weights=[0.3, 0.3, 0.4])


def test_repr(portfolio_rebalanced_year):
    value = pd.Series(
        dict(
            symbol="pf1.PF",
            assets="[RGBITR.INDX, MCFTR.INDX]",
            weights="[0.5, 0.5]",
            rebalancing_period="year",
            currency="RUB",
            inflation="RUB.INFL",
            first_date="2015-01",
            last_date="2020-01",
            period_length="5 years, 1 months",
        )
    )
    assert repr(portfolio_rebalanced_year) == repr(value)


def test_symbol_failing(portfolio_rebalanced_year):
    with pytest.raises(
        ValueError,
        match='portfolio symbol must be a string ending with ".PF" namespace.',
    ):
        portfolio_rebalanced_year.symbol = 1
    with pytest.raises(ValueError, match='portfolio symbol must be a string ending with ".PF" namespace.'):
        portfolio_rebalanced_year.symbol = "Not_a_good_symbol_for_portfolio.US"
    with pytest.raises(ValueError, match="portfolio text symbol should not have whitespace characters."):
        portfolio_rebalanced_year.symbol = "Not a good symbol for portfolio.PF"


def test_symbol_setter(portfolio_rebalanced_year):
    portfolio_rebalanced_year.symbol = "portfolio_1.PF"
    assert portfolio_rebalanced_year.symbol == "portfolio_1.PF"


def test_ror_rebalance(portfolio_rebalanced_year, portfolio_not_rebalanced):
    assert portfolio_rebalanced_year.ror[-2] == approx(0.03052, rel=1e-1)
    assert portfolio_not_rebalanced.ror[-1] == approx(0.01167, rel=1e-1)


def test_ror(portfolio_rebalanced_month):
    portfolio_sample = pd.read_pickle(conftest.data_folder / "portfolio.pkl")
    actual = portfolio_rebalanced_month.ror
    assert_series_equal(actual, portfolio_sample, atol=1e-01)


def test_wealth_index(portfolio_rebalanced_year):
    assert portfolio_rebalanced_year.wealth_index.iloc[-1, 1] == approx(1310.60, rel=1e-2)


def test_wealth_index_with_assets(portfolio_rebalanced_year, portfolio_no_inflation):
    result = portfolio_rebalanced_year.wealth_index_with_assets.iloc[-1, :].values
    assert_allclose(np.array(result), np.array([2259.244689, 2056.11199, 2889.930097, 1310.606208]), rtol=1e-02)


def test_weights(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.weights == [0.5, 0.5]


def test_weights_ts_rebalanced_month(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.weights_ts["RGBITR.INDX"].iloc[-1] == approx(0.5, rel=1e-2)


def test_weights_ts_rebalanced_year(portfolio_rebalanced_year, portfolio_not_rebalanced):
    assert portfolio_rebalanced_year.weights_ts["RGBITR.INDX"].iloc[-2] == approx(0.4645, rel=1e-2)


def test_weights_ts_not_rebalanced(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.weights_ts["RGBITR.INDX"].iloc[-1] == approx(0.4156, rel=1e-2)


def test_mean_return(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.mean_return_monthly == approx(0.01536, rel=1e-2)
    assert portfolio_rebalanced_month.mean_return_annual == approx(0.20080, rel=1e-2)


def test_real_mean_return(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.real_mean_return == approx(0.13746, rel=1e-2)


@mark.parametrize(
    "window, real, expected",
    [(1, True, 0.01100), (12, False, 0.24604), (12, True, 0.2165)],
)
def test_get_rolling_cumulative_return(portfolio_rebalanced_month, window, real, expected):
    assert portfolio_rebalanced_month.get_rolling_cumulative_return(window=window, real=real).iloc[-1, 0] == approx(
        expected, abs=1e-1
    )


def test_assets_close_monthly(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.assets_close_monthly.iloc[-1, 0] == approx(578.19, rel=1e-2)  # RGBITR.INDX
    assert portfolio_not_rebalanced.assets_close_monthly.iloc[-1, 1] == 5245.6  # MCFTR.INDX


def test_close_monthly(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.close_monthly.iloc[-1] == approx(2269.20, rel=1e-2)


def test_get_assets_dividends(portfolio_dividends):
    assert portfolio_dividends._get_assets_dividends().iloc[-1, 0] == approx(0, abs=1e-2)
    # T.US 2020-01=$0.3927 , RUBUSD=63.03 (  http://joxi.ru/823dnYWizBvEOA  )
    # T.US 2020-01=$0.5200 , RUBUSD=63.03 ( http://joxi.ru/Grqjdaliz5Ow9m )  04.09.2022
    # T.US 2020-01-09, 0.5200 from EOD
    assert portfolio_dividends._get_assets_dividends().iloc[-1, 1] == approx(32.77, rel=1e-2)
    assert portfolio_dividends._get_assets_dividends().iloc[-1, 2] == approx(0, rel=1e-2)


def test_number_of_securities(portfolio_not_rebalanced, portfolio_dividends):
    assert portfolio_not_rebalanced.number_of_securities.iloc[-1, 0] == approx(1.6312, rel=1e-2)  # RGBITR.INDX
    assert portfolio_not_rebalanced.number_of_securities.iloc[-1, 1] == approx(0.2527, abs=1e-2)  # MCFTR.INDX
    # with dividends
    assert portfolio_dividends.number_of_securities.iloc[-1, 0] == approx(3.63, rel=1e-2)  # SBER.MOEX
    assert portfolio_dividends.number_of_securities.iloc[-1, 1] == approx(0.3892, abs=1e-2)  # T.US
    assert portfolio_dividends.number_of_securities.iloc[-1, 2] == approx(0.004137, abs=1e-2)  # GNS.LSE


def test_dividends(portfolio_dividends):
    assert portfolio_dividends.dividends.iloc[-1] == approx(12.75, rel=1e-2)


def test_dividend_yield(portfolio_dividends):
    assert portfolio_dividends.dividend_yield.iloc[-1] == approx(0.0396, abs=1e-2)


def test_dividends_annual(portfolio_dividends):
    assert portfolio_dividends.dividends_annual.iloc[-1].sum() == approx(32.778668, rel=1e-3)


def test_dividend_yield_annual(portfolio_dividends):
    assert portfolio_dividends.dividend_yield_annual.iloc[0, 0] == approx(0.004444, abs=1e-3)


def test_risk(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.risk_monthly.iloc[-1] == approx(0.02233, rel=1e-1)
    assert portfolio_rebalanced_month.risk_annual.iloc[-1] == approx(0.091634, rel=1e-1)


def test_semideviation(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.semideviation_monthly == approx(0.02080, abs=1e-2)
    assert portfolio_rebalanced_month.semideviation_annual == approx(0.04534, abs=1e-2)


def test_get_var_historic(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.get_var_historic(time_frame=1, level=5) == approx(0.01500, abs=1e-2)
    assert portfolio_rebalanced_month.get_var_historic(time_frame=5, level=1) == approx(0.0491, abs=1e-2)


def test_get_cvar_historic(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.get_cvar_historic(time_frame=1, level=5) == approx(0.02577, abs=1e-2)
    assert portfolio_rebalanced_month.get_cvar_historic(time_frame=5, level=1) == approx(0.04918, abs=1e-2)


def test_drawdowns(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.drawdowns.min() == approx(-0.0560, rel=1e-2)


def test_recovery_period(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.recovery_period.max() == 6


def test_get_cagr(portfolio_rebalanced_month, portfolio_no_inflation):
    values1 = pd.Series({"pf1.PF": 0.1974, "RUB.INFL": 0.055480})
    actual1 = portfolio_rebalanced_month.get_cagr()
    assert_series_equal(actual1, values1, atol=1e-2)
    # no inflation
    values2 = pd.Series({"pf1.PF": 0.1974})
    actual2 = portfolio_no_inflation.get_cagr()
    assert_series_equal(actual2, values2, atol=1e-2)
    # failing if wrong period
    with pytest.raises(TypeError):
        portfolio_rebalanced_month.get_cagr(period="one year")


cagr_testdata1 = [
    (1, 0.21655),
    (None, 0.13446),
]


@mark.parametrize(
    "input_data, expected",
    cagr_testdata1,
    ids=["1 year", "full period"],
)
def test_get_cagr_real(portfolio_rebalanced_month, input_data, expected):
    assert portfolio_rebalanced_month.get_cagr(period=input_data, real=True).values[0] == approx(expected, abs=1e-2)


def test_get_cagr_real_no_inflation_exception(portfolio_no_inflation):
    with pytest.raises(ValueError):
        portfolio_no_inflation.get_cagr(period=1, real=True)


@mark.parametrize(
    "period, real, expected",
    [("YTD", False, 0.01505), (1, False, 0.24604), (2, True, 0.2742)],
    ids=["YTD - nominal", "1 year - nominal", "2 years - real"],
)
def test_cumulative_return(portfolio_rebalanced_month, period, real, expected):
    assert portfolio_rebalanced_month.get_cumulative_return(period=period, real=real).iloc[0] == approx(
        expected, abs=1e-2
    )


cumulative_return_fail = [
    (1.5, False, TypeError),
    (-1, False, ValueError),
    (1, True, ValueError),
]


@mark.parametrize("period, real, exception", cumulative_return_fail)
def test_cumulative_return_error(portfolio_no_inflation, period, real, exception):
    with pytest.raises(exception):
        portfolio_no_inflation.get_cumulative_return(period=period, real=real)


@mark.xfail
def test_describe_inflation(portfolio_rebalanced_month):
    description = portfolio_rebalanced_month.describe()
    description_sample = pd.read_pickle(conftest.data_folder / "portfolio_description.pkl")
    assert_frame_equal(description, description_sample, check_dtype=False, check_column_type=False, atol=1e-2)


@mark.xfail
def test_describe_no_inflation(portfolio_no_inflation):
    description = portfolio_no_inflation.describe()
    description_sample = pd.read_pickle(conftest.data_folder / "portfolio_description_no_inflation.pkl")
    assert_frame_equal(description, description_sample, check_dtype=False, check_column_type=False, atol=1e-2)


def test_percentile_from_history(portfolio_rebalanced_month, portfolio_no_inflation, portfolio_short_history):
    assert portfolio_rebalanced_month.percentile_history_cagr(years=1).iloc[0, 1] == approx(0.173181, abs=1e-2)
    assert portfolio_no_inflation.percentile_history_cagr(years=1).iloc[0, 1] == approx(0.17318, abs=1e-2)
    with pytest.raises(
        ValueError,
        match="Time series does not have enough history to forecast. "
        "Period length is 0.90 years. At least 2 years are required.",
    ):
        portfolio_short_history.percentile_history_cagr(years=1)


@mark.parametrize(
    "distribution, expected",
    [("hist", 0), ("norm", 0.9), ("lognorm", 0.7)],
)
def test_percentile_inverse_cagr(portfolio_rebalanced_month, distribution, expected):
    assert portfolio_rebalanced_month.percentile_inverse_cagr(distr=distribution, years=1, score=0, n=5000) == approx(
        expected, abs=1e-0
    )


def test_table(portfolio_rebalanced_month):
    assert_array_equal(
        portfolio_rebalanced_month.table["ticker"].values,
        np.array(["RGBITR.INDX", "MCFTR.INDX"]),
    )


@mark.parametrize(
    "window, real, expected",
    [(12, False, 0.1290), (24, True, 0.17067)],
)
def test_get_rolling_cagr(portfolio_rebalanced_month, window, real, expected):
    assert portfolio_rebalanced_month.get_rolling_cagr(window=window, real=real).iloc[0, -1] == approx(
        expected, abs=1e-2
    )


def test_get_rolling_cagr_failing_short_window(portfolio_not_rebalanced):
    with pytest.raises(ValueError, match="window size must be at least 1 year"):
        portfolio_not_rebalanced.get_rolling_cagr(window=1)


def test_get_rolling_cagr_failing_long_window(portfolio_not_rebalanced):
    with pytest.raises(ValueError, match="window size is more than data history depth"):
        portfolio_not_rebalanced.get_rolling_cagr(window=100)


def test_get_rolling_cagr_failing_no_inflation(portfolio_no_inflation):
    with pytest.raises(
        ValueError,
        match="Real return is not defined. Set inflation=True when initiating the class.",
    ):
        portfolio_no_inflation.get_rolling_cagr(real=True)


def test_monte_carlo_wealth(portfolio_rebalanced_month):
    df = portfolio_rebalanced_month._monte_carlo_wealth(distr="norm", years=1, n=1000)
    assert df.shape == (12, 1000)
    assert df.iloc[-1, :].mean() == approx(2915.55, rel=1e-1)


def test_monte_carlo_returns_ts(portfolio_rebalanced_month):
    df = portfolio_rebalanced_month.monte_carlo_returns_ts(distr="lognorm", years=1, n=1000)
    assert df.shape == (12, 1000)
    assert df.iloc[-1, :].mean() == approx(0.0156, abs=1e-1)


@mark.parametrize(
    "distribution, expected",
    [("hist", 2897.72), ("norm", 2940.70), ("lognorm", 2932.56)],
)
def test_percentile_wealth(portfolio_rebalanced_month, distribution, expected):
    dic = portfolio_rebalanced_month.percentile_wealth(distr=distribution, years=1, n=100, percentiles=[50])
    assert dic[50] == approx(expected, rel=1e-1)


def test_forecast_monte_carlo_cagr(portfolio_rebalanced_month):
    dic = portfolio_rebalanced_month.percentile_distribution_cagr(years=2, distr="lognorm", n=100, percentiles=[50])
    assert dic[50] == approx(0.1905, abs=5e-2)


def test_skewness(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.skewness.iloc[-1] == approx(0.4980, abs=1e-1)


def test_rolling_skewness(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.skewness_rolling(window=24).iloc[-1] == approx(0.4498, abs=1e-1)


def test_kurtosis(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.kurtosis.iloc[-1] == approx(1.46, rel=1e-2)


def test_kurtosis_rolling(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.kurtosis_rolling(window=24).iloc[-1] == approx(-0.2498, rel=1e-1)


def test_jarque_bera(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.jarque_bera["statistic"] == approx(6.3657, rel=1e-1)


def test_get_sharpe_ratio(portfolio_no_inflation):
    assert portfolio_no_inflation.get_sharpe_ratio(rf_return=0.05) == approx(1.6457, abs=1e-1)


def test_get_sortino_ratio(portfolio_no_inflation):
    assert portfolio_no_inflation.get_sortino_ratio(t_return=0.05) == approx(2.2766, rel=1e-2)


def test_diversification_ratio(portfolio_no_inflation):
    assert portfolio_no_inflation.diversification_ratio == approx(1.2961, rel=1e-2)


# This test should be a last one, as it changes the weights
def test_init_portfolio_failing():
    with pytest.raises(
        ValueError,
        match=r"Number of tickers \(2\) should be equal to the weights number \(3\)",
    ):
        ok.Portfolio(["RGBITR.INDX", "MCFTR.INDX"], weights=[0.1, 0.2, 0.7])
    with pytest.raises(ValueError, match="Weights sum is not equal to one."):
        ok.Portfolio(["RGBITR.INDX", "MCFTR.INDX"], weights=[0.1, 0.2])


# DCF Methods
def test_dcf_discount_rate(
    portfolio_cashflows_inflation, portfolio_cashflows_NO_inflation, portfolio_cashflows_NO_inflation_NO_discount_rate
):
    assert portfolio_cashflows_inflation.discount_rate == approx(0.0554, abs=1e-3)  # average inflation
    assert portfolio_cashflows_NO_inflation.discount_rate == approx(0.09, abs=1e-3)  # defined discount rate
    assert portfolio_cashflows_NO_inflation_NO_discount_rate.discount_rate == approx(0.05, abs=1e-3)  # default rate


def test_dcf_wealth_index(portfolio_cashflows_inflation, portfolio_cashflows_NO_inflation):
    assert portfolio_cashflows_inflation.dcf.wealth_index.iloc[-1, 0] == approx(164459.78, rel=1e-2)
    assert portfolio_cashflows_inflation.dcf.wealth_index.iloc[-1, 1] == approx(100050.78, rel=1e-2)
    assert portfolio_cashflows_NO_inflation.dcf.wealth_index.iloc[-1, 0] == approx(139454.34, rel=1e-2)


def test_survival_period(portfolio_cashflows_inflation):
    assert portfolio_cashflows_inflation.dcf.survival_period == approx(5.0, rel=1e-2)


def test_monte_carlo_survival_period(portfolio_cashflows_inflation_large_cf):
    result = portfolio_cashflows_inflation_large_cf.dcf.monte_carlo_survival_period(
        distr="norm",
        years=25,
        n=100
    )
    assert result.min() == approx(4.5, rel=1e-1)
    assert result.max() == approx(8.1, rel=1e-1)


def test_survival_date(portfolio_cashflows_inflation):
    assert portfolio_cashflows_inflation.dcf.survival_date == pd.to_datetime("2020-01")


def test_cashflow_pv(portfolio_cashflows_inflation, portfolio_cashflows_NO_inflation_NO_discount_rate):
    assert portfolio_cashflows_inflation.dcf.cashflow_pv == approx(-76.33, rel=1e-2)
    assert portfolio_cashflows_NO_inflation_NO_discount_rate.dcf.cashflow_pv == approx(-78.35, rel=1e-2)


def test_initial_amount_pv(portfolio_cashflows_inflation, portfolio_cashflows_NO_inflation_NO_discount_rate):
    assert portfolio_cashflows_inflation.dcf.initial_amount_pv == approx(76339.31, rel=1e-2)
    assert portfolio_cashflows_NO_inflation_NO_discount_rate.dcf.initial_amount_pv == approx(78352.61, rel=1e-2)
