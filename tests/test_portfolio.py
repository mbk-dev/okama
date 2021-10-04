import numpy as np
import pandas as pd
import pytest
from pytest import approx
from pytest import mark
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal, assert_frame_equal

import okama as ok

from .conftest import data_folder


def test_initialization_failing():
    with pytest.raises(ValueError, match=r"Number of tickers \(2\) should be equal to the weights number \(3\)"):
        ok.Portfolio(assets=['MCFTR.INDX', 'MCFTR.INDX', 'RUB.FX'], weights=[.3, .3, .4])


def test_repr(portfolio_rebalanced_year):
    value = pd.Series(dict(
        symbol="pf1.PF",
        assets="[RUB.FX, MCFTR.INDX]",
        weights="[0.5, 0.5]",
        rebalancing_period="year",
        currency="RUB",
        inflation="RUB.INFL",
        first_date="2015-01",
        last_date="2020-01",
        period_length="5 years, 1 months",
    ))
    assert repr(portfolio_rebalanced_year) == repr(value)


def test_symbol_failing(portfolio_rebalanced_year):
    with pytest.raises(ValueError, match='portfolio symbol must be a string ending with ".PF" namespace.'):
        portfolio_rebalanced_year.symbol = 1
    with pytest.raises(ValueError, match='portfolio symbol must end with ".PF"'):
        portfolio_rebalanced_year.symbol = 'Not_a_good_symbol_for_portfolio.US'
    with pytest.raises(ValueError, match='portfolio text symbol should not have whitespace characters.'):
        portfolio_rebalanced_year.symbol = 'Not a good symbol for portfolio.PF'


def test_symbol_setter(portfolio_rebalanced_year):
    portfolio_rebalanced_year.symbol = 'portfolio_1.PF'
    assert portfolio_rebalanced_year.symbol == 'portfolio_1.PF'


def test_ror_rebalance(
    portfolio_rebalanced_year, portfolio_not_rebalanced
):
    print(f'portfolio_rebalanced_year={portfolio_rebalanced_year}')
    assert portfolio_rebalanced_year.ror[-2] == approx(0.01361, rel=1e-2)
    assert portfolio_not_rebalanced.ror[-1] == approx(0.01359, rel=1e-2)


def test_ror(portfolio_rebalanced_month):
    portfolio_sample = pd.read_pickle(data_folder / "portfolio.pkl")
    actual = portfolio_rebalanced_month.ror
    actual.rename("portfolio", inplace=True)
    assert_series_equal(actual, portfolio_sample)


def test_weights(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.weights == [0.5, 0.5]


def test_weights_ts_rebalanced_month(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.weights_ts["RUB.FX"].iloc[-1] == approx(0.5, rel=1e-2)


def test_weights_ts_rebalanced_year(portfolio_rebalanced_year, portfolio_not_rebalanced):
    assert portfolio_rebalanced_year.weights_ts["RUB.FX"].iloc[-2] == approx(0.3907, rel=1e-2)


def test_weights_ts_not_rebalanced(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.weights_ts["RUB.FX"].iloc[-1] == approx(0.2770, rel=1e-2)


def test_mean_return(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.mean_return_monthly == approx(0.010854, rel=1e-2)
    assert portfolio_rebalanced_month.mean_return_annual == approx(0.138324, rel=1e-2)


def test_real_mean_return(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.real_mean_return == approx(0.07827, rel=1e-2)


@mark.parametrize(
    "window, real, expected",
    [(1, True, 0.01100), (12, False, 0.12269), (12, True, 0.0961)],
)
def test_get_rolling_cumulative_return(portfolio_rebalanced_month, window, real, expected):
    assert portfolio_rebalanced_month.get_rolling_cumulative_return(window=window, real=real).iloc[-1, 0] == approx(expected, rel=1e-2)


def test_assets_close_monthly(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.assets_close_monthly.iloc[-1, 0] == 63.0359  # RUB.FX
    assert portfolio_not_rebalanced.assets_close_monthly.iloc[-1, 1] == 5245.6  # MCFTR.INDX


def test_close_monthly(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.close_monthly.iloc[-1] == approx(2022, rel=1e-2)


def test_number_of_securities(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.number_of_securities.iloc[-1, 0] == approx(8.88, rel=1e-2)  # RUB.FX
    assert portfolio_not_rebalanced.number_of_securities.iloc[-1, 1] == approx(0.2787, abs=1e-2)  # MCFTR.INDX


def test_dividends(portfolio_dividends):
    assert portfolio_dividends.dividends.iloc[-1] == approx(13.90, rel=1e-2)


def test_dividend_yield(portfolio_dividends):
    assert portfolio_dividends.dividend_yield.iloc[-1] == approx(
        0.03949, rel=1e-2
    )


def test_risk(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.risk_monthly == approx(0.035718, rel=1e-2)
    assert portfolio_rebalanced_month.risk_annual == approx(0.139814, rel=1e-2)


def test_get_var_historic(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.get_var_historic(time_frame=1, level=5) == approx(
        0.03815, rel=1e-2
    )
    assert portfolio_rebalanced_month.get_var_historic(time_frame=5, level=1) == approx(
        0.0969, rel=1e-2
    )


def test_get_cvar_historic(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.get_cvar_historic(
        time_frame=1, level=5
    ) == approx(0.05016, rel=1e-2)
    assert portfolio_rebalanced_month.get_cvar_historic(
        time_frame=5, level=1
    ) == approx(0.10762, rel=1e-2)


def test_drawdowns(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.drawdowns.min() == approx(-0.1265, rel=1e-2)


def test_recovery_period(portfolio_not_rebalanced):
    assert portfolio_not_rebalanced.recovery_period == 12


def test_get_cagr(portfolio_rebalanced_month, portfolio_no_inflation):
    values = pd.Series({"pf1.PF": 0.1303543, "RUB.INFL": 0.05548082428015655})
    actual = portfolio_rebalanced_month.get_cagr()
    assert_series_equal(actual, values, rtol=1e-4)
    # no inflation
    values = pd.Series({"pf1.PF": 0.1303543})
    actual = portfolio_no_inflation.get_cagr()
    assert_series_equal(actual, values, rtol=1e-4)
    # failing if wrong period
    with pytest.raises(TypeError):
        portfolio_rebalanced_month.get_cagr(period="one year")


cagr_testdata1 = [
    (1, 0.0961),
    (None, 0.0710),
]


@mark.parametrize("input_data, expected", cagr_testdata1, ids=["1 year", "full period"],)
def test_get_cagr_real(portfolio_rebalanced_month, input_data, expected):
    assert portfolio_rebalanced_month.get_cagr(period=input_data, real=True).values[0] == approx(expected, abs=1e-2)


def test_get_cagr_real_no_inflation_exception(portfolio_no_inflation):
    with pytest.raises(ValueError):
        portfolio_no_inflation.get_cagr(period=1, real=True)


@mark.parametrize(
    "period, real, expected",
    [("YTD", False, 0.01505), (1, False, 0.12269), (2, True, 0.2299)],
)
def test_cumulative_return(portfolio_rebalanced_month, period, real, expected):
    assert portfolio_rebalanced_month.get_cumulative_return(
        period=period, real=real
    ).iloc[0] == approx(expected, abs=1e-2)


cumulative_return_fail = [
    (1.5, False, TypeError),
    (-1, False, ValueError),
    (1, True, ValueError),
]


@mark.parametrize("period, real, exception", cumulative_return_fail)
def test_cumulative_return_error(portfolio_no_inflation, period, real, exception):
    with pytest.raises(exception):
        portfolio_no_inflation.get_cumulative_return(period=period, real=real)


def test_describe_inflation(portfolio_rebalanced_month):
    description = portfolio_rebalanced_month.describe()
    description_sample = pd.read_pickle(data_folder / "portfolio_description.pkl")
    assert_frame_equal(description, description_sample)


def test_describe_no_inflation(portfolio_no_inflation):
    description = portfolio_no_inflation.describe()
    description_sample = pd.read_pickle(data_folder / "portfolio_description_no_inflation.pkl")
    assert_frame_equal(description, description_sample)


def test_percentile_from_history(portfolio_rebalanced_month, portfolio_no_inflation, portfolio_short_history):
    assert portfolio_rebalanced_month.percentile_history_cagr(years=1).iloc[0, 1] == approx(0.12456, rel=1e-2)
    assert portfolio_no_inflation.percentile_history_cagr(years=1).iloc[0, 1] == approx(0.12456, rel=1e-2)
    with pytest.raises(
        ValueError,
        match="Time series does not have enough history to forecast. "
        "Period length is 0.90 years. At least 2 years are required.",
    ):
        portfolio_short_history.percentile_history_cagr(years=1)


def test_table(portfolio_rebalanced_month):
    assert_array_equal(
        portfolio_rebalanced_month.table["ticker"].values,
        np.array(["RUB.FX", "MCFTR.INDX"]),
    )


@mark.parametrize(
    "window, real, expected",
    [(12, False, 0.1290), (24, True, 0.08505)],
)
def test_get_rolling_cagr(portfolio_rebalanced_month, window, real, expected):
    assert portfolio_rebalanced_month.get_rolling_cagr(
        window=window, real=real).iloc[0, -1] == approx(expected, rel=1e-2)


def test_get_rolling_cagr_failing_short_window(portfolio_not_rebalanced):
    with pytest.raises(
        ValueError,
        match="window size should be at least 1 year"
    ):
        portfolio_not_rebalanced.get_rolling_cagr(window=1)


def test_get_rolling_cagr_failing_long_window(portfolio_not_rebalanced):
    with pytest.raises(
        ValueError,
        match="window size is more than data history depth"
    ):
        portfolio_not_rebalanced.get_rolling_cagr(window=100)


def test_get_rolling_cagr_failing_no_inflation(portfolio_no_inflation):
    with pytest.raises(
        ValueError,
        match="Real return is not defined. Set inflation=True when initiating the class."
    ):
        portfolio_no_inflation.get_rolling_cagr(real=True)


def test_monte_carlo_wealth(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month._monte_carlo_wealth(
        distr='norm',
        years=1,
        n=1000
    ).iloc[-1, :].mean() == approx(2121, rel=1e-1)


@mark.parametrize(
    "distribution, expected",
    [('hist', 2096), ('norm', 2103), ('lognorm', 2093)],
)
def test_percentile_wealth(portfolio_rebalanced_month, distribution, expected):
    dic = portfolio_rebalanced_month.percentile_wealth(distr=distribution, years=1, n=100, percentiles=[50])
    assert dic[50] == approx(expected, rel=1e-1)


def test_forecast_monte_carlo_cagr(portfolio_rebalanced_month):
    dic = portfolio_rebalanced_month.percentile_distribution_cagr(years=2, distr='lognorm', n=100, percentiles=[50])
    assert dic[50] == approx(0.12, abs=5e-2)
    with pytest.raises(ValueError):
        portfolio_rebalanced_month.percentile_distribution_cagr(years=10, distr='lognorm', n=100, percentiles=[50])


def test_skewness(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.skewness.iloc[-1] == approx(2.39731, rel=1e-2)


def test_rolling_skewness(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.skewness_rolling(window=24).iloc[-1] == approx(
        0.82381, rel=1e-2
    )


def test_kurtosis(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.kurtosis.iloc[-1] == approx(13.19578, rel=1e-2)


def test_kurtosis_rolling(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.kurtosis_rolling(window=24).iloc[-1] == approx(
        1.58931, rel=1e-2
    )


def test_jarque_bera(portfolio_rebalanced_month):
    assert portfolio_rebalanced_month.jarque_bera["statistic"] == approx(
        424.1337, rel=1e-2
    )


def test_get_sharpe_ratio(portfolio_no_inflation):
    assert portfolio_no_inflation.get_sharpe_ratio(rf_return=0.05) == approx(0.631, rel=1e-2)


# This test should be a last one, as it changes the weights
def test_init_portfolio_failing():
    with pytest.raises(
        ValueError,
        match=r"Number of tickers \(2\) should be equal to the weights number \(3\)",
    ):
        ok.Portfolio(['RUB.FX', 'MCFTR.INDX'], weights=[0.1, 0.2, 0.7])
    with pytest.raises(ValueError, match="Weights sum is not equal to one."):
        ok.Portfolio(['RUB.FX', 'MCFTR.INDX'], weights=[0.1, 0.2])
