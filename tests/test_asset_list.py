import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

import pytest
from pytest import approx
from pytest import mark

import okama as ok

from tests import conftest


@mark.asset_list
def test_asset_list_init_failing():
    with pytest.raises(ValueError, match=r"Assets must be a list."):
        ok.AssetList(assets=("RUB.FX", "MCFTR.INDX"))
    with pytest.raises(
        ValueError,
        match=r"SBGB.MOEX historical data period length is too short. " r"It must be at least 3 months.",
    ):
        ok.AssetList(assets=["SBGB.MOEX"], last_date="2019-02", inflation=True)


@mark.asset_list
@mark.usefixtures("_init_asset_list")
class TestAssetList:
    def test_repr(self):
        value = pd.Series(
            dict(
                assets="[pf1.PF, USDRUB.CBR, MCFTR.INDX]",
                currency="USD",
                first_date="2019-02",
                last_date="2020-01",
                period_length="1 years, 0 months",
                inflation="USD.INFL",
            )
        )
        assert repr(self.asset_list_with_portfolio) == repr(value)

    def test_len(self):
        assert self.asset_list.__len__() == 2

    def test_iter(self):
        tickers = [asset.symbol for asset in self.asset_list]
        assert tickers == ["USDRUB.CBR", "MCFTR.INDX"]

    def test_tickers(self):
        assert self.asset_list_with_portfolio.tickers == ["pf1", "USDRUB", "MCFTR"]

    def test_ror(self):
        asset_list_sample = pd.read_pickle(conftest.data_folder / "asset_list.pkl")
        asset_list_lt_sample = pd.read_pickle(conftest.data_folder / "asset_list_lt.pkl")
        real_estate_sample = pd.read_pickle(conftest.data_folder / "real_estate.pkl")
        assert_frame_equal(self.asset_list.assets_ror, asset_list_sample, rtol=1e-2)
        assert_frame_equal(self.asset_list_lt.assets_ror, asset_list_lt_sample, rtol=1e-2)
        assert_frame_equal(self.real_estate.assets_ror, real_estate_sample, rtol=1e-2)
        # adj_close and ror are changing over time for assets with dividends & after splits

    def test_currencies(self):
        assert self.currencies.pl.years == 1
        assert self.currencies.first_date == pd.to_datetime("2019-01")
        assert self.currencies.currencies == {
            "RUBUSD.FX": "USD",
            "EURUSD.FX": "USD",
            "CNYUSD.FX": "USD",
            "asset list": "USD",
        }
        assert self.currencies.names == {
            "RUBUSD.FX": "Russian Rouble/US Dollar FX Cross Rate",
            "EURUSD.FX": "EURUSD",
            "CNYUSD.FX": "Chinese Renminbi/US Dollar FX Cross Rate",
        }
        assert self.currencies.describe().iloc[1, -1] == approx(0.02485, rel=1e-2)

    def test_names(self):
        assert list(self.spy.names.values()) == ["SPDR S&P 500 ETF Trust"]

    @mark.smoke
    def test_make_asset_list(self):
        assert self.asset_list.last_date == pd.to_datetime("2020-01")
        assert self.asset_list.assets_first_dates["MCFTR.INDX"].strftime("%Y-%m-%d") == "1997-10-01"
        last_year = int(self.asset_list.last_date.year)
        assert int(self.asset_list.assets_last_dates["MCFTR.INDX"].year) > last_year
        assert self.asset_list.newest_asset == "MCFTR.INDX"
        assert self.asset_list.eldest_asset == "USDRUB.CBR"
        assert list(self.asset_list.assets_ror) == ["USDRUB.CBR", "MCFTR.INDX"]
        assert self.asset_list.assets_ror.columns.name == "Symbols"

    def test_calculate_wealth_indexes(self):
        assert self.asset_list.wealth_indexes.sum(axis=1)[-1] == approx(
            3339.677963676333, rel=1e-2
        )  # last month indexes sum

    def test_risk(self):
        assert self.asset_list.risk_monthly["USDRUB.CBR"] == approx(0.0258, rel=1e-2)
        assert self.asset_list.risk_monthly["MCFTR.INDX"] == approx(0.0264, rel=1e-2)
        assert self.asset_list.risk_annual["USDRUB.CBR"] == approx(0.0825, rel=1e-2)
        assert self.asset_list.risk_annual["MCFTR.INDX"] == approx(0.1222, rel=1e-2)

    def test_semideviation_monthly(self):
        assert self.asset_list.semideviation_monthly[0] == approx(0.01930, abs=1e-3)
        assert self.asset_list.semideviation_monthly[1] == approx(0.01109, abs=1e-3)

    def test_semideviation_annual(self):
        assert self.asset_list.semideviation_annual[0] == approx(0.0679, abs=1e-2)
        assert self.asset_list.semideviation_annual[1] == approx(0.0384, abs=1e-2)

    def test_get_var_historic(self):
        assert self.asset_list.get_var_historic(time_frame=1, level=5)["USDRUB.CBR"] == approx(0.0398, abs=1e-2)
        assert self.asset_list.get_var_historic(time_frame=5, level=1)["MCFTR.INDX"] == approx(-0.1048, abs=1e-2)
        assert self.asset_list_no_infl.get_var_historic(time_frame=1, level=1)["USDRUB.CBR"] == approx(0.04975, abs=1e-2)
        assert self.asset_list_no_infl.get_var_historic(time_frame=1, level=1)["MCFTR.INDX"] == approx(
            0.01229, abs=1e-2
        )

    def test_get_cvar_historic(self):
        assert self.asset_list.get_cvar_historic(level=5, time_frame=12)["USDRUB.CBR"] == approx(0.108, rel=1e-2)
        assert self.asset_list.get_cvar_historic(level=5, time_frame=12)["MCFTR.INDX"] == approx(-0.3130, rel=1e-2)

    def test_drawdowns(self):
        assert self.asset_list.drawdowns.min().sum() == approx(-0.082932, rel=1e-2)

    def test_recovery_periods(self):
        assert self.asset_list.recovery_periods["MCFTR.INDX"] == approx(0, rel=1e-2)
        assert np.isnan(self.asset_list.recovery_periods["USDRUB.CBR"])
        assert self.asset_list_lt.recovery_periods["MCFTR.INDX"] == 45
        assert self.asset_list_lt.recovery_periods["USDRUB.CBR"] == 69

    cagr_testdata1 = [
        (1, -0.0463, 0.3131, 0.0242),
        (None, -0.0857, 0.3651, 0.0318),
    ]

    @mark.parametrize(
        "input_data,expected1,expected2,expected3",
        cagr_testdata1,
        ids=["1 year", "full period"],
    )
    def test_get_cagr(self, input_data, expected1, expected2, expected3):
        assert self.asset_list.get_cagr(period=input_data)["USDRUB.CBR"] == approx(expected1, abs=1e-2)
        assert self.asset_list.get_cagr(period=input_data)["MCFTR.INDX"] == approx(expected2, abs=1e-2)
        assert self.asset_list.get_cagr(period=input_data)["RUB.INFL"] == approx(expected3, abs=1e-2)

    cagr_testdata2 = [
        (1, -0.0688, 0.2820),
        (None, -0.1169, 0.3228),
    ]

    @mark.parametrize(
        "input_data,expected1,expected2",
        cagr_testdata2,
        ids=["1 year", "full period"],
    )
    def test_get_cagr_real(self, input_data, expected1, expected2):
        assert self.asset_list.get_cagr(period=input_data, real=True)["USDRUB.CBR"] == approx(expected1, abs=1e-2)
        assert self.asset_list.get_cagr(period=input_data, real=True)["MCFTR.INDX"] == approx(expected2, abs=1e-2)

    def test_get_cagr_value_error(self):
        with pytest.raises(ValueError):
            self.asset_list.get_cagr(period=3, real=True)

    def test_get_cagr_real_no_inflation_exception(self):
        with pytest.raises(ValueError):
            self.asset_list_no_infl.get_cagr(period=1, real=True)

    @pytest.mark.parametrize("real, expected1, expected2", [(False, 0.05822, 0.2393), (True, 0.0204, 0.1951)])
    def test_get_rolling_cagr(self, real, expected1, expected2):
        assert self.asset_list_lt.get_rolling_cagr(window=24, real=real)["USDRUB.CBR"].iloc[-1] == approx(
            expected1, rel=1e-2
        )
        assert self.asset_list_lt.get_rolling_cagr(window=24, real=real)["MCFTR.INDX"].iloc[-1] == approx(
            expected2, rel=1e-2
        )

    get_rolling_cagr_error_data = [
        (0, False, ValueError),  # window should be at least 12 months for CAGR
        (12.5, False, ValueError),  # not an integer
        (10 * 12, False, ValueError),  # window size should be in the history period
        (
            12,
            True,
            ValueError,
        ),  # real CAGR is defined when AssetList(inflation=True) only
    ]

    @pytest.mark.parametrize("window, real, exception", get_rolling_cagr_error_data)
    def test_get_rolling_cagr_error(self, window, real, exception):
        with pytest.raises(exception):
            self.asset_list_no_infl.get_rolling_cagr(window=window, real=real)

    cumulative_testdata1 = [
        ("YTD", 0.0182, 0.0118, 0.0040),
        (1, -0.0463, 0.3131, 0.0242),
        (None, -0.0957, 0.4009, 0.0345),
    ]

    @mark.parametrize(
        "input_data,expected1,expected2,expected3",
        cumulative_testdata1,
        ids=["YTD", "1 year", "full period"],
    )
    def test_get_cumulative_return(self, input_data, expected1, expected2, expected3):
        assert self.asset_list.get_cumulative_return(period=input_data)["USDRUB.CBR"] == approx(expected1, abs=1e-2)
        assert self.asset_list.get_cumulative_return(period=input_data)["MCFTR.INDX"] == approx(expected2, abs=1e-2)
        assert self.asset_list.get_cumulative_return(period=input_data)["RUB.INFL"] == approx(expected3, abs=1e-2)

    cumulative_testdata2 = [
        ("YTD", 0.01424, 0.0077),
        (1, -0.06885, 0.2820),
        (None, -0.1260, 0.3541),
    ]

    @mark.parametrize(
        "input_data,expected1,expected2",
        cumulative_testdata2,
        ids=["YTD", "1 year", "full period"],
    )
    def test_get_cumulative_return_real(self, input_data, expected1, expected2):
        assert self.asset_list.get_cumulative_return(period=input_data, real=True)["USDRUB.CBR"] == approx(
            expected1, abs=1e-2
        )
        assert self.asset_list.get_cumulative_return(period=input_data, real=True)["MCFTR.INDX"] == approx(
            expected2, abs=1e-2
        )

    def test_get_cumulative_return_value_error(self):
        with pytest.raises(ValueError):
            self.asset_list.get_cumulative_return(period=3, real=True)

    def test_get_cumulative_return_real_no_inflation_exception(self):
        with pytest.raises(ValueError):
            self.asset_list_no_infl.get_cumulative_return(period=1, real=True)

    def test_get_rolling_cumulative_return(self):
        assert self.asset_list_lt.get_rolling_cumulative_return(window=12)["USDRUB.CBR"].iloc[-1] == approx(
            -0.0462, rel=1e-2
        )
        assert self.asset_list_lt.get_rolling_cumulative_return(window=12)["MCFTR.INDX"].iloc[-1] == approx(
            0.3130, rel=1e-2
        )

    def test_mean_return(self):
        assert self.asset_list.mean_return["USDRUB.CBR"] == approx(-0.0854, abs=1e-2)
        assert self.asset_list.mean_return["MCFTR.INDX"] == approx(0.3701, abs=1e-2)
        assert self.asset_list.mean_return["RUB.INFL"] == approx(0.0319, abs=1e-2)

    def test_real_return(self):
        assert self.asset_list.real_mean_return["USDRUB.CBR"] == approx(-0.11366, abs=1e-2)
        assert self.asset_list.real_mean_return["MCFTR.INDX"] == approx(0.3276, abs=1e-2)

    def test_annual_return_ts(self):
        assert self.asset_list.annual_return_ts.iloc[-1, 0] == approx(0.01829, rel=1e-2)
        assert self.asset_list.annual_return_ts.iloc[-1, 1] == approx(0.01180, rel=1e-2)

    def test_describe(self):
        description = self.asset_list.describe(tickers=False).iloc[:-2, :]  # last 2 rows have fresh lastdate
        description_sample = pd.read_pickle(conftest.data_folder / "asset_list_describe.pkl").iloc[:-2, :]
        cols = list(description_sample.columns.values)
        description = description[cols]  # columns order should not be an issue
        assert_frame_equal(description, description_sample, check_dtype=False, check_column_type=False, rtol=1e-2)

    def test_dividend_yield(self):
        assert self.spy.assets_dividend_yield.iloc[-1, 0] == approx(0.0125, abs=1e-3)
        assert self.spy_rub.assets_dividend_yield.iloc[-1, 0] == approx(0.01197, abs=1e-3)
        assert self.asset_list.assets_dividend_yield.iloc[:, 0].sum() == 0
        assert self.asset_list_with_portfolio_dividends.assets_dividend_yield.iloc[-1, 0] == approx(0.0394, abs=1e-2)

    def test_dividends_annual(self):
        assert self.spy.dividends_annual.iloc[-2, 0] == approx(1.4194999999999998, rel=1e-2)
        assert self.asset_list.dividends_annual.iloc[:, 0].sum() == 0

    def test_growing_dividend_years(self):
        assert self.spy.dividend_growing_years.iloc[-1, 0] == 0

    def test_paying_dividend_years(self):
        assert self.spy.dividend_paying_years.iloc[-2, 0] == 2

    def test_get_dividend_mean_growth_rate_valid(self):
        assert self.spy.get_dividend_mean_growth_rate(period=2).iloc[-1] == approx(-0.02765, abs=1e-2)

    def test_get_dividend_mean_growth_rate_value_err(self):
        with pytest.raises(
            ValueError,
            match="'period' \\(3\\) is beyond historical data range \\(2.0\\)",
        ):
            self.spy.get_dividend_mean_growth_rate(period=3)

    def test_tracking_difference_failing(self):
        with pytest.raises(
            ValueError,
            match="At least 2 symbols should be provided to calculate Tracking Difference.",
        ):
            self.spy.tracking_difference()

    @mark.parametrize(
        "window,expected",
        [(None, 0.4967), (6, 0.1537), (12, 0.3594)],
        ids=["None", "6 months", "12 months"],
    )
    def test_tracking_difference(self, window, expected):
        assert self.asset_list.tracking_difference(rolling_window=window)["MCFTR.INDX"].iloc[-1] == approx(
            expected, abs=1e-2
        )

    @mark.parametrize(
        "window,expected",
        [(None, 0.4510), (12, 0.3594)],
        ids=["None", "12 months"],
    )
    def test_tracking_difference_annualized(self, window, expected):
        assert self.asset_list.tracking_difference_annualized(rolling_window=window).iloc[-1, 0] == approx(
            expected, abs=1e-2
        )

    def test_tracking_difference_annual(self):
        assert self.asset_list.tracking_difference_annual.iloc[0, 0] == approx(0.4966, abs=1e-2)

    @mark.parametrize(
        "window,expected",
        [(None, 0.3115), (24, 0.1929)],
        ids=["None", "24 months"],
    )
    def test_tracking_error(self, window, expected):
        assert self.asset_list_lt.tracking_error(rolling_window=window).iloc[-1, 0] == approx(
            expected, abs=1e-2
        )

    def test_tracking_error_failing(self):
        with pytest.raises(
            ValueError,
            match="window size should be at least 1 year",
        ):
            self.asset_list.tracking_error(rolling_window=5)

    def test_index_corr(self):
        assert self.asset_list.index_corr.iloc[-1, 0] == approx(-0.519, abs=1e-2)

    def test_index_beta(self):
        assert self.asset_list.index_beta.iloc[-1, 0] == approx(-0.5052, abs=1e-2)

    def test_skewness(self):
        assert self.asset_list.skewness["USDRUB.CBR"].iloc[-1] == approx(0.47897, abs=1e-2)
        assert self.asset_list.skewness["MCFTR.INDX"].iloc[-1] == approx(0.24876, abs=1e-2)

    def test_rolling_skewness_failing(self):
        with pytest.raises(ValueError, match=r"window size is more than data history depth"):
            self.asset_list.skewness_rolling(window=24)

    def test_kurtosis(self):
        assert self.asset_list.kurtosis["USDRUB.CBR"].iloc[-1] == approx(0.7073, abs=1e-2)
        assert self.asset_list.kurtosis["MCFTR.INDX"].iloc[-1] == approx(-1.32129, rel=1e-2)

    def test_kurtosis_rolling(self):
        assert self.asset_list_lt.kurtosis_rolling(window=24)["USDRUB.CBR"].iloc[-1] == approx(0.8472, abs=1e-2)
        assert self.asset_list_lt.kurtosis_rolling(window=24)["MCFTR.INDX"].iloc[-1] == approx(-0.11495, abs=1e-2)

    def test_jarque_bera(self):
        assert self.asset_list.jarque_bera["USDRUB.CBR"].iloc[-1] == approx(0.8243, abs=1e-2)
        assert self.asset_list.jarque_bera["MCFTR.INDX"].iloc[-1] == approx(0.60333, abs=1e-2)

    def test_get_sharpe_ratio(self):
        sharpe_ratio = self.asset_list.get_sharpe_ratio(rf_return=0.06)
        assert sharpe_ratio.loc["USDRUB.CBR"] == approx(-1.7300, rel=1e-2)
        assert sharpe_ratio.loc["MCFTR.INDX"] == approx(2.53, rel=1e-2)

    def test_get_sortino_ratio(self):
        sortino_ratio = self.asset_list.get_sortino_ratio(t_return=0.02)
        assert sortino_ratio.loc["USDRUB.CBR"] == approx(-1.5306, rel=1e-2)
        assert sortino_ratio.loc["MCFTR.INDX"] == approx(10.36, rel=1e-2)
