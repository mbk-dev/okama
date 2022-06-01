import pandas as pd
import pytest
from pytest import mark
from pytest import approx

import okama as ok


def test_macro_init_failing():
    with pytest.raises(ValueError, match=r"RATE is not in allowed namespaces: \['INFL'\]"):
        ok.Inflation("RUS_RUB.RATE")
    with pytest.raises(ValueError, match=r"INFL is not in allowed namespaces: \['RATE'\]"):
        ok.Rate("USD.INFL")
    with pytest.raises(ValueError, match=r"INFL is not in allowed namespaces: \['RATIO'\]"):
        ok.Indicator("USD.INFL")


@mark.inflation
@mark.usefixtures("_init_inflation")
class TestInflation:
    def test_get_infl_rub_data(self):
        assert self.infl_rub.first_date == pd.to_datetime("1991-01")
        assert self.infl_rub.pl.years == 10
        assert self.infl_rub.pl.months == 1
        assert self.infl_rub.name == "Russia Inflation Rate"
        assert self.infl_rub.type == "inflation"

    def test_get_infl_usd_data(self):
        assert self.infl_usd.first_date == pd.to_datetime("1913-02")
        assert self.infl_usd.pl.years == 10
        assert self.infl_usd.pl.months == 0
        assert self.infl_usd.name == "US Inflation Rate"
        assert self.infl_usd.type == "inflation"

    def test_get_infl_eur_data(self):
        assert self.infl_eur.first_date == pd.to_datetime("1996-02")
        assert self.infl_eur.pl.years == 10
        assert self.infl_eur.pl.years == 10
        assert self.infl_eur.name == "EU Inflation Rate"
        assert self.infl_eur.type == "inflation"

    def test_cumulative_inflation(self):
        assert self.infl_rub.cumulative_inflation[-1] == approx(19576.47386585591, rel=1e-4)
        assert self.infl_eur.cumulative_inflation[-1] == approx(0.20267532488218776, abs=1e-4)
        assert self.infl_usd.cumulative_inflation[-1] == approx(0.7145424753209466, abs=1e-4)

    def test_purchasing_power_1000(self):
        assert self.infl_usd.purchasing_power_1000 == approx(583.2459763429362, rel=1e-4)
        assert self.infl_eur.purchasing_power_1000 == approx(831.4796016106495, rel=1e-4)
        assert self.infl_rub.purchasing_power_1000 == approx(0.05107911300773333, abs=1e-4)

    def test_rolling_inflation(self):
        assert self.infl_eur.rolling_inflation[-1] == approx(0.02317927930197139, abs=1e-4)
        assert self.infl_usd.rolling_inflation[-1] == approx(-0.0058137, abs=1e-4)
        assert self.infl_rub.rolling_inflation[-1] == approx(0.2070533602100877, abs=1e-4)

    def test_annual_inflation_ts(self):
        assert self.infl_rub.annual_inflation_ts.iloc[-1] == approx(0.0276, abs=1e-4)
        assert self.infl_usd.annual_inflation_ts[-1] == approx(-0.0059, abs=1e-4)
        assert self.infl_eur.annual_inflation_ts[-1] == approx(-0.002015, abs=1e-4)

    def test_values_monthly(self):
        assert self.infl_eur.values_monthly[-1] == approx(0.003, abs=1e-4)
        assert self.infl_usd.values_monthly[-1] == approx(-0.0059, abs=1e-4)
        assert self.infl_rub.values_monthly[-1] == approx(0.0276, abs=1e-4)

    def test_describe(self):
        description = self.infl_rub.describe(years=[5])
        assert list(description.columns) == [
            "property",
            "period",
            "RUB.INFL",
        ]
        assert description.loc[0, "RUB.INFL"] == approx(0.02760, abs=1e-4)  # YTD Compound Inflation
        assert description.loc[3, "RUB.INFL"] == approx(3.0414434004010245, rel=1e-4)
        assert description.loc[5, "RUB.INFL"] == approx(247.43634907784974, rel=1e-4)


@mark.rates
@mark.usefixtures("_init_rates")
class TestRates:
    def test_rates_init(self):
        assert self.rates_rub.name == "Max deposit rates (RUB) in Russian banks"
        assert self.rates_rub.first_date == pd.to_datetime("2015-01")
        assert self.rates_rub.last_date == pd.to_datetime("2020-02")

    def test_values_monthly(self):
        assert self.rates_rub.values_monthly[-1] == 0.0639

    def test_values_daily(self):
        assert self.rates_ruonia.values_daily[-1] == 0.0605
        assert self.rates_ruonia.values_daily.shape[0] == 1846
        assert self.rates_cbr_rate.values_daily.shape[0] == 62  # RUS_CBR.RATE has only monthly values

    def test_describe(self):
        description = self.rates_rub.describe(years=[5])
        assert list(description.columns) == [
            "property",
            "period",
            "RUS_RUB.RATE",
        ]
        assert description.loc[0, "RUS_RUB.RATE"] == approx(0.066, abs=1e-4)  # YTD mean
        assert description.loc[3, "RUS_RUB.RATE"] == approx(0.0639, abs=1e-4)
        assert description.loc[5, "RUS_RUB.RATE"] == approx(0.08875, abs=1e-4)


@mark.indicator
@mark.usefixtures("_init_indicator")
class TestIndicator:
    def test_indicator_init(self):
        assert self.cape10_usd.name == "Cyclically adjusted price-to-earnings ratio CAPE10 for USA"
        assert self.cape10_usd.first_date == pd.to_datetime("2021-01")
        assert self.cape10_usd.last_date == pd.to_datetime("2022-02")

    def test_values_monthly(self):
        assert self.cape10_usd.values_monthly[-1] == approx(34.93, rel=1e-4)
