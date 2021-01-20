import pandas as pd
from pytest import mark
from pytest import approx


@mark.inflation
@mark.usefixtures('_init_inflation')
class TestInflation:

    def test_get_infl_rub_data(self):
        assert self.infl_rub.first_date == pd.to_datetime('1991-01')
        assert self.infl_rub.pl.years == 10
        assert self.infl_rub.pl.months == 1
        assert self.infl_rub.name == 'Russia Inflation Rate'
        assert self.infl_rub.type == 'inflation'
        assert self.infl_rub.cumulative_inflation[-1] == approx(19576.47386585591, rel=1e-4)
        assert self.infl_rub.purchasing_power_1000 == approx(0.05107911300773333, rel=1e-4)
        assert self.infl_rub.rolling_inflation[-1] == approx(0.2070533602100877, rel=1e-4)

    def test_get_infl_usd_data(self):
        assert self.infl_usd.first_date == pd.to_datetime('1913-02')
        assert self.infl_usd.pl.years == 10
        assert self.infl_usd.pl.months == 0
        assert self.infl_usd.name == 'US Inflation Rate'
        assert self.infl_usd.type == 'inflation'
        assert self.infl_usd.cumulative_inflation[-1] == approx(0.7145424753209466, rel=1e-4)
        assert self.infl_usd.purchasing_power_1000 == approx(583.2459763429362, rel=1e-4)
        assert self.infl_usd.rolling_inflation[-1] == approx(-0.005813765681402461, rel=1e-4)

    def test_get_infl_eur_data(self):
        assert self.infl_eur.first_date == pd.to_datetime('1996-02')
        assert self.infl_eur.pl.years == 10
        assert self.infl_eur.pl.years == 10
        assert self.infl_eur.name == 'EU Inflation Rate'
        assert self.infl_eur.type == 'inflation'
        assert self.infl_eur.cumulative_inflation[-1] == approx(0.20267532488218776, rel=1e-4)
        assert self.infl_eur.purchasing_power_1000 == approx(831.4796016106495, rel=1e-4)
        assert self.infl_eur.rolling_inflation[-1] == approx(0.02317927930197139, rel=1e-4)

    def test_describe(self):
        description = self.infl_rub.describe(years=[5])
        assert list(description.columns) == ['property', 'period', 'Russia Inflation Rate']
        assert description.loc[3, 'Russia Inflation Rate'] == approx(3.0414434004010245, rel=1e-4)
        assert description.loc[5, 'Russia Inflation Rate'] == approx(247.43634907784974, rel=1e-4)

    def test_annual_inflation_ts(self):
        assert self.infl_rub.annual_inflation_ts.iloc[-1] == approx(0.02760000000000007, rel=1e-4)


@mark.rates
@mark.usefixtures('_init_rates')
class TestRates:

    def test_rates_rub(self):
        assert self.rates_rub.name == 'Max deposit rates (RUB) in Russian banks'
        assert self.rates_rub.first_date == pd.to_datetime('2015-01')
        assert self.rates_rub.last_date == pd.to_datetime('2020-02')

    def test_okid(self):
        assert self.rates_rub.okid.sum() == approx(6376.2308059223915, rel=1e-4)
