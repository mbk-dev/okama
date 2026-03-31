import numpy as np
import pandas as pd
import pytest

import okama as ok
from tests.asset_list.conftest import _FakeAsset, _FakeCurrencyAsset


def test_init_uses_mocked_assets(list_basic_patches):
    dm = list_basic_patches["defaults"]
    al = ok.AssetList(["A.US", "B.US"], ccy="USD", inflation=False)

    # Symbols/tickers
    assert al.symbols == ["A.US", "B.US"]
    assert al.tickers == ["A", "B"]

    # Dates come from mocked series intersection
    assert al.first_date == dm.first_ts
    assert al.last_date == dm.last_ts

    # assets_ror built from our series
    ror = al.assets_ror
    assert list(ror.columns) == ["A.US", "B.US"]
    assert list(ror.index) == list(dm.ror_index)
    assert pytest.approx(float(ror.loc["2020-01", "A.US"])) == 0.01
    assert pytest.approx(float(ror.loc["2020-03", "B.US"])) == 0.02


def test_make_ror_called_with_base_currency(list_basic_patches, mocker):
    # spy on _make_ror to ensure it is invoked for each asset with base currency ticker "USD"
    spy_make_ror = mocker.spy(ok.common.make_asset_list.ListMaker, "_make_ror")

    al = ok.AssetList(["A.US", "B.US"], ccy="USD", inflation=False)
    assert len(spy_make_ror.call_args_list) == 2

    # Each call: args are (self, list_asset, base_currency_name)
    for c in spy_make_ror.call_args_list:
        args = c.args if hasattr(c, "args") else c[0]
        assert args[2] == "USD"
        assert hasattr(args[1], "symbol")  # our FakeAsset

    # also verify that computed DataFrame matches expectations
    assert list(al.assets_ror.columns) == ["A.US", "B.US"]


def test_basic_properties_and_assets_ror(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    assert al.symbols == ["IDX.US", "A.US", "B.US"]
    assert al.tickers == ["IDX", "A", "B"]

    ror = al.assets_ror
    assert isinstance(ror, pd.DataFrame)
    assert list(ror.columns) == ["IDX.US", "A.US", "B.US"]
    assert len(ror.index) == 24


def test_risk_and_rolling_risk_annual(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    # risk_annual is an annualized risk expanding time series for each asset.
    risk_annual = al.risk_annual
    assert isinstance(risk_annual, pd.DataFrame)
    assert list(risk_annual.columns) == ["IDX.US", "A.US", "B.US"]
    # values should be finite and non-negative in the last row
    last = risk_annual.iloc[-1]
    for col in risk_annual.columns:
        v = float(last[col])
        assert np.isfinite(v) and v >= 0.0

    # rolling risk 12 months should have 24-12+1 rows and no NaNs (dropna in implementation)
    rr = al.get_rolling_risk_annual(window=12)
    assert isinstance(rr, pd.DataFrame)
    assert rr.shape[0] == 24 - 12 + 1
    assert rr.notna().all(axis=None)


def test_semideviation_and_mean_returns(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    sd_m = al.semideviation_monthly
    sd_a = al.semideviation_annual
    mean_r = al.mean_return
    assert isinstance(sd_m, pd.Series) and isinstance(sd_a, pd.Series)
    assert isinstance(mean_r, pd.Series)
    assert all(x in sd_m.index for x in ["IDX.US", "A.US", "B.US"])  # columns become index
    assert all(~np.isnan(sd_a.values))


def test_var_cvar_drawdowns_recovery_periods(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    var = al.get_var_historic()
    cvar = al.get_cvar_historic()
    dd = al.drawdowns
    rec = al.recovery_periods
    assert isinstance(var, pd.Series) and isinstance(cvar, pd.Series)
    assert isinstance(dd, pd.DataFrame) and isinstance(rec, pd.Series)
    assert all(x in var.index for x in ["IDX.US", "A.US", "B.US"])  # columns become index


def test_cagr_and_cumulative_returns(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    cagr = al.get_cagr()
    cum = al.get_cumulative_return()
    assert isinstance(cagr, pd.Series) and isinstance(cum, pd.Series)
    assert all(x in cagr.index for x in ["IDX.US", "A.US", "B.US"])  # columns become index


def test_tracking_and_index_metrics(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    td = al.tracking_difference()
    te = al.tracking_error()
    beta = al.index_beta()
    corr = al.index_corr()
    assert isinstance(td, pd.DataFrame) and isinstance(te, pd.DataFrame)
    assert isinstance(beta, pd.DataFrame) and isinstance(corr, pd.DataFrame)


def test_distribution_properties_with_24m_normal_series(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    skew = al.skewness
    kurt = al.kurtosis
    rs = al.kurtosis_rolling(window=12)
    # Normality tests
    jb_test = al.jarque_bera
    ks_test = al.kstest(distr="norm")
    assert isinstance(skew, pd.DataFrame) and isinstance(kurt, pd.DataFrame)
    assert isinstance(jb_test, pd.DataFrame)
    assert isinstance(rs, pd.DataFrame)
    assert isinstance(ks_test, pd.DataFrame)
    # p-value (< 0.05) indicates that null hypothesis (H0) is rejected and the time series is not normally distributed
    assert (jb_test.loc["p-value", :] > 0.05).all()
    assert (ks_test.loc["p-value", :] > 0.05).all()


def test_sharpe_and_sortino(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    sh = al.get_sharpe_ratio()
    so = al.get_sortino_ratio()
    assert isinstance(sh, pd.Series) and isinstance(so, pd.Series)


def test_wealth_indexes_and_risk_monthly(synthetic_env2):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    ror = al.assets_ror
    wi = al.wealth_indexes
    # wealth index has an initial 1000 line
    assert wi.shape[0] == ror.shape[0] + 1
    assert pytest.approx(float(wi.iloc[0, 0])) == 1000.0
    expected_last = (1.0 + ror).cumprod().iloc[-1] * 1000.0
    for c in ror.columns:
        assert pytest.approx(float(wi.iloc[-1][c]), rel=1e-10) == float(expected_last[c])

    rm = al.risk_monthly
    assert isinstance(rm, pd.DataFrame)
    assert rm.shape[0] == ror.shape[0] - 1
    for c in ror.columns:
        assert pytest.approx(float(rm.iloc[-1][c]), rel=1e-10) == float(ror[c].std())


def test_annual_return_ts_last_year_matches_product(synthetic_env2):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    art = al.annual_return_ts
    assert isinstance(art, pd.DataFrame)
    assert list(art.columns) == ["IDX.US", "A.US", "B.US"]
    ror = al.assets_ror
    last_year = ror.index[-1].year
    by_last = (1.0 + ror[ror.index.year == last_year]).prod() - 1.0
    for c in ror.columns:
        assert pytest.approx(float(art.iloc[-1][c]), rel=1e-12) == float(by_last[c])


def test_repr_contains_fields(synthetic_env2):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    rep = repr(al)
    assert "assets" in rep and "currency" in rep and "period_length" in rep
    assert al.first_date.strftime("%Y-%m") in rep
    assert al.last_date.strftime("%Y-%m") in rep


def test_describe_basic_with_zero_dividends(synthetic_env2, mocker):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    zero_df = pd.DataFrame(0.0, index=al.assets_ror.index, columns=al.symbols)
    mocker.patch("okama.common.make_asset_list.ListMaker._get_assets_dividends", return_value=zero_df)

    desc = al.describe()
    props = set(desc["property"].values)
    assert {
        "Compound return",
        "CAGR",
        "Annualized mean return",
        "Dividend yield",
        "Risk",
        "CVAR",
        "Max drawdowns",
        "Max drawdowns dates",
        "Inception date",
        "Last asset date",
        "Common last data date",
    }.issubset(props)
    risk_row = desc.loc[desc["property"] == "Risk"].iloc[0]
    for c in al.symbols:
        assert pytest.approx(float(risk_row[c]), rel=1e-10) == float(al.risk_annual.iloc[-1][c])
    mr_row = desc.loc[desc["property"] == "Annualized mean return"].iloc[0]
    for c in al.symbols:
        assert pytest.approx(float(mr_row[c]), rel=1e-10) == float(al.mean_return[c])
    dy_row = desc.loc[desc["property"] == "Dividend yield"].iloc[0]
    for c in al.symbols:
        assert float(dy_row[c]) == 0.0


def test_real_mean_return_with_mocked_inflation(mocker):
    # Asset with 1%/month; inflation 0.2%/month
    idx = pd.period_range("2020-01", periods=24, freq="M")
    asset_ror = pd.Series(0.01, index=idx, name="A.US")
    infl_monthly = pd.Series(0.002, index=idx.to_timestamp(how="end"), name="USD.INFL")

    class _AssetInfl(_FakeAsset):
        pass

    fake_assets = {"A.US": _AssetInfl("A.US", asset_ror)}
    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets)
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset)

    class _FakeInflation:
        def __init__(self, symbol: str, first_date=None, last_date=None):
            self.symbol = symbol
            self.first_date = infl_monthly.index[0].to_period("M").to_timestamp(how="start")
            self.last_date = infl_monthly.index[-1].to_period("M").to_timestamp(how="start")
            # Use PeriodIndex to align in _add_inflation (concat inner)
            self.values_monthly = infl_monthly.to_period("M")

    mocker.patch("okama.common.make_asset_list.macro.Inflation", side_effect=_FakeInflation)

    al = ok.AssetList(["A.US"], ccy="USD", inflation=True)
    # real_mean_return = (1+mean_ret)/(1+infl_mean) - 1, where means are annualized arithmetic
    mr = float(al.mean_return["A.US"])  # ~0.12
    infl_mean_annual = float(al.inflation_ts.values.mean() * 12)
    expected = (1.0 + mr) / (1.0 + infl_mean_annual) - 1.0
    assert pytest.approx(float(al.real_mean_return["A.US"])) == expected


def test_dividends_and_yield_pipeline(mocker):
    # price=100 monthly, dividends=1 per month -> LTM yield 12/100=0.12
    idx = pd.period_range("2020-01", periods=24, freq="M")
    ror = pd.Series(0.0, index=idx, name="D.US")

    class _AssetWithDiv(_FakeAsset):
        def __init__(self, symbol: str, ror: pd.Series, currency: str = "USD", name: str | None = None):
            # Do NOT call super().__init__ because the base class assigns to
            # `self.close_monthly`, which would conflict with the read-only
            # property defined below. Here we only set the minimal attributes
            # required by ListMaker/AssetList logic in tests.
            self.symbol = symbol
            self.ticker = symbol.split(".")[0]
            self.name = name or f"{self.ticker} name"
            self.currency = currency
            self.ror = ror
            self.first_date = ror.index[0].to_timestamp(how="start")
            self.last_date = ror.index[-1].to_timestamp(how="start")

        @property
        def dividends(self):
            return pd.Series(1.0, index=idx, name=self.symbol)

        @property
        def close_monthly(self):
            # Return monthly closing prices indexed by PeriodIndex (as expected by _assets_dividend_yield)
            return pd.Series(100.0, index=idx, name=self.symbol)

    fake_assets = {"D.US": _AssetWithDiv("D.US", ror)}
    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets)
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset)

    al = ok.AssetList(["D.US"], ccy="USD", inflation=False)
    dy = al.dividend_yield
    da = al.dividends_annual
    dya = al.dividend_yield_annual

    assert isinstance(dy, pd.DataFrame) and "D.US" in dy.columns
    assert isinstance(da, pd.DataFrame) and "D.US" in da.columns
    assert isinstance(dya, pd.DataFrame) and "D.US" in dya.columns

    assert pytest.approx(float(dy["D.US"].iloc[-1]), rel=1e-9) == 0.12
    assert pytest.approx(float(da["D.US"].iloc[-1]), rel=1e-9) == 12.0
    assert pytest.approx(float(dya["D.US"].iloc[-1]), rel=1e-9) == 0.12

    dpy = al.dividend_paying_years
    dgy = al.dividend_growing_years
    assert isinstance(dpy, pd.DataFrame) and isinstance(dgy, pd.DataFrame)
    assert dgy["D.US"].fillna(0).max() == 0
    assert dpy["D.US"].iloc[-1] >= 1

    gmr = al.get_dividend_mean_growth_rate(period=2)
    dmy = al.get_dividend_mean_yield(period=2)
    assert pytest.approx(float(gmr.iloc[0]), abs=1e-12) == 0.0
    assert pytest.approx(float(dmy.iloc[0]), rel=1e-9) == 0.12


def test_tracking_difference_annualized_and_annual(synthetic_env2):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    tda = al.tracking_difference_annualized()
    tdan = al.tracking_difference_annual
    assert isinstance(tda, pd.DataFrame)
    assert isinstance(tdan, pd.DataFrame)
    assert list(tda.columns) == ["A.US", "B.US"]
    assert list(tdan.columns) == ["A.US", "B.US"]
    # Annualized expanding series may have fewer rows due to GIPS 12m requirement
    assert 1 <= tda.shape[0] <= al.assets_ror.shape[0]
    assert tdan.shape[0] >= 1
