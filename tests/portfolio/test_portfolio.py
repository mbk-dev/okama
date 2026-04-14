import numpy as np
import pandas as pd
import pytest
from urllib.parse import parse_qs, urlparse

import okama as ok

# Note: These tests use the global synthetic_env fixture defined in tests/conftest.py
# which patches asset loading and the currency Asset to avoid any external API calls.


@pytest.fixture()
def pf_ab_monthly(synthetic_env):
    """Two-asset Portfolio with monthly rebalancing and no inflation (mocked data)."""
    return ok.Portfolio(["A.US", "B.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))


@pytest.fixture()
def pf_ab_none(synthetic_env):
    """Two-asset Portfolio with no rebalancing (weights drift)."""
    return ok.Portfolio(["A.US", "B.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="none"))


@pytest.fixture()
def pf_three_monthly(synthetic_env):
    """Three-asset Portfolio with monthly rebalancing and no inflation (mocked data)."""
    return ok.Portfolio(
        ["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month")
    )


def test_initialization_failing_weights_number(synthetic_env):
    with pytest.raises(
        ValueError,
        match=r"Number of tickers \(2\) should be equal to the weights number \(3\)",
    ):
        ok.Portfolio(["A.US", "B.US"], weights=[0.2, 0.3, 0.5], ccy="USD", inflation=False)


def test_repr_contains_key_fields(pf_ab_monthly):
    pf_ab_monthly.symbol = "pf_mock.PF"
    r = repr(pf_ab_monthly)
    # Basic sanity: important fields are present in representation
    assert "symbol" in r and "pf_mock.PF" in r
    assert "assets" in r and "A.US" in r and "B.US" in r
    assert "currency" in r and "USD" in r
    assert "rebalancing_period" in r


def test_weights_default_and_setter(pf_ab_monthly):
    # default equal weights
    assert pf_ab_monthly.weights == [0.5, 0.5]
    # change weights to a valid pair
    pf_ab_monthly.weights = [0.3, 0.7]
    assert pf_ab_monthly.weights == [0.3, 0.7]
    # sum must be 1 and length must match number of tickers
    with pytest.raises(ValueError):
        pf_ab_monthly.weights = [0.6, 0.5]


def test_symbol_setter_validation(pf_ab_monthly):
    with pytest.raises(ValueError):
        pf_ab_monthly.symbol = "bad symbol"
    with pytest.raises(ValueError):
        pf_ab_monthly.symbol = "ABC.US"
    pf_ab_monthly.symbol = "pf1.PF"
    assert pf_ab_monthly.symbol == "pf1.PF"


def test_weights_ts_behavior_for_different_rebalancing(pf_ab_monthly, pf_ab_none):
    w_m = pf_ab_monthly.weights_ts
    w_n = pf_ab_none.weights_ts
    # Same shape and columns
    assert list(w_m.columns) == ["A.US", "B.US"]
    assert list(w_n.columns) == ["A.US", "B.US"]
    # Lengths relative to ror: monthly rebalancing aligns with ror length; no rebalancing may include a leading row
    rlen = len(pf_ab_monthly.ror)
    assert len(w_m) == rlen
    assert len(w_n) in {rlen, rlen + 1}
    # Monthly rebalancing keeps weights constant across time
    assert np.allclose(w_m.values, np.tile([0.5, 0.5], (len(w_m), 1)))
    # No rebalancing lets weights drift: last row is unlikely equal to initial weights
    assert not np.allclose(w_n.iloc[-1].values, [0.5, 0.5])
    # Each row must sum to 1
    assert np.allclose(w_m.sum(axis=1).values, 1.0)
    assert np.allclose(w_n.sum(axis=1).values, 1.0)


def test_ror_series_exists_and_length(pf_ab_monthly):
    r = pf_ab_monthly.ror
    assert isinstance(r, pd.Series)
    assert len(r) == 24
    assert np.isfinite(r.values).all()


def test_wealth_index_and_with_assets_shapes(pf_three_monthly):
    wi = pf_three_monthly.wealth_index
    wiwa = pf_three_monthly.wealth_index_with_assets
    # wealth index is a 1-column Series-like DataFrame for the portfolio + name
    assert isinstance(wiwa, pd.DataFrame)
    assert isinstance(wi, pd.DataFrame)
    rlen = len(pf_three_monthly.ror)
    # Wealth index helpers may add an initial row (starting wealth) → allow rlen or rlen+1
    assert wi.shape[0] in {rlen, rlen + 1}
    assert wiwa.shape[0] in {rlen, rlen + 1}
    # Columns include portfolio and each asset symbol
    for col in [pf_three_monthly.name, *pf_three_monthly.symbols]:
        assert col in wiwa.columns


def test_mean_return_and_risk_consistency(pf_ab_monthly):
    mr_m = pf_ab_monthly.mean_return_monthly
    mr_a = pf_ab_monthly.mean_return_annual
    rk_m_ts = pf_ab_monthly.risk_monthly
    rk_a_ts = pf_ab_monthly.risk_annual
    rk_m = float(rk_m_ts.iloc[-1])
    rk_a = float(rk_a_ts.iloc[-1])
    assert isinstance(mr_m, float) and isinstance(mr_a, float)
    assert isinstance(rk_m, float) and isinstance(rk_a, float)
    # Annualization rules
    assert pytest.approx(mr_a, rel=1e-12) == 12 * mr_m
    # Portfolio.risk_annual uses helpers.Float.annualize_risk with expanding mean
    mean_last = float(pf_ab_monthly.ror.expanding().mean().iloc[-1])
    expected_rk_a = ok.common.helpers.helpers.Float.annualize_risk(rk_m, mean_last)
    assert pytest.approx(rk_a, rel=1e-6, abs=1e-10) == expected_rk_a


def test_semideviation_and_tail_risks(pf_three_monthly):
    sd_m = pf_three_monthly.semideviation_monthly
    sd_a = pf_three_monthly.semideviation_annual
    var12 = pf_three_monthly.get_var_historic(time_frame=12, level=1)
    cvar12 = pf_three_monthly.get_cvar_historic(time_frame=12, level=1)
    assert isinstance(sd_m, float) and sd_m >= 0
    assert isinstance(sd_a, float) and sd_a >= 0
    assert isinstance(var12, float)
    assert isinstance(cvar12, float)


def test_drawdowns_and_recovery_period(pf_three_monthly):
    dd = pf_three_monthly.drawdowns
    rp = pf_three_monthly.recovery_period
    # For Portfolio drawdowns is a Series (for AssetList it is a DataFrame)
    assert isinstance(dd, pd.Series)
    # Recovery period is a Series of integers with non-negative values
    assert isinstance(rp, pd.Series)
    assert rp.dtype.kind in {"i", "u"}  # integer types
    assert (rp.values >= 0).all()


def test_cagr_and_cumulative_returns(pf_ab_monthly):
    cagr = pf_ab_monthly.get_cagr()
    cum = pf_ab_monthly.get_cumulative_return()
    # API returns Series with a single value named by portfolio; accept both scalar and 1-len Series
    if isinstance(cagr, pd.Series):
        assert cagr.shape[0] == 1
        _ = float(cagr.iloc[0])
    else:
        assert isinstance(cagr, float)
    if isinstance(cum, pd.Series):
        assert cum.shape[0] == 1
        _ = float(cum.iloc[0])
    else:
        assert isinstance(cum, float)


def test_rolling_cagr_and_cumulative_returns(pf_ab_monthly):
    rc = pf_ab_monthly.get_rolling_cagr(window=12)
    rcr = pf_ab_monthly.get_rolling_cumulative_return(window=12)
    # For Portfolio both functions return a DataFrame with one column (portfolio name)
    assert isinstance(rc, pd.DataFrame)
    assert isinstance(rcr, pd.DataFrame)
    # 24 months -> 24-12+1 values for rolling window
    assert rc.shape[0] == 13
    assert rcr.shape[0] == 13


def test_sharpe_and_sortino_and_diversification(pf_ab_monthly):
    sh = pf_ab_monthly.get_sharpe_ratio(rf_return=0.0)
    so = pf_ab_monthly.get_sortino_ratio(t_return=0.0)
    dr = pf_ab_monthly.diversification_ratio
    assert np.isfinite(sh)
    assert np.isfinite(so)
    assert np.isfinite(dr)


def test_describe_and_table_return_types(pf_ab_monthly):
    d = pf_ab_monthly.describe(years=(1, 2))
    t = pf_ab_monthly.table
    assert isinstance(d, pd.DataFrame)
    assert isinstance(t, pd.DataFrame)
    # table lists assets with names, tickers, weights
    cols = set(t.columns)
    assert {"asset name", "ticker", "weights"}.issubset(cols)


def test_okamaio_link_returns_string(pf_ab_monthly):
    link = pf_ab_monthly.okamaio_link
    assert isinstance(link, str) and len(link) > 0


def test_okamaio_link_serializes_rebalancing_strategy(pf_ab_monthly):
    pf_ab_monthly.rebalancing_strategy = ok.Rebalance(period="quarter", abs_deviation=0.05, rel_deviation=0.1)
    query = parse_qs(urlparse(pf_ab_monthly.okamaio_link).query)

    assert query["rebal"] == ["quarter"]
    assert query["rebalancing_period"] == ["quarter"]
    assert query["rebalancing_abs_deviation"] == ["0.05"]
    assert query["rebalancing_rel_deviation"] == ["0.1"]


def test_okamaio_link_omits_empty_rebalancing_thresholds(pf_ab_monthly):
    query = parse_qs(urlparse(pf_ab_monthly.okamaio_link).query)

    assert query["rebal"] == ["month"]
    assert query["rebalancing_period"] == ["month"]
    assert "rebalancing_abs_deviation" not in query
    assert "rebalancing_rel_deviation" not in query


# ---------------- New tests for additional Portfolio API -----------------


def test_rebalancing_events_and_setter(pf_ab_none):
    # Initially: no rebalancing -> events may be empty
    ev0 = pf_ab_none.rebalancing_events
    assert hasattr(ev0, "index")  # Series-like
    # Change strategy to monthly rebalancing
    pf_ab_none.rebalancing_strategy = ok.Rebalance(period="month")
    ev1 = pf_ab_none.rebalancing_events
    # Expect at least one calendar event within the period
    assert len(ev1) >= 1
    # All values are from the known set
    allowed = {"calendar", "abs", "rel"}
    assert set(ev1.astype(str).unique()).issubset(allowed)
    # Weights become constant at targets with monthly rebalancing
    w = pf_ab_none.weights_ts
    assert np.allclose(w.values, np.tile([0.5, 0.5], (len(w), 1)))


def test_assets_and_portfolio_close_and_counts(pf_ab_monthly):
    # Assets close monthly
    ac = pf_ab_monthly.assets_close_monthly
    assert isinstance(ac, pd.DataFrame)
    assert list(ac.columns) == ["A.US", "B.US"]
    assert (ac.values > 0).all()
    # Portfolio close monthly equals first column of wealth_index
    cm = pf_ab_monthly.close_monthly
    assert isinstance(cm, pd.Series)
    rlen = len(pf_ab_monthly.ror)
    assert cm.shape[0] in {rlen, rlen + 1}
    assert (cm.values > 0).all()
    # Number of securities
    nos = pf_ab_monthly.number_of_securities
    assert isinstance(nos, pd.DataFrame)
    assert list(nos.columns) == ["A.US", "B.US"]
    # The very first row can be NaN (pre-initialization marker); subsequent rows must be finite and non-negative
    if len(nos) > 0:
        assert nos.iloc[1:].map(np.isfinite).values.all()
        assert (nos.iloc[1:].values >= 0).all()


def test_dividends_and_yields_monthly_and_annual(pf_ab_monthly):
    # Monthly portfolio dividends (sum over assets)
    dv = pf_ab_monthly.dividends
    assert isinstance(dv, pd.Series)
    rlen = len(pf_ab_monthly.ror)
    assert len(dv) in {rlen, rlen + 1}
    if len(dv) == rlen + 1:
        # extra leading month is allowed (initial allocation date)
        assert dv.index[0] == (pf_ab_monthly.ror.index[0] - 1)
    assert (dv.values >= 0).all()

    # Monthly LTM dividend yield for portfolio
    dy = pf_ab_monthly.dividend_yield
    assert isinstance(dy, pd.Series)
    assert len(dy) >= 1
    assert np.isfinite(dy.values).all()
    assert (dy.values >= 0).all()

    # Assets monthly dividend yields (LTM)
    ady = pf_ab_monthly.assets_dividend_yield
    assert isinstance(ady, pd.DataFrame)
    assert list(ady.columns) == ["A.US", "B.US"]
    assert (ady.values >= 0).all()

    # Annual sums of dividends (by asset)
    dva = pf_ab_monthly.dividends_annual
    assert isinstance(dva, pd.DataFrame)
    assert list(dva.columns) == ["A.US", "B.US"]
    assert dva.shape[0] >= 1
    assert (dva.values >= 0).all()

    # Annual dividend yields (per asset, year-end LTM)
    dya = pf_ab_monthly.dividend_yield_annual
    assert isinstance(dya, pd.DataFrame)
    assert list(dya.columns) == ["A.US", "B.US"]
    assert dya.shape[0] >= 1
    assert (dya.values >= 0).all()


def test_real_mean_return_raises_without_inflation(pf_ab_monthly):
    with pytest.raises(ValueError, match="Real Return is not defined"):
        _ = pf_ab_monthly.real_mean_return


def test_annual_return_ts_is_series_with_two_years(pf_ab_monthly):
    ar = pf_ab_monthly.annual_return_ts()
    # For 24 months we expect two annual values (2020 and 2021)
    assert isinstance(ar, pd.Series)
    assert len(ar) == 2
    assert ar.notna().all()


def test_percentile_functions(pf_ab_monthly):
    pinv = pf_ab_monthly.percentile_inverse_cagr(years=1, score=0.0)
    assert isinstance(pinv, float)
    assert 0.0 <= pinv <= 100.0

    pc = pf_ab_monthly.percentile_cagr(years=2, percentiles=[10, 50, 90])
    assert isinstance(pc, pd.DataFrame)
    assert list(pc.columns) == [10, 50, 90]
    assert list(pc.index) == [1, 2]
    assert pc.notna().all(axis=None)
