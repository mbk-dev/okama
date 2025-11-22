import pytest
import pandas as pd
import numpy as np

import okama as ok
from tests.asset_list.conftest import _FakeAsset, _FakeCurrencyAsset


@pytest.fixture()
def _inflation_env(mocker):
    """
    Prepare a single-asset environment with deterministic monthly inflation.

    Asset A.US has constant monthly return r_a. Inflation is constant monthly r_i.
    We patch ListMaker assets and macro.Inflation to be deterministic and offline.
    """
    idx = pd.period_range("2020-01", periods=24, freq="M")
    r_a = 0.01  # 1% per month asset
    r_i = 0.002  # 0.2% per month inflation

    asset_ror = pd.Series(r_a, index=idx, name="A.US")
    infl_monthly = pd.Series(r_i, index=idx.to_timestamp(how="end"), name="USD.INFL")

    # Patch assets
    fake_assets = {"A.US": _FakeAsset("A.US", asset_ror)}
    mocker.patch("okama.common.make_asset_list.ListMaker._get_asset_obj_dict", return_value=fake_assets)
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=_FakeCurrencyAsset)

    # Patch Inflation so that AssetList(inflation=True) picks our values
    class _FakeInflation:
        def __init__(self, symbol: str, first_date=None, last_date=None):
            self.symbol = symbol
            # first/last dates are taken from the monthly series index
            self.first_date = infl_monthly.index[0].to_period("M").to_timestamp()
            self.last_date = infl_monthly.index[-1].to_period("M").to_timestamp()
            # ListMaker expects PeriodIndex in _add_inflation path
            self.values_monthly = infl_monthly.to_period("M")

    mocker.patch("okama.common.make_asset_list.macro.Inflation", side_effect=_FakeInflation)

    return {
        "index": idx,
        "asset_ror": asset_ror,
        "infl_periodic": infl_monthly.to_period("M"),
        "r_a": r_a,
        "r_i": r_i,
    }


def test_inflation_basic_pipeline_and_properties(_inflation_env):
    al = ok.AssetList(["A.US"], ccy="USD", inflation=True)

    # Inflation attributes should exist and align with our patched series
    assert hasattr(al, "inflation")
    assert hasattr(al, "inflation_ts")
    assert isinstance(al.inflation_ts, pd.Series)
    # Index should be PeriodIndex with monthly frequency and match assets_ror index (inner join applied)
    assert isinstance(al.inflation_ts.index, pd.PeriodIndex)
    assert list(al.inflation_ts.index) == list(al.assets_ror.index)

    # Name of inflation series should be the inflation symbol
    assert isinstance(al.inflation, str) and al.inflation.endswith(".INFL")
    assert al.inflation_ts.name == al.inflation

    # Mean annualized inflation from monthly 0.2% should be close to 12 * 0.002 for small rates
    # (Exact CAGR is tested below; here only sanity check for arithmetic mean annualization in tests)
    mean_annual = float(al.inflation_ts.values.mean() * 12)
    assert pytest.approx(mean_annual, rel=1e-12) == 0.002 * 12


@pytest.mark.parametrize(
    "compute_metric",
    [
        "real_mean_return",  # property: annualized arithmetic mean adjusted by arithmetic mean inflation
        "real_cagr",         # method: get_cagr(real=True) full period
    ],
)
def test_real_metrics_with_inflation_parametrized(_inflation_env, compute_metric):
    al = ok.AssetList(["A.US"], ccy="USD", inflation=True)

    # Helper to compute expected real mean return using arithmetic means (as in tests elsewhere)
    def expected_real_mean_return():
        # Asset annualized arithmetic mean from AssetList
        mean_asset_annual = float(al.mean_return["A.US"])  # already annualized in implementation
        # Inflation annualized arithmetic mean (monthly -> annual by *12)
        infl_mean_annual = float(al.inflation_ts.values.mean() * 12)
        return (1.0 + mean_asset_annual) / (1.0 + infl_mean_annual) - 1.0

    # Helper to compute expected real CAGR over the full period
    def expected_real_cagr_full():
        ror = al.assets_ror["A.US"]
        n = len(ror)
        wi_last = float((1.0 + ror).prod())
        # Annualize to 12 months per year
        cagr_asset = wi_last ** (12.0 / n) - 1.0

        infl = al.inflation_ts
        wi_infl_last = float((1.0 + infl).prod())
        cagr_infl = wi_infl_last ** (12.0 / n) - 1.0
        return (1.0 + cagr_asset) / (1.0 + cagr_infl) - 1.0

    if compute_metric == "real_mean_return":
        got = float(al.real_mean_return["A.US"])  # property should exist for inflation=True case
        exp = expected_real_mean_return()
        assert pytest.approx(got, rel=1e-12) == exp

    elif compute_metric == "real_cagr":
        got = float(al.get_cagr(real=True)["A.US"])  # full-period real CAGR
        exp = expected_real_cagr_full()
        assert pytest.approx(got, rel=1e-12) == exp

    else:
        raise AssertionError("Unknown metric param")
