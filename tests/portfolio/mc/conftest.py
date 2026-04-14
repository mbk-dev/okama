import pytest  # noqa: I001
import okama as ok
from pathlib import Path
import numpy as np
import pandas as pd

import okama.portfolios.cashflow_strategies
import okama.portfolios.dcf

# Use shared test helpers for deterministic offline assets
from tests.helpers.factories import (
    FakeAsset,
    FakeCurrencyAsset,
    make_period_index,
    make_ror_series,
)

data_folder = Path(__file__).parent / "data"


@pytest.fixture(scope="package", autouse=True)
def _mc_offline_asset_patches():
    """Provide deterministic offline assets for MC tests in this package.

    Patches asset loading so that building Portfolio/DCF objects does not hit
    any external data sources. The patch returns FakeAsset instances with
    synthetic monthly returns for requested symbols.
    """

    # Build a long enough monthly index for all MC periods used in tests
    idx = make_period_index(months=600, start="2000-01")

    # Define default synthetic series factory for arbitrary symbols
    def _series_for(symbol: str):
        # Special-case the symbol used in MC tests to ensure realistic heavy tails
        # so that optimize_df_for_students() converges to df ~ 3 as asserted.
        if symbol == "MCFTR.INDX":
            rng = np.random.default_rng(20241122)
            # Student's t with df=3, scaled to reasonable monthly volatility
            data = rng.standard_t(df=3, size=len(idx)) * 0.04
            return pd.Series(data, index=idx, name=symbol)
        base = 0.007 if symbol.endswith(".INDX") else 0.005
        amp = 0.004 if symbol.endswith(".INDX") else 0.003
        return make_ror_series(symbol, idx, base=base, amp=amp)

    # Cache created FakeAssets by symbol
    _cache: dict[str, FakeAsset] = {}

    def _get_or_make(symbol: str):
        if symbol not in _cache:
            _cache[symbol] = FakeAsset(
                symbol, _series_for(symbol), currency="RUB" if symbol.endswith(".INDX") else "USD"
            )
        return _cache[symbol]

    def _filtered_get_dict(symbols, first_date=None, last_date=None):
        res = {}
        for s in symbols:
            if hasattr(s, "symbol"):
                res[s.symbol] = s
            else:
                res[s] = _get_or_make(s)
        return res

    mp = pytest.MonkeyPatch()
    mp.setattr(
        "okama.common.make_asset_list.ListMaker._get_asset_obj_dict",
        staticmethod(_filtered_get_dict),
    )
    mp.setattr(
        "okama.common.make_asset_list.asset.Asset",
        FakeCurrencyAsset,
    )
    # Also patch the global okama.asset.Asset constructor to avoid any network
    # calls from other code paths (e.g., direct Asset construction outside ListMaker)
    mp.setattr(
        "okama.asset.Asset",
        FakeCurrencyAsset,
    )

    try:
        yield {"index": idx}
    finally:
        mp.undo()


# Monte Carlo Scenarios
@pytest.fixture(scope="package")
def mc_normal_small():
    """Monte Carlo fixture with normal distribution for tests."""
    pf = ok.Portfolio(
        assets=["MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf1.PF",
    )
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, discount_rate=None)
    mc = ok.MonteCarlo(parent=pf_dcf, distribution="norm", distribution_parameters=(None, None), period=1, mc_number=10)
    return mc


@pytest.fixture(scope="package")
def mc_lognormal_small():
    """Monte Carlo fixture with lognormal distribution for tests."""
    pf = ok.Portfolio(
        assets=["MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf1.PF",
    )
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, discount_rate=None)
    mc = ok.MonteCarlo(
        parent=pf_dcf, distribution="lognorm", distribution_parameters=(None, None, None), period=1, mc_number=10
    )
    return mc


@pytest.fixture(scope="package")
def mc_students():
    """Monte Carlo fixture with Student's t distribution for tests."""
    pf = ok.Portfolio(
        assets=["MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf1.PF",
    )
    pf_dcf = okama.portfolios.dcf.PortfolioDCF(pf, discount_rate=None)
    mc = ok.MonteCarlo(
        parent=pf_dcf, distribution="t", distribution_parameters=(None, None, None), period=10, mc_number=100
    )
    return mc
