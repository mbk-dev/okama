import numpy as np  # noqa: I001
import pandas as pd
import pytest

import okama as ok


@pytest.fixture()
def pf_mc():
    """Fresh single-asset portfolio per test (offline assets via package conftest)."""
    return ok.Portfolio(
        ["MCFTR.INDX"],
        ccy="RUB",
        first_date="2005-01",
        last_date="2020-01",
        inflation=True,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="pf_seed.PF",
    )


def test_monte_carlo_returns_ts_is_cached(pf_mc):
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=None)
    first = pf_mc.dcf.mc.monte_carlo_returns_ts
    second = pf_mc.dcf.mc.monte_carlo_returns_ts
    pd.testing.assert_frame_equal(first, second)


def test_monte_carlo_seed_is_reproducible(pf_mc):
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=42)
    first = pf_mc.dcf.mc.monte_carlo_returns_ts.copy()
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=42)
    second = pf_mc.dcf.mc.monte_carlo_returns_ts
    pd.testing.assert_frame_equal(first, second)


def test_monte_carlo_different_seed_changes_draw(pf_mc):
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=1)
    first = pf_mc.dcf.mc.monte_carlo_returns_ts.copy()
    pf_mc.dcf.set_mc_parameters(distribution="norm", period=2, mc_number=20, seed=2)
    second = pf_mc.dcf.mc.monte_carlo_returns_ts
    assert not np.allclose(first.to_numpy(), second.to_numpy())


def test_monte_carlo_seed_invalid_type_raises(pf_mc):
    with pytest.raises((TypeError, ValueError)):
        pf_mc.dcf.mc.seed = "not-an-int"
