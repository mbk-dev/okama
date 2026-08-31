"""The extracted pure generator must reproduce MonteCarlo's own draw exactly.

`FinPlan` draws stage returns through `generate_returns_ts` instead of through
`MonteCarlo`, so any drift in how the random generator is consumed would change
every seeded result in the library without a test failing elsewhere.
"""

import numpy as np  # noqa: I001
import pandas as pd
import pytest
import okama as ok
from okama.portfolios import mc


@pytest.mark.parametrize("distribution", ["norm", "lognorm", "t"])
def test_generate_returns_ts_reproduces_monte_carlo_returns_ts(distribution) -> None:
    pf = ok.Portfolio(["MCFTR.INDX"], ccy="RUB", inflation=False, symbol="pf.PF")
    pf.dcf.set_mc_parameters(distribution=distribution, period=2, mc_number=16, seed=42)
    expected = pf.dcf.mc.monte_carlo_returns_ts

    index = pd.period_range(pf.last_date.to_period("M"), periods=2 * 12, freq="M")
    result = mc.generate_returns_ts(
        ror=pf.ror,
        distribution=distribution,
        distribution_parameters=None,
        n_paths=16,
        index=index,
        rng=np.random.default_rng(42),
    )

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("distribution", ["norm", "lognorm", "t"])
def test_resolve_distribution_parameters_matches_monte_carlo(distribution) -> None:
    pf = ok.Portfolio(["MCFTR.INDX"], ccy="RUB", inflation=False, symbol="pf.PF")
    pf.dcf.set_mc_parameters(distribution=distribution, period=2, mc_number=8, seed=0)

    expected = pf.dcf.mc.get_parameters_for_distribution()
    result = mc.resolve_distribution_parameters(pf.ror, distribution, None)

    assert result == expected


def test_resolve_distribution_parameters_rejects_unknown_distribution() -> None:
    ror = pd.Series([0.01, -0.02, 0.03], index=pd.period_range("2020-01", periods=3, freq="M"))
    with pytest.raises(ValueError, match="Unknown distribution"):
        mc.resolve_distribution_parameters(ror, "cauchy", None)
