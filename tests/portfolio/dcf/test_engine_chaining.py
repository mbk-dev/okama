"""Engine-level behaviour that FinPlan is built on.

The golden invariant lives here rather than at the FinPlan level: a plan draws
each stage from its own random stream, so it can never coincide with one
continuous draw. Chaining the pure engine over two slices of the *same* return
matrix can, and that is what pins the connector.
"""

import numpy as np  # noqa: I001
import pandas as pd
import okama as ok
from okama.portfolios import dcf_calculations


def _ror(n_months: int = 24, n_paths: int = 8, start: str = "2022-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.period_range(start, periods=n_months, freq="M")
    return pd.DataFrame(rng.normal(0.005, 0.03, (n_months, n_paths)), index=index)


def _contributions(pf, frequency: str = "month") -> ok.IndexationStrategy:
    """Contributions only, so no scenario can ruin and raw arrays stay comparable."""
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = frequency
    ind.amount = 100
    ind.indexation = 0.0
    return ind


def test_chained_simulation_matches_single_run_for_monthly_cash_flow(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    strategy = _contributions(pf, "month")
    ror = _ror()

    full_wealth, full_cash_flow = dcf_calculations._simulate_paths_mc(ror, strategy, 0.05)
    wealth_1, cash_flow_1 = dcf_calculations._simulate_paths_mc(ror.iloc[:12], strategy, 0.05)
    wealth_2, cash_flow_2 = dcf_calculations._simulate_paths_mc(
        ror.iloc[12:], strategy, 0.05, initial_balance=wealth_1[-1]
    )

    np.testing.assert_allclose(np.vstack([wealth_1, wealth_2]), full_wealth, rtol=1e-12)
    np.testing.assert_allclose(np.vstack([cash_flow_1, cash_flow_2]), full_cash_flow, rtol=1e-12)


def test_initial_balance_none_keeps_the_strategy_value(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    strategy = _contributions(pf, "none")
    ror = _ror(n_months=1, n_paths=3)

    wealth, _ = dcf_calculations._simulate_paths_mc(ror, strategy, 0.05)

    expected = 10_000 * (1 + ror.to_numpy()[0])
    np.testing.assert_allclose(wealth[0], expected, rtol=1e-12)


def test_initial_balance_accepts_a_per_scenario_vector(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    strategy = _contributions(pf, "none")
    ror = _ror(n_months=1, n_paths=3)
    opening = np.array([1_000.0, 2_000.0, 3_000.0])

    wealth, _ = dcf_calculations._simulate_paths_mc(ror, strategy, 0.05, initial_balance=opening)

    np.testing.assert_allclose(wealth[0], opening * (1 + ror.to_numpy()[0]), rtol=1e-12)
