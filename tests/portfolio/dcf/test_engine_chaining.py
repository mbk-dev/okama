"""Engine-level behaviour that FinPlan is built on.

The golden invariant lives here rather than at the FinPlan level: a plan draws
each stage from its own random stream, so it can never coincide with one
continuous draw. Chaining the pure engine over two slices of the *same* return
matrix can, and that is what pins the connector.
"""

import numpy as np  # noqa: I001
import pytest
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


def _zero_ror(n_months: int, n_paths: int = 1, start: str = "2022-01") -> pd.DataFrame:
    """Zero returns isolate cash flow behaviour from compounding."""
    index = pd.period_range(start, periods=n_months, freq="M")
    return pd.DataFrame(np.zeros((n_months, n_paths)), index=index)


def test_month_offset_indexes_amount_from_the_plan_start(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -1_200
    ind.indexation = 0.10

    _, cash_flow = dcf_calculations._simulate_paths_mc(
        _zero_ror(24), ind, 0.05, month_offset=36
    )

    # 36 months of plan history = 3 whole annual periods already elapsed.
    assert cash_flow[11, 0] == pytest.approx(-1_200 * 1.10**3)
    assert cash_flow[23, 0] == pytest.approx(-1_200 * 1.10**4)


def test_month_offset_compounds_extra_cash_flow_from_the_plan_start(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    ts = ok.TimeSeriesStrategy(pf)
    ts.initial_investment = 10_000
    ts.time_series_dic = {"2022-01": -1_000}
    rate = 0.05
    monthly_rate = (1 + rate) ** (1 / 12) - 1

    _, cash_flow = dcf_calculations._simulate_paths_mc(
        _zero_ror(12), ts, rate, month_offset=24
    )

    assert cash_flow[0, 0] == pytest.approx(-1_000 * (1 + monthly_rate) ** 24)


def test_month_offset_indexes_vds_min_max_from_the_plan_start(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    vds = ok.VanguardDynamicSpending(
        pf,
        initial_investment=20_000,
        percentage=-0.08,
        indexation=0.10,
        min_max_annual_withdrawals=(500.0, 900.0),
        adjust_min_max=True,
    )

    _, cash_flow = dcf_calculations._simulate_paths_mc(
        _zero_ror(12), vds, 0.05, month_offset=12
    )

    # 8% of 20 000 = 1 600 exceeds the ceiling, so the indexed maximum binds.
    assert cash_flow[11, 0] == pytest.approx(-900 * 1.10**1)


def test_month_offset_must_be_a_whole_number_of_periods(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -1_200
    ind.indexation = 0.05

    with pytest.raises(ValueError, match="whole number"):
        dcf_calculations._simulate_paths_mc(_zero_ror(12), ind, 0.05, month_offset=6)


def test_a_boundary_inside_a_calendar_year_splits_the_annual_cash_flow(synthetic_env) -> None:
    """The invariant is exact only for monthly cash flow — this pins the periodic case.

    A stage boundary that falls inside a calendar year cuts that year's resample
    group in two, and each half gets a pro-rated cash flow. Nothing is lost, but
    the payment moves, so the raw trajectories differ from a single continuous
    run around the boundary.
    """
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -1_200
    ind.indexation = 0.0
    # Start in November so the boundary at month 12 lands inside calendar 2023.
    ror = _zero_ror(24, start="2022-11")

    _, full = dcf_calculations._simulate_paths_mc(ror, ind, 0.05)
    _, first = dcf_calculations._simulate_paths_mc(ror.iloc[:12], ind, 0.05)
    _, second = dcf_calculations._simulate_paths_mc(
        ror.iloc[12:], ind, 0.05, initial_balance=np.full(1, 10_000.0), month_offset=12
    )

    # The two pro-rated halves add up to the one full-period withdrawal.
    chained_2023 = first[first.shape[0] - 1, 0] + second[1, 0]
    assert chained_2023 == pytest.approx(full[13, 0])


def test_cwd_drawdown_peak_resets_at_a_stage_boundary(synthetic_env) -> None:
    """Characterization test: a new stage means a new portfolio and a new peak.

    This pins a deliberate semantic choice rather than driving new code, so it
    passes as soon as the signature exists. It must not be skipped: it is the
    reason the golden invariant is documented as not holding for CWD.
    """
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    cwd = ok.CutWithdrawalsIfDrawdown(
        pf,
        initial_investment=10_000,
        amount=-1_200,
        indexation=0.0,
        crash_threshold_reduction=[(0.05, 0.5)],
    )
    cwd.frequency = "year"
    index = pd.period_range("2022-01", periods=24, freq="M")
    returns = np.zeros((24, 1))
    returns[:6, 0] = -0.05  # a deep drawdown confined to the first stage
    ror = pd.DataFrame(returns, index=index)

    _, first = dcf_calculations._simulate_paths_mc(ror.iloc[:12], cwd, 0.05)
    _, second = dcf_calculations._simulate_paths_mc(
        ror.iloc[12:], cwd, 0.05, initial_balance=np.full(1, 5_000.0), month_offset=12
    )

    # Stage two sees flat returns only, so no drawdown and no reduction: the
    # full withdrawal is taken even though stage one was deeply under water.
    assert second[11, 0] == pytest.approx(-1_200)
    assert first[11, 0] != pytest.approx(-1_200)


def test_vds_starts_each_stage_without_a_previous_withdrawal(synthetic_env) -> None:
    """Characterization test: floor/ceiling do not bind on a stage's first period.

    Same status as the CWD test above — it records an agreed contract, so it is
    green from the start.
    """
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    vds = ok.VanguardDynamicSpending(
        pf,
        initial_investment=20_000,
        percentage=-0.08,
        indexation=0.0,
        floor_ceiling=(-0.025, 0.05),
        adjust_floor_ceiling=True,
    )

    _, cash_flow = dcf_calculations._simulate_paths_mc(
        _zero_ror(12), vds, 0.05, initial_balance=np.full(1, 20_000.0), month_offset=24
    )

    # With last_withdrawal == 0 the ceiling is 0 and the percentage rule applies.
    assert cash_flow[11, 0] == pytest.approx(-20_000 * 0.08)

@pytest.mark.parametrize("frequency", ["month", "year", "quarter"])
def test_backtest_task_matches_the_per_path_reference(synthetic_env, frequency) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = frequency
    ind.amount = -50 if frequency == "month" else -600
    ind.indexation = 0.03
    ind.time_series_dic = {"2020-06": -300, "2021-03": 500}
    pf.dcf.cashflow_parameters = ind

    reference = dcf_calculations.get_wealth_indexes_fv_with_cashflow(
        ror=pf.ror.to_frame(),
        portfolio_symbol=pf.ror.name,
        inflation_symbol=None,
        cashflow_parameters=ind,
        task="backtest",
    )
    wealth, _ = dcf_calculations._simulate_paths_mc(
        pf.ror.to_frame(), ind, pf.dcf.discount_rate, task="backtest"
    )

    np.testing.assert_allclose(wealth[:, 0], reference.iloc[1:].to_numpy(), rtol=1e-11, atol=1e-8)


def test_unknown_task_is_rejected(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, symbol="pf.PF")
    strategy = _contributions(pf, "month")

    with pytest.raises(ValueError, match="task"):
        dcf_calculations._simulate_paths_mc(_zero_ror(6), strategy, 0.05, task="hindcast")
