import numpy as np  # noqa: I001
import pandas as pd
import pytest
import okama as ok


def test_history_window_is_the_intersection_of_the_stage_portfolios(equity_portfolio, bond_portfolio) -> None:
    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=5),
            ok.FinPlanStage(bond_portfolio, period=5),
        ],
        initial_investment=10_000,
    )

    first, last = plan.history_window

    assert first == max(equity_portfolio.first_date, bond_portfolio.first_date)
    assert last == min(equity_portfolio.last_date, bond_portfolio.last_date)


def test_wealth_index_starts_at_the_earliest_common_date(equity_portfolio, bond_portfolio) -> None:
    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=5),
            ok.FinPlanStage(bond_portfolio, period=5),
        ],
        initial_investment=10_000,
    )

    wealth = plan.wealth_index()

    first, _ = plan.history_window
    assert wealth.index[0] == first.to_period("M") - 1
    assert wealth.shape[0] == plan.period_months + 1
    assert list(wealth.columns) == [plan.name]
    assert wealth.iloc[0, 0] == 10_000


def test_wealth_index_rejects_a_plan_longer_than_the_common_history(equity_portfolio, bond_portfolio) -> None:
    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=40),
            ok.FinPlanStage(bond_portfolio, period=40),
        ],
        initial_investment=10_000,
    )

    with pytest.raises(ValueError, match="months of history"):
        plan.wealth_index()


def test_single_stage_backtest_matches_the_portfolio_wealth_index(equity_portfolio) -> None:
    years = len(equity_portfolio.ror) // 12
    strategy = ok.IndexationStrategy(equity_portfolio)
    strategy.initial_investment = 10_000
    strategy.frequency = "year"
    strategy.amount = -400
    strategy.indexation = 0.02
    equity_portfolio.dcf.cashflow_parameters = strategy
    reference = equity_portfolio.dcf.wealth_index(discounting="fv", include_negative_values=True)

    plan = ok.FinPlan(
        stages=[ok.FinPlanStage(equity_portfolio, period=years, cashflow_parameters=strategy)],
        initial_investment=10_000,
    )
    result = plan.wealth_index(discounting="fv", include_negative_values=True)

    np.testing.assert_allclose(
        result[plan.name].to_numpy(),
        reference[equity_portfolio.symbol].to_numpy()[: result.shape[0]],
        rtol=1e-11,
        atol=1e-8,
    )


def test_cash_flow_ts_covers_the_whole_plan_window(equity_portfolio, bond_portfolio) -> None:
    contrib = ok.IndexationStrategy(equity_portfolio)
    contrib.frequency = "year"
    contrib.amount = 1_000
    contrib.indexation = 0.0
    pension = ok.IndexationStrategy(bond_portfolio)
    pension.initial_investment = 2_000
    pension.frequency = "year"
    pension.amount = -2_000
    pension.indexation = 0.0
    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=5, cashflow_parameters=contrib),
            ok.FinPlanStage(bond_portfolio, period=5, cashflow_parameters=pension),
        ],
        initial_investment=50_000,
    )

    cash_flow = plan.cash_flow_ts()
    wealth = plan.wealth_index()

    assert isinstance(cash_flow, pd.Series)
    assert cash_flow.shape[0] == plan.period_months
    assert cash_flow[cash_flow > 0].sum() == pytest.approx(5 * 1_000)
    # The plan starts at plan.initial_investment (50_000), not at the stage
    # strategy's initial_investment (1_000 default for contrib, 2_000 for pension).
    # This catches a regression where _run_backtest reads the wrong value.
    # We check the final balance because the opening row is added separately
    # and would not detect the error.
    assert wealth.iloc[-1, 0] == pytest.approx(55234.089, abs=0.01)


def test_present_values_are_discounted_from_the_plan_start(equity_portfolio, bond_portfolio) -> None:
    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=5),
            ok.FinPlanStage(bond_portfolio, period=5),
        ],
        initial_investment=10_000,
    )
    monthly_rate = (1 + plan.discount_rate) ** (1 / 12) - 1

    fv = plan.wealth_index(discounting="fv")
    pv = plan.wealth_index(discounting="pv")

    # The opening balance (row 0) is undiscounted in both FV and PV modes: it
    # sits one period before the window start, so its discount factor is 1.
    # The Monte Carlo test does not have an opening row, so it checks only the
    # last row; here we verify both the opening and the last.
    assert pv.iloc[0, 0] == fv.iloc[0, 0]
    # The exponent is continuous across the stage boundary: the last row of the
    # whole plan is discounted by period_months, not by the last stage's length.
    np.testing.assert_allclose(
        pv.iloc[-1, 0],
        fv.iloc[-1, 0] / (1 + monthly_rate) ** plan.period_months,
        rtol=1e-12,
    )


def test_cash_flow_ts_masks_withdrawals_after_depletion(equity_portfolio) -> None:
    """
    cash_flow_ts should zero out flows after the wealth index goes to zero.
    This matches monte_carlo_cash_flow behavior and prevents reporting
    withdrawals the plan never funded.
    """
    withdraw = ok.IndexationStrategy(equity_portfolio)
    withdraw.frequency = "year"
    withdraw.initial_investment = 200_000  # Must exceed plan initial_investment
    withdraw.amount = -50_000  # Large enough to deplete
    withdraw.indexation = 0.0
    plan = ok.FinPlan(
        stages=[ok.FinPlanStage(equity_portfolio, period=20, cashflow_parameters=withdraw)],
        initial_investment=100_000,
        mc_number=10,
        seed=42,
    )

    wealth = plan.wealth_index(include_negative_values=False)
    # Default: flows are masked after depletion
    cash_flow_masked = plan.cash_flow_ts()
    # Opt-out: flows continue after depletion
    cash_flow_all = plan.cash_flow_ts(remove_if_wealth_index_negative=False)

    wealth_zero_mask = wealth[plan.name] == 0
    assert wealth_zero_mask.any(), "Test setup: plan must deplete historically"
    # With masking, all flows after depletion are zero
    assert (cash_flow_masked[wealth_zero_mask] == 0).all()
    # Without masking, some flows remain
    assert (cash_flow_all[wealth_zero_mask] != 0).any()


def test_indexation_continues_across_stage_boundary(equity_portfolio, bond_portfolio) -> None:
    """
    Indexation anchors at the plan start, not at each stage start.
    A withdrawal in stage two must be indexed from t0, not from the stage boundary.
    This test would fail if month_offset=0 were passed instead of the cumulative offset.
    """
    contrib = ok.IndexationStrategy(equity_portfolio)
    contrib.frequency = "year"
    contrib.amount = 10_000
    contrib.indexation = 0.0  # No indexation in accumulation

    withdraw = ok.IndexationStrategy(bond_portfolio)
    withdraw.initial_investment = 200_000  # Must exceed plan initial_investment
    withdraw.frequency = "year"
    withdraw.amount = -5_000  # Base withdrawal
    withdraw.indexation = 0.03  # 3% annual indexation

    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=10, cashflow_parameters=contrib, name="accumulation"),
            ok.FinPlanStage(bond_portfolio, period=10, cashflow_parameters=withdraw, name="retirement"),
        ],
        initial_investment=100_000,
        mc_number=10,
        seed=42,
    )

    cash_flow = plan.cash_flow_ts(remove_if_wealth_index_negative=False)

    # First withdrawal of stage two occurs at the first due date after the boundary.
    # Annual withdrawals happen in December, so the first withdrawal after month 120
    # (January 2000) is in December 2000 (month 131 within the 0-indexed series).
    # It should equal amount * (1 + indexation) ** 10, not just amount,
    # because indexation anchors at the plan start.
    stage_two_flows = cash_flow.iloc[120:240]
    first_withdrawal = stage_two_flows[stage_two_flows != 0].iloc[0]
    expected_withdrawal = -5_000 * (1 + 0.03) ** 10

    np.testing.assert_allclose(first_withdrawal, expected_withdrawal, rtol=1e-9)


def test_time_series_flows_compound_across_stage_boundary(equity_portfolio, bond_portfolio) -> None:
    """
    A time_series cash flow landing in stage two must compound with the portfolio's
    returns from its date forward. This exercises task="backtest" + non-zero month_offset
    + non-empty time_series, the only place task is observable through the plan.
    The test would fail if task="backtest" were removed from _run_backtest.
    """
    # Stage 1: simple accumulation with regular contributions
    contrib = ok.IndexationStrategy(equity_portfolio)
    contrib.frequency = "year"
    contrib.amount = 5_000
    contrib.indexation = 0.0

    # Stage 2: TimeSeriesStrategy with a single large contribution in the middle
    ts_strategy = ok.TimeSeriesStrategy(bond_portfolio)
    ts_strategy.initial_investment = 100_000  # Must exceed plan initial_investment
    # Add a one-time contribution of 50,000 in December 2005 (month 191 = 15y 11m from 1990-01)
    ts_strategy.time_series_dic = {"2005-12": 50_000}

    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=10, cashflow_parameters=contrib, name="accumulation"),
            ok.FinPlanStage(bond_portfolio, period=10, cashflow_parameters=ts_strategy, name="retirement"),
        ],
        initial_investment=100_000,
        mc_number=10,
        seed=42,
    )

    wealth = plan.wealth_index(include_negative_values=False)
    cash_flow = plan.cash_flow_ts(remove_if_wealth_index_negative=False)

    # The contribution occurs in December 2005 (month 191 in 0-indexed series from 1990-01)
    contribution_month = pd.Period("2005-12", freq="M")
    assert contribution_month in cash_flow.index, "Test setup: contribution must be in the series"
    assert cash_flow[contribution_month] == 50_000, "Contribution should appear in cash flow"

    # The contribution should compound with bond returns from 2005-12 forward.
    # After the contribution, wealth should reflect both the existing balance and the new money.
    # This is only possible if task="backtest" is passed, enabling time_series handling.
    wealth_before_contrib = wealth.loc[contribution_month - 1, plan.name]
    wealth_at_contrib = wealth.loc[contribution_month, plan.name]

    # Contributions happen at month-end: the month's return is applied first,
    # then the contribution is added.
    bond_ror_at_contrib = bond_portfolio.ror.loc[contribution_month]
    expected_growth = wealth_before_contrib * (1 + bond_ror_at_contrib) + 50_000

    np.testing.assert_allclose(wealth_at_contrib, expected_growth, rtol=1e-9)
