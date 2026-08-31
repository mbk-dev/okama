import numpy as np  # noqa: I001
import pandas as pd
import pytest
import okama as ok


def test_history_window_is_the_intersection_of_the_stage_portfolios(
    equity_portfolio, bond_portfolio
) -> None:
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


def test_wealth_index_rejects_a_plan_longer_than_the_common_history(
    equity_portfolio, bond_portfolio
) -> None:
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

    assert isinstance(cash_flow, pd.Series)
    assert cash_flow.shape[0] == plan.period_months
    assert cash_flow[cash_flow > 0].sum() == pytest.approx(5 * 1_000)
