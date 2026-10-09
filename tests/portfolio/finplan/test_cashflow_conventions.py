"""Cash-flow amount conventions exercised through real FinPlan calculations."""

import pandas as pd
import pytest

import okama as ok


def _forecast_plan(portfolio: ok.Portfolio, flag: bool | None = None) -> ok.FinPlan:
    event = str(portfolio.last_date.to_period("M") + 23)
    kwargs = {} if flag is None else {"time_series_discounted_values": flag}
    strategy = ok.TimeSeriesStrategy(portfolio, time_series_dic={event: -800.0}, **kwargs)
    stage = ok.FinPlanStage(
        portfolio,
        period=2,
        cashflow_parameters=strategy,
        distribution="norm",
        distribution_parameters=(0.0, 0.0),
    )
    return ok.FinPlan(stages=[stage], initial_investment=1000.0, discount_rate=0.21, mc_number=4, seed=7)


def test_timeseries_default_keeps_forecast_amounts_verbatim(equity_portfolio) -> None:
    plan = _forecast_plan(equity_portfolio)
    cash_flow = plan.monte_carlo_cash_flow(remove_if_wealth_index_negative=False)
    assert (cash_flow.iloc[-1] == -800.0).all()
    assert plan.probability_of_success() == 1.0


def test_explicit_forecast_conventions_change_success_on_the_same_plan_inputs(equity_portfolio) -> None:
    nominal = _forecast_plan(equity_portfolio, flag=True)
    indexed = _forecast_plan(equity_portfolio, flag=False)
    nominal_flow = nominal.monte_carlo_cash_flow(remove_if_wealth_index_negative=False)
    indexed_flow = indexed.monte_carlo_cash_flow(remove_if_wealth_index_negative=False)
    assert nominal_flow.iloc[-1, 0] == -800.0
    assert indexed_flow.iloc[-1, 0] == pytest.approx(-800.0 * 1.21 ** (23 / 12))
    assert nominal.probability_of_success() == 1.0
    assert indexed.probability_of_success() == 0.0


@pytest.mark.parametrize("flag, expected", [(False, -800.0), (True, -800.0 * 1.21 ** (23 / 12))])
def test_explicit_backtest_conventions_mirror_the_forecast(equity_portfolio, flag, expected) -> None:
    strategy = ok.TimeSeriesStrategy(
        equity_portfolio,
        time_series_dic={"2021-12": -800.0},
        time_series_discounted_values=flag,
    )
    plan = ok.FinPlan(
        stages=[ok.FinPlanStage(equity_portfolio, period=2, cashflow_parameters=strategy)],
        initial_investment=1000.0,
        discount_rate=0.21,
    )
    cash_flow = plan.cash_flow_ts(first_date="2020-01", remove_if_wealth_index_negative=False)
    assert cash_flow.loc[pd.Period("2021-12", freq="M")] == pytest.approx(expected)


def test_finplan_accepts_indexation_withdrawal_exceeding_strategy_opening_balance(equity_portfolio) -> None:
    strategy = ok.IndexationStrategy(equity_portfolio, frequency="month", indexation=0.0)
    strategy.amount = -2000.0
    stage = ok.FinPlanStage(
        equity_portfolio,
        period=1,
        cashflow_parameters=strategy,
        distribution="norm",
        distribution_parameters=(0.0, 0.0),
    )
    plan = ok.FinPlan(stages=[stage], initial_investment=30_000.0, mc_number=4, seed=7)
    assert strategy.initial_investment == 1000.0
    assert (plan.monte_carlo_wealth().iloc[-1] == 6000.0).all()
    assert plan.probability_of_success() == 1.0
