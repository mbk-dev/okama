import pytest

import okama as ok


def _plan(equity_portfolio, bond_portfolio, **kwargs):
    contrib = ok.IndexationStrategy(equity_portfolio)
    contrib.initial_investment = 100_000
    contrib.frequency = "year"
    contrib.amount = 12_000
    contrib.indexation = 0.0

    pension = ok.IndexationStrategy(bond_portfolio)
    pension.initial_investment = 100_000
    pension.frequency = "year"
    pension.amount = -30_000
    pension.indexation = 0.0

    return ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=10, cashflow_parameters=contrib, name="accumulation"),
            ok.FinPlanStage(bond_portfolio, period=10, cashflow_parameters=pension, name="retirement"),
        ],
        initial_investment=kwargs.pop("initial_investment", 100_000),
        mc_number=kwargs.pop("mc_number", 20),
        seed=kwargs.pop("seed", 7),
        **kwargs,
    )


def test_plan_horizon_is_the_sum_of_stage_periods(equity_portfolio, bond_portfolio) -> None:
    plan = _plan(equity_portfolio, bond_portfolio)

    assert plan.period == 20
    assert plan.period_months == 240


def test_plan_starts_at_the_first_stage_portfolio_last_date(equity_portfolio, bond_portfolio) -> None:
    plan = _plan(equity_portfolio, bond_portfolio)

    assert plan.t0 == equity_portfolio.last_date


def test_plan_rejects_stages_in_different_currencies(equity_portfolio) -> None:
    rub_portfolio = ok.Portfolio(["EQ1.US"], ccy="RUB", inflation=False, symbol="rub.PF")

    with pytest.raises(ValueError, match="currency"):
        ok.FinPlan(
            stages=[
                ok.FinPlanStage(equity_portfolio, period=5),
                ok.FinPlanStage(rub_portfolio, period=5),
            ],
            initial_investment=1_000,
        )


def test_plan_rejects_an_empty_stage_list() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        ok.FinPlan(stages=[], initial_investment=1_000)


def test_plan_rejects_a_non_positive_initial_investment(equity_portfolio) -> None:
    with pytest.raises(ValueError, match="initial_investment"):
        ok.FinPlan(stages=[ok.FinPlanStage(equity_portfolio, period=5)], initial_investment=0)
