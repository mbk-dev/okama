import numpy as np  # noqa: I001
import pytest
import okama as ok


def _steady_stage(portfolio, period, monthly_return, strategy=None, name=None):
    """A stage whose returns are exactly `monthly_return` in every scenario."""
    return ok.FinPlanStage(
        portfolio,
        period=period,
        cashflow_parameters=strategy,
        name=name,
        distribution="norm",
        distribution_parameters=(monthly_return, 0.0),
    )


def test_probability_of_success_is_one_when_nothing_is_withdrawn(equity_portfolio, bond_portfolio) -> None:
    plan = ok.FinPlan(
        stages=[
            _steady_stage(equity_portfolio, 5, 0.005, name="grow"),
            _steady_stage(bond_portfolio, 5, 0.002, name="hold"),
        ],
        initial_investment=100_000,
        mc_number=8,
        seed=0,
    )

    assert plan.probability_of_success() == 1.0


def test_probability_of_success_is_zero_when_the_plan_is_drained(equity_portfolio, bond_portfolio) -> None:
    drain = ok.IndexationStrategy(equity_portfolio)
    drain.initial_investment = 60_000  # Must satisfy validator: abs(amount) <= initial_investment
    drain.frequency = "year"
    drain.amount = -60_000
    drain.indexation = 0.0
    pension = ok.IndexationStrategy(bond_portfolio)
    pension.initial_investment = 1_000  # Must satisfy validator: abs(amount) <= initial_investment
    pension.frequency = "year"
    pension.amount = -1_000
    pension.indexation = 0.0
    plan = ok.FinPlan(
        stages=[
            _steady_stage(equity_portfolio, 5, 0.0, strategy=drain),
            _steady_stage(bond_portfolio, 5, 0.0, strategy=pension),
        ],
        initial_investment=100_000,
        mc_number=8,
        seed=0,
    )

    assert plan.probability_of_success() == 0.0


def test_monte_carlo_irr_equals_the_compounded_return_without_cash_flows(equity_portfolio) -> None:
    idle = ok.IndexationStrategy(equity_portfolio)
    idle.frequency = "none"
    plan = ok.FinPlan(
        stages=[_steady_stage(equity_portfolio, 10, 0.01, strategy=idle)],
        initial_investment=10_000,
        mc_number=5,
        seed=0,
    )

    irr = plan.monte_carlo_irr()

    assert len(irr) == 5
    np.testing.assert_allclose(irr.to_numpy(), 1.01**12 - 1, rtol=1e-9)


def test_survival_period_is_the_full_horizon_when_nothing_is_withdrawn(equity_portfolio) -> None:
    idle = ok.IndexationStrategy(equity_portfolio)
    idle.frequency = "none"
    plan = ok.FinPlan(
        stages=[_steady_stage(equity_portfolio, 10, 0.005, strategy=idle)],
        initial_investment=10_000,
        mc_number=5,
        seed=0,
    )

    survival = plan.monte_carlo_survival_period()

    assert len(survival) == 5
    np.testing.assert_allclose(survival.to_numpy(), 10.0, atol=0.15)


def test_balance_percentiles_reports_one_row_per_stage_boundary(equity_portfolio, bond_portfolio) -> None:
    plan = ok.FinPlan(
        stages=[
            _steady_stage(equity_portfolio, 5, 0.005, name="accumulation"),
            _steady_stage(bond_portfolio, 5, 0.002, name="retirement"),
        ],
        initial_investment=100_000,
        mc_number=8,
        seed=0,
    )

    table = plan.balance_percentiles(percentiles=(10, 50, 90))

    assert list(table.index) == ["accumulation", "retirement"]
    assert list(table.columns) == ["date", "10%", "50%", "90%"]
    assert table.loc["accumulation", "50%"] == pytest.approx(100_000 * 1.005**60, rel=1e-9)
    assert table.loc["retirement", "50%"] == pytest.approx(100_000 * 1.005**60 * 1.002**60, rel=1e-9)
