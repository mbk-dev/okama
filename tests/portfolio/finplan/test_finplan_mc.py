import pytest

import okama as ok


def _plan(equity_portfolio, bond_portfolio, **kwargs):
    contrib = ok.IndexationStrategy(equity_portfolio)
    contrib.initial_investment = 12_000
    contrib.frequency = "year"
    contrib.amount = 12_000
    contrib.indexation = 0.0

    pension = ok.IndexationStrategy(bond_portfolio)
    pension.initial_investment = 30_000
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


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from okama.portfolios import dcf_calculations, mc as mc_module  # noqa: E402


def test_monte_carlo_wealth_has_one_row_per_month_plus_the_opening_balance(
    equity_portfolio, bond_portfolio
) -> None:
    plan = _plan(equity_portfolio, bond_portfolio)

    wealth = plan.monte_carlo_wealth()

    assert wealth.shape == (plan.period_months + 1, plan.mc_number)
    assert wealth.index[0] == equity_portfolio.last_date.to_period("M") - 1
    assert wealth.index[1] == equity_portfolio.last_date.to_period("M")
    assert (wealth.iloc[0] == plan.initial_investment).all()
    assert wealth.index.is_monotonic_increasing


def test_single_stage_plan_equals_the_engine_on_the_same_draw(equity_portfolio) -> None:
    strategy = ok.IndexationStrategy(equity_portfolio)
    strategy.frequency = "year"
    strategy.initial_investment = 5_000  # ignored by the plan on purpose
    strategy.amount = -5_000
    strategy.indexation = 0.0
    stage = ok.FinPlanStage(equity_portfolio, period=8, cashflow_parameters=strategy)
    plan = ok.FinPlan(stages=[stage], initial_investment=200_000, mc_number=12, seed=3)

    spawned = np.random.SeedSequence(3).spawn(1)[0]
    index = pd.period_range(equity_portfolio.last_date.to_period("M"), periods=96, freq="M")
    ror = mc_module.generate_returns_ts(
        ror=equity_portfolio.ror,
        distribution="norm",
        distribution_parameters=None,
        n_paths=12,
        index=index,
        rng=np.random.default_rng(spawned),
    )
    expected_wealth, _ = dcf_calculations._simulate_paths_mc(
        ror, strategy, plan.discount_rate, initial_balance=np.full(12, 200_000.0)
    )

    np.testing.assert_allclose(
        plan.monte_carlo_wealth().iloc[1:].to_numpy(), expected_wealth, rtol=1e-12
    )


def test_second_stage_opens_at_the_first_stage_terminal_balance(equity_portfolio, bond_portfolio) -> None:
    # A zero-variance second stage makes the handoff exactly observable: with
    # returns of exactly 0 and no cash flow, its first row is the balance it was
    # handed.
    contrib = ok.IndexationStrategy(equity_portfolio)
    contrib.frequency = "none"
    idle = ok.IndexationStrategy(bond_portfolio)
    idle.frequency = "none"
    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=6, cashflow_parameters=contrib),
            ok.FinPlanStage(
                bond_portfolio,
                period=2,
                cashflow_parameters=idle,
                distribution="norm",
                distribution_parameters=(0.0, 0.0),
            ),
        ],
        initial_investment=50_000,
        mc_number=10,
        seed=1,
    )

    wealth = plan.monte_carlo_wealth()
    boundary = plan.stages[0].period_months  # row 0 is the opening balance

    pd.testing.assert_series_equal(
        wealth.iloc[boundary + 1],
        wealth.iloc[boundary].clip(lower=0),
        check_names=False,
    )


def test_a_ruined_scenario_stays_at_zero_through_the_next_stage(equity_portfolio, bond_portfolio) -> None:
    drain = ok.IndexationStrategy(equity_portfolio)
    drain.frequency = "year"
    drain.initial_investment = 60_000
    drain.amount = -60_000  # exhausts a 100 000 balance within the first stage
    drain.indexation = 0.0
    pension = ok.IndexationStrategy(bond_portfolio)
    pension.frequency = "year"
    pension.amount = -1_000
    pension.indexation = 0.0
    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=5, cashflow_parameters=drain),
            ok.FinPlanStage(bond_portfolio, period=5, cashflow_parameters=pension),
        ],
        initial_investment=100_000,
        mc_number=10,
        seed=2,
    )

    masked = plan.monte_carlo_wealth(include_negative_values=False)

    assert (masked.iloc[-1] == 0).all()


def test_the_same_seed_reproduces_the_plan(equity_portfolio, bond_portfolio) -> None:
    first = _plan(equity_portfolio, bond_portfolio, seed=11).monte_carlo_wealth()
    second = _plan(equity_portfolio, bond_portfolio, seed=11).monte_carlo_wealth()

    pd.testing.assert_frame_equal(first, second)


def test_changing_the_seed_invalidates_the_cache(equity_portfolio, bond_portfolio) -> None:
    plan = _plan(equity_portfolio, bond_portfolio, seed=11)
    before = plan.monte_carlo_wealth().copy()

    plan.seed = 12
    after = plan.monte_carlo_wealth()

    assert not before.equals(after)


def test_monte_carlo_cash_flow_has_one_row_per_month(equity_portfolio, bond_portfolio) -> None:
    plan = _plan(equity_portfolio, bond_portfolio)

    cash_flow = plan.monte_carlo_cash_flow()

    assert cash_flow.shape == (plan.period_months, plan.mc_number)
    assert cash_flow.index[0] == equity_portfolio.last_date.to_period("M")


def test_present_values_are_discounted_from_the_plan_start(equity_portfolio, bond_portfolio) -> None:
    plan = _plan(equity_portfolio, bond_portfolio)
    monthly_rate = (1 + plan.discount_rate) ** (1 / 12) - 1

    fv = plan.monte_carlo_wealth(discounting="fv")
    pv = plan.monte_carlo_wealth(discounting="pv")

    # The exponent is continuous across the stage boundary: the last row of the
    # whole plan is discounted by period_months, not by the last stage's length.
    np.testing.assert_allclose(
        pv.iloc[-1].to_numpy(),
        fv.iloc[-1].to_numpy() / (1 + monthly_rate) ** plan.period_months,
        rtol=1e-12,
    )
