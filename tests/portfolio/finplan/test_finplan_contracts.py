import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

import okama as ok
from okama import settings
from okama.portfolios import mc as mc_module


def make_plan(portfolio: ok.Portfolio, **kwargs: Any) -> ok.FinPlan:
    strategy = ok.TimeSeriesStrategy(portfolio, time_series_dic={}, time_series_discounted_values=True)
    stage = ok.FinPlanStage(portfolio, period=1, cashflow_parameters=strategy, name="goals")
    return ok.FinPlan([stage], mc_number=3, seed=42, **kwargs)


@pytest.mark.parametrize("end", ["2025-11", "2025-10"])
def test_rejects_a_stage_with_a_different_history_end(equity_portfolio: ok.Portfolio, end: str) -> None:
    stale = ok.Portfolio(["BND1.US"], ccy="USD", inflation=False, last_date=end)
    with pytest.raises(ValueError, match=f"retirement.*{end}.*2025-12"):
        ok.FinPlan([ok.FinPlanStage(equity_portfolio, 1), ok.FinPlanStage(stale, 1, name="retirement")])


def test_explicit_t0_is_checked_against_the_first_stage(equity_portfolio: ok.Portfolio) -> None:
    with pytest.raises(ValueError, match="goals.*2025-12.*2026-01"):
        make_plan(equity_portfolio, t0="2026-01")


def test_explicit_t0_accepts_a_day_in_the_same_month(equity_portfolio: ok.Portfolio) -> None:
    assert make_plan(equity_portfolio, t0="2025-12-31").t0 == equity_portfolio.last_date


def test_stage_month_indices_include_the_offset(equity_portfolio: ok.Portfolio, bond_portfolio: ok.Portfolio) -> None:
    plan = ok.FinPlan([ok.FinPlanStage(equity_portfolio, 1), ok.FinPlanStage(bond_portfolio, 1)])
    first, second = plan.stage_month_indices
    assert first.equals(pd.period_range("2025-12", "2026-11", freq="M"))
    assert second.equals(pd.period_range("2026-12", "2027-11", freq="M"))


def test_rejects_a_first_stage_key_given_to_the_second_stage(
    equity_portfolio: ok.Portfolio, bond_portfolio: ok.Portfolio
) -> None:
    strategy = ok.TimeSeriesStrategy(bond_portfolio, time_series_dic={"2026-02": -100})
    with pytest.raises(ValueError, match="retirement.*2026-02.*2026-12.*2027-11"):
        ok.FinPlan(
            [
                ok.FinPlanStage(equity_portfolio, 1),
                ok.FinPlanStage(bond_portfolio, 1, strategy, name="retirement"),
            ]
        ).monte_carlo_wealth()


@pytest.mark.parametrize("reader", ["monte_carlo_wealth", "monte_carlo_cash_flow"])
def test_revalidates_in_place_keys_even_when_results_are_cached(equity_portfolio: ok.Portfolio, reader: str) -> None:
    plan = make_plan(equity_portfolio)
    plan.monte_carlo_wealth()
    plan.stages[0].cashflow_parameters.time_series_dic["2026-12"] = -100
    with pytest.raises(ValueError, match="goals.*2026-12.*2025-12.*2026-11"):
        getattr(plan, reader)()


def test_logs_default_discount_rate_and_keeps_it_readable(
    equity_portfolio: ok.Portfolio, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="okama.portfolios.finplan"):
        plan = make_plan(equity_portfolio)
    assert plan.discount_rate == settings.DEFAULT_DISCOUNT_RATE
    assert str(plan.discount_rate) in caplog.text
    assert "DEFAULT_DISCOUNT_RATE" in caplog.text


def test_logs_inflation_discount_rate(equity_portfolio: ok.Portfolio, caplog: pytest.LogCaptureFixture) -> None:
    equity_portfolio.inflation = "USD.INFL"
    equity_portfolio.inflation_ts = pd.Series(0.01, index=equity_portfolio.ror.index)
    with caplog.at_level(logging.INFO, logger="okama.portfolios.finplan"):
        plan = make_plan(equity_portfolio)
    assert plan.discount_rate == pytest.approx(1.01**12 - 1)
    assert "inflation" in caplog.text
    assert str(plan.discount_rate) in caplog.text


def test_explicit_discount_rate_does_not_log_a_substitution(
    equity_portfolio: ok.Portfolio, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="okama.portfolios.finplan"):
        plan = make_plan(equity_portfolio, discount_rate=0.08)
    assert plan.discount_rate == 0.08
    assert not caplog.records


def test_reused_paths_equal_full_seeded_run_after_flows_change(equity_portfolio: ok.Portfolio) -> None:
    plan = make_plan(equity_portfolio)
    paths = plan.draw_return_paths()
    plan.run_monte_carlo(return_paths=paths)
    plan.stages[0].cashflow_parameters.time_series_dic = {"2026-04": -500}
    plan.run_monte_carlo(return_paths=paths)
    reference = make_plan(equity_portfolio)
    reference.stages[0].cashflow_parameters.time_series_dic = {"2026-04": -500}
    pd.testing.assert_frame_equal(plan.monte_carlo_wealth(), reference.monte_carlo_wealth())
    pd.testing.assert_frame_equal(
        plan.monte_carlo_cash_flow(remove_if_wealth_index_negative=False),
        reference.monte_carlo_cash_flow(remove_if_wealth_index_negative=False),
    )


def test_supplied_paths_are_not_redrawn_and_are_copied(
    equity_portfolio: ok.Portfolio, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = make_plan(equity_portfolio)
    paths = plan.draw_return_paths()
    before = paths[0].copy(deep=True)

    def forbid_draw(**kwargs: Any) -> None:
        raise AssertionError("Supplied paths must not be redrawn")

    monkeypatch.setattr(mc_module, "generate_returns_ts", forbid_draw)
    plan.run_monte_carlo(return_paths=paths)
    baseline = plan.monte_carlo_wealth()
    paths[0].iloc[:, :] = 99
    plan.stages[0].cashflow_parameters.time_series_dic["2026-04"] = -10
    plan.run_monte_carlo(return_paths=plan.draw_return_paths())
    after = plan.monte_carlo_wealth()
    assert not after.equals(baseline)
    assert after.iloc[-1].max() < baseline.iloc[-1].max()
    fresh = plan.draw_return_paths()
    pd.testing.assert_frame_equal(fresh[0], before)
    fresh[0].iloc[:, :] = 88
    pd.testing.assert_frame_equal(plan.draw_return_paths()[0], before)


@pytest.mark.parametrize(
    "invalid", ["count", "shape", "index", "columns", "text", "nan", "infinity", "complex", "boolean"]
)
def test_rejects_invalid_supplied_paths(equity_portfolio: ok.Portfolio, invalid: str) -> None:
    plan = make_plan(equity_portfolio)
    paths = list(plan.draw_return_paths())
    if invalid == "count":
        paths = []
    elif invalid == "shape":
        paths[0] = paths[0].iloc[:-1]
    elif invalid == "index":
        paths[0].index = paths[0].index + 1
    elif invalid == "columns":
        paths[0].columns = ["a", "b", "c"]
    elif invalid == "text":
        paths[0] = paths[0].astype(str)
    elif invalid == "complex":
        paths[0] = paths[0].astype(complex)
    elif invalid == "boolean":
        paths[0] = paths[0] > 0
    else:
        paths[0].iloc[0, 0] = np.nan if invalid == "nan" else np.inf
    with pytest.raises((ValueError, TypeError), match="return_paths.*goals|return_paths.*stage"):
        plan.run_monte_carlo(return_paths=paths)


def test_raw_flows_preserve_a_depleting_withdrawal(equity_portfolio: ok.Portfolio) -> None:
    plan = make_plan(equity_portfolio, initial_investment=100)
    plan.stages[0].cashflow_parameters.time_series_dic = {"2025-12": -200, "2026-01": -50}
    paths = [pd.DataFrame(0.0, index=pd.period_range("2025-12", periods=12, freq="M"), columns=range(3))]
    plan.run_monte_carlo(return_paths=paths)
    raw = plan.monte_carlo_cash_flow(remove_if_wealth_index_negative=False)
    assert (raw.loc["2025-12"] == -200).all()
    assert (raw.loc["2026-01"] == -50).all()
    assert (plan.monte_carlo_cash_flow() == 0).all().all()


def test_backtest_rejects_a_key_outside_its_historical_stage(equity_portfolio: ok.Portfolio) -> None:
    plan = make_plan(equity_portfolio)
    plan.stages[0].cashflow_parameters.time_series_dic = {"2025-12": -10}
    with pytest.raises(ValueError, match="goals.*2025-12.*1990-01.*1990-12"):
        plan.wealth_index()


def test_backtest_refreshes_an_in_place_historical_flow(equity_portfolio: ok.Portfolio) -> None:
    plan = make_plan(equity_portfolio)
    plan.stages[0].cashflow_parameters.time_series_dic["1990-01"] = 50
    assert plan.cash_flow_ts(remove_if_wealth_index_negative=False).iloc[0] == 50


def test_reused_multistage_paths_match_seeded_run(equity_portfolio: ok.Portfolio, bond_portfolio: ok.Portfolio) -> None:
    stages = [ok.FinPlanStage(equity_portfolio, 1), ok.FinPlanStage(bond_portfolio, 2)]
    plan = ok.FinPlan(stages, seed=21, mc_number=4)
    paths = plan.draw_return_paths()
    plan.run_monte_carlo(return_paths=paths)
    reference = ok.FinPlan(stages, seed=21, mc_number=4)
    pd.testing.assert_frame_equal(plan.monte_carlo_wealth(), reference.monte_carlo_wealth())


def test_invalid_paths_leave_previous_results_intact(equity_portfolio: ok.Portfolio) -> None:
    plan = make_plan(equity_portfolio)
    baseline = plan.monte_carlo_wealth()
    paths = list(plan.draw_return_paths())
    paths[0].iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        plan.run_monte_carlo(return_paths=paths)
    pd.testing.assert_frame_equal(plan.monte_carlo_wealth(), baseline)
