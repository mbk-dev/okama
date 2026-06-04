"""Behavior of the root-finding solver in PortfolioDCF.find_the_largest_withdrawals_size.

The solver contract: evaluate the ends of withdrawals_range first and stop
early when the root lies outside the range; stop as soon as
error_rel < tolerance_rel; never spend more than iter_max Monte Carlo
simulations; always restore the cash flow parameters.
"""

import pytest  # noqa: I001
import okama as ok


@pytest.fixture()
def dcf_solver(synthetic_env):
    """Single-asset portfolio with IndexationStrategy and a small, seeded Monte Carlo."""
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -1_200
    ind.indexation = 0.05
    pf.dcf.cashflow_parameters = ind
    pf.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=16, seed=0)
    return pf.dcf


def test_stops_after_one_evaluation_when_largest_withdrawal_sustains_goal(dcf_solver) -> None:
    # Max withdrawal is 0.1% of 10_000 = 10 per year: the portfolio trivially
    # survives 3 of 5 years, so the root lies outside withdrawals_range.
    res = dcf_solver.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.0, 0.001),
        target_survival_period=3,
        percentile=50,
        threshold=0,
        tolerance_rel=0.25,
        iter_max=10,
    )
    # The solver must detect this at the range edge with a single simulation
    # instead of burning the whole iteration budget at the same point.
    assert res.solutions.shape[0] == 1
    assert res.success is False
    assert res.withdrawal_rel == pytest.approx(0.001)
    assert dcf_solver.cashflow_parameters.amount == pytest.approx(-1_200)


def test_stops_after_two_evaluations_when_no_withdrawal_in_range_survives(dcf_solver) -> None:
    # Min withdrawal is 50% of 10_000 = 5_000 per year: the portfolio dies in
    # ~2 years, far below the 4-year target, for every value in the range.
    res = dcf_solver.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.5, 1.0),
        target_survival_period=4,
        percentile=50,
        threshold=0,
        tolerance_rel=0.1,
        iter_max=10,
    )
    # Both range ends fail the goal: two simulations are enough to prove
    # there is no root inside the range.
    assert res.solutions.shape[0] == 2
    assert res.success is False
    assert dcf_solver.cashflow_parameters.amount == pytest.approx(-1_200)


def test_budget_limits_number_of_evaluations(dcf_solver) -> None:
    # Guard: with a practically unreachable tolerance the solver must stop
    # at iter_max evaluations (may already pass before the change).
    res = dcf_solver.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.0, 1.0),
        target_survival_period=4,
        percentile=50,
        threshold=0,
        tolerance_rel=0.0001,
        iter_max=4,
    )
    assert res.solutions.shape[0] <= 4
    assert dcf_solver.cashflow_parameters.amount == pytest.approx(-1_200)


def test_converges_on_smooth_maintain_balance_goal(dcf_solver) -> None:
    # Guard: the maintain-balance residual is smooth, the root is inside the
    # range, and the solver must converge within the budget.
    res = dcf_solver.find_the_largest_withdrawals_size(
        goal="maintain_balance_fv",
        withdrawals_range=(0.0, 1.0),
        target_survival_period=3,
        percentile=50,
        threshold=0,
        tolerance_rel=0.01,
        iter_max=12,
    )
    assert res.success is True
    assert res.error_rel < 0.01
    assert res.solutions.shape[0] <= 12
    assert dcf_solver.cashflow_parameters.amount == pytest.approx(-1_200)


def test_rejects_non_positive_iter_max(dcf_solver) -> None:
    # iter_max is the evaluation budget; zero would crash the best-attempt
    # reporting on an empty solutions history.
    with pytest.raises(ValueError, match=r"iter_max"):
        dcf_solver.find_the_largest_withdrawals_size(
            goal="survival_period",
            withdrawals_range=(0.0, 1.0),
            target_survival_period=3,
            percentile=50,
            threshold=0,
            tolerance_rel=0.25,
            iter_max=0,
        )
