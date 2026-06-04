"""Reproductions and regression tests for reference-engine fixes (issues #81, #82).

All tests build deterministic return series by hand and compute the expected
balances with the canonical recursion `balance = balance * (1 + r) + cash_flow`
applied to every month.
"""

import numpy as np  # noqa: I001
import pandas as pd
import pytest
import okama as ok
from okama.portfolios import dcf_calculations


@pytest.fixture()
def pf_single(synthetic_env):
    return ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))


def test_wealth_index_applies_first_month_return_in_period_with_extra_cash_flows(pf_single) -> None:
    # Issue #81 (wealth side): with an extra cash flow anywhere in a resample
    # period, the first month of that period must still earn its return.
    ind = ok.IndexationStrategy(pf_single)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = 0  # no regular withdrawals: isolate the in-period recursion
    ind.indexation = 0.0
    ind.time_series_dic = {"2022-04": -500}
    ind.time_series_discounted_values = True  # keep the extra flow at face value
    pf_single.dcf.cashflow_parameters = ind

    idx = pd.period_range("2022-01", periods=12, freq="M")
    ror = pd.Series(0.01, index=idx)

    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow(ror, None, None, ind, "monte_carlo")

    balance = 10_000.0
    expected = {}
    for date, r in ror.items():
        balance = balance * (1 + r) + (-500.0 if str(date) == "2022-04" else 0.0)
        expected[date] = balance

    assert result.loc[idx[0]] == pytest.approx(expected[idx[0]])  # 10_100, not 10_000
    assert result.loc[idx[-1]] == pytest.approx(expected[idx[-1]])


def test_cash_flow_fv_sizes_next_period_from_fully_compounded_balance(pf_single) -> None:
    # Issue #81 (cash-flow side): the internal balance tracking must include
    # both the first month's return and the extra cash flow, otherwise the
    # next period's percentage withdrawal is sized from a wrong balance.
    pc = ok.PercentageStrategy(pf_single)
    pc.initial_investment = 10_000
    pc.frequency = "year"
    pc.percentage = -0.12
    pc.time_series_dic = {"2022-01": -500}
    pc.time_series_discounted_values = True
    pf_single.dcf.cashflow_parameters = pc

    idx = pd.period_range("2022-01", periods=24, freq="M")
    ror = pd.Series(0.0, index=idx)  # zero returns keep the math exact

    result = dcf_calculations.get_cash_flow_fv(ror, None, pc, "monte_carlo")

    # Year 2022: start 10_000, extra -500 in January, regular withdrawal
    # -0.12 * 10_000 = -1_200 at year end -> 2023 starts at 8_300.
    # Year 2023: regular withdrawal -0.12 * 8_300 = -996 at year end.
    assert result.loc[pd.Period("2023-12", freq="M")] == pytest.approx(-996.0)


def test_vds_floor_ceiling_binds_in_wealth_index(pf_single) -> None:
    # Issue #82: with floor_ceiling=(0, 0) every withdrawal must equal the
    # previous one. With zero returns: 10_000 -> withdrawal 1_000 (first
    # period, by percentage) -> 9_000 -> withdrawal forced to 1_000 again
    # (not 0.10 * 9_000 = 900) -> 8_000.
    vds = ok.VanguardDynamicSpending(
        pf_single,
        initial_investment=10_000,
        percentage=-0.10,
        floor_ceiling=(0.0, 0.0),
        adjust_floor_ceiling=False,
        indexation=0.0,
    )
    pf_single.dcf.cashflow_parameters = vds

    idx = pd.period_range("2022-01", periods=24, freq="M")
    ror = pd.Series(0.0, index=idx)

    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow(ror, None, None, vds, "monte_carlo")

    assert result.loc[pd.Period("2022-12", freq="M")] == pytest.approx(9_000.0)
    assert result.loc[pd.Period("2023-12", freq="M")] == pytest.approx(8_000.0)


@pytest.mark.parametrize("floor_ceiling", [None, (-0.025, 0.05)])
@pytest.mark.parametrize("min_max", [None, (500.0, 900.0)])
@pytest.mark.parametrize("adjust_floor_ceiling", [False, True])
def test_vds_withdrawal_vector_matches_scalar(pf_single, floor_ceiling, min_max, adjust_floor_ceiling) -> None:
    vds = ok.VanguardDynamicSpending(
        pf_single,
        initial_investment=10_000,
        percentage=-0.08,
        floor_ceiling=floor_ceiling,
        adjust_floor_ceiling=adjust_floor_ceiling,
        min_max_annual_withdrawals=min_max,
        adjust_min_max=True,
        indexation=0.04,
    )
    balances = np.array([100.0, 4_000.0, 8_000.0, 12_000.0, 20_000.0])
    last_withdrawals = np.array([0.0, -300.0, -700.0, -900.0, -1_500.0])
    for n in (0, 3):
        vector = dcf_calculations._vds_withdrawal_vector(vds, balances, last_withdrawals, n)
        scalar = np.array(
            [
                vds._calculate_withdrawal_size(last_withdrawal=lw, balance=b, number_of_periods=n)
                for b, lw in zip(balances, last_withdrawals, strict=True)
            ]
        )
        np.testing.assert_allclose(vector, scalar, rtol=1e-12)
