"""Equivalence of the vectorized MC wealth engine with the per-path reference.

`get_wealth_indexes_fv_with_cashflow_mc` must reproduce
`get_wealth_indexes_fv_with_cashflow` (task="monte_carlo") applied per column —
including the period-fraction scaling for partial periods and the known quirks
of the reference implementation (no return in the first month of a period with
extra cash flows; VDS last_withdrawal pinned to 0). The VDS cases also pin the first-period initialization (last_withdrawal == 0).
"""

import pandas as pd  # noqa: I001
import pytest
import okama as ok
from okama.portfolios import dcf_calculations


def _make_portfolio(last_date=None):
    kwargs = {"ccy": "USD", "inflation": False, "rebalancing_strategy": ok.Rebalance(period="month")}
    if last_date is not None:
        kwargs["last_date"] = last_date
    return ok.Portfolio(["A.US"], **kwargs)


def _indexation(pf, frequency):
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = frequency
    ind.amount = -300 if frequency == "month" else -1_200
    ind.indexation = 0.05
    return ind


def _percentage(pf, frequency):
    pc = ok.PercentageStrategy(pf)
    pc.initial_investment = 10_000
    pc.frequency = frequency
    pc.percentage = -0.08
    return pc


def _time_series_only(pf, extra_dic):
    ts = ok.TimeSeriesStrategy(pf)
    ts.initial_investment = 10_000
    ts.time_series_dic = extra_dic
    return ts


def _cwd(pf, frequency):
    cwd = ok.CutWithdrawalsIfDrawdown(
        pf,
        initial_investment=10_000,
        amount=-1_000,
        indexation=0.02,
        crash_threshold_reduction=[(0.05, 0.3), (0.15, 1.0)],
    )
    cwd.frequency = frequency
    return cwd


def _vds(pf):
    return ok.VanguardDynamicSpending(
        pf,
        initial_investment=10_000,
        percentage=-0.08,
        indexation=0.0,
        min_max_annual_withdrawals=(500.0, 900.0),
        adjust_min_max=True,
    )


def _vds_indexed(pf):
    return ok.VanguardDynamicSpending(
        pf,
        initial_investment=10_000,
        percentage=-0.08,
        indexation=0.03,
        min_max_annual_withdrawals=(500.0, 900.0),
        adjust_min_max=True,
    )


def _vds_floor_ceiling(pf):
    return ok.VanguardDynamicSpending(
        pf,
        initial_investment=10_000,
        percentage=-0.08,
        indexation=0.03,
        floor_ceiling=(-0.025, 0.05),
        adjust_floor_ceiling=True,
        min_max_annual_withdrawals=(500.0, 900.0),
        adjust_min_max=True,
    )


# Each case: (case_id, strategy_builder(pf) -> CashFlow, last_date, mc_period_years, extra_dic)
# extra_dic dates must fall inside the Monte Carlo window (it starts the month
# after the portfolio last_date: 2022-01 for full history, 2021-07 for last_date="2021-06").
CASES = [
    ("indexation_month", lambda pf: _indexation(pf, "month"), None, 3, None),
    ("indexation_year", lambda pf: _indexation(pf, "year"), None, 3, None),
    ("indexation_quarter", lambda pf: _indexation(pf, "quarter"), None, 3, None),
    ("indexation_none", lambda pf: _indexation(pf, "none"), None, 3, None),
    ("indexation_year_partial_periods", lambda pf: _indexation(pf, "year"), "2021-06", 3, None),
    ("indexation_year_extra_cf", lambda pf: _indexation(pf, "year"), None, 3, {"2022-03": -500, "2023-11": 1_000}),
    ("percentage_month", lambda pf: _percentage(pf, "month"), None, 3, None),
    ("percentage_year", lambda pf: _percentage(pf, "year"), None, 3, None),
    ("percentage_halfyear_partial", lambda pf: _percentage(pf, "half-year"), "2021-06", 3, None),
    ("time_series_only", lambda pf: _time_series_only(pf, {"2022-03": -500, "2022-11": 1_000}), None, 3, None),
    ("cwd_year", lambda pf: _cwd(pf, "year"), None, 3, None),
    ("cwd_month", lambda pf: _cwd(pf, "month"), None, 3, None),
    ("vds_year", lambda pf: _vds(pf), None, 3, None),
    ("vds_year_partial", lambda pf: _vds(pf), "2021-06", 3, None),
    ("vds_year_indexed", lambda pf: _vds_indexed(pf), None, 3, None),
    ("vds_year_floor_ceiling", lambda pf: _vds_floor_ceiling(pf), None, 3, None),
    ("vds_year_floor_ceiling_partial", lambda pf: _vds_floor_ceiling(pf), "2021-06", 3, None),
]


def _reference_wealth(dcf, params):
    """Old per-column engine output (the ground truth)."""
    return_ts = dcf.mc.monte_carlo_returns_ts
    return return_ts.apply(
        dcf_calculations.get_wealth_indexes_fv_with_cashflow,
        axis=0,
        args=(None, None, params, "monte_carlo"),
    )


@pytest.mark.parametrize(("case_id", "builder", "last_date", "period", "extra_dic"), CASES, ids=[c[0] for c in CASES])
def test_vectorized_engine_matches_per_path_reference(
    synthetic_env, case_id, builder, last_date, period, extra_dic
) -> None:
    pf = _make_portfolio(last_date)
    params = builder(pf)
    if extra_dic is not None:
        params.time_series_dic = extra_dic
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=period, mc_number=8, seed=0)

    reference = _reference_wealth(pf.dcf, params)
    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(
        pf.dcf.mc.monte_carlo_returns_ts, params, pf.dcf.discount_rate
    )

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8)


def test_vectorized_engine_discounted_extra_cashflows(synthetic_env) -> None:
    # time_series_discounted_values=True skips the FV compounding of extra cash flows.
    pf = _make_portfolio()
    params = _percentage(pf, "quarter")
    params.time_series_dic = {"2022-05": -700}
    params.time_series_discounted_values = True
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=8, seed=0)

    reference = _reference_wealth(pf.dcf, params)
    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(
        pf.dcf.mc.monte_carlo_returns_ts, params, pf.dcf.discount_rate
    )

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8)


def test_vectorized_engine_single_path_short_horizon(synthetic_env) -> None:
    # Edge shapes: one Monte Carlo path (single column) and a single resample group.
    pf = _make_portfolio()
    params = _indexation(pf, "year")
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=1, mc_number=1, seed=0)

    reference = _reference_wealth(pf.dcf, params)
    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(
        pf.dcf.mc.monte_carlo_returns_ts, params, pf.dcf.discount_rate
    )

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8)


def test_monte_carlo_wealth_matches_per_path_reference(synthetic_env) -> None:
    # Guard for the routing: the public API output must stay equal to the
    # old per-path computation (passes before and after the engine switch).
    pf = _make_portfolio()
    params = _indexation(pf, "year")
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=3, mc_number=8, seed=0)

    reference = _reference_wealth(pf.dcf, params)
    pf.dcf.cashflow_parameters = params  # restore after the reference run side effects
    pf.dcf.cashflow_parameters._clear_cf_cache()
    result = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=True)

    pd.testing.assert_frame_equal(result, reference, check_exact=False, rtol=1e-12, atol=1e-8)


def test_monte_carlo_wealth_zeroes_paths_after_first_void(synthetic_env) -> None:
    # Guard for the vectorized negative-value masking: once a path hits a
    # non-positive value, that month and everything after must read 0.
    pf = _make_portfolio()
    params = _indexation(pf, "year")
    params.amount = -3_000  # large withdrawal: paths void within the horizon
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=8, seed=0)

    fv = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=True)
    masked = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=False)

    expected = fv.apply(dcf_calculations.remove_negative_values, axis=0).fillna(0)
    pd.testing.assert_frame_equal(masked, expected, check_exact=False, rtol=1e-12, atol=1e-8)
    assert (masked.to_numpy() >= 0).all()


def test_survival_dates_frame_matches_per_column_scan(synthetic_env) -> None:
    # Guard: the DataFrame overload of get_survival_date must equal the
    # per-column Series overload for both threshold modes.
    from okama.common.helpers import helpers

    pf = _make_portfolio()
    params = _indexation(pf, "year")
    params.amount = -3_000
    pf.dcf.cashflow_parameters = params
    pf.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=8, seed=0)
    wealth = pf.dcf.monte_carlo_wealth(discounting="fv", include_negative_values=False)

    for threshold in (0, 0.1):
        frame_result = helpers.Frame.get_survival_date(wealth, pf.dcf.discount_rate, threshold)
        per_column = pd.Series(
            {col: helpers.Frame.get_survival_date(wealth[col], pf.dcf.discount_rate, threshold) for col in wealth},
        )
        pd.testing.assert_series_equal(frame_result, per_column, check_names=False, check_dtype=False)


def test_survival_dates_frame_rejects_empty_input() -> None:
    # The vectorized scan needs at least one row; fail with a clear message
    # instead of numpy's obscure argmax error.
    from okama.common.helpers import helpers

    empty = pd.DataFrame(columns=[0, 1], dtype=float)
    with pytest.raises(ValueError, match=r"at least one row"):
        helpers.Frame.get_survival_date(empty, 0.05, 0)
