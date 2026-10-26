"""Financial equivalence and bounded-work requirements for audited DCF paths."""

import numpy as np
import pandas as pd
import pytest

import okama as ok
from okama.common.helpers import helpers
from okama.portfolios import dcf_calculations as calc
from okama.portfolios import mc as mc_module
from tests.helpers import dcf_reference as ref

# The frozen scalar reference intentionally retains pandas 2-compatible insertion.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing integer data to PeriodArray/PeriodIndex is deprecated:Warning:"
    "(tests.helpers.dcf_reference|okama.portfolios.dcf_calculations)"
)


@pytest.fixture()
def dcf(synthetic_env):
    p = ok.Portfolio(["A.US", "B.US"], ccy="USD", inflation=False, symbol="perf.PF")
    strategy = ok.IndexationStrategy(p)
    strategy.initial_investment = 10_000
    strategy.amount, strategy.frequency, strategy.indexation = -50, "month", 0.02
    p.dcf.cashflow_parameters = strategy
    p.dcf.set_mc_parameters(period=2, mc_number=17, seed=127)
    return p.dcf


@pytest.fixture()
def constructor_arrays(monkeypatch):
    arrays = []
    original = pd.DataFrame.__init__

    def tracked(frame, data=None, *args, **kwargs):
        original(frame, data, *args, **kwargs)
        if isinstance(data, np.ndarray):
            arrays.append((frame, data))

    monkeypatch.setattr(pd.DataFrame, "__init__", tracked)
    return arrays


@pytest.mark.parametrize(
    "distribution,parameters", [("norm", (0.005, 0.03)), ("t", (6.0, 0.005, 0.03)), ("lognorm", (0.03, -1.0, 1.005))]
)
def test_generated_frame_owns_fresh_sample_without_duplicate_array(dcf, constructor_arrays, distribution, parameters):
    result = mc_module.generate_returns_ts(
        dcf.mc.ror,
        distribution,
        parameters,
        17,
        pd.period_range("2020-01", periods=24, freq="M"),
        np.random.default_rng(127),
    )
    inputs = [a for frame, a in constructor_arrays if frame is result]
    assert len(inputs) == 1
    assert np.shares_memory(result.to_numpy(), inputs[0])
    assert not np.shares_memory(result.to_numpy(), dcf.mc.ror.to_numpy())


@pytest.mark.parametrize("kind", ["wealth", "cashflow", "ruin"])
def test_owned_simulation_result_does_not_duplicate_array(dcf, constructor_arrays, kind):
    returns = dcf.mc.monte_carlo_returns_ts
    if kind == "wealth":
        result = calc.get_wealth_indexes_fv_with_cashflow_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    elif kind == "cashflow":
        result = calc.get_cash_flow_fv_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    else:
        result = calc.zero_wealth_after_first_void(returns)
    inputs = [a for frame, a in constructor_arrays if frame is result]
    assert len(inputs) == 1
    assert np.shares_memory(result.to_numpy(), inputs[0])
    assert not np.shares_memory(result.to_numpy(), returns.to_numpy())


def test_public_result_mutations_do_not_poison_cached_frames(dcf):
    returns = dcf.mc.monte_carlo_returns_ts.copy()
    expected_w = dcf.monte_carlo_wealth().copy()
    expected_c = dcf.monte_carlo_cash_flow("fv").copy()
    wealth = dcf.monte_carlo_wealth()
    cash = dcf.monte_carlo_cash_flow("fv")
    wealth.iloc[:, :] = -123
    cash.iloc[:, :] = -456
    pd.testing.assert_frame_equal(dcf.monte_carlo_wealth(), expected_w)
    pd.testing.assert_frame_equal(dcf.monte_carlo_cash_flow("fv"), expected_c)
    pd.testing.assert_frame_equal(dcf.mc.monte_carlo_returns_ts, returns)


def test_seeded_wealth_and_cashflow_match_frozen_engine(dcf):
    returns = dcf.mc.monte_carlo_returns_ts
    expected_w = ref.get_wealth_indexes_fv_with_cashflow_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    expected_c = ref.get_cash_flow_fv_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    pd.testing.assert_frame_equal(dcf.monte_carlo_wealth(), expected_w, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(dcf.monte_carlo_cash_flow("fv", False), expected_c, rtol=1e-12, atol=1e-12)


def strategy_for(parent, kind: str, frequency: str, extras: bool = False):
    if kind == "index":
        strategy = ok.IndexationStrategy(parent)
        strategy.amount, strategy.indexation = -100, 0.03
    elif kind == "percentage":
        strategy = ok.PercentageStrategy(parent)
        strategy.percentage = -0.08
    elif kind == "vds":
        strategy = ok.VanguardDynamicSpending(
            parent, percentage=-0.08, min_max_annual_withdrawals=(300.0, 900.0), adjust_min_max=True
        )
    elif kind == "cwd":
        strategy = ok.CutWithdrawalsIfDrawdown(
            parent, amount=-100, indexation=0.03, crash_threshold_reduction=[(0.05, 0.3), (0.15, 1.0)]
        )
    else:
        strategy = ok.TimeSeriesStrategy(parent)
    strategy.initial_investment = 10_000
    strategy.frequency = frequency
    if extras or kind == "time_series":
        strategy.time_series_dic = {"2020-05": 250.0, "2021-05": -70.0}
        strategy.time_series_discounted_values = True
    parent.dcf.cashflow_parameters = strategy
    return strategy


STRATEGIES = (
    [("index", f) for f in ("month", "quarter", "half-year", "year", "none")]
    + [("percentage", f) for f in ("month", "quarter", "half-year", "year", "none")]
    + [("cwd", "month"), ("cwd", "year"), ("vds", "year"), ("time_series", "none")]
)


@pytest.mark.parametrize("kind,frequency", STRATEGIES)
@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("task", ["backtest", "monte_carlo"])
def test_historical_helpers_match_original_strategies(dcf, kind, frequency, extras, task):
    strategy = strategy_for(dcf.parent, kind, frequency, extras)
    ror = pd.Series(
        np.random.default_rng(19).normal(0.005, 0.04, 29),
        index=pd.period_range("2020-03", periods=29, freq="M", name="Date"),
        name="sample.PF",
    )
    expected_w = ref.get_wealth_indexes_fv_with_cashflow(ror.copy(), None, None, strategy, task)
    actual_w = calc.get_wealth_indexes_fv_with_cashflow(ror.copy(), None, None, strategy, task)
    expected_c = ref.get_cash_flow_fv(ror.copy(), None, strategy, task)
    actual_c = calc.get_cash_flow_fv(ror.copy(), None, strategy, task)
    pd.testing.assert_series_equal(actual_w, expected_w, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(actual_c, expected_c, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("frequency", ["month", "quarter", "year", "none"])
def test_historical_inflation_and_column_metadata_are_preserved(dcf, frequency):
    strategy = strategy_for(dcf.parent, "index", frequency, True)
    index = pd.period_range("2020-03", periods=29, freq="M", name="Month")
    ror = pd.DataFrame({"sample.PF": np.full(29, 0.01), "USD.INFL": np.full(29, 0.002)}, index=index)
    expected = ref.get_wealth_indexes_fv_with_cashflow(ror.copy(), "sample.PF", "USD.INFL", strategy, "backtest")
    actual = calc.get_wealth_indexes_fv_with_cashflow(ror.copy(), "sample.PF", "USD.INFL", strategy, "backtest")
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


def test_historical_work_does_not_insert_one_series_value_per_month(dcf, mocker):
    ror = dcf.parent.ror
    inserts = mocker.spy(pd.Series, "__setitem__")
    result = calc.get_wealth_indexes_fv_with_cashflow(ror, dcf.parent.symbol, None, dcf.cashflow_parameters, "backtest")
    assert len(result) == len(ror) + 1
    assert inserts.call_count < 4


def test_scalar_helpers_do_not_evict_unchanged_strategy_caches(dcf):
    sentinel = pd.DataFrame([[123.0]])
    dcf._monte_carlo_wealth_fv = sentinel
    calc.get_wealth_indexes_fv_with_cashflow(
        dcf.parent.ror, dcf.parent.symbol, None, dcf.cashflow_parameters, "backtest"
    )
    calc.get_cash_flow_fv(dcf.parent.ror, dcf.parent.symbol, dcf.cashflow_parameters, "backtest")
    assert dcf._monte_carlo_wealth_fv is sentinel
    dcf.cashflow_parameters = dcf.cashflow_parameters
    assert dcf._monte_carlo_wealth_fv.empty


def test_all_asset_wealth_runs_one_array_simulation(dcf, mocker):
    calls = mocker.spy(calc, "_simulate_paths_mc")
    result = dcf.wealth_index_fv_with_assets
    assert result.shape == (len(dcf.parent.ror) + 1, 3)
    assert calls.call_count == 1


@pytest.mark.parametrize("case", ["nan", "float32", "gap", "nullable", "unsorted"])
@pytest.mark.parametrize("frequency", ["month", "quarter", "year"])
def test_nonstandard_historical_inputs_keep_scalar_behavior(dcf, case, frequency):
    dcf.cashflow_parameters.frequency = frequency
    ror = dcf.parent.ror.copy()
    if case == "nan":
        ror.iloc[2] = np.nan
    elif case == "float32":
        ror = ror.astype("float32")
    elif case == "nullable":
        ror = ror.astype("Float64")
    elif case == "gap":
        ror = ror.drop(ror.index[3])
    else:
        ror = ror.iloc[::-1]
    expected = ref.get_wealth_indexes_fv_with_cashflow(
        ror.copy(), dcf.parent.symbol, None, dcf.cashflow_parameters, "backtest"
    )
    actual = calc.get_wealth_indexes_fv_with_cashflow(
        ror.copy(), dcf.parent.symbol, None, dcf.cashflow_parameters, "backtest"
    )
    pd.testing.assert_series_equal(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kind,frequency",
    [
        ("index", "month"),
        ("index", "year"),
        ("vds", "year"),
        ("cwd", "month"),
        ("percentage", "quarter"),
        ("time_series", "none"),
    ],
)
@pytest.mark.parametrize("inflation", [False, True])
def test_all_asset_wealth_matches_frozen_property(synthetic_env, kind, frequency, inflation):
    p = ok.Portfolio(["IDX.US", "A.US", "B.US"], inflation=inflation, ccy="USD", symbol="perf.PF")
    strategy_for(p, kind, frequency, True)
    expected = ref.LegacyPortfolioDCF.wealth_index_fv_with_assets.fget(p.dcf)
    pd.testing.assert_frame_equal(p.dcf.wealth_index_fv_with_assets, expected, rtol=1e-12, atol=1e-12)


def test_custom_strategy_keeps_overridden_scalar_withdrawals(dcf, mocker):
    class CustomCWD(ok.CutWithdrawalsIfDrawdown):
        def _calculate_withdrawal_size(self, drawdown, withdrawal_without_drawdowns):
            return -7.0

    strategy = CustomCWD(dcf.parent, amount=-100)
    strategy.frequency = "month"
    ror = pd.Series([-0.1, -0.2, 0.0], index=pd.period_range("2020-01", periods=3, freq="M"))
    expected = ref.get_wealth_indexes_fv_with_cashflow(ror.copy(), None, None, strategy, "backtest")
    engine = mocker.spy(calc, "_simulate_paths_mc")
    actual = calc.get_wealth_indexes_fv_with_cashflow(ror.copy(), None, None, strategy, "backtest")
    pd.testing.assert_series_equal(actual, expected)
    assert engine.call_count == 0


def test_same_strategy_setter_still_invalidates_both_historical_caches(dcf):
    dcf.wealth_index("fv")
    dcf.cash_flow_ts("fv")
    assert not dcf._wealth_index_fv.empty and not dcf._cash_flow_fv.empty
    dcf.cashflow_parameters = dcf.cashflow_parameters
    assert dcf._wealth_index_fv.empty and dcf._cash_flow_fv.empty


def test_mc_wealth_and_cashflow_share_one_engine_pass(dcf, mocker):
    _ = dcf.mc.monte_carlo_returns_ts
    engine = mocker.spy(calc, "_simulate_paths_mc")
    dcf.monte_carlo_wealth()
    dcf.monte_carlo_cash_flow("fv")
    assert engine.call_count == 1


def test_mc_input_mutations_invalidate_joint_outputs(dcf):
    strategy_for(dcf.parent, "percentage", "month")
    dcf.monte_carlo_wealth()
    returns = dcf.mc.monte_carlo_returns_ts
    returns.iloc[0, 0] += 0.5
    expected_w = ref.get_wealth_indexes_fv_with_cashflow_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    expected_c = ref.get_cash_flow_fv_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    pd.testing.assert_frame_equal(dcf.monte_carlo_wealth(), expected_w, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(dcf.monte_carlo_cash_flow("fv", False), expected_c, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind,frequency", STRATEGIES)
def test_joint_mc_outputs_match_frozen_engine_and_strategy_mutations(dcf, kind, frequency):
    strategy_for(dcf.parent, kind, frequency, True)
    returns = dcf.mc.monte_carlo_returns_ts
    expected_w = ref.get_wealth_indexes_fv_with_cashflow_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    expected_c = ref.get_cash_flow_fv_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    pd.testing.assert_frame_equal(dcf.monte_carlo_wealth(), expected_w, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(dcf.monte_carlo_cash_flow("fv", False), expected_c, rtol=1e-12, atol=1e-12)
    dcf.cashflow_parameters.initial_investment *= 1.5
    expected_w = ref.get_wealth_indexes_fv_with_cashflow_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    pd.testing.assert_frame_equal(dcf.monte_carlo_wealth(), expected_w, rtol=1e-12, atol=1e-12)


def test_survival_period_does_not_call_python_date_function_per_path(dcf, mocker):
    expected = ref.LegacyPortfolioDCF.monte_carlo_survival_period(dcf, 0.2)
    dates = mocker.spy(helpers.Date, "get_period_length")
    actual = dcf.monte_carlo_survival_period(0.2)
    pd.testing.assert_series_equal(actual, expected)
    assert dates.call_count == 0


def test_forecast_plot_without_backtest_preserves_mc_cache(dcf, mocker):
    import matplotlib.pyplot as plt

    dcf.monte_carlo_wealth()
    engine = mocker.spy(calc, "_simulate_paths_mc")
    ax = dcf.plot_forecast_monte_carlo(backtest=False)
    plt.close(ax.figure)
    dcf.monte_carlo_wealth()
    assert engine.call_count == 0


def test_conventional_sparse_irr_avoids_newton_seed_matrix_work(mocker):
    flows = np.zeros((361, 17))
    flows[0], flows[-1] = -10_000, np.linspace(5_000, 100_000, 17)
    expected = ref.irr_of_cashflow_matrix(flows)
    seeds = mocker.spy(calc, "_irr_initial_guess")
    np.testing.assert_allclose(calc.irr_of_cashflow_matrix(flows), expected, rtol=1e-12, atol=1e-12)
    assert seeds.call_count == 0


@pytest.mark.parametrize("case", ["multiple", "negative_terminal", "no_sign", "extreme", "custom_guess"])
def test_general_irr_retains_original_solver(case):
    flows = np.zeros((25, 4))
    flows[0], flows[-1] = -100, 150
    kwargs = {}
    if case == "multiple":
        flows[2], flows[7] = 200, -400
    elif case == "negative_terminal":
        flows[-1] = -10
    elif case == "no_sign":
        flows[0] = 100
    elif case == "extreme":
        flows[-1] = 1e250
    else:
        kwargs["guess"] = 0.01
    expected = ref.irr_of_cashflow_matrix(flows, **kwargs)
    np.testing.assert_allclose(
        calc.irr_of_cashflow_matrix(flows, **kwargs), expected, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_discount_rate_change_refreshes_discounted_historical_cashflows(dcf):
    strategy = strategy_for(dcf.parent, "index", "month", extras=True)
    dcf.cashflow_parameters = strategy
    dcf.discount_rate = 0.01
    dcf.cash_flow_ts("fv")
    dcf.discount_rate = 0.09
    actual = dcf.cash_flow_ts("fv", False)
    expected = ref.get_cash_flow_fv(dcf.parent.ror, dcf.parent.symbol, strategy, "backtest")
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("rate", [-1.0, -2.0, float("nan"), float("inf")])
def test_exceptional_discount_rate_uses_scalar_history(dcf, mocker, rate):
    dcf.discount_rate = rate
    original = mocker.spy(calc, "_get_wealth_indexes_fv_with_cashflow_scalar")
    try:
        calc.get_wealth_indexes_fv_with_cashflow(
            dcf.parent.ror, dcf.parent.symbol, None, dcf.cashflow_parameters, "backtest"
        )
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        pass
    assert original.call_count == 1


def test_mutable_extra_cashflow_invalidates_joint_mc_outputs(dcf):
    strategy = strategy_for(dcf.parent, "index", "month")
    dcf.cashflow_parameters = strategy
    dcf.monte_carlo_wealth()
    strategy.time_series.loc[dcf.mc.monte_carlo_returns_ts.index[1]] = 250.0
    expected = ref.get_wealth_indexes_fv_with_cashflow_mc(dcf.mc.monte_carlo_returns_ts, strategy, dcf.discount_rate)
    pd.testing.assert_frame_equal(dcf.monte_carlo_wealth(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("frequency", ["quarter", "year"])
def test_mixed_historical_frame_retains_scalar_precision(dcf, frequency):
    strategy = strategy_for(dcf.parent, "index", frequency)
    strategy.initial_investment = 100_000
    index = pd.period_range("2020-01", periods=24, freq="M", name="Date")
    ror = pd.DataFrame(
        {dcf.parent.symbol: np.linspace(-0.04, 0.06, 24).astype("float32"), "USD.INFL": 0.002}, index=index
    )
    expected = ref.get_wealth_indexes_fv_with_cashflow(ror.copy(), dcf.parent.symbol, "USD.INFL", strategy, "backtest")
    actual = calc.get_wealth_indexes_fv_with_cashflow(ror.copy(), dcf.parent.symbol, "USD.INFL", strategy, "backtest")
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("previous_cache", [False, True])
def test_unsupported_mc_draw_mutations_use_fresh_outputs(dcf, previous_cache):
    if previous_cache:
        dcf.monte_carlo_wealth()
    dcf.mc._returns_ts_cache = dcf.mc.monte_carlo_returns_ts.astype("float32")
    dcf.monte_carlo_wealth()
    for offset in (0.125, -0.25):
        dcf.mc.monte_carlo_returns_ts.iloc[1, 0] += offset
        expected = ref.get_wealth_indexes_fv_with_cashflow_mc(
            dcf.mc.monte_carlo_returns_ts, dcf.cashflow_parameters, dcf.discount_rate
        )
        pd.testing.assert_frame_equal(dcf.monte_carlo_wealth(), expected, rtol=1e-12, atol=1e-12)
