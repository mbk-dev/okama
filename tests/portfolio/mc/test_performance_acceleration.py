"""Derived MC cache freshness and local statistical work reuse."""

import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import okama as ok
from okama.common.helpers import helpers, tails
from okama.portfolios import mc as mc_module
from tests.helpers import dcf_reference as ref


@pytest.fixture()
def mc(synthetic_env):
    p = ok.Portfolio(["A.US", "B.US"], ccy="USD", inflation=False)
    p.dcf.set_mc_parameters(period=2, mc_number=127, seed=127)
    return p.dcf.mc


@pytest.mark.parametrize("distribution", ["norm", "t", "lognorm"])
def test_backtesting_resolves_distribution_once(mc, mocker, distribution):
    mc.distribution = distribution
    expected = ref.LegacyMonteCarlo.backtesting_error(mc)
    resolve = mocker.spy(mc, "get_parameters_for_distribution")
    actual = mc.backtesting_error()
    assert actual == expected
    assert resolve.call_count == 1


def test_repeated_cagr_queries_reuse_one_derived_distribution(mc, mocker):
    cagr = mocker.spy(helpers.Frame, "get_cagr")
    mc.percentile_distribution_cagr([5, 50, 95])
    mc.percentile_inverse_cagr(0.0)
    mc.percentile_inverse_cagr(0.05)
    assert cagr.call_count == 1


def test_percentiles_are_evaluated_in_one_quantile_call(mc, mocker):
    expected = ref.LegacyMonteCarlo.percentile_distribution_cagr(mc, [5, 50, 95])
    quantile = mocker.spy(pd.Series, "quantile")
    assert mc.percentile_distribution_cagr([5, 50, 95]) == expected
    assert quantile.call_count == 1


@pytest.mark.parametrize("change", ["value", "nan", "columns", "rows", "float32"])
def test_cagr_cache_observes_public_return_matrix_mutation(mc, change):
    mc._get_cagr_distribution()
    returns = mc.monte_carlo_returns_ts
    if change == "value":
        returns.iloc[1, 2] += 0.5
    elif change == "nan":
        returns.iloc[1, 2] = np.nan
    elif change == "columns":
        returns.columns = pd.Index([f"p{i}" for i in returns], name="Path")
    elif change == "rows":
        mc._returns_ts_cache = returns.iloc[:6].copy()
    else:
        mc._returns_ts_cache = returns.astype("float32")
    expected = helpers.Frame.get_cagr(mc.monte_carlo_returns_ts)
    pd.testing.assert_series_equal(mc._get_cagr_distribution(), expected)


def test_cagr_result_is_not_a_mutable_alias_of_cache(mc):
    expected = mc._get_cagr_distribution().copy()
    actual = mc._get_cagr_distribution()
    actual.iloc[:] = -123
    pd.testing.assert_series_equal(mc._get_cagr_distribution(), expected)


@pytest.mark.parametrize("setting,value", [("seed", 99), ("period", 3), ("mc_number", 19), ("distribution", "lognorm")])
def test_cagr_cache_is_invalidated_by_mc_settings(mc, setting, value):
    mc._get_cagr_distribution()
    setattr(mc, setting, value)
    pd.testing.assert_series_equal(mc._get_cagr_distribution(), helpers.Frame.get_cagr(mc.monte_carlo_returns_ts))


@pytest.mark.parametrize("levels", [[], [90, 10, 50, 10], [0, 100], [5]])
def test_percentile_labels_order_and_duplicates_match_original(mc, levels):
    expected = ref.LegacyMonteCarlo.percentile_distribution_cagr(mc, levels)
    assert mc.percentile_distribution_cagr(levels) == expected


def test_plot_bootstrap_does_not_call_python_cvar_per_sample(mc, mocker):
    cvar = mocker.spy(tails, "cvar_of_sample")
    ax = mc.plot_qq(bootstrap_size_var=2000)
    plt.close(ax.figure)
    assert cvar.call_count <= 1


def test_full_plot_matches_original_financial_lines_and_confidence_bands(mc):
    before = ref.LegacyMonteCarlo.plot_qq(mc, bootstrap_size_var=127)
    expected = [(line.get_xdata(), line.get_ydata()) for line in before.lines]
    bands = [(span.get_y(), span.get_height()) for span in before.patches]
    after = mc.plot_qq(bootstrap_size_var=127)
    assert len(after.lines) == len(expected)
    for line, (x, y) in zip(after.lines, expected, strict=True):
        np.testing.assert_allclose(line.get_xdata(), x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(line.get_ydata(), y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose([(p.get_y(), p.get_height()) for p in after.patches], bands, rtol=1e-12, atol=1e-12)
    plt.close(before.figure)
    plt.close(after.figure)


@pytest.mark.parametrize(
    "samples",
    [
        np.array([[1.0, 1.0, 2.0, 3.0], [-2.0, -2.0, 0.0, 3.0]]),
        np.array([[np.nan, 1.0, 2.0, 3.0], [np.nan, np.nan, np.nan, np.nan]]),
        np.zeros((2, 4)),
    ],
)
def test_bootstrap_tail_kernel_preserves_ties_and_nan(samples):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.apply_along_axis(tails.cvar_of_sample, 1, samples, 0.5)
    quantiles, actual = mc_module._bootstrap_tail_statistics(samples, 0.5)
    np.testing.assert_allclose(quantiles, np.quantile(samples, 0.5, axis=1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_cagr_cache_preserves_column_dtype_changes(mc):
    mc._get_cagr_distribution()
    mc.monte_carlo_returns_ts.columns = mc.monte_carlo_returns_ts.columns.astype(float)
    pd.testing.assert_series_equal(mc._get_cagr_distribution(), helpers.Frame.get_cagr(mc.monte_carlo_returns_ts))


def test_percentile_array_matches_list_input(mc):
    levels = np.array([5, 50, 95])
    expected = ref.LegacyMonteCarlo.percentile_distribution_cagr(mc, levels)
    assert mc.percentile_distribution_cagr(levels) == expected


def test_cagr_mixed_return_dtypes_use_fresh_derivation(mc):
    returns = mc.monte_carlo_returns_ts
    returns.iloc[:, :] = 0.125
    mc._get_cagr_distribution()
    returns[0] = returns[0].astype("float32")
    pd.testing.assert_series_equal(mc._get_cagr_distribution(), helpers.Frame.get_cagr(returns), rtol=1e-12, atol=1e-12)
