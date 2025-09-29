# tests/test_mc.py
import re

import pytest
from pytest import approx
import pandas as pd


def test_optimize_df_for_students_valid_case(mc_students):
    result = mc_students.optimize_df_for_students(var_level=5)
    assert isinstance(result, float), "The returned optimized degrees of freedom should be a float."
    assert result == approx(3.04, rel=1e-2)


def test_optimize_df_for_students_invalid_var_level_high(mc_students):
    with pytest.raises(ValueError, match=re.escape("var_level must be in [1, 99]")):
        mc_students.optimize_df_for_students(var_level=100)


# moved from core
def test_monte_carlo_returns_ts(mc_normal_small):
    df = mc_normal_small.monte_carlo_returns_ts
    assert df.shape == (12, 10)
    assert df.iloc[-1, :].mean() == approx(0.0156, abs=1e-1)


def test_forecast_monte_carlo_cagr(mc_students):
    dic = mc_students.percentile_distribution_cagr(percentiles=[50])
    assert dic[50] == approx(0.2275, abs=1e-1)


def test_skewness(mc_normal_small):
    assert mc_normal_small.skewness.iloc[-1] == approx(-0.6448, abs=1e-2)


def test_rolling_skewness(mc_normal_small):
    assert mc_normal_small.skewness_rolling(window=24).iloc[-1] == approx(0.1449, abs=1e-1)


def test_kurtosis(mc_normal_small):
    assert mc_normal_small.kurtosis.iloc[-1] == approx(2.7960, rel=1e-2)


def test_kurtosis_rolling(mc_normal_small):
    assert mc_normal_small.kurtosis_rolling(window=24).iloc[-1] == approx(-0.1149, rel=1e-1)


def test_jarque_bera(mc_normal_small):
    assert mc_normal_small.jarque_bera["statistic"] == approx(66.765, rel=1e-1)


# New tests to extend coverage of MonteCarlo

def test_percentile_inverse_cagr_range(mc_students):
    # Should return a percentile between 0 and 100
    p = mc_students.percentile_inverse_cagr(score=0)
    assert isinstance(p, float)
    assert 0.0 <= p <= 100.0


def test_kstest_structure(mc_students):
    res = mc_students.kstest
    assert set(res.keys()) == {"statistic", "p-value"}
    assert isinstance(res["statistic"], float)
    assert isinstance(res["p-value"], float)
    assert 0.0 <= res["p-value"] <= 1.0


def test_kstest_for_all_distributions(mc_students):
    df = mc_students.kstest_for_all_distributions
    assert isinstance(df, pd.DataFrame)
    # Expect rows for all configured distributions
    assert len(df.index) >= 3
    for col in ("statistic", "p-value"):
        assert col in df.columns


def test_model_risk_structure(mc_students):
    res = mc_students.model_risk(var_level=5)
    assert set(res.keys()) == {"delta_arithmetic_mean", "delta_var", "delta_cvar"}
    for k in res:
        assert isinstance(res[k], float)


# Tests for get_parameters_for_distribution

def test_get_parameters_for_distribution_norm_defaults(mc_normal_small):
    # With None parameters, should use historical mean and std
    mc_normal_small.distribution_parameters = (None, None)
    mu, sigma = mc_normal_small.get_parameters_for_distribution()
    assert isinstance(mu, float) and isinstance(sigma, float)
    assert mu == approx(float(mc_normal_small.ror.mean()), rel=1e-12, abs=0)
    assert sigma == approx(float(mc_normal_small.ror.std()), rel=1e-12, abs=0)


def test_get_parameters_for_distribution_norm_partial_override(mc_normal_small):
    # Override only mu, keep sigma from data
    mc_normal_small.distribution_parameters = (0.01, None)
    mu, sigma = mc_normal_small.get_parameters_for_distribution()
    assert mu == approx(0.01, rel=0, abs=0)
    assert sigma == approx(float(mc_normal_small.ror.std()), rel=1e-12, abs=0)


def test_get_parameters_for_distribution_norm_full_override(mc_normal_small):
    # Full pass-through when both params provided
    mc_normal_small.distribution_parameters = (0.02, 0.05)
    mu, sigma = mc_normal_small.get_parameters_for_distribution()
    assert mu == approx(0.02)
    assert sigma == approx(0.05)


def test_get_parameters_for_distribution_lognorm_defaults(mc_lognormal_small):
    # With None parameters, should fit with loc fixed at -1.0
    mc_lognormal_small.distribution_parameters = (None, None, None)
    shape, loc, scale = mc_lognormal_small.get_parameters_for_distribution()
    assert isinstance(shape, float) and isinstance(loc, float) and isinstance(scale, float)
    assert loc == approx(-1.0, rel=0, abs=0)
    assert shape == approx(0.07, abs=1e-02)
    assert scale == approx(1.012, abs=1e-02)


def test_get_parameters_for_distribution_lognorm_full_override(mc_lognormal_small):
    # Full pass-through for lognormal; returned loc must be preserved
    mc_lognormal_small.distribution_parameters = (0.4, -1.0, 0.1)
    shape, loc, scale = mc_lognormal_small.get_parameters_for_distribution()
    assert shape == approx(0.4)
    assert loc == approx(-1.0, rel=0, abs=0)
    assert scale == approx(0.1)


def test_get_parameters_for_distribution_t_defaults(mc_students):
    # With None parameters, should fit t distribution
    mc_students.distribution_parameters = (None, None, None)
    df, loc, scale = mc_students.get_parameters_for_distribution()
    assert isinstance(df, float) and isinstance(loc, float) and isinstance(scale, float)
    assert df > 2  # df must be > 2 for finite variance
    assert scale > 0


def test_get_parameters_for_distribution_t_full_override(mc_students):
    mc_students.distribution_parameters = (5.0, 0.0, 0.02)
    df, loc, scale = mc_students.get_parameters_for_distribution()
    assert df == approx(5.0)
    assert loc == approx(0.0)
    assert scale == approx(0.02)

def test_get_parameters_for_distribution_lognormal_defaults(mc_lognormal_small):
    mc_lognormal_small.distribution_parameters = (None, None, None)
    shape, loc, scale = mc_lognormal_small.get_parameters_for_distribution()
