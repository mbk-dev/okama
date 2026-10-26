"""Scalar equivalence and repeated-work regressions for GMV and rolling CAGR."""

import numpy as np
import pandas as pd
import pytest
from pandas.core.window.rolling import Rolling

import okama as ok
from okama.common.helpers import helpers
from okama.frontier import single_period
from tests.helpers.gmv_reference import ScalarGMV


@pytest.fixture()
def frontier(synthetic_env) -> ok.EfficientFrontierSingle:
    return ok.EfficientFrontierSingle(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)


@pytest.mark.parametrize("property_name", ["gmv_monthly_weights", "gmv_annual_weights"])
def test_gmv_prepares_covariance_once(frontier, mocker, property_name: str) -> None:
    cov = mocker.spy(frontier.assets_ror, "cov")
    mean = mocker.spy(frontier.assets_ror, "mean")
    result = getattr(frontier, property_name)
    assert np.isclose(sum(result), 1)
    assert cov.call_count == 1
    assert mean.call_count <= 1


@pytest.mark.parametrize("property_name", ["gmv_monthly", "gmv_annualized"])
def test_gmv_result_runs_one_optimization(frontier, mocker, property_name: str) -> None:
    solver = mocker.spy(single_period, "minimize")
    result = getattr(frontier, property_name)
    assert len(result) == 2
    assert solver.call_count == 1


@pytest.mark.parametrize(
    "property_name", ["gmv_monthly_weights", "gmv_annual_weights", "gmv_monthly", "gmv_annualized"]
)
@pytest.mark.parametrize("missing", [False, True])
def test_gmv_matches_original_scalar_objective(frontier, property_name: str, missing: bool) -> None:
    if missing:
        frontier.assets_ror.iloc[1:3, 0] = np.nan
        frontier.assets_ror.iloc[4, 1] = np.nan
    frontier.bounds = ((0.05, 0.5), (0.1, 0.7), (0.0, 0.9))
    expected = getattr(ScalarGMV(frontier.assets_ror, frontier.bounds), property_name)
    np.testing.assert_allclose(getattr(frontier, property_name), expected, rtol=1e-12, atol=1e-12)


def test_gmv_preparation_does_not_cache_mutable_data(frontier) -> None:
    original = frontier.gmv_monthly_weights.copy()
    frontier.assets_ror.iloc[:, 0] *= 20
    expected = ScalarGMV(frontier.assets_ror, frontier.bounds).gmv_monthly_weights
    np.testing.assert_allclose(frontier.gmv_monthly_weights, expected, rtol=1e-12, atol=1e-12)
    assert not np.allclose(original, expected)


@pytest.mark.parametrize("property_name", ["gmv_monthly_weights", "gmv_annual_weights"])
def test_gmv_keeps_unsuccessful_solver_error(frontier, property_name: str) -> None:
    frontier.bounds = ((0.7, 1.0),) * 3
    with pytest.raises(RuntimeError, match="No solutions were found"):
        getattr(frontier, property_name)


@pytest.mark.parametrize("kind", ["assets", "portfolio"])
@pytest.mark.parametrize("real", [False, True])
def test_public_rolling_cagr_avoids_python_window_callbacks(synthetic_env, mocker, kind: str, real: bool) -> None:
    cls = ok.AssetList if kind == "assets" else ok.Portfolio
    obj = cls(["A.US", "B.US"], ccy="USD", inflation=True)
    callback = mocker.spy(Rolling, "apply")
    result = obj.get_rolling_cagr(window=12, real=real)
    assert not result.empty
    assert callback.call_count == 0


def returns_frame(months: int = 120, assets: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        np.random.default_rng(127).normal(0.005, 0.04, (months, assets)),
        index=pd.period_range("1970-03", periods=months, freq="M", name="Date"),
        columns=pd.Index([f"A{i}" for i in range(assets)], name="Asset"),
    )


@pytest.mark.parametrize("window", [12, 13, 60, 120])
@pytest.mark.parametrize("series", [False, True])
@pytest.mark.parametrize("case", ["normal", "nan", "zero", "loss", "negative", "inf", "float32", "nullable", "object"])
def test_rolling_cagr_matches_original_product(case: str, window: int, series: bool) -> None:
    ror = returns_frame()
    if case == "nan":
        ror.iloc[16, 1] = np.nan
    elif case == "zero":
        ror.iloc[:, :] = 0.0
    elif case == "loss":
        ror.iloc[17, 1] = -1.0
    elif case == "negative":
        ror.iloc[17:19, 1] = -1.5
    elif case == "inf":
        ror.iloc[17, 1] = np.inf
    elif case in ("float32", "nullable", "object"):
        ror = ror.astype({"float32": "float32", "nullable": "Float64", "object": "object"}[case])
    if series:
        ror = ror.iloc[:, 1].rename("portfolio")
    with np.errstate(over="ignore", invalid="ignore"):
        expected = helpers.Frame.get_rolling_fn(ror, window, helpers.Frame.get_cagr)
        actual = helpers.Frame.get_rolling_cagr(ror, window)
    assertion = pd.testing.assert_series_equal if series else pd.testing.assert_frame_equal
    assertion(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("value", [1e100, -1 + 1e-15])
def test_rolling_cagr_preserves_intermediate_product_range(value: float) -> None:
    ror = returns_frame(24, 2)
    ror.iloc[:12, :] = value
    with np.errstate(over="ignore", invalid="ignore"):
        expected = helpers.Frame.get_rolling_fn(ror, 12, helpers.Frame.get_cagr)
        actual = helpers.Frame.get_rolling_cagr(ror, 12)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("window", [0, -1, *range(1, 12), 121, 12.0, None])
def test_rolling_cagr_preserves_window_validation(window) -> None:
    ror = returns_frame()
    try:
        helpers.Frame.get_rolling_fn(ror, window, helpers.Frame.get_cagr)
    except Exception as error:
        with pytest.raises(type(error), match=str(error)):
            helpers.Frame.get_rolling_cagr(ror, window)
    else:
        pytest.fail("The reference must reject this window")


def test_generic_rolling_callback_keeps_custom_behavior() -> None:
    ror = returns_frame()

    def fn(x: pd.Series) -> float:
        return x.iloc[-1] - x.iloc[0]

    expected = ror.rolling(12).apply(fn).dropna()
    pd.testing.assert_frame_equal(helpers.Frame.get_rolling_fn(ror, 12, fn), expected)


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("kind", ["assets", "portfolio"])
def test_public_rolling_cagr_preserves_inflation_conversion(synthetic_env, real: bool, kind: str) -> None:
    cls = ok.AssetList if kind == "assets" else ok.Portfolio
    obj = cls(["A.US", "B.US"], ccy="USD", inflation=True)
    data = obj._add_inflation()
    if real:
        data = obj._make_real_return_time_series(data)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    expected = helpers.Frame.get_rolling_fn(data, 12, helpers.Frame.get_cagr)
    pd.testing.assert_frame_equal(obj.get_rolling_cagr(12, real=real), expected, rtol=1e-12, atol=1e-12)


def test_rolling_tracking_difference_uses_fast_cagr(synthetic_env, mocker) -> None:
    obj = ok.AssetList(["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False)
    rolling = helpers.Frame.get_rolling_fn(obj.assets_ror, 12, helpers.Frame.get_cagr)
    expected = rolling.subtract(rolling.iloc[:, 0], axis=0).iloc[:, 1:]
    callback = mocker.spy(Rolling, "apply")
    actual = obj.tracking_difference_annualized(rolling_window=12)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    assert callback.call_count == 0
