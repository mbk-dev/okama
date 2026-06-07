import threading
from math import comb

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from okama.common.helpers import helpers


# --- get_grid_weights tests ---


def test_get_grid_weights_two_assets_step_50():
    """Step 0.5 with 2 assets produces 3 combos: (0,1), (0.5,0.5), (1,0)."""
    result = helpers.Float.get_grid_weights(w_shape=2, step=0.50)
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    for w in result:
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert w.shape == (2,)


def test_get_grid_weights_three_assets_step_50():
    """Step 0.5 with 3 assets produces 6 combos."""
    result = helpers.Float.get_grid_weights(w_shape=3, step=0.50)
    assert len(result) == 6
    for w in result:
        assert_allclose(w.sum(), 1.0, atol=1e-12)


def test_get_grid_weights_two_assets_step_25():
    """Step 0.25 with 2 assets: 0.00,0.25,0.50,0.75,1.00 → 5 combos."""
    result = helpers.Float.get_grid_weights(w_shape=2, step=0.25)
    assert len(result) == 5
    expected = [
        np.array([0.00, 1.00]),
        np.array([0.25, 0.75]),
        np.array([0.50, 0.50]),
        np.array([0.75, 0.25]),
        np.array([1.00, 0.00]),
    ]
    for w, exp in zip(result, expected, strict=True):
        assert_allclose(w, exp, atol=1e-12)


def test_get_grid_weights_with_bounds():
    """Bounds ((0.25,0.75),(0.25,0.75)) with step 0.25 → 3 combos."""
    bounds = ((0.25, 0.75), (0.25, 0.75))
    result = helpers.Float.get_grid_weights(w_shape=2, step=0.25, bounds=bounds)
    assert len(result) == 3
    for w in result:
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert w[0] >= 0.25 - 1e-12
        assert w[0] <= 0.75 + 1e-12
        assert w[1] >= 0.25 - 1e-12
        assert w[1] <= 0.75 + 1e-12


def test_get_grid_weights_all_weights_are_multiples_of_step():
    """Every weight value must be a multiple of the step."""
    step = 0.10
    result = helpers.Float.get_grid_weights(w_shape=3, step=step)
    for w in result:
        for val in w:
            remainder = val % step
            assert remainder < 1e-12 or abs(remainder - step) < 1e-12


def test_get_grid_weights_invalid_step_raises():
    """Step that doesn't divide 1.0 evenly should raise ValueError."""
    with pytest.raises(ValueError, match="step"):
        helpers.Float.get_grid_weights(w_shape=2, step=0.15)


def test_get_grid_weights_step_below_one_percent_raises():
    """Step below 1% (0.01) should raise ValueError."""
    with pytest.raises(ValueError, match="step"):
        helpers.Float.get_grid_weights(w_shape=2, step=0.005)


def test_get_grid_weights_step_out_of_range_raises():
    """Step <= 0 or > 1 should raise ValueError."""
    with pytest.raises(ValueError):
        helpers.Float.get_grid_weights(w_shape=2, step=0.0)
    with pytest.raises(ValueError):
        helpers.Float.get_grid_weights(w_shape=2, step=1.5)


def test_get_grid_weights_scales_to_many_assets():
    """Generation cost must be O(valid points), not O((1/step + 1)**N).

    With 8 assets and step 0.10 there are comb(17, 7) = 19448 valid weight
    vectors, but 11**8 = 214 million points in the full Cartesian product.
    A product-and-filter implementation cannot finish enumerating that in
    seconds; a direct composition enumeration returns almost instantly.
    """
    result_holder: dict[str, pd.Series] = {}

    def _run() -> None:
        result_holder["weights"] = helpers.Float.get_grid_weights(w_shape=8, step=0.10)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=8.0)

    if worker.is_alive():
        pytest.fail("get_grid_weights did not scale: 8 assets did not finish within 8 seconds")

    weights = result_holder["weights"]
    assert len(weights) == comb(17, 7)  # 19448 valid compositions of 10 units into 8 parts
    for w in weights:
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert w.shape == (8,)


def test_get_grid_weights_raises_for_explosive_grid():
    """A grid larger than the default ceiling must fail fast with ValueError
    instead of enumerating hundreds of thousands of points (12 assets at
    step 0.10 = comb(21, 11) = 352716 points)."""
    with pytest.raises(ValueError, match="max_points"):
        helpers.Float.get_grid_weights(w_shape=12, step=0.10)


def test_get_grid_weights_max_points_is_configurable():
    """max_points can be lowered to reject an otherwise-small grid, or raised
    to allow a large one."""
    # 8 assets at step 0.10 = 19448 points; a low ceiling rejects it.
    with pytest.raises(ValueError, match="max_points"):
        helpers.Float.get_grid_weights(w_shape=8, step=0.10, max_points=1_000)
    # Raising the ceiling allows a grid that the default would reject.
    result = helpers.Float.get_grid_weights(w_shape=12, step=0.10, max_points=500_000)
    assert len(result) == comb(21, 11)  # 352716


def test_frame_get_cagr_short_history_returns_float_series_with_nan():
    """For history shorter than 12 months, `Frame.get_cagr` must return a
    Series with float dtype and NaN values (not an object Series of Nones,
    which would silently turn downstream numeric columns into `object`)."""
    idx = pd.period_range("2020-01", periods=6, freq="M")
    ror = pd.DataFrame({"A.US": np.full(6, 0.01), "B.US": np.full(6, 0.02)}, index=idx)

    result = helpers.Frame.get_cagr(ror)

    assert isinstance(result, pd.Series)
    assert result.dtype == np.float64, f"expected float64, got {result.dtype}"
    assert result.isna().all()
    assert list(result.index) == ["A.US", "B.US"]


def test_frame_get_drawdowns_from_wealth_measures_decline_from_peak():
    """`Frame.get_drawdowns_from_wealth` must compute drawdowns directly from a
    wealth index (or price) time series, keeping the first observation."""
    idx = pd.period_range("2020-01", periods=4, freq="M")
    wealth = pd.Series([100.0, 110.0, 99.0, 121.0], index=idx, name="A.US")

    result = helpers.Frame.get_drawdowns_from_wealth(wealth)

    expected = pd.Series([0.0, 0.0, -0.1, 0.0], index=idx, name="A.US")
    pd.testing.assert_series_equal(result, expected)


def test_index_rolling_fn_emits_no_pandas4warning():
    """pd.concat 'copy' keyword is deprecated on pandas 3 — the rolling-window
    concat must stay warning-free (GH #85)."""
    import warnings

    idx = pd.period_range("2020-01", periods=14, freq="M")
    df = pd.DataFrame({"A.US": np.full(14, 0.01)}, index=idx)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.Pandas4Warning)
        result = helpers.Index.rolling_fn(df, window=12, fn=lambda d: d.cumsum())

    assert not result.empty


# --- Index.tracking_error tests ---


def _make_two_asset_ror(months: int = 24) -> pd.DataFrame:
    """Benchmark in the first column + a fund tracking it with noise."""
    rng = np.random.default_rng(42)
    idx = pd.period_range("2020-01", periods=months, freq="M")
    bench = pd.Series(rng.normal(0.01, 0.03, size=months), index=idx, name="BENCH.INDX")
    fund = pd.Series(bench.values + rng.normal(0.001, 0.01, size=months), index=idx, name="FUND.US")
    return pd.concat([bench, fund], axis=1)


def test_tracking_error_rms_default_matches_legacy_formula():
    """Default method and method='rms' produce the historical uncentered RMS values."""
    ror = _make_two_asset_ror()
    d = ror["FUND.US"] - ror["BENCH.INDX"]
    expected_last = np.sqrt((d**2).sum() / len(d)) * np.sqrt(12)
    result_default = helpers.Index.tracking_error(ror)
    result_rms = helpers.Index.tracking_error(ror, method="rms")
    pd.testing.assert_frame_equal(result_default, result_rms)
    assert result_default["FUND.US"].iloc[-1] == pytest.approx(expected_last)
    assert len(result_default) == len(ror)


def test_tracking_error_std_is_centered_with_bessel_correction():
    """method='std' is the centered sample std of differences (ddof=1), annualized."""
    ror = _make_two_asset_ror()
    d = ror["FUND.US"] - ror["BENCH.INDX"]
    expected_last = d.std(ddof=1) * np.sqrt(12)
    result = helpers.Index.tracking_error(ror, method="std")
    assert result["FUND.US"].iloc[-1] == pytest.approx(expected_last)
    # The first expanding point (std of a single observation) is dropped
    assert len(result) == len(ror) - 1


def test_tracking_error_invalid_method_raises_value_error():
    ror = _make_two_asset_ror()
    with pytest.raises(ValueError, match="method"):
        helpers.Index.tracking_error(ror, method="mad")


def test_tracking_error_short_period_raises_for_both_methods():
    from okama.common.error import ShortPeriodLengthError

    ror = _make_two_asset_ror(months=11)
    for m in ("rms", "std"):
        with pytest.raises(ShortPeriodLengthError):
            helpers.Index.tracking_error(ror, method=m)
