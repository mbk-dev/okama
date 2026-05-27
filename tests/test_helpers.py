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


def test_frame_get_cagr_short_history_returns_float_series_with_nan():
    """For history shorter than 12 months, `Frame.get_cagr` must return a
    Series with float dtype and NaN values (not an object Series of Nones,
    which would silently turn downstream numeric columns into `object`)."""
    idx = pd.period_range("2020-01", periods=6, freq="M")
    ror = pd.DataFrame(
        {"A.US": np.full(6, 0.01), "B.US": np.full(6, 0.02)}, index=idx
    )

    result = helpers.Frame.get_cagr(ror)

    assert isinstance(result, pd.Series)
    assert result.dtype == np.float64, f"expected float64, got {result.dtype}"
    assert result.isna().all()
    assert list(result.index) == ["A.US", "B.US"]
