import numpy as np
import pandas as pd

from okama.common.helpers import helpers


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
