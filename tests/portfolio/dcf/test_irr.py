import numpy as np
import pytest

import okama as ok  # noqa: F401  (used by integration tests added in later tasks)
from okama.portfolios import dcf_calculations
from okama.settings import _MONTHS_PER_YEAR, DEFAULT_DISCOUNT_RATE  # noqa: F401


def test_irr_core_single_in_single_out_matches_closed_form():
    # -1000 at t0, +1200 at t12 on a monthly grid -> annual IRR is exactly 20%.
    cf = np.zeros(13)
    cf[0] = -1000.0
    cf[12] = 1200.0
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=_MONTHS_PER_YEAR)
    assert result[0] == pytest.approx(0.2, abs=1e-9)


def test_irr_core_textbook_per_period_rate():
    # -100, +60, +60 with periods_per_year=1 -> per-period IRR from 100 x^2 - 60 x - 60 = 0.
    cf = np.array([-100.0, 60.0, 60.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    x = (60.0 + np.sqrt(3600.0 + 24000.0)) / 200.0
    assert result[0] == pytest.approx(x - 1.0, abs=1e-9)


def test_irr_core_no_sign_change_returns_nan():
    cf = np.array([-100.0, -50.0, -30.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert np.isnan(result[0])


def test_irr_core_vectorized_columns_are_independent():
    # Two columns solved at once; periods_per_year=1 so annual == per-period rate.
    cf = np.array(
        [
            [-1000.0, -1000.0],
            [0.0, 0.0],
            [1200.0, 900.0],
        ]
    )
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert result[0] == pytest.approx(np.sqrt(1.2) - 1.0, abs=1e-9)
    assert result[1] == pytest.approx(np.sqrt(0.9) - 1.0, abs=1e-9)


def test_irr_core_depleted_partial_recovery_is_negative():
    # Invested 1000, recovered only 300 over the period, terminal 0 -> finite, negative IRR.
    cf = np.array([-1000.0, 100.0, 100.0, 100.0, 0.0])
    result = dcf_calculations.irr_of_cashflow_matrix(cf, periods_per_year=1)
    assert np.isfinite(result[0])
    assert result[0] < 0.0
