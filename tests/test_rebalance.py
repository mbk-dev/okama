import pytest
from pytest import approx
from pytest import mark

import numpy as np
from numpy.testing import assert_allclose

import pandas as pd

import okama as ok
from okama import settings


@mark.rebalance
def test_validate_period_failing():
    with pytest.raises(ValueError):
        ok.Rebalance(period="not existing")

@mark.rebalance
def test_validate_abs_deviation_big_failing():
    with pytest.raises(ValueError, match=r"Absolute deviation must be less or equal to 1."):
        ok.Rebalance(abs_deviation=1.5)

@mark.rebalance
def test_validate_abs_deviation_big_failing():
    with pytest.raises(ValueError, match=r"Absolute deviation must be positive."):
        ok.Rebalance(abs_deviation=-100)

@mark.rebalance
def test_validate_rel_deviation_failing():
    with pytest.raises(ValueError, match=r"Relative deviation must be positive."):
        ok.Rebalance(rel_deviation=-100)


def test_repr(init_rebalance_no_rebalancing):
    value = pd.Series(
        dict(
            period="none",
            abs_deviation=None,
            rel_deviation=None,
        )
    )
    assert repr(init_rebalance_no_rebalancing) == repr(value)
