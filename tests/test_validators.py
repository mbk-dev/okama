"""
Tests the validator functions
"""

import pytest

from okama import settings
from okama.common import validators


# validate_integer
def test_validate_integer_valid():
    validators.validate_integer(
        "arg",
        10,
        min_value=0,
        max_value=20,
        inclusive=False,
        custom_min_message="custom min msg",
        custom_max_message="custom max msg",
    )


@pytest.mark.parametrize("object_value, exception", [(1.5, TypeError), (-1, ValueError), (100, ValueError)])
def test_validate_integer_error(object_value, exception):
    with pytest.raises(exception) as ex:
        validators.validate_integer("arg", object_value, min_value=100, inclusive=False)
    assert "arg" in str(ex.value)


def test_validate_integer_min_custom_msg():
    with pytest.raises(ValueError) as ex:
        validators.validate_integer("arg", 10, min_value=100, custom_min_message="custom")
    assert str(ex.value) == "custom"


def test_validate_integer_max_std_err_msg():
    with pytest.raises(ValueError) as ex:
        validators.validate_integer("arg", 10, min_value=1, max_value=5)
    assert "arg" in str(ex.value)
    assert "5" in str(ex.value)


def test_validate_integer_max_custom_err_msg():
    with pytest.raises(ValueError) as ex:
        validators.validate_integer("arg", 10, min_value=1, max_value=5, custom_max_message="custom")
    assert str(ex.value) == "custom"


def test_validate_integer_valid_inclusive():
    validators.validate_integer(
        "arg",
        10,
        min_value=0,
        max_value=20,
        inclusive=True,
        custom_min_message="custom min msg",
        custom_max_message="custom max msg",
    )


def test_validate_integer_valid_inclusive_equal():
    validators.validate_integer(
        "arg",
        10,
        min_value=10,
        max_value=20,
        inclusive=True,
    )


def test_validate_integer_min_std_err_msg_inclusive():
    with pytest.raises(ValueError) as ex:
        validators.validate_integer("arg", 10, min_value=11, inclusive=True)
    assert "arg" in str(ex.value)
    assert "11" in str(ex.value)


def test_validate_integer_max_std_err_msg_inclusive():
    with pytest.raises(ValueError) as ex:
        validators.validate_integer("arg", 10, min_value=1, max_value=5, inclusive=True)
    assert "arg" in str(ex.value)
    assert "5" in str(ex.value)


# validate_real


def test_validate_real_valid():
    validators.validate_real("number", 1)


def test_validate_real_type_error():
    with pytest.raises(TypeError):
        validators.validate_real("arg", "not a number")


# validate_distribution
@pytest.mark.parametrize("distr", list(settings.distributions))
def test_validate_distribution_valid(distr):
    validators.validate_distribution(distr)


def test_validate_distribution_invalid():
    with pytest.raises(ValueError) as ex:
        validators.validate_distribution("invalid")
    msg = str(ex.value)
    assert "distribution must be in" in msg
    # Ensure at least one known option is mentioned in the message
    assert "norm" in msg


# validate_distribution_parameters

def test_validate_distribution_parameters_type_error():
    with pytest.raises(ValueError) as ex:
        validators.validate_distribution_parameters("norm", 123)
    assert "neither a list nor a tuple" in str(ex.value)


def test_validate_distribution_parameters_norm_valid():
    validators.validate_distribution_parameters("norm", (0, 1))


def test_validate_distribution_parameters_norm_wrong_len():
    with pytest.raises(ValueError) as ex:
        validators.validate_distribution_parameters("norm", (0,))
    assert "Normal distribution" in str(ex.value)


def test_validate_distribution_parameters_lognorm_valid():
    validators.validate_distribution_parameters("lognorm", (0.5, 0, 1))


def test_validate_distribution_parameters_lognorm_wrong_len():
    with pytest.raises(ValueError) as ex:
        validators.validate_distribution_parameters("lognorm", (0.5, 0))
    assert "Lognormal distribution" in str(ex.value)


def test_validate_distribution_parameters_t_valid():
    validators.validate_distribution_parameters("t", (10, 0, 1))


def test_validate_distribution_parameters_t_wrong_len():
    with pytest.raises(ValueError) as ex:
        validators.validate_distribution_parameters("t", (10, 0))
    assert "T-distribution" in str(ex.value)


def test_validate_distribution_parameters_t_df_too_small():
    # df <= 2 should raise ValueError
    with pytest.raises(ValueError, match="Degrees of freedom \(df\) for Student's t-distribution must be > 2"):
        validators.validate_distribution_parameters("t", (2, 0, 1))

    with pytest.raises(ValueError, match="Degrees of freedom \(df\) for Student's t-distribution must be > 2"):
        validators.validate_distribution_parameters("t", (1.5, 0, 1))


def test_validate_distribution_parameters_unknown_distribution():
    with pytest.raises(ValueError) as ex:
        validators.validate_distribution_parameters("gamma", (1, 2, 3))
    assert str(ex.value) == "Unknown distribution."
