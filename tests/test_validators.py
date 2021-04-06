"""
Tests the validator functions
"""

import pytest

from okama.common.validators import validate_integer, validate_real


# validate_integer
def test_validate_integer_valid():
    validate_integer(
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
        validate_integer("arg", object_value, min_value=100, inclusive=False)
    assert "arg" in str(ex.value)


def test_validate_integer_min_custom_msg():
    with pytest.raises(ValueError) as ex:
        validate_integer("arg", 10, min_value=100, custom_min_message="custom")
    assert str(ex.value) == "custom"


def test_validate_integer_max_std_err_msg():
    with pytest.raises(ValueError) as ex:
        validate_integer("arg", 10, min_value=1, max_value=5)
    assert "arg" in str(ex.value)
    assert "5" in str(ex.value)


def test_validate_integer_max_custom_err_msg():
    with pytest.raises(ValueError) as ex:
        validate_integer(
            "arg", 10, min_value=1, max_value=5, custom_max_message="custom"
        )
    assert str(ex.value) == "custom"


def test_validate_integer_valid_inclusive():
    validate_integer(
        "arg",
        10,
        min_value=0,
        max_value=20,
        inclusive=True,
        custom_min_message="custom min msg",
        custom_max_message="custom max msg",
    )


def test_validate_integer_valid_inclusive_equal():
    validate_integer(
        "arg", 10, min_value=10, max_value=20, inclusive=True,
    )


def test_validate_integer_min_std_err_msg_inclusive():
    with pytest.raises(ValueError) as ex:
        validate_integer("arg", 10, min_value=11, inclusive=True)
    assert "arg" in str(ex.value)
    assert "11" in str(ex.value)


def test_validate_integer_max_std_err_msg_inclusive():
    with pytest.raises(ValueError) as ex:
        validate_integer("arg", 10, min_value=1, max_value=5, inclusive=True)
    assert "arg" in str(ex.value)
    assert "5" in str(ex.value)


# validate_real


def test_validate_real_valid():
    validate_real("number", 1)


def test_validate_real_type_error():
    with pytest.raises(TypeError):
        validate_real("arg", "not a number")
