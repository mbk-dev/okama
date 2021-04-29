"""Various validators"""
import numbers
import operator
from typing import Optional, Any


def validate_integer(
    arg_name: str,
    arg_value: Any, *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    inclusive: bool = False,
    custom_min_message: Optional[str] = None,
    custom_max_message: Optional[str] = None,
):
    """
    Validate that `arg_value` is an integer, and optionally fall within specific bounds.

    A custom override error message can be provided when min/max bounds are exceeded.

    Parameters
    ----------
    arg_name : str
        The name of the argument (used in default error messages).
    arg_value : Any
        The value being validated.
    min_value : int, optional
        Optional, specifies the minimum value.
    max_value : int, optional
        Optional, specifies the maximum value.
    inclusive : bool, default False
        Specifies whether the bounds limits are included (inclusive). Default value in False.
    custom_min_message : str, optional
        Optional, custom message when value is less than minimum.
    custom_max_message : str, optional
        Optional, custom message when value is less than maximum.

    Returns
    -------
    None
        No exceptions raised if validation passes.

    Raises
    ------
    TypeError
        If `arg_value` is not an integer.
    ValueError
        If `arg_value` does not satisfy the bounds.
    """
    if not isinstance(arg_value, int):
        raise TypeError(f"{arg_name} must be an integer.")

    ops = {'<=': operator.le,
           '>=': operator.ge,
           '>': operator.gt,
           '<': operator.lt}

    if inclusive:
        operator_less = '<'
        operator_greater = '>'
    else:
        operator_less = '<='
        operator_greater = '>='

    if min_value is not None and ops[operator_less](arg_value, min_value):
        if custom_min_message is not None:
            raise ValueError(custom_min_message)
        raise ValueError(f"'{arg_name:s}' must be {operator_greater} {min_value:d}")

    if max_value is not None and ops[operator_greater](arg_value, max_value):
        if custom_max_message is not None:
            raise ValueError(custom_max_message)
        raise ValueError(f"'{arg_name:s}' must be {operator_less} {max_value:d}")


def validate_real(arg_name: str, arg_value: Any) -> None:
    """
    Validate that `arg_value` is a real number.

    Parameters
    ----------
    arg_name : str
        The name of the argument (used in default error messages).
    arg_value : Any
        The value being validated.

    Returns
    -------
    None
        No exceptions raised if validation passes.

    Raises
    ------
    TypeError
        If `arg_value` is not an integer.
    """
    if not isinstance(arg_value, numbers.Real):
        raise TypeError(f"{arg_name} should be a Real number.")
