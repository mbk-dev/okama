from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Result:
    """
    The result of finding a solution for `find_the_largest_withdrawals_size()`.

    Attributes
    ----------
    success : bool
        Whether or not the solver exited successfully.

    withdrawal_abs : float
        The absolute amount of withdrawal size (the best solution found).

    withdrawal_rel : float
        The relative amount of withdrawal size (the best solution found).

    error_rel : float
        Characterizes how accurately the goal is fulfilled. The goal is set in the parameters.

    solutions : pd.DataFrame
        The array of results of attempts to find solutions.
    """
    success: bool
    withdrawal_abs: float
    withdrawal_rel: float
    error_rel: float
    solutions: pd.DataFrame