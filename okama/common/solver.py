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
        It is the first withdrawal value (before it was indexed in IndexationStrategy).

    withdrawal_rel : float
        The relative amount of withdrawal size (the best solution found).
        The first withdrawal value (before it was indexed in IndexationStrategy) divided by the initial investment.

    error_rel : float
        Characterizes how accurately the goal is fulfilled. The goal is set in the parameters.

    solutions : pd.DataFrame
        The history of attempts to find solutions (withdrawal values and error level).
    """

    success: bool
    withdrawal_abs: float
    withdrawal_rel: float
    error_rel: float
    solutions: pd.DataFrame(columns=["withdrawal_abs", "withdrawal_rel", "error_rel", "error_rel_change"])

    def __repr__(self):
        dic = {
            "success": self.success,
            "withdrawal_abs": self.withdrawal_abs,
            "withdrawal_rel": self.withdrawal_rel,
            "error_rel": self.error_rel,
            "attempts": self.solutions.shape[0],
        }
        return repr(pd.Series(dic))
