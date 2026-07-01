"""Portfolio ratios"""

import pandas as pd


def get_sharpe_ratio(
    pf_return: float | pd.Series,
    rf_return: float,
    std_deviation: float | pd.Series,
) -> float | pd.Series:
    """
    Calculate Sharpe ratio.
    """
    return (pf_return - rf_return) / std_deviation


def get_sortino_ratio(
    pf_return: float | pd.Series,
    t_return: float,
    semi_deviation: float | pd.Series,
):
    """
    Calculate Sortino ratio.
    """
    return (pf_return - t_return) / semi_deviation
