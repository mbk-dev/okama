"""Portfolio rates"""


def get_sharpe_ratio(pf_return: float, rf_return: float, std_deviation: float) -> float:
    """
    Calculate Sharpe ratio.
    """
    return (pf_return - rf_return) / std_deviation
