import warnings

import pandas as pd
import okama as ok

import os


os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

pf = ok.Portfolio(
    ["SPY.US", "AGG.US", "GLD.US"],
    weights=[0.60, 0.35, 0.05],
    ccy="USD",
    inflation=True,
    last_date="2024-01",
    rebalancing_strategy=ok.Rebalance(period="year"),
    symbol="My_portfolio.PF",
)

pc = ok.PercentageStrategy(pf)  # create PercentageStrategy linked to the portfolio

pc.initial_investment = 10_000  # initial investments size
pc.frequency = "year"  # withdrawals frequency
pc.percentage = -0.12

pf.dcf.cashflow_parameters = pc  # assign the cash flow strategy to portfolio

pf.dcf.set_mc_parameters(distribution="norm", period=30, number=400)  # simulation period in years

result = pf.dcf.find_the_largest_withdrawals_size(
    goal="maintain_balance_pv",  # The goal of the strategy in this case is to keep the portfolio's real balance (for the whole period)
    percentile=25,  # The percentile of Monte Carlo result distribution where the goal is to be achieved. The 25th percentile is a negative scenario.
    threshold=0.10,  # 10% - is the percentage of initial investments when the portfolio balance is considered voided.
    iter_max=50,  # The maximum number of iterations to find the solution.
    tolerance_rel=0.15,  # The allowed tolerance for the solution. The tolerance is the largest error for the achieved goal.
)

print(result.withdrawal_rel)
