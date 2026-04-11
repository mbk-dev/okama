import warnings

import pandas as pd
import matplotlib.pyplot as plt

import okama as ok

import os

# os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
# warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

pf = ok.Portfolio(
    ["MCFTR.INDX", "RUCBTRNS.INDX", "GC.COMM"],
    weights=[0.60, 0.35, 0.05],
    ccy="RUB",
    inflation=True,
    last_date="2026-02",
    rebalancing_strategy=ok.Rebalance(period="year"),
    symbol="Pension_portfolio.PF",
)

ind = ok.IndexationStrategy(pf)  # создаём стратегию, привязанную к портфелю

ind.initial_investment = 10_000   # размер начальных вложений
ind.amount = -2_500               # размер годового снятия
ind.frequency = "year"            # частота — раз в год
ind.indexation = "inflation"      # индексация на среднюю инфляцию

pf.dcf.cashflow_parameters = ind

cf_percentage = pf.dcf.cash_flow_ts(discounting="fv").resample("Y").sum()
cf_percentage.plot(kind="bar")

plt.show()