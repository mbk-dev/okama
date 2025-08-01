from __future__ import annotations

from typing import Union, Optional, Literal

import pandas as pd
import numpy as np

import okama.common.helpers.helpers as helpers
import okama.portfolios.cashflow_strategies as cf
from okama import settings


def get_wealth_indexes_fv_with_cashflow(
    ror: Union[pd.Series, pd.DataFrame],
    portfolio_symbol: Optional[str],
    inflation_symbol: Optional[str],
    cashflow_parameters: cf.CashFlow,
    task: Literal['backtest', 'monte_carlo'],
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate wealth index Future Values (FV) for a series of returns with cash flows (withdrawals/contributions).

    Values of the wealth index correspond to the beginning of the month.
    """
    dcf_object = cashflow_parameters.parent.dcf
    dcf_object.cashflow_parameters = cashflow_parameters
    period_initial_amount = cashflow_parameters.initial_investment
    period_initial_amount_cached = period_initial_amount
    last_regular_cash_flow = 0
    amount = getattr(cashflow_parameters, "amount", None)
    if isinstance(ror, pd.DataFrame):
        portfolio_position = ror.columns.get_loc(portfolio_symbol)
    else:
        # for Series
        portfolio_position = 0
        ror = ror.to_frame()
    cash_flow_ts = dcf_object.cashflow_parameters.time_series
    # check if iteration needed
    cashflow_iterate_condition = not (cash_flow_ts.empty or (cash_flow_ts == 0).all())
    if cashflow_iterate_condition:
        ror_cashflow_df = ror.assign(cashflow_ts=cash_flow_ts)
        ror_cashflow_df.fillna(0, inplace=True)
        n_rows = ror.shape[0]
        discount_factors = (1.0 + dcf_object.discount_rate / settings._MONTHS_PER_YEAR) ** np.arange(n_rows)
        if task == 'backtest':
            if dcf_object.cashflow_parameters.time_series_discounted_values:
                ror_cashflow_df.loc[:, "cashflow_ts"] = ror_cashflow_df.loc[:, "cashflow_ts"].mul(discount_factors, axis=0)
        elif task =='monte_carlo':
            if not dcf_object.cashflow_parameters.time_series_discounted_values:
                ror_cashflow_df.loc[:, "cashflow_ts"] = ror_cashflow_df.loc[:, "cashflow_ts"].mul(discount_factors, axis=0)
        else:
            raise ValueError(f"Unknown task: {task}. It must be 'monte_carlo' or 'backtest'")
    else:
        ror_cashflow_df = ror.to_frame() if not isinstance(ror, pd.DataFrame) else ror
        ror_cashflow_df.loc[:, "cashflow_ts"] = 0
    cash_flow_ts = ror_cashflow_df["cashflow_ts"]  # cash flow monthly time series
    periods_per_year = settings.frequency_periods_per_year[cashflow_parameters.frequency]
    if cashflow_parameters.frequency == "month" or cashflow_parameters.NAME == "time_series":
    # Fast Calculation
        s = pd.Series(dtype=float, name=portfolio_symbol)
        for n, row in enumerate(ror.itertuples()):
            date = row[0]
            r = row[portfolio_position + 1]
            if cashflow_parameters.NAME == "fixed_amount":
                cashflow = amount * (1 + cashflow_parameters.indexation / settings._MONTHS_PER_YEAR) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow = cashflow_parameters.percentage / periods_per_year * period_initial_amount
            elif cashflow_parameters.NAME == "time_series":
                cashflow = 0
            elif cashflow_parameters.NAME == "VDS":
                cashflow = cashflow_parameters.calculate_withdrawal_size(
                    last_withdrawal=last_regular_cash_flow if n > 0 else 0,
                    balance=period_initial_amount,
                    number_of_periods=n,
                )
            else:
                raise ValueError("Wrong cashflow strategy name value.")
            period_initial_amount = period_initial_amount * (r + 1) + cashflow + cash_flow_ts[date]
            date = row[0]
            s[date] = period_initial_amount
    else:
    # Slow Calculation
        pandas_frequency = cashflow_parameters._pandas_frequency
        wealth_df = pd.DataFrame(dtype=float, columns=[portfolio_symbol])
        for n, x in enumerate(ror_cashflow_df.resample(rule=pandas_frequency, convention="start")):
            ror_ts = x[1].iloc[:, portfolio_position]  # select ror part of the grouped data
            cashflow_ts_local = x[1].loc[:, "cashflow_ts"].copy()
            # CashFlow inside period (Extra withdrawals/contributions)
            if (cashflow_ts_local != 0).any():
                period_wealth_index = pd.Series(dtype=float, name=portfolio_symbol)
                for k, (date, r) in enumerate(ror_ts.items()):
                    if k == 0:
                        month_balance = period_initial_amount + cashflow_ts_local[date]
                    else:
                        month_balance = month_balance * (r + 1) + cashflow_ts_local[date]
                    period_wealth_index[date] = month_balance
            else:
                period_wealth_index = period_initial_amount * (1 + ror_ts).cumprod()
            # CashFlow END period
            if cashflow_parameters.NAME == "fixed_amount":
                cashflow_value = amount * (1 + cashflow_parameters.indexation / periods_per_year) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow_value = cashflow_parameters.percentage / periods_per_year * period_initial_amount
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = cashflow_parameters.calculate_withdrawal_size(
                    last_withdrawal=last_regular_cash_flow if n > 0 else 0,
                    balance=period_initial_amount,
                    number_of_periods=n,
                )
            else:
                raise ValueError("Wrong cashflow_method value.")
            period_final_balance = period_wealth_index.iloc[-1] + cashflow_value
            period_wealth_index.iloc[-1] = period_final_balance
            period_initial_amount = period_final_balance
            wealth_df = pd.concat([None if wealth_df.empty else wealth_df, period_wealth_index], sort=False)
        s = wealth_df.squeeze()
    first_date = s.index[0]
    first_wealth_index_date = first_date - 1  # set first date to one month earlie
    s.loc[first_wealth_index_date] = period_initial_amount_cached
    if inflation_symbol:
        cum_inflation = helpers.Frame.get_wealth_indexes(
            ror=ror.loc[:, inflation_symbol], initial_amount=period_initial_amount_cached
        )
        wealth_index = pd.concat([s, cum_inflation], axis="columns")
    else:
        wealth_index = s
    wealth_index.sort_index(ascending=True, inplace=True)
    return wealth_index

def get_cash_flow_fv(
        ror: Union[pd.Series, pd.DataFrame],
        portfolio_symbol: Optional[str],
        cashflow_parameters: cf.CashFlow,
        task: Literal['backtest', 'monte_carlo'],
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate cash flow future values (FV) for a series of returns according to withdrawal/contributions strategies.
    """
    dcf_object = cashflow_parameters.parent.dcf
    dcf_object.cashflow_parameters = cashflow_parameters
    period_initial_amount = cashflow_parameters.initial_investment
    last_regular_cash_flow = 0
    cs_fv = pd.Series(dtype=float, name="cash_flow_fv")
    amount = getattr(cashflow_parameters, "amount", None)
    if isinstance(ror, pd.DataFrame):
        portfolio_position = ror.columns.get_loc(portfolio_symbol)
    else:
        # for Series
        portfolio_position = 0
        ror = ror.to_frame()
    cash_flow_ts = dcf_object.cashflow_parameters.time_series
    # check if iteration needed
    cashflow_iterate_condition = not (cash_flow_ts.empty or (cash_flow_ts == 0).all())
    if cashflow_iterate_condition:
        ror_cashflow_df = ror.assign(cashflow_ts=cash_flow_ts)
        ror_cashflow_df.fillna(0, inplace=True)
        n_rows = ror.shape[0]
        discount_factors = (1.0 + dcf_object.discount_rate / settings._MONTHS_PER_YEAR) ** np.arange(n_rows)
        if task == 'backtest':
            if dcf_object.cashflow_parameters.time_series_discounted_values:
                ror_cashflow_df.loc[:, "cashflow_ts"] = ror_cashflow_df.loc[:, "cashflow_ts"].mul(discount_factors,
                                                                                                  axis=0)
        elif task == 'monte_carlo':
            if not dcf_object.cashflow_parameters.time_series_discounted_values:
                ror_cashflow_df.loc[:, "cashflow_ts"] = ror_cashflow_df.loc[:, "cashflow_ts"].mul(discount_factors,
                                                                                                  axis=0)
        else:
            raise ValueError(f"Unknown task: {task}. It must be 'monte_carlo' or 'backtest'")
    else:
        ror_cashflow_df = ror.to_frame() if not isinstance(ror, pd.DataFrame) else ror
        ror_cashflow_df.loc[:, "cashflow_ts"] = 0
    cash_flow_ts = ror_cashflow_df["cashflow_ts"]  # cash flow monthly time series
    periods_per_year = settings.frequency_periods_per_year[cashflow_parameters.frequency]
    if cashflow_parameters.frequency == "month" or cashflow_parameters.NAME == "time_series":
        # Fast Calculation
        for n, row in enumerate(ror.itertuples()):
            date = row[0]
            r = row[portfolio_position + 1]
            # Calculate regular cash flow
            if cashflow_parameters.NAME == "fixed_amount":
                cashflow = amount * (1 + cashflow_parameters.indexation / settings._MONTHS_PER_YEAR) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow = cashflow_parameters.percentage / periods_per_year * period_initial_amount
            elif cashflow_parameters.NAME == "time_series":
                cashflow = 0
            elif cashflow_parameters.NAME == "VDS":
                cashflow = cashflow_parameters.calculate_withdrawal_size(
                    last_withdrawal=last_regular_cash_flow if n > 0 else 0,
                    balance=period_initial_amount,
                    number_of_periods=n,
                )
            else:
                raise ValueError("Wrong cashflow strategy name value.")
            # add Extra Withdrawals/Contributions
            last_regular_cash_flow = cashflow
            cs_value = cashflow + cash_flow_ts[date]
            period_initial_amount = period_initial_amount * (r + 1) + cs_value
            cs_fv[date] = cs_value
    else:
        # Slow Calculation
        pandas_frequency = settings.frequency_mapping[cashflow_parameters.frequency]
        for n, x in enumerate(ror_cashflow_df.resample(rule=pandas_frequency, convention="start")):
            ror_ts = x[1].iloc[:, portfolio_position]  # select ror part of the grouped data
            cashflow_ts_local = x[1].loc[:, "cashflow_ts"].copy()
            # CashFlow inside period (Extra cash flow)
            if (cashflow_ts_local != 0).any():
                period_wealth_index = pd.Series(dtype=float, name=portfolio_symbol)
                for k, (date, r) in enumerate(ror_ts.items()):
                    if k == 0:
                        month_balance = period_initial_amount
                    else:
                        month_balance = month_balance * (r + 1) + cashflow_ts_local[date]
                    period_wealth_index[date] = month_balance
            else:
                period_wealth_index = period_initial_amount * (1 + ror_ts).cumprod()
            # CashFlow END period (Regular Cash Flow)
            if cashflow_parameters.NAME == "fixed_amount":
                cashflow_value = amount * (1 + cashflow_parameters.indexation / periods_per_year) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow_value = cashflow_parameters.percentage / periods_per_year * period_initial_amount
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = cashflow_parameters.calculate_withdrawal_size(
                    last_withdrawal=last_regular_cash_flow if n > 0 else 0,
                    balance=period_initial_amount,
                    number_of_periods=n,
                )
            else:
                raise ValueError("Wrong cashflow_method value.")
            period_final_balance = period_wealth_index.iloc[-1] + cashflow_value
            period_wealth_index.iloc[-1] = period_final_balance
            period_initial_amount = period_final_balance
            cashflow_ts_local.iloc[-1] += cashflow_value
            cs_fv = pd.concat([None if cs_fv.empty else cs_fv, cashflow_ts_local], sort=False)
    return cs_fv


def remove_negative_values(input_s: pd.Series) -> pd.Series:
    if not isinstance(input_s, pd.Series):
        raise TypeError("input_s must be a pd.Series")
    s = input_s.copy()
    condition = s <= 0
    try:
        survival_date = s[condition].index[0]
        s[survival_date] = 0
        s[s.index > survival_date] = np.nan
    except IndexError:
        pass
    return s


def discount_monthly_cash_flow(
        cash_flow_fv: Union[pd.Series, pd.DataFrame],
        annual_effective_discount_rate: float
) -> Union[pd.Series, pd.DataFrame]:
    number_of_months = cash_flow_fv.shape[0]
    monlthly_discount_rate = (1 + annual_effective_discount_rate) ** (1 / settings._MONTHS_PER_YEAR) - 1
    discount_factors = (1.0 + monlthly_discount_rate) ** np.arange(number_of_months)
    cash_flow_pv = cash_flow_fv.div(discount_factors, axis=0)
    return cash_flow_pv