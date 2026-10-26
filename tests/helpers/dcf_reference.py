"""Frozen numerical DCF/MC behavior from dev bcd8774, without test-time Git."""

from __future__ import annotations  # noqa: I001

import sys
import copy
import logging
import scipy
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from okama.portfolios import core
from okama.portfolios.dcf import PortfolioDCF
from okama.portfolios.mc import MonteCarlo
from okama.common.helpers import tails

from typing import Literal

import pandas as pd
import numpy as np
from scipy import optimize

import okama.common.helpers.helpers as helpers
import okama.portfolios.cashflow_strategies as cf
from okama import settings


dcf_calculations = sys.modules[__name__]
logger = logging.getLogger(__name__)


def get_wealth_indexes_fv_with_cashflow(  # noqa: C901
    ror: pd.Series | pd.DataFrame,
    portfolio_symbol: str | None,
    inflation_symbol: str | None,
    cashflow_parameters: cf.CashFlow,
    task: Literal["backtest", "monte_carlo"],
) -> pd.Series | pd.DataFrame:
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
    if cashflow_parameters.NAME == "CWD":
        drawdowns = helpers.Frame.get_drawdowns(ror.iloc[:, portfolio_position])
    cash_flow_ts = dcf_object.cashflow_parameters.time_series
    # check if iteration needed
    cashflow_iterate_condition = not (cash_flow_ts.empty or (cash_flow_ts == 0).all())
    if cashflow_iterate_condition:
        ror_cashflow_df = ror.assign(cashflow_ts=cash_flow_ts)
        ror_cashflow_df = ror_cashflow_df.fillna(0)
        n_rows = ror.shape[0]
        monthly_discount_rate = (1 + dcf_object.discount_rate) ** (1 / settings._MONTHS_PER_YEAR) - 1
        discount_factors = (1.0 + monthly_discount_rate) ** np.arange(n_rows)
        if task == "backtest":
            if dcf_object.cashflow_parameters.time_series_discounted_values:
                ror_cashflow_df.loc[:, "cashflow_ts"] = ror_cashflow_df.loc[:, "cashflow_ts"].mul(
                    discount_factors, axis=0
                )
        elif task == "monte_carlo":
            if not dcf_object.cashflow_parameters.time_series_discounted_values:
                ror_cashflow_df.loc[:, "cashflow_ts"] = ror_cashflow_df.loc[:, "cashflow_ts"].mul(
                    discount_factors, axis=0
                )
        else:
            raise ValueError(f"Unknown task: {task}. It must be 'monte_carlo' or 'backtest'")
    else:
        ror_cashflow_df = ror
        ror_cashflow_df.loc[:, "cashflow_ts"] = 0.0
    cash_flow_ts = ror_cashflow_df["cashflow_ts"]  # cash flow monthly time series
    periods_per_year = settings.frequency_periods_per_year[cashflow_parameters.frequency]
    if hasattr(cashflow_parameters, "indexation") and cashflow_parameters.frequency != "none":
        indexation_per_period = (1 + cashflow_parameters.indexation) ** (1 / periods_per_year) - 1
    if cashflow_parameters.frequency == "month" or cashflow_parameters.frequency == "none":
        # Fast Calculation
        s = pd.Series(dtype=float, name=portfolio_symbol)
        for n, row in enumerate(ror.itertuples()):
            date = row[0]
            r = row[portfolio_position + 1]
            if cashflow_parameters.frequency == "none":
                cashflow = 0
            elif cashflow_parameters.NAME == "fixed_amount":
                cashflow = amount * (1 + indexation_per_period) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow = cashflow_parameters.percentage / periods_per_year * period_initial_amount
            elif cashflow_parameters.NAME == "time_series":
                cashflow = 0
            elif cashflow_parameters.NAME == "CWD":
                withdrawal_without_drawdowns = amount * (1 + indexation_per_period) ** n
                if drawdowns[date] < 0:
                    cashflow = cashflow_parameters._calculate_withdrawal_size(
                        drawdown=drawdowns[date],
                        withdrawal_without_drawdowns=withdrawal_without_drawdowns,
                    )
                else:
                    cashflow = withdrawal_without_drawdowns
            else:
                raise ValueError("Wrong cashflow strategy name value.")
            period_initial_amount = period_initial_amount * (r + 1) + cashflow + cash_flow_ts[date]
            date = row[0]
            s[date] = period_initial_amount
    elif cashflow_parameters.frequency != "month" and cashflow_parameters.frequency != "none":
        # Slow Calculation
        pandas_frequency = cashflow_parameters._pandas_frequency
        months_in_full_period = settings._MONTHS_PER_YEAR / cashflow_parameters.periods_per_year
        wealth_chunks = []  # Collect all chunks to concatenate once at the end
        for n, x in enumerate(ror_cashflow_df.resample(rule=pandas_frequency, convention="start")):
            ror_ts = x[1].iloc[:, portfolio_position]  # select ror part of the grouped data
            months_local = ror_ts.shape[0]
            period_fraction = months_local / months_in_full_period  # 1 for a full period
            cashflow_ts_local = x[1].loc[:, "cashflow_ts"].copy()
            last_date = ror_ts.index[-1]
            # CashFlow inside period (Extra withdrawals/contributions)
            if (cashflow_ts_local != 0).any():
                period_wealth_index = pd.Series(dtype=float, name=portfolio_symbol)
                month_balance = period_initial_amount
                for date, r in ror_ts.items():
                    month_balance = month_balance * (r + 1) + cashflow_ts_local[date]
                    period_wealth_index[date] = month_balance
            else:
                period_wealth_index = period_initial_amount * (1 + ror_ts).cumprod()
            # CashFlow END period
            if cashflow_parameters.NAME == "fixed_amount":
                cashflow_value = amount * (1 + indexation_per_period) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow_value = cashflow_parameters.percentage / periods_per_year * period_initial_amount
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = cashflow_parameters._calculate_withdrawal_size(
                    last_withdrawal=last_regular_cash_flow if n > 0 else 0,
                    balance=period_initial_amount,
                    number_of_periods=n,
                )
            elif cashflow_parameters.NAME == "CWD":
                withdrawal_without_drawdowns = amount * (1 + indexation_per_period) ** n
                if drawdowns[last_date] < 0:
                    cashflow_value = cashflow_parameters._calculate_withdrawal_size(
                        drawdown=drawdowns[last_date],
                        withdrawal_without_drawdowns=withdrawal_without_drawdowns,
                    )
                else:
                    cashflow_value = withdrawal_without_drawdowns
            else:
                raise ValueError("Wrong cashflow_method value.")
            cashflow_value *= period_fraction  # adjust cash flow to the period length (months)
            last_regular_cash_flow = cashflow_value
            period_final_balance = period_wealth_index.iloc[-1] + cashflow_value
            period_wealth_index.iloc[-1] = period_final_balance
            period_initial_amount = period_final_balance
            wealth_chunks.append(period_wealth_index)
        wealth_df = (
            pd.concat(wealth_chunks, sort=False)
            if wealth_chunks
            else pd.DataFrame(dtype=float, columns=[portfolio_symbol])
        )
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
    wealth_index = wealth_index.sort_index(ascending=True)
    return wealth_index


def get_cash_flow_fv(  # noqa: C901
    ror: pd.Series | pd.DataFrame,
    portfolio_symbol: str | None,
    cashflow_parameters: cf.CashFlow,
    task: Literal["backtest", "monte_carlo"],
) -> pd.Series | pd.DataFrame:
    """
    Calculate cash flow future values (FV) for a series of returns according to withdrawal/contributions strategies.
    """
    dcf_object = cashflow_parameters.parent.dcf
    dcf_object.cashflow_parameters = cashflow_parameters
    period_initial_amount = cashflow_parameters.initial_investment
    last_regular_cash_flow = 0
    cs_fv = pd.Series(dtype=float, name="cash_flow_fv")
    amount = getattr(cashflow_parameters, "amount", None)
    if cashflow_parameters.NAME == "CWD":
        drawdowns = helpers.Frame.get_drawdowns(ror)
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
        ror_cashflow_df = ror_cashflow_df.fillna(0)
        n_rows = ror.shape[0]
        monthly_discount_rate = (1 + dcf_object.discount_rate) ** (1 / settings._MONTHS_PER_YEAR) - 1
        discount_factors = (1.0 + monthly_discount_rate) ** np.arange(n_rows)
        if task == "backtest":
            if dcf_object.cashflow_parameters.time_series_discounted_values:
                ror_cashflow_df.loc[:, "cashflow_ts"] = ror_cashflow_df.loc[:, "cashflow_ts"].mul(
                    discount_factors, axis=0
                )
        elif task == "monte_carlo":
            if not dcf_object.cashflow_parameters.time_series_discounted_values:
                ror_cashflow_df.loc[:, "cashflow_ts"] = ror_cashflow_df.loc[:, "cashflow_ts"].mul(
                    discount_factors, axis=0
                )
        else:
            raise ValueError(f"Unknown task: {task}. It must be 'monte_carlo' or 'backtest'")
    else:
        ror_cashflow_df = ror
        ror_cashflow_df.loc[:, "cashflow_ts"] = 0.0
    cash_flow_ts = ror_cashflow_df["cashflow_ts"]  # cash flow monthly time series
    periods_per_year = settings.frequency_periods_per_year[cashflow_parameters.frequency]
    if hasattr(cashflow_parameters, "indexation") and cashflow_parameters.frequency != "none":
        indexation_per_period = (1 + cashflow_parameters.indexation) ** (1 / periods_per_year) - 1
    if cashflow_parameters.frequency == "month" or cashflow_parameters.frequency == "none":
        # Fast Calculation
        for n, row in enumerate(ror.itertuples()):
            date = row[0]
            r = row[portfolio_position + 1]
            # Calculate regular cash flow
            if cashflow_parameters.frequency == "none":
                cashflow = 0
            elif cashflow_parameters.NAME == "fixed_amount":
                cashflow = amount * (1 + indexation_per_period) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow = cashflow_parameters.percentage / periods_per_year * period_initial_amount
            elif cashflow_parameters.NAME == "time_series":
                cashflow = 0
            elif cashflow_parameters.NAME == "CWD":
                withdrawal_without_drawdowns = amount * (1 + indexation_per_period) ** n
                if drawdowns[date] < 0:
                    cashflow = cashflow_parameters._calculate_withdrawal_size(
                        drawdown=drawdowns[date],
                        withdrawal_without_drawdowns=withdrawal_without_drawdowns,
                    )
                else:
                    cashflow = withdrawal_without_drawdowns
            else:
                raise ValueError("Wrong cashflow strategy name value.")
            # add Extra Withdrawals/Contributions
            cs_value = cashflow + cash_flow_ts[date]
            period_initial_amount = period_initial_amount * (r + 1) + cs_value
            cs_fv[date] = cs_value
    elif cashflow_parameters.frequency != "month" and cashflow_parameters.frequency != "none":
        # Slow Calculation
        pandas_frequency = settings.frequency_mapping[cashflow_parameters.frequency]
        months_in_full_period = settings._MONTHS_PER_YEAR / cashflow_parameters.periods_per_year
        cashflow_chunks = []  # Collect all chunks to concatenate once at the end
        for n, x in enumerate(ror_cashflow_df.resample(rule=pandas_frequency, convention="start")):
            ror_ts = x[1].iloc[:, portfolio_position]  # select ror part of the grouped data
            months_local = ror_ts.shape[0]
            period_fraction = months_local / months_in_full_period  # 1 for a full period
            cashflow_ts_local = x[1].loc[:, "cashflow_ts"].copy()
            last_date = ror_ts.index[-1]
            # CashFlow inside period (Extra cash flow)
            if (cashflow_ts_local != 0).any():
                period_wealth_index = pd.Series(dtype=float, name=portfolio_symbol)
                month_balance = period_initial_amount
                for date, r in ror_ts.items():
                    month_balance = month_balance * (r + 1) + cashflow_ts_local[date]
                    period_wealth_index[date] = month_balance
            else:
                period_wealth_index = period_initial_amount * (1 + ror_ts).cumprod()
            # CashFlow END period (Regular Cash Flow)
            if cashflow_parameters.NAME == "fixed_amount":
                cashflow_value = amount * (1 + indexation_per_period) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow_value = cashflow_parameters.percentage / periods_per_year * period_initial_amount
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = cashflow_parameters._calculate_withdrawal_size(
                    last_withdrawal=last_regular_cash_flow if n > 0 else 0,
                    balance=period_initial_amount,
                    number_of_periods=n,
                )
            elif cashflow_parameters.NAME == "CWD":
                withdrawal_without_drawdowns = amount * (1 + indexation_per_period) ** n
                if drawdowns[last_date] < 0:
                    cashflow_value = cashflow_parameters._calculate_withdrawal_size(
                        drawdown=drawdowns[last_date],
                        withdrawal_without_drawdowns=withdrawal_without_drawdowns,
                    )
                else:
                    cashflow_value = withdrawal_without_drawdowns
            else:
                raise ValueError("Wrong cashflow_method value.")
            cashflow_value *= period_fraction  # adjust cash flow to the period length (months)
            last_regular_cash_flow = cashflow_value
            period_final_balance = period_wealth_index.iloc[-1] + cashflow_value
            period_wealth_index.iloc[-1] = period_final_balance
            period_initial_amount = period_final_balance
            cashflow_ts_local.iloc[-1] += cashflow_value
            cashflow_chunks.append(cashflow_ts_local)
        cs_fv = pd.concat(cashflow_chunks, sort=False) if cashflow_chunks else cs_fv
    return cs_fv


def _cwd_reduction_factor_matrix(cashflow_parameters: cf.CashFlow, drawdowns: np.ndarray) -> np.ndarray:
    """Per-path CutWithdrawalsIfDrawdown reduction factors (1 - reduction), T x N.

    `_crash_threshold_reduction_series` is sorted by threshold descending and the
    scalar loop picks the first (deepest) threshold with |drawdown| >= threshold.
    Iterating thresholds ascending and overwriting reproduces that choice.
    """
    factors = np.ones_like(drawdowns)
    series = cashflow_parameters._crash_threshold_reduction_series.sort_index(ascending=True)
    for threshold, reduction in series.items():
        factors = np.where(np.abs(drawdowns) >= threshold, 1 - reduction, factors)
    return factors


def _vds_withdrawal_vector(
    cashflow_parameters: cf.CashFlow,
    balance: np.ndarray,
    last_withdrawal: float | np.ndarray,
    number_of_periods: int,
) -> np.ndarray:
    """Vectorized VanguardDynamicSpending withdrawal for one period across paths.

    Faithful translation of `VanguardDynamicSpending._calculate_withdrawal_size`
    to numpy, preserving the scalar branch order (in-range -> percentage;
    above max -> max; below min -> min). Scalar/vector parity is pinned by a
    property test.
    """
    withdrawal_by_percentage = balance * abs(cashflow_parameters.percentage)
    last = np.abs(last_withdrawal)
    has_floor_ceiling = cashflow_parameters.floor_ceiling is not None
    has_min_max = cashflow_parameters.min_max_annual_withdrawals is not None
    if has_floor_ceiling:
        floor, ceiling = cashflow_parameters.floor_ceiling
        adjust = (1 + cashflow_parameters.indexation) if cashflow_parameters.adjust_floor_ceiling else 1.0
        floor_indexed = last * adjust * (1 + floor)
        ceiling_indexed = last * adjust * (1 + ceiling)
    if has_min_max:
        min_withdrawal, max_withdrawal = cashflow_parameters.min_max_annual_withdrawals
        indexation_factor = (
            (1 + cashflow_parameters.indexation) ** number_of_periods if cashflow_parameters.adjust_min_max else 1.0
        )
        min_indexed = abs(min_withdrawal) * indexation_factor
        max_indexed = abs(max_withdrawal) * indexation_factor
    if has_floor_ceiling and has_min_max:
        max_final = np.where(
            ceiling_indexed > max_indexed,
            max_indexed,
            np.where((min_indexed < ceiling_indexed) & (ceiling_indexed <= max_indexed), ceiling_indexed, max_indexed),
        )
        min_final = np.where(floor_indexed > min_indexed, floor_indexed, min_indexed)
    elif has_min_max:
        min_final, max_final = min_indexed, max_indexed
    elif has_floor_ceiling:
        min_final = floor_indexed
        max_final = np.where(ceiling_indexed != 0, ceiling_indexed, withdrawal_by_percentage)
    else:
        return -withdrawal_by_percentage
    withdrawal = np.where(
        (min_final <= withdrawal_by_percentage) & (withdrawal_by_percentage <= max_final),
        withdrawal_by_percentage,
        np.where(withdrawal_by_percentage > max_final, max_final, min_final),
    )
    return -withdrawal


def _resample_slices(index: pd.PeriodIndex, pandas_frequency: str) -> list[tuple[int, int]]:
    """Start/stop integer positions of the groups of `resample(rule, convention="start")`.

    One cheap resample on a marker Series replaces N identical per-path resamples
    and guarantees the exact same grouping as the per-path reference.
    """
    marker = pd.Series(np.zeros(len(index)), index=index)
    slices = []
    position = 0
    for _, group in marker.resample(pandas_frequency, convention="start"):
        if group.shape[0] == 0:
            continue
        slices.append((position, position + group.shape[0]))
        position += group.shape[0]
    return slices


def _simulate_paths_mc(  # noqa: C901
    ror: pd.DataFrame,
    cashflow_parameters: cf.CashFlow,
    discount_rate: float,
    *,
    initial_balance: float | np.ndarray | None = None,
    month_offset: int = 0,
    task: Literal["monte_carlo", "backtest"] = "monte_carlo",
) -> tuple[np.ndarray, np.ndarray]:
    """One vectorized Monte Carlo pass over all paths.

    Returns ``(wealth, cash_flow)`` matrices of shape (T, N): the wealth index
    (without the prepended initial row) and the total cash flow recorded per
    month (regular strategy cash flow at its due month plus extra cash flows
    from `time_series`), matching `get_wealth_indexes_fv_with_cashflow` and
    `get_cash_flow_fv` (task="monte_carlo") respectively.

    ``initial_balance`` overrides the opening balance of every scenario. A
    scalar applies to all of them; an array must have one value per column. It
    is how a multi-stage plan hands the terminal balances of one stage to the
    next. When None the strategy's ``initial_investment`` is used, which is the
    single-portfolio behaviour.

    ``month_offset`` is the number of months elapsed since the start of a
    multi-stage plan. Everything that indexes with time — the compounding of
    extra cash flows, the indexation of ``amount``, the CWD base withdrawal and
    the VDS ``number_of_periods`` — is measured from there rather than from the
    first row of ``ror``, so a later stage's amounts stay expressed in the money
    of the plan's start. It must be a whole number of cash flow periods.

    ``task`` controls how ``time_series`` cash flows are compounded:
    ``"monte_carlo"`` compounds values given in nominal terms, ``"backtest"``
    compounds values given in present-value terms, mirroring the per-path
    reference ``get_wealth_indexes_fv_with_cashflow``.
    """
    if task not in ("monte_carlo", "backtest"):
        raise ValueError(f"Unknown task: {task}. It must be 'monte_carlo' or 'backtest'")
    n_rows, n_cols = ror.shape
    returns = ror.to_numpy(dtype=float)
    frequency = cashflow_parameters.frequency
    periods_per_year = settings.frequency_periods_per_year[frequency]
    amount = getattr(cashflow_parameters, "amount", None)
    indexation_per_period = 0.0
    if hasattr(cashflow_parameters, "indexation") and frequency != "none":
        indexation_per_period = (1 + cashflow_parameters.indexation) ** (1 / periods_per_year) - 1

    if frequency == "none":
        period_offset = 0
    else:
        months_in_period = settings._MONTHS_PER_YEAR / periods_per_year
        if month_offset % months_in_period:
            raise ValueError(
                f"month_offset={month_offset} is not a whole number of '{frequency}' "
                f"periods ({months_in_period:g} months each)."
            )
        period_offset = int(month_offset // months_in_period)

    # Extra cash flows from `time_series`, aligned to the simulation index.
    extra_cf = np.zeros(n_rows)
    cash_flow_ts = cashflow_parameters.time_series
    if not (cash_flow_ts.empty or (cash_flow_ts == 0).all()):
        aligned = cash_flow_ts.reindex(ror.index).fillna(0).to_numpy(dtype=float)
        discounted_values = cashflow_parameters.time_series_discounted_values
        # A backtest compounds values given in present value terms; a forecast
        # compounds values given in nominal terms. Mirrors the per-path engine.
        compound = discounted_values if task == "backtest" else not discounted_values
        if compound:
            monthly_discount_rate = (1 + discount_rate) ** (1 / settings._MONTHS_PER_YEAR) - 1
            aligned = aligned * (1.0 + monthly_discount_rate) ** np.arange(month_offset, month_offset + n_rows)
        extra_cf = aligned

    # Per-path drawdowns and reduction factors for the CWD strategy
    # (they depend on the return paths only, so they are precomputed once).
    if cashflow_parameters.NAME == "CWD":
        wealth_for_dd = 1000 * np.cumprod(1 + returns, axis=0)
        peaks = np.maximum.accumulate(wealth_for_dd, axis=0)
        drawdowns = (wealth_for_dd - peaks) / peaks
        cwd_factors = _cwd_reduction_factor_matrix(cashflow_parameters, drawdowns)

    wealth = np.empty((n_rows, n_cols))
    cash_flow = np.zeros((n_rows, n_cols))
    if initial_balance is None:
        balance = np.full(n_cols, float(cashflow_parameters.initial_investment))
    else:
        balance = np.broadcast_to(np.asarray(initial_balance, dtype=float), (n_cols,)).astype(float, copy=True)

    if frequency in ("month", "none"):
        for n in range(n_rows):
            if frequency == "none":
                cashflow: float | np.ndarray = 0.0
            elif cashflow_parameters.NAME == "fixed_amount":
                cashflow = amount * (1 + indexation_per_period) ** (n + period_offset)
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow = cashflow_parameters.percentage / periods_per_year * balance
            elif cashflow_parameters.NAME == "time_series":
                cashflow = 0.0
            elif cashflow_parameters.NAME == "CWD":
                base_withdrawal = amount * (1 + indexation_per_period) ** (n + period_offset)
                cashflow = base_withdrawal * np.where(drawdowns[n] < 0, cwd_factors[n], 1.0)
            else:
                raise ValueError("Wrong cashflow strategy name value.")
            cs_value = cashflow + extra_cf[n]
            cash_flow[n] = cs_value
            balance = balance * (1 + returns[n]) + cs_value
            wealth[n] = balance
    else:
        months_in_full_period = settings._MONTHS_PER_YEAR / cashflow_parameters.periods_per_year
        last_regular_cashflow: float | np.ndarray = 0.0
        for n, (start, stop) in enumerate(_resample_slices(ror.index, cashflow_parameters._pandas_frequency)):
            start_balance = balance
            cash_flow[start:stop] = extra_cf[start:stop][:, None]
            if np.any(extra_cf[start:stop] != 0):
                for k in range(start, stop):
                    balance = balance * (1 + returns[k]) + extra_cf[k]
                    wealth[k] = balance
            else:
                segment = np.cumprod(1 + returns[start:stop], axis=0) * start_balance
                wealth[start:stop] = segment
                balance = segment[-1]
            # Regular cash flow at the period end.
            if cashflow_parameters.NAME == "fixed_amount":
                cashflow_value: float | np.ndarray = amount * (1 + indexation_per_period) ** (n + period_offset)
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow_value = cashflow_parameters.percentage / periods_per_year * start_balance
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = _vds_withdrawal_vector(
                    cashflow_parameters,
                    start_balance,
                    last_regular_cashflow if n > 0 else 0.0,
                    n + period_offset,
                )
            elif cashflow_parameters.NAME == "CWD":
                base_withdrawal = amount * (1 + indexation_per_period) ** (n + period_offset)
                cashflow_value = base_withdrawal * np.where(drawdowns[stop - 1] < 0, cwd_factors[stop - 1], 1.0)
            else:
                raise ValueError("Wrong cashflow_method value.")
            cashflow_value = cashflow_value * ((stop - start) / months_in_full_period)
            last_regular_cashflow = cashflow_value
            cash_flow[stop - 1] += cashflow_value
            balance = balance + cashflow_value
            wealth[stop - 1] = balance

    return wealth, cash_flow


def get_wealth_indexes_fv_with_cashflow_mc(
    ror: pd.DataFrame,
    cashflow_parameters: cf.CashFlow,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Vectorized Monte Carlo counterpart of `get_wealth_indexes_fv_with_cashflow`.

    Computes wealth index future values (FV) for all random return paths at
    once; one Python loop over time steps with numpy operations across paths.
    Implements the "monte_carlo" task semantics only: extra cash flows from
    `time_series` are compounded with the discount rate unless
    `time_series_discounted_values` is True. Replicates the per-path reference
    exactly; equivalence is pinned by tests. (The two historical reference
    quirks were fixed together with this engine — see GitHub issues #81, #82.)
    """
    wealth, _ = _simulate_paths_mc(ror, cashflow_parameters, discount_rate)
    n_cols = ror.shape[1]
    initial_investment = cashflow_parameters.initial_investment
    out_index = ror.index.insert(0, ror.index[0] - 1)
    data = np.vstack([np.full((1, n_cols), float(initial_investment)), wealth])
    return pd.DataFrame(data, index=out_index, columns=ror.columns)


def get_cash_flow_fv_mc(
    ror: pd.DataFrame,
    cashflow_parameters: cf.CashFlow,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Vectorized Monte Carlo counterpart of `get_cash_flow_fv`.

    Returns the monthly cash flow (regular strategy cash flow plus extra cash
    flows from `time_series`) for all random return paths at once, on the same
    index as `ror` (no prepended initial row). Equivalence with the per-path
    reference is pinned by tests.
    """
    _, cash_flow = _simulate_paths_mc(ror, cashflow_parameters, discount_rate)
    return pd.DataFrame(cash_flow, index=ror.index, columns=ror.columns)


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


def zero_wealth_after_first_void(wealth: pd.DataFrame) -> pd.DataFrame:
    """Vectorized frame counterpart of per-column `remove_negative_values` + `fillna(0)`.

    For every column, the first non-positive value and everything after it are
    replaced with 0. Returns a new DataFrame; the input is not modified.
    """
    values = wealth.to_numpy(dtype=float)
    voided = values <= 0
    has_void = voided.any(axis=0)
    first_void = voided.argmax(axis=0)
    rows = np.arange(values.shape[0])[:, None]
    masked = np.where(has_void & (rows >= first_void), 0.0, values)
    return pd.DataFrame(masked, index=wealth.index, columns=wealth.columns)


def discount_monthly_cash_flow(
    cash_flow_fv: pd.Series | pd.DataFrame,
    annual_effective_discount_rate: float,
    reverse: bool = False,
) -> pd.Series | pd.DataFrame:
    number_of_months = cash_flow_fv.shape[0]
    monthly_discount_rate = (1 + annual_effective_discount_rate) ** (1 / settings._MONTHS_PER_YEAR) - 1
    if not reverse:
        discount_factors = (1.0 + monthly_discount_rate) ** np.arange(number_of_months)
    else:
        discount_factors = (1.0 + monthly_discount_rate) ** np.arange(number_of_months)[::-1]
    return cash_flow_fv.div(discount_factors, axis=0)


def _irr_initial_guess(cashflows: np.ndarray) -> np.ndarray:
    """
    Cheap per-column periodic-rate seed for Newton IRR iteration.

    For the dominant single-outflow/single-inflow case this equals
    ``(total_in / total_out) ** (1 / horizon) - 1``, i.e. the true periodic rate,
    so Newton converges in ~1 iteration.
    """
    n_periods = cashflows.shape[0]
    horizon = max(n_periods - 1, 1)
    inflows = np.where(cashflows > 0.0, cashflows, 0.0).sum(axis=0)
    outflows = -np.where(cashflows < 0.0, cashflows, 0.0).sum(axis=0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = np.where(outflows > 0.0, inflows / outflows, 1.0)
    ratio = np.where(ratio > 0.0, ratio, 1.0)
    return ratio ** (1.0 / horizon) - 1.0


def _irr_brentq_column(cashflow_column: np.ndarray) -> float:
    """Bracketing-solver fallback for a single cash-flow column. Returns NaN if no bracket."""
    n_periods = cashflow_column.shape[0]
    t = np.arange(n_periods, dtype=float)
    # Lower bracket: the rate closest to -1 for which (1 + rate) ** -(n_periods - 1) stays
    # finite (float64 overflows past ~1e308, and inf * 0 would poison the NPV with NaN).
    # Solving (1 + rate) > 10 ** (-300 / (n - 1)) keeps every brentq evaluation overflow-free
    # while preserving the widest possible root range; short columns floor at -1 + 1e-9.
    lower_bracket = -1.0 + max(1e-9, 10.0 ** (-300.0 / max(n_periods - 1, 1)))

    def npv(rate: float) -> float:
        # Overflow is expected for large negative rates and large t; silence the warning.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return float((cashflow_column * (1.0 + rate) ** (-t)).sum())

    try:
        # Upper bracket: any economically meaningful periodic rate is well below 1e6.
        return optimize.brentq(npv, lower_bracket, 1e6, xtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")


def irr_of_cashflow_matrix(
    cashflows: np.ndarray,
    periods_per_year: int = 12,
    guess: np.ndarray | float | None = None,
    xtol: float = 1e-10,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Annualized effective IRR for each column of a cash-flow matrix.

    Solves, per column, the periodic rate ``r`` with ``sum_t cf[t] / (1 + r) ** t = 0``
    (row index ``t`` = period number, row 0 = t0), then annualizes to
    ``(1 + r) ** periods_per_year - 1``.

    The Newton iteration runs over all columns simultaneously with the analytic
    derivative. Columns that do not converge but have a sign change fall back to a
    bracketing solver. Columns with no sign change (no real root) return NaN.

    Parameters
    ----------
    cashflows : np.ndarray
        Shape ``(n_periods, n_series)`` or ``(n_periods,)``. Each column is one series.
    periods_per_year : int, default 12
        Periods per year used for annualization (12 for monthly grids).
    guess : np.ndarray or float or None, default None
        Periodic-rate seed. If None, a ratio-based seed is used.
    xtol : float, default 1e-10
        Convergence tolerance on the Newton step (scale-free).
    max_iter : int, default 50
        Maximum Newton iterations.

    Returns
    -------
    np.ndarray
        Shape ``(n_series,)``. Annualized effective IRR per column; NaN where no root.
        A 1-D input is treated as a single column and returns shape ``(1,)``.
    """
    cf = np.asarray(cashflows, dtype=float)
    if cf.ndim == 1:
        cf = cf[:, None]
    n_periods, n_series = cf.shape
    t = np.arange(n_periods, dtype=float)[:, None]  # (n_periods, 1)

    if guess is None:
        rate = _irr_initial_guess(cf)
    elif np.isscalar(guess):
        rate = np.full(n_series, float(guess))
    else:
        rate = np.asarray(guess, dtype=float).copy()

    eps = 1e-12
    has_sign_change = (cf > 0.0).any(axis=0) & (cf < 0.0).any(axis=0)

    # A clipped iterate (rate -> -1 + eps) makes (1 + rate) ** (-t) overflow for large t.
    # Such columns are detected below and resolved by brentq or set to NaN, so the
    # intermediate overflow/invalid values are expected; silence their warnings.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(max_iter):
            base = np.where(1.0 + rate <= eps, eps, 1.0 + rate)  # (n_series,)
            disc = base[None, :] ** (-t)  # (n_periods, n_series)
            f = (cf * disc).sum(axis=0)  # (n_series,)
            fprime = -(t * cf * disc).sum(axis=0) / base  # (n_series,)
            step = np.where(fprime != 0.0, f / fprime, 0.0)
            rate = rate - step
            rate = np.where(rate <= -1.0 + eps, -1.0 + eps, rate)
            finite_step = step[np.isfinite(step)]
            if finite_step.size and np.abs(finite_step).max() < xtol:
                break

        # Validate convergence (scale-free residual) and retry stragglers with brentq.
        base = np.where(1.0 + rate <= eps, eps, 1.0 + rate)
        residual = (cf * base[None, :] ** (-t)).sum(axis=0)
    scale = np.abs(cf).sum(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    not_converged = np.abs(residual) / scale > 1e-8
    for j in np.flatnonzero(not_converged & has_sign_change):
        rate[j] = _irr_brentq_column(cf[:, j])

    rate = np.where(has_sign_change, rate, np.nan)
    return (1.0 + rate) ** periods_per_year - 1.0


class LegacyMonteCarlo(MonteCarlo):
    def backtesting_error(self, var_level: int = 5) -> dict:
        """
        Calculate Backtesting Error as the difference between empirical and theoretical
        risk measures (VaR, CVaR) and arithmetic mean.

        Parameters
        ----------
        var_level : int, default 5
            Confidence level in percent for Value-at-Risk (VaR) and Conditional
            Value-at-Risk (CVaR). For example, 5 corresponds to 5% left tail.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - 'delta_arithmetic_mean': float
                Difference between the theoretical and empirical arithmetic mean
                of returns.
            - 'delta_var': float
                Difference between empirical and theoretical Value-at-Risk at the
                specified level.
            - 'delta_cvar': float
                Difference between empirical and theoretical Conditional
                Value-at-Risk at the specified level.
        """
        parameters = self.get_parameters_for_distribution()
        # Var CVaR
        var_theor = tails.var_theoretical(distr=self.distribution, alpha=var_level / 100, args=parameters)
        cvar_theor = tails.cvar_theoretical(distr=self.distribution, alpha=var_level / 100, args=parameters)
        var_emp = -helpers.Frame.get_var_historic(ror=self.ror, level=var_level)
        cvar_emp = -helpers.Frame.get_cvar_historic(ror=self.ror, level=var_level)
        delta_var = var_emp - var_theor
        delta_cvar = cvar_emp - cvar_theor
        # Arithmetic mean
        parameters = self.get_parameters_for_distribution()
        match self.distribution:
            case "norm":
                mean_t = parameters[0]
            case "lognorm":
                mean_t = scipy.stats.lognorm.mean(*parameters)
            case "t":
                mean_t = parameters[1]
            case _:
                raise ValueError("Unknown distribution: " + self.distribution)
        delta_arithmetic_mean = mean_t - self.ror.mean()
        return {
            "delta_arithmetic_mean": float(delta_arithmetic_mean),
            "delta_var": float(delta_var),
            "delta_cvar": float(delta_cvar),
        }

    def _get_cagr_distribution(self) -> pd.Series:
        """
        Generate CAGR distribution for the rate of return distribution.

        CAGR is calculated for each of n random returns time series.
        """
        return_ts = self.monte_carlo_returns_ts
        return helpers.Frame.get_cagr(return_ts)

    def percentile_distribution_cagr(
        self,
        percentiles: list[int] = [10, 50, 90],  # noqa: B006
    ) -> dict[int, float]:
        """
        Calculate percentiles for the simulated CAGR distribution.

        CAGR (Compound Annual Growth Rate) is calculated for each Monte Carlo return path.

        Parameters
        ----------
        percentiles : list[int], default [10, 50, 90]
            Percentiles to compute (0-100).

        Returns
        -------
        dict[int, float]
            Mapping `{percentile: value}`.

        Examples
        --------
        >>> pf = ok.Portfolio(
        ...     ["SPY.US", "AGG.US", "GLD.US"],
        ...     weights=[0.60, 0.35, 0.05],
        ...     rebalancing_strategy=ok.Rebalance(period="year"),
        ... )
        >>> pf.dcf.set_mc_parameters(distribution="norm", period=1)
        >>> pf.dcf.mc.percentile_distribution_cagr()
        {10: ..., 50: ..., 90: ...}
        >>> pf.dcf.set_mc_parameters(period=5)
        >>> pf.dcf.mc.percentile_distribution_cagr([5, 10, 20])
        {5: ..., 10: ..., 20: ...}
        """
        cagr_distr = self._get_cagr_distribution()
        results = {}
        for percentile in percentiles:
            value = cagr_distr.quantile(percentile / 100)
            results.update({percentile: value})
        return results

    def plot_qq(
        self,
        var_level: int = 5,
        bootstrap_size_var: int = 2000,
        zoom_to_left_tail: int = 20,
        figsize: tuple | None = None,
    ) -> Axes:
        """
        Generate a quantile-quantile (Q-Q) plot of portfolio monthly rate of return against quantiles of a given
        theoretical distribution.

        A q-q plot is a plot of the quantiles of the portfolio rate of return historical data
        against the quantiles of a given theoretical distribution.

        Bootstrap bands in a Q–Q plot are bootstrap-based confidence envelopes around quantiles that show
        the amount of random sample-to-sample variation one would expect. They bands built by repeatedly resampling
        dataset of a given size and recomputing the Q–Q points.

        Parameters
        ----------
        var_level : int, default 5
            Confidence level in percent for VaR and CVaR.

        bootstrap_size_var : int, default 2000
            Number of bootstrap resamples used to compute confidence intervals for empirical
            VaR and CVaR. If 0, the bootstrap stripe is not drawn. A larger number provides a
            smoother estimate of the confidence bands at the cost of computation time.

        zoom_to_left_tail : int or None, default 20
            Zoom the plot to the left tail by limiting the view to the
            [0.1%, `zoom_to_left_tail`%] percentile range. Must be in [1, 98]. Use `None`
            to show the full range.

        figsize : tuple[float, float], default None
            Figure size in inches (width, height). If `None`, matplotlib default is used.

        Returns
        -------
        Axes
            Matplotlib axes object.

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> pf = ok.Portfolio(
        ...     ["SPY.US", "AGG.US", "GLD.US"],
        ...     weights=[0.60, 0.35, 0.05],
        ...     rebalancing_strategy=ok.Rebalance(period="year"),
        ... )
        >>> pf.dcf.set_mc_parameters(distribution="t")
        >>> pf.dcf.mc.plot_qq(bootstrap_size_var=2000, zoom_to_left_tail=50, figsize=(10, 10))
        >>> plt.show()
        """
        distr = self.distribution
        parameters = self.get_parameters_for_distribution()
        _, ax = plt.subplots(figsize=figsize)
        alpha = var_level / 100
        if zoom_to_left_tail in range(1, 99):
            p_zoom = zoom_to_left_tail / 100
        elif zoom_to_left_tail is None:
            p_zoom = 1
        else:
            raise ValueError("Zoom level must be between 1 and 99 (or None).")
        var_emp = -helpers.Frame.get_var_historic(ror=self.ror, level=int(alpha * 100))
        cvar_emp = -helpers.Frame.get_cvar_historic(ror=self.ror, level=int(alpha * 100))
        if distr == "norm":
            distargs = ()
            distribution = scipy.stats.norm
            x10_theor = scipy.stats.norm.ppf(p_zoom, loc=parameters[0], scale=parameters[1])
            x001_theor = scipy.stats.norm.ppf(0.001, loc=parameters[0], scale=parameters[1])
            title = f"QQ-plot: Normal mu={parameters[0]:.3f}, sigma={parameters[1]:.3f}"
        elif distr == "lognorm":
            distargs = (parameters[0],)
            distribution = scipy.stats.lognorm
            x10_theor = scipy.stats.lognorm.ppf(p_zoom, s=parameters[0], loc=parameters[1], scale=parameters[2])
            x001_theor = scipy.stats.lognorm.ppf(0.001, s=parameters[0], loc=parameters[1], scale=parameters[2])
            title = f"QQ-plot: Lognormal shape={parameters[0]:.3f}, loc={parameters[1]:.3f}, scale={parameters[2]:.3f}"
        elif distr == "t":
            distargs = (parameters[0],)
            distribution = scipy.stats.t
            x10_theor = scipy.stats.t.ppf(p_zoom, df=parameters[0], loc=parameters[1], scale=parameters[2])
            x001_theor = scipy.stats.t.ppf(0.001, df=parameters[0], loc=parameters[1], scale=parameters[2])
            title = f"QQ-plot: Student's t df={parameters[0]:.3f}, loc={parameters[1]:.3f}, scale={parameters[2]:.3f}"
        else:
            raise ValueError('distr should be "norm" (default), "lognorm" or "t".')
        var_theor = tails.var_theoretical(distr=distr, alpha=alpha, args=parameters)
        cvar_theor = tails.cvar_theoretical(distr=distr, alpha=alpha, args=parameters)
        delta_var = var_emp - var_theor
        delta_cvar = cvar_emp - cvar_theor

        # VaR points
        ax.scatter([var_theor], [var_emp], color="tab:red", s=40, zorder=3, label=f"VaR {alpha:.0%}")
        ax.axvline(var_theor, color="tab:red", ls=":", lw=1.2)
        ax.axhline(var_emp, color="tab:red", ls="--", lw=1.2)

        ax.annotate(
            f"delta var={delta_var:.4f}",
            xy=(var_theor, var_emp),
            xytext=(10, -15),
            textcoords="offset points",
            color="tab:red",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "tab:red", "alpha": 0.7},
        )

        # CVAR points
        ax.scatter([cvar_theor], [cvar_emp], color="yellow", s=40, zorder=3, label=f"CVAR {alpha:.0%}")
        ax.axvline(cvar_theor, color="tab:blue", ls=":", lw=1.2)
        ax.axhline(cvar_emp, color="tab:blue", ls="--", lw=1.2)

        ax.annotate(
            f"delta cvar={delta_cvar:.4f}",
            xy=(cvar_theor, cvar_emp),
            xytext=(10, -15),
            textcoords="offset points",
            color="tab:red",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "tab:red", "alpha": 0.7},
        )

        sm.qqplot(
            self.ror,
            dist=distribution,
            distargs=distargs,
            loc=parameters[0] if distr == "norm" else parameters[1],
            scale=parameters[1] if distr == "norm" else parameters[2],
            fit=False,
            line="45",
            markersize=3,
            alpha=0.7,
            ax=ax,
        )
        # BOOTSTRAP VaR & CVaR
        if bootstrap_size_var:
            bootsrap_number = bootstrap_size_var
            rng = np.random.default_rng(seed=11)
            samples = rng.choice(self.ror, size=(bootsrap_number, self.ror.size), replace=True)
            boot_var = np.quantile(samples, alpha, axis=1)
            boot_cvar = np.apply_along_axis(tails.cvar_of_sample, 1, samples, alpha)
            var_ci = np.quantile(boot_var, [0.025, 0.975])
            cvar_ci = np.quantile(boot_cvar, [0.025, 0.975])
            # Stripe 95% CI emp VaR & CVar
            ax.axhspan(var_ci[0], var_ci[1], color="tab:blue", alpha=0.08, label="95% CI Emp-VaR (bootstrap)")
            ax.axhspan(cvar_ci[0], cvar_ci[1], color="tab:green", alpha=0.08, label="95% CI Emp-CVaR (bootstrap)")

        # Zoom to the left tail (large drawdowns)
        if zoom_to_left_tail:
            y10_emp = float(np.quantile(self.ror, p_zoom))
            y001_emp = float(np.quantile(self.ror, 0.001))
            ax.set_xlim(x001_theor, x10_theor)
            ax.set_ylim(y001_emp, y10_emp)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title)
        ax.figure.tight_layout()
        # Log metrics using f-strings
        logger.info(
            f"VaR  {alpha:.0%}: theor={var_theor:.6f}, emp={var_emp:.6f}, delta(emp-theor)={(var_emp - var_theor):.6f}"
        )
        logger.info(
            f"CVAR {alpha:.0%}: theor={cvar_theor:.6f}, emp={cvar_emp:.6f}, delta(emp-theor)={(cvar_emp - cvar_theor):.6f}"
        )
        # Log bootstrap confidence intervals only if they were computed
        if bootstrap_size_var:
            logger.info(f"95% CI empiric VaR (bootstrap): [{var_ci[0]:.6f}, {var_ci[1]:.6f}]")
            logger.info(f"95% CI empiric CVaR (bootstrap): [{cvar_ci[0]:.6f}, {cvar_ci[1]:.6f}]")
        return ax


class LegacyPortfolioDCF(PortfolioDCF):
    def __init__(self, parent: core.Portfolio) -> None:
        super().__init__(parent)
        self.mc = LegacyMonteCarlo(self)

    def wealth_index(
        self, discounting: Literal["fv", "pv"], include_negative_values: bool = False
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate wealth index time series for the portfolio with cash flow (contributions and withdrawals)
        using historical rate of returns.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period considering cash flows.

        Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        If there is no cash flow, Wealth index is obtained from the accumulated return multiplied
        by the initial investments. That is: initial_investment * (Accumulated_Return + 1)

        Parameters
        ----------
        discounting : {'fv', 'pv'}
            Type of discounting to apply:
            - 'fv': Future Values - nominal values without discounting
            - 'pv': Present Values - values discounted to present using the discount rate
        include_negative_values : bool, default False
            Whether to include negative wealth index values in the result.
            If False, negative values are replaced with zeros.

        Returns
        -------
        pd.DataFrame
            Time series of wealth index values for portfolio and accumulated inflation (if applicable).

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> pf = ok.Portfolio(["VOO.US", "GLD.US"], weights=[0.8, 0.2])
        >>> ind = ok.IndexationStrategy(pf)  # Set Cash Flow Strategy parameters
        >>> ind.initial_investment = 100  # initial investments value
        >>> ind.frequency = "year"  # withdrawals frequency
        >>> ind.amount = -6  # initial withdrawals amount
        >>> ind.indexation = "inflation"  # the indexation is equal to inflation
        >>> pf.dcf.cashflow_parameters = ind  # assign the strategy to Portfolio
        >>> pf.dcf.wealth_index(discounting="fv", include_negative_values=False).plot()
        >>> plt.show()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._wealth_index_fv.empty:
            df = self.parent._add_inflation()
            infl_symbol = self.parent.inflation if hasattr(self.parent, "inflation") else None
            df = dcf_calculations.get_wealth_indexes_fv_with_cashflow(
                ror=df,
                portfolio_symbol=self.parent.symbol,
                inflation_symbol=infl_symbol,
                cashflow_parameters=self.cashflow_parameters,
                task="backtest",
            )
            self._wealth_index_fv = self.parent._make_df_if_series(df)
        wealth_index_fv = self._wealth_index_fv.copy()
        if not include_negative_values:
            wealth_index_fv_s = dcf_calculations.remove_negative_values(self._wealth_index_fv[self.parent.name])
            wealth_index_fv[self.parent.name] = wealth_index_fv_s.fillna(0)
        if discounting.lower() == "fv":
            return wealth_index_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(wealth_index_fv, self.discount_rate, reverse=False)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")

    def cash_flow_ts(self, discounting: Literal["fv", "pv"], remove_if_wealth_index_negative: bool = True) -> pd.Series:
        """
        Cash flow Future Values (FV) or Present Values (PV) time series for the portfolio.

        The cash flow time series presents the values of contributions and withdrawals over
        historical time period.

        Parameters
        ----------
        discounting : {'fv', 'pv'}
            Type of discounting to apply:
            - 'fv': Future Values - nominal values without discounting
            - 'pv': Present Values - values discounted to present using the discount rate
        remove_if_wealth_index_negative : bool, default True
            If True, cash flow values are replaced with 0 when wealth index is negative (or zero).

        Returns
        -------
        pd.Series
            Time series of cash flow values for portfolio.

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> pf = ok.Portfolio(["VOO.US", "GLD.US"], weights=[0.8, 0.2])
        >>> ind = ok.IndexationStrategy(pf)  # Set Cash Flow Strategy parameters
        >>> ind.initial_investment = 100  # initial investments value
        >>> ind.frequency = "year"  # withdrawals frequency
        >>> ind.amount = -0.5 * 12  # initial withdrawals amount
        >>> ind.indexation = "inflation"  # the indexation is equal to inflation
        >>> pf.dcf.cashflow_parameters = ind  # assign the strategy to Portfolio
        >>> pf.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=True).plot()
        >>> plt.show()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._cash_flow_fv.empty:
            df = self.parent.ror
            self._cash_flow_fv = dcf_calculations.get_cash_flow_fv(
                ror=df,
                portfolio_symbol=self.parent.symbol,
                cashflow_parameters=self.cashflow_parameters,
                task="backtest",
            )
        cash_flow_fv = self._cash_flow_fv.copy()
        if remove_if_wealth_index_negative:
            wealth_index = self.wealth_index(discounting="fv", include_negative_values=False)
            condition = wealth_index[self.parent.name] == 0
            cash_flow_fv[condition] = 0
        if discounting.lower() == "fv":
            return cash_flow_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(cash_flow_fv, self.discount_rate, reverse=False)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")

    @property
    def wealth_index_fv_with_assets(self) -> pd.DataFrame:
        """
        Wealth index Future Values (FV) time series for the portfolio and all assets considering cash flow (contributions and
        withdrawals).

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period. Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        If there is no cash flow, Wealth index is obtained from the accumulated return multiplied
        by the initial investments. That is: initial_investment * (Accumulated_Return + 1)

        Returns
        -------
        DataFrame
            Time series of wealth index values for portfolio, each asset and accumulated inflation.

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> pf = ok.Portfolio(["VOO.US", "GLD.US"], weights=[0.8, 0.2])
        >>> ind = ok.IndexationStrategy(pf)  # Set Cash Flow Strategy parameters
        >>> ind.initial_investment = 100  # initial investments value
        >>> ind.frequency = "year"  # withdrawals frequency
        >>> ind.amount = -0.5 * 12  # initial withdrawals amount
        >>> ind.indexation = "inflation"  # the indexation is equal to inflation
        >>> pf.dcf.cashflow_parameters = ind  # assign the strategy to Portfolio
        >>> pf.dcf.wealth_index_fv_with_assets.plot()
        >>> plt.show()
        """
        ls = [self.parent.ror, self.parent.assets_ror]
        ror_df = pd.concat(ls, axis=1, join="inner")
        wealth_df = ror_df.apply(
            dcf_calculations.get_wealth_indexes_fv_with_cashflow,
            axis=0,
            args=(None, None, self.cashflow_parameters, "backtest"),  # symbol  # inflation_symbol
        )
        if hasattr(self.parent, "inflation"):
            inflation_wi = helpers.Frame.get_wealth_indexes(
                ror=self.parent.inflation_ts, initial_amount=self.cashflow_parameters.initial_investment
            )
            wealth_df = pd.concat([wealth_df, inflation_wi], axis=1, join="inner")
        return wealth_df

    def monte_carlo_wealth(
        self, discounting: Literal["fv", "pv"] = "fv", include_negative_values: bool = True
    ) -> pd.DataFrame:
        """
        Calculate portfolio random wealth indexes with cash flows (withdrawals/contributions) using Monte Carlo simulation.

        Monte Carlo simulation generates n random monthly time series.
        Each wealth index is calculated with rate of return time series of a given distribution.

        First date of forecasted returns is portfolio last_date.
        First value for the forecasted wealth indexes is the last historical portfolio index value. It is useful
        for a chart with historical wealth index and forecasted values.

        Parameters
        ----------
        discounting : {'fv', 'pv'}, default 'fv'
            'fv' - calculate Future Value (not discounted) wealth indexes.
            'pv' - calculate Present Value (discounted) wealth indexes.
        include_negative_values : bool, default True
            If True, negative values in the wealth index are preserved.
            If False, negative values are replaced with 0.

        Returns
        -------
        DataFrame
            Table with n random wealth indexes monthly time series.

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> pf = ok.Portfolio(
        ...     ["SPY.US", "AGG.US", "GLD.US"],
        ...     weights=[0.60, 0.35, 0.05],
        ...     rebalancing_strategy=ok.Rebalance(period="month"),
        ... )
        >>> pf.dcf.set_mc_parameters(distribution="t", period=10, mc_number=100)  # Set Monte Carlo parameters
        >>> # set cash flow parameters
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> ind.amount = -500  # set withdrawal size
        >>> ind.frequency = "year"  # set withdrawal frequency
        >>> pf.dcf.cashflow_parameters = ind  # assign cash flow strategy to portfolio
        >>> pf.dcf.monte_carlo_wealth(discounting="fv").plot()
        >>> plt.legend("")  # don't show legend for each line
        >>> plt.show()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._monte_carlo_wealth_fv.empty:
            return_ts = self.mc.monte_carlo_returns_ts
            self._monte_carlo_wealth_fv = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(
                return_ts, self.cashflow_parameters, self.discount_rate
            )
        if not include_negative_values:
            monte_carlo_wealth_fv = dcf_calculations.zero_wealth_after_first_void(self._monte_carlo_wealth_fv)
        else:
            monte_carlo_wealth_fv = self._monte_carlo_wealth_fv.copy()
        if discounting.lower() == "fv":
            return monte_carlo_wealth_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(monte_carlo_wealth_fv, self.discount_rate)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")

    def monte_carlo_cash_flow(
        self, discounting: Literal["fv", "pv"], remove_if_wealth_index_negative: bool = True
    ) -> pd.DataFrame:
        """
        Calculate portfolio random cash flow (withdrawals/contributions) time series by Monte Carlo simulation.

        Monte Carlo simulation generates n random monthly time series of cash flows.
        Each cash flow time series corresponds to a wealth index calculated with rate of return time series of a given distribution.

        First date of forecasted returns is portfolio last_date.

        Parameters
        ----------
        discounting : {'fv', 'pv'}
            Type of discounting to apply:
            - 'fv': Future Values - nominal values without discounting
            - 'pv': Present Values - values discounted to present using the discount rate
        remove_if_wealth_index_negative : bool, default True
            If True, cash flow values are replaced with 0 when wealth index is negative (or zero).

        Returns
        -------
        DataFrame
            Table with n random cash flow monthly time series.

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> pf = ok.Portfolio(
        ...     ["SPY.US", "AGG.US", "GLD.US"],
        ...     weights=[0.60, 0.35, 0.05],
        ...     rebalancing_strategy=ok.Rebalance(period="month"),
        ... )
        >>> pf.dcf.set_mc_parameters(distribution="t", period=10, mc_number=100)  # Set Monte Carlo parameters
        >>> # set cash flow parameters
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> ind.amount = -500  # set withdrawal size
        >>> ind.frequency = "year"  # set withdrawal frequency
        >>> pf.dcf.cashflow_parameters = ind  # assign cash flow strategy to portfolio
        >>> pf.dcf.monte_carlo_cash_flow(discounting="fv").plot()
        >>> plt.legend("")  # don't show legend for each line
        >>> plt.show()
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        if self._monte_carlo_cash_flow_fv.empty:
            return_ts = self.mc.monte_carlo_returns_ts
            self._monte_carlo_cash_flow_fv = dcf_calculations.get_cash_flow_fv_mc(
                return_ts, self.cashflow_parameters, self.discount_rate
            )
        mc_cash_flow_fv = self._monte_carlo_cash_flow_fv.copy()
        if remove_if_wealth_index_negative:
            mc_wealth_index = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
            condition = mc_wealth_index == 0
            mc_cash_flow_fv[condition] = 0
        if discounting.lower() == "fv":
            return mc_cash_flow_fv
        elif discounting.lower() == "pv":
            return dcf_calculations.discount_monthly_cash_flow(mc_cash_flow_fv, self.discount_rate)
        else:
            raise ValueError("'discounting' must be either 'fv' or 'pv'")

    def plot_forecast_monte_carlo(
        self,
        backtest: bool = True,
        figsize: tuple | None = None,
    ) -> Axes:
        """
        Plot Monte Carlo simulation for portfolio future wealth indexes optionally together with historical wealth index.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        time period considering cash flows (portfolio withdrawals/contributions).

        Random wealth indexes are generated according to a given distribution.

        Parameters
        ----------
        backtest : bool, default True
            Include historical wealth index if True.

        figsize : tuple of (float, float), default None
            Figure size (width, height) in inches. If None, matplotlib defaults are used.

        Returns
        -------
        Axes
            Matplotlib axes object.

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> pf = ok.Portfolio(
        ...     assets=["SPY.US", "AGG.US", "GLD.US"],
        ...     weights=[0.60, 0.35, 0.05],
        ...     rebalancing_strategy=ok.Rebalance(period="year"),
        ... )
        >>> # Set Monte Carlo parameters
        >>> pf.dcf.set_mc_parameters(distribution="norm", period=50, mc_number=200)
        >>> # set cash flow parameters
        >>> ind = ok.IndexationStrategy(pf)  # create cash flow strategy linked to the portfolio
        >>> ind.initial_investment = 10_000  # add initial investment to cash flow strategy
        >>> ind.amount = -500  # set withdrawal size
        >>> ind.frequency = "year"  # set withdrawal frequency
        >>> pf.dcf.cashflow_parameters = ind  # assign cash flow strategy to portfolio
        >>> pf.dcf.plot_forecast_monte_carlo(backtest=True)
        >>> plt.yscale("log")  # Y-axis has logarithmic scale
        >>> plt.show()
        """
        if backtest:
            ax = self._plot_mc_with_backtest(figsize)
        else:
            s2 = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
            ax = s2.plot(legend=None, figsize=figsize)
        self.cashflow_parameters._clear_cf_cache()
        return ax

    def _plot_mc_with_backtest(self, figsize: tuple | None = None) -> Axes:
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        original_cashflow = self.cashflow_parameters
        s1 = self.wealth_index(discounting="fv", include_negative_values=False)[self.parent.symbol]
        ax = s1.plot(legend=None, figsize=figsize)
        last_backtest_value = s1.iloc[-1]
        if last_backtest_value > 0:
            temp_cashflow = copy.copy(original_cashflow)
            try:
                self.cashflow_parameters = temp_cashflow
                self.cashflow_parameters.initial_investment = last_backtest_value
                if self.cashflow_parameters.NAME == "fixed_amount":
                    # calculate the cash flow size after backtesting period
                    months = helpers.Date.get_difference_in_months(self.parent.last_date, self.parent.first_date).n
                    years = months / settings._MONTHS_PER_YEAR
                    if self.cashflow_parameters.frequency != "none":
                        periods = years / settings.frequency_periods_per_year[self.cashflow_parameters.frequency]
                        self.cashflow_parameters.amount *= (1.0 + self.cashflow_parameters.indexation) ** periods
                s2 = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
                for s in s2:
                    s2[s].plot(legend=None, ax=ax)
            finally:
                self.cashflow_parameters = original_cashflow
        return ax

    def monte_carlo_survival_period(self, threshold: float = 0) -> pd.Series:
        """
        Generate a survival period distribution for a portfolio with cash flows by Monte Carlo simulation.

        Analyzing the result, finding "min", "max" and percentiles it's possible to see for how long
        will last the investment strategy - possible longevity period.

        Parameters
        ----------
        threshold : float, default 0
            The percentage of the initial investments when the portfolio balance considered voided.
            This parameter is important to use in cash flow strategies with a fixed
            withdrawal percentage (PercentageStrategy).

        Returns
        -------
        Series
            Survival period distribution for a portfolio with cash flows.

        Examples
        --------
        >>> pf = ok.Portfolio(["SPY.US", "AGG.US", "GLD.US"], weights=[0.60, 0.35, 0.05])
        >>> # set Monte Carlos parameters
        >>> pf.dcf.set_mc_parameters(
        ...     distribution="t",  # use Student's distribution (t-distribution)
        ...     period=50,  # make forecast for 50 years
        ...     mc_number=200,  # create 200 randow wealth indexes
        ... )
        >>> # Set Cash Flow parameters
        >>> pc = ok.PercentageStrategy(pf)  # create PercentageStrategy linked to the portfolio
        >>> pc.initial_investment = 10_000  # add initial investments size
        >>> pc.frequency = "year"  # set cash flow frequency
        >>> pc.percentage = -0.20  # set withdrawal percentage
        >>> # Assign the strategy to Portfolio
        >>> pf.dcf.cashflow_parameters = pc
        >>> s = pf.dcf.monte_carlo_survival_period(threshold=0.10)  # the balance is considered voided at 10%
        >>> s.min()
        np.float64(10.5)
        >>> s.max()
        np.float64(33.5)
        >>> s.mean()
        np.float64(17.9055)
        >>> s.quantile(50 / 100)
        np.float64(17.5)
        """
        s2 = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
        dates: pd.Series = helpers.Frame.get_survival_date(s2, self.discount_rate, threshold)
        return dates.apply(helpers.Date.get_period_length, args=(self.parent.last_date,))

    def monte_carlo_irr(self) -> pd.Series:
        """
        Distribution of money-weighted IRRs across Monte Carlo forecast paths.

        For each simulated path the nominal internal rate of return (MWRR) is computed
        over the forecast horizon, honoring the configured cash flow strategy. It is the
        forward-looking counterpart of :meth:`irr`.

        All paths share a single Monte Carlo return draw (see `MonteCarlo.seed` for
        reproducibility). The per-path investor cash-flow vector on a monthly grid
        ``t = 0 .. M`` (``M = mc.period * 12``) is:

        - ``t = 0`` (Monte Carlo start): ``-initial_investment``
        - ``t = 1 .. M``: ``-monte_carlo_cash_flow`` (withdrawals become inflows)
        - ``t = M``: additionally ``+`` the terminal wealth value (floored at 0)

        With no intermediate cash flows each path's IRR equals that path's CAGR.

        Returns
        -------
        Series
            One annualized effective IRR per Monte Carlo path (length ``mc_number``).
            NaN for paths whose cash flow has no sign change.

        Examples
        --------
        >>> pf = ok.Portfolio(["SPY.US", "AGG.US"], ccy="USD")
        >>> pf.dcf.set_mc_parameters(distribution="norm", period=20, mc_number=100, seed=0)
        >>> ind = ok.IndexationStrategy(pf)
        >>> ind.initial_investment = 10_000
        >>> ind.frequency = "year"
        >>> ind.amount = -500
        >>> pf.dcf.cashflow_parameters = ind
        >>> pf.dcf.monte_carlo_irr().quantile([0.1, 0.5, 0.9])
        """
        if self.cashflow_parameters is None:
            raise AttributeError("'cashflow_parameters' is not defined.")
        cashflow_parameters = self.cashflow_parameters
        wealth = self.monte_carlo_wealth(discounting="fv", include_negative_values=False)
        cash_flow = self.monte_carlo_cash_flow(discounting="fv", remove_if_wealth_index_negative=False)
        # Zero a path's cash flow once its (floored) wealth is depleted, consistent per path.
        cash_flow = cash_flow.where(wealth.reindex(cash_flow.index) != 0, 0.0)
        terminal = wealth.iloc[-1]
        n_months, n_paths = cash_flow.shape
        flows = np.empty((n_months + 1, n_paths), dtype=float)
        flows[0, :] = -cashflow_parameters.initial_investment
        flows[1:, :] = -cash_flow.to_numpy()
        flows[-1, :] += terminal.reindex(cash_flow.columns).to_numpy()
        irr = dcf_calculations.irr_of_cashflow_matrix(flows, periods_per_year=settings._MONTHS_PER_YEAR)
        return pd.Series(irr, index=cash_flow.columns, name="monte_carlo_irr")
