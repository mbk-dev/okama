from __future__ import annotations  # noqa: I001

from typing import Union, Optional, Literal

import pandas as pd
import numpy as np
from scipy import optimize

import okama.common.helpers.helpers as helpers
import okama.portfolios.cashflow_strategies as cf
from okama import settings


def get_wealth_indexes_fv_with_cashflow(  # noqa: C901
    ror: Union[pd.Series, pd.DataFrame],  # noqa: UP007
    portfolio_symbol: Optional[str],  # noqa: UP045
    inflation_symbol: Optional[str],  # noqa: UP045
    cashflow_parameters: cf.CashFlow,
    task: Literal["backtest", "monte_carlo"],
) -> Union[pd.Series, pd.DataFrame]:  # noqa: UP007
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
    ror: Union[pd.Series, pd.DataFrame],  # noqa: UP007
    portfolio_symbol: Optional[str],  # noqa: UP045
    cashflow_parameters: cf.CashFlow,
    task: Literal["backtest", "monte_carlo"],
) -> Union[pd.Series, pd.DataFrame]:  # noqa: UP007
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


def get_wealth_indexes_fv_with_cashflow_mc(  # noqa: C901
    ror: pd.DataFrame,
    cashflow_parameters: cf.CashFlow,
    discount_rate: float,
) -> pd.DataFrame:
    """
    Vectorized Monte Carlo counterpart of `get_wealth_indexes_fv_with_cashflow`.

    Computes wealth index future values (FV) for all random return paths at
    once: a single Python loop over time steps with numpy operations across
    paths, instead of a per-path pandas apply. Implements the "monte_carlo"
    task semantics only: extra cash flows from `time_series` are compounded
    with the discount rate unless `time_series_discounted_values` is True.

    Replicates the per-path reference exactly; equivalence is pinned by tests.
    (The two historical reference quirks were fixed together with this engine —
    see GitHub issues #81 and #82.)
    """
    initial_investment = cashflow_parameters.initial_investment
    n_rows, n_cols = ror.shape
    returns = ror.to_numpy(dtype=float)
    frequency = cashflow_parameters.frequency
    periods_per_year = settings.frequency_periods_per_year[frequency]
    amount = getattr(cashflow_parameters, "amount", None)
    indexation_per_period = 0.0
    if hasattr(cashflow_parameters, "indexation") and frequency != "none":
        indexation_per_period = (1 + cashflow_parameters.indexation) ** (1 / periods_per_year) - 1

    # Extra cash flows from `time_series`, aligned to the simulation index.
    extra_cf = np.zeros(n_rows)
    cash_flow_ts = cashflow_parameters.time_series
    if not (cash_flow_ts.empty or (cash_flow_ts == 0).all()):
        aligned = cash_flow_ts.reindex(ror.index).fillna(0).to_numpy(dtype=float)
        if not cashflow_parameters.time_series_discounted_values:
            monthly_discount_rate = (1 + discount_rate) ** (1 / settings._MONTHS_PER_YEAR) - 1
            aligned = aligned * (1.0 + monthly_discount_rate) ** np.arange(n_rows)
        extra_cf = aligned

    # Per-path drawdowns and reduction factors for the CWD strategy
    # (they depend on the return paths only, so they are precomputed once).
    if cashflow_parameters.NAME == "CWD":
        wealth_for_dd = 1000 * np.cumprod(1 + returns, axis=0)
        peaks = np.maximum.accumulate(wealth_for_dd, axis=0)
        drawdowns = (wealth_for_dd - peaks) / peaks
        cwd_factors = _cwd_reduction_factor_matrix(cashflow_parameters, drawdowns)

    wealth = np.empty((n_rows, n_cols))
    balance = np.full(n_cols, float(initial_investment))

    if frequency in ("month", "none"):
        for n in range(n_rows):
            if frequency == "none":
                cashflow: float | np.ndarray = 0.0
            elif cashflow_parameters.NAME == "fixed_amount":
                cashflow = amount * (1 + indexation_per_period) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow = cashflow_parameters.percentage / periods_per_year * balance
            elif cashflow_parameters.NAME == "time_series":
                cashflow = 0.0
            elif cashflow_parameters.NAME == "CWD":
                base_withdrawal = amount * (1 + indexation_per_period) ** n
                cashflow = base_withdrawal * np.where(drawdowns[n] < 0, cwd_factors[n], 1.0)
            else:
                raise ValueError("Wrong cashflow strategy name value.")
            balance = balance * (1 + returns[n]) + cashflow + extra_cf[n]
            wealth[n] = balance
    else:
        months_in_full_period = settings._MONTHS_PER_YEAR / cashflow_parameters.periods_per_year
        last_regular_cashflow: float | np.ndarray = 0.0
        for n, (start, stop) in enumerate(_resample_slices(ror.index, cashflow_parameters._pandas_frequency)):
            start_balance = balance
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
                cashflow_value: float | np.ndarray = amount * (1 + indexation_per_period) ** n
            elif cashflow_parameters.NAME == "fixed_percentage":
                cashflow_value = cashflow_parameters.percentage / periods_per_year * start_balance
            elif cashflow_parameters.NAME == "VDS":
                cashflow_value = _vds_withdrawal_vector(
                    cashflow_parameters, start_balance, last_regular_cashflow if n > 0 else 0.0, n
                )
            elif cashflow_parameters.NAME == "CWD":
                base_withdrawal = amount * (1 + indexation_per_period) ** n
                cashflow_value = base_withdrawal * np.where(drawdowns[stop - 1] < 0, cwd_factors[stop - 1], 1.0)
            else:
                raise ValueError("Wrong cashflow_method value.")
            cashflow_value = cashflow_value * ((stop - start) / months_in_full_period)
            last_regular_cashflow = cashflow_value
            balance = balance + cashflow_value
            wealth[stop - 1] = balance

    out_index = ror.index.insert(0, ror.index[0] - 1)
    data = np.vstack([np.full((1, n_cols), float(initial_investment)), wealth])
    return pd.DataFrame(data, index=out_index, columns=ror.columns)


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
    cash_flow_fv: Union[pd.Series, pd.DataFrame],  # noqa: UP007
    annual_effective_discount_rate: float,
    reverse: bool = False,
) -> Union[pd.Series, pd.DataFrame]:  # noqa: UP007
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
    guess: Union[np.ndarray, float, None] = None,  # noqa: UP007
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
            disc = base[None, :] ** (-t)                         # (n_periods, n_series)
            f = (cf * disc).sum(axis=0)                          # (n_series,)
            fprime = -(t * cf * disc).sum(axis=0) / base         # (n_series,)
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
