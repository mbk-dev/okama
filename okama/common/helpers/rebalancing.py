from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

from okama import settings
from okama.common.validators import validate_real


@dataclass(frozen=True)
class Result:
    """
    The result of rebalancing portfolio.

    Attributes
    ----------
    portfolio_wealth_index : Series
        Monthly time series for rebalanced portfolio wealth index.

    assets_wealth_indexes : DataFrame
        The absolute amount of withdrawal size (the best solution found).

    events : Series
        The relative amount of withdrawal size (the best solution found).
    """
    portfolio_wealth_index: pd.Series
    assets_wealth_indexes: pd.DataFrame
    events: pd.Series


class Rebalance:
    """
    Rebalancing strategy for an investment portfolio.

    Rebalancing is the process by which an investor restores their portfolio to its target allocation
    by selling and buying assets. After rebalancing all the assets have original (target) weights.
    """

    def __init__(
        self, period: str = "year", abs_deviation: Optional[float] = None, rel_deviation: Optional[float] = None
    ):
        self.period = period
        self.abs_deviation = abs_deviation
        self.rel_deviation = rel_deviation
        self._pandas_frequency = settings.frequency_mapping.get(self.period)
        self._validate_condition()

    def __str__(self):
        dic = {
            "period": self.period,
            "abs_deviation": self.abs_deviation,
            "rel_deviation": self.rel_deviation,
        }
        return repr(pd.Series(dic))

    def __repr__(self):
        dic = {
            "period": self.period,
            "abs_deviation": self.abs_deviation,
            "rel_deviation": self.rel_deviation,
        }
        return repr(pd.Series(dic))

    def _validate_condition(self):
        if self.period not in settings.frequency_mapping.keys():
            raise ValueError(f"rebalancing_period must be in {list(settings.frequency_mapping.keys())}")
        if self.abs_deviation:
            validate_real(arg_name="abs_deviation", arg_value=self.abs_deviation)
            if self.abs_deviation <= 0:
                raise ValueError("Absolute deviation must be positive.")
            if self.abs_deviation > 1:
                raise ValueError("Absolute deviation must be less or equal to 1.")
        if self.rel_deviation:
            validate_real(arg_name="rel_deviation", arg_value=self.rel_deviation)
            if self.rel_deviation <= 0:
                raise ValueError("Relative deviation must be positive.")


    def wealth_ts(self, target_weights: list, ror: pd.DataFrame, calculate_assets_wealth_indexes:bool = False) -> Result:
        """
        Calculate wealth index time series of rebalanced portfolio given returns time series of the assets.

        Optionally calculate also ASSETS wealth indexes time series inside rebalanced portfolio.
        """
        if isinstance(ror, pd.Series):
            ror = ror.to_frame()
        if len(target_weights) != len(ror.columns):
            raise ValueError("The dimension of target_weights and the number of columns in ror must be equal")
        initial_inv = 1000
        first_date = ror.index[0]
        first_wealth_index_date = first_date - 1
        portfolio_wealth_index = pd.Series(dtype="float64")
        assets_wealth_indexes = pd.DataFrame(columns=ror.columns, dtype="float64")
        events_ts = pd.Series(dtype="float64")
        target_weights_np = np.asarray(target_weights)
        if self.period == "none" and self.abs_deviation is None and self.rel_deviation is None:  # No rebalancing
            initial_allocation = target_weights_np * initial_inv
            assets_wealth_indexes_local = initial_allocation * (1 + ror).cumprod()
            portfolio_wealth_index = assets_wealth_indexes_local.sum(axis=1)
            if calculate_assets_wealth_indexes:
                assets_wealth_indexes = assets_wealth_indexes_local
        elif self.period == "none":  # No calendar rebalancing
            portfolio_wealth_index, assets_wealth_indexes, events_ts = self._rebalance_by_condition(
                ror,
                target_weights,
                initial_inv,
                calculate_assets_wealth_indexes
            )
        else:  # Calendar rebalancing
            rebalancing_by_condition_needed = self.abs_deviation or self.rel_deviation
            rebalancing_condition = False
            for n, x in enumerate(ror.resample(rule=self._pandas_frequency, convention="start")):
                df = x[1]  # select ror part of the grouped data
                if n == 0:
                    initial_allocation = target_weights_np * initial_inv
                else:
                    if (rebalancing_by_condition_needed and rebalancing_condition) or (not rebalancing_by_condition_needed):
                        # rebalancing
                        initial_allocation = target_weights_np * end_period_balance
                        date = x[0]
                        if rebalancing_by_condition_needed:
                            events_ts[date.asfreq('M', how='start') - 1] = 'abs' if condition_abs else 'rel'
                        else:
                            events_ts[date.asfreq('M', how='start') - 1] = 'calendar'
                    elif rebalancing_by_condition_needed and not rebalancing_condition:
                        # skip rebalancing
                        initial_allocation = end_period_weights * end_period_balance
                assets_wealth_indexes_local = initial_allocation * (1 + df).cumprod()
                if calculate_assets_wealth_indexes:
                    assets_wealth_indexes = pd.concat(
                        [assets_wealth_indexes_local, assets_wealth_indexes],
                        verify_integrity=True,
                        sort=False,
                    )
                wealth_index_local = assets_wealth_indexes_local.sum(axis=1)
                portfolio_wealth_index = pd.concat([None if portfolio_wealth_index.empty else portfolio_wealth_index, wealth_index_local], sort=False)
                end_period_balance = portfolio_wealth_index.iloc[-1]
                if rebalancing_by_condition_needed:
                    end_period_weights = assets_wealth_indexes_local.iloc[-1].divide(wealth_index_local.iloc[-1], axis=0)
                    rebalancing_condition, condition_abs = self._check_if_rebalancing_required(assets_wealth_indexes_local,
                                                                                               wealth_index_local,
                                                                                               target_weights)
        # set value for the first date
        portfolio_wealth_index.loc[first_wealth_index_date] = initial_inv
        portfolio_wealth_index.sort_index(ascending=True, inplace=True)
        if calculate_assets_wealth_indexes:
            assets_wealth_indexes.loc[first_wealth_index_date] = target_weights_np * initial_inv
            assets_wealth_indexes.sort_index(ascending=True, inplace=True)
        return Result(
            portfolio_wealth_index=portfolio_wealth_index,
            assets_wealth_indexes=assets_wealth_indexes,
            events=events_ts
        )

    def _rebalance_by_condition(
            self,
            ror,
            target_weights,
            initial_inv,
            calculate_assets_wealth_indexes: bool = False
    ) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
        target_weights_np = np.asarray(target_weights)
        portfolio_wealth_index = pd.Series(dtype="float64")
        assets_wealth_indexes_local = pd.DataFrame(columns=ror.columns, dtype="float64")
        assets_wealth_indexes = pd.DataFrame(columns=ror.columns, dtype="float64")
        events_ts = pd.Series(dtype="float64")
        for n, row in enumerate(ror.itertuples()):
            date = row[0]
            r = pd.Series(row[1:], index=ror.columns, name=date)
            if n == 0:
                initial_allocation = target_weights_np * initial_inv  # initial rebalancing
                assets_wealth_indexes_local = initial_allocation * (1 + r)
            else:
                if rebalancing_condition:
                    assets_wealth_indexes_local = target_weights_np * assets_wealth_indexes_local.sum()
                    events_ts[date - 1] = 'abs' if condition_abs else 'rel'  # set previous month as its EOD data
                assets_wealth_indexes_local *= 1 + r
                assets_wealth_indexes_local.rename(date, inplace=True)
            if calculate_assets_wealth_indexes:
                row = pd.DataFrame(assets_wealth_indexes_local).T
                assets_wealth_indexes = pd.concat([assets_wealth_indexes, row])
            portfolio_wealth_index_local = assets_wealth_indexes_local.sum()
            portfolio_wealth_index[date] = portfolio_wealth_index_local
            # Check if rebalancing required
            rebalancing_condition, condition_abs = self._check_if_rebalancing_required(assets_wealth_indexes_local,
                                                                                       portfolio_wealth_index_local,
                                                                                       target_weights)
        return portfolio_wealth_index, assets_wealth_indexes, events_ts

    def _check_if_rebalancing_required(
            self,
            assets_wealth_indexes_local,
            portfolio_wealth_index_local,
            target_weights,
    ) -> Tuple[bool, bool]:
        try:
            # DataFrame
            weights = assets_wealth_indexes_local.iloc[-1].divide(portfolio_wealth_index_local.iloc[-1], axis=0)
            target_weights_s = pd.Series(target_weights, index=assets_wealth_indexes_local.columns)
        except AttributeError:
            # Series
            weights = assets_wealth_indexes_local.divide(portfolio_wealth_index_local, axis=0)
            target_weights_s = pd.Series(target_weights, index=assets_wealth_indexes_local.index)
        weights_difference_abs = weights - target_weights_s
        weights_difference_abs = weights_difference_abs.abs()
        weights_difference_rel = weights.divide(target_weights_s, axis=0) - 1
        weights_difference_rel = weights_difference_rel.abs()
        condition_abs = False if self.abs_deviation is None else (weights_difference_abs > self.abs_deviation).any()
        condition_rel = False if self.rel_deviation is None else (weights_difference_rel > self.rel_deviation).any()
        rebalancing_condition = condition_abs or condition_rel  # Determined at the end, as it is not needed during the first run.
        return rebalancing_condition, condition_abs

    def assets_weights_ts(self, target_weights: list, ror: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate assets weights monthly time series for rebalanced portfolio.
        """
        reb = self.wealth_ts(target_weights=target_weights,
                             ror=ror,
                             calculate_assets_wealth_indexes=True)
        return reb.assets_wealth_indexes.divide(reb.portfolio_wealth_index, axis=0)

    def return_ror_ts(self, target_weights: Union[list, np.ndarray], ror: pd.DataFrame) -> pd.Series:
        """
        Return monthly rate of return time series of rebalanced portfolio given returns time series of the assets.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'none'.
        """
        # define data of the first period
        first_date = ror.index[0]
        return_first_period = ror.iloc[0] @ target_weights

        wealth_index = self.wealth_ts(target_weights, ror).portfolio_wealth_index
        ror = wealth_index.pct_change()
        ror.loc[first_date] = return_first_period  # replaces NaN with the first period return
        return ror

