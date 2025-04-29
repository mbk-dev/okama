from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from okama import settings
from okama.common.validators import validate_real


class Rebalance:
    """
    Rebalancing strategies for portfolios.
    """

    def __init__(
        self, period: str = "year", abs_deviation: Optional[float] = None, rel_deviation: Optional[float] = None
    ):
        self.period = period
        self.abs_deviation = abs_deviation
        self.rel_deviation = rel_deviation
        self.pandas_frequency = settings.frequency_mapping.get(self.period)
        self.validate_condition()

    def validate_condition(self):
        if self.period not in settings.frequency_mapping.keys():
            raise ValueError(f"rebalancing_period must be in {settings.frequency_mapping.keys()}")
        if self.period != "none" and (self.abs_deviation or self.rel_deviation):
            raise ValueError(f"Rebalancing cannot be both calendar and conditional.")
        if self.abs_deviation:
            validate_real(arg_name="abs_deviation", arg_value=self.abs_deviation)
            if self.abs_deviation <= 0:
                raise ValueError("Absolute deviation must be positive.")
            if self.abs_deviation > 1:
                raise ValueError("Absolute deviation must be less or equal to 1.")
        if self.rel_deviation:
            validate_real(arg_name="abs_deviation", arg_value=self.abs_deviation)
            if self.rel_deviation <= 0:
                raise ValueError("Relative deviation must be positive.")
            if self.rel_deviation > 1:
                raise ValueError("Relative deviation must be less or equal to 1.")


    def wealth_ts(self, target_weights: list, ror: pd.DataFrame, calculate_assets_wealth_indexes:bool = False) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate wealth index time series of rebalanced portfolio given returns time series of the assets.

        Optionally calculate also ASSETS wealth indexes time series inside rebalanced portfolio.
        """
        # Frame.weights_sum_is_one(weights)
        initial_inv = 1000
        portfolio_wealth_index = pd.Series(dtype="float64")
        assets_wealth_indexes = pd.DataFrame(columns=ror.columns, dtype="float64")
        target_weights_np = np.asarray(target_weights)
        if self.period == "none" and self.abs_deviation is None and self.rel_deviation is None:  # No rebalancing
            initial_allocation = target_weights_np * initial_inv
            assets_wealth_indexes_local = initial_allocation * (1 + ror).cumprod()
            portfolio_wealth_index = assets_wealth_indexes_local.sum(axis=1)
            if calculate_assets_wealth_indexes:
                assets_wealth_indexes = assets_wealth_indexes_local
        elif self.period == "none":  # No calendar rebalancing
            portfolio_wealth_index, assets_wealth_indexes = self.rebalance_by_condition(
                ror,
                target_weights,
                initial_inv,
                calculate_assets_wealth_indexes
            )
        elif self.abs_deviation is None and self.rel_deviation is None:  # Calendar rebalancing
            for x in ror.resample(rule=self.pandas_frequency, convention="start"):
                df = x[1]  # select ror part of the grouped data
                initial_allocation = target_weights_np * initial_inv  # rebalancing
                assets_wealth_indexes_local = initial_allocation * (1 + df).cumprod()
                if calculate_assets_wealth_indexes:
                    assets_wealth_indexes = pd.concat(
                        [assets_wealth_indexes_local, assets_wealth_indexes],
                        verify_integrity=True,
                        sort=False,
                    )
                wealth_index_local = assets_wealth_indexes_local.sum(axis=1)
                portfolio_wealth_index = pd.concat([None if portfolio_wealth_index.empty else portfolio_wealth_index, wealth_index_local], sort=False)
                initial_inv = portfolio_wealth_index.iloc[-1]
        return portfolio_wealth_index, assets_wealth_indexes

    def rebalance_by_condition(self, ror, target_weights, initial_inv, calculate_assets_wealth_indexes: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
        target_weights_np = np.asarray(target_weights)
        portfolio_wealth_index = pd.Series(dtype="float64")
        weights_ts = pd.DataFrame(columns=ror.columns)
        assets_wealth_indexes_local = pd.DataFrame(columns=ror.columns, dtype="float64")
        assets_wealth_indexes = pd.DataFrame(columns=ror.columns, dtype="float64")
        for n, row in enumerate(ror.itertuples()):
            date = row[0]
            r = pd.Series(row[1:], index=ror.columns, name=date)
            if n == 0:
                initial_allocation = target_weights_np * initial_inv  # initial rebalancing
                assets_wealth_indexes_local = initial_allocation * (1 + r)
            else:
                if rebalancing_condition:
                    assets_wealth_indexes_local = target_weights_np * assets_wealth_indexes_local.sum()
                assets_wealth_indexes_local *= 1 + r
                assets_wealth_indexes_local.rename(date, inplace=True)
            if calculate_assets_wealth_indexes:
                row = pd.DataFrame(assets_wealth_indexes_local).T
                assets_wealth_indexes = pd.concat([assets_wealth_indexes, row])
            portfolio_wealth_index_value = assets_wealth_indexes_local.sum()
            portfolio_wealth_index[date] = portfolio_wealth_index_value
            # Check if rebalancing required
            weights = assets_wealth_indexes_local.divide(portfolio_wealth_index_value, axis=0)
            weights_s = weights.to_frame().T
            weights_ts = pd.concat([weights_ts, weights_s])
            weights_ts.columns = ror.columns
            target_weights_s = pd.Series(target_weights, index=ror.columns, name=date)
            weights_difference_abs = weights - target_weights_s
            weights_difference_abs = weights_difference_abs.abs()
            weights_difference_rel = weights.divide(target_weights_s, axis=0) - 1
            weights_difference_rel = weights_difference_rel.abs()
            condition_abs = False if self.abs_deviation is None else (weights_difference_abs > self.abs_deviation).any()
            condition_rel = False if self.rel_deviation is None else (weights_difference_rel > self.rel_deviation).any()
            rebalancing_condition = condition_abs or condition_rel  # Determined at the end, as it is not needed during the first run.
        return portfolio_wealth_index, assets_wealth_indexes

    def assets_weights_ts(self, target_weights: list, ror: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate assets weights monthly time series for rebalanced portfolio.
        """
        portfolio_wealth_index, assets_wealth_indexes = self.wealth_ts(target_weights=target_weights,
                                                                       ror=ror,
                                                                       calculate_assets_wealth_indexes=True)
        return assets_wealth_indexes.divide(portfolio_wealth_index, axis=0)

    def return_ror_ts(self, target_weights: Union[list, np.ndarray], ror: pd.DataFrame) -> pd.Series:
        """
        Return monthly rate of return time series of rebalanced portfolio given returns time series of the assets.
        Default rebalancing period is a Year (end of year)
        For not rebalanced portfolio set Period to 'none'.
        """
        # define data of the first period
        first_date = ror.index[0]
        return_first_period = ror.iloc[0] @ target_weights

        wealth_index = self.wealth_ts(target_weights, ror)[0]
        ror = wealth_index.pct_change()
        ror.loc[first_date] = return_first_period  # replaces NaN with the first period return
        return ror
