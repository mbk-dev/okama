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

    Attributes
    ----------
    period : {'none', 'month', 'quarter', 'half-year', 'year'}
        rebalancing period (rebalancing frequency) is predetermined time intervals when
        the investor rebalances the portfolio.
        If 'none' assets weights are not rebalanced.

    abs_deviation : float, optional
        The absolute deviation allowed for the assets weights in the portfolio.
        It is defined for an asset weight as an absolut value of: actual_weight - target_weight.
        It must be more than 0 (0%).
        Max value is 1 (100%).
        if None the parameter is ignored.

    rel_deviation : float, optional
        The relative deviation allowed for the assets weights in the portfolio.
        It is defined for an asset weight as an absolut value of: actual_weight / target_weight - 1.
        it must be positive.
        if None the parameter is ignored.
    """

    def __init__(
        self, period: str = "year", abs_deviation: Optional[float] = None, rel_deviation: Optional[float] = None
    ):
        self.abs_deviation = abs_deviation
        self.rel_deviation = rel_deviation
        self.period = period

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

    @property
    def period(self) -> str:
        """
        The rebalancing period for the investment portfolio.

        It can be: 'none', 'month', 'quarter', 'half-year', 'year'.
        """
        return self._period

    @period.setter
    def period(self, value: str):
        self._period = value
        self._validate_condition()
        self._pandas_frequency = settings.frequency_mapping[self.period]
        self._pandas_frequency_grouper = settings.grouper_frequency_mapping[self.period]

    def wealth_ts(
        self, target_weights: list, ror: pd.DataFrame, calculate_assets_wealth_indexes: bool = False
    ) -> Result:
        """
        Calculate wealth index time series of rebalanced portfolio given returns time series of the assets.

        Optionally calculate also ASSETS wealth indexes time series inside rebalanced portfolio.

        Parameters
        ----------
        target_weights : list of float
            The target weights for assets in the portfolio.

        ror : pd.DataFrame
            Assets rate of return monthly time series.

        calculate_assets_wealth_indexes : bool, optional
            Whether or not to calculate assets wealth indexes after rebalancing.
            When 'True' works slower.

        Returns
        -------
        Result
            Result of rebalancing investment portfolio.
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
                ror, target_weights, initial_inv, calculate_assets_wealth_indexes
            )
        else:  # Calendar rebalancing
            rebalancing_by_condition_needed = self.abs_deviation or self.rel_deviation
            rebalancing_condition = False
            # accumulate chunks and concatenate once after the loop
            pw_chunks: list[pd.Series] = []
            awi_chunks: list[pd.DataFrame] = [] if calculate_assets_wealth_indexes else None
            for n, x in enumerate(ror.resample(rule=self._pandas_frequency, convention="start")):
                df = x[1]  # select ror part of the grouped data
                if n == 0:
                    initial_allocation = target_weights_np * initial_inv
                else:
                    if (rebalancing_by_condition_needed and rebalancing_condition) or (
                        not rebalancing_by_condition_needed
                    ):
                        # rebalancing
                        initial_allocation = target_weights_np * end_period_balance
                        date = x[0]
                        if rebalancing_by_condition_needed:
                            events_ts[date.asfreq("M", how="start") - 1] = "abs" if condition_abs else "rel"
                        else:
                            events_ts[date.asfreq("M", how="start") - 1] = "calendar"
                    elif rebalancing_by_condition_needed and not rebalancing_condition:
                        # skip rebalancing
                        initial_allocation = end_period_weights * end_period_balance
                assets_wealth_indexes_local = initial_allocation * (1 + df).cumprod()
                if calculate_assets_wealth_indexes:
                    # collect asset wealth index chunks for later concatenation
                    awi_chunks.append(assets_wealth_indexes_local.copy())
                wealth_index_local = assets_wealth_indexes_local.sum(axis=1)
                # collect portfolio wealth index chunks for later concatenation
                pw_chunks.append(wealth_index_local.copy())
                # use local last value instead of relying on concatenated series
                end_period_balance = wealth_index_local.iloc[-1]
                if rebalancing_by_condition_needed:
                    end_period_weights = assets_wealth_indexes_local.iloc[-1].divide(
                        wealth_index_local.iloc[-1], axis=0
                    )
                    rebalancing_condition, condition_abs = self._check_if_rebalancing_required(
                        assets_wealth_indexes_local, wealth_index_local, target_weights
                    )
            # perform single concatenation outside the loop
            portfolio_wealth_index = pd.concat(pw_chunks, sort=False) if pw_chunks else portfolio_wealth_index
            if calculate_assets_wealth_indexes and awi_chunks:
                assets_wealth_indexes = pd.concat(awi_chunks, verify_integrity=True, sort=False)
        # set value for the first date
        portfolio_wealth_index.loc[first_wealth_index_date] = initial_inv
        portfolio_wealth_index = portfolio_wealth_index.sort_index(ascending=True)
        if calculate_assets_wealth_indexes:
            assets_wealth_indexes.loc[first_wealth_index_date] = target_weights_np * initial_inv
            assets_wealth_indexes = assets_wealth_indexes.sort_index(ascending=True)
        return Result(
            portfolio_wealth_index=portfolio_wealth_index, assets_wealth_indexes=assets_wealth_indexes, events=events_ts
        )

    def _rebalance_by_condition(
        self, ror, target_weights, initial_inv, calculate_assets_wealth_indexes: bool = False
    ) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
        target_weights_np = np.asarray(target_weights)
        portfolio_wealth_index = pd.Series(dtype="float64")
        assets_wealth_indexes_local = pd.DataFrame(columns=ror.columns, dtype="float64")
        assets_wealth_indexes = pd.DataFrame(columns=ror.columns, dtype="float64")
        events_ts = pd.Series(dtype="float64")
        awi_rows = [] if calculate_assets_wealth_indexes else None
        for n, row in enumerate(ror.itertuples()):
            date = row[0]
            r = pd.Series(row[1:], index=ror.columns, name=date)
            if n == 0:
                initial_allocation = target_weights_np * initial_inv  # initial rebalancing
                assets_wealth_indexes_local = initial_allocation * (1 + r)
            else:
                if rebalancing_condition:
                    assets_wealth_indexes_local = target_weights_np * assets_wealth_indexes_local.sum()
                    events_ts[date - 1] = "abs" if condition_abs else "rel"  # set previous month as its EOD data
                assets_wealth_indexes_local *= 1 + r
                assets_wealth_indexes_local = assets_wealth_indexes_local.rename(date)
            if calculate_assets_wealth_indexes:
                # collect rows for a single concat after the loop
                awi_rows.append(pd.DataFrame(assets_wealth_indexes_local).T)
            portfolio_wealth_index_local = assets_wealth_indexes_local.sum()
            portfolio_wealth_index[date] = portfolio_wealth_index_local
            # Check if rebalancing required
            rebalancing_condition, condition_abs = self._check_if_rebalancing_required(
                assets_wealth_indexes_local, portfolio_wealth_index_local, target_weights
            )
        # perform single concatenation outside the loop
        if calculate_assets_wealth_indexes and awi_rows:
            assets_wealth_indexes = pd.concat(awi_rows, sort=False)
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
        # Ensure Python bools for downstream identity checks in tests
        condition_abs_np = (
            (weights_difference_abs > self.abs_deviation).any() if self.abs_deviation is not None else False
        )
        condition_rel_np = (
            (weights_difference_rel > self.rel_deviation).any() if self.rel_deviation is not None else False
        )
        condition_abs = bool(condition_abs_np)
        condition_rel = bool(condition_rel_np)
        # Determined at the end, as it is not needed during the first run.
        rebalancing_condition = bool(condition_abs or condition_rel)
        return rebalancing_condition, condition_abs

    def assets_weights_ts(self, target_weights: list, ror: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate assets weights monthly time series for rebalanced portfolio.

        Parameters
        ----------
        target_weights : list of float
            The target weights for assets in the portfolio.

        ror : pd.DataFrame
            Assets rate of return monthly time series.

        Returns
        -------
        pd.DataFrame
            Assets weights monthly time series.
        """
        reb = self.wealth_ts(target_weights=target_weights, ror=ror, calculate_assets_wealth_indexes=True)
        return reb.assets_wealth_indexes.divide(reb.portfolio_wealth_index, axis=0)

    def return_ror_ts(self, target_weights: Union[list, np.ndarray], ror: pd.DataFrame) -> pd.Series:
        """
        Return monthly rate of return time series of rebalanced portfolio given returns time series of the assets.

        Parameters
        ----------
        target_weights : list of float
            The target weights for assets in the portfolio.

        ror : pd.DataFrame
            Assets rate of return monthly time series.

        Returns
        -------
        pd.Series
            The monthly rate of return time series of rebalanced portfolio.
        """
        # Calculate wealth index using wealth_ts (which includes an initial pre-first-period point)
        wealth_index = self.wealth_ts(target_weights, ror).portfolio_wealth_index
        # Align behavior with manual baseline: start pct_change from the first input ror date,
        # so the first period in ror.index will be NaN and then dropped.
        first_ror_date = ror.index[0]
        wealth_index_aligned = wealth_index.loc[wealth_index.index >= first_ror_date]
        portfolio_ror = wealth_index_aligned.pct_change()
        portfolio_ror = portfolio_ror.dropna()
        return portfolio_ror

    def wealth_ts_ef(self, weights: list, ror: pd.DataFrame) -> pd.Series:
        """
        Calculate wealth index time series of rebalanced portfolio given returns time series of the assets.

        Simple version without conditional rebalancing (abs_deviation and rel_deviation are not used).
        This method is used for efficient frontier optimization.
        Default rebalancing period is a Year (end of year).
        For not rebalanced portfolio set period to 'none'.

        Parameters
        ----------
        weights : list of float
            The target weights for assets in the portfolio.

        ror : pd.DataFrame
            Assets rate of return monthly time series.

        Returns
        -------
        pd.Series
            Wealth index time series.
        """
        initial_inv = 1000
        weights_array = np.asarray(weights)

        if self.period == "none":  # Not rebalanced portfolio
            inv_period_spread = weights_array * initial_inv
            assets_wealth_indexes = inv_period_spread * (1 + ror).cumprod()
            wealth_index = assets_wealth_indexes.sum(axis=1)
        else:
            # Fully vectorized approach without loops
            period_grouper = pd.Grouper(freq=self._pandas_frequency_grouper, convention="start")
            period_ids = ror.groupby(period_grouper).ngroup()

            growth_factors = 1 + ror
            cumulative_growth_within_period = growth_factors.groupby(period_ids).cumprod()

            wealth_per_asset_normalized = weights_array * cumulative_growth_within_period
            portfolio_wealth_normalized = wealth_per_asset_normalized.sum(axis=1)

            # Calculate ending wealth for each period (growth factor for that period)
            period_ends = portfolio_wealth_normalized.groupby(period_ids).last()

            # Calculate cumulative multiplier for the START of each period
            period_multipliers = period_ends.shift(1).fillna(1.0).cumprod()

            # Map multipliers back to the original time series
            aligned_multipliers = period_ids.map(period_multipliers)

            # Calculate final wealth index
            wealth_index = initial_inv * aligned_multipliers * portfolio_wealth_normalized

        return wealth_index

    def return_ror_ts_ef(self, weights: Union[list, np.ndarray], ror: pd.DataFrame) -> pd.Series:
        """
        Return monthly rate of return time series of rebalanced portfolio given returns time series of the assets.

        Simple version without conditional rebalancing (abs_deviation and rel_deviation are not used).
        This method is used for efficient frontier optimization.
        Default rebalancing period is a Year (end of year).
        For not rebalanced portfolio set period to 'none'.

        Parameters
        ----------
        weights : list of float or np.ndarray
            The target weights for assets in the portfolio.

        ror : pd.DataFrame
            Assets rate of return monthly time series.

        Returns
        -------
        pd.Series
            The monthly rate of return time series of rebalanced portfolio.
        """
        # define data of the first period
        first_date = ror.index[0]
        return_first_period = ror.iloc[0] @ weights

        wealth_index = self.wealth_ts_ef(weights, ror)
        portfolio_ror = wealth_index.pct_change()
        portfolio_ror.loc[first_date] = return_first_period  # replaces NaN with the first period return
        return portfolio_ror
