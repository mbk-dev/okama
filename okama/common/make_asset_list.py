from typing import Dict, Optional, List, Any, Type, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from okama import macro, asset, settings
from okama.common import validators
from okama.common.helpers import helpers

class ListMaker(ABC):
    """
    Abstract class to generate a list of assets with properties.

    Parameters
    ----------
    assets : list, default None
        List of assets. Could include tickers or asset like objects (Asset, Portfolio).
        If None a single asset list with a default ticker is used.

    first_date : str, default None
        First date of monthly return time series.
        If None the first date is calculated automatically as the oldest available date for the listed assets.

    last_date : str, default None
        Last date of monthly return time series.
        If None the last date is calculated automatically as the newest available date for the listed assets.

    ccy : str, default 'USD'
        Base currency for the list of assets. All risk metrics and returns are adjusted to the base currency.

    inflation: bool, default True
        Defines whether to take inflation data into account in the calculations.
        Including inflation could limit available data (last_date, first_date)
        as the inflation data is usually published with a one-month delay.
        With inflation = False some properties like real return are not available.
    """
    def __init__(
        self,
        assets: Optional[List[Union[str, Type]]] = None,
        *,
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
        ccy: str = "USD",
        inflation: bool = True,
    ):
        self._assets = assets
        self._currency = asset.Asset(symbol=f"{ccy}.FX")
        (
            self.asset_obj_dict,
            self.first_date,
            self.last_date,
            self.newest_asset,
            self.eldest_asset,
            self.names,
            self.currencies,
            self.assets_first_dates,
            self.assets_last_dates,
            self.assets_ror,
        ) = self._make_list(ls=self._list_of_asset_like_objects, first_date=first_date, last_date=last_date).values()
        if first_date:
            self.first_date = max(self.first_date, pd.to_datetime(first_date))
        self.assets_ror = self.assets_ror[self.first_date:]
        if last_date:
            self.last_date = min(self.last_date, pd.to_datetime(last_date))
        if inflation:
            self.inflation: str = f"{ccy}.INFL"
            self._inflation_instance = macro.Inflation(
                self.inflation, self.first_date, self.last_date
            )
            self.inflation_first_date: pd.Timestamp = self._inflation_instance.first_date
            self.inflation_last_date: pd.Timestamp = self._inflation_instance.last_date
            self.first_date = max(self.first_date, self.inflation_first_date)
            self.last_date = min(self.last_date, self.inflation_last_date)
            self.inflation_ts: pd.Series = self._inflation_instance.values_ts.loc[self.first_date: self.last_date]
            # Add inflation to the date range dict
            self.assets_first_dates.update({self.inflation: macro.Inflation(self.inflation).first_date})
            self.assets_last_dates.update({self.inflation: macro.Inflation(self.inflation).last_date})
        self.assets_ror: pd.DataFrame = self.assets_ror[
            self.first_date: self.last_date
        ]
        self.period_length: float = round(
            (self.last_date - self.first_date) / np.timedelta64(365, "D"), ndigits=1
        )
        self.pl = settings.PeriodLength(
            self.assets_ror.shape[0] // settings._MONTHS_PER_YEAR,
            self.assets_ror.shape[0] % settings._MONTHS_PER_YEAR,
        )
        self._pl_txt = f"{self.pl.years} years, {self.pl.months} months"
        self._dividend_yield: pd.DataFrame = pd.DataFrame(dtype=float)
        self._assets_dividends_ts: pd.DataFrame = pd.DataFrame(dtype=float)

    @abstractmethod
    def __repr__(self):
        pass

    def __len__(self):
        return len(self.symbols)

    def _make_list(self, ls: list, first_date, last_date) -> dict:
        """
        Make an asset list from a list of symbols.
        """
        base_currency_name: str = self._currency.name
        currency_first_date: pd.Timestamp = self._currency.first_date
        currency_last_date: pd.Timestamp = self._currency.last_date

        asset_obj_dict = {}  # dict of Asset/Portfolio type objects
        first_dates: Dict[str, pd.Timestamp] = {}
        last_dates: Dict[str, pd.Timestamp] = {}
        names: Dict[str, str] = {}
        currencies: Dict[str, str] = {}
        df = pd.DataFrame()
        input_first_date = pd.to_datetime(first_date) if first_date else None
        input_last_date = pd.to_datetime(last_date) if last_date else None
        for i, x in enumerate(ls):
            asset_item = x if hasattr(x, 'symbol') and hasattr(x, 'ror') else asset.Asset(x)
            if asset_item.pl.years == 0 and asset_item.pl.months <= 2:
                raise ValueError(f'{asset_item.symbol} period length is {asset_item.pl.months}. It should be at least 3 months.')
            if i == 0:  # required to use pd.concat below (df should not be empty).
                df = self._make_ror(asset_item, base_currency_name)
            else:
                new = self._make_ror(asset_item, base_currency_name)
                df = pd.concat([df, new], axis=1, join="inner", copy="false")
            # get first and last dates
            asset_first_date = df.index[0].to_timestamp()
            asset_last_date = df.index[-1].to_timestamp()
            # check first and last dates
            fd = [asset_first_date, input_first_date]
            ld = [asset_last_date, input_last_date]
            fd_max = max(x for x in fd if x is not None)
            ld_min = min(x for x in ld if x is not None)
            if helpers.Date.get_difference_in_months(ld_min, fd_max).n < 2:
                raise ValueError(f'{asset_item.symbol} historical data period length is too short. '
                                 f'It must be at least 3 months.')
            # uppend data to dictionaries
            asset_obj_dict[asset_item.symbol] = asset_item
            currencies[asset_item.symbol] = asset_item.currency
            names[asset_item.symbol] = asset_item.name
            first_dates[asset_item.symbol] = asset_first_date
            last_dates[asset_item.symbol] = asset_last_date
        first_dates[base_currency_name] = currency_first_date
        last_dates[base_currency_name] = currency_last_date
        currencies["asset list"] = base_currency_name
        # get first and last dates
        first_date_list = list(first_dates.values()) + [input_first_date]
        last_date_list = list(last_dates.values()) + [input_last_date]
        list_first_date = max(x for x in first_date_list if x is not None)
        list_last_date = min(x for x in last_date_list if x is not None)
        # range of last and first dates not limeted by AssetList first_date & lastdate parameters
        first_dates_sorted: list = sorted(first_dates.items(), key=lambda y: y[1])
        last_dates_sorted: list = sorted(last_dates.items(), key=lambda y: y[1])
        if isinstance(df, pd.Series):
            # required to convert Series to DataFrame for single asset list
            df = df.to_frame()
        return dict(
            asset_obj_list=asset_obj_dict,
            first_date=list_first_date,
            last_date=list_last_date,
            newest_asset=first_dates_sorted[-1][0],
            eldest_asset=first_dates_sorted[0][0],
            names_dict=names,
            currencies_dict=currencies,
            assets_first_dates=dict(first_dates_sorted),
            assets_last_dates=dict(last_dates_sorted),
            ror=df,
        )

    def _make_ror(self, list_asset: asset.Asset, base_currency_name: str) -> pd.Series:
        """
        Make aseet reate of return time series.
        """
        asset_currency_name = list_asset.currency
        if asset_currency_name == base_currency_name:
            ror = list_asset.ror
        else:
            asset_currency = asset.Asset(symbol=f"{asset_currency_name}{base_currency_name}.FX")
            ror = self._adjust_ror_to_currency(returns=list_asset.ror, asset_currency=asset_currency)
        return ror

    @classmethod
    def _adjust_ror_to_currency(cls, returns: pd.Series, asset_currency: asset.Asset) -> pd.Series:
        """
        Adjust returns time series to a certain currency.
        """
        asset_mult = returns + 1.0
        currency_mult = asset_currency.ror + 1.0
        # join dataframes to have the same Time Series Index
        df = pd.concat([asset_mult, currency_mult], axis=1, join="inner", copy="false")
        currency_mult = df.iloc[:, -1]
        asset_mult = df.iloc[:, 0]
        x = asset_mult * currency_mult - 1.0
        x.rename(returns.name, inplace=True)
        return x

    def _adjust_price_to_currency_monthly(self, price: pd.Series, asset_currency: str) -> pd.Series:
        """
        Adjust monthly time series of dividends or close values to a base currency.
        """
        ccy_symbol = f"{asset_currency}{self.currency}.FX"
        currency_rate = asset.Asset(ccy_symbol).close_monthly.to_frame()
        merged = price.to_frame().join(currency_rate, how="left")
        if merged.isnull().values.any():
            # can happen if the first value is missing
            merged.fillna(method='backfill', inplace=True)
        return merged.iloc[:, 0].multiply(merged[ccy_symbol], axis=0)

    @staticmethod
    def _define_symbol_list(assets):
        return [asset.symbol if hasattr(asset, 'symbol') else asset for asset in assets]

    def _add_inflation(self) -> pd.DataFrame:
        """
        Add inflation column to returns DataFrame.
        """
        if hasattr(self, "inflation"):
            return pd.concat(
                [self.assets_ror, self.inflation_ts], axis=1, join="inner", copy="false"
            )
        else:
            return self.assets_ror

    def _validate_period(self, period: Any) -> None:
        """
        Check if conditions are met:
        * period should be an integer
        * period should be positive
        * period should not exceed history period length

        Parameters
        ----------
        period : Any

        Returns
        -------
        None
            No exceptions raised if validation passes.
        """
        validators.validate_integer("period", period, min_value=0, inclusive=False)
        if period > self.pl.years:
            raise ValueError(
                f"'period' ({period}) is beyond historical data range ({self.period_length})."
            )

    def _get_single_asset_dividends(
        self, tick: str, remove_forecast: bool = True
    ) -> pd.Series:
        """
        Get monthly dividend time series for a single symbol and adjust to the currency.
        """
        asset = self.asset_obj_dict[tick]
        s = asset.dividends[self.first_date: self.last_date]
        if asset.currency != self.currency:
            s = self._adjust_price_to_currency_monthly(s, asset.currency)
        if remove_forecast:
            s = s[: pd.Period.now(freq="M")]
        # Create time series with zeros to pad the empty spaces in dividends time series
        index = pd.date_range(start=self.first_date, end=self.last_date, freq="MS")  # 'MS' to include the last period
        period = index.to_period("M")
        pad_s = pd.Series(data=0, index=period)
        return s.add(pad_s, fill_value=0)

    def _get_assets_dividends(self, remove_forecast=True) -> pd.DataFrame:
        """
        Get monthly dividend time series for all assets.

        If `remove_forecast=True` all forecasted (future) data is removed from the time series.
        """
        if self._assets_dividends_ts.empty:
            dic = {}
            for tick in self.symbols:
                s = self._get_single_asset_dividends(tick, remove_forecast=remove_forecast)
                dic[tick] = s
            self._assets_dividends_ts = pd.DataFrame(dic)
        return self._assets_dividends_ts

    def _make_real_return_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate real monthly return time series.

        Rate of return monthly data is adjusted for inflation.
        """
        if not hasattr(self, "inflation"):
            raise ValueError(
                "Real return is not defined. Set inflation=True when initiating the class."
            )
        df = (1.0 + df).divide(1.0 + self.inflation_ts, axis=0) - 1.0
        df.drop(columns=[self.inflation], inplace=True)
        return df

    @property
    def assets_dividend_yield(self) -> pd.DataFrame:
        """
        Calculate last twelve months (LTM) dividend yield time series (monthly) for each asset.

        LTM dividend yield is the sum trailing twelve months of common dividends per share divided by
        the current price per share.

        All yields are calculated in the asset list base currency after adjusting the dividends and price time series.
        Forecasted (future) dividends are removed.
        Zero value time series are created for assets without dividends.

        Returns
        -------
        DataFrame
            Time series of LTM dividend yield for each asset.

        Examples
        --------
        >>> x = ok.AssetList(['T.US', 'XOM.US'], first_date='1984-01', last_date='1994-12')
        >>> x.assets_dividend_yield
                   T.US    XOM.US
        1984-01  0.000000  0.000000
        1984-02  0.000000  0.002597
        1984-03  0.002038  0.002589
        1984-04  0.001961  0.002346
                   ...       ...
        1994-09  0.018165  0.012522
        1994-10  0.018651  0.011451
        1994-11  0.018876  0.012050
        1994-12  0.019344  0.011975
        [132 rows x 2 columns]
        """
        if self._dividend_yield.empty:
            frame = {}
            df = self._get_assets_dividends(remove_forecast=True)
            for tick in self.symbols:
                div_monthly = df[tick]
                if div_monthly.sum() != 0:
                    asset = self.asset_obj_dict[tick]
                    price_monthly_ts = asset.close_monthly.loc[self.first_date: self.last_date]
                    if asset.currency != self.currency:
                        price_monthly_ts = self._adjust_price_to_currency_monthly(price_monthly_ts, asset.currency)
                else:
                    # skipping prices if no dividends
                    div_yield = div_monthly
                    frame.update({tick: div_yield})
                    continue
                # Get dividend yield time series
                div_yield = pd.Series(dtype=float)
                div_monthly.index = div_monthly.index.to_timestamp()
                for date in price_monthly_ts.index.to_timestamp(how="End"):
                    ltm_div = div_monthly[:date].last("12M").sum()
                    last_price = price_monthly_ts.loc[:date].iloc[-1]
                    value = ltm_div / last_price
                    div_yield.at[date] = value
                div_yield.index = div_yield.index.to_period("M")
                frame.update({tick: div_yield})
            self._dividend_yield = pd.DataFrame(frame)
        return self._dividend_yield

    @property
    def _list_of_asset_like_objects(self) -> List[Union[str, Type]]:
        """
        Return list which may include tickers or asset like objects (Portfolio, Asset).

        Returns
        -------
        list
        """
        assets = [settings.default_ticker] if not self._assets else self._assets
        if not isinstance(assets, list):
            raise ValueError("Assets must be a list.")
        return assets

    @property
    def symbols(self) -> List[str]:
        """
        Return a list of financial symbols used to set the AssetList.

        Symbols are similar to tickers but have a namespace information:

        * SPY.US is a symbol
        * SPY is a ticker

        Returns
        -------
        list of str
            List of symbols included in the Asset List.
        """
        return self._define_symbol_list(self._list_of_asset_like_objects)

    @property
    def tickers(self) -> List[str]:
        """
        Return a list of tickers (symbols without a namespace) used to set the AssetList.

        tickers are similar to symbols but do not have namespace information:

        * SPY is a ticker
        * SPY.US is a symbol

        Returns
        -------
        list of str
            List of tickers included in the Asset List.
        """
        return [x.split(".", 1)[0] for x in self.symbols]

    @property
    def currency(self) -> str:
        """
        Return the base currency of the Asset List.

        Such properties as rate of return and risk are adjusted to the base currency.

        Returns
        -------
        okama.Asset
            Base currency of the Asset List in form of okama.Asset class.
        """
        return self._currency.currency

    def plot_assets(
        self,
        kind: str = "mean",
        tickers: Union[str, list] = "tickers",
        pct_values: bool = False,
    ) -> plt.axes:
        """
        Plot the assets points on the risk-return chart with annotations.

        Annualized values for risk and return are used.
        Risk is a standard deviation of monthly rate of return time series.
        Return can be an annualized mean return (expected return) or CAGR (Compound annual growth rate).

        Returns
        -------
        Axes : 'matplotlib.axes._subplots.AxesSubplot'

        Parameters
        ----------
        kind : {'mean', 'cagr'}, default 'mean'
            Type of Return: annualized mean return (expected return) or CAGR (Compound annual growth rate).

        tickers : {'tickers', 'names'} or list of str, default 'tickers'
            Annotation type for assets.
            'tickers' - assets symbols are shown in form of 'SPY.US'
            'names' - assets names are used like - 'SPDR S&P 500 ETF Trust'
            To show custom annotations for each asset pass the list of names.

        pct_values : bool, default False
            Risk and return values in the axes:
            Algebraic annotation (False)
            Percents (True)

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> al = ok.AssetList(['SPY.US', 'AGG.US'], ccy='USD', inflation=False)
        >>> al.plot_assets()
        >>> plt.show()

        Plotting with default parameters values shows expected return, ticker annotations and algebraic values
        for risk and return.
        To use CAGR instead of expected return use kind='cagr'.

        >>> al.plot_assets(kind='cagr',
        ...               tickers=['US Stocks', 'US Bonds'],  # use custom annotations for the assets
        ...               pct_values=True  # risk and return values are in percents
        ...               )
        >>> plt.show()
        """
        risk_monthly = self.assets_ror.std()
        mean_return_monthly = self.assets_ror.mean()
        risks = helpers.Float.annualize_risk(risk_monthly, mean_return_monthly)
        if kind == "mean":
            returns = helpers.Float.annualize_return(self.assets_ror.mean())
        elif kind == "cagr":
            returns = helpers.Frame.get_cagr(self.assets_ror).loc[self.symbols]
        else:
            raise ValueError('kind should be "mean" or "cagr".')
        # set lists for single point scatter
        if len(self.symbols) < 2:
            risks = [risks]
            returns = [returns]
        # set the plot
        ax = plt.gca()
        plt.autoscale(enable=True, axis="year", tight=False)
        ax.margins(.05, .1)  # increase margins on Y-axis from 5% to 10% as `annotate` moves text upwards
        m = 100 if pct_values else 1
        ax.scatter(risks * m, returns * m)
        # Set the labels
        if tickers == "tickers":
            asset_labels = self.symbols
        elif tickers == "names":
            asset_labels = list(self.names.values())
        else:
            if not isinstance(tickers, list):
                raise ValueError(
                    "tickers parameter should be a list of string labels."
                )
            if len(tickers) != len(self.symbols):
                raise ValueError("labels and tickers must be of the same length")
            asset_labels = tickers
        # draw the points and print the labels
        for label, x, y in zip(asset_labels, risks, returns):
            ax.annotate(
                label,  # this is the text
                (x * m, y * m),  # this is the point to label
                textcoords="offset points",  # how to position the text
                xytext=(0, 10),  # distance from text to points (x,y)
                ha="center",  # horizontal alignment can be left, right or center
            )
        return ax
