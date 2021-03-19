from typing import Union, Optional, List, Dict, Tuple

import pandas as pd
import numpy as np

from .macro import Inflation
from .common.helpers import Float, Frame, Date, Index
from .settings import default_ticker, PeriodLength, _MONTHS_PER_YEAR
from .api.data_queries import QueryData
from .api.namespaces import get_assets_namespaces


class Asset:
    """
    A financial asset, that could be used in a list of assets or in portfolio.
    """

    def __init__(self, symbol: str = default_ticker):
        if symbol is None or len(str(symbol).strip()) == 0:
            raise ValueError('Symbol can not be empty')
        self._symbol = str(symbol).strip()
        self._check_namespace()
        self._get_symbol_data(symbol)
        self.ror: pd.Series = QueryData.get_ror(symbol)
        self.first_date: pd.Timestamp = self.ror.index[0].to_timestamp()
        self.last_date: pd.Timestamp = self.ror.index[-1].to_timestamp()
        self.period_length: float = round((self.last_date - self.first_date) / np.timedelta64(365, 'D'), ndigits=1)

    def __repr__(self):
        dic = {
            'symbol': self.symbol,
            'name': self.name,
            'country': self.country,
            'exchange': self.exchange,
            'currency': self.currency,
            'type': self.type,
            'first date': self.first_date.strftime("%Y-%m"),
            'last date': self.last_date.strftime("%Y-%m"),
            'period length': "{:.2f}".format(self.period_length)
        }
        return repr(pd.Series(dic))

    def _check_namespace(self):
        namespace = self._symbol.split('.', 1)[-1]
        allowed_namespaces = get_assets_namespaces()
        if namespace not in allowed_namespaces:
            raise ValueError(f'{namespace} is not in allowed assets namespaces: {allowed_namespaces}')

    @property
    def symbol(self):
        return self._symbol

    def _get_symbol_data(self, symbol) -> None:
        x = QueryData.get_symbol_info(symbol)
        self.ticker: str = x['code']
        self.name: str = x['name']
        self.country: str = x['country']
        self.exchange: str = x['exchange']
        self.currency: str = x['currency']
        self.type: str = x['type']
        self.inflation: str = f'{self.currency}.INFL'

    @property
    def price(self) -> float:
        """
        Live price of an asset.
        """
        return QueryData.get_live_price(self.symbol)

    @property
    def dividends(self) -> pd.Series:
        """
        Dividends time series daily data.
        Not defined for namespaces: 'PIF', 'INFL', 'INDX', 'FX', 'COMM'
        """
        div = QueryData.get_dividends(self.symbol)
        if div.empty:
            # Zero time series for assets where dividend yield is not defined.
            index = pd.date_range(start=self.first_date, end=self.last_date, freq='MS', closed=None)
            period = index.to_period('D')
            div = pd.Series(data=0, index=period)
            div.rename(self.symbol, inplace=True)
        return div

    @property
    def nav_ts(self) -> pd.Series:
        """
        NAV time series (monthly) for mutual funds when available in data.
        """
        if self.exchange == 'PIF':
            return QueryData.get_nav(self.symbol)
        return np.nan


class AssetList:
    """
    The list of financial assets implementation.
    """
    def __init__(self,
                 symbols: Optional[List[str]] = None, *,
                 first_date: Optional[str] = None,
                 last_date: Optional[str] = None,
                 ccy: str = 'USD',
                 inflation: bool = True):
        self.__symbols = symbols
        self.__tickers: List[str] = [x.split(".", 1)[0] for x in self.symbols]
        self.__currency: Asset = Asset(symbol=f'{ccy}.FX')
        self.__make_asset_list(self.symbols)
        if inflation:
            self.inflation: str = f'{ccy}.INFL'
            self._inflation_instance: Inflation = Inflation(self.inflation, self.first_date, self.last_date)
            self.inflation_ts: pd.Series = self._inflation_instance.values_ts
            self.inflation_first_date: pd.Timestamp = self._inflation_instance.first_date
            self.inflation_last_date: pd.Timestamp = self._inflation_instance.last_date
            self.first_date = max(self.first_date, self.inflation_first_date)
            self.last_date: pd.Timestamp = min(self.last_date, self.inflation_last_date)
            # Add inflation to the date range dict
            self.assets_first_dates.update({self.inflation: self.inflation_first_date})
            self.assets_last_dates.update({self.inflation: self.inflation_last_date})
        if first_date:
            self.first_date = max(self.first_date, pd.to_datetime(first_date))
        self.ror = self.ror[self.first_date:]
        if last_date:
            self.last_date = min(self.last_date, pd.to_datetime(last_date))
        self.ror: pd.DataFrame = self.ror[self.first_date: self.last_date]
        self.period_length: float = round((self.last_date - self.first_date) / np.timedelta64(365, 'D'), ndigits=1)
        self.pl = PeriodLength(self.ror.shape[0] // _MONTHS_PER_YEAR, self.ror.shape[0] % _MONTHS_PER_YEAR)
        self._pl_txt = f'{self.pl.years} years, {self.pl.months} months'
        self._dividend_yield: pd.DataFrame = pd.DataFrame(dtype=float)
        self._dividends_ts: pd.DataFrame = pd.DataFrame(dtype=float)

    def __repr__(self):
        dic = {
            'symbols': self.symbols,
            'currency': self.currency.ticker,
            'first date': self.first_date.strftime("%Y-%m"),
            'last_date': self.last_date.strftime("%Y-%m"),
            'period length': self._pl_txt,
            'inflation': self.inflation if hasattr(self, 'inflation') else 'None',
        }
        return repr(pd.Series(dic))

    def __len__(self):
        return len(self.symbols)

    def __make_asset_list(self, ls: list) -> None:
        """
        Make an asset list from a list of symbols.
        """
        first_dates: Dict[str, pd.Timestamp] = {}
        last_dates: Dict[str, pd.Timestamp] = {}
        names: Dict[str, str] = {}
        currencies: Dict[str, str] = {}
        df = pd.DataFrame()
        for i, x in enumerate(ls):
            asset = Asset(x)
            if i == 0:
                if asset.currency == self.currency.name:
                    df = asset.ror
                else:
                    df = self._set_currency(returns=asset.ror, asset_currency=asset.currency)
            else:
                if asset.currency == self.currency.name:
                    new = asset.ror
                else:
                    new = self._set_currency(returns=asset.ror, asset_currency=asset.currency)
                df = pd.concat([df, new], axis=1, join='inner', copy='false')
            currencies.update({asset.symbol: asset.currency})
            names.update({asset.symbol: asset.name})
            first_dates.update({asset.symbol: asset.first_date})
            last_dates.update({asset.symbol: asset.last_date})
        # Add currency to the date range dict
        first_dates.update({self.currency.name: self.currency.first_date})
        last_dates.update({self.currency.name: self.currency.last_date})

        first_dates_sorted = sorted(first_dates.items(), key=lambda y: y[1])
        last_dates_sorted = sorted(last_dates.items(), key=lambda y: y[1])
        self.first_date: pd.Timestamp = first_dates_sorted[-1][1]
        self.last_date: pd.Timestamp = last_dates_sorted[0][1]
        self.newest_asset: str = first_dates_sorted[-1][0]
        self.eldest_asset: str = first_dates_sorted[0][0]
        self.names: Dict[str, str] = names
        currencies.update({'asset list': self.currency.currency})
        self.currencies: Dict[str, str] = currencies
        self.assets_first_dates: Dict[str, pd.Timestamp] = dict(first_dates_sorted)
        self.assets_last_dates: Dict[str, pd.Timestamp] = dict(last_dates_sorted)
        if isinstance(df, pd.Series):  # required to convert Series to DataFrame for single asset list
            df = df.to_frame()
        self.ror: pd.DataFrame = df

    def _set_currency(self, returns: pd.Series, asset_currency: str) -> pd.Series:
        """
        Set return to a certain currency. Input is a pd.Series of mean returns and a currency symbol.
        """
        currency = Asset(symbol=f'{asset_currency}{self.currency.name}.FX')
        asset_mult = returns + 1.
        currency_mult = currency.ror + 1.
        # join dataframes to have the same Time Series Index
        df = pd.concat([asset_mult, currency_mult], axis=1, join='inner', copy='false')
        currency_mult = df.iloc[:, -1]
        asset_mult = df.iloc[:, 0]
        x = asset_mult * currency_mult - 1.
        x.rename(returns.name, inplace=True)
        return x

    def _add_inflation(self) -> pd.DataFrame:
        """
        Add inflation column to returns DataFrame.
        """
        if hasattr(self, 'inflation'):
            return pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            return self.ror

    def _remove_inflation(self, time_frame: int) -> pd.DataFrame:
        """
        Remove inflation column from rolling returns if exists.
        Parameters
        """
        if hasattr(self, 'inflation'):
            return self.get_rolling_cumulative_return(window=time_frame).drop(columns=[self.inflation])
        else:
            return self.get_rolling_cumulative_return(window=time_frame)

    @property
    def symbols(self) -> List[str]:
        """
        Return a list of financial symbols used to set the AssetList.
        """
        symbols = [default_ticker] if not self.__symbols else self.__symbols
        if not isinstance(symbols, list):
            raise ValueError('Symbols should be a list of string values.')
        return symbols

    @property
    def tickers(self) -> List[str]:
        """
        Return a list of tickers (symbols without a namespace) used to set the AssetList.
        """
        return self.__tickers

    @property
    def currency(self) -> str:
        """
        Return the base currency. Such properties as rate of return and volatility are adjusted to the base currency.
        """
        return self.__currency

    @property
    def wealth_indexes(self) -> pd.DataFrame:
        """
        Return wealth index time series for the assets and accumulated inflation.
        Wealth index is obtained from the accumulated return multiplicated by the initial investments (1000).
        """
        df = self._add_inflation()
        return Frame.get_wealth_indexes(df)

    @property
    def risk_monthly(self) -> pd.Series:
        """
        Return monthly risks (standard deviation) for each asset.
        """
        return self.ror.std()

    @property
    def risk_annual(self) -> pd.Series:
        """
        Return annualized risks (standard deviation) for each asset.
        """
        risk = self.ror.std()
        mean_return = self.ror.mean()
        return Float.annualize_risk(risk, mean_return)

    @property
    def semideviation_monthly(self) -> pd.Series:
        """
        Return semideviation monthly values for each asset.
        """
        return Frame.get_semideviation(self.ror)

    @property
    def semideviation_annual(self) -> float:
        """
        Return semideviation annualized values for each asset.
        """
        return Frame.get_semideviation(self.ror) * 12 ** 0.5

    def get_var_historic(self, time_frame: int = 12, level: int = 5) -> pd.Series:
        """
        Calculate historic Value at Risk (VaR) for the assets.

        The VaR calculates the potential loss of an investment with a given time frame and confidence level.
        Loss is a positive number (expressed in cumulative return).
        If VaR is negative there are gains at this confidence level.

        Parameters
        ----------
        time_frame : int, default 12 (12 months)
        level : int, default 5 (5% quantile)

        Returns
        -------
        Series

        Examples
        --------
        >>> x = ok.AssetList(['SPY.US', 'AGG.US'])
        >>> x.get_var_historic(time_frame=60, level=1)
        SPY.US    0.2101
        AGG.US    -0.0867
        Name: VaR, dtype: float64
        """
        df = self._remove_inflation(time_frame)
        return Frame.get_var_historic(df, level)

    def get_cvar_historic(self, time_frame: int = 12, level: int = 5) -> pd.Series:
        """
        Calculate historic Conditional Value at Risk (CVAR, expected shortfall) for the assets.

        CVaR is the average loss over a specified time period of unlikely scenarios beyond the confidence level.
        Loss is a positive number (expressed in cumulative return).
        If CVaR is negative there are gains at this confidence level.

        Parameters
        ----------
        time_frame : int, default 12 (12 months)
        level : int, default 5 (5% quantile)

        Returns
        -------
        Series

        Examples
        --------
        >>> x = ok.AssetList(['SPY.US', 'AGG.US'])
        >>> x.get_cvar_historic(time_frame=60, level=1)
        SPY.US    0.2574
        AGG.US   -0.0766
        dtype: float64
        Name: VaR, dtype: float64
        """
        df = self._remove_inflation(time_frame)
        return Frame.get_cvar_historic(df, level)

    @property
    def drawdowns(self) -> pd.DataFrame:
        """
        Calculate drawdowns time series for the assets.

        The drawdown is the percent decline from a previous peak in wealth index.
        """
        return Frame.get_drawdowns(self.ror)

    def get_cagr(self, period: Union[int, None] = None) -> pd.Series:
        """
        Calculate assets Compound Annual Growth Rate (CAGR) for a given trailing period.

        Annual inflation data is shown for the same period if inflation=True (default) in the AssetList.
        CAGR is not defined for periods less than 1 year.

        Parameters
        ----------
        period: int, optional
            CAGR trailing period in years. None for full time CAGR.

        Returns
        -------
        Series

        Examples
        --------
        >>> x = ok.AssetList()
        >>> x.get_cagr(period=5)
        SPY.US    0.1510
        USD.INFL   0.0195
        dtype: float64
        """
        df = self._add_inflation()
        dt0 = self.last_date

        if not period:
            cagr = Frame.get_cagr(df)
        elif isinstance(period, int) and period > 0:
            dt = Date.subtract_years(dt0, period)
            if dt >= self.first_date:
                cagr = Frame.get_cagr(df[dt:])
            else:
                row = {x: None for x in df.columns}
                cagr = pd.Series(row)
        else:
            raise ValueError(f'{period} is not a valid value for period')
        return cagr

    def get_rolling_cagr(self, window: int = 12) -> pd.DataFrame:
        """
        Calculate rolling CAGR (Compound Annual Growth Rate) for each asset.

        Parameters
        ----------
        window : int, default 12
            Window size in months. Window size should be at least 12 months for CAGR.

        Returns
        -------
        DataFrame
            Time series of rolling CAGR.
        """
        df = self._add_inflation()
        return Frame.get_rolling_fn(df, window=window, fn=Frame.get_cagr)

    def get_cumulative_return(self, period: Union[str, int, None] = None) -> pd.Series:
        """
        Calculate cumulative return of return for the assets.

        Annual inflation data is shown for the same period if inflation=True (default) in the AssetList.

        Parameters
        ----------
        period: str, int or None, default None
            Trailing period in years.
            None - full time cumulative return.
            'YTD' - (Year To Date) period of time beginning the first day of the calendar year up to the last month.

        Returns
        -------
        Series

        Examples
        --------
        >>> x = ok.AssetList(['MCFTR.INDX'], ccy='RUB')
        >>> x.get_cumulative_return(period='YTD')
        MCFTR.INDX   0.1483
        RUB.INFL     0.0485
        dtype: float64
        """
        df = self._add_inflation()
        dt0 = self.last_date

        if not period:
            cr = Frame.get_cumulative_return(df)
        elif str(period).lower() == 'ytd':
            year = dt0.year
            cr = (df[str(year):] + 1.).prod() - 1.
        elif isinstance(period, int) and period > 0:
            dt = Date.subtract_years(dt0, period)
            if dt >= self.first_date:
                cr = Frame.get_cumulative_return(df[dt:])
            else:
                row = {x: None for x in df.columns}
                cr = pd.Series(row)
        else:
            raise ValueError(f'{period} is not a valid value for period')
        return cr

    def get_rolling_cumulative_return(self, window: int = 12) -> pd.DataFrame:
        """
        Calculate rolling cumulative return for each asset.

        Parameters
        ----------
        window : int, default 12
            Window size in months.

        Returns
        -------
            DataFrame
            Time series of rolling cumulative return.
        """
        df = self._add_inflation()
        return Frame.get_rolling_fn(df,
                                    window=window,
                                    fn=Frame.get_cumulative_return,
                                    window_below_year=True)

    @property
    def annual_return_ts(self) -> pd.DataFrame:
        """
        Calculate annual rate of return time series for each asset.

        Rate of return is calculated for each calendar year.
        """
        return Frame.get_annual_return_ts_from_monthly(self.ror)

    def describe(self, years: Tuple[int, ...] = (1, 5, 10), tickers: bool = True) -> pd.DataFrame:
        """
        Generate descriptive statistics for a list of assets.

        Statistics includes:
        - YTD (Year To date) compound return
        - CAGR for a given list of periods
        - Dividend yield - yield for last 12 months (LTM)

        Risk metrics (full available period):
        - risk (standard deviation)
        - CVAR
        - max drawdowns (and dates)

        Statistics also shows for each asset:
        - inception date - first date available for each asset
        - last asset date - available for each asset date
        - Common last data date - common for the asset list data (may be set by last_date manually)

        Parameters
        ----------
        years : tuple of (int,), default (1, 5, 10)
            List of periods for CAGR.

        tickers : bool, default True
            Defines whether show tickers (True) or assets names in the header.

        Returns
        -------
            DataFrame

        See Also
        --------
            get_cumulative_return : Calculate cumulative return.
            get_cagr : Calculate assets Compound Annual Growth Rate (CAGR).
            dividend_yield : Calculate dividend yield (LTM).
            risk_annual : Return annualized risks (standard deviation).
            get_cvar : Calculate historic Conditional Value at Risk (CVAR, expected shortfall).
            drawdowns : Calculate drawdowns.
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        df = self._add_inflation()
        # YTD return
        ytd_return = self.get_cumulative_return(period='YTD')
        row = ytd_return.to_dict()
        row.update({'period': 'YTD'})
        row.update({'property': 'Compound return'})
        description = description.append(row, ignore_index=True)
        # CAGR for a list of periods
        if self.pl.years >= 1:
            for i in years:
                dt = Date.subtract_years(dt0, i)
                if dt >= self.first_date:
                    row = self.get_cagr(period=i).to_dict()
                else:
                    row = {x: None for x in df.columns}
                row.update({'period': f'{i} years'})
                row.update({'property': 'CAGR'})
                description = description.append(row, ignore_index=True)
            # CAGR for full period
            row = self.get_cagr(period=None).to_dict()
            row.update({'period': self._pl_txt})
            row.update({'property': 'CAGR'})
            description = description.append(row, ignore_index=True)
            # Dividend Yield
            row = self.dividend_yield.iloc[-1].to_dict()
            row.update({'period': 'LTM'})
            row.update({'property': 'Dividend yield'})
            description = description.append(row, ignore_index=True)
        # risk for full period
        row = self.risk_annual.to_dict()
        row.update({'period': self._pl_txt})
        row.update({'property': 'Risk'})
        description = description.append(row, ignore_index=True)
        # CVAR
        if self.pl.years >= 1:
            row = self.get_cvar_historic().to_dict()
            row.update({'period': self._pl_txt})
            row.update({'property': 'CVAR'})
            description = description.append(row, ignore_index=True)
        # max drawdowns
        row = self.drawdowns.min().to_dict()
        row.update({'period': self._pl_txt})
        row.update({'property': 'Max drawdowns'})
        description = description.append(row, ignore_index=True)
        # max drawdowns dates
        row = self.drawdowns.idxmin().to_dict()
        row.update({'period': self._pl_txt})
        row.update({'property': 'Max drawdowns dates'})
        description = description.append(row, ignore_index=True)
        # inception dates
        row = {}
        for ti in self.symbols:
            # short_ticker = ti.split(".", 1)[0]
            value = self.assets_first_dates[ti].strftime("%Y-%m")
            row.update({ti: value})
        row.update({'period': None})
        row.update({'property': 'Inception date'})
        if hasattr(self, 'inflation'):
            row.update({self.inflation: self.inflation_first_date.strftime("%Y-%m")})
        description = description.append(row, ignore_index=True)
        # last asset date
        row = {}
        for ti in self.symbols:
            # short_ticker = ti.split(".", 1)[0]
            value = self.assets_last_dates[ti].strftime("%Y-%m")
            row.update({ti: value})
        row.update({'period': None})
        row.update({'property': 'Last asset date'})
        if hasattr(self, 'inflation'):
            row.update({self.inflation: self.inflation_last_date.strftime("%Y-%m")})
        description = description.append(row, ignore_index=True)
        # last data date
        row = {x: self.last_date.strftime("%Y-%m") for x in df.columns}
        row.update({'period': None})
        row.update({'property': 'Common last data date'})
        description = description.append(row, ignore_index=True)
        # rename columns
        if hasattr(self, 'inflation'):
            description.rename(columns={self.inflation: 'inflation'}, inplace=True)
            description = Frame.change_columns_order(description, ['inflation'], position='last')
        description = Frame.change_columns_order(description, ['property', 'period'], position='first')
        if not tickers:
            for ti in self.symbols:
                # short_ticker = ti.split(".", 1)[0]
                description.rename(columns={ti: self.names[ti]}, inplace=True)
        return description

    @property
    def mean_return(self) -> pd.Series:
        """
        Calculate annualized mean return (arithmetic mean) for the assets.
        """
        df = self._add_inflation()
        mean = df.mean()
        return Float.annualize_return(mean)

    @property
    def real_mean_return(self) -> pd.Series:
        """
        Calculates annualized real mean return (arithmetic mean) for the assets.

        Real rate of return is adjusted for inflation.
        """
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            raise Exception('Real Return is not defined. Set inflation=True to calculate.')
        infl_mean = Float.annualize_return(self.inflation_ts.values.mean())
        ror_mean = Float.annualize_return(df.loc[:, self.symbols].mean())
        return (1. + ror_mean) / (1. + infl_mean) - 1.

    def _get_asset_dividends(self, tick, remove_forecast=True) -> pd.Series:
        """
        Get dividend time series for a single symbol.
        """
        first_period = pd.Period(self.first_date, freq='M')
        first_day = first_period.to_timestamp(how='Start')
        last_period = pd.Period(self.last_date, freq='M')
        last_day = last_period.to_timestamp(how='End')
        s = Asset(tick).dividends[first_day: last_day]  # limit divs by first_day and last_day
        if remove_forecast:
            s = s[:pd.Period.now(freq='D')]
        # Create time series with zeros to pad the empty spaces in dividends time series
        index = pd.date_range(start=first_day, end=last_day, freq='D')
        period = index.to_period('D')
        pad_s = pd.Series(data=0, index=period)
        return s.add(pad_s, fill_value=0)

    def _get_dividends(self, remove_forecast=True) -> pd.DataFrame:
        """
        Get dividend time series for all assets.

        If remove_forecast=True all forecasted (future) data is removed from time series.
        """
        if self._dividends_ts.empty:
            dic = {}
            for tick in self.symbols:
                s = self._get_asset_dividends(tick, remove_forecast=remove_forecast)
                dic.update({tick: s})
            self._dividends_ts = pd.DataFrame(dic)
        return self._dividends_ts

    @property
    def dividend_yield(self) -> pd.DataFrame:
        """
        Calculate last twelve months (LTM) dividend yield time series (monthly) for each asset.

        All yields are calculated in the original asset currency (not adjusting to AssetList currency).
        Forecast dividends are removed.
        Zero value time series are created for assets without dividends.
        """
        if self._dividend_yield.empty:
            frame = {}
            df = self._get_dividends(remove_forecast=True)
            for tick in self.symbols:
                # Get dividends time series
                div = df[tick]
                # Get close (not adjusted) values time series.
                # If the last_date month is current month live price of assets is used.
                if div.sum() != 0:
                    div_monthly = div.resample('M').sum()
                    price = QueryData.get_close(tick, period='M').loc[self.first_date: self.last_date]
                else:
                    # skipping prices if no dividends
                    div_yield = div.asfreq(freq='M')
                    frame.update({tick: div_yield})
                    continue
                if price.index[-1] == pd.Period(pd.Timestamp.today(), freq='M'):
                    price.loc[f'{pd.Timestamp.today().year}-{pd.Timestamp.today().month}'] = Asset(tick).price
                # Get dividend yield time series
                div_yield = pd.Series(dtype=float)
                div_monthly.index = div_monthly.index.to_timestamp()
                for date in price.index.to_timestamp(how='End'):
                    ltm_div = div_monthly[:date].last('12M').sum()
                    last_price = price.loc[:date].iloc[-1]
                    value = ltm_div / last_price
                    div_yield.at[date] = value
                div_yield.index = div_yield.index.to_period('M')
                # Currency adjusted yield
                # if self.currencies[tick] != self.currency.name:
                #     div_yield = self._set_currency(returns=div_yield, asset_currency=self.currencies[tick])
                frame.update({tick: div_yield})
            self._dividend_yield = pd.DataFrame(frame)
        return self._dividend_yield

    @property
    def dividends_annual(self) -> pd.DataFrame:
        """
        Return calendar year dividends time series for each asset.
        """
        return self._get_dividends().resample('Y').sum()

    @property
    def dividend_growing_years(self) -> pd.DataFrame:
        """
        Return the number of growing dividend years for each asset.

        TODO: finish description. Insert an example
        """
        div_growth = self.dividends_annual.pct_change()[1:]
        df = pd.DataFrame()
        for name in div_growth:
            s = div_growth[name]
            s1 = s.where(s > 0).notnull().astype(int)
            s1_1 = s.where(s > 0).isnull().astype(int).cumsum()
            s2 = s1.groupby(s1_1).cumsum()
            df = pd.concat([df, s2], axis=1, copy='false')
        return df

    @property
    def dividend_paying_years(self) -> pd.DataFrame:
        """
        Return the number of years of consecutive dividend payments for each asset.
        """
        div_annual = self.dividends_annual
        frame = pd.DataFrame()
        df = frame
        for name in div_annual:
            s = div_annual[name]
            s1 = s.where(s != 0).notnull().astype(int)
            s1_1 = s.where(s != 0).isnull().astype(int).cumsum()
            s2 = s1.groupby(s1_1).cumsum()
            df = pd.concat([df, s2], axis=1, copy='false')
        return df

    def get_dividend_mean_growth_rate(self, period=5) -> pd.Series:
        """
        Calculate geometric mean of dividends growth rate time series for a given period.
        Period should be integer and not exceed the available data period_length.
        """
        if period > self.pl.years or not isinstance(period, int):
            raise TypeError(f'{period} is not a valid value for period')
        growth_ts = self.dividends_annual.pct_change().iloc[1:-1]  # Slice the last year for full dividends
        dt0 = self.last_date
        dt = Date.subtract_years(dt0, period)
        return ((growth_ts[dt:] + 1.).prod()) ** (1 / period) - 1.

    # index methods
    @property
    def tracking_difference(self):
        """
        Returns tracking difference for the rate of return of assets.
        Assets are compared with the index or another benchmark.
        Index should be in the first position (first column).
        """
        accumulated_return = Frame.get_wealth_indexes(self.ror)  # we don't need inflation here
        return Index.tracking_difference(accumulated_return)

    @property
    def tracking_difference_annualized(self):
        """
        Annualizes the values of tracking difference time series.
        Annual values are available for periods of more than 12 months.
        Returns for less than 12 months can't be annualized.
        """
        return Index.tracking_difference_annualized(self.tracking_difference)

    @property
    def tracking_error(self):
        """
        Returns tracking error for the rate of return time series of assets.
        Assets are compared with the index or another benchmark.
        Index should be in the first position (first column).
        """
        return Index.tracking_error(self.ror)

    @property
    def index_corr(self):
        """
        Compute expanding correlation with the index (or benchmark) time series for the assets.
        Index should be in the first position (first column).
        The period should be at least 12 months.
        """
        return Index.cov_cor(self.ror, fn='corr')

    def index_rolling_corr(self, window: int = 60):
        """
        Compute rolling correlation with the index (or benchmark) time series for the assets.
        Index should be in the first position (first column).
        The period should be at least 12 months.
        window - the rolling window size in months (default is 5 years).
        """
        return Index.rolling_cov_cor(self.ror, window=window, fn='corr')

    @property
    def index_beta(self):
        """
        Compute beta coefficient time series for the assets.
        Index (or benchmark) should be in the first position (first column).
        Rolling window size should be at least 12 months.
        """
        return Index.beta(self.ror)

    # distributions
    @property
    def skewness(self):
        """
        Compute expanding skewness of the return time series for each asset returns.
        For normally distributed data, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.
        """
        return Frame.skewness(self.ror)

    def skewness_rolling(self, window: int = 60):
        """
        Compute rolling skewness of the return time series for each asset returns.
        For normally distributed data, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.

        window - the rolling window size in months (default is 5 years).
        The window size should be at least 12 months.
        """
        return Frame.skewness_rolling(self.ror, window=window)

    @property
    def kurtosis(self):
        """
        Calculate expanding Fisher (normalized) kurtosis time series for each asset returns.
        Kurtosis is the fourth central moment divided by the square of the variance.
        Kurtosis should be close to zero for normal distribution.
        """
        return Frame.kurtosis(self.ror)

    def kurtosis_rolling(self, window: int = 60):
        """
        Calculate rolling Fisher (normalized) kurtosis time series for each asset returns.
        Kurtosis is the fourth central moment divided by the square of the variance.
        Kurtosis should be close to zero for normal distribution.

        window - the rolling window size in months (default is 5 years).
        The window size should be at least 12 months.
        """
        return Frame.kurtosis_rolling(self.ror, window=window)

    @property
    def jarque_bera(self):
        """
        Perform Jarque-Bera test for normality of assets returns historical data.
        It shows whether the returns have the skewness and kurtosis matching a normal distribution.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
            Low statistic numbers correspond to normal distribution.
        """
        return Frame.jarque_bera_dataframe(self.ror)

    def kstest(self, distr: str = 'norm') -> pd.DataFrame:
        """
        Perform Kolmogorov-Smirnov test for goodness of fit the asset returns to a given distribution.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
            Low statistic numbers correspond to normal distribution.
        """
        return Frame.kstest_dataframe(self.ror, distr=distr)
