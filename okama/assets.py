from typing import Union, Optional, List, Dict

import pandas as pd
import numpy as np

from .macro import Inflation
from .helpers import Float, Frame, Date, Index
from .settings import default_ticker, PeriodLength, _MONTHS_PER_YEAR
from .api.data_queries import QueryData
from .api.namespaces import get_assets_namespaces


class Asset:
    """
    An asset, that could be used in a list of assets or in portfolio.
    Works with monthly end of day historical rate of return data.
    """

    def __init__(self, symbol: str = default_ticker):
        self.symbol: str = symbol
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
        namespace = self.symbol.split('.', 1)[-1]
        allowed_namespaces = get_assets_namespaces()
        if namespace not in allowed_namespaces:
            raise Exception(f'{namespace} is not in allowed assets namespaces: {allowed_namespaces}')

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
    The list of assets implementation.
    Works with monthly end of day historical rate of return data.
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
            self.first_date: pd.Timestamp = max(self.first_date, self.inflation_first_date)
            self.last_date: pd.Timestamp = min(self.last_date, self.inflation_last_date)
            # Add inflation to the date range dict
            self.assets_first_dates.update({self.inflation: self.inflation_first_date})
            self.assets_last_dates.update({self.inflation: self.inflation_last_date})
        if first_date:
            self.first_date: pd.Timestamp = max(self.first_date, pd.to_datetime(first_date))
        self.ror = self.ror[self.first_date:]
        if last_date:
            # TODO: self.assets_last_dates should be less or equal to self.last_date
            self.last_date: pd.Timestamp = min(self.last_date, pd.to_datetime(last_date))
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
        Makes an asset list from a list of symbols. Returns DataFrame of returns (monthly) as an attribute.
        """
        first_dates: Dict[str, pd.Timestamp] = {}
        last_dates: Dict[str, pd.Timestamp] = {}
        names: Dict[str, str] = {}
        currencies: Dict[str, str] = {}
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

        first_dates_sorted = sorted(first_dates.items(), key=lambda x: x[1])
        last_dates_sorted = sorted(last_dates.items(), key=lambda x: x[1])
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

    @property
    def symbols(self):
        symbols = [default_ticker] if not self.__symbols else self.__symbols
        if not isinstance(symbols, list):
            raise ValueError('Symbols should be a list of string values.')
        return symbols

    @property
    def tickers(self):
        return self.__tickers

    @property
    def currency(self):
        return self.__currency

    @property
    def wealth_indexes(self) -> pd.DataFrame:
        """
        Wealth index time series for the assets and accumulated inflation.
        Wealth index is obtained from the accumulated return multiplicated by the initial investments (1000).
        """
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.ror
        return Frame.get_wealth_indexes(df)

    @property
    def risk_monthly(self) -> pd.Series:
        """
        Takes assets returns DataFrame and calculates monthly risks (std) for each asset.
        """
        return self.ror.std()

    @property
    def risk_annual(self) -> pd.Series:
        """
        Takes assets returns DataFrame and calculates annulized risks (std) for each asset.
        """
        risk = self.ror.std()
        mean_return = self.ror.mean()
        return Float.annualize_risk(risk, mean_return)

    @property
    def semideviation_monthly(self) -> pd.Series:
        """
        Returns semideviation monthly values for each asset (full period).
        """
        return Frame.get_semideviation(self.ror)

    @property
    def semideviation_annual(self) -> float:
        """
        Returns semideviation annual values for each asset (full period).
        """
        return Frame.get_semideviation(self.returns_ts) * 12 ** 0.5

    def get_var_historic(self, level: int = 5) -> pd.Series:
        """
        Calculates historic VAR for the assets (full period).
        VAR levels could be set by level attribute (integer).
        """
        return Frame.get_var_historic(self.ror, level)

    def get_cvar_historic(self, level: int = 5) -> pd.Series:
        """
        Calculates historic CVAR for the assets (full period).
        CVAR levels could be set by level attribute (integer).
        """
        return Frame.get_cvar_historic(self.ror, level)

    @property
    def drawdowns(self) -> pd.DataFrame:
        """
        Calculates drawdowns time series for the assets.
        """
        return Frame.get_drawdowns(self.ror)

    def get_cagr(self, period: Union[str, int, None] = None) -> pd.Series:
        """
        Calculates Compound Annual Growth Rate (CAGR) for a given period:
        None: full time
        'YTD': Year To Date compound rate of return (formally not a CAGR)
        Integer: several years
        """
        if hasattr(self, 'inflation'):
            df: pd.DataFrame = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.ror
        dt0 = self.last_date

        if not period:
            cagr = Frame.get_cagr(df)
        elif period == 'YTD':
            year = dt0.year
            cagr = (df[str(year):] + 1.).prod() - 1.
        elif isinstance(period, int):
            dt = Date.subtract_years(dt0, period)
            if dt >= self.first_date:
                cagr = Frame.get_cagr(df[dt:])
            else:
                row = {x: None for x in df.columns}
                cagr = pd.Series(row)
        else:
            raise ValueError(f'{period} is not a valid value for period')
        return cagr

    @property
    def annual_return_ts(self) -> pd.DataFrame:
        """
        Calculates annual rate of return time series for the assets.
        """
        return Frame.get_annual_return_ts_from_monthly(self.ror)

    def describe(self, years: tuple = (1, 5, 10), tickers: bool = True) -> pd.DataFrame:
        """
        Generate descriptive statistics for a given list of tickers.
        Statistics includes:
        - YTD compound return
        - CAGR for a given list of periods
        - Dividend yield - yield for last 12 months (LTM)
        - risk (std) for a full period
        - CVAR for a full period
        - max drawdowns (and dates) for a full period
        - inception date - first date available for each asset
        - last asset date - available for each asset date
        - last data data - common for all assets data (may be set by last_date manually)
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.ror
        # YTD return
        ytd_return = self.get_cagr(period='YTD')
        row = ytd_return.to_dict()
        row.update({'period': 'YTD'})
        row.update({'property': 'Compound return'})
        description = description.append(row, ignore_index=True)
        # CAGR for a list of periods
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
        Calculates mean return (arithmetic mean) for the assets.
        """
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            df = self.ror
        mean: pd.Series = df.mean()
        return Float.annualize_return(mean)

    @property
    def real_mean_return(self) -> pd.Series:
        """
        Calculates real mean return (arithmetic mean) for the assets.
        """
        if hasattr(self, 'inflation'):
            df = pd.concat([self.ror, self.inflation_ts], axis=1, join='inner', copy='false')
        else:
            raise Exception('Real Return is not defined. Set inflation=True to calculate.')
        infl_mean = Float.annualize_return(self.inflation_ts.values.mean())
        ror_mean = Float.annualize_return(df.loc[:, self.symbols].mean())
        return (1. + ror_mean) / (1. + infl_mean) - 1.

    def _get_asset_dividends(self, tick, remove_forecast=True) -> pd.Series:
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
        Dividend yield (LTM) time series monthly.
        Calculates yield assuming original asset currency (not adjusting to AssetList currency).
        Forecast dividends are removed.
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
        Time series of dividends for a calendar year.
        """
        return self._get_dividends().resample('Y').sum()

    @property
    def dividend_growing_years(self) -> pd.DataFrame:
        """
        Returns the number of growing dividend years for each asset.
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
        Returns the number of years of consecutive dividend payments.
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
        Calculates geometric mean of dividends growth rate time series for a certain period.
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

    def kstest(self, distr: str = 'norm') -> dict:
        """
        Perform Kolmogorov-Smirnov test for goodness of fit the asset returns to a given distribution.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
            Low statistic numbers correspond to normal distribution.
        """
        return Frame.kstest_dataframe(self.ror, distr=distr)
