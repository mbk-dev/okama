from typing import Union, Optional, List, Dict, Tuple, Any

import pandas as pd
import numpy as np

from .common.validators import validate_integer
from .macro import Inflation
from .common.helpers import Float, Frame, Date, Index
from .settings import default_ticker, PeriodLength, _MONTHS_PER_YEAR
from .api.data_queries import QueryData
from .api.namespaces import get_assets_namespaces


class Asset:
    """
    A financial asset, that could be used in a list of assets or in portfolio.

    Parameters
    ----------
    symbol: str, default "SPY.US"
        Symbol is an asset ticker with namespace after dot. The default value is "SPY.US" (SPDR S&P 500 ETF Trust).

    Examples
    --------
    >>> asset = ok.Asset()
    >>> asset
    symbol                           SPY.US
    name             SPDR S&P 500 ETF Trust
    country                             USA
    exchange                      NYSE ARCA
    currency                            USD
    type                                ETF
    first date                      1993-02
    last date                       2021-03
    period length                      28.1
    dtype: object

    An Asset object could be easy created whithout specifying a symbol Asset() using the default symbol.
    """

    def __init__(self, symbol: str = default_ticker):
        if symbol is None or len(str(symbol).strip()) == 0:
            raise ValueError("Symbol can not be empty")
        self._symbol = str(symbol).strip()
        self._check_namespace()
        self._get_symbol_data(symbol)
        self.ror: pd.Series = QueryData.get_ror(symbol)
        self.first_date: pd.Timestamp = self.ror.index[0].to_timestamp()
        self.last_date: pd.Timestamp = self.ror.index[-1].to_timestamp()
        self.period_length: float = round(
            (self.last_date - self.first_date) / np.timedelta64(365, "D"), ndigits=1
        )

    def __repr__(self):
        dic = {
            "symbol": self.symbol,
            "name": self.name,
            "country": self.country,
            "exchange": self.exchange,
            "currency": self.currency,
            "type": self.type,
            "first date": self.first_date.strftime("%Y-%m"),
            "last date": self.last_date.strftime("%Y-%m"),
            "period length": "{:.2f}".format(self.period_length),
        }
        return repr(pd.Series(dic))

    def _check_namespace(self):
        namespace = self._symbol.split(".", 1)[-1]
        allowed_namespaces = get_assets_namespaces()
        if namespace not in allowed_namespaces:
            raise ValueError(
                f"{namespace} is not in allowed assets namespaces: {allowed_namespaces}"
            )

    @property
    def symbol(self) -> str:
        """
        Return a symbol of the asset.

        Returns
        -------
        str
        """
        return self._symbol

    def _get_symbol_data(self, symbol) -> None:
        x = QueryData.get_symbol_info(symbol)
        self.ticker: str = x["code"]
        self.name: str = x["name"]
        self.country: str = x["country"]
        self.exchange: str = x["exchange"]
        self.currency: str = x["currency"]
        self.type: str = x["type"]
        self.inflation: str = f"{self.currency}.INFL"

    @property
    def price(self) -> Optional[float]:
        """
        Return live price of an asset.

        Live price is delayed (15-20 minutes).
        For certain namespaces (FX, INDX, PIF etc.) live price is not supported.

        Returns
        -------
        float, None
            Live price of the asset. Returns None if not defined.
        """
        return QueryData.get_live_price(self.symbol)

    @property
    def dividends(self) -> pd.Series:
        """
        Return dividends time series historical daily data.

        Returns
        -------
        Series
            Time series of dividends historical data (daily).

        Examples
        --------
        >>> x = ok.Asset('VNQ.US')
        >>> x.dividends
                Date
        2004-12-22    1.2700
        2005-03-24    0.6140
        2005-06-27    0.6440
        2005-09-26    0.6760
                       ...
        2020-06-25    0.7590
        2020-09-25    0.5900
        2020-12-24    1.3380
        2021-03-25    0.5264
        Freq: D, Name: VNQ.US, Length: 66, dtype: float64
        """
        div = QueryData.get_dividends(self.symbol)
        if div.empty:
            # Zero time series for assets where dividend yield is not defined.
            index = pd.date_range(
                start=self.first_date, end=self.last_date, freq="MS", closed=None
            )
            period = index.to_period("D")
            div = pd.Series(data=0, index=period)
            div.rename(self.symbol, inplace=True)
        return div

    @property
    def nav_ts(self) -> Optional[pd.Series]:
        """
        Return NAV time series (monthly) for mutual funds.
        """
        if self.exchange == "PIF":
            return QueryData.get_nav(self.symbol)
        return np.nan


class AssetList:
    """
    The list of financial assets implementation.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        *,
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
        ccy: str = "USD",
        inflation: bool = True,
    ):
        self.__symbols = symbols
        self.__tickers: List[str] = [x.split(".", 1)[0] for x in self.symbols]
        self.__currency: Asset = Asset(symbol=f"{ccy}.FX")
        self.__make_asset_list(self.symbols)
        if inflation:
            self.inflation: str = f"{ccy}.INFL"
            self._inflation_instance: Inflation = Inflation(
                self.inflation, self.first_date, self.last_date
            )
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
        self.ror = self.ror[self.first_date :]
        if last_date:
            self.last_date = min(self.last_date, pd.to_datetime(last_date))
        self.ror: pd.DataFrame = self.ror[self.first_date: self.last_date]
        self.period_length: float = round(
            (self.last_date - self.first_date) / np.timedelta64(365, "D"), ndigits=1
        )
        self.pl = PeriodLength(
            self.ror.shape[0] // _MONTHS_PER_YEAR, self.ror.shape[0] % _MONTHS_PER_YEAR
        )
        self._pl_txt = f"{self.pl.years} years, {self.pl.months} months"
        self._dividend_yield: pd.DataFrame = pd.DataFrame(dtype=float)
        self._dividends_ts: pd.DataFrame = pd.DataFrame(dtype=float)

    def __repr__(self):
        dic = {
            "symbols": self.symbols,
            "currency": self.currency.ticker,
            "first date": self.first_date.strftime("%Y-%m"),
            "last_date": self.last_date.strftime("%Y-%m"),
            "period length": self._pl_txt,
            "inflation": self.inflation if hasattr(self, "inflation") else "None",
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
            if i == 0:  # required to use pd.concat below (df should not be empty).
                if asset.currency == self.currency.name:
                    df = asset.ror
                else:
                    df = self._set_currency(
                        returns=asset.ror, asset_currency=asset.currency
                    )
            else:
                if asset.currency == self.currency.name:
                    new = asset.ror
                else:
                    new = self._set_currency(
                        returns=asset.ror, asset_currency=asset.currency
                    )
                df = pd.concat([df, new], axis=1, join="inner", copy="false")
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
        self.names = names
        currencies.update({"asset list": self.currency.currency})
        self.currencies: Dict[str, str] = currencies
        self.assets_first_dates: Dict[str, pd.Timestamp] = dict(first_dates_sorted)
        self.assets_last_dates: Dict[str, pd.Timestamp] = dict(last_dates_sorted)
        if isinstance(
            df, pd.Series
        ):  # required to convert Series to DataFrame for single asset list
            df = df.to_frame()
        self.ror = df

    def _set_currency(self, returns: pd.Series, asset_currency: str) -> pd.Series:
        """
        Set return to a certain currency.
        """
        currency = Asset(symbol=f"{asset_currency}{self.currency.name}.FX")
        asset_mult = returns + 1.0
        currency_mult = currency.ror + 1.0
        # join dataframes to have the same Time Series Index
        df = pd.concat([asset_mult, currency_mult], axis=1, join="inner", copy="false")
        currency_mult = df.iloc[:, -1]
        asset_mult = df.iloc[:, 0]
        x = asset_mult * currency_mult - 1.0
        x.rename(returns.name, inplace=True)
        return x

    def _add_inflation(self) -> pd.DataFrame:
        """
        Add inflation column to returns DataFrame.
        """
        if hasattr(self, "inflation"):
            return pd.concat(
                [self.ror, self.inflation_ts], axis=1, join="inner", copy="false"
            )
        else:
            return self.ror

    def _remove_inflation(self, time_frame: int) -> pd.DataFrame:
        """
        Remove inflation column from rolling returns if exists.
        """
        if hasattr(self, "inflation"):
            return self.get_rolling_cumulative_return(window=time_frame).drop(
                columns=[self.inflation]
            )
        else:
            return self.get_rolling_cumulative_return(window=time_frame)

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
        symbols = [default_ticker] if not self.__symbols else self.__symbols
        if not isinstance(symbols, list):
            raise ValueError("Symbols must be a list of string values.")
        return symbols

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
        return self.__tickers

    @property
    def currency(self) -> Asset:
        """
        Return the base currency of the Asset List.

        Such properties as rate of return and risk are adjusted to the base currency.

        Returns
        -------
        okama.Asset
            Base currency of the Asset List in form of okama.Asset class.
        """
        return self.__currency

    @property
    def wealth_indexes(self) -> pd.DataFrame:
        """
        Calculate wealth index time series for the assets and accumulated inflation.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of each asset over
        historical time period. Accumulated inflation time series is added if `inflation=True` in the AssetList.

        Wealth index is obtained from the accumulated return multiplicated by the initial investments.
        That is: 1000 * (Acc_Return + 1)
        Initial investments are taken as 1000 units of the AssetList base currency.

        Returns
        -------
        DataFrame
            Time series of wealth index values for each asset and accumulated inflation.
        """
        df = self._add_inflation()
        return Frame.get_wealth_indexes(df)

    @property
    def risk_monthly(self) -> pd.Series:
        """
        Calculate monthly risks (standard deviation) for each asset.

        Monthly risk of the asset is a standard deviation of the rate of return time series.
        Standard deviation (sigma σ) is normalized by N-1.

        Returns
        -------
        Series
            Monthly risk (standard deviation) values for each asset in form of Series.

        See Also
        --------
        risk_annual : Calculate annualized risks.
        semideviation_monthly : Calculate semideviation monthly values.
        semideviation_annual : Calculate semideviation annualized values.
        get_var_historic : Calculate historic Value at Risk (VaR).
        get_cvar_historic : Calculate historic Conditional Value at Risk (CVaR).
        drawdowns : Calculate drawdowns.

        Examples
        --------
        >>> al = ok.AssetList(['GC.COMM', 'SHV.US'], ccy='USD', last_date='2021-01')
        >>> al.risk_monthly
        GC.COMM    0.050864
        SHV.US     0.001419
        dtype: float64
        """
        return self.ror.std()

    @property
    def risk_annual(self) -> pd.Series:
        """
        Calculate annualized risks (standard deviation) for each asset.

        Returns
        -------
        Series
            Annualized risk (standard deviation) values for each asset in form of Series.
        """
        risk = self.ror.std()
        mean_return = self.ror.mean()
        return Float.annualize_risk(risk, mean_return)

    @property
    def semideviation_monthly(self) -> pd.Series:
        """
        Calculate semideviation monthly values for each asset.

        Returns
        -------
        Series
            Monthly semideviation values for each asset in form of Series.
        """
        return Frame.get_semideviation(self.ror)

    @property
    def semideviation_annual(self) -> pd.Series:
        """
        Return semideviation annualized values for each asset.

        Returns
        -------
        Series
            Annualized semideviation values for each asset in form of Series.
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
        time_frame : int, default 12
            Time period size in months
        level : int, default 5
            Confidence level in percents to calculate the VaR. Default value is 5%.
        Returns
        -------
        Series
            VaR values for each asset in form of Series.

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
        time_frame : int, default 12
            Time period size in months
        level : int, default 5
            Confidence level in percents to calculate the VaR. Default value is 5%.

        Returns
        -------
        Series
            CVaR values for each asset in form of Series.

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

        Returns
        -------
        DataFrame
            Time series of drawdowns.
        """
        return Frame.get_drawdowns(self.ror)

    def get_cagr(self, period: Optional[int] = None, real: bool = False) -> pd.Series:
        """
        Calculate assets Compound Annual Growth Rate (CAGR) for a given trailing period.

        Compound annual growth rate (CAGR) is the rate of return that would be required for an investment to grow from
        its initial to its final value, assuming all incomes were reinvested.

        Inflation adjusted annualized returns (real CAGR) are shown with `real=True` option.

        Annual inflation value is calculated for the same period if inflation=True in the AssetList.
        CAGR is not defined for periods less than 1 year.

        Parameters
        ----------
        period: int, optional
            CAGR trailing period in years. None for the full time CAGR.
        real: bool, default False
            CAGR is adjusted for inflation (real CAGR) if True.
            AssetList should be initiated with Inflation=True for real CAGR.

        Returns
        -------
        Series
            CAGR values for each asset and annualized inflation (optional).

        Examples
        --------
        >>> x = ok.AssetList()
        >>> x.get_cagr(period=5)
        SPY.US    0.1510
        USD.INFL   0.0195
        dtype: float64

        To get inflation adjusted return (real annualized return) add `real=True` option:
        >>> x = ok.AssetList(['EURUSD.FX', 'CNYUSD.FX'], inflation=True)
        >>> x.get_cagr(period=5, real=True)
        EURUSD.FX    0.000439
        CNYUSD.FX   -0.017922
        dtype: float64
        """
        df = self._add_inflation()
        dt0 = self.last_date
        if period is None:
            dt = self.first_date
        else:
            self._validate_period(period)
            dt = Date.subtract_years(dt0, period)
        cagr = Frame.get_cagr(df[dt:])
        if real:
            if not hasattr(self, "inflation"):
                raise Exception(
                    "Real CAGR is not defined. Set inflation=True in AssetList to calculate it."
                )
            mean_inflation = Frame.get_cagr(self.inflation_ts[dt:])
            cagr = (1. + cagr) / (1. + mean_inflation) - 1.
            cagr.drop(self.inflation, inplace=True)
        return cagr

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
        validate_integer("period", period, min_value=0, inclusive=False)
        if period > self.pl.years:
            raise ValueError(
                f"'period' ({period}) is beyond historical data range ({self.period_length})."
            )

    def _make_real_return_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate real monthly return time series.

        Rate of return monthly data is adjusted for inflation.
        """
        if not hasattr(self, "inflation"):
            raise Exception(
                "Real return is not defined. Set inflation=True in AssetList to calculate it."
            )
        df = (1. + df).divide(1. + self.inflation_ts, axis=0) - 1.
        df.drop(columns=[self.inflation], inplace=True)
        return df

    def get_rolling_cagr(self, window: int = 12, real: bool = False) -> pd.DataFrame:
        """
        Calculate rolling CAGR (Compound Annual Growth Rate) for each asset.

        Parameters
        ----------
        window : int, default 12
            Size of the moving window in months. Window size should be at least 12 months for CAGR.
        real: bool, default False
            CAGR is adjusted for inflation (real CAGR) if True.
            AssetList should be initiated with Inflation=True for real CAGR.

        Returns
        -------
        DataFrame
            Time series of rolling CAGR.

        Examples
        --------
        Get inflation adjusted rolling return (real annualized return) win 5 years window:
        >>> x = ok.AssetList(['DXET.XETR', 'DBXN.XETR'], ccy='EUR', inflation=True)
        >>> x.get_rolling_cagr(window=5*12, real=True)
                         DXET.XETR  DBXN.XETR
        2013-09   0.012148   0.034538
        2013-10   0.058834   0.034235
        2013-11   0.072305   0.027890
        2013-12   0.056456   0.022916
                    ...        ...
        2020-12   0.038441   0.020781
        2021-01   0.045849   0.012216
        2021-02   0.062271   0.006188
        2021-03   0.074446   0.006124
        """
        df = self._add_inflation()
        if real:
            df = self._make_real_return_time_series(df)
        return Frame.get_rolling_fn(df, window=window, fn=Frame.get_cagr)

    def get_cumulative_return(self, period: Union[str, int, None] = None, real: bool = False) -> pd.Series:
        """
        Calculate cumulative return over a given trailing period for each asset.

        The cumulative return is the total change in the asset price during the investment period.

        Inflation adjusted cumulative returns (real cumulative returns) are shown with `real=True` option.
        Annual inflation data is calculated for the same period if `inflation=True` in the AssetList.

        Parameters
        ----------
        period: str, int or None, default None
            Trailing period in years. Period should be more then 0.
            None - full time cumulative return.
            'YTD' - (Year To Date) period of time beginning the first day of the calendar year up to the last month.
        real: bool, default False
            Cumulative return is adjusted for inflation (real cumulative return) if True.
            AssetList should be initiated with `Inflation=True` for real cumulative return.

        Returns
        -------
        Series
            Cumulative return values for each asset and cumulative inflation (if inflation=True in AssetList).

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

        if period is None:
            dt = self.first_date
        elif str(period).lower() == "ytd":
            year = dt0.year
            dt = str(year)
        else:
            self._validate_period(period)
            dt = Date.subtract_years(dt0, period)

        cr = Frame.get_cumulative_return(df[dt:])
        if real:
            if not hasattr(self, "inflation"):
                raise Exception(
                    "Real cumulative return is not defined (no inflation information is available)."
                    "Set inflation=True in AssetList to calculate it."
                )
            cumulative_inflation = Frame.get_cumulative_return(self.inflation_ts[dt:])
            cr = (1. + cr) / (1. + cumulative_inflation) - 1.
            cr.drop(self.inflation, inplace=True)
        return cr

    def get_rolling_cumulative_return(self, window: int = 12, real: bool = False) -> pd.DataFrame:
        """
        Calculate rolling cumulative return for each asset.

        The cumulative return is the total change in the asset price.

        Parameters
        ----------
        window : int, default 12
            Size of the moving window in months.
        real: bool, default False
            Cumulative return is adjusted for inflation (real cumulative return) if True.
            AssetList should be initiated with `Inflation=True` for real cumulative return.

        Returns
        -------
        DataFrame
            Time series of rolling cumulative return.
        """
        df = self._add_inflation()
        if real:
            df = self._make_real_return_time_series(df)
        return Frame.get_rolling_fn(
            df, window=window, fn=Frame.get_cumulative_return, window_below_year=True
        )

    @property
    def annual_return_ts(self) -> pd.DataFrame:
        """
        Calculate annual rate of return time series for each asset.

        Rate of return is calculated for each calendar year.

        Returns
        -------
        DataFrame
            Calendar annual rate of return time series.
        """
        return Frame.get_annual_return_ts_from_monthly(self.ror)

    def describe(
        self, years: Tuple[int, ...] = (1, 5, 10), tickers: bool = True
    ) -> pd.DataFrame:
        """
        Generate descriptive statistics for a list of assets.

        Statistics includes:
        - YTD (Year To date) compound return
        - CAGR for a given list of periods
        - Dividend yield - yield for last 12 months (LTM)

        Risk metrics (full period):
        - risk (standard deviation)
        - CVAR
        - max drawdowns (and dates of the drawdowns)

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
            Table of descriptive statistics for a list of assets.

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
        ytd_return = self.get_cumulative_return(period="YTD")
        row = ytd_return.to_dict()
        row.update(period="YTD", property="Compound return")
        description = description.append(row, ignore_index=True)
        # CAGR for a list of periods
        if self.pl.years >= 1:
            for i in years:
                dt = Date.subtract_years(dt0, i)
                if dt >= self.first_date:
                    row = self.get_cagr(period=i).to_dict()
                else:
                    row = {x: None for x in df.columns}
                row.update(period=f"{i} years", property="CAGR")
                description = description.append(row, ignore_index=True)
            # CAGR for full period
            row = self.get_cagr(period=None).to_dict()
            row.update(period=self._pl_txt, property="CAGR")
            description = description.append(row, ignore_index=True)
            # Dividend Yield
            row = self.dividend_yield.iloc[-1].to_dict()
            row.update(period="LTM", property="Dividend yield")
            description = description.append(row, ignore_index=True)
        # risk for full period
        row = self.risk_annual.to_dict()
        row.update(period=self._pl_txt, property="Risk")
        description = description.append(row, ignore_index=True)
        # CVAR
        if self.pl.years >= 1:
            row = self.get_cvar_historic().to_dict()
            row.update(period=self._pl_txt, property="CVAR")
            description = description.append(row, ignore_index=True)
        # max drawdowns
        row = self.drawdowns.min().to_dict()
        row.update(period=self._pl_txt, property="Max drawdowns")
        description = description.append(row, ignore_index=True)
        # max drawdowns dates
        row = self.drawdowns.idxmin().to_dict()
        row.update(period=self._pl_txt, property="Max drawdowns dates")
        description = description.append(row, ignore_index=True)
        # inception dates
        row = {}
        for ti in self.symbols:
            # short_ticker = ti.split(".", 1)[0]
            value = self.assets_first_dates[ti].strftime("%Y-%m")
            row.update({ti: value})
        row.update(period=None, property="Inception date")
        if hasattr(self, "inflation"):
            row.update({self.inflation: self.inflation_first_date.strftime("%Y-%m")})
        description = description.append(row, ignore_index=True)
        # last asset date
        row = {}
        for ti in self.symbols:
            # short_ticker = ti.split(".", 1)[0]
            value = self.assets_last_dates[ti].strftime("%Y-%m")
            row.update({ti: value})
        row.update(period=None, property="Last asset date")
        if hasattr(self, "inflation"):
            row.update({self.inflation: self.inflation_last_date.strftime("%Y-%m")})
        description = description.append(row, ignore_index=True)
        # last data date
        row = {x: self.last_date.strftime("%Y-%m") for x in df.columns}
        row.update(period=None, property="Common last data date")
        description = description.append(row, ignore_index=True)
        # rename columns
        if hasattr(self, "inflation"):
            description.rename(columns={self.inflation: "inflation"}, inplace=True)
            description = Frame.change_columns_order(
                description, ["inflation"], position="last"
            )
        description = Frame.change_columns_order(
            description, ["property", "period"], position="first"
        )
        if not tickers:
            for ti in self.symbols:
                # short_ticker = ti.split(".", 1)[0]
                description.rename(columns={ti: self.names[ti]}, inplace=True)
        return description

    @property
    def mean_return(self) -> pd.Series:
        """
        Calculate annualized mean return (arithmetic mean) for the rate of return time series (each asset).

        Mean return calculated for the full history period. Arithmetic mean for the inflation is also shown
        if there is an `inflation=True` option in AssetList.

        Returns
        -------
        Series
            Mean return value for each asset.

        Examples
        --------
        >>> x = ok.AssetList(['MCFTR.INDX', 'RGBITR.INDX'], ccy='RUB', inflation=True)
        >>> x.mean_return
        MCFTR.INDX     0.209090
        RGBITR.INDX    0.100133
        RUB.INFL       0.081363
        dtype: float64
        """
        df = self._add_inflation()
        mean = df.mean()
        return Float.annualize_return(mean)

    @property
    def real_mean_return(self) -> pd.Series:
        """
        Calculate annualized real mean return (arithmetic mean) for the rate of return time series (each assets).

        Real rate of return is adjusted for inflation. Real return is defined if
        there is an `inflation=True` option in AssetList.

        Returns
        -------
        Series
            Mean real return value for each asset.

        Examples
        --------
        >>> x = ok.AssetList(['MCFTR.INDX', 'RGBITR.INDX'], ccy='RUB', inflation=True)
        >>> x.real_mean_return
        MCFTR.INDX     0.118116
        RGBITR.INDX    0.017357
        dtype: float64
        """
        if not hasattr(self, "inflation"):
            raise Exception(
                "Real Return is not defined. Set inflation=True to calculate."
            )
        df = pd.concat(
            [self.ror, self.inflation_ts], axis=1, join="inner", copy="false"
        )
        infl_mean = Float.annualize_return(self.inflation_ts.values.mean())
        ror_mean = Float.annualize_return(df.loc[:, self.symbols].mean())
        return (1. + ror_mean) / (1. + infl_mean) - 1.

    def _get_asset_dividends(self, tick: str, remove_forecast: bool = True) -> pd.Series:
        """
        Get dividend time series for a single symbol.
        """
        first_period = pd.Period(self.first_date, freq="M")
        first_day = first_period.to_timestamp(how="Start")
        last_period = pd.Period(self.last_date, freq="M")
        last_day = last_period.to_timestamp(how="End")
        s = Asset(tick).dividends[
            first_day:last_day
        ]  # limit divs by first_day and last_day
        if remove_forecast:
            s = s[: pd.Period.now(freq="D")]
        # Create time series with zeros to pad the empty spaces in dividends time series
        index = pd.date_range(start=first_day, end=last_day, freq="D")
        period = index.to_period("D")
        pad_s = pd.Series(data=0, index=period)
        return s.add(pad_s, fill_value=0)

    def _get_dividends(self, remove_forecast=True) -> pd.DataFrame:
        """
        Get dividend time series for all assets.

        If `remove_forecast=True` all forecasted (future) data is removed from the time series.
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

        All yields are calculated in the original asset currency (not adjusting to AssetList base currency).
        Forecasted (future) dividends are removed.
        Zero value time series are created for assets without dividends.

        Returns
        -------
        DataFrame
            Time series of LTM dividend yield for each asset.

        Examples
        --------
        >>> x = ok.AssetList(['T.US', 'XOM.US'], first_date='1984-01', last_date='1994-12')
        >>> x.dividend_yield
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
            df = self._get_dividends(remove_forecast=True)
            for tick in self.symbols:
                # Get dividends time series
                div = df[tick]
                # Get close (not adjusted) values time series.
                # If the last_date month is current month live price of assets is used.
                if div.sum() != 0:
                    div_monthly = div.resample("M").sum()
                    price = QueryData.get_close(tick, period="M").loc[
                        self.first_date : self.last_date
                    ]
                else:
                    # skipping prices if no dividends
                    div_yield = div.asfreq(freq="M")
                    frame.update({tick: div_yield})
                    continue
                if price.index[-1] == pd.Period(pd.Timestamp.today(), freq="M"):
                    price.loc[
                        f"{pd.Timestamp.today().year}-{pd.Timestamp.today().month}"
                    ] = Asset(tick).price
                # Get dividend yield time series
                div_yield = pd.Series(dtype=float)
                div_monthly.index = div_monthly.index.to_timestamp()
                for date in price.index.to_timestamp(how="End"):
                    ltm_div = div_monthly[:date].last("12M").sum()
                    last_price = price.loc[:date].iloc[-1]
                    value = ltm_div / last_price
                    div_yield.at[date] = value
                div_yield.index = div_yield.index.to_period("M")
                # Currency adjusted yield
                # if self.currencies[tick] != self.currency.name:
                #     div_yield = self._set_currency(returns=div_yield, asset_currency=self.currencies[tick])
                frame.update({tick: div_yield})
            self._dividend_yield = pd.DataFrame(frame)
        return self._dividend_yield

    @property
    def dividends_annual(self) -> pd.DataFrame:
        """
        Return calendar year dividends sum time series for each asset.

        Returns
        -------
        DataFrame
            Annual dividends time series for each asset.
        """
        return self._get_dividends().resample("Y").sum()

    @property
    def dividend_growing_years(self) -> pd.DataFrame:
        """
        Return the number of years when the annual dividend was growing for each asset.

        Returns
        -------
        DataFrame
            Dividend growth length periods time series for each asset.

        Examples
        --------
        >>> x = ok.AssetList(['T.US', 'XOM.US'], first_date='1984-01', last_date='1994-12')
        >>> x.dividend_growing_years
                T.US  XOM.US
        1985     1       1
        1986     2       2
        1987     3       3
        1988     0       4
        1989     1       5
        1990     2       6
        1991     3       7
        1992     4       8
        1993     5       9
        1994     6      10
        """
        div_growth = self.dividends_annual.pct_change()[1:]
        df = pd.DataFrame()
        for name in div_growth:
            s = div_growth[name]
            s1 = s.where(s > 0).notnull().astype(int)
            s1_1 = s.where(s > 0).isnull().astype(int).cumsum()
            s2 = s1.groupby(s1_1).cumsum()
            df = pd.concat([df, s2], axis=1, copy="false")
        return df

    @property
    def dividend_paying_years(self) -> pd.DataFrame:
        """
        Return the number of years of consecutive dividend payments for each asset.

        Returns
        -------
        DataFrame
            Dividend payment period length time series for each asset.

        Examples
        --------
        >>> x = ok.AssetList(['T.US', 'XOM.US'], first_date='1984-01', last_date='1994-12')
        >>> x.dividend_paying_years
              T.US  XOM.US
        1984     1       1
        1985     2       2
        1986     3       3
        1987     4       4
        1988     5       5
        1989     6       6
        1990     7       7
        1991     8       8
        1992     9       9
        1993    10      10
        1994    11      11
        """
        div_annual = self.dividends_annual
        frame = pd.DataFrame()
        df = frame
        for name in div_annual:
            s = div_annual[name]
            s1 = s.where(s != 0).notnull().astype(int)
            s1_1 = s.where(s != 0).isnull().astype(int).cumsum()
            s2 = s1.groupby(s1_1).cumsum()
            df = pd.concat([df, s2], axis=1, copy="false")
        return df

    def get_dividend_mean_growth_rate(self, period=5) -> pd.Series:
        """
        Calculate geometric mean of dividends growth rate time series for a given trailing period.

        Parameters
        ----------
        period : int, default 5
            Growth rate trailing period in years. Period should be a positive integer
            and not exceed the available data period_length.

        Returns
        -------
        Series
            Dividend growth geometric mean values for each asset.

        Examples
        --------
        >>> x = ok.AssetList(['T.US', 'XOM.US'], first_date='1984-01', last_date='1994-12')
        >>> x.get_dividend_mean_growth_rate(period=3)
        T.US      0.020067
        XOM.US    0.024281
        dtype: float64
        """
        self._validate_period(period)
        growth_ts = self.dividends_annual.pct_change().iloc[
            1:-1
        ]  # Slice the last year for full dividends
        dt0 = self.last_date
        dt = Date.subtract_years(dt0, period)
        return ((growth_ts[dt:] + 1.0).prod()) ** (1 / period) - 1.0

    # index methods
    @property
    def tracking_difference(self) -> pd.DataFrame:
        """
        Return tracking difference for the rate of return of assets.


        Tracking difference is calculated by measuring the accumulated difference between the returns of a benchmark
        and those of the ETF replicating it (could be mutual funds, or other types of assets).
        
        Benchmark should be in the first position of the symbols list in AssetList parameters.

        Returns
        -------
        DataFrame
            Tracking diffirence time series for each asset.

        Examples
        --------
        >>> x = ok.AssetList(['SP500TR.INDX', 'SPY.US', 'VOO.US'], last_date='2021-01')
        >>> x.tracking_difference
                   SPY.US    VOO.US
        Date
        2011-01  0.000000  0.000000
        2011-02 -0.000004 -0.001143
        2011-03 -0.000322 -0.001566
        2011-04 -0.000967 -0.001824
        2011-05 -0.000847 -0.002239
                   ...       ...
        2020-09 -0.037189 -0.022919
        2020-10 -0.030695 -0.018732
        2020-11 -0.036266 -0.020783
        2020-12 -0.042560 -0.025097
        2021-01 -0.042493 -0.025209
        """
        accumulated_return = Frame.get_wealth_indexes(
            self.ror
        )  # we don't need inflation here
        return Index.tracking_difference(accumulated_return)

    @property
    def tracking_difference_annualized(self) -> pd.DataFrame:
        """
        Calculate annualized tracking difference time series for the rate of return of assets.

        Tracking difference is calculated by measuring the accumulated difference between the returns of a benchmark
        and those of the ETF replicating it (could be mutual funds, or other types of assets).

        Benchmark should be in the first position of the symbols list in AssetList parameters.

        Annual values are available for history periods of more than 12 months.
        Returns for less than 12 months can't be annualized According to the CFA
        Institute's Global Investment Performance Standards (GIPS).

        Returns
        -------
        DataFrame
            Annualized tracking diffirence time series for each asset.
        
        Examples
        --------
        >>> x = ok.AssetList(['SP500TR.INDX', 'SPY.US', 'VOO.US'], last_date='2021-01')
        >>> x.tracking_difference
                   SPY.US    VOO.US
        Date
        2011-12 -0.002198 -0.002230
        2012-01 -0.000615 -0.002245
        2012-02 -0.000413 -0.002539
        2012-03 -0.001021 -0.002359
                   ...       ...
        2020-10 -0.003079 -0.001889
        2020-11 -0.003599 -0.002076
        2020-12 -0.004177 -0.002482
        2021-01 -0.004136 -0.002472
        """
        return Index.tracking_difference_annualized(self.tracking_difference)

    @property
    def tracking_error(self) -> pd.DataFrame:
        """
        Calculate tracking error time series for the rate of return of assets.

        Tracking error is defined as the standard deviation of the difference between the returns of the asset
        and the returns of the benchmark.

        Benchmark should be in the first position of the symbols list in AssetList parameters.

        Returns
        -------
        DataFrame
            Tracking error time series for each asset.
        
        Examples
        --------
        >>> x = ok.AssetList(['SP500TR.INDX', 'SPY.US', 'VOO.US'], last_date='2021-01')
        >>> x.tracking_error
                   SPY.US    VOO.US
        Date
        2010-10  0.000346  0.001039
        2010-11  0.000346  0.003030
        2010-12  0.000283  0.005400
        2011-01  0.000735  0.005350
                   ...       ...
        2020-10  0.003132  0.003370
        2020-11  0.003127  0.003356
        2020-12  0.003144  0.003357
        2021-01  0.003132  0.003343
        """
        return Index.tracking_error(self.ror)

    @property
    def index_corr(self) -> pd.DataFrame:
        """
        Compute expanding correlation with the index (or benchmark) time series for the assets.

        Benchmark should be in the first position of the symbols list in AssetList parameters.
        There should be at least 12 months of historical data.

        Returns
        -------
        DataFrame
            Expanding correlation with the index (or benchmark) time series for each asset.

        Examples
        --------
        >>> sp = ok.AssetList(['SP500TR.INDX', 'VBMFX.US', 'GC.COMM', 'VNQ.US'])
        >>> sp.names
        {'SP500TR.INDX': 'S&P 500 (TR)',
        'VBMFX.US': 'VANGUARD TOTAL BOND MARKET INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold',
        'VNQ.US': 'Vanguard Real Estate Index Fund ETF Shares'}
        >>> sp.index_corr
                 VBMFX.US   GC.COMM    VNQ.US
        2005-10 -0.217992  0.103308  0.681394
        2005-11 -0.171918  0.213368  0.683557
        2005-12 -0.191054  0.183656  0.687335
        2006-01 -0.204574  0.250068  0.699323
                   ...       ...       ...
        2020-11 -0.004154  0.065746  0.721346
        2020-12 -0.006035  0.069420  0.721324
        2021-01 -0.002942  0.070801  0.721216
        2021-02 -0.007533  0.067011  0.721464
        """
        return Index.cov_cor(self.ror, fn="corr")

    def index_rolling_corr(self, window: int = 60) -> pd.DataFrame:
        """
        Compute rolling correlation with the index (or benchmark) time series for the assets.
        
        Index (benchmark) should be in the first position of the symbols list in AssetList parameters.
        There should be at least 12 months of historical data.

        Parameters
        ----------
        window : int, default 60
            Rolling window size in months. This is the number of observations used for calculating the statistic.
            
        Returns
        -------
        DataFrame
            Rolling correlation with the index (or benchmark) time series for each asset.

        Examples
        --------
        >>> sp = ok.AssetList(['SP500TR.INDX', 'VBMFX.US', 'GC.COMM', 'VNQ.US'])
        >>> sp.names
        {'SP500TR.INDX': 'S&P 500 (TR)',
        'VBMFX.US': 'VANGUARD TOTAL BOND MARKET INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold',
        'VNQ.US': 'Vanguard Real Estate Index Fund ETF Shares'}
        >>> sp.index_rolling_corr(window=24)
                 VBMFX.US   GC.COMM    VNQ.US
        2006-09 -0.072073  0.209741  0.639184
        2006-10 -0.053556  0.196464  0.657984
        2006-11  0.048231  0.173406  0.666584
        2006-12 -0.001431  0.227669  0.634478
                   ...       ...       ...
        2020-11 -0.038417  0.122855  0.837298
        2020-12  0.033282  0.204574  0.820935
        2021-01  0.046599  0.205193  0.816003
        2021-02  0.033039  0.181227  0.816178
        """
        return Index.rolling_cov_cor(self.ror, window=window, fn="corr")

    @property
    def index_beta(self) -> pd.DataFrame:
        """
        Compute beta coefficient time series for the assets.

        Beta coefficient is defined in Capital Asset Pricing Model (CAPM). It is a measure of how
        an individual asset moves (on average) when the benchmark increases or decreases. When beta is positive,
        the asset price tends to move in the same direction as the benchmark,
        and the magnitude of beta tells by how much.

        Index (benchmark) should be in the first position of the symbols list in AssetList parameters.
        There should be at least 12 months of historical data.

        Returns
        -------
        DataFrame
            Beta coefficient time series for each asset.

        See Also
        --------
        index_corr : Compute correlation with the index (or benchmark).
        index_rolling_corr : Compute rolling correlation with the index (or benchmark).
        index_beta : Compute beta coefficient.

        Examples
        --------
        >>> sp = ok.AssetList(['SP500TR.INDX', 'VBMFX.US', 'GC.COMM', 'VNQ.US'])
        >>> sp.names
        {'SP500TR.INDX': 'S&P 500 (TR)',
        'VBMFX.US': 'VANGUARD TOTAL BOND MARKET INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold',
        'VNQ.US': 'Vanguard Real Estate Index Fund ETF Shares'}
        >>> sp.index_beta
                 VBMFX.US   GC.COMM    VNQ.US
        2005-10 -0.541931  0.064489  0.346571
        2005-11 -0.450691  0.131065  0.364683
        2005-12 -0.490117  0.110731  0.366512
        2006-01 -0.531695  0.132016  0.359480
        2006-02 -0.540665  0.135381  0.360091
                   ...       ...       ...
        2020-10 -0.063057  0.069050  0.465525
        2020-11 -0.018408  0.055676  0.472042
        """
        return Index.beta(self.ror)

    # distributions
    @property
    def skewness(self) -> pd.DataFrame:
        """
        Compute expanding skewness of the return time series for each asset returns.

        Skewness is a measure of the asymmetry of the probability distribution
        of a real-valued random variable about its mean. The skewness value can be positive, zero, negative, 
        or undefined.

        For normally distributed returns, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.

        Returns
        -------
        Dataframe
            Expanding skewness time series for each asset.

        See Also
        --------
        skewness_rolling : Compute rolling skewness.
        kurtosis : Calculate expanding Fisher (normalized) kurtosis.
        kurtosis_rolling : Calculate rolling Fisher (normalized) kurtosis.
        jarque_bera : Perform Jarque-Bera test for normality.
        kstest : Perform Kolmogorov-Smirnov test for different types of distributions.

        Examples
        --------
        >>> al = ok.AssetList(['VFINX.US', 'GC.COMM'], last_date='2021-01')
        >>> al.names
        {'VFINX.US': 'VANGUARD 500 INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold'}
        >>> al.skewness
                 VFINX.US   GC.COMM
        1981-02 -0.537554  0.272718
        1981-03 -0.642592  0.128630
        1981-04 -0.489567  0.231292
        1981-05 -0.471067  0.219311
                   ...       ...
        2020-10 -0.629908  0.107989
        2020-11 -0.610480  0.111627
        2020-12 -0.613742  0.107515
        2021-01 -0.611421  0.110552
        """
        return Frame.skewness(self.ror)

    def skewness_rolling(self, window: int = 60) -> pd.DataFrame:
        """
        Compute rolling skewness of the return time series for each asset.

        Skewness is a measure of the asymmetry of the probability distribution
        of a real-valued random variable about its mean. The skewness value can be positive, zero, negative,
        or undefined.

        For normally distributed returns, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.

        Parameters
        ----------
        window : int, default 60
            Rolling window size in months. This is the number of observations used for calculating the statistic.
            The window size should be at least 12 months.

        Returns
        -------
        DataFrame
            Rolling skewness time series for each asset.

        See Also
        --------
        skewness : Compute skewness.
        kurtosis : Calculate expanding Fisher (normalized) kurtosis.
        kurtosis_rolling : Calculate rolling Fisher (normalized) kurtosis.
        jarque_bera : Perform Jarque-Bera test for normality.
        kstest : Perform Kolmogorov-Smirnov test for different types of distributions.

        Examples
        --------
        >>> al = ok.AssetList(['VFINX.US', 'GC.COMM'], last_date='2021-01')
        >>> al.names
        {'VFINX.US': 'VANGUARD 500 INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold'}
        >>> al.skewness_rolling(window=24)
                 VFINX.US   GC.COMM
        1982-01 -0.144778  0.303309
        1982-02 -0.049833  0.353829
        1982-03  0.173783  1.198266
        1982-04  0.176163  1.123462
                   ...       ...
        2020-10 -0.547946  0.181045
        2020-11 -0.473080  0.071605
        2020-12 -0.597739  0.065503
        2021-01 -0.480090  0.205303
        """
        return Frame.skewness_rolling(self.ror, window=window)

    @property
    def kurtosis(self) -> pd.DataFrame:
        """
        Calculate expanding Fisher (normalized) kurtosis of the return time series for each asset.

        Kurtosis is the fourth central moment divided by the square of the variance. It is a measure of the "tailedness"
        of the probability distribution of a real-valued random variable.

        Kurtosis should be close to zero for normal distribution.

        Returns
        -------
        DataFrame
            Expanding kurtosis time series for each asset.

        See Also
        --------
        skewness : Compute skewness.
        skewness_rolling : Compute rolling skewness.
        kurtosis_rolling : Calculate rolling Fisher (normalized) kurtosis.
        jarque_bera : Perform Jarque-Bera test for normality.
        kstest : Perform Kolmogorov-Smirnov test for different types of distributions.

        Examples
        --------
        >>> al = ok.AssetList(['GC.COMM', 'FNER.INDX'], first_date='2000-01', last_date='2021-01')
        >>> al.names
        {'GC.COMM': 'Gold',
        'FNER.INDX': 'FTSE NAREIT All Equity REITs'}
        >>> al.kurtosis
                  GC.COMM  FNER.INDX
        date
        2001-01  0.141457  -0.424810
        2001-02  0.255112  -0.486316
        2001-03  0.264453  -0.275661
        2001-04 -0.102208  -0.107295
                   ...        ...
        2020-10  0.705098   7.485606
        2020-11  0.679793   7.400417
        2020-12  0.663579   7.439888
        2021-01  0.664566   7.475272
        """
        return Frame.kurtosis(self.ror)

    def kurtosis_rolling(self, window: int = 60):
        """
        Calculate rolling Fisher (normalized) kurtosis of the return time series for each asset.

        Kurtosis is the fourth central moment divided by the square of the variance. It is a measure of the "tailedness"
        of the probability distribution of a real-valued random variable.

        Kurtosis should be close to zero for normal distribution.

        Parameters
        ----------
        window : int, default 60
            Rolling window size in months. This is the number of observations used for calculating the statistic.
            The window size should be at least 12 months.

        Returns
        -------
        DataFrame
            Rolling kurtosis time series for each asset.

        See Also
        --------
        skewness : Compute skewness.
        skewness_rolling : Compute rolling skewness.
        kurtosis : Calculate expanding Fisher (normalized) kurtosis.
        jarque_bera : Perform Jarque-Bera test for normality.
        kstest : Perform Kolmogorov-Smirnov test for different types of distributions.

        Examples
        --------
        >>> al = ok.AssetList(['GC.COMM', 'FNER.INDX'], first_date='2000-01', last_date='2021-01')
        >>> al.names
        {'GC.COMM': 'Gold',
        'FNER.INDX': 'FTSE NAREIT All Equity REITs'}
        >>> al.kurtosis_rolling(window=12)
                  GC.COMM  FNER.INDX
        date
        2000-12 -0.044261  -0.640834
        2001-01 -0.034628  -0.571309
        2001-02  1.089403  -0.639850
        2001-03  1.560623  -0.601771
                   ...        ...
        2020-10 -0.153749   3.867389
        2020-11 -0.262682   2.854431
        2020-12 -0.695676   2.865679
        2021-01 -0.754352   2.801018
        """
        return Frame.kurtosis_rolling(self.ror, window=window)

    @property
    def jarque_bera(self) -> pd.DataFrame:
        """
        Perform Jarque-Bera test for normality of assets returns historical data.

        Jarque-Bera test shows whether the returns have the skewness and kurtosis
        matching a normal distribution (null hypothesis or H0).

        Returns
        -------
        DataFrame
            Returns test statistic and the p-value for the hypothesis test.
            large Jarque-Bera statistics and tiny p-value indicate that null hypothesis (H0) is rejected and
            the time series are not normally distributed.
            Low statistic numbers correspond to normal distribution.
            
        See Also
        --------
        skewness : Compute skewness.
        skewness_rolling : Compute rolling skewness.
        kurtosis : Calculate expanding Fisher (normalized) kurtosis.
        kurtosis_rolling : Calculate rolling Fisher (normalized) kurtosis.
        kstest : Perform Kolmogorov-Smirnov test for different types of distributions.

        Examples
        --------
        >>> al = ok.AssetList(['GC.COMM', 'FNER.INDX'], first_date='2000-01', last_date='2021-01')
        >>> al.names
        {'GC.COMM': 'Gold',
        'FNER.INDX': 'FTSE NAREIT All Equity REITs'}
        >>> al.jarque_bera
                    GC.COMM   FNER.INDX
        statistic  4.507287  593.633047
        p-value    0.105016    0.000000

        Gold return time series (GC.COMM) distribution have small p-values (H0 is not rejected).
        Null hypothesis (H0) is rejected for FTSE NAREIT Index (FNER.INDX) as Jarque-Bera test shows very small p-value
        and large statistic.
        """
        return Frame.jarque_bera_dataframe(self.ror)

    def kstest(self, distr: str = "norm") -> pd.DataFrame:
        """
        Perform Kolmogorov-Smirnov test for goodness of fit the asset returns to a given distribution.

        Kolmogorov-Smirnov is a test of the distribution of assets returns historical data against a
        given distribution. Under the null hypothesis (H0), the two distributions are identical.

        Parameters
        ----------
        distr : {'norm', 'lognorm'}, default 'norm'
            Type of distributions. Can be 'norm' - for normal distribution or 'lognorm' - for lognormal distribtion.

        Returns
        -------
        DataFrame
            Returns test statistic and the p-value for the hypothesis test.
            Large test statistics and tiny p-value indicate that null hypothesis (H0) is rejected.

        Examples
        --------
        >>> al = ok.AssetList(['EDV.US'], last_date='2021-01')
        >>> al.kstest(distr='lognorm')
                     EDV.US
        p-value    0.402179
        statistic  0.070246

        H0 is not rejected for EDV ETF and it seems to have lognormal distribution.
        """
        return Frame.kstest_dataframe(self.ror, distr=distr)
