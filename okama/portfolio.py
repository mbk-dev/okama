from random import randint
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

from .common.helpers import Frame, Rebalance, Float, Date
from .common.make_asset_list import ListMaker
from .common.validators import validate_real
from .settings import _MONTHS_PER_YEAR


class Portfolio(ListMaker):
    """
    Implementation of investment portfolio.

    Investments portfolio is a type of financial asset.
    Arguments are similar to AssetList (weights are added), but different behavior.
    Works with monthly end of day historical rate of return data.

    The rebalancing is the action of bringing the portfolio that has deviated away
    from original target asset allocation back into line. After rebalancing the portfolio assets
    have weights set with Portfolio(weights=[...]).
    Different rebalancing periods are allowed for portfolio: 'month' (default), 'year' or 'none'.

    Parameters
    ----------
    rebalancing_period : {"month", "year", "none"}, default "month"
        Portfolio rebalancing periods. 'none' is for not rebalanced portfolio.

    # TODO: Finish description.
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        *,
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
        ccy: str = "USD",
        inflation: bool = True,
        weights: Optional[List[float]] = None,
        rebalancing_period: str = "month",
        symbol: str = None,
    ):
        super().__init__(
            assets,
            first_date=first_date,
            last_date=last_date,
            ccy=ccy,
            inflation=inflation,
        )
        self._weights = None
        self.weights = weights
        self.assets_weights = dict(zip(self.symbols, self.weights))
        self._rebalancing_period = None
        self.rebalancing_period = rebalancing_period
        self._symbol = symbol or f'portfolio_{randint(1000, 9999)}.PF'

    def __repr__(self):
        dic = {
            "symbol": self.symbol,
            "assets": self.symbols,
            "weights": self.weights,
            "rebalancing_period": self.rebalancing_period,
            "currency": self.currency,
            "inflation": self.inflation if hasattr(self, "inflation") else "None",
            "first_date": self.first_date.strftime("%Y-%m"),
            "last_date": self.last_date.strftime("%Y-%m"),
            "period_length": self._pl_txt,
        }
        return repr(pd.Series(dic))

    def _add_inflation(self):
        if hasattr(self, "inflation"):
            return pd.concat(
                [self.ror, self.inflation_ts], axis=1, join="inner", copy="false"
            )
        else:
            return self.ror

    @property
    def weights(self) -> Union[list, tuple]:
        """
        Get or set assets weights in portfolio.

        If not defined equal weights are used for each asset.

        Weights must be a list (or tuple) of float values.

        Returns
        -------
            Values for the weights of assets in portfolio.

        Examples
        --------
        >>> x = ok.Portfolio(['SPY.US', 'BND.US'])
        >>> x.weights
        [0.5, 0.5]
        """
        return self._weights

    @weights.setter
    def weights(self, weights: Optional[List[float]]):
        if weights is None:
            # Equally weighted portfolio
            n = len(self.symbols)  # number of assets
            weights = list(np.repeat(1 / n, n))
        else:
            [validate_real("weight", weight) for weight in weights]
            Frame.weights_sum_is_one(weights)
            if len(weights) != len(self.symbols):
                raise ValueError(
                    f"Number of tickers ({len(self.symbols)}) should be equal "
                    f"to the weights number ({len(weights)})"
                )
        self._weights = weights

    @property
    def weights_ts(self) -> pd.DataFrame:
        """
        Calculate assets weights time series.

        Returns
        -------
        DataFrame
            Weights of assets time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US'], weights=[0.5, 0.5], rebalancing_period='none')
        >>> pf.weights
        [0.5, 0.5]
        >>> pf.weights_ts
                   SPY.US    AGG.US
        Date
        2003-10  0.515361  0.484639
        2003-11  0.517245  0.482755
        2003-12  0.527056  0.472944
                   ...       ...
        2021-02  0.731292  0.268708
        2021-03  0.742147  0.257853
        2021-04  0.750528  0.249472
        [211 rows x 2 columns]
        """
        if self.rebalancing_period != 'month':
            return Rebalance.assets_weights_ts(ror=self.assets_ror, period=self.rebalancing_period, weights=self.weights)
        values = np.tile(self.weights, (self.ror.shape[0], 1))
        return pd.DataFrame(values, index=self.ror.index, columns=self.symbols)

    @property
    def rebalancing_period(self) -> str:
        """
        Return rebalancing period of the portfolio.

        Rebalancing is the process by which an investor restores their portfolio to its target allocation
        by selling and buying assets. After rebalancing all the assets have original weights.

        Rebalancing period (rebalancing frequency) is predetermined time intervals when
        the investor rebalances the portfolio.

        Returns
        -------
        str
            Portfolio rebalancing period.
        """
        return self._rebalancing_period

    @rebalancing_period.setter
    def rebalancing_period(self, rebalancing_period: str):
        if rebalancing_period in {'none', 'month', 'year'}:
            self._rebalancing_period = rebalancing_period
        else:
            raise ValueError('rebalancing_period must be "year", "month" or "none"')

    @property
    def symbol(self) -> str:
        """
        Return a text symbol of portfolio.

        Symbols are similar to tickers but have a namespace information:

        * SPY.US is a symbol
        * SPY is a ticker

        Portfolios have '.PF' as a namespace.

        Returns
        -------
        str
            Text symbol of the portfolio.
        """
        return self._symbol

    @symbol.setter
    def symbol(self, text_symbol: str):
        if isinstance(text_symbol, str) and '.' in text_symbol:
            if " " in text_symbol:
                raise ValueError('portfolio text symbol should not have whitespace characters.')
            namespace = text_symbol.split(".", 1)[-1]
            if namespace == 'PF':
                self._symbol = text_symbol
            else:
                raise ValueError('portfolio symbol must end with ".PF"')
        else:
            raise ValueError('portfolio symbol must be a string ending with ".PF" namespace.')

    @property
    def name(self) -> str:
        """
        Return text name of portfolio.

        For portfolio name is equal to symbol.

        Returns
        -------
        str
            Text name of the portfolio.
        """
        return self.symbol

    @property
    def ror(self) -> pd.Series:
        """
        Calculate rate of return time series for portfolio.

        Returns
        -------
        Series
            Rate of return time series for portfolio.
        """
        if self.rebalancing_period == 'month':
            s = Frame.get_portfolio_return_ts(self.weights, self.assets_ror)
        else:
            s = Rebalance.return_ts(
                self.weights, self.assets_ror, period=self.rebalancing_period
            )
        return s.rename(self.symbol, inplace=True)

    @property
    def wealth_index(self) -> pd.DataFrame:
        """
        Calculate wealth index time series for the portfolio and accumulated inflation.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period. Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        Wealth index is obtained from the accumulated return multiplicated by the initial investments.
        That is: 1000 * (Acc_Return + 1)
        Initial investments are taken as 1000 units of the Portfolio base currency.

        Returns
        -------
            Time series of wealth index values for portfolio and accumulated inflation.

        Examples
        --------
        >>> x = ok.Portfolio(['SPY.US', 'BND.US'])
        >>> x.wealth_index
                    portfolio     USD.INFL
        2007-05  1000.000000  1000.000000
        2007-06  1004.034950  1008.011590
        2007-07   992.940364  1007.709187
        2007-08  1006.642941  1005.895310
                      ...          ...
        2020-12  2561.882476  1260.242835
        2021-01  2537.800781  1265.661880
        2021-02  2553.408256  1272.623020
        2021-03  2595.156481  1281.658643
        [167 rows x 2 columns]
        """
        df = self._add_inflation()
        df = Frame.get_wealth_indexes(df)
        df = self._make_df_if_series(df)
        return df

    def _make_df_if_series(self, ts):
        if isinstance(ts, pd.Series):  # should always return a DataFrame
            ts = ts.to_frame()
            ts.rename({1: self.symbol}, axis="columns", inplace=True)
        return ts

    @property
    def wealth_index_with_assets(self) -> pd.DataFrame:
        """
        Calculate wealth index time series for the portfolio, all assets and accumulated inflation.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period. Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        Wealth index is obtained from the accumulated return multiplicated by the initial investments.
        That is: 1000 * (Acc_Return + 1)
        Initial investments are taken as 1000 units of the Portfolio base currency.

        Returns
        -------
        DataFrame
            Time series of wealth index values for portfolio, each asset and accumulated inflation.

        Examples
        --------
        >>> pf = ok.Portfolio(['VOO.US', 'GLD.US'], weights=[0.8, 0.2])
        >>> pf.wealth_index_with_assets
                   portfolio       VOO.US       GLD.US     USD.INFL
        2010-10  1000.000000  1000.000000  1000.000000  1000.000000
        2010-11  1041.065584  1036.658420  1058.676480  1001.600480
        2010-12  1103.779375  1108.395183  1084.508186  1003.303201
        2011-01  1109.298272  1133.001556  1015.316564  1008.119056
                      ...          ...          ...          ...
        2020-12  3381.729677  4043.276231  1394.513920  1192.576493
        2021-01  3332.356424  4002.034813  1349.610572  1197.704572
        2021-02  3364.480340  4112.891178  1265.124950  1204.291947
        2021-03  3480.083884  4301.261594  1250.702526  1212.842420
        """
        if hasattr(self, "inflation"):
            df = pd.concat(
                [self.ror, self.assets_ror, self.inflation_ts],
                axis=1,
                join="inner",
                copy="false",
            )
        else:
            df = pd.concat(
                [self.ror, self.assets_ror], axis=1, join="inner", copy="false"
            )
        return Frame.get_wealth_indexes(df)

    @property
    def mean_return_monthly(self) -> float:
        """
        Calculate monthly mean return (arithmetic mean) for the portfolio rate of return time series.

        Mean return calculated for the full history period.

        Returns
        -------
        Float
            Mean return value.

        Examples
        --------
        >>> pf = ok.Portfolio(['ISF.LSE', 'XGLE.LSE'], weights=[0.6, 0.4], ccy='GBP')
        >>> pf
        0.0001803312727272665
        """
        return Frame.get_portfolio_mean_return(self.weights, self.assets_ror)

    @property
    def mean_return_annual(self) -> float:
        """
        Calculate annualized mean return (arithmetic mean) for the portfolio rate of return time series.

        Mean return calculated for the full history period.

        Returns
        -------
        Float
           Mean return value.

        Examples
        --------
        >>> pf = ok.Portfolio(['XCS6.XETR', 'PHAU.LSE'], weights=[0.85, 0.15], ccy='USD')
        >>> pf.names
        {'XCS6.XETR': 'Xtrackers MSCI China UCITS ETF 1C', 'PHAU.LSE': 'WisdomTree Physical Gold'}
        >>> pf.mean_return_annual
        0.09005826844072184
        """
        return Float.annualize_return(self.mean_return_monthly)

    @property
    def annual_return_ts(self) -> pd.Series:
        """
        Calculate annual rate of return time series for portfolio.

        Rate of return is calculated for each calendar year.

        Returns
        -------
        DataFrame
            Calendar annual rate of return time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['VOO.US', 'AGG.US'], weights=[0.4, 0.6])
        >>> pf.annual_return_ts
        Date
        2010    0.034299
        2011    0.056599
        2012    0.086613
        2013    0.107111
        2014    0.090420
        2015    0.010381
        2016    0.063620
        2017    0.105450
        2018   -0.013262
        2019    0.174182
        2020    0.124668
        2021    0.030430
        Freq: A-DEC, Name: portfolio_5364.PF, dtype: float64
        """
        return Frame.get_annual_return_ts_from_monthly(self.ror)

    def get_cagr(self, period: Optional[int] = None, real: bool = False) -> pd.Series:
        """
        Calculate portfolio Compound Annual Growth Rate (CAGR) for a given trailing period.

        Compound annual growth rate (CAGR) is the rate of return that would be required for an investment to grow from
        its initial to its final value, assuming all incomes were reinvested.

        Inflation adjusted annualized returns (real CAGR) are shown with `real=True` option.

        Annual inflation value is calculated for the same period if inflation=True in the AssetList.

        Parameters
        ----------
        period: int, optional
            CAGR trailing period in years. None for the full time CAGR.
        real: bool, default False
            CAGR is adjusted for inflation (real CAGR) if True.
            Portfolio should be initiated with Inflation=True for real CAGR.

        Returns
        -------
        Series
            Portfolio CAGR value and annualized inflation (optional).

        Notes
        -----
        CAGR is not defined for periods less than 1 year (NaN values are returned).

        Examples
        --------
        >>> pf = ok.Portfolio(['XCS6.XETR', 'PHAU.LSE'], weights=[0.85, 0.15], ccy='USD')
        >>> pf.names
        {'XCS6.XETR': 'Xtrackers MSCI China UCITS ETF 1C', 'PHAU.LSE': 'WisdomTree Physical Gold'}

        To get inflation adjusted return (real annualized return) add `real=True` option:
        >>> pf.get_cagr(period=5, real=True)
        portfolio_5625.PF    0.121265
        dtype: float64
        """
        ts = self._add_inflation()
        df = self._make_df_if_series(ts)
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
                    "Real CAGR is not defined. Set inflation=True in Portfolio to calculate it."
                )
            mean_inflation = Frame.get_cagr(self.inflation_ts[dt:])
            cagr = (1. + cagr) / (1. + mean_inflation) - 1.
            cagr.drop(self.inflation, inplace=True)
        return cagr

    def get_rolling_cagr(self, window: int = 12, real: bool = False) -> pd.DataFrame:
        """
        Calculate rolling CAGR (Compound Annual Growth Rate) for the portfolio.

        Parameters
        ----------
        window : int, default 12
            Size of the moving window in months. Window size should be at least 12 months for CAGR.
        real: bool, default False
            CAGR is adjusted for inflation (real CAGR) if True.
            Portfolio should be initiated with Inflation=True for real CAGR.

        Returns
        -------
        DataFrame
            Time series of rolling CAGR and mean inflation (optionaly).

        Notes
        -----
        CAGR is not defined for periods less than 1 year (NaN values are returned).

        Examples
        --------
        Get inflation adjusted rolling CAGR (real annualized return) win 5 years window:
        >>> x = ok.Portfolio(['DXET.XETR', 'DBXN.XETR'], ccy='EUR', inflation=True)
        >>> x.get_rolling_cagr(window=5*12, real=True)
                 portfolio_7645.PF
        2013-09           0.029914
        2013-10           0.052435
        2013-11           0.055651
        2013-12           0.045180
        2014-01           0.063153
                            ...
        2021-01           0.032734
        2021-02           0.037779
        2021-03           0.043811
        2021-04           0.043729
        2021-05           0.042704
        """
        df = self._add_inflation()
        if real:
            df = self._make_real_return_time_series(df)
        return Frame.get_rolling_fn(df, window=window, fn=Frame.get_cagr)

    def get_cumulative_return(self, period: Union[str, int, None] = None, real: bool = False) -> pd.Series:
        """
        Calculate cumulative return over a given trailing period for the portfolio.

        The cumulative return is the total change in the portfolio price during the investment period.

        Inflation adjusted cumulative returns (real cumulative returns) are shown with `real=True` option.
        Annual inflation data is calculated for the same period if `inflation=True` in the AssetList.

        Parameters
        ----------
        period: str, int or None, default None
            Trailing period in years.
            None - full time cumulative return.
            'YTD' - (Year To Date) period of time beginning the first day of the calendar year up to the last month.
        real: bool, default False
            Cumulative return is adjusted for inflation (real cumulative return) if True.
            Portfolio should be initiated with `Inflation=True` for real cumulative return.

        Returns
        -------
        Series
            Cumulative rate of return values for portfolio and cumulative inflation (if inflation=True in Portfolio).

        Examples
        --------
        >>> pf = ok.Portfolio(['BTC-USD.CC', 'LTC-USD.CC'], weights=[.8, .2], last_date='2021-03')
        >>> pf.get_cumulative_return(period=2)
        portfolio_6232.PF    9.920432
        USD.INFL             0.042121
        dtype: float64

        To get inflation adjusted return (real annualized return) add `real=True` option:
        >>> pf.get_cumulative_return(period=2, real=True)
        portfolio_6232.PF    9.39381
        dtype: float64
        """
        ts = self._add_inflation()
        df = self._make_df_if_series(ts)
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
                    "Set inflation=True in Portfolio to calculate it."
                )
            cumulative_inflation = Frame.get_cumulative_return(self.inflation_ts[dt:])
            cr = (1. + cr) / (1. + cumulative_inflation) - 1.
            cr.drop(self.inflation, inplace=True)
        return cr

    def get_rolling_cumulative_return(self, window: int = 12, real: bool = False) -> pd.DataFrame:
        """
        Calculate rolling cumulative return.

        The cumulative return is the total change in the portfolio price.

        Parameters
        ----------
        window : int, default 12
            Size of the moving window in months.
        real: bool, default False
            Cumulative return is adjusted for inflation (real cumulative return) if True.
            Portfolio should be initiated with `Inflation=True` for real cumulative return.

        Returns
        -------
        DataFrame
            Time series of rolling cumulative return and inflation (optional).
        """
        ts = self._add_inflation()
        if real:
            ts = self._make_real_return_time_series(ts)
        df = self._make_df_if_series(ts)
        return Frame.get_rolling_fn(
            df,
            window=window,
            fn=Frame.get_cumulative_return,
            window_below_year=True,
        )

    @property
    def assets_close_monthly(self) -> pd.DataFrame:
        """
        Show assets monthly close time series adjusted to the base currency.

        Returns
        -------
        DataFrame
            Assets monthly close time series adjusted to the base currency.
        """
        assets_close_monthly = pd.DataFrame(dtype=float)
        for i, x in enumerate(self.asset_obj_dict.values()):
            if i == 0:  # required to use pd.concat below (df should not be empty).
                assets_close_monthly = x.close_monthly if x.currency == self.currency else self._adjust_price_to_currency_monthly(x.close_monthly, x.currency)
                assets_close_monthly.rename(x.symbol, inplace=True)
            else:
                new = x.close_monthly if x.currency == self.currency else self._adjust_price_to_currency_monthly(x.close_monthly, x.currency)
                new.rename(x.symbol, inplace=True)
                assets_close_monthly = pd.concat([assets_close_monthly, new], axis=1, join="inner", copy="false")
        if isinstance(assets_close_monthly, pd.Series):
            assets_close_monthly = assets_close_monthly.to_frame()
        assets_close_monthly = assets_close_monthly[self.first_date: self.last_date]
        return assets_close_monthly

    @property
    def close_monthly(self) -> pd.Series:
        """
        Portfolio size monthly time series.

        Portfolio size is shown in base currency units. It is similar to the close value of an asset.
        Initial portfolio value is equal to 1000 units of base currency.

        Returns
        -------
        pd.Series
            Monthly portfolio size time series.
        """
        return self.wealth_index.iloc[:, 0]

    @property
    def number_of_securities(self) -> pd.DataFrame:
        """
        Calculate the number of securities monthly time series for the portfolio assets.

        Number of securities is changing over time as the dividends are reinvested.
        Portfolio rebalancing also affects the number of securities.

        Initial number of securities depends on the portfolio size in base currency (1000 units).

        Returns
        -------
        DataFrame
            Number of securities monthly time series for the portfolio assets.
        """
        return self.weights_ts.mul(self.wealth_index.iloc[:, 0], axis=0).div(self.assets_close_monthly, axis=0)

    @property
    def dividends(self) -> pd.Series:
        """
        Calculate portfolio dividends monthly time series.

        Portfolio dividends are obtained by summing asset dividends adjusted to the base currency.
        Dividends size depends on the portfolio value and number of securities.

        Returns
        -------
        Series
            Portfolio dividends monthly time series.
        """
        s = (self._get_assets_dividends() * self.number_of_securities).sum(axis=1)
        s.rename(self.symbol, inplace=True)
        return s

    @property
    def dividend_yield(self) -> pd.Series:
        """
        Calculate last twelve months (LTM) dividend yield time series for the portfolio. Time series has monthly values.

        Portfolio dividend yield is a weighted sum of the assets dividend yields (adjusted to
        the portfolio base currency).

        For an asset LTM dividend yield is the sum trailing twelve months of common dividends per share divided by
        the current price per share.

        Returns
        -------
        Series
            Portfolio LTM dividend yield monthly time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['T.US', 'XOM.US'], weights=[0.8, 0.2], first_date='2010-01', last_date='2021-01', ccy='USD')
        >>> pf.dividend_yield
        2010-01    0.013249
        2010-02    0.014835
        2010-03    0.014257
                     ...
        2020-11    0.076132
        2020-12    0.074743
        2021-01    0.073643
        Freq: M, Name: portfolio_8836.PF, Length: 133, dtype: float64
        """
        df = self.assets_dividend_yield @ self.weights_ts.T
        div_yield_series = pd.Series(np.diag(df), index=df.index)
        div_yield_series.rename(self.symbol, inplace=True)
        return div_yield_series

    @property
    def real_mean_return(self) -> float:
        """
        Calculate annualized real mean return (arithmetic mean) for the rate of return time series.

        Real rate of return is adjusted for inflation. Real return is defined if
        there is an `inflation=True` option in Portfolio.

        Returns
        -------
        float
            Annualized value of the mean for the real rate of return time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['MSFT.US', 'AAPL.US'])
        >>> pf.real_mean_return
        0.3088967455111862
        """
        if not hasattr(self, "inflation"):
            raise Exception(
                "Real Return is not defined. Set inflation=True to calculate."
            )
        infl_mean = Float.annualize_return(self.inflation_ts.mean())
        ror_mean = Float.annualize_return(self.ror.mean())
        return (1.0 + ror_mean) / (1.0 + infl_mean) - 1.0

    @property
    def risk_monthly(self) -> float:
        """
        Calculate monthly risk (standard deviation of return) for Portfolio.

        Monthly risk of portfolio is a standard deviation of the rate of return time series.
        Standard deviation (sigma σ) is normalized by N-1.

        Returns
        -------
        float
            Standard deviation value of the monthly return time series.

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
        >>> pf = ok.Portfolio(['MSFT.US', 'AAPL.US'])
        >>> pf.risk_monthly
        0.09415483565833212
        """
        return self.ror.std()

    @property
    def risk_annual(self) -> float:
        """
        Calculate annualized risk (return standard deviation) for portfolio.

        Returns
        -------
        float
            Annualized standard deviation value of the monthly return time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['MSFT.US', 'AAPL.US'])
        >>> pf.risk_annual
        0.4374591902169046
        """
        return Float.annualize_risk(self.risk_monthly, self.mean_return_monthly)

    @property
    def semideviation_monthly(self) -> float:
        """
        Calculate semi-deviation monthly value for portfolio rate of return time series.

        Semi-deviation (Downside risk) is the risk of the return being below the expected return.

        Returns
        -------
        float
            Semi-deviation monthly value for portfolio rate of return time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['MSFT.US', 'AAPL.US'])
        >>> pf.semideviation_monthly
        0.05601433676604449
        """
        return Frame.get_semideviation(self.ror)

    @property
    def semideviation_annual(self) -> float:
        """
        Return semideviation annualized value for portfolio rate of return time series.

        Semi-deviation (Downside risk) is the risk of the return being below the expected return.

        Returns
        -------
        float
            Annualized semi-deviation monthly value for portfolio rate of return time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['MSFT.US', 'AAPL.US'])
        >>> pf.semideviation_annual
        0.1940393544621248
        """
        return Frame.get_semideviation(self.ror) * 12 ** 0.5

    def get_var_historic(self, time_frame: int = 12, level=1) -> float:
        """
        Calculate historic Value at Risk (VaR) for the portfolio.

        The VaR calculates the potential loss of an investment with a given time frame and confidence level.
        Loss is a positive number (expressed in cumulative return).
        If VaR is negative there are expected gains at this confidence level.

        Parameters
        ----------
        time_frame : int, default 12
            Time frame for VAR. Default is 12 months.
        level : int, default 1
            Confidence level in percents. Default value is 1%.

        Returns
        -------
        Float
            Historic Value at Risk (VaR) value for the portfolio.

        Examples
        --------
        >>> x = ok.Portfolio(['SP500TR.INDX', 'SP500BDT.INDX'], last_date='2021-01')
        >>> x.get_var_historic(time_frame=12, level=1)
        0.24030006476701732
        """
        # remove inflation column from rolling return
        df = self.get_rolling_cumulative_return(window=time_frame).loc[:, [self.symbol]]
        return Frame.get_var_historic(df, level).iloc[0]

    def get_cvar_historic(self, time_frame: int = 12, level=1) -> float:
        """
        Calculate historic Conditional Value at Risk (CVAR, expected shortfall) for the portfolio.

        CVaR is the average loss over a specified time period of unlikely scenarios beyond the confidence level.
        Loss is a positive number (expressed in cumulative return).
        If CVaR is negative there are expected gains at this confidence level.

        Parameters
        ----------
        time_frame : int, default 12 (12 months)
        level : int, default 1 (1% quantile)

        Returns
        -------
        Float
            Historic Conditional Value at Risk (CVAR, expected shortfall) value for the portfolio.

        Examples
        --------
        >>> x = ok.Portfolio(['USDEUR.FX', 'BTC-USD.CC'], last_date='2021-01')
        >>> x.get_cvar_historic(time_frame=2, level=1)
        0.3566909250442616
        """
        # remove inflation column form rolling return
        df = self.get_rolling_cumulative_return(window=time_frame).loc[:, [self.symbol]]
        return Frame.get_cvar_historic(df, level).iloc[0]

    @property
    def drawdowns(self) -> pd.Series:
        """
        Calculate drawdowns time series for the portfolio.

        The drawdown is the percent decline from a previous peak in wealth index.

        Returns
        -------
        Series
            Drawdowns time series for the portfolio
        """
        return Frame.get_drawdowns(self.ror)

    @property
    def recovery_period(self) -> int:
        """
        Calculate the longest recovery period for the portfolio assets value.

        The recovery period (drawdown duration) is the number of months to reach the value of the last maximum.

        Returns
        -------
        Integer
            Max recovery period for the protfolio assets value in months.

        Notes
        -----
        If the last maximum value is not recovered NaN is returned.
        The largest recovery period does not necessary correspond to the max drawdown.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US'], weights=[0.5, 0.5])
        >>> pf.recovery_period
        35

        See Also
        --------
        drawdowns : Calculate drawdowns time series.
        """
        if hasattr(self, "inflation"):
            w_index = self.wealth_index.drop(columns=[self.inflation])
        else:
            w_index = self.wealth_index
        if isinstance(w_index, pd.DataFrame):
            # time series should be a Series to use groupby
            w_index = w_index.squeeze()
        cummax = w_index.cummax()
        s = cummax.pct_change()[1:]
        s1 = s.where(s == 0).notnull().astype(int)
        s1_1 = s.where(s == 0).isnull().astype(int).cumsum()
        s2 = s1.groupby(s1_1).cumsum()
        # Max recovery period date should not be in the border (means it's not recovered)
        max_period = s2.max() if s2.idxmax().to_timestamp() != self.last_date else np.NAN
        return max_period

    def describe(self, years: Tuple[int] = (1, 5, 10)) -> pd.DataFrame:
        """
        Generate descriptive statistics for the portfolio.

        Statistics includes:
        - YTD (Year To date) compound return
        - CAGR for a given list of periods
        - LTM Dividend yield - last twelve months dividend yield

        Risk metrics (full available period):
        - risk (standard deviation)
        - CVAR
        - max drawdowns (and dates)

        Parameters
        ----------
        years : tuple of (int,), default (1, 5, 10)
            List of periods for CAGR.

        Returns
        -------
        DataFrame
            Table of descriptive statistics for the portfolio.

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
        row.update(period="YTD", property="compound return")
        description = description.append(row, ignore_index=True)
        # CAGR for a list of periods
        if self.pl.years >= 1:
            for i in years:
                dt = Date.subtract_years(dt0, i)
                if dt >= self.first_date:
                    row = self.get_cagr(period=i).to_dict()
                else:
                    row = (
                        {x: None for x in df.columns}
                        if hasattr(self, "inflation")
                        else {self.symbol: None}
                    )
                row.update(period=f"{i} years", property="CAGR")
                description = description.append(row, ignore_index=True)
            # CAGR for full period
            row = self.get_cagr(period=None).to_dict()
            row.update(period=self._pl_txt, property="CAGR",)
            description = description.append(row, ignore_index=True)
            # Dividend Yield
            value = self.dividend_yield.iloc[-1]
            row = {self.symbol: value}
            row.update(period="LTM", property=f"Dividend yield",)
            description = description.append(row, ignore_index=True)
        # risk (standard deviation)
        row = {self.symbol: self.risk_annual}
        row.update(
            period=self._pl_txt, property="Risk"
        )
        description = description.append(row, ignore_index=True)
        # CVAR
        if self.pl.years >= 1:
            row = {self.symbol: self.get_cvar_historic()}
            row.update(
                period=self._pl_txt,
                property="CVAR",
            )
            description = description.append(row, ignore_index=True)
        # max drawdowns
        row = {self.symbol: self.drawdowns.min()}
        row.update(
            period=self._pl_txt,
            property="Max drawdown",
        )
        description = description.append(row, ignore_index=True)
        # max drawdowns dates
        row = {self.symbol: self.drawdowns.idxmin()}
        row.update(
            period=self._pl_txt,
            property="Max drawdown date",
        )
        description = description.append(row, ignore_index=True)
        if hasattr(self, "inflation"):
            description.rename(columns={self.inflation: "inflation"}, inplace=True)
        description = Frame.change_columns_order(
            description, ["property", "period", self.symbol]
        )
        return description

    @property
    def table(self) -> pd.DataFrame:
        """
        Return security name - ticker - weight table.

        Returns
        -------
        DataFrame
            Security name - ticker - weight table.

        Examples
        --------
        >>> pf = ok.Portfolio(['MSFT.US', 'AAPL.US'])
        >>> pf.table
                        asset name   ticker  weights
        0  Microsoft Corporation  MSFT.US      0.5
        1              Apple Inc  AAPL.US      0.5
        """
        x = pd.DataFrame(
            data={
                "asset name": list(self.names.values()),
                "ticker": list(self.names.keys()),
            }
        )
        x["weights"] = self.weights
        return x

    # Forecasting

    def _test_forecast_period(self, years):
        max_period_years = round(self.period_length / 2)
        if max_period_years < 1:
            raise ValueError(
                f"Time series does not have enough history to forecast. "
                f"Period length is {self.period_length:.2f} years. At least 2 years are required."
            )
        if not isinstance(years, int) or years == 0:
            raise ValueError("years must be an integer number (not equal to zero).")
        if years > max_period_years:
            raise ValueError(
                f"Forecast period {years} years is not credible. "
                f"It should not exceed 1/2 of portfolio history period length {self.period_length / 2} years"
            )

    def percentile_inverse(
        self,
        distr: str = "norm",
        years: int = 1,
        score: float = 0,
        n: Optional[int] = None,
    ) -> float:
        """
        Compute the percentile rank of a score (CAGR value) in a given time frame.

        If percentile_inverse of, for example, 0% (CAGR value) is equal to 8% for 1 year time frame
        it means that 8% of the CAGR values in the distribution are negative in 1 year periods. Or in other words
        the probability of getting negative result after 1 year of investments is 8%.

        Args:
            distr: norm, lognorm, hist - distribution type (normal or lognormal) or hist for CAGR array from history
            years: period length when CAGR is calculated
            score: score that is compared to the elements in CAGR array.
            n: number of random time series (for 'norm' or 'lognorm' only)

        Returns:
            Percentile-position of score (0-100) relative to distr.
        """
        if distr == "hist":
            cagr_distr = self.get_rolling_cagr(years)
        elif distr in ["norm", "lognorm"]:
            if not n:
                n = 1000
            cagr_distr = self._get_monte_carlo_cagr_distribution(
                distr=distr, years=years, n=n
            )
        else:
            raise ValueError('distr should be one of "norm", "lognorm", "hist".')
        return scipy.stats.percentileofscore(cagr_distr, score, kind="rank")

    def percentile_from_history(
        self, years: int, percentiles: List[int] = [10, 50, 90]
    ) -> pd.DataFrame:
        """
        Calculate given percentiles for portfolio CAGR (annualized rolling returns) distribution from the historical data.
        Each percentile is calculated for a period range from 1 year to 'years'.

        years - max window size for rolling CAGR (limited with half history of period length).
        percentiles - list of percentiles to be calculated
        """
        self._test_forecast_period(years)
        period_range = range(1, years + 1)
        returns_dict = {}
        for percentile in percentiles:
            percentile_returns_list = [
                self.get_rolling_cagr(years * 12).loc[:, self.symbol].quantile(percentile / 100)
                for years in period_range
            ]
            returns_dict.update({percentile: percentile_returns_list})
        df = pd.DataFrame(returns_dict, index=list(period_range))
        df.index.rename("years", inplace=True)
        return df

    def forecast_wealth_history(
        self, years: int = 1, percentiles: List[int] = [10, 50, 90]
    ) -> pd.DataFrame:
        """
        Compute accumulated wealth for each CAGR derived by 'percentile_from_history' method.
        CAGRs are taken from the historical data.

        Initial portfolio wealth is adjusted to the last known historical value (from wealth_index). It is useful
        for a chart with historical wealth index and forecasted values.

        Args:
            years:
            percentiles:

        Returns:
            Dataframe of percentiles for period range from 1 to 'years'
        """
        first_value = self.wealth_index[self.symbol].values[-1]
        percentile_returns = self.percentile_from_history(
            years=years, percentiles=percentiles
        )
        return first_value * (percentile_returns + 1.0).pow(
            percentile_returns.index.values, axis=0
        )

    def _forecast_preparation(self, years: int):
        self._test_forecast_period(years)
        period_months = years * _MONTHS_PER_YEAR
        # make periods index where the shape is max_period
        start_period = self.last_date.to_period("M")
        end_period = self.last_date.to_period("M") + period_months - 1
        ts_index = pd.period_range(start_period, end_period, freq="M")
        return period_months, ts_index

    def forecast_monte_carlo_returns(
        self, distr: str = "norm", years: int = 1, n: int = 100
    ) -> pd.DataFrame:
        """
        Generates N random monthly returns time series with normal or lognormal distributions.
        Forecast period should not exceed 1/2 of portfolio history period length.
        """
        period_months, ts_index = self._forecast_preparation(years)
        # random returns
        if distr == "norm":
            random_returns = np.random.normal(
                self.mean_return_monthly, self.risk_monthly, (period_months, n)
            )
        elif distr == "lognorm":
            std, loc, scale = scipy.stats.lognorm.fit(self.ror)
            random_returns = scipy.stats.lognorm(std, loc=loc, scale=scale).rvs(
                size=[period_months, n]
            )
        else:
            raise ValueError('distr should be "norm" (default) or "lognorm".')
        return pd.DataFrame(data=random_returns, index=ts_index)

    def forecast_monte_carlo_wealth_indexes(
        self, distr: str = "norm", years: int = 1, n: int = 100
    ) -> pd.DataFrame:
        """
        Generates N future random wealth indexes.
        Random distribution could be normal or lognormal.

        First value for the forecasted wealth indexes is the last historical portfolio index value. It is useful
        for a chart with historical wealth index and forecasted values.
        """
        if distr not in ["norm", "lognorm"]:
            raise ValueError('distr should be "norm" (default) or "lognorm".')
        return_ts = self.forecast_monte_carlo_returns(distr=distr, years=years, n=n)
        first_value = self.wealth_index[self.symbol].values[-1]
        return Frame.get_wealth_indexes(return_ts, first_value)

    def _get_monte_carlo_cagr_distribution(
        self, distr: str = "norm", years: int = 1, n: int = 100,
    ) -> pd.Series:
        """
        Generate random CAGR distribution.
        CAGR is calculated for each of N future random returns time series.
        Random distribution could be normal or lognormal.
        """
        if distr not in ["norm", "lognorm"]:
            raise ValueError('distr should be "norm" (default) or "lognorm".')
        return_ts = self.forecast_monte_carlo_returns(distr=distr, years=years, n=n)
        return Frame.get_cagr(return_ts)

    def forecast_monte_carlo_cagr(
        self,
        distr: str = "norm",
        years: int = 1,
        percentiles: List[int] = [10, 50, 90],
        n: int = 10000,
    ) -> dict:
        """
        Calculate percentiles for forecasted CAGR distribution.
        CAGR is calculated for each of N future random returns time series.
        Random distribution could be normal or lognormal.
        """
        if distr not in ["norm", "lognorm"]:
            raise ValueError('distr should be "norm" (default) or "lognorm".')
        cagr_distr = self._get_monte_carlo_cagr_distribution(
            distr=distr, years=years, n=n
        )
        results = {}
        for percentile in percentiles:
            value = cagr_distr.quantile(percentile / 100)
            results.update({percentile: value})
        return results

    def forecast_wealth(
        self,
        distr: str = "norm",
        years: int = 1,
        percentiles: List[int] = [10, 50, 90],
        today_value: Optional[int] = None,
        n: int = 1000,
    ) -> Dict[int, float]:
        """
        Calculate percentiles of forecasted random accumulated wealth distribution.
        Random distribution could be normal lognormal or from history.

        today_value - the value of portfolio today (before forecast period). If today_value is None
        the last value of the historical wealth indexes is taken.
        """
        if distr == "hist":
            results = (
                self.forecast_wealth_history(years=years, percentiles=percentiles)
                .iloc[-1]
                .to_dict()
            )
        elif distr in ["norm", "lognorm"]:
            results = {}
            wealth_indexes = self.forecast_monte_carlo_wealth_indexes(
                distr=distr, years=years, n=n
            )
            for percentile in percentiles:
                value = wealth_indexes.iloc[-1, :].quantile(percentile / 100)
                results.update({percentile: value})
        else:
            raise ValueError('distr should be "norm", "lognorm" or "hist".')
        if today_value:
            modifier = today_value / self.wealth_index[self.symbol].values[-1]
            results.update((x, y * modifier) for x, y in results.items())
        return results

    def plot_forecast(
        self,
        distr: str = "norm",
        years: int = 5,
        percentiles: List[int] = [10, 50, 90],
        today_value: Optional[int] = None,
        n: int = 1000,
        figsize: Optional[tuple] = None,
    ):
        """
        Plots forecasted ranges of wealth indexes (lines) for a given set of percentiles.

        distr - the distribution model type:
            norm - normal distribution
            lognorm - lognormal distribution
            hist - percentiles are taken from historical data
        today_value - the value of portfolio today (before forecast period)
        n - number of random wealth time series used to calculate percentiles (not needed if distr='hist')
        """
        wealth = self.wealth_index
        x1 = self.last_date
        x2 = x1.replace(year=x1.year + years)
        y_start_value = wealth[self.symbol].iloc[-1]
        y_end_values = self.forecast_wealth(
            distr=distr, years=years, percentiles=percentiles, n=n
        )
        if today_value:
            modifier = today_value / y_start_value
            wealth *= modifier
            y_start_value = y_start_value * modifier
            y_end_values.update((x, y * modifier) for x, y in y_end_values.items())
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            wealth.index.to_timestamp(),
            wealth[self.symbol],
            linewidth=1,
            label="Historical data",
        )
        for percentile in percentiles:
            x, y = [x1, x2], [y_start_value, y_end_values[percentile]]
            if percentile == 50:
                ax.plot(x, y, color="blue", linestyle="-", linewidth=2, label="Median")
            else:
                ax.plot(
                    x,
                    y,
                    linestyle="dashed",
                    linewidth=1,
                    label=f"Percentile {percentile}",
                )
        ax.legend(loc="upper left")
        return ax

    def plot_forecast_monte_carlo(
        self,
        distr: str = "norm",
        years: int = 1,
        n: int = 20,
        figsize: Optional[tuple] = None,
    ):
        """
        Plots N random wealth indexes and historical wealth index.
        Forecasted indexes are generated accorded to a given distribution (Monte Carlo simulation).
        Normal and lognormal distributions could be used for Monte Carlo simulation.
        """
        s1 = self.wealth_index
        s2 = self.forecast_monte_carlo_wealth_indexes(distr=distr, years=years, n=n)
        s1[self.symbol].plot(legend=None, figsize=figsize)
        for n in s2:
            s2[n].plot(legend=None)

    # distributions
    @property
    def skewness(self):
        """
        Compute expanding skewness of the return time series.
        For normally distributed data, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.
        """
        return Frame.skewness(self.ror)

    def skewness_rolling(self, window: int = 60):
        """
        Compute rolling skewness of the return time series.
        For normally distributed data, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.

        window - the rolling window size in months (default is 5 years).
        The window size should be at least 12 months.
        """
        return Frame.skewness_rolling(self.ror, window=window)

    @property
    def kurtosis(self):
        """
        Calculate expanding Fisher (normalized) kurtosis time series for portfolio returns.
        Kurtosis is the fourth central moment divided by the square of the variance.
        Kurtosis should be close to zero for normal distribution.
        """
        return Frame.kurtosis(self.ror)

    def kurtosis_rolling(self, window: int = 60):
        """
        Calculate rolling Fisher (normalized) kurtosis time series for portfolio returns.
        Kurtosis is the fourth central moment divided by the square of the variance.
        Kurtosis should be close to zero for normal distribution.

        window - the rolling window size in months (default is 5 years).
        The window size should be at least 12 months.
        """
        return Frame.kurtosis_rolling(self.ror, window=window)

    @property
    def jarque_bera(self):
        """
        Performs Jarque-Bera test for normality.
        It shows whether the returns have the skewness and kurtosis matching a normal distribution.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
            Low statistic numbers correspond to normal distribution.
        """
        return Frame.jarque_bera_series(self.ror)

    def kstest(self, distr: str = "norm") -> dict:
        """
        Performs Kolmogorov-Smirnov test on portfolio returns and evaluate goodness of fit.
        Test works with normal and lognormal distributions.

        Returns:
            (The test statistic, The p-value for the hypothesis test)
        """
        return Frame.kstest_series(self.ror, distr=distr)

    def plot_percentiles_fit(
        self, distr: str = "norm", figsize: Optional[tuple] = None
    ):
        """
        Generates a probability plot of portfolio returns against percentiles of a specified
        theoretical distribution (the normal distribution by default).
        Works with normal and lognormal distributions.
        """
        plt.figure(figsize=figsize)
        if distr == "norm":
            scipy.stats.probplot(self.ror, dist=distr, plot=plt)
        elif distr == "lognorm":
            scipy.stats.probplot(
                self.ror,
                sparams=(scipy.stats.lognorm.fit(self.ror)),
                dist=distr,
                plot=plt,
            )
        else:
            raise ValueError('distr should be "norm" (default) or "lognorm".')
        plt.show()

    def plot_hist_fit(self, distr: str = "norm", bins: int = None):
        """
        Plots historical distribution histogram and theoretical PDF (Probability Distribution Function).
        Lognormal and normal distributions could be used.

        normal distribution - 'norm'
        lognormal distribution - 'lognorm'
        """
        data = self.ror
        # Plot the histogram
        plt.hist(data, bins=bins, density=True, alpha=0.6, color="g")
        # Plot the PDF.Probability Density Function
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        if distr == "norm":  # Generate PDF
            mu, std = scipy.stats.norm.fit(data)
            p = scipy.stats.norm.pdf(x, mu, std)
        elif distr == "lognorm":
            std, loc, scale = scipy.stats.lognorm.fit(data)
            mu = np.log(scale)
            p = scipy.stats.lognorm.pdf(x, std, loc, scale)
        else:
            raise ValueError('distr should be "norm" (default) or "lognorm".')
        plt.plot(x, p, "k", linewidth=2)
        title = "Fit results: mu = %.3f,  std = %.3f" % (mu, std)
        plt.title(title)
        plt.show()
