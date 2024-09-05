from random import randint
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

from okama import settings
from okama.common import make_asset_list, validators
from okama.common.helpers import helpers, ratios


class Portfolio(make_asset_list.ListMaker):
    """
    Implementation of investment portfolio.

    Investments portfolio is a type of financial asset (same as stocks, ETF, mutual funds, currencies etc.).
    Arguments are similar to AssetList, however Portfolio additionally has:

    - weights
    - rebalancing_period
    - initial_amount
    - cashflow
    - discount_rate
    - symbol

    Portfolio is defined by the investment strategy, which includes:
    - asset allocation (financial assets and their proportions in the portfolio)
    - the initial investment (`initial_amount` parameter)
    - cash flows or withdrawals/contributions (`cashflow` parameter)
    - discount rate (`discount_rate` parameter)
    - the rebalancing strategy (`rebalancing_period` parameter)

    The rebalancing is the action of bringing the portfolio that has deviated away
    from original target asset allocation back into line. After rebalancing the portfolio assets
    have original weights.

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

    weights : list of float, default None
        List of assets weights.
        The weight of an asset is the percent of an investment portfolio that corresponds to the asset.
        If weights = None an equally weighted portfolio is created (all weights are equal).

    rebalancing_period : {'none', 'month', 'quarter', 'half-year', 'year'}, default 'month'
        Rebalancing period (rebalancing frequency) is predetermined time intervals when
        the investor rebalances the portfolio. If 'none' assets weights are not rebalanced.

    initial_amount : float, default 1000
        Portfolio initial investment FV (at last_date).

    cashflow : float, default 0
        Portfolio monthly cash flow FV (at last_date). Negative value corresponds to withdrawals.
        Positive value corresponds to contributions. Cash flow value is indexed each month by discount_rate.

    discount_rate : float or None, default None
        Cash flow discount rate required to calculate PV values. If not provided geometric mean of inflation is taken.
        For portfolios without inflation the default value from settings is used.

    symbol : str, default None
        Text symbol of portfolio. It is similar to tickers but have a namespace information.
        Portfolio symbol must end with .PF (all_weather_portfolio.PF).
        If None a random symbol is generated (portfolio_7802.PF).
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
        self.weights = weights
        self.assets_weights = dict(zip(self.symbols, self.weights))
        self.rebalancing_period = rebalancing_period
        self.symbol = symbol or f"portfolio_{randint(1000, 9999)}.PF"
        self._ror = pd.DataFrame(dtype=float)
        self.dcf = PortfolioDCF(self)

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
            return pd.concat([self.ror, self.inflation_ts], axis=1, join="inner", copy="false")
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
            [validators.validate_real("weight", weight) for weight in weights]
            helpers.Frame.weights_sum_is_one(weights)
            if len(weights) != len(set(self.symbols)):
                raise ValueError(
                    f"Number of tickers ({len(set(self.symbols))}) should be equal "
                    f"to the weights number ({len(weights)})"
                )
        self._ror = pd.DataFrame(dtype=float)
        #TODO: check why it's not working:
        # self.dcf.mc._monte_carlo_wealth = pd.DataFrame()
        # self.dcf._wealth_index = pd.DataFrame()
        self._weights = weights

    @property
    def weights_ts(self) -> pd.DataFrame:
        """
        Calculate assets weights time series.

        The weights of assets in Portfolio are not constant if rebalancing_period is different from 'month'.

        Returns
        -------
        DataFrame
            Weights of assets time series.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> reb_period='none'  # The Portfolio is not rebalanced.
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US'], weights=[0.5, 0.5], rebalancing_period=reb_period)
        >>> pf.weights_ts.plot()
        >>> plt.show()

        The weights of assets time series will differ significantly if the portfolio rebalancing_period is 1 year.

        >>> pf.rebalancing_period = 'year'  # set a new rebalancing period
        >>> pf.weights_ts.plot()
        >>> plt.show()
        """
        if self.rebalancing_period != "month":
            return helpers.Rebalance(period=self.rebalancing_period).assets_weights_ts(
                ror=self.assets_ror,
                weights=self.weights,
            )
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
        if rebalancing_period in settings.frequency_mapping.keys():
            self._ror = pd.DataFrame(dtype=float)
            #TODO: check why it's not working:
            # self.dcf._monte_carlo_wealth = pd.DataFrame()
            # self.dcf._wealth_index = pd.DataFrame()
            self._rebalancing_period = rebalancing_period
        else:
            raise ValueError(f"rebalancing_period must be in {settings.frequency_mapping.keys()}")

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

        Examples
        --------
        >>> p = ok.Portflio()
        >>> p.symbol  # a randomly generated symbol will be shown
        'portfolio_5312.PF'

        >>> p.symbol = 'spy_portfolo.PF'  # The symbol can be customized after initialization

        New Portfolio can have a custom symbol.

        >>> p = ok.Portfolio(symbol='aggressive.PF')
        >>> p.symbol
        'aggressive.PF'
        """
        return self._symbol

    @symbol.setter
    def symbol(self, text_symbol: str):
        if isinstance(text_symbol, str) and text_symbol.endswith(".PF"):
            if " " in text_symbol:
                raise ValueError("portfolio text symbol should not have whitespace characters.")
            self._symbol = text_symbol
        else:
            raise ValueError('portfolio symbol must be a string ending with ".PF" namespace.')

    @property
    def name(self) -> str:
        """
        Return text name of portfolio.

        For Portfolio name is equal to symbol.

        Returns
        -------
        str
            Text name of the portfolio.

        >>> p = ok.Portfolio()
        >>> p.name
        'portfolio_5312.PF'
        """
        return self.symbol

    @property
    def ror(self) -> pd.Series:
        """
        Calculate monthly rate of return time series for portfolio.

        Returns
        -------
        Series
            Rate of return monthly time series for portfolio.

        Examples
        --------
        >>> pf = ok.Portfolio(first_date='2020-01', last_date='2020-12')
        >>> pf.ror
        Date
        2020-01   -0.0004
        2020-02   -0.0792
        2020-03   -0.1249
        2020-04    0.1270
        2020-05    0.0476
        2020-06    0.0177
        2020-07    0.0589
        2020-08    0.0698
        2020-09   -0.0374
        2020-10   -0.0249
        2020-11    0.1088
        2020-12    0.0370
        Freq: M, Name: portfolio_4669.PF, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.ror.plot(kind='bar')
        >>> plt.show()
        """
        if self._ror.empty:
            if self.rebalancing_period == "month":
                s = helpers.Frame.get_portfolio_return_ts(self.weights, self.assets_ror)
            else:
                s = helpers.Rebalance(period=self.rebalancing_period).return_ror_ts(self.weights, self.assets_ror)
            s.rename(self.symbol, inplace=True)
            self._ror = s
        return self._ror

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
        >>> import matplotlib.pyplot as plt
        >>> x = ok.Portfolio(['SPY.US', 'BND.US'])
        >>> x.wealth_index.plot()
        >>> plt.show()
        """
        df = self._add_inflation()
        df = helpers.Frame.get_wealth_indexes(df)
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
        Сash flows are not taken into account.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period. Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        Wealth index is obtained from the accumulated return multiplicated by the initial investments.
        initial_amount_pv * (Acc_Return + 1)

        initial_amount_pv is the discounted value of the initial investments (initial_amount).

        Returns
        -------
        DataFrame
            Time series of wealth index values for portfolio, each asset and accumulated inflation.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['VOO.US', 'GLD.US'], weights=[0.8, 0.2])
        >>> pf.wealth_index_with_assets.plot()
        >>> plt.show()
        """
        ls = [self.ror, self.assets_ror]
        if hasattr(self, "inflation"):
            ls.append(self.inflation_ts)
        df = pd.concat(ls, axis=1, join="inner", copy="false")
        return helpers.Frame.get_wealth_indexes(df)

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
        return helpers.Frame.get_portfolio_mean_return(self.weights, self.assets_ror)

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
        return helpers.Float.annualize_return(self.mean_return_monthly)

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
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['VOO.US', 'AGG.US'], weights=[0.4, 0.6])
        >>> pf.annual_return_ts.plot(kind='bar')
        >>> plt.show()

        Plot annual returns for portfolio with EUR as the base currency.

        >>> pf = ok.Portfolio(['VOO.US', 'AGG.US'], weights=[0.4, 0.6], ccy='EUR')
        >>> pf.annual_return_ts.plot(kind='bar')
        >>> plt.show()
        """
        return helpers.Frame.get_annual_return_ts_from_monthly(self.ror)

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
        # TODO: add option assets=False
        ts = self._add_inflation()
        df = self._make_df_if_series(ts)
        dt0 = self.last_date
        if period is None:
            dt = self.first_date
        else:
            self._validate_period(period)
            dt = helpers.Date.subtract_years(dt0, period)
        cagr = helpers.Frame.get_cagr(df[dt:])
        if real:
            if not hasattr(self, "inflation"):
                raise ValueError("Real CAGR is not defined. Set inflation=True in Portfolio to calculate it.")
            mean_inflation = helpers.Frame.get_cagr(self.inflation_ts[dt:])
            cagr = (1.0 + cagr) / (1.0 + mean_inflation) - 1.0
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
        >>> x = ok.Portfolio(['DXET.XFRA', 'DBXN.XFRA'], ccy='EUR', inflation=True)
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
        df_or_ts = self._add_inflation()
        if real:
            df_or_ts = self._make_real_return_time_series(df_or_ts)
        df = self._make_df_if_series(df_or_ts)
        return helpers.Frame.get_rolling_fn(df, window=window, fn=helpers.Frame.get_cagr)

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
            dt = helpers.Date.subtract_years(dt0, period)

        cr = helpers.Frame.get_cumulative_return(df[dt:])
        if real:
            if not hasattr(self, "inflation"):
                raise ValueError(
                    "Real cumulative return is not defined (no inflation information is available)."
                    "Set inflation=True in Portfolio to calculate it."
                )
            cumulative_inflation = helpers.Frame.get_cumulative_return(self.inflation_ts[dt:])
            cr = (1.0 + cr) / (1.0 + cumulative_inflation) - 1.0
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

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.6, .35, .05], rebalancing_period='year')
        >>> pf.get_rolling_cumulative_return(window=24, real=True)
                 portfolio_9012.PF
        2006-11           0.125728
        2006-12           0.104348
        2007-01           0.129601
        2007-02           0.110680
        2007-03           0.132610
                            ...
        2021-03           0.263755
        2021-04           0.275474
        2021-05           0.322736
        2021-06           0.264963
        2021-07           0.273801
        [177 rows x 1 columns]
        """
        ts = self._add_inflation()
        if real:
            ts = self._make_real_return_time_series(ts)
        df = self._make_df_if_series(ts)
        return helpers.Frame.get_rolling_fn(
            df,
            window=window,
            fn=helpers.Frame.get_cumulative_return,
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

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'BND.US'], ccy='USD')
        >>> pf.assets_close_monthly.plot()
        >>> plt.show()
        """
        assets_close_monthly = pd.DataFrame(dtype=float)
        for i, x in enumerate(self.asset_obj_dict.values()):
            if i == 0:  # required to use pd.concat below (df should not be empty).
                assets_close_monthly = (
                    x.close_monthly
                    if x.currency == self.currency
                    else self._adjust_price_to_currency_monthly(x.close_monthly, x.currency)
                )
                assets_close_monthly.rename(x.symbol, inplace=True)
            else:
                new = (
                    x.close_monthly
                    if x.currency == self.currency
                    else self._adjust_price_to_currency_monthly(x.close_monthly, x.currency)
                )
                new.rename(x.symbol, inplace=True)
                assets_close_monthly = pd.concat([assets_close_monthly, new], axis=1, join="inner", copy="false")
        if isinstance(assets_close_monthly, pd.Series):
            assets_close_monthly = assets_close_monthly.to_frame()
        assets_close_monthly = assets_close_monthly[self.first_date : self.last_date]
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

        Notes
        -----
        'close_mothly' shows the same output as the 'wealth_index'.
        This property is required as Portfolio must have the same attributes as an Asset.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'BND.US'], ccy='USD')
        >>> pf.close_monthly.plot()
        >>> plt.show()
        """
        return self.wealth_index.iloc[:, 0]

    @property
    def number_of_securities(self) -> pd.DataFrame:
        """
        Calculate the number of securities monthly time series for the portfolio assets.

        The number of securities in the Portfolio is changing over time as the dividends are reinvested.
        Portfolio rebalancing also affects the number of securities.

        Initial number of securities depends on the portfolio size in base currency (1000 units).

        Returns
        -------
        DataFrame
            Number of securities monthly time series for the portfolio assets.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'BND.US'], ccy='USD', last_date='07-2021')
        >>> pf.number_of_securities
                   SPY.US     BND.US
        Date
        2007-05  3.261153   6.687174
        2007-06  3.337216   6.758447
        2007-07  3.407015   6.643519
        2007-08  3.410268   6.663862
        2007-09  3.372630   6.798730
                   ...        ...
        2021-03  3.273521  15.313911
        2021-04  3.204779  15.685601
        2021-05  3.196768  15.749127
        2021-06  3.186124  15.879056
        2021-07  3.166335  16.003569
        [171 rows x 2 columns]
        """
        df = self.weights_ts.mul(self.wealth_index.iloc[:, 0], axis=0).div(self.assets_close_monthly, axis=0)
        return helpers.Frame.change_columns_order(df, self.symbols)

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

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'BND.US'], ccy='USD', last_date='07-2021')
        >>> pf.dividends
        2007-05    0.849271
        2007-06    3.928855
        2007-07    1.551262
        2007-08    2.023148
        2007-09    4.423416
                     ...
        2021-03    6.155337
        2021-04    3.019478
        2021-05    2.056836
        2021-06    6.519521
        2021-07    2.114071
        Freq: M, Name: portfolio_2951.PF, Length: 171, dtype: float64
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

        >>> import matplotlib.pyplot as plt
        >>> pf.dividend_yield.plot()
        >>> plt.show()
        """
        df = self._assets_dividend_yield @ self.weights_ts.T
        div_yield_series = pd.Series(np.diag(df), index=df.index)  # faster than df1.mul(df2).sum(axis=1)
        div_yield_series.rename(self.symbol, inplace=True)
        return div_yield_series

    @property
    def dividends_annual(self) -> pd.DataFrame:
        """
        Return calendar year dividends sum time series for each asset.

        Returns
        -------
        DataFrame
            Annual dividends time series for each asset.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'BND.US'], ccy='USD', last_date='07-2021')
        >>> pf.dividends_annual.plot(kind='bar')
        >>> plt.show()
        """
        return self._get_assets_dividends().resample("Y").sum()

    @property
    def dividend_yield_annual(self):
        """
        Calculate last twelve months (LTM) dividend yield annual time series.

        Time series is based on the dividend yield for the end of calendar year.

        LTM dividend yield is the sum trailing twelve months of common dividends per share divided by
        the current price per share.

        All yields are calculated in the asset list base currency after adjusting the dividends and price time series.
        Forecasted (future) dividends are removed.

        Returns
        -------
        DataFrame
            Time series of LTM dividend yield for each asset.

        See Also
        --------
        dividend_yield : Dividend yield time series.
        dividends_annual : Calendar year dividends time series.
        dividend_paying_years : Number of years of consecutive dividend payments.
        dividend_growing_years : Number of years when the annual dividend was growing.
        get_dividend_mean_yield : Arithmetic mean for annual dividend yield.
        get_dividend_mean_growth_rate : Geometric mean of annual dividends growth rate.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'BND.US'], ccy='USD', last_date='07-2021')
        >>> pf.dividend_yield_annual.plot(kind='bar')
        >>> plt.show()
        """
        return self._assets_dividend_yield.resample(rule="Y").last()

    @property
    def assets_dividend_yield(self):
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
            Monthly time series of LTM dividend yield for each asset.

        See Also
        --------
        dividend_yield_annual : Calendar year dividend yield time series.
        dividends_annual : Calendar year dividends time series.
        dividend_paying_years : Number of years of consecutive dividend payments.
        dividend_growing_years : Number of years when the annual dividend was growing.
        get_dividend_mean_yield : Arithmetic mean for annual dividend yield.
        get_dividend_mean_growth_rate : Geometric mean of annual dividends growth rate.

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
        return super()._assets_dividend_yield

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
            raise ValueError("Real Return is not defined. Set inflation=True to calculate.")
        infl_mean = helpers.Float.annualize_return(self.inflation_ts.mean())
        ror_mean = helpers.Float.annualize_return(self.ror.mean())
        return (1.0 + ror_mean) / (1.0 + infl_mean) - 1.0

    @property
    def risk_monthly(self) -> pd.Series:
        """
        Calculate monthly risk expanding time series for Portfolio.

        Monthly risk of portfolio is a standard deviation of the rate of return time series.
        Standard deviation (sigma σ) is normalized by N-1.

        Returns
        -------
        Series
            Standard deviation of the monthly return expanding time series.

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
        date
        1986-05    0.020117
        1986-06    0.122032
        1986-07    0.130113
        1986-08    0.116642
                     ...
        2023-08    0.092875
        2023-09    0.092861
        2023-10    0.092759
        2023-11    0.092763
        2023-12    0.092665
        Freq: M, Name: portfolio_1094.PF, Length: 453, dtype: float64
        """
        return self.ror.expanding().std().iloc[1:]

    @property
    def risk_annual(self) -> pd.Series:
        """
        Calculate annualized risk expanding time series for portfolio.

        Risk is a standard deviation of the rate of return.

        Annualized risk is calculated for rate of retirun time series for the sample from 'first_date' to
        'last_date'.

        Returns
        -------
        Series
            Annualized standard deviation of the monthly return expanding time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['MSFT.US', 'AAPL.US'])
        >>> pf.risk_annual
        date
        1986-05    0.285175
        1986-06    0.890909
        1986-07    0.616876
        1986-08    0.632270
        1986-09    0.509642
                     ...
        2023-08    0.428297
        2023-09    0.427350
        2023-10    0.426961
        2023-11    0.427930
        """
        risk_ts = self.ror.expanding().std()
        mean_return_ts = self.ror.expanding().mean()
        return helpers.Float.annualize_risk(risk_ts, mean_return_ts).iloc[1:]

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
        return helpers.Frame.get_semideviation(self.ror)

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
        return helpers.Frame.get_semideviation(self.ror) * 12**0.5

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
        return helpers.Frame.get_var_historic(df, level).iloc[0]

    def get_cvar_historic(self, time_frame: int = 12, level=1) -> float:
        """
        Calculate historic Conditional Value at Risk (CVAR, expected shortfall) for the portfolio.

        CVaR is the average loss over a specified time period of unlikely scenarios beyond the confidence level.
        Loss is a positive number (expressed in cumulative return).
        If CVaR is negative there are expected gains at this confidence level.

        Parameters
        ----------
        time_frame : int, default 12 (12 months)
            Time period size in months
        level : int, default 1
            Confidence level in percents to calculate the VaR. Default value is 1% (1% quantile).

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
        return helpers.Frame.get_cvar_historic(df, level).iloc[0]

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
        return helpers.Frame.get_drawdowns(self.ror)

    @property
    def recovery_period(self) -> pd.Series:
        """
        Get recovery period time series for the portfolio value.

        The recovery period (drawdown duration) is the number of months to reach the value of the last maximum.

        Returns
        -------
        pd.Series
            Recovery period time series for the portfolio value

        Notes
        -----
        The largest recovery period does not necessary correspond to the max drawdown.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US'], weights=[0.5, 0.5])
        >>> pf.recovery_period.nlargest()
        date
        2010-10    35
        2004-10     7
        2012-01     7
        2019-03     6
        2018-07     5
        Freq: M, Name: portfolio_5724.PF, dtype: int32

        See Also
        --------
        drawdowns : Calculate drawdowns time series.
        """
        w_index = self.wealth_index_with_assets[self.symbol]
        cummax = w_index.cummax()
        s = cummax.pct_change()[1:]
        s1 = s.where(s == 0).notnull().astype(int)
        s1_1 = s.where(s == 0).isnull().astype(int).cumsum()
        s2 = s1.groupby(s1_1).cumsum()
        return s2[s2.shift(-1) < s2]

    def describe(self, years: Tuple[int] = (1, 5, 10)) -> pd.DataFrame:
        """
        Generate descriptive statistics for the portfolio.

        Statistics includes:

        - YTD (Year To date) compound return
        - CAGR for a given list of periods and full available period
        - Annualized mean rate of return (full available period)
        - LTM Dividend yield - last twelve months dividend yield

        Risk metrics (full available period):

        - risk (standard deviation)
        - CVAR (timeframe is 1 year)
        - max drawdowns (and dates)

        Parameters
        ----------
        years : tuple of (int,), default (1, 5, 10)
            List of periods for CAGR statistics.

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

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'BND.US'], ccy='USD', last_date='07-2021')
        >>> pf.describe(years=[2, 5, 7])  # 'years' customizes the timeframe for the CAGR
                    property              period portfolio_2951.PF  inflation
        0    compound return                 YTD          0.084098   0.048154
        1               CAGR             2 years          0.141465   0.031566
        2               CAGR             5 years          0.102494   0.025582
        3               CAGR             7 years          0.091694   0.019656
        4               CAGR  14 years, 3 months          0.074305   0.019724
        5     Dividend yield                 LTM          0.016504        NaN
        6               Risk  14 years, 3 months          0.086103        NaN
        7               CVAR  14 years, 3 months          0.214207        NaN
        8       Max drawdown  14 years, 3 months         -0.266915        NaN
        9  Max drawdown date  14 years, 3 months           2009-02        NaN
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        df = self._add_inflation()
        # YTD return
        ytd_return = self.get_cumulative_return(period="YTD")
        row = ytd_return.to_dict()
        row.update(period="YTD", property="compound return")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # CAGR for a list of periods
        if self.pl.years >= 1:
            for i in years:
                dt = helpers.Date.subtract_years(dt0, i)
                if dt >= self.first_date:
                    row = self.get_cagr(period=i).to_dict()
                else:
                    row = {x: None for x in df.columns} if hasattr(self, "inflation") else {self.symbol: None}
                row.update(period=f"{i} years", property="CAGR")
                description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
            # CAGR for full period
            row = self.get_cagr(period=None).to_dict()
            row.update(
                period=self._pl_txt,
                property="CAGR",
            )
            description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
            # Mean rate of return (arithmetic mean)
            value = self.mean_return_annual
            row = {self.symbol: value}
            row.update(
                period=self._pl_txt,
                property="Annualized mean return",
            )
            description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
            # Dividend Yield
            value = self.dividend_yield.iloc[-1]
            row = {self.symbol: value}
            row.update(
                period="LTM",
                property="Dividend yield",
            )
            description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # risk (standard deviation)
        row = {self.symbol: self.risk_annual.iloc[-1]}
        row.update(period=self._pl_txt, property="Risk")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # CVAR
        if self.pl.years >= 1:
            row = {self.symbol: self.get_cvar_historic()}
            row.update(
                period=self._pl_txt,
                property="CVAR",
            )
            description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # max drawdowns
        row = {self.symbol: self.drawdowns.min()}
        row.update(
            period=self._pl_txt,
            property="Max drawdown",
        )
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # max drawdowns dates
        row = {self.symbol: self.drawdowns.idxmin()}
        row.update(
            period=self._pl_txt,
            property="Max drawdown date",
        )
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        if hasattr(self, "inflation"):
            description.rename(columns={self.inflation: "inflation"}, inplace=True)
        description = helpers.Frame.change_columns_order(description, ["property", "period", self.symbol])
        return description

    @property
    def table(self) -> pd.DataFrame:
        """
        Return table with security name, ticker, weight for assets in the portfolio.

        Returns
        -------
        DataFrame
            Security name - ticker - weight table.

        Examples
        --------
        >>> pf = ok.Portfolio(["MSFT.US", "AAPL.US"])
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
                "Time series does not have enough history to forecast. "
                f"Period length is {self.period_length:.2f} years. At least 2 years are required."
            )
        if not isinstance(years, int) or years == 0:
            raise ValueError("years must be an integer number (not equal to zero).")

    def percentile_inverse_cagr(
        self,
        distr: str = "norm",
        years: int = 1,
        score: float = 0,
        n: Optional[int] = None,
    ) -> float:
        """
        Compute the percentile rank of a score (CAGR value).

        Percentile rank can be calculated for given distribution type or for hsitorical distribution of CAGR.

        If percentile_inverse of, for example, 0% (CAGR value) is equal to 8% for 1 year time frame
        it means that 8% of the CAGR values in the distribution are negative in 1 year periods. Or in other words
        the probability of getting negative result after 1 year of investments is 8%.

        Parameters
        ----------
        distr: {'norm', 'lognorm', 't', 'hist'}, default 'norm'
            The rate of teturn distribution type.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's t distribution.
            'hist' - percentiles are taken from the historical data.

        years: int, default 1
            Period length (time frame) in years when CAGR is calculated.

        score: float, default 0
            Score that is compared to the elements in CAGR array.

        n: int, optional
            Number of random time series with the defined distributions (for 'norm' or 'lognorm' only).
            Larger argument values can be used to increase the precision of the calculation. But this will lead
            to slower performance.
            Is not required for historical distribution (dist='hist').
            For 'norm' or 'lognorm' distribution default value n=1000 is used.

        Returns
        -------
        float
            Percentile-position of score (0-100) relative to distribution.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='year')
        >>> pf.percentile_inverse_cagr(distr='lognorm', score=0, years=1, n=5000)
        18.08
        The probability of getting negative result (score=0) in 1 year period for lognormal distribution.
        """
        if distr == "hist":
            cagr_distr = self.get_rolling_cagr(years * settings._MONTHS_PER_YEAR).loc[:, [self.symbol]].squeeze()
        elif distr in ["norm", "lognorm", "t"]:
            if not n:
                n = 1000
            cagr_distr = self._get_cagr_distribution(distr=distr, years=years, n=n)
        else:
            raise ValueError('distr should be one of "norm", "lognorm", "t" or "hist".')
        return scipy.stats.percentileofscore(cagr_distr, score, kind="rank")

    def percentile_history_cagr(self, years: int, percentiles: List[int] = [10, 50, 90]) -> pd.DataFrame:
        """
        Calculate given percentiles for portfolio rolling CAGR distribution from the historical data.

        CAGR - Compound Annual Growth Rate.
        Each percentile is calculated for a period range from 1 year to 'years'.

        Parameters
        ----------
        years: int, default 1
            Max window size for rolling CAGR in the distribution in years.
            It should not exceed 1/2 of the portfolio history period length 'period_length'.

        percentiles: list of int, default [10, 50, 90]
            List of percentiles to be calculated.

        Returns
        -------
        DataFrame
            Table with percentiles values for each period from 1 to 'years'.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='none')
        >>> pf.percentile_history_cagr(years=5, percentiles=[1, 50, 99])
                     1         50        99
        years
        1     -0.231327  0.098693  0.295343
        2     -0.101689  0.091824  0.206471
        3     -0.036771  0.085428  0.157833
        4     -0.007674  0.085178  0.142195
        5      0.030933  0.082865  0.134496
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

    def percentile_wealth_history(self, years: int = 1, percentiles: List[int] = [10, 50, 90]) -> pd.DataFrame:
        """
        Calculate portfolio wealth index percentiles.

        Percentiles are derived from rolling CAGR historical distribution.
        CAGR - Compound Annual Growth Rate.
        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over a given
        time period.

        Actual portfolio wealth is adjusted to the last known historical value (from 'wealth_index'). It is useful
        for a chart with historical wealth index and forecasted values.

        Parameters
        ----------
        years: int, default 1
            Time frame for portfolio wealth index percentiles.
            It should not exceed 1/2 of the portfolio history period length 'period_length'.
            Percentiles are calculated for periods from 1 to 'years'.

        percentiles: list of int, default [10, 50, 90]
            List of percentiles to be calculated.

        Returns
        -------
        DataFrame
            Table with portfolio wealth index percentiles for each period from 1 to 'years'.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='month')
        >>> pf.percentile_wealth_history(years=5)
                        10           50           90
        years
        1      3815.660408  4202.758919  4457.210561
        2      3727.946026  4540.888480  5005.291952
        3      3797.214674  4855.631902  5384.216628
        4      4173.503054  5274.584657  6018.571025
        5      4613.287195  5706.343210  6694.576137
        """
        first_value = self.wealth_index[self.symbol].values[-1]
        percentile_returns = self.percentile_history_cagr(years=years, percentiles=percentiles)
        return first_value * (percentile_returns + 1.0).pow(percentile_returns.index.values, axis=0)

    def _forecast_preparation(self, years: int):
        self._test_forecast_period(years)
        period_months = years * settings._MONTHS_PER_YEAR
        # make periods index where the shape is max_period
        start_period = self.last_date.to_period("M")
        end_period = self.last_date.to_period("M") + period_months - 1
        ts_index = pd.period_range(start_period, end_period, freq="M")
        return period_months, ts_index

    def monte_carlo_returns_ts(self, distr: str = "norm", years: int = 1, n: int = 100) -> pd.DataFrame:
        """
        Generate portfolio monthly rate of return time series with Monte Carlo simulation.

        Monte Carlo simulation generates n random monthly time series with a given distribution.
        Forecast period should not exceed 1/2 of portfolio history period length.

        First date of forecaseted returns is portfolio last_date.

        Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            Distribution type for the rate of return of portfolio.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.

        years : int, default 1
            Forecast period for portfolio monthly rate of return time series.
            It should not exceed 1/2 of the portfolio history period length 'period_length'.

        n : int, default 100
            Number of random rate of return time series to generate with Monte Carlo simulation.

        Returns
        -------
        DataFrame
            Table with n random rate of return monthly time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='month')
        >>> pf.monte_carlo_returns_ts(years=8, distr='norm', n=5000)
                     0         1         2     ...      4997      4998      4999
        2021-07 -0.008383 -0.013167 -0.031659  ...  0.046717  0.065675  0.017933
        2021-08  0.038773 -0.023627  0.039208  ... -0.016075  0.034439  0.001856
        2021-09  0.005026 -0.007195 -0.003300  ... -0.041591  0.021173  0.114225
        2021-10 -0.007257  0.003013 -0.004958  ...  0.037057 -0.009689 -0.003242
        2021-11 -0.005006  0.007090  0.020741  ...  0.026509 -0.023554  0.010271
                   ...       ...       ...  ...       ...       ...       ...
        2029-02 -0.065898 -0.003673  0.001198  ...  0.039293  0.015963 -0.050704
        2029-03  0.021215  0.008783 -0.017003  ...  0.035144  0.002169  0.015055
        2029-04  0.002454 -0.016281  0.017004  ...  0.032535  0.027196 -0.029475
        2029-05  0.011206  0.023396 -0.013757  ... -0.044717 -0.025613 -0.002066
        2029-06 -0.016740 -0.007955  0.002862  ... -0.027956 -0.012339  0.048974
        [96 rows x 5000 columns]
        """
        period_months, ts_index = self._forecast_preparation(years)
        # random returns
        if distr == "norm":
            random_returns = np.random.normal(self.mean_return_monthly, self.risk_monthly.iloc[-1], (period_months, n))
        elif distr == "lognorm":
            std, loc, scale = scipy.stats.lognorm.fit(self.ror)
            random_returns = scipy.stats.lognorm(std, loc=loc, scale=scale).rvs(size=[period_months, n])
        elif distr == "t":
            df, loc, scale = scipy.stats.t.fit(self.ror)
            random_returns = scipy.stats.t(loc=loc, scale=scale, df=df).rvs(size=[period_months, n])
        else:
            raise ValueError('"distr" must be "norm" (default), "lognorm" or "t".')
        return pd.DataFrame(data=random_returns, index=ts_index)

    def monte_carlo_wealth(self, distr: str = "norm", years: int = 1, n: int = 100) -> pd.DataFrame:
        """
        Generate portfolio wealth index with Monte Carlo simulation.

        Monte Carlo simulation generates n random monthly time series.
        Each wealth index is calculated with rate of return time series of a given distribution.

        Forecast period should not exceed 1/2 of portfolio history period length.
        First date of forecasted returns is portfolio last_date.
        First value for the forecasted wealth indexes is the last historical portfolio index value. It is useful
        for a chart with historical wealth index and forecasted values.

        Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            Distribution type for the rate of return of portfolio.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.

        years : int, default 1
            Forecast period for portfolio wealth index time series.
            It should not exceed 1/2 of the portfolio history period length 'period_length'.

        n : int, default 100
            Number of random wealth indexes to generate with Monte Carlo simulation.

        Returns
        -------
        DataFrame
            Table with n random wealth indexes monthly time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='month')
        >>> pf.monte_carlo_wealth(distr='lognorm', years=5, n=1000)
                         0            1    ...          998          999
        2021-07  3895.377293  3895.377293  ...  3895.377293  3895.377293
        2021-08  3869.854680  4004.814981  ...  3874.455244  3935.913516
        2021-09  3811.125717  3993.783034  ...  3648.925159  3974.103856
        2021-10  4053.024519  4232.141143  ...  3870.099003  4082.189688
        2021-11  4179.544897  4156.839698  ...  3899.249696  4097.003962
        2021-12  4237.030690  4351.305114  ...  3916.639721  4042.011774
        """
        validators.validate_distribution(distr)
        return_ts = self.monte_carlo_returns_ts(distr=distr, years=years, n=n)
        first_value = self.wealth_index[self.symbol].values[-1]
        return helpers.Frame.get_wealth_indexes(return_ts, first_value)

    def _get_cagr_distribution(
        self,
        distr: str = "norm",
        years: int = 1,
        n: int = 100,
    ) -> pd.Series:
        """
        Generate CAGR distribution for the rate of return distribution of a given type.

        CAGR is calculated for each of n random returns time series.
        """
        validators.validate_distribution(distr)
        return_ts = self.monte_carlo_returns_ts(distr=distr, years=years, n=n)
        return helpers.Frame.get_cagr(return_ts)

    def percentile_distribution_cagr(
        self,
        distr: str = "norm",
        years: int = 1,
        percentiles: List[int] = [10, 50, 90],
        n: int = 10000,
    ) -> Dict[int, float]:
        """
        Calculate percentiles for a given CAGR distribution.

        CAGR - Compound Annual Growth Rate.
        CAGR is calculated for each of n random returns time series of a given distribution. Random time series are
        generated with Monte Carlo simulation.
        CAGR time frame should not exceed 1/2 of portfolio history period length.

        Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            Distribution type for the rate of return of portfolio.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.

        years: int, default 1
            Time frame for portfolio CAGR.
            It should not exceed 1/2 of the portfolio history period length 'period_length'.

        percentiles: list of int, default [10, 50, 90]
            List of percentiles to be calculated.

        n : int, default 10000
            Number of random time series to generate with Monte Carlo simulation.
            Larger argument values can be used to increase the precision of the calculation. But this will lead
            to slower performance.

        Returns
        -------
        dict
            Dictionary {Percentile: value}

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='year')
        >>> pf.percentile_distribution_cagr()
        {10: -0.0329600265453808, 50: 0.08247141141668779, 90: 0.21338327078214836}
        Forecast CAGR according to normal distribution within 1 year period.
        >>> pf.percentile_distribution_cagr(years=5)
        {10: 0.030625112922274055, 50: 0.08346815557550402, 90: 0.13902575176654647}
        Forecast CAGR according to normal distribution within 5 year period.
        """
        validators.validate_distribution(distr)
        cagr_distr = self._get_cagr_distribution(distr=distr, years=years, n=n)
        results = {}
        for percentile in percentiles:
            value = cagr_distr.quantile(percentile / 100)
            results.update({percentile: value})
        return results

    def percentile_wealth(
        self,
        distr: str = "norm",
        years: int = 1,
        percentiles: List[int] = [10, 50, 90],
        today_value: Optional[int] = None,
        n: int = 1000,
    ) -> Dict[int, float]:
        """
        Calculate percentiles for portfolio wealth indexes distribution.

        Portfolio wealth indexes are derived from the rate of return time series of a given distribution type.

        Parameters
        ----------
        distr : {'hist', 'norm', 'lognorm', 't'}, default 'norm'
            Distribution type for the rate of return of portfolio.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's t distribution.
            'hist' - percentiles are taken from the historical data.

        years : int, default 1
            Investment period length to calculate wealth index.
            It should not exceed 1/2 of the portfolio history period length 'period_length'.

        percentiles : list of int, default [10, 50, 90]
            List of percentiles to be calculated.

        today_value :  int, optional
            Initial value of the wealth index.
            If today_value is None the last value of the historical wealth indexes is taken. It can be useful to plot
            the forecast of wealth index togeather with the hitorical data.

        n : int, default 1000
            Number of random time series to generate with Monte Carlo simulation (for 'norm' or 'lognorm' only).
            Larger argument values can be used to increase the precision of the calculation. But this will lead
            to slower performance.
            Is not required for historical distribution (dist='hist').

        Returns
        -------
        dict
            Dictionary {Percentile: value}

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='year')
        >>> pf.percentile_wealth(distr='hist', years=5, today_value=1000, n=5000)
        {10: 1228.3741255659957, 50: 1491.7857161011104, 90: 1745.1130920663286}
        Percentiles values for the wealth index 5 years forecast if the initial value is 1000.
        """
        # TODO: check "today value"
        if distr == "hist":
            results = self.percentile_wealth_history(years=years, percentiles=percentiles).iloc[-1].to_dict()
        elif distr in ["norm", "lognorm", "t"]:
            results = {}
            wealth_indexes = self.monte_carlo_wealth(distr=distr, years=years, n=n)
            for percentile in percentiles:
                value = wealth_indexes.iloc[-1, :].quantile(percentile / 100)
                results.update({percentile: value})
        else:
            raise ValueError('distr should be "norm", "lognorm", "t" or "hist".')
        if today_value:
            modifier = today_value / self.wealth_index[self.symbol].values[-1]
            results.update((x, y * modifier) for x, y in results.items())
        return results

    # distributions
    @property
    def skewness(self) -> pd.Series:
        """
        Compute expanding skewness time series for portfolio rate of return.

        For normally distributed data, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.

        Returns
        -------
        Series
            Expanding skewness time series

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.skewness
        Date
        2008-05   -0.134193
        2008-06   -0.022349
        2008-07    0.081412
        2008-08   -0.020978
                     ...
        2021-04    0.441430
        2021-05    0.445772
        2021-06    0.437383
        2021-07    0.425247
        Freq: M, Name: portfolio_8378.PF, Length: 159, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.skewness.plot()
        >>> plt.show()
        """
        return helpers.Frame.skewness(self.ror)

    def skewness_rolling(self, window: int = 60):
        """
        Compute rolling skewness of the return time series.

        For normally distributed rate of return, the skewness should be about zero.
        A skewness value greater than zero means that there is more weight in the right tail of the distribution.

        Parameters
        ----------
        window : int, default 60
            Size of the moving window in months.
            The window size should be at least 12 months.

        Returns
        -------
        Series
            Expanding skewness time series

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.skewness_rolling(window=12*10)
        Date
        2017-04    0.464916
        2017-05    0.446095
        2017-06    0.441211
        2017-07    0.453947
        2017-08    0.464805
        ...
        2021-02    0.007622
        2021-03    0.000775
        2021-04    0.002308
        2021-05    0.022543
        2021-06   -0.006534
        2021-07   -0.012192
        Freq: M, Name: portfolio_8378.PF, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.skewness_rolling(window=12*10).plot()
        >>> plt.show()
        """
        return helpers.Frame.skewness_rolling(self.ror, window=window)

    @property
    def kurtosis(self):
        """
        Calculate expanding Fisher (normalized) kurtosis time series for portfolio rate of return.

        Kurtosis is a measure of whether the rate of return are heavy-tailed or light-tailed
        relative to a normal distribution.
        It should be close to zero for normally distributed rate of return.
        Kurtosis is the fourth central moment divided by the square of the variance.

        Returns
        -------
        Series
            Expanding kurtosis time series

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.kurtosis
        Date
        2008-05   -0.815206
        2008-06   -0.718330
        2008-07   -0.610741
        2008-08   -0.534105
                     ...
        2021-04    2.821322
        2021-05    2.855267
        2021-06    2.864717
        2021-07    2.850407
        Freq: M, Name: portfolio_4411.PF, Length: 159, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.kurtosis.plot()
        >>> plt.show()
        """
        return helpers.Frame.kurtosis(self.ror)

    def kurtosis_rolling(self, window: int = 60):
        """
        Calculate rolling Fisher (normalized) kurtosis time series for portfolio rate of return.

        Kurtosis is a measure of whether the rate of return are heavy-tailed or light-tailed
        relative to a normal distribution.
        It should be close to zero for normally distributed rate of return.
        Kurtosis is the fourth central moment divided by the square of the variance.

        Parameters
        ----------
        window : int, default 60
            Size of the moving window in months.
            The window size should be at least 12 months.

        Returns
        -------
        Series
            Expanding kurtosis time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.kurtosis_rolling(window=12*10)
        Date
        2017-04    4.041599
        2017-05    4.133518
        2017-06    4.165099
        2017-07    4.205125
        2017-08    4.313773
        ...
        2021-03    0.362184
        2021-04    0.409680
        2021-05    0.455760
        2021-06    0.457315
        2021-07    0.496168
        Freq: M, Name: portfolio_4411.PF, dtype: float64

        >>> import matplotlib.pyplot as plt
        >>> pf.kurtosis_rolling(window=12*10).plot()
        >>> plt.show()
        """
        return helpers.Frame.kurtosis_rolling(self.ror, window=window)

    @property
    def jarque_bera(self) -> Dict[str, float]:
        """
        Perform Jarque-Bera test for normality of portfolio returns time series.

        Jarque-Bera shows whether the returns have the skewness and kurtosis
        matching a normal distribution (null hypothesis or H0).

        Returns
        -------
        dict
            Jarque-Bera test statistics and p-value.

        Notes
        -----
        Test returns statistics (first row) and p-value (second row).
        p-value is the probability of obtaining test results, under the assumption that the null hypothesis is correct.
        In general, a large Jarque-Bera statistics and tiny p-value indicate that null hypothesis is rejected
        and the time series are not normally distributed.

        Examples
        --------
        >>> pf = ok.Portfolio(['BND.US'])
        >>> pf.jarque_bera
        {'statistic': 58.27670538027455, 'p-value': 2.2148949341271873e-13}
        """
        return helpers.Frame.jarque_bera_series(self.ror)

    def kstest(self, distr: str = "norm") -> Dict[str, float]:
        """
        Perform one sample Kolmogorov-Smirnov test on portfolio returns and evaluate goodness of fit
        for a given distribution.

        The one-sample Kolmogorov-Smirnov test compares the rate of return time series against a given distribution.

        Returns
        -------
        dict
            Kolmogorov-Smirnov test statistics and p-value.

        Parameters
        ----------
        distr : {'norm', 'lognorm'}, default 'norm'
            The name of a distribution to fit.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.


        Notes
        -----
        Like in Jarque-Bera test returns statistic (first row) and p-value (second row).
        Null hypotesis (two distributions are similar) is not rejected when p-value is high enough.
        5% threshold can be used.

        Examples
        --------
        >>> pf = ok.Portfolio(['GLD.US'])
        >>> pf.kstest(distr='lognorm')
        {'statistic': 0.05001344986084533, 'p-value': 0.6799422889377373}

        >>> pf.kstest(distr='norm')
        {'statistic': 0.09528000069992831, 'p-value': 0.047761781235967415}

        Kolmogorov-Smirnov test shows that GLD rate of return time series fits lognormal distribution
        better than normal one.
        """
        return helpers.Frame.kstest_series(self.ror, distr=distr)

    def get_sharpe_ratio(self, rf_return: float = 0) -> float:
        """
        Calculate Sharpe ratio.

        The Sharpe ratio is the average annual return in excess of the risk-free rate
        per unit of risk (annualized standard deviation).

        Risk-free rate should be taken according to the Portfolio base currency.

        Parameters
        ----------
        rf_return : float, default 0
            Risk-free rate of return.

        Returns
        -------
        float

        Examples
        --------
        >>> pf = ok.Portfolio(['VOO.US', 'BND.US'], weights=[0.40, 0.60])
        >>> pf.get_sharpe_ratio(rf_return=0.04)
        0.7412193684695373
        """
        return ratios.get_sharpe_ratio(
            pf_return=self.mean_return_annual,
            rf_return=rf_return,
            std_deviation=self.risk_annual.iloc[-1],
        )

    def get_sortino_ratio(self, t_return: float = 0) -> float:
        """
        Calculate Sortino ratio for the portfolio with specified target return.

        Sortion ratio measures the risk-adjusted return of portfolio. It is a modification of the Sharpe ratio
        but penalizes only those returns falling below a specified target rate of return, while
        the Sharpe ratio penalizes both upside and downside volatility equally.

        Parameters
        ----------
        t_return : float, default 0
            Traget rate of return.

        Returns
        -------
        float

        Examples
        --------
        >>> pf = ok.Portfolio(['VOO.US', 'BND.US'], last_date='2021-12')
        >>> pf.get_sortino_ratio(t_return=0.02)
        1.4377728903230174
        """
        semideviation = helpers.Frame.get_below_target_semideviation(ror=self.ror, t_return=t_return) * 12**0.5
        return ratios.get_sortino_ratio(
            pf_return=self.mean_return_annual,
            t_return=t_return,
            semi_deviation=semideviation,
        )

    @property
    def diversification_ratio(self) -> float:
        """
        Calculate Diversification Ratio for the portfolio.

        The Diversification Ratio is the ratio of the weighted average of assets risks divided by the portfolio risk.
        In this case risk is the annuilized standatd deviation for the rate of return .

        Returns
        -------
        float

        Examples
        --------
        >>> pf = ok.Portfolio(['VOO.US', 'BND.US'], weights=[0.7, 0.3], last_date='2021-12')
        >>> pf.diversification_ratio
        1.1264305597257505
        """
        assets_risk = self.assets_ror.std()
        assets_mean_return = self.assets_ror.mean()
        assets_annualized_risk = helpers.Float.annualize_risk(assets_risk, assets_mean_return)
        weights = np.asarray(self.weights)
        sigma_weighted_sum = weights.T @ assets_annualized_risk
        return sigma_weighted_sum / self.risk_annual.iloc[-1]

    def plot_percentiles_fit(self, distr: str = "norm", figsize: Optional[tuple] = None) -> None:
        """
        Generate a quantile-quantile (Q-Q) plot of portfolio monthly rate of return against quantiles of a given
        theoretical distribution.

        A q-q plot is a plot of the quantiles of the portfolio rate of return historical data
        against the quantiles of a given theoretical distribution.

        Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            The name of a distribution to fit.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.


        figsize : (float, float), optional
            Width and height of plot in inches.
            If None default matplotlib figsize value is used.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='year')
        >>> pf.plot_percentiles_fit(distr='lognorm')
        >>> plt.show()
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
        elif distr == "t":
            scipy.stats.probplot(
                self.ror,
                sparams=(scipy.stats.t.fit(self.ror)),
                dist=scipy.stats.t,
                plot=plt,
            )
        else:
            raise ValueError('distr should be "norm" (default) or "lognorm".')
        plt.show()

    def plot_hist_fit(self, distr: str = "norm", bins: int = None) -> None:
        """
        Plot historical distribution histogram for ptrtfolio monthly rate of return time series
        and theoretical PDF (Probability Distribution Function).

        Can be used with Normal, Lognormal and Stident's T distributions.

        Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            The name of a distribution to fit.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SP500TR.INDX'])
        >>> pf.plot_hist_fit(distr='norm')
        >>> plt.show()
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
        elif distr == "t":
            df, loc, scale = scipy.stats.t.fit(data)
            p = scipy.stats.t.pdf(x, loc=loc, scale=scale, df=df)
        else:
            raise ValueError('distr must be "norm" (default) or "lognorm".')
        plt.plot(x, p, "k", linewidth=2)
        title = "Fit results: mu = %.3f,  std = %.3f" % (mu, std)
        plt.title(title)
        plt.show()

    def plot_forecast(
        self,
        distr: str = "norm",
        years: int = 5,
        percentiles: List[int] = [10, 50, 90],
        today_value: Optional[int] = None,
        n: int = 1000,
        figsize: Optional[tuple] = None,
    ) -> plt.axes:
        """
        Plot forecasted ranges of wealth indexes (lines) for a given set of percentiles.
        Historical wealth index is shown in the same chart.

        Parameters
        ----------
        distr : {'hist', 'norm', 'lognorm', 't'}, default 'norm'
            Distribution type for the rate of return of portfolio.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's t distribution.
            'hist' - percentiles are taken from the historical data.

        years : int, default 1
            Investment period length to calculate wealth index.
            It should not exceed 1/2 of the portfolio history period length 'period_length'.

        percentiles : list of int, default [10, 50, 90]
            List of percentiles to be calculated.

        today_value :  int, optional
            Initial value of the wealth index.
            If today_value is None the last value of the historical wealth indexes is taken. It can be useful to plot
            the forecast of wealth index togeather with the hitorical data.

        n : int, default 1000
            Number of random time series to generate with Monte Carlo simulation (for 'norm' or 'lognorm' only).
            Larger argument values can be used to increase the precision of the calculation. But this will lead
            to slower performance.
            Is not required for historical distribution (dist='hist').

        Returns
        -------
        Axes : 'matplotlib.axes._subplots.AxesSubplot'

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='year')
        >>> pf.plot_forecast()
        >>> plt.show()
        """
        wealth = self.wealth_index
        x1 = self.last_date
        x2 = x1.replace(year=x1.year + years)
        y_start_value = wealth[self.symbol].iloc[-1]
        y_end_values = self.percentile_wealth(distr=distr, years=years, percentiles=percentiles, n=n)
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
    ) -> None:
        """
        Plot Monte Carlo simulation for portfolio wealth indexes together with historical wealth index.

        Random wealth indexes are generated according to a given distribution.

        Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            Distribution type for the rate of return of portfolio.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.

        years : int, default 1
            Investment period length for new wealth indexes
            It should not exceed 1/2 of the portfolio history period length 'period_length'.

        n : int, default 20
            Number of random wealth indexes to generate with Monte Carlo simulation.

        figsize : (float, float), optional
            Width, height in inches.
            If None default matplotlib figsize value is used.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'],
        ...                    weights=[.60, .35, .05],
        ...                    rebalancing_period='year')
        >>> pf.plot_forecast_monte_carlo(years=5, distr='lognorm', n=100)
        >>> plt.show()
        """
        s1 = self.wealth_index
        s2 = self.monte_carlo_wealth(distr=distr, years=years, n=n)
        s1[self.symbol].plot(legend=None, figsize=figsize)
        for n in s2:
            s2[n].plot(legend=None)


class PortfolioDCF:
    """
    Class to access discounted cash flow (DCF) methods of Portfolio.
    All methods can be used in Portfolio instances trough construction:
    ```
    pf = Portfolio()
    pf.dcf.weatlh_index
    pf.dсf.cashflow_pv
    ```
    """

    def __init__(self,
                 parent: Portfolio,
                 discount_rate: Optional[float] = None):
        self.parent = parent
        self.discount_rate = discount_rate
        self._wealth_index = pd.DataFrame(dtype=float)
        self._monte_carlo_wealth = pd.DataFrame(dtype=float)
        self.mc = MonteCarlo(self)
        self.cashflow_parameters = CashFlow(self)

    @property
    def discount_rate(self) -> float:
        """
        Portfolio cash flow discount rate.

        Returns
        -------
        float
            Cash flow discount rate.
        """
        return self._discount_rate

    @discount_rate.setter
    def discount_rate(self, discount_rate: Optional[float]):
        if discount_rate is None and hasattr(self.parent, "inflation"):
            self._discount_rate = self.parent.get_cagr().loc[self.parent.inflation]
        elif discount_rate is None and not hasattr(self.parent, "inflation"):
            self._discount_rate = settings.DEFAULT_DISCOUNT_RATE
        else:
            validators.validate_real("discount rate", discount_rate)
            self._discount_rate = discount_rate

    def set_mc_parameters(self,
                          distribution: str,
                          period: int,
                          number: int):
        self.mc.distribution = distribution
        self.mc.period = period
        self.mc.number = number

    def set_cashflow_parameters(self,
                                initial_investment: float,
                                method: str,
                                frequency: Optional[int] = None,
                                amount: int = None,
                                indexation: float = None,
                                percentage: float = None,
                                ):
        self.cashflow_parameters.initial_investment = initial_investment
        self.cashflow_parameters.method = method
        self.cashflow_parameters.frequency = frequency
        # fixed amount
        self.cashflow_parameters.amount = amount
        self.cashflow_parameters.indexation = indexation
        # fixed percentage
        self.cashflow_parameters.percentage = percentage

    @property
    def wealth_index(self) -> pd.DataFrame:
        """
        Calculate wealth index time series for the portfolio with contributions and
        withdrawals.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        historical time period considering cash flows.

        Accumulated inflation time series is added if `inflation=True` in the Portfolio.

        If there are no cashflows, Wealth index is obtained from the accumulated return multiplicated
        by the initial investments. That is: initial_amount_pv * (Acc_Return + 1)

        initial_amount_pv is the discounted value of the initial investments (initial_amount).

        Values of the wealth index correspond to the beginning of the month.

        Returns
        -------
            Time series of wealth index values for portfolio and accumulated inflation.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = ok.Portfolio(['SPY.US', 'BND.US'])
        >>> x.dcf.wealth_index.plot()
        >>> plt.show()
        """
        if self._wealth_index.empty:
            df = self.parent._add_inflation()
            infl = self.parent.inflation if hasattr(self.parent, "inflation") else None
            df = helpers.Frame.get_wealth_indexes_with_cashflow(
                ror=df,
                portfolio_symbol=self.parent.symbol,
                inflation_symbol=infl,
                cashflow_parameters=self.cashflow_parameters
            )
            self._wealth_index = self.parent._make_df_if_series(df)
        return self._wealth_index

    @property
    def survival_period_hist(self) -> float:
        """
        Calculate the period when the portfolio has positive balance considering withdrawals on the historical data.

        The portfolio survival period (longevity period) depends on the investment strategy: asset allocation,
        rebalancing, withdrawals rate etc.

        The withdrawals are defined by the `cashflow` parameter of the portfolio.

        Returns
        -------
        float
            The portfolio survival period (longevity period) in years.

        Examples
        --------
        >>> pf = ok.Portfolio(
        ...    ['SPY.US', 'AGG.US'],
        ...    ccy='USD',
        ...    first_date='2010-01',
        ...    initial_amount=100_000,
        ...    cashflow=-1_000
        ...)
        >>> pf.dcf.survival_period_hist
        11.6
        """
        return helpers.Date.get_period_length(last_date=self.survival_date_hist, first_date=self.parent.first_date)

    @property
    def survival_date_hist(self) -> pd.Timestamp:
        """
        Get the date when the portfolio balance become negative considering withdrawals on the historical data.

        The portfolio survival date (longevity date) depends on the investment strategy: asset allocation,
        rebalancing, withdrawals rate etc.

        The withdrawals are defined by the `cashflow` parameter of the portfolio.

        Returns
        -------
        pd.Timestamp
            The portfolio survival date (longevity period) in years.

        Examples
        --------
        >>> pf = ok.Portfolio(
        ...    ['SPY.US', 'AGG.US'],
        ...    ccy='USD',
        ...    first_date='2010-01',
        ...    initial_amount=100_000,
        ...    cashflow=-1_000
        ...)
        >>> pf.dcf.survival_date_hist
        Timestamp('2021-08-01 00:00:00')
        """
        return helpers.Frame.get_survival_date(self.wealth_index.loc[:, self.parent.symbol])

    @property
    def amount_pv(self) -> Optional[float]:
        """
        Calculate the discounted value (PV) of the contributions/withdrawals value at the historical first date.

        The future value (FV) is defined by `amount` parameter.

        Returns
        -------
        float
            The discounted value (PV) of the contributions/withdrawals at the historical first date.

        Examples
        --------
        >>> pf = ok.Portfolio(['EQMX.MOEX', 'SBGB.MOEX'], ccy='RUB')
        >>> pf.dcf.amount
        73650
        """
        return self.cashflow_parameters.amount / (1.0 + self.discount_rate) ** self.parent.period_length

    @property
    def initial_investment_pv(self) -> Optional[float]:
        """
        Calculate the discounted value (PV) of the initial investments at the historical first date.

        The future value (FV) is defined by `initial_amount` parameter.
        The discount rate is the average annual inflation.

        Returns
        -------
        float
            The discounted value (PV) of the initial investments at the historical first date.

        Examples
        --------
        >>> pf = ok.Portfolio(['EQMX.MOEX', 'SBGB.MOEX'], ccy='RUB', initial_amount=100_000)
        >>> pf.dcf.initial_investment_pv
        73650
        """
        return self.cashflow_parameters.initial_investment / (1.0 + self.discount_rate) ** self.parent.period_length

    @property
    def initial_investment_fv(self) -> Optional[float]:
        """
        Calculate the discounted value (PV) of the initial investments at the historical first date.

        The future value (FV) is defined by `initial_amount` parameter.
        The discount rate is the average annual inflation.

        Returns
        -------
        float
            The discounted value (PV) of the initial investments at the historical first date.

        Examples
        --------
        >>> pf = ok.Portfolio(['EQMX.MOEX', 'SBGB.MOEX'], ccy='RUB', initial_amount=100_000)
        >>> pf.dcf.initial_investment_pv
        73650
        """
        return self.cashflow_parameters.initial_investment * (1.0 + self.discount_rate) ** self.mc.period

    @property
    def cashflow_pv(self) -> Optional[float]:
        """
        Calculate the discounted value (PV) of the cash flow amount at the historical first date.

        The future value (FV) is defined by `cashflow` parameter.
        The discount rate is the average annual inflation.

        Returns
        -------
        float
            The discounted value (PV) of the cash flow amount at the historical first date.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US'], ccy='USD', initial_amount=100_000, cashflow=-5_000)
        >>> pf.dcf.cashflow_pv
        -3004
        """
        return self.cashflow_parameters.amount / (1.0 + self.cashflow_parameters.indexation) ** self.parent.period_length

    @property
    def monte_carlo_wealth(self) -> pd.DataFrame:
        """
        Generate portfolio wealth indexes with cash flows (withdrawals/contributions) by Monte Carlo simulation.

        Monte Carlo simulation generates n random monthly time series.
        Each wealth index is calculated with rate of return time series of a given distribution.

        First date of forecasted returns is portfolio last_date.
        First value for the forecasted wealth indexes is the last historical portfolio index value. It is useful
        for a chart with historical wealth index and forecasted values.

        Returns
        -------
        DataFrame
            Table with n random wealth indexes monthly time series.

        Examples
        --------
        >>> pf = ok.Portfolio(['SPY.US', 'AGG.US', 'GLD.US'], weights=[.60, .35, .05], rebalancing_period='month')
        >>> pf.dcf.monte_carlo_wealth
                         0            1    ...          998          999
        2021-07  3895.377293  3895.377293  ...  3895.377293  3895.377293
        2021-08  3869.854680  4004.814981  ...  3874.455244  3935.913516
        2021-09  3811.125717  3993.783034  ...  3648.925159  3974.103856
        2021-10  4053.024519  4232.141143  ...  3870.099003  4082.189688
        2021-11  4179.544897  4156.839698  ...  3899.249696  4097.003962
        2021-12  4237.030690  4351.305114  ...  3916.639721  4042.011774
        """

        if self._monte_carlo_wealth.empty:
            return_ts = self.parent.monte_carlo_returns_ts(distr=self.mc.distribution,
                                                           years=self.mc.period,
                                                           n=self.mc.number)
            df = return_ts.apply(
                helpers.Frame.get_wealth_indexes_with_cashflow,
                axis=0,
                args=(
                    None,  # portfolio_symbol
                    None,  #  inflation_symbol
                    self.cashflow_parameters
                ),
            )

            def remove_negative_values(s):
                condition = s <= 0
                try:
                    survival_date = s[condition].index[0]
                    s[survival_date] = 0
                    s[s.index > survival_date] = np.nan
                except IndexError:
                    pass
                return s

            df = df.apply(remove_negative_values, axis=0)
            all_cells_are_nan = df.isna().all(axis=1)
            self._monte_carlo_wealth = df[~all_cells_are_nan]
        return self._monte_carlo_wealth

    @property
    def monte_carlo_wealth_pv(self) -> pd.DataFrame:
        #TODO: add method for discounted cashflow values
        pass

    def plot_forecast_monte_carlo(
        self,
        backtest: bool = True,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        Plot Monte Carlo simulation for portfolio future wealth indexes optionally together with historical wealth index.

        Wealth index (Cumulative Wealth Index) is a time series that presents the value of portfolio over
        time period considering cash flows (portfolio withdrawals/contributions).

        Random wealth indexes are generated according to a given distribution.

        Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            The type of a distribution to generate random portfolios.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.

        years : int, default 1
            Investment period length for new wealth indexes

        backtest : bool, default 'True'
            Include historical wealth index if 'True'.

        n : int, default 20
            Number of random wealth indexes to generate with Monte Carlo simulation.

        figsize : (float, float), optional
            Width, height in inches.
            If None default matplotlib figsize value is used.

        Returns
        -------
        None

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> pf = ok.Portfolio(assets=['SPY.US', 'AGG.US', 'GLD.US'],
        ...                   weights=[.60, .35, .05],
        ...                   rebalancing_period='year',
        ...                   initial_amount=300_000,
        ...                   cashflow=-1_000)
        >>> pf.dcf.plot_forecast_monte_carlo(backtest=True)
        >>> plt.show()
        """
        if backtest:
            s1 = self.wealth_index[self.parent.symbol]
            s1.plot(legend=None, figsize=figsize)
            last_backtest_value = s1.iloc[-1]
            if last_backtest_value > 0:
                self.cashflow_parameters.initial_investment = last_backtest_value
                s2 = self.monte_carlo_wealth
                for s in s2:
                    s2[s].plot(legend=None)
        else:
            s2 = self.monte_carlo_wealth
            s2.plot(legend=None)

    def monte_carlo_survival_period(
        self,
        threshold: float
    ) -> pd.Series:
        """
        Generate a survival period distribution for a portfolio with cash flows by Monte Carlo simulation.

        It's possible to analyze the distribution finding "min", "max" and percentiles to see for how long will last
        the investment strategy - possible longevity period.

        Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            The type of a distribution to generate random portfolios.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.

        years : int, default 1
            Forecast period for portfolio wealth index time series.

        n : int, default 100
            Number of random wealth indexes to generate with Monte Carlo simulation.

        Returns
        -------
        Series
            Survival period distribution for a portfolio with cash flows.

        Examples
        --------
        >>> pf = ok.Portfolio(
            ['SPY.US', 'AGG.US', 'GLD.US'],
            weights=[.60, .35, .05],
            rebalancing_period='year',
            initial_amount=300_000,
            cashflow=-10_000
        )
        >>> s = pf.dcf.monte_carlo_survival_period(
            distr="norm",
            years=10,
            n=100,
        )
        >>> s.min()
        2.2
        >>> s.max()
        3.6
        >> s.mean()
        2.737
        >> s.quantile(50 / 100)
        2.7
        """
        s2 = self.monte_carlo_wealth
        dates: pd.Series = helpers.Frame.get_survival_date(s2, self.discount_rate, threshold)
        return dates.apply(helpers.Date.get_period_length, args=(self.parent.last_date,))

    def find_the_largest_withdrawals_size(
            self,
            min_amount: float,
            max_amount: float,
            withdrawal_steps: int,
            confidence_level: float,
            goal: str,
            target_survival_period: int = 25
    ):
        amount_range = np.linspace(min_amount, max_amount, withdrawal_steps)
        saved_amount = self.cashflow_parameters.amount
        max_withdrawal = 0
        for a in amount_range:
            # print("testing:", a)
            self.cashflow_parameters.amount = a
            sp_at_quantile = self.monte_carlo_survival_period.quantile(confidence_level)
            if goal == "maintain_balance":
                wealth_at_quantile = self.monte_carlo_wealth.iloc[-1, :].quantile(confidence_level)
                # print(f"{wealth_at_quantile=}")
                condition = (sp_at_quantile == self.mc.period) and (wealth_at_quantile >= self.initial_investment_fv)
                if condition:
                    max_withdrawal = a
                    break
            elif goal == "survival_period":
                condition = sp_at_quantile >= target_survival_period
                if condition:
                    max_withdrawal = a
                    break

        self.cashflow_parameters.amount = saved_amount
        # print(f"{max_withdrawal=}")
        return max_withdrawal


class MonteCarlo:
    """
    Parameters
        ----------
        distr : {'norm', 'lognorm', 't'}, default 'norm'
            Distribution type for the rate of return of portfolio.
            'norm' - for normal distribution.
            'lognorm' - for lognormal distribution.
            't' - for Student's T distribution.

        years : int, default 1
            Forecast period for portfolio wealth index time series.
            It should not exceed 1/2 of the portfolio history period length 'period_length'.

        n : int, default 100
            Number of random wealth indexes to generate with Monte Carlo simulation.
    """
    def __init__(self, parent: PortfolioDCF):
        self.parent = parent
        self._distribution: str = "norm"
        self._period: int = 25
        self._mc_number: int = 100

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        validators.validate_distribution(distribution)
        self.parent._monte_carlo_wealth = pd.DataFrame()
        self._distribution = distribution

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, period):
        validators.validate_integer("period", period)
        self.parent._monte_carlo_wealth = pd.DataFrame()
        self._period = period

    @property
    def number(self):
        return self._mc_number

    @number.setter
    def number(self, mc_number):
        validators.validate_integer("mc_number", mc_number)
        self.parent._monte_carlo_wealth = pd.DataFrame()
        self._mc_number = mc_number

    def clear_wealth_data(self):
        self.parent._monte_carlo_wealth = pd.DataFrame()


class CashFlow:
    def __init__(self, parent: PortfolioDCF):
        self.parent = parent
        self.initial_investment: float = 1000.
        self.method: str = "fixed_amount"
        self.frequency: str = "quarter"
        self._pandas_frequency = settings.frequency_mapping.get(self.frequency)
        # fixed amount
        self.amount: Optional[float] = None
        self.indexation: Optional[Union[float, str]] = None
        # fixed percentage
        self.percentage: Optional[float] = None
        # predefined cashflow series
        self.time_series: Optional[dict] = None

    methods_list = ["fixed_amount", "fixed_percentage", "time_series"]
    amount_default_value = 0

    @property
    def initial_investment(self):
        return self._initial_investment

    @initial_investment.setter
    def initial_investment(self, initial_investment):
        validators.validate_real("initial_investment", initial_investment)
        self.parent._monte_carlo_wealth = pd.DataFrame()
        self.parent._wealth_index = pd.DataFrame()
        self._initial_investment = initial_investment

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, method):
        if method in self.methods_list:
            self.parent._monte_carlo_wealth = pd.DataFrame()
            self.parent._wealth_index = pd.DataFrame()
            self._method = method
        else:
            raise ValueError(f"method must be in {self.methods_list}")

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        if frequency in settings.frequency_mapping.keys():
            self.parent._monte_carlo_wealth = pd.DataFrame()
            self.parent._wealth_index = pd.DataFrame()
            self._frequency = frequency
        else:
            raise ValueError(f"frequency must be in {settings.frequency_mapping.keys()}")

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, amount):
        self.parent._monte_carlo_wealth = pd.DataFrame()
        self.parent._wealth_index = pd.DataFrame()
        if self.method == "fixed_amount":
            self._amount = amount if amount else self.amount_default_value
        else:
            self._amount = None

    @property
    def indexation(self) -> float:
        """
        Portfolio cash flow indexation rate.

        Returns
        -------
        float
            Cash flow indexation rate.
        """
        return self._indexation

    @indexation.setter
    def indexation(self, indexation: Optional[float]):
        if indexation in [None, "inflation"] and hasattr(self.parent.parent, "inflation"):
            self._indexation = self.parent.parent.get_cagr().loc[self.parent.parent.inflation]
        elif indexation == "inflation" and not hasattr(self.parent.parent, "inflation"):
            raise ValueError("There is no information about historical inflation. Set inflation=True to calculate.")
        elif indexation is None and not hasattr(self.parent.parent, "inflation"):
            self._indexation = settings.DEFAULT_DISCOUNT_RATE
        else:
            validators.validate_real("indexation", indexation)
            self._indexation = indexation
