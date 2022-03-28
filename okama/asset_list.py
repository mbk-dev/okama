from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd

import okama.common.helpers.ratios as ratios
from okama.common.helpers import helpers
from okama.common import make_asset_list


class AssetList(make_asset_list.ListMaker):
    """
    The list of financial assets implementation.

    AssetList can include stocks, ETF, mutual funds, commodities, currencies and stock indexes (benchmarks).

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

    inflation : bool, default True
        Defines whether to take inflation data into account in the calculations.
        Including inflation could limit available data (last_date, first_date)
        as the inflation data is usually published with a one-month delay.
        With inflation = False some properties like real return are not available.
    """

    def __repr__(self):
        dic = {
            "assets": self.symbols,
            "currency": self._currency.ticker,
            "first_date": self.first_date.strftime("%Y-%m"),
            "last_date": self.last_date.strftime("%Y-%m"),
            "period_length": self._pl_txt,
            "inflation": self.inflation if hasattr(self, "inflation") else "None",
        }
        return repr(pd.Series(dic))

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

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = ok.AssetList(['SPY.US', 'BND.US'])
        >>> x.wealth_indexes.plot()
        >>> plt.show()
        """
        df = self._add_inflation()
        return helpers.Frame.get_wealth_indexes(df)

    @property
    def risk_monthly(self) -> pd.Series:
        """
        Calculate monthly risk (standard deviation of return) for each asset.

        Monthly risk of the asset is a standard deviation of the rate of return time series.
        Standard deviation (sigma σ) is normalized by N-1.

        Monthly risk is calculated for rate of retirun time series for the sample from 'first_date' to
        'last_date'.

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
        return self.assets_ror.std()

    @property
    def risk_annual(self) -> pd.Series:
        """
        Calculate annualized risks (standard deviation) for each asset.

        Annualized risk is calculated for rate of retirun time series for the sample from 'first_date' to
        'last_date'.

        Returns
        -------
        Series
            Annualized risk (standard deviation) values for each asset in form of Series.

        See Also
        --------
        risk_monthly : Calculate montly risk for each asset.
        risk_annual : Calculate annualized risks.
        semideviation_monthly : Calculate semideviation monthly values.
        semideviation_annual : Calculate semideviation annualized values.
        get_var_historic : Calculate historic Value at Risk (VaR).
        get_cvar_historic : Calculate historic Conditional Value at Risk (CVaR).
        drawdowns : Calculate drawdowns.

        Notes
        -----
        CFA recomendations are used to annualize risk values [1]_.

        .. [1] `What’s Wrong with Multiplying by the Square Root of Twelve. <https://www.cfainstitute.org/en/research/cfa-digest/2013/11/whats-wrong-with-multiplying-by-the-square-root-of-twelve-digest-summary>`_ Paul D. Kaplan, CFA Institute Journal Review, 2013

        Examples
        --------
        >>> al = ok.AssetList(['GC.COMM', 'SHV.US'], ccy='USD', last_date='2021-01')
        >>> al.risk_annual
        GC.COMM    0.195236
        SHV.US     0.004960
        dtype: float64
        """
        risk = self.assets_ror.std()
        mean_return = self.assets_ror.mean()
        return helpers.Float.annualize_risk(risk, mean_return)

    @property
    def semideviation_monthly(self) -> pd.Series:
        """
        Calculate semi-deviation monthly values for each asset.

        Semi-deviation (Downside risk) is the risk of the return being below the expected return.

        Semi-deviation is calculated for rate of retirun time series for the sample from 'first_date' to
        'last_date'.

        Returns
        -------
        Series
            Monthly semideviation values for each asset in form of Series.

        See Also
        --------
        risk_monthly : Calculate montly risk for each asset.
        risk_annual : Calculate annualized risks.
        semideviation_annual : Calculate semideviation annualized values.
        get_var_historic : Calculate historic Value at Risk (VaR).
        get_cvar_historic : Calculate historic Conditional Value at Risk (CVaR).
        drawdowns : Calculate drawdowns.

        Examples
        --------
        >>> al = ok.AssetList(['GC.COMM', 'SHV.US'], ccy='USD', last_date='2021-01')
        >>> al.semideviation_monthly
        GC.COMM    0.039358
        SHV.US     0.000384
        dtype: float64
        """
        return helpers.Frame.get_semideviation(self.assets_ror)

    @property
    def semideviation_annual(self) -> pd.Series:
        """
        Return semideviation annualized values for each asset.

        Semi-deviation (Downside risk) is the risk of the return being below the expected return.

        Semi-deviation is calculated for rate of retirun time series for the sample from 'first_date' to
        'last_date'.

        Returns
        -------
        Series
            Annualized semideviation values for each asset in form of Series.

        See Also
        --------
        risk_monthly : Calculate montly risk for each asset.
        risk_annual : Calculate annualized risks.
        semideviation_monthly : Calculate semideviation monthly values.
        get_var_historic : Calculate historic Value at Risk (VaR).
        get_cvar_historic : Calculate historic Conditional Value at Risk (CVaR).
        drawdowns : Calculate drawdowns.

        Examples
        --------
        >>> al = ok.AssetList(['GC.COMM', 'SHV.US'], ccy='USD', last_date='2021-01')
        >>> al.semideviation_annual
        GC.COMM    0.115302
        SHV.US     0.000560
        dtype: float64
        """
        return helpers.Frame.get_semideviation(self.assets_ror) * 12 ** 0.5

    def get_var_historic(self, time_frame: int = 12, level: int = 1) -> pd.Series:
        """
        Calculate historic Value at Risk (VaR) for the assets with a given timeframe.

        The VaR calculates the potential loss of an investment with a given time frame and confidence level.
        Loss is a positive number (expressed in cumulative return).
        If VaR is negative there are expected gains at this confidence level.

        Parameters
        ----------
        time_frame : int, default 12
            Time period size in months
        level : int, default 1
            Confidence level in percents. Default value is 1%.

        Returns
        -------
        Series
            VaR values for each asset in form of Series.

        See Also
        --------
        risk_monthly : Calculate montly risk for each asset.
        risk_annual : Calculate annualized risks.
        semideviation_monthly : Calculate semideviation monthly values.
        semideviation_annual : Calculate semideviation annualized values.
        get_cvar_historic : Calculate historic Conditional Value at Risk (CVaR).
        drawdowns : Calculate drawdowns.

        Examples
        --------
        >>> x = ok.AssetList(['SPY.US', 'AGG.US'])
        >>> x.get_var_historic(time_frame=60, level=1)
        SPY.US    0.2101
        AGG.US    -0.0867
        Name: VaR, dtype: float64
        """
        df = self.get_rolling_cumulative_return(window=time_frame).loc[:, self.symbols]
        return helpers.Frame.get_var_historic(df, level)

    def get_cvar_historic(self, time_frame: int = 12, level: int = 1) -> pd.Series:
        """
        Calculate historic Conditional Value at Risk (CVAR, expected shortfall) for the assets with a given timeframe.

        CVaR is the average loss over a specified time period of unlikely scenarios beyond the confidence level.
        Loss is a positive number (expressed in cumulative return).
        If CVaR is negative there are expected gains at this confidence level.

        Parameters
        ----------
        time_frame : int, default 12
            Time period size in months
        level : int, default 1
            Confidence level in percents to calculate the VaR. Default value is 5%.

        Returns
        -------
        Series
            CVaR values for each asset in form of Series.

        See Also
        --------
        risk_monthly : Calculate montly risk for each asset.
        risk_annual : Calculate annualized risks.
        semideviation_monthly : Calculate semideviation monthly values.
        semideviation_annual : Calculate semideviation annualized values.
        get_var_historic : Calculate historic Value at Risk (VaR).
        drawdowns : Calculate drawdowns.

        Examples
        --------
        >>> x = ok.AssetList(['SPY.US', 'AGG.US'])
        >>> x.get_cvar_historic(time_frame=60, level=1)
        SPY.US    0.2574
        AGG.US   -0.0766
        dtype: float64
        Name: VaR, dtype: float64
        """
        df = self.get_rolling_cumulative_return(window=time_frame).loc[:, self.symbols]
        return helpers.Frame.get_cvar_historic(df, level)

    @property
    def drawdowns(self) -> pd.DataFrame:
        """
        Calculate drawdowns time series for the assets.

        The drawdown is the percent decline from a previous peak in wealth index.

        Returns
        -------
        DataFrame
            Time series of drawdowns.

        See Also
        --------
        risk_monthly : Calculate montly risk for each asset.
        risk_annual : Calculate annualized risks.
        semideviation_monthly : Calculate semideviation monthly values.
        semideviation_annual : Calculate semideviation annualized values.
        get_var_historic : Calculate historic Value at Risk (VaR).
        get_cvar_historic : Calculate historic Conditional Value at Risk (CVaR).

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> al = ok.AssetList(['SPY.US', 'BND.US'], last_date='2021-08')
        >>> al.drawdowns.plot()
        >>> plt.show()
        """
        return helpers.Frame.get_drawdowns(self.assets_ror)

    @property
    def recovery_periods(self) -> pd.Series:
        """
        Calculate the longest recovery periods for the assets.

        The recovery period (drawdown duration) is the number of months to reach the value of the last maximum.

        Returns
        -------
        Series
            Max recovery period for each asset (in months).

        See Also
        --------
        drawdowns : Calculate drawdowns time series.

        Notes
        -----
        If the last asset maximum value is not recovered NaN is returned.
        The largest recovery period does not necessary correspond to the max drawdown.

        Examples
        --------
        >>> x = ok.AssetList(['SPY.US', 'AGG.US'])
        >>> x.recovery_periods
        SPY.US    52
        AGG.US    15
        dtype: int32
        """
        cummax = self.wealth_indexes.cummax()
        growth = cummax.pct_change()[1:]
        max_recovery_periods = pd.Series(dtype=int)
        for name in self.symbols:
            namespace = name.split(".", 1)[-1]
            if namespace == 'INFL':
                continue
            s = growth[name]
            s1 = s.where(s == 0).notnull().astype(int)
            s1_1 = s.where(s == 0).isnull().astype(int).cumsum()
            s2 = s1.groupby(s1_1).cumsum()
            # Max recovery period date should not be in the border (it's not recovered)
            max_period = s2.max() if s2.idxmax().to_timestamp() != self.last_date else np.NAN
            ser = pd.Series(max_period, index=[name])
            max_recovery_periods = pd.concat([max_recovery_periods, ser])
        return max_recovery_periods

    def get_cagr(self, period: Optional[int] = None, real: bool = False) -> pd.Series:
        """
        Calculate assets Compound Annual Growth Rate (CAGR) for a given trailing period.

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
            AssetList should be initiated with Inflation=True for real CAGR.

        Returns
        -------
        Series
            CAGR values for each asset and annualized inflation (optional).

        See Also
        --------
        get_rolling_cagr : Calculate rolling CAGR.

        Notes
        -----
        CAGR is not defined for periods less than 1 year (NaN values are returned).

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
            dt = helpers.Date.subtract_years(dt0, period)
        cagr = helpers.Frame.get_cagr(df[dt:])
        if real:
            if not hasattr(self, "inflation"):
                raise ValueError(
                    "Real CAGR is not defined. Set inflation=True in AssetList to calculate it."
                )
            mean_inflation = helpers.Frame.get_cagr(self.inflation_ts[dt:])
            cagr = (1.0 + cagr) / (1.0 + mean_inflation) - 1.0
            cagr.drop(self.inflation, inplace=True)
        return cagr

    def get_rolling_cagr(self, window: int = 12, real: bool = False) -> pd.DataFrame:
        """
        Calculate rolling CAGR for each asset.

        Compound annual growth rate (CAGR) is the rate of return that would be required for an investment to grow from
        its initial to its final value, assuming all incomes were reinvested.

        Inflation adjusted annualized returns (real CAGR) are shown with `real=True` option.

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
            Time series of rolling CAGR and mean inflation (optionally).

        See Also
        --------
        get_rolling_cagr : Calculate rolling CAGR.
        get_cagr : Calculate CAGR.
        get_rolling_cumulative_return : Calculate rolling cumulative return.
        annual_return : Calculate annualized mean return (arithmetic mean).

        Notes
        -----
        CAGR is not defined for periods less than 1 year (NaN values are returned).

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = ok.AssetList(['DXET.XETR', 'DBXN.XETR'], ccy='EUR', inflation=True)
        >>> x.get_rolling_cagr(window=5*12).plot()
        >>> plt.show()

        For inflation adjusted rolling CAGR add 'real=True' option:

        >>> x.get_rolling_cagr(window=5*12, real=True).plot()
        >>> plt.show()
        """
        df = self._add_inflation()
        if real:
            df = self._make_real_return_time_series(df)
        return helpers.Frame.get_rolling_fn(df, window=window, fn=helpers.Frame.get_cagr)

    def get_cumulative_return(
        self, period: Union[str, int, None] = None, real: bool = False
    ) -> pd.Series:
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

        See Also
        --------
        get_rolling_cagr : Calculate rolling CAGR.
        get_cagr : Calculate CAGR.
        get_rolling_cumulative_return : Calculate rolling cumulative return.
        annual_return : Calculate annualized mean return (arithmetic mean).

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
            dt = helpers.Date.subtract_years(dt0, period)

        cr = helpers.Frame.get_cumulative_return(df[dt:])
        if real:
            if not hasattr(self, "inflation"):
                raise ValueError(
                    "Real cumulative return is not defined (no inflation information is available)."
                    "Set inflation=True in AssetList to calculate it."
                )
            cumulative_inflation = helpers.Frame.get_cumulative_return(self.inflation_ts[dt:])
            cr = (1.0 + cr) / (1.0 + cumulative_inflation) - 1.0
            cr.drop(self.inflation, inplace=True)
        return cr

    def get_rolling_cumulative_return(
        self, window: int = 12, real: bool = False
    ) -> pd.DataFrame:
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

        See Also
        --------
        get_rolling_cagr : Calculate rolling CAGR.
        get_cagr : Calculate CAGR.
        get_cumulative_return : Calculate cumulative return.
        annual_return : Calculate annualized mean return (arithmetic mean).

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = ok.AssetList(['DXET.XETR', 'DBXN.XETR'], ccy='EUR', inflation=True)
        >>> x.get_rolling_cumulative_return(window=5*12).plot()
        >>> plt.show()

        For inflation adjusted rolling cumulative return add 'real=True' option:

        >>> x.get_rolling_cumulative_return(window=5*12, real=True).plot()
        >>> plt.show()
        """
        df = self._add_inflation()
        if real:
            df = self._make_real_return_time_series(df)
        return helpers.Frame.get_rolling_fn(
            df, window=window, fn=helpers.Frame.get_cumulative_return, window_below_year=True
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

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> al = ok.AssetList(['SPY.US', 'BND.US'], last_date='2021-08')
        >>> al.annual_return_ts.plot(kind='bar')
        >>> plt.show()
        """
        return helpers.Frame.get_annual_return_ts_from_monthly(self.assets_ror)

    def describe(
        self, years: Tuple[int, ...] = (1, 5, 10), tickers: bool = True
    ) -> pd.DataFrame:
        """
        Generate descriptive statistics for a list of assets.

        Statistics includes:

        - YTD (Year To date) compound return
        - CAGR for a given list of periods
        - LTM Dividend yield - last twelve months dividend yield

        Risk metrics (full period):

        - risk (standard deviation)
        - CVAR (timeframe is 1 year)
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

        Examples
        --------
        >>> al = ok.AssetList(['SPY.US', 'AGG.US'], last_date='2021-08')
        >>> al.describe(years=[1, 10, 15])
                         property               period    AGG.US    SPY.US inflation
        0         Compound return                  YTD -0.005620  0.180519  0.048154
        1                    CAGR              1 years -0.007530  0.363021  0.053717
        2                    CAGR             10 years  0.032918  0.152310  0.019136
        3                    CAGR             15 years  0.043013  0.107598  0.019788
        4                    CAGR  17 years, 10 months  0.039793  0.107972  0.022002
        5          Dividend yield                  LTM  0.018690  0.012709       NaN
        6                    Risk  17 years, 10 months  0.037796  0.158301       NaN
        7                    CVAR  17 years, 10 months  0.023107  0.399398       NaN
        """
        description = pd.DataFrame()
        dt0 = self.last_date
        df = self._add_inflation()
        # YTD return
        ytd_return = self.get_cumulative_return(period="YTD")
        row = ytd_return.to_dict()
        row.update(period="YTD", property="Compound return")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # CAGR for a list of periods
        if self.pl.years >= 1:
            for i in years:
                dt = helpers.Date.subtract_years(dt0, i)
                if dt >= self.first_date:
                    row = self.get_cagr(period=i).to_dict()
                else:
                    row = {x: None for x in df.columns}
                row.update(period=f"{i} years", property="CAGR")
                description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
            # CAGR for full period
            row = self.get_cagr(period=None).to_dict()
            row.update(period=self._pl_txt, property="CAGR")
            description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
            # Dividend Yield
            row = self.assets_dividend_yield.iloc[-1].to_dict()
            row.update(period="LTM", property="Dividend yield")
            description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # risk for full period
        row = self.risk_annual.to_dict()
        row.update(period=self._pl_txt, property="Risk")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # CVAR
        if self.pl.years >= 1:
            row = self.get_cvar_historic().to_dict()
            row.update(period=self._pl_txt, property="CVAR")
            description = pd.concat([description,pd.DataFrame(row, index=[0])], ignore_index=True)
        # max drawdowns
        row = self.drawdowns.min().to_dict()
        row.update(period=self._pl_txt, property="Max drawdowns")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # max drawdowns dates
        row = self.drawdowns.idxmin().to_dict()
        row.update(period=self._pl_txt, property="Max drawdowns dates")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # inception dates
        row = {}
        for ti in self.symbols:
            # short_ticker = ti.split(".", 1)[0]
            value = self.assets_first_dates[ti].strftime("%Y-%m")
            row.update({ti: value})
        row.update(period=None, property="Inception date")
        if hasattr(self, "inflation"):
            row.update({self.inflation: self.inflation_first_date.strftime("%Y-%m")})
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # last asset date
        row = {}
        for ti in self.symbols:
            # short_ticker = ti.split(".", 1)[0]
            value = self.assets_last_dates[ti].strftime("%Y-%m")
            row.update({ti: value})
        row.update(period=None, property="Last asset date")
        if hasattr(self, "inflation"):
            row.update({self.inflation: self.inflation_last_date.strftime("%Y-%m")})
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # last data date
        row = {x: self.last_date.strftime("%Y-%m") for x in df.columns}
        row.update(period=None, property="Common last data date")
        description = pd.concat([description, pd.DataFrame(row, index=[0])], ignore_index=True)
        # rename columns
        if hasattr(self, "inflation"):
            description.rename(columns={self.inflation: "inflation"}, inplace=True)
            description = helpers.Frame.change_columns_order(
                description, ["inflation"], position="last"
            )
        description = helpers.Frame.change_columns_order(
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
        return helpers.Float.annualize_return(mean)

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
        # TODO: make a single method with mean_return
        if not hasattr(self, "inflation"):
            raise ValueError(
                "Real Return is not defined. Set inflation=True to calculate."
            )
        df = pd.concat(
            [self.assets_ror, self.inflation_ts], axis=1, join="inner", copy="false"
        )
        infl_mean = helpers.Float.annualize_return(self.inflation_ts.values.mean())
        ror_mean = helpers.Float.annualize_return(df.loc[:, self.symbols].mean())
        return (1.0 + ror_mean) / (1.0 + infl_mean) - 1.0

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
        >>> x = ok.AssetList(['T.US', 'XOM.US'], first_date='2010-01', last_date='2020-12')
        >>> x.dividends_annual.plot(kind='bar')
        >>> plt.show()
        """
        return self._get_assets_dividends().resample("Y").sum()

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
        >>> import matplotlib.pyplot as plt
        >>> x = ok.AssetList(['T.US', 'XOM.US'], first_date='1984-01', last_date='1994-12')
        >>> x.dividend_growing_years.plot(kind='bar')
        >>> plt.show()
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
        >>> import matplotlib.pyplot as plt
        >>> x = ok.AssetList(['T.US', 'XOM.US'], first_date='1984-01', last_date='1994-12')
        >>> x.dividend_paying_years.plot(kind='bar')
        >>> plt.show()
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
        Calculate geometric mean of annual dividends growth rate time series for a given trailing period.

        Growth rate is taken for full calendar annual dividends.

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
        growth_ts.replace([np.inf, -np.inf, np.nan], 0, inplace=True)  # replace possible nan and inf
        dt0 = self.last_date
        dt = helpers.Date.subtract_years(dt0, period)
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
        >>> import matplotlib.pyplot as plt
        >>> x = ok.AssetList(['SP500TR.INDX', 'SPY.US', 'VOO.US'], last_date='2021-01')
        >>> x.tracking_difference.plot()
        >>> plt.show()
        """
        accumulated_return = helpers.Frame.get_wealth_indexes(
            self.assets_ror
        )  # we don't need inflation here
        return helpers.Index.tracking_difference(accumulated_return)

    @property
    def tracking_difference_annualized(self) -> pd.DataFrame:
        """
        Calculate annualized tracking difference time series for the rate of return of assets.

        Tracking difference is calculated by measuring the accumulated difference between the returns of a benchmark
        and ETFs replicating it (could be mutual funds, or other types of assets).

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
        >>> import matplotlib.pyplot as plt
        >>> x = ok.AssetList(['SP500TR.INDX', 'SPY.US', 'VOO.US'], last_date='2021-01')
        >>> x.tracking_difference_annualized.plot()
        """
        return helpers.Index.tracking_difference_annualized(self.tracking_difference)

    @property
    def tracking_difference_annual(self) -> pd.DataFrame:
        """
        Calculate tracking difference for each calendar year.

        Tracking difference is calculated by measuring the accumulated difference between the returns of a benchmark
        and ETFs replicating it (could be mutual funds, or other types of assets).

        Benchmark should be in the first position of the symbols list in AssetList parameters.

        Returns
        -------
        DataFrame
            Time series with tracking difference for each calendar year period.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> al = ok.AssetList(['SP500TR.INDX', 'VOO.US', 'SPXS.LSE'], inflation=False)
        >>> al.tracking_difference_annual.plot(kind='bar')
        """
        result = pd.DataFrame()
        for x in self.assets_ror.resample('Y'):
            df = x[1]
            wealth_index = helpers.Frame.get_wealth_indexes(df)
            row = helpers.Index.tracking_difference(wealth_index).iloc[[-1]]
            result = pd.concat([result, row], ignore_index=False)
        result.index = result.index.asfreq('Y')
        return result

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
        >>> import matplotlib.pyplot as plt
        >>> x = ok.AssetList(['SP500TR.INDX', 'SPY.US', 'VOO.US'], last_date='2021-01')
        >>> x.tracking_error.plot()
        >>> plt.show()
        """
        return helpers.Index.tracking_error(self.assets_ror)

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
        >>> import matplotlib.pyplot as plt
        >>> sp = ok.AssetList(['SP500TR.INDX', 'VBMFX.US', 'GC.COMM', 'VNQ.US'])
        >>> sp.names
        {'SP500TR.INDX': 'S&P 500 (TR)',
        'VBMFX.US': 'VANGUARD TOTAL BOND MARKET INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold',
        'VNQ.US': 'Vanguard Real Estate Index Fund ETF Shares'}
        >>> sp.index_corr.plot()
        >>> plt.show()
        """
        return helpers.Index.cov_cor(self.assets_ror, fn="corr")

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
        >>> import matplotlib.pyplot as plt
        >>> sp = ok.AssetList(['SP500TR.INDX', 'VBMFX.US', 'GC.COMM'])
        >>> sp.names
        {'SP500TR.INDX': 'S&P 500 (TR)',
        'VBMFX.US': 'VANGUARD TOTAL BOND MARKET INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold'}
        >>> sp.index_rolling_corr(window=24).plot()
        >>> plt.show()
        """
        return helpers.Index.rolling_cov_cor(self.assets_ror, window=window, fn="corr")

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
        >>> import matplotlib.pyplot as plt
        >>> sp = ok.AssetList(['SP500TR.INDX', 'VBMFX.US', 'GC.COMM', 'VNQ.US'])
        >>> sp.names
        {'SP500TR.INDX': 'S&P 500 (TR)',
        'VBMFX.US': 'VANGUARD TOTAL BOND MARKET INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold',
        'VNQ.US': 'Vanguard Real Estate Index Fund ETF Shares'}
        >>> sp.index_beta.plot()
        >>> plt.show()
        """
        return helpers.Index.beta(self.assets_ror)

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
        >>> import matplotlib.pyplot as plt
        >>> al = ok.AssetList(['VFINX.US', 'GC.COMM'], last_date='2021-01')
        >>> al.names
        {'VFINX.US': 'VANGUARD 500 INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold'}
        >>> al.skewness.plot()
        >>> plt.show()
        """
        return helpers.Frame.skewness(self.assets_ror)

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
        >>> import matplotlib.pyplot as plt
        >>> al = ok.AssetList(['VFINX.US', 'GC.COMM'], last_date='2021-01')
        >>> al.names
        {'VFINX.US': 'VANGUARD 500 INDEX FUND INVESTOR SHARES',
        'GC.COMM': 'Gold'}
        >>> al.skewness_rolling(window=12*5).plot()
        >>> plt.show()
        """
        return helpers.Frame.skewness_rolling(self.assets_ror, window=window)

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
        >>> import matplotlib.pyplot as plt
        >>> al = ok.AssetList(['GC.COMM', 'FNER.INDX'], first_date='2000-01', last_date='2021-01')
        >>> al.names
        {'GC.COMM': 'Gold',
        'FNER.INDX': 'FTSE NAREIT All Equity REITs'}
        >>> al.kurtosis.plot()
        >>> plt.show()
        """
        return helpers.Frame.kurtosis(self.assets_ror)

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
        >>> import matplotlib.pyplot as plt
        >>> al = ok.AssetList(['GC.COMM', 'FNER.INDX'], first_date='2000-01', last_date='2021-01')
        >>> al.names
        {'GC.COMM': 'Gold',
        'FNER.INDX': 'FTSE NAREIT All Equity REITs'}
        >>> al.kurtosis_rolling(window=12*5).plot()
        >>> plt.show()
        """
        return helpers.Frame.kurtosis_rolling(self.assets_ror, window=window)

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
        return helpers.Frame.jarque_bera_dataframe(self.assets_ror)

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
        return helpers.Frame.kstest_dataframe(self.assets_ror, distr=distr)

    def get_sharpe_ratio(self, rf_return: float = 0) -> pd.Series:
        """
        Calculate Sharpe ratio for the assets.

        The Sharpe ratio is the average annual return in excess of the risk-free rate
        per unit of risk (annualized standard deviation).

        Risk-free rate should be taken according to the AssetList base currency.

        Parameters
        ----------
        rf_return : float, default 0
            Risk-free rate of return.

        Returns
        -------
        pd.Series

        Examples
        --------
        >>> al = ok.AssetList(['VOO.US', 'BND.US'])
        >>> al.get_sharpe_ratio(rf_return=0.02)
        VOO.US    0.962619
        BND.US    0.390814
        dtype: float64
        """
        mean_return = self.mean_return.drop(self.inflation) if self.inflation else self.mean_return
        return ratios.get_sharpe_ratio(
            pf_return=mean_return,
            rf_return=rf_return,
            std_deviation=self.risk_annual)

    def get_sortino_ratio(self, t_return: float = 0) -> pd.Series:
        """
        Calculate Sortino ratio for the assets with specified target return.

        Sortion ratio measures the risk-adjusted return of each asset. It is a modification of the Sharpe ratio
        but penalizes only those returns falling below a specified target rate of return, while
        the Sharpe ratio penalizes both upside and downside volatility equally.

        Parameters
        ----------
        t_return : float, default 0
            Traget rate of return.

        Returns
        -------
        pd.Series

        Examples
        --------
        >>> al = ok.AssetList(['VOO.US', 'BND.US'], last_date='2021-12')
        >>> al.get_sortino_ratio(t_return=0.03)
        VOO.US    1.321951
        BND.US    0.028969
        dtype: float64
        """
        mean_return = self.mean_return.drop(self.inflation) if self.inflation else self.mean_return
        semideviation = helpers.Frame.get_below_target_semideviation(ror=self.assets_ror, t_return=t_return) * 12 ** 0.5
        return ratios.get_sortino_ratio(
            pf_return=mean_return,
            t_return=t_return,
            semi_deviation=semideviation)
