import pandas as pd
import numpy as np

from okama import settings
from okama.api import data_queries, namespaces
from okama.common import validators
from okama.common.helpers import helpers


class Asset:
    """
    A financial asset, that could be used in a list of assets or in portfolio.

    Parameters
    ----------
    symbol: str, default 'SPY.US'
        Symbol is an asset ticker with a namespace after dot. The default value is 'SPY.US' (SPDR S&P 500 ETF Trust).

    first_date : str, pd.Timestamp, None, default None
        First date of the rate of return time series.
        If None, the first available date will be used.

    last_date : str, pd.Timestamp, None, default None
        Last date of the rate of return time series.
        If None, the last available date will be used.
    """

    def __init__(
        self,
        symbol: str = settings.default_ticker,
        first_date: str | pd.Timestamp | None = None,
        last_date: str | pd.Timestamp | None = None,
    ):
        if symbol is None or len(str(symbol).strip()) == 0:
            raise ValueError("Symbol can not be empty")
        self._symbol = str(symbol).strip()
        self._check_namespace()
        self._get_symbol_data(symbol)
        self._first_date = first_date
        self._last_date = last_date
        self.ror: pd.Series = data_queries.QueryData.get_ror(
            symbol,
            first_date=first_date if first_date else "1913-01-01",
            last_date=last_date if last_date else "2100-01-01",
            period="M",
        )
        self._set_first_last_dates()

    def _set_first_last_dates(self) -> None:
        """
        Set first_date, last_date, period_length and pl attributes based on ror data.

        Converts Period index to Timestamp using 'start' parameter to ensure
        the timestamp represents the beginning of the month.
        """
        self.first_date: pd.Timestamp = self.ror.index[0].to_timestamp(how="start")
        self.last_date: pd.Timestamp = self.ror.index[-1].to_timestamp(how="start")
        self.period_length: float = round((self.last_date - self.first_date) / np.timedelta64(365, "D"), ndigits=1)
        self.pl = settings.PeriodLength(
            self.ror.shape[0] // settings._MONTHS_PER_YEAR,
            self.ror.shape[0] % settings._MONTHS_PER_YEAR,
        )

    def __repr__(self):
        dic = {
            "symbol": self.symbol,
            "name": self.name,
            "local_name": self.local_name,
            "country": self.country,
            "exchange": self.exchange,
            "currency": self.currency,
            "type": self.type,
            "isin": self.isin,
            "first date": self.first_date.strftime("%Y-%m"),
            "last date": self.last_date.strftime("%Y-%m"),
            "period length": f"{self.period_length:.2f}",
        }
        return repr(pd.Series(dic))

    def _check_namespace(self):
        namespace = self._symbol.split(".")[-1]
        allowed_namespaces = namespaces.get_assets_namespaces()
        if namespace not in allowed_namespaces:
            raise ValueError(f"{namespace} is not in allowed assets namespaces: {allowed_namespaces}")

    def _get_symbol_data(self, symbol) -> None:
        x = data_queries.QueryData.get_symbol_info(symbol)
        self.info: dict = x
        self.ticker: str = x["code"]
        self.name: str = x["name"]
        self.local_name: str | None = x.get("local_name")
        self.country: str = x["country"]
        self.exchange: str = x["exchange"]
        self.currency: str = x["currency"]
        self.type: str = x["type"]
        self.isin: str = x["isin"]
        self.inflation: str = f"{self.currency}.INFL"

    @property
    def symbol(self) -> str:
        """
        Return a symbol of the asset.

        Returns
        -------
        str
        """
        return self._symbol

    @property
    def price(self) -> float | None:
        """
        Return live price of an asset.

        Live price is delayed (15-20 minutes).
        For certain namespaces (FX, INDX, PIF etc.) live price is not supported.

        Returns
        -------
        float, None
            Live price of the asset. Returns None if not defined.
        """
        return data_queries.QueryData.get_live_price(self.symbol)

    @property
    def close_daily(self) -> pd.Series:
        """
        Return close price time series historical daily data.

        Returns
        -------
        Series
            Time series of close price historical data (daily).
        """
        return data_queries.QueryData.get_close(self.symbol, period="D")

    @property
    def close_monthly(self) -> pd.Series:
        """
        Return close price time series historical monthly data.

        Monthly close time series not adjusted to for corporate actions: dividends and splits.

        Returns
        -------
        Series
            Time series of close price historical data (monthly).

        Examples
        --------
        >>> import matplotlib.pyplot as plt

        >>> x = ok.Asset("VOO.US")
        >>> x.close_monthly.plot()
        >>> plt.show()
        """
        return data_queries.QueryData.get_close(self.symbol, period="M")

    def get_close_monthly(self, month: str | pd.Timestamp | pd.Period) -> float:
        """
        Return the monthly close price for a requested month.

        Parameters
        ----------
        month : str, Timestamp or Period
            Requested month, for example "2026-09". Dates are converted to months.

        Returns
        -------
        float
            Monthly close price in the asset currency.

        Raises
        ------
        KeyError
            If the month is unavailable; the message includes the requested month
            and the available close-price range. The range is independent of the
            first_date and last_date used to select rate-of-return history.
        """
        requested = pd.Period(month, freq="M")
        close = self.close_monthly
        try:
            return float(close.loc[requested])
        except KeyError as error:
            bounds = f"{close.index.min()} to {close.index.max()}" if not close.empty else "empty history"
            raise KeyError(
                f"Monthly close for {self.symbol} at {requested} is unavailable; available range: {bounds}."
            ) from error

    def get_cagr(self, period: int | None = None, real: bool = False) -> float:
        """
        Return the scalar Compound Annual Growth Rate in the asset currency.

        Parameters
        ----------
        period : int, default None
            Trailing period in whole years ending at the last available month.
            None measures all selected return history between first_date and
            last_date, rather than a fixed number of years.
        real : bool, default False
            Adjust for inflation in the asset currency. The history is restricted
            to months shared by the asset and inflation before selecting the
            trailing period; its last month can precede the asset's last_date.
            Nominal calculations do not load inflation or shorten asset history.

        Returns
        -------
        float
            CAGR, or NaN when fewer than 12 months are available. For nominal
            returns this matches the final asset row of AssetList.get_cagr with
            the same currency and selected history and inflation=False.

        Raises
        ------
        TypeError
            If period is not an integer.
        ValueError
            If period is not positive or exceeds the available history, or if
            there are no months shared with inflation for a real calculation.
        """
        returns = self.ror
        inflation = None
        if real:
            from okama.macro import Inflation

            inflation = Inflation(self.inflation, self.first_date, self.last_date).values_monthly
            common_index = returns.index.intersection(inflation.index)
            if common_index.empty:
                raise ValueError("Real CAGR is not defined: asset and inflation have no common months.")
            returns = returns.loc[common_index]
            inflation = inflation.loc[common_index]
        if period is not None:
            validators.validate_integer("period", period, min_value=0, inclusive=False)
            years = len(returns) // settings._MONTHS_PER_YEAR
            if period > years:
                raise ValueError(f"'period' ({period}) is beyond historical data range ({years} years).")
            start = helpers.Date.subtract_years(returns.index[-1].to_timestamp(), period)
            returns = returns.loc[start:]
            if inflation is not None:
                inflation = inflation.loc[start:]
        if len(returns) < settings._MONTHS_PER_YEAR:
            return float("nan")
        cagr = helpers.Frame.get_cagr(returns)
        if inflation is not None:
            cagr = (1.0 + cagr) / (1.0 + helpers.Frame.get_cagr(inflation)) - 1.0
        return float(cagr)

    @property
    def adj_close(self) -> pd.Series:
        """
        Return adjusted close price time series historical daily data.

        The adjusted closing price amends a stock's closing price after accounting
        for corporate actions: dividends and splits. All values are adjusted by reducing the price
        prior to the dividend payment (or split).

        Returns
        -------
        Series
            Time series of adjusted close price historical data (daily).
        """
        return data_queries.QueryData.get_adj_close(self.symbol, period="D")

    @property
    def dividends(self) -> pd.Series:
        """
        Return dividends time series historical monthly data.

        Returns
        -------
        Series
            Time series of dividends historical data (monthly).

        Examples
        --------
        >>> x = ok.Asset("VNQ.US")
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
        div = data_queries.QueryData.get_dividends(self.symbol)
        if div.empty:
            # Zero time series for assets where dividend yield is not defined.
            index = pd.date_range(start=self.first_date, end=self.last_date, freq="MS", inclusive="neither")
            period = index.to_period("D")
            div = pd.Series(data=0, index=period)
            div = div.rename(self.symbol)
        return div.resample("M").sum()

    @property
    def nav_ts(self) -> pd.Series | None:
        """
        Return NAV time series (monthly) for mutual funds.
        """
        if self.exchange == "PIF":
            return data_queries.QueryData.get_nav(self.symbol)
        return np.nan
