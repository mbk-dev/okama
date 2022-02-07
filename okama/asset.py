from typing import Optional

import pandas as pd
import numpy as np

from okama import settings
from okama.api import data_queries, namespaces
from okama.common.helpers import helpers


class Asset:
    """
    A financial asset, that could be used in a list of assets or in portfolio.

    Parameters
    ----------
    symbol: str, default "SPY.US"
        Symbol is an asset ticker with namespace after dot. The default value is "SPY.US" (SPDR S&P 500 ETF Trust).
    """

    def __init__(self, symbol: str = settings.default_ticker):
        if symbol is None or len(str(symbol).strip()) == 0:
            raise ValueError("Symbol can not be empty")
        self._symbol = str(symbol).strip()
        self._check_namespace()
        self._get_symbol_data(symbol)
        self.ror: pd.Series = data_queries.QueryData.get_ror(symbol)
        self.first_date: pd.Timestamp = self.ror.index[0].to_timestamp()
        self.last_date: pd.Timestamp = self.ror.index[-1].to_timestamp()
        self.period_length: float = round(
            (self.last_date - self.first_date) / np.timedelta64(365, "D"), ndigits=1
        )
        self.pl = settings.PeriodLength(
            self.ror.shape[0] // settings._MONTHS_PER_YEAR,
            self.ror.shape[0] % settings._MONTHS_PER_YEAR,
        )

    def __repr__(self):
        dic = {
            "symbol": self.symbol,
            "name": self.name,
            "country": self.country,
            "exchange": self.exchange,
            "currency": self.currency,
            "type": self.type,
            "isin": self.isin,
            "first date": self.first_date.strftime("%Y-%m"),
            "last date": self.last_date.strftime("%Y-%m"),
            "period length": "{:.2f}".format(self.period_length),
        }
        return repr(pd.Series(dic))

    def _check_namespace(self):
        namespace = self._symbol.split(".", 1)[-1]
        allowed_namespaces = namespaces.get_assets_namespaces()
        if namespace not in allowed_namespaces:
            raise ValueError(
                f"{namespace} is not in allowed assets namespaces: {allowed_namespaces}"
            )

    def _get_symbol_data(self, symbol) -> None:
        x = data_queries.QueryData.get_symbol_info(symbol)
        self.ticker: str = x["code"]
        self.name: str = x["name"]
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
        return data_queries.QueryData.get_live_price(self.symbol)

    @property
    def close_daily(self):
        """
        Return close price time series historical daily data.

        Returns
        -------
        Series
            Time series of close price historical data (daily).
        """
        return data_queries.QueryData.get_close(self.symbol, period='D')

    @property
    def close_monthly(self):
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
        >>> x = ok.Asset('VOO.US')
        >>> x.close_monthly.plot()
        >>> plt.show()
        """
        return helpers.Frame.change_period_to_month(self.close_daily)

    @property
    def adj_close(self):
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
        return data_queries.QueryData.get_adj_close(self.symbol, period='D')

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
        div = data_queries.QueryData.get_dividends(self.symbol)
        if div.empty:
            # Zero time series for assets where dividend yield is not defined.
            index = pd.date_range(
                start=self.first_date, end=self.last_date, freq="MS", closed=None
            )
            period = index.to_period("D")
            div = pd.Series(data=0, index=period)
            div.rename(self.symbol, inplace=True)
        return div.resample("M").sum()

    @property
    def nav_ts(self) -> Optional[pd.Series]:
        """
        Return NAV time series (monthly) for mutual funds.
        """
        if self.exchange == "PIF":
            return data_queries.QueryData.get_nav(self.symbol)
        return np.nan
