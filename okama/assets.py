import time

import pandas as pd
import numpy as np

from .helpers import Float, Frame, Rebalance, String
from .settings import default_ticker
from .data import get_eod_data, get_ticker_data


class Asset:
    """
    An asset, that could be used in a list or portfolio.
    """

    def __init__(self, symbol=default_ticker):
        self.ticker = symbol
        self.ror = self._get_monthly_ror(symbol)
        self.market = self._define_market(symbol)
        self.asset_currency = self._define_currency()
        self.first_date = self.ror.index[0].to_timestamp()
        self.last_date = self.ror.index[-1].to_timestamp()
        # self.asset_type = self._define_type()

    # def _define_type(self):
    #     if self.market == 'US':
    #         return 'stock'
    #     elif self.market == 'FOREX':
    #         return 'currency'
    #     elif self.market == 'RUFUND':
    #         return 'mutual fund'
    #     elif self.market == 'MCX':
    #         return 'stock'
    #     elif self.market == 'INDX':
    #         return 'index'
    #     else:
    #         raise ValueError(f'asset type is not defined for {self.market}')

    def _get_monthly_ror(self, ticker: str) -> pd.Series:
        """
        Calculate monthly mean return time series given the ticker.
        Time Series with low number of days (<15) in the first month are sliced (monthly data begins from th second month)
        """
        s = get_eod_data(ticker)
        name = s.name

        y = s.index[0].year
        m = s.index[0].month
        first_month = str(y)+'-'+str(m)
        cut = s[first_month].count() < 15

        s = s.resample('M').apply(lambda x: (np.prod(1 + x) - 1))
        if cut:
            s = s.iloc[1:]
        s.rename(name, inplace=True)
        return s

    def _define_market(self, char):
        return String.define_market(char)[1]
        
    def _define_currency(self):
        if self.market == 'US':
            return 'USD'
        elif self.market == 'FOREX':
            return 'USD'
        elif self.market == 'RUFUND':
            return 'RUB'
        elif self.market == 'MCX':
            if self.ticker.split('.',1)[0] in ['AKSP', 'AKNX', 'TUSD', 'VTBU']:
                return 'USD'
            else:
                return 'RUB'
        elif self.market == 'INDX':
            if self.ticker.split('.',1)[0] in ['IMOEX', 'MCFTR', 'RGBI']:
                return 'RUB'
            if self.ticker.split('.',1)[0] in ['GSPC', 'SP500TR', 'SP500NTR']:
                return 'USD'
        else:
            raise ValueError(self.market + ' is not a known namespace')

    @property
    def nav_ts(self) -> pd.Series:
        """
        NAV time series (monthly) for mutual funds when available in data.
        """
        if self.market == 'RUFUND':
            s = get_eod_data(self.ticker, type='nav')
            s = s.resample('M').last()
            return s
        return np.nan

    @property
    def name(self) -> str:
        return get_ticker_data(self.ticker)['name']

    @property
    def exchange(self) -> str:
        return get_ticker_data(self.ticker)['exchange']

    @property
    def asset_type(self) -> str:
        return get_ticker_data(self.ticker)['type']

            
class AssetList:
    """
    The list of assets implementation.
    """
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD'):
        main_start_time = time.time()
        self.tickers = symbols
        self.currency = curr
        self._make_asset_list(symbols)
        if first_date:
            self.first_date = max(self.first_date, pd.to_datetime(first_date))
        self.ror = self.ror[self.first_date:]
        if last_date:
            self.ror = self.ror[:pd.to_datetime(last_date)]
        self.last_date = self.ror.index[-1].to_timestamp()
        main_end_time = time.time()
        print(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
             
    def _make_asset_list(self, ls: list):
        """
        Makes an asset list from a list of symbols. Returns dataframe (even for one asset) of returns (monthly)
        as an attribute.
        """
        for i, x in enumerate(ls):
            asset = Asset(x)
            if i == 0:
                dates = {}
                if asset.asset_currency == self.currency:
                    df = asset.ror
                else:
                    df = self._set_currency(asset.ror, self.currency)                  
            else:
                if asset.asset_currency == self.currency:
                    new = asset.ror
                else:
                    new = self._set_currency(asset.ror, self.currency)                   
                df = pd.concat([df,new], axis = 1, join='inner', copy='false')
            dates.update({asset.ticker: asset.first_date})
        dates = sorted(dates.items(), key=lambda x: x[1])
        if isinstance(df, pd.Series):  # required to convert Series to DataFrame for single asset list
            df = df.to_frame()
        self.first_date = dates[-1][1]
        self.newest_asset = dates[-1][0]
        self.eldest_asset = dates[0][0]
        self.assets_first_dates = dates
        self.ror = df
    
    def _set_currency(self, returns: pd.Series, currency: str):
        """
        Set return to a certain currency. Input is a pd.Series of mean returns and a currency symbol.
        """
        currency_returns = Asset('RUB.FOREX').ror
        df = pd.concat([returns, currency_returns], axis = 1, join='inner', copy='false') # join dataframes to have the same Time Series Index
        currency_returns = df.iloc[:,-1]
        df = df.drop(columns=['RUB.FOREX'])
        if currency == 'USD':
            y = currency_returns + 1.
            x = (df+1.).mul(1/y, axis=0) - 1.
        elif currency == 'RUB':
            y = currency_returns + 1.
            x = (df+1.).mul(y, axis=0) - 1.
        return x

    @property
    def wealth_indexes(self):
        return Frame.wealth_indexes(self.ror)

    def calculate_risk(self, annualize=False, periods_per_year=12) -> pd.Series:
        """
        Takes assets returns DataFrame and calculates risks (std) for each asset.
        Default risk is monthly standard deviation.
        If annualize=True calculate annualized values.
        """
        risk = self.ror.std()
        mean_return = self.ror.mean()
        if annualize:
            risk = Float.annualize_risk(risk, mean_return, periods_per_year)
        return risk

    @property
    def semideviation(self):
        return Frame.get_semideviation(self.ror)

    def get_var_historic(self, level=5):
        return Frame.get_var_historic(self.ror, level)

    def get_cvar_historic(self, level=5):
        return Frame.get_cvar_historic(self.ror, level)

    @property
    def drawdowns(self):
        return Frame.get_drawdowns(self.ror)

    @property
    def cagr(self):
        return Frame.get_cagr(self.ror)


class Portfolio:
    """
    Implementation of investment portfolio.
    Arguments are similar to AssetList (weights are added), but different behavior.
    """
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD', weights=None):
        self.currency = curr
        self.tickers = symbols
        if weights is None:
            n = len(symbols)  # number of assets
            weights = list(np.repeat(1/n, n))
            self.weights = weights
        else:
           self.weights = weights
        self._ror = AssetList(symbols=symbols, first_date=first_date, last_date=last_date, curr=curr).ror
        self.first_date = self._ror.index[0].to_timestamp()
        self.last_date = self._ror.index[-1].to_timestamp()

    @property
    def returns_ts(self) -> pd.Series:
        return Frame.get_portfolio_return_ts(self.weights, self._ror)

    @property
    def wealth_index(self) -> pd.Series:
        return Frame.wealth_indexes(self.returns_ts)

    def get_rebalanced_portfolio_return_ts(self, period='Y') -> pd.Series:
        return Rebalance.rebalanced_portfolio_return_ts(self.weights, self._ror, period=period)

    @property
    def mean_return_monthly(self) -> float:
        return Frame.get_portfolio_mean_return(self.weights, self._ror)

    @property
    def mean_return_annual(self) -> float:
        return Float.annualize_return(self.mean_return_monthly)

    @property
    def cagr(self):
        return Frame.get_cagr(self.returns_ts)

    @property
    def risk_monthly(self):
        return Frame.get_portfolio_risk(self.weights, self._ror)

    @property
    def risk_annual(self) -> float:
        return Float.annualize_risk(self.risk_monthly, self.mean_return_monthly)

    @property
    def semideviation(self):
        return Frame.get_semideviation(self.returns_ts)

    def get_var_historic(self, level=5):
        rolling = self.returns_ts.rolling(12).apply(Frame.get_cagr)
        var = Frame.get_var_historic(rolling, level)
        return var

    def get_cvar_historic(self, level=5):
        rolling = self.returns_ts.rolling(12).apply(Frame.get_cagr)
        cvar = Frame.get_cvar_historic(rolling, level)
        return cvar

    @property
    def drawdowns(self):
        return Frame.get_drawdowns(self._ror)