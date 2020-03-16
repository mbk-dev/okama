import math

import pandas as pd
import numpy as np

from .helpers import Float, Frame, Rebalance
from .settings import default_ticker
from .data import get_eod_data


class Asset:
    """
    An asset, that could be used in a list or portfolio.
    """

    def __init__(self, symbol=default_ticker):
        self.ticker = symbol
        self.ror = self._get_monthly_ror(symbol)
        self.market = self._define_market(symbol)
        self.asset_currency = self._define_currency()

    def _get_monthly_ror(self, ticker: str) -> pd.Series:
        """
        Calculate monthly mean return time series given the ticker.
        """
        s = get_eod_data(ticker)
        name = s.name
        s = s.resample('M').apply(lambda x: (np.prod(1 + x) - 1))
        s.rename(name, inplace=True)
        return s
        
    def _define_market(self, char):
        if '.' not in char:
            return 'US'
        else:
            return char.split('.',1)[-1]
        
    def _define_currency(self):
        if self.market == 'US':
            return 'USD'
        elif self.market == 'FOREX':
            return 'USD'
        elif self.market == 'RUFUND':
            return 'RUB'
        elif self.market == 'MCX':
            return 'RUB'
        else:
            raise ValueError(self.market + ' is not a known namespace') # ERR: self.market is a text

            
class AssetList:
    """
    The list of assets implementation.
    """
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD'):
        self.tickers = symbols
        self.currency = curr
        self._make_asset_list(symbols)
        if first_date:
            self.ror = self.ror[pd.to_datetime(first_date):]
        if last_date:
            self.ror = self.ror[:pd.to_datetime(last_date)]
        self.first_date = self.ror.index[0].to_timestamp()
        self.last_date = self.ror.index[-1].to_timestamp()
             
    def _make_asset_list(self, ls: list):
        """
        Makes an asset list from a list of symbols. Returns dataframe (even for one asset) of returns (monthly)
        as an attribute.
        """
        for i, x in enumerate(ls):
            asset = Asset(x)
            if i == 0:
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
        if isinstance(df, pd.Series):  # required to convert Series to DataFrame for single asset list
            df = df.to_frame()
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
        """
        Returns wealth index for a list of assets.
        """
        initial_investments = 1000
        return initial_investments * (self.ror + 1).cumprod()

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
    def cagr(self):
        return ((self.ror + 1.).prod()) ** (12 / self.ror.shape[0]) - 1


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
        """
        Returns mean rate of return time series.
        """
        return Frame.get_portfolio_return_ts(self.weights, self._ror)

    def get_rebalanced_portfolio_return_ts(self, period='Y') -> pd.Series:
        """
        Return meant rate of return for rebalanced (annual) portfolio.
        For not rebalanced portfolio set Period to 'N'
        """
        return Rebalance.rebalanced_portfolio_return_ts(self.weights, self._ror, period=period)

    @property
    def mean_return_monthly(self) -> float:
        """
        Calculates mean monthly return (weighted sum of asset returns).
        """
        return Frame.get_portfolio_mean_return(self.weights, self._ror)

    @property
    def mean_return_annual(self) -> float:
        """
        Annualizes monthly mean return.
        """
        return Float.annualize_return(self.mean_return_monthly)

    @property
    def risk_monthly(self):
        """
        Calculates monthly risk (standard deviation) using covariance.
        """
        return Frame.get_portfolio_risk(self.weights, self._ror)

    @property
    def risk_annual(self) -> float:
        """
        Calculates annualized risk (standard deviation) given monthly risk and mean return.
        """
        return Float.annualize_risk(self.risk_monthly, self.mean_return_monthly)
