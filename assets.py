import math

import pandas as pd
import numpy as np

from .settings import default_ticker
from .data import get_eod_data

def get_portfolio_return_ts(weights:list, ror: pd.DataFrame) -> pd.Series:
    if isinstance(ror, pd.Series): # required for a single asset portfolio
        return ror
    return_ts = ror @ weights
    return return_ts


def get_portfolio_mean_return(weights: list, ror: pd.DataFrame) -> float:
    """
    Computes mean return of a portfolio (month scale). Returns a single float number.
    """
    weights = np.asarray(weights)
    if isinstance(ror.mean(), float): # required for a single asset portfolio
        return ror.mean()
    return weights.T @ ror.mean()


def get_portfolio_risk(weights: list, ror: pd.DataFrame) -> float:
    """
    Computes the std of portfolio returns.
    """
    if isinstance(ror, pd.Series): # required for a single asset portfolio
        return ror.std()
    weights = np.array(weights)
    covmat = ror.cov()
    return math.sqrt(weights.T @ covmat @ weights)


def rebalanced_portfolio_return_ts(weights: list, ror: pd.DataFrame, *, period='Y'):
    """
    Returns the rate of return time serie of rebalanced portfolio.
    Default rebalancing period is a Year (end of year)
    For not rebalanced portfolio set Period to 'None'
    """
    initial_inv = 1000
    if period == 'None':  # Not rebalanced portfolio
        inv_period = initial_inv
        inv_period_spread = np.asarray(weights) * inv_period
        assets_wealth_indexes = inv_period_spread * (1 + ror).cumprod()
        wealth_index = assets_wealth_indexes.sum(axis=1)
        ror = wealth_index.pct_change()
        ror = ror.iloc[1:]
        #print(f'{period=}')
        return ror
    grouped = ror.resample(period)
    for i, x in enumerate(grouped):
        if i == 0:
            inv_period = 1000
            wealth_index = pd.Series(dtype='float64')
            wealth_index_local = pd.Series(dtype='float64')
        df = x[1]
        inv_period_spread = np.asarray(weights) * inv_period # rebalancing
        assets_wealth_indexes = inv_period_spread * (1 + df).cumprod()
        wealth_index_local = assets_wealth_indexes.sum(axis=1)
        wealth_index = pd.concat([wealth_index, wealth_index_local], verify_integrity=True, sort=True)
        inv_period = wealth_index.iloc[-1]
    ror = wealth_index.pct_change()
    ror = ror.iloc[1:]
    return ror

# Functions with float argument


def annualize_return(rate_of_return: float, periods_per_year=12) -> float:
    """
    Annualizes a return.
    Default annualization is from month to year.
    """
    return (rate_of_return+1.)**periods_per_year - 1.


def annualize_risk(risk: float, mean_return: float, periods_per_year=12) -> float:
    """
    Annualizes Risk.
    Annualization from month to year (from standard deviation) is by default. Mean return is also required.
    Works with DataFrame inputs (in math.sqrt is not used)
    """
    annualized_std = ((risk**2+(1+mean_return)**2)**12 - (1 + mean_return)**24)**0.5
    return annualized_std


def approx_return_risk_adjusted(mean_return: float, std: float) -> float:
    """
    Approximates geometric mean return given mean return and std.
    """
    return np.exp(np.log(1. + mean_return) - 0.5 * std ** 2 / (1. + mean_return) ** 2) - 1.

# Classes


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
            raise ValueError(self.market + ' is not a known namespace')

            
class AssetList:
    """
    The list of assets implementation.
    """
    def __init__(self, symbols=[default_ticker], curr='USD'):
        self.tickers = symbols
        self.currency = curr
        self._make_asset_list(symbols)
        self._calculate_wealth_indexes()
             
    def _make_asset_list(self, l:list):
        """
        Makes an asset list from a list of symbols. Returns dataframe (or series if one ticker) of returns (monthly) as an attribute.
        """
        for i,x in enumerate(l): 
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
        self.ror = df
    
    def _set_currency(self, returns:pd.Series, currency: str):
        """
        Set return to a certain currency. Works with a list of assets.
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

    def _calculate_wealth_indexes(self, initial_investments=1000):
        """
        Returns wealth index for a list of assets.
        """
        self.wealth_indexes = initial_investments * (self.ror + 1).cumprod()

    def calculate_risk(self, annualize=False, periods_per_year=12) -> pd.Series:
        """
        Takes assets returns DataFrame and calcultes risks (std) for each asset. Default risk is monthly standard deviation.
        If annualize=True calculate annualized values.
        """
        risk = self.ror.std()
        mean_return = self.ror.mean()
        if annualize: risk = annualize_risk(risk, mean_return, periods_per_year)
        return risk

    def calculate_cagr(self):
        return ((self.ror + 1.).prod()) ** (12 / self.ror.shape[0]) - 1


class Portfolio:
    """
    Implementation of investment portfolio. Investment portfolio is an AssetList + Weights
    """
    def __init__(self, *, symbols=[default_ticker], curr='USD', weights=None):
        self.currency = curr
        self.tickers = symbols
        if weights is None:
            n = len(symbols) # number of assets
            weights = list(np.repeat(1/n, n))
            self.weights = weights
        else:
           self.weights = weights
        self._ror = AssetList(symbols, curr).ror # AssetsList returns pd.DataFrame
        self.returns_ts = get_portfolio_return_ts(self.weights, self._ror)
        self.rebalanced_portfolio_return_ts = rebalanced_portfolio_return_ts(self.weights, self._ror)
        self.mean_return_monthly = get_portfolio_mean_return(self.weights, self._ror)
        self.mean_return_annual = annualize_return(self.mean_return_monthly)
        self.risk_monthly = get_portfolio_risk(self.weights, self._ror)
        self.risk_annual = annualize_risk(self.risk_monthly, self.returns_ts.mean())

    #
    # def risk_annual(self, risk: float, mean_return: float, periods_per_year=12) -> float:
    #     """
    #     Annualizes Risk.
    #     Default is annualization from month to year from standard deviation. Mean return is also required.
    #     """
    #     annualized_std = math.sqrt((risk**2+(1+mean_return)**2)**12 - (1 + mean_return)**24)
    #     return annualized_std