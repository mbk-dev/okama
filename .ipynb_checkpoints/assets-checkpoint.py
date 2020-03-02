import math

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Get EOD Data
import requests
from urllib3.exceptions import InsecureRequestWarning
from io import StringIO

# Settings
default_ticker = 'SPY.US'
EOD_url = 'https://eodhistoricaldata.com/api/eod/'
api_token = '5cf7a8132af958.27978804'

# Get Data functions
def get_eod_data(symbol=default_ticker, session=None) -> pd.Series:
    '''
    Get rate of return for a set of assets in the same currency.
    '''
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    if session is None:
        session = requests.Session()
    # url = 'https://eodhistoricaldata.com/api/eod/%s' % symbol
    url = EOD_url + symbol
    params = {'api_token': api_token}
    r = session.get(url, params=params, verify=False)
    if r.status_code == requests.codes.ok:
        df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine='python')
        if symbol.split('.',1)[-1] == 'RUFUND':
            df = df['Price']
        else:
            df = df['Adjusted_close']
        df.index = df.index.to_period('D')
        df.sort_index(ascending = True, inplace=True)
        df = df.pct_change()
        df = df.iloc[1:]
        if df.isna().any(): raise Exception("NaN values in data")
        df.rename(symbol, inplace=True)
        return df
    else:
        raise Exception(r.status_code, r.reason, url)

# Classes
class Asset:
    '''
    An asset, that could be used in a list or portfolio.
    '''
    def __init__(self, symbol=default_ticker):
        self.ticker = symbol
        self.ror = self._get_monly_ror(symbol)
        self.market = self._define_market(symbol)
        self.asset_currency = self._define_curency()
    def _get_monly_ror(self, ticker: str) -> pd.Series:
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
        
    def _define_curency(self):
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
    '''
    The list of assets implementation.
    '''
    def __init__(self, symbols=[default_ticker], curr='USD'):
        self.currency = curr
        self.tickers = symbols
        self._make_asset_list(symbols)
        self._calculate_wealth_indexes()
             
    def _make_asset_list(self, l:list):
        '''
        Makes an asset list from a list of symbols. Returns dataframe (or series if one ticker) of returns (monthly) as an attribute.
        '''
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
        '''
        Set return to a certain currency. Works with a list of assets.
        '''
        currency_returns = Asset('RUB.FOREX').ror
        df = pd.concat([returns, currency_returns], axis = 1, join='inner', copy='false') # join dataframes to have the same Time Series Index
        currency_returns = df.iloc[:,-1]
        df = df.drop(columns=['RUB.FOREX'])
        if currency == 'USD':
            y = currency_returns + 1
            x = (df+1).mul(1/y, axis=0) - 1
        elif currency == 'RUB':
            y = currency_returns + 1
            x = (df+1).mul(y, axis=0) - 1
        return x

    def _calculate_wealth_indexes(self, initial_investments=1000):
        '''
        Returns wealth index for a list of assets.
        '''
        self.wealth_indexes = initial_investments * (self.ror + 1).cumprod()

        
class Portfolio:
    '''
    Implementation of investment portfolio. Investment portfolio is an AssetList + Weights
    '''
    def __init__(self, *, symbols=[default_ticker], curr='USD', weights=None):
        self.currency = curr
        self.tickers = symbols
        if weights == None:
            n = len(symbols) # number of assets
            weights = list(np.repeat(1/n, n))
            self.weights = weights
        else:
           self.weights = weights
        self._ror = AssetList(symbols, curr).ror # AssetsList returns pd.DataFrame
        self.returns = self._get_portfolio_returns(self._ror, self.weights)
        self.mean_return_monthly = self._get_portfolio_mean_return(self._ror, self.weights)
        self.mean_return_annual = self._annualize_return(self.mean_return_monthly)
        self.risk_monthly = self._get_portfolio_risk(self._ror, self.weights)
        self.risk_annual = self._annualize_risk(self.risk_monthly, self.returns.mean())[0]
        
    def _get_portfolio_mean_return(self, ror: pd.DataFrame, weights:list) -> float:
        """
        Computes mean return of a portfolio. Returns a single float number.
        """
        weights = np.asarray(weights)
        if isinstance(ror.mean(), float): # required for a single asset portfolio as AssetList(symbols, curr).ror.mean() returns float
            return ror.mean()
        return weights.T @ ror.mean()
    
    def _get_portfolio_returns(self, ror: pd.DataFrame, weights:list) -> pd.Series:
        if isinstance(ror, pd.Series): # required for a single asset portfolio as AssetList(symbols, curr).ror.mean() returns float
            return ror
        returns = ror @ weights
        return returns
    
    def _annualize_return(self, ror: float, periods_per_year=12) -> float:
        """
        Annualizes a return.
        Default is annulization from month to year.
        """
        return (ror+1.)**periods_per_year - 1. 
    
    def _get_portfolio_risk(self, ror:pd.DataFrame, weights:list) -> float:
        """
        Computes the std of portfolio returns.
        """
        if isinstance(ror, pd.Series): # required for a single asset portfolio as AssetList(symbols, curr).ror.mean() returns float
            return ror.std()
        weights = np.array(weights)
        covmat = ror.cov()
        return math.sqrt(weights.T @ covmat @ weights)
    
    def _annualize_risk(self, risk: float, mean_return: float, periods_per_year=12) -> float:
        """
        Annualizes Rsik.
        Default is annulization from month to year from standard deviation. Mean return is also required.     
        """
        wrong_annualized_std = risk * math.sqrt(periods_per_year)
        correct_annulized_std = math.sqrt((risk**2+(1+mean_return)**2)**12 - (1 + mean_return)**24)
        return (correct_annulized_std, wrong_annualized_std)

'''
class EfficientFrontier:
    def __init__(self, ror: pd.DataFrame):
        self.ror = ror
        
    def minimize_risk(target_return, self.ror):
        """
        Returns the optimal weights that achieve the target return
        given a set of expected returns and a covariance matrix
        """
        n = self.ror.shape[1] # number of assets

        init_guess = np.repeat(1/n, n) # initial weights
        bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples! Weights constrains
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
        }
        return_is_target = {'type': 'eq',
                            'args': (assets,),
                            'fun': lambda weights, assets: target_return - portfolio_return(weights,assets)
        }
        weights = minimize(portfolio_risk, init_guess,
                           args=(assets,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,return_is_target),
                           bounds=bounds)
        return weights.x    
    
    def optimal_weights(n_points, assets):
        """
        Returns a set of weights for Efficient Frontier
        """
        er = assets.mean()
        cov = assets.cov()
        target_rs = np.linspace(er.min(), er.max(), n_points)
        weights = [minimize_risk(target_return, assets) for target_return in target_rs]
        return weights
'''        