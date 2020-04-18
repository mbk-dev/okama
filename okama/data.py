import requests
from urllib3.exceptions import InsecureRequestWarning
from io import StringIO
import json
import time

import pandas as pd

from .helpers import String
from .settings import default_ticker, EOD_url, eod_search_url, eod_exchanges_url, api_token


def get_eod_data(symbol=default_ticker, type='return', session=None, check_zeros=True) -> pd.Series:
    """
    Get rate of return for a set of ror in the same currency. Returns daily data.
    type: return - Rate of Return pd.Series
    type: nav - Net Asset Value (works with RUFUND only)
    type: close - Adjusted Close values
    """
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    if session is None:
        session = requests.session()
    url = EOD_url + symbol
    params = {'api_token': api_token}
    r = session.get(url, params=params, verify=False)
    if r.status_code == requests.codes.ok:
        r.connection.close()
        df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine='python')
        if symbol.split('.',1)[-1] == 'RUFUND':
            if type == 'nav':
                df = df['Nav']
                df.rename(symbol, inplace=True)
                return df
            df = df['Price']
        else:
            df = df['Adjusted_close']
        df.index = df.index.to_period('D')
        df.sort_index(ascending = True, inplace=True)
        if check_zeros:
            if (df == 0).any():
                raise Exception("Zero close values in data")
            if df.isna().any():
                raise Exception("NaN values in data")
        if type == 'close':
            df.rename(symbol, inplace=True)
            return df
        if type == 'nav':
            raise Exception("NAV is not available for this type of data")
        if type == 'return':
            df = df.pct_change()
            df = df.iloc[1:]
        if df.isna().any():
            raise Exception("NaN values in data")
        df.rename(symbol, inplace=True)
        return df
    else:
        raise Exception(r.status_code, r.reason, url)


def search(string: str, session=None):
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    if session is None:
        session = requests.Session()
    url = eod_search_url + string
    params = {'api_token': api_token}
    r = session.get(url, params=params, verify=False)
    if r.status_code == requests.codes.ok:
        r.connection.close()
        d = json.loads(r.text)
        return d
    else:
        raise Exception(r.status_code, r.reason, url)


def get_market_data(market: str, session=None):
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    if session is None:
        session = requests.Session()
    url = eod_exchanges_url + market
    params = {'api_token': api_token}
    r = session.get(url, params=params, verify=False)
    if r.status_code == requests.codes.ok:
        r.connection.close()
        df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine='python')
        return df
    else:
        raise Exception(r.status_code, r.reason, url)


def get_ticker_data(char: str) -> str:
    # main_start_time = time.time()
    ticker = String.define_market(char)[0]
    market = String.define_market(char)[1]
    df = get_market_data(market)
    x = {
        'name': df.loc[ticker, 'Name'],
        'country': df.loc[ticker, 'Country'],
        'exchange': df.loc[ticker, 'Exchange'],
        'currency': df.loc[ticker, 'Currency'],
        'type': df.loc[ticker, 'Type']
    }
    # main_end_time = time.time()
    # print(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
    return x

