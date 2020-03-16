import pandas as pd

# Get EOD Data
import requests
from urllib3.exceptions import InsecureRequestWarning
from io import StringIO

from .settings import default_ticker, EOD_url, api_token


def get_eod_data(symbol=default_ticker, type='return', session=None) -> pd.Series:
    """
    Get rate of return for a set of ror in the same currency.
    """
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    if session is None:
        session = requests.session()
        # session.config['keep_alive'] = False
    url = EOD_url + symbol
    params = {'api_token': api_token}
    r = session.get(url, params=params, verify=False)
    if r.status_code == requests.codes.ok:
        r.connection.close()  # TODO: check if closing connection slows down multiple requests
        df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine='python')
        if symbol.split('.',1)[-1] == 'RUFUND':
            df = df['Price']
        else:
            df = df['Adjusted_close']
        df.index = df.index.to_period('D')
        df.sort_index(ascending = True, inplace=True)
        if type == 'return':
            df = df.pct_change()
            df = df.iloc[1:]
        if df.isna().any(): raise Exception("NaN values in data")
        df.rename(symbol, inplace=True)
        return df
    else:
        raise Exception(r.status_code, r.reason, url)

def get_eod_close(symbol=default_ticker, session=None):
    requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    if session is None:
        session = requests.Session()
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
        if df.isna().any(): raise Exception("NaN values in data")
        df.rename(symbol, inplace=True)
        return df
    else:
        raise Exception(r.status_code, r.reason, url)