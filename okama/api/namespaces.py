import json
from functools import lru_cache

import pandas as pd

from .api_methods import API
from ..settings import default_namespace


@lru_cache()
def get_namespaces():
    string_response = API.get_namespaces()
    return json.loads(string_response)


@lru_cache()
def symbols_in_namespace(namespace: str = default_namespace, response_format: str = 'frame') -> pd.DataFrame:
    string_response = API.get_symbols_in_namespace(namespace.upper())
    list_of_symbols = json.loads(string_response)
    if response_format.lower() == 'frame':
        df = pd.DataFrame(list_of_symbols[1:], columns=list_of_symbols[0])
        return df.astype("string", copy=False)
    elif response_format.lower() == 'json':
        return list_of_symbols
    else:
        raise ValueError('response_format must be "json" or "frame"')


@lru_cache()
def get_assets_namespaces():
    string_response = API.get_assets_namespaces()
    return json.loads(string_response)


@lru_cache()
def get_macro_namespaces():
    string_response = API.get_macro_namespaces()
    return json.loads(string_response)


@lru_cache()
def no_dividends_namespaces():
    string_response = API.get_no_dividends_namespaces()
    return json.loads(string_response)


namespaces = get_namespaces()
assets_namespaces = get_assets_namespaces()
macro_namespaces = get_macro_namespaces()
