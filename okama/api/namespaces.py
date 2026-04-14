import json  # noqa: I001
from functools import lru_cache

import pandas as pd

from okama.api import api_methods
from okama import settings


@lru_cache()  # noqa: UP011
def get_namespaces():
    """
    Return the namespace mapping exposed as ``ok.namespaces``.

    The returned dictionary maps namespace codes to human-readable
    descriptions.

    Returns
    -------
    dict
        Dictionary where keys are namespace codes and values are namespace
        descriptions.

    Examples
    --------
    >>> "US" in ok.namespaces
    True
    >>> isinstance(ok.namespaces["US"], str)
    True
    """
    string_response = api_methods.API.get_namespaces()
    return json.loads(string_response)


@lru_cache()  # noqa: UP011
def symbols_in_namespace(namespace: str = settings.default_namespace, response_format: str = "frame") -> pd.DataFrame:
    """
    Return all symbols available in a namespace.

    Parameters
    ----------
    namespace : str, default settings.default_namespace
        Namespace code such as ``"US"``, ``"XETR"``, or ``"INFL"``.

    response_format : {'frame', 'json'}, default 'frame'
        Format of the returned namespace contents.

    Returns
    -------
    pandas.DataFrame or list
        Namespace contents.

        - Returns a ``DataFrame`` when ``response_format='frame'``.
        - Returns the parsed API JSON payload as a list when
          ``response_format='json'``.

    Raises
    ------
    ValueError
        If ``response_format`` is not ``'frame'`` or ``'json'``.

    Examples
    --------
    >>> symbols = ok.symbols_in_namespace("US")
    >>> symbols.empty
    False
    >>> {"symbol", "name", "currency"}.issubset(symbols.columns)
    True
    """
    string_response = api_methods.API.get_symbols_in_namespace(namespace.upper())
    list_of_symbols = json.loads(string_response)
    if response_format.lower() == "frame":
        df = pd.DataFrame(list_of_symbols[1:], columns=list_of_symbols[0])
        return df.astype("string", copy=False)
    elif response_format.lower() == "json":
        return list_of_symbols
    else:
        raise ValueError('response_format must be "json" or "frame"')


@lru_cache()  # noqa: UP011
def get_assets_namespaces():
    string_response = api_methods.API.get_assets_namespaces()
    return json.loads(string_response)


@lru_cache()  # noqa: UP011
def get_macro_namespaces():
    string_response = api_methods.API.get_macro_namespaces()
    return json.loads(string_response)


@lru_cache()  # noqa: UP011
def no_dividends_namespaces():
    string_response = api_methods.API.get_no_dividends_namespaces()
    return json.loads(string_response)


# namespaces = get_namespaces()
# assets_namespaces = get_assets_namespaces()
# macro_namespaces = get_macro_namespaces()
