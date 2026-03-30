import json
from typing import Optional
from functools import lru_cache

import pandas as pd

from okama.api import api_methods, namespaces


@lru_cache()
def search(search_string: str, namespace: Optional[str] = None, response_format: str = "frame") -> json:
    """
    Search symbols by ticker, name, or ISIN.

    When ``namespace`` is provided, the search is performed within the cached
    table returned by ``ok.symbols_in_namespace(namespace)``. Otherwise the
    query is delegated to the API search endpoint across all namespaces.

    Parameters
    ----------
    search_string : str
        Case-insensitive text used to match symbol names, tickers, and ISINs.

    namespace : str, optional
        Namespace code such as ``"US"`` or ``"XETR"``. If omitted, all
        available namespaces are searched.

    response_format : {'frame', 'json'}, default 'frame'
        Format of the returned search results.

    Returns
    -------
    pandas.DataFrame or str or list
        Search results.

        - Returns a ``DataFrame`` when ``response_format='frame'``.
        - Returns a JSON string in pandas ``records`` orientation when
          ``namespace`` is provided and ``response_format='json'``.
        - Returns the parsed API JSON payload as a list when ``namespace`` is
          omitted and ``response_format='json'``.

    Raises
    ------
    ValueError
        If ``response_format`` is not ``'frame'`` or ``'json'``.

    Examples
    --------
    >>> result = ok.search("SPY", namespace="US")
    >>> result.empty
    False
    >>> {"symbol", "ticker", "name"}.issubset(result.columns)
    True
    """
    # search for string in a single namespace
    if namespace:
        df = namespaces.symbols_in_namespace(namespace.upper())
        condition1 = df["name"].str.contains(search_string, case=False)
        condition2 = df["ticker"].str.contains(search_string, case=False)
        condition3 = df["isin"].str.contains(search_string, case=False)
        frame_response = df[condition1 | condition2 | condition3]
        if response_format.lower() == "frame":
            return frame_response
        elif response_format.lower() == "json":
            return frame_response.to_json(orient="records")
        else:
            raise ValueError('response_format must be "json" or "frame"')
    # search for string in all namespaces
    string_response = api_methods.API.search(search_string)
    json_response = json.loads(string_response)
    if response_format.lower() == "frame":
        df = pd.DataFrame(json_response[1:], columns=json_response[0])
        return df
    elif response_format.lower() == "json":
        return json_response
    else:
        raise ValueError('response_format must be "json" or "frame"')
