import json
from typing import Optional

import pandas as pd

from .api_methods import API
from .namespaces import symbols_in_namespace


def search(search_string: str, namespace: Optional[str] = None, response_format: str = 'frame') -> json:
    # search for string in a single namespace
    if namespace:
        df = symbols_in_namespace(namespace.upper())
        condition1 = df['name'].str.contains(search_string, case=False)
        condition2 = df['ticker'].str.contains(search_string, case=False)
        frame_response = df[condition1 | condition2]
        if response_format.lower() == 'frame':
            return frame_response
        elif response_format.lower() == 'json':
            return frame_response.to_json(orient='records')
        else:
            raise ValueError('response_format must be "json" or "frame"')
    # search for string in all namespaces
    string_response = API.search(search_string)
    json_response = json.loads(string_response)
    if response_format.lower() == 'frame':
        df = pd.DataFrame(json_response[1:], columns=json_response[0])
        return df
    elif response_format.lower() == 'json':
        return json_response
    else:
        raise ValueError('response_format must be "json" or "frame"')
