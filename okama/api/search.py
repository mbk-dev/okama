import json

from .api_methods import API


def search(search_string: str) -> json:
    string_response = API.search(search_string)
    return json.loads(string_response)
