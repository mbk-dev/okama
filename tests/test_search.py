"""
Tests the search
"""
import json

import pytest

from okama.api import search


def test_search_namespace_json():
    x = search.search(
        "aeroflot",
        namespace="MOEX",
        response_format="json",
    )
    assert json.loads(x)[0]["symbol"] == "AFLT.MOEX"


def test_search_namespace_frame():
    x = search.search(
        "aeroflot",
        namespace="MOEX",
        response_format="frame",
    )
    assert x["symbol"].values[0] == "AFLT.MOEX"


def test_search_all_json():
    x = search.search(
        "lkoh",
        response_format="json",
    )
    assert {x[1][0], x[2][0]} == {"LKOH.MOEX", "LKOH.LSE"}


def test_search_all_frame():
    x = search.search(
        "lkoh",
        response_format="frame",
    )
    assert {x["symbol"].iloc[0], x["symbol"].iloc[1]} == {"LKOH.MOEX", "LKOH.LSE"}


def test_search_error():
    with pytest.raises(ValueError):
        search.search("arg", response_format="txt")
