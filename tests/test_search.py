import json  # noqa: I001
import pandas as pd
import pytest

from okama.api import search as search_module


@pytest.fixture(autouse=True)
def clear_cache():
    # Ensure LRU cache on search() does not leak between tests
    search_module.search.cache_clear()


def test_search_namespace_json_mocked(mocker):
    # Prepare a fake namespace DataFrame
    df = pd.DataFrame(
        [
            {
                "name": "PJSC Aeroflot",
                "ticker": "AFLT",
                "isin": "RU0009062285",
                "symbol": "AFLT.MOEX",
            },
            {
                "name": "Another Corp",
                "ticker": "ANTR",
                "isin": "RU0000000000",
                "symbol": "ANTR.MOEX",
            },
        ]
    )

    mocker.patch(
        "okama.api.namespaces.symbols_in_namespace",
        return_value=df,
    )

    res = search_module.search("aero", namespace="MOEX", response_format="json")
    parsed = json.loads(res)
    assert isinstance(parsed, list)
    assert parsed[0]["symbol"] == "AFLT.MOEX"


def test_search_namespace_frame_mocked(mocker):
    df = pd.DataFrame(
        [
            {
                "name": "PJSC Aeroflot",
                "ticker": "AFLT",
                "isin": "RU0009062285",
                "symbol": "AFLT.MOEX",
            },
            {
                "name": "Another Corp",
                "ticker": "ANTR",
                "isin": "RU0000000000",
                "symbol": "ANTR.MOEX",
            },
        ]
    )

    mocker.patch(
        "okama.api.namespaces.symbols_in_namespace",
        return_value=df,
    )

    res = search_module.search("aero", namespace="MOEX", response_format="frame")
    # Only rows that contain the search string should be returned
    assert isinstance(res, pd.DataFrame)
    assert res.shape[0] >= 1
    assert res.iloc[0]["symbol"] == "AFLT.MOEX"


def test_search_all_json_mocked(mocker):
    # API.search returns a JSON string; the first row is header
    payload = [
        ["symbol", "name"],
        ["LKOH.MOEX", "Lukoil PJSC"],
        ["LKOH.LSE", "Lukoil PLC"],
    ]
    mocker.patch(
        "okama.api.api_methods.API.search",
        return_value=json.dumps(payload),
    )

    res = search_module.search("lkoh", response_format="json")
    assert isinstance(res, list)
    # After json.loads in the implementation, we expect the same nested list
    assert {res[1][0], res[2][0]} == {"LKOH.MOEX", "LKOH.LSE"}


def test_search_all_frame_mocked(mocker):
    payload = [
        ["symbol", "name"],
        ["LKOH.MOEX", "Lukoil PJSC"],
        ["LKOH.LSE", "Lukoil PLC"],
    ]
    mocker.patch(
        "okama.api.api_methods.API.search",
        return_value=json.dumps(payload),
    )

    res = search_module.search("lkoh", response_format="frame")
    assert isinstance(res, pd.DataFrame)
    assert set(res["symbol"].tolist()) == {"LKOH.MOEX", "LKOH.LSE"}


def test_search_namespace_matches_local_name(mocker):
    """Namespace search also matches the native-language local_name column."""
    df = pd.DataFrame(
        [
            {
                "name": "Aeroflot",
                "ticker": "AFLT",
                "isin": "RU0009062285",
                "symbol": "AFLT.MOEX",
                "local_name": "Аэрофлот",
            },
            {
                "name": "Another Corp",
                "ticker": "ANTR",
                "isin": "RU0000000000",
                "symbol": "ANTR.MOEX",
                "local_name": None,  # a null local_name must not break the boolean mask
            },
        ]
    )
    mocker.patch(
        "okama.api.namespaces.symbols_in_namespace",
        return_value=df,
    )

    res = search_module.search("Аэрофлот", namespace="MOEX", response_format="frame")
    # matched only via local_name — name/ticker/isin do not contain the Cyrillic string
    assert list(res["symbol"]) == ["AFLT.MOEX"]


def test_search_namespace_without_local_name_column(mocker):
    """Namespaces whose data has no local_name column still search by name/ticker/isin."""
    df = pd.DataFrame(
        [
            {
                "name": "PJSC Aeroflot",
                "ticker": "AFLT",
                "isin": "RU0009062285",
                "symbol": "AFLT.MOEX",
            },
        ]
    )
    mocker.patch(
        "okama.api.namespaces.symbols_in_namespace",
        return_value=df,
    )

    res = search_module.search("aero", namespace="MOEX", response_format="frame")
    assert list(res["symbol"]) == ["AFLT.MOEX"]


def test_search_error_invalid_format(mocker):
    # Prevent real network call by mocking API.search
    mocker.patch(
        "okama.api.api_methods.API.search",
        return_value='[["symbol","name"],["AAA.NS","Asset AAA"]]',
    )
    with pytest.raises(ValueError):
        search_module.search("anything", response_format="txt")
