import json
import warnings

import pandas as pd
import pytest

from okama.api import namespaces


@pytest.fixture(autouse=True)
def clear_symbols_cache():
    # symbols_in_namespace is lru_cached: clear before and after so the mocked
    # payload never leaks into (or from) other tests.
    namespaces.symbols_in_namespace.cache_clear()
    yield
    namespaces.symbols_in_namespace.cache_clear()


def test_symbols_in_namespace_frame_emits_no_pandas4warning(mocker):
    """astype 'copy' keyword is deprecated on pandas 3 — the frame path must stay warning-free (GH #85)."""
    mocker.patch(
        "okama.api.api_methods.API.get_symbols_in_namespace",
        return_value=json.dumps([["symbol", "name"], ["AAA.US", "Aaa Corp"]]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.Pandas4Warning)
        df = namespaces.symbols_in_namespace("US")
    assert df.shape == (1, 2)
    assert df.dtypes.eq("string").all()
