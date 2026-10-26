"""Offline transport lifetime and scoped metadata regressions."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
import requests

import okama as ok
from okama.api.api_methods import API
from okama.api.data_queries import QueryData


@pytest.fixture()
def transport(monkeypatch, mocker):
    calls = []
    factory = mocker.spy(requests, "session")
    close = mocker.spy(requests.Session, "close")
    monkeypatch.setattr("okama.api.namespaces.get_assets_namespaces", lambda: ["US", "FX"])

    def get(session, url, params, verify, timeout):
        calls.append(
            {
                "session": session,
                "thread": threading.get_ident(),
                "url": url,
                "params": params.copy(),
                "verify": verify,
                "timeout": timeout,
            }
        )
        symbol = url.rsplit("/", 1)[-1]
        response = requests.Response()
        response.status_code, response.url = 200, url
        if "/api/symbol/" in url:
            body = json.dumps(
                {
                    "code": symbol.split(".")[0],
                    "name": symbol,
                    "country": "US",
                    "exchange": "X",
                    "currency": "USD",
                    "type": "ETF",
                    "isin": "test",
                }
            )
        else:
            body = pd.Series(0.001, index=pd.date_range("2010-01-01", periods=36, freq="MS"), name=symbol).to_csv()
        response._content = body.encode()
        return response

    monkeypatch.setattr(requests.Session, "get", get)
    return calls, factory, close


def test_asset_constructor_reuses_one_session(transport) -> None:
    calls, factory, close = transport
    asset = ok.Asset("A.US")
    assert len(asset.ror) == 36
    assert len(calls) == 2
    assert factory.call_count == 1
    assert close.call_count == 1


def test_scoped_metadata_is_decoded_fresh_and_expires(transport) -> None:
    calls, factory, close = transport
    with ok.data_request_scope():
        first = QueryData.get_symbol_info("A.US")
        first["name"] = "mutated"
        second = QueryData.get_symbol_info("A.US")
        assert second["name"] == "A.US"
        assert len(calls) == 1
        assert close.call_count == 0
    assert close.call_count == 1
    with ok.data_request_scope():
        QueryData.get_symbol_info("A.US")
    assert len(calls) == factory.call_count == close.call_count == 2


def test_nested_scope_preserves_outer_session_until_exit(transport) -> None:
    calls, factory, close = transport
    with ok.data_request_scope():
        QueryData.get_symbol_info("A.US")
        with ok.data_request_scope():
            QueryData.get_symbol_info("B.US")
        assert close.call_count == 0
        QueryData.get_symbol_info("C.US")
    assert len(calls) == 3
    assert factory.call_count == close.call_count == 1


def test_timeseries_ranges_are_not_cached(transport) -> None:
    calls, _, _ = transport
    with ok.data_request_scope():
        API.get_ror("USD.FX", first_date="2005-01", last_date="2025-12")
        API.get_ror("USD.FX", first_date="1913-01-01", last_date="2100-01-01")
        API.get_ror("USD.FX", first_date="2005-01", last_date="2025-12")
    assert len(calls) == 3
    assert calls[0]["params"] != calls[1]["params"]


def test_scope_closes_sessions_when_body_raises(transport) -> None:
    _, factory, close = transport
    with pytest.raises(RuntimeError, match="stop"):
        with ok.data_request_scope():
            QueryData.get_symbol_info("A.US")
            raise RuntimeError("stop")
    assert factory.call_count == close.call_count == 1


def test_listmaker_workers_borrow_scope_with_separate_sessions(transport) -> None:
    calls, factory, close = transport
    with ok.data_request_scope():
        assets = ok.AssetList(["A.US", "B.US", "C.US"], inflation=False)
        assert len(assets.assets_ror) == 36
        assert close.call_count == 0
        before = len(calls)
        QueryData.get_symbol_info("A.US")
        assert len(calls) == before
    sessions_by_thread = {}
    for call in calls:
        sessions_by_thread.setdefault(call["thread"], set()).add(id(call["session"]))
    assert all(len(sessions) == 1 for sessions in sessions_by_thread.values())
    assert len({id(call["session"]) for call in calls}) == len(sessions_by_thread)
    assert factory.call_count == close.call_count == len(sessions_by_thread)


def test_independent_thread_scopes_do_not_share_sessions_or_metadata(transport) -> None:
    calls, factory, close = transport
    barrier = threading.Barrier(2)

    def worker() -> str:
        with ok.data_request_scope():
            barrier.wait(timeout=5)
            QueryData.get_symbol_info("A.US")
            return QueryData.get_symbol_info("A.US")["name"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(lambda _: worker(), range(2))) == ["A.US", "A.US"]
    assert len(calls) == factory.call_count == close.call_count == 2
    assert len({id(c["session"]) for c in calls}) == 2


def test_cache_separates_api_class_and_url(transport, monkeypatch) -> None:
    calls, _, _ = transport

    class OtherAPI(API):
        pass

    with ok.data_request_scope():
        API.get_symbol_info("A.US")
        OtherAPI.get_symbol_info("A.US")
        monkeypatch.setattr(API, "api_url", "https://other.example")
        API.get_symbol_info("A.US")
    assert len(calls) == 3


@pytest.mark.parametrize("status", [400, 404, 500])
def test_failed_metadata_is_not_cached(transport, monkeypatch, status: int) -> None:
    calls, factory, close = transport
    get = requests.Session.get
    attempts = 0

    def fail_once(session, url, **kwargs):
        nonlocal attempts
        response = get(session, url, **kwargs)
        attempts += 1
        if attempts == 1:
            response.status_code = status
            response.reason = "test error"
        return response

    monkeypatch.setattr(requests.Session, "get", fail_once)
    with ok.data_request_scope():
        with pytest.raises(requests.HTTPError):
            API.get_symbol_info("A.US")
        assert QueryData.get_symbol_info("A.US")["name"] == "A.US"
    assert len(calls) == 2
    assert factory.call_count == close.call_count == 1


def test_malformed_json_is_not_cached(transport, monkeypatch) -> None:
    calls, _, _ = transport
    get = requests.Session.get
    attempts = 0

    def malformed_once(session, url, **kwargs):
        nonlocal attempts
        response = get(session, url, **kwargs)
        attempts += 1
        if attempts == 1:
            response._content = b"not JSON"
        return response

    monkeypatch.setattr(requests.Session, "get", malformed_once)
    with ok.data_request_scope():
        with pytest.raises(ValueError):
            QueryData.get_symbol_info("A.US")
        assert QueryData.get_symbol_info("A.US")["name"] == "A.US"
    assert len(calls) == 2


@pytest.mark.parametrize("limit", ["entries", "bytes"])
def test_metadata_limit_falls_back_to_fresh_requests(transport, monkeypatch, limit: str) -> None:
    from okama.api.request_scope import _RequestScope

    calls, _, _ = transport
    monkeypatch.setattr(_RequestScope, "_MAX_METADATA_ENTRIES" if limit == "entries" else "_MAX_METADATA_BYTES", 1)
    with ok.data_request_scope():
        for name in ("A.US", "B.US", "B.US", "A.US"):
            QueryData.get_symbol_info(name)
    assert len(calls) == (3 if limit == "entries" else 4)


def test_unscoped_calls_remain_fresh(transport) -> None:
    calls, factory, close = transport
    API.get_symbol_info("A.US")
    API.get_symbol_info("A.US")
    assert len(calls) == factory.call_count == close.call_count == 2


def test_transport_keeps_timeouts_tls_retries_and_timestamp_parameters(transport) -> None:
    calls, _, _ = transport
    API.get_ror("A.US", first_date=pd.Timestamp("2010-01-01"), last_date=pd.Timestamp("2012-12-01"))
    call = calls[0]
    assert call["params"] == {"first_date": "2010-01-01", "last_date": "2012-12-01", "period": "m"}
    assert call["verify"] is True
    assert call["timeout"] == API.default_timeout
    retries = call["session"].adapters["https://"].max_retries
    assert retries.total == 3
    assert retries.backoff_factor == 0.1
    assert retries.status_forcelist == [429, 500, 502, 503, 504]


def test_metadata_cache_key_preserves_complete_request_parameters(transport) -> None:
    calls, _, _ = transport
    with ok.data_request_scope():
        API.connect(endpoint=API.endpoint_symbol, symbol="A.US", first_date="2010-01-01", period="m")
        API.connect(endpoint=API.endpoint_symbol, symbol="A.US", first_date="2011-01-01", period="m")
        API.connect(endpoint=API.endpoint_symbol, symbol="A.US", first_date="2011-01-01", period="d")
        API.connect(endpoint=API.endpoint_symbol, symbol="A.US", first_date=pd.Timestamp("2010-01-01"), period="m")
    assert len(calls) == 3
