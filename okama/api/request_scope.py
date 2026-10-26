"""Operation-local HTTP connections and bounded symbol metadata reuse."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock, get_ident

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_MetadataKey = tuple[type, str, str, str, str | None, str | None, str]


class _RequestScope:
    _MAX_METADATA_ENTRIES = 256
    _MAX_METADATA_BYTES = 4 * 1024 * 1024

    def __init__(self) -> None:
        self.pid = os.getpid()
        self._sessions: dict[int, requests.Session] = {}
        self._metadata: dict[_MetadataKey, str] = {}
        self._metadata_bytes = 0
        self._lock = RLock()
        self._closed = False

    def session(self) -> requests.Session:
        with self._lock:
            if self._closed:
                raise RuntimeError("The data request scope is closed")
            thread = get_ident()
            if thread not in self._sessions:
                session = requests.session()
                retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
                adapter = HTTPAdapter(max_retries=retries)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                self._sessions[thread] = session
            return self._sessions[thread]

    def metadata(self, key: _MetadataKey) -> str | None:
        with self._lock:
            return self._metadata.get(key)

    def remember_metadata(self, key: _MetadataKey, text: str) -> None:
        try:
            if not isinstance(json.loads(text), dict):
                return
        except (ValueError, TypeError):
            return
        size = len(text.encode("utf-8"))
        with self._lock:
            if (
                key not in self._metadata
                and len(self._metadata) < self._MAX_METADATA_ENTRIES
                and self._metadata_bytes + size <= self._MAX_METADATA_BYTES
            ):
                self._metadata[key] = text
                self._metadata_bytes += size

    def close(self) -> None:
        with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._metadata.clear()
            self._metadata_bytes = 0
        for session in sessions:
            session.close()


_scope: ContextVar[_RequestScope | None] = ContextVar("okama_data_request_scope", default=None)


def _get_request_scope() -> _RequestScope | None:
    scope = _scope.get()
    return scope if scope is not None and scope.pid == os.getpid() else None


@contextmanager
def _bind_request_scope(scope: _RequestScope | None) -> Iterator[None]:
    """Borrow an operation scope inside one joined loading worker."""
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


@contextmanager
def data_request_scope() -> Iterator[_RequestScope]:
    """Reuse thread-isolated HTTP connections and metadata during related operations.

    Asset and AssetList constructors create scopes automatically. An optional
    outer context also reuses metadata between successive constructors. Only
    successful JSON-object symbol metadata is cached, for this context's lifetime;
    time series, prices and search responses remain fresh requests. Exiting the
    outer context closes its sessions. Join borrowed loading workers before exit.
    """
    existing = _get_request_scope()
    if existing is not None:
        yield existing
        return
    scope = _RequestScope()
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)
        scope.close()
