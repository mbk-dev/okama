# Scoped API connection and metadata reuse

Earlier calculation improvements were delivered into dev through [PR134](https://github.com/mbk-dev/okama/pull/134). This follow-up reduces transport setup during actual public Asset/AssetList/frontier/Portfolio loading without a persistent cache or time-series reuse.

## Behavior

`Asset.__init__` and `ListMaker.__init__` automatically create an operation-local `data_request_scope`. Each thread gets its own requests.Session with the same retry count 3, backoff factor 0.1 and status list, TLS verification and per-request timeout as before. Metadata and return requests within one constructor can reuse a connection. Joined ListMaker loading workers borrow the operation scope but never share a Session across concurrent threads. Sessions close at outer exit, including exceptions. Unscoped standalone API calls remain fresh independent operations.

An optional outer `okama.data_request_scope()` also reuses successful symbol metadata across successive operations:

```python
import okama as ok

with ok.data_request_scope():
    frontier = ok.EfficientFrontierSingle(["SPY.US", "BND.US", "GLD.US"], ccy="USD", inflation=False)
    portfolio = ok.Portfolio(list(frontier.asset_obj_dict.values()), ccy="USD", inflation=False)
```

Only JSON-object metadata text is cached, bounded to 256 entries and 4 MiB. Existing QueryData methods decode a fresh dictionary each call. Keys include API class, API URL, endpoint, symbol, normalized dates and period. Nested scopes borrow ownership; later unrelated contexts do not see the cache. Forked processes do not borrow inherited scopes. Keep a scope limited to one logical operation and join borrowed workers before exiting.

Time series, live prices and search are never cached. USD.FX return requests with different date ranges remain distinct; even repeated identical series requests are sent afresh. HTTP failures and malformed JSON are not cached. No global Session, credentials, process configuration or dependencies are added.

## Measurements

[benchmark_api_request_scope.py](https://github.com/mbk-dev/okama/blob/v4.0.1/analytics/benchmark_api_request_scope.py) compares the original API.connect from dev31db96c, automatic constructor scopes and an explicit outer scope. Earlier computational improvements are shared. Every workflow creates a new 20-point frontier for SPY/BND/GLD, geometric tangency and an annually rebalanced Portfolio, then evaluates returns, CAGR, risk history, rolling CAGR and events. Complete financial outputs/metadata compare at rtol=atol=1e-12. The original connect is patched only inside the benchmark process, with transport cleanup after timing.

Deterministic offline transport counts were:

| Mode | HTTP Sessions | Logical HTTP requests |
|---|---:|---:|
| Original connect | 11 | 11 |
| Automatic scopes | 5 | 11 |
| Explicit outer scope | 4 | 10 |

The outer context removes the second USD.FX metadata request. It retains both different USD return-range requests. Offline timings are diagnostic only; a fake transport cannot establish network speedup.

The final live experiment used one warm triple followed by three rotating triples on a shared host, with `first_date="2005-01"` / `last_date="2025-12"` and a shared 224-month history. All 12 observations succeeded and all compared financial values matched exactly. Python 3.14.8; command-local single BLAS/OpenMP thread and 3-second API timeout. Final medians, excluding warmup:

| Mode | Complete workflow median |
|---|---:|
| Original connect | 611.5 ms |
| Automatic scopes | 519.1 ms |
| Explicit outer scope | 572.8 ms |

Automatic scopes were about 1.18× faster in this experiment. The outer context used fewer requests but was not faster than automatic scopes in the final sample: its observations ranged from 376 to 775 ms. Network latency, remote caching and scheduling dominate this small sample; neither a general speedup nor an advantage on every call is established. A pre-hardening experiment had different timings and remains in private evidence; final published observations match the accepted source hashes.

Full receipts retain warmup, order, individual timings, request arguments, failures (none in accepted runs), counts and source/script hashes: [offline](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/api-offline-request-scope.json), [live](https://github.com/mbk-dev/okama/blob/v4.0.1/docs/performance/api-live-request-scope.json). Request durations can overlap in parallel loading and must not be summed as network wall time. Logical calls are API-level requests, not a count of internal retry attempts.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m analytics.benchmark_api_request_scope --offline --output tmp/api-offline.json
OKAMA_API_TIMEOUT=3 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m analytics.benchmark_api_request_scope --output tmp/api-live.json
```

## Verification

RED/GREEN first demonstrated duplicate session construction and missing scope behavior, then tested complete metadata-key parameters separately before hardening the key. Tests cover nested ownership, cleanup on errors, independent threads, worker scope propagation, changed API classes/URLs, dictionary mutation, metadata limits, invalid JSON/HTTP errors, date normalization, retry/TLS/timeout preservation and unchanged time-series calls. Final verification on Python 3.14.8: **922 passed, 3 skipped, 172 existing pandas warnings**. Both `uv run --no-sync ruff check .` and `uv run --no-sync ruff format --check .` passed; `git diff --check` passed. An independent Grok Build review found no blocking lifetime/concurrency issues. A second focused review confirmed the final full-parameter key and normalization; successful-JSON filtering remains in the scope store. Final benchmarks match accepted source/script hashes.
