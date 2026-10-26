"""Transport counts and bounded rotating live comparisons of scoped API loading."""

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from threading import get_ident
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import okama as ok
from analytics.benchmark_frontier_workflow import compare
from okama import settings
from okama.api import namespaces
from okama.api.api_methods import API


BASE = "31db96c4411e24b30656d1f76f94e95849251e01"
SYMBOLS = ["SPY.US", "BND.US", "GLD.US"]


def original_connect() -> classmethod:
    source = subprocess.check_output(["git", "show", f"{BASE}:okama/api/api_methods.py"], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "API")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "connect")
    method.decorator_list = []
    namespace = {
        "pd": pd,
        "requests": requests,
        "Retry": Retry,
        "HTTPAdapter": HTTPAdapter,
        "settings": settings,
        "endpoint_ror": API.endpoint_ror,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "original_api_connect.py", "exec"), namespace)
    return classmethod(namespace["connect"])


def workflow() -> dict[str, Any]:
    frontier = ok.EfficientFrontierSingle(
        SYMBOLS, first_date="2005-01", last_date="2025-12", ccy="USD", inflation=False, n_points=20
    )
    points = frontier.ef_points
    tangency = frontier.get_tangency_portfolio(rate_of_return="cagr", rf_return=0.01)
    portfolio = ok.Portfolio(
        list(frontier.asset_obj_dict.values()),
        weights=tangency["Weights"].tolist(),
        ccy="USD",
        inflation=False,
        rebalancing_strategy=ok.Rebalance(period="year"),
        symbol="request_scope.PF",
    )
    return {
        "frontier": points,
        "tangency": tangency,
        "returns": portfolio.ror,
        "cagr": portfolio.get_cagr(),
        "risk": portfolio.risk_annual,
        "rolling": portfolio.get_rolling_cagr(60),
        "events": portfolio.rebalancing_events,
    }


def offline_transport() -> Any:
    index = pd.date_range("2005-01-01", periods=252, freq="MS")
    data = np.random.default_rng(127).normal(0.005, 0.03, (252, 3))
    payloads = {s: pd.Series(data[:, i], index=index, name=s).to_csv() for i, s in enumerate(SYMBOLS)}
    payloads["USD.FX"] = pd.Series(0.0, index=index, name="USD.FX").to_csv()

    def get(session, url, **kwargs):
        symbol = url.rsplit("/", 1)[-1]
        if "/assets_namespaces/" in url:
            body = json.dumps(["US", "FX"])
        elif "/symbol/" in url:
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
            body = payloads[symbol]
        response = requests.Response()
        response.status_code, response.url, response._content = 200, url, body.encode()
        return response

    return get


def observe(mode: str, old_connect: classmethod, fake_get: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    sessions, calls = [], []
    factory = requests.session
    get = fake_get or requests.Session.get

    def new_session():
        session = factory()
        sessions.append(session)
        return session

    def observed_get(session, url, **kwargs):
        entry = {
            "path": urlsplit(url).path,
            "thread": get_ident(),
            "session": sessions.index(session),
            "params": kwargs.get("params"),
            "success": False,
        }
        begin = time.perf_counter()
        try:
            response = get(session, url, **kwargs)
            entry.update(success=response.ok, status=response.status_code)
            return response
        finally:
            entry["seconds"] = time.perf_counter() - begin
            calls.append(entry)

    namespaces.get_assets_namespaces.cache_clear()
    output = None
    receipt: dict[str, Any] = {"mode": mode, "success": False, "calls": calls}
    begin = time.perf_counter()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(requests, "session", new_session))
            stack.enter_context(patch.object(requests.Session, "get", observed_get))
            if mode == "baseline":
                stack.enter_context(patch.object(API, "connect", old_connect))
            with ok.data_request_scope() if mode == "outer_scope" else nullcontext():
                output = workflow()
        receipt["success"] = True
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    finally:
        receipt.update(seconds=time.perf_counter() - begin, sessions=len(sessions), logical_requests=len(calls))
        for session in sessions:
            session.close()
    return output, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "baseline_revision": BASE,
        "protocol": "Offline transport counts or live warm triple plus 3 rotating triples; complete fresh public workflow",
        "offline": args.offline,
        "observations": [],
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "okama/api/request_scope.py",
                "okama/api/api_methods.py",
                "okama/asset.py",
                "okama/common/make_asset_list.py",
            )
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    old = original_connect()
    fake = offline_transport() if args.offline else None
    modes = ["baseline", "automatic", "outer_scope"]
    reference = None
    try:
        for repeat in range(1 if args.offline else 4):
            for mode in modes[repeat % 3 :] + modes[: repeat % 3]:
                value, receipt = observe(mode, old, fake)
                receipt.update(repeat=repeat, warmup=not args.offline and repeat == 0)
                result["observations"].append(receipt)
                if value is not None:
                    if reference is None:
                        reference = value
                    receipt["max_absolute_difference"] = compare(value, reference)
                print({k: v for k, v in receipt.items() if k != "calls"}, flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        result["all_successful"] = all(r["success"] for r in result["observations"])
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
