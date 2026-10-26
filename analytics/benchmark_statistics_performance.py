"""Paired actual-method GMV and rolling CAGR timings against frozen source."""

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import textwrap
import time
import types
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import okama as ok
from analytics.benchmark_calendar_batching import Currency, returns_frame
from okama.common.helpers import helpers
from tests.helpers.factories import FakeAsset


BASELINE = "b33ca12bdd81b4107c2de484fff581eda1023ce1"


def original_class() -> type:
    source = subprocess.check_output(["git", "show", f"{BASELINE}:okama/frontier/single_period.py"], text=True)
    module = types.ModuleType("original_single_frontier_statistics")
    exec(compile(source, "original_single_frontier_statistics.py", "exec"), module.__dict__)
    return module.EfficientFrontierSingle


def original_method(path: str, cls_name: str, method_name: str) -> Any:
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    code = textwrap.dedent("\n".join(source.splitlines()[method.lineno - 1 : method.end_lineno]))
    namespace = {"pd": pd, "np": np, "helpers": helpers}
    exec(compile(code, f"original_{method_name}.py", "exec"), namespace)
    return namespace[method_name]


def compare(a: Any, b: Any) -> float:
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, rtol=1e-12, atol=1e-12)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def measure(before: Any, after: Any) -> dict[str, Any]:
    difference = compare(after(), before())
    timings = {"baseline_seconds": [], "candidate_seconds": []}
    for n in range(5):
        pair = [("baseline_seconds", before), ("candidate_seconds", after)]
        for label, call in pair if n % 2 == 0 else pair[::-1]:
            start = time.perf_counter()
            value = call()
            timings[label].append(time.perf_counter() - start)
            assert value is not None
    baseline, candidate = (float(np.median(v)) for v in timings.values())
    return {
        **timings,
        "baseline_median_seconds": baseline,
        "candidate_median_seconds": candidate,
        "ratio": baseline / candidate,
        "max_absolute_difference": difference,
    }


def make_object(cls: type, data: pd.DataFrame) -> Any:
    assets = [FakeAsset(c, data[c]) for c in data]
    with patch("okama.asset.Asset", Currency):
        obj = cls(assets, ccy="USD", inflation=False, first_date=str(data.index[0]), last_date=str(data.index[-1]))
    assert obj.assets_ror.shape == data.shape
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_cls = original_class()
    result: dict[str, Any] = {
        "baseline_revision": BASELINE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "okama/frontier/single_period.py",
                "okama/common/helpers/helpers.py",
                "okama/asset_list.py",
                "okama/portfolios/core.py",
            )
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": "Offline actual public calls; one warm pair, five alternating pairs; rtol=atol=1e-12",
        "cases": {},
    }
    for assets in (3, 32):
        data = returns_frame(252, assets)
        before, after = make_object(old_cls, data), make_object(ok.EfficientFrontierSingle, data)
        for prop in ("gmv_monthly_weights", "gmv_annual_weights", "gmv_monthly", "gmv_annualized"):
            key = f"{prop}_{assets}_assets"
            row = measure(partial(getattr, before, prop), partial(getattr, after, prop))
            result["cases"][key] = row
            print(key, row, flush=True)
    for months, assets, window in ((252, 3, 12), (600, 16, 60), (6000, 16, 120)):
        data = returns_frame(months, assets)
        obj = make_object(ok.AssetList, data)
        old_method = original_method("okama/asset_list.py", "AssetList", "get_rolling_cagr")
        key = f"assetlist_rolling_cagr_{months}x{assets}_window{window}"
        row = measure(partial(old_method, obj, window), partial(obj.get_rolling_cagr, window))
        result["cases"][key] = row
        print(key, row, flush=True)
    data = returns_frame(600, 16)
    portfolio = make_object(ok.Portfolio, data)
    old_method = original_method("okama/portfolios/core.py", "Portfolio", "get_rolling_cagr")
    row = measure(lambda: old_method(portfolio, 60), lambda: portfolio.get_rolling_cagr(60))
    result["cases"]["portfolio_rolling_cagr_600x16_window60"] = row
    print("portfolio_rolling_cagr_600x16_window60", row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
