"""Compare actual ListMaker paths with frozen Git source on offline data.

Run from the repo root: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LOKY_MAX_CPU_COUNT=4 poetry run python -m analytics.benchmark_listmaker
--output tmp/listmaker.json --baseline b33ca12bdd81b4107c2de484fff581eda1023ce1
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import types
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import okama as ok
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset


class Currency(FakeCurrencyAsset):
    """Keep all synthetic input months without contacting the API."""

    def __init__(self, symbol: str, first_date: str | None = None, last_date: str | None = None) -> None:
        super().__init__(symbol, first_date or "1900-01", last_date or "2100-01")


def make_assets(months: int, count: int) -> list[FakeAsset]:
    rng = np.random.default_rng(127)
    index = pd.period_range("1970-03", periods=months, freq="M")
    values = rng.normal(0.004, 0.04, (months, count))
    return [FakeAsset(f"A{i}.US", pd.Series(values[:, i], index=index)) for i in range(count)]


def frozen_class(revision: str) -> type:
    source = subprocess.check_output(["git", "show", f"{revision}:okama/common/make_asset_list.py"], text=True)
    module = types.ModuleType("baseline_listmaker")
    exec(compile(source, f"{revision}:make_asset_list.py", "exec"), module.__dict__)

    class BaselineList(module.ListMaker):
        """Expose the original inherited constructor and dividend calculation."""

        def __repr__(self) -> str:
            return "Frozen baseline ListMaker"

        @property
        def dividend_yield(self) -> pd.DataFrame:
            return self._assets_dividend_yield

    return BaselineList


def compare(baseline: Callable, candidate: Callable, repeats: int = 5) -> dict[str, Any]:
    expected, actual = baseline(), candidate()
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
    result: dict[str, Any] = {
        "baseline_seconds": [],
        "candidate_seconds": [],
        "max_absolute_difference": float(np.max(np.abs(actual.to_numpy() - expected.to_numpy()))),
    }
    for repeat in range(repeats):
        order = (("baseline_seconds", baseline), ("candidate_seconds", candidate))
        for label, call in order if repeat % 2 == 0 else order[::-1]:
            start = time.perf_counter()
            call()
            result[label].append(time.perf_counter() - start)
    result["baseline_median_seconds"] = float(np.median(result["baseline_seconds"]))
    result["candidate_median_seconds"] = float(np.median(result["candidate_seconds"]))
    result["ratio"] = result["baseline_median_seconds"] / result["candidate_median_seconds"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="b33ca12bdd81b4107c2de484fff581eda1023ce1")
    args = parser.parse_args()
    baseline_cls = frozen_class(args.baseline)
    result: dict[str, Any] = {
        "baseline_revision": args.baseline,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "candidate_source_sha256": hashlib.sha256(Path("okama/common/make_asset_list.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": "1 warmup, 5 alternating pairs, offline float64 inputs; constructor setup excluded only for dividend cases",
        "cases": {},
    }
    with patch("okama.common.make_asset_list.asset.Asset", Currency), warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
        for count in (3, 32, 128):
            objects = make_assets(252, count)
            result["cases"][f"ready_constructor_{count}"] = compare(
                lambda objects=objects: baseline_cls(objects, ccy="USD", inflation=False).assets_ror,
                lambda objects=objects: ok.AssetList(objects, ccy="USD", inflation=False).assets_ror,
            )
        for months, count in ((252, 3), (600, 16)):
            objects = make_assets(months, count)
            baseline = baseline_cls(objects, ccy="USD", inflation=False)
            candidate = ok.AssetList(objects, ccy="USD", inflation=False)
            assert len(baseline.assets_ror) == len(candidate.assets_ror) == months

            def cold(obj: Any) -> pd.DataFrame:
                obj._dividend_yield = pd.DataFrame()
                return obj.dividend_yield

            key = f"cold_yield_{months}x{count}"
            result["cases"][key] = compare(
                lambda baseline=baseline: cold(baseline), lambda candidate=candidate: cold(candidate)
            )
            start = time.perf_counter()
            for _ in range(100):
                _ = candidate.dividend_yield
            result["cases"][key]["cached_seconds_per_call"] = (time.perf_counter() - start) / 100
            print(key, result["cases"][key], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
