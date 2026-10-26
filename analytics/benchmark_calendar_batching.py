"""Frozen-source comparisons of actual calendar and multi-period frontier methods."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import tracemalloc
import types
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import okama as ok
from analytics.benchmark_rebalance import compare_results, frozen_class
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset


class Currency(FakeCurrencyAsset):
    def __init__(self, symbol: str, first_date: str | None = None, last_date: str | None = None) -> None:
        super().__init__(symbol, first_date or "1900-01", last_date or "2100-01")


def returns_frame(months: int = 252, assets: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(127)
    return pd.DataFrame(
        rng.normal(0.004, 0.04, (months, assets)),
        index=pd.period_range("1970-03", periods=months, freq="M", name="Date"),
        columns=[f"A{i}.US" for i in range(assets)],
    )


def measure_pair(before: Callable, after: Callable) -> dict[str, Any]:
    baseline, candidate = before(), after()
    difference = compare_results(candidate, baseline)
    result: dict[str, Any] = {"baseline_seconds": [], "candidate_seconds": [], "max_absolute_difference": difference}
    for repeat in range(5):
        order = (("baseline_seconds", before), ("candidate_seconds", after))
        for label, call in order if repeat % 2 == 0 else order[::-1]:
            start = time.perf_counter()
            call()
            result[label].append(time.perf_counter() - start)
    result["baseline_median_seconds"] = float(np.median(result["baseline_seconds"]))
    result["candidate_median_seconds"] = float(np.median(result["candidate_seconds"]))
    result["ratio"] = result["baseline_median_seconds"] / result["candidate_median_seconds"]
    return result


def allocation_peak(call: Callable) -> int:
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def baseline_frontier(revision: str, rebalance_class: type) -> type:
    source = subprocess.check_output(["git", "show", f"{revision}:okama/frontier/multi_period.py"], text=True)
    module = types.ModuleType("benchmark_original_multi_frontier")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:multi_period.py", "exec"), module.__dict__)
    module.Rebalance = rebalance_class
    return module.EfficientFrontier


def calendar_cases(result: dict[str, Any], baseline_class: type) -> None:
    ror = returns_frame(600, 16)
    weights = np.full(16, 1 / 16)
    for period in ("month", "quarter", "half-year", "year"):
        for condition in (None, 0.05):
            before = baseline_class(period=period, abs_deviation=condition)
            after = ok.Rebalance(period=period, abs_deviation=condition)
            for assets in (False, True):
                key = f"calendar_{period}_abs{condition}_assets{assets}"
                old_call, new_call = (
                    partial(before.wealth_ts, weights, ror, assets),
                    partial(after.wealth_ts, weights, ror, assets),
                )
                row = measure_pair(old_call, new_call)
                if period == "month" and condition is None and assets:
                    row["baseline_peak_bytes"] = allocation_peak(old_call)
                    row["candidate_peak_bytes"] = allocation_peak(new_call)
                result[key] = row
                print(key, row, flush=True)


def make_frontiers(cls: type, baseline_rebalance: type, assets: int = 3) -> tuple[Any, Any]:
    ror = returns_frame(252, assets)
    objects = [FakeAsset(name, ror[name]) for name in ror]
    with patch("okama.common.make_asset_list.asset.Asset", Currency):
        before = cls(objects, ccy="USD", inflation=False, rebalancing_strategy=baseline_rebalance(period="year"))
        after = ok.EfficientFrontier(
            objects, ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="year")
        )
    assert len(before.assets_ror) == len(after.assets_ror) == 252
    return before, after


def monte_carlo(frontier: Any, n: int) -> pd.DataFrame:
    np.random.seed(127)
    return frontier.get_monte_carlo(n=n)


def frontier_cases(result: dict[str, Any], cls: type, baseline_rebalance: type) -> None:
    before, after = make_frontiers(cls, baseline_rebalance)
    for n in (1, 8, 128, 2048, 8192):
        old_call, new_call = partial(monte_carlo, before, n), partial(monte_carlo, after, n)
        key = f"frontier_year_mc_{n}"
        row = measure_pair(old_call, new_call)
        if n == 2048:
            row["baseline_peak_bytes"] = allocation_peak(old_call)
            row["candidate_peak_bytes"] = allocation_peak(new_call)
        if n == 8192:
            row["candidate_peak_bytes"] = allocation_peak(new_call)
        result[key] = row
        print(key, row, flush=True)
    for divisions in (15, 63, 100):
        key = f"frontier_year_grid_step1over{divisions}"
        row = measure_pair(
            partial(before.get_grid_portfolios, step=1 / divisions),
            partial(after.get_grid_portfolios, step=1 / divisions),
        )
        row["portfolios"] = len(after.get_grid_portfolios(step=1 / divisions))
        result[key] = row
        print(key, row, flush=True)
    for period in ("none", "month", "quarter", "half-year"):
        before.rebalancing_strategy = baseline_rebalance(period=period)
        after.rebalancing_strategy = ok.Rebalance(period=period)
        key = f"frontier_{period}_mc_128"
        result[key] = measure_pair(partial(monte_carlo, before, 128), partial(monte_carlo, after, 128))
        print(key, result[key], flush=True)
    before, after = make_frontiers(cls, baseline_rebalance, assets=32)
    result["frontier_year_mc_2048_32assets"] = measure_pair(
        partial(monte_carlo, before, 2048), partial(monte_carlo, after, 2048)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="b33ca12bdd81b4107c2de484fff581eda1023ce1")
    args = parser.parse_args()
    old_rebalance = frozen_class(args.baseline)
    old_frontier = baseline_frontier(args.baseline, old_rebalance)
    result: dict[str, Any] = {
        "baseline_revision": args.baseline,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("okama/common/helpers/rebalancing.py", "okama/frontier/multi_period.py")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": "1 warm pair, 5 alternating timed pairs; offline inputs and actual complete public methods",
        "memory": "Separate warm tracemalloc calls; allocated output included, resident imports/inputs excluded; not process RSS",
        "cases": {},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
        calendar_cases(result["cases"], old_rebalance)
        frontier_cases(result["cases"], old_frontier, old_rebalance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
