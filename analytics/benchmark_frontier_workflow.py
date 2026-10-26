"""Incremental optimizer and complete CPU workflow comparisons; HTTP is excluded."""

import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import okama as ok
from analytics.benchmark_calendar_batching import Currency, returns_frame
from okama.api.data_queries import QueryData
from tests.helpers.factories import FakeAsset
from tests.helpers.optimizer_reference import ScalarOptimizer


def compare(a: Any, b: Any) -> float:
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, rtol=1e-12, atol=1e-12)
        return float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))
    if isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, rtol=1e-12, atol=1e-12)
        return float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy()))) if pd.api.types.is_numeric_dtype(a) else 0.0
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max(compare(a[k], b[k]) for k in a)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def measure(before: Callable, after: Callable, workflow: bool = False) -> dict[str, Any]:
    baseline, candidate = before(), after()
    difference = compare(candidate[0] if workflow else candidate, baseline[0] if workflow else baseline)
    result: dict[str, Any] = {"baseline_seconds": [], "candidate_seconds": [], "max_absolute_difference": difference}
    stages: dict[str, list[dict[str, float]]] = {"baseline": [], "candidate": []}
    for repeat in range(5):
        pair = [("baseline", before), ("candidate", after)]
        for label, call in pair if repeat % 2 == 0 else pair[::-1]:
            start = time.perf_counter()
            value = call()
            result[f"{label}_seconds"].append(time.perf_counter() - start)
            if workflow:
                stages[label].append(value[1])
    for label in ("baseline", "candidate"):
        result[f"{label}_median_seconds"] = float(np.median(result[f"{label}_seconds"]))
    result["ratio"] = result["baseline_median_seconds"] / result["candidate_median_seconds"]
    if workflow:
        result["stage_seconds"] = stages
        result["stage_median_seconds"] = {
            label: {k: float(np.median([r[k] for r in rows])) for k in rows[0]} for label, rows in stages.items()
        }
    return result


def asset_payloads(data: pd.DataFrame) -> dict[str, str]:
    return {name: data[name].set_axis(data.index.to_timestamp()).to_csv() for name in data}


def load_assets(payloads: dict[str, str]) -> list[FakeAsset]:
    return [FakeAsset(name, QueryData.csv_to_series(payload, "M")) for name, payload in payloads.items()]


def workflow(cls: type, payloads: dict[str, str], n_points: int) -> tuple[dict[str, Any], dict[str, float]]:
    stages: dict[str, float] = {}
    start = time.perf_counter()
    assets = load_assets(payloads)
    stages["csv_load_and_ready_assets"] = time.perf_counter() - start
    start = time.perf_counter()
    with patch("okama.asset.Asset", Currency):
        frontier = cls(assets, ccy="USD", inflation=False, n_points=n_points)
    assert len(frontier.assets_ror) == len(assets[0].ror)
    stages["frontier_setup"] = time.perf_counter() - start
    start = time.perf_counter()
    points = frontier.ef_points
    tangency = frontier.get_tangency_portfolio(rate_of_return="cagr", rf_return=0.01)
    stages["frontier_and_tangency"] = time.perf_counter() - start
    start = time.perf_counter()
    with patch("okama.asset.Asset", Currency):
        portfolio = ok.Portfolio(
            assets,
            weights=tangency["Weights"].tolist(),
            ccy="USD",
            inflation=False,
            rebalancing_strategy=ok.Rebalance(period="year"),
            symbol="workflow.PF",
        )
    stages["portfolio_setup"] = time.perf_counter() - start
    start = time.perf_counter()
    result = {
        "frontier": points,
        "tangency": tangency,
        "returns": portfolio.ror,
        "cagr": portfolio.get_cagr(),
        "annual_risk": portfolio.risk_annual,
        "rolling_cagr": portfolio.get_rolling_cagr(window=60),
        "events": portfolio.rebalancing_events,
    }
    stages["portfolio_metrics"] = time.perf_counter() - start
    return result, stages


def profile_top(call: Callable) -> list[dict[str, Any]]:
    profiler = cProfile.Profile()
    profiler.runcall(call)
    rows = []
    for (filename, line, name), (_primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": name,
                "calls": calls,
                "own_seconds": own,
                "cumulative_seconds": cumulative,
            }
        )
    return sorted(rows, key=lambda row: row["own_seconds"], reverse=True)[:15]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "baseline": "Previous completed local stage: original remaining optimizer methods; earlier improvements shared",
        "reference_revision": "b33ca12bdd81b4107c2de484fff581eda1023ce1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "protocol": "One warm pair then five alternating pairs; new objects per complete workflow; financial equivalence rtol=atol=1e-12",
        "loading_boundary": "CSV payloads already resident; actual QueryData parser plus FakeAsset construction. No HTTP, metadata lookup or real-market-data claim",
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in ("okama/frontier/single_period.py", "tests/helpers/optimizer_reference.py")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": {},
        "profiles": {},
    }
    for assets in (3, 32):
        data = returns_frame(600, assets)
        payloads = asset_payloads(data)
        with patch("okama.asset.Asset", Currency):
            old = ScalarOptimizer(load_assets(payloads), ccy="USD", inflation=False)
            new = ok.EfficientFrontierSingle(load_assets(payloads), ccy="USD", inflation=False)
        target = float(data.mean().mean())
        calls = {
            "minimize_risk": lambda obj, target=target: obj.minimize_risk(target, monthly_return=True),
            "tangency_cagr": lambda obj: obj.get_tangency_portfolio(rate_of_return="cagr"),
            "tangency_mean": lambda obj: obj.get_tangency_portfolio(rate_of_return="mean_return"),
            "optimize_return": lambda obj: obj.optimize_return(),
        }
        for name, call in calls.items():
            key = f"{name}_{assets}_assets"
            row = measure(partial(call, old), partial(call, new))
            result["cases"][key] = row
            print(key, row, flush=True)
        key = f"workflow_600x{assets}_20_points"
        old_call = partial(workflow, ScalarOptimizer, payloads, 20)
        new_call = partial(workflow, ok.EfficientFrontierSingle, payloads, 20)
        row = measure(old_call, new_call, workflow=True)
        result["cases"][key] = row
        result["profiles"][key] = profile_top(new_call)
        print(key, row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
