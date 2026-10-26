"""Offline issue-127 benchmark. Run from the repo root with python -m analytics.benchmark_frontier_batching."""

import argparse
import cProfile
import io
import hashlib
import json
import os
import platform
import pstats
import subprocess
import time
import tracemalloc
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

import okama as ok
from okama.common.helpers import helpers
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset


def make_frontier(multi: bool = False, months: int = 252, cls: Any = None) -> Any:
    """Initialize real frontiers with fixed offline assets; replace only currency lookup."""
    rng = np.random.default_rng(127)
    index = pd.period_range("2000-03", periods=months, freq="M")
    returns = rng.normal(0.005, 0.04, (months, 3))
    assets = [FakeAsset(f"A{i}.US", pd.Series(returns[:, i], index=index)) for i in range(3)]
    cls = cls or (ok.EfficientFrontier if multi else ok.EfficientFrontierSingle)
    with patch("okama.common.make_asset_list.asset.Asset", FakeCurrencyAsset):
        return cls(assets, ccy="USD", inflation=False)


def measure(call: Any, repeats: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Warm once, time repetitions including generation/preparation/output, then trace allocations separately."""
    call()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        timings.append(time.perf_counter() - start)
    tracemalloc.start()
    result = call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "seconds": timings,
        "median_seconds": float(np.median(timings)),
        "peak_bytes": peak,
        "portfolios": len(result),
    }, result


def paired_measure(candidate: Any, revision: str, repeats: int) -> dict[str, Any]:
    """Alternate real baseline/candidate calls on the same inputs, without timing assertions."""
    source = subprocess.check_output(["git", "show", f"{revision}:okama/frontier/single_period.py"], text=True)
    module = types.ModuleType("frontier_benchmark_baseline")
    exec(compile(source, f"{revision}:single_period.py", "exec"), module.__dict__)
    baseline = make_frontier(cls=module.EfficientFrontierSingle)
    cases = {}
    for n in (1, 8, 128, 2048, 8192):
        timings: dict[str, list[float]] = {"baseline": [], "candidate": []}
        expected = None
        for repeat in range(repeats + 1):
            # Alternate order too; the first pair is warmup only.
            order = (("baseline", baseline), ("candidate", candidate))
            if repeat % 2:
                order = order[::-1]
            for label, ef in order:
                np.random.seed(127)
                start = time.perf_counter()
                result = ef.get_monte_carlo(n=n)
                elapsed = time.perf_counter() - start
                if repeat:
                    timings[label].append(elapsed)
                if label == "baseline":
                    expected = result
                elif expected is not None:
                    pd.testing.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)
        cases[str(n)] = timings
        print(f"paired {n}: {timings}", flush=True)
    return {"baseline_revision": revision, "method": "get_monte_carlo(kind=mean)", "cases": cases}


def measure_phases(frontier: Any, repeats: int) -> dict[str, list[float]]:
    # Separate generation, statistics preparation, calculation and output construction.
    weights = np.stack(helpers.Float.get_random_weights(8192, 3, frontier.bounds))
    cov = frontier.assets_ror.cov().to_numpy()
    means = frontier.assets_ror.mean().to_numpy()
    risk = helpers.Float.annualize_risk(np.sqrt(np.sum((weights @ cov) * weights, axis=1)), weights @ means)
    annual_return = helpers.Float.annualize_return(weights @ means)
    phases = {
        "weight_generation": lambda: helpers.Float.get_random_weights(8192, 3, frontier.bounds),
        "statistics_preparation": lambda: (frontier.assets_ror.cov(), frontier.assets_ror.mean()),
        "batch_calculation": lambda: (
            helpers.Float.annualize_risk(np.sqrt(np.sum((weights @ cov) * weights, axis=1)), weights @ means),
            helpers.Float.annualize_return(weights @ means),
        ),
        "output_construction": lambda: pd.DataFrame(weights, columns=frontier.symbols).assign(
            Risk=risk, Return=annual_return
        ),
    }
    result = {}
    for name, call in phases.items():
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            call()
            times.append(time.perf_counter() - start)
        result[name] = times
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--paired-base", help="Compare against this Git revision, alternating warm actual calls")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frontier = make_frontier()
    config = io.StringIO()
    with patch("sys.stdout", config):
        np.show_config()
    report: dict[str, Any] = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--", "okama/"])).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "okama": ok.__version__,
        "blas_config": config.getvalue(),
        "thread_env": {
            key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "months": 252,
        "assets": 3,
        "warmups": 1,
        "repeats": args.repeats,
        "memory": "tracemalloc peak allocated during an extra warm call, including returned DataFrame; excludes resident inputs/imports",
        "cases": {},
    }
    if args.paired_base:
        report["paired"] = paired_measure(frontier, args.paired_base, args.repeats)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    arrays = {}
    for kind in ("mean", "cagr"):
        for n, divisions in ((128, 15), (2048, 63), (8192, 100)):

            def monte_carlo(n: int = n, kind: str = kind) -> pd.DataFrame:
                np.random.seed(127)
                return frontier.get_monte_carlo(n=n, kind=kind)

            def grid(divisions: int = divisions, kind: str = kind) -> pd.DataFrame:
                return frontier.get_grid_portfolios(step=1 / divisions, kind=kind)

            for method, call in (("monte_carlo", monte_carlo), ("grid", grid)):
                key = f"single_{kind}_{method}_{n}"
                report["cases"][key], result = measure(call, args.repeats)
                arrays[key] = result.to_numpy()
                print(f"{key}: {report['cases'][key]}", flush=True)
    # Profile both actual classes, preserving the slower class's annual rebalancing.
    for multi in (False, True):
        ef = make_frontier(multi=multi)
        for method in ("monte_carlo", "grid"):
            call = (
                (lambda ef=ef: ef.get_monte_carlo(n=128))
                if method == "monte_carlo"
                else (lambda ef=ef: ef.get_grid_portfolios(step=0.1))
            )
            profile = cProfile.Profile()
            profile.runcall(call)
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(25)
            report[f"profile_{'multi' if multi else 'single'}_{method}"] = stream.getvalue()
    report["phases_seconds"] = measure_phases(frontier, args.repeats)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
