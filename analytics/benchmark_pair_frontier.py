"""Measure complete pair plots against frozen Git source with full solver-output parity."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import types
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cloudpickle

import okama as ok
from analytics.benchmark_calendar_batching import Currency, returns_frame
from tests.helpers.factories import FakeAsset

BASELINE = "48a0e7c8eb139a8b8b3b0eeb6e6a1dbbcbc8e3aa"
CASES = [(4, 120, 12, 1), (8, 120, 12, 1), (12, 120, 12, 1), (4, 600, 12, 1), (4, 120, 80, 1), (4, 120, 12, 4)]


def load_original() -> type:
    source = subprocess.check_output(["git", "show", f"{BASELINE}:okama/frontier/multi_period.py"], text=True)
    module = types.ModuleType("pair_frontier_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, "baseline/multi_period.py", "exec"), module.__dict__)
    cloudpickle.register_pickle_by_value(module)
    return module.EfficientFrontier


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported result type: {type(value)}")


def tree_rss() -> int:
    """Sum Linux resident bytes for this process and its own worker descendants."""
    todo = [os.getpid()]
    total = 0
    while todo:
        pid = todo.pop()
        try:
            total += int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            todo.extend(int(v) for v in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return total


def measure_side(side: str, case: tuple[int, ...], repeats: int) -> dict:
    assets_count, months, points, jobs = case
    cls = load_original() if side == "baseline" else ok.EfficientFrontier
    data = returns_frame(months, assets_count)
    assets = [FakeAsset(c, data[c]) for c in data]

    def construct() -> object:
        return cls(assets, ccy="USD", inflation=False, n_points=points, rebalancing_strategy=ok.Rebalance())

    def plot(frontier: object) -> list:
        ax = frontier.plot_pair_ef()
        lines = [line.get_xydata().tolist() for line in ax.lines]
        plt.close(ax.figure)
        return lines

    captured = []
    original_calculate = cls._get_ef_points

    def capture(frontier: object) -> None:
        original_calculate(frontier)
        captured.append(frontier._ef_points.to_dict(orient="list"))

    with patch("okama.asset.Asset", Currency), patch("okama.settings.resolve_n_jobs", return_value=jobs):
        # Untimed full-output validation also warms imports and worker processes.
        with patch.object(cls, "_get_ef_points", capture):
            lines = plot(construct())
        samples = []
        warm_samples = []
        for _ in range(repeats):
            frontier = construct()  # Parent construction is outside timing on both sides.
            start = time.perf_counter()
            plot(frontier)
            samples.append(time.perf_counter() - start)
            if side == "current":
                start = time.perf_counter()
                warm_lines = plot(frontier)
                warm_samples.append(time.perf_counter() - start)
                compare(warm_lines, lines)
        # Measure process-tree RSS in a separate complete call, excluding sampler overhead from timings.
        peak = [tree_rss()]
        stopped = threading.Event()

        def sample() -> None:
            while not stopped.wait(0.01):
                peak[0] = max(peak[0], tree_rss())

        sampler = threading.Thread(target=sample)
        sampler.start()
        try:
            plot(construct())
        finally:
            stopped.set()
            sampler.join()
        return {
            "samples_seconds": samples,
            "warm_samples_seconds": warm_samples,
            "main_process_lifetime_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "sampled_process_tree_peak_rss_mib": peak[0] / 1024**2,
            "tables": captured,
            "lines": lines,
        }


def compare(left: object, right: object) -> float:
    """Compare every numerical/diagnostic field, including all optimizer weights."""
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return max((compare(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, list):
        assert len(left) == len(right)
        return max((compare(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float)):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
        return abs(left - right) if np.isfinite(left) and np.isfinite(right) else 0.0
    assert left == right
    return 0.0


def run(args: argparse.Namespace) -> None:
    if args.side:
        result = measure_side(args.side, tuple(map(int, args.case.split(","))), args.repeats)
        args.output.write_text(json.dumps(result, default=json_default) + "\n")
        return
    selected = [tuple(map(int, args.case.split(",")))] if args.case else CASES
    args.scratch.mkdir(parents=True, exist_ok=True)
    manifest_path = args.scratch / "baseline-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if args.reuse_baseline else None
    if manifest is not None:
        assert manifest["baseline"] == BASELINE
    result = {
        "baseline": BASELINE,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in [
                "okama/frontier/multi_period.py",
                "okama/common/helpers/rebalancing.py",
                "analytics/benchmark_pair_frontier.py",
            ]
        },
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
        "protocol": "Offline seed127; parent excluded; complete child construction/optimization/Agg plot included. "
        f"One untimed validation call then independent processes per side; {args.repeats} timed cold calls. "
        "Baseline recreates pairs on every call; current warm samples repeat each cold parent. "
        "RSS sampler runs in a separate call; summed RSS includes shared pages more than once, not PSS.",
        "repeats": args.repeats,
        "baseline_reused": args.reuse_baseline,
        "cases": [],
    }
    for index, case in enumerate(selected):
        sides = {}
        for side in ["baseline", "current"] if index % 2 == 0 else ["current", "baseline"]:
            output = args.scratch / f"{','.join(map(str, case))}-{side}.json"
            command = [
                sys.executable,
                "-m",
                "analytics.benchmark_pair_frontier",
                "--side",
                side,
                "--case",
                ",".join(map(str, case)),
                "--repeats",
                str(args.repeats),
                "--output",
                str(output),
            ]
            if side == "baseline" and manifest is not None:
                assert hashlib.sha256(output.read_bytes()).hexdigest() == manifest["files"][output.name]
            else:
                subprocess.run(command, check=True, timeout=900)
            sides[side] = json.loads(output.read_text())
        difference = compare(sides["baseline"]["tables"], sides["current"]["tables"])
        compare(sides["baseline"]["lines"], sides["current"]["lines"])
        for values in sides.values():
            values["full_output_sha256"] = hashlib.sha256(
                json.dumps(values.pop("tables"), sort_keys=True).encode()
            ).hexdigest()
            values.pop("lines")
            for kind in ["samples", "warm_samples"]:
                samples = values[f"{kind}_seconds"]
                if samples:
                    values[f"{kind}_median_seconds"] = float(np.median(samples))
                    values[f"{kind}_p95_seconds"] = float(np.percentile(samples, 95))
        row = {
            "assets": case[0],
            "months": case[1],
            "points": case[2],
            "jobs": case[3],
            "pairs": case[0] * (case[0] - 1) // 2,
            "max_absolute_difference": difference,
            **sides,
        }
        row["cold_speedup"] = sides["baseline"]["samples_median_seconds"] / sides["current"]["samples_median_seconds"]
        row["repeat_speedup"] = (
            sides["baseline"]["samples_median_seconds"] / sides["current"]["warm_samples_median_seconds"]
        )
        result["cases"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"{case}: cold {row['cold_speedup']:.2f}x, repeat {row['repeat_speedup']:.2f}x, full difference {difference:.3g}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["baseline", "current"])
    parser.add_argument("--case")
    parser.add_argument("--reuse-baseline", action="store_true", help="Reuse this run's preserved baseline raw outputs")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("tmp/pair-benchmark"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
