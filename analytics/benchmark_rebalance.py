"""Actual conditional Rebalance methods versus frozen Git source, with full outputs."""

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
from functools import partial
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import okama as ok


def frozen_class(revision: str) -> type:
    source = subprocess.check_output(["git", "show", f"{revision}:okama/common/helpers/rebalancing.py"], text=True)
    module = types.ModuleType("benchmark_rebalance_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:rebalancing.py", "exec"), module.__dict__)
    return module.Rebalance


def compare_results(candidate: Any, baseline: Any) -> float:
    if isinstance(candidate, pd.Series):
        pd.testing.assert_series_equal(candidate, baseline, rtol=1e-12, atol=1e-12)
        return float(np.max(np.abs(candidate.to_numpy() - baseline.to_numpy())))
    if isinstance(candidate, pd.DataFrame):
        pd.testing.assert_frame_equal(candidate, baseline, rtol=1e-12, atol=1e-12)
        return float(np.max(np.abs(candidate.to_numpy() - baseline.to_numpy())))
    pd.testing.assert_series_equal(
        candidate.portfolio_wealth_index, baseline.portfolio_wealth_index, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(
        candidate.assets_wealth_indexes, baseline.assets_wealth_indexes, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_series_equal(candidate.events, baseline.events)
    return float(
        np.max(np.abs(candidate.portfolio_wealth_index.to_numpy() - baseline.portfolio_wealth_index.to_numpy()))
    )


def measure_pair(baseline: Callable, candidate: Callable) -> dict[str, Any]:
    error = compare_results(candidate(), baseline())
    result: dict[str, Any] = {"baseline_seconds": [], "candidate_seconds": [], "max_wealth_difference": error}
    for repeat in range(5):
        order = (("baseline_seconds", baseline), ("candidate_seconds", candidate))
        for label, call in order if repeat % 2 == 0 else order[::-1]:
            start = time.perf_counter()
            call()
            result[label].append(time.perf_counter() - start)
    result["baseline_median_seconds"] = float(np.median(result["baseline_seconds"]))
    result["candidate_median_seconds"] = float(np.median(result["candidate_seconds"]))
    result["ratio"] = result["baseline_median_seconds"] / result["candidate_median_seconds"]
    return result


def peak_bytes(call: Callable) -> int:
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


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
        "candidate_source_sha256": hashlib.sha256(Path("okama/common/helpers/rebalancing.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": "Offline, seed127, 1 warm pair then 5 alternating timed pairs; actual complete public methods",
        "cases": {},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
        for months, assets in ((252, 3), (600, 16), (1200, 32)):
            rng = np.random.default_rng(127)
            ror = pd.DataFrame(
                rng.normal(0.004, 0.04, (months, assets)),
                index=pd.period_range("1900-03", periods=months, freq="M", name="Date"),
                columns=[f"A{i}" for i in range(assets)],
            )
            ror.columns.name = "Symbols"
            weights = np.full(assets, 1 / assets)
            modes = (("abs", 0.05, None), ("rel", None, 0.1), ("both", 0.05, 0.1))
            for mode, absolute, relative in modes:
                before = baseline_cls(period="none", abs_deviation=absolute, rel_deviation=relative)
                after = ok.Rebalance(period="none", abs_deviation=absolute, rel_deviation=relative)
                for with_assets in (False, True):
                    baseline = partial(before.wealth_ts, weights, ror, with_assets)
                    candidate = partial(after.wealth_ts, weights, ror, with_assets)
                    key = f"wealth_{mode}_{months}x{assets}_assets{with_assets}"
                    result["cases"][key] = measure_pair(baseline, candidate)
                    if (months, assets, mode, with_assets) == (600, 16, "abs", True):
                        result["cases"][key]["baseline_peak_bytes"] = peak_bytes(baseline)
                        result["cases"][key]["candidate_peak_bytes"] = peak_bytes(candidate)
                    print(key, result["cases"][key], flush=True)
                if months == 600 and mode == "both":
                    for method in ("return_ror_ts", "assets_weights_ts"):
                        key = f"{method}_{months}x{assets}"
                        result["cases"][key] = measure_pair(
                            lambda before=before, ror=ror, weights=weights, method=method: getattr(before, method)(
                                weights, ror
                            ),
                            lambda after=after, ror=ror, weights=weights, method=method: getattr(after, method)(
                                weights, ror
                            ),
                        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
