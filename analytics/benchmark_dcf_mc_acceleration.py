"""Compare complete public DCF/MC calls with the unmodified pre-change modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
import tracemalloc
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics.benchmark_dcf_mc_opportunities import compare, make_dcf, clear_hist, clear_mc

BASELINE = "bcd877421262d107b7b35a9a771498802dc39c0b"
SOURCES = ("okama/portfolios/dcf_calculations.py", "okama/portfolios/dcf.py", "okama/portfolios/mc.py")


def baseline_modules() -> types.ModuleType:
    """Execute exact baseline source, with its own MC and numerical module bindings."""
    modules = {}
    for name in ("dcf_calculations", "mc", "dcf"):
        module = types.ModuleType(f"_dcf_perf_baseline_{name}")
        sys.modules[module.__name__] = module
        source = subprocess.check_output(["git", "show", f"{BASELINE}:okama/portfolios/{name}.py"], text=True)
        exec(compile(source, f"baseline/{name}.py", "exec"), module.__dict__)
        modules[name] = module
    modules["dcf"].mc = modules["mc"]
    modules["dcf"].dcf_calculations = modules["dcf_calculations"]
    return modules["dcf"]


def make_pair(baseline: types.ModuleType, **kwargs: Any) -> tuple[Any, Any]:
    old, new = make_dcf(**kwargs), make_dcf(**kwargs)
    portfolio, strategy = old.parent, old.cashflow_parameters
    old = baseline.PortfolioDCF(portfolio, discount_rate=0.02)
    portfolio.dcf = old
    old.cashflow_parameters = strategy
    old.set_mc_parameters(distribution="norm", period=30, mc_number=kwargs.get("paths", 128), seed=127)
    new.discount_rate = 0.02
    return old, new


def measure(before: Callable[[], Any], after: Callable[[], Any], repeats: int) -> dict[str, Any]:
    difference = compare(before(), after())
    samples: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for n in range(repeats):
        calls = [("baseline", before), ("candidate", after)]
        for name, call in calls if n % 2 == 0 else calls[::-1]:
            start = time.perf_counter()
            call()
            samples[name].append(time.perf_counter() - start)
    medians = {name: float(np.median(values)) for name, values in samples.items()}
    return {
        "samples_seconds": samples,
        "median_seconds": medians,
        "speedup": medians["baseline"] / medians["candidate"],
        "max_absolute_difference": difference,
    }


def historical(dcf: Any, chain: bool = False) -> Any:
    clear_hist(dcf)
    if chain:
        return dcf.wealth_index("fv"), dcf.cash_flow_ts("fv"), dcf.wealth_index("fv"), dcf.cash_flow_ts("fv")
    return dcf.wealth_index("fv")


def mc_pair(dcf: Any, cold: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cold:
        dcf.mc._clear_cf_cache()
    else:
        clear_mc(dcf)
    return dcf.monte_carlo_wealth(include_negative_values=False), dcf.monte_carlo_cash_flow("fv")


def cagr_bundle(mc: Any, cold: bool) -> tuple[Any, ...]:
    if cold:
        mc._clear_cf_cache()
    return (
        mc.percentile_distribution_cagr([5, 10, 25, 50, 75, 90, 95]),
        mc.percentile_inverse_cagr(0),
        mc.percentile_inverse_cagr(0.03),
        mc.percentile_inverse_cagr(0.06),
    )


def qq(mc: Any) -> tuple[Any, ...]:
    ax = mc.plot_qq(bootstrap_size_var=2000)
    result = tuple((line.get_xdata(), line.get_ydata()) for line in ax.lines)
    bands = np.array([(span.get_y(), span.get_height()) for span in ax.patches])
    plt.close(ax.figure)
    return result, bands


def solve(dcf: Any) -> dict[str, Any]:
    result = dcf.find_the_largest_withdrawals_size(
        goal="maintain_balance_pv",
        withdrawals_range=(0, 0.1),
        percentile=50,
        threshold=0.2,
        tolerance_rel=0.02,
        iter_max=10,
    )
    return {
        "success": result.success,
        "withdrawal_abs": result.withdrawal_abs,
        "withdrawal_rel": result.withdrawal_rel,
        "error_rel": result.error_rel,
        "solutions": result.solutions.astype(float),
    }


def cold_irr(dcf: Any) -> pd.Series:
    dcf.mc._clear_cf_cache()
    return dcf.monte_carlo_irr()


def allocations(dcf: Any) -> dict[str, int]:
    dcf.mc._clear_cf_cache()
    tracemalloc.start()
    mc_pair(dcf, cold=True)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"retained_bytes": retained, "peak_bytes": peak}


def run(repeats: int) -> dict[str, Any]:
    baseline = baseline_modules()
    results: dict[str, Any] = {}

    def record(name: str, old: Any, new: Any, call: Callable[[Any], Any]) -> None:
        results[name] = measure(lambda: call(old), lambda: call(new), repeats)
        print(f"{name}: {results[name]['speedup']:.2f}x", flush=True)

    for kind, frequency in (
        ("index", "month"),
        ("index", "year"),
        ("percentage", "quarter"),
        ("cwd", "month"),
        ("vds", "year"),
    ):
        old, new = make_pair(baseline, kind=kind, frequency=frequency)
        record(f"history_600_{kind}_{frequency}", old, new, historical)
    old, new = make_pair(baseline)
    record("history_chain_600", old, new, lambda x: historical(x, chain=True))
    old, new = make_pair(baseline, assets=16)
    record("history_with_16_assets_600", old, new, lambda x: x.wealth_index_fv_with_assets)
    for paths in (128, 2048, 8192):
        old, new = make_pair(baseline, paths=paths)
        record(f"mc_pair_360x{paths}_warm_returns", old, new, lambda x: mc_pair(x, cold=False))
        record(f"mc_pair_360x{paths}_cold", old, new, lambda x: mc_pair(x, cold=True))
        record(f"survival_360x{paths}_cached_wealth", old, new, lambda x: x.monte_carlo_survival_period())
        results[f"memory_mc_pair_360x{paths}"] = {"baseline": allocations(old), "candidate": allocations(new)}
        record(f"cagr_bundle_360x{paths}_warm_returns", old.mc, new.mc, lambda x: cagr_bundle(x, cold=False))
        if paths == 8192:
            record(f"cagr_bundle_360x{paths}_cold", old.mc, new.mc, lambda x: cagr_bundle(x, cold=True))
            old.cashflow_parameters.frequency = new.cashflow_parameters.frequency = "none"
            record("irr_no_intermediate_cashflows_360x8192_cold", old, new, cold_irr)
    old, new = make_pair(baseline, paths=2048)
    record("withdrawal_solver_360x2048", old, new, solve)
    old, new = make_pair(baseline)
    old.mc.distribution = new.mc.distribution = "t"
    record("backtesting_error_student_t", old.mc, new.mc, lambda x: x.backtesting_error())
    old.mc.distribution = new.mc.distribution = "norm"
    record("qq_full_plot_600_bootstrap2000", old.mc, new.mc, qq)
    return {
        "baseline_commit": BASELINE,
        "candidate_source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES},
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "repeats": repeats,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.disable(logging.INFO)
    args.output.write_text(json.dumps(run(args.repeats), indent=2) + "\n")


if __name__ == "__main__":
    main()
