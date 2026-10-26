"""Offline PortfolioDCF/MonteCarlo audit: bounded prototypes, no production edits."""

import argparse
import ast
import cProfile
import hashlib
import inspect
import json
import platform
import pstats
import subprocess
import textwrap
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial, lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.stats

import okama as ok
from analytics.benchmark_calendar_batching import Currency, returns_frame
from okama.common.helpers import helpers, tails
import matplotlib.pyplot as plt
from okama.portfolios import dcf_calculations as calc
from okama.portfolios import mc as mc_module
from tests.helpers.factories import FakeAsset

_ORIGINAL_IRR = calc.irr_of_cashflow_matrix


def compare(a: Any, b: Any) -> float:
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, rtol=1e-12, atol=1e-12)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, rtol=1e-12, atol=1e-12)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((compare(a[k], b[k]) for k in a), default=0.0)
    elif isinstance(a, tuple):
        assert len(a) == len(b)
        return max((compare(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
    if np.asarray(a).dtype.kind == "b":
        return float(a != b)
    values = np.abs(np.asarray(a) - np.asarray(b))
    return float(np.nanmax(values)) if values.size and np.isfinite(values).any() else 0.0


def measure(before: Callable, after: Callable, repeats: int = 5) -> dict[str, Any]:
    expected, actual = before(), after()
    difference = compare(actual, expected)
    result: dict[str, Any] = {"baseline_seconds": [], "prototype_seconds": [], "max_absolute_difference": difference}
    for n in range(repeats):
        pair = [("baseline", before), ("prototype", after)]
        for name, call in pair if n % 2 == 0 else pair[::-1]:
            start = time.perf_counter()
            call()
            result[name + "_seconds"].append(time.perf_counter() - start)
    for name in ("baseline", "prototype"):
        result[name + "_median_seconds"] = float(np.median(result[name + "_seconds"]))
    result["ratio"] = result["baseline_median_seconds"] / result["prototype_median_seconds"]
    return result


def peak(call: Callable) -> int:
    tracemalloc.start()
    call()
    _, maximum = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return maximum


def profile(call: Callable) -> list[dict[str, Any]]:
    profiler = cProfile.Profile()
    profiler.runcall(call)
    rows = []
    for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
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
    return sorted(rows, key=lambda r: r["cumulative_seconds"], reverse=True)[:15]


def make_dcf(kind: str = "index", frequency: str = "month", paths: int = 128, assets: int = 3) -> Any:
    data = returns_frame(600, assets)
    with patch("okama.asset.Asset", Currency):
        portfolio = ok.Portfolio([FakeAsset(c, data[c]) for c in data], ccy="USD", inflation=False, symbol="audit.PF")
    if kind == "index":
        strategy = ok.IndexationStrategy(portfolio)
        strategy.initial_investment, strategy.amount, strategy.indexation = 100_000, -100, 0.02
    elif kind == "percentage":
        strategy = ok.PercentageStrategy(portfolio)
        strategy.initial_investment, strategy.percentage = 100_000, -0.04
    elif kind == "vds":
        strategy = ok.VanguardDynamicSpending(
            portfolio,
            initial_investment=100_000,
            percentage=-0.04,
            indexation=0.02,
            min_max_annual_withdrawals=(2_000.0, 6_000.0),
            adjust_min_max=True,
        )
    else:
        strategy = ok.CutWithdrawalsIfDrawdown(
            portfolio,
            initial_investment=100_000,
            amount=-100,
            indexation=0.02,
            crash_threshold_reduction=[(0.05, 0.3), (0.15, 1.0)],
        )
    strategy.frequency = frequency
    dcf = portfolio.dcf
    dcf.cashflow_parameters = strategy
    dcf.set_mc_parameters(distribution="norm", period=30, mc_number=paths, seed=127)
    assert len(portfolio.ror) == 600
    return dcf


def clear_mc(dcf: Any) -> None:
    dcf._monte_carlo_wealth_fv = pd.DataFrame(dtype=float)
    dcf._monte_carlo_cash_flow_fv = pd.DataFrame(dtype=float)


def clear_hist(dcf: Any) -> None:
    dcf._wealth_index_fv = pd.DataFrame(dtype=float)
    dcf._cash_flow_fv = pd.DataFrame(dtype=float)


def fill_joint_mc(dcf: Any) -> None:
    returns = dcf.mc.monte_carlo_returns_ts
    wealth, cash_flow = calc._simulate_paths_mc(returns, dcf.cashflow_parameters, dcf.discount_rate)
    dcf._monte_carlo_wealth_fv = pd.DataFrame(
        np.vstack([np.full((1, returns.shape[1]), dcf.cashflow_parameters.initial_investment), wealth]),
        index=returns.index.insert(0, returns.index[0] - 1),
        columns=returns.columns,
    )
    dcf._monte_carlo_cash_flow_fv = pd.DataFrame(cash_flow, index=returns.index, columns=returns.columns)


def mc_pair(dcf: Any, joint: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    clear_mc(dcf)
    if joint:
        fill_joint_mc(dcf)
    return dcf.monte_carlo_wealth(include_negative_values=False), dcf.monte_carlo_cash_flow("fv")


def historical_wealth(dcf: Any, array: bool = False) -> pd.DataFrame:
    clear_hist(dcf)
    if array:
        returns = dcf.parent.ror
        wealth, _ = calc._simulate_paths_mc(
            returns.to_frame(), dcf.cashflow_parameters, dcf.discount_rate, task="backtest"
        )
        index = returns.index.insert(0, returns.index[0] - 1)
        if dcf.cashflow_parameters.frequency in ("month", "none"):
            index = index.rename(None)
        dcf._wealth_index_fv = pd.DataFrame(
            np.concatenate([[dcf.cashflow_parameters.initial_investment], wealth[:, 0]]),
            index=index,
            columns=[dcf.parent.name],
        )
    return dcf.wealth_index("fv", include_negative_values=True)


def historical_assets(dcf: Any, vector: bool) -> pd.DataFrame:
    if not vector:
        return dcf.wealth_index_fv_with_assets
    returns = pd.concat([dcf.parent.ror, dcf.parent.assets_ror], axis=1, join="inner")
    wealth, _ = calc._simulate_paths_mc(returns, dcf.cashflow_parameters, dcf.discount_rate, task="backtest")
    index = returns.index.insert(0, returns.index[0] - 1)
    if dcf.cashflow_parameters.frequency in ("month", "none"):
        index = index.rename(None)
    return pd.DataFrame(
        np.vstack([np.full((1, len(returns.columns)), dcf.cashflow_parameters.initial_investment), wealth]),
        index=index,
        columns=returns.columns,
    )


def guarded_helper(fn: Callable) -> Callable:
    """Prototype: avoid reassigning the same strategy from within a scalar helper."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    function = tree.body[0]
    for n, node in enumerate(function.body):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "cashflow_parameters"
        ):
            condition = ast.parse("dcf_object.cashflow_parameters is not cashflow_parameters", mode="eval").body
            function.body[n] = ast.If(test=condition, body=[node], orelse=[])
    namespace = vars(calc).copy()
    exec(compile(ast.fix_missing_locations(tree), "prototype_guarded_helper.py", "exec"), namespace)
    return namespace[fn.__name__]


def historical_chain(dcf: Any, guards: tuple[Callable, Callable] | None = None) -> tuple:
    clear_hist(dcf)
    with ExitStack() as stack:
        if guards:
            stack.enter_context(patch.object(calc, "get_wealth_indexes_fv_with_cashflow", guards[0]))
            stack.enter_context(patch.object(calc, "get_cash_flow_fv", guards[1]))
        return (dcf.wealth_index("fv"), dcf.cash_flow_ts("fv"), dcf.wealth_index("fv"), dcf.cash_flow_ts("fv"))


def survival_array(dcf: Any, threshold: float = 0) -> pd.Series:
    wealth = dcf.monte_carlo_wealth("fv", include_negative_values=False)
    dates = helpers.Frame.get_survival_date(wealth, dcf.discount_rate, threshold)
    return ((dates - dcf.parent.last_date) / np.timedelta64(365, "D")).round(1)


def survival_call(dcf: Any, array: bool, cold: bool) -> pd.Series:
    if cold:
        clear_mc(dcf)
    return survival_array(dcf, 0.2) if array else dcf.monte_carlo_survival_period(0.2)


def sparse_irr(flows: np.ndarray, periods_per_year: int = 12, **kwargs: Any) -> np.ndarray:
    values = np.asarray(flows, dtype=float)
    if (
        values.ndim == 2
        and len(values) >= 2
        and np.isfinite(values).all()
        and np.all(values[0] < 0)
        and np.all(values[-1] > 0)
        and not np.any(values[1:-1])
        and not kwargs
    ):
        rate = (values[-1] / -values[0]) ** (1 / (len(values) - 1)) - 1
        return (1 + rate) ** periods_per_year - 1
    return _ORIGINAL_IRR(flows, periods_per_year=periods_per_year, **kwargs)


def irr_public(dcf: Any, sparse: bool) -> pd.Series:
    clear_mc(dcf)
    if not sparse:
        return dcf.monte_carlo_irr()
    with patch.object(calc, "irr_of_cashflow_matrix", sparse_irr):
        return dcf.monte_carlo_irr()


def percentile_bundle(mc: Any, reused: bool) -> tuple:
    levels, scores = [5, 10, 50, 90, 95], [0.0, 0.04, 0.08]
    if not reused:
        return mc.percentile_distribution_cagr(levels), tuple(mc.percentile_inverse_cagr(s) for s in scores)
    distribution = mc._get_cagr_distribution()
    values = distribution.quantile(np.asarray(levels) / 100)
    return dict(zip(levels, values, strict=True)), tuple(
        scipy.stats.percentileofscore(distribution, s, kind="rank") for s in scores
    )


def student_backtest(mc: Any, reused: bool) -> dict:
    if not reused:
        return mc.backtesting_error()
    parameters = mc.get_parameters_for_distribution()
    with patch.object(mc, "get_parameters_for_distribution", return_value=parameters):
        return mc.backtesting_error()


def solver(dcf: Any, vector_dates: bool) -> dict:
    with ExitStack() as stack:
        if vector_dates:
            stack.enter_context(patch.object(dcf, "monte_carlo_survival_period", partial(survival_array, dcf)))
        result = dcf.find_the_largest_withdrawals_size(
            goal="maintain_balance_pv",
            withdrawals_range=(0.0, 0.1),
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


def bootstrap_values(samples: np.ndarray, alpha: float, vector: bool) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.quantile(samples, alpha, axis=1)
    if not vector:
        return quantiles, np.apply_along_axis(tails.cvar_of_sample, 1, samples, alpha)
    selected = samples <= quantiles[:, None]
    counts = selected.sum(axis=1)
    cvar = np.full(len(samples), np.nan)
    np.divide(np.where(selected, samples, 0.0).sum(axis=1), counts, out=cvar, where=counts > 0)
    return quantiles, cvar


def qq_snapshot(mc: Any, vector: bool) -> tuple:
    original = np.apply_along_axis

    def apply(fn: Callable, axis: int, values: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        if fn is tails.cvar_of_sample and axis == 1 and values.ndim == 2 and len(args) == 1 and not kwargs:
            return bootstrap_values(values, args[0], True)[1]
        return original(fn, axis, values, *args, **kwargs)

    with ExitStack() as stack:
        if vector:
            stack.enter_context(patch.object(np, "apply_along_axis", apply))
        ax = mc.plot_qq(bootstrap_size_var=2000)
        result = (
            tuple((line.get_xdata(), line.get_ydata()) for line in ax.lines),
            tuple((span.get_y(), span.get_height()) for span in ax.patches),
        )
        plt.close(ax.figure)
        return result


@lru_cache(maxsize=1)
def fresh_array_functions() -> tuple[tuple[Any, str, Callable], ...]:
    targets = [
        (calc, "get_wealth_indexes_fv_with_cashflow_mc"),
        (calc, "get_cash_flow_fv_mc"),
        (calc, "zero_wealth_after_first_void"),
        (mc_module, "generate_returns_ts"),
    ]
    results = []
    for module, name in targets:
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(module, name))))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pd"
                and node.func.attr == "DataFrame"
            ):
                node.keywords.append(ast.keyword(arg="copy", value=ast.Constant(value=False)))
        namespace = vars(module).copy()
        exec(compile(ast.fix_missing_locations(tree), "prototype_fresh_array.py", "exec"), namespace)
        results.append((module, name, namespace[name]))
    return tuple(results)


@contextmanager
def fresh_array_scope() -> Iterator[None]:
    with ExitStack() as stack:
        for module, name, fn in fresh_array_functions():
            stack.enter_context(patch.object(module, name, fn))
        yield


def copy_pair(dcf: Any, fresh: bool, cold_returns: bool = False) -> tuple:
    if cold_returns:
        dcf.mc._returns_ts_cache = None
    with fresh_array_scope() if fresh else nullcontext():
        return mc_pair(dcf)


def copy_draw(mc: Any, fresh: bool) -> pd.DataFrame:
    mc._returns_ts_cache = None
    with fresh_array_scope() if fresh else nullcontext():
        return mc.monte_carlo_returns_ts


def copy_survival(dcf: Any, fresh: bool) -> pd.Series:
    with fresh_array_scope() if fresh else nullcontext():
        return dcf.monte_carlo_survival_period(0.2)


def copy_solver(dcf: Any, fresh: bool) -> dict:
    with fresh_array_scope() if fresh else nullcontext():
        return solver(dcf, fresh)


def copy_cases(record: Callable) -> None:
    fresh_array_functions()  # Compile prototypes outside the public-call timers.
    for paths in (128, 2048, 8192):
        dcf = make_dcf(paths=paths)
        _ = dcf.mc.monte_carlo_returns_ts
        record(
            f"fresh_array_mc_pair_360x{paths}",
            partial(copy_pair, dcf, False),
            partial(copy_pair, dcf, True),
            memory=paths == 8192,
            profiling=paths == 8192,
        )
        if paths >= 2048:
            record(
                f"fresh_array_mc_pair_cold_360x{paths}",
                partial(copy_pair, dcf, False, True),
                partial(copy_pair, dcf, True, True),
            )
        record(
            f"fresh_array_draw_norm_360x{paths}", partial(copy_draw, dcf.mc, False), partial(copy_draw, dcf.mc, True)
        )
        clear_mc(dcf)
        dcf.mc._returns_ts_cache = None
        _ = dcf.mc.monte_carlo_returns_ts
        dcf.monte_carlo_wealth()
        record(
            f"fresh_array_survival_warm_360x{paths}",
            partial(copy_survival, dcf, False),
            partial(copy_survival, dcf, True),
        )
    dcf = make_dcf(paths=2048)
    dcf.discount_rate = 0.02
    _ = dcf.mc.monte_carlo_returns_ts
    record(
        "fresh_arrays_and_dates_solver_360x2048",
        partial(copy_solver, dcf, False),
        partial(copy_solver, dcf, True),
        profiling=True,
    )


def historical_cases(record: Callable) -> None:
    for assets in (3, 16):
        dcf = make_dcf(assets=assets)
        record(
            f"historical_with_assets_600x{assets}",
            partial(historical_assets, dcf, False),
            partial(historical_assets, dcf, True),
        )
    for kind, frequency in [
        ("index", "month"),
        ("index", "year"),
        ("percentage", "month"),
        ("percentage", "quarter"),
        ("vds", "year"),
        ("cwd", "month"),
    ]:
        dcf = make_dcf(kind, frequency)
        record(
            f"historical_{kind}_{frequency}_600",
            partial(historical_wealth, dcf),
            partial(historical_wealth, dcf, True),
        )
    guards = (guarded_helper(calc.get_wealth_indexes_fv_with_cashflow), guarded_helper(calc.get_cash_flow_fv))
    for frequency in ("month", "year"):
        dcf = make_dcf(frequency=frequency)
        record(
            f"historical_cache_chain_{frequency}_600",
            partial(historical_chain, dcf),
            partial(historical_chain, dcf, guards),
            profiling=True,
        )


def mc_cases(record: Callable) -> None:
    for paths in (128, 2048, 8192):
        for frequency in ("month", "year"):
            dcf = make_dcf(frequency=frequency, paths=paths)
            _ = dcf.mc.monte_carlo_returns_ts
            record(
                f"joint_mc_{frequency}_360x{paths}",
                partial(mc_pair, dcf),
                partial(mc_pair, dcf, True),
                memory=paths == 2048 and frequency == "month",
            )
        for cold in (False, True):
            dcf.monte_carlo_wealth()
            record(
                f"survival_{'cold' if cold else 'warm'}_360x{paths}",
                partial(survival_call, dcf, False, cold),
                partial(survival_call, dcf, True, cold),
            )
    for paths in (2048, 8192):
        dcf = make_dcf(frequency="none", paths=paths)
        wealth = dcf.monte_carlo_wealth()
        flows = np.zeros((361, paths))
        flows[0], flows[-1] = -100_000, wealth.iloc[-1]
        record(
            f"sparse_irr_kernel_361x{paths}",
            partial(calc.irr_of_cashflow_matrix, flows),
            partial(sparse_irr, flows),
            memory=paths == 2048,
        )
        record(f"no_cashflow_irr_public_360x{paths}", partial(irr_public, dcf, False), partial(irr_public, dcf, True))
    dcf = make_dcf(paths=2048)
    dcf.discount_rate = 0.02
    _ = dcf.mc.monte_carlo_returns_ts
    record("withdrawal_solver_360x2048", partial(solver, dcf, False), partial(solver, dcf, True), profiling=True)


def diagnostics_cases(record: Callable) -> None:
    dcf_plot = make_dcf()
    samples = np.random.default_rng(11).choice(dcf_plot.mc.ror, size=(2000, len(dcf_plot.mc.ror)))
    record(
        "bootstrap_cvar_2000x600",
        partial(bootstrap_values, samples, 0.05, False),
        partial(bootstrap_values, samples, 0.05, True),
    )
    record(
        "plot_qq_full_600_history", partial(qq_snapshot, dcf_plot.mc, False), partial(qq_snapshot, dcf_plot.mc, True)
    )
    for paths in (2048, 8192):
        dcf = make_dcf(paths=paths)
        _ = dcf.mc.monte_carlo_returns_ts
        record(
            f"cagr_queries_360x{paths}",
            partial(percentile_bundle, dcf.mc, False),
            partial(percentile_bundle, dcf.mc, True),
        )
    dcf = make_dcf()
    dcf.mc.distribution = "t"
    record(
        "student_backtesting_600_history",
        partial(student_backtest, dcf.mc, False),
        partial(student_backtest, dcf.mc, True),
        profiling=True,
    )


def work_counts() -> dict[str, Any]:
    result = {}
    dcf = make_dcf(paths=2048)
    _ = dcf.mc.monte_carlo_returns_ts
    for joint in (False, True):
        with patch.object(calc, "_simulate_paths_mc", wraps=calc._simulate_paths_mc) as counter:
            mc_pair(dcf, joint)
            result["mc_engine_joint" if joint else "mc_engine_baseline"] = counter.call_count
    originals = calc.get_wealth_indexes_fv_with_cashflow, calc.get_cash_flow_fv
    guards = tuple(guarded_helper(fn) for fn in originals)
    for label, functions in (("baseline", originals), ("guard", guards)):
        with patch.object(calc, "get_wealth_indexes_fv_with_cashflow", wraps=functions[0]) as wealth:
            with patch.object(calc, "get_cash_flow_fv", wraps=functions[1]) as cash:
                historical_chain(dcf)
                result["historical_" + label] = {"wealth_calls": wealth.call_count, "cashflow_calls": cash.call_count}
    dcf.mc.distribution = "t"
    for reused in (False, True):
        with patch.object(scipy.stats.t, "fit", wraps=scipy.stats.t.fit) as fit:
            student_backtest(dcf.mc, reused)
            result["student_fit_reused" if reused else "student_fit_baseline"] = fit.call_count
    dcf.mc.distribution = "norm"
    _ = dcf.mc.monte_carlo_returns_ts
    for reused in (False, True):
        with patch.object(dcf.mc, "_get_cagr_distribution", wraps=dcf.mc._get_cagr_distribution) as cagr:
            percentile_bundle(dcf.mc, reused)
            result["cagr_reused" if reused else "cagr_baseline"] = cagr.call_count
    dcf.discount_rate = 0.02
    original = solver(dcf, False)
    with fresh_array_scope():
        optimized = solver(dcf, True)
    compare(optimized, original)
    result["solver"] = {
        "baseline_evaluations": len(original["solutions"]),
        "prototype_evaluations": len(optimized["solutions"]),
        "success": original["success"],
    }
    for distribution in ("norm", "t", "lognorm"):
        dcf.mc.distribution = distribution
        dcf.mc.seed = 127
        baseline = percentile_bundle(dcf.mc, False)
        dcf.mc._returns_ts_cache = None
        with fresh_array_scope():
            candidate = percentile_bundle(dcf.mc, False)
        result["layout_cagr_difference_" + distribution] = compare(candidate, baseline)
    values = np.zeros((12, 8))
    result["pandas_default_constructor_shares_numpy"] = bool(np.shares_memory(pd.DataFrame(values).to_numpy(), values))
    result["explicit_no_copy_shares_numpy"] = bool(
        np.shares_memory(pd.DataFrame(values, copy=False).to_numpy(), values)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=["all", "historical", "mc", "diagnostics", "copy"], default="all")
    parser.add_argument("--counts", action="store_true")
    args = parser.parse_args()
    result: dict[str, Any] = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "protocol": "Narrow offline audit prototypes only; 1 warm pair then5 alternating pairs, strict1e-12 numerical/metadata checks",
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "okama/portfolios/dcf.py",
                "okama/portfolios/dcf_calculations.py",
                "okama/portfolios/mc.py",
                "okama/common/helpers/helpers.py",
            )
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": {},
        "profiles": {},
    }

    def record(name: str, before: Callable, after: Callable, memory: bool = False, profiling: bool = False) -> None:
        row = measure(before, after)
        if memory:
            row["baseline_peak_bytes"], row["prototype_peak_bytes"] = peak(before), peak(after)
        result["cases"][name] = row
        if profiling:
            result["profiles"][name] = profile(before)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(name, row, flush=True)

    if args.group in ("all", "copy"):
        copy_cases(record)
    if args.group in ("all", "historical"):
        historical_cases(record)
    if args.group in ("all", "mc"):
        mc_cases(record)
    if args.group in ("all", "diagnostics"):
        diagnostics_cases(record)
    if args.counts:
        result["work_counts"] = work_counts()
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
