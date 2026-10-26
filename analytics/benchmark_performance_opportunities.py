"""Offline exploratory prototypes, NOT production-ready replacements.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 poetry run python -m
analytics.benchmark_performance_opportunities --output tmp/opportunities.json
No library methods or public behavior are modified. Synthetic, finite returns
and complete monthly histories are used; see the report for untested cases.
"""

import argparse
import cProfile
import hashlib
import io
import json
import math
import platform
import pstats
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any
from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pandas as pd

import okama as ok
from okama.api.data_queries import QueryData
from okama.common.helpers import helpers
from okama.common.make_asset_list import ListMaker
from tests.helpers.factories import FakeAsset, FakeCurrencyAsset


class AuditCurrencyAsset(FakeCurrencyAsset):
    """Keep the base currency fixture from truncating pre-1990 research histories."""

    def __init__(self, symbol: str, first_date: str | None = None, last_date: str | None = None) -> None:
        super().__init__(symbol, first_date=first_date or "1900-01", last_date=last_date or "2100-01")


def returns_frame(months: int, assets: int) -> pd.DataFrame:
    rng = np.random.default_rng(127)
    return pd.DataFrame(
        rng.normal(0.004, 0.04, (months, assets)),
        index=pd.period_range("1970-03", periods=months, freq="M"),
        columns=[f"A{i}.US" for i in range(assets)],
    )


def measure_pair(baseline: Callable, prototype: Callable, repeats: int = 5) -> dict[str, Any]:
    """One warm pair, alternating execution order, equality before collecting timings."""
    expected, actual = baseline(), prototype()
    if isinstance(expected, dict):
        assert list(expected) == list(actual)
        assert all(expected[k] is actual[k] for k in expected)
    else:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-11, atol=1e-11)
    result: dict[str, Any] = {"baseline_seconds": [], "prototype_seconds": []}
    for i in range(repeats):
        order = (("baseline_seconds", baseline), ("prototype_seconds", prototype))
        for key, call in order if i % 2 == 0 else order[::-1]:
            start = time.perf_counter()
            call()
            result[key].append(time.perf_counter() - start)
    result["baseline_median_seconds"] = float(np.median(result["baseline_seconds"]))
    result["prototype_median_seconds"] = float(np.median(result["prototype_seconds"]))
    result["ratio"] = result["baseline_median_seconds"] / result["prototype_median_seconds"]
    return result


def profile(call: Callable) -> str:
    profiler = cProfile.Profile()
    profiler.runcall(call)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(18)
    return stream.getvalue()


def conditional_numpy(ror: pd.DataFrame, weights: np.ndarray, deviation: float) -> np.ndarray:
    """Prototype for period=none, absolute threshold only, positive finite growth."""
    values = ror.to_numpy()
    allocation = weights * 1000
    wealth = np.empty(len(values) + 1)
    wealth[0] = 1000
    rebalance = False
    for i, row in enumerate(values):
        if rebalance:
            allocation = weights * allocation.sum()
        allocation *= 1 + row
        total = allocation.sum()
        wealth[i + 1] = total
        rebalance = bool(np.any(np.abs(allocation / total - weights) > deviation))
    return wealth


def ef_batch(ror: pd.DataFrame, weights: np.ndarray, period: str) -> np.ndarray:
    """Prototype includes preparation; dense periods x portfolios, not chunked."""
    strategy = ok.Rebalance(period=period)
    if period == "none":
        wealth = (1 + ror).cumprod().to_numpy() @ weights.T
    else:
        ids = ror.groupby(pd.Grouper(freq=strategy._pandas_frequency_grouper, convention="start")).ngroup()
        growth = (1 + ror).groupby(ids).cumprod().to_numpy()
        local = growth @ weights.T
        ends = np.r_[np.flatnonzero(np.diff(ids.to_numpy())), len(ids) - 1]
        multipliers = np.vstack([np.ones(len(weights)), local[ends[:-1]]]).cumprod(axis=0)
        wealth = multipliers[ids.to_numpy()] * local
    result = np.empty_like(wealth)
    result[0] = ror.iloc[0].to_numpy() @ weights.T
    result[1:] = wealth[1:] / wealth[:-1] - 1
    return result


class SequentialParallel:
    """Execute the real joblib-delayed validation calls without starting threads."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, calls: Any) -> list[Any]:
        return [fn(*args, **kwargs) for fn, args, kwargs in calls]


def offline_list(ror: pd.DataFrame) -> ok.AssetList:
    assets = [FakeAsset(name, ror[name]) for name in ror]
    with patch("okama.common.make_asset_list.asset.Asset", AuditCurrencyAsset):
        return ok.AssetList(assets, ccy="USD", inflation=False)


def dividend_rolling(asset_list: ok.AssetList) -> pd.DataFrame:
    """Prototype for complete monthly prices and positive dividends in base currency."""
    dividends = asset_list._get_assets_dividends()
    return pd.DataFrame(
        {
            name: dividends[name].rolling(12, min_periods=1).sum()
            / obj.close_monthly.loc[asset_list.first_date : asset_list.last_date]
            for name, obj in asset_list.asset_obj_dict.items()
        }
    )


def csv_c_parser(csv: str, period: str) -> pd.Series:
    frame = pd.read_csv(
        io.StringIO(csv),
        delimiter=",",
        index_col=0,
        parse_dates=[0],
        dayfirst=False,
        date_format="mixed",
        dtype={1: np.float64},
        engine="c",
    )
    if not frame.empty:
        frame.index = frame.index.to_period(period.upper())
        frame = frame.squeeze("columns")
    return frame


def rebalance_cases(cases: dict[str, Any], profiles: dict[str, str]) -> None:
    for months, assets in ((252, 3), (600, 16)):
        ror = returns_frame(months, assets)
        weights = np.full(assets, 1 / assets)
        threshold = ok.Rebalance(period="none", abs_deviation=0.05)

        def reference(threshold: Any = threshold, weights: np.ndarray = weights, ror: pd.DataFrame = ror) -> pd.Series:
            return threshold.wealth_ts(weights, ror).portfolio_wealth_index

        key = f"rebalance_conditional_{months}x{assets}"
        cases[key] = measure_pair(reference, lambda ror=ror, weights=weights: conditional_numpy(ror, weights, 0.05))
        profiles[key] = profile(reference)
        for period in ("month", "year"):
            strategy = ok.Rebalance(period=period)

            # Existing EF method gives the same portfolio wealth for unconditional calendars.
            def fast_wealth(
                strategy: Any = strategy, weights: np.ndarray = weights, ror: pd.DataFrame = ror
            ) -> np.ndarray:
                return np.r_[1000, strategy.wealth_ts_ef(weights, ror).to_numpy()]

            key = f"rebalance_calendar_{period}_{months}x{assets}"

            def reference(
                strategy: Any = strategy, weights: np.ndarray = weights, ror: pd.DataFrame = ror
            ) -> pd.Series:
                return strategy.wealth_ts(weights, ror).portfolio_wealth_index

            cases[key] = measure_pair(reference, fast_wealth)
            profiles[key] = profile(reference)
    ror = returns_frame(252, 3)
    for period in ("none", "month", "quarter", "half-year", "year"):
        strategy = ok.Rebalance(period=period)
        for n in (128, 2048):
            weights = np.random.default_rng(127).dirichlet(np.ones(3), n)
            key = f"rebalance_ef_batch_{period}_{n}"

            def reference(strategy: Any = strategy, weights: np.ndarray = weights) -> np.ndarray:
                return np.column_stack([strategy.return_ror_ts_ef(row, ror).to_numpy() for row in weights])

            cases[key] = measure_pair(
                reference, lambda weights=weights, period=period: ef_batch(ror, weights, period), repeats=3
            )
            print(key, cases[key]["ratio"], flush=True)


def listmaker_cases(cases: dict[str, Any], profiles: dict[str, str]) -> None:
    for assets in (3, 32, 128):
        ror = returns_frame(252, assets)
        objects = [FakeAsset(name, ror[name]) for name in ror]

        def reference(objects: Any = objects) -> dict:
            return ListMaker._get_asset_obj_dict(objects)

        def sequential(objects: Any = objects) -> dict:
            with patch("okama.common.make_asset_list.Parallel", SequentialParallel):
                return ListMaker._get_asset_obj_dict(objects)

        cases[f"listmaker_ready_objects_{assets}"] = measure_pair(reference, sequential)

        def normal_init(objects: Any = objects) -> pd.DataFrame:
            with patch("okama.common.make_asset_list.asset.Asset", AuditCurrencyAsset):
                return ok.AssetList(objects, ccy="USD", inflation=False).assets_ror

        def sequential_init(normal_init: Callable = normal_init) -> pd.DataFrame:
            with patch("okama.common.make_asset_list.Parallel", SequentialParallel):
                return normal_init()

        cases[f"listmaker_full_init_{assets}"] = measure_pair(normal_init, sequential_init)
    for months, assets in ((252, 3), (600, 16)):
        asset_list = offline_list(returns_frame(months, assets))
        assert len(asset_list.assets_ror) == months

        def cold_yield(asset_list: ok.AssetList = asset_list) -> pd.DataFrame:
            asset_list._dividend_yield = pd.DataFrame()
            return asset_list._assets_dividend_yield

        key = f"listmaker_dividend_yield_{months}x{assets}"
        cases[key] = measure_pair(cold_yield, lambda asset_list=asset_list: dividend_rolling(asset_list), repeats=3)
        profiles[key] = profile(cold_yield)
        # Both calculations above reuse the existing cached dividend data.
        start = time.perf_counter()
        for _ in range(100):
            _ = asset_list._assets_dividend_yield
        cases[key]["warm_cached_seconds_per_call"] = (time.perf_counter() - start) / 100
        print(key, cases[key]["ratio"], flush=True)


def adjacent_cases(cases: dict[str, Any]) -> None:
    for months in (252, 6000):
        series = returns_frame(months, 1).iloc[:, 0]
        frame = series.to_frame()
        frame.index = frame.index.to_timestamp()
        csv = frame.to_csv()
        pd.testing.assert_series_equal(QueryData.csv_to_series(csv, "M"), csv_c_parser(csv, "M"))
        cases[f"query_csv_parser_{months}"] = measure_pair(
            lambda csv=csv: QueryData.csv_to_series(csv, "M"), lambda csv=csv: csv_c_parser(csv, "M")
        )
    ror = returns_frame(600, 16)
    cases["rolling_cagr_600x16_window60"] = measure_pair(
        lambda: helpers.Frame.get_rolling_fn(ror, 60, helpers.Frame.get_cagr),
        lambda: np.expm1(np.log1p(ror).rolling(60).sum() / 5).dropna(),
        repeats=3,
    )
    # Count actual per-currency constructor calls, with the network replaced only at its boundary.
    assets = [FakeAsset(name, ror[name], currency="EUR") for name in ror]
    calls = []

    def currency(symbol: str, **kwargs: Any) -> FakeCurrencyAsset:
        calls.append(symbol)
        return AuditCurrencyAsset(symbol, **kwargs)

    with patch("okama.common.make_asset_list.asset.Asset", side_effect=currency):
        ok.AssetList(assets, ccy="USD", inflation=False)
    cases["listmaker_fx_constructor_counts"] = {"assets": len(assets), "calls": calls}


def optimizer_cases() -> dict[str, Any]:
    """Profile actual SLSQP calls, changing only statistics preparation in a scoped prototype."""
    cases = {}
    for assets in (3, 32):
        ror = returns_frame(252, assets)
        objects = [FakeAsset(name, ror[name]) for name in ror]
        with patch("okama.common.make_asset_list.asset.Asset", AuditCurrencyAsset):
            ef = ok.EfficientFrontierSingle(objects, ccy="USD", inflation=False)

        def reference(ef: Any = ef) -> np.ndarray:
            return ef.gmv_monthly_weights

        def prepared(ef: Any = ef) -> np.ndarray:
            covariance = ef.assets_ror.cov().to_numpy()

            def risk(weights: np.ndarray, assets_ror: pd.DataFrame) -> float:
                return math.sqrt(weights.T @ covariance @ weights)

            with patch.object(helpers.Frame, "get_portfolio_risk", staticmethod(risk)):
                return ef.gmv_monthly_weights

        key = f"gmv_monthly_{assets}_assets"
        cases[key] = measure_pair(reference, prepared)
        with patch.object(ef.assets_ror, "cov", wraps=ef.assets_ror.cov) as calls:
            reference()
            cases[key]["baseline_covariance_calls"] = calls.call_count
        print(key, cases[key]["ratio"], flush=True)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizers-only", action="store_true")
    parser.add_argument("--listmaker-only", action="store_true")
    args = parser.parse_args()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Exploratory offline prototypes; no production modifications",
        "cases": {},
        "profiles": {},
    }
    if args.optimizers_only:
        report["cases"] = optimizer_cases()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    # Existing pandas warnings can dominate these experiments; suppress only known deprecations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if hasattr(pd.errors, "Pandas4Warning"):
            warnings.simplefilter("ignore", pd.errors.Pandas4Warning)
        if not args.listmaker_only:
            rebalance_cases(report["cases"], report["profiles"])
        listmaker_cases(report["cases"], report["profiles"])
        adjacent_cases(report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
