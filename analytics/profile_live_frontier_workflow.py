"""One bounded live workflow observation; not a paired network speedup claim."""

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import okama as ok
from okama.api.api_methods import API


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial-ready-assets", action="store_true")
    args = parser.parse_args()
    result: dict[str, Any] = {
        "protocol": "One current-candidate live observation; no baseline or warm/cold network equivalence claim",
        "loading_mode": "serial_public_assets" if args.serial_ready_assets else "frontier_symbol_constructor",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "symbols": ["SPY.US", "BND.US", "GLD.US"],
        "first_date": "2005-01",
        "last_date": "2025-12",
        "stages_seconds": {},
        "requests": [],
    }
    stage = "initializing"
    original_connect = API.connect

    def connect(*args: Any, **kwargs: Any) -> str:
        start = time.perf_counter()
        request = {
            "stage": stage,
            "endpoint": kwargs.get("endpoint"),
            "symbol": kwargs.get("symbol"),
            "first_date": str(kwargs["first_date"]) if kwargs.get("first_date") is not None else None,
            "last_date": str(kwargs["last_date"]) if kwargs.get("last_date") is not None else None,
            "period": kwargs.get("period"),
        }
        try:
            response = original_connect(*args, **kwargs)
            request["success"] = True
            return response
        except Exception as error:
            request["success"] = False
            request["error_type"] = type(error).__name__
            raise
        finally:
            request["seconds"] = time.perf_counter() - start
            result["requests"].append(request)

    def timed(name: str, call: Callable) -> Any:
        nonlocal stage
        stage = name
        start = time.perf_counter()
        try:
            return call()
        finally:
            result["stages_seconds"][name] = time.perf_counter() - start

    begin = time.perf_counter()
    try:
        with patch.object(API, "connect", side_effect=connect):
            if args.serial_ready_assets:
                assets = timed(
                    "asset_load",
                    lambda: [
                        ok.Asset(symbol, first_date="2005-01", last_date="2025-12") for symbol in result["symbols"]
                    ],
                )
                frontier = timed(
                    "frontier_setup",
                    lambda: ok.EfficientFrontierSingle(assets, ccy="USD", inflation=False, n_points=20),
                )
            else:
                frontier = timed(
                    "frontier_load_and_setup",
                    lambda: ok.EfficientFrontierSingle(
                        result["symbols"],
                        first_date="2005-01",
                        last_date="2025-12",
                        ccy="USD",
                        inflation=False,
                        n_points=20,
                    ),
                )
                assets = list(frontier.asset_obj_dict.values())
            points = timed("frontier_solve", lambda: frontier.ef_points)
            tangency = timed("tangency", lambda: frontier.get_tangency_portfolio(rate_of_return="cagr", rf_return=0.01))
            portfolio = timed(
                "portfolio_setup",
                lambda: ok.Portfolio(
                    assets,
                    ccy="USD",
                    inflation=False,
                    weights=tangency["Weights"].tolist(),
                    rebalancing_strategy=ok.Rebalance(period="year"),
                    symbol="live_workflow.PF",
                ),
            )

            def metrics() -> tuple[Any, ...]:
                return (
                    portfolio.ror,
                    portfolio.get_cagr(),
                    portfolio.risk_annual,
                    portfolio.get_rolling_cagr(60),
                    portfolio.rebalancing_events,
                )

            timed("portfolio_metrics", metrics)
            result.update(success=True, frontier_rows=len(points), return_months=len(frontier.assets_ror))
    except Exception as error:
        result.update(success=False, failed_stage=stage, error_type=type(error).__name__, error=str(error))
    finally:
        result["total_seconds"] = time.perf_counter() - begin
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()
