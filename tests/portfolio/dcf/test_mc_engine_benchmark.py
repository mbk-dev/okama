"""Manual benchmark: vectorized MC engine vs the per-path reference.

Not part of the regular suite. Run with:

    OKAMA_BENCH=1 poetry run pytest tests/portfolio/dcf/test_mc_engine_benchmark.py -s
"""

import os  # noqa: I001
import time

import pytest
import okama as ok
from okama.portfolios import dcf_calculations

pytestmark = pytest.mark.skipif(not os.environ.get("OKAMA_BENCH"), reason="manual benchmark; set OKAMA_BENCH=1")


@pytest.mark.parametrize("frequency", ["year", "month"])
def test_benchmark_engine_vs_reference(synthetic_env, frequency) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = frequency
    ind.amount = -100 if frequency == "month" else -1_200
    ind.indexation = 0.05
    pf.dcf.cashflow_parameters = ind
    pf.dcf.set_mc_parameters(distribution="norm", period=30, mc_number=1_000, seed=0)
    return_ts = pf.dcf.mc.monte_carlo_returns_ts  # draw once, outside the timers

    started = time.perf_counter()
    reference = return_ts.apply(
        dcf_calculations.get_wealth_indexes_fv_with_cashflow,
        axis=0,
        args=(None, None, ind, "monte_carlo"),
    )
    reference_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = dcf_calculations.get_wealth_indexes_fv_with_cashflow_mc(return_ts, ind, pf.dcf.discount_rate)
    engine_seconds = time.perf_counter() - started

    assert result.shape == reference.shape
    assert engine_seconds < reference_seconds
    print(
        f"\n[{frequency}] mc_number=1000, period=30y: reference {reference_seconds:.2f}s, "
        f"engine {engine_seconds:.3f}s, speedup x{reference_seconds / engine_seconds:.0f}"
    )


def test_benchmark_monte_carlo_irr(synthetic_env) -> None:
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -500
    ind.indexation = 0.05
    pf.dcf.cashflow_parameters = ind
    pf.dcf.set_mc_parameters(distribution="norm", period=30, mc_number=1_000, seed=0)
    return_ts = pf.dcf.mc.monte_carlo_returns_ts  # draw once, outside the timers

    started = time.perf_counter()
    reference_wealth = return_ts.apply(
        dcf_calculations.get_wealth_indexes_fv_with_cashflow, axis=0, args=(None, None, ind, "monte_carlo")
    )
    reference_cash_flow = return_ts.apply(dcf_calculations.get_cash_flow_fv, axis=0, args=(None, ind, "monte_carlo"))
    reference_seconds = time.perf_counter() - started
    assert reference_wealth.shape[1] == reference_cash_flow.shape[1]

    ind._clear_cf_cache()
    started = time.perf_counter()
    irr = pf.dcf.monte_carlo_irr()
    routed_seconds = time.perf_counter() - started

    assert irr.shape[0] == 1_000
    assert routed_seconds < reference_seconds
    print(
        f"\n[irr] mc_number=1000, period=30y: per-path matrices {reference_seconds:.2f}s, "
        f"routed monte_carlo_irr {routed_seconds:.3f}s, speedup x{reference_seconds / routed_seconds:.0f}"
    )
