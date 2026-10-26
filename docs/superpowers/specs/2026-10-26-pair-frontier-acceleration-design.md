# Proposed acceleration of multi-period pairwise efficient frontiers

Status: **implemented after user approval**, in `24d3147`. This document preserves the design; see [measured results and compatibility boundaries](../../performance/pair-frontier-acceleration.md) for the final implementation.

## Problem and evidence

`EfficientFrontier.plot_pair_ef()` creates a new `EfficientFrontier` for each asset pair and reads its `ef_points`. The numerical path is:

`ef_points -> minimize_risk -> _get_portfolio_ror_ts -> Rebalance.return_ror_ts_ef -> wealth_ts_ef`.

These hot-path bodies are unchanged from `b33ca12`. Existing grid/Monte Carlo batching uses `_get_portfolio_points` and is not reached by this optimization path. Ordinary `Rebalance.wealth_ts` acceleration also does not replace `wealth_ts_ef`.

The [pairwise audit](../../performance/pair-frontier-audit.md) measured six pair curves, 120 months and 12 requested points. Multi-period plotting remained about 3–4 seconds. A separate profile recorded 1,300 resampler and 3,900 groupby constructions; risk minimization took 6.63 seconds of a 7.06-second instrumented call, versus about 0.01 seconds in pair constructors. Constructor-only changes therefore miss the main CPU cost. The two plotted outputs matched exactly.

Consumers that construct pair objects and read `ef_points` directly need the same inner-engine improvement. A cache or faster renderer exclusively inside `plot_pair_ef` would not accelerate their first calculation.

## Approaches

1. **Prepare return data once, then reuse it across optimizer evaluations — recommended first.** Calendar membership, period ends and cumulative asset growth depend on the return history and effective rebalance period, not on weights. Remove their repetition while retaining SLSQP, its constraints, targets, tolerances, starting points and corner handling. This benefits both full frontiers and pairwise consumers.
2. **Cache completed pair tables — useful second.** This removes repeated calculation when only plotting labels or figure size changes. It does not speed the first pair calculation, and downstream services may already cache the response.
3. **Replace the two-asset optimizer or parallelize pairs — defer.** One free weight can enable a specialized scalar solver, but non-monotonic CAGR/risk, both frontier branches and constrained endpoints require separate mathematical validation. Additional outer parallelism can multiply existing point-level joblib/BLAS concurrency and memory. Neither is a prerequisite for approach 1.

The user approved the first two approaches; specialized solvers and additional outer parallelism remain deferred.

## Recommended implementation

### 1. Reuse preparation in the shared numerical engine

Primary files: `okama/frontier/multi_period.py` and `okama/common/helpers/rebalancing.py`.

- Introduce a private calculation-scoped prepared view containing the resolved monthly index, original column order, period IDs/ends, within-period cumulative asset growth and first-month returns. Start from the existing `Rebalance._prepare_ef_data` machinery, preserving its EF-specific Grouper boundaries; do not substitute ordinary calendar-wealth boundaries.
- Build it once per stable frontier input snapshot, outside weight/objective callbacks. `_get_portfolio_ror_ts` should use a private prepared evaluator on cache misses, retaining the existing weight-result cache and rounding policy. The prepared state must be available to point workers without recomputing groupings for every trial weight.
- **Do not call `_PreparedRebalance.return_ror` unchanged for every SLSQP trial.** That sampling helper requires nonnegative weights summing to one within `1e-12`. Finite-difference trials can be temporarily off the sum constraint. Normalizing or clipping them changes the objective and gradient. The optimizer evaluator must preserve original off-constraint arithmetic, with explicit coverage for finite-difference perturbations and original fallback behavior.
- Initially retain the original statistical reduction order and pandas result wrapping where that keeps solver trajectories stable. Removing repeated grouping is valuable without changing the annual-risk/CAGR formulas. Reuse prepared numeric arrays only after full frontier equivalence has been demonstrated.
- Preserve the existing scalar path for unsupported dtypes (including mixed and nullable data), missing/non-finite returns, non-contiguous or duplicate indices, exceptional growth/wealth and custom strategy/subclass behavior.

### 2. Give prepared state an explicit lifetime

- Prefer one immutable snapshot for a complete numerical calculation. Rebuild at the next external calculation boundary if the underlying data or effective settings changed. Do not validate or copy an entire matrix on every objective evaluation.
- Existing public configuration setters must clear both prepared state and affected result caches. Observe changes to values, dtypes, row/column labels, resolved dates and rebalancing semantics; object identity alone is insufficient for mutable pandas inputs.
- Bounds, target grid, `n_points`, `full_frontier` and strategy changes invalidate final frontier/pair tables even where return preparation can mathematically be reused.
- Define ownership when sharing state with joblib workers: private immutable input arrays, no mutable global cache and no nested all-core executors. Measure serialization and worker memory as part of acceptance.

### 3. Separate pair calculation from rendering

Primary entry point: `EfficientFrontier.plot_pair_ef`.

- Extract a private iterator/helper producing pair identity plus numerical frontier tables; plotting consumes those tables. Keep the public plotting signature and Axes behavior.
- Preserve the parent's resolved common period, asset order, pair-specific bounds, labels and rebalance behavior. Do not silently expand each pair to its own longer history.
- Optionally retain only the latest validated collection of pair tables, with a bounded size/lifetime. Re-render from copies without retaining matplotlib figures. Changing label mode or figure size should not rerun numerical optimization; changing financial inputs must.
- Apply shared preparation to numerical consumers as well as plotting. A future public numerical pair API is optional and should be justified by actual consumers, not introduced solely for this refactor.
- Reusing existing assets/currency/metadata can follow a measured network bottleneck. It is secondary to the repeated inner calculations in the current offline profile.

## Verification required before shipping

Use the repository's RED -> GREEN workflow and independent frozen-reference comparisons. Do not replace observed behavior with a cleaner mathematical model without a separately approved change.

1. Prove calendar/growth preparation count is independent of optimizer objective evaluations within one stable calculation; prove it refreshes after input or settings changes.
2. Compare complete `ef_points`, weights, risk/CAGR, both frontier branches, pair order and rendered coordinates against the original. Cover all effective rebalance periods, partial calendar bins, constrained/degenerate pairs, unequal original asset histories, multiple dates, short histories, NaN/total-loss/extreme values, mixed/nullable dtypes, custom subclasses and all existing corner regressions.
3. Pin off-constraint finite-difference evaluations and failure behavior. Check near-zero risk and nearly equal asset CAGRs; do not accept a fast kernel if it changes solver convergence, targets or endpoint weights.
4. Prove repeat-plot reuse and invalidation separately; prevent mutation of returned data from corrupting stored results. Distinguish an existing service response cache from new library reuse.
5. Measure complete cold pair construction and warm replot, not just an array kernel: 4/8/12 assets (6/28/66 pairs), 120/600 months, 12/80 requested points, normal point parallelism plus a controlled single-CPU comparison. Record median/p95, failures, peak RSS, cache state and source/dependency versions. Compare full numerical tables at `rtol=atol=1e-12`; investigate discrepancies rather than widening tolerances automatically.

The implementation report contains the completed measurements. Existing MC/grid gains remain separate evidence and must not be multiplied by the pairwise results.
