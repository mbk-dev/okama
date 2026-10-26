# Conditional Rebalance performance design

Continue the approved Rebalance stage from the performance audit. Implement the measured chronological threshold path in `Rebalance._rebalance_by_condition()`; calendar and multi-weight EF paths remain separate changes.

Use a float64 NumPy loop with preallocated portfolio and optional per-asset outputs. Check thresholds through the existing `_check_if_rebalancing_required()` hook, adding ndarray support without changing pandas callers. Build pandas containers once, preserving the initial investment point, metadata, event dates/types, absolute-before-relative priority, strict comparisons, NaN sums and zero target weights.

Keep the original pandas implementation as a fallback for nonstandard input types/indexes and custom subclasses. Use the array path only for standard Rebalance instances, sorted unique monthly PeriodIndexes, float64 return columns and positional float64 weights (or binary integer weights with exact initial allocations). No formula, period/threshold, public API, dependency or process-setting changes.

Alternatives: vectorizing all time steps would break state-dependent thresholds; JIT dependencies are unnecessary. Calendar batching and EF preparation are valid later candidates but widen this stage's financial scope.

Earlier local frontier/ListMaker changes are retained. No commits, pushes, deployment or tracker mutation are authorized.
