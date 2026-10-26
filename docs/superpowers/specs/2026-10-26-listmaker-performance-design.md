# ListMaker performance design

Implement the two ListMaker opportunities approved after the offline audit in `docs/performance/additional-opportunities.md`. Preserve all earlier local issue-127 changes.

- Ready asset objects: execute the existing validation directly when every item already supplies `symbol` and `ror`. Retain the existing threaded loader for lists containing symbol strings, preserving input/key order, object identity, duplicate resolution and short-history errors.
- LTM dividend yield: calculate calendar-complete monthly rolling sums and align them to monthly price observations, retaining the existing currency conversion, forecast filtering, zero/net-zero dividend branch and result cache. Restrict the fast path to nonempty, sorted, unique monthly PeriodIndexes and float64 dividends without infinities and numeric NumPy prices. Retain the original scalar calculation for unusual inputs outside that contract.

Alternatives: global caching would introduce freshness/invalidation and mutation risks; it is unnecessary for either measured bottleneck. Replacing the full dividend pipeline would widen the financial scope. Use the narrow fast paths and verify them against hand-calculated cases and the frozen original algorithm.

No public API changes, dependencies, network/service changes, commits, tracker mutation or deployment. Rebalance and the other opportunities remain follow-up work.
