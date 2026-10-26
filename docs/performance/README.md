# Calculation performance work

These reports describe implementation stages for issue 127 and the follow-up performance audit. Each report states its own measurement protocol, tested source digests and limitations; timings from different stages are separate experiments. Individual reports record the local delivery status at their measurement date. Integration targets the `dev` branch. Historical measurements used Poetry/Python 3.11; the current `dev` environment uses uv, so reproduction commands should use `uv run` after syncing the required groups. Different dependency versions can change timings.

- [Single-period frontier batching](frontier-cpu-batching.md): sampled Monte Carlo and grid portfolios.
- [ListMaker preparation and dividend yield](listmaker-optimizations.md): ready asset objects and ordinary monthly dividend calculations.
- [Conditional Rebalance](rebalance-conditional.md): chronological threshold-based portfolio wealth.
- [Calendar Rebalance and multi-period frontier batching](calendar-batching.md): calendar periods plus repeated Monte Carlo/grid weights.
- [GMV statistics and rolling CAGR](statistics-performance.md): per-solve covariance/means, repeated-result reuse and guarded rolling compounding.
- [Remaining frontier optimizers and complete workflow](frontier-workflow.md): incremental optimizer/CPU workflow comparisons plus a separate live API observation.
- [Scoped API request reuse](api-request-scope.md): automatic thread-isolated HTTP connection reuse and optional operation-scoped metadata reuse.
- [PortfolioDCF and MonteCarlo audit](dcf-mc-opportunities.md): 44 paired research cases; new opportunities only, without production changes.
- [Original opportunity audit](additional-opportunities.md): historical prototypes and candidate prioritization, not proof of production implementation.

The benchmark scripts are in `analytics/`. Ordinary financial behavior is compared to original scalar implementations; unsupported and numerically exceptional inputs retain the original paths. Exact coverage and numerical tolerances are documented in each report.

Still outside the implemented stages: most-diversified objective preparation, CSV parser changes, currency time-series coverage reuse and persistent caches. No GPU/JIT dependency or global runtime configuration is introduced.
