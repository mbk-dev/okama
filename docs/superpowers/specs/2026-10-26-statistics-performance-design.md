# GMV statistics and rolling CAGR acceleration

Continue the approved performance audit with two bounded changes, without altering financial definitions, solver settings, dependencies or public result schemas. Existing task worktree and earlier local changes are retained; no commits, publication or tracker writes are authorized.

GMV: prepare covariance (and annual-objective means) once per solve. Retain pandas covariance multiplication, square-root risk objective, SLSQP settings, bounds, initial guess and errors to avoid numerical changes in solver trajectories. Reuse weights once in `gmv_monthly` and its result once in `gmv_annualized`. Statistics are request-local, never cached across mutations.

Rolling CAGR: introduce an explicit `Frame.get_rolling_cagr` helper used by AssetList, Portfolio and rolling tracking-difference CAGR. Generic `get_rolling_fn` remains unchanged for arbitrary callbacks. For ordinary float64 positive growth, compute log of the same rounded `1 + ror` factors, rolling sums and annualized expm1. Preserve window validation, min-periods, dropna, index/columns/name and real-return conversion. Nonstandard dtypes, nonpositive/infinite growth and histories whose conservative log bound could underflow/overflow intermediate products retain the original callback implementation. The safe fast path is intentionally narrower than the full API.

Alternatives: persistent statistics caching needs invalidation; variance objectives/analytic gradients alter solver trajectories; raw rolling products still scale with window size; unrestricted logarithms change exceptional product semantics. None are selected. CPU NumPy/pandas only, Python >=3.11.

Acceptance: strict TDD for repeated covariance/solve work and rolling callback elimination; original-scalar comparisons including edge cases and mutations; actual public-call paired benchmarks with distributions, complete equivalence and source hashes; full pytest and both Ruff commands; independent final review and Git lock check. Document precision limits and bounded scope.
