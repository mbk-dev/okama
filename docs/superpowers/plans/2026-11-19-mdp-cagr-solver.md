# MDP target CAGR recovery

[Issue #158](https://github.com/mbk-dev/okama/issues/158) reproduces a line-search failure on three assets. The offline fixture records the 114 aligned RUB monthly returns served on 2026-11-19 for the issue's date range.

- Verify all solver candidates against weight bounds, the weight sum and the requested CAGR. Calculate returned metrics at the accepted weights.
- Keep the single-asset corner fallback. For an interior target, retry from a verified minimum-risk solution that already reaches that target.
- For the global MDP marker, retry from the annualized minimum-volatility portfolio when equal weights fail.
- Report infeasibility only when the target is outside an independently established return bound. Return a typed solver failure for unresolved targets and preserve the all-or-error line.
- Check the original data, ordinary MDP lines and the backend response for a failed calculation.
