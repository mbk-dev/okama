# Feasible target CAGRs under weight bounds

Issue [#132](https://github.com/mbk-dev/okama/issues/132) used SPY.US and GLD.US
over 2015-01 through 2020-12, annual rebalancing, weights constrained to 20–80%,
and three requested frontier points. The old target construction used individual
asset returns even when those 100% asset corners were prohibited.

For restrictive bounds, `EfficientFrontier` now searches within the box and the
sum-to-one constraint for the low-CAGR endpoint of the full left branch and the
maximum-risk endpoint of the right branch. Multiple feasible starting portfolios
reduce dependence on one local SLSQP start. The default unrestricted-corner
construction remains covered by the existing regression tests. This is numerical
optimization, not a proof of a global optimum for every possible return path.

A live repeat of the issue's fixed-date example on 2026-10-26 returns three feasible
portfolios in both modes instead of raising `RuntimeError`:

| full_frontier | Target CAGRs | SPY weights |
|---|---|---|
| True | 8.8222%, 10.2779%, 11.7335% | 0.200000, 0.495526, 0.799999 |
| False | 10.2067%, 10.9701%, 11.7335% | 0.480865, 0.639171, 0.799999 |

The largest target-versus-recomputed CAGR difference in this probe is below
0.000001 in decimal annual return. The previously attempted 12.6659% target is
outside the feasible range and is no longer generated. Synthetic offline tests
also cover asymmetric two- and three-asset bounds and both frontier modes.

Reproduce the live probe (data can be revised upstream):

```python
import okama as ok

ok.settings.n_jobs = 1
for full in (True, False):
    frontier = ok.EfficientFrontier(
        assets=["SPY.US", "GLD.US"], ccy="USD",
        first_date="2015-01", last_date="2020-12",
        bounds=((0.2, 0.8), (0.2, 0.8)), n_points=3,
        inflation=False, full_frontier=full,
        rebalancing_strategy=ok.Rebalance(period="year"),
    )
    print(f"full_frontier={full}")
    print(frontier.ef_points[["CAGR", "Risk", "Weights"]])
```
