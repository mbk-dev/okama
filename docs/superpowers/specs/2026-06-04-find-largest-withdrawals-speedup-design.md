# Speeding up `Portfolio.dcf.find_the_largest_withdrawals_size` — design

Date: 2026-06-04
Status: approved (brainstorming session)

## Problem

`PortfolioDCF.find_the_largest_withdrawals_size` (`okama/portfolios/dcf.py:892`)
finds the largest withdrawal size that satisfies a goal
(`maintain_balance_pv`, `maintain_balance_fv`, `survival_period`) at a given
Monte Carlo percentile. The current implementation is slow for two independent
reasons:

1. **Linear-convergence search.** A hand-rolled bisection runs up to
   `iter_max=20` iterations, each gaining one bit of precision.
2. **Expensive iterations.** Every iteration re-simulates all Monte Carlo
   wealth paths from scratch: the `amount`/`percentage` setter clears the
   `_monte_carlo_wealth_fv` cache (`cashflow_strategies.py:169`), and
   `monte_carlo_wealth` rebuilds it via
   `return_ts.apply(get_wealth_indexes_fv_with_cashflow, axis=0)` — a pure
   Python loop over months *inside each of N paths*
   (`dcf_calculations.py:71` itertuples branch) or pandas `resample` chunks
   per period (`dcf_calculations.py:101` branch for non-monthly frequencies).

Key enabling facts confirmed in the code:

- Random return scenarios **are** cached across iterations
  (`mc.py:436`, `_returns_ts_cache` is not cleared by withdrawal setters), so
  the objective function is deterministic during the search. Fast
  interpolation-based root finders are therefore safe (no MC noise between
  evaluations).
- `scipy` (>= 1.9) and `joblib` are already project dependencies; no new
  dependencies are needed.
- For `IndexationStrategy` and `PercentageOnDrawdownsStrategy` (CWD) the
  wealth recursion without absorption is **linear** in `amount` (the CWD
  drawdown-based reduction factor depends only on the return path,
  `cashflow_strategies.py:958`). Extra cash flows from `time_series` do not
  depend on `amount`, so they preserve linearity.
- For `PercentageStrategy` the recursion is **multiplicative** in the
  percentage; extra `time_series` cash flows break this structure.
- For `VanguardDynamicSpendingStrategy` (VDS) the floor/ceiling logic versus
  the previous withdrawal (`cashflow_strategies.py:714`) breaks both
  structures.

## Goals

- Substantially reduce wall-clock time of
  `find_the_largest_withdrawals_size` for all supported strategies.
- Speed up every `monte_carlo_*` method as a side effect (shared engine).

## Non-goals / compatibility contract

- Public signature of `find_the_largest_withdrawals_size` and the `Result`
  shape (`success`, `withdrawal_abs`, `withdrawal_rel`, `error_rel`,
  `solutions` with columns `withdrawal_abs`, `withdrawal_rel`, `error_rel`,
  `error_rel_change`) are preserved.
- Bit-for-bit reproduction of intermediate steps is **not** required: the
  number and values of recorded attempts may change; the found root may
  differ within `tolerance_rel`.
- The stopping criterion stays a *function-value* tolerance
  (`error_rel < tolerance_rel`), not a bracket-width tolerance.

## Architecture overview

`find_the_largest_withdrawals_size` keeps its signature and dispatches to a
solver based on a structural capability of the cash flow strategy:

```
find_the_largest_withdrawals_size(...)
 │
 ├─ strategy capability (class attribute `_withdrawal_structure`):
 │   • "linear"         → IndexationStrategy, CWD            (stage 3)
 │   • "multiplicative" → PercentageStrategy, only when the
 │     extra `time_series` cash flows are absent/zero        (stage 3)
 │   • None             → VDS, unknown subclasses,
 │     PercentageStrategy with extra cash flows → generic path (stage 1)
 │
 └─ generic path: Brent root finding on a signed residual,
     one full MC simulation per evaluation (made fast by stage 2)
```

The capability is declared as a class attribute (`_withdrawal_structure =
"linear" | "multiplicative" | None`), not an `isinstance` chain: future
subclasses default to `None` and safely take the generic path.

Delivery is staged; each stage is a separate TDD cycle and a separately
shippable change:

1. **Stage 1** — replace bisection with Brent (generic path).
2. **Stage 2** — vectorized Monte Carlo wealth engine.
3. **Stage 3** — structural shortcut solvers with final verification.

## Stage 1 — Brent instead of bisection (generic path)

**Signed residual.** Parametrize by the main parameter value `m` (`amount` or
`percentage`, negative for withdrawals). Replace the
`(condition: bool, error_rel: abs)` pair with a signed function `f(m)`:

- `survival_period`: `f(m) = sp_at_quantile(m) − target_survival_period`
- `maintain_balance_pv/fv`: `f(m) = wealth_at_quantile(m) − start_investment`;
  if additionally `sp_at_quantile < mc.period` (the compound condition at
  `dcf.py:1147`), force the residual negative ("withdrawal too large"). This
  keeps the sign of `f` equivalent to the current `condition`.

`f` is monotonically increasing in `m` (smaller withdrawal → longer survival
/ more wealth).

**Bracketing.** Evaluate `f` at both ends of `withdrawals_range` first:

- `f(m_min) < 0` (even the smallest withdrawal fails the goal) → no root in
  range: return `success=False` with the better endpoint (mirrors the current
  iter-exhaustion outcome).
- `f(m_max) > 0` (even the largest withdrawal sustains the goal) → root is
  outside the range: return `m_max`; `success` follows from the actual
  `error_rel < tolerance_rel`.
- Sign change → `scipy.optimize.brentq(f, m_min, m_max, maxiter=...)`.

**Early stop on tolerance.** `brentq` stops on `xtol`, but the public
contract stops on function value. A wrapper around `f` computes `error_rel`
on every call, appends a row to `solutions`, and raises an internal
`_Converged` exception carrying the result when `error_rel < tolerance_rel`;
the caller catches it and returns `Result(success=True, ...)`. `iter_max`
becomes the limit on **function evaluations** (including the 2 endpoint
evaluations) — i.e. it still means "how many MC simulations at most"; the
docstring is updated accordingly.

**`solutions` history.** Same 4 columns, one row per evaluation of `f`. Step
values differ from the old midpoints (Brent interpolation points) — allowed
by the compatibility contract.

**Known caveat.** For `survival_period` the residual is a step function (the
survival-period quantile is discrete with ~1 month granularity). Brent
degrades towards bisection on steps — i.e. worst case is no worse than
today; on the smooth maintain-balance residuals the speedup is full.
Parameter restore in `finally` is unchanged.

Expected effect: ~2–3× fewer simulations per search for all strategies.

## Stage 2 — vectorized Monte Carlo wealth engine

**What changes.** `monte_carlo_wealth` currently applies
`get_wealth_indexes_fv_with_cashflow` once per path. A new function in
`dcf_calculations` (e.g. `get_wealth_indexes_fv_with_cashflow_mc(ror_df,
cashflow_parameters, discount_rate)`) computes the whole matrix at once:

- `ror` → numpy matrix T×N; state is a balance vector `W` of length N; **a
  single loop over T months** with vector operations only:
  `W = W * (1 + R[t]) + cashflow_t + extra_cf[t]`.
- Regular cash flow per strategy (all vectorizable):
  - `fixed_amount`: scalar `amount·(1+idx)^n` at period-boundary months;
  - `fixed_percentage`: `p/periods_per_year · W_period_start` (vector of
    period-start balances);
  - `CWD`: per-path reduction factor from that path's drawdown; the drawdown
    matrix is precomputed vectorized (columnwise cumprod/cummax);
  - `VDS`: per-path state vectors (`last_withdrawal`, balance) with
    floor/ceiling via `np.clip`.
- **The "slow branch" semantics are reproduced exactly**: cash flow applied
  at period end, partial periods scaled by
  `months_local / months_in_full_period`, period boundaries precomputed from
  the index (same logic as `resample(convention="start")`). Extra
  `time_series` cash flows are discounted exactly as today
  (`discount_factors`).
- Output: a DataFrame of the same shape — same index with the prepended
  initial-investment row, same columns.

**Routing.** `monte_carlo_wealth` switches to the new engine. The old
per-column function remains for the backtest/Series path
(`wealth_index_with_assets` etc.); migrating it is a separate later change to
keep the risk surface small.

**Adjacent cleanups in `monte_carlo_wealth`:** drop the double `.copy()`
(copy only when mutating), replace the per-column
`apply(remove_negative_values)` with a vectorized mask (first non-positive
month per path via boolean cummax), vectorize `get_survival_date` (argmax
over a mask instead of apply).

**Correctness guarantee — equivalence tests:** new engine vs old on a grid:
4 strategies × frequencies (`month`, `quarter`, `half-year`, `year`, `none`)
× with/without extra cash flows × histories with partial first/last periods;
match with tight tolerance (`atol≈1e-9`). Existing `monte_carlo_*` tests must
pass unchanged (fix scenarios with `MonteCarlo.seed`).

Expected effect: one simulation gets 1–2 orders of magnitude faster on
typical parameters (mc_number=200–5000, period=30–50 years); all
`monte_carlo_*` methods benefit, not only the search.

## Stage 3 — structural shortcut solvers

**Linear solver (Indexation, CWD).** The wealth recursion without absorption
(`include_negative_values=True`) is linear in `amount`:
`W_t(m) = A_t + m·B_t`. Both matrices come from **two simulations** with the
new engine: `A` at `m=0` (extra `time_series` cash flows are included here),
and `B = (W(m₁) − A)/m₁` at a reference `m₁` (e.g. −1% of the initial
investment). Then:

- **`survival_period`**: for each path, the critical withdrawal at which the
  path survives exactly to the target date:
  `m_i = min_{t≤T_target} (thr_abs − A_t)/B_t`, where
  `thr_abs = threshold · initial_investment` (vectorized, O(T×N)). The
  answer is the percentile of the `{m_i}` distribution: the monotone coupling
  "quantile of survival periods ↔ quantile of critical withdrawals" makes
  this equivalent to the current formulation. **Zero search iterations.**
- **`maintain_balance_pv/fv`**:
  `f(m) = quantile_p(A_T + m·B_T) − start_investment` — one evaluation costs
  O(N) (microseconds); Brent over this cheap function converges instantly.
  The compound "survived to the end" condition is a vector check
  `min_t(A_t + m·B_t) > 0` on the same matrices.

**Multiplicative solver (Percentage without extra cash flows).** No
closed-form per path needed: with monthly frequency
`W(p) = W₀·cumprod(1 + r_t + p/12)`; with periodic frequency the wealth at
period ends is `W₀·Π_k(G_k + p_eff)` where the period growth factors `G_k`
are collected by **one simulation** at `p=0` and
`p_eff = p / periods_per_year · period_fraction` (the same partial-period
scaling as the current slow branch); intra-period wealth is a cumprod within
the period. Any candidate `p` is re-evaluated by vectorized
cumprods in milliseconds → Brent over the cheap re-evaluation.

**Final verification (the safety mechanism of both shortcuts).** The found
`m` is set on the parameters, **one** full simulation runs through the
standard engine, and `condition`/`error_rel` are computed by the existing
`_calculate_goal_metrics` — `Result` is always filled by the same metric as
the generic path. If verification yields `error_rel ≥ tolerance_rel`
(theoretically possible due to quantile interpolation), silently fall back to
the generic Brent path. Path divergences therefore cannot reach the result.

**`solutions`**: same 4 columns; for shortcuts the history records the cheap
Brent evaluations and the verification row.

## Testing

TDD per repository rules (RED → GREEN per stage):

- Equivalence "shortcut vs generic path": grid of
  {Indexation, CWD, Percentage} × {3 goals} × {percentiles 20/50}, fixed
  `MonteCarlo.seed`, `withdrawal_rel` matches within tolerance.
- Existing tests pass without expectation changes:
  `test_find_the_largest_withdrawals_size_converges`,
  `..._supports_cwid`, `..._supports_vds`,
  `..._rejects_target_above_period_with_tolerance`.
- The VDS test doubles as the dispatcher test (falls back to the generic
  path).
- Stage 2 equivalence tests: new engine vs old per-column implementation
  (grid described above).

## Benchmarks

A manual `timeit` script (no new dependencies, not in CI): typical
configurations mc_number=200/1000, period=30/50 years, frequencies
year/month, all three goals. Before/after numbers are recorded in each
stage's PR description.

## Risks

- **Slow-branch semantics** (partial periods, `convention="start"`, period
  fractions) must be replicated exactly in the vectorized engine — mitigated
  by the equivalence-test grid including partial first/last periods.
- **Step-function residual** for the survival goal can degrade Brent towards
  bisection — bounded by "no worse than today".
- **Quantile interpolation** differences between the per-path critical
  withdrawal distribution and the survival-period quantile — neutralized by
  the final verification + fallback.
- **Compound condition** (`wealth ≥ start` AND `sp == period`) must keep the
  residual sign consistent with the current `condition` — covered by the
  forced-negative rule in stage 1 and by equivalence tests.
