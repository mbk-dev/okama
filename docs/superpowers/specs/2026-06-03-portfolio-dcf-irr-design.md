# Design: `Portfolio.dcf.irr` — money-weighted IRR of portfolio cash flow

Date: 2026-06-03
Status: Approved (brainstorming), pending implementation plan

## Goal

Add a method that computes the **internal rate of return (IRR / money-weighted
return, MWRR)** of a portfolio's cash flow over the full historical tracked
period, honoring the configured `CashFlow` strategy.

The computation must be **fast and vectorizable**, because the same core will be
reused for Monte Carlo forecast analysis (`monte_carlo_irr`), where the input is
a large matrix of simulated cash-flow paths.

## Scope

In scope:
- `PortfolioDCF.irr() -> float` (historical, single series, **nominal** annual
  effective rate).
- A shared vectorized core `irr_of_cashflow_matrix(...)` in
  `okama/portfolios/dcf_calculations.py` that solves IRR for an `(N+1, M)`
  matrix of cash flows at once (`M=1` for the historical case).

Out of scope (separate future task, but the core is designed to serve it):
- `monte_carlo_irr` on `PortfolioDCF` — will consume the same core with the
  Monte Carlo cash-flow matrix and terminal wealth row.

## Why nominal-only (no FV/PV switch)

IRR is a **rate**, not an amount of money. "Future value / present value"
discounting is a concept for money quantities (`wealth_index`, `cash_flow_ts`),
not for a rate. Solving IRR on PV-discounted flows does not give "the same IRR
expressed in PV" — it gives the **real** rate, which differs from the nominal
rate by exactly the Fisher factor (`(1+nominal)/(1+discount_rate) - 1`), equal
only when `discount_rate = 0`. Putting a `discounting="fv"/"pv"` switch on a rate
is therefore the wrong frame.

The codebase already has the right precedent for an inflation-adjusted rate:
`Portfolio.get_cagr(period=None, real=False)` uses a `real: bool` flag, not
`fv/pv`. For now `irr()` returns the single canonical **nominal** rate; a real
variant, if needed later, should mirror `get_cagr(real=True)` rather than reuse
the `discounting=` pattern.

## Definitions

### Cash-flow vector (investor perspective)

Built from existing, tested building blocks — no strategy-specific logic lives in
IRR itself, because the strategy only shapes `cash_flow_ts`. All components are
**nominal (FV)**:

- `cf = dcf.cash_flow_ts("fv", remove_if_wealth_index_negative=True)` — monthly
  regular + extra flows; contributions `+`, withdrawals `−`.
- `terminal = dcf.wealth_index("fv", include_negative_values=False).iloc[-1]`
  — terminal balance, floored at 0 (we withdraw "only if there is anything left").
- `initial_investment = cashflow_parameters.initial_investment`.

Monthly grid `t = 0 .. N`, where `N = portfolio.ror.shape[0]`:

| t | time point | investor cash flow |
|---|------------|--------------------|
| `0` | `first_date − 1 month` | `−initial_investment` |
| `1 .. N` | `first_date .. last_date` | `−cf[t]` (withdrawal → `+`, contribution → `−`) |
| `N` | `last_date` | additionally `+ terminal` (liquidation) |

So `v[0] = −initial_investment`, `v[k] = −cf[k]` for `k = 1..N`, and
`v[N] += terminal`.

### Solving and annualization

Solve for the monthly rate `r` such that `NPV(r) = Σ_t v[t] / (1+r)^t = 0`.
Annualize to the effective annual rate (okama convention, matches
`discount_monthly_cash_flow` and `get_cagr`):

```
annualized_IRR = (1 + r)^12 − 1
```

## Correctness invariant (primary regression test)

With **no intermediate cash flows** (`frequency="none"`, no `time_series_dic`),
the vector is only `v[0] = −II` (exponent 0) and `v[N] = +TV` (exponent
`N = ror.shape[0]`), where `TV = II · Π(1+rₜ)`. Then:

```
(1+r)^N = TV/II = Π(1+rₜ)
annualized_IRR = (Π(1+rₜ))^(12/N) − 1
```

which is **literally** `helpers.Frame.get_cagr` (`Π(1+rₜ))^(_MONTHS_PER_YEAR / ror.shape[0]) − 1`).
So `irr()` equals `get_cagr` exactly (to solver tolerance), because MWRR with a
single inflow and single outflow is identically the TWR/CAGR.

**This fixes the time grid:** `initial_investment` at exponent `0`, terminal at
exponent `N = ror.shape[0]`, monthly flows in between. A ±1-month shift would
break the equality — so the CAGR test is also the regression test that the grid
is correct.

## API

```python
def irr(self) -> float:
    ...
```

- `pf.dcf.irr()` — nominal annual effective IRR.
- A **method** (not a property): it raises `AttributeError` when
  `cashflow_parameters is None` (consistent with `wealth_index`/`cash_flow_ts`),
  and a property that raises `AttributeError` would silently break `hasattr`.
- Returns `NaN` when the cash flow has no sign change (no real root).

## Vectorized core

In `okama/portfolios/dcf_calculations.py`:

```python
def irr_of_cashflow_matrix(
    cashflows: np.ndarray,        # shape (N+1, M); each column is a series
    periods_per_year: int = 12,
    guess: np.ndarray | float | None = None,
) -> np.ndarray:                  # shape (M,), annualized effective IRR per column
    ...
```

Algorithm:
- **Newton–Raphson with analytic derivative**, iterating over all `M` columns
  simultaneously as numpy matrix operations.
  - `f(r)  = Σ_t c[t] · (1+r)^(−t)`
  - `f'(r) = −Σ_t t · c[t] · (1+r)^(−t−1)`
  - Vectorized: discount matrix `(1+r)[None,:] ** (−t[:,None])`.
- **Seed** (`guess`): a cheap ratio-based monthly guess computed from the merged
  matrix — `r0 = (Σ inflows / Σ |outflows|)^(1/T) − 1`, `T = N`. For the dominant
  single-outflow/single-inflow case this equals `(TV/II)^(1/N) − 1`, i.e. the true
  monthly rate, so Newton converges in ~1 iteration. (The core only receives the
  merged cash-flow matrix, so it cannot use modified Dietz, which needs `V0`/`V_end`
  separated from the flows.)
- **Guards:** clip `1 + r > eps` each step; convergence on the Newton **step**
  (`max|Δr| < xtol`, scale-free) or `max_iter` reached.
- **Fallback:** columns not converged within `max_iter` are retried with
  `scipy.optimize.brentq` on a bracket `[-1 + eps, large]`.
- **No real root** (no sign change in the column) → `NaN`.
- Returns annualized effective rate per column: `(1 + r)^periods_per_year − 1`.

`PortfolioDCF.irr` calls this with `M = 1` and reads element `[0]`. The future
`monte_carlo_irr` feeds the `monte_carlo_cash_flow` matrix plus the terminal
`monte_carlo_wealth` row — same core, no duplicated solver.

## Edge cases

- **No sign change** (pure accumulation, `terminal = 0`, no withdrawals) → `NaN`.
- **`initial_investment = 0`** (default of `TimeSeriesStrategy`) → `v[0] = 0`;
  IRR is solved from the first non-zero flow. Valid.
- **History < 2 points** → `NaN` (cannot solve).
- **Multiple sign changes** (contributions and withdrawals interleaved) → the
  equation may have several real roots; Newton returns the root nearest the seed.
  This behavior is documented in the method docstring.
- **Depleted portfolio** (wealth index hits 0 mid-history): `terminal = 0` and
  post-depletion flows are zeroed (`remove_if_wealth_index_negative=True`); IRR is
  defined and negative.

## Testing (TDD)

New file `tests/portfolio/dcf/test_irr.py`:

Core (pure function, deterministic, no fixtures):
1. **Single in / single out** matches closed form (`-1000` at t0, `+1200` at t12,
   monthly → annual 20%).
2. **Textbook per-period rate** (`-100, 60, 60`, `periods_per_year=1`).
3. **No sign change** → `NaN`.
4. **Vectorized columns are independent** (two columns solved at once).
5. **Depleted/partial recovery** → finite, negative IRR.

`PortfolioDCF.irr` (via `synthetic_env`, mocked data):
6. **CAGR oracle (primary):** no intermediate flows → `irr()` equals
   `pf.get_cagr().iloc[-1].loc[pf.symbol]`, exact to `abs=1e-9`.
7. **Reference cross-check:** with `IndexationStrategy` and `PercentageStrategy`,
   the vectorized Newton result matches a slow reference `scipy.optimize.brentq`
   solve on the same vector to `1e-8`.
8. **`cashflow_parameters is None`** → `AttributeError`.

## Files touched

- `okama/portfolios/dcf.py` — add `PortfolioDCF.irr`.
- `okama/portfolios/dcf_calculations.py` — add `irr_of_cashflow_matrix` (+ the
  `_irr_initial_guess` and `_irr_brentq_column` helpers).
- `tests/portfolio/dcf/test_irr.py` — new test module.

No new third-party dependency: `numpy` + `scipy` (already required) are
sufficient; `numpy-financial` is intentionally not added.
