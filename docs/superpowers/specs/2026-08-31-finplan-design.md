# Design: `FinPlan` — a financial plan as a sequence of portfolios

Date: 2026-08-31
Status: Approved (brainstorming), pending implementation plan

## Goal

Add a new top-level class `FinPlan`: a financial plan modelled as an ordered
sequence of stages, where each stage is a `Portfolio` with its own cash flow
strategy and its own duration. The terminal balance of stage *k* becomes the
starting balance of stage *k+1*, per Monte Carlo scenario.

The motivating case is a retirement plan: an aggressive accumulation portfolio
for N years, then a conservative decumulation portfolio for M years, with the
pension funded by whatever the accumulation stage produced.

This is the headline feature of the next major okama version.

## Public API

```python
import okama as ok

acc = ok.Portfolio(["SPY.US", "AGG.US"], weights=[0.8, 0.2], ccy="USD")
ret = ok.Portfolio(["AGG.US", "SPY.US"], weights=[0.7, 0.3], ccy="USD")

contrib = ok.IndexationStrategy(acc)          # contributions
contrib.frequency = "year"
contrib.amount = 12_000
contrib.indexation = "inflation"

pension = ok.IndexationStrategy(ret)          # withdrawals
pension.frequency = "month"
pension.amount = -3_000
pension.indexation = "inflation"

plan = ok.FinPlan(
    stages=[
        ok.FinPlanStage(acc, period=20, cashflow_parameters=contrib, name="accumulation"),
        ok.FinPlanStage(ret, period=30, cashflow_parameters=pension, name="retirement"),
    ],
    initial_investment=100_000,
    mc_number=1_000,
    seed=0,
)

plan.monte_carlo_wealth(discounting="fv")
plan.probability_of_success()
plan.balance_percentiles()
plan.plot_forecast_monte_carlo()
```

### `FinPlanStage`

A value object, not three parallel lists. Fields:

| Field | Meaning |
|---|---|
| `portfolio` | `Portfolio` instance active during the stage |
| `period` | Stage length in whole years (int ≥ 1) |
| `cashflow_parameters` | `CashFlow` strategy; its `parent` must be `portfolio` |
| `name` | Optional label used in `__repr__` and on plots |
| `distribution` | Monte Carlo distribution for this stage, default `"norm"` |
| `distribution_parameters` | Optional tuple, same semantics as `MonteCarlo` |

The distribution settings live on the stage because each stage has its own
portfolio and therefore its own return history to fit.

A stage object is what makes the v2 conditional transition (`until=...`) an
added field rather than a signature break.

### `FinPlan`

Plan-level, deliberately not per stage:

| Property | Rationale |
|---|---|
| `initial_investment` | A property of the plan's start, not of any stage |
| `discount_rate` | One rate for the whole horizon, defaults to the base portfolio's inflation CAGR (as `PortfolioDCF` does today) |
| `mc_number` | All stages must share the scenario count |
| `seed` | One plan seed; per-stage seeds are spawned from it |
| `name` | Label for the wealth column, the plot and `__repr__`; default `"plan"` |

`FinPlan` never reads `initial_investment` from any stage's cash flow strategy.
`CashFlow.__init__` always assigns `settings.DEFAULT_INITIAL_INVESTMENT`
(1000.0), so "unset" is indistinguishable from "deliberately set to 1000" and no
warning can be reliable. The single source is `FinPlan.initial_investment`;
this is stated in the class docstring and visible in `__repr__`.

## Design decisions taken during brainstorming

1. **Stage boundaries are fixed durations** in v1. Conditional transitions
   ("switch when the balance reaches X") are deferred to v2, but the data model
   is shaped to accept them.
2. **Ruin handoff is a floor at zero.** A scenario whose balance went to zero or
   below during stage *k* enters stage *k+1* with a balance of 0.
3. **One plan-level discount rate.** Per-stage rates would make stage PVs
   non-additive, since each would be expressed in different money.
4. **Indexation is anchored at the plan start.** `amount` in a later stage's
   strategy is money of the plan's `t0`, matching what `amount` already means in
   a single-portfolio forecast.
5. **v1 scope** — plan-level Monte Carlo tables, plan-level metrics, and a
   calendar-history backtest. Per-stage views are deferred to v2.
6. **The historical mode is a calendar backtest**: stages consume the common
   history window sequentially.

## Engine changes (`okama/portfolios/dcf_calculations.py`)

The public `Portfolio.dcf` API is unchanged. Exactly one function gains
parameters, all keyword-only with defaults that reproduce current behaviour, so
every existing call site and the existing equivalence tests stay valid.

### `_simulate_paths_mc(ror, cashflow_parameters, discount_rate, *, initial_balance=None, month_offset=0, task="monte_carlo")`

**`initial_balance: float | np.ndarray | None`** — replaces the hardcoded
scalar broadcast at `dcf_calculations.py:422`
(`balance = np.full(n_cols, float(initial_investment))`). An array must have
length `n_cols`. `None` keeps today's behaviour (read from
`cashflow_parameters.initial_investment`). **This parameter is the stage
connector**; no separate connector object is needed.

**`month_offset: int`** — months elapsed since the plan's `t0`. Four sites
currently derive a time exponent from the stage-local row position and would
silently restart at each boundary:

| Site | Fix |
|---|---|
| `:409` compounding of `time_series` extra cash flows | `np.arange(month_offset, month_offset + n_rows)` |
| `:429`, `:459` indexation of `amount` | `(1 + i) ** (n + period_offset)` |
| `:435`, `:467` CWD base withdrawal | same |
| `:336` `_vds_withdrawal_vector` `number_of_periods` | `n + period_offset`, otherwise VDS `min_max_annual_withdrawals` re-indexes from zero at each stage |

`period_offset = month_offset // months_in_full_period`, guarded by a
`ValueError` when `month_offset % months_in_full_period != 0`.

The division is exact, and the reason is worth recording because it is not
obvious. `_resample_slices` (`:362`) groups by **calendar** periods, so the
group boundaries do not start at `t0`. But stage durations are whole years, so
the start of every stage falls on the same month-of-year as `t0`; the number of
calendar periods between them is therefore exactly
`month_offset / months_in_full_period` (240 months → 20 years → 80 quarters).
Deriving the offset instead from the cumulative slice count of previous stages
would be wrong: a 20-year stage starting in November spans 21 annual groups
(a partial first year, 19 full ones, a partial last one), which would push the
next stage one period too far and give the shared boundary year two different
indices.

**`task: {"monte_carlo", "backtest"}`** — selects the `time_series`
compounding convention, mirroring `get_wealth_indexes_fv_with_cashflow`. This is
the only behavioural difference between the two engines and it is what lets the
historical mode reuse this function.

`get_wealth_indexes_fv_with_cashflow_mc` and `get_cash_flow_fv_mc` are **not**
modified: `FinPlan` calls `_simulate_paths_mc` directly, obtaining wealth and
cash flow from one pass, and prepends the `t0 - 1` initial row itself.

### Semantics that stay stage-local

Both are deliberate and both are pinned by tests and documented:

- **CWD drawdowns.** `:415` builds `wealth_for_dd` from the current stage's
  returns, so the peak resets at each boundary — a new portfolio has a new peak.
  Consequence: **the "two stages equal one long run" invariant does not hold for
  `CutWithdrawalsIfDrawdown`.** The golden test uses `IndexationStrategy`.
- **VDS `last_withdrawal`.** Zero on the first period of a stage, so
  floor/ceiling do not bind there — same as a freshly started portfolio.
- **The boundary splits a calendar period.** With a periodic `frequency`
  (`year`, `half-year`, `quarter`) the calendar period containing the stage
  boundary is cut in two: stage 1 ends with a partial slice and stage 2 opens
  with another, each receiving a pro-rated cash flow via the existing
  `period_fraction` scaling. Nothing is lost — the two fractions sum to one full
  period — but the payment moves to the boundary month, and for
  `PercentageStrategy` the two halves are computed off two different opening
  balances. This is the reason the golden invariant is exact only for `month`
  and `none` (see Testing).

### Ruin handling

`FinPlan` passes `np.maximum(terminal_wealth, 0)` to the next stage. Within a
stage, negative balances are retained as they are today and masked only on
output by `zero_wealth_after_first_void`.

The equivalence with one long run therefore holds at the **masked** level, not
the raw one: with withdrawals a negative balance never crosses back above zero,
so once a scenario is void it stays void in both, and the masked outputs
coincide. The raw arrays do not — the continuous run keeps compounding the
negative while the plan floors it and descends again. The Testing section splits
this into two tests accordingly.

## Monte Carlo orchestration

### Extracting the return generator

`MonteCarlo._generate_returns_ts` (`mc.py:439`) is bound to its parent: it reads
`self.ror`, derives the index from `parent.parent.last_date`, and uses
`self.seed`. Extract a module-level pure function:

```python
def generate_returns_ts(ror, distribution, distribution_parameters, n_paths, index, rng) -> pd.DataFrame
```

plus a second pure function for the resolution step it depends on:

```python
def resolve_distribution_parameters(ror, distribution, distribution_parameters) -> tuple[float, ...]
```

Splitting them keeps fitting separate from drawing, which makes the
byte-for-byte reproduction test easier to localize. The split is safe: the whole
resolution chain (`get_parameters_for_distribution`, `_get_params_for_norm` /
`_lognormal` / `_t`, and `optimize_df_for_students` with `backtesting_error`)
reads only `self.ror`, `self.distribution` and `self.distribution_parameters`.
`self.parent` appears in `mc.py` in just three places — `__repr__`, cache
clearing, and `_forecast_preparation` — none of them in the resolution path. So
`FinPlan` obtains resolved parameters per stage without touching
`stage.portfolio.dcf.mc` at all, which is the coupling this design exists to
avoid.

`MonteCarlo.get_parameters_for_distribution` and
`MonteCarlo._generate_returns_ts` become calls to these. The order of `rng`
consumption must be preserved byte-for-byte, or every existing seeded test
shifts silently. Therefore the first implementation step is a test proving the
extracted function reproduces `pf.dcf.mc.monte_carlo_returns_ts` for a fixed
seed; `FinPlan` wires up to it only after that test is green.

### The plan clock

`t0 = stages[0].portfolio.last_date`, the same anchor a single-portfolio
forecast uses (`mc.py:462`). This makes the single-stage equivalence test an
exact equality rather than an equality up to a shift. The other portfolios'
`last_date` values do not matter for Monte Carlo — their history is used only to
fit distribution parameters.

### The chaining loop

```python
seeds = np.random.SeedSequence(self.seed).spawn(len(self.stages))
t0_period = self.t0.to_period("M")            # t0 = stages[0].portfolio.last_date
balance, offset = np.full(self.mc_number, float(self.initial_investment)), 0
for stage, s in zip(self.stages, seeds):
    index = pd.period_range(t0_period + offset, periods=stage.period * 12, freq="M")
    ror = mc.generate_returns_ts(stage.portfolio.ror, stage.distribution,
                                 stage.distribution_parameters, self.mc_number,
                                 index, np.random.default_rng(s))
    wealth, cash_flow = dcf_calculations._simulate_paths_mc(
        ror, stage.cashflow_parameters, self.discount_rate,
        initial_balance=balance, month_offset=offset)
    balance = np.maximum(wealth[-1], 0.0)
    offset += stage.period * 12
```

Per-stage seeds come from `SeedSequence.spawn()`, not `seed + i`: adjacent
integer seeds produce statistically related streams.

**Stages draw independently.** Scenario *i* in stage 2 is unrelated to scenario
*i* in stage 1 beyond the handed-over balance. This is a defensible modelling
assumption — the stages have different portfolios and different distributions —
but it must be stated in the documentation: the plan does not model a prolonged
bear market spanning both stages. Correlated scenarios across stages are a
deferred item.

### Discounting

Applied once to the concatenated frame. `discount_monthly_cash_flow`
(`:550`) raises the factor to the power of the row position; the plan's frame
starts at `t0`, so the exponent is correct by construction. Per-stage
discounting would be wrong exactly here.

### Validation at construction

- non-empty `stages`; each `period` an int ≥ 1;
- all stage portfolios share the same `currency` — otherwise balances in
  different currencies would be chained;
- `stage.cashflow_parameters.parent is stage.portfolio` — otherwise
  `indexation="inflation"` silently resolves against a foreign portfolio's
  inflation.

Not validated, deliberately: the rule "the forecast period should not exceed
half the portfolio history" appears only as prose in the
`monte_carlo_returns_ts` docstring (`mc.py:394`). `MonteCarlo.period`'s setter
(`mc.py:172`) checks nothing but integrality, so there is no guard for `FinPlan`
to bypass, and adding one here would make the plan stricter than the rest of the
library rather than consistent with it.

### Caching

`FinPlan` caches the concatenated FV tables. Plan-level setters (`mc_number`,
`seed`, `discount_rate`, `initial_investment`) clear the cache. In-place edits
to a stage's strategy (`pension.amount = -4000`) cannot be intercepted; a public
`FinPlan.clear_cache()` covers that case and the limitation is documented.

## Historical mode (calendar backtest)

`plan.wealth_index(discounting, include_negative_values=False)` and
`plan.cash_flow_ts(discounting)` — names in parity with `PortfolioDCF`.

**Window.** The common history is the intersection of the stage portfolios'
histories: `first = max(pf.first_date)`, `last = min(pf.last_date)`. The plan
needs `sum(period) * 12` months. If the history is shorter, raise `ValueError`
naming both numbers and the available date range, rather than truncating the
last stage.

**Window placement.** Anchored at the start: the plan begins at
`max(first_date)` and uses the longest available run. An optional `first_date=`
argument shifts the window. Anchoring at the end would silently discard the
oldest history, which is usually the reason for running a backtest at all.

**Engine.** The same `_simulate_paths_mc` with a single-column `ror` and
`task="backtest"`, chained identically (`initial_balance` from the previous
stage, `month_offset` from `t0`, floor at zero on the boundary). No second
simulation code path and no per-path loop.

This reuse is only sound because GitHub issues #81 and #82 are fixed in the
per-path reference: `dcf_calculations.py:107` applies the month's return in the
first month of a period containing extra cash flows, and `:141` updates
`last_regular_cash_flow`. Had those quirks survived, a plan backtest would
diverge from `pf.dcf.wealth_index()` on identical inputs. A test pins the
absence of divergence (see Testing).

**Accumulated inflation column.** `Portfolio.dcf.wealth_index` returns it
alongside the wealth index (`dcf.py:255`). All stage portfolios share a
currency, so the inflation series is one and the same; it is taken from the base
portfolio over the plan window. If the portfolios were built with
`inflation=False`, the column is absent, as today.

**Deferred:** rolling windows (running the plan from every possible start date
and reporting the distribution of outcomes).

## Metrics and plotting

| Method | Returns | Parity with |
|---|---|---|
| `monte_carlo_wealth(discounting, include_negative_values=True)` | DataFrame T×N | `dcf.py:560` |
| `monte_carlo_cash_flow(discounting, remove_if_wealth_index_negative=True)` | DataFrame T×N | `dcf.py:625` |
| `monte_carlo_survival_period(threshold=0)` | Series per scenario, years to first void, capped at the plan horizon | `dcf.py:770` |
| `monte_carlo_irr()` | Series per scenario, flows from `t0` (initial investment as a negative flow) to the plan end | `dcf.py:819` |
| `wealth_index(...)`, `cash_flow_ts(...)` | Historical backtest | `dcf.py:206`, `dcf.py:274` |

Two metrics exist only because of staging:

- **`probability_of_success(threshold=0)`** — the share of scenarios that reach
  the end of the plan with a balance above the threshold. For a retirement plan
  this is the headline number.
- **`balance_percentiles(percentiles=(10, 50, 90))`** — a DataFrame of
  "stage boundary × percentile": the balance at the accumulation-to-retirement
  transition and at the plan end. This is the distribution behind the informal
  "terminal balance of stage 1".

**`plot_forecast_monte_carlo(figsize=None)`** draws the scenario cloud with
vertical lines at the stage boundaries and `FinPlanStage.name` labels over the
segments. Without them a multi-stage chart gives no hint of where the regime
changes.

It deliberately drops the `backtest` parameter that `dcf.py:689` carries.
`_plot_mc_with_backtest` (`dcf.py:744`) prepends history by temporarily
replacing the portfolio's cash flow strategy, rescaling its
`initial_investment` to the backtest's terminal value and restoring it in a
`finally` block. For a plan that mutate-and-restore dance would have to reach
into a stage's strategy — exactly the coupling this design avoids — and the
"history" it would prepend is the base portfolio's alone, not the plan's. The
plan's own history has a separate and better-defined method, `wealth_index()`,
which the user can plot directly.

**`__repr__`** — a table of stages (name, portfolio symbol, years, strategy)
plus horizon, `mc_number` and discount rate, following `PortfolioDCF.__repr__`.

## Testing

All work follows the repository's TDD requirement (RED → verify RED → GREEN →
verify GREEN → REFACTOR).

### The golden invariant is stated at the engine level

"Two stages equal one long run" cannot be tested through `FinPlan` with a shared
seed: the plan draws each stage from a separate `SeedSequence.spawn()` stream
and cannot coincide with one continuous draw by construction. The invariant is
therefore pinned on the pure function, with no seeds involved:

```
_simulate_paths_mc(ror_full, strategy, rate)                     # T = a + b
  ==
_simulate_paths_mc(ror_full[:a], strategy, rate)                 # stage 1
  → _simulate_paths_mc(ror_full[a:], strategy, rate,
                       initial_balance=<stage 1 terminal>, month_offset=a)
```

Fully deterministic, and it catches all four positional-time sites at once.

**The invariant has two documented limits, and the test must respect both.**

*Frequency.* It is an exact equality only for `frequency` in `{"month",
"none"}`, where the cash flow is applied every month and no calendar grouping is
involved. For `year` / `half-year` / `quarter` the boundary splits a calendar
period into two pro-rated slices (see "Semantics that stay stage-local"), so the
raw trajectories legitimately differ around the boundary. The periodic case is
covered by a weaker assertion instead: the cash flow the plan applies across the
two boundary slices sums to the single full-period amount the continuous run
applies, and the trajectories coincide again from the next full period onward.

*Ruin.* `_simulate_paths_mc` returns **unmasked** wealth — the flooring lives in
`zero_wealth_after_first_void`, in the wrapper. A scenario that ruins inside
stage 1 therefore diverges at the raw level: the continuous run keeps
compounding a negative balance, while the plan floors it to zero and descends
again on a different trajectory. So the raw-array invariant test must use
parameters under which no path ruins (contributions-only first stage, or a large
initial balance). The flooring is covered by its own test, below.

On top of it, a `FinPlan`-level wiring test for a **single-stage** plan. It
cannot compare against `pf.dcf.monte_carlo_wealth()` directly: the plan draws
each stage from `SeedSequence(seed).spawn(...)`, which differs from
`default_rng(seed)` even for one stage, so a one-stage plan is *similar to* but
not bit-identical with a portfolio forecast — a property worth documenting
rather than engineering around. The test instead reproduces the stage's draw
with the same spawned seed and asserts the plan equals
`get_wealth_indexes_fv_with_cashflow_mc` on it, which pins the `t0` anchor, the
prepended initial row, and the plan-level `initial_investment`.

The stage handoff gets a deterministic test of its own: give the second stage
`distribution="norm"` with `distribution_parameters=(0.0, 0.0)`, so its returns
are exactly zero, and its opening row must equal the first stage's terminal row
clipped at zero.

### Remaining tests

- **Generator extraction, first of all:** `generate_returns_ts` reproduces
  `pf.dcf.mc.monte_carlo_returns_ts` for a fixed seed. `FinPlan` does not use it
  until this is green.
- **Indexation from `t0`:** the first retirement withdrawal equals
  `amount * (1 + i) ** (periods since plan start)`, not `amount`.
- **`time_series` extra cash flow** in year 25 of the plan is compounded from
  `t0`.
- **PV:** the discount exponent is continuous across the stage boundary.
- **Handoff:** a scenario ruined in stage 1 enters stage 2 with zero, and
  `zero_wealth_after_first_void` applied to the plan matches the same masking
  applied to the continuous run — the equality that the raw invariant test
  cannot make.
- **Stage-local semantics**, as an agreed contract rather than an accident: the
  CWD drawdown peak resets at the boundary; VDS `last_withdrawal` is zero on the
  first period of a stage.
- **Backtest:** a single-stage `FinPlan.wealth_index()` equals
  `pf.dcf.wealth_index()` over the same window.
- **Validation:** mixed currencies, a strategy whose `parent` is another
  portfolio, insufficient history, an indivisible `month_offset` — each with its
  own message.
- **Reproducibility:** the same seed gives the same result; different seeds give
  different results.

## File layout

- `okama/portfolios/finplan.py` — new module holding `FinPlan` and
  `FinPlanStage`. `core.py` is already 1778 lines and `dcf.py` 1241; neither
  should grow.
- `okama/__init__.py` — export `FinPlan`, `FinPlanStage`.
- `tests/portfolio/finplan/` — mirrors the source layout.
- `docs/` — an autosummary page for the new classes.
- `CHANGELOG.md` — entry for the major release.
- A `/examples` notebook follows as a separate task, after implementation, since
  it needs a live data run.

## Explicitly out of scope for v1

Named here so they are understood as deferred rather than forgotten:

- per-stage views, `plan.stages[i].monte_carlo_wealth()` — v2;
- conditional stage transitions (`until=`) — v2, a field on `FinPlanStage`;
- a **plan-level solver**, the analogue of `find_the_largest_withdrawals_size`
  (`dcf.py:870`): "the largest sustainable pension given 20 years of
  accumulation". The obvious next step, but the existing method is built on
  mutating and restoring portfolio state
  (`_restore_cashflow_parameters_from_backup`, `dcf.py:1185`) and would have to
  be rewritten for a plan, not wrapped;
- rolling-window backtests;
- correlated scenarios across stages.

## Unrelated fix noticed along the way

The module docstring of `tests/portfolio/dcf/test_mc_engine_equivalence.py:1`
still claims the test pins "known quirks of the reference implementation (no
return in the first month of a period with extra cash flows; VDS
last_withdrawal pinned to 0)". Both quirks were fixed (issues #81, #82) and no
longer exist in the code, so the docstring misleads the next reader. To be
corrected either as part of this work or separately.
