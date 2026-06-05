# Stage 1 — Brent root finding in `find_the_largest_withdrawals_size` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rolled bisection in `PortfolioDCF.find_the_largest_withdrawals_size` with `scipy.optimize.brentq` over a signed residual, cutting the number of Monte Carlo simulations per search ~2–3× while preserving the public signature and `Result` shape.

**Architecture:** The public method keeps its signature. Internally, a closure `_evaluate(m)` runs one MC simulation per *fresh* withdrawal parameter `m`, records the attempt in the `solutions` DataFrame, raises `_SolverConverged` when `error_rel < tolerance_rel` (preserving the function-value stopping contract) and `_SolverBudgetExhausted` when `iter_max` evaluations are spent. The two ends of `withdrawals_range` are evaluated first: if the root lies outside the range, the search stops after 1–2 evaluations instead of burning the whole budget. A memo dict makes brentq's re-evaluation of the endpoints free. The dead bisection helpers are deleted.

**Tech Stack:** Python, scipy (`optimize.brentq`, already a dependency, same import style as `okama/portfolios/dcf_calculations.py:7`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-06-04-find-largest-withdrawals-speedup-design.md` (this plan covers Stage 1 only; Stages 2–3 get their own plans after this one lands).

**Branch:** work happens on `dev` (current development branch).

---

## Context for the engineer (read first)

- The method under change: `okama/portfolios/dcf.py` — `find_the_largest_withdrawals_size` (starts at line ~892; the executable body is after the long docstring, lines ~1013–1090).
- Sign conventions: withdrawal parameters are **negative** (`amount` or `percentage`). `_get_withdrawal_bounds` (`dcf.py:1233`) returns `(expected_min_withdrawal, expected_max_withdrawal)` where `expected_min_withdrawal` (the *smallest* withdrawal) is numerically the **larger** value (closer to 0) and `expected_max_withdrawal` (the *largest* withdrawal) is numerically the **smaller** (more negative). For `brentq(f, a, b)` with `a < b`: `a = expected_max_withdrawal`, `b = expected_min_withdrawal`.
- The signed residual is `error_rel if condition else -error_rel`, where `(condition, error_rel)` come from the existing `_calculate_goal_metrics` (`dcf.py:1130`). For the survival goal this equals `(sp_quantile − target)/target` exactly; for maintain-balance goals the compound condition (`wealth ≥ start` AND `sp == mc.period`) forces the residual negative when the portfolio dies early, per the spec. The residual is monotonically increasing in `m`.
- Setting the parameter (`_set_main_parameter`, `dcf.py:1268`) clears the wealth cache, so every fresh `m` costs one full MC simulation. The random scenario draw is cached across evaluations (`mc.py:436`), so the objective is deterministic.
- `Result` dataclass: `okama/common/solver.py:7` (frozen; fields `success`, `withdrawal_abs`, `withdrawal_rel`, `error_rel`, `solutions`).
- Tests run via `poetry run pytest -q`. Lint via `poetry run ruff check .`.
- Existing tests that must keep passing without expectation changes:
  - `tests/portfolio/dcf/test_cashflow.py::test_find_the_largest_withdrawals_size_converges`
  - `tests/portfolio/dcf/test_cashflow.py::test_find_the_largest_withdrawals_size_rejects_target_above_period_with_tolerance`
  - `tests/portfolio/dcf/test_advanced_cashflow.py::test_find_the_largest_withdrawals_size_supports_cwid`
  - `tests/portfolio/dcf/test_advanced_cashflow.py::test_find_the_largest_withdrawals_size_supports_vds`
- Validation gotcha for new tests: `_validate_parameters` (`dcf.py:1092`) requires `target_survival_period <= mc.period * (1 - tolerance_rel)` **for every goal**, including maintain-balance. Pick `target_survival_period` accordingly.
- The `synthetic_env` fixture (global, `tests/conftest.py:92`) mocks asset data; portfolios are built as in `tests/portfolio/dcf/test_advanced_cashflow.py:8`.

## File structure

- Create: `tests/portfolio/dcf/test_find_largest_withdrawals.py` — solver-behavior tests (early exit, budget, brentq success path).
- Modify: `okama/portfolios/dcf.py` — new solver internals; delete `_update_parameter` (lines ~1118–1128) and `_bisection_iteration` (lines ~1184–1219); docstring update.
- Modify: `CHANGELOG.md` — Unreleased entry.

---

### Task 1: Write the failing solver-behavior tests

**Files:**
- Create: `tests/portfolio/dcf/test_find_largest_withdrawals.py`

- [ ] **Step 1: Create the test file with two RED driver tests and two guard tests**

```python
"""Behavior of the root-finding solver in PortfolioDCF.find_the_largest_withdrawals_size.

The solver contract: evaluate the ends of withdrawals_range first and stop
early when the root lies outside the range; stop as soon as
error_rel < tolerance_rel; never spend more than iter_max Monte Carlo
simulations; always restore the cash flow parameters.
"""

import pytest  # noqa: I001
import okama as ok


@pytest.fixture()
def dcf_solver(synthetic_env):
    """Single-asset portfolio with IndexationStrategy and a small, seeded Monte Carlo."""
    pf = ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))
    ind = ok.IndexationStrategy(pf)
    ind.initial_investment = 10_000
    ind.frequency = "year"
    ind.amount = -1_200
    ind.indexation = 0.05
    pf.dcf.cashflow_parameters = ind
    pf.dcf.set_mc_parameters(distribution="norm", period=5, mc_number=16, seed=0)
    return pf.dcf


def test_stops_after_one_evaluation_when_largest_withdrawal_sustains_goal(dcf_solver) -> None:
    # Max withdrawal is 0.1% of 10_000 = 10 per year: the portfolio trivially
    # survives 3 of 5 years, so the root lies outside withdrawals_range.
    res = dcf_solver.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.0, 0.001),
        target_survival_period=3,
        percentile=50,
        threshold=0,
        tolerance_rel=0.25,
        iter_max=10,
    )
    # The solver must detect this at the range edge with a single simulation
    # instead of burning the whole iteration budget at the same point.
    assert res.solutions.shape[0] == 1
    assert res.success is False
    assert res.withdrawal_rel == pytest.approx(0.001)
    assert dcf_solver.cashflow_parameters.amount == pytest.approx(-1_200)


def test_stops_after_two_evaluations_when_no_withdrawal_in_range_survives(dcf_solver) -> None:
    # Min withdrawal is 50% of 10_000 = 5_000 per year: the portfolio dies in
    # ~2 years, far below the 4-year target, for every value in the range.
    res = dcf_solver.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.5, 1.0),
        target_survival_period=4,
        percentile=50,
        threshold=0,
        tolerance_rel=0.1,
        iter_max=10,
    )
    # Both range ends fail the goal: two simulations are enough to prove
    # there is no root inside the range.
    assert res.solutions.shape[0] == 2
    assert res.success is False
    assert dcf_solver.cashflow_parameters.amount == pytest.approx(-1_200)


def test_budget_limits_number_of_evaluations(dcf_solver) -> None:
    # Guard: with a practically unreachable tolerance the solver must stop
    # at iter_max evaluations (may already pass before the change).
    res = dcf_solver.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.0, 1.0),
        target_survival_period=4,
        percentile=50,
        threshold=0,
        tolerance_rel=0.0001,
        iter_max=4,
    )
    assert res.solutions.shape[0] <= 4
    assert dcf_solver.cashflow_parameters.amount == pytest.approx(-1_200)


def test_converges_on_smooth_maintain_balance_goal(dcf_solver) -> None:
    # Guard: the maintain-balance residual is smooth, the root is inside the
    # range, and the solver must converge within the budget.
    res = dcf_solver.find_the_largest_withdrawals_size(
        goal="maintain_balance_fv",
        withdrawals_range=(0.0, 1.0),
        target_survival_period=3,
        percentile=50,
        threshold=0,
        tolerance_rel=0.01,
        iter_max=12,
    )
    assert res.success is True
    assert res.error_rel < 0.01
    assert res.solutions.shape[0] <= 12
    assert dcf_solver.cashflow_parameters.amount == pytest.approx(-1_200)
```

- [ ] **Step 2: Run the new tests and verify the two driver tests fail for the right reason**

Run: `poetry run pytest tests/portfolio/dcf/test_find_largest_withdrawals.py -v`

Expected:
- `test_stops_after_one_evaluation_when_largest_withdrawal_sustains_goal` — **FAIL** with `AssertionError` on `res.solutions.shape[0] == 1` (current bisection records 10 rows: it stalls at the range edge because the bisection delta is zero there).
- `test_stops_after_two_evaluations_when_no_withdrawal_in_range_survives` — **FAIL** with `AssertionError` on `res.solutions.shape[0] == 2` (current bisection spends all 10 iterations).
- The two guard tests may PASS or FAIL — note their status; they must pass after the implementation.

If a driver test fails with `ImportError`/`TypeError` instead of `AssertionError`, fix the test, not the code.

- [ ] **Step 3: Commit the RED tests**

```bash
git add tests/portfolio/dcf/test_find_largest_withdrawals.py
git commit -m "test(dcf): add solver-behavior tests for find_the_largest_withdrawals_size

Two driver tests encode the new early-exit contract (root outside
withdrawals_range detected at the range ends in 1-2 evaluations) and are
RED against the current bisection; two guard tests pin the budget and the
smooth maintain-balance convergence."
```

---

### Task 2: Implement the Brent-based solver

**Files:**
- Modify: `okama/portfolios/dcf.py`

- [ ] **Step 1: Add the scipy import**

In the import block at the top of `okama/portfolios/dcf.py` (after `import pandas as pd`, line ~8), add (same style as `okama/portfolios/dcf_calculations.py:7`):

```python
from scipy import optimize
```

- [ ] **Step 2: Add the internal solver-control exceptions**

Right after the `logger = logging.getLogger(...)` line near the top of `okama/portfolios/dcf.py` (before the `PortfolioDCF` class), add:

```python
class _SolverConverged(Exception):
    """Internal signal: error_rel dropped below tolerance_rel during the root search."""


class _SolverBudgetExhausted(Exception):
    """Internal signal: the root search spent all iter_max objective evaluations."""
```

- [ ] **Step 3: Replace the executable body of `find_the_largest_withdrawals_size`**

Keep the signature and the docstring untouched for now (docstring is Task 3). Replace everything from the `# Validation` comment (line ~1013) down to and including the `finally:` block (line ~1090) with:

```python
        # Validation
        self._validate_parameters(withdrawals_range, target_survival_period, percentile, threshold, tolerance_rel)

        # Initialization
        backup_obj = self.cashflow_parameters
        backup_main_parameter = self._get_main_parameter()
        start_investment = self.cashflow_parameters.initial_investment
        expected_min_withdrawal, expected_max_withdrawal = self._get_withdrawal_bounds(
            withdrawals_range, start_investment
        )
        # Both bounds are negative. The largest withdrawal (expected_max_withdrawal)
        # is numerically the smallest value, which makes it the left end for brentq.
        lower_m, upper_m = expected_max_withdrawal, expected_min_withdrawal

        solutions = pd.DataFrame(columns=["withdrawal_abs", "withdrawal_rel", "error_rel", "error_rel_change"])
        residual_cache: dict[float, float] = {}

        def _evaluate(m: float) -> float:
            """Signed residual of the goal at withdrawal parameter `m` (one MC simulation).

            Positive when the goal is met (the withdrawal can be increased),
            negative otherwise. Every fresh evaluation is recorded in `solutions`;
            repeated points are served from the cache without spending budget.
            """
            if m in residual_cache:
                return residual_cache[m]
            iteration = solutions.shape[0]
            if iteration >= iter_max:
                raise _SolverBudgetExhausted
            self._set_main_parameter(m)
            condition, error_rel = self._calculate_goal_metrics(
                goal, percentile, threshold, start_investment, target_survival_period
            )
            withdrawal_abs, withdrawal_rel = self._calculate_withdrawal_metrics(m, start_investment)
            solutions.at[iteration, "withdrawal_abs"] = withdrawal_abs
            solutions.at[iteration, "withdrawal_rel"] = withdrawal_rel
            solutions.at[iteration, "error_rel"] = error_rel
            gradient = (
                solutions.at[iteration, "error_rel"] - solutions.at[iteration - 1, "error_rel"]
                if iteration > 0
                else 0
            )
            solutions.at[iteration, "error_rel_change"] = gradient
            logger.info(f"Evaluation {iteration}: m={m:.6f}, error_rel={error_rel:.3f}, gradient={gradient:.3f}")
            if error_rel < tolerance_rel:
                raise _SolverConverged
            residual = error_rel if condition else -error_rel
            residual_cache[m] = residual
            return residual

        try:
            if _evaluate(lower_m) > 0:
                # Even the largest allowed withdrawal sustains the goal:
                # the root lies outside withdrawals_range.
                return self._best_attempt_result(solutions)
            if _evaluate(upper_m) < 0:
                # Even the smallest allowed withdrawal fails the goal:
                # there is no root inside withdrawals_range.
                return self._best_attempt_result(solutions)
            xtol = max(abs(upper_m - lower_m) * 1e-6, 1e-12)
            optimize.brentq(_evaluate, lower_m, upper_m, xtol=xtol, maxiter=iter_max)
            # brentq located the sign change with xtol precision, but error_rel
            # never dropped below tolerance_rel (e.g. a steep step in the
            # survival-period goal): report the best attempt.
            return self._best_attempt_result(solutions)
        except _SolverConverged:
            last = solutions.iloc[-1]
            logger.info(
                f"Solution found: {last['withdrawal_abs']:.2f} or {last['withdrawal_rel'] * 100:.2f}% "
                f"after {solutions.shape[0]} evaluations."
            )
            return Result(
                success=True,
                withdrawal_abs=float(last["withdrawal_abs"]),
                withdrawal_rel=float(last["withdrawal_rel"]),
                error_rel=float(last["error_rel"]),
                solutions=solutions,
            )
        except _SolverBudgetExhausted:
            return self._best_attempt_result(solutions)
        finally:
            self._restore_cashflow_parameters_from_backup(backup_obj, backup_main_parameter)
```

- [ ] **Step 4: Add the `_best_attempt_result` helper method**

Add to `PortfolioDCF` right after `find_the_largest_withdrawals_size` (i.e. before `_validate_parameters`, line ~1092):

```python
    def _best_attempt_result(self, solutions: pd.DataFrame) -> Result:
        """Build a failure Result from the recorded attempt with the smallest error."""
        best_idx = solutions["error_rel"].idxmin()
        best_result = solutions.loc[best_idx]
        logger.warning(
            f"Solution not found after {solutions.shape[0]} evaluations. "
            f"Best withdrawal: {best_result['withdrawal_abs']:.2f} ({best_result['withdrawal_rel'] * 100:.2f}%) "
            f"with error: {best_result['error_rel'] * 100:.2f}%"
        )
        return Result(
            success=False,
            withdrawal_abs=float(best_result["withdrawal_abs"]),
            withdrawal_rel=float(best_result["withdrawal_rel"]),
            error_rel=float(best_result["error_rel"]),
            solutions=solutions,
        )
```

- [ ] **Step 5: Delete the dead bisection helpers**

Delete two now-unused methods from `PortfolioDCF` (verify exact locations first — line numbers may have shifted):

- `_update_parameter` (was `dcf.py:1118-1128`)
- `_bisection_iteration` (was `dcf.py:1184-1219`)

Then verify nothing references them:

Run: `grep -rn "_bisection_iteration\|_update_parameter" okama/ tests/ docs/ --include="*.py"`
Expected: no output.

- [ ] **Step 6: Run the new tests — all four must pass**

Run: `poetry run pytest tests/portfolio/dcf/test_find_largest_withdrawals.py -v`
Expected: 4 passed.

If `test_converges_on_smooth_maintain_balance_goal` fails because convergence needs more evaluations, do NOT loosen the implementation; check the residual sign logic first (a sign error makes brentq bracket the wrong region).

- [ ] **Step 7: Run the full dcf test package**

Run: `poetry run pytest tests/portfolio/dcf/ -q`
Expected: all pass, no warnings/errors in the summary. The four pre-existing solver tests listed in the context section must pass unchanged.

- [ ] **Step 8: Commit**

```bash
git add okama/portfolios/dcf.py
git commit -m "perf(dcf): replace bisection with Brent root finding in find_the_largest_withdrawals_size

Evaluate the ends of withdrawals_range first and exit early when the root
is outside the range; run scipy.optimize.brentq on the signed goal residual
otherwise. Stopping contract is unchanged (error_rel < tolerance_rel);
iter_max now caps objective evaluations including the two endpoint checks.
A memo dict makes brentq's endpoint re-evaluations free. Dead bisection
helpers (_bisection_iteration, _update_parameter) removed."
```

---

### Task 3: Update the docstring and CHANGELOG

**Files:**
- Modify: `okama/portfolios/dcf.py` (docstring of `find_the_largest_withdrawals_size`)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update the algorithm sentence in the docstring**

In the docstring of `find_the_largest_withdrawals_size`, replace:

```
        The algorithm uses the bisection method to find the largest withdrawal size.
```

with:

```
        The algorithm evaluates the goal at both ends of `withdrawals_range` first
        and stops early if the solution lies outside the range; otherwise it finds
        the withdrawal size with Brent's method (`scipy.optimize.brentq`) over the
        cached set of Monte Carlo scenarios.
```

- [ ] **Step 2: Update the `iter_max` parameter description**

Replace:

```
        iter_max : int, default 20
            The maximum number of iterations to find the solution.
```

with:

```
        iter_max : int, default 20
            The maximum number of objective evaluations (each runs one Monte Carlo
            simulation), including the two evaluations at the ends of `withdrawals_range`.
```

- [ ] **Step 3: Annotate the `solutions` example in the docstring**

The `>>> res.solutions` example table shows historical bisection midpoints. Replace the sentence introducing it:

```
        If the solution was not found, it is still possible to see the intermediate steps.
```

with:

```
        If the solution was not found, it is still possible to see the intermediate steps.
        The intermediate withdrawal values depend on the root-finding steps
        (the values below are illustrative).
```

- [ ] **Step 4: Add the CHANGELOG entry**

In `CHANGELOG.md`, under `## [Unreleased]`, add a `### Changed` section if absent and append (keep the existing Keep-a-Changelog style):

```markdown
### Changed
- `Portfolio.dcf.find_the_largest_withdrawals_size()` is faster: the bisection
  search is replaced with Brent's method (`scipy.optimize.brentq`) on a signed
  goal residual, and both ends of `withdrawals_range` are checked first so the
  solver exits after 1–2 Monte Carlo simulations when the solution lies outside
  the range. The public signature, the `Result` shape and the stopping rule
  (`error_rel < tolerance_rel`) are unchanged; `iter_max` now caps objective
  evaluations (Monte Carlo simulations), including the two range-end checks, so
  the history of intermediate attempts in `Result.solutions` differs from the
  former bisection midpoints.
```

- [ ] **Step 5: Commit**

```bash
git add okama/portfolios/dcf.py CHANGELOG.md
git commit -m "docs(dcf): document Brent-based solver in find_the_largest_withdrawals_size"
```

---

### Task 4: Lint and full test suite

**Files:** none (verification only)

- [ ] **Step 1: Run ruff**

Run: `poetry run ruff check .`
Expected: `All checks passed!`

If `find_the_largest_withdrawals_size` is flagged with C901 (complexity) after the rewrite, add `# noqa: C901` on its `def` line with the rationale comment `# noqa: C901  # solver orchestration: validation + bracketing + result assembly`, per the repo rule allowing targeted suppression instead of risky restructuring.

- [ ] **Step 2: Run the full test suite**

Run: `poetry run pytest -q`
Expected: all tests pass, clean output.

- [ ] **Step 3: Commit (only if lint fixes were needed)**

```bash
git add -u
git commit -m "style(dcf): satisfy ruff after solver rewrite"
```

---

## Out of scope (later stages)

- Stage 2 (vectorized MC wealth engine) and Stage 3 (structural shortcut solvers + `_withdrawal_structure` dispatcher) are planned separately after this stage lands — see the spec.
- A manual before/after benchmark belongs to the Stage 2 plan, where the per-simulation speedup makes the comparison meaningful; for this stage the PR description should cite the evaluation counts from the new tests (1–2 vs 10 in the early-exit cases).
