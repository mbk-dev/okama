Reading and reusing a financial plan
====================================

Calendar and cash flow alignment
--------------------------------

``FinPlan.t0`` is the last date of the first stage's portfolio. Every stage
portfolio must end in the same month. The optional ``t0`` constructor argument
checks that expected month without changing the existing positional arguments.
``plan.stage_month_indices`` returns the resolved monthly ``PeriodIndex`` for
each forecast stage, including its offset from the start of the plan.

Cash flow dictionary keys belong to the stage that owns the strategy. They are
matched against that stage's own index, rather than merged into the whole plan.
Before a forecast, keys outside ``stage_month_indices`` raise ``ValueError``
naming the stage, key and allowed range. Historical backtests validate against
the stage's actual historical window instead. Construction accepts historical
ledgers because the calculation mode is not yet known. Use a ledger appropriate
to the calculation's calendar; requesting a forecast with historical keys raises.
Keys are checked again after edits, including dictionary edits in place.

When ``discount_rate=None``, the plan logs its automatically chosen annual rate
at INFO level. It uses the first stage's inflation CAGR when available, otherwise
``settings.DEFAULT_DISCOUNT_RATE``. The effective value is readable and can be
changed through ``plan.discount_rate``.

Reading a goal month
--------------------

``monte_carlo_wealth()`` has an opening row dated one month before ``t0``.
Use month labels rather than row positions to read a goal's result.

With ``include_negative_values=False``, a scenario becomes zero at its first
non-positive balance and remains zero thereafter. Therefore a positive balance
share is **cumulative survival**: it covers all earlier goals and the current
goal, with positive wealth remaining. Spending exactly the remaining balance
counts as depletion. It is not the probability of funding that goal in isolation.

.. code-block:: python

    wealth = plan.monte_carlo_wealth(include_negative_values=False)
    previous_goal = "2035-12"
    goal_month = "2040-12"
    reached_previous = wealth.loc[previous_goal] > 0
    reached_goal = wealth.loc[goal_month] > 0
    cumulative_share = reached_goal.mean()
    conditional_share = (
        reached_goal[reached_previous].mean() if reached_previous.any() else float("nan")
    )

``conditional_share`` is survival from the previous goal through this goal
among scenarios that survived the previous goal. It still includes all intervening
cash flows and is not an independent goal probability. Raw negative wealth does
not provide that independent probability either: depleted stage balances are
floored at zero when handed to the next stage.

Verifying declared flows
------------------------

``monte_carlo_cash_flow()`` defaults to a presentation that zeroes flows when a
scenario's cumulative wealth is zero, including the withdrawal that depleted it.
For a ledger comparison, explicitly disable this masking:

.. code-block:: python

    raw = plan.monte_carlo_cash_flow(
        discounting="fv", remove_if_wealth_index_negative=False
    )
    strategy = plan.stages[0].cashflow_parameters
    # For a TimeSeriesStrategy containing nominal forecast amounts:
    # time_series_discounted_values=True preserves the amounts verbatim.
    expected = pd.Series(strategy.time_series_dic, dtype=float)
    expected.index = pd.to_datetime(expected.index).to_period("M")
    expected = expected.reindex(plan.stage_month_indices[0], fill_value=0.0)
    np.testing.assert_allclose(raw.loc[expected.index, 0], expected)

This example assumes ``import pandas as pd`` and ``import numpy as np`` and a
``TimeSeriesStrategy`` with ``time_series_discounted_values=True``. Other
strategies may also add regular or balance-dependent flows. With the flag False,
forecast dictionary amounts are compounded from the plan start by the discount
rate, so a nominal-ledger comparison must account for that transformation.

Reusing returns for a goal-size search
--------------------------------------

``draw_return_paths()`` returns one DataFrame per stage, in stage order. Each
contains monthly returns on ``stage_month_indices`` and scenario columns numbered
from zero. Repeated calls return defensive copies of the retained draw.
``run_monte_carlo(return_paths=paths)`` reapplies the current cash flow strategies
without drawing returns again. Both wealth and cash flow caches, and the metrics
that read them, then describe this pass. Supplied paths are copied and must have
the expected shape, monthly index, scenario columns, and real finite numeric
values. The same plan seed gives the same paths as a full run.

After editing a strategy, explicitly call ``run_monte_carlo`` to refresh results.
``clear_cache()`` and plan-level setters discard the retained draw as well as
results; retain the returned ``paths`` yourself to reuse them across such changes.

The following example finds the largest one-time withdrawal that leaves a
positive terminal balance in at least 90 percent of scenarios. It uses a single
stage with zero regular flows and nominal ``TimeSeriesStrategy`` amounts. The
end-date portfolio is fixed so the goal month falls within the stage's calendar.

.. code-block:: python

    import okama as ok

    portfolio = ok.Portfolio(
        ["SPY.US", "AGG.US"], weights=[0.6, 0.4], ccy="USD", last_date="2025-12"
    )
    strategy = ok.TimeSeriesStrategy(
        portfolio, time_series_dic={}, time_series_discounted_values=True
    )
    plan = ok.FinPlan(
        [ok.FinPlanStage(portfolio, period=10, cashflow_parameters=strategy)],
        initial_investment=100_000,
        discount_rate=0.03,
        mc_number=1_000,
        seed=7,
        t0="2025-12",
    )
    paths = plan.draw_return_paths()  # One draw for the entire search.
    goal_month = str(plan.stage_month_indices[0][-1])
    target = 0.90

    def success(amount: float) -> float:
        strategy.time_series_dic = {goal_month: -amount}
        plan.run_monte_carlo(return_paths=paths)
        return plan.probability_of_success()

    low, high = 0.0, 100_000.0
    if success(low) < target:
        raise ValueError("The plan misses the target even without this goal.")
    while success(high) >= target:
        high *= 2.0  # Establish a failing upper bound before bisection.
    while high - low > 100.0:
        middle = (low + high) / 2.0
        if success(middle) >= target:
            low = middle
        else:
            high = middle
    success(low)  # Leave the plan caches describing the accepted goal size.
    largest_goal = low

All probes use identical returns and differ only in the withdrawal. The bracket
and bisection rely on this example's success share decreasing as that single
withdrawal grows; arbitrary changes to several strategies need not have that
property. The result is conditional on the finite scenario sample and chosen
amount tolerance.
