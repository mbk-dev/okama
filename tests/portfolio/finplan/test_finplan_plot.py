import matplotlib  # noqa: I001

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import okama as ok  # noqa: E402


def test_plot_draws_one_line_per_scenario_and_one_boundary_marker(
    equity_portfolio, bond_portfolio
) -> None:
    plan = ok.FinPlan(
        stages=[
            ok.FinPlanStage(equity_portfolio, period=5, name="accumulation"),
            ok.FinPlanStage(bond_portfolio, period=5, name="retirement"),
        ],
        initial_investment=10_000,
        mc_number=6,
        seed=0,
    )

    ax = plan.plot_forecast_monte_carlo()
    wealth = plan.monte_carlo_wealth(discounting="fv", include_negative_values=False)

    # One line per scenario plus one vertical marker for the single internal
    # boundary (`axvline` appends to `ax.lines` too).
    assert len(ax.lines) == plan.mc_number + 1
    assert [text.get_text() for text in ax.texts] == ["accumulation", "retirement"]

    # The boundary marker must sit at the first month of the second stage.
    stage_one_months = plan.stages[0].period_months
    boundary_line = ax.lines[plan.mc_number]
    expected_x = wealth.index[stage_one_months + 1].ordinal
    actual_x = boundary_line.get_xdata()[0]
    assert actual_x == expected_x

    plt.close("all")
