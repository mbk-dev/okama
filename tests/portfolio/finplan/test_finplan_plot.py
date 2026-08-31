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

    # One line per scenario plus one vertical marker for the single internal
    # boundary (`axvline` appends to `ax.lines` too).
    assert len(ax.lines) == plan.mc_number + 1
    assert [text.get_text() for text in ax.texts] == ["accumulation", "retirement"]
    plt.close("all")
