import pytest  # noqa: I001
import okama as ok


def test_stage_exposes_its_portfolio_period_and_strategy(equity_portfolio) -> None:
    strategy = ok.IndexationStrategy(equity_portfolio)
    stage = ok.FinPlanStage(equity_portfolio, period=20, cashflow_parameters=strategy, name="accumulation")

    assert stage.portfolio is equity_portfolio
    assert stage.period == 20
    assert stage.period_months == 240
    assert stage.cashflow_parameters is strategy
    assert stage.name == "accumulation"


def test_stage_creates_a_default_strategy_when_none_is_given(equity_portfolio) -> None:
    stage = ok.FinPlanStage(equity_portfolio, period=5)

    assert isinstance(stage.cashflow_parameters, ok.IndexationStrategy)
    assert stage.cashflow_parameters.parent is equity_portfolio


def test_stage_rejects_a_strategy_built_on_another_portfolio(equity_portfolio, bond_portfolio) -> None:
    foreign = ok.IndexationStrategy(bond_portfolio)

    with pytest.raises(ValueError, match="cashflow_parameters"):
        ok.FinPlanStage(equity_portfolio, period=5, cashflow_parameters=foreign)


@pytest.mark.parametrize("period", [0, -1])
def test_stage_rejects_a_non_positive_period(equity_portfolio, period) -> None:
    with pytest.raises(ValueError, match="period"):
        ok.FinPlanStage(equity_portfolio, period=period)


def test_stage_rejects_an_unknown_distribution(equity_portfolio) -> None:
    with pytest.raises(ValueError, match="distribution"):
        ok.FinPlanStage(equity_portfolio, period=5, distribution="cauchy")


def test_stage_rejects_t_distribution_with_df_below_threshold(equity_portfolio) -> None:
    """
    Student's t-distribution with df <= 2 has infinite variance.
    The validator must reject it at construction, not later.
    """
    with pytest.raises(ValueError, match="Degrees of freedom"):
        ok.FinPlanStage(
            equity_portfolio,
            period=10,
            distribution="t",
            distribution_parameters=(1.5, None, None),
        )


def test_stage_rejects_norm_distribution_with_wrong_tuple_length(equity_portfolio) -> None:
    """
    Normal distribution requires exactly 2 parameters (mu, sigma).
    Passing (0.01,) should raise at construction, not crash later inside the generator.
    """
    with pytest.raises(ValueError, match="length 2"):
        ok.FinPlanStage(
            equity_portfolio,
            period=10,
            distribution="norm",
            distribution_parameters=(0.01,),
        )
