from pytest import approx
import pandas as pd

from okama.settings import DEFAULT_DISCOUNT_RATE


def test_dcf_discount_rate(portfolio_dcf, portfolio_dcf_no_inflation, portfolio_dcf_discount_rate):
    assert portfolio_dcf.discount_rate == approx(0.05548, rel=1e-2)  # average inflation
    assert portfolio_dcf_no_inflation.discount_rate == DEFAULT_DISCOUNT_RATE  # no inflation
    assert portfolio_dcf_discount_rate.discount_rate == approx(0.08, abs=1e-3)  # defined discount_rate


def test_wealth_index(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.wealth_index.iloc[0, 0] == 10_000
    assert portfolio_dcf_indexation.wealth_index.iloc[-1, 0] == approx(11047.67, rel=1e-2)


def test_wealth_index_time_series_strategy(portfolio_dcf_time_series):
    assert portfolio_dcf_time_series.wealth_index.iloc[0, 0] == 1_000
    assert portfolio_dcf_time_series.wealth_index.iloc[-1, 0] == approx(5214.73, rel=1e-2)


def test_wealth_index_with_assets(portfolio_dcf_percentage):
    df = portfolio_dcf_percentage.wealth_index_with_assets
    assert df.shape == (62, 4)
    # discounted initial investments
    assert portfolio_dcf_percentage.wealth_index_with_assets.iloc[0, 0] == approx(76339.3156, rel=1e-2)
    # FV
    assert portfolio_dcf_percentage.wealth_index_with_assets.iloc[-1, 0] == approx(232477.6576, rel=1e-2)


def test_survival_period_hist(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.survival_period_hist(threshold=0) == approx(5.1, rel=1e-2)


def test_survival_date_hist(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.survival_date_hist(threshold=0) == pd.Timestamp("2020-01-31 00:00:00")


def test_initial_investment_pv(portfolio_dcf_indexation):
    assert portfolio_dcf_indexation.initial_investment_pv == approx(7633.93, rel=1e-2)


def test_initial_investment_fv(portfolio_dcf_percentage):
    assert portfolio_dcf_percentage.initial_investment_fv == approx(171594.5442, rel=1e-2)


def test_cashflow_pv(portfolio_dcf_indexation, portfolio_dcf_percentage):
    assert portfolio_dcf_indexation.cashflow_pv == approx(-1145.08, rel=1e-2)
    assert portfolio_dcf_percentage.cashflow_pv is None


def test_monte_carlo_wealth_fv(portfolio_dcf_indexation):
    df = portfolio_dcf_indexation.monte_carlo_wealth_fv
    assert df.shape == (121, 100)
    assert df.iloc[-1, :].mean() == approx(11965, rel=1e-0)


def test_monte_carlo_wealth_pv(portfolio_dcf_percentage):
    df = portfolio_dcf_percentage.monte_carlo_wealth_pv
    assert df.shape == (121, 100)
    assert df.iloc[-1, :].mean() == approx(471464, rel=1e-1)


# def test_plot_forecast_monte_carlo(portfolio_dcf_indexation):
#     data = portfolio_dcf_indexation.plot_forecast_monte_carlo(backtest=False)
#     axes_data = np.array(data)
#     expected = np.array([[0, 0.042512], [0.02, 0.159596]])
#     assert_allclose(axes_data, expected, atol=1e-2)


def test_monte_carlo_survival_period(portfolio_dcf_percentage):
    s = portfolio_dcf_percentage.monte_carlo_survival_period()
    assert s.shape == (100,)
    assert s.mean() == approx(10, rel=1e-1)


def test_find_the_largest_withdrawals_size(portfolio_dcf_indexation_small):
    r = portfolio_dcf_indexation_small.find_the_largest_withdrawals_size(
        goal="survival_period",
        target_survival_period=1,
        percentile=25,
    )
    assert r.success is True
    assert r.withdrawal_abs == approx(-833.33, rel=1e-2)
    assert r.withdrawal_rel == approx(1, rel=1e-2)
    assert r.error_rel == approx(0, abs=1e-2)
    assert isinstance(r.solutions) == pd.DataFrame
    assert r.solutions.columns.tolist() == ["withdrawal_abs", "withdrawal_rel", "error_rel", "error_rel_change"]
