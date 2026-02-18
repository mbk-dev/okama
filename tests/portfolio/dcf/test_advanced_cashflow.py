import pytest
import okama as ok


@pytest.fixture()
def pf_single_monthly(synthetic_env):
    """Single-asset portfolio with monthly rebalancing and mocked data."""
    return ok.Portfolio(["A.US"], ccy="USD", inflation=False, rebalancing_strategy=ok.Rebalance(period="month"))


def _configure_small_mc(dcf: ok.PortfolioDCF, period: int = 1, mc_number: int = 16) -> None:
    """Configure a small Monte Carlo setup for fast deterministic solver checks."""
    dcf.mc.distribution = "norm"
    dcf.mc.period = period
    dcf.mc.mc_number = mc_number


def test_vds_frequency_is_year_and_setter_raises(pf_single_monthly):
    vds = ok.VanguardDynamicSpending(pf_single_monthly)
    assert vds.frequency == "year"
    with pytest.raises(AttributeError, match=r"frequency.*year"):
        vds.frequency = "month"


def test_vds_calculate_withdrawal_size_limits(pf_single_monthly):
    vds = ok.VanguardDynamicSpending(
        pf_single_monthly,
        percentage=-0.10,
        min_max_annual_withdrawals=(500.0, 900.0),
        adjust_min_max=False,
        floor_ceiling=(-0.10, 0.10),
        adjust_floor_ceiling=False,
        indexation=0.0,
    )
    # Cap by ceiling when percentage-based withdrawal is too high.
    withdrawal = vds._calculate_withdrawal_size(last_withdrawal=-800.0, balance=10_000.0, number_of_periods=0)
    assert withdrawal == pytest.approx(-880.0)
    # Enforce floor when percentage-based withdrawal is too low.
    withdrawal_low = vds._calculate_withdrawal_size(last_withdrawal=-800.0, balance=4_000.0, number_of_periods=0)
    assert withdrawal_low == pytest.approx(-720.0)


def test_vds_cash_flow_ts_yearly_entries(pf_single_monthly):
    vds = ok.VanguardDynamicSpending(
        pf_single_monthly,
        percentage=-0.06,
        min_max_annual_withdrawals=(200.0, 2_000.0),
        floor_ceiling=(-0.10, 0.10),
        adjust_min_max=False,
        adjust_floor_ceiling=False,
        indexation=0.0,
    )
    pf_single_monthly.dcf.cashflow_parameters = vds
    cfts = pf_single_monthly.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)
    non_zero = cfts[cfts != 0]
    # With 24 months of data and yearly frequency, there should be two cash flow entries.
    assert len(non_zero) == 2
    assert (non_zero <= 0).all()


def test_cwid_calculate_withdrawal_size_reduction(pf_single_monthly):
    cwid = ok.CutWithdrawalsIfDrawdown(
        pf_single_monthly,
        amount=-1000.0,
        indexation=0.0,
        crash_threshold_reduction=[(0.20, 0.40), (0.50, 1.0)],
    )
    assert cwid._calculate_withdrawal_size(drawdown=-0.25, withdrawal_without_drawdowns=-1000.0) == pytest.approx(-600.0)
    assert cwid._calculate_withdrawal_size(drawdown=-0.55, withdrawal_without_drawdowns=-1000.0) == pytest.approx(0.0)


def test_cwid_crash_threshold_reduction_validation(pf_single_monthly):
    cwid = ok.CutWithdrawalsIfDrawdown(pf_single_monthly, amount=-1000.0, indexation=0.0)
    with pytest.raises(ValueError, match=r"threshold"):
        cwid.crash_threshold_reduction = [(0.0, 0.4)]
    with pytest.raises(ValueError, match=r"reductiuon"):
        cwid.crash_threshold_reduction = [(0.2, 1.1)]


def test_cwid_cash_flow_ts_yearly_entries(pf_single_monthly):
    cwid = ok.CutWithdrawalsIfDrawdown(
        pf_single_monthly,
        amount=-1000.0,
        indexation=0.0,
        crash_threshold_reduction=[(0.10, 0.30)],
    )
    pf_single_monthly.dcf.cashflow_parameters = cwid
    cfts = pf_single_monthly.dcf.cash_flow_ts(discounting="fv", remove_if_wealth_index_negative=False)
    non_zero = cfts[cfts != 0]
    assert len(non_zero) == 2
    assert (non_zero <= 0).all()
    assert non_zero.abs().max() <= 1000.0
    

def test_vds_percentage_validation_positive_assignment_raises(pf_single_monthly):
    """VDS should raise an error if a positive percentage is assigned."""
    vds = ok.VanguardDynamicSpending(pf_single_monthly)
    with pytest.raises(ValueError, match=r"Percentage must be negative or zero"):
        vds.percentage = 0.1


def test_find_the_largest_withdrawals_size_supports_cwid(pf_single_monthly) -> None:
    cwid = ok.CutWithdrawalsIfDrawdown(
        pf_single_monthly,
        initial_investment=10_000.0,
        amount=-1_000.0,
        indexation=0.0,
        crash_threshold_reduction=[(0.1, 0.3), (0.3, 1.0)],
    )
    pf_single_monthly.dcf.cashflow_parameters = cwid
    _configure_small_mc(pf_single_monthly.dcf, period=5)
    initial_amount = cwid.amount

    res = pf_single_monthly.dcf.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.0, 1.0),
        target_survival_period=3,
        percentile=50,
        threshold=0,
        tolerance_rel=0.25,
        iter_max=10,
    )

    assert isinstance(res.withdrawal_abs, float)
    assert isinstance(res.withdrawal_rel, float)
    assert isinstance(res.error_rel, float)
    assert res.withdrawal_abs <= 0
    assert res.solutions.shape[0] >= 1
    assert cwid.amount == pytest.approx(initial_amount)


def test_find_the_largest_withdrawals_size_supports_vds(pf_single_monthly) -> None:
    vds = ok.VanguardDynamicSpending(
        pf_single_monthly,
        initial_investment=10_000.0,
        percentage=-0.08,
        indexation=0.0,
    )
    pf_single_monthly.dcf.cashflow_parameters = vds
    _configure_small_mc(pf_single_monthly.dcf, period=10)
    initial_percentage = vds.percentage

    res = pf_single_monthly.dcf.find_the_largest_withdrawals_size(
        goal="survival_period",
        withdrawals_range=(0.0, 0.2),
        target_survival_period=10,
        percentile=50,
        threshold=0,
        tolerance_rel=0.25,
        iter_max=10,
    )

    assert isinstance(res.withdrawal_abs, float)
    assert isinstance(res.withdrawal_rel, float)
    assert isinstance(res.error_rel, float)
    assert res.solutions.shape[0] >= 1
    assert vds.percentage == pytest.approx(initial_percentage)
