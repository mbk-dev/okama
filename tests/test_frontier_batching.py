"""Equivalence and bounded-work tests for single-period repeated portfolios."""

import numpy as np
import pandas as pd
import pytest

import okama as ok
from okama.common.helpers import helpers


@pytest.fixture()
def frontier(synthetic_env) -> ok.EfficientFrontierSingle:
    return ok.EfficientFrontierSingle(
        ["IDX.US", "A.US", "B.US"], ccy="USD", inflation=False, bounds=((0.1, 0.5), (0.2, 0.6), (0.0, 0.7))
    )


def scalar_points(ef: ok.EfficientFrontierSingle, weights: np.ndarray, kind: str) -> pd.DataFrame:
    """Frozen scalar algorithm from b33ca12, using the actual financial helpers."""
    rows = []
    for row_weights in weights:
        monthly_risk = helpers.Frame.get_portfolio_risk(row_weights, ef.assets_ror)
        monthly_mean = helpers.Frame.get_portfolio_mean_return(row_weights, ef.assets_ror)
        risk = helpers.Float.annualize_risk(monthly_risk, monthly_mean)
        mean = helpers.Float.annualize_return(monthly_mean)
        row = dict(zip(ef.get_assets_tickers(), row_weights, strict=True))
        row["Risk"] = risk
        row["Return" if kind == "mean" else "CAGR"] = (
            mean if kind == "mean" else helpers.Float.approx_return_risk_adjusted(mean, risk)
        )
        rows.append(row)
    return helpers.Frame.change_columns_order(
        pd.DataFrame.from_records(rows), ["Risk", "Return" if kind == "mean" else "CAGR"]
    )


@pytest.mark.parametrize("method", ["get_monte_carlo", "get_grid_portfolios"])
def test_covariance_prepared_once_per_request(frontier, mocker, method: str) -> None:
    """Real covariance work must not grow with the number of weight vectors."""
    covariance = mocker.spy(frontier.assets_ror, "cov")
    result = getattr(frontier, method)(**({"n": 19} if method == "get_monte_carlo" else {"step": 0.1}))
    assert len(result) > 1
    assert covariance.call_count == 1


@pytest.mark.parametrize("method", ["get_monte_carlo", "get_grid_portfolios"])
@pytest.mark.parametrize("kind", ["mean", "cagr"])
@pytest.mark.parametrize("labels", ["ticker", "name", "local_name"])
def test_batched_points_match_scalar_formulas(frontier, method: str, kind: str, labels: str) -> None:
    frontier.labels = labels
    result = getattr(frontier, method)(kind=kind, **({"n": 31} if method == "get_monte_carlo" else {"step": 0.1}))
    weights = result.iloc[:, 2:].to_numpy()
    expected = scalar_points(frontier, weights, kind)
    pd.testing.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)
    for column, (low, high) in zip(weights.T, frontier.bounds, strict=True):
        assert np.all(column >= low - 1e-12)
        assert np.all(column <= high + 1e-12)


@pytest.mark.parametrize("kind", ["mean", "cagr"])
def test_chunk_sizes_produce_equivalent_points(frontier, monkeypatch, kind: str) -> None:
    # Also exercise a partial final chunk and exactly one portfolio per chunk.
    expected = frontier.get_grid_portfolios(step=0.1, kind=kind)
    for size in (1, 7, 4096):
        monkeypatch.setattr(frontier, "_BATCH_SIZE", size)
        result = frontier.get_grid_portfolios(step=0.1, kind=kind)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)


def test_missing_returns_keep_pairwise_covariance_semantics(frontier) -> None:
    frontier.assets_ror.iloc[2, 0] = np.nan
    frontier.assets_ror.iloc[4:6, 1] = np.nan
    result = frontier.get_grid_portfolios(step=0.1)
    expected = scalar_points(frontier, result.iloc[:, 2:].to_numpy(), "mean")
    pd.testing.assert_frame_equal(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind, error", [("invalid", ValueError), ("MEAN", KeyError)])
def test_kind_validation_remains_compatible(frontier, kind: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        frontier.get_grid_portfolios(step=0.1, kind=kind)


def test_uppercase_cagr_remains_supported(frontier) -> None:
    pd.testing.assert_frame_equal(
        frontier.get_grid_portfolios(step=0.1, kind="CAGR"), frontier.get_grid_portfolios(step=0.1, kind="cagr")
    )


def test_duplicate_asset_labels_keep_last_weight(frontier) -> None:
    frontier.names = dict.fromkeys(frontier.symbols, "Same name")
    frontier.labels = "name"
    result = frontier.get_grid_portfolios(step=0.1)
    weights = np.stack(helpers.Float.get_grid_weights(3, step=0.1, bounds=frontier.bounds))
    pd.testing.assert_frame_equal(result, scalar_points(frontier, weights, "mean"), rtol=1e-12, atol=1e-12)


def test_zero_portfolios_keep_empty_result_validation(frontier) -> None:
    with pytest.raises(KeyError):
        frontier.get_monte_carlo(n=0)


def test_negative_pairwise_portfolio_variance_keeps_domain_error(frontier) -> None:
    # Pairwise deletion can produce an indefinite covariance matrix.
    frontier.assets_ror.iloc[:, :] = np.nan
    frontier.assets_ror.iloc[:6, :] = pd.DataFrame(
        {
            "IDX.US": [-0.1, 0.1, -0.1, 0.1, np.nan, np.nan],
            "A.US": [0.1, -0.1, np.nan, np.nan, -0.1, 0.1],
            "B.US": [np.nan, np.nan, 0.1, -0.1, 0.1, -0.1],
        }
    ).to_numpy()
    with pytest.raises(ValueError, match="math domain error"):
        frontier.get_grid_portfolios(step=0.1)


@pytest.mark.parametrize("value", [0.0, 0.01])
def test_constant_assets_match_scalar_risk(frontier, value: float) -> None:
    frontier.assets_ror.iloc[:, :] = value
    # The original annualization may take sqrt of a negative rounding residue.
    with np.errstate(invalid="ignore"):
        result = frontier.get_grid_portfolios(step=0.1)
        pd.testing.assert_frame_equal(
            result, scalar_points(frontier, result.iloc[:, 2:].to_numpy(), "mean"), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("assets", [2, 32])
@pytest.mark.parametrize("kind", ["mean", "cagr"])
def test_larger_covariance_matches_scalar_with_identical_weights(synthetic_env, assets: int, kind: str) -> None:
    from tests.helpers.factories import FakeAsset

    rng = np.random.default_rng(127)
    index = pd.period_range("2000-03", periods=252, freq="M")
    ror = rng.normal(0.005, 0.04, size=(252, assets))
    objects = [FakeAsset(f"T{i}.US", pd.Series(ror[:, i], index=index)) for i in range(assets)]
    ef = ok.EfficientFrontierSingle(objects, ccy="USD", inflation=False)
    np.random.seed(127)
    weights = np.stack(helpers.Float.get_random_weights(128, assets, ef.bounds))
    np.random.seed(127)
    actual = ef.get_monte_carlo(128, kind=kind)
    pd.testing.assert_frame_equal(actual, scalar_points(ef, weights, kind), rtol=1e-12, atol=1e-12)
