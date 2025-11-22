import numpy as np
import pandas as pd


def make_period_index(months: int = 24, start: str = "2020-01") -> pd.PeriodIndex:
    """Create a monthly PeriodIndex of the given length starting from start."""
    return pd.period_range(start=start, periods=months, freq="M")


def make_ror_series(
    symbol: str,
    idx: pd.PeriodIndex,
    base: float = 0.005,
    amp: float = 0.01,
) -> pd.Series:
    """Create a synthetic monthly rate-of-return series with mild seasonality."""
    n = len(idx)
    season = amp * np.sin(np.linspace(0, 2 * np.pi, n))
    return pd.Series(base + season, index=idx, name=symbol)


class ListDefaults:
    """Provide minimal defaults for two synthetic assets with short history."""

    def __init__(self) -> None:
        self.ror_index = pd.period_range("2020-01", "2020-03", freq="M")
        self.ror_a = pd.Series([0.01, 0.02, 0.03], index=self.ror_index, name="A.US")
        self.ror_b = pd.Series([0.00, -0.01, 0.02], index=self.ror_index, name="B.US")
        self.first_ts = self.ror_index[0].to_timestamp()
        self.last_ts = self.ror_index[-1].to_timestamp()


class FakeAsset:
    """Minimal Asset-like object compatible with ListMaker expectations in tests."""

    def __init__(self, symbol: str, ror: pd.Series, currency: str = "USD", name: str | None = None):
        self.symbol = symbol
        self.ticker = symbol.split(".")[0]
        self.name = name or f"{self.ticker} name"
        self.currency = currency
        self.ror = ror
        # Synthetic close prices (monthly) and dividends for tests (PeriodIndex 'M').
        try:
            self.close_monthly = 100.0 * (1.0 + ror).cumprod()
            self.close_monthly.name = symbol
        except Exception:
            self.close_monthly = pd.Series(dtype=float, name=symbol)
        try:
            self.dividends = 0.002 * self.close_monthly
            self.dividends.name = symbol
        except Exception:
            self.dividends = pd.Series(dtype=float, name=symbol)
        self.first_date = ror.index[0].to_timestamp()
        self.last_date = ror.index[-1].to_timestamp()


class FakeCurrencyAsset:
    """Minimal currency Asset stub used by ListMaker during tests."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticker = symbol.split(".")[0]
        self.currency = self.ticker
        self.first_date = pd.Timestamp("1990-01-01")
        self.last_date = pd.Timestamp("2100-01-01")
