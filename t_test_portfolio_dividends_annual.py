import pytest
import okama as ok

tickers = [
    "BND.US",
    "VTI.US",
    "VXUS.US"
]

w = [0.34, 0.33, 0.33]

portfolio = ok.Portfolio(tickers, weights=w, rebalancing_period="none", inflation=False)

def test_dividend_yield():
    return portfolio.assets_dividend_yield

def test_dividends_annual():
    return portfolio.dividends_annual

# portfolio.assets_dividend_yield
# portfolio.dividends_annual