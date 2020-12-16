import itertools
from typing import List, Optional

from matplotlib import pyplot as plt

from .assets import AssetList, Portfolio
from .helpers import Float
from .frontier import EfficientFrontier
from .settings import default_ticker


class Plots(AssetList):
    """
    Several tools to plot Efficient Frontier, Assets and Transition map.
    """
    def __init__(self,
                 symbols: List[str] = [default_ticker],
                 first_date: Optional[str] = None,
                 last_date: Optional[str] = None,
                 curr: str = 'USD',
                 inflation: bool = True):
        super().__init__(symbols, first_date=first_date, last_date=last_date, curr=curr, inflation=inflation)
        self.ax = None
        self._bool_inflation = inflation

    def _verify_axes(self):
        if not self.ax:
            self.ax = plt.gca()
        else:
            del self.ax
            self.ax = plt.gca()

    def plot_assets(self, kind='mean', tickers='tickers', pct_values=False) -> plt.axes:
        """
        Plots assets scatter (annual risks, annual returns) with the tickers annotations.
        type:
        mean - mean return
        cagr_app - CAGR by approximation
        cagr - CAGR from monthly returns time series
        tickers:
        - 'tickers' - shows tickers values (default)
        - 'names' - shows assets names from database
        - list of string labels
        """
        if kind == 'mean':
            risks = self.risk_annual
            returns = Float.annualize_return(self.ror.mean())
        elif kind == 'cagr_app':
            risks = self.risk_annual
            returns = Float.approx_return_risk_adjusted(Float.annualize_return(self.ror.mean()), risks)
        elif kind == 'cagr':
            risks = self.risk_annual
            returns = self.get_cagr().loc[self.symbols]
        # set lists for single point scatter
        if len(self.symbols) < 2:
            risks = [risks]
            returns = [returns]
        # set the plot
        self._verify_axes()
        plt.autoscale(enable=True, axis='y', tight=False)
        if pct_values:
            m = 100
        else:
            m = 1
        self.ax.scatter(risks * m, returns * m)
        # Set the labels
        if tickers == 'tickers':
            asset_labels = self.symbols
        elif tickers == 'names':
            asset_labels = list(self.names.values())
        else:
            if not isinstance(tickers, list):
                raise ValueError(f'tickers parameter should be a list of string labels.')
            if len(tickers) != len(self.symbols):
                raise ValueError(f'The number of labels ({len(tickers)}) should be equal '
                                 f'to the number of tickers ({len(self.symbols)}).')
            else:
                asset_labels = tickers
        # draw the points and print the labels
        for n, x, y in zip(asset_labels, risks, returns):
            label = n
            self.ax.annotate(label,  # this is the text
                        (x * m, y * m),  # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha='center')  # horizontal alignment can be left, right or center
        return self.ax

    def plot_transition_map(self, bounds=None, full_frontier=False, cagr=True) -> plt.axes:
        """
        Plots EF weights transition map given a EF points DataFrame.
        cagr - sets X axe to CAGR (if true) or to risk (if false).
        """
        ef = EfficientFrontier(symbols=self.symbols,
                               first_date=self.first_date,
                               last_date=self.last_date,
                               curr=self.currency.name,
                               inflation=self._bool_inflation,
                               bounds=bounds,
                               full_frontier=full_frontier,
                               n_points=10
                               ).ef_points
        self._verify_axes()
        if cagr:
            x_axe = 'CAGR (approx)'
        else:
            x_axe = 'Risk'
        fig = plt.figure(figsize=(12, 6))
        for i in ef:
            if i not in ('Risk', 'Mean return', 'CAGR (approx)'):  # select only columns with tickers
                self.ax.plot(ef[x_axe], ef.loc[:, i], label=i)
        self.ax.set_xlim(ef[x_axe].min(), ef[x_axe].max())
        if cagr:
            self.ax.set_xlabel('CAGR (compound annual growth rate)')
        else:
            self.ax.set_xlabel('Risk (volatility)')
        self.ax.set_ylabel('Weights of assets')
        self.ax.legend(loc='upper left', frameon=False)
        fig.tight_layout()
        return self.ax

    def plot_pair_ef(self, tickers='tickers', bounds=None) -> plt.axes:
        """
        Plots efficient frontier of every pair of assets in a set.
        tickers:
        - 'tickers' - shows tickers values (default)
        - 'names' - shows assets names from database
        - list of string labels
        """
        if len(self.symbols) < 3:
            raise ValueError('The number of symbols cannot be less than 3')
        self._verify_axes()
        for i in itertools.combinations(self.symbols, 2):
            sym_pair = list(i)
            index0 = self.symbols.index(sym_pair[0])
            index1 = self.symbols.index(sym_pair[1])
            if bounds:
                bounds_pair = (bounds[index0], bounds[index1])
            else:
                bounds_pair = None
            ef = EfficientFrontier(symbols=sym_pair,
                                   curr=self.currency.currency,
                                   first_date=self.first_date,
                                   last_date=self.last_date,
                                   inflation=self._bool_inflation,
                                   full_frontier=True,
                                   bounds=bounds_pair).ef_points
            self.ax.plot(ef['Risk'], ef['Mean return'])
        self.plot_assets(kind='mean', tickers=tickers)
        return self.ax
