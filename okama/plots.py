import itertools
from typing import List, Optional, Union

from matplotlib import pyplot as plt

from .asset_list import AssetList
from .common.helpers import Float
from .frontier.single_period import EfficientFrontier
from .settings import default_ticker


class Plots(AssetList):
    """
    Plotting tools collection to use with financial charts (Efficient Frontier, Assets and Transition map etc.)

    Parameters
    ----------
    assets : list, default None
        List of assets. Could include tickers or asset like objects (Asset, Portfolio).
        If None a single asset list with a default ticker is used.

    first_date : str, default None
        First date of monthly return time series.
        If None the first date is calculated automatically as the oldest available date for the listed assets.

    last_date : str, default None
        Last date of monthly return time series.
        If None the last date is calculated automatically as the newest available date for the listed assets.

    ccy : str, default 'USD'
        Base currency for the list of assets. All risk metrics and returns are adjusted to the base currency.

    inflation : bool, default True
        Defines whether to take inflation data into account in the calculations.
        Including inflation could limit available data (last_date, first_date)
        as the inflation data is usually published with a one-month delay.
        With inflation = False some properties like real return are not available.
    """

    def __init__(
        self,
        assets: List[str] = [default_ticker],
        first_date: Optional[str] = None,
        last_date: Optional[str] = None,
        ccy: str = "USD",
        inflation: bool = True,
    ):
        super().__init__(
            assets,
            first_date=first_date,
            last_date=last_date,
            ccy=ccy,
            inflation=inflation,
        )
        self.ax = None
        self._bool_inflation = inflation

    def _verify_axes(self):
        if self.ax:
            del self.ax
        self.ax = plt.gca()

    def plot_assets(
        self,
        kind: str = "mean",
        tickers: Union[str, list] = "tickers",
        pct_values: bool = False,
    ) -> plt.axes:
        """
        Plot the assets points on the risk-return chart with annotations.

        Annualized values for risk and return are used.
        Risk is a standard deviation of monthly rate of return time series.
        Return can be an annualized mean return (expected return) or CAGR (Compound annual growth rate).

        Returns
        -------
        Axes : 'matplotlib.axes._subplots.AxesSubplot'

        Parameters
        ----------
        kind : {'mean', 'cagr'}, default 'mean'
            Type of Return: annualized mean return (expected return) or CAGR (Compound annual growth rate).

        tickers : {'tickers', 'names'} or list of str, default 'tickers'
            Annotation type for assets.
            'tickers' - assets symbols are shown in form of 'SPY.US'
            'names' - assets names are used like - 'SPDR S&P 500 ETF Trust'
            To show custom annotations for each asset pass the list of names.

        pct_values : bool, default False
            Risk and return values in the axes:
            Algebraic annotation (False)
            Percents (True)

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = ok.Plots(['SPY.US', 'AGG.US'], ccy='USD', inflation=False)
        >>> x.plot_assets()
        >>> plt.show()

        Plotting with default parameters values shows expected return, ticker annotations and algebraic values
        for risk and return.
        To use CAGR instead of expected return use kind='cagr'.

        >>> x.plot_assets(kind='cagr',
        ...               tickers=['US Stocks', 'US Bonds'],  # use custom annotations for the assets
        ...               pct_values=True  # risk and return values are in percents
        ...               )
        >>> plt.show()
        """
        if kind == "mean":
            risks = self.risk_annual
            returns = Float.annualize_return(self.assets_ror.mean())
        elif kind == "cagr":
            risks = self.risk_annual
            returns = self.get_cagr().loc[self.symbols]
        else:
            raise ValueError('kind should be "mean" or "cagr".')
        # set lists for single point scatter
        if len(self.symbols) < 2:
            risks = [risks]
            returns = [returns]
        # set the plot
        self._verify_axes()
        plt.autoscale(enable=True, axis="year", tight=False)
        m = 100 if pct_values else 1
        self.ax.scatter(risks * m, returns * m)
        # Set the labels
        if tickers == "tickers":
            asset_labels = self.symbols
        elif tickers == "names":
            asset_labels = list(self.names.values())
        else:
            if not isinstance(tickers, list):
                raise ValueError(
                    f"tickers parameter should be a list of string labels."
                )
            if len(tickers) != len(self.symbols):
                raise ValueError("labels and tickers must be of the same length")
            asset_labels = tickers
        # draw the points and print the labels
        for label, x, y in zip(asset_labels, risks, returns):
            self.ax.annotate(
                label,  # this is the text
                (x * m, y * m),  # this is the point to label
                textcoords="offset points",  # how to position the text
                xytext=(0, 10),  # distance from text to points (x,y)
                ha="center",  # horizontal alignment can be left, right or center
            )
        return self.ax

    def plot_transition_map(
        self, bounds=None, full_frontier=False, cagr=True
    ) -> plt.axes:
        """
        Plot Transition Map for optimized portfolios on the single period Efficient Frontier.

        Transition Map shows the relation between asset weights and optimized portfolios properties:

        - CAGR (Compound annual growth rate)
        - Risk (annualized standard deviation of return)

        Wights are displayed on the y-axis.
        CAGR or Risk - on the x-axis.

        Constrained optimization with weights bounds is available.

        Returns
        -------
        Axes : 'matplotlib.axes._subplots.AxesSubplot'

        Parameters
        ----------
        bounds: tuple of ((float, float),...)
            Bounds for the assets weights. Each asset can have weights limitation from 0 to 1.0.
            If an asset has limitation for 10 to 20%, bounds are defined as (0.1, 0.2).
            bounds = ((0, .5), (0, 1)) shows that in Portfolio with two assets first one has weight limitations
            from 0 to 50%. The second asset has no limitations.

        full_frontier : bool, default False
            Defines whether to show the Transition Map for portfolios on the full Efficient Frontier or
            only on its upper part.
            If 'False' only portfolios with the return above Global Minimum Volatility (GMV) point are shown.

        cagr : bool, default True
            Show the relation between weights and CAGR (if True) or between weights and Risk (if False).
            of - sets X axe to CAGR (if true) or to risk (if false).
            CAGR or Risk are displayed on the x-axis.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> x = ok.Plots(['SPY.US', 'AGG.US', 'GLD.US'], ccy='USD', inflation=False)
        >>> x.plot_transition_map()
        >>> plt.show()

        Transition Map with default setting show the relation between Return (CAGR) and assets weights for optimized portfolios.
        The same relation for Risk can be shown setting cagr=False.

        >>> x.plot_transition_map(cagr=False,
        ...                       full_frontier=True,  # to see the relation for the full Efficient Frontier
        ...                       )
        >>> plt.show()
        """
        ef = EfficientFrontier(
            assets=self.symbols,
            first_date=self.first_date,
            last_date=self.last_date,
            ccy=self.currency,
            inflation=self._bool_inflation,
            bounds=bounds,
            full_frontier=full_frontier,
            n_points=20,
        ).ef_points
        self._verify_axes()
        linestyle = itertools.cycle(("-", "--", ":", "-."))
        x_axe = "CAGR" if cagr else "Risk"
        fig = plt.figure(figsize=(12, 6))
        for i in ef:
            if i not in (
                "Risk",
                "Mean return",
                "CAGR",
            ):  # select only columns with tickers
                self.ax.plot(
                    ef[x_axe], ef.loc[:, i], linestyle=next(linestyle), label=i
                )
        self.ax.set_xlim(ef[x_axe].min(), ef[x_axe].max())
        if cagr:
            self.ax.set_xlabel("CAGR (Compound Annual Growth Rate)")
        else:
            self.ax.set_xlabel("Risk (volatility)")
        self.ax.set_ylabel("Weights of assets")
        self.ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        return self.ax

    def plot_pair_ef(self, tickers="tickers", bounds=None) -> plt.axes:
        """
        Plot Efficient Frontier of every pair of assets.

        Efficient Frontier is a set of portfolios which satisfy the condition that no other portfolio exists
        with a higher expected return but with the same risk (standard deviation of return).

        Arithmetic mean (expected return) is used for optimized portfolios.

        Returns
        -------
        Axes : 'matplotlib.axes._subplots.AxesSubplot'

        Parameters
        ----------
        tickers : {'tickers', 'names'} or list of str, default 'tickers'
            Annotation type for assets.
            'tickers' - assets symbols are shown in form of 'SPY.US'
            'names' - assets names are used like - 'SPDR S&P 500 ETF Trust'
            To show custom annotations for each asset pass the list of names.

        bounds: tuple of ((float, float),...)
            Bounds for the assets weights. Each asset can have weights limitation from 0 to 1.0.
            If an asset has limitation for 10 to 20%, bounds are defined as (0.1, 0.2).
            bounds = ((0, .5), (0, 1)) shows that in Portfolio with two assets first one has weight limitations
            from 0 to 50%. The second asset has no limitations.

        Notes
        -----
        It should be at least 3 assets.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> ls4 = ['SPY.US', 'BND.US', 'GLD.US', 'VNQ.US']
        >>> curr = 'USD'
        >>> last_date = '07-2021'
        >>> ok.Plots(ls4, ccy=curr, last_date=last_date).plot_pair_ef()
        >>> plt.show()

        It can be useful to plot the full Efficent Frontier (EF) with optimized 4 assets portfolios
        together with the EFs for each pair of assets.

        >>> ef4 = ok.EfficientFrontier(assets=ls4, ccy=curr, n_points=100)
        >>> df4 = ef4.ef_points
        >>> fig = plt.figure()
        >>> # Plot Efficient Frontier of every pair of assets. Optimized portfolios will have 2 assets.
        >>> ok.Plots(ls4, ccy=curr, last_date=last_date).plot_pair_ef()  # mean return is used for optimized portfolios.
        >>> ax = plt.gca()
        >>> # Plot the full Efficient Frontier for 4 asset portfolios.
        >>> ax.plot(df4['Risk'], df4['Mean return'], color = 'black', linestyle='--')
        >>> plt.show()
        """
        if len(self.symbols) < 3:
            raise ValueError("The number of symbols cannot be less than 3")
        self._verify_axes()
        for i in itertools.combinations(self.symbols, 2):
            sym_pair = list(i)
            index0 = self.symbols.index(sym_pair[0])
            index1 = self.symbols.index(sym_pair[1])
            if bounds:
                bounds_pair = (bounds[index0], bounds[index1])
            else:
                bounds_pair = None
            ef = EfficientFrontier(
                assets=sym_pair,
                ccy=self.currency,
                first_date=self.first_date,
                last_date=self.last_date,
                inflation=self._bool_inflation,
                full_frontier=True,
                bounds=bounds_pair,
            ).ef_points
            self.ax.plot(ef["Risk"], ef["Mean return"])
        self.plot_assets(kind="mean", tickers=tickers)
        return self.ax
