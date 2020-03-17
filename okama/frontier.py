import itertools

import pandas as pd
import numpy as np

from scipy.optimize import minimize
import matplotlib. pyplot as plt

from .assets import AssetList
from .helpers import Float, Frame, Rebalance
from okama.settings import default_ticker, n_points


class EfficientFrontier(AssetList):
    """
    Efficient Frontier (EF) with classic MVA implementation.
    n - is a number of points in the EF.
    full_frontier = False - shows only the points with the return above GMV
    """
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD', full_frontier=True, n=20):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols, first_date, last_date, curr)
        self.full_frontier = full_frontier
        self.n_points = n

    @property
    def gmv_weights(self):
        """
        Returns the weights of the Global Minimum Volatility portfolio
        """
        n = self.ror.shape[1]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        weights = minimize(Frame.get_portfolio_risk, init_guess,
                           args=(self.ror,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x

    @property
    def gmv(self):
        """
        Returns the annualized risk and return of the Global Minimum Volatility portfolio
        """
        gmv_monthly = (
            Frame.get_portfolio_risk(self.gmv_weights, self.ror),
            Frame.get_portfolio_mean_return(self.gmv_weights, self.ror)
        )

        gmv_annualized = (
            Float.annualize_risk(gmv_monthly[0], gmv_monthly[1]),
            Float.annualize_return(gmv_monthly[1])
        )
        return gmv_annualized

    def _minimize_risk(self, target_return: float) -> float:
        """
        Returns the optimal weights that achieve the target return
        given a DataFrame of return time series.
        """
        ror = self.ror
        n = ror.shape[1]  # number of assets

        init_guess = np.repeat(1 / n, n)  # initial weights

        # Set the objective function
        def objective_function(w, r):
            return Frame.get_portfolio_risk(w, r)

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        return_is_target = {'type': 'eq',
                            'args': (ror,),
                            'fun': lambda weights, ror: target_return - Frame.get_portfolio_mean_return(weights, ror)
                            }
        weights = minimize(objective_function, init_guess,
                           args=(ror,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1, return_is_target),
                           bounds=bounds)
        # Calculate points of EF given optimal weights
        risk = objective_function(weights.x, ror)
        r = Frame.get_portfolio_mean_return(weights.x, ror)
        a_r = Float.annualize_return(r)
        # Risk adjusted return approximation
        r_gmean = Float.approx_return_risk_adjusted(a_r, r)

        point = {x: y for x, y in zip(self.tickers, weights.x)}
        point['Return'] = a_r
        point['Return (risk adjusted approx)'] = r_gmean
        point['Risk'] = Float.annualize_risk(risk, r)
        return point

    @property
    def ef_points(self) -> list:
        """
        Returns a set of weights for Efficient Frontier
        """
        ror = self.ror
        er = ror.mean()
        target_rs = np.linspace(er.min(), er.max(), self.n_points)
        for (i, x) in enumerate(target_rs):
            if i == 0: df = pd.DataFrame()
            row = self._minimize_risk(x)
            df = df.append(row, ignore_index=True)
        # Put Risk, Return and "Return (risk adjusted approx)" columns in the beginning
        cols = list(df.columns.values)  # Make a list of all of the columns in the df
        cols.pop(cols.index('Risk'))  # Remove from list
        cols.pop(cols.index('Return'))
        cols.pop(cols.index('Return (risk adjusted approx)'))
        # Create new DataFrame with columns in the right order
        df = df[['Risk', 'Return', 'Return (risk adjusted approx)'] + cols]
        if not self.full_frontier:
            df = df[df['Return'] >= self.gmv[1]]
        return df


class EfficientFrontierReb(AssetList):
    """
    Efficient Frontier (EF) with rebalanced portfolio implementation.
    Objective Function is accumulated_return.
    """
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD', period='Y', n=20):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols, first_date, last_date, curr)
        self.period = period
        self.n_points = n

    @property
    def gmv_weights(self):
        """
        Returns the weights of the Global Minimum Volatility portfolio
        """
        n = self.ror.shape[1]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!

        # Set the objective function
        def objective_function(w, ror, per):
            risk = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=per).std()
            return risk

        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        weights = minimize(objective_function, init_guess,
                           args=(self.ror, self.period), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x

    def _get_gmv_monthly(self) -> tuple:
        """
        Returns the risk and return (mean, monthly) of the Global Minimum Volatility portfolio
        """
        gmv_monthly = (
            Rebalance.rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.period).std(),
            Rebalance.rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.period).mean()
        )
        return gmv_monthly

    @property
    def gmv(self):
        """
        Returns the annualized risk and CAGR of the Global Minimum Volatility portfolio
        TODO: Apparently very small difference with quick gmv calculation by get_portfolio_risk function. Replace?
        """
        returns = Rebalance.rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.period)
        gmv_annualized = (
            Float.annualize_risk(self._get_gmv_monthly()[0], self._get_gmv_monthly()[1]),
            (returns + 1).prod()**(12/self.ror.shape[0]) - 1
        )
        return gmv_annualized

    @property
    def max_return(self):
        """
        Returns the weights of the maximum return portfolio
        """
        n = self.ror.shape[1]  # Number of assets
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!

        # Set the objective function
        def objective_function(w, ror, per):
            returns = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=per)
            accumulated_return = (returns + 1.).prod() - 1.
            return - accumulated_return

        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        weights = minimize(objective_function, init_guess,
                           args=(self.ror, self.period), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        mean_return = Rebalance.rebalanced_portfolio_return_ts(weights.x, self.ror).mean()
        portfolio_risk = Rebalance.rebalanced_portfolio_return_ts(weights.x, self.ror, period=self.period).std()
        # print(portfolio_risk) # monthly risk for debugging
        point = {
            'Weights': weights.x,
            'CAGR': (1 - objective_function(weights.x, self.ror, self.period)) ** (12 / self.ror.shape[0]) - 1,
            'Risk': Float.annualize_risk(portfolio_risk, mean_return)
        }
        return point

    def _maximize_return(self, target_risk: float) -> float:
        """
        Returns the optimal weights that achieve max return at the target risk
        given a DataFrame of return time series.
        """
        ror = self.ror
        period = self.period
        n = ror.shape[1]  # number of assets

        init_guess = np.repeat(1 / n, n)  # initial weights

        # Set the objective function
        def objective_function(w, ror, per):
            objective_function.returns = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=per)
            accumulated_return = (objective_function.returns + 1.).prod() - 1.
            return - accumulated_return

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        risk_is_target = {'type': 'eq',
                          'args': (ror,),
                          'fun': lambda weights, ror: target_risk - Rebalance.rebalanced_portfolio_return_ts(weights, ror, period=period).std()
                          }
        weights = minimize(objective_function, init_guess,
                           args=(ror, period), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1, risk_is_target),
                           bounds=bounds)
        # Calculate points of EF given optimal weights
        cagr = (1. - objective_function(weights.x, ror, period))**(12/ror.shape[0]) - 1.
        returns = objective_function.returns
        risk = Float.annualize_risk(target_risk, returns.mean())
        point = {x: y for x, y in zip(self.tickers, weights.x)}
        point['CAGR'] = cagr
        point['Risk'] = risk
        return point
    
    @property
    def ef_points(self) -> pd.DataFrame:
        """
        Returns a DataFrame of points for Efficient Frontier when the Objective Function is the rate of return
        for rebalanced portfolio.
        Each point has:
        - Weights (float) for each ticker
        - CAGR (float)
        - Risk (float)
        TODO: make a function to place set of columns in the first place
        """
        min_std = self._get_gmv_monthly()[0]
        max_std = self.ror.std().max()

        target_range = np.linspace(min_std, max_std, self.n_points)
        for (i, target_risk) in enumerate(target_range):
            if i == 0: df = pd.DataFrame()
            row = self._maximize_return(target_risk)
            df = df.append(row, ignore_index=True)
        # Put Risk, Return and "Return (risk adjusted approx)" columns in the beginning
        cols = list(df.columns.values)  # Make a list of all of the columns in the df
        cols.pop(cols.index('Risk'))  # Remove from list
        cols.pop(cols.index('CAGR'))
        # Create new DataFrame with columns in the right order
        df = df[['Risk', 'CAGR'] + cols]
        return df


class Plots(AssetList):
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD'):
        super().__init__(symbols, first_date, last_date, curr)
        self.ax = None

    def plot_assets(self, type='mean'):
        """
        Plots assets scatter (annual risks, annual returns) with the tickers annotations.
        type:
        mean - mean return
        cagr_app - CAGR by approximation
        cagr - CAGR from monthly returns time series
        """
        if type == 'mean':
            risks = self.calculate_risk(annualize=True)
            returns = Float.annualize_return(self.ror.mean())
        elif type == 'cagr_app':
            risks = self.calculate_risk(annualize=True)
            returns = Float.approx_return_risk_adjusted(Float.annualize_return(self.ror.mean()), risks)
        elif type == 'cagr':
            risks = self.calculate_risk(annualize=True)
            returns = self.cagr
        # set lists for single point scatter
        if len(self.tickers) < 2:
            risks = [risks]
            returns = [returns]
        # Plotting
        if not self.ax:
            self.ax = plt.axes()
        plt.autoscale(enable=True, axis='y', tight=False)
        self.ax.scatter(risks, returns)

        for n, x, y in zip(self.tickers, risks, returns):
            label = n
            self.ax.annotate(label,  # this is the text
                        (x, y),  # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha='center')  # horizontal alignment can be left, right or center
        return self.ax

    @staticmethod
    def transition_map(ef: pd.DataFrame):
        """
        Plots EF weights transition map given a EF points DataFrame.
        """
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes()
        for i in list(ef.columns.values):
            if i not in ('Risk', 'Return', 'Return (risk adjusted approx)', 'CAGR'):
                ax.plot(ef['Risk'], ef.loc[:, i], label=i)
        ax.set_xlim(ef.Risk.min(), ef.Risk.max())
        ax.legend(loc='upper left', frameon=False)
        fig.tight_layout()
        return ax

    def plot_pair_ef(self):
        """
        Plots efficient frontier of every pair of assets in a set.
        """
        if not self.ax:
            self.ax = plt.axes()
        for i in list(itertools.combinations(self.tickers, 2)):
            ef = EfficientFrontier(symbols=i, curr=self.currency, first_date=self.first_date, last_date=self.last_date).ef_points
            self.ax.plot(ef.Risk, ef.Return)
        self.plot_assets()
        return self.ax
