import itertools

import pandas as pd
import numpy as np

from scipy.optimize import minimize
import matplotlib. pyplot as plt

from .assets import AssetList, get_portfolio_mean_return, get_portfolio_risk, annualize_return, annualize_risk, rebalanced_portfolio_return_ts, approx_return_risk_adjusted
from okama.settings import default_ticker


class EfficientFrontier(AssetList):
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD'):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols, first_date, last_date, curr)

    n_points = 20  # number of points for EF

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
        weights = minimize(get_portfolio_risk, init_guess,
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
            get_portfolio_risk(self.gmv_weights, self.ror),
            get_portfolio_mean_return(self.gmv_weights, self.ror)
        )

        gmv_annualized = (
            annualize_risk(gmv_monthly[0], gmv_monthly[1]),
            annualize_return(gmv_monthly[1])
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
            return get_portfolio_risk(w, r)

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        return_is_target = {'type': 'eq',
                            'args': (ror,),
                            'fun': lambda weights, ror: target_return - get_portfolio_mean_return(weights, ror)
                            }
        weights = minimize(objective_function, init_guess,
                           args=(ror,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1, return_is_target),
                           bounds=bounds)
        # Calculate points of EF given optimal weights
        risk = objective_function(weights.x, ror)
        r = get_portfolio_mean_return(weights.x, ror)
        a_r = annualize_return(r)
        # Risk adjusted return approximation
        r_gmean = approx_return_risk_adjusted(a_r, r)

        point = {x: y for x, y in zip(self.tickers, weights.x)}
        point['Return'] = a_r
        point['Return (risk adjusted approx)'] = r_gmean
        point['Risk'] = annualize_risk(risk, r)
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
        return df


class EfficientFrontierReb(AssetList):
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD', period='Y'):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols, first_date, last_date, curr)
        self.period = period
        #self.gmv = None
        #self.gmv_annualized = None

    n_points = 40  # number of points for EF

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
            risk = rebalanced_portfolio_return_ts(w, ror, period=per).std()
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
            rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.period).std(),
            rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.period).mean()
        )
        return gmv_monthly

    @property
    def gmv(self):
        """
        Returns the annualized risk and CAGR of the Global Minimum Volatility portfolio
        NOTE: Apparently very small difference with quick gmv calculation by get_portfolio_risk function.
        TODO: Check the Note
        """
        returns = rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.period)
        gmv_annualized = (
            annualize_risk(self._get_gmv_monthly()[0], self._get_gmv_monthly()[1]),
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
            returns = rebalanced_portfolio_return_ts(w, ror, period=per)
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
        mean_return = rebalanced_portfolio_return_ts(weights.x, self.ror).mean()
        portfolio_risk = rebalanced_portfolio_return_ts(weights.x, self.ror, period=self.period).std()
        print(portfolio_risk)
        point = {
            'Weights': weights.x,
            'CAGR': (1 - objective_function(weights.x, self.ror, self.period)) ** (12 / self.ror.shape[0]) - 1,
            'Risk': annualize_risk(portfolio_risk, mean_return)
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
            objective_function.returns = rebalanced_portfolio_return_ts(w, ror, period=per)
            accumulated_return = (objective_function.returns + 1.).prod() - 1.
            return - accumulated_return

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        risk_is_target = {'type': 'eq',
                          'args': (ror,),
                          'fun': lambda weights, ror: target_risk - rebalanced_portfolio_return_ts(weights, ror, period=period).std()
                          }
        weights = minimize(objective_function, init_guess,
                           args=(ror, period), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1, risk_is_target),
                           bounds=bounds)
        # Calculate points of EF given optimal weights
        cagr = (1. - objective_function(weights.x, ror, period))**(12/ror.shape[0]) - 1.
        returns = objective_function.returns
        risk = annualize_risk(target_risk, returns.mean())
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
        """
        min_std = self._get_gmv_monthly()[0]
        max_std = self.ror.std().max()

        target_range = np.linspace(min_std, max_std, self.n_points)
        for (i, target_risk) in enumerate(target_range):
            if i == 0: df = pd.DataFrame()
            row = self._maximize_return(target_risk)
            df = df.append(row, ignore_index=True)
        return df


class Plots(AssetList):
    def __init__(self, symbols=[default_ticker], first_date=None, last_date=None, curr='USD'):
        super().__init__(symbols, first_date, last_date, curr)

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
            returns = annualize_return(self.ror.mean())
        elif type == 'cagr_app':
            risks = self.calculate_risk(annualize=True)
            returns = approx_return_risk_adjusted(annualize_return(self.ror.mean()), risks)
        elif type == 'cagr':
            risks = self.calculate_risk(annualize=True)
            returns = self.calculate_cagr()
        # set lists for single point scatter
        if len(self.tickers) < 2:
            risks = [risks]
            returns = [returns]
        # Plotting
        ax = plt.axes()
        plt.autoscale(enable=True, axis='y', tight=False)
        ax.scatter(risks, returns)

        for n, x, y in zip(self.tickers, risks, returns):
            label = n
            ax.annotate(label,  # this is the text
                        (x, y),  # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha='center')  # horizontal alignment can be left, right or center
        return ax

    def transition_map(self, ef: pd.DataFrame):
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes()
        for i in list(ef.columns.values):
            if i not in ('Risk', 'Return', 'Return (risk adjusted approx)'):
                ax.plot(ef['Risk'], ef.loc[:, i], label=i)
        ax.set_xlim(ef.Risk.iloc[0], ef.Risk.iloc[-1])
        ax.legend(loc='upper left', frameon=False)
        fig.tight_layout()
        return ax

    def plot_pair_ef(self):
        """
        Plots efficient frontier of every pair of assets in a set.
        """
        ax = plt.axes()
        for i in list(itertools.combinations(self.tickers, 2)):
            ef = EfficientFrontier(symbols=i, curr=self.currency, first_date=self.first_date, last_date=self.last_date).ef_points
            ax.plot(ef.Risk, ef.Return)
        self.plot_assets()
        return ax
