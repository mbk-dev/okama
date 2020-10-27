from typing import Optional, List, Tuple, Dict
import time

import pandas as pd
import numpy as np

from scipy.optimize import minimize

import mystic as my
from mystic.penalty import quadratic_equality, linear_equality
from mystic.constraints import as_constraint
from mystic.solvers import diffev2, fmin_powell
from mystic.math import almostEqual

from .assets import AssetList
from .helpers import Float, Frame, Rebalance
from .settings import default_tickers_list


class EfficientFrontier(AssetList):
    """
    Efficient Frontier (EF) with classic MVA implementation.
    n - is a number of points in the EF.
    full_frontier = False - shows only the points with the return above GMV
    tickers = True - labels of data in DataFrame are tickers (asset long names if False)
    """
    def __init__(self,
                 symbols: str = default_tickers_list,
                 first_date: Optional[str] = None,
                 last_date: Optional[str] = None,
                 curr: str = 'USD',
                 bounds: Optional[Tuple[Tuple[float]]] = None,
                 inflation: bool = True,
                 full_frontier: bool = True,
                 n: int = 20,
                 tickers: bool = True):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols, first_date, last_date, curr, inflation)
        self._bounds = None
        self.bounds = bounds
        self.full_frontier: bool = full_frontier
        self.n_points: int = n
        self.labels_are_tickers: bool = tickers

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if bounds:
            if len(bounds) != len(self.symbols):
                raise ValueError(f'The number of symbols ({len(self.symbols)}) '
                                 f'and the length of bounds ({len(bounds)}) should be equal.')
            self._bounds = bounds
        else:
            self._bounds = ((0.0, 1.0),) * len(self.symbols)  # an N-tuple of 2-tuples!

    @property
    def gmv_weights(self) -> np.ndarray:
        """
        Returns the weights of the Global Minimum Volatility portfolio
        """
        n = self.ror.shape[1]
        init_guess = np.repeat(1 / n, n)
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        weights = minimize(Frame.get_portfolio_risk,
                           init_guess,
                           args=(self.ror,),
                           method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=self.bounds)
        if weights.success:
            assert np.around(np.sum(weights.x), decimals=3) == 1.
            return weights.x
        else:
            raise Exception('No solutions where found')

    @property
    def gmv_monthly(self) -> Tuple[float]:
        """
        Returns the monthly risk and return of the Global Minimum Volatility portfolio
        """
        gmv_monthly = (
            Frame.get_portfolio_risk(self.gmv_weights, self.ror),
            Frame.get_portfolio_mean_return(self.gmv_weights, self.ror)
        )
        return gmv_monthly

    @property
    def gmv_annualized(self) -> Tuple[float]:
        """
        Returns the annualized risk and return of the Global Minimum Volatility portfolio
        """
        gmv_annualized = (
            Float.annualize_risk(self.gmv_monthly[0], self.gmv_monthly[1]),
            Float.annualize_return(self.gmv_monthly[1])
        )
        return gmv_annualized

    def optimize_return(self, option: str ='max') -> dict:
        """
        Finds global max or min for the rate of return.
        Returns monthly values for the risk, mean return and the weights.
        'max' - search for global maximum
        'min' - search for global minimum
        """
        n = self.ror.shape[1]  # Number of assets
        init_guess = np.repeat(1 / n, n)
        # Set the objective function
        if option == 'max':
            def objective_function(w, ror):
                month_return_value = Frame.get_portfolio_mean_return(w, ror)
                return - month_return_value
        elif option == 'min':
            def objective_function(w, ror):
                month_return_value = Frame.get_portfolio_mean_return(w, ror)
                return month_return_value
        else:
            raise ValueError('option should be "max" or "min"')
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        weights = minimize(objective_function,
                           init_guess,
                           args=(self.ror,),
                           method='SLSQP',
                           constraints=(weights_sum_to_1,),
                           bounds=self.bounds,
                           options={'disp': False,
                                    'ftol': 1e-08}  # 1e-06 is not enough to optimize monthly returns
                           )
        if weights.success:
            portfolio_risk = Frame.get_portfolio_risk(weights.x, self.ror)
            if option == 'max':
                optimized_return = -weights.fun
            else:
                optimized_return = weights.fun
            point = {
                'Weights': weights.x,
                'Mean_return_monthly': optimized_return,
                'Risk_monthly': portfolio_risk
            }
            return point
        else:
            raise Exception('No solutions where found')

    def minimize_risk(self,
                      target_return: float,
                      monthly_return: bool = False,
                      tolerance: float = 1e-08
                      ) -> Dict[str, float]:
        """
        Finds minimal risk given the target return.
        Returns a "point" with monthly values:
        - weights
        - mean return
        - aproximate vaue for the CAGR
        - risk (std)
        Target return is a monthly or annual value:
        monthly_return = False / True
        tolerance - sets the accuracy for the solver
        """
        if not monthly_return:
            target_return = Float.get_monthly_return_from_annual(target_return)
        ror = self.ror
        n = ror.shape[1]  # number of assets
        init_guess = np.repeat(1 / n, n)  # initial weights

        def objective_function(w):
            return Frame.get_portfolio_risk(w, ror)

        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        return_is_target = {'type': 'eq',
                            'fun': lambda weights: target_return - Frame.get_portfolio_mean_return(weights, ror)
                            }
        weights = minimize(objective_function,
                           init_guess,
                           method='SLSQP',
                           constraints=(weights_sum_to_1, return_is_target),
                           bounds=self.bounds,
                           options={'disp': False,
                                    'ftol': tolerance}
                           )
        if weights.success:
            # Calculate point of EF given optimal weights
            risk = weights.fun
            # Annualize risk and return
            a_r = Float.annualize_return(target_return)
            a_risk = Float.annualize_risk(risk=risk, mean_return=target_return)
            # Risk adjusted return approximation
            r_gmean = Float.approx_return_risk_adjusted(mean_return=a_r, std=a_risk)
            if not self.labels_are_tickers:
                asset_labels = list(self.names.values())
            else:
                asset_labels = self.symbols
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            point['Mean return'] = a_r
            point['CAGR (approx)'] = r_gmean
            point['Risk'] = a_risk
        else:
            raise Exception("No solutions were found")
        return point

    @property
    def mean_return_range(self) -> np.ndarray:
        """
        Returns the range of mean monthly returns (from the min to max).
        """
        if self.full_frontier:
            if not self.bounds:
                er = self.ror.mean()
                min_return = er.min()
                max_return = er.max()
            else:
                min_return = self.optimize_return(option='min')['Mean_return_monthly']
                max_return = self.optimize_return(option='max')['Mean_return_monthly']
        else:
            min_return = self.gmv_monthly[1]
            if not self.bounds:
                er = self.ror.mean()
                max_return = er.max()
            else:
                max_return = self.optimize_return(option='max')['Mean_return_monthly']
        return_range = np.linspace(min_return, max_return, self.n_points)
        return return_range

    @property
    def ef_points(self) -> pd.DataFrame:
        """
        DataFrame of weights and risk/return values for the Efficient Frontier.
        The columns of the DataFrame:
        - weights
        - mean return
        - aproximate vaue for the CAGR
        - risk (std)
        All the values are annualized.
        """
        target_rs = self.mean_return_range
        df = pd.DataFrame(dtype='float')
        for x in target_rs:
            row = self.minimize_risk(x, monthly_return=True)
            df = df.append(row, ignore_index=True)
        df = Frame.change_columns_order(df, ['Risk', 'Mean return', 'CAGR (approx)'])
        return df


class EfficientFrontierReb(AssetList):
    """
    Efficient Frontier (EF) with rebalanced portfolio implementation.
    Objective Function is accumulated_return.
    Default rebalancing period is Year - 'Y' ('M' - for month)
    """
    def __init__(self,
                 symbols: List[str] = default_tickers_list,
                 first_date: str = None,
                 last_date: str = None,
                 curr: str = 'USD',
                 inflation: bool = True,
                 period: str = 'Y',
                 n: int = 20,
                 tickers: bool = True):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols, first_date, last_date, curr, inflation)
        self.reb_period: str = period
        self.n_points: int = n
        self.gmv_monthly: Tuple[float] = self._get_gmv_monthly()
        self.labels_are_tickers: bool = tickers

    @property
    def gmv_weights(self) -> np.ndarray:
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
                           args=(self.ror, self.reb_period), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x

    def _get_gmv_monthly(self) -> Tuple[float]:
        """
        Returns the risk and return (mean, monthly) of the Global Minimum Volatility portfolio
        """
        gmv_monthly = (
            Rebalance.rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.reb_period).std(),
            Rebalance.rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.reb_period).mean()
        )
        return gmv_monthly

    @property
    def gmv(self) -> Tuple[float]:
        """
        Returns the annualized risk and CAGR of the Global Minimum Volatility portfolio
        TODO: Apparently very small difference with quick gmv calculation by get_portfolio_risk function. Replace?
        """
        returns = Rebalance.rebalanced_portfolio_return_ts(self.gmv_weights, self.ror, period=self.reb_period)
        gmv_annualized = (
            Float.annualize_risk(self._get_gmv_monthly()[0], self._get_gmv_monthly()[1]),
            (returns + 1.).prod()**(12/self.ror.shape[0]) - 1.
        )
        return gmv_annualized

    @property
    def max_return(self) -> dict:
        """
        Returns the weights and risk|return of the maximum return portfolio.
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
                           args=(self.ror, self.reb_period), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        mean_return = Rebalance.rebalanced_portfolio_return_ts(weights.x, self.ror).mean()
        portfolio_risk = Rebalance.rebalanced_portfolio_return_ts(weights.x, self.ror, period=self.reb_period).std()
        point = {
            'Weights': weights.x,
            'CAGR': (1 - objective_function(weights.x, self.ror, self.reb_period)) ** (12 / self.ror.shape[0]) - 1,
            'Risk': Float.annualize_risk(portfolio_risk, mean_return),
            'Risk_monthly': portfolio_risk
        }
        return point

    def _maximize_return(self, target_risk: float) -> Dict[str, float]:
        """
        Returns the optimal weights that achieve max return at the target risk
        given a DataFrame of return time series.
        """
        ror = self.ror
        period = self.reb_period
        n = ror.shape[1]  # number of assets

        init_guess = np.repeat(1 / n, n)  # initial weights

        # Set the objective function
        def objective_function(w, ror, per):
            objective_function.returns = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=per)
            if objective_function.returns.mean() > self.gmv_monthly[1]:  # makes OF convex
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
        if weights.success:
            cagr = (1. - weights.fun)**(12 / ror.shape[0]) - 1.
            returns = objective_function.returns
            risk = Float.annualize_risk(target_risk, returns.mean())
            if not self.labels_are_tickers:
                asset_labels = list(self.names.values())
            else:
                asset_labels = self.symbols
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            point['CAGR'] = cagr
            point['Risk'] = risk
        else:
            raise Exception(f'There is no solution for target month risk {target_risk}.')
        return point

    def maximize_return_penalty_mysty(self, target_risk):
        """
        Testing Mystic nonconvex optimizer
        """
        main_start_time = time.time()
        assets_ror_ts = self.ror
        period = self.reb_period
        n = assets_ror_ts.shape[1]  # number of assets

        init_guess = [0, 1.]
        # init_guess = np.repeat(1 / n, n)  # initial guess

        def objective(w):
            objective.returns = Rebalance.rebalanced_portfolio_return_ts(w, assets_ror_ts, period=period)
            accumulated_return = (objective.returns + 1.).prod() - 1.
            return - accumulated_return  # inverted to minimize

        def penalty1(weights):
            return np.sum(weights) - 1.

        def penalty2(weights):
            return Rebalance.rebalanced_portfolio_return_ts(weights, assets_ror_ts, period=period).std() - target_risk

        @linear_equality(penalty1, k=1e12)  # == 0
        @linear_equality(penalty2, k=1e12)
        def penalty(x):
            return 0.0

        solver = as_constraint(penalty)

        bounds = [(0, 1)] * n

        # result = diffev2(objective, x0=init_guess, bounds=bounds, penalty=penalty, npop=40, ftol=5e-3, gtol=100, disp=False, full_output=True)

        # result = fmin_powell(objective, x0=init_guess, bounds=bounds, penalty=penalty, disp=False, ftol=5e-3, full_output=True)

        result = fmin_powell(objective, x0=init_guess, bounds=bounds, constraints=solver, disp=False, ftol=5e-3, full_output=True)

        # assert almostEqual(np.sum(result[0]), 1., rel=1e-2)
        main_end_time = time.time()
        print(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
        return result

    def maximize_return_constraints_mysty(self, target_risk):
        """
        Testing Mystic nonconvex optimizer
        """
        main_start_time = time.time()
        assets_ror_ts = self.ror
        period = self.reb_period
        n = assets_ror_ts.shape[1]  # number of assets

        init_guess = np.repeat(1 / n, n)  # initial guess

        def objective(w):
            objective.returns = Rebalance.rebalanced_portfolio_return_ts(w, assets_ror_ts, period=period)
            accumulated_return = (objective.returns + 1.).prod() - 1.
            return - accumulated_return  # inverted to minimize

        # @my.constraints.normalized()
        def constraint1(weights):
            return my.constraints.impose_sum(np.sum(weights), 1.)

        def rebal_std(weights):
            return Rebalance.rebalanced_portfolio_return_ts(weights, assets_ror_ts, period=period).std()

        def constraint2(weights):
            return my.constraints.impose_sum(rebal_std(weights), - target_risk)

        cons = my.constraints.and_(constraint1, constraint2)

        bounds = [(0, 1)] * n

        # result = diffev2(objective, x0=init_guess, bounds=bounds, penalty=penalty, npop=40, ftol=5e-3, gtol=100, disp=False, full_output=True)

        result = fmin_powell(objective, x0=init_guess, bounds=bounds, constraints=cons, disp=False, ftol=5e-3, full_output=True)
        assert almostEqual(np.sum(result[0]), 1., rel=1e-2)
        main_end_time = time.time()
        print(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
        return result

    @property
    def _target_risk_range(self) -> np.ndarray:
        """
        Range of monthly risk values (from min risk to max risk).
        """
        min_std = self._get_gmv_monthly()[0]

        ticker_with_largest_mean_return = self.ror.mean().nlargest(1, keep='first').index.values[0]
        max_std = self.risk_monthly.loc[ticker_with_largest_mean_return]
        print(f'Starting optimization: min risk = {min_std}, max risk = {max_std}')

        # max_std = self.ror.std().max()

        # max_std = self.max_return['Risk_monthly']  # Risk limit is a max return point risk

        # criterion = self.ror.mean() > self.gmv_monthly[1]  # Select only R above GMV. Works only if such asset exist
        # max_std = self.ror.loc[:, criterion].std().max()

        target_range = np.linspace(min_std, max_std, self.n_points)
        return target_range

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
        main_start_time = time.time()

        target_range = self._target_risk_range
        for (i, target_risk) in enumerate(target_range):
            if i == 0:
                df = pd.DataFrame()
            start_time = time.time()
            row = self._maximize_return(target_risk)
            df = df.append(row, ignore_index=True)
            end_time = time.time()
            print(f"EF point #{i+1}/{self.n_points} is done in {end_time - start_time:.2f} sec.")
        df = Frame.change_columns_order(df, ['Risk', 'CAGR'])
        main_end_time = time.time()
        print(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
        return df

    def get_monte_carlo(self, n: int = 100) -> pd.DataFrame:
        """
        Calculates random risk, cagr point for rebalanced portfolios for a given asset list.
        Risk and cagr are calculated from a set of random weights.
        """
        main_start_time = time.time()
        # Random weights
        rand_nos = np.random.rand(n, self.ror.shape[1])
        weights_transposed = rand_nos.transpose() / rand_nos.sum(axis=1)
        weights = weights_transposed.transpose()
        weights_df = pd.DataFrame(weights)
        # weights_df = weights_df.aggregate(list, axis=1)  # Converts df to DataFrame of lists
        weights_df = weights_df.aggregate(np.array, axis=1)  # Converts df to DataFrame of np.array

        # Portfolio risk and cagr for each set of weights
        portfolios_ror = weights_df.aggregate(Rebalance.rebalanced_portfolio_return_ts, ror=self.ror, period=self.reb_period)
        for index, data in portfolios_ror.iterrows():
            if index == 0:
                random_portfolios = pd.DataFrame()
            risk_monthly = data.std()
            mean_return = data.mean()
            risk = Float.annualize_risk(risk_monthly, mean_return)
            cagr = Frame.get_cagr(data)
            row = {
                'Risk': risk,
                'CAGR': cagr
            }
            random_portfolios = random_portfolios.append(row, ignore_index=True)
        main_end_time = time.time()
        print(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
        return random_portfolios


