from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from .helpers import Float, Frame, Rebalance
from .assets import AssetList


class EfficientFrontierReb(AssetList):
    """
    Efficient Frontier (EF) for rebalanced portfolios.
    Rebalancing periods could be:
    'Y' - one Year (default)
    'N' - not rebalanced portfolios
    Asset labels are set with 'tickers':
    True - for tickers
    False - for full asset names
    """
    def __init__(self,
                 symbols: List[str], *,
                 first_date: str = None,
                 last_date: str = None,
                 curr: str = 'USD',
                 inflation: bool = True,
                 reb_period: str = 'Y',
                 n_points: int = 20,
                 tickers: bool = True):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols=symbols, first_date=first_date, last_date=last_date, curr=curr, inflation=inflation)
        self.reb_period: str = reb_period
        self.n_points: int = n_points
        self.tickers: bool = tickers

    @property
    def n_points(self):
        return self._n_points

    @n_points.setter
    def n_points(self, n_points: int):
        if not isinstance(n_points, int):
            raise ValueError('n_points should be an integer')
        self._n_points = n_points

    @property
    def reb_period(self):
        return self._reb_period

    @reb_period.setter
    def reb_period(self, reb_period: str):
        if reb_period not in ['Y', 'N']:
            raise ValueError('reb_period: Rebalancing period should be "Y" - year or "N" - not rebalanced.')
        self._reb_period = reb_period

    @property
    def tickers(self):
        return self._tickers

    @tickers.setter
    def tickers(self, tickers: bool):
        if not isinstance(tickers, bool):
            raise ValueError('tickers should be True or False')
        self._tickers = tickers

    @property
    def gmv_monthly_weights(self) -> np.ndarray:
        """
        Returns the weights of the Global Minimum Volatility portfolio with monthly values of risk / return
        """
        ror = self.ror
        period = self.reb_period
        n = self.ror.shape[1]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!

        # Set the objective function
        def objective_function(w):
            risk = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period).std()
            return risk

        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        weights = minimize(objective_function,
                           init_guess,
                           method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x

    @property
    def gmv_annual_weights(self) -> np.ndarray:
        """
        Returns the weights of the Global Minimum Volatility portfolio with annualized values of risk / return
        """
        ror = self.ror
        period = self.reb_period
        n = self.ror.shape[1]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!

        # Set the objective function
        def objective_function(w):
            ts = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period)
            mean_return = ts.mean()
            risk = ts.std()
            return Float.annualize_risk(risk=risk, mean_return=mean_return)

        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        weights = minimize(objective_function,
                           init_guess,
                           method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x

    def _get_gmv_monthly(self) -> Tuple[float]:
        """
        Returns the risk and return (mean, monthly) of the Global Minimum Volatility portfolio
        """
        gmv_monthly = (
            Rebalance.rebalanced_portfolio_return_ts(self.gmv_monthly_weights, self.ror, period=self.reb_period).std(),
            Rebalance.rebalanced_portfolio_return_ts(self.gmv_monthly_weights, self.ror, period=self.reb_period).mean()
        )
        return gmv_monthly

    @property
    def gmv_annual_values(self) -> Tuple[float]:
        """
        Returns the annual risk (std) and CAGR of the Global Minimum Volatility portfolio.
        """
        returns = Rebalance.rebalanced_portfolio_return_ts(self.gmv_annual_weights, self.ror, period=self.reb_period)
        gmv = (
            Float.annualize_risk(returns.std(), returns.mean()),
            (returns + 1.).prod()**(12/returns.shape[0]) - 1.
        )
        return gmv

    @property
    def max_return(self) -> dict:
        """
        Returns the weights and risk / return of the maximum return portfolio.
        """
        ror = self.ror
        period = self.reb_period
        n = self.ror.shape[1]  # Number of assets
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!

        # Set the objective function
        def objective_function(w):
            # Accumulated return for rebalanced portfolio time series
            objective_function.returns = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period)
            accumulated_return = (objective_function.returns + 1.).prod() - 1.
            return - accumulated_return

        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        weights = minimize(objective_function,
                           init_guess,
                           method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        portfolio_ts = objective_function.returns
        mean_return = portfolio_ts.mean()
        portfolio_risk = portfolio_ts.std()
        point = {
            'Weights': weights.x,
            'CAGR': (1 - weights.fun) ** (12 / self.ror.shape[0]) - 1,
            'Risk': Float.annualize_risk(portfolio_risk, mean_return),
            'Risk_monthly': portfolio_risk
        }
        return point

    def minimize_risk(self, target_return: float) -> Dict[str, float]:
        """
        Returns the optimal weights and risk / cagr values for a min risk at the target cagr.
        """
        ror = self.ror
        period = self.reb_period
        n = ror.shape[1]  # number of assets

        init_guess = np.repeat(1 / n, n)  # initial weights

        def objective_function(w):
            # annual risk
            ts = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period)
            risk_monthly = ts.std()
            mean_return = ts.mean()
            result = Float.annualize_risk(risk_monthly, mean_return)
            return result

        def cagr(w):
            ts = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period)
            acc_return = (ts + 1.).prod() - 1.
            return (1. + acc_return)**(12 / ror.shape[0]) - 1.

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        cagr_is_target = {'type': 'eq',
                          'fun': lambda weights: target_return - cagr(weights)
                          }

        weights = minimize(objective_function,
                           init_guess,
                           method='SLSQP',
                           options={'disp': False,
                                    'maxiter': 100,
                                    'ftol': 1e-06,
                                    },
                           constraints=(weights_sum_to_1, cagr_is_target),
                           bounds=bounds)

        # Calculate points of EF given optimal weights
        if weights.success:
            if not self.tickers:
                asset_labels = list(self.names.values())
            else:
                asset_labels = self.symbols
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            point['CAGR'] = target_return
            point['Risk'] = weights.fun
        else:
            raise Exception(f'There is no solution for target cagr {target_return}.')
        return point

    def _maximize_risk_trust_constr(self, target_return: float) -> Dict[str, float]:
        """
        Returns the optimal weights and rick / cagr values for a max risk at the target cagr.
        """
        ror = self.ror
        period = self.reb_period
        n = ror.shape[1]  # number of assets

        init_guess = np.repeat(0, n)
        init_guess[self.max_annual_risk_asset['list_position']] = 1.
        risk_limit = self.gmv_annual_values[0]

        def objective_function(w):
            # - annual risk
            ts = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period)
            risk_monthly = ts.std()
            mean_return = ts.mean()
            result = - Float.annualize_risk(risk_monthly, mean_return)
            return result

        def cagr(w):
            ts = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period)
            acc_return = (ts + 1.).prod() - 1.
            return (1. + acc_return)**(12 / ror.shape[0]) - 1.

        # def constr_hess(x, v):
        #     return np.zeros([n, n])

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        cagr_is_target = {'type': 'eq',
                          'fun': lambda weights: target_return - cagr(weights)
                          }

        risk_is_above = {'type': 'ineq',
                         'fun': lambda weights: - objective_function(weights) - risk_limit
                         }

        weights = minimize(objective_function,
                           init_guess,
                           method='trust-constr',
                           options={'disp': False,
                                    'gtol': 1e-6,
                                    'xtol': 1e-8,
                                    # 'barrier_tol': 1e-01,
                                    'maxiter': 100,
                                    'factorization_method': 'QRFactorization',
                                    'verbose': 0,
                                    },
                           constraints=(weights_sum_to_1, cagr_is_target, risk_is_above),
                           bounds=bounds)

        # Calculate points of EF given optimal weights
        if weights.success:
            if not self.tickers:
                asset_labels = list(self.names.values())
            else:
                asset_labels = self.symbols
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            point['CAGR'] = target_return
            point['Risk'] = - weights.fun
        else:
            raise Exception(f'There is no solution for target cagr {target_return}.')
        return point

    def maximize_risk(self, target_return: float) -> Dict[str, float]:
        """
        Returns the optimal weights and rick / cagr values for a max risk at the target cagr.
        """
        ror = self.ror
        period = self.reb_period
        n = ror.shape[1]  # number of assets

        init_guess = np.repeat(0, n)
        init_guess[self.max_cagr_asset_right_to_max_cagr['list_position']] = 1.

        def objective_function(w):
            # annual risk
            ts = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period)
            risk_monthly = ts.std()
            mean_return = ts.mean()
            result = - Float.annualize_risk(risk_monthly, mean_return)
            return result

        def cagr(w):
            ts = Rebalance.rebalanced_portfolio_return_ts(w, ror, period=period)
            acc_return = (ts + 1.).prod() - 1.
            return (1. + acc_return)**(12 / ror.shape[0]) - 1.

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        cagr_is_target = {'type': 'eq',
                          'fun': lambda weights: target_return - cagr(weights)
                          }

        weights = minimize(objective_function,
                           init_guess,
                           method='SLSQP',
                           options={'disp': False,
                                    'ftol': 1e-06,
                                    'maxiter': 100,
                                    },
                           constraints=(weights_sum_to_1, cagr_is_target),
                           bounds=bounds)

        # Calculate points of EF given optimal weights
        if weights.success:
            if not self.tickers:
                asset_labels = list(self.names.values())
            else:
                asset_labels = self.symbols
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            point['CAGR'] = target_return
            point['Risk'] = - weights.fun
        else:
            raise Exception(f'There is no solution for target cagr {target_return}.')
        return point

    @property
    def target_cagr_range_left(self) -> np.ndarray:
        """
        Full range of cagr values (from min to max).
        """
        max_cagr = self.max_return['CAGR']
        min_cagr = Frame.get_cagr(self.ror).min()
        target_range = np.linspace(min_cagr, max_cagr, self.n_points)
        return target_range

    @property
    def max_cagr_asset(self):
        max_asset_cagr = Frame.get_cagr(self.ror).max()
        ticker_with_largest_cagr = Frame.get_cagr(self.ror).nlargest(1, keep='first').index.values[0]
        return {'max_asset_cagr': max_asset_cagr,
                'ticker_with_largest_cagr': ticker_with_largest_cagr,
                'list_position': self.symbols.index(ticker_with_largest_cagr)
                }

    @property
    def max_cagr_asset_right_to_max_cagr(self) -> Optional[dict]:
        """
        The asset with max CAGR lieing to the right of max CAGR point (risk is more than self.max_return['Risk']).
        """
        condition = self.risk_annual.values > self.max_return['Risk']
        ror_selected = self.ror.loc[:, condition]
        if not ror_selected.empty:
            cagr_selected = Frame.get_cagr(ror_selected)
            max_asset_cagr = cagr_selected.max()
            ticker_with_largest_cagr = cagr_selected.nlargest(1, keep='first').index.values[0]
            return {'max_asset_cagr': max_asset_cagr,
                    'ticker_with_largest_cagr': ticker_with_largest_cagr,
                    'list_position': self.symbols.index(ticker_with_largest_cagr)
                    }

    @property
    def max_annual_risk_asset(self):
        max_risk = self.risk_annual.max()
        ticker_with_largest_risk = self.risk_annual.nlargest(1, keep='first').index.values[0]
        return {'max_annual_risk': max_risk,
                'ticker_with_largest_risk': ticker_with_largest_risk,
                'list_position': self.symbols.index(ticker_with_largest_risk)
                }

    @property
    def target_cagr_range_right(self) -> Optional[np.ndarray]:
        """
        Range of cagr values from the global CAGR max to the max asset cagr
        to the right of the max CAGR point (if exists).
        """
        if self.max_cagr_asset_right_to_max_cagr:
            ticker_cagr = self.max_cagr_asset_right_to_max_cagr['max_asset_cagr']
            max_cagr = self.max_return['CAGR']
            if not np.isclose(max_cagr, ticker_cagr, rtol=1e-3, atol=1e-05):
                k = abs((self.target_cagr_range_left[0] - self.target_cagr_range_left[-1]) / (max_cagr - ticker_cagr))
                number_of_points = round(self.n_points / k) + 1
                target_range = np.linspace(max_cagr, ticker_cagr, number_of_points)
                return target_range[1:]  # skip the first point (max cagr) as it presents in the left part of the EF

    @property
    def target_risk_range(self) -> np.ndarray:
        """
        Range of annual risk values (from min risk to max risk).
        """
        min_std = self.gmv_annual_values[0]
        ticker_with_largest_risk = self.ror.std().nlargest(1, keep='first').index.values[0]
        max_std_monthly = self.ror.std().max()
        mean_return = self.ror.loc[:, ticker_with_largest_risk].mean()
        max_std = Float.annualize_risk(max_std_monthly, mean_return)
        target_range = np.linspace(min_std, max_std, self.n_points)
        return target_range

    @property
    def ef_points(self) -> pd.DataFrame:
        """
        Returns a DataFrame of points for Efficient Frontier when the Objective Function is the risk (std)
        for rebalanced portfolio.
        Each point has:
        - Weights (float)
        - CAGR (float)
        - Risk (float)
        """
        df = pd.DataFrame()
        # left part
        for target_cagr in self.target_cagr_range_left:
            row = self.minimize_risk(target_cagr)
            df = df.append(row, ignore_index=True)
        # right part
        range_right = self.target_cagr_range_right
        if range_right is not None:  # range_right can be a DataFrame. Should put and explicit "is not None"
            n = len(range_right)
            for target_cagr in range_right:
                row = self.maximize_risk(target_cagr)
                df = df.append(row, ignore_index=True)
        df = Frame.change_columns_order(df, ['Risk', 'CAGR'])
        return df

    def get_monte_carlo(self, n: int = 100) -> pd.DataFrame:
        """
        Calculates random risk / cagr point for rebalanced portfolios for a given asset list.
        Risk and cagr are calculated for a set of random weights.
        """
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
        return random_portfolios
