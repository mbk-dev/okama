import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from okama.common.helpers import Float, Frame, Rebalance
from ..assets import AssetList
from ..settings import _MONTHS_PER_YEAR


class EfficientFrontierReb(AssetList):
    """
    Efficient Frontier (EF) for rebalanced portfolios.
    Rebalancing periods could be:
    'year' - one Year (default)
    'none' - not rebalanced portfolios
    Asset labels are set with 'tickers':
    True - for tickers
    False - for full asset names
    TODO: Add bounds
    """
    def __init__(self,
                 symbols: List[str], *,
                 first_date: str = None,
                 last_date: str = None,
                 ccy: str = 'USD',
                 inflation: bool = True,
                 reb_period: str = 'year',
                 n_points: int = 20,
                 verbose: bool = False,
                 tickers: bool = True,
                 ):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols=symbols, first_date=first_date, last_date=last_date, ccy=ccy, inflation=inflation)
        self.reb_period = reb_period
        self.n_points = n_points
        self.tickers = tickers
        self.verbose = verbose
        self._ef_points = None

    def __repr__(self):
        dic = {
            'symbols': self.symbols,
            'currency': self.currency.ticker,
            'first date': self.first_date.strftime("%Y-%m"),
            'last_date': self.last_date.strftime("%Y-%m"),
            'period length': self._pl_txt,
            'rebalancing period': self.reb_period,
            'inflation': self.inflation if hasattr(self, 'inflation') else 'None',
        }
        return repr(pd.Series(dic))

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
        """
        Rebalancing period for multi-period Efficient Frontier.

        Rebalancing periods could be:
        'year' - one Year (default)
        'none' - not rebalanced portfolios

        Returns
        -------
        pd.DataFrame
        """
        return self._reb_period

    @reb_period.setter
    def reb_period(self, reb_period: str):
        if reb_period not in ['year', 'none']:
            raise ValueError('reb_period: Rebalancing period should be "year" - year or "none" - not rebalanced.')
        self._ef_points = None
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
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        if not isinstance(verbose, bool):
            raise ValueError('verbose should be True or False')
        self._verbose = verbose

    @property
    def gmv_monthly_weights(self) -> np.ndarray:
        """
        Returns the weights of the Global Minimum Volatility portfolio with monthly values of risk / return
        """
        ror = self.ror
        period = self.reb_period
        n = self.ror.shape[1]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples

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
        return (
            Rebalance.rebalanced_portfolio_return_ts(
                self.gmv_monthly_weights, self.ror, period=self.reb_period
            ).std(),
            Rebalance.rebalanced_portfolio_return_ts(
                self.gmv_monthly_weights, self.ror, period=self.reb_period
            ).mean(),
        )

    @property
    def gmv_annual_values(self) -> Tuple[float]:
        """
        Returns the annual risk (std) and CAGR of the Global Minimum Volatility portfolio.
        """
        returns = Rebalance.rebalanced_portfolio_return_ts(self.gmv_annual_weights, self.ror, period=self.reb_period)
        return (
            Float.annualize_risk(returns.std(), returns.mean()),
            (returns + 1.0).prod() ** (_MONTHS_PER_YEAR / returns.shape[0]) - 1.0,
        )

    @property
    def max_return(self) -> dict:
        """
        Returns the weights and risk / CAGR of the maximum return portfolio point.
        """
        ror = self.ror
        period = self.reb_period
        n = self.ror.shape[1]  # Number of assets
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n

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
            'CAGR': (1 - weights.fun) ** (_MONTHS_PER_YEAR / self.ror.shape[0]) - 1,
            'Risk': Float.annualize_risk(portfolio_risk, mean_return),
            'Risk_monthly': portfolio_risk
        }
        return point

    def _get_cagr(self, weights):
        ts = Rebalance.rebalanced_portfolio_return_ts(weights, self.ror, period=self.reb_period)
        acc_return = (ts + 1.).prod() - 1.
        return (1. + acc_return) ** (_MONTHS_PER_YEAR / ts.shape[0]) - 1.

    def minimize_risk(self, target_return: float) -> Dict[str, float]:
        """
        Returns the optimal weights and risk / cagr values for a min risk at the target cagr.
        """
        n = self.ror.shape[1]  # number of assets

        init_guess = np.repeat(1 / n, n)  # initial weights

        def objective_function(w):
            # annual risk
            ts = Rebalance.rebalanced_portfolio_return_ts(w, self.ror, period=self.reb_period)
            risk_monthly = ts.std()
            mean_return = ts.mean()
            return Float.annualize_risk(risk_monthly, mean_return)

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        cagr_is_target = {'type': 'eq',
                          'fun': lambda weights: target_return - self._get_cagr(weights)
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
            asset_labels = self.symbols if self.tickers else list(self.names.values())
            point = {x: y for x, y in zip(asset_labels, weights.x)}
            # TODO: rename columns
            point['CAGR'] = target_return
            point['Risk'] = weights.fun
        else:
            raise Exception(f'There is no solution for target cagr {target_return}.')
        return point

    def _maximize_risk_trust_constr(self, target_return: float) -> Dict[str, float]:
        """
        Returns the optimal weights and rick / cagr values for a max risk at the target cagr.
        """
        n = self.ror.shape[1]  # number of assets

        init_guess = np.repeat(0, n)
        init_guess[self.max_annual_risk_asset['list_position']] = 1.
        risk_limit = self.gmv_annual_values[0]

        def objective_function(w):
            # - annual risk
            ts = Rebalance.rebalanced_portfolio_return_ts(w, self.ror, period=self.reb_period)
            risk_monthly = ts.std()
            mean_return = ts.mean()
            result = - Float.annualize_risk(risk_monthly, mean_return)
            return result

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        cagr_is_target = {'type': 'eq',
                          'fun': lambda weights: target_return - self._get_cagr(weights)
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
            asset_labels = self.symbols if self.tickers else list(self.names.values())
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
        n = self.ror.shape[1]  # number of assets

        init_guess = np.repeat(0, n)
        init_guess[self.max_cagr_asset_right_to_max_cagr['list_position']] = 1.

        def objective_function(w):
            # annual risk
            ts = Rebalance.rebalanced_portfolio_return_ts(w, self.ror, period=self.reb_period)
            risk_monthly = ts.std()
            mean_return = ts.mean()
            result = - Float.annualize_risk(risk_monthly, mean_return)
            return result

        # construct the constraints
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples for Weights constrains
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights: np.sum(weights) - 1
                            }
        cagr_is_target = {'type': 'eq',
                          'fun': lambda weights: target_return - self._get_cagr(weights)
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
            asset_labels = self.symbols if self.tickers else list(self.names.values())
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
        return np.linspace(min_cagr, max_cagr, self.n_points)

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
        Max return point should not be an asset.
        """
        tolerance = 0.01  # assets CAGR should be less than max CAGR with certain tolerance
        max_cagr_is_not_asset = (self.get_cagr() < self.max_return['CAGR'] * (1 - tolerance)).all()
        if max_cagr_is_not_asset:
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
        return np.linspace(min_std, max_std, self.n_points)

    @property
    def ef_points(self):
        """
        Return a DataFrame of points for Efficient Frontier when the Objective Function is the risk (std)
        for rebalanced portfolio.

        Each point has:
        - Weights (float)
        - CAGR (float)
        - Risk (float)
        ... and the weights for each asset.
        """
        if self._ef_points is None:
            self.get_ef_points()
        return self._ef_points

    def get_ef_points(self):
        """
        Get all the points for the Efficient Frontier running optimizer.

        If verbose=True calculates elapsed time for each point and the total elapsed time.
        """
        main_start_time = time.time()
        df = pd.DataFrame()
        # left part of the EF
        for i, target_cagr in enumerate(self.target_cagr_range_left):
            start_time = time.time()
            row = self.minimize_risk(target_cagr)
            df = df.append(row, ignore_index=True)
            end_time = time.time()
            if self.verbose:
                print(f"left EF point #{i + 1}/{self.n_points} is done in {end_time - start_time:.2f} sec.")
        # right part of the EF
        range_right = self.target_cagr_range_right
        if range_right is not None:  # range_right can be a DataFrame. Must put an explicit "is not None"
            n = len(range_right)
            for i, target_cagr in enumerate(range_right):
                start_time = time.time()
                row = self.maximize_risk(target_cagr)
                df = df.append(row, ignore_index=True)
                end_time = time.time()
                if self.verbose:
                    print(f"right EF point #{i + 1}/{n} is done in {end_time - start_time:.2f} sec.")
        df = Frame.change_columns_order(df, ['Risk', 'CAGR'])
        main_end_time = time.time()
        if self.verbose:
            print(f"Total time taken is {(main_end_time - main_start_time) / 60:.2f} min.")
        self._ef_points = df

    def get_monte_carlo(self, n: int = 100) -> pd.DataFrame:
        """
        Generate N random risk / cagr point for rebalanced portfolios.
        Risk and cagr are calculated for a set of random weights.
        """
        weights_df = Float.get_random_weights(n, self.ror.shape[1])

        # Portfolio risk and cagr for each set of weights
        portfolios_ror = weights_df.aggregate(Rebalance.rebalanced_portfolio_return_ts, ror=self.ror, period=self.reb_period)
        random_portfolios = pd.DataFrame()
        for _, data in portfolios_ror.iterrows():
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
