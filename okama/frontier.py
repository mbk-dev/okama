from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np

from scipy.optimize import minimize

from .assets import AssetList
from .helpers import Float, Frame
from .settings import default_tickers_list


class EfficientFrontier(AssetList):
    """
    Efficient Frontier (EF) with classic MVA implementation.
    n - is a number of points in the EF.
    full_frontier = False - shows only the points with the return above GMV
    tickers = True - labels of data in DataFrame are tickers (asset long names if False)
    """
    def __init__(self,
                 symbols: str = default_tickers_list, *,
                 first_date: Optional[str] = None,
                 last_date: Optional[str] = None,
                 curr: str = 'USD',
                 bounds: Optional[Tuple[Tuple[float]]] = None,
                 inflation: bool = True,
                 full_frontier: bool = True,
                 n_points: int = 20,
                 tickers: bool = True):
        if len(symbols) < 2:
            raise ValueError('The number of symbols cannot be less than two')
        super().__init__(symbols, first_date=first_date, last_date=last_date, curr=curr, inflation=inflation)
        self._bounds = None
        self.bounds = bounds
        self.full_frontier: bool = full_frontier
        self.n_points: int = n_points
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
