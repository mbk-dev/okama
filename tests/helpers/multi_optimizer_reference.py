"""Original weight-cache evaluation from 48a0e7c, independent of Git at test time."""

import numpy as np
import pandas as pd

from okama.frontier.multi_period import EfficientFrontier
from tests.helpers.rebalance_reference import ScalarCalendarRebalance


class ScalarMultiOptimizer(EfficientFrontier):
    """Use frozen pandas rebalance arithmetic with unchanged public optimizer methods."""

    def _get_portfolio_ror_ts(self, weights: np.ndarray) -> pd.Series:
        weights_key = tuple(np.round(weights, decimals=10))
        if weights_key not in self._ror_cache:
            rebalance = ScalarCalendarRebalance(period=self.rebalancing_strategy.period)
            self._ror_cache[weights_key] = rebalance.return_ror_ts_ef(weights, self.assets_ror)
        return self._ror_cache[weights_key]
