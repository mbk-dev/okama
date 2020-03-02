import matplotlib. pyplot as plt
from assets import Asset, AssetList, Portfolio
from frontier import EfficientFrontier

# p = Portfolio(symbols=['SPY.US', '0890-94127385.RUFUND'], curr='RUB', weights=None)
# # p_spy = Portfolio()
# print(p.returns_ts.tail())
# print(p.rebalanced_portfolio_return_ts.tail())
# print(p.weights)

# assets = AssetList(symbols=['SPY.US', '0890-94127385.RUFUND'], curr='RUB')
# print(assets.ror.head())

ef = EfficientFrontier(symbols=['SPY.US', 'GLD.US', '0890-94127385.RUFUND', '0165-70287767.RUFUND'], curr='RUB')

print('max mean return = ',ef.ror.mean().max())
# print('max rebalanced return = ',ef.get_max_return())
# print('risk monthly = ',ef.get_gmv()[0])
# print('return monthly = ',ef.get_gmv()[1])
# print(ef.get_gmv_annualized())


# fig = plt.figure(figsize=(12,6))
# ax = plt.axes()
# ax.plot(ef.ef_risk_return.Risk, ef.ef_risk_return.Return)
# plt.show()