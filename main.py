import matplotlib. pyplot as plt
from okama.assets import Asset, AssetList, Portfolio
from okama.frontier import EfficientFrontier, EfficientFrontierReb, Plots

p = 'test'
# # p_spy = Portfolio()
# print(p.returns_ts.tail())
# print(p.rebalanced_portfolio_return_ts.tail())
# print(p.weights)

# assets = AssetList(symbols=['SPY.US', '0890-94127385.RUFUND'], curr='RUB')
print(p)

ef = EfficientFrontier(symbols=['SPY.US', '0890-94127385.RUFUND'], curr='RUB').ef_points

# x = Plots(['SPY.US', 'GLD.US', 'VNQ.US', '0890-94127385.RUFUND', '0165-70287767.RUFUND'], first_date='2009-01', curr='RUB')
# x.plot_pair_ef()


# fig = plt.figure(figsize=(12,6))
# ax = plt.axes()
# ax.plot(ef.ef_risk_return.Risk, ef.ef_risk_return.Return)
# plt.show()