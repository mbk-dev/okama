.. _multi_period:

=========================
Multi-Period Optimization
=========================

In single period optimization portfolio is always rebalanced and has original weights. However, in real life portfolios
are not rebalanced every day or every moth.

In multi-period approach portfolio is rebalanced to the original allocation with a certain frequency (annually, quarterly etc.) or not rebalanced at all.

EfficientFrontierReb class can be used for multi-period optimization. Two rebalancing frequencies can be usd (reb_period parameter):

- 'year' - one Year (default)
- 'none' - not rebalanced portfolios


.. autoclass:: okama.EfficientFrontierReb
    :members:
