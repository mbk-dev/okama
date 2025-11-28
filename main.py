# import warnings

import pandas as pd
from matplotlib import pyplot as plt

import okama as ok

import os

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
# warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.float_format", lambda x: "%.2f" % x)

a = ok.Inflation('RUB.INFL' ,first_date="2000-01-01")

print(a)

