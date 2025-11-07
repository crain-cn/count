import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

def parser(x):
    dt_index = pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
    return dt_index

df = pd.read_csv('gold.csv',  index_col=0,  header=0,  date_parser=parser)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

temps = DataFrame(series.values)
width = 2
shifted = temps.shift(width - 1)
print(shifted.head(12))

window = shifted.rolling(window=3)
# 分别取过去四天的最小值，均值，均方误差，平均绝对误差，最大值，正常值
# 增加移动加权均值 weighted_ma
weights = np.array([0.6, 0.3, 0.1])
def weighted_ma(x):
    if np.any(pd.isnull(x)):
        return np.nan
    return np.dot(x, weights)
weighted_ma_series = shifted.rolling(window=3).apply(weighted_ma, raw=True)
dataframe = concat([
    window.min(),
    window.mean(),
    temps - window.mean(),
    np.fabs(temps - window.mean()),
    window.max(),
    temps,
    weighted_ma_series
], axis=1)
dataframe.columns = ['min', 'mean', 'mse' , 'mae' , 'max', 't+1', 'weighted_ma']
print(dataframe.head(50))


plt.figure(figsize=(10,6))
X_full = np.arange(len(series)).reshape(-1, 1)

# t+1 trend
plt.plot(series.index, series.values, label='t+1 trend', linestyle='-')

# ma trend（去NaN）
plt.plot(series.index, window.mean(), label='ma trend', linestyle='--')

# weighted_ma trend（去NaN）
plt.plot(series.index, weighted_ma_series, label='weighted_ma trend', linestyle=':')

plt.legend()
pyplot.show()

