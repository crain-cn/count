import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


df = pd.read_csv('path_to_file.csv',  index_col=0, skiprows=1,  header=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)
temps = DataFrame(series.values)
width = 2
shifted = temps.shift(width - 1)
print(shifted.head(12))

window = temps.expanding()
# 分别取过去四天的最小值，均值，最大值
dataframe = concat([window.min(), window.mean(), temps - window.mean(), np.fabs(temps - window.mean()), window.max(), temps], axis=1)
dataframe.columns = ['min', 'mean', 'mse' , 'mae' , 'max', 't+1']
print(dataframe.head(50))


plt.figure(figsize=(10,6))
plt.plot(temps, label='t+1')

plt.plot(window.mean(), label='mean')

plt.legend()
pyplot.show()

