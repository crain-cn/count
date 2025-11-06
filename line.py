import pandas as pd
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
import matplotlib.pyplot as plt

def parser(x):
    dt_index = pd.to_datetime(x, format='mixed', dayfirst=True)
    return dt_index

df = pd.read_csv('path_to_file.csv',  index_col=0,  date_parser=parser)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)
print(series)
temps = DataFrame(series.values)
# 使用expanding方法，对先前所有值进行统计
window = temps.expanding()
# window.mean 代表累计平均, t+1 代表当前值
# dataframe = concat([window.min(), window.mean(), window.max(), temps.shift(0)], axis=1)
# dataframe.columns = ['min', 'mean', 'max', 't+1']
dataframe = concat([window.mean(), temps.shift(0)], axis=1)
dataframe.columns = [ 'mean', 't+1']
print(dataframe)

# plt.figure(figsize=(10,6))
# plt.plot(series, label='Observed')
# plt.legend()
# pyplot.show()