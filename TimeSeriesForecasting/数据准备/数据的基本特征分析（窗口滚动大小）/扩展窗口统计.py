# create expanding window features
import pandas as pd
from pandas import DataFrame
from pandas import concat
df = pd.read_csv('daily-minimum-temperatures-in-me.csv',  index_col=0,  header=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

print(series.head(15))
temps = DataFrame(series.values)
# 使用expanding方法，对先前所有值进行统计
window = temps.expanding()
# window.mean 代表累计平均, t+1 代表当前值
dataframe = concat([window.min(), window.mean(), window.max(), temps.shift(0)], axis=1)
dataframe.columns = ['min', 'mean', 'max', 't+1']
print(dataframe.head(15))