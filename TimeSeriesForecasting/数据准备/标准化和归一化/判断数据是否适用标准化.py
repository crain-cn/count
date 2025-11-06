import pandas as pd
from matplotlib import pyplot

df = pd.read_csv('daily-minimum-temperatures-in-me.csv',  index_col=0,  header=0, skiprows=1)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

series.hist()
# 看看画出的数据图是否符合高斯钟型分布
pyplot.show()