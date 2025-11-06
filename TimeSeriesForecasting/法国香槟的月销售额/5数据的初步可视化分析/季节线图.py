import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot


def parser(x):
    return pd.to_datetime(x, format='%Y-%m-%d')

df = pd.read_csv('dataset.csv',  index_col=0,  date_parser=parser)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)
print(series)
groups = series['1964':'1970'].groupby(pd.Grouper( freq='A'))

years = DataFrame()
pyplot.figure()

i = 1
n_groups = len(groups)
for name, group in groups:
    pyplot.subplot((n_groups*100) + 10 + i)
    i += 1
    pyplot.plot(group)
pyplot.show()
