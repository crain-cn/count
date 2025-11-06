import pandas as pd
from matplotlib import pyplot


df = pd.read_csv('dataset.csv',  index_col=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

series.plot()
pyplot.show()