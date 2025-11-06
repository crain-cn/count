import pandas as pd
from matplotlib import pyplot

def parser(x):
	return pd.to_datetime('190'+x, format='%Y-%m')

df = pd.read_csv('shampoo-sales.csv',  index_col=0, date_parser=parser)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)
series.plot()
pyplot.show()