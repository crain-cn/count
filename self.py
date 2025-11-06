import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot


def parser(x):
	return pd.to_datetime('190'+x, format='%Y-%m')

df = pd.read_csv('shampoo-sales.csv',  index_col=0,  date_parser=parser)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

autocorrelation_plot(series)
pyplot.show()