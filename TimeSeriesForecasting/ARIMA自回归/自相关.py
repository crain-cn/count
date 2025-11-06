import pandas as pd
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

def parser(x):
	return pd.to_datetime('190'+x, format='%Y-%m')

series = pd.read_csv('shampoo-sales.csv', header=0, index_col=0, date_parser=parser)
autocorrelation_plot(series)
pyplot.show()