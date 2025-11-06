import pandas as pd
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

df = pd.read_csv('daily-minimum-temperatures.csv',  index_col=0,  header=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

autocorrelation_plot(series)
pyplot.show()