import pandas as pd

from pandas import DataFrame
import statsmodels.api as sm
from matplotlib import pyplot

def parser(x):
	return pd.to_datetime('190'+x, format='%Y-%m')

series = pd.read_csv('shampoo-sales.csv', header=0,index_col=0, date_parser=parser)
# fit model
model = sm.tsa.arima.ARIMA(series, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())