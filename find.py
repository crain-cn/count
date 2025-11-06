import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot

def parser(x):
	return pd.to_datetime('190'+x, format='%Y-%m')

df = pd.read_csv('shampoo-sales.csv',  index_col=0,  date_parser=parser)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series_again = pd.Series(vals, index=indx)
print(series_again)
# fit model
model = sm.tsa.arima.ARIMA(series_again, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())