import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


df = pd.read_csv('daily-minimum-temperatures.csv',  index_col=0,  header=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
print(train)
# train autoregression
model = AutoReg(train, lags=1)
model_fit = model.fit()
# 滞后长度
print('Lag: %s' % model_fit.ar_lags)
# 系数
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()