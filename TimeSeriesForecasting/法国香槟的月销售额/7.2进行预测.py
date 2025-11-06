import pandas as pd
import statsmodels.api as sm
import statsmodels as sl
import numpy

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

df = pd.read_csv('dataset.csv',  index_col=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

months_in_year = 12
# 加载之前保存的模型参数文件
model_fit = sl.tsa.arima.model.ARIMAResults.load('model.pkl')
# 加载误差值
bias = numpy.load('model_bias.npy')
yhat = float(model_fit.forecast()[0])
# 预测结果加上误差值作为新的预测结果
yhat = bias + inverse_difference(series.values, yhat, months_in_year)
print('Predicted: %.3f' % yhat)