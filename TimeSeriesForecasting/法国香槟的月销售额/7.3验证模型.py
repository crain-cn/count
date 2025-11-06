import pandas as pd
from matplotlib import pyplot
import statsmodels.api as sm
import statsmodels as sl
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# load and prepare datasets
df = pd.read_csv('dataset.csv',  index_col=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
dataset = pd.Series(vals, index=indx)


X = dataset.values.astype('float32')
history = [x for x in X]
months_in_year = 12

# load and prepare datasets
df = pd.read_csv('validation.csv',  index_col=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
validation = pd.Series(vals, index=indx)
y = validation.values.astype('float32')
# load model
model_fit = sl.tsa.arima.model.ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
    # 差分
    months_in_year = 12
    diff = difference(history, months_in_year)
    # predict
    # 配置模型参数
    model = sm.tsa.arima.ARIMA(diff, order=(0, 0, 1))
    # 训练模型
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # 获得预测值，保存和修正预测值
    yhat = bias + inverse_difference(history, yhat, months_in_year)
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(y, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.show()
