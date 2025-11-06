from pandas import Series
import statsmodels.api as sm
from scipy.stats import boxcox
import numpy
import pandas as pd

# monkey patch around bug in ARIMA class
# def __getnewargs__(self):
#	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

# sm.tsa.arima.ARIMA.__getnewargs__ = __getnewargs__

# 计算差分序列
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# load data
df = pd.read_csv('dataset.csv',  index_col=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

# prepare data
X = series.values
X = X.astype('float32')
# 数据差分
months_in_year = 12
diff = difference(X, months_in_year)
# 训练模型
model = sm.tsa.arima.ARIMA(diff, order=(0,0,1))
model_fit = model.fit()
# 偏差常数，可以从样本内平均残差计算得出
bias = 165.904728
# save model
model_fit.save('model.pkl')
# 将残差作为一个参数存储起来
numpy.save('model_bias.npy', [bias])