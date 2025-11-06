# Normalize time series data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# load the dataset and print the first 5 rows
df = pd.read_csv('daily-minimum-temperatures-in-me.csv',  index_col=0,  header=0, skiprows=1)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

print(series.head())
# 准备归一化数据
values = series.values
values = values.reshape((len(values), 1))
# 定义缩放范围(0,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
# 归一化数据集并打印前5行
normalized = scaler.transform(values)
for i in range(5):
	print(normalized[i])
# 逆变换并打印前5行
inversed = scaler.inverse_transform(normalized)
for i in range(5):
	print(inversed[i])