# Standardize time series data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import sqrt
# load the dataset and print the first 5 rows
df = pd.read_csv('daily-minimum-temperatures-in-me.csv',  index_col=0,  header=0, skiprows=1)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

print(series.head())
# 准备数据
values = series.values
values = values.reshape((len(values), 1))
# 定义标准化模型
scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
# 开始标准化，打印前五行
normalized = scaler.transform(values)
for i in range(5):
	print(normalized[i])
# 逆标准化数据
inversed = scaler.inverse_transform(normalized)
for i in range(5):
	print(inversed[i])