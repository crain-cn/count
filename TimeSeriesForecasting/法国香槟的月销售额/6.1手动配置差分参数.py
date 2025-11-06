import pandas as pd
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot

# 根据给出的步长interval，创建一个差分序列，间隔12个数字相减
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return  pd.Series(diff)


df = pd.read_csv('dataset.csv',  index_col=0)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

"""
1964-01-01    2815
1964-02-01    2672
1964-03-01    2755
1964-04-01    2721
1964-05-01    2946
"""
X = series.values
rc = pd.read_csv('dataset.csv')
indx = rc.index
vals = [rc.iloc[i, 0] for i in range(len(indx))]
r = pd.Series(vals, index=indx)
"""
   1964-01-01  2815
0  1964-02-01  2672
1  1964-03-01  2755
2  1964-04-01  2721
3  1964-05-01  2946
4  1964-06-01  3036
"""
r = r.values
X = X.astype('float32')
# difference data
months_in_year = 12
stationary = difference(X, months_in_year)
stationary.index = series.index[months_in_year:]
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv')
# plot
stationary.plot()
pyplot.show()