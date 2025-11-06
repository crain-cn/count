import tssearch as ts
import numpy as np

# 创建一组示例时间序列
np.random.seed(0)
series = [np.random.rand(100) for _ in range(10)]

# 创建时间序列搜索对象
search = ts.search.TimeSeriesSearch(series)

# 查询与第一个时间序列相似的序列
similar_sequences = search.query(series[0], threshold=0.1)
pyplot.plot(similar_sequences)
pyplot.show()