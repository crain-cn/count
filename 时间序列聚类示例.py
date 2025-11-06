from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

# 加载示例数据集
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

# 创建时间序列聚类模型
model = TimeSeriesKMeans(n_clusters=3)
model.fit(X_train)

# 聚类预测
labels = model.predict(X_test)
pyplot.plot(labels)
pyplot.show()