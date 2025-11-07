import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
# 增加带交互的数据点悬停提示
import mplcursors

def parser(x):
    dt_index = pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
    return dt_index

df = pd.read_csv('gold.csv',  index_col=0,  header=0,  date_parser=parser)
indx = df.index
vals = [df.iloc[i, 0] for i in range(len(indx))]
series = pd.Series(vals, index=indx)

temps = DataFrame(series.values)
width = 2
shifted = temps.shift(width - 1)
print(shifted.head(12))

window = shifted.rolling(window=3)
# 分别取过去四天的最小值，均值，均方误差，平均绝对误差，最大值，正常值
# 增加移动加权均值 weighted_ma
weights = np.array([0.6, 0.3, 0.1])
def weighted_ma(x):
    if np.any(pd.isnull(x)):
        return np.nan
    return np.dot(x, weights)
weighted_ma_series = shifted.rolling(window=3).apply(weighted_ma, raw=True)
dataframe = concat([
    window.min(),
    window.mean(),
    temps - window.mean(),
    np.fabs(temps - window.mean()),
    window.max(),
    temps,
    weighted_ma_series
], axis=1)
dataframe.columns = ['min', 'mean', 'mse' , 'mae' , 'max', 't+1', 'weighted_ma']
print(dataframe.head(50))


# 贝叶斯网络分析辅助 window.mean & weighted_ma_series（离散化后用 *_bin 字段训练和推断）
num_bins = 10
bn_df = pd.DataFrame({
    't+1': series.values.flatten(),
    'ma': window.mean().values.flatten(),
    'weighted_ma': weighted_ma_series.values.flatten()
}).dropna()

# 离散化
bn_df['t+1_bin'], bin_edges = pd.cut(bn_df['t+1'], bins=num_bins, labels=False, retbins=True, duplicates='drop')
bn_df['ma_bin'] = pd.cut(bn_df['ma'], bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')
bn_df['weighted_ma_bin'] = pd.cut(bn_df['weighted_ma'], bins=bin_edges, labels=False, include_lowest=True, duplicates='drop')

# 只用离散特征建模
bn_df_discrete = bn_df[['t+1_bin', 'ma_bin', 'weighted_ma_bin']]

# 贝叶斯结构学习及参数估计
from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
hc = HillClimbSearch(bn_df_discrete)
best_model = hc.estimate(scoring_method=K2Score(bn_df_discrete))
print("贝叶斯网络结构:", best_model.edges())

model = BayesianNetwork(best_model.edges())
model.fit(bn_df_discrete, estimator=MaximumLikelihoodEstimator)

for cpd in model.get_cpds():
    print(cpd)

# 可视化 t+1、ma、weighted_ma 三组数据
# plt.figure(figsize=(10,6))
# plt.plot(series.index, series.values, label='t+1', linestyle='-')
# plt.plot(series.index, window.mean(), label='ma', linestyle='--')
# plt.plot(series.index, weighted_ma_series, label='weighted_ma', linestyle=':')
# plt.legend()
# plt.title('t+1, MA, Weighted MA 对比')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10,6))
# X_full = np.arange(len(series)).reshape(-1, 1)
#
# # t+1 trend
# plt.plot(series.index, series.values, marker='o', label='t+1 trend', linestyle='-')
#
# # ma trend（去NaN）
# plt.plot(series.index, window.mean(), marker='o', label='ma trend', linestyle='--')
#
# # weighted_ma trend（去NaN）
# plt.plot(series.index, weighted_ma_series, marker='o', label='weighted_ma trend', linestyle=':')
#
# plt.legend()
#
#
# mplcursors.cursor(hover=True)
#
# pyplot.show()


# -----------------------------
# 贝叶斯修正预测 t+1（根据bin列）
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)

# 用bin中点代表值
bin_means = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]

# 贝叶斯修正ma：P(t+1_bin | ma_bin)，概率加权均值
preds_ma = []
for mb in bn_df['ma_bin'].astype(int):
    q = infer.query(variables=['t+1_bin'], evidence={'ma_bin': int(mb)}, show_progress=False)
    cur_bin_means = bin_means[:len(q.values)]
    pred = np.sum(q.values * cur_bin_means)
    preds_ma.append(pred)

# 贝叶斯修正weighted_ma：P(t+1_bin | weighted_ma_bin)，概率加权均值
preds_wma = []
for wb in bn_df['weighted_ma_bin'].astype(int):
    q = infer.query(variables=['t+1_bin'], evidence={'weighted_ma_bin': int(wb)}, show_progress=False)
    cur_bin_means = bin_means[:len(q.values)]
    pred = np.sum(q.values * cur_bin_means)
    preds_wma.append(pred)


ts_true = bn_df['t+1'].values
valid_idx = bn_df.index

# 绘图对比
plt.figure(figsize=(10,6))
plt.plot(valid_idx, ts_true, label=' t+1', linestyle='-', marker='o')
plt.plot(valid_idx, preds_ma, label='MA', linestyle='--', marker='s')
plt.plot(valid_idx, preds_wma, label='Weighted MA', linestyle=':', marker='^')
plt.legend()
plt.title('line')
plt.xlabel('Date Index')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
mplcursors.cursor(hover=True)
plt.show()


