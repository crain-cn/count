import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 创建一个简单的时间序列
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')


data = pd.Series(range(len(date_rng)), index=date_rng)

print(data)
# 拟合 ARIMA 模型
model = ARIMA(data, order=(5,1,0))
model_fit = model.fit()

# 预测未来值
forecast = model_fit.forecast(steps=30)

# 可视化原始数据和预测结果
plt.figure(figsize=(10,6))
plt.plot(data, label='Observed')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.show()
