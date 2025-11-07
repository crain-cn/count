import yfinance as yf

# 下载黄金ETF（GLD）的全部历史数据
gold = yf.Ticker("GLD")
hist = gold.history(period="3mo")

# 保存至 gold.csv
hist.to_csv("gold.csv")
print("黄金历史数据已保存为 gold.csv")