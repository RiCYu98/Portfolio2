# Statistical Arbitrage (Pairs Trading) with AAPL & NVDA
# Compatible with Python 3.13

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd

tickers = ["AAPL", "AMZN"]
data = yf.download(tickers, start="2020-01-01", end="2025-01-01")

#  Case 1: MultiIndex columns (usual when multiple tickers)
if isinstance(data.columns, pd.MultiIndex):
    if "Adj Close" in data.columns.levels[0]:
        data = data["Adj Close"].copy()
    elif "Close" in data.columns.levels[0]:
        data = data["Close"].copy()  # fallback if only Close exists
    else:
        raise KeyError(f"Available top-level columns: {data.columns.levels[0]}")

#  Case 2: SingleIndex columns (rare case)
else:
    if "Adj Close" in data.columns:
        data = data[["Adj Close"]].copy()
    elif "Close" in data.columns:
        data = data[["Close"]].copy()
    else:
        raise KeyError(f"Available columns: {data.columns}")

#  Rename to ticker symbols
data.columns = tickers
data.dropna(inplace=True)

print(data.head())


# Extract series
dbs = data["AAPL"]
ocbc = data["AMZN"]

# -------------------------------
# 2. Cointegration Test
# -------------------------------
score, pvalue, _ = coint(dbs, ocbc)
print("Cointegration test p-value:", pvalue)

# -------------------------------
# 3. Hedge Ratio (OLS Regression)
# -------------------------------
model = sm.OLS(dbs, sm.add_constant(ocbc)).fit()
hedge_ratio = model.params[1]
print("Hedge ratio:", hedge_ratio)

# -------------------------------
# 4. Build Spread & Z-Score
# -------------------------------
spread = dbs - hedge_ratio * ocbc
zscore = (spread - spread.mean()) / spread.std()

# -------------------------------
# 5. Generate Trading Signals
# -------------------------------
signals = pd.DataFrame(index=data.index)
signals["zscore"] = zscore
signals["longs"] = signals["zscore"] < -1
signals["shorts"] = signals["zscore"] > 1
signals["exits"] = abs(signals["zscore"]) < 0.5

# -------------------------------
# 6. Backtesting Positions
# -------------------------------
positions = pd.DataFrame(index=signals.index)
positions["dbs"] = 0
positions["ocbc"] = 0

# Long spread (buy AAPL, short NVDA)
positions.loc[signals["longs"], "dbs"] = 1
positions.loc[signals["longs"], "ocbc"] = -hedge_ratio

# Short spread (sell AAPL, buy NVDA)
positions.loc[signals["shorts"], "dbs"] = -1
positions.loc[signals["shorts"], "ocbc"] = hedge_ratio

# -------------------------------
# 7. Calculate Returns
# -------------------------------
returns = (positions.shift(1) * data.pct_change()).sum(axis=1)
cumulative_returns = (1 + returns).cumprod()

# -------------------------------
# 8. Plot Results
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(cumulative_returns, label="Strategy")
plt.plot((1 + data.pct_change().mean(axis=1)).cumprod(), label="Market Avg", alpha=0.7)
plt.title("Statistical Arbitrage: AAPL vs AMZN")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 9. Performance Metrics
# -------------------------------
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
