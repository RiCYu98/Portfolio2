# Starting the GARCH

# Step A — Load STI data (no CSV needed)
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model

## Download STI data
sti = yf.download("^STI", start="2020-01-01", end="2026-01-12", auto_adjust=True)

## Log returns in percent
returns = 100 * np.log(sti["Close"]).diff().dropna()

# Step B — Specify Student-t GJR-GARCH
model = arch_model(
    returns,
    mean="AR",
    lags=1,
    vol="GARCH",
    p=1,
    o=1,     # GJR term (asymmetry)
    q=1,
    dist="t" # Student-t errors
)

res = model.fit(disp="off")

print(res.summary())

plt.figure(figsize=(10, 4))
plt.plot(res.conditional_volatility)
plt.title("STI Student-t GJR-GARCH Conditional Volatility")
plt.ylabel("Volatility (%)")
plt.grid(True)
plt.show()

#Step C — Fit the model
forecast = res.forecast(horizon=10)

var_forecast = forecast.variance.iloc[-1]
vol_forecast = np.sqrt(var_forecast)

print("10-day ahead volatility forecast:")
print(vol_forecast)

# Standard GARCH for comparison
garch = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")

print("GARCH AIC:", garch.aic)
print("GJR-t AIC:", res.aic)
