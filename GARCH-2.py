# Add Student-t / GJR-GARCH for STI


import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

# STI index (Singapore)
sti = yf.download("^STI", start="2020-01-01", end="2026-01-12", auto_adjust=True)

# Log returns (% scale)
returns = 100 * np.log(sti["Close"]).diff().dropna()

# GARCH(1,1)
model = arch_model(
    returns,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

res = model.fit(disp="off")
print(res.summary())

# Plot conditional volatility
plt.figure(figsize=(10, 4))
plt.plot(res.conditional_volatility)
plt.title("STI GARCH(1,1) Conditional Volatility")
plt.grid(True)
plt.show()
