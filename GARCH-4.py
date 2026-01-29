# Step A — Fit GJR-GARCH-t to STI
import numpy as np
import yfinance as yf
from arch import arch_model
from scipy.stats import norm

# # Load STI data
sti = yf.download("^STI", start="2020-01-01",end="2026-01-22", auto_adjust=True)
returns = 100 * np.log(sti["Close"]).diff().dropna()

# GJR-GARCH with Student-t
model = arch_model(
    returns,
    mean="AR",
    lags=1,
    vol="GARCH",
    p=1,
    o=1,
    q=1,
    dist="t"
)

res = model.fit(disp="off")

# Step B — Forecast volatility to option maturity
T = 0.25
days = int(252 * T)

forecast = res.forecast(horizon=days)
daily_var = forecast.variance.iloc[-1].values

# Annualised volatility
sigma_garch = np.sqrt(daily_var.sum()) / 100

# Step C — Black–Scholes pricing using GARCH σ
def black_scholes_call(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Step D — Price STI option
S0 = sti["Close"].iloc[-1].item()  # current STI
K = S0                      # ATM option
r = 0.03                    # Singapore risk-free (approx)

call_price = black_scholes_call(
    S=S0,
    K=K,
    r=r,
    T=T,
    sigma=sigma_garch
)

print(f"GJR-GARCH BS Call Price: {call_price:.2f}")

