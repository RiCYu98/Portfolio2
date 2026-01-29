# need GARCH to Black-Scholes by the start = "2020-01-01", end="2026-01-12" for
# 1. Data: STI with fixed time window
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib as plt
from scipy.stats import t
from scipy.stats import norm

# Download STI data for the required period
sti = yf.download(
    "^STI",
    start="2020-01-01",
    end="2026-01-20",
    auto_adjust=True
)

# Daily log-returns in percent
returns = 100 * np.log(sti["Close"]).diff().dropna()

# 2. Student-t GJR-GARCH model (correct specification)
model = arch_model(
    returns,
    mean="Zero",      # ← bank-standard for VaR / ES
    vol="GARCH",
    p=1,
    o=1,              # GJR asymmetry
    q=1,
    dist="t"          # Student-t innovations
)

res = model.fit(disp="off")
print(res.summary())

# 3. VaR and Expected Shortfall (1-day, 99%)
sigma_t = res.conditional_volatility.iloc[-1]  # last conditional σ (%)
nu = res.params["nu"]
mu_t = 0.0                                     # intentional

alpha = 0.01  # 99%

q_alpha = t.ppf(alpha, df=nu)

VaR_99 = -(mu_t + sigma_t * q_alpha)

pdf_q = t.pdf(q_alpha, df=nu)
ES_99 = -(
    mu_t
    + sigma_t
    * (nu + q_alpha**2)
    / ((nu - 1) * alpha)
    * pdf_q
)

print(f"99% 1-day VaR: {VaR_99:.4f}%")
print(f"99% 1-day ES : {ES_99:.4f}%")

# 4. GJR-GARCH volatility → Black–Scholes
## 4.1 Forecast volatility to option maturity
T = 0.25
days = int(252 * T)

forecast = res.forecast(horizon=days)
daily_var = forecast.variance.iloc[-1].values

# Annualised forward volatility (decimal)
sigma_garch = np.sqrt(daily_var.sum()) / 100

## 4.2 Black–Scholes pricing function
def black_scholes_call(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# 4.3 Price an STI option using GJR-GARCH volatility
S0 = float(sti["Close"].iloc[-1])  # ensure scalar

sigma_daily = np.sqrt(res.conditional_variance.iloc[-1])
sigma_annual = sigma_daily * np.sqrt(252)
print(f"GJR-GARCH Annualised Volatility: {sigma_annual/100:.2%}")

K = S0                            # ATM
r = 0.03    
T = 0.5                          # SGD risk-free (approx.)

call_price = black_scholes_call(
    S=S0,
    K=K,
    r=r,
    T=T,
    sigma=sigma_garch
)

print(f"GJR-GARCH Black-Scholes Call Price: {call_price:.2f}")
