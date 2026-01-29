# VaR / Expected Shortfall from GJR-GARCH

# Step A — importGARCGH
import numpy as np
import yfinance as yf
from arch import arch_model
from scipy.stats import t


# Step B — fit GJR-GARCH-t (STI)
sti = yf.download("^STI", start="2020-01-01", end="2026-01-12" , auto_adjust=True)

returns = 100 * np.log(sti["Close"]).diff().dropna()

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


alpha = 0.01

# Step C — extract model outputs



sigma_t = res.conditional_volatility.iloc[-1]
nu = res.params["nu"]

mu_t = 0.0

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


# Step D — VaR & ES computation (99%)
#alpha = 0.01  # 99% VaR

# Student-t quantile
#q_alpha = t.ppf(alpha, df=nu)

# Value at Risk
#VaR_99 = -(mu_t + sigma_t * q_alpha)

# Expected Shortfall
#pdf_q = t.pdf(q_alpha, df=nu)
#ES_99 = -(
    #mu_t
    #+ sigma_t
    #* (nu + q_alpha**2)
    #/ ((nu - 1) * alpha)
    #* pdf_q
#)

print(f"99% VaR (1-day): {VaR_99:.4f}%")
print(f"99% ES  (1-day): {ES_99:.4f}%")


# Model validation checks (important)
print("Persistence:",
      res.params["alpha[1]"]
      + res.params["beta[1]"]
      + 0.5 * res.params["gamma[1]"])


