# monte_carlo_sgx_option.py
# Python 3.13 compatible
import numpy as np
import pandas as pd
import yfinance as yf
from math import log, sqrt, exp
from scipy.stats import norm

# ---------------------------
# User inputs (editable)
# ---------------------------
ticker = "D05.SI"          # DBS on SGX (change to any SGX ticker)
start_hist = "2024-01-01"  # historical window start (for sigma estimate)
end_hist   = "2025-05-31"  # historical window end (use last trading day in July 2025)
S0_override = None         # optionally override S0 from market (SGD), None => use last close
strike = 50.0              # strike in SGD (edit as needed)
T = 0.5                    # time to maturity in years (e.g., 6 months)
r_market = 0.0168          # risk-free rate (annual) ~1.68% from 1-yr T-bill in July 2025
n_sims = 200_000           # number of Monte Carlo paths (vectorized)
seed = 2025                # RNG seed for reproducibility
option_type = "call"       # "call" or "put"

# ---------------------------
# Helper functions
# ---------------------------
def black_scholes_price(S, K, T, r, sigma, option="call"):
    """Return Black-Scholes price for European call/put (non-dividend paying)."""
    if T <= 0:
        return max(S-K, 0.0) if option=="call" else max(K-S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ---------------------------
# 1) Fetch price history and compute S0 & historical vol
# ---------------------------
print("Fetching historical prices for", ticker)
df = yf.download(ticker, start=start_hist, end=end_hist, progress=True)

if df.empty:
    raise RuntimeError("No historical data fetched. Check ticker and network.")

last_close = df["Close"].iloc[-1]
S0 = S0_override if (S0_override is not None) else float(last_close)
print(f"Using S0 = {S0:.4f} (last close on {df.index[-1].date()})")

# compute daily log returns and annualized realized volatility (sample std)
df["logret"] = np.log(df["Close"] / df["Close"].shift(1)) # using logarithm function inside numpy
hist_days = df["logret"].dropna()
trading_days = 252.0
sigma_hat = hist_days.std(ddof=1) * np.sqrt(trading_days)
print(f"Estimated annualized volatility (from {len(hist_days)} returns): {sigma_hat:.4%}")

# ---------------------------
# 2) Monte Carlo with variance reduction
#    - Antithetic variates
#    - Control variate: use Black-Scholes analytic price (since european option)
# ---------------------------
np.random.seed(seed)
M = n_sims // 2  # we'll generate M normals and create antithetic pairs => n_sims total
Z = np.random.randn(M)
Z_full = np.concatenate([Z, -Z])  # antithetic

# simulate terminal prices under risk-neutral measure (GBM)
mu = (r_market - 0.5 * sigma_hat**2) * T
sigma_sqrtT = sigma_hat * np.sqrt(T)
ln_ST = np.log(S0) + mu + sigma_sqrtT * Z_full
ST = np.exp(ln_ST)

# compute payoffs
if option_type == "call":
    payoffs = np.maximum(ST - strike, 0.0)
else:
    payoffs = np.maximum(strike - ST, 0.0)

# discount
discount_factor = np.exp(-r_market * T)
discounted_payoffs = discount_factor * payoffs

# basic Monte Carlo estimate
mc_mean = discounted_payoffs.mean()
mc_std_error = discounted_payoffs.std(ddof=1) / np.sqrt(n_sims)
ci_low = mc_mean - 1.96*mc_std_error
ci_high = mc_mean + 1.96*mc_std_error

# ---------------------------
# 3) Control variate (use analytic BS price as control)
#    Derivation: use ST-based control variable = discounted payoff of geometric or
#    use price difference between MC and BS closed-form with same S0,sigma_hat,r,T.
#    Here use simple regression-free control variate with analytic BS (zero-mean adjust).
# ---------------------------
bs_price = black_scholes_price(S0, strike, T, r_market, sigma_hat, option_type)
# control variate technique (simple): adjust estimator by (BS - sample_bs_estimate)
# compute sample estimate of option via analytical function of ST? Simpler approach:
# compute expectation of (discounted payoff - analytic d payoff with same ST) is complex.
# Instead do linear control variate using ST (its known E[ST]=S0*exp(rT))
# use control X = ST (not discounted). Known E[X] = S0*exp(r_market*T)
X = ST
X_mean = X.mean()
X_true = S0 * np.exp(r_market * T)
cov = np.cov(discounted_payoffs, X, ddof=1)[0,1]
var_X = X.var(ddof=1)
b_hat = cov / var_X

# adjusted estimator
mc_cv = mc_mean - b_hat * (X_mean - X_true) * discount_factor  # discount factor applied earlier -> keep consistent
# standard error approximation (delta method)
adj_payoffs = discounted_payoffs - b_hat * discount_factor * (X - X_true)
mc_cv_se = adj_payoffs.std(ddof=1) / np.sqrt(n_sims)
ci_cv_low, ci_cv_high = mc_cv - 1.96*mc_cv_se, mc_cv + 1.96*mc_cv_se

# ---------------------------
# 4) Output summary
# ---------------------------
print("\nMonte Carlo estimate (plain antithetic):")
print(f"  Price = {mc_mean:.6f} SGD")
print(f"  Std error = {mc_std_error:.6f}")
print(f"  95% CI = [{ci_low:.6f}, {ci_high:.6f}]")

print("\nMonte Carlo with control variate adjustment:")
print(f"  Price (CV) = {mc_cv:.6f} SGD")
print(f"  Std error (CV) = {mc_cv_se:.6f}")
print(f"  95% CI (CV) = [{ci_cv_low:.6f}, {ci_cv_high:.6f}]")

print("\nBlack-Scholes benchmark (S0, sigma_hat, r):")
print(f"  BS Price = {bs_price:.6f} SGD")

# Save results to CSV for your resume/report
out = {
    "S0": S0, "strike": strike, "T": T, "r": r_market, "sigma_hat": sigma_hat,
    "MC_price": mc_mean, "MC_se": mc_std_error, "MC_CI_low": ci_low, "MC_CI_high": ci_high,
    "MC_CV_price": mc_cv, "MC_CV_se": mc_cv_se, "MC_CV_CI_low": ci_cv_low, "MC_CV_CI_high": ci_cv_high,
    "BS_price": bs_price
}
pd.DataFrame([out]).to_csv("mc_option_result.csv", index=False)
print("\nResults saved to mc_option_result.csv")
