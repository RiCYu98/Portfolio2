"""
black_scholes_real_option.py
Python 3.13 - Black-Scholes pricing + implied volatility + optional yfinance fetch for real option (e.g. July 2025 expiry)

Usage:
 - Edit the example parameters at the bottom (ticker, expiry_date, strike, option_type).
 - Or call functions from another script / REPL.

Author: ChatGPT (rigorous implementation)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton

# Optional: yfinance for live data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

# ---------------------------
# Black-Scholes core functions
# ---------------------------

def year_fraction(from_date: date, to_date: date) -> float:
    """Actual/365 day count for year fraction (simple)."""
    delta_days = (to_date - from_date).days
    return max(delta_days / 365.0, 0.0)

def d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float]:
    if sigma <= 0 or T <= 0:
        # avoid division by zero; returned d1/d2 won't be used for sigma=0 in formula
        return (float("inf"), float("inf"))
    sroot = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sroot
    d2 = d1 - sroot
    return d1, d2

def black_scholes_price(S: float, K: float, r: float, q: float, sigma: float, T: float, option_type: str = "call") -> float:
    """
    Compute European Black-Scholes price.
    S: spot price
    K: strike
    r: continuously compounded risk-free rate (decimal)
    q: continuous dividend yield (decimal)
    sigma: volatility (annual, decimal)
    T: time to expiry in years (>=0)
    option_type: "call" or "put"
    """
    if T <= 0:
        # Option at or past expiry -> intrinsic value
        if option_type.lower() == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    if sigma == 0:
        # In the limit sigma -> 0, option value is discounted intrinsic for deterministic stock drift =>
        # but under BS assumptions, this degenerates; we return intrinsic discounted appropriately:
        if option_type.lower() == "call":
            return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        else:
            return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)

    d1, d2 = d1_d2(S, K, r, q, sigma, T)
    if option_type.lower() == "call":
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        # put-call parity or direct formula
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return float(price)

def bs_vega(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Vega = partial derivative of price wrt sigma. Returns per 1.0 vol (i.e., scaling)."""
    if T <= 0:
        return 0.0
    d1, _ = d1_d2(S, K, r, q, sigma, T)
    return float(S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T))

# ---------------------------
# Implied volatility solvers
# ---------------------------

def implied_volatility(market_price: float, S: float, K: float, r: float, q: float, T: float,
                       option_type: str = "call", tol: float = 1e-8, maxiter: int = 200) -> float:
    """
    Compute implied volatility by solving Black-Scholes(price, sigma) = market_price.
    Uses Brent solver on [1e-12, 5.0] as a robust bracket (0% to 500% vol).
    Fallback to Newton if bracket fails.
    """
    if market_price <= 0:
        return 0.0

    def f(sigma):
        return black_scholes_price(S, K, r, q, sigma, T, option_type) - market_price

    # brackets: low_vol, high_vol
    low_vol = 1e-12
    high_vol = 5.0  # 500% annual vol is wide
    try:
        # ensure f(low)*f(high) < 0; if not, expand high
        f_low, f_high = f(low_vol), f(high_vol)
        if f_low * f_high > 0:
            # try increasing high_vol
            for hv in (10.0, 20.0, 50.0):
                f_high = f(hv)
                if f_low * f_high <= 0:
                    high_vol = hv
                    break

        sigma_imp = brentq(f, low_vol, high_vol, xtol=tol, maxiter=maxiter)
        return float(sigma_imp)
    except Exception:
        # Fallback to Newton with initial guess
        sigma = 0.2 if S > 0 else 0.2
        for i in range(maxiter):
            price = black_scholes_price(S, K, r, q, sigma, T, option_type)
            diff = price - market_price
            if abs(diff) < tol:
                return float(max(sigma, 0.0))
            v = bs_vega(S, K, r, q, sigma, T)
            if v == 0:
                break
            sigma = sigma - diff / v
            if sigma <= 0 or sigma > 200:
                sigma = max(min(sigma, 200), 1e-12)
        raise RuntimeError("Implied vol did not converge")

# ---------------------------
# Historical volatility estimator
# ---------------------------

def historical_volatility_from_prices(prices: np.ndarray, window_days: int = None) -> float:
    """
    Estimate annualized historical volatility from a series of prices.
    Uses log returns: annualized stdev = std(log returns) * sqrt(252).
    If window_days provided, use the last window_days data points (days).
    """
    if window_days is not None:
        prices = prices[-window_days:]
    if len(prices) < 2:
        return 0.0
    logrets = np.diff(np.log(prices))
    # sample standard deviation
    sigma_daily = np.std(logrets, ddof=1)
    # annualize assuming 252 trading days (or 365 if you prefer)
    sigma_annual = sigma_daily * math.sqrt(252)
    return float(sigma_annual)

# ---------------------------
# Optional: fetch real data via yfinance
# ---------------------------

def fetch_underlying_and_options(ticker: str, expiry_dt: date) -> dict:
    """
    Use yfinance to fetch:
      - last underlying price
      - option chain for the expiry date (if available)
    Returns dictionary with 'spot', 'options' (DataFrame-like), 'expiry_found' boolean.
    Requires yfinance installed.
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance not installed or failed to import. pip install yfinance")

    t = yf.Ticker(ticker)
    info = t.history(period="1d")
    if info.empty:
        raise RuntimeError(f"No price data for {ticker}")
    spot = float(info['Close'].iloc[-1])

    # option expirations available
    exps = t.options  # list of expiry strings like '2025-07-18'
    expiry_str = expiry_dt.strftime("%Y-%m-%d")
    if expiry_str not in exps:
        # try other formats or find nearest
        expiry_found = False
        options_df = None
    else:
        expiry_found = True
        opt_chain = t.option_chain(expiry_str)
        # returns a namedtuple with calls, puts DataFrame
        options_df = {
            "calls": opt_chain.calls,
            "puts": opt_chain.puts
        }
    return {"spot": spot, "expiry_found": expiry_found, "options": options_df, "available_expiries": exps}

# ---------------------------
# Example usage: real option in July 2025
# ---------------------------

if __name__ == "__main__":
    # Example parameters (edit these to your real option)
    ticker = "AAPL"                       # change to your ticker
    expiry_str = "2025-07-18"             # typical 3rd Friday; change to exact expiration you care about
    strike = 170.0                        # change strike
    option_type = "call"                  # "call" or "put"
    # valuation date (when you price it)
    valuation_date = date(2025, 7, 1)     # example: 1 July 2025
    expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()

    # User inputs / assumptions (edit)
    r = 0.05      # continuous risk-free rate (5% p.a.) - change to real short-rate if available
    q = 0.0       # continuous dividend yield (set e.g. 0.005 or stock's yield)
    # If you don't know sigma, you can estimate it from historical prices below, or provide implied vol if known:
    sigma_assumed = None  # if None, we'll attempt to estimate via historical vol (if yfinance available)

    # Compute time to expiry in years
    T = year_fraction(valuation_date, expiry_date)
    print(f"Valuation date: {valuation_date}, Expiry date: {expiry_date}, T = {T:.6f} years")
    print("----")

    
    # Try to fetch real spot & option chain (optional)
    spot_price = None
    if HAS_YFINANCE:
        try:
            fetch = fetch_underlying_and_options(ticker, expiry_date)
            spot_price = fetch["spot"]
            print(f"Fetched spot for {ticker}: {spot_price:.4f}")
            print("----")
            if not fetch["expiry_found"]:
                print(f"Expiry {expiry_str} not found in yfinance option chain list. Available expiries: {fetch['available_expiries']}")
                print("------")
            else:
                print("Option chain for expiry retrieved (calls and puts DataFrames).")
                print("------")
        except Exception as e:
            print("Warning: yfinance fetch failed:", e)
            print("-----")
    else:
        print("yfinance not available; skipping live fetch. Set HAS_YFINANCE by installing yfinance to fetch live data.")
        print("-----")

    # If no spot fetched, ask user (or set example)
    if spot_price is None:
        spot_price = 172.50  # fallback examplprint("/n")e spot price; replace with real spot
        print(f"Using fallback spot: {spot_price}")
        print("-----")

    # If sigma not provided, estimate from historical (attempt using yfinance if available)
    sigma = sigma_assumed
    if sigma is None and HAS_YFINANCE:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")['Close'].to_numpy()
            sigma = historical_volatility_from_prices(hist)
            print(f"Estimated historical vol (annual) from 1y data: {sigma:.4%}")
            print("----")
        except Exception as e:
            print("Historical vol estimation failed:", e)
            print("/n")
    if sigma is None:
        sigma = 0.25
        print(f"Using fallback sigma = {sigma:.4%}")
        print("/n")

    # Compute theoretical BS price
    bs_price = black_scholes_price(spot_price, strike, r, q, sigma, T, option_type)
    print(f"Black-Scholes {option_type} price (S={spot_price}, K={strike}, r={r}, q={q}, sigma={sigma:.4%}, T={T:.6f}) = {bs_price:.4f}")

    # Example: if you have a market option price and want implied vol:
    # market_price = 7.30
    # try:
    #     impv = implied_volatility(market_price, spot_price, strike, r, q, T, option_type)
    #     print(f"Implied volatility from market price {market_price} = {impv:.4%}")
    # except Exception as e:
    #     print("Implied vol solve failed:", e)

    # End of script
