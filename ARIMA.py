import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Download data (Straits Times Index - ^STI)
data = yf.download('^STI', start='2020-01-01', end='2025-05-31')
data = data['Close'].dropna()

# Step 2: Plot original data
plt.figure(figsize=(10, 4))
plt.plot(data, label='STI Close Price')
plt.title('STI Stock Index (Singapore)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Step 3: Fit ARIMA model
model = ARIMA(data, order=(5, 1, 0))  # (p,d,q) can be tuned
model_fit = model.fit()

# Step 4: Forecast for 252 trading days (~1 year)
forecast = model_fit.forecast(steps=252)
forecast_index = pd.date_range(start='2025-01-01', periods=252, freq='B')

# Step 5: Plot forecast
plt.figure(figsize=(10, 4))
plt.plot(data[-100:], label='Historical (last 100 days)')
plt.plot(forecast_index, forecast, color='red', label='Forecast 2025')
plt.title('STI Forecast for 2025')
plt.xlabel('Date')
plt.ylabel('Forecasted Closing Price')
plt.legend()
plt.show()
