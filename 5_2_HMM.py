import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# Step 1: Data Collection - Get financial data (e.g., S&P 500, Apple stock, etc.)
ticker = 'AAPL'  # You can replace this with any stock ticker symbol
start_date = '2010-01-01'
end_date = '2024-01-01'

# Download historical stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Preprocessing - Calculate daily returns (log returns or percentage returns)
data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

# Drop NaN values resulting from the shift
data.dropna(inplace=True)

# Step 3: Fit the Gaussian HMM to the returns data
# We assume there are 2 hidden states, representing two regimes (e.g., low volatility and high volatility)
n_components = 2  # Number of hidden states (regimes)
model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)

# Reshape the data to fit the HMM model (it expects a 2D array)
returns = data['Log_Returns'].values.reshape(-1, 1)

# Fit the model
model.fit(returns)

# Step 4: Predict the hidden states (market regimes) using the trained HMM model
hidden_states = model.predict(returns)

# Step 5: Visualization of the Results
# Plot the stock price and the predicted market regimes
plt.figure(figsize=(14, 7))

# Plot the stock price
plt.subplot(2, 1, 1)
plt.plot(data.index, data['Close'], label='Stock Price (AAPL)', color='blue')
plt.title('AAPL Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)

# Plot the hidden states (market regimes)
plt.subplot(2, 1, 2)
plt.plot(data.index, hidden_states, label='Hidden States (Market Regimes)', color='green')
plt.title('Predicted Market Regimes (Hidden States)')
plt.xlabel('Date')
plt.ylabel('Hidden State')
plt.grid(True)

plt.tight_layout()
plt.show()

# Step 6: Interpretation of Results
# You can analyze the mean and covariance of each state
print("Means of the hidden states:")
print(model.means_)

print("\nCovariances of the hidden states:")
print(model.covars_)

# Visualizing the distribution of returns for each state
plt.figure(figsize=(14, 7))
plt.hist(returns[hidden_states == 0], bins=50, alpha=0.5, label="State 0 (Low Volatility)", color='red')
plt.hist(returns[hidden_states == 1], bins=50, alpha=0.5, label="State 1 (High Volatility)", color='blue')
plt.title('Distribution of Returns in Different Market Regimes')
plt.xlabel('Log Returns')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.grid(True)
plt.show()
