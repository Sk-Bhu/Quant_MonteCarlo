!pip install numpy pandas scikit-learn matplotlib yfinance
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

# Step 1: Monte Carlo Simulation for Stock Price Forecasting
def simulate_stock_price(S0, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    prices = np.zeros((n_steps, n_simulations))
    prices[0] = S0
    for t in range(1, n_steps):
        Z = np.random.standard_normal(n_simulations)
        prices[t] = prices[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return prices

# Step 2: Black-Scholes Option Pricing Formula
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Step 3: Machine Learning Predictions
def load_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    return data.dropna()

def prepare_features(data, window_size=30):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data['Returns'].iloc[i-window_size:i].values)
        y.append(data['Close'].iloc[i])
    return np.array(X), np.array(y)

# Step 4: Combine Simulated Results with Machine Learning
if __name__ == "__main__":
    # Parameters for Monte Carlo Simulation
    S0 = 100      # Initial stock price
    K = 105       # Strike price
    T = 1         # Time to maturity (in years)
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility
    n_simulations = 1000
    n_steps = 252 # Number of steps (daily)

    # Simulate stock prices
    stock_prices = simulate_stock_price(S0, T, r, sigma, n_simulations, n_steps)
    
    # Price the options using Black-Scholes formula
    option_prices = [black_scholes_call(S0, K, T, r, sigma) for S in stock_prices[-1]]

    # Plot simulated stock price paths
    plt.plot(stock_prices)
    plt.title('Monte Carlo Simulation of Stock Price Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.show()

    # Download and preprocess stock data (using Apple stock just as an example)
    stock_data = load_stock_data('AAPL', start='2020-01-01', end='2024-01-01')

    # Prepare features and targets
    X, y = prepare_features(stock_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a machine learning model (Random Forest)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Predict and evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.title('Machine Learning Model Predictions')
    plt.xlabel('Test Samples')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()

    # Predict option price for the next day using the machine learning model
    predicted_stock_price = model.predict(X_test[-1].reshape(1, -1))[0]
    predicted_option_price = black_scholes_call(predicted_stock_price, K, T, r, sigma)
    print(f'Predicted Option Price: {predicted_option_price}')
