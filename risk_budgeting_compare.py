import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define a function to compute ERC weights using scipy.optimize
def compute_erc_weights(cov_matrix):
    n = cov_matrix.shape[0]

    def objective(w):
        marginal_contrib = cov_matrix @ w
        risk_contrib = w * marginal_contrib
        risk_diffs = risk_contrib - np.mean(risk_contrib)
        return np.sum(risk_diffs**2)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'ineq', 'fun': lambda w: w})

    initial_weights = 1 / np.sqrt(np.diag(cov_matrix))
    initial_weights /= initial_weights.sum()

    result = minimize(objective, initial_weights, constraints=constraints, method='SLSQP')
    return result.x

# Load your data (returns DataFrame should already be prepared)
data_path = 'risk-budgeting.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
returns = data.pct_change().dropna()

# Parameters for backtesting
window_size = 63  # 3 months rolling window (assuming ~21 trading days/month)
rebalance_freq = 126  # Rebalance semiannually (~6 months)

dates = returns.index
weights_history = []
weights_dates = []
portfolio_returns = []
equal_weight_returns = []

# Backtesting loop
for start in range(0, len(returns) - window_size, rebalance_freq):
    # Rolling window data
    window_data = returns.iloc[start:start + window_size]

    # Check if the rolling window contains sufficient data
    if window_data.empty or window_data.shape[0] < 2:
        print(f"Insufficient data for covariance calculation at index {start}. Skipping...")
        continue

    # Validate data for NaN or infinite values
    if window_data.isnull().values.any() or np.isinf(window_data.values).any():
        print(f"Data contains NaN or infinite values at index {start}. Skipping...")
        continue

    cov_matrix = window_data.cov()

    # Validate covariance matrix
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        print(f"Invalid covariance matrix dimensions at index {start}. Skipping...")
        continue

    if not np.allclose(cov_matrix, cov_matrix.T) or np.any(np.linalg.eigvals(cov_matrix) < 0):
        print(f"Covariance matrix is not valid at index {start}. Skipping...")
        continue

    # Add regularization to the covariance matrix
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8

    # Calculate ERC weights
    try:
        weights = compute_erc_weights(cov_matrix)
    except Exception as e:
        print(f"Optimization failed at index {start} due to: {e}")
        continue

    weights_history.append(weights)
    weights_dates.append(dates[start + window_size - 1])

    # Equal weights for comparison
    equal_weights = np.ones(len(returns.columns)) / len(returns.columns)

    # Apply weights for the next rebalance period
    for t in range(start + window_size, min(start + window_size + rebalance_freq, len(returns))):
        daily_return = np.dot(weights, returns.iloc[t].values)
        equal_weight_return = np.dot(equal_weights, returns.iloc[t].values)
        portfolio_returns.append(daily_return)
        equal_weight_returns.append(equal_weight_return)

# Convert portfolio returns to a DataFrame
if len(portfolio_returns) == 0:
    print("No portfolio returns were computed. Check data or backtesting logic.")
else:
    portfolio_returns = pd.Series(portfolio_returns, index=dates[window_size:len(portfolio_returns) + window_size])
    equal_weight_returns = pd.Series(equal_weight_returns, index=dates[window_size:len(equal_weight_returns) + window_size])

    # Calculate cumulative returns starting with $10,000
    initial_capital = 10000
    cumulative_returns = initial_capital * (1 + portfolio_returns).cumprod()
    cumulative_equal_returns = initial_capital * (1 + equal_weight_returns).cumprod()

    # Calculate performance metrics
    summary = {
        "Total Return (ERC)": (1 + portfolio_returns).prod() - 1,
        "Total Return (Equal Weight)": (1 + equal_weight_returns).prod() - 1,
        "Annualized Return (ERC)": (1 + portfolio_returns).prod()**(252 / len(portfolio_returns)) - 1,
        "Annualized Return (Equal Weight)": (1 + equal_weight_returns).prod()**(252 / len(equal_weight_returns)) - 1,
        "Annualized Volatility (ERC)": np.std(portfolio_returns) * np.sqrt(252),
        "Annualized Volatility (Equal Weight)": np.std(equal_weight_returns) * np.sqrt(252),
        "Sharpe Ratio (ERC)": (np.mean(portfolio_returns) / np.std(portfolio_returns)) * np.sqrt(252),
        "Sharpe Ratio (Equal Weight)": (np.mean(equal_weight_returns) / np.std(equal_weight_returns)) * np.sqrt(252),
    }

    # Display performance metrics
    print(summary)

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Portfolio (ERC)')
    plt.plot(cumulative_equal_returns, label='Portfolio (Equal Weight)', linestyle='--')
    plt.title('Portfolio Performance ($10,000 Initial Capital)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # Plot weights over time
    weights_df = pd.DataFrame(weights_history, index=weights_dates, columns=returns.columns)
    weights_df.plot(kind='line', figsize=(12, 6))
    plt.title('Asset Weights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend(title='Assets', loc='upper right')
    plt.grid()
    plt.show()
