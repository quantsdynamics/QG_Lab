import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Define a function to compute ERC weights using VaR as risk contribution
def compute_erc_weights_var(cov_matrix, mean_returns, confidence_level=0.05):
    n = cov_matrix.shape[0]

    def risk_contribution(weights):
        portfolio_variance = weights @ cov_matrix @ weights
        z_score = norm.ppf(confidence_level)
        marginal_contrib = cov_matrix @ weights
        rc = weights * (mean_returns + z_score * marginal_contrib / np.sqrt(portfolio_variance))
        return rc

    def objective(w):
        rc = risk_contribution(w)
        risk_diffs = rc - np.mean(rc)
        return np.sum(risk_diffs**2)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'ineq', 'fun': lambda w: w})

    initial_weights = np.ones(n) / n

    result = minimize(objective, initial_weights, constraints=constraints, method='SLSQP')
    return result.x

# Define a function to compute mean-variance weights (max Sharpe ratio)
def compute_mvo_weights(cov_matrix, mean_returns, risk_free_rate=0.0):
    n = len(mean_returns)

    def objective(w):
        portfolio_return = np.dot(w, mean_returns)
        portfolio_var = np.dot(w.T, np.dot(cov_matrix, w))
        return -(portfolio_return - risk_free_rate) / np.sqrt(portfolio_var)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'ineq', 'fun': lambda w: w})

    initial_weights = np.ones(n) / n

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

# Backtesting variables
dates = returns.index
weights_history_erc_var = []
weights_history_mvo = []
weights_dates = []
portfolio_returns_erc_var = []
portfolio_returns_mvo = []
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
    mean_returns = window_data.mean()

    # Validate covariance matrix
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        print(f"Invalid covariance matrix dimensions at index {start}. Skipping...")
        continue

    if not np.allclose(cov_matrix, cov_matrix.T) or np.any(np.linalg.eigvals(cov_matrix) < 0):
        print(f"Covariance matrix is not valid at index {start}. Skipping...")
        continue

    # Add regularization to the covariance matrix
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8

    # Calculate ERC (using VaR) and MVO weights
    try:
        weights_erc_var = compute_erc_weights_var(cov_matrix, mean_returns)
        weights_mvo = compute_mvo_weights(cov_matrix, mean_returns)
    except Exception as e:
        print(f"Optimization failed at index {start} due to: {e}")
        continue

    weights_history_erc_var.append(weights_erc_var)
    weights_history_mvo.append(weights_mvo)
    weights_dates.append(dates[start + window_size - 1])

    # Equal weights for comparison
    equal_weights = np.ones(len(returns.columns)) / len(returns.columns)

    # Apply weights for the next rebalance period
    for t in range(start + window_size, min(start + window_size + rebalance_freq, len(returns))):
        daily_return_erc_var = np.dot(weights_erc_var, returns.iloc[t].values)
        daily_return_mvo = np.dot(weights_mvo, returns.iloc[t].values)
        equal_weight_return = np.dot(equal_weights, returns.iloc[t].values)

        portfolio_returns_erc_var.append(daily_return_erc_var)
        portfolio_returns_mvo.append(daily_return_mvo)
        equal_weight_returns.append(equal_weight_return)

# Convert portfolio returns to a DataFrame
if len(portfolio_returns_erc_var) == 0:
    print("No portfolio returns were computed. Check data or backtesting logic.")
else:
    portfolio_returns_erc_var = pd.Series(portfolio_returns_erc_var, index=dates[window_size:len(portfolio_returns_erc_var) + window_size])
    portfolio_returns_mvo = pd.Series(portfolio_returns_mvo, index=dates[window_size:len(portfolio_returns_mvo) + window_size])
    equal_weight_returns = pd.Series(equal_weight_returns, index=dates[window_size:len(equal_weight_returns) + window_size])

    # Calculate cumulative returns starting with $10,000
    initial_capital = 10000
    cumulative_returns_erc_var = initial_capital * (1 + portfolio_returns_erc_var).cumprod()
    cumulative_returns_mvo = initial_capital * (1 + portfolio_returns_mvo).cumprod()
    cumulative_equal_returns = initial_capital * (1 + equal_weight_returns).cumprod()

    # Calculate performance metrics
    confidence_level = 0.05
    summary = {
        "Total Return (ERC VaR)": (1 + portfolio_returns_erc_var).prod() - 1,
        "Total Return (MVO)": (1 + portfolio_returns_mvo).prod() - 1,
        "Total Return (Equal Weight)": (1 + equal_weight_returns).prod() - 1,
        "Annualized Return (ERC VaR)": (1 + portfolio_returns_erc_var).prod()**(252 / len(portfolio_returns_erc_var)) - 1,
        "Annualized Return (MVO)": (1 + portfolio_returns_mvo).prod()**(252 / len(portfolio_returns_mvo)) - 1,
        "Annualized Return (Equal Weight)": (1 + equal_weight_returns).prod()**(252 / len(equal_weight_returns)) - 1,
        "Sharpe Ratio (ERC VaR)": (np.mean(portfolio_returns_erc_var) / np.std(portfolio_returns_erc_var)) * np.sqrt(252),
        "Sharpe Ratio (MVO)": (np.mean(portfolio_returns_mvo) / np.std(portfolio_returns_mvo)) * np.sqrt(252),
        "Sharpe Ratio (Equal Weight)": (np.mean(equal_weight_returns) / np.std(equal_weight_returns)) * np.sqrt(252),
    }

    # Display performance metrics
    print(summary)

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns_erc_var, label='Portfolio (ERC VaR)')
    plt.plot(cumulative_returns_mvo, label='Portfolio (MVO)', linestyle='--')
    plt.plot(cumulative_equal_returns, label='Portfolio (Equal Weight)', linestyle=':')
    plt.title('Portfolio Performance ($10,000 Initial Capital)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (USD)')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    # Plot weights over time
    weights_df_erc_var = pd.DataFrame(weights_history_erc_var, index=weights_dates, columns=returns.columns)
    weights_df_mvo = pd.DataFrame(weights_history_mvo, index=weights_dates, columns=returns.columns)

    plt.figure(figsize=(12, 6))
    weights_df_erc_var.plot(kind='line', ax=plt.gca())
    plt.title('ERC (VaR) Portfolio Weights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend(title='Assets', loc='upper right')
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    weights_df_mvo.plot(kind='line', ax=plt.gca())
    plt.title('MVO Portfolio Weights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend(title='Assets', loc='upper right')
    plt.grid()
    plt.show()
