import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm

# Define a function to compute ERC weights using VaR-based Risk Contribution
def compute_erc_weights_var(cov_matrix, mean_returns, confidence_level=0.95):
    n = cov_matrix.shape[0]

    def risk_contribution(weights):
        portfolio_variance = weights @ cov_matrix @ weights
        z_score = norm.ppf(confidence_level)
        marginal_contrib = cov_matrix @ weights
        rc = weights * (mean_returns + z_score * marginal_contrib / portfolio_variance)
        return rc

    def objective(w):
        rc = risk_contribution(w)
        avg_rc = np.mean(rc)
        return np.sum((rc - avg_rc) ** 2)

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

# Parameters for sensitivity analysis
rolling_windows = range(30, 91, 10)  # Rolling windows from 30 to 90 days
rebalance_frequencies = [63, 126, 252]  # Quarterly, Semiannually, Annually
confidence_level = 0.95  # Confidence level for VaR

dates = returns.index
sharpe_ratios = pd.DataFrame(index=rolling_windows, columns=rebalance_frequencies)

# Sensitivity Analysis Loop
for window_size in rolling_windows:
    for rebalance_freq in rebalance_frequencies:
        portfolio_returns = []

        for start in range(0, len(returns) - window_size, rebalance_freq):
            # Rolling window data
            window_data = returns.iloc[start:start + window_size]

            # Check if the rolling window contains sufficient data
            if window_data.empty or window_data.shape[0] < 2:
                continue

            # Validate data for NaN or infinite values
            if window_data.isnull().values.any() or np.isinf(window_data.values).any():
                continue

            cov_matrix = window_data.cov()
            mean_returns = window_data.mean().values

            # Validate covariance matrix
            if cov_matrix.shape[0] != cov_matrix.shape[1]:
                continue

            if not np.allclose(cov_matrix, cov_matrix.T) or np.any(np.linalg.eigvals(cov_matrix) < 0):
                continue

            # Add regularization to the covariance matrix
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8

            # Calculate ERC weights based on VaR Risk Contribution
            try:
                weights = compute_erc_weights_var(cov_matrix.values, mean_returns, confidence_level=confidence_level)
            except Exception:
                continue

            # Apply weights for the next rebalance period
            for t in range(start + window_size, min(start + window_size + rebalance_freq, len(returns))):
                daily_return = np.dot(weights, returns.iloc[t].values)
                portfolio_returns.append(daily_return)

        # Calculate Sharpe Ratio for the combination
        if portfolio_returns:
            portfolio_returns = np.array(portfolio_returns)
            sharpe_ratio = (np.mean(portfolio_returns) / np.std(portfolio_returns)) * np.sqrt(252)
            sharpe_ratios.loc[window_size, rebalance_freq] = sharpe_ratio

# Convert the DataFrame to numeric, replacing any non-numeric entries with NaN
sharpe_ratios = sharpe_ratios.apply(pd.to_numeric, errors='coerce')

# Plot Sharpe Ratio Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(sharpe_ratios, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Sharpe Ratio'})
plt.title('Sharpe Ratio Heatmap by Rolling Window and Rebalance Frequency')
plt.xlabel('Rebalance Frequency (Days)')
plt.ylabel('Rolling Window Size (Days)')
plt.show()
