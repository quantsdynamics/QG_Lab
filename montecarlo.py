import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Reload the uploaded file
file_path = 'risk-budgeting.csv'
data = pd.read_csv(file_path)

# Convert date column to datetime and sort by date
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data = data.sort_values('Date')

# Select asset prices (excluding the date column)
assets = data.columns[1:]
prices = data[assets].dropna()

# Calculate daily returns for all assets
returns = prices.pct_change().dropna()

# Monte Carlo Simulation using Geometric Brownian Motion (GBM)
n_simulations = 10000  # Number of Monte Carlo paths
n_days = 126  # 6 months (252 trading days in a year divided by 2)
initial_prices = prices.iloc[-1].values  # Use the last observed prices as starting points
dt = 1 / 252  # Time step (daily)

# Calculate mean and standard deviation of returns
mean_returns = returns.mean().values
std_devs = returns.std().values

# Initialize array to store simulations
simulated_prices = np.zeros((n_simulations, n_days, len(assets)))

# Generate GBM simulations for each asset
for i in range(len(assets)):
    drift = mean_returns[i] - (0.5 * std_devs[i]**2)
    for j in range(n_simulations):
        random_shocks = np.random.normal(0, 1, n_days)
        simulated_prices[j, :, i] = initial_prices[i] * np.exp(
            np.cumsum(drift * dt + std_devs[i] * np.sqrt(dt) * random_shocks)
        )

# Calculate VaR for each asset at 95% confidence level
confidence_level = 0.95
z_score = norm.ppf(confidence_level)
final_prices = simulated_prices[:, -1, :]
portfolio_returns = (final_prices - initial_prices) / initial_prices

# Calculate VaR for each asset
var_values = np.percentile(portfolio_returns, (1-confidence_level)*100, axis=0)

# Calculate mean VaR
mean_var = np.mean(var_values)
print(f"Mean Value at Risk (VaR) across all assets: {mean_var:.4f}")

# Equal Risk Contribution Optimization for VaR
weights = np.ones(len(assets)) / len(assets)  # Initial guess for weights
tolerance = 1e-6
max_iterations = 100

# Optimize weights iteratively for equal VaR contribution
for _ in range(max_iterations):
    portfolio_var = np.dot(weights, var_values)
    marginal_contributions = var_values / portfolio_var
    risk_contributions = weights * marginal_contributions
    diff = risk_contributions - risk_contributions.mean()

    if np.all(np.abs(diff) < tolerance):
        break

    weights -= 0.01 * diff
    weights = np.maximum(weights, 0)  # Ensure weights are non-negative
    weights /= weights.sum()  # Normalize weights

# Visualize 1000 Monte Carlo simulation paths for all assets
plt.figure(figsize=(14, 10))
for asset_index, asset_name in enumerate(assets):
    plt.subplot(4, 2, asset_index + 1)
    simulation_paths = simulated_prices[:1000, :, asset_index]  # Take the first 1000 simulations
    for path in simulation_paths:
        plt.plot(path, color="blue", alpha=0.1)
    plt.title(f"Monte Carlo Simulation Paths for {asset_name} (First 1000 Paths)")
    plt.xlabel("Days")
    plt.ylabel("Simulated Price")
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Display final weights
final_weights = {asset: weight for asset, weight in zip(assets, weights)}
print("Final Weights for Equal VaR Contribution:")
print(final_weights)

# Export final weights to CSV
weights_df = pd.DataFrame(list(final_weights.items()), columns=["Asset", "Weight"])
weights_df.to_csv("gbm_weights.csv", index=False)
