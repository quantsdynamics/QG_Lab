import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import riskfolio as rp
# riskfolio-lib does not have a PlotFunctions module, using rp directly instead.

# Load the provided data
data_path = 'risk-budgeting.csv'
data = pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True)
data.set_index('Date', inplace=True)

# Calculate daily returns
returns = data.pct_change().dropna()

rolling_window = 60  # 60 trading days
rebalance_freq = 126  # 126 trading days
initial_capital = 10000

# Define the backtest function

def backtest_erc(returns, rolling_window, rebalance_freq, initial_capital=10000):
    portfolio_values = [initial_capital]
    dates = returns.index
    weights_history = []

    for start in range(0, len(returns) - rolling_window, rebalance_freq):
        end = start + rolling_window
        train_data = returns.iloc[start:end]

        # Initialize Portfolio object
        port = rp.Portfolio(returns=train_data)

        # Estimate statistics
        port.assets_stats(method_mu='hist', method_cov='hist')

        # Optimize portfolio for Equal Risk Contribution
        weights = port.rp_optimization(model='Classic', rm='MV', rf=0, b=None, hist=True)
        weights_history.append(weights)

        # Apply weights to the next period
        for t in range(end, min(end + rebalance_freq, len(returns))):
            daily_return = np.dot(weights.T, returns.iloc[t].values)
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)

    # Ensure the lengths match by trimming the excess value if present
    if len(portfolio_values) > len(dates[rolling_window:]):
        portfolio_values = portfolio_values[:len(dates[rolling_window:])]

    return pd.Series(portfolio_values, index=dates[rolling_window:len(portfolio_values) + rolling_window]), weights_history

# Run the backtest
portfolio_values, weights_history = backtest_erc(returns, rolling_window, rebalance_freq, initial_capital)

# Plot portfolio value over time
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label='Portfolio Value')
plt.title('Portfolio Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Use riskfolio-lib's built-in risk contribution plot
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu='hist', method_cov='hist')
w_rp = port.rp_optimization(model='Classic', rm='MV', rf=0, b=None, hist=True)
ax = rp.plot_risk_con(
    w=w_rp,
    cov=port.cov,
    returns=port.returns,
    rm='MV',
    rf=0,
    alpha=0.01,
    color="tab:blue",
    height=6,
    width=10,
    ax=None
)
