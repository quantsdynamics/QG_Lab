import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import riskfolio as rp

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

# Date range
start = '2018-02-01'
end = '2024-12-17'

# Tickers of assets
assets = ['SPY', 'EFA', 'TLT', 'JNK', 'GLD', 'USO', 'PSP', 'TBLL']
assets.sort()

# Tickers of factors
factors = ['MTUM', 'QUAL', 'VLUE', 'SIZE', 'USMV']
factors.sort()

tickers = assets + factors
tickers.sort()

# Downloading data
data = yf.download(tickers, start=start, end=end)
data = data.loc[:, ('Adj Close', slice(None))]
data.columns = tickers

# Calculating returns
X = data[factors].pct_change().dropna()
Y = data[assets].pct_change().dropna()

display(X.head())

# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:
method_mu = 'hist'  # Method to estimate expected returns based on historical data.
method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov)

# Define factors statistics
port.factors = X
port.factors_stats(method_mu=method_mu, method_cov=method_cov)

# Estimate optimal portfolio
model = 'FC'  # Factor Contribution Model
rm = 'MV'     # Variance risk measure
rf = 0         # Risk-free rate
b_f = None     # Risk factor contribution vector

# Perform risk-parity optimization
w = port.rp_optimization(model=model, rm=rm, rf=rf, b_f=b_f)

# Display the weights
display(w.T)

# Plotting the composition of the portfolio
ax = rp.plot_bar(w,
                 title='Risk Factor Parity Portfolio - Variance',
                 kind="v",
                 others=0.05,
                 nrow=25,
                 height=6,
                 width=10,
                 ax=None)