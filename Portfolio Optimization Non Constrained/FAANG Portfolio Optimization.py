# -*- coding: utf-8 -*-
"""
@author: Ronnawat ,CQF ,CFAII, AFPT

"""
import math
import pandas as pd
import xlwings as xw
import numpy as np
from numpy import *
from numpy.linalg import multi_dot
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import scipy.optimize as sco

rcParams['figure.figsize'] = 16, 8

# FAANG stocks
symbols = ['AAPL', 'AMZN', 'FB', 'GOOG', 'NFLX' ]
numofasset = len(symbols)
numofportfolio = 5000

df = pd.read_csv('C:/Users/Ronnawat/Desktop/Trading with Python\From Medium/Portfolio Optimization/faang_cprice.csv', index_col=0, parse_dates=True)['2018':]
xw.view(df)
wb = xw.Book(r'C:/Users/Ronnawat/Desktop/Trading with Python\From Medium/Portfolio Optimization/portfolio.xlsx') 
summary = df.describe().T
#summary.to_excel("portfolio.xlsx",sheet_name='Sheet1')
fig = plt.figure(figsize=(16,8))
ax = plt.axes()

#Movement Visualization
ax.set_title('Normalized Price Plot')
ax.plot(df[-252:]/df.iloc[-252] * 100)
ax.legend(df.columns, loc='upper left')
ax.grid(True)

# Fill MA Values
returns = df.pct_change().fillna(0)

# Calculate annual returns
annual_returns = (returns.mean() * 252)

# Visualize the data
fig = plt.figure()
ax =plt.axes()

ax.bar(annual_returns.index, annual_returns*100, color='royalblue', alpha=0.75)
ax.set_title('Annualized Returns (in %)');

# Calculate Volatility
vols = returns.std()
annual_vols = vols*math.sqrt(252)

# Visualize the data
fig = plt.figure()
ax = plt.axes()

ax.bar(annual_vols.index, annual_vols*100, color='orange', alpha=0.5)
ax.set_title('Annualized Volatility (in %)');

#Equal Weighted Portfolio Simulation
wts = numofasset * [1./numofasset]
wts = array(wts)[:,newaxis]
array(returns.mean() * 252)[:,newaxis]    

# Portfolio returns for Equal Weighted Portfolio
wts.T @ array(returns.mean() * 252)[:,newaxis]      

# Covariance matrix
returns.cov() * 252

# Portfolio variance
multi_dot([wts.T,returns.cov()*252,wts])

# Portfolio volatility
sqrt(multi_dot([wts.T,returns.cov()*252,wts]))

#Random Weighted Portfolio Simulation
w = random.random(numofasset)[:, newaxis]
w /= sum(w)
w.flatten()
rets = []; vols = []; wts = []

# Simulate 5,000 portfolios
for i in range (5000):
    
    # Generate random weights
    weights = random.random(numofasset)[:, newaxis]
    
    # Set weights such that sum of weights equals 1
    weights /= sum(weights)
    
    # Portfolio statistics
    rets.append(weights.T @ array(returns.mean() * 252)[:, newaxis])        
    vols.append(sqrt(multi_dot([weights.T, returns.cov()*252, weights])))
    wts.append(weights.flatten())

# Record values     
port_rets = array(rets).flatten()
port_vols = array(vols).flatten()
port_wts = array(wts)

# Create a dataframe for analysis
mc_df = pd.DataFrame({'returns': port_rets,
                      'volatility': port_vols,
                      'sharpe_ratio': port_rets/port_vols,
                      'weights': list(port_wts)})

# Max sharpe ratio portfolio 
msrp = mc_df.iloc[mc_df['sharpe_ratio'].idxmax()]

# Max sharpe ratio portfolio weights
max_sharpe_port_wts = mc_df['weights'][mc_df['sharpe_ratio'].idxmax()]

# Allocation to achieve max sharpe ratio portfolio
dict(zip(symbols,np.around(max_sharpe_port_wts*100,2)))

# Visualize the simulated portfolio for risk and return
fig = plt.figure()
ax = plt.axes()

ax.set_title('Monte Carlo Simulated Allocation')

# Simulated portfolios
fig.colorbar(ax.scatter(port_vols, port_rets, c=port_rets / port_vols, 
                        marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

# Maximum sharpe ratio portfolio
ax.scatter(msrp['volatility'], msrp['returns'], c='red', marker='*', s = 300, label='Max Sharpe Ratio')

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
ax.grid(True)

# Visualize the simulated portfolio for risk and return
fig = plt.figure()
ax = plt.axes()

ax.set_title('Monte Carlo Simulated Allocation')

# Simulated portfolios
fig.colorbar(ax.scatter(port_vols, port_rets, c=port_rets / port_vols, 
                        marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

# Maximum sharpe ratio portfolio
ax.scatter(msrp['volatility'], msrp['returns'], c='red', marker='*', s = 300, label='Max Sharpe Ratio')

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
ax.grid(True)

def portfolio_stats(weights):
    
    weights = array(weights)[:,newaxis]
    port_rets = weights.T @ array(returns.mean() * 252)[:,newaxis]    
    port_vols = sqrt(multi_dot([weights.T, returns.cov() * 252, weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

# Max sharpe ratio portfolio 
msrp = mc_df.iloc[mc_df['sharpe_ratio'].idxmax()]