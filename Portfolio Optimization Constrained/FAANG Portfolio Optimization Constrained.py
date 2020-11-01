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

"""
Constrained Optimization

"""
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(numofasset))
initial_wts = numofasset*[1./numofasset]

# Optimizing for minimum Sharpe
opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
# Portfolio weights
list(zip(symbols,np.around(opt_sharpe['x']*100,2)))
# Portfolio stats
stats = ['Returns', 'Volatility', 'Sharpe Ratio']
list(zip(stats,np.around(portfolio_stats(opt_sharpe['x']),4)))

# Optimizing for minimum variance
opt_var = sco.minimize(min_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=cons)
# Portfolio weights
list(zip(symbols,np.around(opt_var['x']*100,2)))
# Portfolio stats
list(zip(stats,np.around(portfolio_stats(opt_var['x']),4)))

#Efficient Frontier Within Range

targetrets = linspace(0.22,0.45,200)
tvols = []

for tr in targetrets:
    
    ef_cons = ({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - tr},
               {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    opt_ef = sco.minimize(min_volatility, initial_wts, method='SLSQP', bounds=bnds, constraints=ef_cons)
    
    tvols.append(opt_ef['fun'])

targetvols = array(tvols)

# Visualize the simulated portfolio for risk and return
fig = plt.figure()
ax = plt.axes()

ax.set_title('Efficient Frontier Portfolio')

# Efficient Frontier
fig.colorbar(ax.scatter(targetvols, targetrets, c=targetrets / targetvols, 
                        marker='x', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 

# Maximum Sharpe Portfolio
ax.plot(portfolio_stats(opt_sharpe['x'])[1], portfolio_stats(opt_sharpe['x'])[0], 'r*', markersize =15.0)

# Minimum Variance Portfolio
ax.plot(portfolio_stats(opt_var['x'])[1], portfolio_stats(opt_var['x'])[0], 'b*', markersize =15.0)

ax.set_xlabel('Expected Volatility')
ax.set_ylabel('Expected Return')
ax.grid(True)


# Maximizing sharpe ratio
def min_sharpe_ratio(weights):
    return -portfolio_stats(weights)[2]


def portfolio_stats(weights):
    
    weights = array(weights)[:,newaxis]
    port_rets = weights.T @ array(returns.mean() * 252)[:,newaxis]    
    port_vols = sqrt(multi_dot([weights.T, returns.cov() * 252, weights])) 
    
    return np.array([port_rets, port_vols, port_rets/port_vols]).flatten()

# Minimize the variance
def min_variance(weights):
    return portfolio_stats(weights)[1]**2

# Minimize the volatility  within Range
def min_volatility(weights):
    return portfolio_stats(weights)[1]




