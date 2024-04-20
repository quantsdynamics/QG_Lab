# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:36:32 2024

@author: Aaron Desktop
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import riskfolio as rp
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import norm
import pmdarima as pm
np.random.seed(1)

df = pd.read_csv('test_garch.csv')
df['time']      = pd.to_datetime(df['time'])
df['time']      = pd.to_datetime(df['time'].dt.strftime('%Y-%m-%d'), format='%Y-%m-%d')
df = df.set_index('time')
df['logret'] = np.log(df['SPX']) - np.log(df['SPX'].shift(1))
df = df.dropna()

df['roll_sd'] = df['logret'].rolling(window=20).std()

def EWMA_Volatility(rets, lam):
    sq_rets_sp500 = (rets**2).values
    EWMA_var = np.zeros(len(sq_rets_sp500))
    
    for r in range(1, len(sq_rets_sp500)):
        EWMA_var[r] = (1-lam)*sq_rets_sp500[r] + lam*EWMA_var[r-1]

    EWMA_vol = np.sqrt(EWMA_var)
    return pd.Series(EWMA_vol, index=rets.index, name ="EWMA Vol {}".format(lam))[1:]

df['EWMA_95'] = EWMA_Volatility(df['logret'], 0.95)
df['EWMA_90'] = EWMA_Volatility(df['logret'], 0.90)

df[['roll_sd','EWMA_95','EWMA_90']].plot(figsize=(16,6))

#Maximum Likelihood Estimation

# Log likelihood function

# GARCH(1,1) function
def garch(omega, alpha, beta, ret):
    
    length = len(ret)
    
    var = []
    for i in range(length):
        if i==0:
            var.append(omega/np.abs(1-alpha-beta))
        else:
            var.append(omega + alpha * ret[i-1]**2 + beta * var[i-1])
            
    return np.array(var)

def likelihood(params, ret):
    
    length = len(ret)
    omega = params[0]
    alpha = params[1]
    beta = params[2]
    
    variance = garch(omega, alpha, beta, ret)
    
    llh = []
    for i in range(length):
        llh.append(np.log(norm.pdf(ret[i], 0, np.sqrt(variance[i]))))
    
    return -np.sum(np.array(llh))

# Specify optimization input
param = ['omega', 'alpha', 'beta']
initial_values = (np.var(df['logret']), 0.1,0.8)
res = minimize(likelihood, initial_values, args = df['logret'], 
                   method='Nelder-Mead', options={'disp':False})

# Parameters
omega = res['x'][0] 
alpha = res['x'][1]
beta = res['x'][2]

# Variance
var = garch(res['x'][0],res['x'][1],res['x'][2],df['logret'])

# Annualised conditional volatility
ann_vol = np.sqrt(var*252) * 100
ann_vol

# Visualise GARCH volatility and VIX
plt.title('Annualized Volatility')
plt.plot(df.index, ann_vol, color='orange', label='GARCH')
plt.plot(df.index, df['VIX'], color='blue', label = 'VIX')
plt.legend()

g1 = arch_model(df['logret'], vol='GARCH', mean='Zero', p=1, o=0, q=1, dist='Normal')
model1 = g1.fit()
# Model output
print(model1)
# Plot annualised vol
fig = model1.plot(annualize='D')

g2 = arch_model(df['logret'], vol='GARCH', mean='Zero', p=1, o=0, q=1, dist='GED')
model2 = g2.fit()
# Model output
print(model2)
# Plot annualised vol
fig = model2.plot(annualize='D')