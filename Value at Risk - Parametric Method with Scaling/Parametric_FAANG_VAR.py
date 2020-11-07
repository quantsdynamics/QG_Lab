# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:16:12 2020

@author: Ronnawat
"""
import pandas as pd
import numpy as np
from numpy.linalg import multi_dot

from scipy.stats import norm
from tabulate import tabulate

import matplotlib.pyplot as plt

# Load locally stored data
df = pd.read_csv('C:/Users/Ronnawat/Desktop/Trading with Python/ลงทุนเป็นระบบ Finlab/Value at Risk/faang15toCurrent.csv', parse_dates=True, index_col=0)['2015':]

# Calculate daily returns
returns = df.pct_change().dropna()

# Visualize FB daily returns
plt.plot(returns['FB'], color='orange')
plt.axhline(y=0.10, ls='dotted', color='black')
plt.axhline(y=-0.10, ls='dotted', color='black')
plt.title('FB Daily Returns')
plt.grid(True)

"""

Parametric VAR

"""

# Calculate mean and standard deviation 
mean = np.mean(returns['FB'])
stdev = np.std(returns['FB'])

# Calculate VaR at difference confidence level
VaR_90 = norm.ppf(1-0.90,mean,stdev)
VaR_95 = norm.ppf(1-0.95,mean,stdev) 
VaR_99 = norm.ppf(1-0.99,mean,stdev)

# Ouput results in tabular format
table = [['90%', VaR_90],['95%', VaR_95],['99%', VaR_99] ]
header = ['Confidence Level', 'Value At Risk']
#print(tabulate(table,headers=header))

# VaR function
def VaR(symbol, cl=0.95):
    mean = np.mean(returns[symbol])
    stdev = np.std(returns[symbol])
    
    return np.around(100*norm.ppf(1-cl,mean,stdev),4)


# VaR for stocks
print('VaR for FAANG Stocks')
print('---'*11)
[print(f'VaR at 95% CL for {stock:4} : {VaR(stock)}%') for stock in df.columns][0]

num_of_shares = 10000
price = df['FB'].iloc[-1]
position = num_of_shares * price 

fb_var = position * VaR_99

# Visualize VaR at 95% confidence level

print(f'FB Holding Value: {position}')
print(f'FB VaR at 99% confidence level is: {fb_var}')


"""

Scaling VAR

"""

forecast_days = 5
f_VaR_90 = VaR_90*np.sqrt(forecast_days)
f_VaR_95 = VaR_95*np.sqrt(forecast_days)
f_VaR_99 = VaR_99*np.sqrt(forecast_days)

ftable = [['90%', f_VaR_90],['95%', f_VaR_95],['99%', f_VaR_99] ]
fheader = ['Confidence Level', '5-Day Forecast Value At Risk']
#print(tabulate(ftable,headers=fheader))

fb_var_5days = position * f_VaR_99
print(f'FB Holding Value: {position}')
print(f'FB VaR at 99% confidence level is: {fb_var_5days}')

# Scaled VaR over different time horizon
plt.figure(figsize=(8,6))
plt.plot(range(100),[-100*VaR_95*np.sqrt(x) for x in range(100)])
plt.xlabel('Horizon')
plt.ylabel('Var 95 (%)')
plt.title('VaR_95 Scaled by Time');

# Scaled VaR over different time horizon
plt.figure(figsize=(8,6))
plt.plot(range(100),[-100*VaR_95*np.sqrt(x) for x in range(100)])
plt.xlabel('Horizon')
plt.ylabel('Var 95 (%)')
plt.title('FB_VaR_95 Scaled by Time');


