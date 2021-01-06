# -*- coding: utf-8 -*-
"""
@author: Ronnawat, CQF ,CFAII ,AFPT

"""

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 16, 8

import warnings
warnings.filterwarnings('ignore')

'''
ts = TimeSeries(key='ZODH1LMZM80QKB0C', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
data['4. close'].plot()
plt.title('Intraday Times Series for the MSFT stock (1 min)')
plt.show()
'''
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

# Log likelihood function
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

cc = CryptoCurrencies(key='ZODH1LMZM80QKB0C', output_format='pandas')
data, meta_data = cc.get_digital_currency_daily(symbol='BTC', market='CNY')
sdata = data['4b. close (USD)'].iloc[1:]
sdata.plot()
returns = np.log(sdata).diff().fillna(0)
plt.tight_layout()
plt.title('Daily close value for bitcoin (BTC)')
plt.grid()
plt.show()

# Specify optimization input
param = ['omega', 'alpha', 'beta']
initial_values = (0.00000790570, 0.1,0.8)

res = minimize(likelihood, initial_values, args = returns, 
                   method='Nelder-Mead', options={'disp':False})

# GARCH parameters
dict(zip(param,np.around(res['x']*100,4)))

# Parameters
omega = res['x'][0] 
alpha = res['x'][1]
beta = res['x'][2]

# Variance
var = garch(res['x'][0],res['x'][1],res['x'][2],returns)

# Annualised conditional volatility
ann_vol = np.sqrt(var*252) * 100

# Visualise GARCH volatility of BTC/USD
plt.title('Annualized Volatility of BTC/USD')
plt.plot(returns.index, ann_vol, color='orange', label='GARCH')
plt.legend(loc=2)
plt.grid(True)

# Calculate N-day forecast
longrun_variance = omega/(1-alpha-beta)
 
fvar = []
for i in range(1,360):    
    fvar.append(longrun_variance + (alpha+beta)**i * (var[-1] - longrun_variance))

var = np.array(fvar)

# Plot volatility forecast over different time horizon
plt.axhline(y=np.sqrt(longrun_variance*252)*100, color='blue')
plt.plot(np.sqrt(var*252)*100, color='red')

plt.xlabel('Horizon (in days)')
plt.ylabel('Volatility (%)')

plt.annotate('GARCH Forecast', xy=(650,15.60), color='red')
plt.annotate('Longrun Volatility =' + str(np.around(np.sqrt(longrun_variance*252)*100,2)) + '%', 
             xy=(0,15.70), color='blue')

plt.title('Volatility Forecast : N-days Ahead')
plt.grid(axis='x')
