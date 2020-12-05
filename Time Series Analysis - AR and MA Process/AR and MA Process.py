# -*- coding: utf-8 -*-
"""
@author: Ronnawat, CQF ,CFAII ,AFPT

"""
import os
import sys

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.stats as sms

import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_csv('C:/Users/Ronnawat/Desktop/Trading with Python/ลงทุนเป็นระบบ Finlab/Time Series Analysis/Time Series Analysis - AR and MA Process/ETHUSDT.csv', index_col=0, parse_dates=True)
lrets = np.log(df['close']/df['close'].shift(1)).fillna(0)

def tsplot(y, lags=None, figsize=(15, 10), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 

# Select best lag order for AAPL returns

max_lag = 30
mdl = smt.ARMA(lrets, order=(1,1)).fit(maxlag=max_lag, method='mle', trend='nc')
print(mdl.summary())
      
_ = tsplot(lrets, max_lag)
_ = tsplot(mdl.resid, max_lag)

from statsmodels.stats.stattools import jarque_bera

score, pvalue, _, _ = jarque_bera(mdl.resid)

if pvalue < 0.10:
    print ('We have reason to suspect the residuals are not normally distributed.')
else:
    print ('The residuals seem normally distributed.')