# -*- coding: utf-8 -*-
"""
@author: Ronnawat, CQF ,AFPT , CFA Charterholder Candidate

"""

import os
import sys

import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.stats as sms
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

d1 = pd.read_csv('C:/Users/Ronnawat/Desktop/Trading with Python/ลงทุนเป็นระบบ Finlab/Skewness Test/FTX Case/FTX_BULLUSDT.csv', index_col=0, parse_dates=True)
d2 = pd.read_csv('C:/Users/Ronnawat/Desktop/Trading with Python/ลงทุนเป็นระบบ Finlab/Skewness Test/FTX Case/FTX_ETHBULLUSDT.csv', index_col=0, parse_dates=True)

d1['LRet'] = np.log(d1['close']/d1['close'].shift(1)).dropna()
d1 = d1.iloc[1:]
d2['LRet'] = np.log(d2['close']/d2['close'].shift(1)).dropna()
d2 = d2.iloc[1:]

fig, ax = plt.subplots(3)
ax[0].grid()
ax[0].set_title("Distribution Plot: BTCUSDT Vs. BULLUSDT")
ax[0].set_xlabel("Log Return Bins")
ax[0].set_ylabel("Frequency in Bins")
ax[0].hist(d1['LRet'], bins=31, histtype='step', label='BTCUSDT', color='r')
ax[0].hist(d2['LRet'], bins=31, histtype='step', label='BULLUSDT', color='b')
ax[0].legend()

sm.qqplot(d1['LRet'], line='s',ax=ax[1])
ax[1].set_title("QQ Plot - BTCUSDT")
ax[1].legend(['BTCUSDT'])

sm.qqplot(d2['LRet'], line='s',ax=ax[2])
ax[2].set_title("QQ Plot - BULLUSDT")
ax[2].legend(['BULLUSDT'])

'''
#Skewness and Kurtosis Test
 skew(d1['LRet'])
 skew(d2['LRet'])
 kurtosis(d1['LRet'])
 kurtosis(d2['LRet'])
 d3 = d1['LRet']
 d4 = d2['LRet']
 d5 = pd.concat([d3, d4], axis=1)
 plt.boxplot(d5,vert=True,patch_artist=True,labels = ["BTCUSDT","BULLUSDT"]);
'''
