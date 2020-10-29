# -*- coding: utf-8 -*-
"""
@author: Ronnawat, CQF, AFPT, CFA(II)
"""
#Import libraries
import math
import time
import numpy as np
import pandas as pd
import datetime as dt
import cufflinks as cf
import matplotlib.pyplot as plt

#Loading SPY data
data = pd.read_csv('SET 100 Historical Data.csv', index_col=0, parse_dates=True)
#Light Feature Engineering on Returns
data['Change %'] = data['Change %'].map(lambda x: x.rstrip('%')).astype(float) / 100
data.dropna(inplace=True)
data.tail()
mu = data['Change %'].mean() * 252 #Calculates the annualized return.
sigma = (data['Change %'].std() * 252 ** 0.5)  #Calculates the annualized volatility.
r = 0.0141 #10 yr GOV Bond from TBMA
f = (mu - r) / sigma ** 2 
equs = [] # preallocating space for our simulations
def kelly_strategy(f):
    global equs
    equ = 'equity_{:.2f}'.format(f)
    equs.append(equ)
    cap = 'capital_{:.2f}'.format(f)
    data[equ] = 1  #Generates a new column for equity and sets the initial value to 1.
    data[cap] = data[equ] * f  #Generates a new column for capital and sets the initial value to 1·f∗.
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]  #Picks the right DatetimeIndex value for the previous values.
        data.loc[t, cap] = data[cap].loc[t_1] * math.exp(data['Change %'].loc[t])
        data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * f 
kelly_strategy(f*0.5)
kelly_strategy(f * 0.66) # Values for 2/3 KC
kelly_strategy(f) # Values for optimal KC
ax = data['Change %'].cumsum().apply(np.exp).plot(legend=True,figsize=(10, 6))         
data[equs].plot(ax=ax, legend=True);
plt.title('Varied KC Values on SET100')
plt.xlabel('Years');
plt.ylabel('$ Return')