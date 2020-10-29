# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:46:48 2020

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

np.random.seed(1) # For reproducibility
plt.style.use('seaborn') # I think this looks pretty
#Coin flip variable set up
p = 0.55
f = p - (1-p)
# Preparing our simulation of coin flips with variables
I = 50 #The number of series to be simulated.
n = 100 #The number of trials per series.
def run_simulation(f):
    c = np.zeros((n, I)) #Instantiates an ndarray object to store the simulation results.
    c[0] = 100 #Initializes the starting capital with 100.
    for i in range(I): #Outer loop for the series simulations.
        for t in range(1,n): #Inner loop for the series itself.
            o = np.random.binomial(1, p) #Simulates the tossing of a coin.
            if o > 0: #If 1, i.e., heads …
                c[t, i] = (1+f) * c[t-1,i] #… then add the win to the capital.
            else: #If 0, i.e., tails …
                c[t, i] = (1-f) * c[t-1,i] #… then subtract the loss from the capital.
    return c

c_1 = run_simulation(f)
c_1.round(2) #Looking at a simulation

plt.figure(figsize=(10,6))
plt.plot(c_1, 'b', lw=0.5) #Plots all 50 series.
plt.plot(c_1.mean(axis=1), 'r', lw=2.5); #Plots the average over all 50 series.
plt.title('50 Simulations of Rigged Coin Flips');
plt.xlabel('Number of trials');
plt.ylabel('$ Amount Won/Lost');