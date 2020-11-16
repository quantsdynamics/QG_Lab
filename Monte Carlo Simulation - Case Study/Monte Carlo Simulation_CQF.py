# -*- coding: utf-8 -*-
"""
@author: Ronnawat, CQF, CFAII, AFPT

"""
# Importing libraries
import pandas as pd
from numpy import *
import numpy as np

# Libraries for plotting
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
import cufflinks as cf
cf.set_config_file(offline=True)

def simulate_path(s0, mu, sigma, horizon, timesteps, n_sims):
    
    # Set the random seed for reproducibility
    # Same seed leads to the same set of random values
    random.seed(10000) 

    # Read parameters
    S0 = s0         # initial spot level
    r = mu          # mu = rf in risk neutral framework 
    T = horizon     # time horizion
    t = timesteps   # number of time steps
    n = n_sims      # number of simulation
    
    # Define dt
    dt = T/t        # length of time interval  
    
    # Simulating 'n' asset price paths with 't' timesteps
    S = zeros((t+1, n))
    S[0] = 100.

    for i in range(1, t+1):
        z = random.standard_normal(n)     # psuedo random numbers
        S[i] = S[i-1] * exp((r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z) # vectorized operation per timesteps
        
    return S

# Assign simulated price path to dataframe for analysis and plotting
price_path = pd.DataFrame(simulate_path(100, 0.05, 0.20, 1, 252, 100000))


# Plot the histogram of the simulated price path at maturity
price_path.iloc[-1].iplot(kind='histogram', title= 'Simulated Geometric Brownian Motion at Maturity', bins=100)

# Plot simulated price paths 
price_path.iloc[:,:100].iplot(title='Simulated Geometric Brownian Motion Paths', xTitle='Time Steps', yTitle='Index Level



                                                   
                                                 
 

