# -*- coding: utf-8 -*-
"""
@author: Ronnawat, CQF, CFAII, AFPT

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

def Download_Data(Symbol, Source, Date_Start, Date_End):
    from datetime import datetime
    import pandas_datareader as web
    fecha_ini = datetime.strptime(Date_Start, '%d-%m-%Y')
    fecha_fin = datetime.strptime(Date_End, '%d-%m-%Y')
    df = web.DataReader(Symbol, data_source=Source, start=fecha_ini, end=fecha_fin)
    return df


def Annual_Volatility(Symbol, Source, Date_Start, Date_End):
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from pandas_datareader import data
    
    df_data = Download_Data(Symbol, Source, Date_Start,Date_End)
    quote = df_data.filter(['Close'])
    quote['Returns'] = quote['Close'].pct_change()
    vol = quote['Returns'].std()*np.sqrt(252)
    return vol

def Compound_Annual_Growth_Rate(Symbol, Source, Date_Start, Date_End):
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from pandas_datareader import data
    
    df_dato = Download_Data(Symbol, Source, Date_Start,Date_End)
    quote = df_dato.filter(['Close'])
    days = (quote.index[-1] - quote.index[0]).days
    cagr = ((((quote['Close'][-1]) / quote['Close'][1])) ** (365.0/days)) - 1
    return cagr

def Advanced_Montecarlo_Simulation(Symbol, Source, Date_Start, Date_End,Simulation_Num,Day_Window):
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import pandas as pd
    
    result = []
    
    df_dato = Download_Data(Symbol, Source, Date_Start,Date_End)
    quote = df_dato.filter(['Close'])
    
    S = quote['Close'][-1]
    T = Day_Window #Number of trading days
    mu = Compound_Annual_Growth_Rate(Symbol, Source, Date_Start,Date_End)
    vol = Annual_Volatility(Symbol, Source, Date_Start,Date_End)
    
    plt.subplot(2,1,1)
    plt.title('Tesla Price Simulation Path')
    plt.xlabel('Day')
    plt.ylabel('Price Level')
    for i in range(Simulation_Num):
        daily_returns=np.random.normal(mu/T,vol/math.sqrt(T),T)+1
        price_list = [S]
        for x in daily_returns:
            price_list.append(price_list[-1]*x)
        plt.plot(price_list)
        result.append(price_list[-1])
    
    plt.subplot(2,1,2)
    plt.ylabel('Accumulation')
    plt.xlabel('Price Level')
    plt.hist(result,bins=100)
    plt.axvline(np.percentile(result,5), color='black', linestyle='dashed', linewidth=2)
    plt.axvline(np.percentile(result,95), color='black', linestyle='dashed', linewidth=2)
    
    cabeceras = ['Percentile 5%', 'Percentile 95%', 'CAGR', 'Annual Volatility']
    df_percentile = pd.DataFrame(columns=cabeceras)
    
    df_percentile.loc[len(df_percentile)]=[np.percentile(result,5),np.percentile(result,95),mu,vol]
    
    return df_percentile

x = Advanced_Montecarlo_Simulation('TSLA', 'yahoo', '01-01-2018', '01-11-2020',100000,252)