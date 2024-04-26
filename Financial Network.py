# Copyright (C) 2023 Gennis
'''
Initiated Date    : 2023/11/07
Last Updated Date : 2023/11/28
Aim: Compare between the naive strategy and constrained Markowitz portfolio.
Input Data: Component stock prices of Hang Seng Index.
'''


#%% Enviornment

from bs4 import BeautifulSoup
from copy import deepcopy
# import ffn
# import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import requests
import scipy
# import sklearn
import yfinance as yf
# from functools import reduce

## Parameters
# Length of total period
start_date = '2013-01-01'
end_date   = '2019-05-31'
# Length of period of rolling window
M = '3 months'
# Lenght of holding period
H = '1 months'

# To simplify, I currently hold the units of the 3 types of period as fixed.
T = (int(end_date[:4]) - int(start_date[:4]))*12 + int(end_date[5:7]) - int(start_date[5:7]) + 1
M = int(M[0])
H = int(H[0])

# Risk free rate
rf = 0.01

# Number of stocks selected for constructing portfolio.
NumStock = 10

# ρ threshold for ρ-dependent strategy.
Rho_tilde = 'Pearson'



#%% Data Access

## Download the information regarding the Hang Seng Index from Wiki.
wikiURL = 'https://en.wikipedia.org/wiki/Hang_Seng_Index'


session = requests.Session()
page = session.get(wikiURL)
c = page.content

soup = BeautifulSoup(c, 'html.parser')

mydivs = soup.findAll('table',{'class':'wikitable'})

## Extract the list of the stock components included in the table from the information crawled.
WikiTable = pd.read_html(str(mydivs))
# Convert list to dataframe
WikiTable = pd.DataFrame(WikiTable[2])




#%% Data Cleaning

WikiTable['Ticker'] = WikiTable['Ticker'].apply( lambda x: int(x[6:]) )

def Func_Ticker( num ):
    if num < 10:
        tick = '000' + str(num)
    elif num < 100:
        tick = '00'  + str(num)
    elif num < 1000:
        tick = '0'   + str(num)
    else:
        tick = str(num)
    
    return tick + '.hk'


WikiTable['Ticker_yf']   = WikiTable['Ticker'].apply( lambda x: Func_Ticker(x) )
WikiTable['Ticker_node'] = WikiTable['Ticker_yf'].apply( lambda x: x[:-3] )


stockList = ' '.join( WikiTable['Ticker_yf'].values )


## Add the component stocks that existed in the index previously.
# except 0004.hk, 百麗國際, 0494.hk, 0013.hk, 1199.hk, ...
histStock = ' 0006.hk 0008.hk 0010.hk 0014.hk 0017.hk 0019.hk 0023.hk 0041.hk 0069.hk 0083.hk 0097.hk 0135.hk 0142.hk 0144.hk 0151.hk 0179.hk 0291.hk 0293.hk 0315.hk 0322.hk 0330.hk 0363.hk 0511.hk 0551.hk 0863.hk 0992.hk 1038.hk 1088.hk 1898.hk 2018.hk 2038.hk 2600.hk 3328.hk'

stockListFull = stockList + histStock

Stock = yf.download(stockListFull, start=start_date, end=end_date)

## Extract the adjusted closed prices.
Stock_close = deepcopy( Stock['Adj Close'] )

## Extract the adjusted closed prices.
Stock_open = deepcopy( Stock['Open'] )

## Remove the stocks whose price data is insufficient.
Stock_close = Stock_close.dropna(axis='columns', thresh=int( len(Stock_close)*0.99 ))
Stock_open  = Stock_open.dropna(axis='columns', thresh=int( len(Stock_open)*0.99 ))

# Stock_close.isnull().sum().sum()
## Fill the remaining missing values by interpolation.
# Stock_close = Stock_close.interpolate()

## Retrieve the Index.
HS = yf.download('^HSI', start=start_date, end=end_date)
HSI = deepcopy(HS)['Adj Close']
HS_open = deepcopy(HS)['Open']



#%% Local Data Import


## Set the path of data.
# _Path = r'C:\Users\Gennis\JupyterNB\FIn Network Analysis\Centrality_6.5'
# os.chdir(_Path)

# Stock_close = pd.read_csv('Stock_HK.csv')

# Stock_close['Date'] = Stock_close['Date'].apply(lambda x: pd.Timestamp(x))

# Stock_close = Stock_close.set_index('Date')

# ## Remove the stocks whose price data is insufficient.
# Stock_close = Stock_close.dropna(axis='columns', thresh=int( len(Stock_close)*0.99 ))




#%% Period Setting

N = len(Stock_close.columns)
Returns = Stock_close.pct_change()[1:]

## Find the last trading days each month by counting the intertemporal changes of the day of dates.
Stock_close.index = pd.to_datetime(Stock_close.index)
Stock_open.index = pd.to_datetime(Stock_open.index)
day_shift = Stock_close.index.day[1:] - Stock_close.index.day[:-1]

Last_dates = Stock_close.index[ np.where( day_shift < 0 ) ]

## Find the first trading days each month by the trading day after the last trading days.
First_dates = Stock_close.index[ np.where( day_shift < 0 )[0] + 1 ]

## For convinience, just retain the data with complete months.
# Check first month of the data.
if Stock_close.index.day[0] > 5:
    Stock_close = Stock_close.loc[First_dates[0]:]
    Stock_open  = Stock_open.loc[First_dates[0]:]
    HSI         = HSI.loc[First_dates[0]:]
    
    Last_dates = Last_dates[1:]
else:
    Last_dates = Last_dates.append(pd.DatetimeIndex([Stock_close.index[-1]]))

# Check last month of the data.
if Stock_close.index.day[-1] < 25:
    Stock_close = Stock_close.loc[ :Last_dates[-1] ]
    Stock_open  = Stock_open.loc[ :Last_dates[-1] ]
    HSI         = HSI.loc[ :Last_dates[-1] ]
    
    First_dates = First_dates[:-1]
else:
    First_dates = pd.DatetimeIndex([Stock_close.index.tolist()[0]]).append(First_dates)
    
# Note:
# The first M periods of data and the final rolling window may not be used in case.




#%% Naive Strategy

## The weight of naive strategy.
Weight_naive = np.array([1/N]*N)

## Calculate the rolling windows for returns.
Rolling_r = []
for i in range(0, T - M + 1, H):
    print(i)
    windows = Returns.loc[ First_dates[i] : Last_dates[i + M - 1] ]
    Rolling_r.append(windows)


## Calculate the exposures at the end of each holding period.
Exposures_naive = pd.DataFrame()
Holding_naive = pd.DataFrame()
CumulativeR = pd.DataFrame()
for _period in range(0, T - M, H):
    ## Compute the non-cumulated number of shares of stock of naive strategy at the beginning of each holding period.
    Shares_t = 1/N / Stock_open.loc[ First_dates[_period+M] ]
    
    ## Compute the holding period exposures if the initial inv is $1 at the begining.
    Holding_naive_t = Shares_t * Stock_close.loc[ Last_dates[_period+M] ]
    Holding_naive = pd.concat([ Holding_naive, Holding_naive_t ], axis = 1)
    
    ## Calculate the exposures at the end of each holding period.
    # Scale up to the change of wealth accumulated across each ending of holding period.
    if _period > 0:
        Exposure_t = Holding_naive_t * Exposures_naive.iloc[:, _period-1].sum()
    else:
        Exposure_t = Holding_naive_t
    
    Exposures_naive = pd.concat([ Exposures_naive, Exposure_t ], axis = 1)
    
    ## Compute the cummulative return of the stock.
    _cumR_t = (Stock_close.loc[ Last_dates[_period+M] ] - Stock_open.loc[ First_dates[0] ] ) / Stock_open.loc[ First_dates[0] ]
    CumulativeR = pd.concat([CumulativeR, _cumR_t], axis=1)
    
    
Exposures_naive.columns = Last_dates.tolist()[M:]
Holding_naive.columns = Last_dates.tolist()[M:]
CumulativeR.columns = Last_dates.tolist()[M:]


## Calculate the weights at the end of each holding period.
Weight_end = pd.DataFrame()
for _period in range(len(Exposures_naive.columns)):
    Weight_end = pd.concat([ Weight_end, Exposures_naive.iloc[:, _period] / Exposures_naive.iloc[:, _period].sum() ], axis = 1)

## Calculate the turnover.
Weights_diff = Weight_end - 1/N
Turnover = Weights_diff.abs().sum().sum() / (T - H - M)


## Calculate the final return.
Exposures_naive.iloc[:, -1].sum()

## Calculate the total portfolio Sharpe ratio.
var_naive = Weight_naive.T @ Returns.cov() @ Weight_naive
Sharpe_naive = (Exposures_naive.iloc[:, -1].sum() - 1 - rf) / var_naive**0.5

## Calculate the portfolio cumulative return.
Cum_R_naive = Weight_naive @ CumulativeR


## Calculate the portfolio Sharpe ratios and portfolio variances across holding periods.
Vars_naive = []
Sharpes_naive = []
for _period in range(len(Rolling_r)-1):
    _var = Weight_naive.T @ Rolling_r[_period].cov() @ Weight_naive
    Vars_naive.append( _var )
    
    _sharpe = (Holding_naive.iloc[:, _period].sum() - 1 - rf) / _var**0.5
    Sharpes_naive.append( _sharpe )

Vars_naive    = pd.DataFrame(Vars_naive)
Sharpes_naive = pd.DataFrame(Sharpes_naive)

Sharpes_naive.mean()
Sharpes_naive.std()
Vars_naive.mean()
Vars_naive.std()


# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, (Holding_naive.sum()-1) * 100, label='single period return')
plt.axhline(0, c='r', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
# plt.title('The Change in Birth Rate and Public Finance from 2001 to 2022',
#           fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=13)
plt.show()


# Create a plot for cummulative returns across holding periods.
plt.figure(figsize=(12, 6))
plt.plot(Cum_R_naive.index, Cum_R_naive * 100, label='cummulative period return')
plt.axhline(0, c='r', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
# plt.title('The Change in Birth Rate and Public Finance from 2001 to 2022',
#           fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=13)
plt.show()




#%% Constrained Markowitz - Minimum Variance Portfolio

## Solve the optimal weights of minimum variance portfolio.
def Func_OptWeight_MinV(windows):
    # Objective function: minimum portfolio variance.
    obj_func = lambda w: w.T @ windows.cov() @ w
    
    # Set constraints
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights equals 1
                   {'type': 'ineq', 'fun': lambda w: w})  # Non-negativity constraint
    
    # Specify options with 'maxiter'
    options = {'maxiter': 1000000}
    
    # Solve the quadratic programming problem.
    # Using equal weights as the initial guess.
    result = scipy.optimize.minimize(
        obj_func, Weight_naive, method='SLSQP', constraints=constraints, options=options)
    w_MinV = pd.DataFrame( result.x )
    
    # Remove small numerical calculated weights.
    w_MinV[ w_MinV < 10 ** -4 ] = 0

    return w_MinV


## Calculate the optimal weight using rolling windows.
# No optimal weight calculated by the last rolling window.
Weight_MinV = pd.DataFrame()
for _period in range(len(Rolling_r)-1): 
    weight_t = Func_OptWeight_MinV( Rolling_r[_period] )
    ## Scale up to make sure the sum of actual weights is 100%.
    weight_t = weight_t / weight_t.sum()
    Weight_MinV = pd.concat([ Weight_MinV, weight_t ], axis=1)

Weight_MinV = Weight_MinV.T
Weight_MinV.index = First_dates[M:]
Weight_MinV.columns = Stock_close.columns


## Construct dynamics of the component stocks in the portfolio across rolling windows.
def Func_Holding(weight):
    ## Calculate the exposures at the end of each holding period.
    Exposures = pd.DataFrame()
    Holding = pd.DataFrame()
    for period in range(0, T - M, H):
        ## Compute the non-cumulated number of shares of stock of naive strategy at the beginning of each holding period.
        Shares_t = 1 * weight.loc[ First_dates[period+M] ] / Stock_open.loc[ First_dates[period+M] ]
        
        ## Compute the holding period exposures if the initial inv is $1 at the begining.
        holding_t = Shares_t * Stock_close.loc[ Last_dates[period+M] ]
        Holding = pd.concat([ Holding, holding_t ], axis = 1)

        ## Calculate the exposures at the end of each holding period.
        # Scale up to the change of wealth accumulated across each ending of holding period.
        if period > 0:
            exposure_t = holding_t * Exposures.iloc[:, period-1].sum()
        else:
            exposure_t = holding_t
        
        Exposures = pd.concat([ Exposures, exposure_t ], axis = 1)

    Exposures.columns = Last_dates.tolist()[M:]
    Holding.columns = Last_dates.tolist()[M:]

    ## Calculate the weights at the end of each holding period.
    Weight_end = pd.DataFrame()
    for period in range(len(Exposures.columns)):
        Weight_end = pd.concat(
            [Weight_end, Exposures.iloc[:, period] / Exposures.iloc[:, period].sum()
             ], axis = 1)
    
    return Exposures, Holding, Weight_end

Exposures_MinV, Holding_MinV, Weight_MinV_end = Func_Holding(Weight_MinV)

def Func_Performance(w_begin, w_end, exposure):
    ## Calculate the turnover.
    w_diff = w_begin.T
    w_diff.columns = w_end.columns
    w_diff = w_end - w_diff
    turnover = w_diff.abs().sum().sum() / (T - H - M)

    ## Calculate the final return.
    final_r = Exposures_MinV.iloc[:, -1].sum()

    ## Calculate the portfolio Sharpe ratio and portfolio variance.
    Var = pd.DataFrame()
    Sharpe = pd.DataFrame()
    for period in range(len(Rolling_r)-1):
        _var = w_begin.iloc[period].T @ Rolling_r[period + 1].cov() @ w_begin.iloc[period]
        Var = pd.concat([ Var, pd.DataFrame([_var]) ], axis=1)
        
        _sharpe = (exposure.iloc[:, period].sum() - 1 - rf) / _var**0.5
        Sharpe = pd.concat([ Sharpe, pd.DataFrame([_sharpe]) ], axis=1)

    Var = Var.T
    Var.index = Last_dates.tolist()[M:]
    Sharpe = Sharpe.T
    Sharpe.index = Last_dates.tolist()[M:]

    return turnover, Sharpe, Var, final_r


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_MinV, Var_MinV, _ = Func_Performance(Weight_MinV, Weight_MinV_end, Exposures_MinV)

Sharpe_MinV.mean()
Sharpe_MinV.std()

Var_MinV.mean()
Var_MinV.std()


## Calculate the portfolio cumulative return.
Cum_R_MinV = Weight_MinV @ CumulativeR
Cum_R_MinV = np.diag(Cum_R_MinV.values)


# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, (Holding_naive.sum()-1) * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, (Holding_MinV.sum()-1) * 100, label='Constrained Markowitz \n(Min-Var)', color='r')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()


# Create a plot for cummulative returns across holding periods.
plt.figure(figsize=(12, 6))
plt.plot(Cum_R_naive.index, Cum_R_naive * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, Cum_R_MinV * 100, label='Constrained Markowitz \n(Min-Var)', color='r')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Cummulative Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()



#%% HSI as Benchmark

## Compute the return rate of index.
HSI_t = []
HSI_cum = []
for _period in range(0, T - M, H):
    HSI_t.append( (HSI.loc[Last_dates[_period+M]] - HS_open.loc[First_dates[_period+M]]) / HS_open.loc[First_dates[_period+M]] )
    HSI_cum.append( (HSI.loc[Last_dates[_period+M]] - HS_open.loc[First_dates[0]]) / HS_open.loc[First_dates[0]] )
    
HSI_t = pd.DataFrame(HSI_t, index=Last_dates[M:])
HSI_cum = pd.DataFrame(HSI_cum, index=Last_dates[M:])


plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, (Holding_naive.sum()-1) * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, (Holding_MinV.sum()-1) * 100, label='Constrained Markowitz \n(Min-Var)', color='r')
plt.plot(Last_dates[M:], HSI_t * 100, label='Hang Seng Index', color='k')
plt.axhline(0, c='b', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(Cum_R_naive.index, Cum_R_naive * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, Cum_R_MinV * 100, label='Constrained Markowitz \n(Min-Var)', color='r')
plt.plot(Last_dates[M:], HSI_cum * 100, label='Hang Seng Index', color='k')
plt.axhline(0, c='b', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Cummulative Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()




#%% Information Ratio - Naïve Strategy

## Calculate the active returns and information ratio.
Act_r_naive = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_naive.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_naive = pd.concat([ Act_r_naive, _act_r ], axis=1)
    
Act_r_naive = Act_r_naive.T

Act_r_naive.mean()
Act_r_naive.std()

Act_r_naive.mean() / Act_r_naive.std()



#%% Information Ratio - Constrained Markowitz

## Calculate the active returns and information ratio.
Act_r_MinV = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_MinV.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_MinV = pd.concat([ Act_r_MinV, _act_r ], axis=1)
    
Act_r_MinV = Act_r_MinV.T

Act_r_MinV.mean()
Act_r_MinV.std()

Act_r_MinV.mean() / Act_r_MinV.std()


plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, Act_r_naive * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, Act_r_MinV * 100, label='Constrained Markowitz \n(Min-Var)', color='r')
plt.axhline(0, c='b', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Active Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()

r_t = (Rolling_r[0].iloc[ -1 ] - Rolling_r[0].iloc[ 0 ] ) / Rolling_r[0].iloc[ 0 ]


#%% Tangency Portfolio

## Solve the optimal weights of tangency portfolio.
def Func_OptWeight_TangencyP(windows):
    r_t = (windows.iloc[-1] - windows.iloc[0] ) / windows.iloc[0]
    
    # Objective function: maximize portfolio Sharpe on the tangency point.
    obj_func = lambda w: - w.T @ r_t / np.sqrt(w.T @ windows.cov() @ w)
    
    # Set constraints
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights equals 1
                   {'type': 'ineq', 'fun': lambda w: w})  # Non-negativity constraint
    
    # Specify options with 'maxiter'
    options = {'maxiter': 1000000}
    
    # Solve the quadratic programming problem.
    # Using equal weights as the initial guess.
    result = scipy.optimize.minimize(
        obj_func, Weight_naive, method='SLSQP', constraints=constraints, options=options)
    w_Min = pd.DataFrame( result.x )
    
    # Remove small numerical calculated weights.
    w_Min[ w_Min < 10 ** -4 ] = 0

    return w_Min


## Calculate the optimal weight using rolling windows.
# No optimal weight calculated by the last rolling window.
Weight_TP = pd.DataFrame()
for _period in range(len(Rolling_r)-1): 
    weight_t = Func_OptWeight_TangencyP( Rolling_r[_period] )
    ## Scale up to make sure the sum of actual weights is 100%.
    weight_t = weight_t / weight_t.sum()
    Weight_TP = pd.concat([ Weight_TP, weight_t ], axis=1)

Weight_TP = Weight_TP.T
Weight_TP.index = First_dates[M:]
Weight_TP.columns = Stock_close.columns




#%% Highest Sharpe Strategy

## Compute the Sharpe ratios of each stock over rolling windows.
Sharpe_roll = pd.DataFrame()
for _period in range(M, T-1, H):
    Sharpe_t = (Stock_close.loc[ Last_dates[_period] ] - \
                    Stock_open.loc[ First_dates[_period] ]) / \
                Stock_open.loc[ First_dates[_period] ]
    Sharpe_roll = pd.concat([ Sharpe_roll, Sharpe_t ], axis=1) 
Sharpe_roll.columns = Last_dates[M:]


## Construct the weights of the Highest Sharpe Strategy that acts as a benchmark.
Weight_Sharpe = pd.DataFrame(0, index=Returns.columns, columns=range(len(Rolling_r)-1))
for _period in range(len(Rolling_r)-1):
    ## Find the index of the stocks with the highest Sharpe ratio among each holding periods.
    idx = Sharpe_roll.iloc[:, _period].nlargest(NumStock).index
    
    ## Naively invest stocks above for each holding period.
    Weight_Sharpe.loc[idx, _period] = 1 / NumStock
    
    
Weight_Sharpe.columns = First_dates[M:]
Weight_Sharpe = Weight_Sharpe.T


## Compute the changes in the holding within the holding period over time.
Exposures_Sharpe, Holding_Sharpe, Weight_Sharpe_end = Func_Holding(Weight_Sharpe)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_HighestSharpe, Var_HighestSharpe, _ = \
    Func_Performance(Weight_Sharpe, Weight_Sharpe_end, Exposures_Sharpe)

Sharpe_HighestSharpe.mean()
Sharpe_HighestSharpe.std()

Var_HighestSharpe.mean()
Var_HighestSharpe.std()


# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.plot(Last_dates[M:], HSI_t * 100, label='Hang Seng Index', color='k')
plt.plot(Holding_naive.columns, (Holding_naive.sum()-1) * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, (Holding_MinV.sum()-1) * 100, label='Constrained Markowitz \n(Min-Var)', color='b')
plt.plot(Holding_Sharpe.columns, (Holding_Sharpe.sum()-1) * 100, label='Highest Sharpe', color='r')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()




#%%% Information Ratio - Highest Sharpe Strategy

## Calculate the active returns and information ratio.
Act_r_Sharpe = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_Sharpe.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_Sharpe = pd.concat([ Act_r_Sharpe, _act_r ], axis=1)
    
Act_r_Sharpe = Act_r_Sharpe.T

Act_r_Sharpe.mean()
Act_r_Sharpe.std()

Act_r_Sharpe.mean() / Act_r_Sharpe.std()


plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, Act_r_naive * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, Act_r_MinV * 100, label='Constrained Markowitz \n(Min-Var)', color='r')
plt.plot(Holding_Sharpe.columns, Act_r_Sharpe * 100, label='Highest \nSharpe', color='k')
plt.axhline(0, c='b', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Active Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()




#%% Minimum Spanning Trees over Whole Period

## Compute the optimal weights of minimum variance portfolio using whole period data.
Weight_MinV_Whole = Func_OptWeight_MinV( Returns.corr() )
Weight_MinV_Whole.index = Returns.columns


## Construct the adjacency matrix and thus centrality by MST.
# If volatilty measure is adopted, rolling windows must be the input.
def Func_AdjM(dta, var='return', corr='Spear'):
    if corr == 'Spear':
        ## Construct the Spearman correlation matrix instead of Pearson.
        if var == 'return':
            Spear, _ = scipy.stats.spearmanr(dta)
        
        elif var == 'volatility':
            # Volatility must be collected period by period instead of daily data.
            rolling = pd.DataFrame()
            ## Collect the features from each rolling window.
            for period in range(len(dta)):
                rolling = pd.concat([rolling, dta[period].var()], axis=1)
            # rolling.columns = Last_dates[M-1 :]
            rolling = rolling.T
            
            Spear, _ = scipy.stats.spearmanr(rolling)
            
        ## Construct the distance matrix.
        Distance_mat = np.sqrt( 2 * (1 - Spear) )
            
    ## When Pearson correlation coefficient is adopted.
    elif corr == 'Pearson':
        if var == 'return':
            Distance_mat = np.sqrt( 2 * (1 - dta.corr()) )


    ## Construct the MST.
    MST = scipy.sparse.csgraph.minimum_spanning_tree(Distance_mat).toarray()

    ## Construct the adjacency matrix by MST.
    Adj_mat = (MST != 0) * 1
    Adj_mat = Adj_mat + Adj_mat.T


    ## Construct network.
    rows, cols = np.where(Adj_mat == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    ## Compute centrality.
    EC = nx.eigenvector_centrality(gr, max_iter=2000)
    Centrality = [ value for value in EC.values() ] 

    return MST, Adj_mat, Centrality, gr


## The index of the inputs must be the same.
# Compute the MST through rank correlation among returns.
MST_r, Adj_mat_r, Centrality_r, gr_r = Func_AdjM(Returns, var='return')
# Compute the MST through rank correlation among volatility.
MST_v, Adj_mat_v, Centrality_v, gr_v = Func_AdjM(Rolling_r, var='volatility')


## Set the style of the network sketched.
Style_network_r = pd.DataFrame( deepcopy(Centrality_r), index=Returns.columns )
Style_network_r = pd.concat([ Style_network_r, Weight_MinV_Whole ], axis=1)
# Style_network = pd.concat([ Style_network, Weight_MinV.iloc[0].T ], axis=1)
Style_network_r.columns = ['Centrality', 'Optimal Weights']

Style_network_v = pd.DataFrame( deepcopy(Centrality_v), index=Returns.columns )
Style_network_v = pd.concat([ Style_network_v, Weight_MinV_Whole ], axis=1)
# Style_network = pd.concat([ Style_network, Weight_MinV.iloc[0].T ], axis=1)
Style_network_v.columns = ['Centrality', 'Optimal Weights']


## Sketch the adjacency matrix as a network.
def Func_Network(dta, gr, opacity=1, node_MinSize=100, node_MaxSize=1000, figsize=(15,9), plot_style=1):
    # Specify colors of the nodes.
    dta['Color_1'] = 1
    dta['Color_2'] = 1
    dta['Color_3'] = 1
    dta['Color_4'] = opacity

    dta['Color Scaler'] = dta['Centrality']
    dta['Color Scaler'] = \
        3 * ( dta['Color Scaler'] - dta['Color Scaler'].min() ) / \
            ( dta['Color Scaler'].max() - dta['Color Scaler'].min() )
    scalar = dta['Color Scaler']

    dta.loc[:, 'Color_1'][ scalar <= 1 ] = 1 - scalar
    dta.loc[:, 'Color_1'][ scalar > 1 ]  = 0

    dta.loc[:, 'Color_2'][ (scalar > 1) & (scalar <= 2) ] = 2 - scalar
    dta.loc[:, 'Color_2'][ scalar > 2 ]  = 0

    dta.loc[:, 'Color_3'][ scalar > 2 ] = 3 - scalar
    # Make sure no numerical error occurred.
    dta.loc[:, 'Color_3'][ dta['Color_3'] < 10**-6 ] = 0

    colors = dta[['Color_1', 'Color_2', 'Color_3', 'Color_4']]
    colors = colors.reset_index(drop=True)
    colors.columns = [*range(4)]

    # Specify sizes of the nodes
    dta['Size Scaler'] = dta['Optimal Weights']
    dta['Size Scaler'] = \
        ( dta['Size Scaler'] - dta['Size Scaler'].min() ) / \
            ( dta['Size Scaler'].max() - dta['Size Scaler'].min() )
    sizes = node_MinSize + dta['Size Scaler'] * node_MaxSize

    plt.figure(figsize = figsize)
    ## Sketch the network.
    if plot_style == 1:
        layout = nx.spring_layout(gr,seed=942)  #105 #32 #942
    else:
        layout = nx.drawing.nx_agraph.graphviz_layout(gr, prog='twopi')
    nx.draw(gr, node_size=sizes, with_labels=False, node_color=colors, cmap='hsv', edgecolors='black', pos=layout)
    plt.show()

    ## Use for checking the setting of plotting style.
    return dta

Style_network_r = Func_Network(Style_network_r, gr_r, opacity=0.85, node_MinSize=100, node_MaxSize=3000, figsize=(15,9), plot_style=1)

Style_network_v = Func_Network(Style_network_v, gr_v, opacity=0.85, node_MinSize=100, node_MaxSize=3000, figsize=(15,9), plot_style=1)


## Print the rank correlation between centrality and the Markowitz optimal weights.
scipy.stats.spearmanr( Style_network_r['Centrality'], Style_network_r['Optimal Weights'] )[0]
scipy.stats.spearmanr( Style_network_v['Centrality'], Style_network_v['Optimal Weights'] )[0]




#%% Minimum Spanning Trees of the Individual Period.

MST_r1, Adj_mat_r1, Centrality_r1, gr_r1 = Func_AdjM(Rolling_r[0], var='return')

## Set the style of the network sketched.
Style_network_r1 = pd.DataFrame( deepcopy(Centrality_r1), index=Returns.columns )
Style_network_r1 = pd.concat([ Style_network_r1, Weight_MinV.iloc[0].T ], axis=1)
Style_network_r1.columns = ['Centrality', 'Optimal Weights']
Style_network_r1 = Func_Network(Style_network_r1, gr_r1, opacity=0.85, node_MinSize=100, node_MaxSize=3000, figsize=(15,9), plot_style=1)

scipy.stats.spearmanr( Style_network_r1['Centrality'], Style_network_r1['Optimal Weights'] )[0]


## Collect the Spearman correlations between centrality and optimal weights for each rolling windows.
Spearmans = []
for _period in range(len(Rolling_r)-1):
    _, _, c_t, _ = Func_AdjM(Rolling_r[_period], var='return')   
    spear_t = scipy.stats.spearmanr( c_t, Weight_MinV.iloc[_period].T )[0]
    Spearmans.append(spear_t)
Spearmans = pd.DataFrame(Spearmans, index=Last_dates[M:])

Spearmans[(Spearmans>0.1)].dropna()



## Sketch the MST of the rolling window with the highest Spearman correlations
# between centrality and optimal weights.
_, _, c_t, G = Func_AdjM(Rolling_r[22], var='return')
scipy.stats.spearmanr( c_t, Weight_MinV.iloc[22].T )[0]

Style_network_rh = pd.DataFrame( deepcopy(c_t), index=Returns.columns )
Style_network_rh = pd.concat([ Style_network_rh, Weight_MinV.iloc[22].T ], axis=1)
Style_network_rh.columns = ['Centrality', 'Optimal Weights']
Style_network_rh = Func_Network(Style_network_rh, G, opacity=0.85, node_MinSize=100, node_MaxSize=3000, figsize=(9,9), plot_style=1)


## Sketch the MST of the rolling window with the lowest Spearman correlations
# between centrality and optimal weights.
_, _, c_t, G = Func_AdjM(Rolling_r[27], var='return')
scipy.stats.spearmanr( c_t, Weight_MinV.iloc[27].T )[0]

Style_network_rl = pd.DataFrame( deepcopy(c_t), index=Returns.columns )
Style_network_rl = pd.concat([ Style_network_rl, Weight_MinV.iloc[27].T ], axis=1)
Style_network_rl.columns = ['Centrality', 'Optimal Weights']
Style_network_rl = Func_Network(Style_network_rl, G, opacity=0.85, node_MinSize=100, node_MaxSize=3000, figsize=(9,9), plot_style=1)


## Use Pearson correlation to sketch the MST
# of the rolling window with the highest Spearman correlations between centrality and optimal weights.
_, _, c_t, G = Func_AdjM(Rolling_r[22], var='return', corr='Pearson')
scipy.stats.spearmanr( c_t, Weight_MinV.iloc[22].T )[0]

Style_network_rh = pd.DataFrame( deepcopy(c_t), index=Returns.columns )
Style_network_rh = pd.concat([ Style_network_rh, Weight_MinV.iloc[22].T ], axis=1)
Style_network_rh.columns = ['Centrality', 'Optimal Weights']
Style_network_rh = Func_Network(Style_network_rh, G, opacity=0.85, node_MinSize=100, node_MaxSize=3000, figsize=(9,9), plot_style=1)


## Use Pearson correlation to sketch the MST
# of the rolling window with the lowest Spearman correlations between centrality and optimal weights.
_, _, c_t, G = Func_AdjM(Rolling_r[27], var='return', corr='Pearson')
scipy.stats.spearmanr( c_t, Weight_MinV.iloc[27].T )[0]

Style_network_rl = pd.DataFrame( deepcopy(c_t), index=Returns.columns )
Style_network_rl = pd.concat([ Style_network_rl, Weight_MinV.iloc[27].T ], axis=1)
Style_network_rl.columns = ['Centrality', 'Optimal Weights']
Style_network_rl = Func_Network(Style_network_rl, G, opacity=0.85, node_MinSize=100, node_MaxSize=3000, figsize=(9,9), plot_style=1)


## Collect the Spearman correlations between centrality and optimal weights for each rolling windows
# while using Pearson correlation of the stock returns instead of Spearman corr.
Pearson = []
for _period in range(len(Rolling_r)-1):
    _, _, c_t, _ = Func_AdjM(Rolling_r[_period], var='return', corr='Pearson')   
    pear_t = scipy.stats.spearmanr( c_t, Weight_MinV.iloc[_period].T )[0]
    Pearson.append(pear_t)
Pearson = pd.DataFrame(Pearson, index=Last_dates[M:])

Pearson[(Pearson>=-0.3)&(Pearson<-0.2)].dropna()
Pearson[(Pearson<-0.3)].dropna()




#%% ρ across Time - Spearman Correlation of Returns.

## Compute the ρ between Sharpe ratio and centrality through Spearman correlation.
Sharpe_HSI = (Stock_close.iloc[-1] - Stock_open.iloc[0]) / Stock_open.iloc[0] - rf
var_HSI = Returns.var()
Sharpe_HSI = Sharpe_HSI / var_HSI

_, _, Centrality_r, _ = Func_AdjM(Returns, var='return')

rho_hk_spear, _ = scipy.stats.spearmanr(Centrality_r, Sharpe_HSI)


## Create 60-days rolling windows for ρ computaiton time-seriesly.
Rolling_rho_60 = []
for _period in range(len(Last_dates)-1):
    ## Extract the 60-days data of daily returns.
    rolling_t = Returns.loc[ First_dates[_period] : Last_dates[_period+1] ]
    _, _, c_t, _ = Func_AdjM(rolling_t, var='return', corr='Spear')

    ## Compute the 60-days Sharpe ratio for each component stock.
    sharpe_t = Stock_close.loc[ Last_dates[_period+1] ] - Stock_open.loc[ First_dates[_period] ]
    sharpe_t = sharpe_t / Stock_open.loc[ First_dates[_period] ] - rf
    var_t = rolling_t.var()
    sharpe_t = sharpe_t / var_t
    
    rho_t, _ = scipy.stats.spearmanr(c_t, sharpe_t)
    Rolling_rho_60.append(rho_t)

Rolling_rho_60 = pd.DataFrame(Rolling_rho_60, index=Last_dates[1:])


## Create 120-days rolling windows for ρ computaiton time-seriesly.
Rolling_rho_120 = []
for _period in range(len(Last_dates)-3):
    ## Extract the 120-days data of daily returns.
    rolling_t = Returns.loc[ First_dates[_period] : Last_dates[_period+3] ]
    _, _, c_t, _ = Func_AdjM(rolling_t, var='return', corr='Spear')

    ## Compute the 120-days Sharpe ratio for each component stock.
    sharpe_t = Stock_close.loc[ Last_dates[_period+1] ] - Stock_open.loc[ First_dates[_period] ]
    sharpe_t = sharpe_t / Stock_open.loc[ First_dates[_period] ] - rf
    var_t = rolling_t.var()
    sharpe_t = sharpe_t / var_t
    
    rho_t, _ = scipy.stats.spearmanr(c_t, sharpe_t)
    Rolling_rho_120.append(rho_t)

Rolling_rho_120 = pd.DataFrame(Rolling_rho_120, index=Last_dates[3:])


## Compute the 120-days rolling means of the ρ under 60-days rolling windows.
Mean_rolling_rho = Rolling_rho_60.rolling(3)
Mean_rolling_rho = Mean_rolling_rho.mean()[2:]


## Sketch a plot for the time-series data of ρ's.
plt.figure(figsize=(12, 6))
plt.plot(Last_dates[1:], Rolling_rho_60, label='ρ', color='b')
plt.plot(Last_dates[3:], Mean_rolling_rho, label='Rolling Means (120) days', color='r')
plt.axhline(0, c='k', ls='--')
plt.axhline(rho_hk_spear, c='g', ls='--')
plt.xlabel('Last Date of the Rolling Windows', fontsize=20)
plt.ylabel('ρ', fontsize=20)
plt.title('Values of ρ',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()




#%% ρ across Time - Pearson Correlation of Returns.

## Compute the ρ between Sharpe ratio and centrality through Spearman correlation.
Sharpe_HSI = (Stock_close.iloc[-1] - Stock_open.iloc[0]) / Stock_open.iloc[0] - rf
var_HSI = Returns.var()
Sharpe_HSI = Sharpe_HSI / var_HSI

_, _, Centrality_r, _ = Func_AdjM(Returns, var='return', corr='Pearson')

rho_hk_pearson, _ = scipy.stats.spearmanr(Centrality_r, Sharpe_HSI)


## Create 60-days rolling windows for ρ computaiton time-seriesly.
Rolling_rho_60 = []
for _period in range(len(Last_dates)-1):
    ## Extract the 60-days data of daily returns.
    rolling_t = Returns.loc[ First_dates[_period] : Last_dates[_period+1] ]
    _, _, c_t, _ = Func_AdjM(rolling_t, var='return', corr='Pearson')

    ## Compute the 60-days Sharpe ratio for each component stock.
    sharpe_t = Stock_close.loc[ Last_dates[_period+1] ] - Stock_open.loc[ First_dates[_period] ]
    sharpe_t = sharpe_t / Stock_open.loc[ First_dates[_period] ] - rf
    var_t = rolling_t.var()
    sharpe_t = sharpe_t / var_t
    
    rho_t, _ = scipy.stats.spearmanr(c_t, sharpe_t)
    Rolling_rho_60.append(rho_t)

Rolling_rho_60 = pd.DataFrame(Rolling_rho_60, index=Last_dates[1:])


## Create 120-days rolling windows for ρ computaiton time-seriesly.
Rolling_rho_120 = []
for _period in range(len(Last_dates)-3):
    ## Extract the 120-days data of daily returns.
    rolling_t = Returns.loc[ First_dates[_period] : Last_dates[_period+3] ]
    _, _, c_t, _ = Func_AdjM(rolling_t, var='return', corr='Pearson')

    ## Compute the 120-days Sharpe ratio for each component stock.
    sharpe_t = Stock_close.loc[ Last_dates[_period+1] ] - Stock_open.loc[ First_dates[_period] ]
    sharpe_t = sharpe_t / Stock_open.loc[ First_dates[_period] ] - rf
    var_t = rolling_t.var()
    sharpe_t = sharpe_t / var_t
    
    rho_t, _ = scipy.stats.spearmanr(c_t, sharpe_t)
    Rolling_rho_120.append(rho_t)

Rolling_rho_120 = pd.DataFrame(Rolling_rho_120, index=Last_dates[3:])


## Sketch a plot for the time-series data of ρ's.
plt.figure(figsize=(12, 6))
plt.plot(Last_dates[1:], Rolling_rho_60, label='ρ', color='b')
plt.plot(Last_dates[3:], Rolling_rho_120, label='Rolling Means (120) days', color='r')
plt.axhline(0, c='k', ls='--')
plt.axhline(rho_hk_pearson, c='g', ls='--')
plt.xlabel('Last Date of the Rolling Windows', fontsize=20)
plt.ylabel('ρ', fontsize=20)
plt.title('Values of ρ',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()




#%% Highest Centrality Strategy - Spearman Correlation of Returns

## Require to specify which type of strategy is adopted.
# strategy = 'high'    : Highest centrality strategy;
# strategy = 'low'     : Lowest centrality strategy;
# strategy = 'rho'     : ρ-dependent strategy.
# strategy = 'reverse' : reverse ρ-dependent strategy.
def Func_CentralityPortf(dta, strategy, var='return', corr='Spear'):
    ## Collect the centrality and the corresponding ρ between the centrality and w*.
    Centralities = pd.DataFrame()
    rhos = []
    for period in range(len(dta)-1):
        ## When volatility measure is adopted to calculate the Spearman correlation of the stock returns,
        # each rolling window is further divided to monthly subperiods to calculate the 
        # dynamics of the volatility of the stock returns within current rolling window.
        if var == 'volatility':
            ## Collect the monthly subperiods of current rolling window.
            rolling_t = []
            for _i in range(M):
                windows = dta[period].loc[ First_dates[period + _i] : Last_dates[period + _i] ]
                rolling_t.append(windows)
                
            _, _, c_t, G_t = Func_AdjM(rolling_t, var=var, corr=corr)
            
        elif var == 'return':
            _, _, c_t, G_t = Func_AdjM(dta[period], var=var, corr=corr)
        
        c_t = pd.DataFrame(c_t)
        Centralities = pd.concat([Centralities, c_t], axis=1)
        
        ## Compute the Sharpe ratio for each component stock of current rolling window.
        close = Stock_close.loc[ dta[period].index[-1] ]
        openPrice = Stock_open.loc[dta[period].index[0] ]
        excessRet = (close - openPrice) / openPrice
        var_t = dta[period].var()
        sharpe_t = (excessRet - rf) / var_t

        rho_t, _ = scipy.stats.spearmanr(c_t, sharpe_t)
        rhos.append(rho_t)

    Centralities.index = Returns.columns
    Centralities.columns = Last_dates[M:]

    ## Construct the weights of the Highest Centrality Strategy that acts as a benchmark.
    weight_t = pd.DataFrame(0, index=Returns.columns, columns=range(len(Rolling_r)-1))
    for period in range(len(dta)-1):
        ## Find the index of the stocks with the highest/lowest centralities among each holding periods.
        if strategy == 'high' or (strategy=='rho' and rhos[period]>Rho_tilde) or (strategy=='reverse' and rhos[period]<=Rho_tilde):
            idx = Centralities.iloc[:, period].nlargest(NumStock).index
        elif strategy == 'low' or (strategy=='rho' and rhos[period]<=Rho_tilde) or (strategy=='reverse' and rhos[period]>Rho_tilde):
            idx = Centralities.iloc[:, period].nsmallest(NumStock).index
        
        ## Naively invest stocks above for each holding period.
        weight_t.loc[idx, period] = 1 / NumStock

    return weight_t


## Execute the highest centrality strategy with Spearman correlation of stock returns.
Weight_HCentralSpear_r = Func_CentralityPortf(Rolling_r, strategy='high', var='return', corr='Spear')

Weight_HCentralSpear_r = Weight_HCentralSpear_r.T
Weight_HCentralSpear_r.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_HCentralSpear_r, Holding_HCentralSpear_r, Weight_HCentralSpear_r_end = Func_Holding(Weight_HCentralSpear_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_HCentralSpear_r, Var_HCentralSpear_r, _ = \
    Func_Performance(Weight_HCentralSpear_r, Weight_HCentralSpear_r_end, Exposures_HCentralSpear_r)

Sharpe_HCentralSpear_r.mean()
Sharpe_HCentralSpear_r.std()

Var_HCentralSpear_r.mean()
Var_HCentralSpear_r.std()



#%%% Active Return - Spearman Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_HCentralSpear_r = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_HCentralSpear_r.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_HCentralSpear_r = pd.concat([ Act_r_HCentralSpear_r, _act_r ], axis=1)
    
Act_r_HCentralSpear_r = Act_r_HCentralSpear_r.T

Act_r_HCentralSpear_r.mean()
Act_r_HCentralSpear_r.std()

Act_r_HCentralSpear_r.mean() / Act_r_HCentralSpear_r.std()



#%% Highest Centrality Strategy - Spearman Correlation of Volatility

## Execute the highest centrality strategy with 
# Spearman correlation of the volatility of the stock returns.
Weight_HCentralSpear_var = Func_CentralityPortf(Rolling_r, strategy='high', var='volatility', corr='Spear')

Weight_HCentralSpear_var = Weight_HCentralSpear_var.T
Weight_HCentralSpear_var.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_HCentralSpear_var, Holding_HCentralSpear_var, Weight_HCentralSpear_var_end = Func_Holding(Weight_HCentralSpear_var)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_HCentralSpear_var, Var_HCentralSpear_var, _ = \
    Func_Performance(Weight_HCentralSpear_var, Weight_HCentralSpear_var_end, Exposures_HCentralSpear_var)

Sharpe_HCentralSpear_var.mean()
Sharpe_HCentralSpear_var.std()

Var_HCentralSpear_var.mean()
Var_HCentralSpear_var.std()



#%%% Active Return - Spearman Correlation of Volatility

## Calculate the active returns and information ratio.
Act_r_HCentralSpear_var = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_HCentralSpear_var.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_HCentralSpear_var = pd.concat([ Act_r_HCentralSpear_var, _act_r ], axis=1)
    
Act_r_HCentralSpear_var = Act_r_HCentralSpear_var.T

Act_r_HCentralSpear_var.mean()
Act_r_HCentralSpear_var.std()

Act_r_HCentralSpear_var.mean() / Act_r_HCentralSpear_var.std()




#%% Highest Centrality Strategy - Pearson Correlation of Returns

## Execute the highest centrality strategy with Pearson correlation coefficient of stock returns.
Weight_HCentralPearson_r = Func_CentralityPortf(Rolling_r, strategy='high', var='return', corr='Pearson')

Weight_HCentralPearson_r = Weight_HCentralPearson_r.T
Weight_HCentralPearson_r.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_HCentralPearson_r, Holding_HCentralPearson_r, Weight_HCentralPearson_r_end = Func_Holding(Weight_HCentralPearson_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_HCentralPearson_r, Var_HCentralPearson_r, _ = \
    Func_Performance(Weight_HCentralPearson_r, Weight_HCentralPearson_r_end, Exposures_HCentralPearson_r)

Sharpe_HCentralPearson_r.mean()
Sharpe_HCentralPearson_r.std()

Var_HCentralPearson_r.mean()
Var_HCentralPearson_r.std()



#%%% Active Return - Pearson Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_HCentralPearson_r = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_HCentralPearson_r.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_HCentralPearson_r = pd.concat([ Act_r_HCentralPearson_r, _act_r ], axis=1)
    
Act_r_HCentralPearson_r = Act_r_HCentralPearson_r.T

Act_r_HCentralPearson_r.mean()
Act_r_HCentralPearson_r.std()

Act_r_HCentralPearson_r.mean() / Act_r_HCentralPearson_r.std()




#%% Graph: Highest Centrality Strategy

# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.plot(Last_dates[M:], HSI_t * 100, label='Hang Seng Index', color='k')
plt.plot(Holding_naive.columns, (Holding_naive.sum()-1) * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, (Holding_MinV.sum()-1) * 100, label='Constrained Markowitz \n(Min-Var)', color='b')
plt.plot(Holding_Sharpe.columns, (Holding_Sharpe.sum()-1) * 100, label='Highest Sharpe', color='r')
plt.plot(Holding_HCentralPearson_r.columns, (Holding_HCentralPearson_r.sum()-1) * 100, label='Highest Centrality', color='m')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, Act_r_naive * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, Act_r_MinV * 100, label='Constrained Markowitz \n(Min-Var)', color='r')
plt.plot(Holding_Sharpe.columns, Act_r_Sharpe * 100, label='Highest Sharpe', color='k')
plt.plot(Holding_HCentralPearson_r.columns, Act_r_HCentralPearson_r * 100, label='Highest Centrality', color='m')
plt.axhline(0, c='b', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Active Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()




#%% Lowest Centrality Strategy - Spearman Correlation of Returns

## Execute the lowest centrality strategy with Spearman correlation of stock returns.
Weight_LCentralSpear_r = Func_CentralityPortf(Rolling_r, strategy='low', var='return', corr='Spear')

Weight_LCentralSpear_r = Weight_LCentralSpear_r.T
Weight_LCentralSpear_r.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_LCentralSpear_r, Holding_LCentralSpear_r, Weight_LCentralSpear_r_end = Func_Holding(Weight_LCentralSpear_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_LCentralSpear_r, Var_LCentralSpear_r, _ = \
    Func_Performance(Weight_LCentralSpear_r, Weight_LCentralSpear_r_end, Exposures_LCentralSpear_r)

Sharpe_LCentralSpear_r.mean()
Sharpe_LCentralSpear_r.std()

Var_LCentralSpear_r.mean()
Var_LCentralSpear_r.std()


    
#%%% Active Return - Spearman Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_LCentralSpear_r = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_LCentralSpear_r.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_LCentralSpear_r = pd.concat([ Act_r_LCentralSpear_r, _act_r ], axis=1)
    
Act_r_LCentralSpear_r = Act_r_LCentralSpear_r.T

Act_r_LCentralSpear_r.mean()
Act_r_LCentralSpear_r.std()

Act_r_LCentralSpear_r.mean() / Act_r_LCentralSpear_r.std()




#%% Lowest Centrality Strategy - Spearman Correlation of Volatility

## Execute the lowest centrality strategy with 
# Spearman correlation of the volatility of the stock returns.
Weight_LCentralSpear_var = Func_CentralityPortf(Rolling_r, strategy='low', var='volatility', corr='Spear')

Weight_LCentralSpear_var = Weight_LCentralSpear_var.T
Weight_LCentralSpear_var.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_LCentralSpear_var, Holding_LCentralSpear_var, Weight_LCentralSpear_var_end = Func_Holding(Weight_LCentralSpear_var)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_LCentralSpear_var, Var_LCentralSpear_var, _ = \
    Func_Performance(Weight_LCentralSpear_var, Weight_LCentralSpear_var_end, Exposures_LCentralSpear_var)

Sharpe_LCentralSpear_var.mean()
Sharpe_LCentralSpear_var.std()

Var_LCentralSpear_var.mean()
Var_LCentralSpear_var.std()



#%%% Active Return - Spearman Correlation of Volatility

## Calculate the active returns and information ratio.
Act_r_LCentralSpear_var = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_LCentralSpear_var.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_LCentralSpear_var = pd.concat([ Act_r_LCentralSpear_var, _act_r ], axis=1)
    
Act_r_LCentralSpear_var = Act_r_LCentralSpear_var.T

Act_r_LCentralSpear_var.mean()
Act_r_LCentralSpear_var.std()

Act_r_LCentralSpear_var.mean() / Act_r_LCentralSpear_var.std()




#%% Lowest Centrality Strategy - Pearson Correlation of Returns

## Execute the lowest centrality strategy with Pearson correlation coefficient of stock returns.
Weight_LCentralPearson_r = Func_CentralityPortf(Rolling_r, strategy='low', var='return', corr='Pearson')

Weight_LCentralPearson_r = Weight_LCentralPearson_r.T
Weight_LCentralPearson_r.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_LCentralPearson_r, Holding_LCentralPearson_r, Weight_LCentralPearson_r_end = Func_Holding(Weight_LCentralPearson_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_LCentralPearson_r, Var_LCentralPearson_r, _ = \
    Func_Performance(Weight_LCentralPearson_r, Weight_LCentralPearson_r_end, Exposures_LCentralPearson_r)

Sharpe_LCentralPearson_r.mean()
Sharpe_LCentralPearson_r.std()

Var_LCentralPearson_r.mean()
Var_LCentralPearson_r.std()



#%%% Active Return - Pearson Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_LCentralPearson_r = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_LCentralPearson_r.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_LCentralPearson_r = pd.concat([ Act_r_LCentralPearson_r, _act_r ], axis=1)
    
Act_r_LCentralPearson_r = Act_r_LCentralPearson_r.T

Act_r_LCentralPearson_r.mean()
Act_r_LCentralPearson_r.std()

Act_r_LCentralPearson_r.mean() / Act_r_LCentralPearson_r.std()




#%% Graph: Lowest Centrality Strategy

# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.plot(Last_dates[M:], HSI_t * 100, label='Hang Seng Index', color='k')
plt.plot(Holding_naive.columns, (Holding_naive.sum()-1) * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, (Holding_MinV.sum()-1) * 100, label='Constrained Markowitz \n(Min-Var)', color='b')
plt.plot(Holding_Sharpe.columns, (Holding_Sharpe.sum()-1) * 100, label='Highest Sharpe', color='r')
plt.plot(Holding_HCentralPearson_r.columns, (Holding_HCentralPearson_r.sum()-1) * 100, label='Highest Centrality', color='m')
plt.plot(Holding_LCentralSpear_var.columns, (Holding_LCentralSpear_var.sum()-1) * 100, label='Lowest Centrality', color='cyan')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, Act_r_naive * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, Act_r_MinV * 100, label='Constrained Markowitz \n(Min-Var)', color='r')
plt.plot(Holding_Sharpe.columns, Act_r_Sharpe * 100, label='Highest Sharpe', color='k')
plt.plot(Holding_HCentralPearson_r.columns, Act_r_HCentralPearson_r * 100, label='Highest Centrality', color='m')
plt.plot(Holding_LCentralSpear_var.columns, Act_r_LCentralSpear_var * 100, label='Lowest Centrality', color='cyan')
plt.axhline(0, c='b', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Active Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()




#%% ρ-Dependent Strategy - Spearman Correlation of Returns

## Decide which threshold is used.
if Rho_tilde == 'Spear':
    Rho_tilde = rho_hk_spear
elif Rho_tilde == 'Pearson':
    Rho_tilde = rho_hk_pearson

## Execute the ρ-dependent strategy with Spearman correlation of stock returns.
Weight_rhoSpear_r = Func_CentralityPortf(Rolling_r, strategy='rho', var='return', corr='Spear')

Weight_rhoSpear_r = Weight_rhoSpear_r.T
Weight_rhoSpear_r.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_rhoSpear_r, Holding_rhoSpear_r, Weight_rhoSpear_r_end = Func_Holding(Weight_rhoSpear_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_rhoSpear_r, Var_rhoSpear_r, _ = \
    Func_Performance(Weight_rhoSpear_r, Weight_rhoSpear_r_end, Exposures_rhoSpear_r)

Sharpe_rhoSpear_r.mean()
Sharpe_rhoSpear_r.std()

Var_rhoSpear_r.mean()
Var_rhoSpear_r.std()


    
#%%% Active Return - Spearman Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_rhoSpear_r = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_rhoSpear_r.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_rhoSpear_r = pd.concat([ Act_r_rhoSpear_r, _act_r ], axis=1)
    
Act_r_rhoSpear_r = Act_r_rhoSpear_r.T

Act_r_rhoSpear_r.mean()
Act_r_rhoSpear_r.std()

Act_r_rhoSpear_r.mean() / Act_r_rhoSpear_r.std()




#%% ρ-Dependent Strategy - Spearman Correlation of Volatility

## Execute the ρ-dependent strategy with 
# Spearman correlation of the volatility of the stock returns.
Weight_rhoSpear_var = Func_CentralityPortf(Rolling_r, strategy='rho', var='volatility', corr='Spear')

Weight_rhoSpear_var = Weight_rhoSpear_var.T
Weight_rhoSpear_var.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_rhoSpear_var, Holding_rhoSpear_var, Weight_rhoSpear_var_end = Func_Holding(Weight_rhoSpear_var)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_rhoSpear_var, Var_rhoSpear_var, _ = \
    Func_Performance(Weight_rhoSpear_var, Weight_rhoSpear_var_end, Exposures_rhoSpear_var)

Sharpe_rhoSpear_var.mean()
Sharpe_rhoSpear_var.std()

Var_rhoSpear_var.mean()
Var_rhoSpear_var.std()


    
#%%% Active Return - Spearman Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_rhoSpear_var = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_rhoSpear_var.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_rhoSpear_var = pd.concat([ Act_r_rhoSpear_var, _act_r ], axis=1)
    
Act_r_rhoSpear_var = Act_r_rhoSpear_var.T

Act_r_rhoSpear_var.mean()
Act_r_rhoSpear_var.std()

Act_r_rhoSpear_var.mean() / Act_r_rhoSpear_var.std()




#%% ρ-Dependent Strategy - Pearson Correlation of Returns

## Execute the ρ-dependent strategy with Pearson correlation coefficient of stock returns.
Weight_rhoPearson_r = Func_CentralityPortf(Rolling_r, strategy='rho', var='return', corr='Pearson')

Weight_rhoPearson_r = Weight_rhoPearson_r.T
Weight_rhoPearson_r.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_rhoPearson_r, Holding_rhoPearson_r, Weight_rhoPearson_r_end = Func_Holding(Weight_rhoPearson_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_rhoPearson_r, Var_rhoPearson_r, _ = \
    Func_Performance(Weight_rhoPearson_r, Weight_rhoPearson_r_end, Exposures_rhoPearson_r)

Sharpe_rhoPearson_r.mean()
Sharpe_rhoPearson_r.std()

Var_rhoPearson_r.mean()
Var_rhoPearson_r.std()



#%%% Active Return - Pearson Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_rhoPearson_r = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_rhoPearson_r.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_rhoPearson_r = pd.concat([ Act_r_rhoPearson_r, _act_r ], axis=1)
    
Act_r_rhoPearson_r = Act_r_rhoPearson_r.T

Act_r_rhoPearson_r.mean()
Act_r_rhoPearson_r.std()

Act_r_rhoPearson_r.mean() / Act_r_rhoPearson_r.std()



#%% Graph: ρ-Dependent Strategy

# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.plot(Last_dates[M:], HSI_t * 100, label='Hang Seng Index', color='k')
plt.plot(Holding_naive.columns, (Holding_naive.sum()-1) * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, (Holding_MinV.sum()-1) * 100, label='Constrained Markowitz', color='b')
plt.plot(Holding_Sharpe.columns, (Holding_Sharpe.sum()-1) * 100, label='Highest Sharpe', color='r')
plt.plot(Holding_HCentralPearson_r.columns, (Holding_HCentralPearson_r.sum()-1) * 100, label='Highest Centrality', color='m')
plt.plot(Holding_LCentralSpear_var.columns, (Holding_LCentralSpear_var.sum()-1) * 100, label='Lowest Centrality', color='cyan')
plt.plot(Holding_rhoPearson_r.columns, (Holding_rhoPearson_r.sum()-1) * 100, label='ρ-dependent', color='blueviolet')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=15)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, Act_r_naive * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, Act_r_MinV * 100, label='Constrained Markowitz', color='r')
plt.plot(Holding_Sharpe.columns, Act_r_Sharpe * 100, label='Highest Sharpe', color='k')
plt.plot(Holding_HCentralPearson_r.columns, Act_r_HCentralPearson_r * 100, label='Highest Centrality', color='m')
plt.plot(Holding_LCentralSpear_var.columns, Act_r_LCentralSpear_var * 100, label='Lowest Centrality', color='cyan')
plt.plot(Holding_rhoPearson_r.columns, Act_r_rhoPearson_r * 100, label='ρ-dependent', color='blueviolet')
plt.axhline(0, c='b', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Active Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()



#%% Reverse ρ-Dependent Strategy - Spearman Correlation of Returns

## Execute the reverse ρ-dependent strategy with Spearman correlation of stock returns.
Weight_ReverseRhoSpear_r = Func_CentralityPortf(Rolling_r, strategy='reverse', var='return', corr='Spear')

Weight_ReverseRhoSpear_r = Weight_ReverseRhoSpear_r.T
Weight_ReverseRhoSpear_r.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_ReverseRhoSpear_r, Holding_ReverseRhoSpear_r, Weight_ReverseRhoSpear_r_end = Func_Holding(Weight_ReverseRhoSpear_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_ReverseRhoSpear_r, Var_ReverseRhoSpear_r, _ = \
    Func_Performance(Weight_ReverseRhoSpear_r, Weight_ReverseRhoSpear_r_end, Exposures_ReverseRhoSpear_r)

Sharpe_ReverseRhoSpear_r.mean()
Sharpe_ReverseRhoSpear_r.std()

Var_ReverseRhoSpear_r.mean()
Var_ReverseRhoSpear_r.std()


    
#%%% Active Return - Spearman Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_ReverseRhoSpear_r = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_ReverseRhoSpear_r.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_ReverseRhoSpear_r = pd.concat([ Act_r_ReverseRhoSpear_r, _act_r ], axis=1)
    
Act_r_ReverseRhoSpear_r = Act_r_ReverseRhoSpear_r.T

Act_r_ReverseRhoSpear_r.mean()
Act_r_ReverseRhoSpear_r.std()

Act_r_ReverseRhoSpear_r.mean() / Act_r_ReverseRhoSpear_r.std()




#%% Reverse ρ-Dependent Strategy - Spearman Correlation of Volatility

## Execute the reverse ρ-dependent strategy with 
# Spearman correlation of the volatility of the stock returns.
Weight_ReverseRhoSpear_var = Func_CentralityPortf(Rolling_r, strategy='reverse', var='volatility', corr='Spear')

Weight_ReverseRhoSpear_var = Weight_ReverseRhoSpear_var.T
Weight_ReverseRhoSpear_var.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_ReverseRhoSpear_var, Holding_ReverseRhoSpear_var, Weight_ReverseRhoSpear_var_end = Func_Holding(Weight_ReverseRhoSpear_var)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_ReverseRhoSpear_var, Var_ReverseRhoSpear_var, _ = \
    Func_Performance(Weight_ReverseRhoSpear_var, Weight_ReverseRhoSpear_var_end, Exposures_ReverseRhoSpear_var)

Sharpe_ReverseRhoSpear_var.mean()
Sharpe_ReverseRhoSpear_var.std()

Var_ReverseRhoSpear_var.mean()
Var_ReverseRhoSpear_var.std()


    
#%%% Active Return - Spearman Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_ReverseRhoSpear_var = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_ReverseRhoSpear_var.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_ReverseRhoSpear_var = pd.concat([ Act_r_ReverseRhoSpear_var, _act_r ], axis=1)
    
Act_r_ReverseRhoSpear_var = Act_r_ReverseRhoSpear_var.T

Act_r_ReverseRhoSpear_var.mean()
Act_r_ReverseRhoSpear_var.std()

Act_r_ReverseRhoSpear_var.mean() / Act_r_ReverseRhoSpear_var.std()




#%% Reverse ρ-Dependent Strategy - Pearson Correlation of Returns

## Execute the reverse ρ-dependent strategy with Pearson correlation coefficient of stock returns.
Weight_rhoPearson_r = Func_CentralityPortf(Rolling_r, strategy='reverse', var='return', corr='Pearson')

Weight_rhoPearson_r = Weight_rhoPearson_r.T
Weight_rhoPearson_r.index = First_dates[M:]


## Compute the changes in the holding within the holding period over time.
Exposures_rhoPearson_r, Holding_rhoPearson_r, Weight_rhoPearson_r_end = Func_Holding(Weight_rhoPearson_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_rhoPearson_r, Var_rhoPearson_r, _ = \
    Func_Performance(Weight_rhoPearson_r, Weight_rhoPearson_r_end, Exposures_rhoPearson_r)

Sharpe_rhoPearson_r.mean()
Sharpe_rhoPearson_r.std()

Var_rhoPearson_r.mean()
Var_rhoPearson_r.std()



#%%% Active Return - Pearson Correlation of Returns

## Calculate the active returns and information ratio.
Act_r_rhoPearson_r = pd.DataFrame()
for _period in range(len(Rolling_r)-1):
    _act_r = Holding_rhoPearson_r.iloc[:, _period].sum() - 1 - HSI_t.iloc[_period]
    Act_r_rhoPearson_r = pd.concat([ Act_r_rhoPearson_r, _act_r ], axis=1)
    
Act_r_rhoPearson_r = Act_r_rhoPearson_r.T

Act_r_rhoPearson_r.mean()
Act_r_rhoPearson_r.std()

Act_r_rhoPearson_r.mean() / Act_r_rhoPearson_r.std()




#%% Graph: Reverse ρ-Dependent Strategy

# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.plot(Last_dates[M:], HSI_t * 100, label='Hang Seng Index', color='k')
plt.plot(Holding_naive.columns, (Holding_naive.sum()-1) * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, (Holding_MinV.sum()-1) * 100, label='Constrained Markowitz', color='b')
plt.plot(Holding_Sharpe.columns, (Holding_Sharpe.sum()-1) * 100, label='Highest Sharpe', color='r')
plt.plot(Holding_HCentralPearson_r.columns, (Holding_HCentralPearson_r.sum()-1) * 100, label='Highest Centrality', color='m')
plt.plot(Holding_LCentralSpear_var.columns, (Holding_LCentralSpear_var.sum()-1) * 100, label='Lowest Centrality', color='cyan')
plt.plot(Holding_ReverseRhoSpear_var.columns, (Holding_ReverseRhoSpear_var.sum()-1) * 100, label='Reverse ρ', color='goldenrod')
plt.plot(Holding_rhoPearson_r.columns, (Holding_rhoPearson_r.sum()-1) * 100, label='ρ-dependent', color='blueviolet')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=15)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(Holding_naive.columns, Act_r_naive * 100, label='Naïve Strategy', color='g')
plt.plot(Holding_MinV.columns, Act_r_MinV * 100, label='Constrained Markowitz', color='r')
plt.plot(Holding_Sharpe.columns, Act_r_Sharpe * 100, label='Highest Sharpe', color='k')
plt.plot(Holding_HCentralPearson_r.columns, Act_r_HCentralPearson_r * 100, label='Highest Centrality', color='m')
plt.plot(Holding_LCentralSpear_var.columns, Act_r_LCentralSpear_var * 100, label='Lowest Centrality', color='cyan')
plt.plot(Holding_ReverseRhoSpear_var.columns, Act_r_ReverseRhoSpear_var * 100, label='Reverse ρ', color='goldenrod')
plt.plot(Holding_rhoPearson_r.columns, Act_r_rhoPearson_r * 100, label='ρ-dependent', color='blueviolet')
plt.axhline(0, c='b', ls='--')
plt.xlabel('The Ending Day of Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Active Return of Different Strategies',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()





