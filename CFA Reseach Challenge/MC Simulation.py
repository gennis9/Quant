# Copyright (C) 2024 Gennis
'''
Initiated Date    : 2024/09/04
Last Updated Date : 2024/11/10
Aim: Run Monte Carlo simulation for the stock price of MediaTek.
    1. Revenue YoY by departmeent.
'''



# %% Import
import os
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import statsmodels.tsa as tsa
import statsmodels.graphics.tsaplots as tsaplot
import statsmodels.tsa.arima.model
import statsmodels.api as sm
import scipy
import scipy.stats as stat
import numpy as np
import random

%matplotlib inline

# Set the working directory of the main folder.
# Folder = r'D:\03Clouds\Google Drive\NSYSU\00CFARC\Modeling'
Folder = r'D:\03Programs_Clouds\Google Drive\NSYSU\00CFARC\Modeling'
os.chdir(Folder)

# Import the revenue data.
Table_Rev = pd.read_excel('PQ Model.xlsx', sheet_name='PQ Model')
Table_IS = pd.read_excel('PQ Model.xlsx', sheet_name='Quarterly')


# %% Data Preprocess

# Revenue data.
Data_Rev = deepcopy(Table_Rev)
# Earning data.
Data_IS  = deepcopy(Table_IS)


Data_Rev = Data_Rev.iloc[ [1, 7, 11, 17], 1:29]
Data_Rev.columns = Table_Rev.iloc[0, 1:29]
Data_Rev = Data_Rev.T.astype('float')
Data_Rev.columns = ['Mobile', 'IOT', 'Smart Home', 'PMIC']

# Revenue growth rate.
Data_QoQ = Data_Rev.pct_change()

## Start from 2021Q1.
Data_QoQ = Data_QoQ.iloc[4:]
Data_Rev = Data_Rev.iloc[4:]


# Gross margin rate.
Data_FS = pd.DataFrame( deepcopy(Table_Rev.iloc[76, 5:29]).T.astype('float') )
Data_FS.index = Data_Rev.index
Data_FS.columns = ['Margin Rate']


# Financial statment.
Data_FS['OC-NOI'] = (Data_IS.iloc[ 16, 55:79 ] - Data_IS.iloc[ 25, 55:79 ]).to_list()
Data_FS['Minority Interest'] = (Data_IS.iloc[ 74, 55:79 ]).to_list()


## Unify indices.
idx = [ i[:4] for i in Data_Rev.index.to_list() ]
Data_QoQ.index = idx
Data_Rev.index = idx
Data_FS.index = idx



# %% Parameters

## Use historical data to compute the future variance.
Hist_QoQ = Data_QoQ.iloc[:-9]
Var_QoQ  = Hist_QoQ.var() * len(Hist_QoQ) / ( len(Hist_QoQ) - 1 )
SD_QoQ   = Var_QoQ ** 0.5 / 10

Hist_GM = pd.DataFrame( Data_FS['Margin Rate'].iloc[:-9] )
Var_GM  = Hist_GM.var() * len(Hist_GM) / ( len(Hist_GM) - 1 )
SD_GM   = Var_GM ** 0.5


## The predicted values act as the possible means.
Mean_QoQ = Data_QoQ.iloc[-9:]
Mean_GM  = pd.DataFrame( Data_FS['Margin Rate'].iloc[-9:] )


# ESG Integration premium.
ESG_int = 0.1

## Exploit the historical values.
TaxRate   = 3242 / 29197
Share_out = 1588.728
EPS_2023  = 48.51

Current_price = 1415

Seed = 2024
Num_iteration = 100000




# %% 1. Stock Price Simulation

### Revenue Simulation
def Func_Norm(mean, sd, seed):
    np.random.seed(seed)
    return np.random.normal(mean, sd)

Prices = [] # 2025Q4 stock price
Paths = []
Prices_2026 = [] # 2026Q4 stock price
Seed = 2024
seed_count = 0
for t in range(Num_iteration):
    # t = 0
    
    ### Revenue and Gross Profit Simulation
    ## Generate a new (real) QoQ table by a simulation.
    sim_QoQ = pd.DataFrame( index = Mean_QoQ.index, columns = Mean_QoQ.columns )
    
    for col in range( len(sim_QoQ.columns) ):
        for row in range( len(sim_QoQ) ):
            ## Generate randomness with replicability.
            seed_count += 1
            seed_sim = Seed + seed_count
            
            sim_QoQ.iloc[row, col] = Func_Norm(Mean_QoQ.iloc[row, col], SD_QoQ[col], seed_sim)
            
    sim_QoQ = sim_QoQ.astype(float)
    
    ## Convert the QoQ table to revenue.
    Revenues = pd.DataFrame( Data_Rev.iloc[14].multiply((1+sim_QoQ.iloc[0]), axis=0), columns = ['4Q24'] )
    for row in range( 1, len(sim_QoQ) ):
        _col = idx[ row + 15 ]
        Revenues[_col] = Revenues.iloc[:, row-1].multiply((1+sim_QoQ.iloc[row]), axis=0)
    Revenues = Revenues.T
    
    
    ## Simulate the gross profits.
    sim_GM = pd.DataFrame( index = Mean_GM.index, columns = Mean_GM.columns )
    for row in range( len(sim_GM) ):
        seed_count += 1
        seed_sim = Seed + seed_count
        
        sim_GM.iloc[row] = Func_Norm(Mean_GM.iloc[row], SD_GM, seed_sim)
        
    sim_GM = sim_GM.astype(float)


    ### Earning Simulation
    ## Concatenate the revenues and gross profits of past quarters in 2024.
    Revenues = pd.concat([ Data_Rev.iloc[-12:-9], Revenues ], axis = 0)
    GMR      = pd.concat([ Hist_GM.iloc[-3:], sim_GM ], axis = 0)
    ## Compute the total revenue for each quarter.
    Revenues['Revenue'] = Revenues.sum(axis = 1)
    
    ## After-tax earning (common equity).
    # Operating margin.
    Earning_q = GMR.multiply(Revenues['Revenue'], axis=0)
    # Operating profit.
    Earning_q = Earning_q.sub( Data_FS['OC-NOI'][12:], axis = 0 )
    # After-tax earning.
    Earning_q = Earning_q * (1 - TaxRate)
    # After-tax earning (common equity).
    Earning_q = Earning_q.sub( Data_FS['Minority Interest'][12:], axis = 0 )
    
    ## Annual earnings.
    # Earning = [ Earning_q.iloc[4*i : 4*(i+1)].sum() for i in range(3) ]
    # Earning = pd.DataFrame(Earning, index = [2024, 2025, 2026], columns = ['Earning'])
    Earning = [ Earning_q.iloc[i : i+4].sum()[0] for i in range( len(Earning_q)-3 ) ]
    Earning = pd.DataFrame(Earning, index = Earning_q.index[3:], columns = ['Earning'])
    
    
    ### Stock Price Simulation
    EPS_terminal = Earning.iloc[-1][0] / Share_out
    Earning['EPS_current'] = Earning / Share_out
    Earning['Time'] = [ 1 + i * 0.25 for i in range( len(Earning) ) ]
    Earning['EPS_CAGR'] = (Earning['EPS_current'] / EPS_2023) ** (1/Earning['Time']) - 1
    Earning['Stock price'] = Earning['EPS_CAGR'] * 100 * Earning['EPS_current']
    # ESG integration.
    Earning['Stock price'] = Earning['Stock price'] * (1 + ESG_int)
    
    ## Normalize the stock prices simluated.
    Earning['Index'] = range(0, len(Earning))
    Earning['Baseline'] = Current_price + (Earning['Stock price'].iloc[-1] - Current_price) * (Earning['Index']/8)
    Earning['Norm_Price'] = Earning['Baseline'] + (Earning['Stock price'] - Earning['Baseline'])/(8.1-Earning['Index']) #* ((Earning['Index']+2)/10)**0.7
    Earning.loc['4Q24', 'Norm_Price'] = Current_price
    
    
    ## Price cannot be lower than null.
    Earning['Norm_Price'] = Earning['Norm_Price'].apply( lambda x: 0 if x<0 else x)
    # temp = pd.Series( [Current_price] + Earning['Norm_Price'].to_list(), index = idx[14:] )
    Paths.append( Earning['Norm_Price'] )
    Prices.append( Earning['Norm_Price'].iloc[4] )
    Prices_2026.append( Earning['Norm_Price'].iloc[8] )
    
    ## Time count.
    if t%100 == 0:
        print(t)




# %% Visualization - Price Distribution


# Prices = pd.read_excel('.\MC Results\\MonteCarlo_StockPrices_10K.xlsx', header = None)
# Prices = Prices.iloc[:, 0].to_list()

Current_price = 1415
Target_price  = 1628
Num_iteration = 100000


## Summary statistics.
pd.Series(Prices).describe()


## MC target prices.
plt.figure(figsize=(10, 6))

num_bin = 90
bin_edges = np.histogram_bin_edges(Prices, bins=num_bin)
colors = ['orange' if edge < Current_price else 'tomato' if edge > Target_price else 'peru' for edge in bin_edges[:-1]]

# Plot the histogram with the custom bin colors
n, bins, patches = plt.hist(Prices, bins=bin_edges, density=True, edgecolor='white')

# Color each bin according to its range
for i, patch in enumerate(patches):
    patch.set_facecolor(colors[i])


density = stat.gaussian_kde(Prices)
x_vals = np.linspace(min(Prices), max(Prices), 1000)
plt.plot(x_vals, density(x_vals), color='blue', lw=3)

plt.axvline(x=Current_price, color='black', linestyle='-', linewidth=4)
plt.axvline(x=Target_price, color='black', linestyle='-', linewidth=4)

_title_CP = 'Current Price \n $' + str(Current_price)
_title_TP = 'Target Price \n $' + str(Target_price)

plt.text(Current_price-50, -0.0001, _title_CP, ha='center', va='center', fontsize=12, color='black')
plt.text(Target_price+100, -0.0001, _title_TP, ha='center', va='center', fontsize=12, color='black')

# plt.text(-100, -0.0001, r'$\mathit{Source: Team \; Analysis}$', ha='center', va='center', fontsize=12, color='black')

plt.xticks([])

# plt.xlabel('Stock Price', fontsize=14)
y_label = plt.ylabel('Probability Density', fontsize=12, rotation=0, labelpad=-10)
y_label.set_position((y_label.get_position()[0], 1))

ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)    # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['bottom'].set_visible(False)  # Keep bottom spine
ax.spines['left'].set_visible(False)    # Keep left spine

# plt.grid()
plt.show()


## Summary statistics.
Stat = pd.Series(Prices)
Stat.describe()
np.sqrt( Stat.var() * Num_iteration / (Num_iteration - 1) )
Stat.skew()
Stat.kurtosis()

# Stat.quantile(0.01)
# Stat.quantile(0.99)
# # Stat.quantile(0.95)

# for i in range(1, 20):
#     print( Stat.quantile(0.05 * i) )


Prob_LowerCurrent = sum(x < Current_price for x in Prices) / Num_iteration
Prob_HigherTarget = sum(x > Target_price for x in Prices) / Num_iteration
Prob_Bull = sum(x > 1900 for x in Prices) / Num_iteration
Prob_Bear = sum(x < 870 for x in Prices) / Num_iteration
# Prob of positive return strategy with lower than target price.
1 - Prob_LowerCurrent - Prob_HigherTarget
# Prob of positive return strategy with more than target price.
1 - Prob_HigherTarget

Stat.quantile(1 - Prob_HigherTarget)



# %% Vis - PPT MC Distribution

## % of Sale
Prob_Sale = sum(x < Current_price * 0.85 for x in Prices) / Num_iteration
## % of Buy
Prob_Buy = sum(x > Current_price * 1.15 for x in Prices) / Num_iteration
## % of Hold
1 - Prob_Sale - Prob_Buy


## MC target prices.
plt.figure(figsize=(15, 6))

num_bin = 90
bin_edges = np.histogram_bin_edges(Prices, bins=num_bin)
colors = []
for edge in bin_edges[:-1]:
    if edge > 1900:
        colors.append('gold')
    elif edge < Current_price:
        colors.append('orange')
    elif edge > Target_price - 10:
        colors.append('tomato')
    else:
        colors.append('peru')


# Plot the histogram with the custom bin colors
n, bins, patches = plt.hist(Prices, bins=bin_edges, density=True, edgecolor='white')

# Color each bin according to its range
for i, patch in enumerate(patches):
    patch.set_facecolor(colors[i])


# density = stat.gaussian_kde(Prices)
# x_vals = np.linspace(min(Prices), max(Prices), 1000)
# plt.plot(x_vals, density(x_vals), color='blue', lw=3)

# plt.axvline(x=Current_price, color='black', linestyle='-', linewidth=4)
# plt.axvline(x=Target_price, color='black', linestyle='-', linewidth=4)

_title_CP = 'Current Price \n $' + str(Current_price)
_title_TP = 'Target Price \n $' + str(Target_price)

# plt.text(Current_price-30, -0.0001, _title_CP, ha='center', va='center', fontsize=12, color='black')
# plt.text(Target_price+30, -0.0001, _title_TP, ha='center', va='center', fontsize=12, color='black')
# plt.text(1000, -0.0001, '$1000', ha='center', va='center', fontsize=12, color='black')
# plt.text(2000, -0.0001, '$2000', ha='center', va='center', fontsize=12, color='black')

# plt.text(-100, -0.0001, r'$\mathit{Source: Team \; Analysis}$', ha='center', va='center', fontsize=12, color='black')

plt.xticks([])
plt.yticks([])

# plt.xlabel('Stock Price', fontsize=14)
# y_label = plt.ylabel('Probability Density', fontsize=12, rotation=0, labelpad=-10)
# y_label.set_position((y_label.get_position()[0], 1))

ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)    # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['bottom'].set_visible(False)  # Keep bottom spine
ax.spines['left'].set_visible(False)    # Keep left spine

# plt.grid()
plt.show()




# %% Visualization - Brownian Motion

## Simulated paths of stock price.
colors = ['red', 'gold', 'yellow', 'darkorange', 'peru', 'darkred', 'tomato']

plt.figure(figsize=(6, 6))

# Plot each stock path
for i, stock_path in enumerate(Paths):
    color = random.choice(colors)
    plt.plot(stock_path, color=color, alpha=0.1)

# Add title and labels
# plt.title('1000 Possible Stock Price Evolution Paths')
# plt.xlabel('Time')
y_label = plt.ylabel('Stock Price', fontsize=10, rotation=0, labelpad=-20)
y_label.set_position((y_label.get_position()[0] + 0.1, 1.01))
# plt.text(0.4, -700, r'$\mathit{Source: Team \; Analysis}$', ha='center', va='center', fontsize=12, color='black')

ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)    # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['bottom'].set_visible(False)  # Keep bottom spine
ax.spines['left'].set_visible(False)    # Keep left spine

# Show the plot
plt.show()




# %% Result Description of Brownian Motion

# Prices_2026 = pd.read_excel('.\MC Results\\MonteCarlo_StockPrices_2026_10K.xlsx', header = None)
# Prices_2026 = Prices_2026.iloc[:, 0].to_list()

Current_price = 1415
Target_price  = 1628
Num_iteration = 100000

## Discount the Brownian prices in 2026Q4 back to 2025Q4.
Prices_Discount = pd.DataFrame({'2026': Prices_2026})
Prices_Discount['Discount Factor'] = np.sqrt(Prices_Discount['2026'] / Current_price)
Prices_Discount['2025'] = Prices_Discount['Discount Factor'] * Current_price


## % of Sale
Prob_Sale_Brown = sum(x < Current_price * 0.9 for x in Prices_Discount['2025']) / Num_iteration
## % of Buy
Prob_Buy_Brown = sum(x > Current_price * 1.1 for x in Prices_Discount['2025']) / Num_iteration
## % of Hold
1 - Prob_Sale_Brown - Prob_Buy_Brown


Prices_Discount['2025'].describe()




# %% 2. Risk-adjusted Prices

def Func_Norm(mean, sd, seed):
    np.random.seed(seed)
    return np.random.normal(mean, sd)

def Func_Binary(prob_risk, seed):
    np.random.seed(seed)
    return np.random.choice([0, 1], p=[1 - prob_risk, prob_risk])

## Parameter setting.
Params = pd.DataFrame(index = ['Prob of Occurrence', 'Mean', 'SD'])
# MR1: Geopolitics Risk (Global, since 2025)
Params['GeoPol']      = [0.8, 0.025, 0.01]
# MR2: Business Cycle and Recession Risk (Global, since 2025)
Params['BusCycle']    = [0.1, 0.1, 0.05]
# MR3: Interest Rate Risk and Exchange Rate Risk (Global, Since 2025)
Params['Interest&Ex'] = [0.6, 0.003, 0.0015]
# CR1: Water Supply Restriction (Revenue, Q2)
Params['Water']       = [0.01, 15*1000, 5*1000]
# IR1: Saturated Market in Mobile Phone (Mobile)
Params['Mobile']      = [0.6, 0.1, 0.05]
# FR1: Supply Chain Cost Pressure (Global, Since 2025)
Params['SupplyChain'] = [0.6, 0.005, 0.002]
# FR2: Issue of Compatibility of ARM Architecture (IoT, Since 2Q26)
Params['ARM']         = [0.8, 0.4, 0.2]

## Test region.
# temp = []
# for i in range(100):
#     temp.append( Func_Norm(0.1, 0.08, i+100) )
# temp = pd.Series( temp ) * 100


### Risk-adjuste Revenue Simulation
Prices_adj = [] # 2025Q3 stock price
Paths_adj = []
Seed = 2024
# Num_iteration = 3000
for t in range(Num_iteration):
    # t = 0
    
    ### Revenue and Gross Profit Simulation
    ## Generate a new (real) QoQ table by a simulation.
    sim_QoQ = pd.DataFrame( index = Mean_QoQ.index, columns = Mean_QoQ.columns )
    
    ## Decide the occurrence of risks.
    Seed += 1
    dummy_geo = Func_Binary(Params.loc['Prob of Occurrence', 'GeoPol'], Seed)
    
    Seed += 1
    dummy_bus = Func_Binary(Params.loc['Prob of Occurrence', 'BusCycle'], Seed)
    
    Seed += 1
    dummy_iEx = Func_Binary(Params.loc['Prob of Occurrence', 'Interest&Ex'], Seed)
    
    Seed += 1
    dummy_mob = Func_Binary(Params.loc['Prob of Occurrence', 'Mobile'], Seed)
    
    Seed += 1
    dummy_arm = Func_Binary(Params.loc['Prob of Occurrence', 'ARM'], Seed)
    
    
    for col in range( len(sim_QoQ.columns) ):
        for row in range( len(sim_QoQ) ):
            # Pre-adjusted simulated QoQ.
            Seed += 1
            temp_QoQ = Func_Norm(Mean_QoQ.iloc[row, col], SD_QoQ[col], Seed)
            
            # Risk-adjustment.
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_geo * Func_Norm(Params.loc['Mean', 'GeoPol'], Params.loc['SD', 'GeoPol'], Seed))
            
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_bus * Func_Norm(Params.loc['Mean', 'BusCycle'], Params.loc['SD', 'BusCycle'], Seed))
            
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_iEx * Func_Norm(Params.loc['Mean', 'Interest&Ex'], Params.loc['SD', 'Interest&Ex'], Seed))
            
            
            if col == 0 or col == 3:
                Seed += 1
                temp_QoQ = temp_QoQ * (1 - dummy_mob * Func_Norm(Params.loc['Mean', 'Mobile'], Params.loc['SD', 'Mobile'], Seed))
            
            elif (col == 1 or col == 3) and row > 5:
                Seed += 1
                temp_QoQ = temp_QoQ * (1 - dummy_arm * Func_Norm(Params.loc['Mean', 'ARM'], Params.loc['SD', 'ARM'], Seed))
            
            
            ## Record the risk-adjusted QoQ.
            sim_QoQ.iloc[row, col] = temp_QoQ
            
    sim_QoQ = sim_QoQ.astype(float)

    
    ## Convert the QoQ table to revenue.
    Revenues = pd.DataFrame( Data_Rev.iloc[14].multiply((1+sim_QoQ.iloc[0]), axis=0), columns = ['4Q24'] )
    for row in range( 1, len(sim_QoQ) ):
        _col = idx[ row + 15 ]
        Revenues[_col] = Revenues.iloc[:, row-1].multiply((1+sim_QoQ.iloc[row]), axis=0)        
    Revenues = Revenues.T
    
    ## Simulate the gross profits.
    sim_GM = pd.DataFrame( index = Mean_GM.index, columns = Mean_GM.columns )
    for row in range( len(sim_GM) ):
        Seed += 1
        sim_GM.iloc[row] = Func_Norm(Mean_GM.iloc[row], SD_GM, Seed)
        
        ## Consider supply chain cost pressure.
        Seed += 1
        if Func_Binary(Params.loc['Prob of Occurrence', 'SupplyChain'], Seed) == 1:
            sim_GM.iloc[row] -= Func_Norm(Params.loc['Mean', 'SupplyChain'], Params.loc['SD', 'SupplyChain'], Seed)
        
    sim_GM = sim_GM.astype(float)


    ## Concatenate the revenues and gross profits of past quarters in 2024.
    Revenues = pd.concat([ Data_Rev.iloc[-12:-9], Revenues ], axis = 0)
    GMR      = pd.concat([ Hist_GM.iloc[-3:], sim_GM ], axis = 0)
    ## Compute the total revenue for each quarter.
    Revenues['Revenue'] = Revenues.sum(axis = 1)
    
    ## Consider climate risk.
    Revenues['QoQ'] = Revenues['Revenue'].pct_change()
    # 2025Q2.
    Seed += 1
    if Func_Binary(Params.loc['Prob of Occurrence', 'Water'], Seed) == 1:
        Revenues.loc['2Q25', 'Revenue'] -= Func_Norm(Params.loc['Mean', 'Water'], Params.loc['SD', 'Water'], Seed)
        for row in range(6):
            Revenues.iloc[6 + row, 4] = Revenues.iloc[5 + row, 4] * (1 + Revenues.iloc[6 + row, 5])
    # 2026Q2.
    Seed += 1
    if Func_Binary(Params.loc['Prob of Occurrence', 'Water'], Seed) == 1:
        Revenues.loc['2Q26', 'Revenue'] -= Func_Norm(Params.loc['Mean', 'Water'], Params.loc['SD', 'Water'], Seed)
        for row in range(2):
            Revenues.iloc[10 + row, 4] = Revenues.iloc[9 + row, 4] * (1 + Revenues.iloc[10 + row, 5])
    
    
    ### Earning Simulation
    ## After-tax earning (common equity).
    # Operating margin.
    Earning_q = GMR.multiply(Revenues['Revenue'], axis=0)
    # Operating profit.
    Earning_q = Earning_q.sub( Data_FS['OC-NOI'][12:], axis = 0 )
    # After-tax earning.
    Earning_q = Earning_q * (1 - TaxRate)
    # After-tax earning (common equity).
    Earning_q = Earning_q.sub( Data_FS['Minority Interest'][12:], axis = 0 )
    
    ## Annual earnings.
    # Earning = [ Earning_q.iloc[4*i : 4*(i+1)].sum() for i in range(3) ]
    # Earning = pd.DataFrame(Earning, index = [2024, 2025, 2026], columns = ['Earning'])
    Earning = [ Earning_q.iloc[i : i+4].sum()[0] for i in range( len(Earning_q)-3 ) ]
    Earning = pd.DataFrame(Earning, index = Earning_q.index[3:], columns = ['Earning'])
    
    
    ### Stock Price Simulation
    EPS_terminal = Earning.iloc[-1][0] / Share_out
    Earning['EPS_current'] = Earning / Share_out
    Earning['Time'] = [ 1 + i * 0.25 for i in range( len(Earning) ) ]
    Earning['EPS_CAGR'] = (Earning['EPS_current'] / EPS_2023) ** (1/Earning['Time']) - 1
    Earning['Stock price'] = Earning['EPS_CAGR'] * 100 * Earning['EPS_current']
    # ESG integration.
    Earning['Stock price'] = Earning['Stock price'] * (1 + ESG_int)
    
    ## Normalize the stock prices simluated.
    Earning['Index'] = range(1, len(Earning)+1)
    Earning['Baseline'] = Current_price + (Earning['Stock price'].iloc[-1] - Current_price) * (Earning['Index']/9)
    Earning['Norm_Price'] = Earning['Baseline'] + (Earning['Stock price'] - Earning['Baseline'])/(8.1-Earning['Index'])
    Earning.loc['4Q24', 'Norm_Price'] = Current_price
    
    
    ## Price cannot be lower than null.
    Earning['Norm_Price'] = Earning['Norm_Price'].apply( lambda x: 0 if x<0 else x)
    # temp = pd.Series( [Current_price] + Earning['Norm_Price'].to_list(), index = idx[14:] )
    Paths_adj.append( Earning['Norm_Price'] )
    Prices_adj.append( Earning['Norm_Price'].iloc[4] )
    
    ## Time count.
    if t%100 == 0:
        print(t)


    
# %% Visualization - Price Distribution (Risk Adjusted)


# Prices_adj = pd.read_excel('.\MC Results\\MonteCarlo_StockPrices_RiskAdj_10K.xlsx', header = None)
# Prices_adj = Prices_adj.iloc[:, 0].to_list()

Current_price = 1415
Target_price  = 1628
# Num_iteration = 3000


## Summary statistics.
pd.Series(Prices_adj).describe()


## MC target prices.
plt.figure(figsize=(10, 6))

num_bin = 70
bin_edges = np.histogram_bin_edges(Prices_adj, bins=num_bin)
colors = ['orange' if edge < Current_price else 'tomato' if edge > Target_price else 'peru' for edge in bin_edges[:-1]]

# Plot the histogram with the custom bin colors
n, bins, patches = plt.hist(Prices_adj, bins=bin_edges, density=True, edgecolor='white')

# Color each bin according to its range
for i, patch in enumerate(patches):
    patch.set_facecolor(colors[i])


density = stat.gaussian_kde(Prices_adj)
x_vals = np.linspace(min(Prices_adj), max(Prices_adj), 1000)
plt.plot(x_vals, density(x_vals), color='blue', lw=3)

plt.axvline(x=Current_price, color='black', linestyle='-', linewidth=4)
plt.axvline(x=Target_price, color='black', linestyle='-', linewidth=4)

_title_CP = 'Current Price \n $' + str(Current_price)
_title_TP = 'Target Price \n $' + str(Target_price)

plt.text(Current_price-80, -0.0001, _title_CP, ha='center', va='center', fontsize=12, color='black')
plt.text(Target_price+80, -0.0001, _title_TP, ha='center', va='center', fontsize=12, color='black')

# plt.text(-100, -0.0001, r'$\mathit{Source: Team \; Analysis}$', ha='center', va='center', fontsize=12, color='black')

plt.xticks([])

# plt.xlabel('Stock Price', fontsize=14)
y_label = plt.ylabel('Probability Density', fontsize=12, rotation=0, labelpad=-10)
y_label.set_position((y_label.get_position()[0], 0.95))

ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)    # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['bottom'].set_visible(False)  # Keep bottom spine
ax.spines['left'].set_visible(False)    # Keep left spine

# plt.grid()
plt.show()


## Summary statistics.
Stat = pd.Series(Prices_adj)
Stat.describe()
np.sqrt( Stat.var() * Num_iteration / (Num_iteration - 1) )
Stat.kurtosis()

Stat.quantile(0.01)
Stat.quantile(0.99)
# Stat.quantile(0.95)

for i in range(1, 20):
    print( Stat.quantile(0.05 * i) )


Prob_LowerCurrent_adj = sum(x < Current_price for x in Prices_adj) / Num_iteration
Prob_HigherTarget_adj = sum(x > Target_price for x in Prices_adj) / Num_iteration
# Prob of positive return strategy with lower than target price.
1 - Prob_LowerCurrent_adj - Prob_HigherTarget_adj
# Prob of positive return strategy with more than target price.
1 - Prob_HigherTarget_adj

Price_2025_adj = Stat.quantile(1 - Prob_HigherTarget)
EPS_DropRate = - (Price_2025_adj / Target_price - 1)
print( EPS_DropRate )




# %% Visualization - Brownian Motion (Risk Adjusted)

## Simulated paths of stock price.
colors = ['red', 'gold', 'yellow', 'darkorange', 'peru', 'darkred', 'tomato']

plt.figure(figsize=(6, 6))

# Plot each stock path
for i, stock_path in enumerate(Paths_adj):
    color = random.choice(colors)
    plt.plot(stock_path, color=color, alpha=0.1)

# Add title and labels
# plt.title('1000 Possible Stock Price Evolution Paths_adj')
# plt.xlabel('Time')
y_label = plt.ylabel('Stock Price', fontsize=10, rotation=0, labelpad=-20)
y_label.set_position((y_label.get_position()[0] + 0.1, 1.02))
# plt.text(0.4, -700, r'$\mathit{Source: Team \; Analysis}$', ha='center', va='center', fontsize=12, color='black')

ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)    # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['bottom'].set_visible(True)  # Keep bottom spine
ax.spines['left'].set_visible(True)    # Keep left spine

# Show the plot
plt.show()




# %% Waterfall Chart

## Parameter setting.
PE_FA = 21
PE_ESGInt = PE_FA * ESG_int
Current_price = 1415
Target_price  = 1628

## Allocate the waterfall among various risks.
Params = Params.T
Params['Risk Proportion'] = Params['Prob of Occurrence'].multiply(Params['Mean'])
Params.loc['Mobile', 'Risk Proportion'] *= 0.3
Params.loc['ARM', 'Risk Proportion'] *= 0.05
Params.loc['Water', 'Risk Proportion'] = Params.loc['Water', 'Prob of Occurrence'] * 0.1
Params.loc['SupplyChain', 'Risk Proportion'] *= 1/0.48
# Params['EPS Drop Proportion'] = Params['Risk Proportion'] * EPS_DropRate / Params['Risk Proportion'].sum() * (PE_FA + PE_ESGInt)
Params['Price Drop Proportion'] = Params['Risk Proportion'] * EPS_DropRate / Params['Risk Proportion'].sum() * Target_price

# Params['Risk Proportion'].sum()
# Params['Risk Proportion'].sort_values()
# Params['EPS Drop Proportion'].sum()


plt.figure(figsize=(5, 2.5))

## Sketch the waterfall chart.
categories = ['Target \n Price', 
              'MR1', 'MR2', 'MR3', 'CR1', 'IR1', 'FR1', 'FR2', 'Price \n under \n Risks']
values = [ -round(num, 1) for num in Params['Price Drop Proportion'] ]
values = [Target_price] + values + [int( Target_price * (1 - EPS_DropRate))]
coordinates, temp = [0, Target_price], Target_price
for _i in range(1, len(values) - 2):
    temp += values[_i]
    coordinates.append( temp )
coordinates.append( 0 )

# Colors for up and down
colors = ['peru'] + ['tomato'] * 7 + ['peru']

# Plot the bars
bars = plt.bar(categories, values, bottom=coordinates, color=colors)

# Add value labels on top of each bar
# coord_text = [Target_price] + [Target_price] * 7 + [values[-1]]
coord_text_fallProp = [Target_price] + coordinates[1:-1] + [values[-1]]

for i, (bar, value) in enumerate(zip(bars, values)):
    y = coord_text_fallProp[i]
    value = int(round(value, 0))
    if i == 0 or i == len(values) - 1:  # For the first and last bars, do not show '+'
        plt.text(bar.get_x() + bar.get_width() / 2, y, f'{value}', ha='center', va='bottom')
    else:
        plt.text(bar.get_x() + bar.get_width() / 2, y, f'{value:+}', ha='center', va='bottom')

plt.ylim(1400, 1650)


# Add a horizontal line for current price and the corresponding text.
plt.axhline(y=Current_price, color='k', linestyle='--', linewidth=2)
plt.text(len(categories)/4, Current_price + 10, "Current Price: 1415", color='k', va='center', ha='left', fontsize=10)



# Modify the spines to remove right, left, and top boundaries
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

# Change the bottom spine color to orange
ax.spines['bottom'].set_color('orange')

# Adjust y-axis ticks to only show specific values
# ax.set_yticks([19, 20, 21, 22])

# plt.title('Waterfall Chart')
y_label = plt.ylabel('Price (NT$)', fontsize=12, rotation=0, labelpad=-10)
y_label.set_position((y_label.get_position()[0], 1.03))
plt.show()




# %% 3. Bull Market Simulation

def Func_Binary(prob_risk, seed):
    np.random.seed(seed)
    return np.random.choice([0, 1], p=[1 - prob_risk, prob_risk])

## The growth in revenue is 1.33x in bear market expectation.
Bull_QoQ = Mean_QoQ + abs( Mean_QoQ.mean() - Hist_QoQ.mean() ) / 3
Bull_GM  = Mean_GM + 0.003


## Parameter setting.
Params = pd.DataFrame(index = ['Prob of Occurrence', 'Mean', 'SD'])
# MR1: Geopolitics Risk (Global, since 2025)
Params['GeoPol']      = [0.8, 0.02, 0.01]
# MR2: Business Cycle and Recession Risk (Global, since 2025)
Params['BusCycle']    = [0.02, 0.15, 0.05]
# MR3: Interest Rate Risk and Exchange Rate Risk (Global, Since 2025)
Params['Interest&Ex'] = [0.5, 0.003, 0.0015]
# CR1: Water Supply Restriction (Revenue, Q2)
Params['Water']       = [0.005, 15*1000, 5*1000]
# IR1: Saturated Market in Mobile Phone (Mobile)
Params['Mobile']      = [0.5, 0.1, 0.05]
# FR1: Highly Competitive Intensity and Product Underperformance (Global, Since 2025)
Params['Compet']      = [0.1, 0.2, 0.05]
# FR2: Issue of Compatibility of ARM Architecture (IoT, Since 2Q26)
Params['ARM']         = [0.8, 0.2, 0.07]

## 5% increase in occurrence prob of risks, and double the variance.
Params.loc['Prob of Occurrence'] += 0.05
Params.loc['SD'] *= np.sqrt(2)


### Risk-adjuste Revenue Simulation
Prices_bull = [] # 2025Q3 stock price
Paths_bull = []
Seed = 2024
for t in range(Num_iteration):
    # t = 0
    
    ### Revenue and Gross Profit Simulation
    ## Generate a new (real) QoQ table by a simulation.
    sim_QoQ = pd.DataFrame( index = Mean_QoQ.index, columns = Mean_QoQ.columns )
    
    ## Decide the occurrence of risks.
    Seed += 1
    dummy_geo = Func_Binary(Params.loc['Prob of Occurrence', 'GeoPol'], Seed)
    
    Seed += 1
    dummy_bus = Func_Binary(Params.loc['Prob of Occurrence', 'BusCycle'], Seed)
    
    Seed += 1
    dummy_iEx = Func_Binary(Params.loc['Prob of Occurrence', 'Interest&Ex'], Seed)
    
    Seed += 1
    dummy_com = Func_Binary(Params.loc['Prob of Occurrence', 'Compet'], Seed)
    
    Seed += 1
    dummy_mob = Func_Binary(Params.loc['Prob of Occurrence', 'Mobile'], Seed)
    
    Seed += 1
    dummy_arm = Func_Binary(Params.loc['Prob of Occurrence', 'ARM'], Seed)
    
    
    for col in range( len(sim_QoQ.columns) ):
        for row in range( len(sim_QoQ) ):
            # Pre-adjusted simulated QoQ.
            Seed += 1
            temp_QoQ = Func_Norm(Bull_QoQ.iloc[row, col], SD_QoQ[col], Seed)
            
            # Risk-adjustment.
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_geo * Func_Norm(Params.loc['Mean', 'GeoPol'], Params.loc['SD', 'GeoPol'], Seed))
            
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_bus * Func_Norm(Params.loc['Mean', 'BusCycle'], Params.loc['SD', 'BusCycle'], Seed))
            
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_iEx * Func_Norm(Params.loc['Mean', 'Interest&Ex'], Params.loc['SD', 'Interest&Ex'], Seed))
            
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_com * Func_Norm(Params.loc['Mean', 'Compet'], Params.loc['SD', 'Compet'], Seed))
            
            
            if col == 0:
                Seed += 1
                temp_QoQ = temp_QoQ * (1 - dummy_mob * Func_Norm(Params.loc['Mean', 'Mobile'], Params.loc['SD', 'Mobile'], Seed))
            
            elif col == 1 and row > 5:
                Seed += 1
                temp_QoQ = temp_QoQ * (1 - dummy_arm * Func_Norm(Params.loc['Mean', 'ARM'], Params.loc['SD', 'ARM'], Seed))
            
            
            ## Record the risk-adjusted QoQ.
            sim_QoQ.iloc[row, col] = temp_QoQ
            
    sim_QoQ = sim_QoQ.astype(float)

    
    ## Convert the QoQ table to revenue.
    Revenues = pd.DataFrame( Data_Rev.iloc[14].multiply((1+sim_QoQ.iloc[0]), axis=0), columns = ['4Q24'] )
    for row in range( 1, len(sim_QoQ) ):
        _col = idx[ row + 15 ]
        Revenues[_col] = Revenues.iloc[:, row-1].multiply((1+sim_QoQ.iloc[row]), axis=0)        
    Revenues = Revenues.T
    
    ## Simulate the gross profits.
    sim_GM = pd.DataFrame( index = Bull_GM.index, columns = Bull_GM.columns )
    for row in range( len(sim_GM) ):
        Seed += 1
        sim_GM.iloc[row] = Func_Norm(Bull_GM.iloc[row], SD_GM, Seed)
        
    sim_GM = sim_GM.astype(float)


    ## Concatenate the revenues and gross profits of past quarters in 2024.
    Revenues = pd.concat([ Data_Rev.iloc[-12:-9], Revenues ], axis = 0)
    GMR      = pd.concat([ Hist_GM.iloc[-3:], sim_GM ], axis = 0)
    ## Compute the total revenue for each quarter.
    Revenues['Revenue'] = Revenues.sum(axis = 1)
    
    ## Consider climate risk.
    Revenues['QoQ'] = Revenues['Revenue'].pct_change()
    # 2025Q2.
    Seed += 1
    if Func_Binary(Params.loc['Prob of Occurrence', 'Water'], Seed) == 1:
        Revenues.loc['2Q25', 'Revenue'] -= Func_Norm(Params.loc['Mean', 'Water'], Params.loc['SD', 'Water'], Seed)
        for row in range(6):
            Revenues.iloc[6 + row, 4] = Revenues.iloc[5 + row, 4] * (1 + Revenues.iloc[6 + row, 5])
    # 2026Q2.
    Seed += 1
    if Func_Binary(Params.loc['Prob of Occurrence', 'Water'], Seed) == 1:
        Revenues.loc['2Q26', 'Revenue'] -= Func_Norm(Params.loc['Mean', 'Water'], Params.loc['SD', 'Water'], Seed)
        for row in range(2):
            Revenues.iloc[10 + row, 4] = Revenues.iloc[9 + row, 4] * (1 + Revenues.iloc[10 + row, 5])
    
    
    ### Earning Simulation
    ## After-tax earning (common equity).
    # Operating margin.
    Earning_q = GMR.multiply(Revenues['Revenue'], axis=0)
    # Operating profit.
    Earning_q = Earning_q.sub( Data_FS['OC-NOI'][12:], axis = 0 )
    # After-tax earning.
    Earning_q = Earning_q * (1 - TaxRate)
    # After-tax earning (common equity).
    Earning_q = Earning_q.sub( Data_FS['Minority Interest'][12:], axis = 0 )
    
    ## Annual earnings.
    # Earning = [ Earning_q.iloc[4*i : 4*(i+1)].sum() for i in range(3) ]
    # Earning = pd.DataFrame(Earning, index = [2024, 2025, 2026], columns = ['Earning'])
    Earning = [ Earning_q.iloc[i : i+4].sum()[0] for i in range( len(Earning_q)-3 ) ]
    Earning = pd.DataFrame(Earning, index = Earning_q.index[3:], columns = ['Earning'])
    
    
    ### Stock Price Simulation
    EPS_terminal = Earning.iloc[-1][0] / Share_out
    Earning['EPS_current'] = Earning / Share_out
    Earning['Time'] = [ 1 + i * 0.25 for i in range( len(Earning) ) ]
    Earning['EPS_CAGR'] = (Earning['EPS_current'] / EPS_2023) ** (1/Earning['Time']) - 1
    Earning['Stock price'] = Earning['EPS_CAGR'] * 100 * Earning['EPS_current']
    # ESG integration.
    Earning['Stock price'] = Earning['Stock price'] * (1 + ESG_int)
    
    ## Normalize the stock prices simluated.
    Earning['Index'] = range(1, len(Earning)+1)
    Earning['Baseline'] = Current_price + (Earning['Stock price'].iloc[-1] - Current_price) * (Earning['Index']/9)
    Earning['Norm_Price'] = Earning['Baseline'] + (Earning['Stock price'] - Earning['Baseline'])/(9.1-Earning['Index'])
    
    
    ## Price cannot be lower than null.
    Earning['Norm_Price'] = Earning['Norm_Price'].apply( lambda x: 0 if x<0 else x)
    temp = pd.Series( [Current_price] + Earning['Norm_Price'].to_list(), index = idx[14:] )
    Paths_bull.append(temp)
    Prices_bull.append( temp.iloc[4] )
    
    ## Time count.
    if t%100 == 0:
        print(t)


## Summary statistics.
Stat = pd.Series(Prices_bull)
Stat.describe()

Prob_LowerCurrent = sum(x < Current_price for x in Prices) / Num_iteration
Prob_HigherTarget = sum(x > Target_price for x in Prices) / Num_iteration
# Prob of positive return strategy with lower than target price.
1 - Prob_LowerCurrent - Prob_HigherTarget
# Prob of positive return strategy with more than target price.
1 - Prob_HigherTarget

## Bull Price
Bull_price = Stat.quantile(1 - Prob_HigherTarget)




# %% 4. Bear Market Simulation

def Func_Binary(prob_risk, seed):
    np.random.seed(seed)
    return np.random.choice([0, 1], p=[1 - prob_risk, prob_risk])

## The growth in revenue is 1.33x in bear market expectation.
Bear_QoQ = Mean_QoQ - abs( Mean_QoQ.mean() - Hist_QoQ.mean() ) / 2
Bear_GM  = Mean_GM - 0.003


## Parameter setting.
Params = pd.DataFrame(index = ['Prob of Occurrence', 'Mean', 'SD'])
# MR1: Geopolitics Risk (Global, since 2025)
Params['GeoPol']      = [0.8, 0.02, 0.01]
# MR2: Business Cycle and Recession Risk (Global, since 2025)
Params['BusCycle']    = [0.02, 0.15, 0.05]
# MR3: Interest Rate Risk and Exchange Rate Risk (Global, Since 2025)
Params['Interest&Ex'] = [0.5, 0.003, 0.0015]
# CR1: Water Supply Restriction (Revenue, Q2)
Params['Water']       = [0.005, 15*1000, 5*1000]
# IR1: Saturated Market in Mobile Phone (Mobile)
Params['Mobile']      = [0.5, 0.1, 0.05]
# FR1: Highly Competitive Intensity and Product Underperformance (Global, Since 2025)
Params['Compet']      = [0.1, 0.2, 0.05]
# FR2: Issue of Compatibility of ARM Architecture (IoT, Since 2Q26)
Params['ARM']         = [0.8, 0.2, 0.07]

## 5% increase in occurrence prob of risks, and double the variance.
Params.loc['Prob of Occurrence'] += 0.05
Params.loc['SD'] *= np.sqrt(2)


### Risk-adjuste Revenue Simulation
Prices_bear = [] # 2025Q3 stock price
Paths_bear = []
Seed = 2024
for t in range(Num_iteration):
    # t = 0
    
    ### Revenue and Gross Profit Simulation
    ## Generate a new (real) QoQ table by a simulation.
    sim_QoQ = pd.DataFrame( index = Mean_QoQ.index, columns = Mean_QoQ.columns )
    
    ## Decide the occurrence of risks.
    Seed += 1
    dummy_geo = Func_Binary(Params.loc['Prob of Occurrence', 'GeoPol'], Seed)
    
    Seed += 1
    dummy_bus = Func_Binary(Params.loc['Prob of Occurrence', 'BusCycle'], Seed)
    
    Seed += 1
    dummy_iEx = Func_Binary(Params.loc['Prob of Occurrence', 'Interest&Ex'], Seed)
    
    Seed += 1
    dummy_com = Func_Binary(Params.loc['Prob of Occurrence', 'Compet'], Seed)
    
    Seed += 1
    dummy_mob = Func_Binary(Params.loc['Prob of Occurrence', 'Mobile'], Seed)
    
    Seed += 1
    dummy_arm = Func_Binary(Params.loc['Prob of Occurrence', 'ARM'], Seed)
    
    
    for col in range( len(sim_QoQ.columns) ):
        for row in range( len(sim_QoQ) ):
            # Pre-adjusted simulated QoQ.
            Seed += 1
            temp_QoQ = Func_Norm(Bear_QoQ.iloc[row, col], SD_QoQ[col], Seed)
            
            # Risk-adjustment.
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_geo * Func_Norm(Params.loc['Mean', 'GeoPol'], Params.loc['SD', 'GeoPol'], Seed))
            
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_bus * Func_Norm(Params.loc['Mean', 'BusCycle'], Params.loc['SD', 'BusCycle'], Seed))
            
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_iEx * Func_Norm(Params.loc['Mean', 'Interest&Ex'], Params.loc['SD', 'Interest&Ex'], Seed))
            
            Seed += 1
            temp_QoQ = temp_QoQ * (1 - dummy_com * Func_Norm(Params.loc['Mean', 'Compet'], Params.loc['SD', 'Compet'], Seed))
            
            
            if col == 0:
                Seed += 1
                temp_QoQ = temp_QoQ * (1 - dummy_mob * Func_Norm(Params.loc['Mean', 'Mobile'], Params.loc['SD', 'Mobile'], Seed))
            
            elif col == 1 and row > 5:
                Seed += 1
                temp_QoQ = temp_QoQ * (1 - dummy_arm * Func_Norm(Params.loc['Mean', 'ARM'], Params.loc['SD', 'ARM'], Seed))
            
            
            ## Record the risk-adjusted QoQ.
            sim_QoQ.iloc[row, col] = temp_QoQ
            
    sim_QoQ = sim_QoQ.astype(float)

    
    ## Convert the QoQ table to revenue.
    Revenues = pd.DataFrame( Data_Rev.iloc[14].multiply((1+sim_QoQ.iloc[0]), axis=0), columns = ['4Q24'] )
    for row in range( 1, len(sim_QoQ) ):
        _col = idx[ row + 15 ]
        Revenues[_col] = Revenues.iloc[:, row-1].multiply((1+sim_QoQ.iloc[row]), axis=0)        
    Revenues = Revenues.T
    
    ## Simulate the gross profits.
    sim_GM = pd.DataFrame( index = Bear_GM.index, columns = Bear_GM.columns )
    for row in range( len(sim_GM) ):
        Seed += 1
        sim_GM.iloc[row] = Func_Norm(Bear_GM.iloc[row], SD_GM, Seed)
        
    sim_GM = sim_GM.astype(float)


    ## Concatenate the revenues and gross profits of past quarters in 2024.
    Revenues = pd.concat([ Data_Rev.iloc[-12:-9], Revenues ], axis = 0)
    GMR      = pd.concat([ Hist_GM.iloc[-3:], sim_GM ], axis = 0)
    ## Compute the total revenue for each quarter.
    Revenues['Revenue'] = Revenues.sum(axis = 1)
    
    ## Consider climate risk.
    Revenues['QoQ'] = Revenues['Revenue'].pct_change()
    # 2025Q2.
    Seed += 1
    if Func_Binary(Params.loc['Prob of Occurrence', 'Water'], Seed) == 1:
        Revenues.loc['2Q25', 'Revenue'] -= Func_Norm(Params.loc['Mean', 'Water'], Params.loc['SD', 'Water'], Seed)
        for row in range(6):
            Revenues.iloc[6 + row, 4] = Revenues.iloc[5 + row, 4] * (1 + Revenues.iloc[6 + row, 5])
    # 2026Q2.
    Seed += 1
    if Func_Binary(Params.loc['Prob of Occurrence', 'Water'], Seed) == 1:
        Revenues.loc['2Q26', 'Revenue'] -= Func_Norm(Params.loc['Mean', 'Water'], Params.loc['SD', 'Water'], Seed)
        for row in range(2):
            Revenues.iloc[10 + row, 4] = Revenues.iloc[9 + row, 4] * (1 + Revenues.iloc[10 + row, 5])
    
    
    ### Earning Simulation
    ## After-tax earning (common equity).
    # Operating margin.
    Earning_q = GMR.multiply(Revenues['Revenue'], axis=0)
    # Operating profit.
    Earning_q = Earning_q.sub( Data_FS['OC-NOI'][12:], axis = 0 )
    # After-tax earning.
    Earning_q = Earning_q * (1 - TaxRate)
    # After-tax earning (common equity).
    Earning_q = Earning_q.sub( Data_FS['Minority Interest'][12:], axis = 0 )
    
    ## Annual earnings.
    # Earning = [ Earning_q.iloc[4*i : 4*(i+1)].sum() for i in range(3) ]
    # Earning = pd.DataFrame(Earning, index = [2024, 2025, 2026], columns = ['Earning'])
    Earning = [ Earning_q.iloc[i : i+4].sum()[0] for i in range( len(Earning_q)-3 ) ]
    Earning = pd.DataFrame(Earning, index = Earning_q.index[3:], columns = ['Earning'])
    
    
    ### Stock Price Simulation
    EPS_terminal = Earning.iloc[-1][0] / Share_out
    Earning['EPS_current'] = Earning / Share_out
    Earning['Time'] = [ 1 + i * 0.25 for i in range( len(Earning) ) ]
    Earning['EPS_CAGR'] = (Earning['EPS_current'] / EPS_2023) ** (1/Earning['Time']) - 1
    Earning['Stock price'] = Earning['EPS_CAGR'] * 100 * Earning['EPS_current']
    # ESG integration.
    Earning['Stock price'] = Earning['Stock price'] * (1 + ESG_int)
    
    ## Normalize the stock prices simluated.
    Earning['Index'] = range(1, len(Earning)+1)
    Earning['Baseline'] = Current_price + (Earning['Stock price'].iloc[-1] - Current_price) * (Earning['Index']/9)
    Earning['Norm_Price'] = Earning['Baseline'] + (Earning['Stock price'] - Earning['Baseline'])/(9.1-Earning['Index'])
    
    
    ## Price cannot be lower than null.
    Earning['Norm_Price'] = Earning['Norm_Price'].apply( lambda x: 0 if x<0 else x)
    temp = pd.Series( [Current_price] + Earning['Norm_Price'].to_list(), index = idx[14:] )
    Paths_bear.append(temp)
    Prices_bear.append( temp.iloc[4] )
    
    ## Time count.
    if t%100 == 0:
        print(t)


## Summary statistics.
Stat = pd.Series(Prices_bear)
Stat.describe()

Prob_LowerCurrent = sum(x < Current_price for x in Prices) / Num_iteration
Prob_HigherTarget = sum(x > Target_price for x in Prices) / Num_iteration
# Prob of positive return strategy with lower than target price.
1 - Prob_LowerCurrent - Prob_HigherTarget
# Prob of positive return strategy with more than target price.
1 - Prob_HigherTarget

## Bull Price
Bear_price = Stat.quantile(1 - Prob_HigherTarget)




# %% Export
'''
_FileName = 'MonteCarlo_StockPrices_10K.xlsx'
with pd.ExcelWriter(_FileName) as _writer:  
    pd.Series(Prices).to_excel(_writer, sheet_name='MTK', index=False, header=False)
    
_FileName = 'MonteCarlo_StockPrices_RiskAdj_10K.xlsx'
with pd.ExcelWriter(_FileName) as _writer:  
    pd.Series(Prices_adj).to_excel(_writer, sheet_name='MTK', index=False, header=False)
    
_FileName = 'MonteCarlo_StockPrices_2026_10K.xlsx'
with pd.ExcelWriter(_FileName) as _writer:  
    pd.Series(Prices_2026).to_excel(_writer, sheet_name='MTK', index=False, header=False)

'''













