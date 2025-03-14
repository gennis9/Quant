# Copyright (C) 2024 Gennis
'''
Initiated Date    : 2024/05/16
Last Updated Date : 2024/05/22
Aim: Fork the paris trading model of Marianna Brunetti & Roberta De Luca (2023)
Input Data: Component stock prices of TWSE 50 and S&P 100.
'''

#%% Enviornment

import yfinance as yf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import seaborn as sns
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd # library for data analysis
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents

## Suppress the warning.
pd.options.mode.chained_assignment = None  # default='warn'




#%% S&P 100

## Web crawler.
wikiurl="https://en.wikipedia.org/wiki/S%26P_100"
table_class="wikitable sortable jquery-tablesorter"
response=requests.get(wikiurl)
# print(response.status_code)
# parse data from the html into a beautifulsoup object
soup = BeautifulSoup(response.text, 'html.parser')
indiatable=soup.find_all('table',{'class':"wikitable"})
df=pd.read_html(str(indiatable))
# convert list to dataframe
df=pd.DataFrame(df[1])
print(df.head())
all_symbol = " ".join(df["Symbol"].values)
print(all_symbol)

tmp_info = yf.Tickers(all_symbol)
data =yf.download(all_symbol, start='2018-01-01' , end='2023-12-31')



#%% Data Cleaning

tmp_info = yf.Tickers(all_symbol)
data_info = pd.DataFrame([tmp_info.tickers[tmp_symbol].info for tmp_symbol in df["Symbol"].values])
data_info.index = df["Symbol"].values
data_daily_prices = data["Adj Close"]
tmp_na_count = data_daily_prices.isna().sum()
data_daily_prices2 = data_daily_prices.loc[:,tmp_na_count==0] ## Remove missing values
data_info2 = data_info.loc[tmp_na_count==0,:]
data_daily_returns = data_daily_prices2.pct_change() # Return rate: (Pt = (pt-pt_-1)/pt_-1)
# Closing prices
data_1_adj = data_daily_prices
# Return rates
new_data_return = data_daily_returns
# Log prices
log_data = np.log(data_1_adj)
## Remove missing values.
drop_na_return = new_data_return.dropna()
drop_na_return = drop_na_return.reset_index()




#%% TWSE 50

taiwan50_symbols =  ("2330.TW", "2317.TW", "2454.TW", "6505.TW", "2412.TW",
                      "3008.TW", "1301.TW", "1303.TW", "2308.TW", "2002.TW",
                      "2881.TW", "2882.TW", "2303.TW", "2301.TW", "1216.TW",
                      "5871.TW", "2886.TW", "2891.TW", "3045.TW", "2395.TW",
                      "2884.TW", "2880.TW", "4904.TW", "2885.TW", "1101.TW",
                      "2892.TW", "6415.TW", "2883.TW", "2890.TW", "3711.TW",
                      "5876.TW", "2887.TW", "2888.TW", "1210.TW", "3044.TW",
                      "3037.TW", "2357.TW", "2327.TW", "1102.TW", "2207.TW",
                      "2382.TW", "3231.TW", "9914.TW", "1402.TW", "2603.TW",
                      "2610.TW", "3702.TW", "2383.TW", "2324.TW", "2474.TW")

## Download closing prices.
data_tw = yf.download(taiwan50_symbols, start='2018-01-01' , end='2023-12-31')
data_tw = data_tw.dropna(axis = 0)
# Closing prices
data_tw_adj = data_tw["Adj Close"]
# Return rates
data_tw_return = data_tw_adj.copy()
data_tw_return = data_tw_return.pct_change()

# Log price
log_data_tw = np.log(data_tw_adj)
## Remove missing values.
drop_na_return_tw = data_tw_return.dropna()
drop_na_return_tw = drop_na_return_tw.reset_index()




#%% Summary Statistics

data['Adj Close'].describe(include='all')
data['Adj Close'].describe(include=None)

zz=data_tw['Adj Close']

data.dropna(axis=1)['Adj Close'].describe(include='all')


pd.melt(data.dropna(axis=1)['Volume']).describe(include='all').astype(str)


pd.melt(data_tw.dropna(axis=1)['Adj Close']).describe(include='all').astype(str)

zz=data_tw['Adj Close']


#%% hrt & hpt

## Set parameter values
k = 0.06
initial_sigma = 0.017
## Compute hr_t
hr_t_df = drop_na_return[['Date']].copy()

# hr_t formula
for col in drop_na_return.columns:
    if col == 'Date':
        continue # Skip date

    # Demean
    u_t = drop_na_return[col] - drop_na_return[col].mean()
    sigma_t = np.full(len(drop_na_return), initial_sigma)

    for i in range(1, len(sigma_t)):
        sigma_t[i] = np.sqrt(k * sigma_t[i - 1]**2 + (1 - k) * u_t.iloc[i]**2)

    hr_t = drop_na_return[col] / sigma_t
    hr_t_df[col] = hr_t
hr_t_df.set_index('Date', inplace=True)
hr_t_df = hr_t_df / 100



def extract_periods(start_year, hr_t_df):
    periods = []
    start_date = pd.to_datetime(f"{start_year}-01-01")
    end_date = hr_t_df.index.max()
    while start_date + pd.DateOffset(months=12) <= end_date:
        formation_start = start_date
        formation_end = start_date + pd.DateOffset(months=12) - pd.DateOffset(days=1)
        formation_period = hr_t_df.loc[formation_start:formation_end]
        periods.append(formation_period)
        start_date = start_date + pd.DateOffset(months=6)
    return periods

def calculate_correlations(df):
    return df.corr()

def johansen_coint_test(df, det_order=-1, k_ar_diff=1):
    result = coint_johansen(df, det_order, k_ar_diff)
    return result

def get_top_pairs(corr_matrix, top_n=4851):
    pairs = []
    sorted_pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    for i in range(1, top_n + 1):
        pairs.append(sorted_pairs.index[i])
    return pairs

def calculate_mu_beta_sigma(df, pairs):
    results = {}
    for pair in pairs:
        stock1 = df[pair[0]]
        stock2 = df[pair[1]]
        test_result = johansen_coint_test(df[[pair[0], pair[1]]])
        beta_vector = test_result.evec[:, 0]
        mu = beta_vector[0]
        beta = beta_vector[1]
        spread = stock1 - beta * stock2 - mu
        mean_spread = spread.mean()
        sigma = spread.std()
        results[pair] = (mu, beta, sigma,mean_spread)
    return results

def calculate_mean(df):
    return df.mean()




#%% Formation period information

start_year = 2018

# Formation period
periods = extract_periods(start_year, hr_t_df)

results = {}
johansen_test_counts = {}

for formation_period in periods:
    corr_matrix = calculate_correlations(formation_period)
    top_pairs = get_top_pairs(corr_matrix)

    valid_pairs = []
    test_count = 0
    for pair in top_pairs:
        test_result = johansen_coint_test(log_data[list(pair)])
        test_count += 1
        if test_result.lr1[0] > test_result.cvt[0, 1]:
            valid_pairs.append(pair)
        if len(valid_pairs) == 20:
            break

    pair_params = calculate_mu_beta_sigma(log_data, valid_pairs)
    results[(formation_period.index[0], formation_period.index[-1])] = pair_params
    johansen_test_counts[(formation_period.index[0], formation_period.index[-1])] = test_count

    # Mean of formation period
    formation_mean = calculate_mean(formation_period)
    print(f"Formation period {formation_period.index[0].date()} to {formation_period.index[-1].date()} mean values:")
    print(formation_mean)
    print()

# Result
for period, pairs in results.items():
    print(f"Period {period[0].date()} to {period[1].date()}:")
    for pair, params in pairs.items():
        mu, beta, sigma, mean_spread = params
        print(f"  {pair}: mu={mu:.6f}, beta={beta:.6f}, sigma={sigma:.6f}, mean_spread={mean_spread:.6f}")
    print(f"Johansen cointegration tests run: {johansen_test_counts[period]}")
    print()




#%% Trading

## Save the performance of trading methods.
MeanReturn = []
StdDeviation = []
Sharpe = []
PercentOfPositiveMonths = []

## Slice the trading period.
Trading_StartDates, Trading_EndDates = [], []
for _key1, _key2 in results.keys():
    Trading_StartDates.append(_key1)
    Trading_EndDates.append(_key2)

Trading_StartDates = Trading_StartDates[2:] + [Trading_EndDates[-2] + pd.DateOffset(days=4)] + \
                     [Trading_EndDates[-1] + pd.DateOffset(days=1)] 

Trading_EndDates = Trading_EndDates[1:] + [log_data.index[-1]]


Trading_periods = []
for _t in range(len(Trading_StartDates)):
    Trading_periods.append( log_data.loc[ Trading_StartDates[_t] : Trading_EndDates[_t] ] )



## Extract the information from previous steps (pairs and parameters).
Info_allPeriods = []
for _value in results.values():
    Info_allPeriods.append(_value)


## Define trading rules.
def Func_Trading(period, pairs, parmas):    
    ## Extract the historical prices of the cointegration pair during the trading period.
    Pairs_current = Trading_periods[ period ][pairs]
    Pairs_current['Price_A'] = np.exp( Pairs_current[ pairs[0] ] )
    Pairs_current['Price_B'] = np.exp( Pairs_current[ pairs[1] ] )
    
    ## Compute standardized spreads.
    Pairs_current['Spread'] = Pairs_current[ pairs[1] ] - (parmas[0] + parmas[1] * Pairs_current[ pairs[0] ])
    Pairs_current['Spread'] = ( Pairs_current['Spread'] - parmas[3] ) / parmas[2]
    
    ## Detect signals.
    Pairs_current['Signal'] = 0
    Pairs_current['Signal'][ Pairs_current['Spread'] >= 2 ] = 1  # Long stock A, short stock B.
    Pairs_current['Signal'][ Pairs_current['Spread'] <= -2 ] = -1  # Long stock B, short stock A.
    
    
    ## Compute the number of stock, unrealized profit/loss, and realized wealth in hands.
    Pairs_current['#Share_A'] = 0
    Pairs_current['#Share_B'] = 0
    Pairs_current['Unrealized P/L'] = 0
    Pairs_current['Realized Wealth'] = 0
    
    
    ## Check for the first day.
    # Open interest.
    if Pairs_current.iloc[0, 5] == 1:
        Pairs_current.iloc[0, 6] = 1 / Pairs_current.iloc[0, 2]
        Pairs_current.iloc[0, 7] = -1 / Pairs_current.iloc[0, 3]
    elif Pairs_current.iloc[0, 5] == -1:
        Pairs_current.iloc[0, 6] = -1 / Pairs_current.iloc[0, 2]
        Pairs_current.iloc[0, 7] = 1 / Pairs_current.iloc[0, 3]
            
            
    ## During the period, for each trading day.
    for _row in range(1, len(Pairs_current) - 1):
        # Open interest.
        if Pairs_current.iloc[_row, 5] == 1 and Pairs_current.iloc[_row-1, 5] == 0:
            # Buy stock A, Sell stock B.
            Pairs_current.iloc[_row, 6] = 1 / Pairs_current.iloc[_row, 2]
            Pairs_current.iloc[_row, 7] = -1 / Pairs_current.iloc[_row, 3]
            Pairs_current.iloc[_row, 9] = Pairs_current.iloc[_row-1, 9]
        
        elif Pairs_current.iloc[_row, 5] == -1 and Pairs_current.iloc[_row-1, 5] == 0:
            # Buy stock B, Sell stock A.
            Pairs_current.iloc[_row, 6] = -1 / Pairs_current.iloc[_row, 2]
            Pairs_current.iloc[_row, 7] = 1 / Pairs_current.iloc[_row, 3]
            Pairs_current.iloc[_row, 9] = Pairs_current.iloc[_row-1, 9]
            
        # Holding.
        elif (Pairs_current.iloc[_row, 5] == 1 and Pairs_current.iloc[_row-1, 5] == 1) or (Pairs_current.iloc[_row, 5] == -1 and Pairs_current.iloc[_row-1, 5] == -1):
            # Maintain exposure.
            Pairs_current.iloc[_row, 6] = Pairs_current.iloc[_row-1, 6]
            Pairs_current.iloc[_row, 7] = Pairs_current.iloc[_row-1, 7]
            Pairs_current.iloc[_row, 9] = Pairs_current.iloc[_row-1, 9]
            # Update current unrealized profit and loss.
            Pairs_current.iloc[_row, 8] = Pairs_current.iloc[_row-1, 9] + \
                                            Pairs_current.iloc[_row, 6] * Pairs_current.iloc[_row, 2] + \
                                            Pairs_current.iloc[_row, 7] * Pairs_current.iloc[_row, 3]
        
        # Close position.
        elif Pairs_current.iloc[_row, 5] == 0 and (Pairs_current.iloc[_row-1, 5] == 1 or Pairs_current.iloc[_row-1, 5] == -1):
            # Realize the profit and loss.
            Pairs_current.iloc[_row, 9] = Pairs_current.iloc[_row-1, 9]+ \
                                            Pairs_current.iloc[_row-1, 6] * Pairs_current.iloc[_row, 2] + \
                                            Pairs_current.iloc[_row-1, 7] * Pairs_current.iloc[_row, 3]
    
        # Zero exposure.
        elif Pairs_current.iloc[_row, 5] == 0 and Pairs_current.iloc[_row-1, 5] == 0:
            # Maintain wealth.
            Pairs_current.iloc[_row, 9] = Pairs_current.iloc[_row-1, 9]
    
    
        ## Close all position at the end of the period.
        if Pairs_current.iloc[-1, 5] != 0:
            Pairs_current.iloc[-1, 9] = Pairs_current.iloc[-2, 9]+ \
                                            Pairs_current.iloc[-2, 6] * Pairs_current.iloc[-1, 2] + \
                                            Pairs_current.iloc[-2, 7] * Pairs_current.iloc[-1, 3]


    ## Output the final wealth.
    return Pairs_current


### Detect trading signals through standardized spreads.
Trading_returns = []
positive_counts = 0
for _t in range(len(Trading_periods)):
    returns_pair = 0
    # For each cointegration pair in current trading period.
    for _j in range(20):
        ## Extract each cointegration pair for corresponding trading period.
        stock_A = list(Info_allPeriods[ _t ].keys())[ _j ][0]
        stock_B = list(Info_allPeriods[ _t ].keys())[ _j ][1]
        
        ## Extract the historical parameters.
        params_mu = list(Info_allPeriods[ _t ].values())[ _j ][0]
        params_beta = list(Info_allPeriods[ _t ].values())[ _j ][1]
        params_sigma = list(Info_allPeriods[ _t ].values())[ _j ][2]
        params_mean = list(Info_allPeriods[ _t ].values())[ _j ][3]
        
        ## Start to trade.
        Trading_Record = Func_Trading(_t, [stock_A, stock_B], [params_mu, params_beta, params_sigma, params_mean])
        
        # Record the final gain of the current pairs trading in current period.
        returns_pair += Trading_Record.iloc[-1, 9]
        
        if Trading_Record.iloc[-1, 9] > 0:
            positive_counts += 1
        
    # Record the cumulative wealth for all of 20 pairs trading of current period.
    Trading_returns.append( returns_pair )

MeanReturn.append( sum(Trading_returns) / len(Trading_periods) / 100 )
StdDeviation.append( np.std(Trading_returns, ddof=1) / 10 )
Sharpe.append( sum(Trading_returns) / np.std(Trading_returns, ddof=1) / 10 / len(Trading_periods) )
PercentOfPositiveMonths.append( positive_counts / 20 / len(Trading_periods) )




#%% S&P100: Method 2

zeros_df = pd.DataFrame(0, index=data_1_adj.index, columns=data_1_adj.columns)
zeros_df.iloc[0, :] = data_1_adj.iloc[0, :]
for i in range(1, len(data_1_adj)):
    zeros_df.iloc[i, :] = zeros_df.iloc[i - 1, :] * (1 + hr_t_df.iloc[i - 1, :])

hp_t = zeros_df
# log-price

log_hp_t = np.log(hp_t)
periods_2 = extract_periods(start_year, log_hp_t)


## Save results.
results_2 = {}
johansen_test_counts_2 = {}

# Second formation period
for formation_period in periods_2:
    corr_matrix = calculate_correlations(formation_period)
    top_pairs = get_top_pairs(corr_matrix)

    valid_pairs = []
    test_count = 0
    for pair in top_pairs:
        test_result = johansen_coint_test(log_data[list(pair)])
        test_count += 1
        if test_result.lr1[0] > test_result.cvt[0, 1]:
            valid_pairs.append(pair)
        if len(valid_pairs) == 20:
            break

    pair_params = calculate_mu_beta_sigma(log_data, valid_pairs)
    results_2[(formation_period.index[0], formation_period.index[-1])] = pair_params
    johansen_test_counts_2[(formation_period.index[0], formation_period.index[-1])] = test_count

    # Formation period's mean
    formation_mean = calculate_mean(formation_period)
    print(f"Formation period {formation_period.index[0].date()} to {formation_period.index[-1].date()} mean values:")
    print(formation_mean)
    print()

# Result
print("Results for log_hp_t:")
for period, pairs in results_2.items():
    print(f"Period {period[0].date()} to {period[1].date()}:")
    for pair, params in pairs.items():
        mu, beta, sigma, mean_spread = params
        print(f"  {pair}: mu={mu:.6f}, beta={beta:.6f}, sigma={sigma:.6f}, mean_spread={mean_spread:.6f}")
    print(f"Johansen cointegration tests run: {johansen_test_counts_2[period]}")
    print()



## Extract the information from previous steps (pairs and parameters).
Info_allPeriods = []
for _value in results_2.values():
    Info_allPeriods.append(_value)


### Detect trading signals through standardized spreads.
Trading_returns = []
positive_counts = 0
for _t in range(len(Trading_periods)):
    returns_pair = 0
    # For each cointegration pair in current trading period.
    for _j in range(20):
        ## Extract each cointegration pair for corresponding trading period.
        stock_A = list(Info_allPeriods[ _t ].keys())[ _j ][0]
        stock_B = list(Info_allPeriods[ _t ].keys())[ _j ][1]
        
        ## Extract the historical parameters.
        params_mu = list(Info_allPeriods[ _t ].values())[ _j ][0]
        params_beta = list(Info_allPeriods[ _t ].values())[ _j ][1]
        params_sigma = list(Info_allPeriods[ _t ].values())[ _j ][2]
        params_mean = list(Info_allPeriods[ _t ].values())[ _j ][3]
        
        ## Start to trade.
        Trading_Record = Func_Trading(_t, [stock_A, stock_B], [params_mu, params_beta, params_sigma, params_mean])
        
        # Record the final gain of the current pairs trading in current period.
        returns_pair += Trading_Record.iloc[-1, 9]
        
        if Trading_Record.iloc[-1, 9] > 0:
            positive_counts += 1
        
    # Record the cumulative wealth for all of 20 pairs trading of current period.
    Trading_returns.append( returns_pair )

MeanReturn.append( sum(Trading_returns) / len(Trading_periods) / 100 )
StdDeviation.append( np.std(Trading_returns, ddof=1) / 10 )
Sharpe.append( sum(Trading_returns) / np.std(Trading_returns, ddof=1) / 10 / len(Trading_periods) )
PercentOfPositiveMonths.append( positive_counts / 20 / len(Trading_periods) )




#%% TWSE hr_t

## Set parameters.
k = 0.06
initial_sigma = 0.017
## Save hr_t
hr_t_df_tw = drop_na_return_tw[['Date']].copy()

### hr_t formula
for col in drop_na_return_tw.columns:
    if col == 'Date':
        continue 

    # Demean
    u_t = drop_na_return_tw[col] - drop_na_return_tw[col].mean()
    sigma_t = np.full(len(drop_na_return_tw), initial_sigma)

    for i in range(1, len(sigma_t)):
        sigma_t[i] = np.sqrt(k * sigma_t[i - 1]**2 + (1 - k) * u_t.iloc[i]**2)

    hr_t_tw = drop_na_return_tw[col] / sigma_t
    hr_t_df_tw[col] = hr_t_tw
hr_t_df_tw.set_index('Date', inplace=True)
hr_t_df_tw = hr_t_df_tw / 100
###


def get_top_pairs(corr_matrix, top_n=1225):
    pairs = []
    sorted_pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
    for i in range(1, top_n + 1):
        pairs.append(sorted_pairs.index[i])
    return pairs


periods_3 = extract_periods(start_year, hr_t_df_tw)
results_3 = {}
johansen_test_counts_3 = {}
for formation_period in periods_3:
    corr_matrix = calculate_correlations(formation_period)
    top_pairs = get_top_pairs(corr_matrix)

    valid_pairs = []
    test_count = 0
    for pair in top_pairs:
        test_result = johansen_coint_test(log_data_tw[list(pair)])
        test_count += 1
        if test_result.lr1[0] > test_result.cvt[0, 1]:
            valid_pairs.append(pair)
        if len(valid_pairs) == 20:
            break

    pair_params = calculate_mu_beta_sigma(log_data_tw, valid_pairs)
    results_3[(formation_period.index[0], formation_period.index[-1])] = pair_params
    johansen_test_counts_3[(formation_period.index[0], formation_period.index[-1])] = test_count

    formation_mean = calculate_mean(formation_period)
    print(f"Formation period {formation_period.index[0].date()} to {formation_period.index[-1].date()} mean values:")
    print(formation_mean)
    print()

print("Results for hr_t_df_tw:")
for period, pairs in results_3.items():
    print(f"Period {period[0].date()} to {period[1].date()}:")
    for pair, params in pairs.items():
        mu, beta, sigma, mean_spread = params
        print(f"  {pair}: mu={mu:.6f}, beta={beta:.6f}, sigma={sigma:.6f}, mean_spread={mean_spread:.6f}")
    print(f"Johansen cointegration tests run: {johansen_test_counts_3[period]}")
    print()


#%% Trading

## Slice the trading period.
Trading_StartDates, Trading_EndDates = [], []
for _key1, _key2 in results_3.keys():
    Trading_StartDates.append(_key1)
    Trading_EndDates.append(_key2)

Trading_StartDates = Trading_StartDates[2:] + [Trading_EndDates[-2] + pd.DateOffset(days=4)] + \
                     [Trading_EndDates[-1] + pd.DateOffset(days=1)] 

Trading_EndDates = Trading_EndDates[1:] + [log_data.index[-1]]


Trading_periods = []
for _t in range(len(Trading_StartDates)):
    Trading_periods.append( log_data_tw.loc[ Trading_StartDates[_t] : Trading_EndDates[_t] ] )



## Extract the information from previous steps (pairs and parameters).
Info_allPeriods = []
for _value in results_3.values():
    Info_allPeriods.append(_value)


### Detect trading signals through standardized spreads.
Trading_returns = []
positive_counts = 0
for _t in range(len(Trading_periods)):
    returns_pair = 0
    # For each cointegration pair in current trading period.
    for _j in range(20):
        ## Extract each cointegration pair for corresponding trading period.
        stock_A = list(Info_allPeriods[ _t ].keys())[ _j ][0]
        stock_B = list(Info_allPeriods[ _t ].keys())[ _j ][1]
        
        ## Extract the historical parameters.
        params_mu = list(Info_allPeriods[ _t ].values())[ _j ][0]
        params_beta = list(Info_allPeriods[ _t ].values())[ _j ][1]
        params_sigma = list(Info_allPeriods[ _t ].values())[ _j ][2]
        params_mean = list(Info_allPeriods[ _t ].values())[ _j ][3]
        
        ## Start to trade.
        Trading_Record = Func_Trading(_t, [stock_A, stock_B], [params_mu, params_beta, params_sigma, params_mean])
        
        # Record the final gain of the current pairs trading in current period.
        returns_pair += Trading_Record.iloc[-1, 9]
        
        if Trading_Record.iloc[-1, 9] > 0:
            positive_counts += 1
        
    # Record the cumulative wealth for all of 20 pairs trading of current period.
    Trading_returns.append( returns_pair )

MeanReturn.append( sum(Trading_returns) / len(Trading_periods) / 100 )
StdDeviation.append( np.std(Trading_returns, ddof=1) / 10 )
Sharpe.append( sum(Trading_returns) / np.std(Trading_returns, ddof=1) / 10 / len(Trading_periods) )
PercentOfPositiveMonths.append( positive_counts / 20 / len(Trading_periods) )




#%% TWSE: Method 2

##
zeros_df_tw = pd.DataFrame(0, index=data_tw_adj.index, columns=data_tw_adj.columns)
zeros_df_tw.iloc[0, :] = data_tw_adj.iloc[0, :]
for i in range(1, 1459):
    zeros_df_tw.iloc[i, :] = zeros_df_tw.iloc[i - 1, :] * (1 + hr_t_df_tw.iloc[i - 1, :])

hp_t_tw_no = zeros_df_tw
hp_t_tw = np.log(hp_t_tw_no)


periods_4 = extract_periods(start_year, hp_t_tw)
results_4 = {}
johansen_test_counts_4 = {}

for formation_period in periods_4:
    corr_matrix = calculate_correlations(formation_period)
    top_pairs = get_top_pairs(corr_matrix)

    valid_pairs = []
    test_count = 0
    for pair in top_pairs:
        test_result = johansen_coint_test(log_data_tw[list(pair)])
        test_count += 1
        if test_result.lr1[0] > test_result.cvt[0, 1]:
            valid_pairs.append(pair)
        if len(valid_pairs) == 20:
            break

    pair_params = calculate_mu_beta_sigma(log_data_tw, valid_pairs)
    results_4[(formation_period.index[0], formation_period.index[-1])] = pair_params
    johansen_test_counts_4[(formation_period.index[0], formation_period.index[-1])] = test_count


    formation_mean = calculate_mean(formation_period)
    print(f"Formation period {formation_period.index[0].date()} to {formation_period.index[-1].date()} mean values:")
    print(formation_mean)
    print()

print("Results for hr_t_df_tw:")
for period, pairs in results_4.items():
    print(f"Period {period[0].date()} to {period[1].date()}:")
    for pair, params in pairs.items():
        mu, beta, sigma, mean_spread = params
        print(f"  {pair}: mu={mu:.6f}, beta={beta:.6f}, sigma={sigma:.6f}, mean_spread={mean_spread:.6f}")
    print(f"Johansen cointegration tests run: {johansen_test_counts_4[period]}")
    print()



## Extract the information from previous steps (pairs and parameters).
Info_allPeriods = []
for _value in results_4.values():
    Info_allPeriods.append(_value)


### Detect trading signals through standardized spreads.
Trading_returns = []
positive_counts = 0
for _t in range(len(Trading_periods)):
    returns_pair = 0
    # For each cointegration pair in current trading period.
    for _j in range(20):
        ## Extract each cointegration pair for corresponding trading period.
        stock_A = list(Info_allPeriods[ _t ].keys())[ _j ][0]
        stock_B = list(Info_allPeriods[ _t ].keys())[ _j ][1]
        
        ## Extract the historical parameters.
        params_mu = list(Info_allPeriods[ _t ].values())[ _j ][0]
        params_beta = list(Info_allPeriods[ _t ].values())[ _j ][1]
        params_sigma = list(Info_allPeriods[ _t ].values())[ _j ][2]
        params_mean = list(Info_allPeriods[ _t ].values())[ _j ][3]
        
        ## Start to trade.
        Trading_Record = Func_Trading(_t, [stock_A, stock_B], [params_mu, params_beta, params_sigma, params_mean])
        
        # Record the final gain of the current pairs trading in current period.
        returns_pair += Trading_Record.iloc[-1, 9]
        
        if Trading_Record.iloc[-1, 9] > 0:
            positive_counts += 1
        
    # Record the cumulative wealth for all of 20 pairs trading of current period.
    Trading_returns.append( returns_pair )

MeanReturn.append( sum(Trading_returns) / len(Trading_periods) / 100 )
StdDeviation.append( np.std(Trading_returns, ddof=1) / 10 )
Sharpe.append( sum(Trading_returns) / np.std(Trading_returns, ddof=1) / 10 / len(Trading_periods) )
PercentOfPositiveMonths.append( positive_counts / 20 / len(Trading_periods) )

Sharpe = [ i / 100 for i in Sharpe ]


#%% Performance Evaluation

Metrics = [r'$\rho^r$ (S&P100)', r'$\rho^p$ (S&P100)', r'$\rho^r$ (TWSE50)', r'$\rho^p$ (TWSE50)']


plt.figure(figsize=(14, 14))

plt.subplot(2, 2, 1)  # 1 row, 3 columns, 1st subplot
plt.bar(Metrics, MeanReturn)
plt.title('Panel A: Mean Returns', fontsize=15)
plt.xlabel('Pre-selection metrics (Stock Market)', fontsize=15)
plt.xticks(fontsize=12)
bars = plt.bar(Metrics, MeanReturn)
for bar, value in zip(bars, MeanReturn):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value*100:.2f}%', 
             ha='center', va='bottom', fontsize=15)

plt.subplot(2, 2, 2)  # 1 row, 3 columns, 1st subplot
plt.bar(Metrics, StdDeviation)
plt.title('Panel B: Standard Deviation', fontsize=15)
plt.xlabel('Pre-selection metrics (Stock Market)', fontsize=15)
plt.xticks(fontsize=12)
bars = plt.bar(Metrics, StdDeviation)
for bar, value in zip(bars, StdDeviation):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value*100:.2f}%', 
             ha='center', va='bottom', fontsize=15)

plt.subplot(2, 2, 3)  # 1 row, 3 columns, 1st subplot
plt.bar(Metrics, Sharpe)
plt.title('Panel C: Sharpe Ratio', fontsize=15)
plt.xlabel('Pre-selection metrics (Stock Market)', fontsize=15)
plt.xticks(fontsize=12)
bars = plt.bar(Metrics, Sharpe)
for bar, value in zip(bars, Sharpe):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value*100:.3f}', 
             ha='center', va='bottom', fontsize=15)


plt.subplot(2, 2, 4)  # 1 row, 3 columns, 1st subplot
plt.bar(Metrics, PercentOfPositiveMonths)
plt.title('Panel D: % Positive Excess Returns', fontsize=15)
plt.xlabel('Pre-selection metrics (Stock Market)', fontsize=15)
plt.xticks(fontsize=12)
bars = plt.bar(Metrics, PercentOfPositiveMonths)
for bar, value in zip(bars, PercentOfPositiveMonths):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value*100:.2f}%', 
             ha='center', va='bottom', fontsize=15)

plt.show()




Pairs_data = pd.DataFrame({'Potential Pairs': [4851, 4851, 1225, 1225],
                           'Tested Pairs': [363, 189, 427, 170]
    }).T

Pairs_data.columns = ['ρ^r (S&P100)', 'ρ^p (S&P100)', 'ρ^r (TWSE 50)', 'ρ^p (TWSE 50)']



#%% Examples

## Slice the trading period.
Trading_StartDates, Trading_EndDates = [], []
for _key1, _key2 in results.keys():
    Trading_StartDates.append(_key1)
    Trading_EndDates.append(_key2)

Trading_StartDates = Trading_StartDates[2:] + [Trading_EndDates[-2] + pd.DateOffset(days=4)] + \
                     [Trading_EndDates[-1] + pd.DateOffset(days=1)] 

Trading_EndDates = Trading_EndDates[1:] + [log_data.index[-1]]


Trading_periods = []
for _t in range(len(Trading_StartDates)):
    Trading_periods.append( log_data.loc[ Trading_StartDates[_t] : Trading_EndDates[_t] ] )



## Extract the information from previous steps (pairs and parameters).
Info_allPeriods = []
for _value in results.values():
    Info_allPeriods.append(_value)


_t = 0
_j = 15
stock_A = list(Info_allPeriods[ _t ].keys())[ _j ][0]
stock_B = list(Info_allPeriods[ _t ].keys())[ _j ][1]

## Extract the historical parameters.
params_mu = list(Info_allPeriods[ _t ].values())[ _j ][0]
params_beta = list(Info_allPeriods[ _t ].values())[ _j ][1]
params_sigma = list(Info_allPeriods[ _t ].values())[ _j ][2]
params_mean = list(Info_allPeriods[ _t ].values())[ _j ][3]

## Start to trade.
Trading_Record = Func_Trading(_t, [stock_A, stock_B], [params_mu, params_beta, params_sigma, params_mean])





#%% Standardized Spread

def calculate_standardized_spread(log_data, pair, mu, beta):
    ## Extract the log stock prices in the pairs trading.
    stock1 = log_data[pair[0]]
    stock2 = log_data[pair[1]]

    ## Compute the spread.
    spread = stock2 - (mu + beta * stock1)

    ## Compute the mean and SD of the spreads.
    mean_spread = np.mean(spread)
    sigma = np.std(spread)

    ## Compute standardized spreads.
    standardized_spread = (spread - mean_spread) / sigma

    return standardized_spread

def plot_standardized_spread(standardized_spread, pair):
    plt.figure(figsize=(10, 6))
    plt.plot(standardized_spread, label=f'Standardized Spread={pair[0]}, {pair[1]}')
    plt.axhline(y=2, color='b', linestyle='-')
    plt.axhline(y=-2, color='b', linestyle='-')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Standardized Spread & Boundaries')
    plt.xlabel('Time')
    plt.ylabel('Standardized Spread')
    plt.legend(loc='upper right')
    plt.show()

for period, pairs in results.items():
    print(f"Period {period[0].date()} to {period[1].date()}:")
    for pair, params in pairs.items():
        mu, beta, sigma, mean = params
        print(f"  {pair}: mu={mu:.6f}, beta={beta:.6f}, sigma={sigma:.6f}")
        standardized_spread = calculate_standardized_spread(log_data, pair, mu, beta)
        plot_standardized_spread(standardized_spread, pair)


#%% Essay Raw Results

## Create data.
data_before = {
    'Pre-selection metrics': ['SSD', 'P̂R', 'ρ̂p', 'ρ̂r', 'COVp', 'COVr', 'ĈΔp(0)'],
    'Mean': [0.0033, 0.0070, 0.0073, 0.0032, 0.0076, 0.0056, 0.0019],
    'Std': [0.0235, 0.0284, 0.0369, 0.0266, 0.1082, 0.0371, 0.0319],
    'Sharpe Ratio': [0.14, 0.25, 0.20, 0.11, 0.07, 0.15, 0.06],
    '% Positive Returns': [51.91, 54.47, 54.47, 56.17, 46.81, 50.64, 48.94],
}

data_after = {
    'Pre-selection metrics': ['SSD', 'P̂R', 'ρ̂p', 'ρ̂r', 'COVp', 'COVr', 'ĈΔp(0)'],
    'Mean': [0.0003, 0.0055, 0.0108, 0.0067, 0.0031, 0.0031, 0.0003],
    'Std': [0.0225, 0.0323, 0.0438, 0.0250, 0.0952, 0.0331, 0.0307],
    'Sharpe Ratio': [0.01, 0.17, 0.25, 0.10, 0.05, 0.10, 0.06],
    '% Positive Returns': [46.81, 52.34, 51.56, 49.36, 46.36, 48.94, 43.47],
}


df_before = pd.DataFrame(data_before)
df_after = pd.DataFrame(data_after)


sns.set(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Before Commissions and Cut Rules', fontsize=16)

## Plot mean.
sns.barplot(x='Pre-selection metrics', y='Mean', data=df_before, ax=axes[0, 0])
axes[0, 0].set_title('Mean')

## Plot Standard Deviation
sns.barplot(x='Pre-selection metrics', y='Std', data=df_before, ax=axes[0, 1])
axes[0, 1].set_title('Standard Deviation')

## Plot Sharpe Ratio.
sns.barplot(x='Pre-selection metrics', y='Sharpe Ratio', data=df_before, ax=axes[1, 0])
axes[1, 0].set_title('Sharpe Ratio')

## Plot % of Positive Returns
sns.barplot(x='Pre-selection metrics', y='% Positive Returns', data=df_before, ax=axes[1, 1])
axes[1, 1].set_title('% Positive Excess Returns')
axes[1, 1].set_ylim(40 , 60)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Figures Including Commissions and Cut Rules
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Including Commissions and Cut Rules', fontsize=16)

# Mean
sns.barplot(x='Pre-selection metrics', y='Mean', data=df_after, ax=axes[0, 0])
axes[0, 0].set_title('Mean')

# Standard Deviation
sns.barplot(x='Pre-selection metrics', y='Std', data=df_after, ax=axes[0, 1])
axes[0, 1].set_title('Standard Deviation')

# Sharpe Ratio
sns.barplot(x='Pre-selection metrics', y='Sharpe Ratio', data=df_after, ax=axes[1, 0])
axes[1, 0].set_title('Sharpe Ratio')

# % Positive Returns
sns.barplot(x='Pre-selection metrics', y='% Positive Returns', data=df_after, ax=axes[1, 1])
axes[1, 1].set_title('% Positive Excess Returns')
axes[1, 1].set_ylim(40, 60)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


