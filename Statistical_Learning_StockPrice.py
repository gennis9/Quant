# Copyright (C) 2023 Gennis
'''
Initiated Date    : 2023/11/07
Last Updated Date : 2023/12/19
Aim: (1) Find the hyperparameters through learning algorithm;
     (2) Find the best investment strategy.
Input Data: Component stock prices of FTSE TWSE Taiwan 50 Index.
'''


#%% Enviornment

from bs4 import BeautifulSoup
from copy import deepcopy
# import ffn
# import matplotlib as mpl
import cvxopt
import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import requests
import scipy
import scipy.interpolate as sci
import scipy.optimize as sco
import scipy.stats.mstats as scs
import sklearn as sk
import yfinance as yf
# from functools import reduce
%matplotlib inline

import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import random,requests,re
import matplotlib.pyplot as plt
import time,math
import yfinance as yf
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
from tqdm import tqdm

from scipy.optimize import minimize
from numpy import linalg as LA

import seaborn as sns
from scipy.sparse.csgraph import minimum_spanning_tree
from datetime import datetime 

from sklearn.covariance import ShrunkCovariance
from sklearn.datasets import make_gaussian_quantiles

from pyvis.network import Network
import networkx as nx
import planarity

from pypfopt import efficient_frontier,expected_returns,objective_functions
import cvxpy as cp
from collections import defaultdict
import datetime

import functools
import copy
import statistics


## Parameters
# Length of total period
start_date = '2010-01-01'
end_date   = '2023-11-30'

# Risk free rate
rf = 0.01

# Size of testing set.
test_size = 0.4

# The initial ratio between the length of rolling window and the length of holding period.
# Adopt the ratio presented with the daily data in Peralta & Zareei (2016).
# kappa = 1000 / 20
# kappa = 192 / 12
kappa = 3

# The interval multiplier of the rolling window.
# Day
interval_rw = 25

# The interval multiplier of the holding period.
interval_hp = 20

# Set the random seed.
random_state = 2023

# Set the threshold to kill missing values.
miss_thresh = 0

# Set the closed-form formula for computaiton of optimal weights of the tangency portfolio.
method = 3



#%% Data Access

## Download the information regarding the FTSE TWSE Taiwan 50 Index from Wiki.
wikiURL = 'https://zh.wikipedia.org/zh-tw/%E8%87%BA%E7%81%A350%E6%8C%87%E6%95%B8'


session = requests.Session()
page = session.get(wikiURL)
c = page.content

soup = BeautifulSoup(c, 'html.parser')

mydivs = soup.findAll('table',{'class':'wikitable'})

## Extract the list of the stock components included in the table from the information crawled.
WikiTable = pd.read_html(str(mydivs))
# Convert list to dataframe
df = pd.DataFrame(WikiTable[0])




#%% Data Cleaning

df_new = df[['股票代號.1', '名稱.1']]
df_new.columns = ['股票代號', '名稱']
df = df.drop(columns=['股票代號.1', '名稱.1'])
df = df.append(df_new, ignore_index=True)
df['股票代號'] = df['股票代號'].str.replace('臺證所：', '')
df['股票代號'] = df['股票代號']+'.TW'

## Inconsistent stock data.
df = df[ df['股票代號'] != '4938.TW' ]

stockList = " ".join(df["股票代號"].values)

## Add the component stocks that existed in the index previously.
# 
histStock = ''

stockListFull = stockList + histStock

Stock = yf.download(stockListFull, start=start_date, end=end_date)

## Extract the adjusted closed prices.
Stock_close = deepcopy( Stock['Adj Close'] )

## Extract the adjusted closed prices.
Stock_open = deepcopy( Stock['Open'] )

## Remove the stocks whose price data is insufficient.
Stock_close = Stock_close.dropna(axis='columns', thresh=int( len(Stock_close)*(1-miss_thresh) ))
Stock_open  = Stock_open.dropna(axis='columns', thresh=int( len(Stock_open)*(1-miss_thresh) ))
Stock       = Stock.dropna(axis='columns', thresh=int( len(Stock_open)*(1-miss_thresh) ))

# Stock_close.isnull().sum().sum()
## Fill the remaining missing values by interpolation.
# Stock_close = Stock_close.interpolate()
# Stock_open  = Stock_open.interpolate()

## Retrieve the Index.
TW50 = yf.download('0050.tw', start=start_date, end=end_date)
TW50_close = deepcopy(TW50)['Adj Close']
TW50_open = deepcopy(TW50)['Open']


#%%% Export
''' 
Path = r'D:\03Programs_Clouds\Google Drive\NSYSU\01Stat Learning\Final Report'
os.chdir(Path)

Stock_close.to_csv('Close_Stocks_TW.csv', header=True, index=True)
Stock_open.to_csv('Open_Stocks_TW.csv', header=True, index=True)
TW50_close.to_csv('Close_TW50.csv', header=True, index=True)
TW50_open.to_csv('Open_TW50.csv', header=True, index=True)
'''




#%% Data Preprocessing

## Build initial settings.
Returns = Stock_close.pct_change()[1:]
# Returns = Returns.loc[:, (Returns == 0).sum() <= 200]
N = len(Returns.columns)
ones = np.ones(N)
zeros = np.zeros(N)
# Returns.isnull().sum().sum()
# For balanced panel.
Stock_close = Stock_close[1:]
Stock_open  = Stock_open[1:]
Stock       = Stock[1:]

Stock_close.index = pd.to_datetime(Stock_close.index)
Stock_open.index = pd.to_datetime(Stock_open.index)

Dates = Stock.index


## Split data for cross-validation.
# Test set.
sizeOfTest  = int(len(Stock_close) * test_size)
Test_close  = Stock_close.iloc[ -sizeOfTest: ]
Test_open   = Stock_open.iloc[ -sizeOfTest: ]
Test_return = Returns.loc[ Test_close.index[0]: ]
Test_TW50   = TW50_close.loc[ Test_close.index[0]: ]

# Training set.
Train_close  = Stock_close.iloc[ :-sizeOfTest ]
Train_open   = Stock_open.iloc[ :-sizeOfTest ]
Train_return = Returns.loc[ :Train_close.index[-1] ]
Train_TW50   = TW50_close.loc[ :Train_close.index[-1] ]



#%% Tangency Portfolio Construction

# Sharpe ratio with portfolio
def portfolio_SR_weight(tmp_data, portfolio_weight, rf = 0):
    mus = (tmp_data+1).prod()**(1/(len(tmp_data))) - 1
    portfolio_return = np.dot(tmp_data, portfolio_weight)
    Portfolio_Variance = statistics.variance(portfolio_return)
    Portfolio_Return = np.dot(mus, portfolio_weight)

    SR = (Portfolio_Return-rf) / np.sqrt(Portfolio_Variance)
    return SR


# calculate portfolio return (periord)
def cal_portfolio_return(tmp_data, portfolio_weight):
    period_return = np.dot(tmp_data, portfolio_weight)
    return period_return


def calculate_tangency_weights(tmp_data, rf = 0, bound=(0,1)):
    mus = ((tmp_data+1).prod()) - 1
    shrink_cov = tmp_data.cov()
    ttt = efficient_frontier.EfficientFrontier(expected_returns = mus, cov_matrix = shrink_cov,
                                               verbose = False, weight_bounds = bound)

#     weight = list(ttt.max_sharpe(risk_free_rate = rf).items())
    weight = list(ttt.nonconvex_objective(
        objective_functions.sharpe_ratio,
        objective_args=(ttt.expected_returns, ttt.cov_matrix),
        weights_sum_to_one=True).items())
    
    tangency_weights = pd.DataFrame(weight,columns=['a','b'])['b']
    return tangency_weights


def corr_dist(data_cor):
    r,c = data_cor.shape
    cor_dist = copy.deepcopy(data_cor)
    for i in range(r):
        for j in range(c):
            cor_dist.iloc[i,j] = np.sqrt(((data_cor.iloc[i,:] - data_cor.iloc[j,:])**2).sum())
    
    return cor_dist

def MST_adjacency(data_shrinkage_cor):
    corr_dist_data = corr_dist(data_shrinkage_cor)
    MST = minimum_spanning_tree(corr_dist_data)
    new_corr = pd.DataFrame(MST.toarray(), columns = data_shrinkage_cor.columns, index = data_shrinkage_cor.columns)
    adjacenay_matrix = pd.DataFrame(new_corr != 0).astype(int)
    finall_adjacenay_matrix = adjacenay_matrix + adjacenay_matrix.T
    
    return MST, finall_adjacenay_matrix


# covariance to correlation

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


# sharpe ratio with stock
def stocks_SR(tmp_M_data, rf = 0):
    # M_mus = ((tmp_M_data+1).prod()) - 1
    M_mus = (tmp_M_data+1).prod()**(1/(len(tmp_M_data))) - 1
    # use covariance
    shrink_cov = tmp_M_data.cov()
    SR = (M_mus-rf)/(np.diag(shrink_cov)**0.5)
    
    return SR




#%% Parameter Construction
''' The aim of this subsection is setting the parameters that are,
at least partially, decided by other parameters rather than absoultely manually. '''

## Set for the minimum length of time for a training observation.
# Holding period + rolling window.
kappa = int(kappa)

## Set the number of times for slicing windows + bootstrapping.
numOfBoots = int( len(Train_close)//(1+kappa) * 2 )

## To preserve sufficient large sample space, 
# the maximum length of training period (i.e., rolling window + holding period) is limited.
SampleSize = int( len( Stock_close ) * 0.1 )
# T - (M+H) + 1 > SampleSize.
maxTrainL = int( len(Train_close) - SampleSize + 1)

maxRolling = int( maxTrainL//(1+kappa) * kappa )
maxHolding = int( maxRolling / kappa )

## Debug setting.
# numOfBoots = 30      # For debug.
# method = 3




#%% Search of hyperparameter - Greedy for Length of Rolling Window

## To shorten the searching time, use a list of candidates instead of interval searching.
M_list = [ 75 * (multiplier+1) for multiplier in range( int(maxRolling / 75) ) ]
M_list = [25, 50] + M_list


def Func_Search(initial_date, method=method):
    ## Compute the optimal weights under tangency portfolio using the return data of rolling window.
    rolling = Returns.iloc[initial_date : initial_date + width_rolling]

    weight = Func_OptWeight(rolling, method=method)
    
    # Remove the weights that are too small.
    weight[ np.abs(weight) < 0.001 ] = 0
    # Keep the sum of the weight as 100%.
    weight = weight / weight.sum()
    
    ## Compute the number of shares bought at the beginning of the holding period.
    shares = 1 * weight / Train_open.iloc[initial_date + width_rolling]
    ## Compute the share holding at the end of the holding period.
    holding = shares * Train_close.iloc[initial_date + width_train - 1]
    
    ## Compute the Sharpe ratio for the holding period.
    var = Returns.iloc[initial_date + width_rolling : initial_date + width_train].cov()
    var = weight.T @ var @ weight
    
    r_single = holding.sum() - 1

    return r_single, var


TrainingSets = []
Sharpes_1 = []
PositiveCounts = []
VarCounts = []
Ms = []

## Search the best length of rolling window (M) by loop.
for _i in M_list:
    width_rolling = _i
    width_holding = int( width_rolling / kappa )
    width_train = width_holding + width_rolling
    _periods = []
    _sharpes = []
    _positiveSharpe = 0

    ## 1. Search M through the method of slicing window first.
    _startIdx = 0
    _endIdx = width_train
    _positiveSharpe = 0
    for _j in range( len(Train_close)//width_train ):
        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)

        _startIdx += width_train

    ## Utilize the remaining training data.
    if len(Train_close)//width_train != 0:
        _startIdx = len(Train_close) - width_train

        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)


    ## 2. Bootstrap the training periods for remaining number of loops.
    boots = numOfBoots - len(_sharpes)

    random.seed(random_state + _i)
    idx = [ random.randint(0, len(Train_close) - width_train) for _ in range(boots) ]

    for _startIdx in idx:
        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)

    ## Winsorize the Sharpes to prevent the dominance of the outliers.
    _sharpes = scs.winsorize( np.array(_sharpes), limits=[0.05, 0.05] )

    TrainingSets.append(_periods)
    Sharpes_1.append( np.exp(np.mean(_sharpes) * 252/width_holding) - 1 )
    Ms.append(width_rolling)
    PositiveCounts.append(_positiveSharpe)

    # print(_i, '/', maxRolling+1)

M_temp = Ms[ np.argmax(Sharpes_1) ]
# M_temp = 675




#%% Search of hyperparameter - Length of Holding Period

## Update the maximum length of the holding period to be tested.
maxHolding = maxTrainL - M_temp
# interval_hp = 1

TrainingSets = []
Sharpes_2 = []
PositiveCounts = []
Hs = []

## Search the best length of holding period (H) by loop.
for _i in range( interval_hp, maxHolding+1, interval_hp ):
    width_holding = _i
    width_rolling = M_temp
    width_train = width_holding + width_rolling
    _periods = []
    _sharpes = []
    _positiveSharpe = 0

    ## 1. Search H through the method of slicing window first.
    _startIdx = 0
    _endIdx = width_train
    _positiveSharpe = 0
    for _j in range( len(Train_close)//width_train ):
        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)

        _startIdx += width_train

    ## Utilize the remaining training data.
    if len(Train_close)//width_train != 0:
        _startIdx = len(Train_close) - width_train

        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)


    ## 2. Bootstrap the training periods for remaining number of loops.
    boots = numOfBoots - len(_sharpes)

    random.seed(random_state + _i)
    idx = [ random.randint(0, len(Train_close) - width_train) for _ in range(boots) ]

    for _startIdx in idx:
        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)

    ## Winsorize the Sharpes to prevent the dominance of the outliers.
    _sharpes = scs.winsorize( np.array(_sharpes), limits=[0.05, 0.05] )

    TrainingSets.append(_periods)
    Sharpes_2.append( np.exp(np.mean(_sharpes) * 252/width_holding) - 1 )
    Hs.append(width_holding)
    PositiveCounts.append(_positiveSharpe)

    # print(_i, '/', maxHolding+1)


H_target = Hs[ np.argmax(Sharpes_2) ]
# H_target = 20




#%% Search of hyperparameter - Length of Rolling Window

## Update the maximum length of the holding period to be tested.
# Not exceed the maximum length for preservation of sufficient large sample space.
if maxTrainL - H_target < maxRolling:
    maxRolling = maxTrainL - H_target

TrainingSets = []
Sharpes_3 = []
PositiveCounts = []
Ms = []

## Search the best length of rolling window (M) by loop.
for _i in range( interval_rw, maxRolling+1, interval_rw ):
    width_rolling = _i
    width_holding = H_target
    width_train = width_holding + width_rolling
    _periods = []
    _sharpes = []
    _positiveSharpe = 0

    ## 1. Search M through the method of slicing window first.
    _startIdx = 0
    _endIdx = width_train
    _positiveSharpe = 0
    for _j in range( len(Train_close)//width_train ):
        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)

        _startIdx += width_train

    ## Utilize the remaining training data.
    if len(Train_close)//width_train != 0:
        _startIdx = len(Train_close) - width_train

        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)


    ## 2. Bootstrap the training periods for remaining number of loops.
    boots = numOfBoots - len(_sharpes)

    random.seed(random_state + _i)
    idx = [ random.randint(0, len(Train_close) - width_train) for _ in range(boots) ]

    for _startIdx in idx:
        ## Save the period for later bootstrapping.
        _periods.append( Returns.iloc[_startIdx : _startIdx + width_train] )

        ## Compute the position at the end of the holding period
        # and the corresponding variance of the portfolio constructed.
        # The start index gives the period of rolling window and holding period.
        tmp_holding, tmp_var = Func_Search(_startIdx)

        # if tmp_var < 0.01:
        tmp_sharpe = (tmp_holding.sum() - 1 - rf) / tmp_var
        _sharpes.append(tmp_sharpe)

        ## Record the occurrence of the positive Sharpe ratios.
        if tmp_holding.sum() - 1 - rf > 0:
            _positiveSharpe += 1

        # print(tmp_sharpe)

    ## Winsorize the Sharpes to prevent the dominance of the outliers.
    _sharpes = scs.winsorize( np.array(_sharpes), limits=[0.05, 0.05] )

    TrainingSets.append(_periods)
    Sharpes_3.append( np.exp(np.mean(_sharpes) * 252/width_holding) - 1 )
    Ms.append(width_rolling)
    PositiveCounts.append(_positiveSharpe)

    # print(_i, '/', maxRolling+1)

M_target = Ms[ np.argmax(Sharpes_3) ]
# M_target = 1650
# M_target = 1050




#%% Parameter Recall

H_target = 25
M_target = 165

# Number of stocks selected for constructing portfolio.
NumStock = 20

# The length of time to roll window.
SpeedOfRolling = H_target
# SpeedOfRolling = 20

# Randomly pick n holding periods for comparison the strategies.
NumOfTestSample = 8



#%% In-Sample and Out-of-Sample Performance

## Calculate the rolling windows for computing returns over test set.
# 1. Only holding periods are in test set, 
# it is allowed the rolling windows to get info from the training set.
# Rolling_test = []
# for _i in range(len(Train_close)-M_target, len(Stock_close)-(M_target+H_target), SpeedOfRolling):
#     windows = Returns.iloc[ _i : _i + M_target ]
#     Rolling_test.append(windows)
    
# 2. Both rolling windows and holding periods and are in test set.
Rolling_test = []
for _i in range(0, len(Test_close)-(M_target+H_target), SpeedOfRolling):
    windows = Test_return.iloc[ _i : _i + M_target ]
    Rolling_test.append(windows)

# 3. Randomly pick n holding periods for comparison the strategies.
random.seed(random_state)
Rolling_test = [ Rolling_test[i] for i in random.sample(range(0, len(Rolling_test)), NumOfTestSample) ]

    
## Calculate the rolling windows for computing returns over full dataset.
Rolling_full = []
for _i in range(0, len(Stock_close)-(M_target+H_target), SpeedOfRolling):
    windows = Returns.iloc[ _i : _i + M_target ]
    Rolling_full.append(windows)

## The first day to invest in test set.
firstInvDate = Dates[ Dates.get_loc(Rolling_test[0].iloc[-1].name) + 1 ]




#%% New

## Calculate the exposures at the end of each holding period.
Exposures_naive = pd.DataFrame()
Holding_naive = pd.DataFrame()
Vars_naive = []
Sharpes_naive = []
_col = []
for _period in range(len(Rolling_test)):
    # Final date of the rolling window.
    finalRolling = Dates.get_loc(Rolling_test[_period].iloc[-1].name)
    # First date of the holding period.
    firstHolding = Dates[ finalRolling + 1 ]
    # Final date of the holding period.
    finalHolding = Dates[ finalRolling + H_target ]
    _col.append(finalHolding)


    # 計算 covariance 並藉由 covariance matrix 轉換 correlation matrix
    shrink_cov = train_data.cov()
    # MST 對絞線沒有扣除（應該？）
    shrink_corr = correlation_from_covariance(shrink_cov)
    
    # MST
    MST, adjacency_data = MST_adjacency(shrink_corr)
    
    # centrality
    w,v = np.linalg.eigh(adjacency_data,  UPLO='U')
    centrality = abs(v[:,w.argmax()])
    
    # val_data 
    val_SSR = stocks_SR(val_data)
    
    # calculate rho
    rho = np.corrcoef(val_SSR, centrality)[0,1]
    
    # calculate HS rho and LS rho
    sort_centrality_data = pd.DataFrame(centrality, index = adjacency_data.columns, columns = ['centrality'])
    sort_centrality_data = sort_centrality_data.sort_values(by=['centrality'], ascending=False)
    
    HS_list = sort_centrality_data.index[:PARAMETERS['n_stock']]
    HS_centrality = list(sort_centrality_data[:PARAMETERS['n_stock']]['centrality'])
    LS_list = sort_centrality_data.index[-PARAMETERS['n_stock']:]
    LS_centrality = list(sort_centrality_data[-PARAMETERS['n_stock']:]['centrality'])
    
    HS_train_data = train_data[HS_list]
    LS_train_data = train_data[LS_list]
    
    HS_val_data = val_data[HS_list]
    LS_val_data = val_data[LS_list]
    
    HS_mus = (HS_val_data+1).prod()**(1/(len(HS_val_data))) - 1
    LS_mus = (LS_train_data+1).prod()**(1/(len(LS_train_data))) - 1
    
    HS_SP = stocks_SR(HS_val_data)
    LS_SP = stocks_SR(LS_val_data)
    
    rho_HS = np.corrcoef(HS_SP, HS_centrality)[0,1]
    rho_LS = np.corrcoef(LS_SP, LS_centrality)[0,1]
    















#%% Naïve Strategy

## The weight of naive strategy.
Weight_naive = np.array([1/N]*N)


## Calculate the exposures at the end of each holding period.
Exposures_naive = pd.DataFrame()
Holding_naive = pd.DataFrame()
Vars_naive = []
Sharpes_naive = []
_col = []
for _period in range(len(Rolling_test)):
    # Final date of the rolling window.
    finalRolling = Dates.get_loc(Rolling_test[_period].iloc[-1].name)
    # First date of the holding period.
    firstHolding = Dates[ finalRolling + 1 ]
    # Final date of the holding period.
    finalHolding = Dates[ finalRolling + H_target ]
    _col.append(finalHolding)
    
    ## Compute the non-cumulated number of shares of stock of naive strategy at the beginning of each holding period.
    Shares_t = 1/N / Stock_open.loc[ firstHolding ]
    
    ## Compute the holding period exposures if the initial inv is $1 at the begining.
    Holding_naive_t = Shares_t * Stock_close.loc[ finalHolding ]
    Holding_naive = pd.concat([ Holding_naive, Holding_naive_t ], axis = 1)
    
    ## Calculate the exposures at the end of each holding period.
    # Scale up to the change of wealth accumulated across each ending of holding period.
    if _period > 0:
        Exposure_t = Holding_naive_t * Exposures_naive.iloc[:, _period-1].sum()
    else:
        Exposure_t = Holding_naive_t
    
    Exposures_naive = pd.concat([ Exposures_naive, Exposure_t ], axis = 1)
    
    ## Calculate the portfolio Sharpe ratios and portfolio variances across holding periods.
    _var = Weight_naive.T @ Returns.loc[firstHolding: finalHolding].cov() @ Weight_naive
    Vars_naive.append( _var )
    
    _sharpe = (Holding_naive.iloc[:, _period].sum() - 1 - rf) / _var**0.5
    Sharpes_naive.append( _sharpe )
    
Exposures_naive.columns = _col
Holding_naive.columns   = _col
Vars_naive    = pd.DataFrame(Vars_naive)
Sharpes_naive = pd.DataFrame(Sharpes_naive)


## Calculate the weights at the end of each holding period.
Weight_end = pd.DataFrame()
for _period in range(len(Exposures_naive.columns)):
    Weight_end = pd.concat([ Weight_end, Exposures_naive.iloc[:, _period] / Exposures_naive.iloc[:, _period].sum() ], axis = 1)

## Calculate the turnover.
Weights_diff = Weight_end - 1/N
Turnover_naive = Weights_diff.abs().sum().sum() / (len(Rolling_test))

Sharpes_naive.mean()
Sharpes_naive.std()
Vars_naive.mean()
Vars_naive.std()


# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(Rolling_test)+1), (Holding_naive.sum()-1) * 100, color='blue')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The $n^{th}$ Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Naïve Strategy',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
# plt.legend(fontsize=20)
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
        obj_func, zeros, method='SLSQP', constraints=constraints, options=options)
    w_MinV = pd.DataFrame( result.x )
    
    # Remove small numerical calculated weights.
    w_MinV[ w_MinV < 10 ** -4 ] = 0
    # w_MinV = w_MinV / w_MinV.sum()

    return w_MinV

## Alternative to solve the optimal weights of minimum variance portfolio.
def Func_OptWeight_MinV2(windows):
    # Covariance matrix
    cov_matrix = windows.cov().to_numpy()
    # Number of assets
    n = cov_matrix.shape[0]
    # Variable to optimize
    # Variable to optimize
    weights = cp.Variable(n, nonneg=True)
    # Set the initial guess to equal weights (equal probability)
    weights.value = zeros
    # Objective function: minimum portfolio variance
    objective = cp.Minimize(cp.quad_form(weights, cov_matrix))
    # Constraints
    constraints = [cp.sum(weights) == 1, weights >= 0]
    # Define the problem
    problem = cp.Problem(objective, constraints)
    # Solve the problem
    problem.solve()
    # Extract optimized weights
    w_MinV = pd.DataFrame(weights.value, index=windows.columns)
    # Remove small numerical calculated weights
    w_MinV[w_MinV < 1e-4] = 0
    
    ## Make sure that 100% wealth are actually invested.
    w_MinV = w_MinV / w_MinV.sum()
    
    return w_MinV


## Calculate the optimal weight using rolling windows.
Weight_MinV = pd.DataFrame()
_col = []
for _period in Rolling_test: 
    weight_t = Func_OptWeight_MinV2( _period )
    ## Scale up to make sure the sum of actual weights is 100%.
    weight_t = weight_t / weight_t.sum()
    Weight_MinV = pd.concat([ Weight_MinV, weight_t ], axis=1)
    _col.append( _period.index[-1] )

Weight_MinV = Weight_MinV.T
Weight_MinV.index = _col
Weight_MinV.columns = Stock_close.columns


## Construct dynamics of the component stocks in the portfolio across rolling windows.
def Func_Holding(weight):
    ## Calculate the exposures at the end of each holding period.
    Exposures = pd.DataFrame()
    Holding = pd.DataFrame()
    col = []
    for period in range( len(Rolling_test) ):
        ## Get the critical dates.
        # Final date of the rolling window.
        finalRolling = Dates.get_loc(weight.iloc[ period ].name)
        # First date of the holding period.
        firstHolding = Dates[ finalRolling + 1 ]
        # Final date of the holding period.
        finalHolding = Dates[ finalRolling + H_target ]
        col.append(finalHolding)

        ## Compute the non-cumulated number of shares of stock of naive strategy at the beginning of each holding period.
        Shares_t = 1 * weight.loc[ weight.iloc[ period ].name ] / Stock_open.loc[ firstHolding ]
        
        ## Compute the holding period exposures if the initial inv is $1 at the begining.
        holding_t = Shares_t * Stock_close.loc[ finalHolding ]
        Holding = pd.concat([ Holding, holding_t ], axis = 1)

        ## Calculate the exposures at the end of each holding period.
        # Scale up to the change of wealth accumulated across each ending of holding period.
        if period > 0:
            exposure_t = holding_t * Exposures.iloc[:, period-1].sum()
        else:
            exposure_t = holding_t
        
        Exposures = pd.concat([ Exposures, exposure_t ], axis = 1)

    Exposures.columns = col
    Holding.columns = col

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
    turnover = w_diff.abs().sum().sum() / len(Rolling_test)

    ## Calculate the final return.
    final_r = exposure.iloc[:, -1].sum()

    ## Calculate the portfolio Sharpe ratio and portfolio variance.
    Var = pd.DataFrame()
    Sharpe = pd.DataFrame()
    CumR = pd.DataFrame()
    for period in range( len(Rolling_test) ):
        holdingPeriod = \
            Returns.loc[ Dates[ Dates.get_loc(exposure.columns[period]) - H_target + 1 ]: exposure.columns[period] ]
        
        _var = w_begin.iloc[period].T @ holdingPeriod.cov() @ w_begin.iloc[period]
        Var = pd.concat([ Var, pd.DataFrame([_var]) ], axis=1)
        
        _sharpe = (exposure.iloc[:, period].sum() - 1 - rf) / _var**0.5
        Sharpe = pd.concat([ Sharpe, pd.DataFrame([_sharpe]) ], axis=1)
        
        ## Compute the cummulative return of the stock.
        # _cumR_t = (Stock_close.loc[ exposure.columns[period] ] - Stock_open.loc[ firstInvDate ] ) / Stock_open.loc[ firstInvDate ]
        # CumR = pd.concat([CumR, _cumR_t], axis=1)

    Var = Var.T
    Var.index = exposure.columns
    Sharpe = Sharpe.T
    Sharpe.index = exposure.columns
    # CumR.columns = exposure.columns

    return turnover, Sharpe, Var, final_r #, CumR


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover_MinV, Sharpe_MinV, Var_MinV, _ = Func_Performance(Weight_MinV, Weight_MinV_end, Exposures_MinV)

Sharpe_MinV.mean()
Sharpe_MinV.std()

Var_MinV.mean()
Var_MinV.std()


## Calculate the portfolio cumulative return.
# Cum_R_MinV = Weight_MinV @ Cum_R_MinV
# Cum_R_MinV = np.diag(Cum_R_MinV.values)


# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(Rolling_test)+1), (Holding_MinV.sum()-1) * 100, color='blue')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The $n^{th}$ Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Constrained Minimum Variance Strategy',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
# plt.legend(fontsize=20)
plt.show()




#%% Tangency Portfolio

## Calculate the optimal weight using rolling windows.
Weight_Tan = pd.DataFrame()
_col = []
for _period in Rolling_test: 
    weight_t = Func_OptWeight( _period, method=2 )
    ## Scale up to make sure the sum of actual weights is 100%.
    weight_t = weight_t / weight_t.sum()
    Weight_Tan = pd.concat([ Weight_Tan, pd.DataFrame(weight_t) ], axis=1)
    _col.append( _period.index[-1] )

Weight_Tan = Weight_Tan.T
Weight_Tan.index = _col
Weight_Tan.columns = Stock_close.columns

Exposures_Tan, Holding_Tan, Weight_Tan_end = Func_Holding(Weight_Tan)

## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover_Tan, Sharpe_Tan, Var_Tan, _ = Func_Performance(Weight_Tan, Weight_Tan_end, Exposures_Tan)

Sharpe_Tan.mean()
Sharpe_Tan.std()

Var_Tan.mean()
Var_Tan.std()

# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(Rolling_test)+1), (Holding_Tan.sum()-1) * 100, color='blue')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The $n^{th}$ Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Tangency Strategy',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
# plt.legend(fontsize=20)
plt.show()




#%% Highest Sharpe Strategy

## The number of stocks picked.
NumStock = int(N/10)

## Compute the Sharpe ratios of each stock over rolling windows.
Sharpe_roll = pd.DataFrame()
_col = []
for _period in Rolling_test:
    ## Get the critical dates.
    # Final date of the rolling window.
    finalRolling = _period.iloc[-1].name
    # First date of the rolling window.
    firstRolling = _period.iloc[0].name
    _col.append( finalRolling )
    
    Sharpe_t = (Stock_close.loc[ finalRolling ] - \
                    Stock_open.loc[ firstRolling ]) / \
                Stock_open.loc[ firstRolling ]
    var_t = Returns.loc[firstRolling : finalRolling].var()
    Sharpe_t = Sharpe_t / var_t
                
    Sharpe_roll = pd.concat([ Sharpe_roll, Sharpe_t ], axis=1) 
Sharpe_roll.columns = _col


## Construct the weights of the Highest Sharpe Strategy that acts as a benchmark.
Weight_Sharpe = pd.DataFrame(0, index=Returns.columns, columns=range(len(Rolling_test)))
for _period in range(len(Rolling_test)):
    ## Find the index of the stocks with the highest Sharpe ratio among each holding periods.
    idx = Sharpe_roll.iloc[:, _period].nlargest(NumStock).index
    
    ## Naively invest stocks above for each holding period.
    Weight_Sharpe.loc[idx, _period] = 1 / NumStock
    
    
Weight_Sharpe.columns = _col
Weight_Sharpe = Weight_Sharpe.T


## Compute the changes in the holding within the holding period over time.
Exposures_Sharpe, Holding_Sharpe, Weight_Sharpe_end = Func_Holding(Weight_Sharpe)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover_Sharpe, Sharpe_HighestSharpe, Var_HighestSharpe, _ = \
    Func_Performance(Weight_Sharpe, Weight_Sharpe_end, Exposures_Sharpe)

Sharpe_HighestSharpe.mean()
Sharpe_HighestSharpe.std()

Var_HighestSharpe.mean()
Var_HighestSharpe.std()


# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(Rolling_test)+1), (Holding_Sharpe.sum()-1) * 100, color='blue')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The $n^{th}$ Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Highest Sharpe Strategy',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
# plt.legend(fontsize=20)
plt.show()




#%% ρ-Dependent Strategy

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


## Compute the ρ between Sharpe ratio and centrality through Spearman correlation.
Sharpe_TW = (Stock_close.iloc[-1] - Stock_open.iloc[0]) / Stock_open.iloc[0] - rf
var_TW = Returns.var()
Sharpe_TW = Sharpe_TW / var_TW

_, _, Centrality_r, _ = Func_AdjM(Returns, var='return', corr='Pearson')
Rho_tilde, _ = scipy.stats.spearmanr(Centrality_r, Sharpe_TW)


## Require to specify which type of strategy is adopted.
# strategy = 'high'    : Highest centrality strategy;
# strategy = 'low'     : Lowest centrality strategy;
# strategy = 'rho'     : ρ-dependent strategy.
# strategy = 'reverse' : reverse ρ-dependent strategy.
def Func_CentralityPortf(dta, strategy, var='return', corr='Spear'):
    ## Collect the centrality and the corresponding ρ between the centrality and w*.
    Centralities = pd.DataFrame()
    rhos = []
    _col = []
    for period in range(len(dta)):
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
        
        ## Update column name.
        _col.append(dta[period].index[-1])

    Centralities.index = Returns.columns
    Centralities.columns = _col

    ## Construct the weights of the Highest Centrality Strategy that acts as a benchmark.
    weight_t = pd.DataFrame(0, index=Returns.columns, columns=range(len(Rolling_test)))
    for period in range(len(dta)):
        ## Find the index of the stocks with the highest/lowest centralities among each holding periods.
        if strategy == 'high' or (strategy=='rho' and rhos[period]>Rho_tilde) or (strategy=='reverse' and rhos[period]<=Rho_tilde):
            idx = Centralities.iloc[:, period].nlargest(NumStock).index
        elif strategy == 'low' or (strategy=='rho' and rhos[period]<=Rho_tilde) or (strategy=='reverse' and rhos[period]>Rho_tilde):
            idx = Centralities.iloc[:, period].nsmallest(NumStock).index
        
        ## Naively invest stocks above for each holding period.
        weight_t.loc[idx, period] = 1 / NumStock

    return weight_t

#%%
## Execute the ρ-dependent strategy with Pearson correlation coefficient of stock returns.
Weight_rhoPearson_r = Func_CentralityPortf(Rolling_test, strategy='rho', var='return', corr='Pearson')

Weight_rhoPearson_r = Weight_rhoPearson_r.T
Weight_rhoPearson_r.index = InvDates


## Compute the changes in the holding within the holding period over time.
Exposures_rhoPearson_r, Holding_rhoPearson_r, Weight_rhoPearson_r_end = Func_Holding(Weight_rhoPearson_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_rhoPearson_r, Var_rhoPearson_r, _ = \
    Func_Performance(Weight_rhoPearson_r, Weight_rhoPearson_r_end, Exposures_rhoPearson_r)

Sharpe_rhoPearson_r.mean()
Sharpe_rhoPearson_r.std()

Var_rhoPearson_r.mean()
Var_rhoPearson_r.std()




#%% Lowest Centrality Strategy - Pearson Correlation of Returns

## Execute the lowest centrality strategy with Pearson correlation coefficient of stock returns.
Weight_LCentralPearson_r = Func_CentralityPortf(Rolling_test, strategy='low', var='return', corr='Pearson')

Weight_LCentralPearson_r = Weight_LCentralPearson_r.T
Weight_LCentralPearson_r.index = InvDates


## Compute the changes in the holding within the holding period over time.
Exposures_LCentralPearson_r, Holding_LCentralPearson_r, Weight_LCentralPearson_r_end = Func_Holding(Weight_LCentralPearson_r)


## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover, Sharpe_LCentralPearson_r, Var_LCentralPearson_r, _ = \
    Func_Performance(Weight_LCentralPearson_r, Weight_LCentralPearson_r_end, Exposures_LCentralPearson_r)

Sharpe_LCentralPearson_r.mean()
Sharpe_LCentralPearson_r.std()

Var_LCentralPearson_r.mean()
Var_LCentralPearson_r.std()



#%% New Strategy - Naive

Thresh_67 = Train_TW50.quantile(0.67)
Thresh_33 = Train_TW50.quantile(0.33)

## Collect the centrality and the corresponding ρ between the centrality and w*.
Centralities = pd.DataFrame()
_col = []
for period in range(len(Rolling_test)):
    _, _, c_t, G_t = Func_AdjM(Rolling_test[period])
    
    c_t = pd.DataFrame(c_t)
    Centralities = pd.concat([Centralities, c_t], axis=1)
    
    ## Update column name.
    _col.append(Rolling_test[period].index[-1])

Centralities.index = Returns.columns
Centralities.columns = _col


## Construct the weights of the Highest Centrality Strategy that acts as a benchmark.
Weight_New_Naive = pd.DataFrame(0, index=Returns.columns, columns=range(len(Rolling_test)))
for period in range(len(Rolling_test)):
    ## Find the index of the stocks with the highest/lowest centralities among each holding periods.
    if TW50_open.loc[ Rolling_test[period].index[0] ] >= Thresh_67:
        idx = Sharpe_roll.iloc[:, _period].nlargest(NumStock).index
    elif TW50_open.loc[ Rolling_test[period].index[0] ] <= Thresh_33:
        idx = Centralities.iloc[:, period].nsmallest(NumStock).index
    else:
        # idx = Centralities.iloc[:, period].nsmallest(NumStock).index
        idx = Sharpe_roll.iloc[:, _period].nlargest(NumStock).index
    
    ## Naively invest stocks above for each holding period.
    Weight_New_Naive.loc[idx, period] = 1 / NumStock

Weight_New_Naive.columns = _col
Weight_New_Naive = Weight_New_Naive.T


Exposures_New_N, Holding_New_N, Weight_New_N_end = Func_Holding(Weight_New_Naive)

## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover_New_N, Sharpe_New_N, Var_New_N, _ = Func_Performance(Weight_New_Naive, Weight_New_N_end, Exposures_New_N)

Sharpe_New_N.mean()
Sharpe_New_N.std()

Var_New_N.mean()
Var_New_N.std()

# Cmr_New_N = Cmr_fake(Holding_New_N)

# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
# plt.bar(range(1, len(Rolling_test)+1), (Holding_New_N.sum()-1) * 100, color='blue')
plt.plot(_col, (Exposures_New_N.sum()-1) * 100, label='New', color='k')
plt.plot(_col, (Exposures_Sharpe.sum()-1) * 100, label='High Sharpe', color='r')
plt.plot(_col, (Exposures_LCentralPearson_r.sum()-1) * 100, label='Low Central', color='g')
plt.plot(_col, (Exposures_rhoPearson_r.sum()-1) * 100, label='ρ-dependent', color='cyan')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The $n^{th}$ Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Cumulative Return of New Strategy',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
plt.legend(fontsize=20)
plt.show()


#%% Environment 2

import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
import yfinance as yf
import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, KFold
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import to_tree
from sklearn.cluster import DBSCAN

# 計算dtw距離
def cal_dtw_distance(s_a, s_b):
    d=lambda x, y: abs(x - y)
    max_warping_window = 10000
    s_a, s_b = np.array(s_a), np.array(s_b)
    M, N = len(s_a), len(s_b)
    cost = sys.maxsize * np.ones((M, N))
    cost[0, 0] = d(s_a[0], s_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(s_a[i], s_b[0])
    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(s_a[0], s_b[j])
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window),
                       min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(s_a[i], s_b[j])
    return cost[-1, -1]
# 計算nodes深度
def get_leaf_depths(node):
    depths = {}
    def traverse(current_node, current_depth):
        if current_node:
            if current_node.is_leaf():
                depths[current_node.id] = current_depth
            else:
                traverse(current_node.left, current_depth + 1)
                traverse(current_node.right, current_depth + 1)
    traverse(node, 0)
    return depths
def stand(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    return data

def dtw_matrix(data):
  X = data.index.values
  m = np.zeros((len(X),len(X)))
  for i in range(len(X)):
    for j in range(len(X)):
      m[i,j] = cal_dtw_distance(data.iloc[i,:],data.iloc[j,:])
  return pd.DataFrame(m,columns=data.index, index=data.index)


## Calculate the rolling windows for computing returns over test set.
# 2. Both rolling windows and holding periods and are in test set.
Rolling_close = []
for _i in range(0, len(Test_close)-(M_target+H_target), SpeedOfRolling):
    windows = Test_close.iloc[ _i : _i + M_target ]
    Rolling_close.append(windows)

# 3. Randomly pick n holding periods for comparison the strategies.
random.seed(random_state)
Rolling_close = [ Rolling_close[i] for i in random.sample(range(0, len(Rolling_close)), NumOfTestSample) ]




#%% EDA

## Use dynamic time warpping to Clustering
Y = Stock_close.corr()

# X= stand(Stock_close.T)
X= stand(Stock_close.iloc[-1000:].T)
X = dtw_matrix(X)

plt.figure(figsize=(8, 6))
plt.title('Correlation Matrix')
sns.heatmap(abs(Y), cmap='coolwarm', annot=False, square=True, cbar=True,vmin=0, vmax=1)
plt.show()

plt.figure(figsize=(8, 6))
plt.title('DTW Distance Matrix')
sns.heatmap((X.max()-X)/X.max(), cmap='coolwarm', annot=False, square=True, cbar=True)
plt.show()

# Hierarchical Clustering Demo
Z = linkage(stand(Stock_close.iloc[-1000:].T), metric = cal_dtw_distance)  # 'complete' linkage method
plt.figure(figsize=(10, 6))
tickers = Stock_close.columns
dendrogram(Z, labels=tickers, truncate_mode='lastp', p=len(tickers))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()




#%% Hierarchical Clustering

# ma is a rolling window data for "Adj Close"
def Window_for_Hierarchical(ma):
  #抓股票的名稱以建立權重
  weights = pd.DataFrame(ma.T.index,columns = ['stock'])
  Adj_close = stand(ma.T)
  G = linkage(Adj_close, metric = cal_dtw_distance)
  #將G轉為Tree
  tree = to_tree(G)
  leaf_depths = get_leaf_depths(tree)
  depths = pd.DataFrame(list(leaf_depths.items()), columns=['Leaf Node', 'Depth'])
  depths['weight'] = (1 / 2)**depths['Depth']
  weights[f'Period'] = pd.Series(depths['weight'].values, index=depths['Leaf Node']).reindex(weights.index)
  return weights.set_index('stock')


## Calculate the optimal weight using rolling windows.
Weight_Hierarch = pd.DataFrame()
_col = []
for _period in Rolling_close: 
    weight_t = Window_for_Hierarchical( _period )
    ## Scale up to make sure the sum of actual weights is 100%.
    # Remove small numerical calculated weights.
    weight_t[ weight_t < 10 ** -4 ] = 0
    weight_t = weight_t / weight_t.sum()
    Weight_Hierarch = pd.concat([ Weight_Hierarch, weight_t ], axis=1)
    _col.append( _period.index[-1] )
    
Weight_Hierarch = Weight_Hierarch.T
Weight_Hierarch.index = _col
Weight_Hierarch.columns = Stock_close.columns


Exposures_Hierarch, Holding_Hierarch, Weight_Hierarch_end = Func_Holding(Weight_Hierarch)

## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover_Hierarch, Sharpe_Hierarch, Var_Hierarch, _ = Func_Performance(Weight_Hierarch, Weight_Hierarch_end, Exposures_Hierarch)

Sharpe_Hierarch.mean()
Sharpe_Hierarch.std()

Var_Hierarch.mean()
Var_Hierarch.std()

# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(Rolling_test)+1), (Holding_Hierarch.sum()-1) * 100, color='blue')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The $n^{th}$ Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of Hierarchical Clustering Strategy',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
# plt.legend(fontsize=20)
plt.show()




#%% Search of hyperparmeter - K

#we first choose parameter k
#data is adj close train data
Adj_close = Stock_close.T
SSE = []
kvalues = range(1,21)
for k in kvalues:
    estimator = KMeans(n_clusters=k,n_init=10)
    estimator.fit(Adj_close)
    SSE.append(estimator.inertia_)
X = range(1,21)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()



#%% DBSCAN


def Window_for_DBSCAN(ma):
  weights = pd.DataFrame(ma.T.index,columns = ['stock'])
  Adj_close = stand(ma.T)
  clustering = DBSCAN()
  labels = clustering.fit_predict(Adj_close)
  unique_labels, label_counts = np.unique(labels, return_counts=True)
  non_noise_labels = unique_labels[unique_labels != -1]
  num_clusters = len(non_noise_labels)
  weights = np.zeros(len(labels))
  for label in non_noise_labels:
      stocks_in_label = Adj_close.index[labels == label]
      num_stocks_in_label = len(stocks_in_label)
      weights[labels == label] = 1 / (num_clusters * num_stocks_in_label)
  weights_df = pd.DataFrame({'stock': Adj_close.index, 'weight': weights})
  return weights_df.set_index('stock')


## Calculate the optimal weight using rolling windows.
Weight_DBSCAN = pd.DataFrame()
_col = []
for _period in Rolling_close: 
    weight_t = Window_for_DBSCAN( _period )
    ## Scale up to make sure the sum of actual weights is 100%.
    # Remove small numerical calculated weights.
    # weight_t[ weight_t < 10 ** -4 ] = 0
    # weight_t = weight_t / weight_t.sum()
    Weight_DBSCAN = pd.concat([ Weight_DBSCAN, weight_t ], axis=1)
    _col.append( _period.index[-1] )
    
Weight_DBSCAN = Weight_DBSCAN.T
Weight_DBSCAN.index = _col



Exposures_DBSCAN, Holding_DBSCAN, Weight_DBSCAN_end = Func_Holding(Weight_DBSCAN)

## The mean and dispersion of the poortfolio Sharpes and variances.
Turnover_DBSCAN, Sharpe_DBSCAN, Var_DBSCAN, _ = Func_Performance(Weight_DBSCAN, Weight_DBSCAN_end, Exposures_DBSCAN)

Sharpe_DBSCAN.mean()
Sharpe_DBSCAN.std()

Var_DBSCAN.mean()
Var_DBSCAN.std()

# Create a plot for single period return after each holding period.
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(Rolling_test)+1), (Holding_DBSCAN.sum()-1) * 100, color='blue')
plt.axhline(0, c='k', ls='--')
plt.xlabel('The $n^{th}$ Holding Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.title('The Single Period Return of DBSCAN Strategy',
          fontsize=20)
plt.grid(True)
# plt.axis([2000, 2025, 0, 60])
# plt.legend(fontsize=20)
plt.show()




#%% Label

# data is train set we use
def optimal_k_model(data):
  Adj_close = stand(data["Adj Close"].T)
  Volumne = stand(data["Volume"].T)
  High = stand(data["High"].T)
  Low = stand(data["Low"].T)
  Open = stand(data["Open"].T)
  #在Rolling window下,給每隻Stock進行label((倒數730天內的最高價-倒數第730天的Adj_close)/倒數第730天的Adj_close > 0.1,則為Profit stock)
  citeration = ((data["High"].T.iloc[:,-730:].max(axis=1)-data["Adj Close"].T.iloc[:,-730])/data["Adj Close"].T.iloc[:,-730] >= 0.1).astype(int)
  k_values = range(1,21)
  cv_scores = []
  for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k,metric = cal_dtw_distance)
    scores = cross_val_score(knn,Adj_close,citeration,cv=10,scoring = 'accuracy')
    cv_scores.append(scores.mean())
  optimal_k = k_values[cv_scores.index(max(cv_scores))]
  knn = KNeighborsClassifier(n_neighbors=optimal_k,metric=cal_dtw_distance)
  model1 = knn.fit(Adj_close,citeration)
  model2 = knn.fit(High,citeration)
  model3 = knn.fit(Low,citeration)
  model4 = knn.fit(Volumne,citeration)
  model5 = knn.fit(Open,citeration)
  return model1,model2,model3,model4,model5

# model1,model2,model3,model4,model5 = optimal_k_model(Stock)

# m is rolling period, and model1 to model5 is determined by optimal_k_model()
def window_for_knn(m):
  Adj_close = stand(m["Adj Close"].T)
  Volumne = stand(m["Volume"].T)
  High = stand(m["High"].T)
  Low = stand(m["Low"].T)
  Open = stand(m["Open"].T)
  weights = pd.DataFrame(Adj_close.index,columns = ['stock'])
  
  ## New!
  model1,model2,model3,model4,model5 = optimal_k_model(m)
  ## New!
  
  predictions = model1.predict(Adj_close) + model2.predict(High) + model3.predict(Low) + model4.predict(Volumne) + model5.predict(Open)
  p = (predictions >= 3).astype(int)
  weights[f'Period'] = p/p.sum()
  p_sum = p.sum()
  if p_sum == 0:
      weights[f'Period'] = np.zeros_like(p)
  else:
      weights[f'Period'] = p / p_sum
  return weights

window_for_knn(Stock.iloc[0:1050,:])




#%% EDA

### Cumulative Return

## Compute and sketch the cumulative return of Taiwan 50 index.
TW50_cumR = (1 + TW50_close.pct_change()[1:]).cumprod() - 1

plt.figure(figsize=(12, 6))
# plt.plot(TW50_close.index[1:], TW50_close.pct_change()[1:] * 100, color='k')
plt.plot(TW50_cumR.index, TW50_cumR * 100, color='k')
plt.axhline(0, c='b', ls='--')
plt.xlabel('Time Period', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
# plt.title('The Cumulative Daily Return of the Taiwan 50',
#           fontsize=20)
plt.grid(True)
# plt.axis([2010, 2024, -7, 8.5])
# plt.legend(fontsize=20)
plt.show()


### Index Value

# TW50_r = TW50_close.pct_change()[1:]

plt.figure(figsize=(12, 6))
plt.plot(TW50_close.index[1:], TW50_close.pct_change()[1:] * 100, color='k')
# plt.plot(TW50_r.index, TW50_r, color='k')
# plt.axhline(0, c='b', ls='--')
plt.xlabel('Time Period', fontsize=20)
plt.ylabel('Index Value', fontsize=20)
plt.title('The Index Value of Taiwan 50',
          fontsize=20)
plt.grid(True)
# plt.axis([2010, 2024, -7, 8.5])
# plt.legend(fontsize=20)
plt.show()



