# Copyright (C) 2024 Gennis
'''
Initiated Date    : 2024/04/13
Last Updated Date : 2024/04/17
Aim: Interaction Analysis of Three Way Contingency Tables.
Input Data: loan data of Lending Club.
    https://github.com/dosei1/Lending-Club-Loan-Data/tree/master
'''


#%% Enviornment

from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import os
import pandas as pd
import pingouin as pg
import re
import seaborn as sns
import scipy
import scipy.stats as stat
# import sklearn as sk
from sklearn import linear_model as skl
from sklearn import model_selection as skm
from sklearn import preprocessing as skp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
%matplotlib inline


## Set the path of data.
_Path = r'D:\03Programs_Clouds\Google Drive\NSYSU\02Discrete Data Analysis\Mid-Term'
os.chdir(_Path)

## Correctly show the Mandarin in the graphs.
matplotlib.rc('font', family='Microsoft JhengHei')
## Correctly show the negative sign in the graphs.
plt.rcParams['axes.unicode_minus'] = False




# %% Data Access

_FileName = 'filtered_loans.csv'
File = pd.read_csv(_FileName)




# %% Extract-Transform-Load

Data = deepcopy(File)
## Remove the data with undefined status of home ownership.
# Lose 3 obs.
Data = Data[ Data['home_ownership_NONE'] == 0 ]
# Lose 96 obs.
Data = Data[ Data['home_ownership_OTHER'] == 0 ]


## Select the categorical variables used for analysis.
Data = Data[['loan_status', 
             'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT',
             'verification_status_Not Verified', 'verification_status_Source Verified',
             'verification_status_Verified',
             'purpose_car',
             'purpose_credit_card',
             'purpose_debt_consolidation',
             'purpose_educational',
             'purpose_home_improvement',
             'purpose_house',
             'purpose_major_purchase',
             'purpose_medical',
             'purpose_moving',
             'purpose_other',
             'purpose_renewable_energy',
             'purpose_small_business',
             'purpose_vacation',
             'purpose_wedding',
             'term_ 36 months',
             'term_ 60 months']]




#%% Convergency Table Construction

## Non-ranked data set.
Data_1 = Data[['loan_status', 
               'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT',
               'verification_status_Not Verified', 'verification_status_Source Verified',
               'verification_status_Verified']]

# Data_1.columns = ['Status', 'Mortgage', 'Own', 'Rent', 'NV', 'Source V', 'V']


## Construct conditions.
# i
cond_i1 = Data_1['loan_status'] == 0
cond_i2 = Data_1['loan_status'] == 1

# j
cond_j1 = Data_1['home_ownership_OWN'] == 1
cond_j2 = Data_1['home_ownership_MORTGAGE'] == 1
cond_j3 = Data_1['home_ownership_RENT'] == 1

# k
cond_k1 = Data_1['verification_status_Verified'] == 1
cond_k2 = Data_1['verification_status_Source Verified'] == 1
cond_k3 = Data_1['verification_status_Not Verified'] == 1


## Compute the elements of a 2x3x3 contingency table.
# For n_ijk,
# i = 1: Good standing; i = 2: Not in good standing.
# j = 1: Own; j = 2: Mortgate; j = 3: Rent.
# k = 1: Verified; k = 2: Source verified; k = 3: Not verified.
n_111 = len(Data_1.loc[cond_i1 & cond_j1 & cond_k1])
n_211 = len(Data_1.loc[cond_i2 & cond_j1 & cond_k1])
n_121 = len(Data_1.loc[cond_i1 & cond_j2 & cond_k1])
n_131 = len(Data_1.loc[cond_i1 & cond_j3 & cond_k1])
n_221 = len(Data_1.loc[cond_i2 & cond_j2 & cond_k1])
n_231 = len(Data_1.loc[cond_i2 & cond_j3 & cond_k1])
n_112 = len(Data_1.loc[cond_i1 & cond_j1 & cond_k2])
n_113 = len(Data_1.loc[cond_i1 & cond_j1 & cond_k3])
n_212 = len(Data_1.loc[cond_i2 & cond_j1 & cond_k2])
n_213 = len(Data_1.loc[cond_i2 & cond_j1 & cond_k3])
n_122 = len(Data_1.loc[cond_i1 & cond_j2 & cond_k2])
n_123 = len(Data_1.loc[cond_i1 & cond_j2 & cond_k3])
n_222 = len(Data_1.loc[cond_i2 & cond_j2 & cond_k2])
n_223 = len(Data_1.loc[cond_i2 & cond_j2 & cond_k3])
n_132 = len(Data_1.loc[cond_i1 & cond_j3 & cond_k2])
n_133 = len(Data_1.loc[cond_i1 & cond_j3 & cond_k3])
n_232 = len(Data_1.loc[cond_i2 & cond_j3 & cond_k2])
n_233 = len(Data_1.loc[cond_i2 & cond_j3 & cond_k3])


## Construct the 2x3x3 contingency table.
# Construct 2x2 tables by k.
# When k = 1:
Table_k1 = pd.DataFrame({'Status: Good': [n_111, n_121, n_131],
                         'Status: Not Good': [n_211, n_221, n_231],
                         }, index = ['Home: Own', 'Home: Mortgage', 'Home: Rent']
                        )

# When k = 2:
Table_k2 = pd.DataFrame({'Status: Good': [n_112, n_122, n_132],
                         'Status: Not Good': [n_212, n_222, n_232],
                         }, index = ['Home: Own', 'Home: Mortgage', 'Home: Rent']
                        )

# When k = 3:
Table_k3 = pd.DataFrame({'Status: Good': [n_113, n_123, n_133],
                         'Status: Not Good': [n_213, n_223, n_233],
                         }, index = ['Home: Own', 'Home: Mortgage', 'Home: Rent']
                        )


## Summing over the i and j dimensions.
Tables_k = []
for _table in deepcopy([Table_k1, Table_k2, Table_k3]):
    ## Sum over the j dimension.
    _table['Sum: Home'] = _table.sum(axis=1)
    ## Sum over the i dimension.
    _table = _table.T
    _table['Sum: Status'] = _table.sum(axis=1)
    _table = _table.T

    Tables_k.append(_table)

## Check the data.
Tables_k[0].iloc[-1, -1] + Tables_k[1].iloc[-1, -1]  + Tables_k[2].iloc[-1, -1] == len(Data_1)



## Use long table to represent the contingency table.
Long = pd.DataFrame({'Home': ['Own', 'Own', 'Own', 
                              'Mortgage', 'Mortgage', 'Mortgage', 'Rent', 'Rent', 'Rent'],
                     'Verification': ['Verified', 'Source', 'Not', 
                                      'Verified', 'Source', 'Not', 'Verified', 'Source', 'Not'],
                     'Status: Good': [n_111, n_112, n_113, n_121, n_122, n_123, n_131, n_132, n_133],
                     'Status: Not Good': [n_211, n_212, n_213, n_221, n_222, n_223, n_231, n_232, n_233]
                     })



## Use array to construct the 2x3x3 contingency table.
# Contingency[i, j, k]
Contingency = np.array([
                        [ [n_111, n_112, n_113],
                          [n_121, n_122, n_123],
                          [n_131, n_132, n_133], ],
                        [ [n_211, n_212, n_213],
                          [n_221, n_222, n_223],
                          [n_231, n_232, n_233], ],
                       ])





# %% Mutual Independence Test

## Set the parameters.
n = Contingency.sum()
I = 2
J = 3
K = 3


# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, :, :].sum() * Contingency[:, j, :].sum() * Contingency[:, :, k].sum() / n**2
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (I - 1) + (J - 1) + (K - 1) )

## Compute the p-value.
# Null hypothesis: mutual independence.
1 - stat.chi2.cdf(X2, df)




# %% Joint Independence Test

# %%% (AB, C)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, j, :].sum() * Contingency[:, :, k].sum() / n
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (I * J - 1) + (K - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)



# %%% (AC, B)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, :, k].sum() * Contingency[:, j, :].sum() / n
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (I * K - 1) + (J - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)



# %%% (BC, A)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[:, j, k].sum() * Contingency[i, :, :].sum() / n
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (J * K - 1) + (I - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)




# %% Conditional Independence Test

# %%% (AB, AC)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, j, :].sum() * Contingency[i, :, k].sum() / Contingency[i, :, :].sum()
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (I - 1) + I * (J - 1) + I * (K - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)



# %%% (AB, BC)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, j, :].sum() * Contingency[:, j, k].sum() / Contingency[:, j, :].sum()
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (J - 1) + J * (I - 1) + J * (K - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)



# %%% (AC, BC)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, :, k].sum() * Contingency[:, j, k].sum() / Contingency[:, :, k].sum()
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (K - 1) + K * (I - 1) + K * (J - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)




# %% Homogeneous Association Test (AB, BC, AC)

## Examine each 2x2x3 tables in turn.
M = [ [0, 1], [0, 2], [1, 2] ]

# %%% Log Odds Ratio Test

# %%%% Division w.r.t. Dimension j

Woolf = []
p_values = []

## For each 2x2x3 table,
for m in M:
    numerator = 0
    denominator = 0
    statistics = 0
    ## Run the test.
    for k in range(K):
        ## Compute log odds ratio.
        n_11k = Contingency[:, m, k][0, 0]
        n_22k = Contingency[:, m, k][1, 1]
        n_12k = Contingency[:, m, k][0, 1]
        n_21k = Contingency[:, m, k][1, 0]
        
        logOR_k = np.log( n_11k * n_22k / (n_12k * n_21k) )
        
        ## Compute the weight.
        w_k = 1 / ( 1/n_11k + 1/n_22k + 1/n_12k + 1/n_21k )
        
        ## Calculate the common odds ratio.
        numerator += w_k * logOR_k
        denominator += w_k
        
    logOR_bar = numerator / denominator
    
    for k in range(K):
        ## Compute the test statistics of Woolf's method.
        n_11k = Contingency[:, m, k][0, 0]
        n_22k = Contingency[:, m, k][1, 1]
        n_12k = Contingency[:, m, k][0, 1]
        n_21k = Contingency[:, m, k][1, 0]
        logOR_k = np.log( n_11k * n_22k / (n_12k * n_21k) )
        w_k = 1 / ( 1/n_11k + 1/n_22k + 1/n_12k + 1/n_21k )
        
        statistics += w_k * ( logOR_k - logOR_bar )**2
          
    ## Collect the test statistics and p-values.
    Woolf.append( statistics )
    p_values.append( 1 - stat.chi2.cdf(statistics, K-1) )



# %%%% Division w.r.t. Dimension k

Woolf = []
p_values = []

## For each 2x2x3 table,
for m in M:
    numerator = 0
    denominator = 0
    statistics = 0
    ## Run the test.
    for j in range(J):
        ## Compute log odds ratio.
        n_1j1 = Contingency[:, j, m][0, 0]
        n_2j2 = Contingency[:, j, m][1, 1]
        n_1j2 = Contingency[:, j, m][0, 1]
        n_2j1 = Contingency[:, j, m][1, 0]
        
        logOR_j = np.log( n_1j1 * n_2j2 / (n_1j2 * n_2j1) )
        
        ## Compute the weight.
        w_j = 1 / ( 1/n_1j1 + 1/n_2j2 + 1/n_1j2 + 1/n_2j1 )
        
        ## Calculate the common odds ratio.
        numerator += w_j * logOR_j
        denominator += w_j
        
    logOR_bar = numerator / denominator
    
    for j in range(J):
        ## Compute the test statistics of Woolf's method.
        n_1j1 = Contingency[:, j, m][0, 0]
        n_2j2 = Contingency[:, j, m][1, 1]
        n_1j2 = Contingency[:, j, m][0, 1]
        n_2j1 = Contingency[:, j, m][1, 0]
        logOR_j = np.log( n_1j1 * n_2j2 / (n_1j2 * n_2j1) )
        w_j = 1 / ( 1/n_1j1 + 1/n_2j2 + 1/n_1j2 + 1/n_2j1 )
        
        statistics += w_j * ( logOR_j - logOR_bar )**2
          
    ## Collect the test statistics and p-values.
    Woolf.append( statistics )
    p_values.append( 1 - stat.chi2.cdf(statistics, J-1) )



# %%% Cochran-Mantel-Haenszel Test

# %%%% Division w.r.t. Dimension j

CMH = []
p_values = []

## For each 2x2x3 table,
for m in M:
    diff = 0
    var = 0
    ## Run the Cochran-Mantel-Haenszel test.
    for k in range(K):
        ## Compute expected count.
        n_01k = Contingency[:, m, k][:, 0].sum()
        n_10k = Contingency[:, m, k][0, :].sum()
        n_00k = Contingency[:, m, k].sum()
        
        mu = n_01k * n_10k / n_00k
        
        ## Compute the difference between the observed count and the expected count.
        diff += Contingency[:, m, k][0, 0] - mu
        
        
        ## Compute the variance.
        n_02k = Contingency[:, m, k][:, 1].sum()
        n_20k = Contingency[:, m, k][1, :].sum()
        
        var += n_01k / n_00k * n_10k / n_00k / (n_00k - 1) * n_02k * n_20k   
        
        
    ## Compute the test statistics and p-values.
    # CMH.append( (abs(diff) - 0.5) ** 2 / var )
    # p_values.append( 1 - stat.chi2.cdf( (abs(diff) - 0.5) ** 2 / var, 1) )
    CMH.append( diff ** 2 / var )
    p_values.append( 1 - stat.chi2.cdf(diff ** 2 / var, 1) )



# %%%% Division w.r.t. Dimension k

CMH = []
p_values = []

## For each 2x2x3 table,
for m in M:
    diff = 0
    var = 0
    ## Run the Cochran-Mantel-Haenszel test.
    for j in range(J):
        ## Compute expected count.
        n_0j1 = Contingency[:, j, m][:, 0].sum()
        n_1j0 = Contingency[:, j, m][0, :].sum()
        n_0j0 = Contingency[:, j, m].sum()
        
        mu = n_0j1 * n_1j0 / n_0j0
        
        ## Compute the difference between the observed count and the expected count.
        diff += Contingency[:, j, m][0, 0] - mu
        
        
        ## Compute the variance.
        n_0j2 = Contingency[:, m, k][:, 1].sum()
        n_2j0 = Contingency[:, m, k][1, :].sum()
        
        var += n_0j1 / n_0j0 * n_1j0 / n_0j0 * n_0j2 / (n_0j0 - 1) * n_2j0
        
        
    ## Compute the test statistics and p-values.
    CMH.append( diff ** 2 / var )
    p_values.append( 1 - stat.chi2.cdf(diff ** 2 / var, 1) )




# %% Second Dataset

## Ranked data set.
Data_2 = Data[['loan_status', 
               'purpose_car',
               'purpose_credit_card',
               'purpose_debt_consolidation',
               'purpose_educational',
               'purpose_home_improvement',
               'purpose_house',
               'purpose_major_purchase',
               'purpose_medical',
               'purpose_moving',
               'purpose_other',
               'purpose_renewable_energy',
               'purpose_small_business',
               'purpose_vacation',
               'purpose_wedding',
               'term_ 36 months',
               'term_ 60 months']]


## Construct conditions.
# i
cond_i1 = Data_1['loan_status'] == 0
cond_i2 = Data_1['loan_status'] == 1

# j
cond_j1 = Data_2['term_ 60 months'] == 1
cond_j2 = Data_2['term_ 36 months'] == 1

# k
cond_k1 = Data_2['purpose_car'] == 1
cond_k2 = Data_2['purpose_credit_card'] == 1
cond_k3 = Data_2['purpose_debt_consolidation'] == 1
cond_k4 = Data_2['purpose_educational'] == 1
cond_k5 = Data_2['purpose_home_improvement'] == 1
cond_k6 = Data_2['purpose_house'] == 1
cond_k7 = Data_2['purpose_major_purchase'] == 1
cond_k8 = Data_2['purpose_medical'] == 1
cond_k9 = Data_2['purpose_moving'] == 1
cond_kA = Data_2['purpose_other'] == 1
cond_kB = Data_2['purpose_renewable_energy'] == 1
cond_kC = Data_2['purpose_small_business'] == 1
cond_kD = Data_2['purpose_vacation'] == 1
cond_kE = Data_2['purpose_wedding'] == 1


## Compute the elements of a 2x3x3 contingency table.
# For n_ijk,
# i = 1: Good standing; i = 2: Not in good standing.
# j = 1: 60M term; j = 2: 36M term.
# k: Purposes of borrowing.
n_111 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k1])
n_211 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k1])
n_121 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k1])
n_221 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k1])
n_112 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k2])
n_212 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k2])
n_122 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k2])
n_222 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k2])
n_113 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k3])
n_213 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k3])
n_123 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k3])
n_223 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k3])
n_114 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k4])
n_214 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k4])
n_124 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k4])
n_224 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k4])
n_115 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k5])
n_215 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k5])
n_125 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k5])
n_225 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k5])
n_116 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k6])
n_216 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k6])
n_126 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k6])
n_226 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k6])
n_117 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k7])
n_217 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k7])
n_127 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k7])
n_227 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k7])
n_118 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k8])
n_218 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k8])
n_128 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k8])
n_228 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k8])
n_119 = len(Data_2.loc[cond_i1 & cond_j1 & cond_k9])
n_219 = len(Data_2.loc[cond_i2 & cond_j1 & cond_k9])
n_129 = len(Data_2.loc[cond_i1 & cond_j2 & cond_k9])
n_229 = len(Data_2.loc[cond_i2 & cond_j2 & cond_k9])
n_11A = len(Data_2.loc[cond_i1 & cond_j1 & cond_kA])
n_21A = len(Data_2.loc[cond_i2 & cond_j1 & cond_kA])
n_12A = len(Data_2.loc[cond_i1 & cond_j2 & cond_kA])
n_22A = len(Data_2.loc[cond_i2 & cond_j2 & cond_kA])
n_11B = len(Data_2.loc[cond_i1 & cond_j1 & cond_kB])
n_21B = len(Data_2.loc[cond_i2 & cond_j1 & cond_kB])
n_12B = len(Data_2.loc[cond_i1 & cond_j2 & cond_kB])
n_22B = len(Data_2.loc[cond_i2 & cond_j2 & cond_kB])
n_11C = len(Data_2.loc[cond_i1 & cond_j1 & cond_kC])
n_21C = len(Data_2.loc[cond_i2 & cond_j1 & cond_kC])
n_12C = len(Data_2.loc[cond_i1 & cond_j2 & cond_kC])
n_22C = len(Data_2.loc[cond_i2 & cond_j2 & cond_kC])
n_11D = len(Data_2.loc[cond_i1 & cond_j1 & cond_kD])
n_21D = len(Data_2.loc[cond_i2 & cond_j1 & cond_kD])
n_12D = len(Data_2.loc[cond_i1 & cond_j2 & cond_kD])
n_22D = len(Data_2.loc[cond_i2 & cond_j2 & cond_kD])
n_11E = len(Data_2.loc[cond_i1 & cond_j1 & cond_kE])
n_21E = len(Data_2.loc[cond_i2 & cond_j1 & cond_kE])
n_12E = len(Data_2.loc[cond_i1 & cond_j2 & cond_kE])
n_22E = len(Data_2.loc[cond_i2 & cond_j2 & cond_kE])


## Use array to construct the 2x2x14 contingency table.
# Contingency[i, j, k]
Contingency = np.array([
                        [ [n_111, n_112, n_113, n_114, n_115, n_116, n_117, n_118, n_119, 
                           n_11A, n_11B, n_11C, n_11D, n_11E],
                          [n_121, n_122, n_123, n_124, n_125, n_126, n_127, n_128, n_129,
                           n_12A, n_12B, n_12C, n_12D, n_12E] ],
                        [ [n_211, n_212, n_213, n_214, n_215, n_216, n_217, n_218, n_219,
                           n_21A, n_21B, n_21C, n_21D, n_21E],
                          [n_221, n_222, n_223, n_224, n_225, n_226, n_227, n_228, n_229,
                           n_22A, n_22B, n_22C, n_22D, n_22E] ]
                       ])


## Use long table to represent the contingency table.
Long2 = pd.DataFrame({'Term': ['36M', '36M', '36M', '36M', '36M',
                               '36M', '36M', '36M', '36M', '36M',
                               '36M', '36M', '36M', '36M', 
                               '60M', '60M', '60M', '60M', '60M',
                               '60M', '60M', '60M', '60M', '60M',
                               '60M', '60M', '60M', '60M'],
                      'Purpose': ['Car', 'Credit Card', 'Debt Consolidation',
                                  'Educational', 'Home Improvement', 'House',
                                  'Major Purchase', 'Medical', 'Moving',
                                  'Other', 'Renewable Energy', 'Small Business',
                                  'Vacation', 'Wedding',
                                  'Car', 'Credit Card', 'Debt Consolidation',
                                  'Educational', 'Home Improvement', 'House',
                                  'Major Purchase', 'Medical', 'Moving',
                                  'Other', 'Renewable Energy', 'Small Business',
                                  'Vacation', 'Wedding'],
                      'Status: Good': [n_111, n_112, n_113, n_114, n_115,
                                       n_116, n_117, n_118, n_119, n_11A,
                                       n_11B, n_11C, n_11D, n_11E,
                                       n_121, n_122, n_123, n_124, n_125,
                                       n_126, n_127, n_128, n_129, n_12A,
                                       n_12B, n_12C, n_12D, n_12E],
                      'Status: Not Good': [n_211, n_212, n_213, n_214, n_215,
                                           n_216, n_217, n_218, n_219, n_21A,
                                           n_21B, n_21C, n_21D, n_21E,
                                           n_221, n_222, n_223, n_224, n_225,
                                           n_226, n_227, n_228, n_229, n_22A,
                                           n_22B, n_22C, n_22D, n_22E]
                      })





# %% Mutual Independence Test

## Set the parameters.
n = Contingency.sum()
I = 2
J = 2
K = 14


# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, :, :].sum() / n * Contingency[:, j, :].sum() / n * Contingency[:, :, k].sum()
            
            X2 += ( Contingency[i, j, k] - mu ) / mu * ( Contingency[i, j, k] - mu ) 

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (I - 1) + (J - 1) + (K - 1) )

## Compute the p-value.
# Null hypothesis: mutual independence.
1 - stat.chi2.cdf(X2, df)




# %% Joint Independence Test

# %%% (AB, C)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, j, :].sum() * Contingency[:, :, k].sum() / n
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (I * J - 1) + (K - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)



# %%% (AC, B)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, :, k].sum() * Contingency[:, j, :].sum() / n
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (I * K - 1) + (J - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)



# %%% (BC, A)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[:, j, k].sum() * Contingency[i, :, :].sum() / n
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (J * K - 1) + (I - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)




# %% Conditional Independence Test

# %%% (AB, AC)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, j, :].sum() * Contingency[i, :, k].sum() / Contingency[i, :, :].sum()
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (I - 1) + I * (J - 1) + I * (K - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)



# %%% (AB, BC)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, j, :].sum() * Contingency[:, j, k].sum() / Contingency[:, j, :].sum()
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (J - 1) + J * (I - 1) + J * (K - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)



# %%% (AC, BC)

# Perform Pearson's chi-squared test.
## Compute the chi-squared statistics.
X2 = 0
for i in range(I):
    for j in range(J):
        for k in range(K):
            ## Compute the expected counts.
            mu = Contingency[i, :, k].sum() * Contingency[:, j, k].sum() / Contingency[:, :, k].sum()
            
            X2 += ( Contingency[i, j, k] - mu )**2 / mu

## Compute the degree of freedom.
df = (I * J * K - 1) - ( (K - 1) + K * (I - 1) + K * (J - 1) )

## Compute the p-value.
1 - stat.chi2.cdf(X2, df)




# %% Homogeneous Association Test (AB, BC, AC)


numerator = 0
denominator = 0
statistics = 0
## Run the test.
for k in range(K):
    ## Compute log odds ratio.
    n_11k = Contingency[:, :, k][0, 0]
    n_22k = Contingency[:, :, k][1, 1]
    n_12k = Contingency[:, :, k][0, 1]
    n_21k = Contingency[:, :, k][1, 0]
    
    logOR_k = np.log( n_11k * n_22k / (n_12k * n_21k) )
    
    ## Compute the weight.
    w_k = 1 / ( 1/n_11k + 1/n_22k + 1/n_12k + 1/n_21k )
    
    ## Calculate the common odds ratio.
    numerator += w_k * logOR_k
    denominator += w_k
    
logOR_bar = numerator / denominator

for k in range(K):
    ## Compute the test statistics of Woolf's method.
    n_11k = Contingency[:, :, k][0, 0]
    n_22k = Contingency[:, :, k][1, 1]
    n_12k = Contingency[:, :, k][0, 1]
    n_21k = Contingency[:, :, k][1, 0]
    logOR_k = np.log( n_11k * n_22k / (n_12k * n_21k) )
    w_k = 1 / ( 1/n_11k + 1/n_22k + 1/n_12k + 1/n_21k )
    
    statistics += w_k * ( logOR_k - logOR_bar )**2
      
## Show the test statistics and p-values.
statistics
1 - stat.chi2.cdf(statistics, K-1)




# %% Saturated Model Test


