'''
Initiated Date    : 2024/03/29
Last Updated Date : 2024/04/12
Aim: Comupter the MLEs of different time series model.
Input Data: KOSPI and USD/JPD exchange rate.
'''

# %% Enviornment

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.optimize as sco


## Set the path of data
_Path = r'D:\03Programs_Clouds\Google Drive\NSYSU\02Fin DA and Trading\Cpu Workshop 1'
os.chdir(_Path)

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)




# %% Data Access
"                               ..... Data Access .....                             "

_FileName = 'KOSPI.xlsx'
KOSPI = pd.read_excel(_FileName)

_FileName = 'Data_CWS1_JPYUSD.xlsx'
Ex_rate = pd.read_excel(_FileName)




# %% Task 1: Simulation of Annualized Volatility through AR(1)

## Set parameters.
sd_annual = 0.1
days = 252
phi = 0.9839        # AR parameter
var_epsilon = 0.0027
T = 500

seed = 2024

## Calculate the parameters.
vol_annual = sd_annual ** 2
vol_daily = vol_annual / days
sd_daily = np.sqrt(vol_daily)

## Generate white noise.
np.random.seed(seed)
Epsilons = np.random.normal(0, np.sqrt(var_epsilon), T)


## Set the initial values of the simulation.
alpha = np.log(sd_daily)

# ln(sigma_{t-1})
ln_sigma_last = alpha
SDs_daily = [sd_daily]

for t in range(1, T):
    ## Run AR(1) process to generate the ln(sigma_t).
    ln_sigma = phi * (ln_sigma_last - alpha) + Epsilons[t] + alpha
    
    ## Save the daily volatility generated.
    SDs_daily.append(np.exp(ln_sigma))
    
    ## Save current daily volatility as the one of last day for next round simulation.
    ln_sigma_last = ln_sigma

## Calculate the annualized volatility.
SDs_annual = [ sd * np.sqrt(days) for sd in SDs_daily ]

# Plot the graph for annualized volatility.

## Create x-axis values.
x_values = range(T)

## Plot the line plot.
plt.plot(x_values, SDs_annual)

plt.xlabel('The Number of Days')
plt.ylabel('Annualized Volatility')
plt.title('Annualized Volatility Simulated by AR(1)')

plt.show()




# %% Task 2: MA(1) Simulation

## Set parameters.
days = 252
theta = 0.8        # MA parameter
var_epsilon = 1
T = 1000
mu = 0
seed = 2024


## Generate white noise.
np.random.seed(seed)
Epsilons = np.random.normal(0, np.sqrt(var_epsilon), T+1)

# ln(WN_{t-1})
Epsilons_last = Epsilons[1:] * theta

## Run MA(1) process to generate the X_t.
X = [ Epsilons[t] + Epsilons_last[t] for t in range(T) ]

## Generate the white noise with the same variance as X_t.
WN = np.random.normal(0, np.sqrt( np.var(X) ), T)

## Compute the variances of the sequences.
np.var(Epsilons) * T / (T-1)
np.var(X)        * (T-1) / (T-2)
np.var(WN)       * (T-1) / (T-2)


## Plot the X_t and the white noise with the same variance.
fig, axs = plt.subplots(2)

axs[0].plot(X)
axs[0].set_title(r'$X_t$')

axs[1].plot(WN)
axs[1].set_title('White Noise')

plt.tight_layout()
plt.show()




# %% Task 3: Autocorrelations of Exchange Rate and of KOSPI index

## Set parameters.


# Run exchange rate.
data = Ex_rate

## Calculate the parameters.
T = len(data)

Return_t = [ np.log(data.iloc[t+1, 1] / data.iloc[t, 1]) for t in range(T-1) ]


## Compute autocorrelations.
lags = 10
Autocorrs = [ np.corrcoef(Return_t[lag:], Return_t[:-lag])[0, 1] for lag in range(1, lags) ]



## Plot the scatter plot for the lagged returns.
lag = 1
plt.scatter(Return_t[lag:], Return_t[:-lag])

plt.xlabel('Current Returns')
plt.ylabel('Lagged Returns')
plt.title('Scatter Plot between Current and Lagged Returns')
plt.grid(True, which='both', linestyle='--')
plt.show()

# Absolute returns.
plt.scatter([ abs(r) for r in Return_t[lag:] ], [ abs(r) for r in Return_t[:-lag] ])

plt.xlabel('Current Absolute Returns')
plt.ylabel('Lagged Absolute Returns')
plt.title('Scatter Plot between Current and Lagged Absolute Returns')
plt.grid(True, which='both', linestyle='--')
plt.show()




# %% Task 4: Find MLE of the Parameters of ARCH(1)

## Set parameters.
alpha = 0.5


## Compute the daily return of KOSPI.
Daily_r = KOSPI.set_index('Date')
Daily_r = Daily_r.pct_change().iloc[1:]
Daily_r.columns = ['R_t']


## Calculate the parameters.
mean = Daily_r.mean()[0]
var = Daily_r.var()[0]
mu = mean
omega = var*(1-alpha)
h_1 = var


## Compute h_t and z_t daily.
h_t = omega + alpha * (Daily_r - mu)**2
h_t = pd.Series(h_t.iloc[:, 0]).tolist()
h_t = [h_1] + h_t[:-1]
Daily_r['h_t'] = h_t

Daily_r['z_t'] = (Daily_r['R_t'] - mu) / np.sqrt(Daily_r['h_t'])


## Compute likelihood.
l_t = -0.5 * ( np.log(Daily_r['h_t']) + Daily_r['z_t'] ** 2 )
Daily_r['l_t'] = l_t

# The log likelihood.
obj_MLE = l_t.sum()


## Solve for the MLE.
def Func_Likelihood(params, data):
    ## Set the values of the parameter set.
    mu, alpha, omega, h_1 = params

    ## Compute h_t and z_t daily.
    h_t = omega + alpha * (data - mu)**2
    h_t = pd.Series(h_t.iloc[:, 0]).tolist()
    h_t = [h_1] + h_t[:-1]
    data['h_t'] = h_t

    data['z_t'] = (data['R_t'] - mu) / np.sqrt(data['h_t'])

    ## Compute likelihood.
    l_t = -0.5 * ( np.log(data['h_t']) + data['z_t'] ** 2 )
    
    return -l_t.sum()


## Compute the daily return of KOSPI.
data = KOSPI.set_index('Date')
data = data.pct_change().iloc[1:]
data.columns = ['R_t']

## Set initial guess for parameters.
initial_guess = [mean, alpha, var*(1-alpha), var]

## Give the bounds for parameters.
bounds = [(None, None), (0, 1), (0, None), (0, None)]

## Maximize the negative log likelihood function.
result = sco.minimize(Func_Likelihood, initial_guess, args=(data,), bounds=bounds)

## Extract the MLE.
mu_opt, alpha_opt, omega_opt, h_1_opt = result.x
print("Optimized mu:", mu_opt)
print("Optimized alpha:", alpha_opt)
print("Optimized omega:", omega_opt)
print("Optimized h_1:", h_1_opt)

## Show the likelihood maximized under MLE.
print("Optimized likehihood:", -Func_Likelihood(result.x, data))




# %% Task 5: Find MLE of the Parameters of GARCH(1, 1)

## Set parameters.
beta = 0.8
alpha = 0.1
omega = var * (1- alpha - beta)


## Compute the daily return of KOSPI.
Daily_r_2 = KOSPI.set_index('Date')
Daily_r_2 = Daily_r_2.pct_change().iloc[1:]
Daily_r_2.columns = ['R_t']


## Compute h_t and z_t daily.
h_t = [h_1]
_last = h_1
for _t in range(len(Daily_r_2) - 1):
    _next = omega + alpha * (Daily_r_2.iloc[_t][0] - mu)**2 + beta * _last
    h_t.append(_next)
    
    _last = _next

Daily_r_2['h_t'] = h_t

Daily_r_2['z_t'] = (Daily_r_2['R_t'] - mu) / np.sqrt(Daily_r_2['h_t'])


## Compute likelihood.
l_t = -0.5 * ( np.log(Daily_r_2['h_t']) + Daily_r_2['z_t'] ** 2 )
Daily_r_2['l_t'] = l_t

# The log likelihood.
obj_MLE = l_t.sum()



## Solve for the MLE.
def Func_Likelihood2(params, data):
    ## Set the values of the parameter set.
    mu, alpha, beta, omega, h_1 = params

    ## Compute h_t and z_t daily.
    h_t = [h_1]
    _last = h_1
    for _t in range(len(data) - 1):
        _next = omega + alpha * (data.iloc[_t][0] - mu)**2 + beta * _last
        h_t.append(_next)
        
        _last = _next

    data['h_t'] = h_t

    data['z_t'] = (data['R_t'] - mu) / np.sqrt(data['h_t'])

    ## Compute likelihood.
    l_t = -0.5 * ( np.log(data['h_t']) + data['z_t'] ** 2 )
    
    return -l_t.sum()


## Compute the daily return of KOSPI.
data = KOSPI.set_index('Date')
data = data.pct_change().iloc[1:]
data.columns = ['R_t']

## Set initial guess for parameters.
initial_guess = [mean, alpha, beta, var*(1-alpha), var]

## Give the bounds for parameters.
bounds = [(None, None), (0, 1), (0, 1), (0, None), (0, None)]

# Constraint: alpha + beta < 1
def Func_Constraint(params):
    return 1 - (params[1] + params[2])

con = {'type': 'ineq', 'fun': Func_Constraint}

## Maximize the negative log likelihood function.
result = sco.minimize(Func_Likelihood2, initial_guess, args=(data,), bounds=bounds)
result = sco.minimize(Func_Likelihood2, initial_guess, args=(data,), bounds=bounds, constraints=con)

## Extract the MLE.
mu_opt, alpha_opt, beta_opt, omega_opt, h_1_opt = result.x
print("Optimized mu:", mu_opt)
print("Optimized alpha:", alpha_opt)
print("Optimized beta:", beta_opt)
print("Optimized omega:", omega_opt)
print("Optimized h_1:", h_1_opt)

## Show the likelihood maximized under MLE.
print("Optimized likehihood:", -Func_Likelihood2(result.x, data))


## Renew h_t with the MLE.
h_t = [h_1_opt]
_last = h_1_opt
for _t in range(len(Daily_r_2) - 1):
    _next = omega_opt + alpha_opt * (Daily_r_2.iloc[_t][0] - mu_opt)**2 + beta_opt * _last
    h_t.append(_next)
    
    _last = _next

Daily_r_2['h_t_renew'] = h_t

## Compute the annualized standard deviation.
Daily_r_2['SD'] = np.sqrt( Daily_r_2['h_t'] * 252 )
Daily_r_2['SD_renew'] = np.sqrt( Daily_r_2['h_t_renew'] * 252 )


## Plot the graph for the annualized standard deviation.
plt.figure(figsize=(8, 5))
# plt.plot(Daily_r_2.index, Daily_r_2['SD'] * 100, color='red', linewidth=2, label='Annualized SD')
plt.plot(Daily_r_2.index, Daily_r_2['SD_renew'] * 100, color='blue', linewidth=2, label='Annualized SD_MLE')
plt.title('Annualized Standard Deviation of KOSPI')
plt.xlabel('Date')
plt.ylabel('Standard Deviation (%)')
plt.legend()
plt.grid(True)
plt.show()





































































