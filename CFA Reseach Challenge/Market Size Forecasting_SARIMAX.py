# Copyright (C) 2024 Gennis
'''
Initiated Date    : 2024/09/04
Last Updated Date : 2024/09/08
Aim: Forecasting the market size of automobile.
'''



# %% Import
import os
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import statsmodels.tsa as tsa
import statsmodels.graphics.tsaplots as tsaplot
import statsmodels.tsa.arima.model

%matplotlib inline

# Set the working directory of the main folder.
Folder = r'D:\03Programs_Clouds\Google Drive\NSYSU\00CFARC\Modeling'
os.chdir(Folder)

# Import the market data.
Table = pd.read_excel('Inventory Forecasting.xlsx')




# %% Data Preprocess

Data = deepcopy(Table.T)
Data.columns = Data.iloc[0]
Data = Data.iloc[19:-1]

Data = Data.apply(pd.to_numeric, errors='coerce')

Data = Data.iloc[:, -8:]

Data['Covid'] = [0] * 6 + [1] * 12 + [0] * 5



# %% Forecast: 01: Ultra Low-End (<$100)

Forecasts = pd.DataFrame()
CIs = pd.DataFrame()


## Check stationarity.
tsa.stattools.adfuller(Data.iloc[:, 0].dropna())

plt.clf()
fig, ax=plt.subplots(2,1,figsize=(10,8))
tsaplot.plot_acf(Data.iloc[:, 0], lags = 10, ax=ax[0])
tsaplot.plot_pacf(Data.iloc[:, 0], lags = 10, ax=ax[1])
plt.show()


## Set hyperparameters.
arima_orders = []
for p in range(4):
    for q in range(4):
        arima_orders.append((p, 0, q))

seasonal_orders = []
for P in range(4):
    for D in range(2):
        for Q in range(4):
            seasonal_orders.append((P, D, Q, 4))


## Extract data for analysis.
df = Data.iloc[:, [0, -1] ]


## Run grid search.
col_orders, col_sorders, col_models, col_aic = [], [], [], []

for order in arima_orders:
    for sorder in seasonal_orders:
        try:
            model = tsa.statespace.sarimax.SARIMAX(df.iloc[:, 0], 
                                       exog = df.iloc[:, 1:],
                                       order = order,
                                       seasonal_order = sorder,
                                       enforce_invertibility=True, 
                                       enforce_stationarity=True
                                      ).fit(disp=False)        # full_output=False
    
            col_orders.append( order )
            col_sorders.append( sorder )
            col_models.append( model )
            col_aic.append( model.aic )
        
        except:
            continue


Grid = pd.DataFrame({'orders': col_orders,
                     'sorders': col_sorders,
                     'aic': col_aic
                    })


## Choose the model with min AIC.
Grid['aic'].min()

Grid.iloc[ Grid['aic'].idxmin() ]

model_final = col_models[ Grid['aic'].idxmin() ]

model_final.summary()

order = col_orders[ Grid['aic'].idxmin() ]
sorder = col_sorders[ Grid['aic'].idxmin() ]
# order = (2, 0, 3)
# sorder = (2, 1, 0, 4)

## Forecasting.
NumOfPeriod = 11
model_forecast = tsa.statespace.sarimax.SARIMAX(df.iloc[:, 0], 
                           exog = df.iloc[:, 1:],
                           order = order,
                           seasonal_order = sorder,
                           enforce_invertibility=True, 
                           enforce_stationarity=True
                          ).fit(disp=False)        # full_output=False


# prediction = model_forecast.get_prediction()

pred_values_1 = model_forecast.get_forecast(steps = NumOfPeriod, exog = [0] * NumOfPeriod).predicted_mean

CI_1 = model_forecast.get_forecast(steps = NumOfPeriod, exog = [0] * NumOfPeriod).conf_int()


## Collect predicted results.
Forecasts = pd.concat([Forecasts, pred_values_1], axis = 1)
CIs = pd.concat([CIs, CI_1], axis = 1)



# %% Forecast: 02: Low-End ($100<$200)

## Check stationarity.
tsa.stattools.adfuller(Data.iloc[:, 1])

diff_y = Data.iloc[:, 1].diff(1).dropna()
tsa.stattools.adfuller(diff_y)


plt.clf()
fig, ax=plt.subplots(2,1,figsize=(10,8))
tsaplot.plot_acf(diff_y, lags = 10, ax=ax[0])
tsaplot.plot_pacf(diff_y, lags = 10, ax=ax[1])
plt.show()


## Set hyperparameters.
arima_orders = []
for p in range(4):
    for q in range(4):
        arima_orders.append((p, 1, q))

seasonal_orders = []
for P in range(4):
    for D in range(2):
        for Q in range(4):
            seasonal_orders.append((P, D, Q, 4))


## Extract data for analysis.
df = Data.iloc[:, [1, -1] ]


## Run grid search.
col_orders, col_sorders, col_models, col_aic = [], [], [], []

for order in arima_orders:
    for sorder in seasonal_orders:
        try:
            model = tsa.statespace.sarimax.SARIMAX(df.iloc[:, 0], 
                                       exog = df.iloc[:, 1:],
                                       order = order,
                                       seasonal_order = sorder,
                                       enforce_invertibility=True, 
                                       enforce_stationarity=True
                                      ).fit(disp=False)        # full_output=False
    
            col_orders.append( order )
            col_sorders.append( sorder )
            col_models.append( model )
            col_aic.append( model.aic )
        
        except:
            continue


Grid = pd.DataFrame({'orders': col_orders,
                     'sorders': col_sorders,
                     'aic': col_aic
                    })


## Choose the model with min AIC.
Grid['aic'].min()

Grid.iloc[ Grid['aic'].idxmin() ]

model_final = col_models[ Grid['aic'].idxmin() ]

model_final.summary()

order = col_orders[ Grid['aic'].idxmin() ]
sorder = col_sorders[ Grid['aic'].idxmin() ]
# order = (2, 1, 2)
# sorder = (2, 1, 3, 4)

## Forecasting.
NumOfPeriod = 11
model_forecast = tsa.statespace.sarimax.SARIMAX(df.iloc[:, 0], 
                           exog = df.iloc[:, 1:],
                           order = order,
                           seasonal_order = sorder,
                           enforce_invertibility=True, 
                           enforce_stationarity=True
                          ).fit(disp=False)        # full_output=False


# prediction = model_forecast.get_prediction()

pred_values_2 = model_forecast.get_forecast(steps = NumOfPeriod, exog = [0] * NumOfPeriod).predicted_mean

CI_2 = model_forecast.get_forecast(steps = NumOfPeriod, exog = [0] * NumOfPeriod).conf_int()


## Collect predicted results.
Forecasts = pd.concat([Forecasts, pred_values_2], axis = 1)
CIs = pd.concat([CIs, CI_2], axis = 1)




# %% Forecast: 03: Mid-Range ($200<$400)






# %% Forecast: 04: Mid- to High-End ($400<$600)







# %% Forecast: 05: High-End ($600<$800)


# %% Forecast: 06: Premium ($800<$1000)



# %% Forecast: 07: High Premium ($1000<$1600)

# %% Forecast: 08: Ultra Premium ($1600+)

































