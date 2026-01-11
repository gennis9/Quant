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
import statsmodels.api as sm

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
Data['Time'] = [1] * 2 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4 + [6] * 4 + [7]

Data['Season'] = Data.index.str[1]
Data = pd.get_dummies(Data, columns=['Season'], prefix='Q')
Data.iloc[:, -4:] = Data.iloc[:, -4:].astype(int)
Data = Data.iloc[:, :-1]


# %% Forecast: 01: Ultra Low-End (<$100)

period = ['2024Q2', '2024Q3', '2024Q4',
          '2025Q1', '2025Q2', '2025Q3', '2025Q4',
          '2026Q1', '2026Q2', '2026Q3', '2026Q4']
Forecasts = pd.DataFrame(index = period)


result_1 = sm.OLS( Data.iloc[:, 0], sm.add_constant(Data.iloc[:, -5:]) ).fit()

result_1.summary()

forecast_1 = []

forecast_1.append( result_1.params[0] + result_1.params[2] * 7 + result_1.params[4] )
forecast_1.append( result_1.params[0] + result_1.params[2] * 7 + result_1.params[5] )
forecast_1.append( result_1.params[0] + result_1.params[2] * 7 )
forecast_1.append( result_1.params[0] + result_1.params[2] * 8 + result_1.params[3] )
forecast_1.append( result_1.params[0] + result_1.params[2] * 8 + result_1.params[4] )
forecast_1.append( result_1.params[0] + result_1.params[2] * 8 + result_1.params[5] )
forecast_1.append( result_1.params[0] + result_1.params[2] * 8 )
forecast_1.append( result_1.params[0] + result_1.params[2] * 9 + result_1.params[3] )
forecast_1.append( result_1.params[0] + result_1.params[2] * 9 + result_1.params[4] )
forecast_1.append( result_1.params[0] + result_1.params[2] * 9 + result_1.params[5] )
forecast_1.append( result_1.params[0] + result_1.params[2] * 9 )

Forecasts = pd.concat([Forecasts, pd.DataFrame(forecast_1, index = period)], axis = 1)




# %% Forecast: 02: Low-End ($100<$200)

result_2 = sm.OLS( Data.iloc[:, 1], sm.add_constant(Data.iloc[:, -5:]) ).fit()

result_2.summary()

forecast = []

forecast.append( result_2.params[0] + result_2.params[2] * 7 + result_2.params[4] )
forecast.append( result_2.params[0] + result_2.params[2] * 7 + result_2.params[5] )
forecast.append( result_2.params[0] + result_2.params[2] * 7 )
forecast.append( result_2.params[0] + result_2.params[2] * 8 + result_2.params[3] )
forecast.append( result_2.params[0] + result_2.params[2] * 8 + result_2.params[4] )
forecast.append( result_2.params[0] + result_2.params[2] * 8 + result_2.params[5] )
forecast.append( result_2.params[0] + result_2.params[2] * 8 )
forecast.append( result_2.params[0] + result_2.params[2] * 9 + result_2.params[3] )
forecast.append( result_2.params[0] + result_2.params[2] * 9 + result_2.params[4] )
forecast.append( result_2.params[0] + result_2.params[2] * 9 + result_2.params[5] )
forecast.append( result_2.params[0] + result_2.params[2] * 9 )

Forecasts = pd.concat([Forecasts, pd.DataFrame(forecast, index = period)], axis = 1)



# %% Forecast: 03: Mid-Range ($200<$400)

result = sm.OLS( Data.iloc[:, 2], sm.add_constant(Data.iloc[:, -5:]) ).fit()

result.summary()

forecast = []

forecast.append( result.params[0] + result.params[2] * 7 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 7 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 7 )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 8 )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 9 )

Forecasts = pd.concat([Forecasts, pd.DataFrame(forecast, index = period)], axis = 1)




# %% Forecast: 04: Mid- to High-End ($400<$600)

result = sm.OLS( Data.iloc[:, 3], sm.add_constant(Data.iloc[:, -5:]) ).fit()

result.summary()

forecast = []

forecast.append( result.params[0] + result.params[2] * 7 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 7 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 7 )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 8 )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 9 )

Forecasts = pd.concat([Forecasts, pd.DataFrame(forecast, index = period)], axis = 1)





# %% Forecast: 05: High-End ($600<$800)

result = sm.OLS( Data.iloc[:, 4], sm.add_constant(Data.iloc[:, -5:]) ).fit()

result.summary()

forecast = []

forecast.append( result.params[0] + result.params[2] * 7 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 7 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 7 )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 8 )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 9 )

Forecasts = pd.concat([Forecasts, pd.DataFrame(forecast, index = period)], axis = 1)






# %% Forecast: 06: Premium ($800<$1000)

result = sm.OLS( Data.iloc[:, 5], sm.add_constant(Data.iloc[:, -5:]) ).fit()

result.summary()

forecast = []

forecast.append( result.params[0] + result.params[2] * 7 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 7 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 7 )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 8 )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 9 )

Forecasts = pd.concat([Forecasts, pd.DataFrame(forecast, index = period)], axis = 1)





# %% Forecast: 07: High Premium ($1000<$1600)

result = sm.OLS( Data.iloc[:, 6], sm.add_constant(Data.iloc[:, -5:]) ).fit()

result.summary()

forecast = []

forecast.append( result.params[0] + result.params[2] * 7 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 7 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 7 )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 8 )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 9 )

Forecasts = pd.concat([Forecasts, pd.DataFrame(forecast, index = period)], axis = 1)






# %% Forecast: 08: Ultra Premium ($1600+)

result = sm.OLS( Data.iloc[:, 7], sm.add_constant(Data.iloc[:, -5:]) ).fit()

result.summary()

forecast = []

forecast.append( result.params[0] + result.params[2] * 7 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 7 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 7 )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 8 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 8 )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[3] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[4] )
forecast.append( result.params[0] + result.params[2] * 9 + result.params[5] )
forecast.append( result.params[0] + result.params[2] * 9 )

Forecasts = pd.concat([Forecasts, pd.DataFrame(forecast, index = period)], axis = 1)



Forecasts.columns = Data.columns[:8]

Forecasts = Forecasts.astype(int)


# %% Export

Full = pd.concat([Data.iloc[:, :8], Forecasts], axis = 0)

'''
_FileName = 'Mobilephone Market Size with Forecast.xlsx'
with pd.ExcelWriter(_FileName) as _writer:  
    Full.to_excel(_writer, sheet_name='Global')
'''















