" The following program is used to show various forecasting methods and takes stock return rates, real GDP, real consumption, and real investment of the US as examples. "


### Enviornment
from arch import arch_model
import ffn
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.tsa as tsa
from scipy.stats import norm, kstest
import yfinance as yf
import matplotlib.pyplot as plt
%matplotlib inline



### Autoregressive Moving Average (ARMA) Model
# Estimate the index return rate of S&P 500 by ARMA.

## Get the index data from Yahoo Finance.
spy = ffn.get("^GSPC", start = '2015-01-01', end = '2021-12-31')
## Sketch the index rate.
spy.plot.line(grid = True)

## Calculate the return rates of S&P 500 index.
spy["Return rate"] = spy['gspc'].pct_change()
spy.dropna(inplace = True)

## Run Augmented Dickey-Fuller unit root test.
# The test can be used to test for a unit root in a univariate process in the presence of serial correlation.
# The 1st element in the result is the test statistic (adf); the 2nd is p-value; the 3rd is number of lags used; the 4th is number of observations; the 5th dict is critical values for the test statistic at the 1 %, 5 %, and 10 % levels; the last one is the maximized information criterion if autolag is not None.
sm.tsa.stattools.adfuller(spy["Return rate"].dropna())

## Draw the histogram to show the return rates.
spy["Return rate"].plot.hist(density = True, bins = 100)

## Run Kolmogorov-Smirnov test for goodness of fit.
loc, scale = norm.fit(spy["Return rate"])
n = norm(loc = loc, scale = scale)
kstest(spy["Return rate"].dropna(), n.cdf)

## Plot the autocorrelation function. Plots lags on the horizontal and the correlations on vertical axis. The shaped area is the confidence interval (95% default) and the bar exceeded the shaped area represent autocorrelation significantly.
smg.tsaplots.plot_acf(spy["Return rate"], lags = 20)
## Plot the partial autocorrelation function.
smg.tsaplots.plot_pacf(spy["Return rate"], lags = 20)

## Compute information criteria for many ARMA models.
tsa.stattools.arma_order_select_ic(spy["Return rate"], ic = ['aic', 'bic'], trend = 'nc')

## Run ARMA(2, 2) model on the return rates and fit it using exact maximum likelihood via Kalman filter.
model = sm.tsa.ARMA(spy["Return rate"].dropna(), (2, 2)).fit(disp = False)
print(model.summary())

## Plot forecasts of the time-series data under the regressed result of ARMA model.
model.plot_predict(start = 1600, end = 1780)



### Generalized Autoregressive Conditional Heteroskedasticity (GARCH) Model
## Get the price data of S&P 500 from Yahoo Finance.
spy = yf.download("^gspc", start = "2020-01-01", end = "2021-12-31")
## Calculate the return rates of the index.
return_rates = spy["Close"].pct_change().dropna()

## Show the return rates and the squared return rates that are employed in GARCH model.
return_rates.plot(grid = True)
(return_rates ** 2).plot(grid = True)

## Run the GARCH(1, 1) model on the return rates and show the result table.
model = arch_model(return_rates * 100, vol = 'garch', p = 1, o = 0, q = 1, dist = 'Normal')
result = model.fit()
print(result.summary())

## Plot the standardized residuals and the conditional volatility of the GARCH model run.
fig = result.plot(annualize = 'D')
fig.set_size_inches(12, 6)

## Run the GARCH(2, 1) model on the return rates and show the result table.
model2 = arch_model(return_rates * 100, vol = 'garch', p = 2, o = 0, q = 1, dist = 'Normal')
result2 = model2.fit()
print(result2.summary())

## Plot the standardized residuals and the conditional volatility of GARCH(2, 1).
fig = result2.plot(annualize = 'D')
fig.set_size_inches(12, 6)

## Show the forecasted variance.
yhat = result2.forecast(horizon = 10)
plt.plot(yhat.variance.values[-1, :])



### Vector Autoregression (VAR)
## Download the macroeconomic data of the US from pandas database.
mdata = sm.datasets.macrodata.load_pandas().data
## Extract the time data.
dates = mdata[['year', 'quarter']].astype(int).astype(str)
## Generate quarter data.
quarterly = dates["year"] + "Q" + dates["quarter"]

## Convert the quarter data to datatime format.
quarterly = tsa.base.datetools.dates_from_str(quarterly)
## Extract the data of real GDP, real comsumption, and real investment.
mdata = mdata[['realgdp','realcons','realinv']]
## Employ the timestamp data as the index of the macroeconomic data to be investigated.
mdata.index = pd.DatetimeIndex(quarterly)

## Calculate the percentage rate of change of the macroeconomic data.
data = np.log(mdata).diff().dropna()

## Fit VAR(1) process and do lag order selection.
model = tsa.api.VAR(data)
results = model.fit(1)
results.summary()

## Plot the results of the macroeconomic data under the VAR(1) process.
results.plot()

## Choose the order(p) of the VAR model based on each of the available information criteria with the lowest scores attained and with maximum 15 lag orders.
# The available information criteria include AIC, BIC, FPE, and HQIC in this case.
result_all = model.select_order(15)
## Demonstrate the result.
print(result_all.summary())

## Choose the order(p) of the VAR model using the lowest Bayesian information criterion with maximum 15 lag orders.
results = model.fit(maxlags = 15, ic = "bic")

## Get the order of the VAR process.
lag_order = results.k_ar
## Produce linear minimum MSE forecasts for desired number of steps ahead, using the values with the order chosen by the lowest BIC score.
results.forecast(data.values[-lag_order:], 5)

## Plot the forecasting of the macrodata, real GDP, real comsumption, and real investment, with 10 periods.
results.plot_forecast(10)