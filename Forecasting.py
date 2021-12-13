from arch import arch_model
import ffn
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.graphics as smg
import statsmodels.tsa as tsa
from scipy.stats import norm, kstest
import yfinance as yf



## Another example of ARMA - S&P500
spy = ffn.get("^GSPC", start = '2017-01-01', end = '2020-12-31')
spy.plot.line(grid = True)

sm.tsa.stattools.adfuller(spy["gspc"].dropna()) # return values: test statistic, p-value, ...

spy["Return rate"] = spy.pct_change()
spy.dropna(inplace = True)
sm.tsa.stattools.adfuller(spy["Return rate"].dropna()) # return values: test statistic, p-value, ...

spy["Return rate"].plot.hist(density = True, bins = 100)

loc, scale = norm.fit(spy["Return rate"])
n = norm(loc = loc, scale = scale)
kstest(spy["Return rate"].dropna(), n.cdf)

loc, scale = norm.fit(spy["Return rate"])
n = norm(loc = loc, scale = scale)
kstest(spy["Return rate"].dropna(), n.cdf)

smg.tsaplots.plot_acf(spy["Return rate"], lags = 20)
smg.tsaplots.plot_pacf(spy["Return rate"], lags = 20)

tsa.stattools.arma_order_select_ic(spy["Return rate"], ic = ['aic', 'bic'], trend = 'nc')

model = sm.tsa.ARMA(spy["Return rate"].dropna(), (2, 2)).fit(disp = False)
print(model.summary())

model.plot_predict(start = 900, end = 950)


#### Generalized Autoregressive Conditional Heteroskedasticity (GARCH) Model
spy = yf.download("^gspc", start = "2020-01-01", end = "2020-12-31") # could be used to collect monthly data or alike.
return_rates = spy["Close"].pct_change().dropna()

return_rates.plot(grid = True)
(return_rates ** 2).plot(grid = True)

model = arch_model(return_rates * 100, vol = 'garch', p = 1, o = 0, q = 1, dist = 'Normal')
result = model.fit()
print(result.summary())

fig = result.plot(annualize = 'D')
# fig.set_size_inches(12, 6)


model2 = arch_model(return_rates * 100, vol = 'garch', p = 2, o = 0, q = 1, dist = 'Normal')
result2 = model2.fit()
print(result2.summary())

fig = result2.plot(annualize = 'D')


#### Vector Autoregression (VAR)
mdata = sm.datasets.macrodata.load_pandas().data
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]

quarterly = tsa.base.datetools.dates_from_str(quarterly)
mdata = mdata[['realgdp','realcons','realinv']]
mdata.index = pd.DatetimeIndex(quarterly)

data = np.log(mdata).diff().dropna()

model = tsa.api.VAR(data)
results = model.fit(1)
results.summary()

results.plot()

# results.plot_acorr()

result_all = model.select_order(15)
print(result_all.summary())

results = model.fit(maxlags = 15, ic = "bic")

lag_order = results.k_ar
results.forecast(data.values[-lag_order:], 5)

results.plot_forecast(10)

