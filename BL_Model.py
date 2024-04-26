" The following program is used to show the computation of the weighted average of reversed optimized and Markowitz optimized stock weights of the HK stocks selected by Black-Litterman model. " 


### Enviornment
import cvxopt as opt
from cvxopt import solvers
import ffn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import scipy.optimize as sco
%matplotlib inline


## Set default parameters.
# Choose the stock candidates and the time period of stock prices that used in MVO.
Stocks = '0868.hk, 0992.hk, 0027.hk, 2020.hk, 1038.hk'
_startDate = '2010-01-01'
_endDate   = '2022-09-30'
## How many different sets of assets to be shown as Markowitz mean-variance analysis of portfolios.
portfNum = 10000
## Specify the annual risk free rate.
rf = 0.01
## Input the numbers of outstanding market shares of market portfolio (in billions).
mktShares = np.array([4.02, 12.13, 4.36, 2.71, 2.52])



### Data Access
## Get the stock prices from Yahoo Finance.
Data = ffn.get(Stocks, start = _startDate, end = _endDate)
## Extract the size of panel data downloaded.
_days = Data.shape[0]



### Data Cleaning
## Remove the missing values while keeping the balance panel.
ReturnRates = Data.pct_change().dropna()
test = ((1 + ReturnRates).cumprod() ** (1 / _days))[-1:] ** 252 - 1 - rf


### Porfolios Generation
## Create function for generating a stock portfolio with random weight and give out the basic parameters for MVO.
def Func_GenOnePortf(dailyReturn):
    ## Generate a random weight for stock portfolio.
    weight = np.random.random(dailyReturn.shape[1])
    weight /= np.sum(weight)               # normalization
    
    ## Calculate the average annual excess return rate of the portfolio generated.
    annualReturn = ((1 + dailyReturn).cumprod() ** (1 / _days))[-1:] ** 252 - 1
    excessReturn = annualReturn - rf
    mu = excessReturn.dot(weight).tolist()[0]
    
    ## Caculate the covariance matrix and annualized variance of the daily excess returns.
    covar = dailyReturn.cov()
    sigma2 = weight.T.dot(covar.dot(weight)) * 252
    
    return mu, sigma2 ** 0.5

## Generate 5000 means and standard deviations of the corresponding 5000 allocations of assets (portfolios).
_means, _sds = np.column_stack([Func_GenOnePortf(ReturnRates) for i in range(portfNum)])


## Sketch the 5000 sets of portfolios.
def Func_SketchPortfs(Sharpe = False):
    plt.figure(figsize = (12, 6))
    plt.scatter(_sds * 100, _means * 100, c = _means / _sds, marker = 'o')
    plt.colorbar(label = 'Sharpe ratio')
    plt.xlabel('Standard deviation (%)')
    plt.ylabel('Excess return rate (%)')
    plt.title('Mean-Variance Analysis of Portfolios')
    plt.grid()

    if Sharpe == True:
        ## Find the portfolio with the highest Sharpe ratio along the 5000 sets of portfolios.
        maxSharpe_idx = (_means/_sds).argmax()
        ## Mark the portfolio with the highest Sharpe ratio on the graph.
        plt.scatter(_sds[maxSharpe_idx] * 100, _means[maxSharpe_idx] * 100, c = 'r', marker = 'x')
    plt.show()

Func_SketchPortfs()



### Efficient Frontier
def Func_OptPortf(r):
    solvers.options['show_progress'] = False

    assetNum = r.shape[1]         # Num of assets in each portfolio.
    riskLvs = 30                 # Divide how many different levels of risk willing to take (risk aversion).
    riskAversionLv = [10 ** (2 * i / riskLvs - 2) for i in range(riskLvs)]
    
    # Calculate the parameters of convex optimization.
    q = opt.matrix((((1 + r).cumprod() ** (1 / _days))[-1:] ** 252 - 1 - rf).values.T)
    P = opt.matrix(r.cov().values * 252)
    
    G = opt.matrix(-np.eye(assetNum))
    h = opt.matrix(0.0, (assetNum, 1))
    A = opt.matrix(1.0, (1, assetNum))
    b = opt.matrix(1.0)
    
    ## Calculate the optimized weights of the allocation of assets in the portfolio where maximize {E(r) - 1/2 λ Var(r)} under different levels of risk taken (i.e., diff λ).
    # To solve the maximization problem, the objective function takes negative so that it can be solved under convex optimization.
    weights = [solvers.qp(2 * P, -level * q, G, h, A, b)['x'] for level in riskAversionLv]
    
    ## Calculate the return rates and risk levels of the optimized allocation of assets under different risk aversion levels.
    returns = np.array([np.dot(q.T, x) * 100 for x in weights])
    vols = np.array([np.sqrt(np.dot(x.T, P * x)) * 100 for x in weights])

    # Find the weights and index w.r.t. the greatest Sharpe ratio.
    idx = (returns / vols).argmax()
    optwt =  weights[idx]
    
    return idx, optwt, returns, vols

OptIdx, OptWeight, Returns, Risks = Func_OptPortf(ReturnRates)

## Find the starting index of efficient frontier that rules out the sub-optimal frontier.
_eIdx = np.argmin(Risks)
_eVols = Risks[_eIdx:]
_eReturns = Returns[_eIdx:]
## Calculate the spline (interpolated curve) parameters based on the optimized weights against different risk aversion levels.
_tck = sci.splrep(_eVols, _eReturns)

## Find the y-coordinate(s) against input x on the spline (interpolated curve).
def Func_F(x):
    return sci.splev(x, _tck, der = 0)

## Find the y-coordinate(s) of 1st derviative of the spline against input x.
def Func_dF(x):
    return sci.splev(x, _tck, der = 1)

## Solve the simultaneous equations to find (1) the slope of the capital market line, and (2) x-coordinate of the tangency portfolio with risk-free rate.
def Func_Equations(f, rf = rf):
    eq1 = rf + f[0] * f[1] - Func_F(f[1])     # 0 = c + mx - y
    eq2 = f[0] - Func_dF(f[1])                # 0 = m - y'
    return eq1, eq2

Optimum = sco.fsolve(Func_Equations, [0.5, 3])     # initial values for iteration to finding roots.



### Black-Litterman Model
## Calculate the weights of the market portfolio.
_weight_mkt = Data.iloc[-1] * mktShares
_weight_mkt /= np.sum(_weight_mkt)

## Calculate the mean annual excess market return.
_exReturn = ((1 + ReturnRates).cumprod() ** (1 / _days))[-1:] ** 252 - 1 - rf
_return_mkt = _exReturn.dot(_weight_mkt).tolist()[0]

## Caculate the covariance matrix and annualized variance of the daily excess returns.
_covar = ReturnRates.cov()
_var_mkt = _weight_mkt.T.dot(_covar.dot(_weight_mkt)) * 252

## Derive "view" which would induce the market portfolio.
# The risk aversion parameter.
_riskAdv = _return_mkt / _var_mkt
_mean_mkt = _riskAdv * _covar.dot(_weight_mkt)


## Calculates the Black-Litterman mixture mean excess return and covariance matrix.
def Func_BL(tau, mu_bl, mu_mvo, cov_mvo):
    covInv_bl = np.linalg.inv(tau * cov_mvo)
    covInv_mvo = np.linalg.inv(cov_mvo)
    
    mu_tilde = np.linalg.solve( covInv_mvo + covInv_bl, covInv_mvo.dot(mu_bl) + covInv_bl.dot(mu_mvo) )
    
    return mu_tilde

## Set the mixing parameter to choose how heavily to express the "view".
tau = 600
_mean_bl = Func_BL(tau, _mean_mkt.values.reshape(5, 1), np.array(_exReturn).reshape(5, 1) * OptWeight, _covar * 252)

## Compute the portfolio weights recommended by the Black-Litterman model.
Weight_bl = np.linalg.solve(_riskAdv * _covar * 252, _mean_bl)
Weight_bl /= np.sum(Weight_bl)

Return_bl = _exReturn.dot(Weight_bl)[0].tolist()[0]
Risk_bl = np.sqrt(Weight_bl.T.dot(_covar.dot(Weight_bl)) * 252).tolist()[0][0]


## Draw the efficient frontier, optimal portfolios, and capital market line.
def Func_SketchOptPortf(cml = True):
    plt.figure(figsize = (12, 6))
    plt.scatter(_sds * 100, _means * 100, c = _means / _sds, marker = 'o')
    plt.colorbar(label = 'Sharpe ratio')
    plt.plot(Risks[:, 0, 0], Returns[:, 0, 0], 'c-o')
    plt.xlabel('Standard deviation (%)')
    plt.ylabel('Excess return rate (%)')
    plt.title('Portfolios under Black-Litterman Model and MVO')
    
    plt.scatter(Risks[OptIdx], Returns[OptIdx], c = 'r', marker = 'x', s = 200, label = 'MVO Optimized Portfolio')
    plt.scatter(_var_mkt**0.5 * 100, _return_mkt * 100, c = 'm', marker = '*', s = 200, label = 'Market Portfolio')
    plt.scatter(Risk_bl * 100, Return_bl * 100, c = 'b', marker = '^', s = 200, label = 'Black-Litterman Optimized Portfolio')
    plt.legend(loc = 'best')
    
    if cml == True:
        plt.plot([0, Risk_bl * 100], [rf * 100, Return_bl * 100], 'b--')
        plt.grid()
        plt.xlim(0, _sds.max()*100+1)
    plt.grid()
    plt.show()

Func_SketchOptPortf(cml=False)



### Result Visualization
## Use pie chart to show the optimal weights of the constituent stocks.
plt.figure(figsize = (8, 8))
plt.subplot(211)
plt.pie(list(OptWeight), labels = ReturnRates.columns.values, autopct = '%1.2f%%', shadow=True)
plt.title('Ingredient of Portfolio under MVO')

plt.subplot(212)
plt.pie(list(Weight_bl.reshape(5)), labels = ReturnRates.columns.values, autopct = '%1.2f%%', shadow=True)
plt.title('Ingredient of Portfolio under BL model')
plt.show()
## Show the optimal weights of the constituent stocks.
print(Weight_bl.reshape(5))
