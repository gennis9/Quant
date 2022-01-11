" The following program is used to show the computation of the Markowitz optimized stock weights of the HK stocks selected by mean-variance optimization (MVO). "


### Enviornment
import cvxopt as opt
from cvxopt import solvers
import ffn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import scipy.optimize as sco
import seaborn as sns
%matplotlib inline



### Data Access
## Get the stock prices from Yahoo Finance.
data = ffn.get('0005.hk, 0016.hk, 0027.hk, 0386.hk, 1038.hk', start = '2015-01-01', end='2021-12-31')
## Extract the size of panel data downloaded.
days = data.shape[0]
assets = data.shape[1]



### Data Cleaning
## Remove the missing values while keeping the balance panel.
return_rates = data.pct_change().dropna()



### Exploratory Data Analysis
## Sketch the price fluctuation and have a glance on the volatility of prices. 
ax = return_rates.apply(lambda x : 100 * x).plot.line(grid = True, figsize = (12, 6))
ax.set_ylabel('Return rate (%)')


## Use heatmap to show the correlation matrix among stock returns, which is a critical factor toward MVO.
plt.figure(figsize=(10,5))
sns.heatmap(return_rates.corr(),cmap='BrBG',annot=True)



### Porfolios Generation
## Create function for generating a stock portfolio with random weight and give out the basic parameters for MVO.
def Func_GenOnePortf(daily_return_rates):
    ## Generate a random weight for stock portfolio.
    weight = np.random.random(daily_return_rates.shape[1])
    weight /= np.sum(weight)               # normalization
    
    ## Calculate the average annual return rate of the portfolio generated.
    annual_ret = ((1 + daily_return_rates).cumprod() ** (1 / days))[-1:] ** 252 - 1
    mu = annual_ret.dot(weight).tolist()[0]
    
    ## Caculate the covariance matrix and annualized variance of the daily returns.
    C = daily_return_rates.cov()
    sigma2 = weight.T.dot(C.dot(weight)) * 252
    
    return weight, mu, sigma2

## How many different sets of assets to be shown as Markowitz mean-variance analysis of portfolios.
n_portfolios = 5000
## Generate 5000 weights, means, and vars of the corresponding 5000 allocations of assets (portfolios).
weights, means, sig2s = np.column_stack([Func_GenOnePortf(return_rates) for _ in range(n_portfolios)])
stds = sig2s ** 0.5


## Sketch the 5000 sets of portfolios.
def Func_SketchPortfs(sharpe = False):
    plt.figure(figsize = (12, 6))
    plt.scatter(stds * 100, means * 100, c = means / stds, marker = 'o')
    plt.colorbar(label = 'Sharpe ratio')
    plt.xlabel('Standard deviation (%)')
    plt.ylabel('Return rate (%)')
    plt.title('Mean-Variance Analysis of Portfolios')
    plt.grid()
    if sharpe == True:
        ## Find the portfolio with the highest Sharpe ratio along the 5000 sets of portfolios.
        maxSharpe_idx = (means/stds).argmax()
        ## Mark the portfolio with the highest Sharpe ratio on the graph.
        plt.scatter(stds[maxSharpe_idx] * 100, means[maxSharpe_idx] * 100, c = 'r', marker = 'x')
    plt.show()

Func_SketchPortfs()



### Efficient Frontier
def Func_OptPortf(r):
    solvers.options['show_progress'] = False

    n = r.shape[1]         # Num of assets in each portfolio.
    N = 50                 # Divide how many different levels of risk willing to take (risk aversion).
    risk_aversion_levels = [10 ** (2 * t / N - 2) for t in range(N)]
    
    # Calculate the parameters of convex optimization.
    p = opt.matrix((((1 + r).cumprod() ** (1 / days))[-1:] ** 252 - 1).values.T)
    S = opt.matrix(r.cov().values * 252)
    
    G = opt.matrix(-np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    ## Calculate the optimized weights of the allocation of assets in the portfolio where maximize {E(r) - 1/2 λ Var(r)} under different levels of risk taken (i.e., diff λ).
    # To solve the maximization problem, the objective function takes negative so that it can be solved under convex optimization.
    weights = [solvers.qp(2 * S, -level * p, G, h, A, b)['x'] for level in risk_aversion_levels]
    
    ## Calculate the return rates and risk levels of the optimized allocation of assets under different risk aversion levels.
    returns = np.array([np.dot(p.T, x) * 100 for x in weights])
    vols = np.array([np.sqrt(np.dot(x.T, S * x)) * 100 for x in weights])

    # Find the weights and index w.r.t. the greatest Sharpe ratio.
    idx = (returns / vols).argmax()
    wt =  weights[idx]
    
    return idx, wt, returns, vols

opt_idx, opt_weight, returns, risks = Func_OptPortf(return_rates)

## Find the starting index of efficient frontier that rules out the sub-optimal frontier.
ind = np.argmin(risks)
evols = risks[ind:]
erets = returns[ind:]
## Calculate the spline (interpolated curve) parameters based on the optimized weights against different risk aversion levels.
tck = sci.splrep(evols, erets)

## Find the y-coordinate(s) against input x on the spline (interpolated curve).
def Func_F(x):
    return sci.splev(x, tck, der = 0)

## Find the y-coordinate(s) of 1st derviative of the spline against input x.
def Func_dF(x):
    return sci.splev(x, tck, der = 1)

## Solve the simultaneous equations to find (1) the vertical intercept and the slope of the capital market line, and (2) x-coordinate of the tangency portfolio with risk-free rate equal to 1%.
def Func_Equations(p, rf = 0.01):
    eq1 = rf - p[0]                      # 0 = rf - c
    eq2 = rf + p[1] * p[2] - Func_F(p[2])     # 0 = c + mx - y
    eq3 = p[1] - Func_dF(p[2])                # 0 = m - y'
    return eq1, eq2, eq3

opt = sco.fsolve(Func_Equations, [0.01, 0.5, 3])     # initial values for iteration to finding roots.
print(opt) # opt[0]: risk-free rate, opt[1]: maximum Sharpe ratio, opt[2]: risk level of tangency portfolio.


## Draw the efficient frontier, optimal portfolio, and capital market line.
def Func_SketchOptPortf(cml = True):
    plt.figure(figsize = (12, 6))
    plt.scatter(risks[opt_idx], returns[opt_idx], c = 'r', marker = 'x', s = 100)
    plt.legend(['Optimal Portfolio'], loc = 'best')
    plt.scatter(stds * 100, means * 100, c = means / stds, marker = 'o')
    plt.colorbar(label = 'Sharpe ratio')
    plt.plot(risks[:, 0, 0], returns[:, 0, 0], 'y-o')
    plt.xlabel('Standard deviation (%)')
    plt.ylabel('Mean (%)')
    plt.title('Mean-Variance Analysis of Portfolios')
    if cml == True:
        plt.plot([0, risks[opt_idx]], [1, returns[opt_idx]], 'r--')
        plt.grid()
        plt.xlim(0, stds.max()*100+1)
    plt.show()

Func_SketchOptPortf(cml=True)



### Result Visualization
## Use pie chart to show the optimal weights of the constituent stocks.
plt.figure(figsize = (8, 8))
plt.pie(list(opt_weight), labels = return_rates.columns.values, autopct = '%1.2f%%', shadow=True)
plt.title('Ingredient of Portfolio')
plt.show()
## Show the optimal weights of the constituent stocks.
print(opt_weight)