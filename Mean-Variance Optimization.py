import cvxopt as opt
from cvxopt import solvers
import ffn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import scipy.optimize as sco
%matplotlib inline

#### Efficient Frontier
data = ffn.get("0005.hk, 0016.hk, 0027.hk, 0386.hk, 1038.hk", start = "2015-01-01", end='2021-01-01')
days = data.shape[0]
assets = data.shape[1]
print(days)

return_rates = data.pct_change().dropna()
return_rates.head()

ax = return_rates.apply(lambda x : 100 * x).plot.line(grid = True, figsize = (12, 6))
ax.set_ylabel("Return rate (%)")

annualized_return_rates = ((1 + return_rates).cumprod() ** (1 / days))[-1:] ** 252 - 1
print(annualized_return_rates)

w = np.random.random(assets) # try a random weight vector
w /= np.sum(w)               # normalization
print(w)
print(w.dot(annualized_return_rates.T))

C = return_rates.cov() * 252 # annualized
print(C)

sigma = np.sqrt(w.T.dot(C.dot(w)))
print(sigma)

def generate_one_portfolio(daily_return_rates):
    
    weight = np.random.random(daily_return_rates.shape[1])
    weight /= np.sum(weight)
    
    annual_ret = ((1 + daily_return_rates).cumprod() ** (1 / days))[-1:] ** 252 - 1
    mu = annual_ret.dot(weight).tolist()[0]

    C = daily_return_rates.cov()
    sigma2 = weight.T.dot(C.dot(weight)) * 252
    
    return weight, mu, sigma2

# How many different sets of assets to be shown as Markowitz mean-variance analysis of portfolios.
n_portfolios = 3000
# Generate 3000 weights, means, and vars of corresponding 3000 allocations of assets (portfolios).
weights, means, sig2s = np.column_stack([generate_one_portfolio(return_rates) for _ in range(n_portfolios)])
stds = sig2s ** 0.5
# Find the highest Sharpe ratio along 3000 sets of portfolios.
maxSharpe_idx = (means/stds).argmax()

plt.figure(figsize = (12, 6))
plt.scatter(stds * 100, means * 100, c = means / stds, marker = 'o')
plt.colorbar(label = 'Sharpe ratio')
plt.scatter(stds[maxSharpe_idx] * 100, means[maxSharpe_idx] * 100, c = 'r', marker = 'x')
plt.xlabel('Standard deviation (%)')
plt.ylabel('Return rate (%)')
plt.title('Mean-Variance Analysis of Portfolios')
plt.grid()

def opt_portfolio(r):

    solvers.options['show_progress'] = False

    n = r.shape[1]         # Num of assets in each portfolio.
    N = 50                 # Divide how many different lvs of risk willing to take (risk aversion).
    risk_aversion_levels = [10 ** (2 * t / N - 2) for t in range(N)]
    
    # Calculate the parameters of convex optimization.
    p = opt.matrix((((1 + r).cumprod() ** (1 / days))[-1:] ** 252 - 1).values.T)
    S = opt.matrix(r.cov().values * 252)
    
    G = opt.matrix(-np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate the optimized weights of the allocation of assets in the portfolio where maximize {E(r) - 1/2 λ Var(r)} under different lvs of risk taken (i.e., diff λ).
    weights = [solvers.qp(2 * S, -level * p, G, h, A, b)['x'] for level in risk_aversion_levels]
    
    # Calculate the return rates and risk lvs of the optimized allocation of assets under different risk aversion lvs.
    returns = np.array([np.dot(p.T, x) * 100 for x in weights])
    vols = np.array([np.sqrt(np.dot(x.T, S * x)) * 100 for x in weights])

    # Find the weights and index w.r.t. the greatest Sharpe ratio.
    idx = (returns / vols).argmax()
    wt =  weights[idx]
    
    return idx, wt, returns, vols

opt_idx, opt_weight, returns, risks = opt_portfolio(return_rates)

# In order to draw the efficient frontier, find the starting index that rules out the sub-optimal frontier.
ind = np.argmin(risks)
evols = risks[ind:]
erets = returns[ind:]
# Calculate the spline (interpolated curve) parameters based on the optimized weights against different risk aversion lvs.
tck = sci.splrep(evols, erets)

# Find the y-coordinate(s) against input x on the spline (interpolated curve).
def f(x):
    return sci.splev(x, tck, der = 0)

# Find the y-coordinate(s) of 1st derviative of the spline against input x.
def df(x):
    return sci.splev(x, tck, der = 1)

def equations(p, rf = 0.01):
    eq1 = rf - p[0]                      # 0 = rf - c
    eq2 = rf + p[1] * p[2] - f(p[2])     # 0 = c + mx - y
    eq3 = p[1] - df(p[2])                # 0 = m - y'
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.01, 0.5, 3])
print(opt) # opt[0]: risk-free rate, opt[1]: maximum Sharpe ratio, opt[2]: risk lv of tangency portfolio.

plt.figure(figsize = (12, 6))
plt.scatter(risks[opt_idx], returns[opt_idx], c = "r", marker = 'x', s = 100)
plt.legend(["Optimal Portfolio"], loc = "best")
plt.scatter(stds * 100, means * 100, c = means / stds, marker = 'o')
plt.colorbar(label = 'Sharpe ratio')
plt.plot(risks[:, 0, 0], returns[:, 0, 0], 'y-o')
plt.xlabel('Std (%)')
plt.ylabel('Mean (%)')
plt.title('Mean-Variance Analysis of Portfolios')

plt.plot([0, risks[opt_idx]], [1, returns[opt_idx]], "r--")
plt.grid()

plt.figure(figsize = (8, 8))
plt.pie(list(opt_weight), labels = return_rates.columns.values, autopct = "%1.2f%%", shadow=True)
plt.title("Ingredient of Portfolio")
print(opt_weight)