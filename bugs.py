from sympy import symbols, diff, sqrt,log,simplify,loggamma
import numpy as np

# 正态分布的对数概率密度函数
def dnorm(x, mu, tau):
    return 0.5 * tau * (x - mu)**2 + 0.5 * log(2 * np.pi) - 0.5 * log(tau)

# Gamma分布的对数概率密度函数
def dgamma(x, a, b):
    return b * x - a * log(b) + log(x) - a * log(x) + loggamma(a)

# 分类分布的对数概率密度函数
def dcat(x, theta):
    return -np.log(theta[x])

# import numpy as np
# from scipy.special import gammaln, logsumexp, loggamma, logit, expit
# from scipy.stats import norm, beta, gamma, binom, poisson, uniform, weibull_min

# # Dirichlet分布的对数概率密度函数
# def ddirich(x, theta):
#     return -(np.sum((theta - 1) * np.log(x)) + loggamma(np.sum(theta)) - np.sum(loggamma(theta)))

# # 多项分布的对数概率密度函数
# def dmulti(x, theta):
#     return -(np.sum(x * np.log(theta)) + loggamma(np.sum(x) + 1) - np.sum(loggamma(x + 1)))

# # 伯努利分布的对数概率密度函数
# def dbern(x, p):
#     return (-1 + x) * np.log(1 - p) - x * np.log(p)

# # Beta分布的对数概率密度函数
# def dbeta(x, a, b):
#     return -np.log((1 - x)**(-1 + b) * x**(-1 + a) / beta(a, b))

# # 二项分布的对数概率密度函数
# def dbin(x, theta, n):
#     return (-n + x) * np.log(1 - theta) - x * np.log(theta) - loggamma(n + 1) + loggamma(x + 1) + loggamma(n - x + 1)

# # 对数正态分布的对数概率密度函数
# def dlnorm(x, mu, tau):
#     return -np.log(norm.pdf(np.log(x), loc=mu, scale=1/np.sqrt(tau)))



# # 指数分布的对数概率密度函数
# def dexp(x, theta):
#     return x * theta - np.log(theta)

# # 泊松分布的对数概率密度函数
# def dpois(x, theta):
#     return theta - x * np.log(theta) + gammaln(x + 1)



# # Weibull分布的对数概率密度函数
# def dweib(x, a, b):
#     return b * x**a - np.log(a) - np.log(b) + np.log(x) - a * np.log(x)

# # Weibull分布的随机数生成
# def rweib(a, b, size=1):
#     return weibull_min.rvs(a, scale=b, size=size)

# # 均匀分布的对数概率密度函数
# def dunif(x, l, r):
#     return -np.log(r - l)

# # Logit函数
# def logit(theta):
#     return np.log(theta / (1 - theta))

# # Sigmoid函数
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # t分布的对数概率密度函数
# def dt(x, mu, tau, k):
#     return 0.5 * (np.log(k) - np.log(tau) - (1 + k) * np.log(k + (x - mu)**2 * tau) + 2 * loggamma(k/2) - 2 * loggamma(0.5))

# # Gamma分布的对数概率密度函数
# def dgamma(x, a, b):
#     return b * x - a * np.log(b) + np.log(x) - a * np.log(x) + loggamma(a)

# # Pareto分布的对数概率密度函数
# def dpar(x, a, b):
#     return -np.log(a) - a * np.log(b) - (-1 - a) * np.log(x)

# # 二项分布的随机数生成
# def rbin(p, n, size=1):
#     return binom.rvs(n, p, size=size)
