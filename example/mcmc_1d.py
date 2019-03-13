import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../src')
from mcmc import MCMC
import pickle
import scipy

# MCMC setup
truth = np.array([0.0])
cov = np.array([[1]])
prior = [{'distribution': 'uniform', 'min': -10, 'max': 10}]
width = np.array([1])
param_init = truth

mcmc = MCMC(truth, cov, prior, width, param_init)

# MCMC
iter = 0
iter_max = 50000
while iter < iter_max:
    g = mcmc.param
    mcmc.sampler(g)
    iter += 1
    if np.mod(iter, 10000) == 0:
        print 'Iteration ', iter, ' of ', iter_max, ','\
            ' acceptance rate = ', mcmc.compute_acceptance_rate()

# Actual normal distribution
def compute_normal(p, mean, cov):
    return scipy.stats.multivariate_normal(mean,cov).pdf(p)

x = np.arange(-10, 10, 0.1)
data = compute_normal(x, truth, cov)

# Plot results
fig = plt.figure()
plt.hist(mcmc.posterior, density=True, range=[-4, 4], bins=100)
plt.plot(x, data)
plt.xlabel('x')
plt.show()
