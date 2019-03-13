import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../src')
from mcmc import MCMC
import pickle
import scipy


truth = np.array([0.0, 0.0])
cov = np.diag(np.array([2, 1])**2)
prior = [{'distribution': 'uniform', 'min': -10, 'max': 10},
         {'distribution': 'uniform', 'min': -10, 'max': 10}]
width = np.array([1,1])
param_init = truth

mcmc = MCMC(truth, cov, prior, width, param_init)

iter = 0
iter_max = 10000
while iter < iter_max:
    g = mcmc.param
    mcmc.sampler(g)
    iter += 1
    if np.mod(iter, 10000) == 0:
        print mcmc.compute_acceptance_rate()

x1 = mcmc.posterior[:,0]
x2 = mcmc.posterior[:,1]

fig = plt.figure(figsize=(4,4))
hist, xedges, yedges = np.histogram2d(x1, x2, bins=30)
hist /= hist.sum()
cs = plt.contourf(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='Blues')
cs.cmap.set_under('white')
cs.set_clim(hist.max()/8, hist.max())
cb = plt.colorbar(cs)

plt.xlabel(r'x$_1$')
plt.ylabel(r'x$_2$')
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.show()


# Plot normal distribution
def compute_normal(p, mean, cov):
    return scipy.stats.multivariate_normal(mean,cov).pdf(p)

x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)
xx, yy = np.meshgrid(x, y)
X = np.array(zip(xx.ravel(), yy.ravel()))
data = compute_normal(X, truth, cov).reshape(xx.shape)

fig = plt.figure(figsize=(4,4))
cs = plt.contourf(xx, yy, data, 30, cmap='Blues')
cs.cmap.set_under('white')
cb = plt.colorbar(cs)

plt.xlabel('x')
plt.ylabel('y')
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.show()
