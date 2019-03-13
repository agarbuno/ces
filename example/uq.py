import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append('../src')
from gp import GP
from mcmc import MCMC
import pickle
import scipy.stats
from eki import EKI

########################################################################
# OVERVIEW                                                             #
# ---------------------------------------------------------------------#
#                                                                      #
# 0) Problem setup: function, truth, parameter domain, and noise       #
# 1) Generate training data (EKI)                                      #
# 2) Run & validate GP.                                                #
# 3) UQ OF INVERSE PROBLEM: Run MCMC and plot posterior.               #
# 4) UQ OF FORWARD PROBLEM: Sample from posterior and evaluate model.  #
#                                                                      #
########################################################################

np.random.seed(42)

############################
# Section 0: Problem setup #
############################

# Model
def g(u):
    return 2*u

# Parameter domain
npar = 2
umin = np.array([-10] * npar)
umax = np.array([10] * npar)

# Truth
ut = np.random.uniform(-1, 1, size=npar)
yt = g(ut)

# Noise
nvar = yt.shape[0]
A = np.random.uniform(-1, 1, size=(nvar, nvar))
cov = A.dot(A.T)
def noise(n=1):
    return scipy.stats.multivariate_normal(np.zeros(nvar), cov).rvs(n)

##############################
# Section 1: Training points #
##############################

# Setup EKI
J = 50
u_ens = np.random.uniform(umin, umax, size=(J,npar))
eki = EKI(u_ens, yt, cov)

# EKI optimization
iter = 0
while iter < 10:
    u_ens = eki.get_u()
    g_ens = g(u_ens) + noise(n=J)
    eki.update(g_ens)
    eki.compute_error()
    iter += 1

# EKI results
print 'True parameters:', ut
print 'EKI results:', eki.u[-1].mean(0)
print ''

# Training points
u_tp = np.concatenate(eki.u[:-1], axis=0)
g_tp = np.concatenate(eki.g, axis=0)



#######################
# Section 2: GP model #
#######################

normalizer = False

# svd = False
# truth = None
# covgp = None

svd = True
# truth = False
truth = yt
covgp = cov

sparse = False
# sparse = True
n_induce = 10

gp = GP(u_tp, g_tp,
        normalizer=normalizer,             # Normalize data
        svd=svd, truth=truth, cov=covgp,   # SVD, must decompose COV, shift by truth optional
        sparse=sparse, n_induce=n_induce)  # Sparse GP

gp.initialize_models()

# Validation
nval = 100
uval = np.random.uniform(umin, umax, size=(nval, nvar))
gval = g(uval) + noise(n=nval)
gpred, sigpred = gp.predict(uval)
for y, y_pred in zip(gval.T, gpred.T):
    vmin = min(y.min(), y_pred.min())
    vmax = max(y.max(), y_pred.max())

    line = [y.min(), y.max()]
    
    plt.figure()
    plt.scatter(y, y_pred)
    plt.plot(line, line, 'k-')
    plt.xlabel('Model')
    plt.ylabel('GP')

    plt.tight_layout()
    plt.show()



###################
# Section 3: MCMC #
###################

# Setup
# u0 = np.random.uniform(umin, umax, size=npar)
u0 = np.copy(ut)
prior = [{'distribution': 'uniform', 'min': a, 'max': b} for a,b in zip(umin, umax)]
width = 0.1 * np.sqrt(np.diag(cov))

mcmc = MCMC(yt, cov, prior, width, u0)

print 'Beginning MCMC'
iter = 0
iter_max = 5000
while iter < iter_max:
    
    param = mcmc.param[None,:]

    g_pred, _ = gp.predict(param)
    g_pred = np.asarray(g_pred).squeeze()

    mcmc.sampler(g_pred)

    iter += 1

    if (iter % 1000 == 0):
        print 'Iteration ', iter, ' of ', iter_max,\
            '; acceptance rate = ', mcmc.compute_acceptance_rate()

# Plot posterior
burnin = 1000
u1 = mcmc.posterior[1000:,0]
u2 = mcmc.posterior[1000:,1]

fig = plt.figure()
hist, xedges, yedges = np.histogram2d(u1, u2, bins=30)
hist /= hist.sum()
cs = plt.contourf(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='Blues')
cs.cmap.set_under('white')
cs.set_clim(hist.max()/8, hist.max())
cb = plt.colorbar(cs)

plt.scatter(ut[0], ut[1], s=64, color='k')
plt.scatter(ut[0], ut[1], s=50, color='r', label='True Parameters')

plt.xlabel(r'u$_1$')
plt.ylabel(r'u$_2$')
plt.legend()
plt.show()


# Print stats
print ''
print 'Posterior Mean:'
print np.mean(mcmc.posterior, axis=0)
print ''
print 'Posterior Uncertainty:'
print np.cov(mcmc.posterior.T)
print ''



#########################
# Section 4: Forward UQ #
#########################


# Sample from posterior distribution
u1 = mcmc.posterior[:,0]
u2 = mcmc.posterior[:,1]
hist, u1_bins, u2_bins = np.histogram2d(u1, u2, bins=(50, 50))
u1_bin_midpoints = u1_bins[:-1] + np.diff(u1_bins)/2
u2_bin_midpoints = u2_bins[:-1] + np.diff(u2_bins)/2
cdf = np.cumsum(hist.ravel())
cdf = cdf / cdf[-1]

values = np.random.rand(10000)
value_bins = np.searchsorted(cdf, values)
u1_idx, u2_idx = np.unravel_index(value_bins,
                                (len(u1_bin_midpoints),
                                 len(u2_bin_midpoints)))
random_from_cdf = np.column_stack((u1_bin_midpoints[u1_idx],
                                   u2_bin_midpoints[u2_idx]))
u1_new, u2_new = random_from_cdf.T

# Evaluate model
u_sample = np.column_stack([u1_new, u2_new])
y_sample = g(u_sample) + noise(n=u_sample.shape[0])
yt_sample = g(ut) + noise(n=u_sample.shape[0])

print 'Forward Model UQ using posterior distribution:'
print np.cov(y_sample.T)
print ''

print 'Forward Model UQ at true parameters:'
print np.cov(yt_sample.T)
print ''

print 'Noise Covariance:'
print cov
