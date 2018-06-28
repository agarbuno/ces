import pytest
import numpy as np
import sys
sys.path.append('../src')

from enki import EnKI

def error(u, u_t):
    N = u.shape[0]
    return np.sqrt(np.sum((u - u_t)**2) / N)

def enki_run(J=100, n=5, r=0.1):

    # Maximum iterations
    iter_max = 1000
    
    # Linear operator
    A = np.random.normal(loc=0, scale=2, size=n*n).reshape(n,n)

    # Truth
    u_t = np.random.normal(loc=0, scale=1, size=n)
    g_t = A.dot(u_t)

    # Generate ensembles
    u_ens = np.random.normal(loc=0, scale=2, size=J*n).reshape(J,n)
    g_ens = np.array([A.dot(u) for u in u_ens])

    # Covariance -- no noise
    cov = np.zeros((n,n))
    
    iter = 0
    diff = 1
    err_prev = 0
    while diff > 0.000000001:
        u_ens = EnKI(u_ens, g_ens, g_t, cov)
        err = error(u_ens.mean(0), u_t)
        if iter == iter_max: break
        iter += 1
        diff = abs(err-err_prev)
        err_prev = err

    return iter, err

# J: ensemble size
# n: number of parameters
# r: rescale covariance
def test_no_noise():
    J = 100
    n = 5
    r = 0.1
    iter, err = enki_run(J=J, n=n, r=r)
    assert iter == 1
    assert err < 0.1
