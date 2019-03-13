import pytest
import numpy as np
import sys
sys.path.append('../src')

from eki import EKI

def eki_run(J=100, n=5):

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

    # Ensemble Kalman Inversion object
    eki = EKI(u_ens, g_t, cov)
    
    # Iterate
    iter = 0
    while iter < iter_max:

        eki.update(g_ens)
        eki.compute_error()
        if eki.error[-1] < 1e-10: break

        # New model evaluations
        u_ens = eki.get_u()
        g_ens = np.array([A.dot(u) for u in u_ens])
        iter += 1

    return iter, eki.error[-1]

# J: ensemble size
# n: number of parameters
def test_no_noise():
    J = 100
    n = 5
    iter, err = eki_run(J=J, n=1)
    assert iter == 1
    assert err < 0.1

test_no_noise()
