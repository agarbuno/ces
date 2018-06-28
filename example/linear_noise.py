import sys
sys.path.append('../src')
import numpy as np
from enki import EnKI

def error(u, u_t):
    N = u.shape[0]
    return np.sqrt(np.sum((u - u_t)**2) / N)
    
def main():

    # Maximum iterations
    iter_max = 1000
    
    # Ensemble size
    J = 100

    # Number of parameters
    n = 5

    # Rescale covariance (cov <-- r**2 * cov)
    r = 0.01
    
    # Linear operator
    A = np.random.normal(loc=0, scale=2, size=n*n).reshape((n,n))

    # Truth
    u_t = np.random.normal(loc=0, scale=3, size=n)
    g_t = A.dot(u_t)

    # Covariance -- positive definite and rescale
    cov = np.random.normal(loc=0, scale=2, size=n*n).reshape((n,n))
    cov = cov.T.dot(cov)
    cov *= r*r
    
    # Generate ensembles
    u_ens = np.random.normal(loc=0, scale=2, size=J*n).reshape(J,n)
    g_ens = np.array([A.dot(u) for u in u_ens])

    # Ensemble Kalman Inversion
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

    print('')
    if iter == 1:
        print(iter,'iteration')
    else:
        print(iter,'iterations')

    print('')
    print('Mean parameters')
    print(u_ens.mean(0))
    print('')
    print('Truth')
    print(u_t)
    print('')
    print('Error')
    print(err)
    print('')
    
main()
