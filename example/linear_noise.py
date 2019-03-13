import sys
sys.path.append('../src')
import numpy as np
from eki import EKI
import matplotlib.pyplot as plt

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
    A = np.random.normal(loc=0, scale=2, size=(n,n))

    # Truth
    u_t = np.random.normal(loc=0, scale=3, size=n)
    g_t = A.dot(u_t)

    # Covariance -- positive definite and rescale
    cov = np.random.normal(loc=0, scale=2, size=(n,n))
    cov = cov.T.dot(cov)
    cov *= r*r
    
    # Generate ensembles
    u_ens = np.random.normal(loc=0, scale=2, size=(J,n))
    g_ens = np.array([A.dot(u) for u in u_ens])

    # Ensemble Kalman Inversion object
    eki = EKI(u_ens, g_t, cov)
    
    # Iterate
    iter = 0
    while iter < iter_max:
        eki.update(g_ens)
        eki.compute_error()
        if eki.error[-1] < 0.01: break
        iter += 1
        u_ens = eki.get_u()
        g_ens = np.array([A.dot(u) for u in u_ens])

    print('')
    if iter == 1:
        print(iter,'iteration')
    else:
        print(iter,'iterations')

    print('')
    print('Mean parameters')
    print(u_ens.mean(0))
    print('')
    print('Truth parameters')
    print(u_t)
    print('')
    print('Error')
    print(eki.error[-1])
    print('')

    eki.plot_error()
    plt.show()

main()
