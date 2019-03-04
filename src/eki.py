import numpy as np

# u: parameters
# g: data, i.e. g(u)
# g_t: measurement mean, i.e. y = g_t + N(0, cov)
def EKI(u, g, g_t, cov):

    # Ensemble size
    J = u.shape[0]

    # Sizes of u and p
    us = u[0].size
    ps = g[0].size
    
    # means and covariances
    u_bar = np.zeros(us)
    p_bar = np.zeros(ps)
    c_up = np.zeros((us, ps))
    c_pp = np.zeros((ps, ps))

    # Loop through ensemble to start computing means and covariances
    # (all the summations only)
    for j in range(J):

        u_hat = u[j]
        p_hat = g[j]

        # Means
        u_bar += u_hat
        p_bar += p_hat
        
        # Covariance matrices
        c_up += np.tensordot(u_hat, p_hat, axes=0)
        c_pp += np.tensordot(p_hat, p_hat, axes=0)
        
    # Finalize means and covariances
    # (divide by J, subtract of means from covariance sum terms)
    u_bar = u_bar / J
    p_bar = p_bar / J
    c_up  = c_up  / J - np.tensordot(u_bar, p_bar, axes=0)
    c_pp  = c_pp  / J - np.tensordot(p_bar, p_bar, axes=0)

    # Update u
    for j in range(J):

        noise = np.random.multivariate_normal(np.zeros(ps), cov)
        y = g_t + noise
        
        # Solve for parameter correction
        tmp = np.linalg.solve(c_pp + cov, y - g[j])

        # Update parameters
        u[j] += c_up.dot(tmp)

    return u
