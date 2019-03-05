import numpy as np
import matplotlib.pyplot as plt

class EKI:

    # y = g(u) + N(0, cov)
    def __init__(self, parameters, truth, cov):

        # Check inputs
        assert (parameters.ndim == 2), \
            'EKI init: parameters must be 2d array, num_ensembles x num_parameters'
        assert (truth.ndim == 1), 'EKI init: truth must be 1d array'
        assert (cov.ndim == 2), 'EKI init: covariance must be 2d array'
        assert (truth.size == cov.shape[0] and truth.size == cov.shape[1]),\
            'EKI init: truth and cov are not the correct sizes'
        
        # Truth statistics
        self.g_t = truth
        self.cov = cov

        # Parameters
        self.u = parameters[np.newaxis]

        # Ensemble size
        self.J = parameters.shape[0]
        
        # Store observations during iterations
        self.g = np.empty((0,self.J)+truth.shape)

        # Error
        self.error = np.empty(0)


    # Parameters corresponding to minimum error.
    # Returns mean and standard deviation.
    def get_u(self):
        return self.u[-1]

    # Minimum error
    def get_error(self):
        try:
            idx = self.error.argmin()
            return self.error[idx]
        except ValueError:
            print('No errors computed.')

    # Compute error using mean of ensemble model evaluations.
    def compute_error(self):
        diff = self.g_t - self.g[-1].mean(0)
        try:
            error = diff.dot(np.linalg.solve(self.cov, diff))
        except np.linalg.LinAlgError:
            error = diff.dot(diff) / diff.size
        self.error = np.append(self.error, error)
        
    # g: data, i.e. g(u), with shape (num_ensembles, num_elements)
    def update(self, g):
        
        u = np.copy(self.u[-1])
        g_t = self.g_t
        cov = self.cov
        
        # Ensemble size
        J = self.J
        
        # Sizes of u and p
        us = u[0].size
        ps = g[0].size
        
        # Ensemble statistics
        u_bar = np.zeros(us)
        p_bar = np.zeros(ps)
        c_up = np.zeros((us, ps))
        c_pp = np.zeros((ps, ps))
        
        for j in range(J):
            
            u_hat = u[j]
            p_hat = g[j]
            
            # Means
            u_bar += u_hat
            p_bar += p_hat
            
            # Covariance matrices
            c_up += np.tensordot(u_hat, p_hat, axes=0)
            c_pp += np.tensordot(p_hat, p_hat, axes=0)
            
        u_bar = u_bar / J
        p_bar = p_bar / J
        c_up  = c_up  / J - np.tensordot(u_bar, p_bar, axes=0)
        c_pp  = c_pp  / J - np.tensordot(p_bar, p_bar, axes=0)
        
        # Update u
        noise = np.array([np.random.multivariate_normal(np.zeros(ps), cov) for _ in range(J)])
        y = g_t + noise
        tmp = np.linalg.solve(c_pp + cov, np.transpose(y-g))
        u += c_up.dot(tmp).T

        # Store parameters and observations
        self.u = np.append(self.u, [u], axis=0)
        self.g = np.append(self.g, [g], axis=0)


    def plot_error(self):
        x = np.arange(self.error.size)
        plt.semilogy(x, self.error)
        plt.xlabel('Iteration')
        plt.ylabel(r'$\parallel y - g(u)\parallel_\Gamma^2$')
