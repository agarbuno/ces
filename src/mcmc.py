#!/usr/bin/env python
import numpy as np
import pickle
import sys
import os
import scipy.stats

class MCMC:

    def __init__(self, truth, covariance, prior, width, param_init, burnin=0):

        self.y = truth
        self.cov = covariance
        self.cov_inv = np.linalg.inv(covariance)
        self.prior = prior
        self.width = width
        self.burnin = burnin

        self.param = param_init
        self.posterior = np.array([param_init])
        self.log_posterior = None
        
        
    def get_proposal(self):
        return self.param
        
    def get_posterior(self):
        return self.posterior[self.burnin:]

    # Ideally between 0.25 and 0.5
    def compute_acceptance_rate(self):
        iterations = self.posterior.shape[0] - self.burnin
        mask = self.posterior[self.burnin+1:] - self.posterior[self.burnin:-1] == 0
        if mask.shape[1] > 1:
            mask = np.product(mask, axis=1)
        accept = mask.sum()
        return float(accept) / iterations

    # Assumed normal distribution
    def log_likelihood(self, g):
        diff = g - self.y
        log_rho = -0.5 * diff.dot(self.cov_inv.dot(diff.T))
        return log_rho

    # Uniform and/or normal priors
    def log_prior(self):
        log_rho = 0.0
        for parameter, prior in zip(self.param, self.prior):
            if prior['distribution'] == 'uniform':
                loc = prior['min']
                scale = prior['max'] - prior['min']
                log_rho += np.log(scipy.stats.uniform(loc, scale).pdf(parameter))
            elif prior['distribution'] == 'normal':
                log_rho += np.log(scipy.stats.norm(prior['mean'], prior['sd']).pdf(parameter))
            else:
                print 'No case exists in log_prior for '+prior['distribution']
                sys.exit()
                
        return log_rho

    # Propose new parameters
    def proposal(self):
        prop = scipy.stats.multivariate_normal(self.posterior[-1], self.width**2).rvs() 
        if np.isscalar(prop):
            return np.array([prop])
        else:
            return prop

        
    # MCMC iteration
    def sampler(self, g):

        log_posterior = self.log_likelihood(g) + self.log_prior()
        if self.log_posterior == None:
            self.log_posterior = log_posterior
        else:
            p_accept = np.exp( log_posterior - self.log_posterior )
            if p_accept > np.random.uniform():
                self.posterior = np.append(self.posterior, [self.param], axis=0)
                self.log_posterior = log_posterior
            else:
                self.posterior = np.append(self.posterior, [self.posterior[-1]], axis=0)
                
        # Propose new parameters
        self.param = self.proposal()
