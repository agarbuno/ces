import numpy as np
import GPy
import sys
import os
import pickle

class GP:

    def __init__(self, parameters, data, normalizer=False, kernel='RBF',\
                 svd=False, truth=None, cov=None,\
                 sparse=False, n_induce=10):
        
        # Training data
        self.X = parameters
        self.Y = data
        self.truth = truth

        # SVD stuff
        self.svd = svd
        if svd is True:
            self.cov = cov
            self.U, self.s, self.Vh = np.linalg.svd(cov, full_matrices=False)
        
        # Regression stuff
        self.m = []
        self.kernel = kernel
        self.normalizer = normalizer
        self.sparse = sparse
        self.n_induce = n_induce
        

    def initialize_models(self):

        Y = np.copy(self.Y)
        
        # Subtract training data from truth
        if not self.truth is None:
            try:
                Y -= self.truth
            except ValueError:
                Y -= self.truth[:,None]

        # Transform data with SVD
        if self.svd:
            D = np.diag(1./np.sqrt(self.s))
            V = self.Vh.T
            Y = Y.dot(V.dot(D))

        # Independent variables
        Y = [y[:,None] for y in Y.T]
        
        # Setup the model
        for i, Yi in enumerate(Y):
            
            # Kernel
            if self.kernel == 'RBF':
                k = GPy.kern.RBF(self.X.shape[1], ARD=True)
            else:
                print 'Not setup for ', self.kernel

            # Regression model
            if self.sparse:
                Z = np.asarray(map(np.random.uniform, self.X.min(0), self.X.max(0),
                                   [self.n_induce]*self.X.shape[1])).T
                lik = GPy.likelihoods.Gaussian()
                m = GPy.core.SparseGP(self.X, Yi, Z, k, lik, normalizer=self.normalizer)
            else:
                m = GPy.models.GPRegression(self.X, Yi, k, normalizer=self.normalizer)

            m.optimize()
            self.m.append(m)

        
    def predict(self, X):

        mean = []
        var = []
        for m in self.m:
            mu, sig2 = m.predict(X)
            mean.append(mu)
            var.append(sig2)

        mean = np.array(mean).squeeze()
        var = np.asarray(var).squeeze().T
        try:
            var = map(np.diag, var)
        except ValueError:
            var = [np.diag(var)]
        
        if self.svd:
            D = np.diag(np.sqrt(self.s))
            V = self.Vh.T
            mean = V.dot(D.dot(mean))
            var = [V.dot(D.dot(v.dot(D.dot(V.T)))) for v in var]
        
        if not self.truth is None:
            try:
                mean += self.truth
            except ValueError:
                mean += self.truth[:,None]
        
        return mean.T, np.array(var)
