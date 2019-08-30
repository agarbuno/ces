from __future__ import print_function
import os
import pickle
import numpy as np
import pandas as pd
from . import calibrate
from . import emulate
import gpflow as gp

from tqdm.autonotebook import tqdm

class MCMC(object):

	def __init__(self):
		self.mute_bar = False

	def gp_rw(self, enka, n_mcmc, prior, delta = 1., enka_scaling = True, **kwargs):
		"""
		GP-based Random Walk Metropolis Hastings.
		Inputs:
			- enka_scaling: Use the enka ensemble to setup the transition probabilities.
			- delta: scale parameter for the transition probabilities
		"""
		if enka_scaling:
			scales = delta * np.linalg.cholesky(np.cov(enka.Ustar))
		else:
			scales = delta * np.eye(enka.p)

		kwargs.get('Gamma', None)

		current = enka.Ustar.mean(axis = 1)
		y = self.y_obs.reshape(-1,1)
		samples = []
		gmean, gvars = emulate.predict_gps(enka, current.reshape(1,-1), kwargs.get('gpmodels', None))
		yG           = gmean - y

		if kwargs.get('Gamma', None) is None:
			Sigma = np.diag(gvars.flatten())
		else:
			Sigma = kwargs.get('Gamma', None)

		phi_current  = (yG * np.linalg.solve(2 * Sigma, yG)).sum()
		phi_current -= prior.logpdf(current)
		if kwargs.get('Gamma', None) is None:
			phi_current += .5 * np.log(gvars).sum()
		samples.append(current)
		accept = 0.

		for k in tqdm(range(n_mcmc), desc ='MCMC samples: ', disable = self.mute_bar):
			proposal = current + np.matmul(scales, np.random.normal(0, 1, enka.p))

			gmean_proposal, gvars_proposal = emulate.predict_gps(enka, proposal.reshape(1,-1), kwargs.get('gpmodels', None))
			yGproposal    = gmean_proposal - y

			if kwargs.get('Gamma', None) is None:
				Sigma = np.diag(gvars_proposal.flatten())
			else:
				Sigma = kwargs.get('Gamma', None)

			phi_proposal  = (yGproposal * np.linalg.solve(2 * Sigma, yGproposal)).sum()
			phi_proposal -= prior.logpdf(proposal)
			if kwargs.get('Gamma', None) is None:
				phi_proposal += .5 * np.log(gvars_proposal).sum()

			if np.random.uniform() < np.exp(phi_current - phi_proposal):
				current = proposal
				phi_current = phi_proposal
				accept += 1.

			samples.append(current)

		self.samples = np.array(samples).T
		self.accept  = accept/n_mcmc
