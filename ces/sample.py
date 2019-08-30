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

	def model_rw(self, n_mcmc, wt, model, prior, enka, Gamma, delta = 1., enka_scaling = True):
		if enka_scaling:
			scales = delta * np.linalg.cholesky(np.cov(enka.Ustar))
		else:
			scales = delta * np.eye(enka.p)

		current = enka.Ustar.mean(axis = 1)
		w_mcmc  = np.copy(wt)

		g = enka.G_pde(np.hstack([current.flatten(), w_mcmc]), model, t)
		w_mcmc = np.copy(g[enka.n_obs:])

		phi_current = ((g[:enka.n_obs] - y_obs) * np.linalg.solve(2 * Gamma, g[:enka.n_obs] - y_obs)).sum()
		phi_current -= prior.logpdf(current.flatten())

		samples.append(current.flatten())
		accept = 0.

		for kk in tqdm(range(n_mcmc), desc ='MCMC samples: ', disable = self.mute_bar):
			proposal   = current + np.matmul(scales, np.random.normal(0, 1, (enka.p,1)))

			g_proposal =  enka.G_pde(np.hstack([proposal.flatten(), w_mcmc]), model, t)
			w_mcmc = np.copy(g_proposal[enka.n_obs:])

			phi_proposal  = ((g_proposal[:enka.n_obs] - y_obs) * np.linalg.solve(2 * Gamma, g_proposal[:enka.n_obs] - y_obs)).sum()
			phi_proposal -= prior.logpdf(proposal.flatten())

			if np.random.uniform() < np.exp(phi_current - phi_proposal):
				current     = proposal
				phi_current = phi_proposal
				accept += 1.

			samples.append(current.flatten())

		self.samples = np.array(samples).T
		self.accept  = accept/n_mcmc
