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

	def gp_mh(self, enka, n_mcmc, prior, delta = 1., enka_scaling = True, **kwargs):
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

		if kwargs.get('model', None) is not None:
			model = kwargs.get('model', None)
			phi_current -= model.logjacobian(current)

		if kwargs.get('Gamma', None) is None:
			phi_current += .5 * np.log(gvars).sum()
		samples.append(current)
		accept = 0.

		for k in tqdm(range(n_mcmc), desc ='MCMC samples: ', disable = self.mute_bar):
			if kwargs.get('update', None) is None:
				proposal = self.random_walk(current, scales, enka.p)
			elif kwargs.get('update', None) == 'pCN':
				proposal = self.pCN(current, scales, enka.p, beta = kwargs.get('beta', 0.5))

			gmean_proposal, gvars_proposal = emulate.predict_gps(enka, proposal.reshape(1,-1), kwargs.get('gpmodels', None))
			yGproposal    = gmean_proposal - y

			if kwargs.get('Gamma', None) is None:
				Sigma = np.diag(gvars_proposal.flatten())
			else:
				Sigma = kwargs.get('Gamma', None)

			phi_proposal  = (yGproposal * np.linalg.solve(2 * Sigma, yGproposal)).sum()
			phi_proposal -= prior.logpdf(proposal)

			if kwargs.get('model', None) is not None:
				phi_proposal -= model.logjacobian(proposal)

			if kwargs.get('Gamma', None) is None:
				phi_proposal += .5 * np.log(gvars_proposal).sum()

			if np.random.uniform() < np.exp(phi_current - phi_proposal):
				current = proposal
				phi_current = phi_proposal
				accept += 1.

			samples.append(current)

		self.samples = np.array(samples).T
		self.accept  = accept/n_mcmc

	def gp_mh_separable(self, enka, n_mcmc, prior, delta = 1., enka_scaling = True, **kwargs):
		"""
		GP-based Random Walk Metropolis Hastings.
		Inputs:
			- enka_scaling: Use the enka ensemble to setup the transition probabilities.
			- delta: scale parameter for the transition probabilities
		"""
		model = kwargs.get('model', None)
		n_inputs = kwargs.get('n_inputs', enka.p)

		if model is not None:
			scales = delta * np.cov(enka.Ustar[model.rank[:n_inputs]])
		elif enka_scaling:
			scales = delta * np.linalg.cholesky(np.cov(enka.Ustar))
		else:
			scales = delta * np.eye(enka.p)

		if model is not None:
			current = enka.Ustar[model.rank[:n_inputs]].mean(axis = 1)
		else:
			current = enka.Ustar.mean(axis = 1)

		y = self.y_obs.reshape(-1,1)
		samples = []

		gmean, gvars = emulate.predict_gps(enka,
						current.reshape(1,-1),
						kwargs.get('gpmodels', None),
						model = kwargs.get('model', None),
						separable = kwargs.get('separable', True))
		yG           = gmean - y

		if kwargs.get('Gamma', None) is None:
			Sigma = np.diag(gvars.flatten())
		else:
			Sigma = kwargs.get('Gamma', None)

		tqdm.write(str(Sigma.shape))
		tqdm.write(str(y.shape))
		tqdm.write(str(gmean.shape))

		phi_current  = (yG * np.linalg.solve(2 * Sigma, yG)).sum()
		phi_current -= prior.logpdf(current)

		if kwargs.get('model', None) is not None:
			model = kwargs.get('model', None)
			phi_current -= model.logjacobian(current)

		if kwargs.get('Gamma', None) is None:
			phi_current += .5 * np.log(gvars).sum()
		samples.append(current)
		accept = 0.

		for k in tqdm(range(n_mcmc), desc ='MCMC samples: ', disable = self.mute_bar):
			if kwargs.get('update', None) is None:
				proposal = self.random_walk(current, scales, enka.p)
			elif kwargs.get('update', None) == 'pCN':
				proposal = self.pCN(current, scales, enka.p, beta = kwargs.get('beta', 0.5))

			gmean_proposal, gvars_proposal = emulate.predict_gps(enka, proposal.reshape(1,-1), kwargs.get('gpmodels', None))
			yGproposal    = gmean_proposal - y

			if kwargs.get('Gamma', None) is None:
				Sigma = np.diag(gvars_proposal.flatten())
			else:
				Sigma = kwargs.get('Gamma', None)

			phi_proposal  = (yGproposal * np.linalg.solve(2 * Sigma, yGproposal)).sum()
			phi_proposal -= prior.logpdf(proposal)

			if kwargs.get('model', None) is not None:
				phi_proposal -= model.logjacobian(proposal)

			if kwargs.get('Gamma', None) is None:
				phi_proposal += .5 * np.log(gvars_proposal).sum()

			if np.random.uniform() < np.exp(phi_current - phi_proposal):
				current = proposal
				phi_current = phi_proposal
				accept += 1.

			samples.append(current)

		self.samples = np.array(samples).T
		self.accept  = accept/n_mcmc

	def model_mh(self, n_mcmc, model, prior, enka, Gamma, delta = 1., enka_scaling = True, **kwargs):
		if enka_scaling:
			scales = delta * np.linalg.cholesky(np.cov(enka.Ustar))
		else:
			scales = delta * np.eye(enka.p)

		current = enka.Ustar.mean(axis = 1)

		if model.type == 'pde':
			w_mcmc  = np.copy(model.wt)
			g = enka.G_pde(np.hstack([current.flatten(), w_mcmc]), model, model.t)
			w_mcmc = np.copy(g[enka.n_obs:])
		else:
			g = enka.G(current.flatten(), model)

		yg = g[:enka.n_obs] - self.y_obs
		phi_current = (yg * np.linalg.solve(2 * Gamma, yg)).sum()
		phi_current -= prior.logpdf(current.flatten())
		try:
			phi_current -= model.logjacobian(current.flatten())
		except AttributeError:
			pass

		samples = []
		samples.append(current.flatten())
		accept = 0.

		for kk in tqdm(range(n_mcmc), desc ='MCMC samples: ', disable = self.mute_bar):
			if kwargs.get('update', None) is None:
				proposal = self.random_walk(current, scales, enka.p)
			elif kwargs.get('update', None) == 'pCN':
				proposal = self.pCN(current, scales, enka.p, beta = kwargs.get('beta', 0.5))

			if model.type == 'pde':
				g_proposal =  enka.G_pde(np.hstack([proposal.flatten(), w_mcmc]), model, model.t)
				w_mcmc = np.copy(g_proposal[enka.n_obs:])
			else:
				g_proposal =  enka.G(proposal.flatten(), model)

			yg = g_proposal[:enka.n_obs] - self.y_obs
			phi_proposal  = (yg * np.linalg.solve(2 * Gamma, yg)).sum()
			phi_proposal -= prior.logpdf(proposal.flatten())
			try:
				phi_proposal -= model.logjacobian(proposal.flatten())
			except AttributeError:
				pass

			if np.random.uniform() < np.exp(phi_current - phi_proposal):
				current     = proposal
				phi_current = phi_proposal
				accept += 1.

			samples.append(current.flatten())

		self.samples = np.array(samples).T
		self.accept  = accept/n_mcmc

	def random_walk(self, current, scales, n_dim):
		return current + np.matmul(scales, np.random.normal(0, 1, n_dim))

	def pCN(self, current, scales, n_dim, beta = 0.5):
		return np.sqrt(1 - beta**2) * current + np.sqrt(beta) * np.matmul(scales, np.random.normal(0, 1, n_dim))
