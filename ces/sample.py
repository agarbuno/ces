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

		try:
			getattr(self, 'samples')
			samples = list(self.samples.T)
			current = samples[-1]
			accept  = 0
		except AttributeError:
			samples = []
			samples.append(current.flatten())
			accept = 0.

		gmean, gvars = emulate.predict_gps(enka, current.reshape(1,-1),
							gpmodels  = kwargs.get('gpmodels', None),
							nugget    = kwargs.get('nugget', True),
							pca_tools = kwargs.get('pca_tools', None))
		yG           = gmean - y

		if kwargs.get('Gamma', None) is None:
			Sigma = np.diag(gvars.flatten())
		elif kwargs.get('noise_compounded', False) and kwargs.get('pca_tools', None) is None:
			Sigma = kwargs.get('Gamma') + np.diag(gvars.flatten())
		elif kwargs.get('pca_tools', None) is not None:
			Sigma = kwargs.get('Gamma') + gvars
		else:
			Sigma = kwargs.get('Gamma')

		phi_current  = (yG * np.linalg.solve(2 * Sigma, yG)).sum()
		phi_current -= prior.logpdf(current)

		if kwargs.get('model', None) is not None:
			model = kwargs.get('model', None)
			try:
				phi_current -= model.logjacobian(current)
			except AttributeError:
				pass

		if kwargs.get('Gamma', None) is None:
			phi_current += .5 * np.log(gvars).sum()
		elif kwargs.get('noise_compounded', False):
			phi_current += .5 * np.log(np.linalg.eigvals(Sigma)).sum()
			# phi_current += .5 * np.log(np.linalg.det(Sigma))

		for k in tqdm(range(n_mcmc), desc ='MCMC samples: ', disable = self.mute_bar):
			if kwargs.get('update', None) is None:
				proposal = self.random_walk(current, scales, enka.p)
			elif kwargs.get('update', None) == 'pCN':
				proposal = self.pCN(current, scales, enka.p, beta = kwargs.get('beta', 0.5))

			gmean_proposal, gvars_proposal = emulate.predict_gps(enka, proposal.reshape(1,-1),
						gpmodels = kwargs.get('gpmodels', None),
						nugget = kwargs.get('nugget', True),
						pca_tools = kwargs.get('pca_tools', None))
			yGproposal    = gmean_proposal - y

			if kwargs.get('Gamma', None) is None:
				Sigma = np.diag(gvars_proposal.flatten())
			elif kwargs.get('noise_compounded', False) and kwargs.get('pca_tools', None) is None:
				Sigma = kwargs.get('Gamma', None) + np.diag(gvars_proposal.flatten())
			elif kwargs.get('pca_tools', None) is not None:
				Sigma = kwargs.get('Gamma') + gvars_proposal
			else:
				Sigma = kwargs.get('Gamma', None)

			phi_proposal  = (yGproposal * np.linalg.solve(2 * Sigma, yGproposal)).sum()
			phi_proposal -= prior.logpdf(proposal)

			if kwargs.get('model', None) is not None:
				try:
					phi_proposal -= model.logjacobian(proposal)
				except AttributeError:
					pass

			if kwargs.get('Gamma', None) is None:
				phi_proposal += .5 * np.log(gvars_proposal).sum()
			elif kwargs.get('noise_compounded', False):
				# phi_proposal += .5 * np.log(np.linalg.det(Sigma))
				phi_proposal += .5 * np.log(np.linalg.eigvals(Sigma)).sum()

			if np.random.uniform() < np.exp(phi_current - phi_proposal):
				current = proposal
				phi_current = phi_proposal
				accept += 1.

			samples.append(current)

		self.samples = np.array(samples).T
		self.accept  = accept/n_mcmc

	def model_mh(self, model, n_mcmc, prior, enka, Gamma, delta = 1., enka_scaling = True, **kwargs):
		if enka_scaling:
			scales = delta * np.linalg.cholesky(np.cov(enka.Ustar))
		else:
			scales = delta * np.eye(enka.p)

		current = enka.Ustar.mean(axis = 1)

		if model.type == 'pde':
			w_mcmc  = np.copy(model.wt)
			g = enka.G_pde(np.hstack([current.flatten(), w_mcmc]), model, model.t)
			# w_mcmc = np.copy(g[enka.n_obs:])
		else:
			g = enka.G(current.flatten(), model)

		yg = g[:enka.n_obs] - self.y_obs
		phi_current = (yg * np.linalg.solve(2 * Gamma, yg)).sum()
		phi_current -= prior.logpdf(current.flatten())
		try:
			phi_current -= model.logjacobian(current.flatten())
		except AttributeError:
			pass

		try:
			getattr(self, 'samples')
			samples = list(self.samples.T)
			current = samples[-1]
			accepy = 0
		except AttributeError:
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
				# w_mcmc = np.copy(g_proposal[enka.n_obs:])
			else:
				g_proposal =  enka.G(proposal.flatten(), model)

			yg = g_proposal[:enka.n_obs] - self.y_obs
			phi_proposal  = (yg * np.linalg.solve(2 * Gamma, yg)).sum()
			phi_proposal -= prior.logpdf(proposal.flatten())
			try:
				phi_proposal -= model.logjacobian(proposal.flatten())
			except AttributeError:
				pass

			if np.log(np.random.uniform()) < phi_current - phi_proposal:
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
