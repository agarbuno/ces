import numpy as np
import pandas as pd
from scipy import integrate
import matlab.engine
from tqdm.autonotebook import tqdm

class model(object):
	def __init__(self, alpha = 2., tau  = 3., Nmesh = 2.**4):
		self.eng = matlab.engine.start_matlab()
		self.eng.addpath(r'./../mfiles','-end');
		self.alpha = alpha
		self.tau   = tau
		self.Nmesh = Nmesh
		self.type  = 'map'
		self.p     = int(self.Nmesh * self.Nmesh)
		self.model_name = 'darcy-flow'

	def __call__(self, xi, full_solution = False):
		"""
		Inputs:
			- xi: (p,) array that gets converted to matlab object.
		Outputs:
			- y: (n_obs,) array of observations at obs_index.
		"""
		theta = self.eval_rf(xi)
		U     = self.solve_pde(theta)

		if full_solution:
			return np.asarray(U).flatten()
		else:
			return np.asarray(U).flatten()[self.obs_index]

	def set_rnd_seed(self, seed = 1):
		self.eng.rng('default');
		self.eng.rng(seed)

	def stop(self):
		self.eng.exit()

	def set_initial(self, seed = 1):
		np.random.seed(seed)
		self.ustar  = np.random.normal(0, 1, int(self.p))

	def eval_rf(self, xi):
		"""
		Input  :
			- xi: random seed in the Karhunen Loeve.
		Outputs:
			- theta: log Gaussian random field.
		"""
		return self.eng.gaussrnd_coarse(matlab.double(xi.reshape(int(self.Nmesh), -1).tolist()), self.alpha, self.tau, self.Nmesh)

	def solve_pde(self, theta):
		return self.eng.solve_gwf(theta)