import numpy as np
import pandas as pd
from scipy import integrate
import matlab.engine
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
import multiprocessing

class model(object):
	def __init__(self, alpha = 2., tau  = 3., Nmesh = 2.**4):
		# MODEL HYPER-PARAMETERS
		self.alpha = alpha
		self.tau   = tau
		self.Nmesh = Nmesh
		self.p     = int(self.Nmesh * self.Nmesh)
		# THIS ARE ADDITIONAL VARIABLES FOR THE EKS CODE
		self.model_name = 'darcy-flow'
		self.type  = 'map'

	def __call__(self, xi, full_solution = False):
		"""
		Inputs:
			- xi: (p,) array that gets converted to matlab object. It's the
					coefficients of the KL expansion.
		Outputs:
			- y: (n_obs,) array of observations at the computational grid or at the
					observation indices `obs_index`.
		"""
		# USE THE KL COEFFICIENTS TO GENERATE THE RANDOM FIELD.
		theta = self.eval_rf(xi)
		# USE THE REALIZATION OF THE RANDOM FIELD TO SOLVE THE PDE.
		U     = self.solve_pde(theta)
		if full_solution:
			# RETURNS THE SOLUTION *AT EVERY* COMPUTATIONAL GRID POINT
			return np.asarray(U).flatten()
		else:
			# RETURNS THE SOLUTION *ONLY* AT SPECIFIED INDEX LOCATIONS
			return np.asarray(U).flatten()[self.obs_index]

	def start(self, mpath = None):
		"""
			mpath: the directory where the matlab files reside.
		"""
		self.eng = matlab.engine.start_matlab("-nojvm -nosplash")
		if mpath is None:
			# THIS SPECIFIES A DEFAULT IN MY MACHINE.
			self.eng.addpath(r'./../mfiles','-end');
		else:
			# USE THE PATH SPECIFIED BY USER.
			self.eng.addpath(mpath,'-end');

	def stop(self):
		"""
			stops the Matlab engine
		"""
		self.eng.quit()
		del self.eng

	def set_rnd_seed(self, seed = 1):
		"""
			setups the random seeds for the matlab engine
		"""
		self.eng.rng('default');
		self.eng.rng(seed)

	def set_initial(self, seed = 1):
		"""
			initialize a random realization of the KL coefficients.
			use seed for reproducibility.
		"""
		np.random.seed(seed)
		self.ustar  = np.random.normal(0, 1, int(self.p))

	def set_rank(self):
		"""
			receives the order in which the eigenvalues should be sorted
		"""
		k = np.arange(int(self.Nmesh))
		K1, K2 = np.meshgrid(k, k)
		self.eigs = (self.tau**(self.alpha-1))*(np.pi**2 * (K1**2 + K2**2) + self.tau**2)**(-self.alpha/2)
		self.eigs[0,0] = 1e-10
		self.rank = (-self.eigs).flatten().argsort()

	def eval_rf(self, xi):
		"""
			Evaluates the random field given the KL coefficients.
			Input  :
				xi    - Coefficients of the Karhunen Loeve expansion.
			Outputs:
				theta - log Gaussian random field.
		"""
		return self.eng.gaussrnd_coarse(matlab.double(xi.reshape(int(self.Nmesh), -1).tolist()),
										self.alpha,
										self.tau,
										self.Nmesh)

	def solve_pde(self, theta):
		return self.eng.solve_gwf(theta)

class model_trunc(model):
	"""
		This uses the truncated KL expansion for the PDE solver.
	"""

	def __init__(self, alpha = 2., tau  = 3., Nmesh = 2.**4, p = 10):
		super().__init__(alpha = alpha, tau = tau, Nmesh = Nmesh)
		super().set_rank()
		self.p = p

	def __call__(self, xi_red, full_solution = False):
		"""
		Inputs:
			- xi: (p,) array that gets converted to matlab object.
		Outputs:
			- y: (n_obs,) array of observations at obs_index.
		"""
		theta = self.eval_rf(xi_red)
		U     = self.solve_pde(theta)
		if full_solution:
			return np.asarray(U).flatten()
		else:
			return np.asarray(U).flatten()[self.obs_index]

	def set_initial(self, seed = 1):
		np.random.seed(seed)
		ustar  = np.random.normal(0, 1, int(self.Nmesh * self.Nmesh))
		self.ustar = ustar[self.rank[:self.p]]

	def eval_rf(self, xi):
		"""
		Input  :
			- xi: random seed in the Karhunen Loeve.
		Outputs:
			- theta: log Gaussian random field.
		"""
		xi__ = np.zeros(int(self.Nmesh * self.Nmesh))
		xi__[self.rank[:self.p]] = np.copy(xi)
		return self.eng.gaussrnd_coarse(matlab.double(xi__.reshape(int(self.Nmesh), -1).tolist()), self.alpha, self.tau, self.Nmesh)
