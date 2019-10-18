from __future__ import print_function
import os
import pickle
import numpy as np
import pandas as pd
# import gpflow as gp

# from tqdm import tqdm
from tqdm.autonotebook import tqdm
from scipy import integrate
from joblib import Parallel, delayed
import multiprocessing

class enka(object):

	def __init__(self, p, n_obs, J):
		self.n_obs     = n_obs        # Dimensionality of statistics (observations)
		self.p         = p            # Dimensionality of theta (parameters)
		self.J         = J            # Number of ensemble particles
		self.epsilon   = 1e-7         # Underflow protection
		self.T         = 30           # Number of maximum iterations
		self.num_cores = multiprocessing.cpu_count()
		self.parallel  = False
		self.mute_bar  = True

	def __repr__(self):
		try:
			return 'enka' + '-' + str(self.J).zfill(4) + '-%s'%getattr(self.__update)
		except AttributeError:
			return 'enka' + '-' + str(self.J).zfill(4) + '-eks'

	def __str__(self):
		print(r'Number of parameters ................. %s'%(self.p))
		print(r'Dimension of forward model output .... %s'%(self.n_obs))
		print(r'Ensemble size ........................ %s'%(self.J))
		print(r'Evaluate G in parallel ............... %s'%(self.parallel))
		print(r'Number of iterations to be run ....... %s'%(self.T))

		try:
			getattr(self, 'directory')
		except AttributeError:
			self.directory = os.getcwd()
		print('Path to save: ......................... %s'%('~/.../'+'/'.join(self.directory.split('/')[-2:])))

		try:
			print(r'Number of iterations EKS has run ..... %s'%(len(self.Uall) - 1))
		except AttributeError:
			print(r'NOTE: EKS has not been run!')

		return str()

	def run(self, y_obs, U0, model, Gamma, Jnoise):
		"""
		Find the minimizer of an inverse problem using the continuous time limit
		of the EnKF.

		Inputs:
		- U0: A numpy array of shape (p, J) initial ensemble; there are J
			ensemble particles each of dimension p.
		- y_obs: A numpy array of shape (n_obs,) of observed data.
		- wt: A numpy array of initial conditions to start the ensemble when
			evaluating the forward model
		- t: A numpy array of time points where the ODE is evaluated
		- ...

		Outputs:
		- None
		"""
		pass

	def run_sde(self, y_obs, U0, model, Gamma, Jnoise):
		"""
		Find the minimizer of an inverse problem using the continuous time limit
		of the EnKF

		Inputs:
		- U0: A numpy array of shape (p, J) initial ensemble; there are J
			ensemble particles each of dimension p.
		- y_obs: A numpy array of shape (n_obs,) of observed data.
		- wt: A numpy array of initial conditions to start the ensemble when
			evaluating the forward model
		- t: A numpy array of time points where the ODE is evaluated
		- ...


		Outputs:
		- None
		"""
		pass

	def eks_update(self, Geval):
		"""
		Perform a single step update of the EKS algorithm.
		"""
		pass

	def G(self, theta, model):
		"""
		General evaluation function without partition of the model parameters.
		To be used as a black box. It evaluates single case.
		Inputs:
			- theta: [p, ] dimensional array
			- model: forward map name
		"""
		g = model(theta)
		return g

	def G_ens(self, theta, model):
		"""
		Evaluates for a collection of particles. If parallel is set to true, then
		it uses the available cores as initialized with the enka object.
		Inputs:
			- theta: [p, N] array where N is the number of particles to be evaluated
				and p is the dimensionality of the parameters.
			- model: model name for the forward model.
		"""
		if self.parallel:
			Geval = Parallel(n_jobs=self.num_cores)(delayed(self.G)(k, model) for k in tqdm(theta.T,
				desc = 'Model evaluations: ',
				disable = self.mute_bar,
				leave = False,
				position = 1))
			return (np.asarray(Geval).T)
		else:
			Gs = np.zeros((self.n_obs, theta.shape[1]))
			for ii, k in tqdm(enumerate(theta.T),
					desc = 'Model evaluations: ',
					disable = self.mute_bar,
					leave = True,
					position = 1):
				Gs[:, ii] = model(k)
			return Gs

	def G_pde(self, k, model, t):
		"""
		Forward model for PDE constrained inverse problem. If parallel is set to
		true, then it uses the available cores as initialized with the enka object.
		(WARNING): Hope to be more general now!

		Inputs:
			- k: [p + n_state, ], p is the dimensionality of the parameters, n_state
				is the dimension of the state used as initial conditions for
				the numerical integrator.
			- t: [t_eks,], array with computational times to evaluate the numerical
				integrator.
			- model: object that can solve the pde/ode equations and compute its
				relevant statistics.

		Outputs:
			- gs: [n_obs + n_state,] array.
		"""
		w0 = k[self.p:]

		ws = model.solve(w0, t, args = tuple(k[:self.p]))
		gs = model.statistics(ws)
		return np.concatenate([gs, ws[-1]])

	def G_pde_ens(self, theta, model, t):
		if self.parallel:
			Geval = Parallel(n_jobs=self.num_cores)(delayed(self.G_pde)(k, model, t) for k in tqdm(theta.T,
					desc = 'Model evaluations: ',
					disable = self.mute_bar,
					leave = False,
					position = 1))
			return (np.asarray(Geval).T)
		else:
			Gs = np.zeros((self.n_obs + model.n_state, theta.shape[1]))
			for ii, k in enumerate(theta.T):
				Gs[:, ii] = self.G_pde(k, model, t)
			return Gs

	def save(self, path = './', file = 'ces/', all = False, reset = True, online = False, counter = 0):
		"""
		All files are stored in pickle format
		Modes:
			- Last:    save only last stage (ensemble and model evaluations), and metrics
			- All:     save last stage, metrics, and ensemble path.
			- Online:  save metrics, ensemble and model evaluations as the algorithm progresses
		"""
		try:
			os.makedirs(path+file)
		except OSError:
			pass

		try:
			getattr(self, 'Uall')
			if not online:
				np.save(path+file+'ensemble' , self.Ustar)
				np.save(path+file+'Gensemble', self.Gstar)
				pickle.dump(self.metrics, open(path+file+'metrics.pkl', "wb"))
				if all:
					np.save(path+file+'ensemble_path' , self.Uall)
					np.save(path+file+'Gensemble_path', self.Gall)
			else:
				np.save(path+file+'ensemble_' +str(counter).zfill(4), self.Uall[-1])
				np.save(path+file+'Gensemble_'+str(counter).zfill(4), self.Gall[-1])
				pickle.dump(self.metrics, open(path+file+'metrics.pkl', "wb"))
		except AttributeError:
			tqdm.write('There is nothing to save')

	def load(self, path = './', eks_dir = 'ces/', ix_ensemble = False, flag_metrics = False):
		"""
		Load files and rebuild an eka object.
		"""
		files = os.listdir(path + eks_dir)
		try:
			self.metrics = pickle.load(open(path + eks_dir + 'metrics.pkl', 'rb'))
		except FileNotFoundError:
			print('Metrics object not found. Could not load EKS object.')
			return False

		if not ix_ensemble:
			try:
				self.Uall = np.load(path + eks_dir + 'ensemble_path.npy')
				self.Gall = np.load(path + eks_dir + 'Gensemble_path.npy')
			except FileNotFoundError:
				print('EKS trajectory files not found.')
				return False
		else:
			try:
				self.Uall = []
				self.Gall = []
				if flag_metrics:
					for iter in range(len(self.metrics['v'])):
						self.Uall.append(np.load(path + eks_dir + 'ensemble_'  + str(iter).zfill(4) + '.npy'))
						self.Gall.append(np.load(path + eks_dir + 'Gensemble_' + str(iter).zfill(4) + '.npy'))
				else:
					for iter in range(np.sum([file.split('_')[0] == 'ensemble' for file in (os.listdir(path + eks_dir))])):
						self.Uall.append(np.load(path + eks_dir + 'ensemble_'  + str(iter).zfill(4) + '.npy'))
						self.Gall.append(np.load(path + eks_dir + 'Gensemble_' + str(iter).zfill(4) + '.npy'))
				self.Uall = np.asarray(self.Uall)
				self.Gall = np.asarray(self.Gall)
				self.Ustar = self.Uall[-1]
				self.Gstar = self.Gall[-1]
				self.J = self.Uall.shape[-1]
			except FileNotFoundError:
				return False

		return True

# ------------------------------------------------------------------------------

class sampling(enka):

	def run(self, y_obs, U0, model, Gamma, Jnoise, save_online = False, trace = True, **kwargs):
		"""
		Find the minimizer of an inverse problem using the continuous time limit
		of the EKnF. The update can selected from a range of options.

		Inputs:
		- U0: A numpy array of shape (p, J) initial ensemble; there are J
			ensemble particles each of dimension p.
		- y_obs: A numpy array of shape (n_obs,) of observed data.
		- model: Forward model to be used. Currently for two types of problems.
			- 'pde' : pde / ode constrained forward models.
			- 'map' : input output functions.
		- Gamma: Noise covariance structure. Shape has to be (n_obs, n_obs)
		- Jnoise: Precomputed cholesky decomposition of Gamma.

		Optional (pde model type only):
		- wt: A numpy array of initial conditions to start the ensemble when
			evaluating the forward model
		- t: A numpy array of time points where the ODE is evaluated

		Outputs:
		- None
		"""
		try:
			getattr(model, 'type')
		except AttributeError:
			raise

		try:
			getattr(self, 'directory')
		except AttributeError:
			self.directory = os.getcwd()

		self.__update = kwargs.get('update', 'eks')

		if trace:
			try:
				getattr(self, 'Uall')
				self.Uall = list(self.Uall)
				self.Gall = list(self.Gall)

			except AttributeError:
				# Storing the ensemble members
				self.Uall = [];
				self.Gall = [];

		if model.type == 'pde':
			wt = kwargs.get('wt', None)
			t  = kwargs.get('t', None)
			if kwargs.get('ws', None) is not None:
				widx = np.random.randint(kwargs.get('ws').shape[0], size = self.J)
				self.W0 = kwargs.get('ws')[widx].T

				self.Wall = [];
				self.Wall.append(widx)
			else:
				self.W0 = np.tile(wt, self.J).reshape(self.J, model.n_state).T

		try :
			getattr(self, 'metrics')
		except AttributeError:
			# Storing metrics
			self.radspec = []
			self.metrics = dict()
			self.metrics['v'] = []			# Tracks collapse in parameter space
			self.metrics['V'] = []			# Tracks collapse after forward model evaln
			self.metrics['R'] = []			# Tracks data-fitting
			self.metrics['r'] = []			# Tracks the collapse towards the truth
			self.metrics['t'] = []

		for i in tqdm(range(self.T), desc = 'EKS iterations (%s):'%str(self.J), position = 1):
			if model.type == 'pde':
				Geval = self.G_pde_ens(np.vstack([U0, self.W0]), model, t)
				if kwargs.get('update_wt', True):
					if kwargs.get('ws', None) is not None:
						widx = np.random.randint(kwargs.get('ws').shape[0], size = self.J)
						self.Wall.append(widx)
						self.W0 = kwargs.get('ws')[widx].T
					else:
						self.W0 = np.copy(Geval[self.n_obs:,:])
			elif model.type == 'map':
				Geval = self.G_ens(U0, model)
			else:
				break # Raise an error

			if trace:
				self.Uall.append(U0)
				self.Gall.append(Geval)

			Geval = Geval[:self.n_obs,:]

			if   self.__update == 'eks':
				U0 = self.eks_update(y_obs, U0, Geval, Gamma, i)
			elif self.__update == 'eks-corrected':
				U0 = self.eks_update_corrected(y_obs, U0, Geval, Gamma, i)
			elif self.__update == 'eks-jacobian':
				U0 = self.eks_update_jac(y_obs, U0, Geval, Gamma, i, **kwargs)
			elif self.__update == 'eks-jacobian-corrected':
				U0 = self.eks_update_jac_cor(y_obs, U0, Geval, Gamma, i, **kwargs)
			elif self.__update == 'eks-jacobian-tempered':
				U0 = self.eks_update_jac_temp(y_obs, U0, Geval, Gamma, i, **kwargs)
			elif self.__update == 'eks-loo':
				U0 = self.eks_update_loo(y_obs, U0, Geval, Gamma, i, **kwargs)

			if save_online:
				try:
					getattr(self, 'nexp')
					self.save(path = self.directory+'/ensembles/',
							  file = model.model_name + '-eks-' + \
							  			str(model.l_window).zfill(3)+ '-' + \
										str(self.J).zfill(4)+ '-' + \
										str(self.nexp).zfill(2) + '/',
							  online = True, counter = i)
				except AttributeError:
					self.save(path = self.directory+'/ensembles/',
							  file = model.model_name + '-eks-' + \
							  			str(model.l_window).zfill(3)+ '-' + \
										str(self.J).zfill(4) + '/',
							  online = True, counter = i)

			if self.metrics['t'][-1] > 2:
				break

		if model.type == 'pde':
			Geval = self.G_pde_ens(np.vstack([U0, self.W0]), model, t)
			if kwargs.get('update_wt', True):
				if kwargs.get('ws', None) is not None:
					self.W0 = kwargs.get('ws')[np.random.randint(kwargs.get('ws').shape[0], size = self.J)].T
				else:
					self.W0 = Geval[self.n_obs:,:]
		elif model.type == 'map':
			Geval = self.G_ens(U0, model)

		if trace:
			self.Uall.append(U0)
			self.Gall.append(Geval);

			self.Uall = np.asarray(self.Uall);
			self.Gall = np.array(self.Gall)

		self.Ustar = U0
		self.Gstar = Geval[:self.n_obs,:]

		try:
			getattr(self, 'nexp')
			self.online_path = self.directory+'/ensembles/'+model.model_name + \
					'-eks-' + str(self.J).zfill(4)+ '-' + str(self.nexp).zfill(2)+'/'
		except AttributeError:
			self.online_path = self.directory+'/ensembles/'+model.model_name + \
					'-eks-' + str(self.J).zfill(4)+ '/'

	def eks_update(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		"""

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		if kwargs.get('adaptive', None) is None:
			self.radspec.append(np.linalg.eigvals(D).real.max())
			hk = 1./self.radspec[-1]
		elif kwargs.get('adaptive') == 'norm':
			hk = 1./(np.linalg.norm(D) + 1e-8)

		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
			U0 - hk * np.matmul(U0 - Umean, D)  + hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)))
		Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
			np.random.normal(0, 1, [self.p, self.J])))

		return Uk

	def eks_update_jac(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		"""
		model = kwargs.get('model', None)

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		self.radspec.append(np.linalg.eigvals(D).real.max())
		hk = 1./self.radspec[-1]

		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])
		Umean = U0.mean(axis = 1)[:, np.newaxis]

		if kwargs.get('delta', None) is not None:
			Ucov  = np.cov(U0) + kwargs.get('delta') * np.identity(self.p)
		else:
			Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		if model is not None:
			grad_logjacobian = model.grad_logjacobian(U0)
		else :
			grad_logjacobian = 0.0 * U0

		Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
			U0 - hk * np.matmul(U0 - Umean, D)  + hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)) + \
			hk * np.matmul(Ucov, grad_logjacobian))
		Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
			np.random.normal(0, 1, [self.p, self.J])))

		return Uk

	def eks_update_jac_cor(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		Uses the gradient of the logJacobian to correct for curvature.
		Uses the finite ensemble correction for the Langevin diffusion.
		"""
		model = kwargs.get('model', None)

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		self.radspec.append(np.linalg.eigvals(D).real.max())
		hk = 1./self.radspec[-1]

		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])
		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		if model is not None:
			grad_logjacobian = model.grad_logjacobian(U0)
		else :
			grad_logjacobian = 0.0 * U0

		Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
			U0 - hk * np.matmul(U0 - Umean, D)  + hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)) + \
			hk * np.matmul(Ucov, grad_logjacobian) + \
			hk * (self.p + 1)/self.J * (U0 - Umean))
		Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul(np.linalg.cholesky(Ucov),
			np.random.normal(0, 1, [self.p, self.J])))

		return Uk

	def eks_update_corrected(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		Reich correction
		"""

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		self.radspec.append(np.linalg.eigvals(D).real.max())

		hk = 1./self.radspec[-1]
		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])
		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
			U0 - hk * np.matmul(U0 - Umean, D)  + hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)) + \
			hk * (self.p + 1)/self.J * (U0 - Umean))
		Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
			np.random.normal(0, 1, [self.p, self.J])))

		return Uk

	def eks_update_loo(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		LOO: Leave-One-Out
		"""
		eta     = np.random.normal(0, 1, [self.p, self.J])
		Umean   = U0.mean(axis = 1)[:, np.newaxis]
		Gmean   = Geval.mean(axis = 1)[:,np.newaxis]
		Ucov    = np.cov(U0) + 1e-8 * np.identity(self.p)
		alpha   = 1./self.J
		D_bar   = np.zeros((self.J - 1, self.J))
		idx     = np.arange(self.J)
		R       = Geval - y_obs[:,np.newaxis]
		Gamma_R = np.linalg.solve(Gamma, R)

		# 1) Build D_bar
		for j in range(self.J):
			Gmean_j    = 1/(1-alpha) * np.copy(Gmean - alpha * Geval[:,j].reshape(-1,1))
			E_j        = np.copy(Geval[:, idx != j] - Gmean_j)
			D_bar[:,j] = 1/(self.J-1) * np.matmul(E_j.T, Gamma_R[:,j])
		self.D_bar = D_bar

		# 2) Compute norm of D_bar
		norm_D = np.linalg.norm(D_bar)
		# norm_D = np.sqrt(self.J * (D_bar**2).sum(axis = 1).mean())
		hk     = 1./(norm_D + 1e-8)

		# 3) Update ensemble members
		Uk = np.zeros_like(U0)
		for j in range(self.J):
			Gmean_j = 1/(1-alpha) * np.copy(Gmean - alpha * Geval[:,j].reshape(-1,1))
			Umean_j = 1/(1-alpha) * np.copy(Umean - alpha * U0[:,j].reshape(-1,1))
			delta_j = U0[:,j].reshape(-1,1) - Umean_j
			Ucov_j  = 1/(1-alpha) * Ucov - alpha * delta_j.dot(delta_j.T)
			Ustar_  = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov_j.T).T,
				U0[:,j].reshape(-1,1) - hk * np.matmul(U0[:,idx != j] - Umean_j, D_bar[:,j].reshape(-1,1)) + \
				hk * np.matmul(Ucov_j, np.linalg.solve(self.sigma, self.mu)))
			Uk[:,j] = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov_j), eta[:,j].reshape(-1,1))).flatten()

		# 4) Build metrics
		self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		# self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		# self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		return Uk

	def eks_update_jac_temp(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		"""
		model = kwargs.get('model', None)
		beta_inv  = 1./kwargs.get('beta',  1.0)

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		self.radspec.append(np.linalg.eigvals(D).real.max())
		hk = 1./self.radspec[-1]

		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])
		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		if model is not None:
			grad_logjacobian = model.grad_logjacobian(U0)
		else :
			grad_logjacobian = 0.0 * U0

		Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
			U0 - hk * np.matmul(U0 - Umean, D)  + hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)) + \
			hk * np.matmul(Ucov, grad_logjacobian))
		Uk     = (Ustar_ + np.sqrt(2*hk*beta_inv) * np.matmul( np.linalg.cholesky(Ucov),
			np.random.normal(0, 1, [self.p, self.J])))

		return Uk

# ------------------------------------------------------------------------------

class inversion(enka):

	def run(self, y_obs, U0, model, Gamma, Jnoise, save_online = False, trace = True, **kwargs):
		"""
		Find the minimizer of an inverse problem using the continuous time limit
		of the EKnF. The update can selected from a range of options.

		Inputs:
		- U0: A numpy array of shape (p, J) initial ensemble; there are J
			ensemble particles each of dimension p.
		- y_obs: A numpy array of shape (n_obs,) of observed data.
		- model: Forward model to be used. Currently for two types of problems.
			- 'pde' : pde / ode constrained forward models.
			- 'map' : input output functions.
		- Gamma: Noise covariance structure. Shape has to be (n_obs, n_obs)
		- Jnoise: Precomputed cholesky decomposition of Gamma.

		Optional (pde model type only):
		- wt: A numpy array of initial conditions to start the ensemble when
			evaluating the forward model
		- t: A numpy array of time points where the ODE is evaluated

		Outputs:
		- None
		"""
		try:
			getattr(model, 'type')
		except AttributeError:
			raise

		try:
			getattr(self, 'directory')
		except AttributeError:
			self.directory = os.getcwd()

		self.__update = kwargs.get('update', 'eki')

		if trace:
			try:
				getattr(self, 'Uall')
				self.Uall = list(self.Uall)
				self.Gall = list(self.Gall)

			except AttributeError:
				# Storing the ensemble members
				self.Uall = [];
				self.Gall = [];

		if model.type == 'pde':
			wt = kwargs.get('wt', None)
			t  = kwargs.get('t', None)
			if kwargs.get('ws', None) is not None:
				widx = np.random.randint(kwargs.get('ws').shape[0], size = self.J)
				self.W0 = kwargs.get('ws')[widx].T

				self.Wall = [];
				self.Wall.append(widx)
			else:
				self.W0 = np.tile(wt, self.J).reshape(self.J, model.n_state).T

		try :
			getattr(self, 'metrics')
		except AttributeError:
			# Storing metrics
			self.radspec = []
			self.metrics = dict()
			self.metrics['v'] = []			# Tracks collapse in parameter space
			self.metrics['V'] = []			# Tracks collapse after forward model evaln
			self.metrics['R'] = []			# Tracks data-fitting
			self.metrics['r'] = []			# Tracks the collapse towards the truth
			self.metrics['t'] = []

		for i in tqdm(range(self.T), desc = 'EKS iterations (%s):'%str(self.J), position = 1):
			if model.type == 'pde':
				Geval = self.G_pde_ens(np.vstack([U0, self.W0]), model, t)
				if kwargs.get('update_wt', True):
					if kwargs.get('ws', None) is not None:
						widx = np.random.randint(kwargs.get('ws').shape[0], size = self.J)
						self.Wall.append(widx)
						self.W0 = kwargs.get('ws')[widx].T
					else:
						self.W0 = np.copy(Geval[self.n_obs:,:])
			elif model.type == 'map':
				Geval = self.G_ens(U0, model)
			else:
				break # Raise an error

			if trace:
				self.Uall.append(U0)
				self.Gall.append(Geval)

			Geval = Geval[:self.n_obs,:]

			if   self.__update == 'eki':
				U0 = self.eki_update(y_obs, U0, Geval, Gamma, Jnoise, i, **kwargs)
			elif self.__update == 'eki-corrected':
				U0 = self.eki_update_corrected(y_obs, U0, Geval, Gamma, Jnoise, i, **kwargs)
			elif self.__update == 'eki-flow':
				U0 = self.eki_update_flow(y_obs, U0, Geval, Gamma, Jnoise, i, model = model)
			elif self.__update == 'eki-jacobian':
				U0 = self.eki_update_jacobian(y_obs, U0, Geval, Gamma, Jnoise, i, **kwargs)

			if save_online:
				try:
					getattr(self, 'nexp')
					self.save(path = self.directory+'/ensembles/',
							  file = model.model_name + '-eki-' + \
										str(model.l_window).zfill(3)+ '-' + \
										str(self.J).zfill(4)+ '-' + \
										str(self.nexp).zfill(2) + '/',
							  online = True, counter = i)
				except AttributeError:
					self.save(path = self.directory+'/ensembles/',
							  file = model.model_name + '-eki-' + \
										str(model.l_window).zfill(3)+ '-' + \
										str(self.J).zfill(4) + '/',
							  online = True, counter = i)

		if model.type == 'pde':
			Geval = self.G_pde_ens(np.vstack([U0, self.W0]), model, t)
			if kwargs.get('update_wt', True):
				if kwargs.get('ws', None) is not None:
					self.W0 = kwargs.get('ws')[np.random.randint(kwargs.get('ws').shape[0], size = self.J)].T
				else:
					self.W0 = Geval[self.n_obs:,:]
		elif model.type == 'map':
			Geval = self.G_ens(U0, model)

		if trace:
			self.Uall.append(U0)
			self.Gall.append(Geval);

			self.Uall = np.asarray(self.Uall);
			self.Gall = np.array(self.Gall)

		self.Ustar = U0
		self.Gstar = Geval[:self.n_obs,:]

		try:
			getattr(self, 'nexp')
			self.online_path = self.directory+'/ensembles/'+model.model_name + \
					'-eki-' + str(self.J).zfill(4)+ '-' + str(self.nexp).zfill(2)+'/'
		except AttributeError:
			self.online_path = self.directory+'/ensembles/'+model.model_name + \
					'-eki-' + str(self.J).zfill(4)+ '/'


	def eki_update(self, y_obs, U0, Geval, Gamma, Jnoise, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKI.
		"""
		if kwargs.get('adaptive', None) is None:
			hk = 1.
		elif kwargs.get('adaptive') == 'norm':
			hk = 1./self.T

		# For ensemble update
		eta   = np.random.normal(0, 1, [self.n_obs, self.J])
		Umean = U0.mean(axis = 1)[:, np.newaxis]

		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]

		Cpp = (1./self.J) * np.matmul(E, E.T)
		Cup = (1./self.J) * np.matmul(U0 - Umean, E.T)

		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma + hk * Cpp, R))

		# Track metrics
		# self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		# self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		# self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		# self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		# self.radspec.append(np.linalg.eigvals(D).real.max())
		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		dU = - hk * np.matmul(U0 - Umean, D)
		dW = np.sqrt(hk) * np.matmul(Cup, np.linalg.solve( hk * Cpp + Gamma,
				np.matmul(Jnoise, eta)
				)
			)

		Uk = U0 + dU + dW

		return Uk

	def eki_update_flow(self, y_obs, U0, Geval, Gamma, Jnoise, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKI.
		"""
		# For ensemble update
		eta   = np.random.normal(0, 1, [self.n_obs, self.J])
		Umean = U0.mean(axis = 1)[:, np.newaxis]

		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]

		Cpp = (1./self.J) * np.matmul(E, E.T)
		Cup = (1./self.J) * np.matmul(U0 - Umean, E.T)

		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))
		self.radspec.append(np.linalg.eigvals(D).real.max())

		hk = 1./self.radspec[-1]
		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		# Track metrics
		# self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		# self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		# self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		# self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		# self.radspec.append(np.linalg.eigvals(D).real.max())

		dU = - hk * np.matmul(U0 - Umean, D)
		dW = np.sqrt(hk) * np.matmul(Cup, np.linalg.solve(Gamma,
				np.matmul(Jnoise, eta)
				)
			)

		Uk = U0 + dU + dW

		return Uk

	def eki_update_jac(self, y_obs, U0, Geval, Gamma, Jnoise, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		Don't use for inversion! PLEASE!!! :8-)
		Still to implement grad logjacobian for EKI
		"""
		model = kwargs.get('model', None)

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		self.radspec.append(np.linalg.eigvals(D).real.max())

		hk = 1./self.radspec[-1]
		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])
		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		if model is not None:
			grad_logjacobian = model.grad_logjacobian(U0)
		else :
			grad_logjacobian = 0.0 * U0

		Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
			U0 - hk * np.matmul(U0 - Umean, D)  + hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)) + \
			hk * np.matmul(Ucov, grad_logjacobian))
		Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
			np.random.normal(0, 1, [self.p, self.J])))

		return np.zeros_like(U0)

	def eki_update_jacobian(self, y_obs, U0, Geval, Gamma, Jnoise, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKI.
		Still to implement grad logjacobian for EKI
		"""
		# For ensemble update
		eta = np.random.normal(0, 1, [self.n_obs, self.J])
		E   = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R   = U0 - U0.mean(axis = 1)[:,np.newaxis]

		Cpp = (1./self.J) * np.matmul(E, E.T)
		Cup = (1./self.J) * np.matmul(R, E.T)

		# Track metrics
		# self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		# self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		# self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		# self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		if kwargs.get('adaptive', None) is None:
			hk = 1.
		elif kwargs.get('adaptive') == 'norm':
			hk = 1./self.T

		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		# This is a temporal solution is very hardcoded! -----------------------
		# Jacobian = 0.0 * U0
		# Jacobian[2] = -np.exp(-U0[2])
		# ----------------------------------------------------------------------

		# dU = - np.matmul(Cup, np.linalg.solve( hk * Cpp + Gamma, y_obs[:,np.newaxis] + \
		# 	np.matmul(np.sqrt(self.T) * Jnoise, eta) - Geval))
		# Uk = U0 - hk * dU

		dU = hk * np.matmul(Cup, np.linalg.solve(hk * Cpp + Gamma, y_obs[:,np.newaxis] - Geval))
		dW = np.sqrt(hk) * np.matmul(Cup, np.linalg.solve( hk * Cpp + Gamma, np.matmul(Jnoise, eta)))
		Uk = U0 + dU + dW
		# hk * np.matmul(Ucov, Jacobian))

		return Uk

# ------------------------------------------------------------------------------
