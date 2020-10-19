from __future__ import print_function
import os
import pickle
import numpy as np
import pandas as pd

from tqdm.autonotebook import tqdm
from scipy import integrate
from joblib import Parallel, delayed
import multiprocessing

class EnsembleKalmanMethod:

	def __init__(self, p, n_obs, J):
		self.n_obs     = n_obs        # Dimensionality of statistics (observations)
		self.p         = p            # Dimensionality of theta (parameters)
		self.J         = J            # Number of ensemble particles
		self.epsilon   = 1e-7         # Underflow protection
		self.T         = 30           # Number of maximum iterations
		self.num_cores = multiprocessing.cpu_count()
		self.parallel  = False
		self.mute_bar  = True

	def __str__(self):
		print(r'Number of parameters ................. %s'%(self.p))
		print(r'Dimension of forward model output .... %s'%(self.n_obs))
		print(r'Ensemble size ........................ %s'%(self.J))
		print(r'Parallel evaluation  ................. %s'%(self.parallel))
		print(r'Number of iterations to be run ....... %s'%(self.T))

		try:
			getattr(self, 'directory')
		except AttributeError:
			self.directory = os.getcwd()
		print('Path to save: ......................... %s'%('~/.../'+'/'.join(self.directory.split('/')[-2:])))

		try:
			__iterations = len(self.Uall) - 1
		except AttributeError:
			__iterations = 0
		print(r'Number of iterations EKS has run ..... %s'%(__iterations))

		return str()

	def run(self, y_obs, U0, model, Gamma, Jnoise):
		"""
		Apply Ensemble Kalman Method to solve a static inverse problem.
			y = G(theta) + eta.

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

	def ensemble_update(self, Geval):
		"""
		Perform a single step update for the ensemble
		"""
		pass

	def forward_model(self, theta, model):
		"""
		General evaluation function. To be used as a black box.
		It evaluates a single case.
		Inputs:
			- theta: [p, ] dimensional array
			- model: forward map name

		Output:
			- G(theta)
		"""
		G = model(theta)
		return G

	def ensemble_forward_model(self, theta, model):
		"""
		Evaluates the forward model for a collection of particles. If parallel
		is set to true, then it uses the available cores as initialized within the
		`EnsembleKalmanMethod` object.
		Inputs:
			- theta: [p, N] array where N is the number of particles to be evaluated
				and p is the dimensionality of the parameters.
			- model: model name for the forward model.
		"""
		if self.parallel:
			Geval = Parallel(n_jobs=self.num_cores)(delayed(self.forward_model)(k, model) for k in tqdm(theta.T,
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
				Gs[:, ii] = self.forward_model(k)
			return Gs

	def choose_initial_conditions(self, model, **kwargs):
		self.wt = kwargs.get('wt', None)
		self.t  = kwargs.get('t', None)
		if kwargs.get('ws', None) is not None:
			# Receives a history of initial conditions
			widx = np.random.randint(kwargs.get('ws').shape[0], size = self.J)
			self.W0 = kwargs.get('ws')[widx].T

			self.Wall = [];
			self.Wall.append(widx)

		else:
			# Receives a state for initial conditions
			self.W0 = np.tile(self.wt, self.J).reshape(self.J, model.n_state).T

	def forward_solve(self, k, model, t):
		"""
		Forward model for PDE constrained inverse problems. If parallel is set to
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

	def ensemble_forward_solve(self, theta, model, t):
		if self.parallel:
			Geval = Parallel(n_jobs=self.num_cores)(delayed(self.forward_solve)(k, model, t) for k in tqdm(theta.T,
					desc = 'Model evaluations: ',
					disable = self.mute_bar,
					leave = False,
					position = 1))
			return (np.asarray(Geval).T)
		else:
			Gs = np.zeros((self.n_obs + model.n_state, theta.shape[1]))
			for ii, k in enumerate(theta.T):
				Gs[:, ii] = self.forward_solve(k, model, t)
			return Gs

	def evaluate_ensemble_master(self, U0, model, **kwargs):
		# Evaluate the forward model / solver
		if model.type == 'pde':
			Geval = self.ensemble_forward_solve(np.vstack([U0, self.W0]), model, self.t)
			if kwargs.get('update_wt', True):
				if kwargs.get('ws', None) is not None:
					widx = np.random.randint(kwargs.get('ws').shape[0], size = self.J)
					self.Wall.append(widx)
					self.W0 = kwargs.get('ws')[widx].T
				else:
					self.W0 = np.copy(Geval[self.n_obs:,:])
		elif model.type == 'map':
			Geval = self.ensemble_forward_model(U0, model)

		if self.track_ensemble:
			self.Uall.append(U0)
			self.Gall.append(Geval)

		return Geval

	def postprocess_ensemble(self, U0, Geval):
		if self.track_ensemble:
			self.Uall = np.asarray(self.Uall);
			self.Gall = np.array(self.Gall)

		self.Ustar = U0
		self.Gstar = Geval[:self.n_obs,:]


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
					for iter in range(len(self.metrics['self-bias'])):
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

	def update_save_directory(self, **kwargs):
		try:
			getattr(self, 'directory')
		except AttributeError:
			self.directory = os.getcwd()

	def ensemble_save_online(self, model, iteration, save_online, **kwargs):
		if save_online:
			try:
				getattr(self, 'nexp')
				self.save(path = self.directory+'/ensembles/',
						  file = model.model_name + '-eks-' + \
						  			str(model.l_window).zfill(3)+ '-' + \
									str(self.J).zfill(4)+ '-' + \
									str(self.nexp).zfill(2) + '/',
						  online = True, counter = iteration)
			except AttributeError:
				self.save(path = self.directory+'/ensembles/',
						  file = model.model_name + '-eks-' + \
						  			str(model.l_window).zfill(3)+ '-' + \
									str(self.J).zfill(4) + '/',
						  online = True, counter = iteration)

# ------------------------------------------------------------------------------

class EnsembleKalmanSampler(EnsembleKalmanMethod):

	def __init__(self, p, n_obs, J, track_ensemble = True,
									track_metrics  = True):
		super().__init__(p, n_obs, J)
		self.update_save_directory()
		self.track_ensemble = track_ensemble
		self.track_metrics  = track_metrics

		if self.track_ensemble:
			self.Uall = [];
			self.Gall = [];

		if self.track_metrics:
			self.radspec = []
			self.metrics = dict()
			self.metrics['self-bias'] = []			# Tracks collapse in parameter space
			self.metrics['self-bias-data'] = []			# Tracks collapse after forward model evaln
			self.metrics['bias-data'] = []			# Tracks data-fitting
			self.metrics['bias'] = []			# Tracks the collapse towards the truth
			self.metrics['t'] = []

	def run(self, y_obs, U0, model, Gamma, Jnoise, save_online = False, trace = True, **kwargs):
		"""
		Run the Ensemble Kalman Sampler as described in:
			https://arxiv.org/pdf/1903.08866.pdf

		Inputs:
		- U0: np.array [p, J] initial ensemble; there are J
			ensemble particles each of dimension p.
		- y_obs: np.arrray [n_obs,] of observed data.
		- model: Forward model to be used. Currently for two types of problems.
			- 'pde' : pde/ode constrained forward models.
			- 'map' : input output functions.
		- Gamma: Noise covariance structure. np.array [n_obs, n_obs]
		- Jnoise: Pre-computed cholesky decomposition of Gamma.

		Optional (**pde model type only**):
		- wt: np.array of initial conditions to start the ensemble when
			evaluating the forward model.
		- t:  np.array of time points where the pde/ode is evaluated.

		Outputs:
		- None: all relevant objects are stored within the object.

		"""

		# Check type of model evaluation
		try:
			getattr(model, 'type')
		except AttributeError:
			raise

		# Check for directory to save ensemble
		self.update_save_directory(**kwargs)

		# Type of update to be used
		self.__update = kwargs.get('update', 'eks')

		# Setting for pde-type inverse problems
		if model.type == 'pde':
			self.choose_initial_conditions(model, **kwargs)

		# Evolving the ensemble
		for i in tqdm(range(self.T), desc = 'EKS iterations (%s):'%str(self.J), position = 1):
			# Evaluates G for the ensemble
			Geval = self.evaluate_ensemble_master(U0, model, **kwargs)
			# Retains the model evaluation and discards initial conditions (if pde)
			Geval = Geval[:self.n_obs,:]

			U0 = self.update_ensemble_master(y_obs, U0, Geval, Gamma, i, **kwargs)

			# Save the ensemble as it evolves
			self.ensemble_save_online(model, i, save_online, **kwargs)

			# Shall we continue to evolve the ensemble?
			if self.metrics['t'][-1] > kwargs.get('t_tol', 2.):
				break

		# Evaluating at the final iteration
		Geval = self.evaluate_ensemble_master(U0, model, **kwargs)

		# Finalize ensemble path
		self.postprocess_ensemble(U0, Geval)

		# File-storing ensemble directory.
		try:
			getattr(self, 'nexp')
			self.online_path = self.directory+'/ensembles/'+model.model_name + \
					'-' + str(self.J).zfill(4)+ '-' + str(self.nexp).zfill(2)+'/'
		except AttributeError:
			self.online_path = self.directory+'/ensembles/'+model.model_name + \
					'-' + str(self.J).zfill(4)+ '/'

	def update_ensemble_master(self, y_obs, U0, Geval, Gamma, i, **kwargs):
		if   self.__update == 'eks':
			U0 = self.ensemble_update(y_obs, U0, Geval, Gamma, i, **kwargs)
		elif self.__update == 'eks-linear':
			U0 = self.ensemble_update_linear(y_obs, U0, Geval, Gamma, i, **kwargs)
		elif self.__update == 'aldi':
			U0 = self.ensemble_update_linear_reich(y_obs, U0, Geval, Gamma, i, **kwargs)
		elif self.__update == 'eks-mix':
			U0 = self.ensemble_update_mix(y_obs, U0, Geval, Gamma, i, **kwargs)
		elif self.__update == 'eks-loo':
			U0 = self.ensemble_update_loo(y_obs, U0, Geval, Gamma, i, **kwargs)

		return U0

	def ensemble_update(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		"""
		self.update_rule = 'ensemble_update'
		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0, bias = True) + 1e-8 * np.identity(self.p)

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['self-bias'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['bias'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['self-bias-data'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['bias-data'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		hk = self.timestep_method(D, Geval, y_obs, Gamma, np.linalg.cholesky(Gamma), **kwargs)
		if kwargs.get('time_step', None) in ['adaptive', 'constant']:
			Cpp = np.cov(Geval, bias = True)
			D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(hk * Cpp + Gamma, R))

		Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
			U0 - hk * np.matmul(U0 - Umean, D)  + \
			hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)))
		Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
			np.random.normal(0, 1, [self.p, self.J])))

		return Uk

	def ensemble_update_linear(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		ALDI's linear correction.
		"""
		self.update_rule = 'ensemble_update_linear'

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['self-bias'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['bias'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['self-bias-data'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['bias-data'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		hk = self.timestep_method(D,  Geval, y_obs, Gamma, np.linalg.cholesky(Gamma), **kwargs)
		if kwargs.get('time_step', None) in ['adaptive', 'constant'] or \
		  (kwargs.get('time_step', None) == 'mix' and self.metrics['t'][-1] > 1):
			Cpp = np.cov(Geval, bias = True)
			D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(hk * Cpp + Gamma, R))

		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)
		try:
			np.linalg.cholesky(Ucov)
		except:
			print(self.metrics['t'][-1])
		alpha_J = ((self.p + 1.)/self.J)

		# ------------------     Implicit prior term  --------------------------
		# Ustar_ = np.linalg.solve(np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
		# 	U0 - hk * np.matmul(U0 - Umean, D)  + \
		# 	hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)) + \
		# 	1.0 * hk * alpha_J * (U0 - Umean))
		# Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
		# 	np.random.normal(0, 1, [self.p, self.J])))

		# ------------------     Implicit prior linear / term ------------------
		# Ustar_ = np.linalg.solve( (1 - hk * alpha_J) * np.eye(self.p) + hk * np.linalg.solve(self.sigma.T, Ucov.T).T,
		# 	U0 - hk * np.matmul(U0 - Umean, D)  + \
		# 	hk * np.matmul(Ucov, np.linalg.solve(self.sigma, self.mu)) - \
		# 	hk * alpha_J * Umean)
		# Uk     = (Ustar_ + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
		# 	np.random.normal(0, 1, [self.p, self.J])))

		# ------------------     Explicit as it can get ------------------------
		Uk = U0 - hk * np.matmul(U0 - Umean, D) - \
			hk * np.matmul(Ucov, np.linalg.solve(self.sigma, U0 - self.mu)) + \
			hk * alpha_J * (U0 - Umean) + \
			np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
				np.random.normal(0, 1, [self.p, self.J]))

		return Uk

	def ensemble_update_linear_reich(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		ALDI's linear correction.
		"""
		self.update_rule = 'ensemble_update_aldi'

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['self-bias'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['bias'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['self-bias-data'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['bias-data'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)
		alpha_J = ((self.p + 1.)/self.J)

		drift = - np.matmul(U0 - Umean, D) - \
			np.matmul(Ucov, np.linalg.solve(self.sigma, U0 - self.mu)) + \
			kwargs.get('switch', 1.) * alpha_J * (U0 - Umean);

		hk = 0.1/np.max(np.abs(drift));
		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		Uk = U0 + hk * drift + \
			np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov),
				np.random.normal(0, 1, [self.p, self.J]))

		return Uk

	def ensemble_update_mix(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKS.
		ALDI's linear correction.
		"""
		self.update_rule = 'ensemble_update_mix'
		eta_p   = np.random.normal(0, 1, [self.p, self.J])
		eta_d   = np.random.normal(0, 1, [self.n_obs, self.J])
		Umean = U0.mean(axis = 1)[:, np.newaxis]
		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		# For ensemble update
		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]
		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		# Track metrics
		self.metrics['self-bias'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['bias'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		self.metrics['self-bias-data'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		self.metrics['bias-data'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		hk = self.timestep_method(D,  Geval, y_obs, Gamma, np.linalg.cholesky(Gamma), **kwargs)
		if kwargs.get('time_step', None) in ['adaptive', 'constant']:
			Cpp = np.cov(Geval, bias = True)
			Cup = (1./self.J) * np.matmul(U0 - Umean, E.T)
			D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(hk * Cpp + Gamma, R))

		s_beta  = 1.0
		s_alpha = 0.01
		alpha_t = 1./(1.+np.exp(-(self.metrics['t'][-1] - s_beta)/s_alpha))
		alpha_J = ((self.p + 1.)/self.J)

		Uk = U0 - hk * np.matmul(U0 - Umean, D) - \
			hk * np.matmul(Ucov, np.linalg.solve(self.sigma, U0 - self.mu)) + \
			hk * alpha_J * (U0 - Umean) + \
			np.sqrt(2 * hk * alpha_t)   * np.matmul(np.linalg.cholesky(Ucov), eta_p) + \
			np.sqrt(hk * (1 - alpha_t)) * np.matmul(Cup, np.linalg.solve( hk * Cpp + Gamma,
				np.matmul(np.linalg.cholesky(Gamma), eta_d)
				)
			)

		return Uk

	def ensemble_update_loo(self, y_obs, U0, Geval, Gamma, iter, **kwargs):
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
		self.metrics['self-bias'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		self.metrics['bias'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		# self.metrics['self-bias-data'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		# self.metrics['bias-data'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		return Uk

	def timestep_method(self, D, Geval, y_obs, Gamma, Jnoise, **kwargs):
		if kwargs.get('time_step', None) is None:
			hk = 1./(np.linalg.norm(D) + 1e-8)
			# self.radspec.append(np.linalg.eigvals(D).real.max())
			# hk = 1./self.radspec[-1]
		elif kwargs.get('time_step') == 'spectral':
			self.radspec.append(np.linalg.eigvals(D).real.max())
			hk = 1./self.radspec[-1]
		elif kwargs.get('time_step') == 'constant':
			hk = kwargs.get('delta_t', 1./(self.T/2))
		elif kwargs.get('time_step') == 'adaptive':
			hk = self.LM_procedure(Geval, y_obs, Gamma, Jnoise, **kwargs)
		elif kwargs.get('time_step') == 'mix':
			if len(self.metrics['t']) == 0 or self.metrics['t'][-1] < kwargs.get('spinup', 4.):
				hk = 1./(np.linalg.norm(D) + 1e-8)
			else:
				hk = kwargs.get('delta_t', 1./(self.T/2))

		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		return hk

	def LM_procedure(self, Geval, y_obs, Gamma, Jnoise, **kwargs):
		rho_LM = kwargs.get('rho_LM', .5)

		Cpp   = np.cov(Geval, bias = True)
		Gmean = Geval.mean(axis = 1)

		lower_LM = rho_LM * np.linalg.norm(np.linalg.solve(Jnoise, Gmean - y_obs[:,np.newaxis]))
		alpha = 5.

		upper_LM = alpha * np.linalg.norm(
					np.matmul(Jnoise,
							np.linalg.solve(Cpp + alpha * Gamma,
								Gmean - y_obs[:,np.newaxis]
						)
					)
				)

		while upper_LM < lower_LM:
			alpha = 2 * alpha
			upper_LM = alpha * np.linalg.norm(
						np.matmul(Jnoise,
								np.linalg.solve(Cpp + alpha * Gamma,
									Gmean - y_obs[:,np.newaxis]
							)
						)
					)

		return 1./alpha

# ------------------------------------------------------------------------------

class EnsembleKalmanInversion(EnsembleKalmanMethod):

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
			self.metrics['self-bias'] = []			# Tracks collapse in parameter space
			self.metrics['self-bias-data'] = []			# Tracks collapse after forward model evaln
			self.metrics['bias-data'] = []			# Tracks data-fitting
			self.metrics['bias'] = []			# Tracks the collapse towards the truth
			self.metrics['t'] = []

		for i in tqdm(range(self.T), desc = 'EKS iterations (%s):'%str(self.J), position = 1):
			if model.type == 'pde':
				Geval = self.ensemble_forward_solve(np.vstack([U0, self.W0]), model, t)
				if kwargs.get('update_wt', True):
					if kwargs.get('ws', None) is not None:
						widx = np.random.randint(kwargs.get('ws').shape[0], size = self.J)
						self.Wall.append(widx)
						self.W0 = kwargs.get('ws')[widx].T
					else:
						self.W0 = np.copy(Geval[self.n_obs:,:])
			elif model.type == 'map':
				Geval = self.ensemble_forward_model(U0, model)
			else:
				break # Raise an error

			if trace:
				self.Uall.append(U0)
				self.Gall.append(Geval)

			Geval = Geval[:self.n_obs,:]

			if   self.__update == 'eki':
				U0 = self.ensemble_update(y_obs, U0, Geval, Gamma, Jnoise, i, **kwargs)
			elif self.__update == 'eki-flow':
				U0 = self.ensemble_update_flow(y_obs, U0, Geval, Gamma, Jnoise, i, model = model, **kwargs)

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
			Geval = self.ensemble_forward_solve(np.vstack([U0, self.W0]), model, t)
			if kwargs.get('update_wt', True):
				if kwargs.get('ws', None) is not None:
					self.W0 = kwargs.get('ws')[np.random.randint(kwargs.get('ws').shape[0], size = self.J)].T
				else:
					self.W0 = Geval[self.n_obs:,:]
		elif model.type == 'map':
			Geval = self.ensemble_forward_model(U0, model)

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
					'-' + str(self.J).zfill(4)+ '-' + str(self.nexp).zfill(2)+'/'
		except AttributeError:
			self.online_path = self.directory+'/ensembles/'+model.model_name + \
					'-' + str(self.J).zfill(4)+ '/'


	def ensemble_update(self, y_obs, U0, Geval, Gamma, Jnoise, iter, **kwargs):
		"""
		Ensemble update based on the continuous time limit of the EKI.
		"""
		if kwargs.get('time_step', None) is None:
			hk = 1.
		elif kwargs.get('time_step') == 'constant':
			hk = kwargs.get('delta_t', 1./self.T)

		# For ensemble update
		eta   = np.random.normal(0, 1, [self.n_obs, self.J])
		Umean = U0.mean(axis = 1)[:, np.newaxis]

		E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		R = Geval - y_obs[:,np.newaxis]

		Cpp = (1./self.J) * np.matmul(E, E.T)
		Cup = (1./self.J) * np.matmul(U0 - Umean, E.T)

		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma + hk * Cpp, R))

		# Track metrics
		# self.metrics['self-bias'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		# self.metrics['bias'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		# self.metrics['self-bias-data'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		# self.metrics['bias-data'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		# self.radspec.append(np.linalg.eigvals(D).real.max())
		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		Uk = U0 - \
			hk * np.matmul(U0 - Umean, D) + \
			kwargs.get('prior', False) * hk * np.matmul(Ucov, np.linalg.solve(self.sigma, U0 - self.mu)) + \
			np.sqrt(hk) * np.matmul(Cup, np.linalg.solve( hk * Cpp + Gamma,
				np.matmul(Jnoise, eta)
				)
			)


		return Uk

	def ensemble_update_flow(self, y_obs, U0, Geval, Gamma, Jnoise, iter, **kwargs):
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

		resid = y_obs[:,np.newaxis] - Geval.mean(axis = 1)[:,np.newaxis]
		hk = self.timestep_method([], Geval, y_obs, Gamma, Jnoise, **kwargs)

		D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(hk * Cpp + Gamma, R))

		# Track metrics
		# self.metrics['self-bias'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		# self.metrics['bias'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		# self.metrics['self-bias-data'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		# self.metrics['bias-data'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		# self.radspec.append(np.linalg.eigvals(D).real.max())

		dU = - hk * np.matmul(U0 - Umean, D)
		dW = np.sqrt(hk) * np.matmul(Cup,  np.linalg.solve(hk * Cpp + Gamma,
				np.matmul(Jnoise, eta)
				)
			)

		Uk = U0 + dU + dW

		return Uk

	def timestep_method(self, D, Geval, y_obs, Gamma, Jnoise, **kwargs):
		if kwargs.get('time_step', None) is None:
			hk = 1.
		elif kwargs.get('time_step') == 'norm':
			hk = 1./(np.linalg.norm(D) + 1e-8)
		elif kwargs.get('time_step') == 'constant':
			hk = kwargs.get('delta_t', 1./self.T)
		elif kwargs.get('time_step') == 'adaptive':
			hk = self.LM_procedure(Geval, y_obs, Gamma, Jnoise, **kwargs)

		if len(self.Uall) == 1:
			self.metrics['t'].append(hk)
		else:
			self.metrics['t'].append(hk + self.metrics['t'][-1])

		return hk

	def LM_procedure(self, Geval, y_obs, Gamma, Jnoise, **kwargs):
		rho_LM = kwargs.get('rho_LM', .5)

		Cpp   = np.cov(Geval, bias = True)
		Gmean = Geval.mean(axis = 1)

		lower_LM = rho_LM * np.linalg.norm(np.linalg.solve(Jnoise, Gmean - y_obs[:,np.newaxis]))
		alpha = 1.

		upper_LM = alpha * np.linalg.norm(np.matmul(Jnoise, np.linalg.solve(Cpp + alpha * Gamma, Gmean - y_obs[:,np.newaxis])))

		while upper_LM < lower_LM:
			alpha = 2 * alpha
			upper_LM = alpha * np.linalg.norm(np.matmul(Jnoise, np.linalg.solve(Cpp + alpha * Gamma, Gmean - y_obs[:,np.newaxis])))

		return 1./alpha

# ------------------------------------------------------------------------------
