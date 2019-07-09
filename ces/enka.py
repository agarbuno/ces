from __future__ import print_function
import numpy as np
import pandas as pd
import gpflow as gp

from tqdm import tqdm
from scipy import integrate
from joblib import Parallel, delayed
import multiprocessing

class eki(object):

	def __init__(self, p, n_obs, J, l_window = 0, freq = 0):
		self.n_obs = n_obs		    # Dimensionality of statistics (observations)
		self.p = p                # Dimensionality of theta (parameters)
		self.J = J                # Number of ensemble particles
		self.epsilon = 1e-7       # Underflow protection
		self.T = 30
		self.num_cores = multiprocessing.cpu_count()
		self.l_window = l_window
		self.freq = freq
		self.scaled = False

	def run_sde(self, y_obs, U0, model, Gamma, Jnoise, verbose = True):
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
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """

	def run(self, y_obs, U0, model, Gamma, Jnoise, verbose = True):
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
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """
		pass

	def run_pde(self, y_obs, U0, wt, t, model, Gamma, Jnoise, verbose = True):
		"""
	    Find the minimizer of an inverse problem using the continuous time limit
	    of the EnKF. For PDE constrained inversion problems.

	    Inputs:
	    - U0: A numpy array of shape (p, J) initial ensemble; there are J
	      	ensemble particles each of dimension p.
	    - y_obs: A numpy array of shape (n_obs,) of observed data.
	    - wt: A numpy array of initial conditions to start the ensemble when
	    	evaluating the forward model
	    - t: A numpy array of time points where the ODE is evaluated
	    - ...
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """
		pass

	def run_data(self, y_obs, data, U0, wt, t, model, Gamma, Jnoise, verbose = True):
		"""
	    Find the minimizer of an inverse problem using the continuous time limit
	    of the EnKF. Data averages are provided as a stream.

	    Inputs:
	    - U0: A numpy array of shape (p, J) initial ensemble; there are J
	      	ensemble particles each of dimension p.
	    - data: A numpy array of shape (n_obs, n_samps) of observed data.
	    - wt: A numpy array of initial conditions to start the ensemble when
	    	evaluating the forward model
	    - t: A numpy array of time points where the ODE is evaluated
	    - ...
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """
		pass

	def G(self, k, model):
		"""
		General evaluation function without partition of the model parameters
		"""
		g = model(k)
		return g

	def Gpar(self, theta, model):
	    Geval = Parallel(n_jobs=self.num_cores)(delayed(self.G)(k, model) for k in theta.T)
	    return (np.asarray(Geval).T)

	def Gnopar(self, theta, model):
	    Gs = np.zeros((self.n_obs, self.J))

	    for ii, k in enumerate(theta.T):
	    	Gs[:, ii] = model(k)

	    return Gs

	def G_pde(self, k, model, t):
		r, b = k[:self.p]
		w0 = k[self.p:]

		ws = integrate.odeint(model, w0, t, args = (r, b))
		xs, ys, zs = ws[:,0], ws[:,1], ws[:,2]
		ws = [xs, ys, zs, xs**2, ys**2, zs**2, xs*ys, xs*zs, ys*zs]

		# With rolling statistics
		gs = [np.asarray(pd.Series(k).rolling(window = self.l_window * self.freq ).mean()) for k in ws]
		gs = np.asarray(gs)[:,-1]

		# With adjacent windows
		# gs = np.array([k[1:].reshape(-1, self.l_window * self.freq).mean(axis = 1) for k in ws])
		# gs = gs[:,-1]


		return np.concatenate([gs, np.asarray([xs[-1], ys[-1], zs[-1]])])

	def Gpar_pde(self, theta, model, t):
	    Geval = Parallel(n_jobs=self.num_cores)(delayed(self.G)(k, model, t) for k in theta.T)
	    return (np.asarray(Geval).T)

	def scale_ensemble(self, factor = 2.):
		self.scale_mean = self.Ustar.mean(axis = 1)[:, np.newaxis]
		self.scale_cov = factor * np.linalg.cholesky(np.cov(self.Ustar))
		self.scale_X = np.linalg.solve(self.scale_cov, self.Ustar - self.scale_mean)
		self.scaled = True

	def predict_gps(self, X):
		"""
		Prediction using independent GP models for multioutput code.
		Eventually, we either decorrelate or learn a multioutput emulator.

		Inputs:
		- gpmodels: list of independent GP models
		- X: numpy array with dimensions [n_points, d]
		"""
		gpmeans = np.empty(shape = (len(self.gpmodels), len(X)))
		gpvars = np.empty(shape = (len(self.gpmodels), len(X)))

		for ii, model in enumerate(self.gpmodels):
			if not self.scaled:
				mean_pred, var_pred = model.predict_y(X)
			else:
				mean_pred, var_pred = model.predict_y(np.linalg.solve(self.scale_cov, X.T - self.scale_mean).T)
			gpmeans[ii,:] = mean_pred.flatten()
			gpvars[ii,:] = var_pred.flatten()

		return [gpmeans, gpvars]

# ------------------------------------------------------------------------------

class flow(eki):

	def run(self, y_obs, U0, model, Gamma, Jnoise, verbose = True):
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
		- verbose: (boolean) If true, print progress during iterations.

		Outputs:
		- None
		"""
		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = [] 			# Tracks the collapse towards the truth
		self.t = []

		for i in tqdm(range(self.T)):
		    Geval = self.Gpar(U0, model)

		    # For ensemble update
		    E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		    R = Geval - y_obs[:,np.newaxis]
		    D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		    # Track metrics
		    self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		    self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		    self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		    self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		    # Compute the update
		    self.radspec.append(np.linalg.eigvals(D).real.max())
		    hk = 1./(np.linalg.norm(D) + 1e-8)
		    #hk = 1./self.radspec[-1]
		    if i == 0:
		    	self.t.append(hk)
		    else:
		    	self.t.append(hk + self.t[-1])
		    Umean = U0.mean(axis = 1)[:, np.newaxis]
		    Ucov  = np.cov(U0)

		    Ustar = np.linalg.solve(np.eye(self.p) + hk/(self.sigma**2) * Ucov,
		    	U0 - hk * np.matmul(U0 - Umean, D) + hk/(self.sigma**2) * np.matmul(Ucov, self.mu))
		    Uk = Ustar + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov), np.random.normal(0, 1, [self.p, self.J]))

		    self.Uall.append(Uk)
		    U0 = Uk

		self.Uall = np.asarray(self.Uall)
		self.Ustar = self.Uall[-1]
		self.Gstar = self.Gpar(self.Ustar, model)

	def run_nopar(self, y_obs, U0, model, Gamma, Jnoise, verbose = True):
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
		- verbose: (boolean) If true, print progress during iterations.

		Outputs:
		- None
		"""
		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = [] 			# Tracks the collapse towards the truth
		self.t = []

		for i in tqdm(range(self.T), desc = 'Ens.size: %s'%(self.J)):
		    Geval = self.Gnopar(U0, model)

		    # For ensemble update
		    E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		    R = Geval - y_obs[:,np.newaxis]
		    D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))

		    # Track metrics
		    self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		    self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		    self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		    self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		    # Compute the update
		    self.radspec.append(np.linalg.eigvals(D).real.max())
		    #hk = 1./(np.linalg.norm(D) + 1e-8)
		    hk = 1./self.radspec[-1]
		    #hk = 0.01

		    if i == 0:
		    	self.t.append(hk)
		    else:
		    	self.t.append(hk + self.t[-1])
		    Umean = U0.mean(axis = 1)[:, np.newaxis]
		    Ucov  = np.cov(U0) + 1e-8 * np.identity(self.p)

		    Ustar = np.linalg.solve(np.eye(self.p) + hk/(self.sigma**2) * Ucov,
		    	U0 - hk * np.matmul(U0 - Umean, D) + hk/(self.sigma**2) * np.matmul(Ucov, self.mu))
		    Uk = Ustar + np.sqrt(2*hk) * np.matmul( np.linalg.cholesky(Ucov), np.random.normal(0, 1, [self.p, self.J]))

		    self.Uall.append(Uk)
		    U0 = Uk

		    if self.t[-1] > 2:
		    	break

		self.Uall = np.asarray(self.Uall)
		self.Ustar = self.Uall[-1]
		self.Gstar = self.Gnopar(self.Ustar, model)

	def run_sde(self, y_obs, U0, model, Gamma, Jnoise, verbose = True):
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
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """

		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = [] 			# Tracks the collapse towards the truth
		self.t = []                 # Tracks simulated time

		for i in tqdm(range(self.T)):
		    Geval = self.Gpar(U0, model)

		    # For ensemble update
		    E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		    R = Geval - y_obs[:,np.newaxis]
		    D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))
		    S = -(1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, \
		    		np.matmul(Jnoise, (np.random.normal(0,1,[self.n_obs, self.J])))
		    		)
		    	)

		    # For tracking metrics
		    self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		    self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		    self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		    self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		    self.radspec.append(np.linalg.eigvals(D).real.max())
		    # hk = 1./(np.linalg.norm(D + S) + 1e-8)
		    hk = 1./self.radspec[-1]
		    if i == 0:
		    	self.t.append(hk)
		    else:
		    	self.t.append(hk + self.t[-1])
		    # Explicit
		    # Uk = U0 - hk * np.matmul(U0, D)
		    Umean = U0.mean(axis = 1)[:, np.newaxis]
		    # Uk = U0 - hk * np.matmul(U0 - Umean, D + S)
		    Uk = U0 - hk * np.matmul(U0 - Umean, D) - np.sqrt(2 * hk) * np.matmul(U0 - Umean, S)
		    # Uk[Uk<0.] = 0.
		    # Implicit
		    # Uk = np.linalg.solve(np.eye(J) + hk * D.T, U0.T).T
		    # Uk = np.linalg.solve(np.eye(J) + hk * D.T, (U0 - U0.mean(axis = 1)[:, np.newaxis]).T).T + U0.mean(axis = 1)[:, np.newaxis]
		    # Uk = np.linalg.solve(np.eye(J) + (hk/2) * D.T, np.matmul(np.eye(J) - (hk/2) * D.T, U0.T)).T
		    # Uk = U0 - (hk / (1 + hk * np.diag(D))) * np.matmul(U0, D)

		    self.Uall.append(Uk)
		    U0 = Uk

		self.Uall = np.asarray(self.Uall)

	def run_sdenopar(self, y_obs, U0, model, Gamma, Jnoise, verbose = True):
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
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """

		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = [] 			# Tracks the collapse towards the truth
		self.t = []                 # Tracks simulated time

		for i in tqdm(range(self.T)):
		    Geval = self.Gnopar(U0, model)

		    # For ensemble update
		    E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		    R = Geval - y_obs[:,np.newaxis]
		    D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))
		    S = -(1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, \
		    		np.matmul(Jnoise, (np.random.normal(0,1,[self.n_obs, self.J])))
		    		)
		    	)

		    # For tracking metrics
		    self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		    self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		    self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		    self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		    self.radspec.append(np.linalg.eigvals(D).real.max())
		    # hk = 1./(np.linalg.norm(D + S) + 1e-8)
		    hk = 1./self.radspec[-1]
		    if i == 0:
		    	self.t.append(hk)
		    else:
		    	self.t.append(hk + self.t[-1])
		    # Explicit
		    # Uk = U0 - hk * np.matmul(U0, D)
		    Umean = U0.mean(axis = 1)[:, np.newaxis]
		    # Uk = U0 - hk * np.matmul(U0 - Umean, D + S)
		    Uk = U0 - hk * np.matmul(U0 - Umean, D) - np.sqrt(2 * hk) * np.matmul(U0 - Umean, S)
		    # Uk[Uk<0.] = 0.
		    # Implicit
		    # Uk = np.linalg.solve(np.eye(J) + hk * D.T, U0.T).T
		    # Uk = np.linalg.solve(np.eye(J) + hk * D.T, (U0 - U0.mean(axis = 1)[:, np.newaxis]).T).T + U0.mean(axis = 1)[:, np.newaxis]
		    # Uk = np.linalg.solve(np.eye(J) + (hk/2) * D.T, np.matmul(np.eye(J) - (hk/2) * D.T, U0.T)).T
		    # Uk = U0 - (hk / (1 + hk * np.diag(D))) * np.matmul(U0, D)

		    self.Uall.append(Uk)
		    U0 = Uk

		self.Uall = np.asarray(self.Uall)
		self.Ustar = self.Uall[-1]
		self.Gstar = self.Gnopar(self.Ustar, model)

	def run_pde(self, y_obs, U0, wt, t, model, Gamma, Jnoise, verbose = True):
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
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """
		self.W0 = np.tile(wt, self.J).reshape(self.J, 3).T

		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = [] 			# Tracks the collapse towards the truth
		self.t = []

		for i in tqdm(range(self.T)):
		    Geval = self.Gpar(np.vstack([U0, self.W0]), model, t)
		    self.W0 = Geval[self.n_obs:,:]
		    Geval = Geval[:self.n_obs,:]

		    # For ensemble update
		    E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		    R = Geval - y_obs[:,np.newaxis]
		    D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))
		    S = -(1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, \
		    		np.matmul(Jnoise, (np.random.normal(0,1,[self.n_obs, self.J])))
		    		)
		    	)

		     # For tracking metrics
		    self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		    self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
		    self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		    self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())

		    self.radspec.append(np.linalg.eigvals(D).real.max())
		    hk = 1./self.radspec[-1]
		    if i == 0:
		    	self.t.append(hk)
		    else:
		    	self.t.append(hk + self.t[-1])
		    # Explicit
		    # Uk = U0 - hk * np.matmul(U0, D)
		    Umean = U0.mean(axis = 1)[:, np.newaxis]
		    #Uk = np.abs(U0 - hk * np.matmul(U0 - Umean, D + S))
		    Uk = np.abs(U0 - hk * np.matmul(U0 - Umean, D) - np.sqrt(2*hk) * np.matmul(U0 - Umean, S))
		    # Uk[Uk<0.] = 0.
		    # Implicit
		    # Uk = np.linalg.solve(np.eye(J) + hk * D.T, U0.T).T
		    # Uk = np.linalg.solve(np.eye(J) + hk * D.T, (U0 - U0.mean(axis = 1)[:, np.newaxis]).T).T + U0.mean(axis = 1)[:, np.newaxis]
		    # Uk = np.linalg.solve(np.eye(J) + (hk/2) * D.T, np.matmul(np.eye(J) - (hk/2) * D.T, U0.T)).T
		    # Uk = U0 - (hk / (1 + hk * np.diag(D))) * np.matmul(U0, D)

		    self.Uall.append(Uk)
		    U0 = Uk

		self.Uall = np.asarray(self.Uall)

	def run_data(self, y_obs, data, U0, wt, t, model, Gamma, Jnoise, verbose = True):
		"""
	    Find the minimizer of an inverse problem using the continuous time limit
	    of the EnKF

	    Inputs:
	    - U0: A numpy array of shape (p, J) initial ensemble; there are J
	      	ensemble particles each of dimension p.
	    - data: A numpy array of shape (n_obs, n_samps) of observed data.
	    - wt: A numpy array of initial conditions to start the ensemble when
	    	evaluating the forward model
	    - t: A numpy array of time points where the ODE is evaluated
	    - ...
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """
		self.W0 = np.tile(wt, self.J).reshape(self.J, 3).T

		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = []
		self.t = []

		for i in tqdm(range(self.T)):
		    Geval = self.Gpar(np.vstack([U0, self.W0]), model, t)
		    self.W0 = Geval[self.n_obs:,:]
		    Geval = Geval[:self.n_obs,:]

		    # For ensemble update
		    E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
		    R = Geval - y_obs[:,np.newaxis]
		    D =  (1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, R))
		    S = -(1.0/self.J) * np.matmul(E.T, np.linalg.solve(Gamma, \
		    		y_obs[:,np.newaxis] - data[:,np.random.randint(data.shape[1])][:,np.newaxis]
		    		)
		    	)

		     # For tracking metrics
		    self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
		    self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
		    self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
		    self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())

		    self.radspec.append(np.linalg.eigvals(D + S).real.max())
		    hk = 1./self.radspec[-1]
		    if i == 0:
		    	self.t.append(hk)
		    else:
		    	self.t.append(hk + self.t[-1])
		    # Explicit
		    # Uk = U0 - hk * np.matmul(U0, D)
		    Umean = U0.mean(axis = 1)[:, np.newaxis]
		    # Uk = np.abs(U0 - hk * np.matmul(U0 - Umean, D + S))
		    Uk = np.abs(U0 - hk * np.matmul(U0 - Umean, D + S))
		    # Uk[Uk<0.] = 0.
		    # Implicit
		    # Uk = np.linalg.solve(np.eye(J) + hk * D.T, U0.T).T
		    # Uk = np.linalg.solve(np.eye(J) + hk * D.T, (U0 - U0.mean(axis = 1)[:, np.newaxis]).T).T + U0.mean(axis = 1)[:, np.newaxis]
		    # Uk = np.linalg.solve(np.eye(J) + (hk/2) * D.T, np.matmul(np.eye(J) - (hk/2) * D.T, U0.T)).T
		    # Uk = U0 - (hk / (1 + hk * np.diag(D))) * np.matmul(U0, D)

		    self.Uall.append(Uk)
		    U0 = Uk

		self.Uall = np.asarray(self.Uall)

# ------------------------------------------------------------------------------

class iterative(eki):

	def run_pde(self, y_obs, U0, wt, t, model, Gamma, Jnoise, verbose = True):
		"""
	    Find the minimizer of an inverse problem using the iterative EnKF

	    Inputs:
	    - U0: A numpy array of shape (p, J) initial ensemble; there are J
	      	ensemble particles each of dimension p.
	    - y_obs: A numpy array of shape (n_obs,) of observed data.
	    - wt: A numpy array of initial conditions to start the ensemble when
	    	evaluating the forward model
	    - t: A numpy array of time points where the ODE is evaluated
	    - ...
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """
		self.W0 = np.tile(wt, self.J).reshape(self.J, 3).T

		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = [] 			# Tracks the collapse towards the truth
		self.t = []

		for i in tqdm(range(self.T)):

			Geval = self.Gpar(np.vstack([U0, self.W0]), model, t)
			self.W0 = Geval[self.n_obs:,:]
			Geval = Geval[:self.n_obs,:]

			# For ensemble update
			E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
			R = U0 - U0.mean(axis = 1)[:,np.newaxis]

			Cpp = (1./self.J) * np.matmul(E, E.T)
			Cup = (1./self.J) * np.matmul(R, E.T)

			dU = np.matmul(Cup, np.linalg.solve( (1./self.T) * Cpp + Gamma, y_obs[:,np.newaxis] + \
				np.matmul(Jnoise, (np.random.normal(0, 1, [self.n_obs, self.J]))) - Geval))
			Uk = np.abs(U0 + (1./self.T) * dU)

			# For tracking metrics
			R = Geval - y_obs[:,np.newaxis]
			self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
			self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
			self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
			self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
			if i == 0:
				self.t.append(1./self.T)
			else:
				self.t.append(1./self.T + self.t[-1])


			self.Uall.append(Uk)
			U0 = Uk

		self.Uall = np.asarray(self.Uall)
		self.Ustar = self.Uall[-1]
		self.Gstar = self.Gpar(self.Ustar, model)

	def run_nopar(self, y_obs, U0, model, Gamma, Jnoise, verbose = True):
		"""
	    Find the minimizer of an inverse problem using the iterative EnKF

	    Inputs:
	    - U0: A numpy array of shape (p, J) initial ensemble; there are J
	      	ensemble particles each of dimension p.
	    - y_obs: A numpy array of shape (n_obs,) of observed data.
	    - wt: A numpy array of initial conditions to start the ensemble when
	    	evaluating the forward model
	    - t: A numpy array of time points where the ODE is evaluated
	    - ...
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """

		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = [] 			# Tracks the collapse towards the truth
		self.t = []

		for i in tqdm(range(self.T)):
			Geval = self.Gnopar(U0, model)

			# For ensemble update
			E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
			R = U0 - U0.mean(axis = 1)[:,np.newaxis]

			Cpp = (1./self.J) * np.matmul(E, E.T)
			Cup = (1./self.J) * np.matmul(R, E.T)

			dU = np.matmul(Cup, np.linalg.solve( (1./self.T) * Cpp + Gamma, y_obs[:,np.newaxis] + \
				np.matmul(Jnoise, (np.random.normal(0, 1, [self.n_obs, self.J]))) - Geval))
			Uk = U0 + (1./self.T) * dU

			# dU = np.matmul(Cup, np.linalg.solve(Cpp + Gamma, y_obs[:,np.newaxis] + \
			# 	np.matmul(Jnoise, (np.random.normal(0, 1, [self.n_obs, self.J]))) - Geval))
			# Uk = U0 + dU

			# For tracking metrics
			R = Geval - y_obs[:,np.newaxis]
			self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
			self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
			self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
			self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
			if i == 0:
				self.t.append(1./self.T)
			else:
				self.t.append(1./self.T + self.t[-1])

			self.Uall.append(Uk)
			U0 = Uk

		self.Uall = np.asarray(self.Uall)
		self.Ustar = self.Uall[-1]
		self.Gstar = self.Gnopar(self.Ustar, model)

	def run_data(self, y_obs, data, U0, wt, t, model, Gamma, Jnoise, verbose = True):
		"""
	    Find the minimizer of an inverse problem using the iterative EnKF

	    Inputs:
	    - U0: A numpy array of shape (p, J) initial ensemble; there are J
	      	ensemble particles each of dimension p.
	    - y_obs: A numpy array of shape (n_obs,) of observed data.
	    - wt: A numpy array of initial conditions to start the ensemble when
	    	evaluating the forward model
	    - t: A numpy array of time points where the ODE is evaluated
	    - ...
	    - verbose: (boolean) If true, print progress during iterations.

	    Outputs:
	    - None
	    """
		self.W0 = np.tile(wt, self.J).reshape(self.J, 3).T

		# Storing the ensemble members
		self.Uall = []; self.Uall.append(U0)
		self.radspec = []

		# Storing metrics
		self.metrics = dict()
		self.metrics['v'] = [] 			# Tracks collapse in parameter space
		self.metrics['V'] = []			# Tracks collapse after forward model evaln
		self.metrics['R'] = []			# Tracks data-fitting
		self.metrics['r'] = [] 			# Tracks the collapse towards the truth
		self.t = []

		for i in tqdm(range(self.T)):

			Geval = self.Gpar(np.vstack([U0, self.W0]), model, t)
			self.W0 = Geval[self.n_obs:,:]
			Geval = Geval[:self.n_obs,:]

			# For ensemble update
			E = Geval - Geval.mean(axis = 1)[:,np.newaxis]
			R = U0 - U0.mean(axis = 1)[:,np.newaxis]

			Cpp = (1./self.J) * np.matmul(E, E.T)
			Cup = (1./self.J) * np.matmul(R, E.T)

			dU = np.matmul(Cup, np.linalg.solve( (1./self.T) * Cpp + Gamma, y_obs[:,np.newaxis] + \
				y_obs[:,np.newaxis] - data[:,np.random.randint(data.shape[1])][:,np.newaxis] - Geval))
			Uk = np.abs(U0 + (1./self.T) * dU)

			# For tracking metrics
			R = Geval - y_obs[:,np.newaxis]
			self.metrics['v'].append(((U0 - U0.mean(axis = 1)[:, np.newaxis])**2).sum(axis = 0).mean())
			self.metrics['r'].append(((U0 - self.ustar)**2).sum(axis = 0).mean())
			self.metrics['V'].append((np.diag(np.matmul(E.T, np.linalg.solve(Gamma, E)))**2).mean())
			self.metrics['R'].append((np.diag(np.matmul(R.T, np.linalg.solve(Gamma, R)))**2).mean())
			if i == 0:
				self.t.append(1./self.T)
			else:
				self.t.append(1./self.T + self.t[-1])

			self.Uall.append(Uk)
			U0 = Uk

		self.Uall = np.asarray(self.Uall)

# ------------------------------------------------------------------------------
