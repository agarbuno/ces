import numpy as np
import pandas as pd

import sys
sys.path.append('./')

from ces.calibrate import *
import ces.darcy as darcy
model = darcy.model()

model.start(mpath = r'./mfiles')
model.set_rnd_seed()
model.set_initial()
model.n_obs = 50
model.model_name = 'darcy-flow'

# Forward model G(u) in D. Take the KL coeffs and returns pressure in D.
U = model(model.ustar, full_solution = True);

# Generate the random locations in the grid
xs, ys = np.meshgrid(np.linspace(0, 1, int(model.Nmesh)), np.linspace(0, 1, int(model.Nmesh)))
np.random.seed(1)
grid_pts    = np.vstack((xs.flatten(), ys.flatten()))           # Flat the (x,y) coordinates
model.obs_index = np.random.choice(int(model.p), model.n_obs, replace=False,
							 p = U /U.sum())                    # Sample without replacement
model.obs_locs = grid_pts[:, model.obs_index]                   # Recover the random (x,y) coordinates

# Generate the truth at the locations
y_obs = model(model.ustar)
y_smooth = y_obs.copy()

#  Add noise to observations for inverse problem setup.
gamma = 0.005
Gamma = gamma**2 * np.identity(model.n_obs)
y_obs = y_obs + 1.0 * gamma * np.random.normal(0,1,model.n_obs)

tqdm.write('Remember, the number of params is: %s'%(len(model.ustar)))

# # Ensemble Kalman Sampler

Jnoise = np.linalg.cholesky(Gamma)

def run_neks(J, model, **kwargs):
	"""
	INPUT:
			 J :  The number of ensemble members.
		 model :  The forward model for the inversion. Needs to be a callable object with property model.type = 'map'
		 kwargs:  Variable dictionary with {keyword: value} arguments.

	OUTPUT:
		eks object with:
			eks.Uall  - trajectory of ensemble
			eks.Ustar - ensemble at last iteration

			eks.Gall  - trajectory of forward evaluations for ensemble members
			eks.Gstar - forward evaluations for last iteration

			eks.metrics - contain several metrics of ensemble performance, mainly to assess collapse towards truth and
						  collapse within ensemble, also, fictitious time

	"""

	# EKS setup:
	# Initialize the object as a sampling ensemble kalman algorithm.
	eks          = sampling(p = model.p, n_obs = model.n_obs, J = J)

	# Pass the truth to compute some statistics for the ensemble.
	eks.ustar    = model.ustar.reshape(model.p,-1)

	# Number of maxmimum iterations
	eks.T        = 200

	# Prior specification
	eks.mu       = 0.0 * np.ones((model.p,)).reshape(model.p,-1)
	eks.sigma    = 100. * np.identity(model.p)

	# Computing variables
	eks.parallel = False                   # No parallel evaluation of G
	eks.mute_bar = True                    # Mutes an additional progress bar
	eks.nexp     = kwargs.get('nexp', '')  # Sets an additional string for saving purposes

	# Location of where to save the ensemble members
	eks.directory = './'

	# Random initial ensemble
	np.random.seed(kwargs.get('nexp', 1))
	U0 = 10 * np.random.normal(0, 1, [eks.p, J])

	# Run the whole EKS procedure until it reaches time t_tol or maximum number of iterations are exhausted.
	# Save online option allows to save the trajectory on disk as numpy objects.
	eks.run(y_obs, U0, model, Gamma, Jnoise, save_online = True, t_tol = 5)

	return eks

model.model_name = 'darcy-flow'

Js = [int(model.p/15), int(model.p/5), int(model.p/2), int(model.p + 2), int(2 * model.p), int(3 * model.p)]

np.random.seed(1)

neks = {}
for J in tqdm(Js, desc = 'Ensembles : '):
	neks['eks-'+str(J).zfill(3)] = []
	for jj in range(1):
		neks['eks-'+str(J).zfill(3)].append(run_neks(J, model, nexp = jj))
