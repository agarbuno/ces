from __future__ import print_function
import os
import pickle
import numpy as np
import pandas as pd
from . import calibrate
import gpflow as gp

from tqdm.autonotebook import tqdm

def scale_ensemble(enka, factor = 2.):
	enka.scale['mean'] = enka.Ustar.mean(axis = 1)[:, np.newaxis]
	enka.scale['cov']  = factor * np.linalg.cholesky(np.cov(enka.Ustar))
	enka.scale['X']    = np.linalg.solve(enka.scale_cov, enka.Ustar - enka.scale_mean)

def predict_gps(enka, X, mute_bar = True, **kwargs):
	"""
	Prediction using independent GP models for multioutput code.
	Eventually, we will either decorrelate or learn a multioutput emulator.

	Inputs:
	- gpmodels: list of independent GP models.
	- X: numpy array with dimensions [n_points, d].
		- n_points: number of training points.
		- d: dimensionality of the parameter vector.
	"""
	try:
		getattr(enka, 'gpmodels')
	except AttributeError:
		tqdm.write('There are no trained GP model(s) in object: %s'%enka)
		return ''

	if kwargs.get('gpmodels', None) is None:
		gpmodels = enka.gpmodels
	else:
		gpmodels = kwargs.get('gpmodels', None)

	gpmeans = np.empty(shape = (len(gpmodels), len(X)))
	gpvars  = np.empty(shape = (len(gpmodels), len(X)))

	for ii, model in tqdm(enumerate(gpmodels),
			desc = 'GP predictions',
			disable = mute_bar,
			position = 0):
		try:
			getattr(enka, 'scaled')
			mean_pred, var_pred = model.predict_y(np.linalg.solve(enka.scale['cov'], X.T - enka.scale['mean']).T)
		except AttributeError:
			mean_pred, var_pred = model.predict_y(X)

		gpmeans[ii,:] = mean_pred.flatten()
		gpvars[ii,:] = var_pred.flatten()

	return [gpmeans, gpvars]

def scale_gppreds(gpmeans, gpvars, Gmean, Gstd):
	"""
	Inputs:
	- gpmeans: list
	- gpvars: list
	- Gmean: numpy array with means used to rescale output due to parameter variability
	- Gstd: numpy array with stds used to rescale output due to parameter variability
	"""
	n_obs = len(gpmeans)

	Gmeans = []
	Gvars = []

	for ii in range(len(gpmeans)):
		if ii in range(2,7):
			mexp = np.exp(gpmeans[ii] * Gstd[ii] + Gmean[ii] + (Gstd[ii]**2 * gpvars[ii])/2)
			vexp = (np.exp(Gstd[ii]**2 * gpvars[ii]) - 1.) * (mexp**2)
		else :
			mexp = gpmeans[ii] * Gstd[ii] + Gmean[ii]
			vexp = Gstd[ii]**2 * gpvars[ii]

		Gmeans.append(mexp)
		Gvars.append(vexp)

	return [np.asarray(Gmeans).reshape(n_obs, -1), np.asarray(Gvars).reshape(n_obs, -1)]
