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
