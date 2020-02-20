import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate
import scipy.optimize as opt

def find_levels(x, y, contours = [.9999, .99, .95, .68], **kwargs):
	# Make a 2d normed histogram
	if kwargs.get('energy', None) is None:
		H,xedges,yedges=np.histogram2d(x,y,bins=20,normed=True)
	else:
		H = np.exp(-kwargs.get('energy'))

	norm=H.sum() # Find the norm of the sum
	# Set target levels as percentage of norm
	targets = [norm * c for c in contours]
	# Take histogram bin membership as proportional to Likelihood
	# This is true when data comes from a Markovian process
	def objective(limit, target):
		w = np.where(H>limit)
		count = H[w]
		return count.sum() - target

	# Find levels by summing histogram to objective
	levels = [opt.bisect(objective, H.min(), H.max(), args=(target,)) for target in targets]

	# For nice contour shading with seaborn, define top level
	levels.append(H.max())

	if kwargs.get('energy', None) is None:
		return levels
	else:
		return -np.log(levels)[::-1]

def plot_kde(x, y, ax, shade_lowest = False, alpha = .5, cmap = 'Blues'):
	sns.kdeplot(x, y, ax = ax, shade = True, shade_lowest=shade_lowest,
				alpha = alpha, cmap = cmap,
				n_levels = find_levels(x, y),
				antialiased=True, normed=True, extend = 'both')

def abline(slope, intercept, ax, **kwargs):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, kwargs)
