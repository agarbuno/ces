import numpy as np
import scipy.optimize as opt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

cmap_blues = mpl.cm.Blues(np.linspace(0,1,20))
cmap_blues = mpl.colors.ListedColormap(cmap_blues[5:,:-1])

cmap_oranges = mpl.cm.Oranges(np.linspace(0,1,20))
cmap_oranges = mpl.colors.ListedColormap(cmap_oranges[5:,:-1])

cmap_greys = mpl.cm.Greys(np.linspace(0,1,20))
cmap_greys = mpl.colors.ListedColormap(cmap_greys[5:,:-1][::-1])

def find_levels(x, y, contours = [.99, .90, .68], **kwargs):
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

def plot_3axes(Phi, Hplot_rec, Hplot_full_rec, axes, **kwargs):
	k = kwargs.get('k', '')
	cmap = plt.get_cmap('gray');
	im = axes[0].contourf(np.exp(log_xs), np.exp(log_ys), np.log((Phi - logprior - logjac).reshape(grid_size, grid_size)),
					  pltlevels, extend = 'min',
					  cmap = cmap_greys);

	axes[0].set_title(r'$\Phi_T(\theta; y)$'+'\n');
	axes[0].set_ylabel(k);
	axes[0].set(xlim=(np.exp(logb_lo), np.exp(logb_up)), ylim=(np.exp(logr_lo), np.exp(logr_up)))

	levels_m = np.log(find_levels(bs, rs, energy = Hplot_rec - logprior - logjac))
	im = axes[1].contourf(np.exp(bs), np.exp(rs), np.log((Hplot_rec - logprior - logjac).reshape(grid_size, grid_size)),
						  levels_m, extend = 'min',
					  cmap = cmap_greys);
	axes[1].set_title(r'$\Phi_{\mathsf{m}}(\theta; y)$'+'\n');
	axes[0].set(xlim=(np.exp(logb_lo), np.exp(logb_up)), ylim=(np.exp(logr_lo), np.exp(logr_up)))

	levels_gp = np.log(find_levels(bs, rs, energy = Hplot_full_rec - logprior -logjac - Hplot_full_rec.min()))
	im = axes[2].contourf(np.exp(bs), np.exp(rs), np.log((Hplot_full_rec - logprior -logjac - Hplot_full_rec.min()).reshape(grid_size, grid_size)),
						  levels_gp, extend = 'min',
					  cmap = cmap_greys);
	axes[2].set_title(r'$\Phi_{\mathsf{GP}}(\theta; y)$'+'\n');
	axes[0].set(xlim=(np.exp(logb_lo), np.exp(logb_up)), ylim=(np.exp(logr_lo), np.exp(logr_up)))

def scatter_samples(eks, mcmcs, emu_type, axes):
	axes[0].scatter(np.exp(mcmc.samples[1]), np.exp(mcmc.samples[0]), color = u'#ff7f0e', marker = '+', s = 30, linewidth = 1)
	axes[1].scatter(np.exp(mcmc.samples[1]), np.exp(mcmc.samples[0]), color = u'#ff7f0e', marker = '+', s = 30, linewidth = 1)
	axes[2].scatter(np.exp(mcmc.samples[1]), np.exp(mcmc.samples[0]), color = u'#ff7f0e', marker = '+', s = 30, linewidth = 1)
	axes[2].scatter(np.exp(mcmcs[emu_type][0].samples[1]), np.exp(mcmcs[emu_type][0].samples[0]),  marker = '+', s = 30, linewidth = 1,)
	axes[1].scatter(np.exp(mcmcs[emu_type][1].samples[1]), np.exp(mcmcs[emu_type][1].samples[0]),  marker = '+', s = 30, linewidth = 1,)
	axes[0].scatter(np.exp(eks.Ustar[1]), np.exp(eks.Ustar[0]), color =  u'#2ca02c',  marker = '+', s = 30, linewidth = 2)

def plot_kde(x, y, ax, shade_lowest = False, alpha = .5, cmap = 'Blues'):
	sns.kdeplot(x, y, ax = ax, shade = True, shade_lowest=shade_lowest,
				alpha = alpha, cmap = cmap,
				n_levels = find_levels(x, y),
				antialiased=True, normed=True, extend = 'both')

def kde_samples(eks, mcmcs, emu_type, axes, shade_lowest = False, n_levels = 4, alpha = 1.):
	plot_kde(np.exp(mcmcs[emu_type][0].samples[1]), np.exp(mcmcs[emu_type][0].samples[0]),
			 axes[2], cmap = cmap_blues, alpha = alpha)
	plot_kde(np.exp(mcmcs[emu_type][1].samples[1]), np.exp(mcmcs[emu_type][1].samples[0]),
			 axes[1], cmap = cmap_blues, alpha = alpha)
	plot_kde(np.exp(mcmc.samples[1]), np.exp(mcmc.samples[0]),
				axes[0], cmap = cmap_oranges, alpha = alpha)
	axes[0].scatter(np.exp(eks.Ustar[1]), np.exp(eks.Ustar[0]), color =  u'#2ca02c',  marker = '+', s = 30, linewidth = 2)

	axes[0].set_title(r'$\Phi_T(\theta; y)$'+'\n');
	axes[1].set_title(r'$\Phi_{\mathsf{m}}(\theta; y)$'+'\n');
	axes[2].set_title(r'$\Phi_{\mathsf{GP}}(\theta; y)$'+'\n');

def ellipse(mcmcs, emu_type, ix):
	mu = mcmcs[emu_type][ix].samples.mean(axis = 1)
	Sigma = np.cov(mcmcs[emu_type][ix].samples)
	xs_ = np.vstack([rs.flatten(), bs.flatten()])
	xs_mu = xs_ - mu.reshape(-1,1)
	dist = (xs_mu * np.linalg.solve(Sigma, xs_mu)).sum(axis = 0)

	return dist

def draw_ellipse(mcmcs, emu_type, axes, color = '#d62728'):
	im = axes[0].contour(np.exp(bs), np.exp(rs),
						 ellipse(mcmcs, 'model', 0).reshape(grid_size, grid_size), clevels,
						 colors = color,
						 extend="both");
	axes[0].clabel(im,  fmt = probs,  inline=1, fontsize=12);

	im = axes[1].contour(np.exp(bs), np.exp(rs),
						 ellipse(mcmcs, emu_type, 1).reshape(grid_size, grid_size), clevels,
						 colors = color,
						 extend="both");
	axes[1].clabel(im,  fmt = probs,  inline=1, fontsize=12);


	im = axes[2].contour(np.exp(bs), np.exp(rs),
						 ellipse(mcmcs, emu_type, 0).reshape(grid_size, grid_size), clevels,
						 colors = color,
						 extend="both");
	axes[2].clabel(im,  fmt = probs,  inline=1, fontsize=12);
