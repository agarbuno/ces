import numpy as np

def banana(u, a = 1., b = .5, flag_noise = False):
    rho = 0.95
    Gamma = np.identity(2)
    Gamma[0,1] = rho
    Gamma[1,0] = rho

    Gamma = (0.55**2) * Gamma
    
    u1, u2 = u
    x = u1 * a
    y = u2/a - b *(u1**2 + a**2)
    return np.array([x, y]) + flag_noise * np.linalg.cholesky(Gamma).dot(np.random.normal(0, 1, [2,]))
    
def elliptic(u, x1 = 1./4, x2 = 3./4, dG = False, flag_noise = True, noise = 0.05):
    u1, u2 = u
    x = (u2 * x1) + (np.exp(-u1) * (-x1**2+x1) * 0.5) + (flag_noise * noise * np.random.normal())
    y = (u2 * x2) + (np.exp(-u1) * (-x2**2+x2) * 0.5) + (flag_noise * noise * np.random.normal())
    
    if dG: 
    	DG = np.array([[-np.exp(-u1) * (-x1**2+x1) * 0.5, x1], [-np.exp(-u1) * (-x2**2+x2) * 0.5, x2]])
    	return DG
    else: 
    	return [x,y]

def lorenz(w, t0, sigma = 10., r = 28., b = 8./3):
	"""
	Original Lorenz 63 model
	"""
	x, y, z = w
	x_dot = sigma * (y - x)
	y_dot = r * x - y - x * z
	z_dot = x * y - b * z
	return [x_dot, y_dot, z_dot]

def lorenz2d(w, t0, r = 28., b = 8./3):
	"""
	Original Lorenz 63 model simplified for just two dimensions
	"""
	x_dot, y_dot, z_dot = lorenz(w, t0, 10., r, b)
	return [x_dot, y_dot, z_dot]

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