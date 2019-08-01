import numpy as np
from scipy import integrate

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

def elliptic(u, x1 = 1./4, x2 = 3./4, dG = False, flag_noise = True, noise = 0.1):
    u1, u2 = u
    x = (u2 * x1) + (np.exp(-u1) * (-x1**2+x1) * 0.5) + (flag_noise * noise * np.random.normal())
    y = (u2 * x2) + (np.exp(-u1) * (-x2**2+x2) * 0.5) + (flag_noise * noise * np.random.normal())

    if dG:
    	DG = np.array([[-np.exp(-u1) * (-x1**2+x1) * 0.5, x1], [-np.exp(-u1) * (-x2**2+x2) * 0.5, x2]])
    	return DG
    else:
    	return [x,y]

class lorenz63(object):
    '''Lorenz '63 model'''
    def __init__(self, l_window = 10, freq = 100):
        """
        Lorenz 63 model initilization
        """
        self.n_state = 3
        self.n_obs = 9
        self.l_window = l_window
        self.freq = freq

    def __call__(self, w, t, r = 28., b = 8./3):
        """
        Reduced Lorenz 63 model with 2 parameters
        """
        x_dot, y_dot, z_dot = self.model(w, t, 10., r, b)
        return [x_dot, y_dot, z_dot]

    def model(self, w, t, sigma = 10., r = 28., b = 8./3):
        """
        Original Lorenz 63 model with 3 parameters
        """
        x, y, z = w
        x_dot = sigma * (y - x)
        y_dot = r * x - y - x * z
        z_dot = x * y - b * z
        return [x_dot, y_dot, z_dot]

    def solve(self, w0, t, args = ()):
        """
        Solve the Lorenz system using the callable object defined in __call__.
        Inputs:
            - w0: [n_state, ] initial conditions
            - t : time vector to collect solution of ODE solver
        """
        ws = integrate.odeint(self, w0, t, args = args)
        return ws

    def statistics(self, ws):
        """
        Compute the relevant statistics from the output of the solution of the
        ODE solver
        """
        xs, ys, zs = ws[:,0], ws[:,1], ws[:,2]
        ws = [xs, ys, zs, xs**2, ys**2, zs**2, xs*ys, xs*zs, ys*zs]
        gs = np.asarray(ws)[:,1:].reshape(self.n_obs, -1,
            int(self.l_window * self.freq)).mean(axis = 2)[:,-1]
        return np.round(gs, 12)

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

def lorenz96_rhs(X, t, h, F, c, b):
    n_slow = 36       # Slow variables
    n_fast = 10       # Fast variables

    Y = X[n_slow:]
    X = X[:n_slow]
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(n_slow):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % n_slow]) - X[k] + F - \
                    (h * c) * np.mean(Y[k * n_fast: (k + 1) * n_fast])
    for j in range(n_fast * n_slow):
        dYdt[j] = -c * b * Y[(j + 1) % (n_fast * n_slow)] * (Y[(j + 2) % (n_fast * n_slow)] - \
                    Y[j-1]) - c * Y[j] + ((h*c)/n_fast) * X[int(j / n_fast)]

    return np.hstack((dXdt, dYdt))

def lorenz96_dim(t, X, h = 1., F = 10., c = 2**7., b = 1.):
    n_slow = 36       # Slow variables
    n_fast = 10       # Fast variables

    Y = X[n_slow:]
    X = X[:n_slow]
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(n_slow):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % n_slow]) - X[k] + F - \
                    (0.8) * np.mean(Y[k * n_fast: (k + 1) * n_fast])
    for j in range(n_fast * n_slow):
        dYdt[j] = - c * Y[(j + 1) % (n_fast * n_slow)] * \
                       (Y[(j + 2) % (n_fast * n_slow)] - Y[j-1]) - \
                    c * Y[j] + c * X[int(j / n_fast)]

    return np.hstack((dXdt, dYdt))

def lorenz96_ivp(t, X, h = 1., F = 10., c = 10., b = 10.):
    n_slow = 36       # Slow variables
    n_fast = 10       # Fast variables

    Y = X[n_slow:]
    X = X[:n_slow]
    dXdt = np.zeros(X.shape)
    dYdt = np.zeros(Y.shape)
    for k in range(n_slow):
        dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % n_slow]) - X[k] + F - \
                    (h * c) * np.mean(Y[k * n_fast: (k + 1) * n_fast])
    for j in range(n_fast * n_slow):
        dYdt[j] = -c * b * Y[(j + 1) % (n_fast * n_slow)] * (Y[(j + 2) % (n_fast * n_slow)] - \
                    Y[j-1]) - c * Y[j] + ((h*c)/n_fast) * X[int(j / n_fast)]

    return np.hstack((dXdt, dYdt))

def generate_statistics(data):
    Phi = np.vstack([data[:n_slow].mean(axis = 1),            # <  X >
                 (data[:n_slow]**2).mean(axis = 1),           # < X2 >
                 data[n_slow:].reshape(n_slow, n_fast, -1).mean(axis = 1).mean(axis = 1),      # < Ybar  >
                 (data[n_slow:]**2).reshape(n_slow, n_fast, -1).mean(axis = 1).mean(axis = 1), # < Y2bar >
                 (data[:n_slow] * (data[n_slow:]).reshape(n_slow, n_fast, -1).mean(axis = 1)).mean(axis = 1)])  # < X Y2bar >

    return Phi.reshape(1,-1)

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
