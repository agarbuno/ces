import numpy as np
import pandas as pd
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
    """
    Lorenz 63 model. PDE model structure:
        - init: to initialize the object
        - call: make the object be a function. This could allow the model to be
            used in a reduced way.
        - model: RHS of the ode system.
        - solve: here we define the method we need to solve the model.
        - statistics: here we define the relevant statistics to be computed.
    """
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
        Reduced Lorenz 63 model with 2 parameters.
        (WARNING): This should be edited somehow to be able to reduce the model
                in an arbitrary way.
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
        # With rolling statistics
        gs = [np.asarray(pd.Series(k).rolling(window = int(self.l_window * self.freq) ).mean()) for k in ws]
        gs = np.asarray(gs)[:,-1]
        # With adyacent windows
        # gs = np.asarray(ws)[:,1:].reshape(self.n_obs, -1,
            # int(self.l_window * self.freq)).mean(axis = 2)[:,-1]
        return gs

class lorenz96(object):
    """
    Lorenz 96 model. PDE model structure:
        - init: to initialize the object
        - generate_initial: generate relevant initial condition.
        - call: make the object be a function. This could allow the model to be
            used in a reduced way.
        - model: RHS of the ode system.
        - solve: here we define the method we need to solve the model.
        - statistics: here we define the relevant statistics to be computed.
    """
    def __init__(self, l_window = 10, freq = 100):
        """
        Lorenz 96 model initilization
        """
        self.n_slow   = 36
        self.n_fast   = 10
        self.n_state  = self.n_slow * (self.n_fast + 1)
        self.l_window = l_window
        self.freq     = freq
        self.solve_init = False


    def __call__(self, t, w,  h = 1., F = 10., c = 10., b = 10.):
        """
        Reduced Lorenz 96 model with 2 parameters.
        (WARNING): This should be edited somehow to be able to reduce the model
                in an arbitrary way.
        """
        ws = self.model(w, t)
        return ws

    def generate_initial(self):
        """
        Generate initial conditions. All fast variables are initialized with the
        value of the associated slow variable.
            - y_{l,k} = x{k} for all l.
        """
        x0 = np.empty(self.n_slow + self.n_slow * self.n_fast)
        x0[:self.n_slow] = np.random.rand(self.n_slow) * 15 - 5
        for k in range(0,self.n_slow):
            x0[self.n_slow + k * self.n_fast : self.n_slow + (k+1) * self.n_fast] = x0[k]

        return x0

    def model(self, X, t, h = 1., F = 10., c = 10., b = 10.):
        """
        Original Lorenz 96 model with 4 parameters
        """
        Y = X[self.n_slow:]
        X = X[:self.n_slow]
        dXdt = np.zeros(X.shape)
        dYdt = np.zeros(Y.shape)

        for k in range(self.n_slow):
            dXdt[k] = -X[k - 1] * (X[k - 2] - X[(k + 1) % self.n_slow]) - X[k] + F - \
                        (h * c) * np.mean(Y[k * self.n_fast: (k + 1) * self.n_fast])

        for j in range(self.n_fast * self.n_slow):
            dYdt[j] = -c * b * Y[(j + 1) % (self.n_fast * self.n_slow)] * (Y[(j + 2) % (self.n_fast * self.n_slow)] - \
                        Y[j-1]) - c * Y[j] + ((h*c)/self.n_fast) * X[int(j / self.n_fast)]

        return np.hstack((dXdt, dYdt))

    def set_solver(self, method = 'RK45', T = 20, dt = 0.1):
        self.method = method
        self.dt = dt
        self.T = T
        self.solve_init = True

    def solve(self, w0, t, args = ()):
        """
        Solve the Lorenz 96 system using the callable object defined in __call__.
        Inputs:
            - w0: [n_state, ] initial conditions
            - t : time vector to collect solution of ODE solver
        """
        if self.solve_init:
            res = integrate.solve_ivp(fun = lambda t, y: self(t, y, *args),
                t_span = [0,self.T], y0 = w0,
                t_eval = t, method = self.method, max_step = self.dt)
        else:
            res = np.empty()
        return res.y

    def statistics(self, data):
        Phi = np.vstack([data[:self.n_slow].mean(axis = 1),
            (data[:self.n_slow]**2).mean(axis = 1),
            data[self.n_slow:].reshape(self.n_slow, self.n_fast, -1).mean(axis = 1).mean(axis = 1),
            (data[self.n_slow:]**2).reshape(self.n_slow, self.n_fast, -1).mean(axis = 1).mean(axis = 1),
            (data[:self.n_slow] * (data[self.n_slow:]).reshape(self.n_slow, self.n_fast, -1).mean(axis = 1)).mean(axis = 1)])

        return Phi.reshape(1,-1)

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