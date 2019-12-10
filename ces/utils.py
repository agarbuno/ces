import numpy as np
import pandas as pd
from scipy import integrate

class lineal(object):
	"""
	"""
	def __init__(self, A, flag_noise = False):
		"""
		"""
		self.A = A
		self.n_obs = A.shape[0]
		self.flag_noise = flag_noise
		self.noise_sigma = np.sqrt(0.1)
		self.model_name = 'lineal'
		self.type = 'map'

	def __repr__(self):
		return self.model_name

	def __str__(self):
		return self.model_name + str(self.n_state)

	def __call__(self, theta):
		"""
		"""
		if self.flag_noise:
			return np.matmul(self.A, theta) + self.noise_sigma * np.random.normal()
		else:
			return np.matmul(self.A, theta)

class elliptic(object):
	"""
	"""
	def __init__(self, flag_noise = False):
		"""
		"""
		self.x1 = 1./4
		self.x2 = 3./4
		self.flag_noise = flag_noise
		self.sigma      = np.sqrt(0.1)
		self.model_name = 'elliptic'
		self.type       = 'map'

	def __repr__(self):
		return self.model_name

	def __str__(self):
		return self.model_name

	def __call__(self, theta, dG = False):
		"""
		"""
		u1, u2 = theta
		x = (u2 * self.x1) + (np.exp(-u1) * (-self.x1**2+self.x1) * 0.5) + (self.flag_noise * self.sigma * np.random.normal())
		y = (u2 * self.x2) + (np.exp(-u1) * (-self.x2**2+self.x2) * 0.5) + (self.flag_noise * self.sigma * np.random.normal())

		if dG:
			DG = np.array([[-np.exp(-u1) * (-self.x1**2+self.x1) * 0.5, self.x1],
						   [-np.exp(-u1) * (-self.x2**2+self.x2) * 0.5, self.x2]])
			return DG
		else:
			return [x,y]

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
		self.solve_init = False
		self.model_name = 'lorenz63'
		self.type = 'pde'

	def __repr__(self):
		return self.model_name

	def __str__(self):
		return self.model_name + str(self.n_state)

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
		gs = np.asarray(ws)[:,1:].reshape(self.n_obs, -1,
			int(self.l_window * self.freq)).mean(axis = 2)[:,-1]
		return gs

class lorenz63_log(lorenz63):
	def __init__(self, l_window = 10, freq = 100):
		super().__init__(l_window = l_window, freq = freq)
		self.model_name = 'lorenz63_log'

	def __call__(self, w, t, log_r = np.log(28.), log_b = np.log(8./3)):
		"""
		Reduced Lorenz 63 model with 2 parameters.
		(WARNING): This should be edited somehow to be able to reduce the model
				in an arbitrary way.
		"""
		x_dot, y_dot, z_dot = self.model(w, t, 10., log_r, log_b)
		return [x_dot, y_dot, z_dot]

	def model(self, w, t, sigma = 10., log_r = np.log(28.), log_b = np.log(8./3)):
		"""
		Original Lorenz 63 model with 3 parameters
		"""
		r = np.exp(log_r)
		b = np.exp(log_b)

		x, y, z = w
		x_dot = sigma * (y - x)
		y_dot = r * x - y - x * z
		z_dot = x * y - b * z
		return [x_dot, y_dot, z_dot]

	def grad_logjacobian(self, params):
			return -np.exp(-params)

	def logjacobian(self, params):
			return - params.sum(axis = 0)

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
	def __init__(self, n_slow = 36, n_fast = 10, l_window = 10, freq = 10, spinup = 10):
		"""
		Lorenz 96 model initilization
		"""
		self.n_slow   = n_slow
		self.n_fast   = n_fast
		self.n_state  = self.n_slow * (self.n_fast + 1)
		self.l_window = l_window
		self.freq     = freq
		self.spinup   = spinup
		self.solve_init = False
		self.model_name = 'lorenz96'
		self.type     = 'pde'

	def __repr__(self):
		return self.model_name + ',' + str(self.n_slow) + ',' + str(self.n_fast)

	def __str__(self):
		"""
		Printing method
		"""
		print('Model: ..................... Lorenz 96')
		print('Number of slow variables ... %s'%(self.n_slow))
		print('Number of fast variables ... %s'%(self.n_fast))
		print('Number of parameters........ %s'%(4))
		print('Solver initialized ......... %s'%(self.solve_init))
		return str()

	def __call__(self, t, w, h = 1., F = 10., log_c = np.log(10.), b = 10.):
		"""
		(WARNING): This should be edited somehow to be able to reduce the model
				in an arbitrary way. I am thinking of some way of choosing the parameters
				with a boolean vecor.
		"""
		ws = self.model(w, t, h, F, log_c, b)
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

	def model(self, X, t, h = 1., F = 10., log_c = np.log(10.), b = 10.):
		"""
		Original Lorenz 96 model with 4 parameters
		"""
		c = np.exp(log_c)

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
		return res.y.T

	def statistics(self, ws):
		ws = ws.T
		data = np.copy(ws[:,(self.spinup * self.freq + 1):].reshape(self.n_state, -1, self.l_window * self.freq))

		Phi = np.vstack([data[:self.n_slow].mean(axis = 2),
			(data[:self.n_slow]**2).mean(axis = 2),
			data[self.n_slow:].reshape(self.n_slow, self.n_fast, -1, self.l_window * self.freq ).mean(axis = 1).mean(axis = 2),
			(data[self.n_slow:]**2).reshape(self.n_slow, self.n_fast, -1, self.l_window * self.freq ).mean(axis = 1).mean(axis = 2),
			(data[:self.n_slow] * (data[self.n_slow:]).reshape(self.n_slow, self.n_fast, -1, self.l_window * self.freq ).mean(axis = 1)).mean(axis = 2)])

		return Phi[:,-1]

	def grad_logjacobian(self, params):
		gradlogjac = np.zeros_like(params)
		gradlogjac[2] = -np.exp(-gradlogjac[2])

		return gradlogjac

class lorenz96_hom(lorenz96):
	def __init__(self):
		super().__init__()
		self.hom = True

	def statistics(self, ws):
		ws = ws.T
		data = np.copy(ws[:,(self.spinup * self.freq + 1):].reshape(self.n_state, -1, self.l_window * self.freq))

		Phi = np.vstack([data[:self.n_slow].mean(axis = 2),
			(data[:self.n_slow]**2).mean(axis = 2),
			data[self.n_slow:].reshape(self.n_slow, self.n_fast, -1, self.l_window * self.freq ).mean(axis = 1).mean(axis = 2),
			(data[self.n_slow:]**2).reshape(self.n_slow, self.n_fast, -1, self.l_window * self.freq ).mean(axis = 1).mean(axis = 2),
			(data[:self.n_slow] * (data[self.n_slow:]).reshape(self.n_slow, self.n_fast, -1, self.l_window * self.freq ).mean(axis = 1)).mean(axis = 2)])

		if self.hom :
			return Phi[:,-1].reshape(5, -1).mean(axis = 1)
		else:
			return Phi[:,-1].reshape(5, -1)[:, 7]

class lorenz96Fc(lorenz96):
	def __init__(self):
		super().__init__()

	def __repr__(self):
		return self.model_name + ',' + str(self.n_slow) + ',' + str(self.n_fast) + ',' + str(2)

	def __str__(self):
		"""
		Printing method
		"""
		print('Model: ..................... Lorenz 96')
		print('Number of slow variables ... %s'%(self.n_slow))
		print('Number of fast variables ... %s'%(self.n_fast))
		print('Number of parameters........ %s'%(2))
		print('Solver initialized ......... %s'%(self.solve_init))
		return str()

	def __call__(self, t, w, F = 10., log_c = np.log(10.)):
		ws = self.model(w, t, 1., F, log_c, 10.)
		return ws

class lorenz96Fb(lorenz96):
	def __repr__(self):
		return self.model_name + ',' + str(self.n_slow) + ',' + str(self.n_fast) + ',' + str(2)

	def __str__(self):
		"""
		Printing method
		"""
		print('Model: ..................... Lorenz 96')
		print('Number of slow variables ... %s'%(self.n_slow))
		print('Number of fast variables ... %s'%(self.n_fast))
		print('Number of parameters........ %s'%(2))
		print('Solver initialized ......... %s'%(self.solve_init))
		return str()

	def __call__(self, t, w, F = 10., b = 10.):
		ws = self.model(w, t, 1., F, np.log(10), b)
		return ws

class lorenz96hFb(lorenz96):
	def __repr__(self):
		return self.model_name + ',' + str(self.n_slow) + ',' + str(self.n_fast) + ',' + str(3)

	def __str__(self):
		"""
		Printing method
		"""
		print('Model: ..................... Lorenz 96')
		print('Number of slow variables ... %s'%(self.n_slow))
		print('Number of fast variables ... %s'%(self.n_fast))
		print('Number of parameters........ %s'%(3))
		print('Solver initialized ......... %s'%(self.solve_init))
		return str()

	def __call__(self, t, w, h = 1., F = 10., b = 10.):
		ws = self.model(w, t, h, F, np.log(10), b)
		return ws

class lorenz96hcb(lorenz96):
	def __repr__(self):
		return self.model_name + ',' + str(self.n_slow) + ',' + str(self.n_fast) + ',' + str(3)

	def __str__(self):
		"""
		Printing method
		"""
		print('Model: ..................... Lorenz 96')
		print('Number of slow variables ... %s'%(self.n_slow))
		print('Number of fast variables ... %s'%(self.n_fast))
		print('Number of parameters........ %s'%(3))
		print('Solver initialized ......... %s'%(self.solve_init))
		return str()

	def __call__(self, t, w, h = 1., log_c = np.log(10.), b = 10.):
		ws = self.model(w, t, h, 10., log_c, b)
		return ws

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
