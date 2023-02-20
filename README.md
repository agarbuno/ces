Tools for Ensemble Kalman Inversion (EKI), Ensemble Kalman Sampler (EKS) and
Gaussian Process Emulation (using `Gpflow`) for Uncertainty Quantification in 
inverse problems.

**Note**: Examples can be found as jupyter notebooks.

To import the module into a python script or project, type

```
import sys
sys.path.append('Your path/ipuq/')

from ces.utils import *
from ces.calibrate import *
```
---

Overview of this module:
- `enka` contains the ensemble Kalman algorithms.
- `utils` contains the additional tools for running the examples. Like test functions,
and PDEs constrained functions. The goal is to solve inverse problems through an approximate Bayesian method.  

The provided code can be used for the following:
- MCMC through Metropolis Hastings. 
- Accelerated MCMC using GPs as surrogate models.

Dependencies:
- tqdm
- numpy
- gpflow
- scipy
- pandas

---

References:

- Garbuno-Inigo, A., Nüsken, N., & Reich, S. (2020). _Affine invariant interacting Langevin dynamics for Bayesian inference_. SIAM Journal on Applied Dynamical Systems, 19(3), 1633-1658.

- Garbuno-Inigo, A., Hoffmann, F., Li, W., & Stuart, A. M. (2020). _Interacting Langevin diffusions: Gradient structure and ensemble Kalman sampler_. SIAM Journal on Applied Dynamical Systems, 19(1), 412-441.

- Iglesias, M. A., Law, K. J., & Stuart, A. M. (2013). _Ensemble Kalman methods for inverse problems_. Inverse Problems, 29(4), 045001.
