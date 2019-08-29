Tools for Ensemble Kalman Inversion (EKI), Ensemble Kalman Sampler (EKS) and
Gaussian Process Emulation (using `Gpflow`) for Uncertainty Quantification and
Inverse Problems.

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

Coming shortly:
- MCMC
- Accelerated MCMC using GPs.

Dependencies:
- tqdm
- numpy
- gpflow
- scipy
- pandas

---

References:
- Garbuno-Inigo, A., Hoffmann, F., Li, W., & Stuart, A. M. _Interacting Langevin Diffusions: Gradient Structure And Ensemble Kalman Approximation_.

- Iglesias et al. _Ensemble Kalman methods for inverse problems._ Inverse
Problems 29.
