Tools for Ensemble Kalman Inversion (EKI), Ensemble Kalman Sampler (EKS) and
Gaussian Process Emulation (using `Gpflow`) for Uncertainty Quantification and
Inverse Problems.

**Note**: Examples can be found as jupyter notebooks.

To import the module into a python script or project, type

```
import sys
sys.path.append('Your path/ipuq/ces/')

from ces.utils import *
from ces.enka import *
```
---

Overview of this module:
- `enka` contains the ensemble Kalman algorithms.
- `utils` contains the additional tools for running the examples. Like test functions,
PDEs constrained functions to solve the inverse problem through an approximate Bayesian method.  

Coming shortly:
- MCMC using GPs.

Dependencies:
- numpy
- gpflow
- scipy
- pandas

---

References:

Iglesias et al. Ensemble Kalman methods for inverse problems. Inverse
Problems 29.
