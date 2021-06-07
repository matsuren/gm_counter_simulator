---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Single source localization

```python
import sys
sys.path.insert(0, "../")
```

```python
from gm_simulator import Source, Detector, World
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
from copy import deepcopy

```

<!-- #region -->
# Localization
Radiation localization assuming only one source.

State: $\mathbf{s} =(\mathbf{x}_{0} ,q_{0} )\in \mathbb{R}^{4}$, where $q_{0}$ is intensity of radiation source, and $\mathbf{x}_{0} =[x_{0} ,y_{0} ,z_{0} ]^{\top }$

Measurement: $p_{i} \in \mathbb{R}$, where $y_{i}$ is the number of count detected by detector $i$ located at $\mathbf{y}_{i} \in \mathbb{R}^{3}$.



The probability of $y_{i}$ is formulated as follows:
\begin{gather*}
y_{i} \sim \mathrm{Pois}( \lambda _{i}) =\frac{\lambda ^{y_{i}} e^{-\lambda }}{y_{i} !} ,\\
\lambda _{i} =\Gamma \frac{q_{0}}{||\mathbf{x} -\mathbf{y}_{i} ||^{2}} ,
\end{gather*}
E.g., the likelihood after knowing three measurements $\mathcal{P} =\{p_{i} :0\leqq i\leqq 2\}$


\begin{gather*}
\mathrm{P}(\mathbf{s} |\mathcal{P}) =\prod _{i \in |\mathcal{P}|}\mathrm{Pois}( \lambda _{i}) ,\\
\log\mathrm{P}(\mathbf{s} |\mathcal{P}) =\sum _{i \in |\mathcal{P}|}( y_{i}\log \lambda _{i} -\lambda _{i} -\log y_{i}!) ,
\end{gather*}
<!-- #endregion -->

```python
def get_pois_lambda(detector: Detector, loc: List[float], q: float) -> float:
    """
    Get lambda for Poisson distribution

    Parameters
    ----------
    detector: Detector
        Detector
    loc: List[float]
        Location of the radiation source such that [x, y, z] [m].
    q: float
        Intensity of the radiation source [MBq]. (default is 1.0)

    Returns
    -------
    lambda: float
        lambda for Poisson
    """

    # Factor
    factor = detector.duration * detector.factor

    # Locations
    detec_loc = np.array(detector.loc)
    src_loc = np.array(loc)

    # distance
    dist = np.linalg.norm(src_loc - detec_loc)
    lmd = factor * q / dist / dist

    return lmd
```

```python
def get_loglikelihood(y: int, lmd: float) -> float:
    """
    Get log likelihood.

    Parameters
    ----------
    y: int
        Measurement count
    lmd: float
        lambda in Poisson distribution

    Returns
    -------
    logli: float
        log likelihood
    """
    # y*log(mu)-mu-log(y!)
    logli = y * np.log(lmd) - lmd - np.log(np.arange(1, y + 1)).sum()

    return logli
```

```python
def sum_loglikelihood(
    cnts: List[int], detectors: List[Detector], target_s: List[float]
) -> float:
    """
    Sum loglikelihood from observation.

    Parameters
    ----------
    cnts: List[int]
        Counts from multiple measurements
    detectors: List[Detector]
        List of detectors.
    target_s:List[float]
        Target state [x, y, z, p]

    Returns
    -------
    logli: float
        Sum of log likelihood from multiple measurements
    """
    sum_logli = []
    for i, y in enumerate(cnts):
        lmd = get_pois_lambda(detectors[i], target_s[:3], target_s[3])
        logli = get_loglikelihood(y, lmd)
        sum_logli.append(logli)
    return sum(sum_logli)
```

## Construct world for example

```python
# Construct world
world = World()
world.add_source(Source(loc=[1, 2.5, 0], intensity=2))
world.visualize_world(figsize=(5, 5))
```

## Particle filter


### Single source localization

```python
# Set detectors (locations are known)
detector_locations = [
    [-1.0, 0, 0],
    [0, 0, 0],
    [1.0, 0, 0],
    [-1.0, 2, 0],
    [0, 2, 0],
    [1.0, 2, 0],
]

detectors = []
for loc in detector_locations:
    detectors.append(Detector(loc=loc))

cnts = world.get_measuments(detectors)
print(f"max_count: {cnts.max()}")
locs = np.array(detector_locations)
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(locs[:, 0], locs[:, 1], c=cnts, cmap=plt.cm.jet)
_ = fig.colorbar(sc, orientation="horizontal")
ax.set_aspect("equal")
```

```python
# Make sure if world is set correctly
assert len(world.sources) == 1, "should be single source"
# Groundtruth
gt_loc = world.sources[0].loc
gt_q = world.sources[0].intensity
gt_s = list(gt_loc + [gt_q])
print(f"groundtruth state: {gt_s}")
world.visualize_world(detectors, figsize=(5, 5), plotsize=2)
```

```python
N_p = 1000  # Number of particle

# Initial particle
# Uniform
xs_loc = np.random.uniform(-5, 5, size=(N_p, 3))
xs_loc[:, 2] = 0
xs_q = np.random.uniform(0.1, 10, size=(N_p, 1))
xs = np.hstack([xs_loc, xs_q])

ws = np.full(shape=(N_p), fill_value=1.0)
```

```python
def importance_sampling(weight, cnt, detectors, x):
    # TODO how to use loglikelihood for sampling?
    logli = sum_loglikelihood(cnts, detectors, x)
    weight_updated = weight * np.exp(logli)
    return weight_updated
```

```python
ax = world.visualize_world(detectors, figsize=(5, 5), plotsize=2)
print("Groundtruth:", gt_s)
for epoch in range(1):
    # Update
    w_updated = []
    for i in range(N_p):
        w_updated.append(importance_sampling(ws[i], cnts, detectors, xs[i]))

    # Normalize weight
    sum_w = sum(w_updated)
    w_updated = [w / sum_w for w in w_updated]

    # Resampling
    new_xs = []
    resamples = np.random.multinomial(N_p, w_updated)
    for i, n in enumerate(resamples):
        for _ in range(n):
            # Add noise because no transition model
            # Noise should be zero-centered
            noise = np.random.multivariate_normal(
                mean=[0, 0, 0, 0], cov=np.diag([0.05, 0.05, 0, 0.1])
            )
            new_xs.append(xs[i] + noise)
    xs = np.array(new_xs)
    xs[xs[:, 3] <= 0] = 0.01  # Intensity is always positive
    ws = np.full(shape=(N_p), fill_value=1.0)

    # Estimation
    est_x = xs.mean(axis=0)
    if not epoch % 10:
        print("Estimation:", est_x)
        ax.plot(est_x[0], est_x[1], "cd")
print("Final estimation:", est_x)
ax.plot(est_x[0], est_x[1], "yd")
ax.scatter(xs[:, 0], xs[:, 1], alpha=0.05)
```

## Fisher information matrix
https://ieeexplore.ieee.org/document/9266110
```
@article{anderson_mobile_2020,
	title = {Mobile {Robotic} {Radiation} {Surveying} with {Recursive} {Bayesian} {Estimation} and {Attenuation} {Modelling}},
	doi = {10.1109/TASE.2020.3036808},
	journal = {IEEE Transactions on Automation Science and Engineering},
	author = {Anderson, Blake and Pryor, Mitch and Abeyta, Adrian and Landsberger, Sheldon},
	month = nov,
	year = {2020},
	note = {Print ISSN: 1545-5955
Electronic ISSN: 1558-3783},
	pages = {1--15},
}
```

Question:  
Why information matrix is averaged in Eq.(31)? So, it doesn't mean more measuments guarantee better results?


```python
import sympy as sp
A, delta_X, delta_Y, delta_Z, r, T, mu = sp.symbols("A, \Delta{X}, \Delta{Y}, \Delta{Z}, r, T, mu")
r_sq = r*r
r_sqsq = r_sq*r_sq
tmp_mat = sp.Matrix([1/r/r, 2*A*delta_X/r_sqsq, 2*A*delta_Y/r_sqsq, 2*A*delta_Z/r_sqsq])
fim = T*T* 1/mu*tmp_mat*tmp_mat.T
fim
```

```python
def get_FIM(detector: Detector, loc: List[float], q: float) -> np.ndarray:
    """
    Get fisher information matrix for Poisson distribution

    Parameters
    ----------
    detector: Detector
        Detector
    loc: List[float]
        Location of the radiation source such that [x, y, z] [m].
    q: float
        Intensity of the radiation source [MBq]. (default is 1.0)

    Returns
    -------
    fim: np.ndarray
        fisher information matrix. fim \in R^{4x4}.
        [Be careful] The order is (q, x, y, z), so different than s (x, y, z, q).

    Reference
    ---------
    https://ieeexplore.ieee.org/document/9266110
    """

    # Factor
    factor = detector.duration * detector.factor

    # Locations
    detec_loc = np.array(detector.loc)
    src_loc = np.array(loc)

    # distance: r
    diff = src_loc - detec_loc
    r = np.linalg.norm(diff)

    # lambda for poisson
    lmd = factor * q / r / r

    # Calculate Fisher information matrix
    r_sq = r * r
    r_sqsq = r_sq * r_sq
    tmp_mat = np.array(
        [
            1 / r / r,
            2 * q * diff[0] / r_sqsq,
            2 * q * diff[1] / r_sqsq,
            2 * q * diff[2] / r_sqsq,
        ]
    )[:, np.newaxis]
    fim = factor * factor / lmd * tmp_mat.dot(tmp_mat.T)
    return fim


def get_average_FIM(detectors: List[Detector], target_s: List[float]) -> np.ndarray:
    """
    Get average FIM from detectors. Eq.(31) in the reference.

    Parameters
    ----------
    detectors: List[Detector]
        List of detectors.
    target_s:List[float]
        Target state [x, y, z, p]

    Returns
    -------
    ave_fim:  np.ndarray
        fisher information matrix. fim \in R^{4x4}.
        [Be careful] The order is (q, x, y, z), so different than s (x, y, z, q).

    Reference
    ---------
    https://ieeexplore.ieee.org/document/9266110
    """
    ave_fim = np.zeros((4,4))
    for detector in detectors:
        ave_fim += get_FIM(detector, target_s[:3], target_s[3])
    return ave_fim/len(detectors)
```

### Single source localization

```python
# Initial detector locations
detector_locations = [
    [0, -1, 0],
    [-1.5, 1, 0],
#     [1, 1, 0],
#     [0, 2, 0],
]

detectors = []
for loc in detector_locations:
    detectors.append(Detector(loc=loc))
cnts = world.get_measuments(detectors)
cnts

print(f"max_count: {cnts.max()}")
locs = np.array(detector_locations)
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(locs[:, 0], locs[:, 1], c=cnts, cmap=plt.cm.jet)
_ = fig.colorbar(sc, orientation="horizontal")
ax.set_aspect("equal")
```

```python
# Make sure if world is set correctly
assert len(world.sources) == 1, "should be single source"
# Groundtruth
gt_loc = world.sources[0].loc
gt_q = world.sources[0].intensity
gt_s = list(gt_loc + [gt_q])
print(f"groundtruth state: {gt_s}")
world.visualize_world(detectors, figsize=(5, 5), plotsize=2)
```

```python
N_p = 1000  # Number of particle

# Initial particle
# Uniform
xs_loc = np.random.uniform(-5, 5, size=(N_p, 3))
xs_loc[:, 2] = 0
xs_q = np.random.uniform(0.1, 10, size=(N_p, 1))
xs = np.hstack([xs_loc, xs_q])
ws = np.full(shape=(N_p), fill_value=1.0)

# Reset particle every time measurement added
is_everytime_reset_required = True

```

```python
def importance_sampling(weight, cnt, detectors, x):
    # TODO how to use loglikelihood for sampling?
    logli = sum_loglikelihood(cnts, detectors, x)
    weight_updated = weight * np.exp(logli)
    return weight_updated
```

```python
print("Groundtruth:", gt_s)
for epoch in range(6):
    # Every time reset particle
    if is_everytime_reset_required:
        # Uniform
        xs_loc = np.random.uniform(-5, 5, size=(N_p, 3))
        xs_loc[:, 2] = 0
        xs_q = np.random.uniform(0.1, 10, size=(N_p, 1))
        xs = np.hstack([xs_loc, xs_q])
        ws = np.full(shape=(N_p), fill_value=1.0)
        
    # Update
    w_updated = []
    for i in range(N_p):
        w_updated.append(importance_sampling(ws[i], cnts, detectors, xs[i]))

    # Normalize weight
    sum_w = sum(w_updated)
    w_updated = [w / sum_w for w in w_updated]

    # Resampling
    new_xs = []
    resamples = np.random.multinomial(N_p, w_updated)
    for i, n in enumerate(resamples):
        for _ in range(n):
            # Add noise because no transition model
            # Noise should be zero-centered
            noise = np.random.multivariate_normal(
                mean=[0, 0, 0, 0], cov=np.diag([0.05, 0.05, 0, 0.1])
            )
            new_xs.append(xs[i] + noise)
    xs = np.array(new_xs)
    xs[xs[:, 3] <= 0] = 0.01  # Intensity is always positive
    ws = np.full(shape=(N_p), fill_value=1.0)

    # Estimation
    est_x = xs.mean(axis=0)
    if not epoch % 1:
        print("Estimation:", est_x)
        ax = world.visualize_world(detectors, figsize=(5, 5), plotsize=2)
        ax.plot(est_x[0], est_x[1], "cd")
        ax.scatter(xs[::2, 0], xs[::2, 1], alpha=0.05)

        
    # Next measurement
    # Current robot position
    last_loc = detectors[-1].loc

    # Generate next step candidates near current position
    N_cand = 100
    steps = np.random.normal(scale=1, size=(N_cand,3))
    steps[:, 2] = 0 # Z is always zero
    steps = steps[np.linalg.norm(steps, axis=1)>0.5] # Remove point if too close
    next_locs = np.array(last_loc) + steps
    
    # Choose next location based on fisher info
    next_detec = deepcopy(detectors[-1])
    hist = []
    best_loc = None
    best_cost = np.inf
    for next_loc in next_locs:
        next_detec.loc = next_loc
        final_fim = np.zeros((4,4))
        for i in range(len(xs)):
            final_fim += get_average_FIM(detectors+[next_detec], xs[i])
        final_fim /= len(xs)
        # Z value is null
        final_fim = final_fim[:3,:3]
        # A-Optimality
        cost = np.trace(np.linalg.inv(final_fim))
        hist.append([next_loc, cost])
        if best_cost > cost:
            best_cost = cost
            best_loc = next_loc

    # Visualize cost
    ax = world.visualize_world(detectors, figsize=(5, 5), plotsize=2)
    ax.set_title("Visualize fisher info. Big magenta circle has higher fisher info")
    for loc, cost in hist:
        ax.scatter(loc[0], loc[1], c='m', s=3/cost, alpha=0.5)
    plt.show()
        
    # Measurement in best location
    next_detec.loc = best_loc
    detectors.append(next_detec)
    cnts = world.get_measuments(detectors)
    

print("Final estimation:", est_x)
ax = world.visualize_world(detectors, figsize=(5, 5), plotsize=2)
ax.plot(est_x[0], est_x[1], "yd")
ax.scatter(xs[::2, 0], xs[::2, 1], alpha=0.01)
```

```python

```

```python

```
