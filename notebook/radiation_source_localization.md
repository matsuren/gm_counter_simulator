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

```python
import sys

sys.path.insert(0, "../")
```

```python
from simulator import Source, Detector, World
import numpy as np
import matplotlib.pyplot as plt
from typing import List
```

```python
# Construct world
world = World()
world.add_source(Source(loc=[5, 8, 0], intensity=1))
world.add_source(Source(loc=[-7, -2, 0], intensity=2.3))

ax = world.visualize_world()
```

```python
# Set detectors
div_num = 10
x_lin = np.linspace(-10, 10, div_num)
y_lin = np.linspace(10, -10, div_num)
x_grid, y_grid = np.meshgrid(x_lin, y_lin)

# Constant height
z = 1.5
detectors = []
for x, y in zip(x_grid.flatten(), y_grid.flatten()):
    detectors.append(Detector(loc=[x, y, z]))

cnts = world.get_measuments(detectors)
world.visualize_world(detectors)

print(f"max_count: {cnts.max()}")
viz = cnts.reshape(div_num, div_num)
plt.figure()
plt.imshow(viz, vmin=0, vmax=np.percentile(cnts, 99))
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

```python
from math import factorial
p = lambda k, l: l ** k * np.exp(-l) / factorial(k)
```

```python
l = 3.0
ret = []
for k in range(15):
    ret.append(p(k, l))
plt.plot(ret)
print(sum(ret))
```

## Construct world for example

```python
# Construct world
world = World()
world.add_source(Source(loc=[0, 1, 0], intensity=1))
world.visualize_world(figsize=(5, 5))
```

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
sc = ax.scatter(locs[:, 0], locs[:,1], c=cnts,cmap=plt.cm.jet)
_ = fig.colorbar(sc, orientation="horizontal")
ax.set_aspect("equal")
```

```python
world.visualize_world(detectors, figsize=(6, 6), plotsize=2)
```

```python
assert len(world.sources) == 1, "should be single source"
# Groundtruth
gt_loc = world.sources[0].loc
gt_q = world.sources[0].intensity
gt_s = list(gt_loc + [gt_q])
print(f"groundtruth state: {gt_s}")
```

```python
print("Check if the groundtruth gives maximum log likelihood")
diff = 0.1
for i in range(-1, 3):

    # Copy to keep original value
    target_s = list(gt_s)

    if i < 0:
        print("Groundtruth:", target_s)
    else:
        # Add perturbation
        target_s[i] += diff
        print("Add perturbation:", target_s)

    sum_logli = sum_loglikelihood(cnts, detectors, target_s)
    print(f" Likelihood: {sum_logli:.3f}")
```

## Cross-entropy method 
Reference: https://youtu.be/mJlAfKc4990?t=4296



```python
def sample(mu: List[float], std: List[float], num: int = 1) -> np.ndarray:
    """
    Sample location from Normal distribution(mu[:3], std[:3]**2)
    and intensity from Exponential distribution(mu[3])

    Parameters
    ----------
    mu:float
    std:float
    num:int

    Returns
    -------
    sample_s: ndarray
        Sampled states whose shape is (num, 4)
    """

    sample_loc = np.random.normal(mu[:3], std[:3], (num, 3))
    sample_q = np.random.exponential(mu[3], (num, 1))
    sample_s = np.hstack([sample_loc, sample_q])

    return sample_s
```

```python
world
```

```python
init_mu = [0, 0, 0, 0.1]
std = [0.5, 0.5, 0] # std[2]=0 so that z=0
eval_num = 100
top_n_percentile = 90

# Initial sample
mu = sample(init_mu, std).squeeze()
# Save current max
curr_max = -np.inf
curr_max_mu = None
for i in range(1000):
    # Sample 
    sample_s = sample(mu, std, eval_num)

    # Calculate loglikehood
    logli_hist = []
    for s in sample_s:
        logli = sum_loglikelihood(cnts, detectors, s)
        logli_hist.append(logli)
    # Decide threshold
    logli_hist = np.array(logli_hist)
    thresh = np.percentile(logli_hist, top_n_percentile)
    
    # Calculate next mu
    mu = sample_s[logli_hist > thresh].mean(axis=0)

    # Update max mu
    max_idx = np.argmax(logli_hist)
    if logli_hist[max_idx]>curr_max:
        curr_max = logli_hist[max_idx]
        curr_max_mu = sample_s[max_idx]
    
    if not i%100:
        print(f"[{i}] Max: {curr_max:.3f}, Param:{curr_max_mu}")

print('Estimation:', curr_max_mu)
print('Groundtruth:', gt_s)
```


