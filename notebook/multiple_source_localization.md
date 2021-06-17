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

# Multiple source localization

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
from collections import defaultdict
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

# Multiple Source Localization
There are $N$ sources.  
State: $\mathbf{s} =(\mathbf{x}_{0} ,q_{0} ,\mathbf{x}_{1} ,q_{1} ,\cdots )\in \mathbb{R}^{4N}$, where $q_{i}$ is intensity of i-th radiation source, and $\mathbf{x}_{i} =[x_{i} ,y_{i} ,z_{i} ]^{\top }$   
Measurement: $p_{i} \in \mathbb{R}$, where $y_{i}$ is the number of count detected by detector $i$ located at $\mathbf{y}_{i} \in \mathbb{R}^{3}$.

The probability of $y_{i}$ is formulated as follows:
\begin{gather*}
y_{i} \sim \mathrm{Pois} (\lambda _{i} )=\frac{\lambda ^{y_{i}} e^{-\lambda }}{y_{i} !} ,\\
\lambda _{i} =\Gamma \sum _{j=0}^{N-1}\frac{q_{j}}{||\mathbf{x}_{j} -\mathbf{y}_{i} ||^{2}} ,
\end{gather*}

E.g., the likelihood after knowing three measurements $\mathcal{P} =\{p_{i} :0\leqq i\leqq 2\}$

\begin{gather*}
\mathrm{P}(\mathbf{s} |\mathcal{P}) =\prod _{i \in |\mathcal{P}|}\mathrm{Pois}( \lambda _{i}) ,\\
\log\mathrm{P}(\mathbf{s} |\mathcal{P}) =\sum _{i \in |\mathcal{P}|}( y_{i}\log \lambda _{i} -\lambda _{i} -\log y_{i}!) ,
\end{gather*}


### Question
If the number of sources varies, how to calculate the mean of the particles.  
Some numbers of sources never come back once the number of sources is dissapear.  
Or generate a lot of number of sources and if the intensity is closer to zero, the candidate is None.

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
    cnts: List[int], detectors: List[Detector], target_s: List[List[float]]
) -> float:
    """
    Sum loglikelihood from observation.

    Parameters
    ----------
    cnts: List[int]
        Counts from multiple measurements
    detectors: List[Detector]
        List of detectors.
    target_s: List[List[float]]
        Target state [[x0, y0, z0, p0], [x1, y1, z1, p1], ....]

    Returns
    -------
    logli: float
        Sum of log likelihood from multiple measurements
    """
    assert len(cnts) == len(detectors)
    sum_logli = []
    for y, d in zip(cnts, detectors):
        # Sum of multiple sources
        lmd = 0.0
        for single_s in target_s:
            lmd += get_pois_lambda(d, single_s[:3], single_s[3])
        # Log likelihood
        logli = get_loglikelihood(y, lmd)
        sum_logli.append(logli)
    return sum(sum_logli)
```

## Construct world for example

```python
# Construct world
world = World()
world.add_source(Source(loc=[2.5, 4, 0], intensity=1))
world.add_source(Source(loc=[-3.5, -1, 0], intensity=2.3))
world.add_source(Source(loc=[3.5, -1, 0], intensity=1.6))
# world.add_source(Source(loc=[-3, 4, 0], intensity=2.3))

ax = world.visualize_world(figsize=(5,5), plotsize=5)
```

```python
# Set detectors (locations are known)
div_num = 5
x_lin = np.linspace(-6, 6, div_num)
y_lin = np.linspace(6, -6, div_num)
x_grid, y_grid = np.meshgrid(x_lin, y_lin)

# Constant height
# z = 0.3
z = 0.2
detectors = []
for x, y in zip(x_grid.flatten(), y_grid.flatten()):
    detectors.append(Detector(loc=[x, y, z]))
world.visualize_world(detectors, figsize=(5,5), plotsize=5.5)

cnts = world.get_measuments(detectors)
print(f"max_count: {cnts.max()}")
locs = np.array([d.loc for d in detectors])
fig, ax = plt.subplots(1,1, figsize=(5,5))
sc = ax.scatter(locs[:, 0], locs[:, 1], c=cnts, cmap=plt.cm.jet)
_ = fig.colorbar(sc, orientation="horizontal")
ax.set_title("Measure count")
ax.set_aspect("equal")
```

```python
print("Number of radiation sources:", len(world.sources))
gt_s: List[List[float]] = []
for src in world.sources:
    print(src)
    gt_s.append(src.loc + [src.intensity])
print("##########################")
print(f"Groundtruth state: {gt_s}")
```

```python
from itertools import product
from copy import deepcopy
print("Check if the groundtruth gives maximum log likelihood")
sum_logli = sum_loglikelihood(cnts, detectors, gt_s)
print(f" Groundtruth Likelihood: {sum_logli:.3f}")
```

```python
# diff = 0.5
# for i, j in product([0, 1], [0, 1, 2]):
#     # Copy to keep original value
#     target_s = deepcopy(gt_s)

#     # Add perturbation
#     target_s[i][j] += diff
#     print("Add perturbation:", target_s)

#     sum_logli = sum_loglikelihood(cnts, detectors, target_s)
#     print(f" Likelihood: {sum_logli:.3f}")
```

### Multiple source localization

```python
# Candidate of how many sources exist in world
num_src_candidate = [2, 3, 4, 5]
# num_src_candidate = [3, 4, 5, 6]
# Number of particle for one candidate
N_p_candidate = 1000
# N_p_candidate = 3000
# Actual particle sizi is N_p_candidate * len(num_src_candidate)
N_p = len(num_src_candidate) * N_p_candidate
```

```python
# Initial particle
# Uniform
xs = []
for num in num_src_candidate:
    xs_loc = np.random.uniform(-7, 7, size=(N_p_candidate, num, 3))
    xs_loc[..., 2] = 0
    xs_q = np.random.uniform(0.1, 10, size=(N_p_candidate, num, 1))
    xs_locq = np.concatenate([xs_loc, xs_q], axis=2)
    xs += xs_locq.tolist()
ws = np.full(shape=(len(xs)), fill_value=1.0)
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
n_epoch = 10
for epoch in range(n_epoch):
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
            new_x = []
            for el in xs[i]:
                # Add noise because no transition model
                # Noise should be zero-centered
                noise = np.random.multivariate_normal(
                    mean=[0, 0, 0, 0], cov=np.diag([0.05, 0.05, 0, 0.1])
                )
                new_el = np.array(el) + noise
                new_el[3] = max(0.01, new_el[3])  # Intensity is always positive
                new_x.append(new_el.tolist())
            new_xs.append(new_x)

    xs = new_xs

    if (not epoch % 2) or (epoch == n_epoch - 1):
        # Mean xs for each candidate (number of sources)
        cand_mean_xs = defaultdict(list)
        for x in xs:
            cand_mean_xs[len(x)].append(x)
        for key in cand_mean_xs.keys():
            cand_mean_xs[key] = np.array(cand_mean_xs[key]).mean(axis=0)
        print(f"[{epoch}] Pred:{cand_mean_xs}")

        # Plot
        if len(cand_mean_xs) == 1:
            for key, val in cand_mean_xs.items():
                ax.plot(val[:, 0], val[:, 1], "cd")
print("Final estimation:", cand_mean_xs.values())
# Plot
if len(cand_mean_xs) == 1:
    for key, val in cand_mean_xs.items():
        ax.plot(val[:, 0], val[:, 1], "yd")
```

# The curse of dimensionality
I guess at least one of the initial particles should be close to the groundtruth. However, it get harder as the dimmention is increase.

```python
N_src = 4
dim = 4 * N_src
N = 100000

# Groundtruth
gt = np.random.uniform(-0.5, 0.5, dim)
# Initial particle
particles = np.random.uniform(-0.5, 0.5, (N, dim))
diff = np.linalg.norm(gt - particles, axis=1)
diff.sort()
```

```python
x = []
y = []
fig, ax = plt.subplots(1, 1)
for ratio in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
    n = (diff < ratio).sum()
    if n > 0:
        print(n, ratio)
        x.append(ratio)
        y.append(n)
    else:
        print("Not found in ", ratio)
ax.plot(x, y)
```

## MCMC
### NumPyro

```python
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
```

```python
numpyro.enable_validation(True)
numpyro.set_platform("cpu")

draws = 10000
chains = 1
N_src = 3
```

```python
# N_src = 3
# Y = None
# with handlers.seed(rng_seed=0):
#     # Prior
#     prior_mu = jnp.array([0, 0, 0])
#     prior_scale = jnp.array([5, 5, 0.01])

#     rad_locs = []
#     rad_qs = []
#     for i in range(N_src):
#         rad_locs.append(
#             numpyro.sample(
#                 f"rad{i}_loc",
#                 dist.Normal(loc=prior_mu, scale=prior_scale),
#             )
#         )
#         rad_qs.append(numpyro.sample(f"rad{i}_q", dist.Exponential(rate=0.1)))

#     # Lambda for Poisson
#     lmds = []
#     for detec in detectors:
#         lmd = 0
#         factor = detec.duration * detec.factor
#         for i in range(N_src):
#             d = jnp.linalg.norm(rad_locs[i] - jnp.array(detec.loc))
#             lmd += factor*rad_qs[i]/d/d
#         lmds.append(lmd)
#     lmds = numpyro.deterministic("lambda", jnp.array(lmds))
#     Y_obs = numpyro.sample("obs", dist.Poisson(lmds), obs=Y)
```

```python
def model(detectors, Y=None, N_src=2):
    # Prior
    prior_mu = jnp.array([0, 0, 0])
    prior_scale = jnp.array([5, 5, 0.01])

    rad_locs = []
    rad_qs = []
    for i in range(N_src):
        rad_locs.append(
            numpyro.sample(
                f"rad{i}_loc",
                dist.Normal(loc=prior_mu, scale=prior_scale),
            )
        )
        rad_qs.append(numpyro.sample(f"rad{i}_q", dist.Exponential(rate=0.2)))

    # Lambda for Poisson
    lmds = []
    for detec in detectors:
        lmd = 0
        factor = detec.duration * detec.factor
        for i in range(N_src):
            d = jnp.linalg.norm(rad_locs[i] - jnp.array(detec.loc))
            lmd += factor*rad_qs[i]/d/d
        lmds.append(lmd)
    lmds = numpyro.deterministic("lambda", jnp.array(lmds))
    Y_obs = numpyro.sample("obs", dist.Poisson(lmds), obs=Y)
    return Y_obs
```

```python
Y = jnp.array(cnts)
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_samples=draws, num_warmup=1000, num_chains=chains)
rng_key = jax.random.PRNGKey(0)
mcmc.run(rng_key, detectors, Y, N_src)
```

```python
print("Groundtruth:", gt_s)
```

```python
mcmc.print_summary()
```

```python
az.plot_trace(mcmc, var_names=["rad0_loc", "rad0_q"], combined=False);
```

```python
az.plot_trace(mcmc, var_names=["rad1_loc", "rad1_q"], combined=False);
```

```python
ax = world.visualize_world(detectors, figsize=(5, 5), plotsize=2)
s = mcmc.get_samples()
for i in range(N_src):
    ax.scatter(s[f"rad{i}_loc"][-100:, 0], s[f"rad{i}_loc"][-100:, 1], c="y", alpha=0.05)
    ax.scatter(s[f"rad{i}_loc"][:, 0].mean(), s[f"rad{i}_loc"][:, 1].mean(), c="c")
```

```python

```
