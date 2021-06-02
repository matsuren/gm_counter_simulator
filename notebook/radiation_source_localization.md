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
```

```python
# Construct world
world = World()
world.add_source(Source(loc=np.array([5,8,0]), intensity=1))
world.add_source(Source(loc=np.array([-7,-2,0]), intensity=2.3))

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
    detectors.append(Detector(loc=np.array([x, y, z])))
    
cnts = world.get_measuments(detectors)
print(f"max_count: {cnts.max()}")
viz = cnts.reshape(div_num, div_num)
plt.imshow(viz, vmin=0, vmax=np.percentile(cnts, 99))
```
