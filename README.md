# lick
[![PyPI](https://img.shields.io/pypi/v/lick)](https://pypi.org/project/lick/)

Line Integral Convolution Knit : Package that uses a Line Integral Convolution library to clothe a 2D field (ex: density field) with a LIC texture, given two vector fields (ex: velocity (vx, vy)).
Version: 0.1.0
Author: Gaylor Wafflard-Fernandez
Author-email: gaylor.wafflard@univ-grenoble-alpes.fr

<p align="center">
    <img src="https://raw.githubusercontent.com/volodia99/lick/main/imgs/lick.png" width="800"></a>
</p>

## Installation

Install with `pip`

```
pip install lick
```

To import lick:

```python
import lick as lk
```

The important functions are ```lick_box``` and ```lick_box_plot```. While ```lick_box``` interpolates the data and perform a line integral convolution, ```lick_box_plot``` directly plots the final image. Use ```lick_box``` if you want to have more control of the plots you want to do with the lic. Use ```lick_box_plot``` if you want to take advantage of the fine-tuning of the pcolormesh parameters.

## Example

```python
import numpy as np
import matplotlib.pyplot as plt
from lick import lick_box_plot

fig, ax = plt.subplots()
x = np.geomspace(0.1, 10, 128)
y = np.geomspace(0.1, 5, 128)
a, b = np.meshgrid(x, y)
v1 = np.cos(a)
v2 = np.sin(b)
field = v1 ** 2 + v2 ** 2
lick_box_plot(
    fig,
    ax,
    x,
    y,
    v1,
    v2,
    field,
    size_interpolated=256,
    xmin=1,
    xmax=9,
    ymin=1,
    ymax=4,
    niter_lic=5,
    kernel_length=64,
    cmap="inferno",
    stream_density=0.5
)
plt.show()
```
