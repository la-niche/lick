from dataclasses import dataclass
from math import pi
from types import FunctionType
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from lick import lick_box_plot

FArray: TypeAlias = NDArray[np.dtype("float64")]


@dataclass(frozen=True, slots=True, kw_only=True)
class Range:
    min_value: float
    max_value: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Inputs:
    x: FArray
    y: FArray
    vx: FunctionType
    vy: FunctionType
    field: FunctionType
    xrange: Range
    yrange: Range


def radius(x: FArray, y: FArray) -> FArray:
    return np.hypot(x, y)


def azimuth(x: FArray, y: FArray) -> FArray:
    return np.arctan2(y, x)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(
            Inputs(
                x=np.linspace(-1, 1, 64),
                y=np.linspace(-1, 1, 64),
                vx=lambda xg, yg: radius(xg, yg) * np.cos(azimuth(xg, yg)),
                vy=lambda xg, yg: radius(xg, yg) * np.sin(azimuth(xg, yg)),
                field=lambda xg, yg, vx, vy: vx**2 + vy**2,
                xrange=Range(min_value=-1.0, max_value=1.0),
                yrange=Range(min_value=-1.0, max_value=1.0),
            ),
            id="radial-velocity",
        ),
        pytest.param(
            Inputs(
                x=np.geomspace(0.01, 10, 120),
                y=np.linspace(-7.0, 7.0, 128),
                vx=lambda xg, yg: np.sin(yg),
                vy=lambda xg, yg: np.cos(xg),
                field=lambda xg, yg, vx, vy: vx**2 + vy**2,
                xrange=Range(min_value=0.0, max_value=2 * pi),
                yrange=Range(min_value=-pi, max_value=pi),
            ),
            id="vortices",
        ),
    ],
)
@pytest.mark.mpl_image_compare()
def test_lick_img(inputs):
    fig, ax = plt.subplots()
    x = inputs.x
    y = inputs.y
    xg, yg = np.meshgrid(x, y, indexing="ij")
    vx = inputs.vx(xg, yg)
    vy = inputs.vy(xg, yg)
    field = inputs.field(xg, yg, vx, vy)
    lick_box_plot(
        fig,
        ax,
        x,
        y,
        vx,
        vy,
        field,
        size_interpolated=256,
        method="linear",
        xmin=inputs.xrange.min_value,
        xmax=inputs.xrange.max_value,
        ymin=inputs.yrange.min_value,
        ymax=inputs.yrange.max_value,
        niter_lic=5,
        kernel_length=64,
        cmap="inferno",
        stream_density=0.5,
    )
    ax.set(aspect="equal")
    return fig
