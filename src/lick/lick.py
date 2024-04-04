#!/usr/bin/env python
import warnings
from importlib.util import find_spec
from typing import TYPE_CHECKING, Optional

import numpy as np

import lick._vendor.vectorplot.core as _lic

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

HAS_MPL = find_spec("matplotlib") is not None


def _equalize_hist(image):
    # adapted from scikit-image
    """Return image after histogram equalization.

    Parameters
    ----------
    image : array
        Image array.

    Returns
    -------
    out : float array
        Image array after histogram equalization.

    Notes
    -----
    This function is adapted from [1]_ with the author's permission.

    References
    ----------
    .. [1] http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    .. [2] https://en.wikipedia.org/wiki/Histogram_equalization

    """
    hist, bin_edges = np.histogram(image.flatten(), bins=256, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])

    cdf = cdf.astype(image.dtype, copy=False)
    out = np.interp(image.flat, bin_centers, cdf)
    out = out.reshape(image.shape)
    # Unfortunately, np.interp currently always promotes to float64, so we
    # have to cast back to single precision when float32 output is desired
    return out.astype(image.dtype, copy=False)


def interpol(
    xx,
    yy,
    v1: np.ndarray,
    v2: np.ndarray,
    field: np.ndarray,
    *,
    method: str = "nearest",
    method_background: str = "nearest",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    size_interpolated: int = 800,
):
    from scipy.interpolate import griddata

    if xmin is None:
        xmin = xx.min()
    if xmax is None:
        xmax = xx.max()
    if ymin is None:
        ymin = yy.min()
    if ymax is None:
        ymax = yy.max()

    # evenly spaced grid (same spacing in x and y directions)
    nyi = size_interpolated
    nxi = int((xmax - xmin) / (ymax - ymin) * nyi)
    if nxi < nyi:
        nxi = size_interpolated
        nyi = int((ymax - ymin) / (xmax - xmin) * nxi)

    x = np.linspace(xmin, xmax, nxi)
    y = np.linspace(ymin, ymax, nyi)

    xi, yi = np.meshgrid(x, y)

    # then, interpolate your data onto this grid:

    px = xx.flatten()
    py = yy.flatten()
    pv1 = v1.flatten()
    pv2 = v2.flatten()
    pfield = field.flatten()

    gv1 = griddata((px, py), pv1, (xi, yi), method=method)
    gv2 = griddata((px, py), pv2, (xi, yi), method=method)
    gfield = griddata((px, py), pfield, (xi, yi), method=method_background)

    return (x, y, gv1, gv2, gfield)


def lick(
    v1: np.ndarray,
    v2: np.ndarray,
    *,
    niter_lic: int = 5,
    kernel_length: int = 101,
    light_source: bool = True,
):
    if v1.ndim != 2:
        raise ValueError(f"Expected a 2D array for v1, got v1 with shape {v1.shape}")
    if v2.ndim != 2:
        raise ValueError(f"Expected a 2D array for v2, got v2 with shape {v2.shape}")

    rng = np.random.default_rng(seed=0)
    texture = rng.normal(0.5, 0.001**0.5, v1.shape).astype(v1.dtype, copy=False)
    kernel = np.sin(np.arange(kernel_length, dtype=v1.dtype) * np.pi / kernel_length)

    out = np.empty_like(v1)
    image = texture
    for _ in range(niter_lic):
        out[:, :] = 0.0
        _lic.line_integral_convolution(v1, v2, image, kernel, out=out)
        image[:, :] = out[:, :]

    image = _equalize_hist(image)
    image /= image.max()

    if light_source:
        warnings.warn(
            "The light_source argument is deprecated and will "
            "be removed in a future version. "
            "Please use matplotlib.colors.LightSource directly",
            category=DeprecationWarning,
            stacklevel=2,
        )
        if not HAS_MPL:
            raise RuntimeError(
                "light_source=True requires matplotlib, which is missing"
            )

        from matplotlib.colors import LightSource

        # Illuminate the scene from the northwest
        ls = LightSource(azdeg=0, altdeg=45)
        image = ls.hillshade(image, vert_exag=5)

    return image


def lick_box(
    x: np.ndarray,
    y: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    field: np.ndarray,
    *,
    size_interpolated: int = 800,
    method: str = "nearest",
    method_background: str = "nearest",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    niter_lic: int = 5,
    kernel_length: int = 101,
    light_source: bool = True,
):
    if x.ndim == y.ndim == 2:
        yy = y
        xx = x
    elif x.ndim == y.ndim == 1:
        yy, xx = np.meshgrid(y, x)
    else:
        raise ValueError(
            f"Received 'x' with shape {x.shape}"
            f"and 'y' with shape {y.shape}. "
            "Expected them to be both 1D or 2D arrays with identical shapes"
        )
    xi, yi, v1i, v2i, fieldi = interpol(
        xx,
        yy,
        v1,
        v2,
        field,
        method=method,
        method_background=method_background,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        size_interpolated=size_interpolated,
    )
    Xi, Yi = np.meshgrid(xi, yi)
    licv = lick(
        v1i,
        v2i,
        niter_lic=niter_lic,
        kernel_length=kernel_length,
        light_source=light_source,
    )
    return (Xi, Yi, v1i, v2i, fieldi, licv)


def lick_box_plot(
    fig: "Figure",
    ax: "Axes",
    x: np.ndarray,
    y: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    field: np.ndarray,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size_interpolated: int = 800,
    method: str = "nearest",
    method_background: str = "nearest",
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    niter_lic: int = 5,
    kernel_length: int = 101,
    log: bool = False,
    cmap=None,
    nbin: Optional[int] = None,
    color_stream: str = "w",
    cmap_stream=None,
    light_source: bool = True,
    stream_density: float = 0,
    alpha_transparency: bool = True,
    alpha: float = 0.3,
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    Xi, Yi, v1i, v2i, fieldi, licv = lick_box(
        x,
        y,
        v1,
        v2,
        field,
        size_interpolated=size_interpolated,
        method=method,
        method_background=method_background,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        niter_lic=niter_lic,
        kernel_length=kernel_length,
        light_source=light_source,
    )

    if log:
        if not alpha_transparency:
            datalicv = np.log10(licv * fieldi)
        fieldi = np.log10(fieldi)
    elif not alpha_transparency:
        datalicv = licv * fieldi

    if vmin is None:
        vmin = fieldi.min()
    if vmax is None:
        vmax = fieldi.max()

    if alpha_transparency:
        im = ax.pcolormesh(
            Xi,
            Yi,
            fieldi,
            cmap=cmap,
            shading="nearest",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.pcolormesh(
            Xi, Yi, licv, cmap="gray", shading="nearest", alpha=alpha, rasterized=True
        )
    else:
        im = ax.pcolormesh(
            Xi, Yi, datalicv, cmap=cmap, shading="nearest", vmin=vmin, vmax=vmax
        )

    # print("pcolormesh")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")  # , format='%.0e')
    if stream_density > 0:
        ax.streamplot(
            Xi,
            Yi,
            v1i,
            v2i,
            density=stream_density,
            arrowstyle="->",
            linewidth=0.8,
            color=color_stream,
            cmap=cmap_stream,
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # print("streamplot")


if __name__ == "__main__":
    if not HAS_MPL:
        raise RuntimeError(f"running {__file__} as a script requires matplotlib")

    import matplotlib.pyplot as plt

    cmap = "inferno"
    fig, ax = plt.subplots()
    x = np.geomspace(0.1, 10, 128)
    y = np.geomspace(0.1, 5, 128)
    a, b = np.meshgrid(x, y)
    v1 = np.cos(a)
    v2 = np.sin(b)
    field = v1**2 + v2**2
    lick_box_plot(
        fig,
        ax,
        x,
        y,
        v1,
        v2,
        field,
        cmap=cmap,
        kernel_length=64,
        stream_density=0.5,
        niter_lic=5,
        size_interpolated=256,
    )
    plt.show()
