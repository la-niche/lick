from typing import TYPE_CHECKING, cast

import numpy as np
import rlic

from lick._interpolation import Grid, Interpolator, Interval, Mesh, Method, Monotonic
from lick._typing import F, FArray1D, FArray2D, FArrayND

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=None)
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
    xx: FArray2D[F],
    yy: FArray2D[F],
    v1: FArray2D[F],
    v2: FArray2D[F],
    field: FArray2D[F],
    *,
    method: Method = "nearest",
    method_background: Method = "nearest",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    size_interpolated: int = 800,
) -> tuple[FArray1D[F], FArray1D[F], FArray2D[F], FArray2D[F], FArray2D[F]]:
    if len(all_dtypes := {_.dtype for _ in (xx, yy, v1, v2, field)}) > 1:
        raise TypeError(f"Received inputs with mixed datatypes ({all_dtypes})")

    if np.ptp(xx[:, 0]) == 0.0:
        input_indexing = "xy"
        x = xx[0, :]
        y = yy[:, 0]
    else:
        input_indexing = "ij"
        x = xx[:, 0]
        y = yy[0, :]

    x_mono = Monotonic(x)
    y_mono = Monotonic(y)

    inputs_grid = Grid(x=x_mono.as_increasing_array(), y=y_mono.as_increasing_array())
    target_grid = Grid.from_intervals(
        x=Interval(
            min=float(xx.min()),
            max=float(xx.max()),
        ).with_overrides(min=xmin, max=xmax),
        y=Interval(
            min=float(yy.min()),
            max=float(yy.max()),
        ).with_overrides(min=ymin, max=ymax),
        small_dim_npoints=size_interpolated,
        dtype=cast(F, xx.dtype),
    )

    interpolator = Interpolator(
        grid=inputs_grid, target_mesh=Mesh.from_grid(target_grid, indexing="xy")
    )

    def interpolate(vals: FArray2D[F], /, *, method: Method) -> FArray2D[F]:
        nonlocal xx, yy
        if x_mono.is_decreasing():
            vals = np.flip(vals, axis=0)
        if y_mono.is_decreasing():
            vals = np.flip(vals, axis=1)
        if input_indexing == "xy":
            # https://github.com/la-niche/lick/issues/246
            vals = vals.T

        ret = interpolator(vals, method=method)

        if input_indexing == "xy":
            ret = ret.T
        if y_mono.is_decreasing():
            ret = np.flip(ret, axis=1)
        if x_mono.is_decreasing():
            ret = np.flip(ret, axis=0)

        return ret

    return (
        target_grid.x,
        target_grid.y,
        interpolate(v1, method=method),
        interpolate(v2, method=method),
        interpolate(field, method=method_background),
    )


def lick(
    v1: FArray2D[F],
    v2: FArray2D[F],
    *,
    niter_lic: int = 5,
    kernel_length: int = 101,
    light_source: bool = True,
) -> FArray2D[F]:
    if len(all_dtypes := {_.dtype for _ in (v1, v2)}) > 1:
        raise TypeError(f"Received inputs with mixed datatypes ({all_dtypes})")
    rng = np.random.default_rng(seed=0)
    texture = rng.normal(0.5, 0.001**0.5, v1.shape).astype(v1.dtype, copy=False)
    kernel = np.sin(np.arange(kernel_length, dtype=v1.dtype) * np.pi / kernel_length)

    image = rlic.convolve(texture, v1, v2, kernel=kernel, iterations=niter_lic)
    image = _equalize_hist(image)
    image /= image.max()

    if light_source:
        from matplotlib.colors import LightSource

        # Illuminate the scene from the northwest
        ls = LightSource(azdeg=0, altdeg=45)
        image = ls.hillshade(image, vert_exag=5)

    return image


def lick_box(
    x: FArrayND[F],
    y: FArrayND[F],
    v1: FArray2D[F],
    v2: FArray2D[F],
    field: FArray2D[F],
    *,
    size_interpolated: int = 800,
    method: Method = "nearest",
    method_background: Method = "nearest",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    niter_lic: int = 5,
    kernel_length: int = 101,
    light_source: bool = True,
) -> tuple[
    FArray2D[F], FArray2D[F], FArray2D[F], FArray2D[F], FArray2D[F], FArray2D[F]
]:
    if len(all_dtypes := {_.dtype for _ in (x, y, v1, v2, field)}) > 1:
        raise TypeError(f"Received inputs with mixed datatypes ({all_dtypes})")

    yy: FArray2D
    xx: FArray2D
    if x.ndim == y.ndim == 2:
        yy = cast(FArray2D, y)
        xx = cast(FArray2D, x)
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
    x: FArrayND[F],
    y: FArrayND[F],
    v1: FArray2D[F],
    v2: FArray2D[F],
    field: FArray2D[F],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    size_interpolated: int = 800,
    method: Method = "nearest",
    method_background: Method = "nearest",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    niter_lic: int = 5,
    kernel_length: int = 101,
    log: bool = False,
    cmap=None,
    color_stream: str = "white",
    cmap_stream=None,
    light_source: bool = True,
    stream_density: float = 0,
    alpha_transparency: bool = True,
    alpha: float = 0.3,
) -> None:
    if len(all_dtypes := {_.dtype for _ in (x, y, v1, v2, field)}) > 1:
        raise TypeError(f"Received inputs with mixed datatypes ({all_dtypes})")

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
