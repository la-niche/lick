__all__ = [
    "interpol",
    "lick",
    "lick_box",
    "lick_box_plot",
]
import sys
from functools import partial
from typing import TYPE_CHECKING, Generic, Literal, cast

import numpy as np
import rlic

from lick import _api
from lick._image_processing import (
    AlphaDict,
    HistogramEqualizer,
    ImageProcessor,
    MixMulDict,
)
from lick._interpolation import Grid, Interval, Mesh, Method
from lick._typing import D, F, FArray, FArray1D, FArray2D

if sys.version_info >= (3, 11):
    from typing import NamedTuple, assert_never
else:
    from typing_extensions import NamedTuple, assert_never

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class InterpolationResults(NamedTuple, Generic[F]):
    x_ticks: FArray1D[F]
    y_ticks: FArray1D[F]
    v1: FArray2D[F]
    v2: FArray2D[F]
    field: FArray2D[F]


def interpol(
    x: FArray[D, F],
    y: FArray[D, F],
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
    indexing: Literal["xy", "ij"] | _api.UnsetType = _api.UNSET,
) -> InterpolationResults[F]:
    if len(all_dtypes := {_.dtype for _ in (x, y, v1, v2, field)}) > 1:
        raise TypeError(f"Received inputs with mixed datatypes ({all_dtypes})")
    input_grid_or_mesh = _api.get_grid_or_mesh(x, y)  # type: ignore[arg-type]

    target_grid = Grid.from_intervals(
        x=Interval(
            min=float(x.min()),
            max=float(x.max()),
        ).with_overrides(min=xmin, max=xmax),
        y=Interval(
            min=float(y.min()),
            max=float(y.max()),
        ).with_overrides(min=ymin, max=ymax),
        small_dim_npoints=size_interpolated,
        dtype=cast(F, x.dtype),
    )

    target_mesh = Mesh.from_grid(target_grid, indexing="xy")
    interpolate = _api.get_interpolator(
        input_grid_or_mesh, target_mesh=target_mesh, indexing=indexing
    )

    return InterpolationResults(
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
    niter_lic: int | _api.UnsetType = _api.UNSET,
    kernel: FArray1D[F]
    | Literal["auto-adjust", "auto-legacy"]
    | _api.UnsetType = _api.UNSET,
    kernel_length: int | _api.UnsetType = _api.UNSET,
    post_lic: Literal[None, "north-west-light-source"]
    | ImageProcessor
    | _api.UnsetType = _api.UNSET,
    light_source: bool | _api.UnsetType = _api.UNSET,
) -> FArray2D[F]:
    if len(all_dtypes := {_.dtype for _ in (v1, v2)}) > 1:
        raise TypeError(f"Received inputs with mixed datatypes ({all_dtypes})")
    niter_lic = _api.get_niter_lic(niter_lic)
    kernel = _api.get_kernel(
        kernel,
        size=kernel_length,
        max_auto_size=min(*v1.shape, *v2.shape),
        dtype=v1.dtype,  # type: ignore[arg-type]
    )
    post_lic = _api.get_post_lic(post_lic, light_source=light_source)

    rng = np.random.default_rng(seed=0)
    texture = rng.normal(0.5, 0.001**0.5, v1.shape).astype(v1.dtype, copy=False)

    image = rlic.convolve(texture, v1, v2, kernel=kernel, iterations=niter_lic)
    processors: list[ImageProcessor] = [HistogramEqualizer(nbins=256), post_lic]
    for ip in processors:
        image = ip.process(image)

    return image


class LickBoxResults(NamedTuple, Generic[F]):
    x_grid: FArray2D[F]
    y_grid: FArray2D[F]
    v1: FArray2D[F]
    v2: FArray2D[F]
    field: FArray2D[F]
    licv: FArray2D[F]


def lick_box(
    x: FArray[D, F],
    y: FArray[D, F],
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
    niter_lic: int | _api.UnsetType = _api.UNSET,
    kernel: FArray1D[F]
    | Literal["auto-adjust", "auto-legacy"]
    | _api.UnsetType = _api.UNSET,
    kernel_length: int | _api.UnsetType = _api.UNSET,
    post_lic: Literal[None, "north-west-light-source"]
    | ImageProcessor
    | _api.UnsetType = _api.UNSET,
    light_source: bool | _api.UnsetType = _api.UNSET,
    indexing: Literal["xy", "ij"] | _api.UnsetType = _api.UNSET,
) -> LickBoxResults[F]:
    if len(all_dtypes := {_.dtype for _ in (x, y, v1, v2, field)}) > 1:
        raise TypeError(f"Received inputs with mixed datatypes ({all_dtypes})")
    grid_or_mesh = _api.get_grid_or_mesh(x, y)  # type: ignore[arg-type]
    niter_lic = _api.get_niter_lic(niter_lic)
    kernel = _api.get_kernel(
        kernel,
        size=kernel_length,
        max_auto_size=size_interpolated,
        dtype=v1.dtype,  # type: ignore[arg-type]
    )
    post_lic = _api.get_post_lic(post_lic, light_source=light_source)

    ir = interpol(
        grid_or_mesh.x,
        grid_or_mesh.y,
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
        indexing=indexing,
    )
    Xi, Yi = np.meshgrid(ir.x_ticks, ir.y_ticks)
    licv = lick(
        ir.v1,
        ir.v2,
        kernel=kernel,
        niter_lic=niter_lic,
        post_lic=post_lic,
    )
    return LickBoxResults(Xi, Yi, ir.v1, ir.v2, ir.field, licv)


def lick_box_plot(
    fig: "Figure",
    ax: "Axes",
    x: FArray[D, F],
    y: FArray[D, F],
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
    niter_lic: int | _api.UnsetType = _api.UNSET,
    kernel: FArray1D[F]
    | Literal["auto-adjust", "auto-legacy"]
    | _api.UnsetType = _api.UNSET,
    kernel_length: int | _api.UnsetType = _api.UNSET,
    post_lic: Literal[None, "north-west-light-source"]
    | ImageProcessor
    | _api.UnsetType = _api.UNSET,
    light_source: bool | _api.UnsetType = _api.UNSET,
    log: bool = False,
    cmap=None,
    color_stream: str = "white",
    cmap_stream=None,
    stream_density: float = 0,
    alpha_transparency: bool | _api.UnsetType = _api.UNSET,
    alpha: float | _api.UnsetType = _api.UNSET,
    layering: AlphaDict | MixMulDict | _api.UnsetType = _api.UNSET,
    indexing: Literal["xy", "ij"] | _api.UnsetType = _api.UNSET,
) -> LickBoxResults[F]:
    if len(all_dtypes := {_.dtype for _ in (x, y, v1, v2, field)}) > 1:
        raise TypeError(f"Received inputs with mixed datatypes ({all_dtypes})")
    grid_or_mesh = _api.get_grid_or_mesh(x, y)  # type: ignore[arg-type]
    niter_lic = _api.get_niter_lic(niter_lic)
    kernel = _api.get_kernel(
        kernel,
        size=kernel_length,
        max_auto_size=size_interpolated,
        dtype=v1.dtype,  # type: ignore[arg-type]
    )
    post_lic = _api.get_post_lic(post_lic, light_source=light_source)
    resolved_layering = _api.get_layering(
        layering, alpha=alpha, alpha_transparency=alpha_transparency
    )

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    lbr = lick_box(
        grid_or_mesh.x,
        grid_or_mesh.y,
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
        kernel=kernel,
        niter_lic=niter_lic,
        post_lic=post_lic,
        indexing=indexing,
    )

    new_field = np.log10(lbr.field) if log else lbr.field
    im_kwargs = {
        "cmap": cmap,
        "vmin": new_field.min() if vmin is None else vmin,
        "vmax": new_field.max() if vmax is None else vmax,
    }
    pcolormesh = partial(
        ax.pcolormesh, lbr.x_grid, lbr.y_grid, rasterized=True, shading="nearest"
    )
    match resolved_layering.mode:
        case _api.LayeringMode.ALPHA:
            im = pcolormesh(new_field, **im_kwargs)
            pcolormesh(lbr.licv, cmap="gray", alpha=resolved_layering.alpha)
        case _api.LayeringMode.MIX_MUL:
            datalicv = lbr.licv * lbr.field
            datalicv = np.log10(datalicv) if log else datalicv
            im = pcolormesh(datalicv, **im_kwargs)
        case _ as unreachable:
            assert_never(unreachable)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    if stream_density > 0:
        ax.streamplot(
            lbr.x_grid,
            lbr.y_grid,
            lbr.v1,
            lbr.v2,
            density=stream_density,
            arrowstyle="->",
            linewidth=0.8,
            color=color_stream,
            cmap=cmap_stream,
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # return the output straight from lick_box
    return lbr
