__all__ = [
    "get_grid_or_mesh",
    "get_indexing",
    "get_interpolator",
    "get_kernel",
    "get_layering",
    "get_niter_lic",
    "get_mesh",
    "get_post_lic",
    "UNSET",
    "UnsetType",
]
import sys
import warnings
from enum import Enum, auto
from typing import Any, Literal, overload

from lick._image_processing import (
    AlphaDict,
    Identity,
    ImageProcessor,
    Layering,
    LayeringMode,
    MixMulDict,
    NorthWestLightSource,
)
from lick._interpolation import (
    Grid,
    Interpolator,
    Mesh,
    RegularGridInterpolator,
    UnstruscturedGridInterpolator,
)
from lick._typing import D, F, FArray, FArray1D, FArray2D

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never


class UnsetType(Enum):
    # a type checker-friendly sentinel value
    UNSET = auto()


UNSET = UnsetType.UNSET


class LegacyDefault(Enum):
    # uniqueness isn't required
    # this enum only provides a namespace to allow matching
    KERNEL_SIZE = 101
    KERNEL = "auto-legacy"
    NITER_LIC = 5
    POST_LIC = "north-west-light-source"
    ALPHA = 0.3
    ALPHA_TRANSPARENCY = True
    INDEXING = "xy"


class NoDefaultType(Enum):
    NODEFAULT = auto()


LEGACY_DEFAULT_USED_MSG = (
    "The {kw} argument was not explicitly specified. "
    "{expect_change} in a future release.\n"
    "To silence this warning, set the argument explicitly."
)


def warn_legacy_default_used(
    kw: str, *, legacy_default: LegacyDefault, future_default: Any | NoDefaultType
) -> None:
    if future_default is NoDefaultType.NODEFAULT:
        change = (
            f"Falling back {legacy_default.value!r} as a default value, "
            "but this parameter will be required"
        )
    else:
        change = f"Its default value will change from {legacy_default.value!r} to {future_default!r}"

    warnings.warn(
        LEGACY_DEFAULT_USED_MSG.format(kw=kw, expect_change=change),
        DeprecationWarning,
        stacklevel=4,
    )


LEGACY_VALUE_USED_MSG = (
    "The {kw} argument was set to {legacy_value!r}. "
    "This value is scheduled for removal in a future version. "
    "To silence this warning, set {kw}={alt}"
)


def warn_legacy_value_used(kw: str, *, legacy_value: Any, alt: str) -> None:
    warnings.warn(
        LEGACY_VALUE_USED_MSG.format(kw=kw, legacy_value=legacy_value, alt=alt),
        PendingDeprecationWarning,
        stacklevel=4,
    )


LEGACY_KW_USED_MSG = (
    "The {kw} keyword argument is deprecated since lick v{since_version} "
    "and will be removed in a future version. Use the {alt_kw} argument instead."
)


def warn_legacy_kw_used(kw: str, *, alt_kw: str, since_version: str) -> None:
    warnings.warn(
        LEGACY_KW_USED_MSG.format(kw=kw, alt_kw=alt_kw, since_version=since_version),
        DeprecationWarning,
        stacklevel=4,
    )


MUTUALLY_EXCLUSIVE_KW_MSG = "{kw} and {alt_kw} keyword arguments are mutually exclusive, but both were received."


def get_niter_lic(niter_lic: int | UnsetType) -> int:
    if niter_lic is UNSET:
        warn_legacy_default_used(
            kw="niter_lic", legacy_default=LegacyDefault.NITER_LIC, future_default=1
        )
        return LegacyDefault.NITER_LIC.value
    else:
        return niter_lic


def get_kernel(
    kernel: FArray1D[F] | Literal["auto-adjust", "auto-legacy"] | UnsetType,
    *,
    size: int | UnsetType,
    max_auto_size: int,
    dtype: F,
) -> FArray1D[F]:
    import numpy as np

    if kernel is not UNSET and size is not UNSET:
        raise TypeError(
            MUTUALLY_EXCLUSIVE_KW_MSG.format(kw="kernel", alt_kw="kernel_length")
        )

    if isinstance(kernel, np.ndarray):
        return kernel

    if size is not UNSET:
        warn_legacy_kw_used(kw="kernel_length", alt_kw="kernel", since_version="0.10.0")
        kernel = "auto-legacy"
    elif kernel is UNSET:
        warn_legacy_default_used(
            kw="kernel",
            legacy_default=LegacyDefault.KERNEL,
            future_default="auto-adjust",
        )
        kernel = LegacyDefault.KERNEL.value
    elif kernel == LegacyDefault.KERNEL.value:
        warn_legacy_value_used(
            kw="kernel",
            legacy_value=LegacyDefault.KERNEL.value,
            alt=f"np.sin(np.linspace(0, np.pi, {LegacyDefault.KERNEL_SIZE.value}, endpoint=False)).astype(<ref_dtype>)",
        )

    match kernel:
        case "auto-adjust":
            size = min(21, max_auto_size)
            kernel_base = np.sin(np.linspace(0, np.pi, size + 2))[1:-1]
        case LegacyDefault.KERNEL.value:
            if size is UNSET:
                size = LegacyDefault.KERNEL_SIZE.value
            kernel_base = np.sin(np.linspace(0, np.pi, size, endpoint=False))
        case _ as unreachable:
            assert_never(unreachable)

    return kernel_base.astype(dtype, copy=False)  # type: ignore[no-any-return]


def get_post_lic(
    post_lic: Literal[None, "north-west-light-source"] | ImageProcessor | UnsetType,
    *,
    light_source: bool | UnsetType,
) -> ImageProcessor:
    if post_lic is not UNSET and light_source is not UNSET:
        raise TypeError(
            MUTUALLY_EXCLUSIVE_KW_MSG.format(kw="post_lic", alt_kw="light_source")
        )

    if light_source is not UNSET:
        warn_legacy_kw_used(
            kw="light_source", alt_kw="post_lic", since_version="0.10.0"
        )
        post_lic = LegacyDefault.POST_LIC.value if light_source else None

    if post_lic is UNSET:
        warn_legacy_default_used(
            kw="post_lic",
            legacy_default=LegacyDefault.POST_LIC,
            future_default=None,
        )
        post_lic = LegacyDefault.POST_LIC.value

    match post_lic:
        case LegacyDefault.POST_LIC.value:
            return NorthWestLightSource()
        case None:
            return Identity()
        case _:
            return post_lic


def get_indexing(indexing: Literal["xy", "ij"] | UnsetType) -> Literal["xy", "ij"]:
    if indexing is UNSET:
        warn_legacy_default_used(
            kw="indexing",
            legacy_default=LegacyDefault.INDEXING,
            future_default=NoDefaultType.NODEFAULT,
        )
        return LegacyDefault.INDEXING.value

    if indexing in ["xy", "ij"]:
        return indexing

    raise ValueError(f"Received invalid {indexing=!r}")


@overload
def get_grid_or_mesh(x: FArray1D[F], y: FArray1D[F]) -> Grid[F]: ...
@overload
def get_grid_or_mesh(x: FArray2D[F], y: FArray2D[F]) -> Mesh[F]: ...
def get_grid_or_mesh(x: FArray[D, F], y: FArray[D, F]) -> Grid[F] | Mesh[F]:
    if x.ndim == y.ndim == 1:
        return Grid(x=x, y=y)  # type: ignore[arg-type]

    if x.ndim == y.ndim == 2:
        return Mesh(x=x, y=y)  # type: ignore[arg-type]

    raise TypeError(
        f"Received {x.shape=} and {y.shape=}. "
        "Expected them to have identical dimensionalities."
    )


def get_mesh(
    x: FArray[D, F], y: FArray[D, F], indexing=Literal["xy", "ij", UNSET]
) -> Mesh[F]:
    match grid_or_mesh := get_grid_or_mesh(x, y):  # type: ignore[arg-type]
        case Grid():
            indexing = get_indexing(indexing)
            return Mesh.from_grid(grid_or_mesh, indexing=indexing)
        case Mesh():
            if indexing is not UNSET:
                warnings.warn(
                    f"{indexing=!r} will be ignored because a mesh is already defined",
                    stacklevel=3,
                )
            return grid_or_mesh
        case _ as unreachable:
            assert_never(unreachable)


def get_interpolator(
    input_grid_or_mesh: Grid | Mesh,
    *,
    target_mesh: Mesh,
    indexing: Literal["xy", "ij"] | UnsetType,
) -> Interpolator:
    match input_grid_or_mesh:
        case Mesh() as m:
            return UnstruscturedGridInterpolator(
                input_mesh=m,
                target_mesh=target_mesh,
            )
        case Grid() as g if indexing == "xy" and g.is_mono_increasing():
            return RegularGridInterpolator(
                input_grid=g,
                target_mesh=target_mesh,
            )
        case Grid() as g:
            indexing = get_indexing(indexing)
            return get_interpolator(
                Mesh.from_grid(g, indexing=indexing),
                target_mesh=target_mesh,
                indexing=UNSET,
            )
        case _ as unreachable:
            assert_never(unreachable)


def get_layering(
    layering: AlphaDict | MixMulDict | UnsetType,
    *,
    alpha: float | UnsetType,
    alpha_transparency: bool | UnsetType,
) -> Layering:
    if layering is not UNSET and (alpha is not UNSET):
        raise TypeError(MUTUALLY_EXCLUSIVE_KW_MSG.format(kw="alpha", alt_kw="layering"))
    if layering is not UNSET and (alpha_transparency is not UNSET):
        raise TypeError(
            MUTUALLY_EXCLUSIVE_KW_MSG.format(kw="alpha_transparency", alt_kw="layering")
        )

    if layering is not UNSET:
        return Layering.from_dict(layering)

    if alpha_transparency is not UNSET:
        warn_legacy_kw_used(
            kw="alpha_transparency", alt_kw="layering", since_version="0.10.0"
        )
    else:
        alpha_transparency = LegacyDefault.ALPHA_TRANSPARENCY.value

    if alpha is not UNSET:
        warn_legacy_kw_used(kw="alpha", alt_kw="layering", since_version="0.10.0")
    else:
        alpha = LegacyDefault.ALPHA.value

    if alpha_transparency:
        return Layering(mode=LayeringMode.ALPHA, alpha=alpha)
    else:
        return Layering(mode=LayeringMode.MIX_MUL)
