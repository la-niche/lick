__all__ = [
    "Grid",
    "Interpolator",
    "Interval",
    "Mesh",
    "Method",
]

from dataclasses import KW_ONLY, dataclass
from math import isfinite
from typing import Generic, Literal, Protocol, TypeAlias, final

import numpy as np

from lick._typing import F, FArray1D, FArray2D

Method: TypeAlias = Literal["nearest", "linear", "cubic"]


@final
@dataclass(kw_only=True, slots=True, frozen=True)
class Interval:
    min: float
    max: float

    def __post_init__(self):
        if not (isfinite(self.min) and isfinite(self.max)):
            msg = "max and min must both be finite"
        elif self.max <= self.min:
            # 0-width intervals are not allowed so we can guarantee that
            # as_evenly_spaced_array always returns unique values
            msg = "max must be greater than min"
        else:
            return
        raise ValueError(f"{msg}. Got min={self.min}, max={self.max}")

    def with_overrides(
        self,
        *,
        min: float | None = None,
        max: float | None = None,
    ) -> "Interval":
        return Interval(
            min=float(min) if min is not None else self.min,
            max=float(max) if max is not None else self.max,
        )

    @property
    def span(self) -> float:
        return self.max - self.min

    def as_evenly_spaced_array(self, size: int, *, dtype: F) -> FArray1D[F]:
        return np.linspace(self.min, self.max, size, dtype=dtype)


@final
@dataclass(kw_only=True, slots=True, frozen=True)
class Grid(Generic[F]):
    x: FArray1D[F]
    y: FArray1D[F]

    def __post_init__(self):
        if self.y.dtype == self.x.dtype and self.y.ndim == self.x.ndim == 1:
            return

        raise TypeError(
            "x and y must be 1D arrays with the same data type. "
            f"Got x.ndim={self.x.ndim}, x.dtype={self.x.dtype!s}, "
            f"y.ndim={self.y.ndim}, y.dtype={self.y.dtype!s}"
        )

    @classmethod
    def from_intervals(
        cls,
        *,
        x: Interval,
        y: Interval,
        small_dim_npoints: int,
        dtype: F,
    ) -> "Grid[F]":
        s = small_dim_npoints
        if s < 2:
            raise ValueError(f"Received {small_dim_npoints=}, expected at least 2")
        if (xy_ratio := x.span / y.span) >= 1:
            size_x = int(s * xy_ratio)
            size_y = s
        else:
            size_x = s
            size_y = int(s / xy_ratio)

        return Grid(
            x=x.as_evenly_spaced_array(size_x, dtype=dtype),
            y=y.as_evenly_spaced_array(size_y, dtype=dtype),
        )

    @property
    def dtype(self) -> np.dtype[F]:
        return self.x.dtype

    def is_mono_increasing(self) -> bool:
        return bool(np.all(np.diff(self.x) > 0.0) and np.all(np.diff(self.y) > 0.0))


@final
@dataclass(kw_only=True, slots=True, frozen=True)
class Mesh(Generic[F]):
    x: FArray2D[F]
    y: FArray2D[F]

    def __post_init__(self):
        if (
            self.y.dtype == self.x.dtype
            and self.y.ndim == self.x.ndim == 2
            and self.y.shape == self.x.shape
        ):
            return

        raise TypeError(
            "x and y must be 2D arrays with the same data type and shape. "
            f"Got x.shape={self.x.shape}, x.dtype={self.x.dtype!s}, "
            f"y.shape={self.y.shape}, y.dtype={self.y.dtype!s}"
        )

    @classmethod
    def from_grid(cls, grid: Grid[F], *, indexing: Literal["xy", "ij"]) -> "Mesh[F]":
        x, y = np.meshgrid(grid.x, grid.y, indexing=indexing)
        return Mesh(x=x, y=y)

    @property
    def dtype(self) -> np.dtype[F]:
        return self.x.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape

    def astype(self, dtype: F, /) -> "Mesh[F]":
        return Mesh(x=self.x.astype(dtype), y=self.y.astype(dtype))


@final
@dataclass(slots=True, frozen=True)
class UnstruscturedGridInterpolator(Generic[F]):
    input_mesh: Mesh[F]
    _: KW_ONLY
    target_mesh: Mesh[F]

    def __post_init__(self):
        if self.target_mesh.dtype == self.input_mesh.dtype:
            return
        raise TypeError(
            "input and target must use the same data type. "
            f"Got input_mesh.dtype={self.input_mesh.dtype!s}, target_mesh.dtype={self.target_mesh.dtype!s}"
        )

    def __call__(
        self,
        vals: FArray2D[F],
        /,
        *,
        method: Method,
    ) -> FArray2D[F]:
        if vals.dtype != self.input_mesh.dtype or vals.shape != self.input_mesh.shape:
            raise TypeError(
                f"Expected values to match the input data type ({self.input_mesh.dtype}) "
                f"and shape {self.input_mesh.shape}. "
                f"Received values with dtype={vals.dtype!s}, shape={vals.shape}"
            )
        from scipy.interpolate import griddata

        return griddata(  # type: ignore[no-any-return]
            points=(
                self.input_mesh.x.flat,
                self.input_mesh.y.flat,
            ),
            values=vals.flat,
            xi=(
                self.target_mesh.x,
                self.target_mesh.y,
            ),
            method=method,
        ).astype(vals.dtype)


class Interpolator(Protocol, Generic[F]):
    def __call__(self, vals: FArray2D[F], /, *, method: Method) -> FArray2D[F]: ...


@final
@dataclass(slots=True, frozen=True)
class RegularGridInterpolator(Generic[F]):
    input_grid: Grid[F]
    _: KW_ONLY
    target_mesh: Mesh[F]

    def __post_init__(self):
        if self.target_mesh.dtype == self.input_grid.dtype:
            return
        raise TypeError(
            "input and target must use the same data type. "
            f"Got input_grid.dtype={self.input_grid.dtype!s}, target_mesh.dtype={self.target_mesh.dtype!s}"
        )

    def __call__(
        self,
        vals: FArray2D[F],
        /,
        *,
        method: Method,
    ) -> FArray2D[F]:
        if (
            vals.shape
            != (input_shape := (self.input_grid.y.size, self.input_grid.x.size))
            or vals.dtype != self.input_grid.dtype
        ):
            raise TypeError(
                f"Expected values to match the input data type ({self.input_grid.dtype}) "
                f"and shape {input_shape}. "
                f"Received values with dtype={vals.dtype!s}, shape={vals.shape}"
            )
        from interpn import interpn

        # TODO: disable extrapolation for backward compat
        # upstream patch required
        return interpn(
            grids=(self.input_grid.y, self.input_grid.x),
            obs=(self.target_mesh.y, self.target_mesh.x),
            vals=vals,
            method=method,
        )
