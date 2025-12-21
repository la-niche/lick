__all___ = [
    "AlphaDict",
    "HistogramEqualizer",
    "Identity",
    "ImageProcessor",
    "MixMulDict",
    "NorthWestLightSource",
]
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal, Protocol, TypedDict

from lick._typing import F, FArray2D

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never


class AlphaDict(TypedDict):
    alpha: float


class MixMulDict(TypedDict):
    mix: Literal["mul"]


class ImageProcessor(Protocol):
    def process(self, image: FArray2D[F]) -> FArray2D[F]: ...


class Identity:
    __slots__ = []

    def process(self, image: FArray2D[F]) -> FArray2D[F]:
        return image


class NorthWestLightSource:
    __slots__ = []

    def process(self, image: FArray2D[F]) -> FArray2D[F]:
        from matplotlib.colors import LightSource

        ls = LightSource(azdeg=0.0, altdeg=45.0)
        return ls.hillshade(image, vert_exag=5).astype(image.dtype, copy=False)  # type: ignore[no-any-return]


@dataclass(kw_only=True, slots=True, frozen=True)
class HistogramEqualizer:
    nbins: int

    def process(self, image: FArray2D[F]) -> FArray2D[F]:
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
        import numpy as np
        from interpn import interpn

        shape = image.shape
        flat_image = image.ravel()

        hist, bin_edges = np.histogram(flat_image, bins=self.nbins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        cdf = hist.cumsum(dtype=image.dtype)
        cdf /= cdf[-1]  # fast normalization

        return interpn(
            grids=[bin_centers],
            obs=[flat_image],
            vals=cdf,
            method="linear",
        ).reshape(shape)  # type: ignore[return-value]


class LayeringMode(Enum):
    ALPHA = auto()
    MIX_MUL = auto()


@dataclass(kw_only=True, slots=True, frozen=True)
class Layering:
    mode: LayeringMode
    alpha: float | None = None

    def __post_init__(self):
        match self.mode, self.alpha:
            case LayeringMode.ALPHA, None:
                raise TypeError(
                    "mode=LayeringMode.ALPHA is not compatible with alpha=None"
                )
            case LayeringMode.ALPHA, float(alpha) if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"{alpha=} is invalid. Expected 0.0 <= alpha <= 1.0")
            case LayeringMode.ALPHA, float():
                pass
            case LayeringMode.MIX_MUL, None:
                pass
            case LayeringMode.MIX_MUL, _:
                raise TypeError("mode=LayeringMode.MIX_MUL requires alpha=None")
            case _ as unreachable:
                assert_never(unreachable)

    @classmethod
    def from_dict(cls, d: AlphaDict | MixMulDict, /) -> "Layering":
        match d:
            case {"alpha": float(alpha)} if len(d) == 1:
                return Layering(mode=LayeringMode.ALPHA, alpha=alpha)
            case {"mix": "mul"} if len(d) == 1:
                return Layering(mode=LayeringMode.MIX_MUL, alpha=None)
            case _:
                raise ValueError(f"Failed to parse layering={d}")
