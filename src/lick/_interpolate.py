__all__ = ["Axis", "Limits"]

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from lick._typing import FArray1D

Method: TypeAlias = Literal["nearest", "linear", "cubic"]
Limit: TypeAlias = float | None


@dataclass(kw_only=True, slots=True, frozen=True)
class Limits:
    min: Limit
    max: Limit


@dataclass(kw_only=True, slots=True, frozen=True)
class Axis:
    ticks: FArray1D

    def as_evenly_spaced(
        self,
        size: int,
        *,
        new_min: Limit = None,
        new_max: Limit = None,
    ) -> "Axis":
        if new_min is None:
            new_min = self.ticks[0]
        if new_max is None:
            new_max = self.ticks[-1]
        dt = np.dtype("float64")

        return Axis(ticks=np.linspace(new_min, new_max, size, dtype=dt))
