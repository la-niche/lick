__all__ = [
    "interpol",
    "lick",
    "lick_box",
    "lick_box_plot",
]
from lick._publib import interpol, lick, lick_box, lick_box_plot

from typing import Any


def __getattr__(item: str) -> Any:
    if item == "__version__":
        from importlib.metadata import version

        return version("lick")
    raise AttributeError(f"module {__name__!r} has no attribute {item!r}.")


del Any
