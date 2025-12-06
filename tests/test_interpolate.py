import numpy as np
import numpy.testing as npt
import pytest

from lick._interpolate import Axis, Limits


@pytest.mark.parametrize(
    "input_ticks",
    [
        np.geomspace(1, 100, 64),
        np.linspace(-10, 10, 100),
    ],
)
@pytest.mark.parametrize("output_size", [5, 8, 64, 128])
def test_axis_as_evenly_spaced(input_ticks, output_size):
    a1 = Axis(ticks=input_ticks)
    a2 = a1.as_evenly_spaced(output_size)

    npt.assert_almost_equal(
        a2.ticks,
        np.linspace(a1.ticks.min(), a1.ticks.max(), output_size, dtype="float64"),
    )


@pytest.mark.parametrize(
    "input_ticks",
    [
        np.geomspace(1, 100, 64),
        np.linspace(-10, 10, 100),
    ],
)
@pytest.mark.parametrize("output_size", [5, 8, 64, 128])
@pytest.mark.parametrize(
    "limits",
    [
        Limits(min=None, max=None),
        Limits(min=-1.0, max=1.0),
        Limits(min=None, max=1.0),
        Limits(min=-1.0, max=None),
    ],
)
def test_axis_as_evenly_spaced_with_limits(input_ticks, output_size, limits):
    a1 = Axis(ticks=input_ticks)
    a2 = a1.as_evenly_spaced(output_size, new_min=limits.min, new_max=limits.max)

    npt.assert_array_almost_equal(np.diff(np.diff(a2.ticks)), 0.0)
    npt.assert_array_almost_equal(np.diff(np.diff(a2.ticks)), 0.0)

    assert a2.ticks.min() == Limits.min or a1.ticks.min()
    assert a2.ticks.max() == Limits.max or a1.ticks.max()
