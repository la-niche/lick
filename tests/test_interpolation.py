import re
from functools import partial
from itertools import permutations

import numpy as np
import numpy.testing as npt
import pytest

from lick._interpolation import Grid, Interpolator, Interval, Mesh, Monotonic
from lick.lick import interpol

f64 = np.float64


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param({"min": +1.0, "max": -1.0}, id="unsorted"),
        pytest.param({"min": 1.0, "max": 1.0}, id="zero-span"),
    ],
)
def test_interval_invalid_extrema(inputs):
    min = inputs["min"]
    max = inputs["max"]
    with pytest.raises(
        ValueError,
        match=rf"^max must be greater than min\. Got {min=}, {max=}$",
    ):
        Interval(min=min, max=max)


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param({"min": float("nan"), "max": 1.0}, id="minnan"),
        pytest.param({"min": 0.0, "max": float("nan")}, id="maxnan"),
        pytest.param({"min": float("nan"), "max": float("nan")}, id="allnans"),
        pytest.param({"min": float("-inf"), "max": 1.0}, id="mininf"),
        pytest.param({"min": 0.0, "max": float("inf")}, id="maxinf"),
        pytest.param({"min": float("inf"), "max": float("inf")}, id="allinfs"),
    ],
)
def test_interval_nonfinite_extrema(inputs):
    min = inputs["min"]
    max = inputs["max"]
    with pytest.raises(
        ValueError,
        match=rf"^max and min must both be finite\. Got {min=}, {max=}$",
    ):
        Interval(min=min, max=max)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"min": -2.0, "max": 2.0}, id="min+max"),
        pytest.param({"min": -2.0}, id="min"),
        pytest.param({"max": 2.0}, id="max"),
        pytest.param({}, id="no-override"),
    ],
)
def test_interval_with_overrides(overrides):
    L1 = Interval(min=-1.0, max=1.0)
    L2 = L1.with_overrides(**overrides)
    assert L2 is not L1
    if overrides:
        assert L2 != L1
    else:
        assert L2 == L1
    assert L2.min == overrides.get("min", L1.min)
    assert L2.max == overrides.get("max", L1.max)


@pytest.mark.parametrize(
    "interval, expected",
    [
        pytest.param(Interval(min=0, max=1.0), 1.0, id="0_pos"),
        pytest.param(Interval(min=-1.0, max=0.0), 1.0, id="neg_0"),
        pytest.param(Interval(min=-1.0, max=1.0), 2.0, id="neg_pos"),
    ],
)
def test_interval_span(interval, expected):
    assert interval.span == expected


@pytest.mark.parametrize(
    "interval",
    [
        pytest.param(Interval(min=0, max=1.0), id="0_pos"),
        pytest.param(Interval(min=-1.0, max=0.0), id="neg_0"),
        pytest.param(Interval(min=-1.0, max=1.0), id="neg_pos"),
    ],
)
@pytest.mark.parametrize("size", [5, 8, 64, 128])
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_interval_as_evenly_spaced_array(interval, size, dtype):
    a = interval.as_evenly_spaced_array(size, dtype=dtype)

    assert a.dtype == dtype
    assert a.shape == (size,)
    npt.assert_almost_equal(np.diff(a, 2), 0.0, decimal=14 if dtype == "float64" else 6)


def test_monotonic_invalid_array():
    array = np.array([1, 2, 4, 3], dtype="float32")
    with pytest.raises(
        ValueError,
        match=(
            r"^Expected a monotonic base array "
            r"\(either increasing or decreasing order\)$"
        ),
    ):
        Monotonic(array)


def test_monotonic_is_decreasing(subtests):
    array = np.arange(5.0, dtype="float32")
    with subtests.test("direct"):
        assert not Monotonic(array).is_decreasing()
    with subtests.test("reversed"):
        assert Monotonic(array[::-1]).is_decreasing()


def test_monotonic_as_inreasing_array(subtests):
    array = np.arange(5.0, dtype="float32")[::-1]
    ref = Monotonic(array)
    with subtests.test("direct"):
        assert ref.is_decreasing()
    with subtests.test("reversed"):
        assert not Monotonic(ref.as_increasing_array()).is_decreasing()


@pytest.mark.parametrize("dtx, dty", permutations(["float32", "float64"]))
def test_grid_mixed_dtype(dtx, dty):
    x = np.geomspace(1, 2, 5, dtype=dtx)
    y = np.linspace(1, 2, 5, dtype=dty)
    with pytest.raises(
        TypeError,
        match=(
            "x and y must be 1D arrays with the same data type. "
            f"Got {x.ndim=}, {x.dtype=!s}, {y.ndim=}, {y.dtype=!s}"
        ),
    ):
        Grid(x=x, y=y)


BASE_INTERVALS = [
    Interval(min=0.0, max=1.0),
    Interval(min=-1.0, max=1.0),
    Interval(min=-1.0, max=0.0),
    Interval(min=0.0, max=128.0),
    Interval(min=-99.0, max=15.0),
]


@pytest.mark.parametrize("x", BASE_INTERVALS)
@pytest.mark.parametrize("y", BASE_INTERVALS)
@pytest.mark.parametrize("small_dim_npoints", [-1, 0, 1])
def test_grid_from_intervals_too_small_npoints(x, y, small_dim_npoints):
    with pytest.raises(
        ValueError, match=rf"^Received {small_dim_npoints=}, expected at least 2$"
    ):
        Grid.from_intervals(
            x=x,
            y=y,
            small_dim_npoints=small_dim_npoints,
            dtype="float64",
        )


@pytest.mark.parametrize("x", BASE_INTERVALS)
@pytest.mark.parametrize("y", BASE_INTERVALS)
@pytest.mark.parametrize("small_dim_npoints", [2, 5, 64])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_grid_from_intervals(x, y, small_dim_npoints, dtype, subtests):
    g = Grid.from_intervals(x=x, y=y, small_dim_npoints=small_dim_npoints, dtype=dtype)
    with subtests.test("check small_dim_npoints"):
        assert min(g.x.size, g.y.size) == small_dim_npoints

    with subtests.test("check array sizes hierarchy"):
        if x.span < y.span:
            assert g.x.size <= g.y.size
        elif x.span > y.span:
            assert g.x.size >= g.y.size
        else:
            assert g.x.size == g.y.size

    with subtests.test("check dtypes"):
        assert g.x.dtype == dtype
        assert g.y.dtype == dtype


@pytest.mark.parametrize(
    "x, y",
    [
        pytest.param(
            np.arange(16, dtype="float32"),
            np.arange(16, dtype="float32"),
            id="1d-arrays",
        ),
        pytest.param(
            np.eye(4, dtype="float32"),
            np.eye(4, dtype="float64"),
            id="mismatched-dtypes",
        ),
        pytest.param(
            np.eye(4, dtype="float32"),
            np.eye(5, dtype="float32"),
            id="mismatched-shapes",
        ),
    ],
)
def test_mesh_invalid_arrays(x, y):
    with pytest.raises(
        TypeError,
        match=re.escape(
            r"x and y must be 2D arrays with the same data type and shape. "
            rf"Got {x.shape=}, {x.dtype=!s}, {y.shape=}, {y.dtype=!s}"
        ),
    ):
        Mesh(x=x, y=y)


def test_mesh_from_grid_invalid_indexing():
    x = np.geomspace(1, 2, 5)
    y = np.linspace(3, 4, 7)
    g = Grid(x=x, y=y)

    with pytest.raises(ValueError, match="indexing"):
        # no exact match: the error message is controled by numpy
        Mesh.from_grid(g, indexing="spam")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("indexing", ["xy", "ij"])
def test_mesh_from_grid(dtype, indexing):
    x = np.geomspace(1, 2, 5, dtype=dtype)
    y = np.linspace(3, 4, 7, dtype=dtype)
    grid = Grid(x=x, y=y)

    match indexing:
        case "xy":
            expected_shape = (y.size, x.size)
        case "ij":
            expected_shape = (x.size, y.size)
        case _ as unreachable:
            raise AssertionError(unreachable)

    mesh = Mesh.from_grid(grid, indexing=indexing)
    assert mesh.dtype == grid.dtype
    assert mesh.shape == expected_shape


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("indexing", ["xy", "ij"])
def test_interpolator_dunder_call(subtests, dtype, indexing):
    x = np.geomspace(1, 2, 5, dtype=dtype)
    y = np.linspace(3, 4, 7, dtype=dtype)
    grid = Grid(x=x, y=y)
    mesh = Mesh.from_grid(grid, indexing=indexing)

    interpolator = Interpolator(grid=grid, target_mesh=mesh)
    for method in ["nearest", "linear", "cubic"]:
        with subtests.test(method=method):
            res = interpolator(
                # https://github.com/la-niche/lick/issues/246
                mesh.x.T if indexing == "xy" else mesh.x,
                method=method,
            )
            npt.assert_array_almost_equal_nulp(res, mesh.x)


@pytest.mark.parametrize("dt1, dt2", permutations(["float32", "float64"]))
def test_interpolator_dunder_call_mixed_dtype(subtests, dt1, dt2):
    x = np.geomspace(1, 2, 5, dtype=dt1)
    y = np.linspace(3, 4, 7, dtype=dt1)
    grid = Grid(x=x, y=y)
    mesh = Mesh.from_grid(grid, indexing="ij")

    interpolator = Interpolator(grid=grid, target_mesh=mesh)
    msg_template = (
        f"Expected all inputs to match this interpolator's grid data type ({grid.dtype!s}). "
        "Received vals.dtype={!s}, target_mesh.dtype={!s}"
    )
    with (
        subtests.test(vals_dtype=dt2, mesh_dtype=dt1),
        pytest.raises(TypeError, match=re.escape(msg_template.format(dt2, dt1))),
    ):
        interpolator(mesh.x.astype(dt2), method="nearest")

    interpolator = Interpolator(grid=grid, target_mesh=mesh.astype(dt2))
    with (
        subtests.test(vals_dtype=dt1, mesh_dtype=dt2),
        pytest.raises(TypeError, match=re.escape(msg_template.format(dt1, dt2))),
    ):
        interpolator(mesh.x, method="nearest")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_variable_precision_interpol_inputs(dtype, indexing, subtests):
    prng = np.random.default_rng(0)
    shape = (4, 3)
    size = np.prod(shape)

    xv = np.linspace(0, 10, shape[0], dtype=dtype)
    yv = np.linspace(0, 20, shape[1], dtype=dtype)
    v1, v2, field = [prng.random(size, dtype=dtype).reshape(shape) for _ in range(3)]
    xx, yy = np.meshgrid(xv, yv, indexing=indexing)
    xo, yo, v1o, v2o, fieldo = interpol(
        xx,
        yy,
        field,
        v1,
        v2,
        xmin=None,
        xmax=10.0,
        ymin=1,
        ymax=None,
        size_interpolated=10,
    )

    with subtests.test():
        assert yo.dtype == xo.dtype == dtype
    with subtests.test():
        assert v2o.dtype == v1o.dtype == dtype
    with subtests.test():
        assert fieldo.dtype == v2o.dtype == dtype


def test_decreasing_coordinates(subtests):
    xv = np.geomspace(2.0, 20.0, 8)
    yv = np.geomspace(1.0, 10.0, 5)
    prng = np.random.default_rng(0)
    shape = (xv.size, yv.size)
    size = np.prod(shape)

    v1, v2, field = [prng.random(size, dtype=xv.dtype).reshape(shape) for _ in range(3)]

    xx0, yy0 = np.meshgrid(xv, yv)
    ref = interpol(xx0, yy0, field, v1, v2, size_interpolated=10)

    xx1, yy1 = np.meshgrid(xv[::-1], yv)
    op1 = rop1 = partial(np.flip, axis=0)
    new_res = interpol(xx1, yy1, op1(field), op1(v1), op1(v2), size_interpolated=10)
    for arr, ref_arr in zip(new_res, ref, strict=True):
        with subtests.test(flipped="x"):
            if ref_arr.ndim == 2:
                comp = rop1(arr)
            else:
                comp = arr
            npt.assert_array_equal(comp, ref_arr)

    xx2, yy2 = np.meshgrid(xv, yv[::-1])
    op2 = rop2 = partial(np.flip, axis=1)
    new_res = interpol(xx2, yy2, op2(field), op2(v1), op2(v2), size_interpolated=10)
    for arr, ref_arr in zip(new_res, ref, strict=True):
        with subtests.test(flipped="x"):
            if ref_arr.ndim == 2:
                comp = rop2(arr)
            else:
                comp = arr
            npt.assert_array_equal(comp, ref_arr)

    xx3, yy3 = np.meshgrid(xv[::-1], yv[::-1])

    def op3(a):
        return np.flip(np.flip(a, 0), 1)

    def rop3(a):
        return np.flip(np.flip(a, 1), 0)

    new_res = interpol(xx3, yy3, op3(field), op3(v1), op3(v2), size_interpolated=10)
    for arr, ref_arr in zip(new_res, ref, strict=True):
        with subtests.test(flipped="xy"):
            if ref_arr.ndim == 2:
                comp = rop3(arr)
            else:
                comp = arr
            npt.assert_array_equal(arr, ref_arr)
