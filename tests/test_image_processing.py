import numpy as np
import numpy.testing as npt
import pytest

from lick._image_processing import (
    HistogramEqualizer,
    Identity,
    Layering,
    LayeringMode,
    NorthWestLightSource,
)


def test_identity():
    prng = np.random.default_rng()
    array = prng.random((8, 7))
    processor = Identity()
    result = processor.process(array)
    npt.assert_array_equal(result, array)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_north_west_list_source(dtype):
    prng = np.random.default_rng()
    array = prng.random((8, 7)).astype(dtype)
    processor = NorthWestLightSource()
    result = processor.process(array)
    assert result.dtype == dtype
    assert not np.any(result == array)


@pytest.mark.parametrize(
    "nbins, min_rms_reduction",
    [
        (12, 2.0),
        (64, 10.0),
        (256, 50.0),
    ],
)
def test_histogram_equalization(nbins, min_rms_reduction):
    # histogram equalization produces a new image whose cumulative
    # distribution function (cdf) should be close(r) to a straight line
    # (i.e., approaching a flat intensity distribution)
    # This test check this property from an initial image made of gaussian
    # noise.
    # Expected rms reduction factors (min_rms_reduction) are empirical, i.e.,
    # slightly looser than what the original implementation was able to achieve

    IMAGE_SHAPE = (256, 128)
    prng = np.random.default_rng(0)
    image = prng.normal(size=np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE)

    equalizer = HistogramEqualizer(nbins=nbins)

    def normalized_cdf(a):
        hist, bin_edges = np.histogram(a.ravel(), bins=equalizer.nbins)
        cdf = hist.cumsum()
        return cdf / float(cdf.max())

    def rms(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    image_eq = equalizer.process(image)
    cdf_in = normalized_cdf(image)
    cdf_eq = normalized_cdf(image_eq)

    id_func = np.linspace(0, 1, equalizer.nbins)
    rms_in = rms(cdf_in, id_func)
    rms_eq = rms(cdf_eq, id_func)
    assert (rms_eq / rms_in) < min_rms_reduction


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_alpha_layering(alpha):
    L = Layering(mode=LayeringMode.ALPHA, alpha=alpha)
    assert L.mode is LayeringMode.ALPHA
    assert L.alpha == alpha


def test_mixmul_layering():
    L = Layering(mode=LayeringMode.MIX_MUL)
    assert L.mode is LayeringMode.MIX_MUL
    assert L.alpha is None


@pytest.mark.parametrize(
    "kwargs, expected_msg",
    [
        pytest.param(
            {"mode": LayeringMode.ALPHA, "alpha": None},
            "mode=LayeringMode.ALPHA is not compatible with alpha=None",
            id="missing-alpha",
        ),
        pytest.param(
            {"mode": LayeringMode.MIX_MUL, "alpha": 0.3},
            "mode=LayeringMode.MIX_MUL requires alpha=None",
            id="inconsistent-params",
        ),
    ],
)
def test_layering_invalid_types(kwargs, expected_msg):
    with pytest.raises(TypeError, match=rf"^{expected_msg}$"):
        Layering(**kwargs)


@pytest.mark.parametrize(
    "alpha", [float("-inf"), -2.0, 1.2, float("inf"), float("nan")]
)
def test_layering_invalid_alpha(alpha):
    with pytest.raises(
        ValueError, match=rf"^{alpha=} is invalid\. Expected 0\.0 <= alpha <= 1\.0$"
    ):
        Layering(mode=LayeringMode.ALPHA, alpha=alpha)


@pytest.mark.parametrize(
    "d, expected",
    [
        pytest.param(
            {"alpha": 0.4}, Layering(mode=LayeringMode.ALPHA, alpha=0.4), id="alphadict"
        ),
        pytest.param(
            {"mix": "mul"}, Layering(mode=LayeringMode.MIX_MUL), id="mixmuldict"
        ),
    ],
)
def test_layering_from_dict(d, expected):
    L = Layering.from_dict(d)
    assert L == expected


@pytest.mark.parametrize(
    "d",
    [
        pytest.param({}, id="empty-dict"),
        pytest.param({"alpha": 0.4, "mix": "mul"}, id="ambiguous-dict"),
        pytest.param({"alpha": 0.4, "any": "thing"}, id="unexpected-keys-alpha"),
        pytest.param({"mix": "mul", "any": "thing"}, id="unexpected-keys-mixmul"),
    ],
)
def test_layering_from_dict_invalid_input(d):
    with pytest.raises(ValueError, match=rf"^Failed to parse layering={d}$"):
        Layering.from_dict(d)
