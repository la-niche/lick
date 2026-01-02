import numpy as np
import pytest

from lick import _api
from lick._image_processing import (
    Identity,
    Layering,
    LayeringMode,
    NorthWestLightSource,
)
from lick._interpolation import Grid, Mesh


def test_get_kernel_invalid_call():
    with pytest.raises(
        TypeError,
        match=(
            r"^kernel and kernel_length keyword arguments are mutually exclusive, "
            r"but both were received\.$"
        ),
    ):
        _api.get_kernel(1, size=2, max_auto_size=10, dtype="float32")


def test_get_kernel_length_depr():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The kernel_length keyword argument is deprecated since lick v0\.10\.0 "
            r"and will be removed in a future version. Use the kernel argument instead\.$"
        ),
    ):
        kernel = _api.get_kernel(
            _api.UNSET,
            size=9,
            max_auto_size=_api.LegacyDefault.KERNEL_SIZE.value,
            dtype="float32",
        )
    assert kernel.size == 9
    assert kernel.dtype == "float32"


def test_get_kernel_default():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The kernel argument was not explicitly specified. "
            r"Its default value will change from 'auto-legacy' to 'auto-adjust' in a future release.\n"
            r"To silence this warning, set the argument explicitly\.$"
        ),
    ):
        kernel = _api.get_kernel(
            _api.UNSET, size=_api.UNSET, max_auto_size=10, dtype="float32"
        )

    # max_auto_size is *supposed* to be ignored by default as of lick 0.10.0
    assert kernel.size == _api.LegacyDefault.KERNEL_SIZE.value
    assert kernel.dtype == "float32"


def test_get_kernel_legacy_value():
    with pytest.warns(
        PendingDeprecationWarning,
        match=(
            r"^The kernel argument was set to 'auto\-legacy'\. "
            r"This value is scheduled for removal in a future version\. "
            r"To silence this warning, set "
            r"kernel=np\.sin\(np\.linspace\(0, np\.pi, 101, endpoint=False\)\)\.astype\(<ref_dtype>\)$"
        ),
    ):
        kernel = _api.get_kernel(
            _api.LegacyDefault.KERNEL.value,
            size=_api.UNSET,
            max_auto_size=10,
            dtype="float32",
        )

    # max_auto_size is *supposed* to be ignored by default as of lick 0.10.0
    assert kernel.size == _api.LegacyDefault.KERNEL_SIZE.value
    assert kernel.dtype == "float32"


@pytest.mark.parametrize("max_auto_size", [1, 2, 10])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_get_kernel_auto_small(max_auto_size, dtype):
    kernel = _api.get_kernel(
        "auto-adjust", size=_api.UNSET, max_auto_size=max_auto_size, dtype=dtype
    )
    assert kernel.size == max_auto_size
    assert kernel.size < _api.LegacyDefault.KERNEL_SIZE.value
    assert kernel.dtype == dtype


@pytest.mark.parametrize(
    "max_auto_size",
    [
        _api.LegacyDefault.KERNEL_SIZE.value + 1,
        _api.LegacyDefault.KERNEL_SIZE.value * 2,
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_get_kernel_auto_large(max_auto_size, dtype):
    kernel = _api.get_kernel(
        "auto-adjust", size=_api.UNSET, max_auto_size=max_auto_size, dtype=dtype
    )
    assert kernel.size < max_auto_size
    assert kernel.size == 21
    assert kernel.dtype == dtype


def test_get_niter_lic_default():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The niter_lic argument was not explicitly specified. "
            r"Its default value will change from 5 to 1 in a future release.\n"
            r"To silence this warning, set the argument explicitly\.$"
        ),
    ):
        n = _api.get_niter_lic(_api.UNSET)

    assert n == _api.LegacyDefault.NITER_LIC.value


@pytest.mark.parametrize("niter", [1, 2, 10])
def test_get_niter_lic_explicit(niter):
    n = _api.get_niter_lic(niter)
    assert n == niter


def test_get_post_lic_invalid_call():
    with pytest.raises(
        TypeError,
        match=(
            r"post_lic and light_source keyword arguments are "
            r"mutually exclusive, but both were received\."
        ),
    ):
        _api.get_post_lic("north-west-light-source", light_source=True)


def test_get_post_lic_all_default():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The post_lic argument was not explicitly specified. "
            r"Its default value will change from 'north-west-light-source' to None in a future release\.\n"
            r"To silence this warning, set the argument explicitly\.$"
        ),
    ):
        post_lic = _api.get_post_lic(_api.UNSET, light_source=_api.UNSET)
    assert isinstance(post_lic, NorthWestLightSource)


@pytest.mark.parametrize(
    "light_source_bool, cls",
    [
        (True, NorthWestLightSource),
        (False, Identity),
    ],
)
def test_get_post_lic_light_source_bool(light_source_bool, cls):
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The light_source keyword argument is deprecated since lick v0\.10\.0 "
            r"and will be removed in a future version\. Use the post_lic argument instead\.$"
        ),
    ):
        post_lic = _api.get_post_lic(_api.UNSET, light_source=light_source_bool)
    assert isinstance(post_lic, cls)


def test_get_post_lic_none():
    post_lic = _api.get_post_lic(None, light_source=_api.UNSET)
    assert isinstance(post_lic, Identity)


@pytest.mark.parametrize("indexing", ["xy", "ij"])
def test_get_indexing(indexing):
    assert _api.get_indexing(indexing) == indexing


def test_get_indexing_unset():
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The indexing argument was not explicitly specified\. "
            r"Falling back 'xy' as a default value, but this parameter "
            r"will be required in a future release\.\n"
            r"To silence this warning, set the argument explicitly\.$"
        ),
    ):
        _api.get_indexing(_api.UNSET)


def test_get_indexing_unknown():
    with pytest.raises(ValueError, match=r"^Received invalid indexing='spam'$"):
        _api.get_indexing("spam")


def test_get_grid_or_mesh_grid():
    res = _api.get_grid_or_mesh(np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 9))
    assert type(res) is Grid


def test_get_grid_or_mesh_mesh():
    res = _api.get_grid_or_mesh(
        np.atleast_2d(np.linspace(-1.0, 1.0, 8)),
        np.atleast_2d(np.linspace(-1.0, 1.0, 8)),
    )
    assert type(res) is Mesh


def test_get_mesh_from_grid():
    _api.get_mesh(
        np.linspace(-1.0, 1.0, 8),
        np.linspace(-1.0, 1.0, 8),
        indexing="ij",
    )


def test_get_mesh_no_indexing():
    _api.get_mesh(
        np.atleast_2d(np.linspace(-1.0, 1.0, 8)),
        np.atleast_2d(np.linspace(-1.0, 1.0, 8)),
        indexing=_api.UNSET,
    )


def test_get_mesh_ignored_indexing():
    with pytest.warns(
        UserWarning,
        match=r"^indexing='ij' will be ignored because a mesh is already defined$",
    ):
        _api.get_mesh(
            np.atleast_2d(np.linspace(-1.0, 1.0, 8)),
            np.atleast_2d(np.linspace(-1.0, 1.0, 8)),
            indexing="ij",
        )


def test_get_grid_or_mesh_mixed_ndims():
    with pytest.raises(
        TypeError,
        match=(
            r"^Received x.shape=\(8,\) and y.shape=\(1, 9\)\. "
            r"Expected them to have identical dimensionalities\.$"
        ),
    ):
        _api.get_grid_or_mesh(
            np.linspace(-1.0, 1.0, 8),
            np.atleast_2d(np.linspace(-1.0, 1.0, 9)),
        )


def test_get_layering_default():
    layering = _api.get_layering(
        _api.UNSET, alpha=_api.UNSET, alpha_transparency=_api.UNSET
    )
    assert layering == Layering(
        mode=LayeringMode.ALPHA, alpha=_api.LegacyDefault.ALPHA.value
    )


@pytest.mark.parametrize(
    "d",
    [
        pytest.param({"alpha": 0.1}, id="alpha-10%"),
        pytest.param({"alpha": 0.4}, id="alpha-40%"),
        pytest.param({"mix": "mul"}, id="mixmul"),
    ],
)
def test_get_layering_explicit(d):
    layering = _api.get_layering(d, alpha=_api.UNSET, alpha_transparency=_api.UNSET)
    assert layering == Layering.from_dict(d)


@pytest.mark.parametrize(
    "kwarg",
    [
        pytest.param({"alpha": 0.5}, id="alpha"),
        pytest.param({"alpha_transparency": False}, id="alpha_transparency"),
    ],
)
def test_get_layering_invalid_call(kwarg):
    kw = next(iter(kwarg))
    kwargs = {"alpha": _api.UNSET, "alpha_transparency": _api.UNSET} | kwarg
    with pytest.raises(
        TypeError,
        match=(
            rf"^{kw} and layering keyword arguments are mutually exclusive, "
            r"but both were received\.$"
        ),
    ):
        _api.get_layering({"alpha": 0.2}, **kwargs)


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.8])
def test_get_layering_from_alpha(alpha):
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The alpha keyword argument is deprecated since lick v0\.10\.0 "
            r"and will be removed in a future version\. Use the layering argument instead\.$"
        ),
    ):
        layering = _api.get_layering(
            _api.UNSET, alpha=alpha, alpha_transparency=_api.UNSET
        )

    assert layering == Layering(mode=LayeringMode.ALPHA, alpha=alpha)


@pytest.mark.parametrize(
    "transparency, expected",
    [
        pytest.param(False, Layering(mode=LayeringMode.MIX_MUL), id="mixmul"),
        pytest.param(
            True,
            Layering(mode=LayeringMode.ALPHA, alpha=_api.LegacyDefault.ALPHA.value),
            id="alpha",
        ),
    ],
)
def test_get_layering_from_alpha_transparency(transparency, expected):
    with pytest.warns(
        DeprecationWarning,
        match=(
            r"^The alpha_transparency keyword argument is deprecated since lick v0\.10\.0 "
            r"and will be removed in a future version\. Use the layering argument instead\.$"
        ),
    ):
        layering = _api.get_layering(
            _api.UNSET, alpha=_api.UNSET, alpha_transparency=transparency
        )

    assert layering == expected
