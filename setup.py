import sys
from distutils.extension import Extension
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

MINIMAL_PYTHON_VERSION = (3, 9)


def _check_python_requires():
    root = Path(__file__).parent
    with open(root / "pyproject.toml", "rb") as fh:
        metadata = tomllib.load(fh)
    pyproject_requires = metadata["project"]["requires-python"]
    setuppy_requires = f">={'.'.join(str(_) for _ in MINIMAL_PYTHON_VERSION)}"
    if pyproject_requires != setuppy_requires:
        raise RuntimeError(
            "mismatched minimal python requirements ! "
            f"pyproject.toml has {pyproject_requires!r}, "
            f"setup.py has {setuppy_requires!r}"
        )


_check_python_requires()


def _get_cpython_tag():
    return f"cp{''.join(str(_) for _ in MINIMAL_PYTHON_VERSION[:2])}"


def _get_hex_minimal_version():
    major, minor, *micro = MINIMAL_PYTHON_VERSION
    if micro:
        raise RuntimeError(
            "setting minimal version with a non-zero micro"
            f" is not implemented (got {MINIMAL_PYTHON_VERSION=})"
        )
    return hex(2**24 * major + 2**16 * minor)


class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            # on CPython, our wheels are abi3
            # and compatible down to Python 3.8
            # (keep in sync with python_requires (pyproject.toml))
            return _get_cpython_tag(), "abi3", plat

        return python, abi, plat


define_macros = [
    ("Py_LIMITED_API", _get_hex_minimal_version()),
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    # keep in sync with runtime requirements (pyproject.toml)
    ("NPY_TARGET_VERSION", "NPY_1_19_API_VERSION"),
]

setup(
    ext_modules=cythonize(
        [
            Extension(
                "lick._vendor.vectorplot.core",
                ["src/lick/_vendor/vectorplot/core.pyx"],
                include_dirs=[numpy.get_include()],
                define_macros=define_macros,
                py_limited_api=True,
            ),
        ],
        compiler_directives={"language_level": 3},
    ),
)
