from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                "lick._vendor.vectorplot.core",
                ["lick/_vendor/vectorplot/core.pyx"],
                include_dirs=[numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
        ],
        compiler_directives={"language_level": 3},
        annotate=True,
    ),
)
