from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
    name="dynamic_python_model",
    sources=[
        "*.pyx",
        "../src/dynamic_model.cc",
        "../src/utility.cc"
    ],
    include_dirs = ['/usr/local/include/eigen3', '../include', 'cpp/', numpy.get_include()],
    language="c++",
    ),
]

setup(
    name="dynamic_model",
    ext_modules = cythonize(ext_modules)
)
