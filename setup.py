from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("entropy.linearModel._linear", ["src/entropy/linearModel/_linear.pyx"]),
    Extension("entropy.linearModel._logistic", ["src/entropy/linearModel/_logistic.pyx"]),
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[np.get_include()]
)