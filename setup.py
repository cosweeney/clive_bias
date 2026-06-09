from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="tangential_pvd_vs_r_engine",
        sources=["notebooks/streaming_model/tangential_pvd_vs_r_engine.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name='clive',
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3"),
)