from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np, sys

compile_args = []
if sys.platform == "darwin":
    compile_args += ["-O3", "-Wno-cpp", "-Wno-unreachable-code-fallthrough"]
else:
    compile_args += ["-O3", "-Wno-cpp", "-Wno-unreachable-code-fallthrough"]

ext = Extension(
    "ns_hmm",
    ["ns_hmm.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
)

setup(
    name="ns_hmm",
    ext_modules=cythonize(ext, language_level=3),
    zip_safe=False,
)
