from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np, sys

compile_args = []
if sys.platform == "darwin":
    compile_args += ["-O3", "-Wno-cpp", "-Wno-unreachable-code-fallthrough"]
else:
    compile_args += ["-O3", "-Wno-cpp", "-Wno-unreachable-code-fallthrough"]

ext = Extension(
    "hmm_bw",
    ["hmm_bw.pyx"],
    include_dirs=[np.get_include()],
    # define_macros=[("NPY_NO_DEPRECATED_API","NPY_1_7_API_VERSION")],  # ← 주석/삭제
    extra_compile_args=compile_args,
)

setup(
    name="hmm_bw",
    ext_modules=cythonize(ext, language_level=3),
    zip_safe=False,
)
