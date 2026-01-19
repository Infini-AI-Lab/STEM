"""
Build script for stem_gather C++ extension.
This creates an installable package that can be built ahead-of-time.

Usage:
    python setup_build.py build_ext --inplace    # Build in-place (recommended)
    python setup_build.py install                # Install to site-packages
    pip install .                                 # Alternative: install via pip
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Get the source file path
_src_path = os.path.join(os.path.dirname(__file__), "src", "stem_gather.cpp")

setup(
    name="stem_gather_ext",
    ext_modules=[
        CppExtension(
            name="stem_gather_ext",
            sources=[_src_path],
            extra_compile_args=["-O3"],
            # If you really want OpenMP, add:
            # extra_compile_args=["-O3", "-fopenmp"],
            # extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=False)
    },
    zip_safe=False,
)

