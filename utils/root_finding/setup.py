from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

sources = ["bind.cpp",
           "root_finding.cu"]

setup(
    name='root_finding',
    ext_modules=[
        CUDAExtension(
            name='root_finding',
            sources=sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("./")],
                "nvcc": ["-O2", "-I{}".format("./")],
            },
            define_macros=[("WITH_CUDA", None)],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
