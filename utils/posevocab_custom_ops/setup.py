from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

sources = ["bind.cpp",
           "near_far_smpl_kernel.cu",
           "nearest_face_kernel.cu",
           "point_mesh.cu"]

setup(
    name='posevocab_custom_ops',
    ext_modules=[
        CUDAExtension(
            name='posevocab_custom_ops',
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
