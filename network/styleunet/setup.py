from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

setup(name='upfirdn2d',
      ext_modules=[cpp_extension.CUDAExtension('upfirdn2d', [
        os.path.join("upfirdn2d.cpp"),
        os.path.join("upfirdn2d_kernel.cu")])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='fused',
      ext_modules=[cpp_extension.CUDAExtension('fused', [
        os.path.join("fused_bias_act.cpp"),
        os.path.join("fused_bias_act_kernel.cu")])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
