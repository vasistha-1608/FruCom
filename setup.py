from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


cxx_args = ['/std:c++17']

setup(
    name='my_cuda_kernel',
    ext_modules=[
        CUDAExtension('my_cuda_kernel', [
            'src/main.cpp',
            'src/kernel.cu',
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': []})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)