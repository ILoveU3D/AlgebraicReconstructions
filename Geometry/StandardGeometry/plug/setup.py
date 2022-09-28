from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name="JITStandardGeometry",
      include_paths=["include"], ext_modules=[
        CUDAExtension(
            "JITStandardGeometry",
            ["jit.cpp", "kernel/forwardKernel.cu", "kernel/backwardKernel.cu"]
        )
    ],
      zip_safe=False,
      cmdclass={
          "build_ext": BuildExtension
      })