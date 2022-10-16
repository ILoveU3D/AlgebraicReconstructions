from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name="JITShenzhenGeometry",
      include_paths=["include"], ext_modules=[
        CUDAExtension(
            "JITShenzhenGeometry",
            ["jit.cpp", "kernel/forwardKernel.cu", "kernel/backwardKernel.cu"]
        )
    ],
      zip_safe=False,
      cmdclass={
          "build_ext": BuildExtension
      })