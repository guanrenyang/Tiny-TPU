from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

module_name = "mylinear_cpp"


ext_module = CppExtension(
    name=module_name,
    sources=["mylinear.cpp"]
)

setup(
    name=module_name,
    ext_modules=[ext_module],
    cmdclass={
        'build_ext': BuildExtension
    }
)