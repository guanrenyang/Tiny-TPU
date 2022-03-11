import os

from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

module_name = "myMM"
csrc_path = "csrc"

def get_sources():
    sources = []

    for file in os.listdir(csrc_path):
        if not file.endswith((".cpp", ".cu")):
            continue
        sources.append(os.path.join(csrc_path, file))

    return sources

ext_module = CUDAExtension(
    name=module_name,
    sources=get_sources()
)

setup(
    name=module_name,
    ext_modules=[ext_module],
    cmdclass={
        'build_ext': BuildExtension
    }
)
