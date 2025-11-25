# setup.py
from setuptools import setup, find_packages

setup(
    name="reality-sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",  # Для визуализации
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x>=12.0.0",  # Для NVIDIA GPU с CUDA 12.x
            # Или для других версий CUDA:
            # "cupy-cuda11x>=11.0.0",  # Для CUDA 11.x
            # "cupy-cuda10x>=10.0.0",  # Для CUDA 10.x
        ],
        "vulkan": [
            "vulkpy>=0.1.0",  # Для AMD/NVIDIA/Intel GPU через Vulkan
        ],
        "opencl": [
            "pyopencl>=2023.1.0",  # Для AMD/NVIDIA/Intel GPU через OpenCL
        ],
        "gpu-all": [
            "cupy-cuda12x>=12.0.0",
            "vulkpy>=0.1.0",
            "pyopencl>=2023.1.0",
        ],
    },
    python_requires=">=3.9",
)
