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
    python_requires=">=3.9",
)
