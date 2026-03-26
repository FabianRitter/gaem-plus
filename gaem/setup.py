from setuptools import setup, find_packages

setup(
    name="gaem",
    version="0.1.0",
    description="GAEM+: Generalized Audio Encoder Merging",
    author="Fabian Ritter",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "scipy>=1.5",
        "numpy",
    ],
    python_requires=">=3.8",
)
