# This file meant to retain compatibility with older version of python
# when installing the package in edit mode
# check pyproject for details about the package

from setuptools import setup


requirements = [
    "torch",
    "torchvision",
    "transformers",
    "diffusers[torch]",
    "lpips",
    "Pillow",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "omegaconf",
    "tqdm",
    "PyYAML",
    "einops",
    "lightning",
    "taming-transformers-rom1504",
    "hydra-core",
    "pandas",
    "pytorch_fid",
    "imscore",
    "sentencepiece",
]


setup(
    name="ding",
    version="0.0.0",
    install_requires=requirements,
)
