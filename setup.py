import pathlib
import sys
from distutils.core import setup
from os import path

import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    read_me_description = fh.read()

with open("requirements.txt") as reqs:
    requirements = reqs.read().split("\n")

non_python_files = [
    "cython_decision_tree_functions.pyx",
    "cython_tsne.pyx",
    "cython_unsupervised_clustering.pyx",
    "cython_naive_bayes.pyx",
    "cython_ensemble_learning.pyx",
    "cython_knn.pyx",
    "cython_mixtures.pyx"
]

version_name = sys.argv[1].replace("refs/tags/", "")
del sys.argv[1]

setup(
    name="sealion",
    packages=setuptools.find_packages(),
    package_data={"": non_python_files},
    include_package_data=True,
    version=version_name,
    license="Apache",
    description="SeaLion is a comprehensive machine learning and data science library for beginners and ml-engineers alike.",
    author="Anish Lakkapragada",
    author_email="anish.lakkapragada@gmail.com",
    url="https://github.com/anish-lakkapragada/SeaLion",
    keywords=["Machine Learning", "Data Science", "Python"],
    install_requires=requirements,
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    python_requires=">=3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
)
