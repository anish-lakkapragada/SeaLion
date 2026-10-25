import os

import setuptools
from setuptools import Extension

with open("README.md", "r") as fh:
    read_me_description = fh.read()

with open("requirements.txt") as reqs:
    requirements = reqs.read().strip().split("\n")

non_python_files = [
    "cython_decision_tree_functions.pyx",
    "cython_tsne.pyx",
    "cython_unsupervised_clustering.pyx",
    "cython_naive_bayes.pyx",
    "cython_ensemble_learning.pyx",
    "cython_knn.pyx",
    "cython_mixtures.pyx",
]

# Release builds set PYPI_VERSION to the git ref (see
# .github/workflows/build.yaml). Local builds fall back to the version
# in sealion/__init__.py instead of failing with a KeyError.
if "PYPI_VERSION" in os.environ:
    version_name = os.path.basename(os.environ["PYPI_VERSION"])
else:
    version_name = "4.4.5"

# Cython extensions are built at install time, not at import time.
# Previously sealion/__init__.py compiled them on every first import
# (see issue #11); that runtime compilation is removed, so a plain
# `import sealion` must never spawn a build.
try:
    import numpy
except ImportError as exc:
    raise SystemExit(
        "numpy is required to build SeaLion's Cython extensions. "
        "Install it first, e.g. `pip install numpy`, then retry."
    ) from exc

try:
    from Cython.Build import cythonize
except ImportError as exc:
    raise SystemExit(
        "Cython is required to build SeaLion's Cython extensions. "
        "Install it first, e.g. `pip install Cython`, then retry, "
        "or install a pre-built wheel."
    ) from exc

cython_modules = [
    "cython_decision_tree_functions",
    "cython_tsne",
    "cython_unsupervised_clustering",
    "cython_naive_bayes",
    "cython_ensemble_learning",
    "cython_knn",
    "cython_mixtures",
]

extensions = cythonize(
    [
        Extension(
            "sealion.{0}".format(module),
            ["sealion/{0}.pyx".format(module)],
            include_dirs=[numpy.get_include()],
        )
        for module in cython_modules
    ],
    compiler_directives={"language_level": "3"},
)

setuptools.setup(
    name="sealion",
    packages=setuptools.find_packages(),
    package_data={"": non_python_files},
    include_package_data=True,
    ext_modules=extensions,
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
