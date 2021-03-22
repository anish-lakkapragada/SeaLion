from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy
import warnings

warnings.filterwarnings("ignore")

files = [
    "cython_decision_tree_functions",
    "cython_tsne",
    "cython_unsupervised_clustering",
    "cython_naive_bayes",
    "cython_knn",
    "cython_ensemble_learning",
    "cython_mixtures"
]


def setup_indiv(file_name):
    extension = Extension(
        file_name, [file_name + ".pyx"], include_dirs=[numpy.get_include()]
    )
    setup(ext_modules=cythonize(extension))


from joblib import parallel_backend, delayed, Parallel

for file_name in files :
    setup_indiv(file_name)

