from Cython.Build import cythonize
from setuptools import setup, Extension
import numpy
import warnings

warnings.filterwarnings("ignore")

files = ["cython_decision_tree_functions", "cython_tsne", "cython_unsupervised_clustering",
         "cython_naive_bayes", "cython_knn", "cython_ensemble_learning"]

def setup_indiv(file_name) :
    extension = Extension(file_name, [file_name + ".pyx"], include_dirs = [numpy.get_include()])
    setup(ext_modules = cythonize(extension))

from joblib import parallel_backend, delayed, Parallel
with parallel_backend("threading", n_jobs=-1) :
    print("finished one")
    Parallel()(delayed(setup_indiv)(file_name) for file_name in files)


