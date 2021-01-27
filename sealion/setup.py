from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension
import joblib
ext_modules=[
    Extension("cython_ensemble_learning",
              ["cython_ensemble_learning.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )
]

setup(ext_modules=cythonize("/Users/anish/Documents/PycharmProjects/playProject/venv/sealion/cython_ensemble_learning.pyx"), include_dirs = [numpy.get_include()])
setup(ext_modules=cythonize("/Users/anish/Documents/PycharmProjects/playProject/venv/sealion/cython_naive_bayes.pyx"), include_dirs = [numpy.get_include()])
setup(ext_modules=cythonize("/Users/anish/Documents/PycharmProjects/playProject/venv/sealion/cython_tsne.pyx"), include_dirs = [numpy.get_include()])
setup(ext_modules=cythonize("/Users/anish/Documents/PycharmProjects/playProject/venv/sealion/cython_knn.pyx"), include_dirs = [numpy.get_include()])
setup(ext_modules=cythonize("/Users/anish/Documents/PycharmProjects/playProject/venv/sealion/cython_decision_tree_functions.pyx"), include_dirs = [numpy.get_include()])

