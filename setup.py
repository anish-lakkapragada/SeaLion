import setuptools
from distutils.core import setup
import pathlib
from setuptools import setup


non_python_files = ['cython_decision_tree_functions.pyx', 'cython_tsne.pyx', 'cython_unsupervised_clustering.pyx',  'cython_naive_bayes.pyx',  'cython_ensemble_learning.pyx',  'cython_knn.pyx']

setup(
  name = 'sealion',
  packages = setuptools.find_packages(),
  package_data = {'' : non_python_files},
  include_package_data=True,
  version = '3.0.8',
  license='MIT',
  description='SeaLion is a comprehensive machine learning and data science library for beginners and ml-engineers alike.',
  author = 'Anish Lakkapragada',
  author_email = 'anish.lakkapragada@gmail.com',
  url = 'https://github.com/anish-lakkapragada/SeaLion',
  download_url = 'https://github.com/anish-lakkapragada/SeaLion/archive/v3.0.8.tar.gz',
  keywords = ['Machine Learning', 'Data Science', 'Python'],
  install_requires=[
          'numpy',
          'joblib',
          'pandas',
          'scipy',
          'tqdm',
          'multiprocess'
      ],
  long_description="README.md",
  python_requires='>=3',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
