import setuptools
from distutils.core import setup

non_python_files = ['cython_ensemble_learning.cpython-38-darwin.so', 'cython_decision_tree_functions.cpython-38-darwin.so', 'cython_naive_bayes.c', 'cython_decision_tree_functions.pyx', 'cython_tsne.pyx', 'cython_unsupervised_clustering.cpython-38-darwin.so', '.DS_Store', 'cython_decision_tree_functions.so', 'cython_unsupervised_clustering.pyx', 'cython_naive_bayes.so', 'cython_tsne.o', 'cython_knn.so', 'cython_knn.c', 'cython_naive_bayes.pyx', 'cython_ensemble_learning.o', 'cython_decision_tree_functions.o', 'cython_ensemble_learning.pyx', 'cython_naive_bayes.o', 'cython_ensemble_learning.so', 'cython_knn.cpython-38-darwin.so', 'cython_tsne.cpython-38-darwin.so', 'cython_tsne.c', 'cython_naive_bayes.cpython-38-darwin.so', 'cython_knn.pyx', 'cython_knn.o', 'cython_tsne.so', 'cython_ensemble_learning.c', 'cython_decision_tree_functions.c']

setup(
  name = 'sealion',
  packages = setuptools.find_packages(),
  package_data = {'' : non_python_files},
  include_package_data=True,
  version = '3.0.5',
  license='MIT',
  description = 'SeaLion is a simple machine learning and data science library for beginners and ml-engineers alike.',
  author = 'Anish Lakkapragada',
  author_email = 'anish.lakkapragada@gmail.com',
  url = 'https://github.com/anish-lakkapragada/SeaLion',
  download_url = 'https://github.com/anish-lakkapragada/SeaLion/archive/v3.0.5.tar.gz',
  keywords = ['Machine Learning', 'Data Science', 'Python'],
  install_requires=[
          'numpy',
          'joblib',
          'pandas',
          'scipy',
          'tqdm',
          'multiprocess'
      ],
  long_description=open('README.md', 'r').read(),
  python_requires='>=3',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
