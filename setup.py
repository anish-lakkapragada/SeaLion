import setuptools
from distutils.core import setup

setup(
  name = 'sealion',
  packages = setuptools.find_packages(),
  include_package_data=True,
  version = '2.8.6',
  license='MIT',
  description = 'SeaLion is a simple machine learning and data science library for beginners and ml-engineers alike.',
  author = 'Anish Lakkapragada',
  author_email = 'anish.lakkapragada@gmail.com',
  url = 'https://github.com/anish-lakkapragada/SeaLion',
  download_url = 'https://github.com/anish-lakkapragada/SeaLion/archive/v2.8.6.tar.gz',
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
  python_requires='>=3.0',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
