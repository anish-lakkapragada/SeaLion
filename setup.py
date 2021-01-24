from distutils.core import setup
setup(
  name = 'sealion',
  packages = ['sealion'],
  version = '2.1',
  license='MIT',
  description = 'sealion is a simple machine learning and data science library for beginners and ml-engineers alike.',
  author = 'Anish Lakkapragada',
  author_email = 'anish.lakkapragada@gmail.com',
  url = 'https://github.com/anish-lakkapragada/sealion',
  download_url = 'https://github.com/anish-lakkapragada/sealion/archive/v2.1.tar.gz',
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
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
