from distutils.core import setup
setup(
  name = 'sealion',
  packages = ['sealion'],
  version = '1.0',
  license='MIT',
  description = 'Sealion is a simple machine learning and data science library for beginners and ml-engineers alike.',
  author = 'Anish Lakkapragada',
  author_email = 'anish.lakkapragada@gmail.com',
  url = 'https://github.com/anish-lakkapragada/sealion',
  download_url = 'https://github.com/anish-lakkapragada/sealion/archive/v2.0.0.tar.gz',
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
